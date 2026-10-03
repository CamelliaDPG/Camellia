#!/usr/bin/env python3
"""Compare ordinary and FEM-mass-weighted TT rounding of a saved DPG solution."""

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import solve_triangular

from plot_vlasov_spectra import assemble_states


BLOCK_NAMES = ("trace_x", "trace_t", "trace_v", "field")
MASS_NAMES = ("trace_x", "trace_t", "trace_v", "field")


def read_cores(h5_file, prefix):
    cores = []
    for dimension in range(3):
        sizes = tuple(
            int(size)
            for size in h5_file[f"{prefix}_core_{dimension}sizes"][()][::-1]
        )
        data = h5_file[f"{prefix}_core_{dimension}data"][()]
        cores.append(data.reshape(sizes))
    return cores


def dense_vector(h5_file, prefix):
    cores = read_cores(h5_file, prefix)
    tensor = cores[0][0, :, :]
    for core in cores[1:]:
        tensor = np.einsum("...a,aib->...ib", tensor, core, optimize=True)
    return tensor[..., 0]


def apply_ttm(cores, tensor):
    """Apply a three-dimensional TT matrix without forming its dense matrix."""
    temporary = np.einsum("ijb,jkl->ibkl", cores[0][0], tensor, optimize=True)
    temporary = np.einsum(
        "ibkl,bmkc->imcl", temporary, cores[1], optimize=True
    )
    return np.einsum(
        "imcl,cnl->imn", temporary, cores[2][:, :, :, 0], optimize=True
    )


def apply_mode_matrix(tensor, matrix, dimension):
    transformed = np.tensordot(matrix, tensor, axes=(1, dimension))
    return np.moveaxis(transformed, 0, dimension)


def apply_factors(tensor, factors, transpose=False, inverse=False):
    transformed = tensor
    for dimension, factor in enumerate(factors):
        matrix = factor.T if transpose else factor
        if inverse:
            moved = np.moveaxis(transformed, dimension, 0)
            shape = moved.shape
            moved = solve_triangular(
                matrix, moved.reshape(shape[0], -1), lower=not transpose
            ).reshape(shape)
            transformed = np.moveaxis(moved, 0, dimension)
        else:
            transformed = apply_mode_matrix(transformed, matrix, dimension)
    return transformed


def read_mass_factors(filename):
    with h5py.File(filename, "r") as mass_h5:
        cores = read_cores(mass_h5, "mass")
    factors = []
    for core in cores:
        if core.shape[0] != 1 or core.shape[3] != 1:
            raise ValueError(f"Mass matrix {filename} is not separable")
        matrix = core[0, :, :, 0]
        matrix = 0.5 * (matrix + matrix.T)
        factors.append(np.linalg.cholesky(matrix))
    return factors


def select_rank(singular_values, relative_tolerance=None, rank_cap=None):
    rank = singular_values.size
    if relative_tolerance is not None:
        rank = int(
            np.count_nonzero(singular_values > relative_tolerance * singular_values[0])
        )
        rank = max(rank, 1)
    if rank_cap is not None:
        rank = min(rank, rank_cap)
    return rank


def tt_round(tensor, relative_tolerance=None, rank_cap=None):
    """Dense reference version of BoBa's right-to-left TT rounding sweep."""
    n0, n1, n2 = tensor.shape

    left, singular_values, right = np.linalg.svd(
        tensor.reshape(n0 * n1, n2), full_matrices=False
    )
    rank2 = select_rank(singular_values, relative_tolerance, rank_cap)
    core2 = right[:rank2, :]
    remainder = (left[:, :rank2] * singular_values[:rank2]).reshape(n0, n1, rank2)

    left, singular_values, right = np.linalg.svd(
        remainder.reshape(n0, n1 * rank2), full_matrices=False
    )
    rank1 = select_rank(singular_values, relative_tolerance, rank_cap)
    core0 = left[:, :rank1]
    core1 = (singular_values[:rank1, None] * right[:rank1, :]).reshape(
        rank1, n1, rank2
    )

    rounded = np.einsum("ia,ajb,bk->ijk", core0, core1, core2, optimize=True)
    return rounded, (rank1, rank2)


def weighted_norm(tensor, factors):
    return np.linalg.norm(apply_factors(tensor, factors, transpose=True))


def tt_storage(shape, ranks):
    n0, n1, n2 = shape
    rank1, rank2 = ranks
    return n0 * rank1 + rank1 * n1 * rank2 + rank2 * n2


def round_block(tensor, factors, method, relative_tolerance=None, rank_cap=None):
    if method == "coefficient":
        return tt_round(tensor, relative_tolerance, rank_cap)
    transformed = apply_factors(tensor, factors, transpose=True)
    rounded, ranks = tt_round(transformed, relative_tolerance, rank_cap)
    recovered = apply_factors(rounded, factors, transpose=True, inverse=True)
    return recovered, ranks


class DPGResidual:
    def __init__(self, prefix):
        with h5py.File(prefix + "stiffness.h5", "r") as stiffness_h5:
            self.gram_cores = read_cores(stiffness_h5, "stiffness_0_0")
            self.constraint = [
                read_cores(stiffness_h5, f"stiffness_{block}_0")
                for block in range(1, 5)
            ]
            self.constraint_transpose = [
                read_cores(stiffness_h5, f"stiffness_0_{block}")
                for block in range(1, 5)
            ]
        with h5py.File(prefix + "rhs.h5", "r") as rhs_h5:
            self.rhs = [dense_vector(rhs_h5, f"rhs_{block}") for block in range(5)]

        self.gram_inverse_factors = [
            np.linalg.inv(core[0, :, :, 0]) for core in self.gram_cores
        ]
        gram_inverse_rhs = apply_factors(self.rhs[0], self.gram_inverse_factors)
        self.schur_rhs = [
            apply_ttm(operator, gram_inverse_rhs) - self.rhs[block]
            for block, operator in enumerate(self.constraint, start=1)
        ]
        self.schur_rhs_norm = self.block_norm(self.schur_rhs)
        self.full_rhs_norm = self.block_norm(self.rhs)

    @staticmethod
    def block_norm(blocks):
        return np.sqrt(sum(np.linalg.norm(block) ** 2 for block in blocks))

    def evaluate(self, physical_blocks):
        constraint_transpose_solution = sum(
            (
                apply_ttm(operator, block)
                for operator, block in zip(
                    self.constraint_transpose, physical_blocks
                )
            ),
            np.zeros_like(self.rhs[0]),
        )
        test_solution = apply_factors(
            self.rhs[0] - constraint_transpose_solution,
            self.gram_inverse_factors,
        )

        schur_residual = [
            apply_ttm(operator, test_solution) - self.rhs[block]
            for block, operator in enumerate(self.constraint, start=1)
        ]
        test_residual = (
            apply_ttm(self.gram_cores, test_solution)
            + constraint_transpose_solution
            - self.rhs[0]
        )
        full_residual = [test_residual, *schur_residual]
        return {
            "relative_schur_residual": self.block_norm(schur_residual)
            / self.schur_rhs_norm,
            "relative_full_residual": self.block_norm(full_residual)
            / self.full_rhs_norm,
        }


def format_ranks(block_ranks):
    return ";".join(
        f"{name}:{ranks[0]}x{ranks[1]}"
        for name, ranks in zip(BLOCK_NAMES, block_ranks)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        default="vlasov_system_nx_16_nt_16_nv_16_p_3_dp_2_L2_norm_",
    )
    parser.add_argument(
        "--solution",
        default="vlasov_system_nx_16_nt_16_nv_16_p_3_dp_2_L2_norm_amen_soln.h5",
    )
    parser.add_argument(
        "--direct-dense",
        action="store_true",
        help="use the dense-direct Schur solution reconstructed from the HDF5 system",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("vlasov_spectra/n16_weighted")
    )
    parser.add_argument(
        "--tolerances", nargs="*", type=float, default=[1e-4, 1e-6, 1e-8, 1e-10]
    )
    parser.add_argument("--rank-caps", nargs="*", type=int, default=[1, 2, 4, 6, 8])
    args = parser.parse_args()

    if args.direct_dense:
        states, direct_residual = assemble_states(args.prefix)
        physical_blocks = [states[name]["direct discrete"] for name in BLOCK_NAMES]
        print(f"dense_direct_relative_schur_residual={direct_residual:.6e}")
    else:
        with h5py.File(args.solution, "r") as solution_h5:
            physical_blocks = [
                dense_vector(solution_h5, f"soln_{block}") for block in range(1, 5)
            ]
    mass_factors = [
        read_mass_factors(args.prefix + f"mass_{name}.h5") for name in MASS_NAMES
    ]
    residual = DPGResidual(args.prefix)

    coefficient_norm = np.sqrt(sum(np.linalg.norm(block) ** 2 for block in physical_blocks))
    fem_norm = np.sqrt(
        sum(
            weighted_norm(block, factors) ** 2
            for block, factors in zip(physical_blocks, mass_factors)
        )
    )

    configurations = [("baseline", "none", None)]
    configurations += [("tolerance", f"{value:.0e}", value) for value in args.tolerances]
    configurations += [("rank_cap", str(value), value) for value in args.rank_caps]

    summary_rows = []
    block_rows = []
    for configuration, value_label, value in configurations:
        methods = ("unrounded",) if configuration == "baseline" else (
            "coefficient",
            "fem_mass",
        )
        for method in methods:
            rounded_blocks = []
            block_ranks = []
            storage = 0
            coefficient_error_squared = 0.0
            fem_error_squared = 0.0
            for name, block, factors in zip(BLOCK_NAMES, physical_blocks, mass_factors):
                if method == "unrounded":
                    rounded = block
                    spectra = (
                        np.linalg.svd(
                            block.reshape(block.shape[0], -1), compute_uv=False
                        ),
                        np.linalg.svd(
                            block.reshape(-1, block.shape[2]), compute_uv=False
                        ),
                    )
                    ranks = tuple(
                        max(1, int(np.count_nonzero(values > 1e-12 * values[0])))
                        for values in spectra
                    )
                else:
                    rounded, ranks = round_block(
                        block,
                        factors,
                        method,
                        # The tiny cutoff prevents dense reconstruction roundoff from
                        # manufacturing ranks above those in the saved TT tensor.
                        relative_tolerance=(
                            value if configuration == "tolerance" else 1.0e-12
                        ),
                        rank_cap=value if configuration == "rank_cap" else None,
                    )
                error = rounded - block
                coefficient_error = np.linalg.norm(error) / np.linalg.norm(block)
                fem_error = weighted_norm(error, factors) / weighted_norm(block, factors)
                coefficient_error_squared += np.linalg.norm(error) ** 2
                fem_error_squared += weighted_norm(error, factors) ** 2
                storage += tt_storage(block.shape, ranks)
                rounded_blocks.append(rounded)
                block_ranks.append(ranks)
                block_rows.append(
                    {
                        "configuration": configuration,
                        "value": value_label,
                        "method": method,
                        "block": name,
                        "rank_1": ranks[0],
                        "rank_2": ranks[1],
                        "tt_storage": tt_storage(block.shape, ranks),
                        "relative_coefficient_error": coefficient_error,
                        "relative_fem_error": fem_error,
                    }
                )

            residuals = residual.evaluate(rounded_blocks)
            row = {
                "configuration": configuration,
                "value": value_label,
                "method": method,
                "ranks": format_ranks(block_ranks),
                "tt_storage": storage,
                "relative_coefficient_error": np.sqrt(coefficient_error_squared)
                / coefficient_norm,
                "relative_fem_error": np.sqrt(fem_error_squared) / fem_norm,
                **residuals,
            }
            summary_rows.append(row)
            print(
                f"{configuration:10s} {value_label:>5s} {method:11s} "
                f"storage={storage:7d} fem_error={row['relative_fem_error']:.3e} "
                f"schur={row['relative_schur_residual']:.3e} "
                f"full={row['relative_full_residual']:.3e} ranks={row['ranks']}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (
        ("rounding_summary.csv", summary_rows),
        ("rounding_blocks.csv", block_rows),
    ):
        path = args.output_dir / filename
        with path.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
