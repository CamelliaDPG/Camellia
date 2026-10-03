#!/usr/bin/env python3
"""Plot TT unfolding spectra for the small dense Vlasov baseline."""

import argparse
import csv
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.linalg import solve

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


BLOCK_NAMES = ("error_test", "trace_x", "trace_t", "trace_v", "field")
CUT_LABELS = (r"$x\mid(t,v)$", r"$(x,t)\mid v$")


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


def dense_matrix(h5_file, prefix):
    cores = read_cores(h5_file, prefix)
    matrix = cores[0][0, :, :, :]
    for core in cores[1:]:
        matrix = np.einsum("...a,aijb->...ijb", matrix, core, optimize=True)
    matrix = np.transpose(matrix[..., 0], (0, 2, 4, 1, 3, 5))
    return matrix.reshape(np.prod(matrix.shape[:3]), np.prod(matrix.shape[3:]))


def gram_inverse(h5_file):
    inverse = np.array([[1.0]])
    for core in read_cores(h5_file, "stiffness_0_0"):
        inverse = np.kron(inverse, np.linalg.inv(core[0, :, :, 0]))
    return inverse


def tt_spectra(tensor):
    """Return the two spectra computed by compute_tt_spectra for a 3D tensor."""
    n0, n1, n2 = tensor.shape
    return (
        np.linalg.svd(tensor.reshape(n0, n1 * n2), compute_uv=False),
        np.linalg.svd(tensor.reshape(n0 * n1, n2), compute_uv=False),
    )


def assemble_states(prefix):
    with h5py.File(prefix + "stiffness.h5", "r") as stiffness_h5, h5py.File(
        prefix + "rhs.h5", "r"
    ) as rhs_h5, h5py.File(prefix + "soln.h5", "r") as solution_h5:
        projected = [dense_vector(solution_h5, f"soln_{block}") for block in range(5)]
        rhs = dense_vector(rhs_h5, "rhs_0").reshape(-1)
        gram_inv = gram_inverse(stiffness_h5)
        constraint_blocks = [
            dense_matrix(stiffness_h5, f"stiffness_{block}_0")
            for block in range(1, 5)
        ]

    constraint = np.vstack(constraint_blocks)
    projected_lambda = np.concatenate([block.reshape(-1) for block in projected[1:]])
    schur = (constraint @ gram_inv) @ constraint.T
    schur_rhs = constraint @ gram_inv @ rhs
    direct_lambda = solve(schur, schur_rhs, assume_a="sym")
    direct_error = gram_inv @ (rhs - constraint.T @ direct_lambda)
    projected_error = gram_inv @ (rhs - constraint.T @ projected_lambda)

    direct = [direct_error.reshape(projected[0].shape)]
    offset = 0
    for projected_block in projected[1:]:
        block_size = projected_block.size
        direct.append(
            direct_lambda[offset : offset + block_size].reshape(projected_block.shape)
        )
        offset += block_size

    relative_residual = np.linalg.norm(schur @ direct_lambda - schur_rhs) / np.linalg.norm(
        schur_rhs
    )

    states = {
        "error_test": {
            "projected exact (zero)": projected[0],
            "projected-induced error": projected_error.reshape(projected[0].shape),
            "direct discrete": direct[0],
        }
    }
    for block, name in enumerate(BLOCK_NAMES[1:], start=1):
        states[name] = {
            "projected exact": projected[block],
            "direct discrete": direct[block],
            "direct - projected": direct[block] - projected[block],
        }
    return states, relative_residual


def apply_modewise_cholesky(tensor, mass_file):
    with h5py.File(mass_file, "r") as mass_h5:
        mass_cores = read_cores(mass_h5, "mass")

    transformed = tensor
    for dimension, core in enumerate(mass_cores):
        if core.shape[0] != 1 or core.shape[3] != 1:
            raise ValueError(f"Mass metric in {mass_file} is not separable")
        mass_matrix = core[0, :, :, 0]
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        lower = np.linalg.cholesky(mass_matrix)
        transformed = np.tensordot(lower.T, transformed, axes=(1, dimension))
        transformed = np.moveaxis(transformed, 0, dimension)
    return transformed


def mass_weight_states(states, mass_prefix):
    mass_names = {
        "error_test": "test",
        "trace_x": "trace_x",
        "trace_t": "trace_t",
        "trace_v": "trace_v",
        "field": "field",
    }
    return {
        variable: {
            state_name: apply_modewise_cholesky(
                tensor, mass_prefix + f"mass_{mass_names[variable]}.h5"
            )
            for state_name, tensor in variable_states.items()
        }
        for variable, variable_states in states.items()
    }


def plot_metric_comparison(coefficient_states, weighted_states, output_directory, cutoff):
    output_directory.mkdir(parents=True, exist_ok=True)
    for variable in BLOCK_NAMES:
        state_name = (
            "AMEn discrete"
            if "AMEn discrete" in coefficient_states[variable]
            else "direct discrete"
        )
        coefficient_spectra = tt_spectra(coefficient_states[variable][state_name])
        weighted_spectra = tt_spectra(weighted_states[variable][state_name])
        figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), sharey=True)
        for cut, axis in enumerate(axes):
            for label, values, color in (
                ("coefficient Euclidean", coefficient_spectra[cut], "tab:blue"),
                ("FEM L2 weighted", weighted_spectra[cut], "tab:orange"),
            ):
                if values[0] == 0.0:
                    continue
                normalized = values / values[0]
                indices = np.arange(1, values.size + 1)
                positive = normalized > 0.0
                axis.loglog(
                    indices[positive],
                    normalized[positive],
                    marker="o",
                    markersize=3.5,
                    linewidth=1.25,
                    color=color,
                    label=label,
                )
            axis.axhline(cutoff, color="0.35", linestyle="--", linewidth=1.0)
            axis.set_title(f"TT cut {cut + 1}: {CUT_LABELS[cut]}")
            axis.set_xlabel("singular-value index")
            axis.set_xlim(0.85, max(1.15, 1.12 * len(coefficient_spectra[cut])))
            axis.set_ylim(1.0e-16, 2.0)
            axis.grid(True, which="both", alpha=0.25)
        axes[0].set_ylabel(r"normalized singular value $\sigma_i/\sigma_1$")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            figure.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2
            )
        figure.suptitle(f"{variable.replace('_', ' ')} metric comparison", y=0.99)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.80))
        figure.savefig(
            output_directory / f"vlasov_spectrum_{variable}_metric_comparison.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_spectra(states, output_directory, cutoff, normalize, metric_label):
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "vlasov_spectra.csv"
    colors = {
        "projected exact": "tab:blue",
        "projected exact (zero)": "tab:blue",
        "projected-induced error": "tab:orange",
        "direct discrete": "tab:green",
        "AMEn discrete": "tab:green",
        "direct - projected": "tab:red",
        "AMEn - projected": "tab:red",
    }

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ("variable", "state", "cut", "index", "singular_value", "normalized_value")
        )

        for variable, variable_states in states.items():
            variable_spectra = {
                state_name: tt_spectra(tensor)
                for state_name, tensor in variable_states.items()
            }
            figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.7), sharey=True)
            zero_states = []
            for state_name, tensor in variable_states.items():
                for cut, (axis, singular_values) in enumerate(
                    zip(axes, variable_spectra[state_name])
                ):
                    leading = singular_values[0] if singular_values.size else 0.0
                    normalized = (
                        singular_values / leading
                        if leading > 0.0
                        else np.zeros_like(singular_values)
                    )
                    for index, (value, normalized_value) in enumerate(
                        zip(singular_values, normalized), start=1
                    ):
                        writer.writerow(
                            (
                                variable,
                                state_name,
                                cut + 1,
                                index,
                                f"{value:.17e}",
                                f"{normalized_value:.17e}",
                            )
                        )
                    if leading == 0.0:
                        zero_states.append(state_name)
                        continue
                    indices = np.arange(1, singular_values.size + 1)
                    plotted_values = normalized if normalize else singular_values
                    positive = plotted_values > 0.0
                    axis.loglog(
                        indices[positive],
                        plotted_values[positive],
                        marker="o",
                        markersize=3.5,
                        linewidth=1.25,
                        color=colors[state_name],
                        label=state_name,
                    )

            for cut, axis in enumerate(axes):
                if normalize:
                    axis.axhline(cutoff, color="0.35", linestyle="--", linewidth=1.0)
                axis.set_title(f"TT cut {cut + 1}: {CUT_LABELS[cut]}")
                axis.set_xlabel("singular-value index")
                maximum_index = max(
                    len(spectra[cut]) for spectra in variable_spectra.values()
                )
                axis.set_xlim(0.85, max(1.15, 1.12 * maximum_index))
                if normalize:
                    axis.set_ylim(1.0e-16, 2.0)
                axis.grid(True, which="both", alpha=0.25)
            axes[0].set_ylabel(
                r"normalized singular value $\sigma_i/\sigma_1$"
                if normalize
                else r"singular value $\sigma_i$"
            )
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                figure.legend(
                    handles,
                    labels,
                    loc="upper center",
                    bbox_to_anchor=(0.5, 0.91),
                    ncol=len(handles),
                )
            if zero_states:
                figure.text(
                    0.5,
                    0.02,
                    ", ".join(sorted(set(zero_states))) + " is identically zero",
                    ha="center",
                    fontsize=9,
                )
            scale_label = "normalized" if normalize else "unnormalized"
            figure.suptitle(
                f"{variable.replace('_', ' ')} ({metric_label}, {scale_label})",
                y=0.99,
            )
            figure.tight_layout(rect=(0.0, 0.06, 1.0, 0.80))
            suffix = "" if normalize else "_raw"
            figure.savefig(
                output_directory / f"vlasov_spectrum_{variable}{suffix}.png",
                dpi=180,
                bbox_inches="tight",
            )
            plt.close(figure)
    return csv_path


def print_rank_summary(states, cutoff):
    for variable, variable_states in states.items():
        print(variable)
        for state_name, tensor in variable_states.items():
            spectra = tt_spectra(tensor)
            ranks = tuple(
                int(np.count_nonzero(values > cutoff * values[0])) if values[0] else 0
                for values in spectra
            )
            print(f"  {state_name:24s} ranks={ranks}")


def assemble_saved_states(projected_solution_file, discrete_solution_file):
    with h5py.File(projected_solution_file, "r") as projected_h5, h5py.File(
        discrete_solution_file, "r"
    ) as discrete_h5:
        projected = [dense_vector(projected_h5, f"soln_{block}") for block in range(5)]
        discrete = [dense_vector(discrete_h5, f"soln_{block}") for block in range(5)]

    states = {
        "error_test": {
            "projected exact (zero)": projected[0],
            "AMEn discrete": discrete[0],
        }
    }
    for block, name in enumerate(BLOCK_NAMES[1:], start=1):
        states[name] = {
            "projected exact": projected[block],
            "AMEn discrete": discrete[block],
            "AMEn - projected": discrete[block] - projected[block],
        }
    return states


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefix",
        default="vlasov_system_nx_2_nt_2_nv_2_p_3_dp_2_L2_norm_",
        help="HDF5 filename prefix, including the trailing underscore",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("vlasov_spectra"), help="plot directory"
    )
    parser.add_argument(
        "--discrete-solution",
        type=Path,
        help="saved full AMEn solution; skips the dense Schur solve when provided",
    )
    parser.add_argument(
        "--mass-prefix",
        help="prefix for mass_test.h5, mass_trace_x.h5, ..., enabling FEM weighting",
    )
    parser.add_argument("--cutoff", type=float, default=1.0e-10)
    args = parser.parse_args()

    if args.discrete_solution:
        states = assemble_saved_states(
            args.prefix + "soln.h5", args.discrete_solution
        )
        relative_residual = None
    else:
        states, relative_residual = assemble_states(args.prefix)
    metric_label = "coefficient"
    if args.mass_prefix:
        coefficient_states = states
        states = mass_weight_states(coefficient_states, args.mass_prefix)
        plot_metric_comparison(
            coefficient_states, states, args.output_dir, args.cutoff
        )
        metric_label = "FEM L2 weighted"
    csv_path = plot_spectra(
        states, args.output_dir, args.cutoff, normalize=True, metric_label=metric_label
    )
    plot_spectra(
        states, args.output_dir, args.cutoff, normalize=False, metric_label=metric_label
    )
    if relative_residual is not None:
        print(f"direct_relative_schur_residual={relative_residual:.6e}")
    print_rank_summary(states, args.cutoff)
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
