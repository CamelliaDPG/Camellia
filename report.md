# VlasovAssemblyBobaDriver: rank / residual notes

## Build + run

Built in `build/vlasov_driver_scratch` (Release) and ran the binary via:

`./build/vlasov_driver_scratch/drivers/pulse/DpgAssemblyBoba/VlasovAssemblyBobaDriver ...`

Note: the driver is MPI-linked; running it directly (no `mpirun`) works, though OpenMPI/UCX may emit warnings.

## Key observation: `--schur_method=exact` is not a direct solve

In `drivers/pulse/DpgAssemblyBoba/schur_solver.hpp`, `schur_method=exact` sets `lambda_solution = exact_lambda` (manufactured/known solution) and then reports the residual of that exact solution. It does **not** solve the Schur system directly. So it cannot be used as a “best achievable residual” baseline.

## Parameter snapshots (L2 Schur solve)

All runs below used `--test_space_norm=L2 --solve_schur` and report:

- `relative_schur_residual`, `relative_full_residual`
- field block compression + TT ranks (block(4) in “Full solution diagnostics”)

Compression below means BoBa’s `compression_rate()` as printed by the driver (`compression=`), i.e. `full_size / stored`.

| nx nt nv | p dp | schur_method | svd tol (rel=abs) | relative_schur_residual | relative_full_residual | field compression | field ranks |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 2 2 | 3 2 | block_amen | 1e-12 | ~1e-8 | ~1e-8 | 0.75  | ( 1, 6, 6, 1 ) |
| 2 2 2 | 3 0 | block_amen | 1e-12 | ~1e-8 | ~1e-8 | 0.75  | ( 1, 6, 6, 1 ) |
| 2 2 2 | 2 0 | block_amen | 1e-12 | ~1e-8 | ~1e-8 | 0.667 | ( 1, 4, 4, 1 ) |
| 2 2 2 | 1 0 | block_amen | 1e-12 | ~1e-9 | ~1e-8 | 0.50  | ( 1, 2, 2, 1 ) |
| 2 2 2 | 3 2 | block_amen | 1e-6  | ~6.5e-4 | ~2.6e-2 | 0.75 | ( 1, 6, 6, 1 ) |
| 2 2 2 | 3 2 | block_amen | 1e-4  | ~1.6e-1 | ~6.6e+0 | 1.24 | ( 1, 4, 5, 1 ) |

Notes:

- For the default discretization (2×2×2, p=3,dp=2), AMEn converges to a very small Schur residual, but ranks are not low (field block rank ~6; some lambda blocks reach rank ~8).
- Lowering polynomial order reduces ranks substantially while keeping residuals small; e.g. p=1,dp=0 gives field ranks (1,2,2,1) with ~1e-8 residuals.
- Loosening SVD tolerances reduces ranks somewhat, but quickly degrades both Schur and full-system residuals.

## nx=nt=nv=16: what changes?

These runs are much larger Schur systems because the trace blocks include interior faces (e.g. trace_x has `(nx-1) * nt*(p+1) * nv*(p+1)` unknowns).

Key takeaways so far:

- Refinement + higher p makes the *rank-1* exact manufactured multipliers much closer to the discrete solution (very small `exact_residual_norm[0]`), but a strictly rank-1 Schur solve still hits a residual floor around ~1e-3–1e-2 depending on (p,dp).
- Relaxing SVD tolerances reduces ranks, but increases `relative_schur_residual` sharply (clear rank↔residual tradeoff).

Selected runs (`--test_space_norm=L2 --solve_schur`):

| nx nt nv | p dp | AMEn (max_sweeps,kick) | svd tol (rel=abs) | lambda compression | lambda ranks (after round) | relative_schur_residual | relative_full_residual |
|---:|---:|---:|---:|---:|---|---:|---:|
| 16 16 16 | 1 0 | (10,0) | 1e-8  | 176.05 | all blocks rank-1 | 7.68e-1 | 1.41e+1 |
| 16 16 16 | 3 2 | (50,0) + exact init | 1e-10 | 514.68 | all blocks rank-1 | 2.43e-3 | 1.69e-1 |
| 16 16 16 | 3 2 | (2,4) + exact init | 1e-10 | 17.21  | x:(1,8,9,1) t:(1,9,9,1) v:(1,9,7,1) f:(1,9,9,1) | 6.59e-4 | 4.58e-2 |
| 16 16 16 | 3 2 | (2,4) + exact init | 1e-8  | 33.50  | x:(1,6,7,1) t:(1,6,6,1) v:(1,7,4,1) f:(1,6,5,1) | 2.24e-3 | 1.56e-1 |
| 16 16 16 | 3 2 | (2,4) + exact init | 1e-6  | 205.51 | x:(1,1,2,1) t:(1,1,2,1) v:(1,2,2,1) f:(1,2,3,1) | 1.24e-2 | 8.65e-1 |

Interpretation: for this discretization, “rank 1–2 *and* residual ~1e-8” does not look achievable; lowering rank via truncation quickly lifts the residual.

## Runtime corner case

`--nx=1 --nt=1 --nv=1` produced zero-size trace blocks and then crashed with a floating point exception (divide by zero). For now I’m avoiding nx/nt/nv values that eliminate trace unknowns.

## Direct dense Schur solve baseline (for a “best achievable residual”)

The driver writes HDF5 dumps like:

- `vlasov_system_..._stiffness.h5`
- `vlasov_system_..._rhs.h5`

Using those, we can build a *true* dense Schur system (for the L2 path) as:

- `G = K(test,test)`
- `B = K(lambda,test)`
- `S = B * G^{-1} * B^T`
- `b = B * G^{-1} * f`, where `f = rhs(test)`

For the default case `nx=nt=nv=2, p=3, dp=2, --test_space_norm=L2`:

- The dense construction gives `||b|| = 2.3980102327`, matching the driver’s `schur_rhs_norm=2.398010233`.
- A direct dense solve achieves `relative_schur_residual ≈ 2.2e-12` (so there’s no mysterious residual floor at ~1e-8).
- Reconstructing the test variable via elimination, the corresponding *full* saddle-point residual is also tiny: `relative_full_residual ≈ 1.5e-10` (machine precision for this problem size).

Implementation note: BoBa’s HDF5 TT backend stores cores with a dimension permutation; when reading `*_data`/`*_sizes` with `h5py`, you need `core = data.reshape(tuple(sizes[::-1]))` to recover BoBa ordering ((l,i,j,r) for matrix cores, (l,i,r) for vector cores).

What I did / strategy:

1. Let the driver assemble and dump `stiffness.h5` + `rhs.h5` (these contain TT cores for every block).
2. In a small Python check (numpy + h5py), reconstruct *dense* `G` and each `B_i` block by contracting the TT cores.
3. Validate the HDF5 read + dense reconstruction by checking `||b||` matches the driver’s `schur_rhs_norm` to ~machine precision.
4. Solve the dense Schur system `S λ = b` directly and compute `||Sλ-b||/||b||`.
5. (Optional) Use the dense-direct `λ` as “truth” to study intrinsic rank via TT-SVD and to quantify how much residual grows if you truncate ranks.

This dense baseline is only feasible for the small cases (e.g. Schur size 408 for `nx=2,p=3`); it is not intended for `nx=16` (Schur size 294912).

## Rank vs residual reality check (default p=3,dp=2)

Even though the manufactured *continuous* solution is low rank, the *discrete* Schur solution for `p=3,dp=2` is not.

Why this happens (in this driver):

- The “exact solution” is separable in `(x,t,v)`, and the driver builds a rank-1 `exact_soln` by construction (outer products of 1D discretizations). But `exact_residual_norm[0]` is not zero for the discrete system, so the true *discrete* solution differs from that rank-1 ansatz.
- Solving a linear system generally destroys separability: even if the operator and RHS are low-rank TT objects, the inverse is not. Here the Schur operator includes `G^{-1}` and products like `B G^{-1} B^T`, which mix dimensions and typically require higher TT ranks in `λ` to represent accurately.

From TT-SVD rank estimates of the dense-direct Schur solution (rel_tol ~1e-12, order-3 per block):

- trace_x: ~8, trace_t: ~4, trace_v: ~4, field: ~6 (i.e., ranks comparable to what AMEn returns).
- Forcing ranks down near 1–2 by truncation makes the Schur residual large (e.g. field ranks ~2–3 gives `relative_schur_residual ~ 0.2`).

So, for “low-rank (1–2) *and* low residual”, lowering discretization order (e.g. `p=1,dp=0`) is effective; for `p=3,dp=2`, low residual requires higher ranks.
