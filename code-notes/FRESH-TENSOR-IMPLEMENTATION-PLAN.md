# Fresh Tensor Implementation Plan

## Purpose

This plan replaces the previous ordering for the tensor-native work.  The immediate objective is not to optimize the existing dense local DPG Schur path.  The immediate objective is to prove, with a small and controlled model problem, that the code can perform true tensor-product matrix-free action and sum-factorized assembly at the expected asymptotic cost.  Only after those kernels are correct and fast should they be connected to the DPG Vlasov formulation.

The first target is standard Bubnov-Galerkin Poisson on tensor-product cells.  This is intentionally simpler than the DPG Vlasov operator: it exercises HGRAD basis values and gradients, cell-local tensor contractions, quadrature-point coefficient/metric multiplication, matrix-free apply, and matrix assembly, without introducing trial/test-space asymmetry, traces, graph norms, or Schur complements.  The implementation should be built directly on the shelved `Intrepid2::PAMatrix` implementation and its partially assembled apply/assemble APIs, rather than on the current dense local `TensorLocalSchurComplementBuilder` hot path.

## Non-negotiable constraints

1. **True matrix-free means no dense element matrix in the apply path.**  The Poisson matrix-free apply must perform coefficient-to-quadrature contractions, point-data multiplication, and quadrature-to-coefficient contractions.  It must not cache or apply dense element matrices.
2. **Use PAMatrix as the kernel substrate.**  Reuse `PAMatrix::apply()`, `PAMatrix::assemble()`, `extractColumn()`, `extractDiagonal()`, workspace allocation, and the existing transformed-basis/point-data model where possible.  New code should wrap or extend this implementation, not duplicate an independent tensor contraction system unless a missing capability is identified and documented.
3. **Express parallelism through Kokkos-compatible data paths.**  Setup may use STL and Teuchos.  Kernel data must be in Kokkos views or PAMatrix/Intrepid2 containers.  Avoid accumulation into `std::vector<double>` in performance paths.
4. **Separate timings.**  Timers must distinguish basis/cache construction, geometry/Jacobian/cell-measure construction, coefficient or point-data evaluation, matrix-free apply, sum-factorized local assembly, CRS scatter/fill, solver/preconditioner setup, and pure solver/preconditioner apply.
5. **Keep dense/CRS fallback guarded.**  Any path with intentionally bad asymptotic scaling is allowed only for tiny regression tests.  Add threshold checks or explicit command-line opt-in so that performance tests cannot accidentally spend hours in an assembled dense path.
6. **Dimension must be runtime-selectable up to 7.**  The demonstration driver must support tensor dimension `d = 1,...,7`, with the understanding that high-dimensional assembled matrices become memory-limited and should be guarded.

## Phase 1: use the PAMatrix-enabled Trilinos fork directly

### Goals

- Build the tensor experiment against the installed Trilinos fork that already contains `Intrepid2::PAMatrix`, its dependent Intrepid2 tensor-data classes, and the PAMatrix assembly/apply APIs.
- Avoid vendoring or adapting PAMatrix into Camellia.  Any local code in Camellia should be a narrow test/driver wrapper around the fork API, not a copy of PAMatrix internals.
- Preserve the ability to compare against standard Intrepid2 integration and existing Camellia assembly on small problems.

### Tasks

1. Point the experimental tensor test/driver targets at the PAMatrix-enabled Intrepid2 source/include directories from the fork, using those headers before older installed Intrepid2 headers when necessary.
2. Use the fork API directly: `Intrepid2::PAMatrix`, `BasisValues`, `TensorData`, `TransformedBasisValues`, `apply()`, `assemble()`, column/diagonal extraction, and PAMatrix workspace allocation.
3. Keep any Camellia-side helper minimal and temporary: it may construct simple tensor-product basis/value containers for tests, but it must not duplicate the PAMatrix contraction implementation.
4. Convert the first smoke test from the old independent synthetic tensor operator to real `Intrepid2::PAMatrix` objects using tensor-product 1D value/derivative matrices and cell measures.

### Acceptance criteria

- The code builds in the configured OpenMP/release build against the PAMatrix-enabled fork.
- A unit test constructs real `Intrepid2::PAMatrix` instances, applies them, assembles them, and compares apply-vs-assembled matvec to near roundoff.
- The plan and tests no longer call for vendoring/adapting PAMatrix as the next step.

## Phase 2: matrix-free Bubnov-Galerkin Poisson driver

### Operator

Implement scalar HGRAD Poisson on tensor-product cells:

- mass term option: `(u, v)` for baseline value-value testing;
- stiffness term: `(grad u, grad v)`;
- affine tensor-product geometry first;
- optional scalar coefficient `a(x)` evaluated at quadrature points after trial coefficients have been converted to quadrature values/gradients.

The matrix-free stiffness apply should be represented as the sum over derivative components:

`A x = sum_k B_k^T W_k B_k x`,

where `B_k` maps HGRAD coefficients to the `k`th physical gradient component at tensor quadrature points, `W_k` contains quadrature weights, metric factors, Jacobian determinant factors, and optional coefficient values, and `B_k^T` maps quadrature data back to basis coefficients.  This is exactly the shape already targeted by PAMatrix.

### Driver requirements

Create a driver such as `drivers/performance/TensorPoissonPAMatrixDriver.cpp` with options:

- `--dim d`, required range 1 through 7;
- `--polyOrder pMin:pMax` or repeated p sweep;
- `--cellsPerDim n` or per-dimension cell counts;
- `--operator mass|stiffness|mass-plus-stiffness`;
- `--mode apply|assemble|compare`;
- `--numApply N` for amortized apply timing;
- `--worksetSize` for PAMatrix apply;
- `--seed` for deterministic input vectors;
- `--maxDenseDofs` / `--maxAssembledEntries` guard thresholds.

### Timings and counters

Log one structured line per run with at least:

- dimension `d`;
- polynomial order `p`;
- cells and local/global DOFs;
- quadrature points per direction;
- setup/basis time;
- geometry/cell-measure time;
- PAMatrix construction or point-data construction time;
- pure matrix-free apply time averaged over `N` applies;
- estimated flops for apply;
- achieved rate;
- peak or major allocated bytes where available.

### Complexity target

For fixed cell count and tensor dimension `d`, local matrix-free apply must scale like `O(p^(d+1))` per cell for each value/gradient component pass.  The driver should generate p-sweep tables and a fitted log-log slope.  Acceptance requires observed slopes close to `d+1` after setup costs are excluded.  The report should explicitly separate first-apply warmup from steady-state apply timing.

### Correctness tests

- Compare PAMatrix matrix-free apply against assembled PAMatrix local matrices for small `d,p`.
- Compare mass and stiffness against standard Intrepid2 integration for `d <= 3` where convenient.
- Verify symmetry and positive definiteness for the assembled small Poisson matrix.

## Phase 3: sum-factorized assembly of Poisson

### Algorithm

Use the same PAMatrix representation, but call the sum-factorized assembly path to fill local element matrices.  For an arbitrary basis vector, matrix-free action costs `O(p^(d+1))`.  Forming all `O(p^d)` columns by applying the matrix to canonical unit vectors yields `O(p^(2d+1))`, matching the standard sum-factorized assembly upper-bound complexity and avoiding the dense `test x trial x quadrature` loop of cost `O(p^(3d))`.

PAMatrix's native `assemble()` should be interpreted against the same standard Intrepid2 sum-factorized target: `O(p^(2d+1))` in the generic case, with possible `O(p^(2d))` behavior for separable/specialized cases.  Dense `O(p^(3d))` is only a guarded fallback or baseline diagnostic, not the expected slope for standard `assemble()`.  If an inspected fork implementation still routes through a placeholder or generic integration path, label that as an implementation-path diagnostic while keeping the theoretical expectation separate.

### Driver mode

Extend the Poisson driver with `--mode assemble` and `--mode compare`:

- local sum-factorized assembly only;
- optional CRS scatter for small-to-medium cases;
- explicit dense legacy assembly only below a strict threshold;
- p-sweep timing and slope fitting.

### Acceptance criteria

- Demonstrate `assemble2()` column-apply proxy timing slopes close to `2d+1` for feasible dimensions/orders.
- Demonstrate standard `assemble()` slopes consistent with Intrepid2's expected sum-factorized upper bound `2d+1`, or with `2d` for separable-specialized cases; do not use dense `3d` as the expected standard-assemble slope.
- Demonstrate that matrix-free apply and assembled local/CRS apply agree for small cases.
- Demonstrate that the dense quadrature fallback is disabled by default above the configured threshold and explicitly labeled when measured.

## Phase 4: H1 Gram matrix assembly only

After Poisson apply and assembly are demonstrably correct and performant, move to the DPG-relevant H1 Gram matrix.  This phase is about `G` only, not `B`, not `K = B^T G^{-1} B`, and not Vlasov terms.

### Operator

Assemble local discontinuous-test-space Gram blocks for an H1-like norm, e.g.

`G(v,w) = alpha (v,w) + beta sum_k (d_k v, d_k w)`

with affine tensor-product geometry and scalar weights first.  The local test space is discontinuous, so global coupling is cell-local except for any later optional diagnostic scatter.

### Implementation

- Reuse PAMatrix value-value and grad-grad constructions.
- Start with the scalar H1 Gram case as a weighted mass-plus-stiffness operator, exposed through the performance driver as `--operator=h1-gram` with configurable `--massWeight=alpha` and `--stiffnessWeight=beta` so Phase 4 measurements are labeled separately from the Poisson model problem while sharing the already-validated PAMatrix construction path.  In HGRAD mode, split the gradient contribution into one PAMatrix term per physical component so value/mass and gradient-component assembly timings are visible separately.
- Assemble each local Gram block by sum factorization.
- Store local Gram matrices only in this assembled-backend phase.
- Use optimized dense linear algebra, e.g. LAPACK Cholesky, when a local factorization is intentionally required by the sum-factorized assembled backend.
- Do not form global dense G.
- Do not form local `G^{-1}` explicitly unless only in a tiny diagnostic test below threshold; factor and solve instead.

### Baseline comparison

Compare against the current baseline Gram assembly for small and moderate local orders.  The acceptance target is to beat the baseline Gram assembly time while matching entries to approximately `1e-10` in a sensible relative/absolute norm.

### Timings

Report separately:

- basis/cache construction;
- point-data/cell-measure construction;
- value/mass local assembly;
- gradient component local assembly;
- local Gram accumulation;
- optional Cholesky factorization;
- optional CRS scatter.

## Phase 5: matrix-free H1 Gram operator and Chebyshev inverse recovery

### Matrix-free Gram operator

Wrap the H1 Gram matrix-free apply in a `Tpetra::Operator` or `RowMatrix`-compatible adapter.  The apply path must use PAMatrix contractions and must not use assembled local Gram matrices.

Required methods/data:

- domain/range maps;
- `apply()` for vectors and, if practical, multivectors;
- diagonal extraction or a reliable diagonal estimate suitable for Ifpack2;
- timing hooks for pure apply;
- optional local/block view for correctness tests only.

### Chebyshev inverse approximation

Use Ifpack2 Chebyshev with the matrix-free Gram operator to approximate `G^{-1} r`.  The point of this phase is to demonstrate that the inverse action can be recovered without forming local `G^{-1}` in the matrix-free backend.

Validation procedure:

1. On small cases, assemble and factor the local Gram matrix as a reference.
2. Apply the exact assembled/factored inverse to random right-hand sides.
3. Apply Chebyshev to the matrix-free Gram operator with the same right-hand sides.
4. Measure relative error versus polynomial degree, estimated spectral bounds, and number of iterations.
5. Confirm that increasing Chebyshev degree/iterations reduces the inverse-action error to the expected tolerance.

Acceptance criteria:

- The matrix-free Gram apply agrees with assembled Gram apply on small cases.
- Chebyshev recovers the assembled inverse action to documented tolerances without forming local inverse matrices in the matrix-free path.
- Timings separate Chebyshev setup, diagonal/spectral estimate, and repeated inverse applications.

## Phase 6: integrate only the useful existing tensor API pieces

Only after the Poisson and H1 Gram kernels are working should the code be connected back into the existing tensor API.  The current metadata pieces are useful; most current floating-point implementation kernels should be considered provisional.

Likely reusable pieces:

- `TensorMeshTopology`, `ComponentMeshTopology1D`, and tensor cell enumeration as setup metadata;
- `TensorDofLayout` shape metadata after mixed-radix/device-ready cleanup;
- `TensorOperatorData` local-to-global views and side/parity metadata;
- `TensorFunction` coefficient classification metadata;
- existing correctness tests that compare tensor metadata and small local matrices.

Likely replacement pieces:

- dense local B/G/K construction in performance paths;
- any coupled coefficient path that falls back to dense basis-pair quadrature loops;
- matrix-free facades that are actually cached dense-cell Schur applies;
- construction of local `G^{-1}` in matrix-free paths.

Integration should proceed through narrow adapters: tensor metadata chooses basis, geometry, maps, and coefficient evaluators; PAMatrix-backed kernels perform the floating-point work.

## Phase 7: DPG Vlasov components, one term at a time

After the H1 Gram work is complete, begin DPG Vlasov work piecewise.  Do not begin with the full Schur complement or full Planar MITL driver.

For each component term:

1. Define the mathematical term and its trial/test spaces.
2. Classify coefficients: constant, single-coordinate, separable, or coupled.
3. Implement matrix-free action using trial-to-quadrature PAMatrix-style contractions, quadrature-point coefficient multiplication, and test-transpose contractions.
4. Implement sum-factorized assembly using the same kernels.
5. Compare against the baseline implementation on tiny tensor meshes.
6. Record timings and fitted p-scaling.

Initial term order:

1. time transport term;
2. spatial transport terms;
3. velocity transport terms with precomputed relativistic factors such as `u_x / gamma` and `u_y / gamma` evaluated at quadrature points;
4. acceleration-field terms with coupled coefficient values at quadrature points;
5. directional trace terms on tensor skeleton families.

The coupled coefficient case is the primary optimization target.  Separable coefficient shortcuts may be added later, but must not distract from the arbitrary quadrature-grid coefficient path.

## Phase 8: matrix-free DPG Schur action

Only after B/B-transpose pieces and the matrix-free Gram inverse approximation are validated should the DPG Schur action be assembled as an operator:

`x -> B x -> approximately solve G y = B x -> B^T y`.

Rules:

- no dense local B in the matrix-free backend;
- no local `G^{-1}` materialization;
- no full local dense Schur block;
- use Chebyshev or a later multigrid/Chebyshev approximation for the Gram inverse;
- preserve small assembled CRS versions only for regression tests and correctness comparisons.

## Phase 9: multigrid and full performance targets

Once the matrix-free Schur action is correct, add solver infrastructure:

- p-multigrid hierarchy using cached interpolation/restriction operators;
- Chebyshev smoothing;
- possible h-multigrid later, likely requiring a new Camellia-specific hierarchy or MueLu integration;
- reuse `GMGOperator` where useful for the sum-factorized assembled operator path.

Only after this point should full planar MITL performance runs be attempted.  Full runs must include phase-separated timings and must be excluded from the normal build/test loop unless they complete within the agreed budget.

## Regression and performance policy

- Unit tests must stay small and should complete in the normal build/test loop in about a minute total for the new tensor suite.
- Performance drivers may contain larger cases but must be opt-in.
- Any O(p^(3d)) dense fallback must have a threshold guard and an explicit diagnostic label.
- Every benchmark name must say whether it measures setup, assembly, apply, preconditioner setup, preconditioner apply, or a combined end-to-end solve.

## Immediate next implementation slice

### Completed in the current direct-fork slice

1. Updated the tensor test target to use the PAMatrix-enabled Intrepid2 fork source include directories (`Discretization/Integration`, `Discretization/Basis`, and `Shared`) before any older installed headers.
2. Replaced the self-contained synthetic tensor operator smoke harness with tests that construct real `Intrepid2::PAMatrix` instances from `BasisValues`, `TensorData`, `TransformedBasisValues`, and tensor-product 1D operator views.
3. Added small direct-PAMatrix checks for apply-vs-assembled action, rectangular left/right spaces, apply-vs-independent explicit tensor-product reference action, `assemble2()` vs `assemble()`, diagonal extraction vs assembled diagonal, and a tensor-contraction work-estimate guard.  The Camellia-side additions header now supplies `allocateDiagonalStorage()`, `assemble2()`, and `extractDiagonal()` definitions for the current fork build.
4. Replaced the initial diagonal-extraction fallback with a PDF-style sum-factorized diagonal contraction for unoriented, scalar-point-data, matching square tensor-product component blocks.  Unsupported cases retain the correctness-oriented `assemble2()`-then-copy fallback.  The direct path is tested against assembled local matrices on square and rectangular smoke cases.
5. Added an opt-in `TensorPoissonPAMatrixDriver` prototype under `drivers/performance`.  It constructs direct fork `Intrepid2::PAMatrix` tensor-product mass and derivative-component stiffness terms for runtime dimensions 1 through 7; supports `apply`, guarded `assemble`, guarded `assemble2`, `diagonal`, and `compare` modes; separates setup, first/steady apply, standard assembly, column-apply assembly, and first/steady diagonal timings; and is compiled by the autotest path through the tensor test target dependency without running expensive sweeps.
6. Moved PAMatrix workspace allocation out of the driver's timed apply loop and avoided unconditional apply timing in non-apply benchmark modes, so diagonal and assembly runs no longer report unrelated apply timings as if they were part of the requested mode.
7. Added small true-HGRAD regression tests comparing PAMatrix local mass and stiffness assembly against standard Intrepid2 integration for affine Cartesian cells in 2D/3D.  The driver `compare` mode still compares matrix-free action to PAMatrix local assembly; the independent standard-integration comparison currently lives in the tensor unit tests.
8. Corrected `TensorPoissonPAMatrixDriver` slope telemetry so standard `assemble()` is compared against Intrepid2's expected sum-factorized `O(p^(2d+1))` upper bound, with separable `O(p^(2d))` reported where applicable; dense `O(p^(3d))` remains documented only as a guarded fallback/baseline diagnostic.
9. Started Phase 4 with a deliberately narrow `--operator=h1-gram` driver alias for scalar H1 Gram blocks.  This reuses the validated HGRAD value-value plus grad-grad PAMatrix construction, includes configurable `--massWeight=alpha` and `--stiffnessWeight=beta` coefficients, splits HGRAD gradient contributions into one term per physical component, reports term weights plus term-separated mass/stiffness local assembly timings and slopes in structured output, and provides smoke coverage without introducing DPG trial/test asymmetry, Gram inversion, or CRS/global assembly.
10. Added an opt-in `--mode=factor` diagnostic for H1 Gram blocks.  It assembles the weighted local Gram matrix, now uses a Teuchos/LAPACK Cholesky factor/solve path appropriate for SPD H1 Gram blocks, and reports `factorSeconds`, `avgSolveSeconds`, and solve residuals separately from assembly timings without forming `G^{-1}`.  Residuals above the prototype tolerance are emitted as structured nonfatal warnings so p-sweeps still archive the failing order's timings and residuals.
11. Added opt-in H1 Gram p-sweep targets for `assemble2`, standard `assemble`, and local assembly/factor/solve timing.  These targets are intentionally not part of default CTest; they provide reproducible commands for slope collection while respecting the guarded assembled-entry thresholds.
12. Added a small 3D HGRAD regression test comparing the weighted, component-split PAMatrix H1 Gram assembly against standard Intrepid2 integration, covering the Phase 4 baseline-comparison path before moving to larger sweeps.
13. Added opt-in synthetic 4D H1 Gram assembly sweep targets for standard `assemble()`, column-apply `assemble2()`, and same-process `assembly-compare` timing.  These are labeled synthetic contraction-kernel stress tests only; true 4D HGRAD/Vlasov geometry and basis support remains a separate fork-side milestone.
14. Added a guarded `assemble2()` fast path for orientation-free scalar tensor-product component integrations.  The path keeps the blocked-column public behavior but evaluates the canonical unit trial columns directly at tensor quadrature points, avoiding the dense all-but-one-zero trial-side coefficient-to-point GEMM sequence; point-data multiplication, test-side point-to-coefficient GEMMs, and unsupported-case fallback remain unchanged.  Added both synthetic multi-column-block and true 3D HGRAD mass/H1 Gram assemble2-vs-standard regression coverage for this path.
15. Added explicit `assemble2Path` telemetry to the performance driver and factored the `assemble2()` path-selection predicate into a shared diagnostic helper.  The first post-diagnostic synthetic 4D `assembly-compare` rerun reported `assemble2Path=unit_column_fast_path` for every archived row, so the p=4:8 timing comparison is now known to exercise the intended sparse unit-column path rather than the blocked `apply()` fallback.
16. Added an opt-in `--checkAssemblyAgreement=1` driver diagnostic for `assembly-compare` mode.  It separately assembles each term with standard `assemble()` and `assemble2()`, emits `ASSEMBLY_AGREEMENT` plus per-row max absolute/relative error columns, and is covered by a small H1 Gram CTest smoke so future timing sweeps can include correctness evidence when the assembled-entry guard allows it.
17. Added a 3D HGRAD H1 Gram `runTensorH1GramPAMatrixAssemblyCompareSweep` target and enabled `--checkAssemblyAgreement=1` on the synthetic 4D `runTensorH1GramPAMatrixSynthetic4DAssemblyCompareSweep` target.  These opt-in sweeps now archive standard-vs-`assemble2()` timings, `assemble2Path`, and entrywise agreement in a single run when the assembled-entry guard allows it.
18. Started Phase 5 with a deliberately local `--mode=inverse-diagnostic` before adding a Tpetra/Ifpack2 adapter.  This diagnostic assembles and factors local H1 Gram blocks only as a labeled reference, then applies a damped diagonal-preconditioned Richardson inverse approximation whose operator action is the matrix-free weighted PAMatrix `apply()` path and whose diagonal comes from `PAMatrix::extractDiagonal()`.  It reports initial/final inverse error, average inverse-iteration apply time, iteration count, a reference Gershgorin upper estimate for the diagonally preconditioned operator, and path labels so the future Chebyshev/Tpetra work starts from a validated local inverse-action harness.

### Recent p-sweep observations

- The archived 3D HGRAD H1 Gram standard `assemble()` sweep for `p=8:12` completed successfully.  It reported total assembly times from about 0.116 s at `p=8` to 1.796 s at `p=12`, with fitted total slope 7.55, mass slope 6.41, and stiffness slope 8.09.  These are broadly consistent with the expected standard sum-factorized upper-bound slope `2d+1 = 7`, with term-level noise and stiffness costs visible.
- The archived 3D HGRAD H1 Gram `assemble2()` sweep for `p=8:12` completed successfully.  It reported total column-apply assembly times from about 0.283 s at `p=8` to 2.444 s at `p=12`, with fitted total slope 6.13, mass slope 5.89, and stiffness slope 6.19 against the expected `2d+1 = 7` column-apply proxy model.
- In those archived 3D data, `assemble2()` is slower than standard `assemble()` but not catastrophically so: at `p=12`, 2.444 s versus 1.796 s in the shelved output.  This supports treating the sparse unit-column optimization as a constant-factor improvement target rather than as a correctness blocker.
- The archived factor sweep failed after the successful `p=3` line.  The emitted `p=3` residual was near roundoff (`maxSolveResidual` and relative residual about `7.35e-14`); the old hard failure therefore hid the failing higher-order line.  The factor diagnostic now uses Cholesky for SPD Gram blocks and emits residual-excess diagnostics nonfatally so future sweeps preserve timing/residual evidence for every attempted p.
- The archived synthetic 4D H1 Gram sweeps for `p=4:8` completed successfully.  In same-process `assembly-compare`, standard `assemble()` ranged from about 0.043 s to 6.251 s and `assemble2()` from about 0.048 s to 4.130 s.  The fitted slopes were 8.49 for standard assembly and 7.47 for `assemble2()` against the synthetic 4D expectations (`2d+1 = 9`, separable `2d = 8`).  The reported `assemble2OverAssembleRatio` dropped below 1 from `p=5` onward and was about 0.66 at `p=8`, so the optimized `assemble2()` path is now faster than standard assembly on the larger synthetic 4D cases in the archived sweep.

### Next work

1. Collect meaningful performance sweeps with enough cells/repetitions to amortize Kokkos launch overhead.  The tiny smoke p-sweeps are correctness/CI guards only and should not be used to judge asymptotic slopes.
2. Extend the direct diagonal path beyond the current scalar-point-data/matching-block subset as needed for vector-valued transforms and orientations; keep fallback timings explicitly labeled.
3. Verify `O(p^(2d+1))` local assembly timing using `assemble2()` as an interim column-apply proxy, and verify standard `assemble()` against the Intrepid2 sum-factorized expectation (`2d+1` upper bound, optionally `2d` in separable cases).  If the active fork implementation path is still a placeholder or unexpectedly dense, keep that label diagnostic and track it separately from the theoretical standard-assemble expectation.
4. Use the newly archived H1 Gram p-sweeps as baseline evidence: the 3D HGRAD standard `assemble()` and `assemble2()` sweeps look reasonable, while the old factor sweep failure was in the temporary driver-local dense solve diagnostic rather than the assembly kernels.  Re-run the factor sweep after the Cholesky update and keep residual warnings in the archived output.
5. Use the archived synthetic 4D H1 Gram assembly targets as Vlasov-relevant contraction stress tests: the same-process `assembly-compare` target now shows `assemble2()` overtaking standard `assemble()` by `p=5` and reaching about 0.66x standard assembly time at `p=8`.  Do not interpret these synthetic 4D timings as validated physical HGRAD/Vlasov results; they stress the high-dimensional PAMatrix contraction and assembly paths while true 4D HGRAD support is pending.
6. Re-run the 3D HGRAD and synthetic 4D assembly-compare sweeps after the agreement-check target updates so archived results contain both path-selection and assemble-vs-assemble2 correctness evidence; use those data to decide whether further `assemble2()` work should target constants, slope, or unsupported path coverage.
7. Run the new `runTensorH1GramPAMatrixInverseDiagnosticSweep` target and inspect whether the damped diagonal-preconditioned Richardson prototype decreases inverse-action error consistently enough to justify replacing the oracle-bound prototype with a Chebyshev recurrence and non-reference spectral estimates.
8. Add true 4D HGRAD support as a separate fork/driver milestone before claiming Vlasov-realistic 4D H1 Gram timings.  The current HGRAD construction path remains intentionally limited to dimensions 1 through 3 in the driver.

This slice is complete only when the driver demonstrates the advertised apply, diagonal, and assembly complexity with construction costs excluded from pure kernel timings, or when any implementation-path mismatch is explicitly labeled with a follow-up milestone.
