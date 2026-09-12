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

## Phase 1: vendor and isolate the PAMatrix dependency

### Goals

- Make the PAMatrix code available to the Camellia/tensor experiment in a narrow integration layer.
- Preserve the ability to compare against standard Intrepid2 integration and existing Camellia assembly on small problems.

### Tasks

1. Add or verify availability of the shelved Intrepid2 files needed by `PAMatrix`, including `Intrepid2_PAMatrix.hpp`, `Intrepid2_PAMatrixDef.hpp`, and the assembly helper patterns in `PAMatrixAssembly.hpp`.
2. Provide a local wrapper namespace or adapter if needed, but do not rewrite PAMatrix internals initially.
3. Add CMake options for the experimental tensor/PAMatrix drivers and tests.
4. Build a tiny synthetic PAMatrix smoke test using tensor-product 1D value/derivative matrices and cell measures.

### Acceptance criteria

- The code builds in the configured OpenMP/release build.
- A unit test constructs a synthetic PAMatrix, applies it, assembles it, and compares apply-vs-assembled matvec to near roundoff.

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

Use the same PAMatrix representation, but call the sum-factorized assembly path to fill local element matrices.  For an arbitrary basis vector, matrix-free action costs `O(p^(d+1))`.  Forming all `O(p^d)` columns by applying the matrix to canonical unit vectors yields `O(p^(2d+1))`, matching optimal sum-factorized assembly complexity and avoiding the dense `test x trial x quadrature` loop of cost `O(p^(3d))`.

Prefer PAMatrix's native `assemble()` when it implements the same structured algorithm.  If the implementation path uses column extraction internally, ensure it does not introduce dense quadrature loops.

### Driver mode

Extend the Poisson driver with `--mode assemble` and `--mode compare`:

- local sum-factorized assembly only;
- optional CRS scatter for small-to-medium cases;
- explicit dense legacy assembly only below a strict threshold;
- p-sweep timing and slope fitting.

### Acceptance criteria

- Demonstrate local assembly timing slopes close to `2d+1` for feasible dimensions/orders.
- Demonstrate that matrix-free apply and assembled local/CRS apply agree for small cases.
- Demonstrate that the dense quadrature fallback is disabled by default above the configured threshold.

## Phase 4: H1 Gram matrix assembly only

After Poisson apply and assembly are demonstrably correct and performant, move to the DPG-relevant H1 Gram matrix.  This phase is about `G` only, not `B`, not `K = B^T G^{-1} B`, and not Vlasov terms.

### Operator

Assemble local discontinuous-test-space Gram blocks for an H1-like norm, e.g.

`G(v,w) = alpha (v,w) + beta sum_k (d_k v, d_k w)`

with affine tensor-product geometry and scalar weights first.  The local test space is discontinuous, so global coupling is cell-local except for any later optional diagnostic scatter.

### Implementation

- Reuse PAMatrix value-value and grad-grad constructions.
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

1. Add the PAMatrix smoke test and driver CMake wiring.
2. Implement `TensorPoissonPAMatrixDriver` for matrix-free mass and stiffness apply.
3. Add p-sweep timing output and slope fitting for dimensions 1 through 7.
4. Add small correctness comparisons against assembled PAMatrix and standard integration.
5. Add sum-factorized local assembly mode and verify `O(p^(2d+1))` timing.

This slice is complete only when the driver demonstrates the advertised apply and assembly complexity with construction costs excluded from pure apply timings.
