# Tensor Mesh Implementation Plan

## Goal

Implement a performance-portable tensor-native path for the Planar MITL relativistic Vlasov problem that first supports fast sum-factorized assembly of the usual DPG Schur-complement matrix K = B G^-1 B^T, then supports matrix-free operator application suitable for Krylov solvers and p-multigrid with Chebyshev smoothing.

The immediate target is a 1D component tensor mesh: x, t, and velocity dimensions are all 1D factors.  The motivating driver is `PlanarMITLRelativisticSpaceTime.cpp` or a new driver that solves the same 1D1T2V problem using the new tensor facilities.

## Design principles

- Keep STL containers, strings, maps, and Teuchos objects in setup code only.
- Inner loops must operate on Kokkos views, integer ordinals, small fixed-size arrays where possible, and precomputed tables.
- Preserve tensor-product structure as long as possible.  Avoid materializing full 4D quadrature or dense element tensors except when explicitly assembling K blocks.
- Start with uniform or piecewise-uniform p and tensor-product grids; add hp adaptivity after the fast path is correct and benchmarked.
- Provide a correctness fallback against the legacy assembled operator for small meshes.
- Add instrumentation from the beginning: setup time, assembly time, apply time, solve time, memory high-water mark, FLOP/byte estimates, and iteration counts.

## Phase 0: establish baselines and acceptance tests

1. Add benchmark configurations for the existing Planar MITL path:
   - small correctness case that can be compared against legacy direct/iterative results;
   - medium case that currently runs slowly but completes;
   - target high-order 4D cases intended to demonstrate 10x to 100x speedup.
2. Record baseline metrics:
   - number of cells per component and total cells;
   - polynomial orders and test enrichment;
   - trial/test DOFs;
   - assembly time for B/G/K or legacy `Solution::solve()` setup;
   - solve iterations and solve time;
   - peak memory.
3. Add automated small tests that compare tensor-native local element matrices and global solutions against legacy integration to tight tolerances.

## Phase 1: compile TensorMesh metadata into device-ready data

Create a setup-only compiler from `TensorMesh`/`TensorDofLayout` to a compact runtime structure, tentatively `TensorOperatorData`.

Required data:

- component cell counts, vertex coordinates, cell widths/Jacobians, and side incidence as Kokkos views;
- active tensor cell tuples represented as integer component-cell ordinals, not vectors of `ComponentCellID1D`;
- per-component p order and basis cardinality tables;
- per-variable integer IDs replacing name lookups;
- local tensor DOF shapes and strides per variable/role/cell-type;
- global Tpetra maps and local-to-global gather/scatter views;
- boundary/skeleton tags needed for Vlasov inflow, initial-time, and velocity-boundary conditions.

Refactor/fix before using in kernels:

- Replace repeated linear `tensorCellOrdinal()` lookups with direct ordinal arithmetic for full tensor products, plus optional maps only for adaptive/non-product subsets.
- Fix or remove `ComponentMeshTopology1D::activeCellsForVertex()` before relying on it.
- Replace `TensorDofLayout` tuple enumeration in ordinal encode/decode with mixed-radix arithmetic.
- Ensure trace skeleton entities carry enough side/orientation/parity data for `t_n_hat` terms.

Deliverables:

- `TensorOperatorData` or equivalent header/source;
- unit tests for mapping, local/global DOF counts, boundary tags, side parity, and trace layouts;
- no performance-critical dependence on `std::vector` or `std::map` after setup.

Status update:

- Completed first Phase 1 metadata-hardening slice: `ComponentMeshTopology1D::activeCellsForVertex()` now exposes only the requested vertex incidence, full-product `TensorMeshTopology::tensorCellOrdinal()` uses direct mixed-radix component ordinals with an explicit-subset fallback, and `TensorDofLayout` cell-order encode/decode no longer allocates/enumerates all tensor-product DOF tuples.
- Added `TensorOperatorData` as the setup-compiled container for component geometry/p-order views, active tensor-cell component ordinals, boundary-side flags, and per-variable coefficient counts/ambient shapes.
- Completed second Phase 1 metadata slice: `TensorOperatorData` now also owns padded device-ready local-to-global index views for volume variables, trace-skeleton local-to-global index views for trace variables, and per-skeleton-entity side parity metadata for flux-like `t_n_hat` terms.  Phase 0 tests compare these views against the existing `TensorCoefficientMap`/`TensorCoefficientTraversal` helpers.
- Validation slice: Phase 0 tensor tests and the small Planar MITL baseline remain the acceptance gates; the medium baseline is intentionally not part of the default loop.

Next slice:

- Completed first Phase 2 basis-cache slice: added `TensorBasisCache1D` and `TensorBasisCacheND` to cache 1D quadrature points/weights, basis values, reference derivative values, affine Jacobian/derivative scaling, tensor-product point strides, physical points, and tensor-product quadrature weights.  Phase 0 tests cover 1D geometry/tabulation metadata and lower-component-fastest ND point ordering.
- Completed second Phase 2 integration slice: `MatrixFreeTensorOperator` now uses `TensorBasisCacheND` for affine volume-term basis/quadrature setup, preserving existing reference-result comparisons while removing duplicate volume quadrature/basis construction logic from the operator hot path.
- Completed first sum-factorized local volume slice: supported volume terms with constant or single-component coefficient dependencies are now applied through products of cached 1D component matrices, including value/value and single-derivative/value forms used by the 1X1T1V Vlasov-like tests.  Unsupported coupled coefficients continue to use the generic cached quadrature fallback.
- Completed reusable local-action slice: factored the tensor-product component-matrix application into `TensorProductLocalOperator`, so per-cell Kronecker-style actions can be tested and reused independently of global gather/scatter.  `MatrixFreeTensorOperator` now uses this helper in the specialized volume path.
- Completed separable-product coefficient slice: the specialized volume path now factors tensor-function products whose child dependencies are constant or single-component into per-component coefficient factors, while still falling back for coupled sums and solution-dependent coefficients.

Next slice:

- Completed Phase 0 tensor-operator instrumentation slice: `MatrixFreeTensorOperator` now accumulates per-apply stage counters/timings for basis-cache construction, coefficient evaluation, 1D contraction construction, local action, and scatter/local-output accumulation.  Regression tests cover reset/accumulation behavior, the sum-factorized 1D-contraction count, fallback paths with no 1D-contraction construction, and unchanged numerical results against the explicit reference operator.

Next slice:

- Completed first explicit local B/G/K construction slice: added a host-side `TensorLocalSchurComplementBuilder` that materializes local B blocks from `TensorProductLocalOperator`, constructs diagonal/mass-like tensor-product G entries from one-dimensional diagonal factors, and forms the local diagonal-test-norm Schur complement using the current test-by-trial local B convention (`K = B^T G^{-1} B`).  Phase 0 tests compare B, G, and K against an explicit affine L2 tensor-quadrature reference on the smallest two-factor cell before any CRS scatter is introduced.

Next slice:

- Completed cached affine volume-cell B/G/K wiring slice: `TensorLocalSchurComplementBuilder` can now build per-cell local B operators directly from `TensorBasisCacheND`, term operator descriptors, and separable coefficient factors, and can construct L2 mass-diagonal G entries from cached test basis data.  The sum-factorized `MatrixFreeTensorOperator` volume path now uses this shared B construction helper instead of duplicating component-matrix setup.  Phase 0 tests compare cache-built B blocks against explicit tensor quadrature for affine two-factor cells and a 1X1T1V-like volume term, and continue checking the local diagonal-test-norm K block symmetry before CRS scatter is introduced.

Next slice:

- Add the first assembled-cell path that loops over affine L2 volume cells, forms local B and diagonal G with `TensorLocalSchurComplementBuilder`, computes local K = B^T G^{-1} B in the trial-coefficient convention, and compares those local K blocks against the legacy/reference dense integration for the smallest tensor meshes.  Keep this local-only until the per-cell K comparison is stable; then add CRS preallocation/scatter as the following slice.

## Phase 2: N-factor basis and quadrature cache

Implement an N-factor tensor basis/evaluator layer for line factors.

Data per component and p level:

- 1D quadrature points and weights;
- 1D basis values and derivatives on reference cells;
- geometry scalings for physical derivatives;
- optional interpolation/prolongation/restriction operators for p-multigrid.

The evaluator should expose tensor contractions for a rank-d tensor cell without constructing a full dense basis table.  For a bilinear term involving a derivative in component k, use derivative basis values in k and value basis values in all other components.  Geometry for affine 1D factors contributes separable Jacobian factors.

Operator implementation target:

- Implement local operator application as MFEM/PAMatrix-style sum factorization: a sequence of 1D contractions, one tensor component at a time, rather than dense basis-by-quadrature loops.
- To apply an operator A to a vector x, view x as coefficients on a D-component tensor basis.  Contracting against the cached 1D basis-value matrix in one component converts that component from basis coefficients to quadrature-point values while leaving the other components in their current tensor-product representation.
- With roughly p basis functions and p quadrature points per component, each component contraction is equivalent to multiplying a (p x p) 1D basis matrix by a (p x p^(d-1)) slab of tensor data.  Each output entry costs O(p) multiply-add work and there are O(p^d) entries, so one component conversion costs O(p^(d+1)).
- A full coefficient-to-quadrature conversion therefore costs about d * p^(d+1) flops.  After applying point-local data such as Jacobians and material/coefficient values at quadrature points, convert back to basis coefficients with transpose-like 1D contractions for another d * p^(d+1) flops.
- Use roughly 2 d * p^(d+1) flops per matrix apply, plus point-local coefficient/Jacobian work, as the verification model for observed local-action timings and operation counters.

Deliverables:

- `TensorBasisCache1D` and `TensorBasisCacheND` or similar;
- tests comparing against `TensorBasis`/legacy basis values for 2D and direct product checks for 4D;
- flop, timing, and memory accounting utilities.
## Phase 3: tensor Vlasov term representation

Introduce a Vlasov-specific tensor weak-form representation rather than trying to make arbitrary legacy `BF` terms sum-factorized immediately.

Represent terms such as:

- mass/time term: `-f, d_t w`;
- spatial transport: `-v_i f, d_{x_i} w`;
- velocity transport: `-(q/m) a_i f, d_{u_i} w`;
- trace term: `t_n_hat, w` on tensor skeletons;
- optional positivity terms only after the base linear problem is working.

Classify coefficients:

- constants;
- single-component coordinate functions, e.g. proper velocity coordinate factors;
- separable products;
- configuration-space solution-dependent fields E/B;
- fully coupled fallbacks.

Use `TensorFunction` metadata as setup input, but compile it into reusable device evaluators/scratch buffers.  Avoid virtual calls and per-term allocation in assembly kernels.

Deliverables:

- tensor-native equivalent of the current relativistic Vlasov bilinear form for 1D1T2V;
- coefficient classification tests;
- local element B and G comparison tests against legacy Camellia on small tensor meshes.

## Phase 4: sum-factorized local assembly

Implement local kernels for B, G, and K.

Recommended first target:

- assemble local B and G using sum-factorized contractions;
- invert or factor G locally using batched dense linear algebra when elementwise test spaces are discontinuous;
- form K_e = B_e G_e^-1 B_e^T locally and scatter into a Tpetra/Crs matrix.

For each cell:

1. Gather per-component basis values/derivatives and weights.
2. Evaluate/classify coefficients with the cheapest available path.
3. Apply tensor contractions dimension-by-dimension.
4. For the DPG Schur complement, solve with local G rather than forming explicit G inverse where possible.
5. Scatter element K blocks to global CRS.

Cost target: for degree p in d dimensions, avoid O(p^(2d) q^d) dense quadrature loops when coefficients and geometry permit O(d p^(d+1)) style tensor contractions.  Dense element matrices may still be stored/scattered, but their construction must exploit tensorization.

Deliverables:

- CPU correctness implementation using Kokkos parallel_for;
- GPU-ready data layout choices, even if first validation is on host/device OpenMP;
- assembled K path selected by a new driver option;
- tests versus legacy K for small p and small meshes.

## Phase 5: new Planar MITL tensor driver

Create a new driver, for example `TensorPlanarMITLRelativisticSpaceTime.cpp`, rather than destabilizing the existing driver initially.

The driver should:

- construct component 1D topologies for x, t, ux, and uy;
- construct a `TensorMesh` with f, w, and `t_n_hat` tensor variables;
- configure the same physical parameters and boundary data as `PlanarMITLRelativisticSpaceTime.cpp`;
- use the tensor-native assembled K path for the Vlasov solve;
- keep the existing configuration-space Maxwell solve initially, coupling through projected moments and fields;
- write the same diagnostic quantities where practical.

Initially disable or postpone:

- h-adaptivity;
- positivity enforcement;
- serendipity basis;
- multi-species support;
- non-affine geometry;
- non-product meshes.

After correctness and performance are established, selectively add these features back.

## Phase 6: matrix-free operators

After assembled sum-factorized K works, add matrix-free operator support.

Implement two paths:

1. Schur-complement apply with approximate G inverse:
   - apply B^T to trial vector;
   - approximately solve element-local G y = r using Chebyshev or another local iterative/factorized method;
   - apply B to y;
   - preserve SPD if the G inverse approximation is symmetric/positive enough for the outer solver assumptions.
2. Mixed-formulation apply:
   - represent the block system involving the error representation unknown epsilon;
   - apply blocks B, B^T, and G matrix-free;
   - use an indefinite Krylov solver/preconditioner path.

For the first matrix-free target, build a `Tpetra::Operator`/RowMatrix-compatible wrapper analogous to the PAMatrix examples, with a diagonal estimate for Ifpack2 Chebyshev.  Reuse p-multigrid facilities for level construction and Ifpack2 Chebyshev as a smoother.  Do not plan on Ifpack2 h-multigrid; if h-multigrid is required later, use MueLu or a Camellia-specific hierarchy.

Deliverables:

- local matrix-free B, B^T, G, and K application tests;
- Tpetra operator with `apply()`, domain/range maps, and diagonal extraction/estimate;
- Ifpack2 Chebyshev smoother configuration;
- p-multigrid integration and convergence tests.

## Phase 7: performance engineering

Instrumentation should report:

- setup/metadata compilation time;
- basis tabulation time;
- coefficient evaluation time;
- local assembly/apply time;
- scatter/import/export time;
- solver setup/apply time;
- memory footprint by major view/matrix.

Optimization priorities:

- reuse basis and coefficient buffers across terms/cells;
- group cells by identical p-order tuple and coefficient structure;
- specialize affine geometry and constant/separable coefficients;
- use Kokkos team policies with scratch memory for tensor contractions;
- minimize host-device synchronization;
- avoid constructing full 4D quadrature point arrays except in fallback paths;
- avoid `Teuchos::RCP`, virtual function calls, and STL lookup in kernels.

## Open questions (answers below each)

1. What exact polynomial spaces should the first tensor-native Vlasov path support: full tensor HGRAD/HVOL only, or also traces matching the current L2 flux trace behavior?
   - Also traces matching the current L^2 flux behavior
2. Should the first assembled K path use the L2 test norm only, or must graph norm be supported immediately?
   - Can start with L^2 test norm, but we do need to have support for graph norm.
3. What is the acceptable first coupling strategy to Maxwell: reuse legacy configuration-space solve and moment projection, or implement tensor-native configuration-space assembly too?
   - legacy is acceptable
4. Are periodic x-boundaries required for the first MITL tensor driver, or can the first target be the nonperiodic diode case shown in the current driver defaults?
   - non-periodic is all we need right now.
5. Can h-adaptivity be disabled for the first performance target?  The tensor topology has some 1D refinement support, but global tensor DOF management and skeleton conformity for adaptive products are not ready.
   - yes, h-adaptivity can be disabled for first performance target.  We just want to replicate the geometry (mesh spacing) that the existing driver is using.
6. Should positivity enforcement be deferred?  It adds nonlinear/proximal variables and solution-dependent exponential coefficients that will complicate the first fast path.
   - yes, positivity enforcement should be deferred
7. Which Trilinos/PAMatrix files should be vendored temporarily into Camellia, and under what include namespace, until the Intrepid2 implementation is available upstream?
   - All PAMatrix implementation files.  You can leave them in the Intrepid2 namespace unless that breaks things for some reason.
8. Which solver stack should be the first supported matrix-free target: Belos plus Ifpack2 Chebyshev/p-multigrid, or integration through existing Camellia `GMGSolver` abstractions?
   - Belos plus Ifpack2 Chebyshev/p-multigrid
9. What hardware is the primary performance target for the first benchmark: multicore CPU, NVIDIA GPU, AMD GPU, or all Kokkos backends from the start?
   - First target is single-core CPU.  Eventually, we want all the rest.  But that's where we start.
10. What numerical equivalence tolerance is acceptable when comparing sum-factorized assembly to legacy quadrature, especially for relativistic coefficients involving gamma and solution-dependent fields?
   - tolerances of roughly 1e-10 using some sensible measurement (Knuth's, maybe).

## Suggested milestone order

1. Correctness-only tensor metadata and DOF-map completion.
2. Local N-factor basis/quadrature and term evaluator tests.
3. Local B/G/K tensor assembly for affine cells and L2 test norm.
4. Global assembled K Vlasov solve in a new tensor Planar MITL driver.
5. Graph norm and required boundary/trace semantics.
6. Performance tuning and scaling benchmarks.
7. Matrix-free Schur-complement apply with Chebyshev approximate G inverse.
8. Mixed-formulation matrix-free operator.
9. p-multigrid plus Ifpack2 Chebyshev smoother integration.
10. Optional features: positivity, adaptivity, serendipity, multi-species, and future base-2D/3D tensor products.

## Proposed first implementation slice

The first code slice should be deliberately narrow:

- 1D1T2V tensor product mesh with uniform p and affine geometry;
- one species;
- variables f, w, and `t_n_hat`;
- L2 test norm;
- assembled K via sum-factorized local B/G/K;
- no adaptivity, no positivity, no serendipity;
- small test comparing local/global matrices against the legacy formulation;
- driver option to run the new tensor path and print timing/memory metrics.

This slice is small enough to validate, but it exercises the essential 4D tensor contractions needed for the MITL speedup target.