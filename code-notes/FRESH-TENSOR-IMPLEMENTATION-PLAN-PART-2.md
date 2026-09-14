# Fresh Tensor Implementation Plan, Part 2

## Purpose of this document

This is the active continuation plan for the tensor-native PAMatrix work.  It carries forward the archived Part 1 plan, records the current diagnostic state, and orders the remaining work.

The immediate objective remains the same as in Part 1: prove correct and fast tensor-product matrix-free action and sum-factorized assembly on controlled model problems before connecting those kernels to the DPG Vlasov formulation.  The production DPG path should not be driven by the existing dense local Schur-complement hot path; it should be reached through narrow adapters around the Intrepid2/PAMatrix tensor kernels once those kernels are correct, measured, and guarded.

## Current status summary

The project has moved from a general implementation sketch to a working direct-PAMatrix experimental path for tensor-product HGRAD Gram operators.  The important pieces now exist in runnable diagnostic form:

- a direct `Intrepid2::PAMatrix` HGRAD Gram driver path in `drivers/performance/TensorPoissonPAMatrixDriver.cpp`;
- real HGRAD tensor basis mode for 3D runs and true higher-dimensional HGRAD support in the fork-side Intrepid2 path;
- synthetic tensor basis mode for scaling experiments;
- matrix-free apply diagnostics;
- standard PAMatrix `assemble()` diagnostics;
- experimental column-apply-style `assemble2()` diagnostics;
- term-separated timing for mass and stiffness components;
- PAMatrix diagonal extraction and inverse-diagonal diagnostics;
- local flat Tpetra `Operator` and `RowMatrix` adapters for smoke testing;
- Ifpack2 Chebyshev smoke diagnostics against a host-side reference recurrence;
- lower-spectral-estimate diagnostics based on assembled-reference inverse iteration for diagnostic runs.

The code is still an experiment and not yet a production global FE solver or preconditioner path.  The current Tpetra adapters are explicitly local-flat/nonproduction adapters with identical local domain/range/row/column maps and no global FE ownership semantics.  That limitation is acceptable for the current diagnostics, but it must not be confused with a finished distributed matrix-free solver integration.

## Current diagnostic interpretation

### 3D standard assembly versus assemble2

For the direct 3D HGRAD runs with `p = 8..12` and one cell, standard `assemble()` was faster than the current `assemble2()` path in the recorded timings.  The standard `assemble()` employs specially-optimized, hand-crafted code specifically for the 3D case.  `assemble2()`, by contrast, has been seen to be faster for 4D cases: it beats the standard `assemble()` in the cases we care most about.  We tried a further-optimized `assemble2()` implementation that saved flops by leveraging the known unit-column structure of the input vector in successive right-applies, but this was done in too generic a way, with too much overhead to realize the potential ~2x performance gain.  We defer this until we have time to carefully craft a pure-BLAS implementation.

### 4D synthetic and true higher-dimensional runs

The synthetic and true higher-dimensional runs are still valuable because they exercise the tensor-data path and guard against dense fallback behavior.  They should remain in the diagnostic suite, but they should not be used to justify the sparse/staged unit-column branch.

The current priority is to keep the standard `assemble()`, matrix-free apply, assembly-compare, diagonal, RowMatrix, and Chebyshev diagnostics correct and well labeled while the unit-column optimization is deferred.

### Tpetra RowMatrix and Ifpack2 smoke paths

The RowMatrix smoke diagnostics show that the current local-flat RowMatrix adapter can reproduce local PAMatrix results in the tested tiny cases.  The path is useful for exercising Tpetra/Ifpack2 interfaces, but it remains nonproduction because it does not implement global FE ownership semantics.

The Ifpack2 Chebyshev smoke outputs show that the Chebyshev recurrence can be exercised against the local-flat diagnostic operator and compared to a matching host-side recurrence.  The separate inverse/action calibration diagnostics may still expose parameter or endpoint-estimate issues; those should be treated as preconditioner-calibration work, not as evidence against the tensor contraction kernels.

## Deferred: unit-column optimization for assemble2

The unit-column optimization is deferred until we can design and implement it as a carefully measured BLAS-only path.  A future implementation should be close to the existing high-performance contraction machinery and should avoid sparse coordinate logic in the hot kernels.

Requirements before resuming this work:

1. Produce a concrete BLAS/GEMM design, preferably expressed as operations on contiguous submatrices or batches of contiguous slabs in the existing tensor data layout.
2. Show where each dimension's right-apply work is performed by GEMM or an equivalently efficient batched dense operation.
3. Avoid per-point or per-entry coordinate decoding in the hot path.
4. Preserve the blocked-column public behavior and conservative fallback guards.
5. Add instrumentation that reports which assembly path is active and gives enough flop/byte estimates to compare against standard `assemble()`.
6. Require before/after timings against the current standard `assemble()` and the earlier `assemble2()` baseline before promoting the path.

Until that design exists, Part 2 should treat standard `assemble()` and matrix-free `apply()` as the reliable kernel targets.  The old dense unit-column point-evaluation path may remain only as a guarded diagnostic/fallback if it is still useful for comparisons.  The sparse/staged unit-column implementation should not be listed as active work.

## Carried-forward plan from Part 1

### Phase 1: direct PAMatrix-enabled Trilinos fork

Keep using the PAMatrix-enabled Intrepid2 fork directly.  Local Camellia-side or driver-side code should remain a narrow wrapper around fork APIs, not a copy of PAMatrix internals.  The fork path must continue to expose `PAMatrix::apply()`, `assemble()`, `extractColumn()`, `extractDiagonal()`, transformed basis values, tensor point data, and workspace allocation.

Acceptance criteria:

- the experimental driver builds against the intended fork headers;
- dense or CRS fallbacks remain explicitly guarded and limited to tiny regression cases;
- timings separate basis/cache construction, geometry/Jacobian/cell-measure construction, coefficient/point-data evaluation, matrix-free apply, sum-factorized local assembly, CRS scatter/fill, solver setup, and pure solver/preconditioner apply.

### Phase 2: Poisson matrix-free apply

Maintain the original model problem discipline.  Standard Bubnov-Galerkin Poisson on tensor-product cells is the first proving ground because it exercises HGRAD values/gradients, cell-local tensor contractions, quadrature-point coefficient/metric multiplication, and coefficient-to-quadrature / quadrature-to-coefficient contractions without DPG trial/test asymmetry, traces, graph norms, or Schur complements.

Rules:

- true matrix-free apply must not cache or apply dense element matrices;
- kernel data must remain in Kokkos views, PAMatrix containers, or Intrepid2 tensor containers;
- setup may use STL/Teuchos, but performance paths should not accumulate through host vectors;
- dimensions must remain runtime-selectable up to the planned high-dimensional range, subject to memory guards for assembled outputs.

### Phase 3: Poisson sum-factorized assembly

Keep standard PAMatrix `assemble()` as the reference sum-factorized local assembly path.  It should be compared against standard Intrepid2 integration and, for small cases only, dense assembled references.

Acceptance criteria:

- local matrices agree with trusted references at small sizes;
- assembled-entry thresholds prevent accidental huge dense/CRS runs;
- timing slopes and path labels make it clear when tensor contractions are being used.

### Phase 4: H1 Gram matrix assembly

The H1 Gram matrix is the next DPG-relevant kernel, but it is still `G` only.  It is not `B`, not `K = B^T G^{-1} B`, and not the full Vlasov operator.

Required coverage:

- mass-only and stiffness-by-component diagnostics;
- full H1 Gram diagnostics;
- matrix-free apply, standard `assemble()`, assembly comparison, diagonal extraction, and inverse-diagonal diagnostics;
- term-separated timing so mass and stiffness costs can be interpreted independently.

### Phase 5: comparison with existing dense local Schur work

Use tiny one-cell comparisons to relate the tensor-native H1 Gram path to the existing dense local Schur-complement infrastructure, but do not optimize the dense Schur path as the main line of work.  These comparisons are for confidence and calibration only.

### Phase 6: tensor-native mesh, geometry, and metadata adapters

Before Vlasov integration, finish narrow adapters for tensor mesh metadata, tensor-product geometry, basis selection, cell maps, and coefficient evaluators.  These adapters should select and feed PAMatrix-backed kernels; they should not introduce another contraction engine.

Current progress: the initial HGRAD/HGRAD tensor-volume bridge covers the constant-coefficient `vlasov-dt` smoke case.  The bridge now lives in a reusable, header-only `TensorPAMatrixBridge.h` adapter that maps `TensorBilinearTerm` descriptors to `Intrepid2::PAMatrix` HGRAD volume terms while preserving the performance driver's existing term-list model.  It supports coefficients that are either constant or depend on a single tensor component, folding the coefficient into that component's tensor cubature weights.  The bridge also decomposes additive separable coefficients with scalar factors, so the Planar MITL-style `-0.5 * (x_0 + x_1) * value(f) * d_2(w)` volume term is expressed as one composed tensor coefficient and then split into supported single-component PAMatrix terms by the adapter layer.  Another 4D smoke case covers the separable `-0.75 * x_1 * value(f) * d_3(w)` partial term from the next Planar MITL volume contribution while leaving its `v_y` coefficient subterm guarded as coupled/nonlinear follow-on work.  Focused `TensorPAMatrixBridge` unit smokes cover descriptor mapping, derivative vector weights, additive separable decomposition, and the coupled-coefficient rejection path; the public bridge header no longer relies on driver-side `PAMatrixAssembly.hpp` include order, so these smokes are now part of the tensor-specific `runTensorTests` target while remaining excluded from the generic `runTests` target that lacks the experimental tensor/PAMatrix fork include paths.  The performance driver now uses the reusable bridge helper for HGRAD tensor field variable construction and no longer carries separate driver-local descriptor/vector-weight or Vlasov adapter/decomposition wrappers; ordinary HGRAD mass and stiffness term creation also routes through the reusable bridge constructor.  A first dense reference check now compares the constant-coefficient `vlasov-dt` HGRAD value/test-derivative bridge term against standard Intrepid2 local assembly in the focused bridge unit tests.  This is intentionally still a narrow separable-coefficient bridge; coupled multi-component products, solution-dependent coefficients, boundary/trace terms, non-HGRAD spaces, and dense references for the coordinate-coefficient variants remain guarded follow-on work.

Important requirements:

- preserve true tensor-product cell structure through the mesh/geometry metadata;
- support high-dimensional hypercube/tensor-product cells without relying on unsupported classic Shards node orderings above 3D;
- keep orientation and topology assumptions explicit;
- maintain guarded fallbacks for cases not yet supported by the tensor path.

### Phase 7: DPG Vlasov components, one term at a time

After Poisson and H1 Gram work are correct and measured, begin the DPG Vlasov integration piecewise.  Do not begin with the full Schur complement or the full Planar MITL driver.

For each Vlasov/DPG term:

1. Identify the exact trial space, test space, value/gradient/div/curl operator combination, trace or volume nature, and coefficient dependence.
2. Express the term in the PAMatrix transformed-basis/point-data model if possible.
3. Add a small reference comparison against the existing dense/standard path.
4. Add matrix-free apply coverage for the term.
5. Add assembly coverage only if the assembled local matrix is needed for diagnostics or preconditioner construction.
6. Record unsupported features explicitly, including orientation, non-tensor topology, trace coupling, or coefficient layout limitations.

The intended order is:

- volume scalar HGRAD/HGRAD-style terms already closest to the H1 Gram machinery, now routed through the reusable `TensorPAMatrixBridge` adapter rather than remaining solely driver-local;
- single-component separable coefficient variants of those volume terms, starting with coordinate coefficients such as the `x_0 * value(f) * d_1(w)` smoke term, then sums decomposed into multiple such terms such as `-0.5 * (x_0 + x_1) * value(f) * d_2(w)`, and separable subterms extracted from larger coefficients such as `-0.75 * x_1 * value(f) * d_3(w)`;
- vector-valued or mixed HGRAD/HCURL/HDIV/HVOL terms that PAMatrix already supports in monolithic tests;
- trace and skeleton terms, with explicit orientation and side-topology handling;
- graph-norm pieces needed by the DPG test solve;
- only then the assembled ingredients needed for `B^T G^{-1} B`.

Rules for this phase:

- no dense local `B` in the matrix-free backend;
- no full local Schur complement as the first tensor-native target;
- each term must have a correctness test before it is combined with other terms;
- performance labels must separate coefficient evaluation, point-data construction, tensor apply/assembly, and any local solve.

### Phase 8: global solve and preconditioning path

Only after the local tensor kernels and individual Vlasov terms are correct should the work move to global FE semantics.  The local-flat Tpetra adapters are smoke-test tools; production integration needs proper maps, owned/shared DoF handling, import/export behavior, and preconditioner setup that respects the FE graph.

Near-term global/preconditioner tasks:

- define global matrix-free operator semantics for the tensor-native FE space;
- add proper Tpetra maps and import/export behavior;
- decide which diagonal, block diagonal, local Gram inverse, or Chebyshev ingredients can be built from PAMatrix data without dense local Schur assembly;
- calibrate spectral estimates and Chebyshev parameters using real global semantics rather than local-flat smoke tests.

## Immediate remaining work

### Phase A: stabilize lazy tag initialization

The monolithic Intrepid2 failures from the lazy tag work must be fixed before more performance work.  The lazy strategy is still desirable, but every compatibility accessor that returns dense tag views must materialize those views first.

Acceptance criteria:

- monolithic BasisEquivalence failures are fixed;
- PAMatrix monolithic tests that depend on dense tag-to-ordinal views are fixed;
- lazy TensorBasis tag generation remains available for individual `getDofTag()`, `getDofCount()`, and `getDofOrdinal()` paths where possible;
- full dense tables are built only when compatibility APIs such as `getAllDofTags()` or `getAllDofOrdinal()` ask for them.

### Phase B: rerun monolithic and focused tensor diagnostics

After the lazy-tag fix:

- rerun the Intrepid2 monolithic test that produced `monolithic-test-out.txt`;
- rerun the configured focused PAMatrix/Basis/TensorBasis tests;
- rerun the HGRAD Gram assembly/apply comparison sweeps only if the code touched the contraction kernels.

### Phase C: keep assemble2 labels honest

Update any diagnostic text that implies an active optimized unit-column path.  The labels should distinguish:

- standard sum-factorized `assemble()`;
- generic/fallback `assemble2()` behavior;
- deferred future BLAS-only unit-column optimization.

No staged sparse right-apply path should be described as an active Part 2 deliverable.

### Phase D: proceed toward Vlasov only after kernel stability

Once the lazy tag fix and current diagnostics are clean, the next productive implementation step is the Phase 6/7 bridge: choose the smallest DPG/Vlasov volume term that maps directly to existing PAMatrix capabilities, implement it as a narrow adapter, and compare it against the existing dense/standard path on tiny tensor-product cells.

## What not to do next

- Do not reintroduce the sparse/staged unit-column branch without a BLAS-only design.
- Do not optimize the dense local DPG Schur-complement path as the main objective.
- Do not treat local-flat RowMatrix/Ifpack2 smoke success as production global FE support.
- Do not combine multiple Vlasov terms before each term has an isolated correctness test.
