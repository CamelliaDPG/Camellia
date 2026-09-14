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

Current progress: the initial HGRAD/HGRAD tensor-volume bridge covers the constant-coefficient `vlasov-dt` smoke case.  The bridge now lives in a reusable, header-only `TensorPAMatrixBridge.h` adapter that maps `TensorBilinearTerm` descriptors to `Intrepid2::PAMatrix` HGRAD volume terms while preserving the performance driver's existing term-list model.  It supports coefficients that are either constant or depend on a single tensor component, folding the coefficient into that component's tensor cubature weights.  The bridge also decomposes additive separable coefficients with scalar factors, so the Planar MITL-style `-0.5 * (x_0 + x_1) * value(f) * d_2(w)` volume term is expressed as one composed tensor coefficient and then split into supported single-component PAMatrix terms by the adapter layer.  Another 4D smoke case covers the separable `-0.75 * x_1 * value(f) * d_3(w)` partial term from the next Planar MITL volume contribution while leaving its `v_y` coefficient subterm guarded as coupled/nonlinear follow-on work.  Focused `TensorPAMatrixBridge` unit smokes cover descriptor mapping, derivative vector weights, additive separable decomposition, and the coupled-coefficient rejection path; the public bridge header no longer relies on driver-side `PAMatrixAssembly.hpp` include order, so these smokes are now part of the tensor-specific `runTensorTests` target while remaining excluded from the generic `runTests` target that lacks the experimental tensor/PAMatrix fork include paths.  The performance driver now uses the reusable bridge helper for HGRAD tensor field variable construction and no longer carries separate driver-local descriptor/vector-weight or Vlasov adapter/decomposition wrappers; ordinary HGRAD mass and stiffness term creation also routes through the reusable bridge constructor, and the old driver-local HGRAD cubature/coefficient-folding/PAMatrix-constructor implementation has been removed.  The focused bridge unit tests now compare constant, scalar coordinate-coefficient, and additive separable Vlasov-style HGRAD value/test-derivative bridge terms against standard Intrepid2 local assembly.  The standard reference path has a small coefficient-aware cubature-weight multiplier hook for bounded scalar tensor-function checks, so the coordinate `x_0 * value(f) * d_1(w)` and additive `-0.5 * (x_0 + x_1) * value(f) * d_2(w)` cases no longer rely only on bridge self-consistency.  This is intentionally still a narrow separable-coefficient HGRAD descriptor bridge.  A generalized bridge transform helper and focused direct PAMatrix-vs-standard probes now exercise HCURL, HDIV, and HVOL volume VALUE/VALUE cases.  The tensor bilinear-form descriptors now carry initial explicit function-space metadata and CURL/DIV operator tags, but non-HGRAD descriptor routing remains guarded until basis selection, scalar/vector shape semantics, and function-space-specific adapter validation are implemented on top of that metadata.  Focused bridge tests now also make the current boundary-term and solution-dependent-coefficient rejection contracts explicit.  The first L2/HVOL volume bridge slice is now explicit: descriptor-backed PAMatrix volume construction is covered for the Planar MITL starter `(f, d/dt w)` shape, including scalar L2/HVOL trial VALUE with HGRAD test derivative, plus focused HGRAD/L2 and L2/L2 value/value smoke comparisons.  `TensorBilinearFormBuilder` has narrow L2 volume helpers for these scalar HVOL VALUE combinations so new L2 volume descriptors do not have to rely on manually spelling function-space metadata in every caller.  Coupled multi-component products, solution-dependent coefficients, boundary/trace terms, and non-HGRAD descriptor integration remain guarded follow-on work until their descriptor, evaluator, and geometry/orientation semantics are implemented rather than inferred.

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

The Vlasov term unguarding order is now:

1. **L2 volume terms first; defer HDIV and HCURL.**  Treat L2 as scalar-valued.  The first concrete term is the Planar MITL Vlasov time term `(f, d/dt w)`: trial `f` is scalar L2/HVOL on the 1D1T2V tensor cell with components ordered `(x,t,ux,uy)`, test `w` is scalar HGRAD on the same four components, the trial operator is VALUE, the test operator is the HGRAD derivative in component `t` (component ordinal 1), the coefficient is constant `+1.0`, and the requested starter sign is positive.  This is the first `(L2,HGRAD)` PAMatrix bridge target.  Keep the existing dense-formulation sign convention visible: `RelativisticVlasovSpaceTimeFormulation::initializeVlasovBilinearForm()` currently adds the integrated-by-parts term as `(-f, w->di(t))`, so any adapter wired into that exact production BF must preserve the BF-provided coefficient sign instead of hard-coding the smoke-test sign.  Legal near-term combinations include scalar-weighted `(HGRAD,HGRAD)`, `(L2,HGRAD)`, `(HGRAD,L2)`, and `(L2,L2)` terms, with `(L2,L2)` included when an L2 test-space inner product is useful.  The bridge test suite now covers the starter `(L2,HGRAD)` time derivative term, a scalar coordinate-weighted `(L2,HGRAD)` derivative term, and constant `(HGRAD,L2)` / `(L2,L2)` value/value terms against standard assembly; the builder exposes narrow helpers for those L2 VALUE descriptors.  Unsupported combinations should fail immediately at descriptor construction or adapter selection rather than falling through to a dense or guessed path.
2. **Trace and skeleton terms by component dimension.**  Add traces one component dimension at a time.  For Vlasov, the immediate variables are Dirichlet trace/flux variables for the `t_n hat` unknowns: non-conforming L2 variables on the mesh skeleton with an implicit outward-facing normal.  These traces are single-valued skeleton unknowns, and boundary-term assembly must weight them by the outward-normal sign for the side being evaluated, i.e. `<sgn(n) t_n, w>` with negative sign on the lower/left/bottom side of the orthogonal component and positive sign on the upper/right/top side.  These traces are subdivided according to the dimension they are orthogonal to, so each trace variable lives in a geometric space that is still a true tensor product.  For Maxwell and Poisson, H1 traces are also needed; in the immediate 1D configuration-space case these are point values at element interfaces.  Skeleton variables are distinct variables, not merely traces of volume variables, and may be associated with combinations of volume variables and material weights without requiring a one-to-one volume-variable correspondence.  The tensor operator metadata now names and tests the 1D outward-normal sign convention explicitly: the existing skeleton side parity is documented as `-1` on lower/left component endpoints and `+1` on upper/right endpoints, with non-breaking `skeletonEntityOutwardNormalSign(s)` aliases for trace kernels that should read this data as normal signs rather than generic parity.  The affine local boundary-B block builder now consumes that outward-normal sign accessor by name for flux-like skeleton variables, so the current trace-aware matrix-free/local-boundary diagnostic path uses the same `<sgn(n) t_n, w>` convention that Phase 7 requires.
3. **Solution-dependent scalar coefficients.**  Add coefficients whose data come from another solution field after the fixed-coefficient L2 and trace paths are correct.  For the near-term planar MITL case, assume these coefficients are effectively 1D when used in Vlasov, matching the electromagnetic fields obtained from the Maxwell solve.  The tensor-function layer now has metadata-first 1D solution-coefficient helpers for configuration-space and named/frozen solution sources, with tests that dependency source metadata and derivative metadata propagate correctly.  A first real `TensorFunctionEvaluationContext` now registers source-specific solution-coefficient evaluators, evaluates scalar solution-coefficient leaves through those evaluators instead of through coordinate placeholders, recursively evaluates sum/product/unary expression trees containing those leaves, caches sampled coefficient values, scopes the cache by `TensorFixedPointLifecycle`, clears per-substep caches on fixed-point substep changes, and invalidates caches when the configuration-space solution epoch changes.  This is intentionally still a narrow adapter: PAMatrix coefficient injection remains guarded until the context is wired into the bridge's coefficient sampling path.
4. **Moment computations from the Vlasov solution.**  Add tensor-native moment reductions needed to form the Maxwell right-hand side, and expose the same moment machinery for diagnostic and visualization output.  The first Maxwell-RHS moments should mirror the existing Planar MITL driver and formulations: density is `solutionMomentFunction(0,0)`, x proper-velocity momentum is `solutionMomentFunction(1,0)`, and the second retained 1D2V component is `solutionMomentFunction(0,1)`; relativistic current accessors use `currentX()` and `currentZ()`, with diagnostic mean velocities computed as `Jx / (q * density)` and `Jz / (q * density)`.  In the relativistic formulation `gamma = sqrt(1 + (ux^2 + uy^2 + ...)/c^2)` enters through the velocity map/current formulas, so tensor-native moments must keep velocity-space weights explicit and should initially use the same exact tensor-product quadrature policy as the current code before relaxing quadrature.  The reduced data layout should be chosen to match the Maxwell coefficient/RHS consumer; for now, treat the output as per-configuration-cell scalar fields keyed by moment/current component rather than as ordinary bilinear terms.  A metadata-first `TensorMomentDescriptor` now records these Planar 1D2V density, proper-velocity moment, and relativistic-current requests, including velocity powers, exact-quadrature intent, relativistic velocity-map component, and charge weight.  A first host-side `TensorMomentReduction` kernel now reduces Vlasov coefficient values over tensor-product velocity quadrature into per-configuration-point scalar outputs keyed by a descriptor, including coupled relativistic-current weights `q*u_i/gamma` without pretending they are separable.  A narrow `TensorMaxwellRHSMomentData` / `TensorMaxwell1D2VMomentAdapter` handoff now packages reduced per-configuration-point density and proper-velocity moments into the same algebraic quantities consumed by `ConfigurationSpaceFormulationMaxwell1D2V::setMoments()`: `rhoRHS = sum_i q_i/eps0*(density_i-n0)`, `Jx = sum_i q_i*M_x_i`, and `Jz = sum_i q_i*M_z_i`, with the final Maxwell RHS signs still owned by the Maxwell formulation.  `TensorMaxwell1D2VMomentReductionPipeline` now connects the host-side Vlasov coefficient-value matrix to that handoff: it runs the descriptor-backed density, x proper-velocity, and retained z proper-velocity reductions for each species and accumulates the Maxwell RHS moment data in one tested tensor-native path.  The remaining work is to obtain those coefficient-value matrices from production tensor basis/storage and bridge the typed handoff container to production `FunctionPtr`/`RHS` or coefficient consumers.
5. **Coupled multi-component scalar coefficients.**  Add genuinely coupled coefficient data only after the separable/scalar paths and moments are stable.  For the 1D2V plus time planar MITL problem, the important coupled coefficient is the relativistic `gamma`, which depends on both velocity dimensions but is still scalar-valued.  `TensorFunction` now has canonical `relativisticGamma()` and `relativisticVelocityComponent()` helpers that represent this 1D2V coupled scalar coefficient shape and preserve the existing separable-expansion rejection tests.  `TensorCoupledCoefficientEvaluator` is the first explicit nonseparable point-data path for these coupled coefficients: it samples gamma and relativistic velocity at quadrature points as point data and rejects separable or solution-dependent fallbacks, so coupled coefficients no longer need to masquerade as tensor-product separable weights.  The standalone point-data evaluator remains the canonical sampler for gamma/velocity values outside the separable PAMatrix path.  Full PAMatrix unguarding for these coupled coefficients remains deferred until the bridge/kernel can inject general point weights into cell measures or operator data without losing the sum-factorized separable path for separable coefficients.  Matrix-valued coefficients are not a canonical tensor coefficient type here; decompose any such data into scalar components before it reaches the tensor bridge.
6. **Fixed-point lifecycle and H1 traces.**  The fixed-point lifecycle starts from a zero field guess unless an existing formulation supplies a previous-slab/continuation value.  Follow the existing Vlasov/Maxwell order and setter contracts: the driver sets fixed-point tolerance/min/max substeps, and the tolerance is the relative L2 difference in `f` between substeps.  Solution-dependent coefficient caches must be scoped to a fixed-point substep and invalidated when the Maxwell/configuration-space solution changes.  A metadata-first `TensorFixedPointLifecycle` now records the initial-guess policy, min/max substeps, relative-L2 tolerance, and coefficient-cache scope/invalidation policy; `TensorFunctionEvaluationContext` is the first actual cache owner using that descriptor for solution-dependent coefficient cache lifetime.  The first Maxwell/Poisson H1 trace targets are the 1D interface point-value traces for each scalar unknown in `ConfigurationSpaceFormulationMaxwell1D2V`: `E_x`, `B_y`, `phi`, and `A`.  A metadata-first `TensorTraceDescriptor` now distinguishes Vlasov nonconforming L2 skeleton traces, which carry the outward-normal sign, from Maxwell/Poisson H1 1D point-value traces, which are interface values and do not carry that normal sign; focused tests tie the Maxwell 1D2V target descriptors to the existing configuration-space field/trace names.  A first `TensorH1PointTraceLayout1D` now turns an H1 point trace descriptor into concrete interface-point value ordinals and lower/upper endpoint adapters without applying an outward-normal sign.  Remaining trace work is to connect this layout to tensor mesh/DofLayout ownership and Maxwell boundary/RHS assembly.

The ordering above is pragmatic rather than mathematical: Vlasov eventually needs all of these pieces, but this sequence maximizes isolated correctness tests and fail-fast adapter boundaries before combining terms.

Rules for this phase:

- no dense local `B` in the matrix-free backend;
- no full local Schur complement as the first tensor-native target;
- each term must have a correctness test before it is combined with other terms;
- performance labels must separate coefficient evaluation, point-data construction, tensor apply/assembly, reductions/moments, and any local solve;
- unsupported function-space/operator/coefficient combinations must fail fast when a descriptor is built or when adapter selection is attempted;
- descriptor metadata should use the existing TensorVar/TensorBilinearTerm representation of values, gradients, curls, divergences, and related operators rather than inventing a parallel metadata scheme; if an operator-space combination is not represented there yet, record that as a specific descriptor gap before implementing the term;
- coefficient data crossing the tensor bridge should be scalar-valued; matrix-valued physics data must be split into scalar component coefficients before tensor adapter selection;
- skeleton variables must be modeled as distinct unknowns on skeleton geometry, not as implicit aliases for volume variable traces.

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

Once the lazy tag fix and current diagnostics are clean, the next productive implementation step is the Phase 6/7 bridge: choose the smallest scalar L2 volume term that maps directly to existing PAMatrix capabilities, implement it as a narrow adapter, and compare it against the existing dense/standard path on tiny tensor-product cells.  Keep HDIV and HCURL guarded while this L2 path is brought up.  After the L2 volume path is stable, proceed to dimension-split skeleton trace terms, then solution-dependent scalar coefficients from Maxwell fields, then Vlasov moment reductions for the Maxwell RHS and diagnostics, and finally the coupled scalar `gamma(v_x,v_y)` coefficient.

The Vlasov-Maxwell coupling should be treated as an explicit fixed-point loop in the plan: electromagnetic fields used by the Vlasov solve are solution coefficients produced by the Maxwell solve, while the Maxwell right-hand side is determined by moments of the Vlasov distribution.  The tensor implementation should therefore distinguish ordinary coefficient evaluation, solution-dependent coefficient refresh from Maxwell state, and Vlasov moment computation feeding the next Maxwell solve.

## Remaining answers that would maximize autonomous progress

Most of the previously open semantic questions are now answered well enough to proceed on isolated L2 volume terms, dimension-split traces, scalar solution-dependent coefficients, moments, and the scalar coupled `gamma` coefficient.  The remaining decisions that would most improve autonomous progress are:

1. **Pick the first concrete L2 volume term.**  Name the exact Vlasov bilinear term to implement first, including trial variable, test variable, operator on each side, coefficient, sign, and which component dimensions it spans.  If there is no preference, start with the smallest scalar constant-coefficient `(L2,L2)` or `(L2,HGRAD)` term already present in the Planar MITL formulation and keep all other terms guarded.
2. **Specify trace orientation/sign conventions in code-facing terms.**  The plan now records outward-facing normals and dimension-orthogonal skeleton variables, but implementation still needs the exact plus/minus, left/right, or inflow/outflow naming convention expected by the existing Vlasov formulation and tests.
3. **Define moment formulas and storage layout.**  For the Maxwell RHS and diagnostics, specify the exact moments needed first, the velocity-space weights including where `gamma` enters, quadrature order expectations, and the target layout for writing the reduced data back into Maxwell coefficient/RHS structures.
4. **Define fixed-point lifecycle details.**  State the intended initial field guess, update order, convergence metric, relaxation policy if any, and whether Vlasov and Maxwell coefficient data should be cached per nonlinear iteration, per time slab, or per assembly/apply call.
5. **Identify the first H1 trace target for Maxwell/Poisson.**  In 1D configuration space these are point values at interfaces, but the first concrete unknown and term should be named before implementation.

These are not blockers for beginning the first fail-fast scalar L2 volume adapter, but answering them soon will reduce rework once traces, moments, and solution-dependent coefficients are unguarded.

## What not to do next

- Do not reintroduce the sparse/staged unit-column branch without a BLAS-only design.
- Do not optimize the dense local DPG Schur-complement path as the main objective.
- Do not treat local-flat RowMatrix/Ifpack2 smoke success as production global FE support.
- Do not combine multiple Vlasov terms before each term has an isolated correctness test.
