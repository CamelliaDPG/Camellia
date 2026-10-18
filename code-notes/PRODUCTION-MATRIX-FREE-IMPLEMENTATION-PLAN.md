# Production Matrix-Free Implementation Plan

## Purpose

This document is the production plan for turning the current tensor-native partial-assembly and matrix-free Vlasov-Maxwell work into a correct, maintainable, and performant implementation path. The immediate branch objective is to support drivers such as `PlanarMITLRelativisticSpaceTime.cpp` with a selectable tensor-native path that can use assembled, partial-assembly, or matrix-free Gram solves as appropriate.

The near-term engineering objective is to stop debugging the matrix-free path primarily through whole-driver traces from `TensorPoissonPAMatrixDriver.cpp`. That driver has become too large and too interleaved to be an efficient correctness vehicle. Its reusable pieces should be moved behind library-style components with explicit contracts and unit tests. The monolithic study and trace drivers should remain useful as integration and performance evidence, but not as the first place where algorithmic defects are isolated.

## Current production-readiness status

Status: not production-ready.

The branch contains substantial partial-assembly and matrix-free infrastructure, including reusable bridge and GMG components, but the production boundary is still incomplete. The matrix-free Gram/GMG path is still being validated mainly through end-to-end trace comparison, and the p=4 evidence shows remaining mismatches between assembled and matrix-free behavior. A production path must be component-tested, fail fast for unsupported configurations, maintain deterministic assembled-reference parity tests, and provide stable performance gates.

## Evidence reviewed

The plan is based on the available FileShelf task context and shelved evidence, including:

- `drivers/performance/TensorPoissonPAMatrixDriver.cpp`, a very large performance driver that still owns too many construction, algorithm, diagnostic, and CLI responsibilities.
- `drivers/performance/VlasovGramGMGStudyDriver.cpp`, which exercises the assembled GMG study path used as a reference.
- `drivers/performance/VlasovGramMatrixFreeStudyDriver.cpp`, which exercises nominally matrix-free Gram/GMG behavior.
- `drivers/performance/CompareVlasovGMGTraceReadinessStandalone5.cmake`, which compares assembled and matrix-free trace readiness.
- `build/fork-release-openmp/perf-p4/vlasov-gram-assembled-gmg-p4.txt` and `build/fork-release-openmp/perf-p4/vlasov-gram-matrix-free-p4.txt`, the p=4 study outputs called out in the task.
- `build/fork-release-openmp/TensorVlasovGramGMGTraceReadinessCompareP4-out.txt`, which shows multiple `agree=0` rows in coarse-correction and post-smoothing metrics after earlier stages agree.
- `src/include/TensorPAMatrixBridge.h` and `src/MatrixUtilities/TensorPAMatrixTpetraAdapters.cpp`, which represent already-extracted bridge and adapter work.
- `src/include/MatrixFreeGMGPreconditioner.h`, `TensorHGradPTransfer.hpp`, and `TensorLocalPMultigrid.hpp`, which represent already-extracted GMG and transfer infrastructure.
- Current test artifacts indicating existing unit-test groups for `MatrixFreeGMGPreconditioner`, comparison, and stage-trace comparison, while also showing that more production-level coverage is still needed.

## Current-state diagnosis

### What is working or partially working

1. The assembled GMG study path exists and remains the correctness reference for Gram inversion studies.
2. A matrix-free study driver exists and can be run for controlled experiments.
3. Matrix-free GMG concepts have already been partially extracted into reusable headers and tests.
4. `TensorPAMatrixBridge` already centralizes some HGRAD cubature, coefficient folding, tensor descriptor mapping, derivative-vector weights, coefficient decomposition, and tensor-term-to-PA-matrix adaptation that used to be embedded directly in the monolithic driver.
5. Tpetra adapter utilities exist for map-aware use of tensor PA matrix objects.
6. Stage-trace comparison utilities exist and should be preserved as regression infrastructure.

### What is not production-ready

1. `TensorPoissonPAMatrixDriver.cpp` remains the gravitational center for too much implementation detail. It should become a consumer of reusable components rather than the owner of those components.
2. Correctness is still being chased through monolithic trace output instead of through small deterministic contracts.
3. p=4 trace-readiness evidence shows unresolved assembled-vs-matrix-free disagreements in later GMG stages. In the trace artifact, coarse-correction and post-smoothing solution and residual metrics include `agree=0` entries.
4. Unsupported matrix-free configurations are not yet presented as a clean, documented support matrix with early rejection and actionable errors.
5. Performance expectations are not yet encoded as stable benchmark gates.
6. Production drivers should not need to know how to assemble ad hoc test-specific matrix-free pieces.

## Target architecture

The production design should be layered so that each layer can be tested independently and used by both performance studies and application drivers.

### Layer 1: Problem descriptors and solver configuration

Introduce small immutable descriptors for the problem and solver choices. These should capture mesh topology, tensor basis data, polynomial order, graph-norm or mass/stiffness term choices, boundary conditions, coefficient fields, operator mode, smoother policy, coarse solve policy, diagnostics, and performance options.

Candidate components:

- `TensorGramProblemDescriptor`
- `VlasovGramProblemDescriptor`
- `TensorSolverConfiguration`
- `MatrixFreeSupportMatrix`

Responsibilities:

- Own user-facing choices once, instead of spreading them through driver-local conditionals.
- Validate combinations before setup begins.
- Record whether the path is assembled, partial-assembly, matrix-free, or assembled-reference-for-test.
- Provide reproducible test fixture construction.

### Layer 2: Tensor operator and bridge factory

Create a factory layer that translates validated descriptors into tensor operator terms and bridge objects. The existing `TensorPAMatrixBridge` work should be kept and strengthened, not duplicated.

Candidate components:

- `TensorOperatorFactory`
- `VlasovGramOperatorFactory`
- `TensorPAMatrixPackage`
- `TensorPAMatrixDiagnostics`

Responsibilities:

- Build mass, stiffness, graph-norm, and mixed derivative terms in a single tested path.
- Produce both matrix-free and assembled-reference operators when a test requests parity.
- Validate Tpetra domain/range maps and local tensor layouts.
- Expose diagonal and inverse-diagonal extraction with clear handling of zero or nonpositive diagonal entries.
- Keep coefficient folding and basis/cubature ownership out of performance drivers.

### Layer 3: Matrix-free apply and adapter layer

The matrix-free operator package should expose a narrow Tpetra-compatible interface for application drivers and solvers.

Responsibilities:

- Apply the operator without unexpected global assembly.
- Provide optional assembled-reference apply only in tests and diagnostics.
- Track whether a call path accidentally assembled a matrix.
- Provide deterministic parity checks for small meshes and p-levels.
- Provide structured diagnostics independent of human-readable trace text.

### Layer 4: GMG hierarchy builder

Move the construction of p-multigrid hierarchy, transfer operators, smoothers, coarse operators, and coarse solvers into a builder that consumes the operator package and solver configuration.

Candidate components:

- `MatrixFreeGMGHierarchyBuilder`
- `MatrixFreeGMGPreconditionerFactory`
- `TensorPTransferFactory`
- `GMGCoarseSolvePolicy`

Responsibilities:

- Construct a `MatrixFreeGMGPreconditioner` or compatible operator for supported configurations.
- Keep hierarchy setup out of benchmark drivers.
- Provide an assembled-reference hierarchy for tests.
- Make smoother, spectral estimate, diagonal scaling, and coarse solve choices explicit and unit-testable.

### Layer 5: Numerical policies

Smoothers and spectral estimates should be policies with clear input/output contracts.

Candidate components:

- `ChebyshevSmootherPolicy`
- `JacobiSmootherPolicy`
- `SpectralRadiusEstimatePolicy`
- `DiagonalScalingPolicy`
- `ResidualComputationPolicy`

Responsibilities:

- Test Chebyshev lambda estimates independently from GMG hierarchy setup.
- Ensure residual computations use the same operator and maps as the apply path.
- Make diagonal scaling behavior explicit and reproducible.
- Fail loudly on invalid diagonals or invalid spectral estimates unless a documented fallback policy is selected.

### Layer 6: Structured diagnostics and traces

Keep trace-readiness infrastructure, but make it a consumer of structured stage records instead of hand-parsed driver output.

Candidate components:

- `GMGStageTraceRecorder`
- `GMGStageTraceComparator`
- `MatrixFreeDiagnosticLevel`

Responsibilities:

- Record stage name, p-level, vector norm, residual norm, smoother parameters, spectral estimates, and coarse solve metrics.
- Compare assembled-reference and matrix-free traces with first-disagreement reporting.
- Keep detailed diagnostics optional and off by default in production drivers.
- Preserve existing CTest trace comparisons as high-level regression tests.

### Layer 7: Application integration

Application drivers should request a solver mode and receive a configured solver/preconditioner. They should not construct low-level tensor terms directly.

Responsibilities:

- `PlanarMITLRelativisticSpaceTime.cpp` should be able to select assembled fallback, partial assembly, or supported matrix-free mode through configuration.
- Unsupported configurations should fail before solve setup with a concise actionable message.
- Production defaults should prefer concise diagnostics and stable performance.
- Study drivers should be thin clients of the same factories used by applications.

## Components that already exist and should be retained

The following existing components should be treated as foundations and hardened:

- `TensorPAMatrixBridge.h`: already owns important bridge responsibilities for HGRAD cubature/coefficient folding, tensor descriptor mapping, derivative-vector weights, coefficient decomposition, and PA matrix construction.
- `TensorPAMatrixTpetraAdapters.cpp` and related headers: should remain the boundary between tensor PA matrix objects and Tpetra solver interfaces.
- `MatrixFreeGMGPreconditioner.h`: should remain the central reusable GMG preconditioner abstraction, with setup moved behind factories where needed.
- `TensorHGradPTransfer.hpp`: should remain the HGRAD p-transfer implementation and receive direct transfer tests.
- `TensorLocalPMultigrid.hpp`: should remain the local p-multigrid algorithmic core and receive direct stage tests.
- Existing `MatrixFreeGMGPreconditioner` unit-test groups and stage-trace comparison tests: should be extended into the primary correctness gate.
- `VlasovGramGMGStudyDriver.cpp` and `VlasovGramMatrixFreeStudyDriver.cpp`: should remain integration and performance drivers, but not own reusable construction logic.
- `CompareVlasovGMGTraceReadinessStandalone5.cmake`: should remain a high-level regression gate after component tests are in place.

## Components that should be added or extracted

1. Descriptor and configuration types for Vlasov Gram problems and matrix-free solver choices.
2. A tensor Gram operator factory that can build both matrix-free and assembled-reference operators from the same descriptor.
3. A Vlasov-specific Gram operator factory for graph-norm terms and application-specific coefficients.
4. A matrix-free GMG hierarchy builder that assembles transfer, smoother, diagonal, coarse solve, and tracing policies.
5. Policy objects for Chebyshev/Jacobi smoothing, spectral estimates, residual computation, diagonal handling, and coarse solve selection.
6. Structured trace record types and comparators independent of driver stdout formatting.
7. Deterministic component fixtures for p=2, p=3, p=4, and eventually p=8 Vlasov graph-norm cases.
8. A production support matrix documenting which dimensions, bases, polynomial orders, coefficient types, boundary conditions, and solver options are supported.
9. Application-level solver factory hooks for `PlanarMITLRelativisticSpaceTime.cpp` and related Vlasov-Maxwell drivers.
10. Performance benchmark gates that report matrix-free apply throughput, setup cost, iteration counts, solve time, memory footprint, and accidental assembly detection.

## Refactoring plan for `TensorPoissonPAMatrixDriver.cpp`

### Step 1: Freeze behavior with tests and fixtures

Before moving code, add characterization tests for the paths currently used by the p=4 studies. The goal is to protect current behavior while making mismatches smaller and easier to reproduce.

Deliverables:

- A deterministic p=4 Vlasov graph-norm fixture with fixed mesh, coefficients, RHS, and initial guess.
- A small assembled-reference apply parity test.
- A direct matrix-free apply parity test for each operator term.
- A stage fixture that can run one pre-smoothing step without the full driver.

### Step 2: Extract descriptor construction

Move argument parsing and problem setup into descriptor creation functions. The driver should parse CLI/options and then create a descriptor. The descriptor should be usable by unit tests without CLI parsing.

Deliverables:

- Descriptor types and validation functions.
- Tests for supported and unsupported configurations.
- Replacement of driver-local ad hoc configuration conditionals with descriptor validation.

### Step 3: Extract operator construction

Move tensor term construction and bridge packaging into factories. The driver should ask for an operator package, not build individual terms manually.

Deliverables:

- Operator factory for generic tensor Poisson/Gram terms.
- Vlasov Gram factory for graph-norm-specific terms.
- Tests that each term matches the assembled reference on small deterministic fixtures.
- Tests for map compatibility and diagonal extraction.

### Step 4: Extract GMG hierarchy setup

Move transfer hierarchy, smoother configuration, coarse solver setup, and trace hooks into a GMG builder.

Deliverables:

- `MatrixFreeGMGHierarchyBuilder` or equivalent.
- Policy injection for smoother, spectral estimate, diagonal scaling, coarse solve, and diagnostics.
- Direct tests for pre-smooth, residual, restriction, coarse correction, prolongation, and post-smooth stages.

### Step 5: Convert driver diagnostics to structured tracing

Replace fragile trace text as the internal diagnostic source with structured records. Human-readable trace output can be generated from those records.

Deliverables:

- Structured trace record type.
- Trace recorder and comparator.
- Existing trace-readiness CTest rewritten or wrapped to compare structured records.
- First-disagreement reports that identify stage, level, vector, metric, and tolerance.

### Step 6: Thin the driver

After extraction, `TensorPoissonPAMatrixDriver.cpp` should primarily coordinate studies:

- Parse options.
- Build descriptor.
- Build operator/preconditioner through factories.
- Run study or solve.
- Emit concise results.

It should no longer own core construction algorithms.

## Correctness remediation plan

### Immediate priority: isolate p=4 mismatch

The p=4 trace-readiness output shows that early metrics can agree while coarse-correction and post-smoothing metrics disagree. The immediate remediation should be a component-scale reproduction that narrows the first divergence.

Actions:

1. Build the p=4 fixture from the same descriptor used by the study drivers.
2. Compare matrix-free and assembled-reference operator apply term by term.
3. Compare inverse diagonal values used by the smoother.
4. Compare Chebyshev spectral estimates and damping parameters.
5. Compare the pre-smooth output vector before residual computation.
6. Compare residual computation using identical vector inputs.
7. Compare restriction to coarse level.
8. Compare coarse operator application and coarse solve result.
9. Compare prolongation and correction update.
10. Compare post-smooth input and output.

Do not continue broad trace instrumentation until this first-divergence component test exists.

### Numerical contracts

Each component should define tolerances and invariants:

- Domain and range maps must match expected Tpetra maps.
- Matrix-free apply must match assembled reference within a documented tolerance for small deterministic cases.
- Diagonal extraction must match assembled diagonal within tolerance.
- Nonpositive diagonal entries must be rejected or handled by an explicitly documented policy.
- Spectral estimates must be finite, positive, and reproducible for deterministic inputs.
- Restriction/prolongation must preserve constants and satisfy expected p-transfer identities where applicable.
- Residual computation must use `r = b - A x` with a tested sign convention.
- Coarse correction must update the fine-level solution with a tested sign and scaling convention.

## Test plan

### Unit tests

Add or extend tests for:

- Descriptor validation and support-matrix rejection.
- Tensor operator term factory construction.
- `TensorPAMatrixBridge` apply parity against assembled reference.
- Tpetra adapter domain/range validation.
- Diagonal and inverse-diagonal extraction.
- Nonpositive diagonal handling.
- HGRAD p-transfer restriction and prolongation.
- Local p-multigrid stage sequencing.
- Chebyshev and Jacobi smoother policies.
- Spectral-radius estimate policy.
- Residual computation sign and map compatibility.
- Coarse correction update sign and scaling.
- Structured trace first-disagreement logic.

### Component regression tests from p=4 evidence

Create tests that reproduce the p=4 trace-readiness failure at component scale:

- Fixed Vlasov graph-norm descriptor.
- Fixed RHS and initial solution.
- Compare assembled-reference and matrix-free pre-smooth output.
- Compare residual immediately after pre-smooth.
- Compare restriction and coarse RHS.
- Compare coarse correction solution.
- Compare post-smooth output.

These tests should become the primary correctness gate for the current mismatch.

### Integration tests

Keep and strengthen:

- `CompareVlasovGMGTraceReadinessStandalone5.cmake` as a final regression gate.
- Assembled GMG study driver tests.
- Matrix-free GMG study driver tests.
- p=1 through p=4 smoke tests.
- Application-level smoke tests for the production solver factory.

p=8 tests should initially be nightly or performance-gated until the lower-order correctness gates are stable.

### Performance tests

Add benchmark gates for:

- Matrix-free apply throughput versus assembled apply on representative tensor meshes.
- GMG setup time.
- Iteration count for p=2, p=3, p=4, and later p=8.
- End-to-end solve time.
- Memory footprint relative to assembled Gram inversion.
- Accidental assembly detection on matrix-free paths.

Performance reports should include absolute values and ratios to assembled or partial-assembly baselines. Production acceptance should require that matrix-free solves avoid unexpected assembly and do not regress iteration counts without explanation.

## Application integration plan

1. Add a solver factory entry point that production drivers can call with a descriptor and solver configuration.
2. Wire `PlanarMITLRelativisticSpaceTime.cpp` to request assembled, partial-assembly, or matrix-free modes through configuration.
3. Preserve assembled solve as a fallback while matrix-free support is phased in.
4. Fail early for unsupported matrix-free configurations with messages that name the unsupported option and the nearest supported alternative.
5. Keep detailed trace output behind a diagnostic flag.
6. Add affordable application smoke tests that prove selection, fallback, and unsupported-configuration behavior.

## Production acceptance criteria

The matrix-free path should be considered production-ready only when all of the following are true:

1. `TensorPoissonPAMatrixDriver.cpp` is no longer the owner of reusable operator, bridge, and GMG construction logic.
2. All reusable components have direct unit tests.
3. p=1 through p=4 assembled-reference parity tests pass for operator apply, diagonal extraction, smoother steps, residuals, coarse correction, and full GMG application.
4. The p=4 trace-readiness comparison no longer contains unexplained `agree=0` entries for supported configurations.
5. Unsupported configurations are documented and rejected before solve setup.
6. Matrix-free application drivers can select the new path without constructing low-level components manually.
7. Performance gates show expected setup time, solve time, iteration count, and memory behavior.
8. Diagnostics are concise by default and detailed only when requested.
9. CI or the configured test workflow runs the relevant unit and integration tests.
10. Documentation explains supported modes, limitations, and how to reproduce performance studies.

## Suggested implementation phases

### Phase 0: Stabilize evidence

- Preserve the current p=4 study and trace artifacts.
- Add expected-failing or issue-linked component tests for the first known mismatch.
- Document the current support matrix and unsupported options.

### Phase 1: Extract descriptors and factories

- Add descriptor/configuration types.
- Extract operator construction into factories.
- Convert study drivers to consume factories while preserving current outputs.

### Phase 2: Direct operator and bridge tests

- Add assembled-reference parity tests for each term.
- Add map and diagonal tests.
- Fix any bridge defects found by these tests.

### Phase 3: Direct GMG stage tests

- Add stage-by-stage p-multigrid tests.
- Isolate and correct the p=4 first divergence.
- Promote the p=4 regression to a required test once fixed.

### Phase 4: Structured diagnostics

- Introduce structured trace records.
- Convert trace-readiness comparison to use structured records where possible.
- Keep human-readable output as a reporting layer.

### Phase 5: Application integration

- Add production solver factory entry points.
- Wire application drivers to select assembled, partial-assembly, or matrix-free modes.
- Add application smoke tests.

### Phase 6: Performance hardening

- Add benchmark gates.
- Tune smoother and hierarchy policies.
- Track memory and accidental assembly.
- Promote stable p=8 performance runs to regular gating when practical.

## Risks and mitigations

- Risk: extracted components change behavior unintentionally. Mitigation: characterization tests before refactoring and parity tests after each extraction.
- Risk: trace comparison remains the only failing signal. Mitigation: reproduce p=4 mismatch at component scale and gate on the first-divergence fixture.
- Risk: performance regresses while correctness is fixed. Mitigation: add benchmark gates early, but do not block correctness fixes on unstable p=8 numbers until lower-order cases are correct.
- Risk: application integration grows new ad hoc construction paths. Mitigation: require production drivers to use the solver factory and descriptor types.
- Risk: unsupported configurations silently fall back to assembled behavior. Mitigation: support-matrix validation must distinguish explicit fallback from accidental unsupported matrix-free setup.

## Definition of done for this plan

This plan is complete when it has been committed under `code-notes/PRODUCTION-MATRIX-FREE-IMPLEMENTATION-PLAN.md` and used to drive follow-on implementation changes. The implementation work is complete only when the production acceptance criteria above are satisfied, the p=4 mismatch is resolved or explicitly scoped out with documented limitations, and production drivers can use the matrix-free path through a tested factory interface.
