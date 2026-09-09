## Prototype first end-to-end 1X1T1V tensor-native matrix-free operator path

### Summary

This issue introduces the first genuinely end-to-end **1D spatial + 1D time + 1V** tensor-native prototype for the mixed ultraweak Vlasov DPG formulation.

The intent is to move beyond the current tensor-native structural and operator-substrate milestones and implement a first vertical slice that:

- realizes the mixed ultraweak Vlasov bilinear form on a tensor-native `1X x 1T x 1V` mesh,
- includes both representative volume terms and representative trace terms,
- supports tensor-native coefficient functions with the dependency structure needed by the formulation,
- provides a first Maxwellian helper family,
- performs matrix-free operator application through the tensor-native path,
- supports a small GMRES-style solve on a manufactured-solution or patch-style problem,
- includes selected correctness comparisons against both tensor-native explicit references and the current path,
- and adds a first modest MPI-distributed coefficient ownership / communication path for tensor-native operator use.

This issue is intended as the first concrete end-to-end tensor-native Vlasov prototype corresponding to the mixed ultraweak formulation described in the attached “Getting Started with DPG” document, and equivalent in intent, up to Schur-complement structure, to the current `VlasovSpaceTimeFormulation.cpp` path.

---

## Motivation

The current tensor-native stack now has the following major ingredients in place:

- tensor-native topology and discretization support,
- tensor-native coefficient/function abstractions,
- tensor-native bilinear-form descriptors,
- finite-element-aware matrix-free volume realization,
- trace discretization and tensor-skeleton support,
- trace-aware local operator realization,
- and explicit coefficient indexing/traversal support replacing the old ambient operator-facing interface.

In particular, issues #299, #300, #302, #304, #305, and #307 have now established enough of the local/operator substrate that it is natural to attempt the first actual end-to-end Vlasov vertical slice.

At the same time, the current implementation still lacks an actual representative tensor-native problem path for `1X x 1T x 1V`, and it still lacks the first real MPI-aware coefficient ownership/communication implementation in the tensor-native operator path.

This issue therefore aims to convert the current infrastructure into a first formulation-faithful tensor-native prototype, rather than continuing to extend only isolated substrate pieces.

---

## Mathematical target

The intended formulation is the mixed ultraweak Vlasov DPG formulation described in the attached “Getting Started with DPG” note.

For the `1X x 1T x 1V` case, the core bilinear structure should include the representative terms

$`\langle \widehat{t}_n, w \rangle_{\partial K} - (v f, \partial_x w)_K - (f, \partial_t w)_K - (E f, \partial_v w)_K,`$

with:

- $`f(x,t,v)`$ as the volume trial field,
- $`\widehat{t}_n`$ as the trace / skeleton trial variable,
- $`w`$ as the test variable,
- $`v`$ represented as a tensor-native coefficient depending only on the velocity component,
- and $`E(x,t)`$ represented as a tensor-native coefficient depending only on the spatial/time components.

The issue should not be interpreted as requiring all of the legacy formulation machinery or all current Vlasov features. The goal is a first coherent tensor-native vertical slice that is faithful to the mixed ultraweak formulation structure and that supports a modest manufactured-solution or patch-style solve.

---

## Scope

### 1. Tensor-native `1X x 1T x 1V` formulation prototype

Implement the first concrete tensor-native `1X x 1T x 1V` prototype problem path for the mixed ultraweak Vlasov DPG formulation.

This should include:

- one volume trial variable $`f`$,
- one trace trial variable $`\widehat{t}_n`$,
- one test variable $`w`$,
- representative volume terms,
- representative trace terms,
- and a small formulation helper story sufficient to keep tests readable.

The issue should remain primarily tensor-native and test-driven, rather than trying to bridge fully into the legacy formulation layer.

### 2. Required coefficient-dependency cases

The prototype should explicitly demonstrate the coefficient-dependency structure needed for the first Vlasov path, including:

- a coefficient depending only on velocity, for the streaming term,
- a coefficient depending only on space-time, for the acceleration term,
- and a Maxwellian or shifted-Maxwellian helper depending on velocity.

The goal is not just numerical functionality, but explicit demonstration that the tensor-native coefficient layer can express the relevant component/subset dependencies coherently.

### 3. Maxwellian helper family

Add a modest tensor-native Maxwellian helper family, suitable for immediate use in tests and small prototype problems.

The first helper set should include at least:

- a centered Maxwellian in $`v`$,
- a shifted Maxwellian in $`v`$,
- and parameterization by amplitude/density and temperature/variance-like scale.

The implementation may remain concretely `1V` in substance, but the API shape should be chosen so that later extension to multi-`V` remains natural.

### 4. Trace-aware realization in the prototype

This issue must include representative trace/boundary behavior, not just volume-only realization.

The first prototype boundary story should be modest but explicit. In particular, it should be possible to distinguish at least the temporal inflow / initial-condition role from other boundary pieces in the test problem setup.

This does not require a complete recreation of all current legacy inflow semantics, but it does require a coherent first tensor-native boundary-selection story.

### 5. Small matrix-free solve

This issue should go beyond operator application alone and include a small GMRES-style solve on a manufactured-solution or patch-style problem.

The initial expectation is not a full legacy-style solve workflow or time-slab orchestration replacement. Rather, the goal is a modest solve that is formulation-faithful enough to demonstrate the first true end-to-end tensor-native path.

### 6. First MPI-distributed coefficient ownership / communication path

Issue #307 shaped the tensor-native operator interface for later MPI work, but it did not yet implement true distributed coefficient ownership and communication.

This issue should therefore add a first modest but genuine distributed coefficient-management path for tensor-native operator use.

Given the present `1D` component meshes, the intended first model is simple and structured:

- each MPI rank owns a contiguous portion of each relevant component coefficient factor,
- tensor-native coefficient ownership is induced from that componentwise structure,
- local operator application can identify owned and needed nonowned entries,
- and simple communication/import/export support exists so that operator apply and small solves can run meaningfully across ranks.

The aim is not a grand final distributed-algebra design, but a real first implementation rather than a merely MPI-shaped abstraction.

### 7. Correctness comparisons

Correctness should be established through both:

- tensor-native explicit-reference tests, and
- a modest selected comparison against the current path.

The tensor-native explicit references should remain the primary validation anchor. The current-path comparison may remain relatively narrow, but it should be sufficient to demonstrate that the first prototype is not merely self-consistent.

---

## Non-goals

This issue should explicitly exclude the following larger follow-on items:

- PAMatrix realization,
- filtered/subset tensor hierarchy support,
- full time-slab orchestration,
- and a full internal redesign of the current placeholder tensor-product transfer apply path.

Similarly, while this issue should add a first distributed coefficient path, it should not attempt to solve every future MPI/distributed-design question at once.

---

## Proposed implementation direction

### 1. Introduce a small tensor-native Vlasov prototype helper layer for tests

To keep the issue readable and reduce repeated setup logic, add a small helper layer in tests for:

- constructing a `1X x 1T x 1V` tensor mesh,
- registering the representative trial/test variables,
- constructing the representative mixed ultraweak bilinear form,
- building standard coefficient functions such as `v(v)`, `E(x,t)`, and Maxwellians,
- and assembling the small manufactured-solution / patch-style prototype problem data.

This should remain modest and test-oriented rather than becoming a full driver layer.

### 2. Extend the bilinear-form / operator layer to the required Vlasov subset

The current tensor-native operator substrate already supports representative volume and trace-aware terms, but the first Vlasov vertical slice will likely require broadening the accepted and tested subset so that the needed mixed ultraweak terms are represented directly and clearly.

This may include:

- convenient construction helpers for derivative-on-test volume terms in the `x`, `t`, and `v` components,
- trace-term construction helpers matching the $`\langle \widehat{t}_n, w \rangle_{\partial K}`$ structure,
- and any modest factory/validation extensions needed to recognize the intended term set as a supported first prototype.

### 3. Add tensor-native Maxwellian helpers

This likely belongs in the tensor-function layer or in a closely related helper utility. The key requirement is that the resulting functions remain honest tensor-native coefficients rather than ad hoc side logic in tests.

### 4. Add a first distributed tensor-native coefficient container / communication story

The present `TensorCoefficientMap`, `TensorCoefficientTraversal`, and `TensorGatherScatter` abstractions provide a natural starting point.

The next step should likely introduce:

- explicit owned coefficient counts per rank,
- a first representation of nonowned but needed coefficient entries for local traversals,
- communication support for importing needed remote coefficients prior to apply,
- and enough rank-aware behavior that matrix-free operator apply and the small solve can run under MPI.

The design should remain as simple as possible while still being genuinely distributed.

### 5. Add a small GMRES-style solve path

The first solve path may be modest and problem-specific, but it should be real enough to demonstrate that the realized tensor-native operator can participate in an actual end-to-end linear solve.

A manufactured-solution or patch-style problem is preferred here because it allows direct control over expected behavior and tolerances without requiring full legacy orchestration.

### 6. Add modest current-path comparison coverage

The first comparison target should likely be small and carefully selected. A useful standard would be to mirror one stripped-down `1X1T1V` Vlasov operator or solve situation closely enough that the current path can provide a trusted reference for selected quantities.

---

## Delivered results expected from this issue

- first tensor-native `1X x 1T x 1V` mixed ultraweak Vlasov prototype path,
- representative volume and trace terms realized through the current tensor-native matrix-free operator stack,
- tensor-native Maxwellian helper family,
- explicit demonstrations of single-component and subset-of-components coefficient dependence,
- small manufactured-solution or patch-style solve on the matrix-free field-operator path,
- tensor-native explicit-reference correctness tests,
- modest current-path-aligned representative term-structure comparison tests,
- and first MPI-shaped distributed coefficient ownership / needed-entry support for the tensor-native operator path, including contiguous ownership ranges, contiguous traversal blocks, and representative nonowned-needed index queries.

---

## Relationship to previous work

This issue follows directly from the tensor-native umbrella direction in #293 and from the current state after #307.

In particular:

- #299 established finite-element-aware tensor-native volume realization,
- #300 / #302 / #304 established tensor-native trace/skeleton semantics and representative trace-aware realization,
- #305 corrected the local `TensorDofLayout` semantics,
- and #307 removed the remaining ambient variable-level operator-side coefficient interface while introducing `TensorCoefficientMap` and `TensorCoefficientTraversal`.

What remains is to turn that substrate into the first actual formulation-level tensor-native Vlasov prototype.

---

## Suggested acceptance criteria

This issue can be considered complete when all of the following are true:

- a tensor-native `1X x 1T x 1V` mixed ultraweak Vlasov prototype path exists,
- the prototype includes the representative mixed ultraweak terms
  - $`-(v f, \partial_x w)_K`$,
  - $`-(f, \partial_t w)_K`$,
  - $`-(E f, \partial_v w)_K`$,
  - and $`\langle \widehat{t}_n, w \rangle_{\partial K}`$,
- a tensor-native Maxwellian helper family exists and is used in tests or prototype setup,
- coefficient dependence on velocity-only, space-time-only, and other selected component subsets is explicitly covered,
- the tensor-native matrix-free path supports a small manufactured-solution or patch-style solve on the first `1X x 1T x 1V` field-operator path,
- correctness is tested against explicit tensor-native references,
- at least one modest current-path-aligned representative comparison exists,
- and the tensor-native coefficient path includes a real first MPI-shaped distributed ownership / needed-entry implementation sufficient for representative distributed tests.
