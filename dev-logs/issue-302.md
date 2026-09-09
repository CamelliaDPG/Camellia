## Define tensor-native trace-aware local operator realization

Parent issue: #293

Prerequisite issues:
- #294
- #295
- #296
- #297
- #298
- #299
- #300

### Summary

This issue is the natural follow-on to #300. The main goal is to begin using the newly added tensor-native trace discretization and skeleton support in the operator-realization layer.

In particular, we now have:
- dimension-specific trace semantics on `TensorVar`
- explicit tensor-native skeleton support in `TensorMeshTopology`
- an initial trace-local layout/runtime interpretation in `TensorDofLayout`
- trace registration compatibility in `TensorMesh`
- and flat-buffer gather/scatter compatibility for trace-local coefficient spaces

What we still do **not** have is tensor-native realization of local trace terms. The next step is therefore to define the first trace-aware local operator substrate that can act on the new tensor-native skeleton support.

This issue should focus on the first FE-aware/local-realization layer for traces, rather than trying to solve every possible trace-term case at once.

---

## Motivation

Issue #299 established the first genuinely FE-aware tensor-native matrix-free **volume** apply path.

Issue #300 then established the missing structural substrate for **trace** variables, including:
- explicit tensor-skeleton support orthogonal to a chosen component dimension
- dimension-specific trace-variable meaning
- and an initial trace-local coefficient/layout interpretation

That means the tensor-native stack now has the necessary topology/discretization/runtime footing to begin implementing trace-aware local operator realization.

Without that next step, trace support remains structural only. Realistic ultraweak, DPG, DG, and mixed volume/trace formulations still require a tensor-native way to realize representative trace terms on the new skeleton domains.

---

## Scope

### In scope
- define the first tensor-native trace-local operator-realization substrate
- introduce lower-dimensional quadrature / point generation on tensor-native skeleton entities
- define trace-local basis interpretation on the new skeleton support
- support a representative subset of trace-aware local terms
- support at least one mixed volume/trace or trace/trace local-apply path
- preserve compatibility with the current descriptor/factory/operator split
- add focused correctness tests for trace-local realization

### Out of scope
- full final support for every trace/flux/operator combination
- final optimization of trace-aware kernels
- full neighbor/orientation machinery for every future use case unless needed by the first representative terms
- final PAMatrix-backed trace realization
- broad higher-dimensional generalization beyond what the current orthogonal-component setting needs

---

## Main design goals

### 1. Use the new tensor-skeleton substrate directly
The new realization path should consume the tensor-native skeleton support from #300 rather than bypassing it or falling back to legacy side-entity logic.

### 2. Keep the geometry tensor-native and lower-dimensional
Trace terms should be realized on lower-dimensional tensor-product domains induced by the skeleton entities. This should remain explicit in the implementation.

### 3. Reuse the current tensor-native operator layering
The work should continue to respect the current split among:
- `TensorBilinearForm`
- `TensorOperatorFactory`
- realized operator objects
- gather/local-apply/scatter staging

### 4. Start with representative terms, not universal coverage
A small but meaningful representative subset is preferable to a large speculative first implementation. For example, likely candidates include:
- trace/value couplings
- flux/trace-like pairings represented through current tensor-native trace semantics
- or a carefully chosen mixed volume/trace term subset

### 5. Leave room for later optimization
The first implementation should be structurally correct and explicit, while still leaving a clear path toward:
- factorized lower-dimensional contractions
- trace-aware sum-factorized kernels
- reference-integral reuse
- and eventual PAMatrix-oriented trace realization

---

## Suggested deliverables

### Deliverable A: trace-local quadrature / point-generation layer
Define how tensor-skeleton entities generate lower-dimensional physical points and quadrature weights.

### Deliverable B: trace-local basis interpretation
Define how trace-local coefficient indices correspond to basis actions on the tensor-native skeleton support.

### Deliverable C: first trace-aware local apply path
Implement a first representative trace-aware local operator-realization path in the tensor-native matrix-free stack.

### Deliverable D: focused correctness tests
Add tests that verify:
- skeleton-entity-local quadrature/point interpretation
- trace-local basis action behavior
- representative mixed volume/trace or trace/trace local apply correctness
- compatibility with the current tensor-native runtime structure

---

## Relationship to previous issues

- #294 established tensor-native topology/component topology.
- #295 established tensor-native discretization/runtime.
- #296 established gather/scatter and transfer.
- #297 established tensor-aware coefficients.
- #298 established the descriptor/factory/operator split.
- #299 established the first FE-aware tensor-native matrix-free volume apply path.
- #300 established tensor-native trace discretization and skeleton support.

This issue should build directly on #300 and begin the next missing step: actual trace-aware local operator realization on the tensor-native skeleton substrate.

---

## Suggested implementation strategy

### Phase 1: define the local geometric/numerical model for tensor-skeleton entities
- specify quadrature and physical-point generation on skeleton entities
- pin down trace-local basis interpretation
- define the first supported trace-aware term subset

### Phase 2: implement representative trace-local apply support
- add the first FE-aware trace-local kernel path
- integrate that path into the current matrix-free realized operator structure

### Phase 3: add focused correctness tests
- validate the local geometric interpretation
- validate representative trace-aware local apply results on small examples

### Phase 4: stabilize and document
- tighten comments and naming
- confirm alignment with the structural semantics introduced in #300
- document what remains deferred

---

## Acceptance criteria

This issue can be considered complete when:
- the tensor-native stack has an initial trace-aware local operator-realization path built on the #300 skeleton support
- representative trace-local or mixed volume/trace terms are supported
- lower-dimensional quadrature / point-generation behavior is explicit and tested
- the resulting implementation remains compatible with the current tensor-native descriptor/factory/operator structure
- focused tests cover the new trace-aware realization behavior
- the design leaves a clear path to later optimized and broader trace-aware operator work
