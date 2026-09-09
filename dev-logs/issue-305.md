## #305 Separate tensor indexing conventions from exact local factor spaces

### Summary

The current tensor-native stack uses `TensorDofLayout` as though it were both:

- a descriptor of tensor-product local coefficient indexing, and
- an exact local coefficient-space layout for realized operator application.

For p-uniform cases, these two roles happen to coincide.  For p-nonuniform cases, however, they do not.  The present implementation resolves the mismatch by using an **envelope cardinality** in each non-trace component, i.e. by taking the maximum 1D basis cardinality over active component cells and using the tensor product of those maxima as a single variable-local coefficient space.

This issue proposes a design correction: we should separate

- **tensor indexing conventions**
- from
- **exact local factor spaces associated with a particular tensor cell tuple or tensor skeleton entity**.

The main goal is not merely to remove the current envelope logic from `TensorDofLayout`, but to eliminate the architectural pressure that created it.

---

## Motivation

A recurring design concern in the current implementation is that local indexing is still too strongly tied to a notion of a single flat variable-local block.  In a genuinely tensor-native design, this is not the right primitive once p-nonuniformity is allowed.

What should remain true is:

- tensor cells and tensor skeleton entities are the correct **geometric carriers** for local realization,
- but the admissible local coefficient indices should be derived **factorwise from the relevant component mesh objects**.

That is, for a volume realization on tensor cell tuple
\[
(K_0, K_1, \dots, K_{d-1}),
\]
the local trial/test coefficient spaces should be determined from the product of the corresponding component-local basis index sets,
\[
I_0(K_0) \times I_1(K_1) \times \cdots \times I_{d-1}(K_{d-1}),
\]
rather than from a mesh-wide ambient tensor-product envelope.

Likewise, for a trace realization on a tensor skeleton entity, the local factor spaces should be determined from:

- the distinguished trace-component semantics, and
- the retained-component cells participating in that skeleton entity.

The current envelope design in `TensorDofLayout` signals that these ideas are not yet separated cleanly.

---

## Problem statement

At present:

- `TensorDofLayout` stores one `VariableLayout` per registered trial/test variable,
- each such layout stores one component cardinality per tensor component,
- and for p-nonuniform non-trace components, the component cardinality is taken to be the **maximum basis cardinality over active cells in that component topology**.

This produces a single flat variable-local tensor block that is large enough to cover every componentwise local basis extent that might arise anywhere in the mesh.

That choice is currently relied upon by:

- `TensorGatherScatter`, which assumes a fixed local dof count per variable and flat offset-based movement, and
- `MatrixFreeTensorOperator`, which currently loops over one fixed local trial/test ordinal range while integrating over varying tensor cells or skeleton entities.

In other words, the current architecture still assumes something close to:

> one variable \(\to\) one fixed local vector space.

For p-nonuniform tensor-native realization, that assumption is not exact.  The current envelope is therefore a workaround.

---

## Design principle

The intended design correction is:

> Tensor-product geometric entities should determine where local realization occurs, while exact admissible coefficient indices should be derived factorwise from the participating component mesh objects and variable semantics.

Equivalently:

- **geometry** belongs to tensor cells and tensor skeleton entities,
- **local algebraic factor spaces** belong to component-local basis data,
- and **flattening/orderings** are conveniences that should be parameterized by the actual factor shape rather than hard-coded as mesh-wide variable envelopes.

---

## Proposed redesign direction

### 1. Reinterpret `TensorDofLayout`

`TensorDofLayout` should stop meaning “the exact local coefficient space for this variable.”

Instead, it should become a descriptor of:

- tensor-product indexing conventions,
- support-kind metadata by component,
- trace-component metadata,
- and shape-dependent flatten/unflatten rules.

In particular, it should no longer own mesh-wide envelope cardinalities for p-nonuniform non-trace components.

A likely end-state is that `TensorDofLayout` becomes **shape-parametric** rather than **shape-owning**.

That is, it should know how to flatten and unflatten a supplied factor-cardinality tuple, but should not define that tuple by scanning the whole mesh and taking maxima.

### 2. Introduce exact local realization contexts

We likely need an explicit new abstraction for the exact local factor space associated with one realized geometric entity.

Representative types might include:

- `TensorLocalVolumeContext`
- `TensorLocalSkeletonContext`

or a more general:

- `TensorLocalEntityContext`

Such a context should carry, for the relevant tensor cell tuple or skeleton entity:

- participating component cell IDs,
- effective trial/test p-orders by component,
- exact 1D trial/test basis cardinalities by component,
- basis handles or cached basis data as appropriate,
- exact trial/test factor cardinality tuples,
- exact local dof counts,
- and any other geometry or quadrature metadata needed by realized operators.

This is where exact local factor-space information should live.

### 3. Move from variable-owned cardinalities to context-supplied cardinalities

Under the corrected design, the relevant cardinality tuple should be supplied by the current local realization context, e.g.
\[
(n_0(K), n_1(K), \dots, n_{d-1}(K))
\]
for a volume cell tuple \(K\), or the analogous tuple for a skeleton entity.

Then `TensorDofLayout` is used only to flatten/unflatten with respect to that supplied shape.

### 4. Rework local operator application around exact factor spaces

`MatrixFreeTensorOperator` should eventually stop applying local contributions by looping over a single variable-level ambient local ordinal range and filtering or implicitly tolerating padded ordinals.

Instead, for each realized geometric entity, it should:

1. build an exact local context,
2. obtain the exact trial/test factor cardinalities,
3. enumerate admissible tensor-product basis tuples directly,
4. gather and scatter with respect to those exact tuples,
5. and apply the local contribution without any envelope padding.

This should remove both the conceptual mismatch and the need for entity-side admissibility logic that exists only because the local space was padded.

### 5. Reconsider the role of `TensorGatherScatter`

The current gather/scatter model assumes one flat local block per variable and offset-based copying.

That is acceptable as an interim abstraction only because the current local coefficient space is padded to a fixed size.

Longer term, gather/scatter likely needs to become more explicitly tied to:

- global algebraic layout, and
- entity-specific global index lists or transfer maps,

rather than to one fixed variable-local block.

This issue does not require that full redesign immediately, but it should make the architectural need explicit.

---

## Scope of this issue

This issue focuses on the **design and interface correction** needed to separate indexing conventions from exact local factor spaces.

It does **not** require us to eliminate the current ambient variable-level flat global interface in one step.  What it does require is that this ambient interface stop being confused with the exact local factor space used in realized operator application.

Concretely, the implemented scope includes:

- redesigning `TensorDofLayout` so that flatten/unflatten is shape-dependent,
- introducing explicit local-context abstractions for exact factor-space description,
- migrating realized local operator logic to consume those exact contexts,
- making gather/scatter shape-parameterized,
- preserving an explicit ambient shape only for the current global/operator container interface,
- and updating tests/documentation to reflect the corrected semantics.

---

## Deliverables

- a clarified semantic contract for `TensorDofLayout` in which it describes tensor indexing conventions rather than mesh-wide exact local factor spaces,
- new local-context support for exact factor cardinalities on tensor cells and tensor skeleton entities,
- migration of realized local operator logic to exact context-derived factor spaces,
- an explicit ambient-shape interface for current global/operator/transfer container sizing where still needed,
- and updated tests that check exact context-derived local factor-space behavior rather than envelope cardinality behavior.

---

## Expected API/design consequences

We expect some or all of the following to result from this work:

- `TensorDofLayout` no longer stores envelope cardinalities for p-nonuniform non-trace components as though they were exact local-space facts,
- shape-dependent flatten/unflatten becomes explicit,
- local operator realization obtains exact factor cardinalities from geometric context,
- the current notion of one exact fixed local vector block per variable is reinterpreted as an ambient interface rather than an exact local-space description,
- and some existing tests that explicitly encode envelope semantics are rewritten or removed.

---

## Relationship to existing work

This issue follows naturally from the current state after #302 and #304.

Those issues clarified important trace-local semantics, including the correction that the distinguished trace component contributes local support extent 1 rather than any global support count.  They also exposed the deeper volume/trace design tension that remains when `TensorDofLayout` is still treated as a single exact local coefficient-space descriptor under p-nonuniformity.

The present issue addresses that broader architectural mismatch directly.

---

## Suggested acceptance criteria

This issue can be considered complete when:

- `TensorDofLayout` no longer needs envelope semantics in order to describe exact local tensor indexing,
- exact local factor-space information is obtained from explicit local geometric/component context rather than from mesh-wide maxima,
- the resulting semantics are clean for both volume and trace realization paths,
- any remaining ambient variable-level shape is used only as a global/operator/transfer container interface rather than as the definition of the exact local factor space,
- and the implementation/test suite reflects the corrected separation between indexing conventions and exact local factor spaces.

---

## Closing note

The key design point is that the envelope itself is not the root problem.  The root problem is that one object is being asked to serve both as:

- a variable-level tensor indexing abstraction, and
- an exact local coefficient-space descriptor across varying p-nonuniform geometric contexts.

This issue is intended to separate those roles cleanly.
