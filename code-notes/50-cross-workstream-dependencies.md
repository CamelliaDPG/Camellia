# Cross-Workstream Dependency Analysis

> **File**: code-notes/50-cross-workstream-dependencies.md
> **Context**: Camellia DPG framework modernization -- dependency map across the five analysis workstreams defined in ANALYSIS-PLAN.md.

---

## 1. Workstream Inventory

The analysis is organized into five numbered tracks, each addressing a distinct subsystem.

| Track | Name | Primary Files | Key Classes |
| ----- | ---- | ------------- | ----------- |
| **10** | DOF Numbering & Adaptive Constraints | GDAMinimumRule.{h,cpp}, GlobalDofAssignment.{h,cpp}, BasisReconciliation.h, LocalDofMapper.h | GDAMinimumRule, GlobalDofAssignment, BasisReconciliation, LocalDofMapper |
| **20** | Curvilinear Geometry | CamelliaCellTools.h | CamelliaCellTools |
| **30** | GMG / hp3d Transfer Options | GMGOperator.h, GMGOperator.cpp, Solution.h | GMGOperator |
| **40** | Mesh Topology Simplification | MeshTopology.{h,cpp}, Mesh.h | MeshTopology, Mesh |
| **50** | Code-Notes Refresh & Dependency Synthesis | All notes in code-notes/ | (documentation) |

---

## 2. Runtime Dependency Chain

The four implementation workstreams share data structures and call paths at runtime. The chain below represents the direction of **data flow and construction-time dependency** (top is depended upon; bottom depends on the above):

~~~
MeshTopology   (Track 40)
     |
     v
Mesh  (facade; owns topology and GDA)
     |
     v
GlobalDofAssignment / GDAMinimumRule   (Track 10)
     |
     +-------> BasisReconciliation / LocalDofMapper   (Track 10 helpers)
     |
     v
GMGOperator   (Track 30)   <--- also depends directly on MeshTopology
     |
     v
Solution   (end-consumer)
~~~

**CamelliaCellTools (Track 20)** is called by multiple layers:
- MeshTopology stores physical-node coordinates and calls cell-tools for side normal computations.
- GDAMinimumRule invokes cell-tools for reference-to-physical mapping during constraint evaluation.
- GMGOperator uses cell-tools (through BasisReconciliation) when computing inter-level basis transformations.
- Solution uses cell-tools when computing element stiffness matrices.

As a result, Track 20 (geometry) has the widest **fan-out** of all workstreams -- changes to the geometry interface ripple into every other track.

---

## 3. Directed Dependency Matrix

Rows = *provider*, Columns = *consumer*. A check (Y) means the row subsystem must be stable before the column work can be finalized.

| Provider -> Consumer | T10 DOF/Constraints | T20 Geometry | T30 GMG Transfer | T40 Topology |
| -------------------- | :-----------------: | :----------: | :--------------: | :----------: |
| **T10 DOF/Constraints** | -- | | Y | |
| **T20 Geometry** | Y | -- | Y | Y |
| **T30 GMG Transfer** | | | -- | |
| **T40 Topology** | Y | Y | Y | -- |

**Reading the matrix:**
- T40 (Topology) must be stabilized before T10, T20, and T30 can be finalized.
- T20 (Geometry) must expose its new GeometryMapping interface before T10, T30, and T40 rewrites can be completed.
- T10 (DOF/Constraints) must be stabilized before T30 (GMG) can be completed, because transfer-operator construction reads the conforming DOF map.
- T30 (GMG) is purely a consumer in this dependency structure -- no other track depends on it.

---

## 4. Interface Coupling Points

### 4.1 T40 -> T10: Topology-to-DOF Boundary

- GDAMinimumRule::getGlobalDofOrdinalsForSubcell iterates over **subcell-adjacency data** owned by MeshTopology.
- GDAMinimumRule calls MeshTopology::getActiveCellCount, getActiveCellIDs, getNeighborsForSide, and several entity-type queries.
- **Risk**: If T40 removes or renames any of those queries, T10 code that calls them must be updated simultaneously. These are the highest-risk coupling points in the entire codebase.
- **Mitigation**: Define a TopologyQueryAdapter interface (thin facade over MeshTopology) so T10 can be tested against a mock before T40 is complete.

### 4.2 T20 -> T40: Geometry-to-Topology Boundary

- MeshTopology currently stores physical-node coordinates for curved mesh support. The T20 plan proposes moving this data into a dedicated GeometryMapping object.
- Until T20 defines and stabilizes the GeometryMapping interface, T40 cannot safely remove the node-coordinate fields from MeshTopology.
- **Sequencing requirement**: Draft the GeometryMapping interface (even if not yet implemented) *before* T40 removes internal geometry data.

### 4.3 T10 -> T30: DOF-Map-to-GMG Boundary

- GMGOperator::constructProlongationOperator takes a DofInterpreter (coarse) and a DofInterpreter (fine) and queries conforming DOF indices from both.
- BasisReconciliation (a T10 component) is invoked during this construction to reconcile basis orientations across the coarse-to-fine boundary.
- **Risk**: Any change to the global DOF index assignment strategy (T10) changes the semantics of the inter-level prolongation matrix (T30).
- **Mitigation**: T10 should stabilize and document its index-assignment contract before T30 restructuring begins.

### 4.4 T20 -> T10: Geometry Used in Constraint Evaluation

- GDAMinimumRule calls CamelliaCellTools (T20) mapping methods when evaluating whether two sides are geometrically compatible for constraint imposition.
- **Risk**: Replacing CamelliaCellTools static calls with an injected GeometryMapping object requires updating every call site in GDAMinimumRule.
- **Mitigation**: Introduce the GeometryMapping injection point in T20 and provide a compatibility shim that delegates to the current static CamelliaCellTools API so T10 can continue compiling unmodified during the transition.

### 4.5 T20 -> T30: Geometry Used in Transfer Construction

- GMGOperator uses BasisReconciliation (which internally calls CamelliaCellTools) when aligning basis function orientations across mesh levels.
- Any change to the geometry-mapping interface (T20) will propagate through BasisReconciliation into the transfer-operator construction (T30).
- **Mitigation**: Stabilize the BasisReconciliation API before beginning T30 restructuring; treat it as a shared service boundary.

---

## 5. Recommended Sequencing

Based on the dependency matrix, the recommended implementation order is:

~~~
1. T40 (Topology simplification) -- foundational; everything depends on it
   a. Define TopologyQueryAdapter interface
   b. Remove redundant bidirectional caches
   c. Validate via existing MeshTopologyTests

2. T20 (Geometry interface) -- wide fan-out; must be stable before T40 removes geometry data
   a. Define GeometryMapping interface
   b. Implement CamelliaCellTools-backed shim
   c. Wire shim into MeshTopology, GDAMinimumRule, GMGOperator call sites

3. T10 (DOF/Constraints) -- depends on stable topology and geometry interfaces
   a. Decouple constraint discovery from index assignment
   b. Introduce ConstraintGraph intermediate representation
   c. Stabilize DofInterpreter contract for GMG consumption

4. T30 (GMG Transfer) -- pure consumer; begin only after T10 is stable
   a. Decompose constructProlongationOperator into Coarse2Macro + Macro2Fine stages
   b. Introduce MultigridTransferBuilder strategy interface
   c. Validate against existing GMGOperatorTests
~~~

---

## 6. Shared Abstractions Required by Multiple Tracks

Three interfaces will be introduced as part of this modernization and are explicitly **shared** across workstream boundaries:

| Interface | Owner Track | Consumers | Purpose |
| --------- | ----------- | --------- | ------- |
| TopologyQueryAdapter | T40 | T10, T30 | Thin facade over MeshTopology; allows tracks 10 and 30 to be tested against a mockable interface independent of MeshTopology internals. |
| GeometryMapping | T20 | T10, T30, T40 | Replaces static CamelliaCellTools calls; provides reference-to-physical mapping as an injectable service. |
| DofInterpreter (stabilized) | T10 | T30 | Already exists in Camellia; needs a documented, stable contract before T30 can safely restructure transfer-operator construction. |

---

## 7. Risk Register

| Risk | Tracks Affected | Likelihood | Mitigation |
| ---- | --------------- | ---------- | ---------- |
| T40 topology changes break T10 subcell queries silently | T10, T40 | High | Introduce TopologyQueryAdapter and write adapter-level unit tests before T40 refactoring. |
| T20 geometry interface change propagates through BasisReconciliation into T30 | T10, T20, T30 | High | Stabilize BasisReconciliation API before T30 work begins; treat it as a shared-service boundary. |
| T10 DOF index reordering breaks T30 prolongation-matrix construction | T10, T30 | Medium | Document and enforce DOF index assignment contract; add cross-level index-consistency tests. |
| CamelliaCellTools static-call sites missed during T20 refactor | T20 | Medium | Add a build-time deprecation shim that produces warnings on each legacy call site. |
| Epetra->Tpetra transition artifacts remain in T10/T30 code | T10, T30 | Low | Flag residual Epetra_ types during T10 DOF-map refactor; do not introduce new Epetra dependencies. |

---

## 8. Dependency Summary (One-Line per Track)

- **T10** depends on T40 (topology queries) and T20 (geometry calls); provides stable DofInterpreter to T30.
- **T20** depends on nothing (foundational geometry layer); must be designed before T40 can remove geometry data.
- **T30** depends on T10 (conforming DOF map), T20 (geometry via BasisReconciliation), and T40 (topology via MeshTopology).
- **T40** depends on T20 for interface design (so geometry data can be extracted safely); feeds T10 and T30.

---

## 9. Cross-Reference to Other Notes

| This Document Section | Related Notes File | Key Content |
| --------------------- | ------------------ | ----------- |
| Section 4.1 Topology-to-DOF | 10-dof-numbering-and-constraints.md | Six overlapping caches in GDAMinimumRule; constraint discovery analysis |
| Section 4.2 Geometry-to-Topology | 20-curvilinear-geometry.md | Current CamelliaCellTools limitations; proposed GeometryMapping interface |
| Section 4.3 DOF-map-to-GMG | 30-gmg-hp3d-transfer-option.md | Coarse2Macro->Macro2Fine pipeline; hp3d Schur-complement trace handling |
| Sections 4.1-4.5 | 40-mesh-topology-simplification.md | Redundant bidirectional relationships; TopologyQueryAdapter proposal |
| Section 2 Dependency Chain | 00-current-inventory.md | Subsystem map; full class ownership table |
| Section 5 Sequencing | ANALYSIS-PLAN.md | Five-phase analysis plan; comparison targets (MFEM, hp3d) |