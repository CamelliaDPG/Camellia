# MFEM / hp3d Source-Level Comparison

*Document: 60-mfem-hp3d-source-comparison.md*  
*Last updated: comprehensive analysis from source files (fespace.hpp ~79 KB, pfespace.hpp ~30 KB, ncmesh.hpp ~57 KB, mesh.hpp ~146 KB, nurbs.hpp ~59 KB, prolongation.F90 ~43 KB, coarse2macro.F90 ~13 KB, macro2coarse.F90 ~13 KB, GDAMinimumRule.h/.cpp, GlobalDofAssignment.h/.cpp, BasisReconciliation.h, LocalDofMapper.h, MeshTopology.h/.cpp, GMGOperator.h, plus code notes 10-40)*

---

## 1. Overview and Scope

This document provides a comprehensive source-level comparison of three hp-adaptive finite element frameworks:

| System | Language | Primary DOF Management | Refinement Model |
| --- | --- | --- | --- |
| **MFEM** | C++ | `FiniteElementSpace` | NCMesh (h-adaptive), h/p conforming |
| **hp3d** | Fortran 90 | Module-level global arrays | hp-adaptive nodal basis |
| **Camellia** | C++ (Trilinos/Teuchos) | `GDAMinimumRule` + `GlobalDofAssignment` + `BasisReconciliation` | Minimum-rule adaptive |

The comparison is organized across five workstreams identified in the Camellia Analysis Plan:

1. Global DOF Numbering and Adaptive Constraints
2. Curvilinear Geometry
3. GMG Transfer Operators
4. Mesh Topology
5. Parallel Scalability

---

## 2. Architecture Overview

### 2.1 MFEM Architecture

MFEM separates concerns cleanly across a small number of key classes:

~~~
Mesh / NCMesh
  └─ FiniteElementSpace (fespace.hpp)
       ├─ GetElementDofs() / GetEdgeDofs() / GetFaceDofs()
       ├─ GetConformingProlongation() → SparseMatrix*
       ├─ GetProlongationMatrix() → Operator*
       ├─ GetRestrictionMatrix() → SparseMatrix*
       ├─ BuildConformingInterpolation()
       └─ DOF ordering: Ordering::byNODES / byVDIM
ParFiniteElementSpace (pfespace.hpp) extends with MPI distribution
~~~

**Key design decisions in MFEM:**
- A single class (`FiniteElementSpace`) owns the complete DOF layout for one space
- Conforming constraints are materialized as explicit sparse matrices (global, precomputed)
- Nonconforming topology is entirely encapsulated in `NCMesh`; `FiniteElementSpace` calls into it on demand
- Parallel extension is a direct subclass (`ParFiniteElementSpace`) rather than a separate system

### 2.2 hp3d Architecture

hp3d uses a module-level global data structure approach:

~~~
Global mesh arrays (elements, vertices, edges, faces)
  └─ hp3d modules/prolongation.F90
       ├─ prolong_dofs_h_ref()   ! h-refinement prolongation
       ├─ prolong_dofs_p_ref()   ! p-refinement prolongation
       ├─ restrict_dofs_residual() ! MG restriction
       └─ build_prolongation_matrix() ! explicit sparse matrix
  └─ prolongation/coarse2macro.F90
       ├─ C2M_SETUP()            ! initialize coarse→macro mapping
       ├─ C2M_DOF_MAP()          ! map DOFs across levels
       └─ C2M_TRANSITION_MAP()   ! per-element transitions
  └─ prolongation/macro2coarse.F90
       └─ M2C_* subroutines      ! reverse (restriction) direction
  └─ util/geom3D.F90 / geom_hp.F90
       └─ Nodal geometry evaluation
~~~

**Key design decisions in hp3d:**
- Decomposed two-stage prolongation pipeline: Coarse→Macro→Fine
- hp-transfers are subroutine-based with explicit matrix construction
- No global 'FE space' object; DOF arrays are maintained globally
- Geometry is nodal (physical node coordinates drive mapping)

### 2.3 Camellia Architecture

Camellia distributes responsibilities across multiple collaborating objects:

~~~
Mesh (Mesh.h)
  ├─ MeshTopology (MeshTopology.h/.cpp)    ← topology/connectivity
  └─ GlobalDofAssignment (GlobalDofAssignment.h/.cpp)
       └─ GDAMinimumRule (GDAMinimumRule.h/.cpp)  ← constraint resolution
            ├─ getCellConstraints(cellID)
            ├─ getDofMapper(cellID, varID, sideOrdinal)
            └─ BasisReconciliation (BasisReconciliation.h)
                 └─ SubBasisReconciliationWeights
LocalDofMapper (LocalDofMapper.h)
  └─ getDofMapper() → LocalDofMapperPtr
GMGOperator (GMGOperator.h)
  └─ setCoarseFineMeshPair()
~~~

**Key design decisions in Camellia:**
- Constraints evaluated lazily, per-cell, on demand (not precomputed globally)
- The 'minimum rule' determines which cell owns each shared subcell
- Six overlapping cache layers in `GDAMinimumRule` contribute to complexity
- Static condensation via `CondensedDofInterpreter` is pluggable

---

## 3. DOF Numbering and Adaptive Constraints

### 3.1 DOF Classification Hierarchies

**MFEM DOF Taxonomy (fespace.hpp:84-209):**

| DOF Type | Definition |
| --- | --- |
| `ldof` (local dof) | Index within element, 0..NumDofs-1 |
| `edof` (element dof) | Global DOF before conforming constraints |
| `fdof` (facet dof) | DOF located on mesh facets/edges |
| `vdof` (vector dof) | DOF index in a vector-valued space |
| `dof` (global dof) | Final global DOF after constraint elimination |
| `tdof` (true dof) | DOF after removing Dirichlet BCs and conforming constraints |

MFEM documentation (fespace.hpp:84) is explicit: edof, ldof, and tdof *must not be confused*; the class doc block warns about this.

**hp3d DOF taxonomy:**
- DOFs are associated with mesh entities (vertices, edges, faces, element interiors)
- No explicit 'true dof' abstraction; constraints are applied at assembly time
- Nodal basis: vertices hold vertex DOFs; edges/faces carry higher-order DOFs

**Camellia DOF taxonomy:**
- `GlobalIndexType` for global DOF indices
- Per-cell `DofOrdering` (via `DofOrderingFactory`) maps variables to local indices
- `LocalDofMapper` bridges local→global; `GDAMinimumRule` enforces ownership

### 3.2 Conforming DOF Handling

| Feature | Camellia | MFEM | hp3d |
| --- | --- | --- | --- |
| **Constraint representation** | `CellConstraints` per cell | `ConstraintMatrix` (sparse, global) | Dense local matrices per element |
| **When computed** | Lazy, per cell on demand | Precomputed, stored globally | At assembly time |
| **Orientation** | Per-subcell permutation | Orientation array per entity | Intrinsic to nodal basis |
| **h-constraint weights** | `SubBasisReconciliationWeights` | `GetConformingInterpolation()` | Basis function overlap integrals |
| **p-constraint** | Via `BasisReconciliation` | `BuildConformingInterpolation()` / `GetConformingProlongation()` | Via hierarchical basis overlaps |
| **MPI** | Non-symmetric peer-to-peer | Collective via `ParFiniteElementSpace` | Collective via MPI global arrays |

### 3.3 MFEM Conforming Prolongation APIs (fespace.hpp)

~~~cpp
// Serial conforming prolongation (returns nullptr if identity)
const SparseMatrix *GetConformingProlongation() const;    // line 677

// Virtual: overridden by ParFiniteElementSpace for parallel
virtual const Operator *GetProlongationMatrix() const     // line 691
{ return GetConformingProlongation(); }

// Restriction (adjoint of prolongation)
virtual const SparseMatrix *GetRestrictionMatrix() const  // line 714
{ return GetConformingRestriction(); }

// Conforming interpolation: L2 projection to conforming space
void BuildConformingInterpolation() const;

// Element-level DOF extraction
void GetElementDofs(int elem, Array<int> &dofs) const;
void GetEdgeDofs(int edge, Array<int> &dofs) const;
void GetFaceDofs(int face, Array<int> &dofs) const;
~~~

**ParFiniteElementSpace overrides** (pfespace.hpp:511-518):
~~~cpp
const Operator *GetProlongationMatrix() const override;      // line 511
const Operator *GetRestrictionOperator() const override;     // line 515
// R matrix restricts local dof vector to true dof vector
const SparseMatrix *GetRestrictionMatrix() const override    // line 517
{ Dof_TrueDof_Matrix(); return R; }
~~~

### 3.4 Camellia Constraint APIs (GDAMinimumRule.h)

~~~cpp
CellConstraints getCellConstraints(GlobalIndexType cellID);   // line 167
LocalDofMapperPtr getDofMapper(GlobalIndexType cellID,        // line 168
                               int varIDToMap = -1,
                               int sideOrdinalToMap = -1);

// Ownership-based assignment (minimum-rule contest among active cells)
// getConstraintPermutation() declared but not implemented - always throws
~~~

### 3.5 hp3d Constraint Handling (prolongation.F90)

hp3d uses a similar subcell-based ownership model, but constraint weights are stored as dense local matrices per element rather than per-subcell, and the ownership is resolved at assembly time rather than at DOF numbering time.

Key subroutines:
- `prolong_dofs_h_ref()`: L2 projection prolongation for h-refinement
- `prolong_dofs_p_ref()`: hierarchical basis prolongation for p-refinement
- `restrict_dofs_residual()`: MG restriction (adjoint of prolongation)
- `build_prolongation_matrix()`: explicit sparse matrix form

### 3.6 GDAMinimumRule Cache Complexity

The analysis of `GDAMinimumRule.cpp` (~185 KB) identified **6 overlapping cache layers**:

1. `_cellConstraints` — per-cell constraint cache
2. `_dofMappers` — per-cell LocalDofMapper cache
3. `_subcellDofIndices` — subcell-level DOF cache
4. Ownership tables — minimum-rule contest results
5. Permutation arrays — orientation tracking
6. Basis reconciliation cache — `SubBasisReconciliationWeights`

**MFEM comparison:** FiniteElementSpace uses a single global conforming interpolation matrix built lazily on first call to `GetConformingProlongation()`, then cached. No per-cell cache partitioning.

---

## 4. Nonconforming Mesh Topology

### 4.1 MFEM NCMesh (ncmesh.hpp)

NCMesh encapsulates the complete nonconforming mesh topology:

**Key data structures:**
- `Element`: mesh entity with `parent`, `children`, edge/face connectivity
- `Node`: geometric node; `p1`, `p2` are parent node indices for hanging nodes
- `ConstraintTable`: maps hanging nodes to parent nodes and linear coefficients

**Key APIs:**
~~~cpp
void Refine(Array<Refinement> &refinements);
bool AnyNonconforming() const;
void GetConstraints(SparseMatrix &cP, SparseMatrix &cR);
void LimitNCLevel(int max_level);
~~~

**Design philosophy:**
- Parent-child element links allow arbitrary-depth recursive h-refinement
- Constraint propagation is automatic on refinement
- Edge constraints: hanging nodes on edges constrained to edge midpoints
- Face constraints: bilinear constraints for hanging nodes on faces
- Minimal overhead: parent-child links and constraint table only

### 4.2 Camellia MeshTopology (MeshTopology.h/.cpp)

Camellia's `MeshTopology` is a 'god object' managing mesh connectivity with significant redundancy:

**Known issues:**
- Bidirectional relationships stored redundantly (cell→subcell AND subcell→cell)
- Nested iteration patterns: O(n²) or worse for common operations
- `getParentCellEntry()` is available but complex to use
- Handles multiple refinement types but no clean abstraction boundary

**Key APIs (MeshTopology.h):**
~~~cpp
// Parent-child relationships for GMG
CellPtr getParent(GlobalIndexType cellID) const;
vector<GlobalIndexType> getChildCellIDs(GlobalIndexType cellID) const;
// Subcell ownership (used by GDAMinimumRule)
vector<GlobalIndexType> getCellsForSide(GlobalIndexType sideID) const;
~~~

### 4.3 Topology Comparison

| Feature | Camellia MeshTopology | MFEM NCMesh | hp3d |
| --- | --- | --- | --- |
| **Storage model** | Bidirectional cell↔subcell maps | Parent-child trees + constraint table | Global element/node arrays |
| **Refinement tracking** | Per-cell refinement history | Per-element parent/children | Per-element hp-state flags |
| **Hanging node handling** | Via GDAMinimumRule constraint system | NCMesh constraint table | Nodal basis constraints |
| **Primary weakness** | God-object bloat, redundant bidirectional data | Already well-structured | Global data complicates parallelism |
| **Recommended action** | Simplify to directed parent-child DAG | N/A (reference design) | N/A |

---

## 5. Curvilinear Geometry

### 5.1 MFEM Geometry (mesh.hpp / nurbs.hpp)

MFEM provides a rich geometry layer:

**High-level architecture:**
- `GeometricFactors`: precomputed Jacobians and determinants at quadrature points
- `NURBSExtension` (nurbs.hpp): full NURBS-based geometry with weights and knot vectors
- `GridFunction`: high-order geometry representation
- `IsoparametricTransformation`: reference element to physical element mapping

**Key capabilities:**
- Full 3D curved geometry via NURBS or polynomial high-order nodes
- Isoparametric and sub/super-parametric element transformations
- `GetDofToQuad()`: precomputed basis-function-to-quadrature-point matrix
- NURBS: patch-based CAD-quality geometry (entire mesh can be NURBS)

### 5.2 hp3d Geometry (geom3D.F90, geom_hp.F90, adaptivity_geom.F90)

hp3d uses a **nodal geometry approach:**
- Physical node coordinates define geometry entirely
- Reference→physical mapping computed from nodal values
- `geom3D.F90`: core 3D geometry transformation routines
- `geom_hp.F90`: hp-adaptive geometry utilities
- `adaptivity_geom.F90`: geometry updates during adaptive refinement

**Key properties:**
- Element Jacobian computed directly from nodal coordinates
- No CAD-quality NURBS support; geometry is polynomial/isoparametric only
- Geometry updates during refinement via `adaptivity_geom.F90`

### 5.3 Camellia Geometry (CamelliaCellTools.h)

- `CamelliaCellTools` for reference↔physical frame mappings
- `mapToPhysicalFrame()`, `mapToReferenceFrame()` are core operations
- **Known limitation:** Only 2D curved edges supported; no full 3D curvilinear geometry
- Recommended direction: extract into a dedicated `GeometryMapping` interface (similar to MFEM's `GeometricFactors`)

### 5.4 Geometry Comparison

| Feature | Camellia | MFEM | hp3d |
| --- | --- | --- | --- |
| **Geometry support** | 2D curved edges only | Full 3D, NURBS, high-order | Full 3D, nodal isoparametric |
| **NURBS/CAD** | None | `NURBSExtension`, full patch support | None |
| **Jacobian computation** | CamelliaCellTools, reference only | `GeometricFactors` precomputed | Direct from nodal coordinates |
| **DofToQuad matrices** | Not available | `GetDofToQuad()` precomputed | Quadrature-point evaluation |
| **Curvilinear adaptation** | 2D edge perturbation only | Full h/p/r adaptive geometry | `adaptivity_geom.F90` handles nodes |
| **Recommended direction** | Dedicated `GeometryMapping` interface | Already well-structured (reference) | Adopt nodal approach for 3D |

---

## 6. GMG Transfer Operators / Prolongation

### 6.1 MFEM Transfer Operators (fespace.hpp / pfespace.hpp)

MFEM provides three abstraction levels for transfers:

**Level 1: Conforming prolongation (h-refinement within conforming space)**
~~~cpp
// Serial: stored SparseMatrix, nullptr if identity
const SparseMatrix *GetConformingProlongation() const;     // fespace.hpp:677
// Parallel: returns MPI-aware operator
const Operator *GetProlongationMatrix() const override;    // pfespace.hpp:511
~~~

**Level 2: Restriction (adjoint, for MG smoothing)**
~~~cpp
const SparseMatrix *GetRestrictionMatrix() const;          // fespace.hpp:714
const SparseMatrix *GetRestrictionMatrix() const override; // pfespace.hpp:517
{ Dof_TrueDof_Matrix(); return R; }
~~~

**Level 3: Abstract operator interface**
~~~cpp
// Matrix-free restriction (potentially optimized)
const Operator *GetRestrictionOperator() const override;   // pfespace.hpp:515
// Transpose of restriction
const Operator *GetRestrictionTransposeOperator() const;   // fespace.hpp:705
~~~

**Key properties:**
- Conforming prolongation materializes as stored sparse matrix
- Parallel version uses MPI-aware operators with collective communication
- Single-level: each `FiniteElementSpace` owns one level's operators
- For multi-level GMG, callers must maintain a hierarchy of fespace objects

### 6.2 hp3d Decomposed Prolongation Pipeline

hp3d uses a **two-stage decomposed** prolongation pipeline:

~~~
Stage 1: Coarse → Macro (coarse2macro.F90)
  C2M_SETUP()          -- initialize coarse→macro mapping
  C2M_DOF_MAP()        -- map DOFs with linear constraints
  C2M_TRANSITION_MAP() -- per-element coarse dependencies

Stage 2: Macro → Fine (prolongation.F90 + macro2coarse.F90)
  prolong_dofs_h_ref()     -- L2 projection for h-refinement
  prolong_dofs_p_ref()     -- hierarchical overlap for p-enrichment
  restrict_dofs_residual() -- restriction for MG correction
~~~

**Algorithm strategies:**

*h-refinement prolongation (L2 projection):*
- Evaluates coarse basis functions at fine quadrature points
- Supports recursive refinement (children of children)
- Sparse structure exploited: only basis functions with non-zero support on refined element contribute

*p-enrichment prolongation (hierarchical overlap):*
- Lower-degree DOFs map directly to subset of higher-degree DOFs
- No integration required; hierarchical basis ensures exact embedding
- Data flow: `(element, p-old)` → `(element, p-new)`

*coarse2macro.F90 data structures:*
- `COARSE2MACRO_MAP` type: encapsulates coarse→macro transition data
- `DOF_TRANSITION_INFO` type: tracks DOF dependencies across levels
- Vertex, edge, face, element mapping arrays
- Hanging node constraints, edge constraints, face constraints

### 6.3 Camellia GMG Transfer (GMGOperator.h)

Camellia's GMG is built on `GMGOperator`:
~~~cpp
void setCoarseFineMeshPair(MeshPtr coarseMesh, MeshPtr fineMesh);
~~~

**Available building blocks for hp3d-style transfer:**

| Required Component | Camellia Source | Status |
| --- | --- | --- |
| Fine/coarse cell parent-child relationships | `MeshTopology::getParentCellEntry()` | Available |
| Per-cell DOF orderings | `GlobalDofAssignment::getDofOrdering()` | Available |
| Sub-basis projection weights | `BasisReconciliation` | Available (shared) |
| Global DOF numbering | `GlobalDofAssignment::globalDofIndicesForCell()` | Available |
| Trace conformity constraints | `GDAMinimumRule` constraint data | Available |
| Static condensation DofInterpreter | `CondensedDofInterpreter` | Available |

### 6.4 Transfer Operator Comparison

| Feature | Camellia GMG | MFEM | hp3d |
| --- | --- | --- | --- |
| **h-refinement prolongation** | `getDofMapper()` + `BasisReconciliation` | `GetProlongationMatrix()` | `prolong_dofs_h_ref()` |
| **p-enrichment prolongation** | `BasisReconciliation` projection | `BuildConformingInterpolation()` | `prolong_dofs_p_ref()` |
| **Restriction operator** | Transpose of prolongation | `GetRestrictionMatrix()` | `restrict_dofs_residual()` |
| **Multi-level support** | Single coarse/fine pair in GMGOperator | Caller manages hierarchy | Decomposed 2-stage pipeline |
| **Materialized matrix** | Not pre-built | Stored `SparseMatrix*` | `build_prolongation_matrix()` |
| **hp-combined transfers** | Not handled together | Not unified | Unified in `prolongation.F90` |

---

## 7. Parallel Scalability

### 7.1 MFEM Parallel Design (pfespace.hpp)

MFEM's parallel design uses `ParFiniteElementSpace` as a direct subclass of `FiniteElementSpace`:

**Parallel data structures:**
- `ParDofToTrueDof`: global DOF numbering for distributed mesh
- `LocalDofs`, `SharedDofs`: local and shared DOF arrays per processor
- `FacetDofGroups`: DOFs on processor boundaries
- `ParProlongation`: distributed operators with MPI communication patterns

**Key differences from serial:**
1. Global DOF numbering: `ParFiniteElementSpace` manages MPI rank offsets
2. Shared constraints: additional conforming constraints for DOFs on processor boundaries
3. `GetSharedDofs()` / `GetSharedFaceDofs()` retrieve DOFs on processor boundaries
4. Parallel prolongation/restriction involve MPI collective operations

### 7.2 Camellia Parallel Design

Camellia's parallel implementation uses a non-symmetric peer-to-peer approach:
- `GlobalDofAssignment::globalDofCount()` returns total global DOFs
- `_partitionDofCounts` array: per-rank DOF counts with `[numRanks]` = total
- Peer-to-peer MPI communication (asymmetric)
- `getDofMapper()` → `LocalDofMapper` bridges local/global per cell

### 7.3 hp3d Parallel Design

hp3d uses MPI collective operations via global Fortran arrays:
- Global element/node arrays distributed across MPI ranks
- Prolongation/restriction use collective MPI
- More conventional HPC pattern than Camellia's peer-to-peer approach

---

## 8. DOF Ordering and Basis Evaluation

### 8.1 MFEM DOF Ordering (fespace.hpp)

| Ordering | Description | Performance Impact |
| --- | --- | --- |
| `Ordering::byNODES` | Order by node type (vertices, edges, faces, interiors), then by element | Better for vertex-based operations |
| `Ordering::byVDIM` | Order by vector component first, then by node type | Better for vector assembly |

**Basis evaluation:**
- `GetDofToQuad()`: precomputed basis-function-to-quadrature-point matrix
- Enables fast tensor-product assembly for structured quadrature
- Key performance optimization: avoids repeated basis evaluation during assembly

### 8.2 Camellia DOF Ordering

Camellia uses `DofOrderingFactory` and `DofOrdering` objects:
- Per-cell `DofOrdering` maps (varID, basisOrdinal) to local DOF index
- No precomputed `DofToQuad` matrices; basis evaluated on-the-fly via Intrepid2
- `GlobalDofAssignment::getDofOrderingFactory()` provides factory access
- `DofOrderingFactory::getDofOrderingFactory()` (also in Mesh.h:186)

### 8.3 hp3d DOF Ordering

hp3d uses a nodal DOF ordering inherent to the Fortran data structures:
- Vertex DOFs first, then edge DOFs (edge-order), face DOFs, element interior DOFs
- Consistent with standard FEM convention for hierarchical hp bases
- No explicit 'ordering mode' — ordering fixed by data structure layout

---

## 9. Complete API Cross-Reference

### 9.1 DOF Management

| Operation | MFEM | hp3d | Camellia |
| --- | --- | --- | --- |
| **Element DOF extraction** | `FiniteElementSpace::GetElementDofs()` | Element-local DOF arrays | `GlobalDofAssignment::globalDofIndicesForCell()` |
| **Edge/Face DOF extraction** | `GetEdgeDofs()`, `GetFaceDofs()` | Nodal edge/face arrays | `LocalDofMapper` interfaces |
| **Conforming projection** | `BuildConformingInterpolation()` | Nodal basis overlaps | `BasisReconciliation` |
| **DOF→True DOF mapping** | `GetTrueDofs()` | Assembly-time constraint | `GDAMinimumRule` + `LocalDofMapper` |
| **Parallel DOF numbering** | `ParFiniteElementSpace` | MPI global arrays | `GlobalDofAssignment::globalDofCount()` |
| **DOF ordering scheme** | `Ordering::byNODES` / `byVDIM` | Fixed nodal order | `DofOrderingFactory` |

### 9.2 Prolongation/Restriction

| Operation | MFEM | hp3d | Camellia |
| --- | --- | --- | --- |
| **h-refinement prolongation** | `GetProlongationMatrix()` | `prolong_dofs_h_ref()` | `GMGOperator` + `BasisReconciliation` |
| **p-enrichment prolongation** | `BuildConformingInterpolation()` | `prolong_dofs_p_ref()` | `BasisReconciliation` projection |
| **Restriction** | `GetRestrictionMatrix()` | `restrict_dofs_residual()` | Transpose of prolongation |
| **Explicit sparse matrix** | `GetConformingProlongation()` → `SparseMatrix*` | `build_prolongation_matrix()` | Not pre-built |
| **Matrix-free restriction** | `GetRestrictionOperator()` | Not available | Not available |

### 9.3 Mesh Topology

| Operation | MFEM | hp3d | Camellia |
| --- | --- | --- | --- |
| **h-refinement** | `Mesh::GeneralRefinement()` via `NCMesh` | `hp3d_refine()` modules | `Mesh::hRefine()` |
| **Parent-child access** | `NCMesh::Element::parent/children` | Global element parent arrays | `MeshTopology::getParentCellEntry()` |
| **Hanging node constraints** | `NCMesh::ConstraintTable` | Nodal constraint tables | `GDAMinimumRule::getCellConstraints()` |
| **Nonconforming check** | `NCMesh::AnyNonconforming()` | Implicit in refinement state | `MeshTopology` queries |
| **Maximum NC level** | `NCMesh::LimitNCLevel()` | Policy in refinement driver | Not directly available |

### 9.4 Curvilinear Geometry

| Operation | MFEM | hp3d | Camellia |
| --- | --- | --- | --- |
| **Reference→physical** | `IsoparametricTransformation` | `geom3D.F90` routines | `CamelliaCellTools::mapToPhysicalFrame()` |
| **Physical→reference** | `ElementTransformation::TransformBack()` | Inverse map routines | `CamelliaCellTools::mapToReferenceFrame()` |
| **NURBS support** | `NURBSExtension` (full patch) | None | None |
| **High-order geometry** | `GridFunction`-based geometry | Nodal isoparametric | 2D edge perturbation only |
| **Jacobian precompute** | `GeometricFactors` | Direct from nodal coords | CamelliaCellTools (on-the-fly) |

---

## 10. Key Findings and Recommendations

### 10.1 DOF Numbering Architecture

**Finding:** MFEM's `FiniteElementSpace` provides the cleanest separation of concerns. A single class owns the complete DOF layout, and conforming constraints are materialized as a single global sparse matrix. Camellia distributes this responsibility across 5+ classes with 6 overlapping cache layers.

**Recommendation:** Consider consolidating Camellia's DOF management into a `FESpace`-like class that:
- Precomputes the conforming interpolation matrix (rather than lazy per-cell evaluation)
- Eliminates cache redundancy by materializing the global prolongation operator at construction time
- Follows MFEM's pattern of `GetProlongationMatrix()` / `GetRestrictionMatrix()` as canonical APIs

### 10.2 Nonconforming Topology

**Finding:** MFEM's `NCMesh` provides a minimal, clean nonconforming topology layer with parent-child trees and a constraint table. Camellia's `MeshTopology` stores redundant bidirectional data and is a god object.

**Recommendation (from 40-mesh-topology-simplification.md):** Simplify `MeshTopology` to:
- Directed parent-child DAG (no bidirectional maps)
- Facet ownership table (single source of truth for shared entities)
- Separate hanging node constraint data into `GDAMinimumRule` exclusively

### 10.3 GMG Transfer Operators

**Finding:** hp3d's two-stage Coarse→Macro→Fine pipeline is a well-designed pattern for hp-adaptive transfers. MFEM's single-level APIs are clean but require callers to manage the hierarchy. Camellia has all the building blocks but no unified transfer layer.

**Recommendation (from 30-gmg-hp3d-transfer-option.md):** Implement hp3d-style two-stage prolongation for Camellia:
1. Stage 1 (Coarse→Macro): use `MeshTopology::getParentCellEntry()` + `BasisReconciliation`
2. Stage 2 (Macro→Fine): use `GDAMinimumRule` constraint data + `getDofMapper()`
3. Expose via MFEM-compatible `GetProlongationMatrix()` / `GetRestrictionMatrix()` API

### 10.4 Curvilinear Geometry

**Finding:** Camellia supports only 2D curved edge geometry. Both MFEM (via `NURBSExtension`, `GeometricFactors`) and hp3d (via `geom3D.F90` nodal mapping) support full 3D curvilinear geometry.

**Recommendation (from 20-curvilinear-geometry.md):** Extract geometry into a dedicated `GeometryMapping` interface:
- Decouple from `CamelliaCellTools`
- Enable pluggable geometry (isoparametric, NURBS, blending functions)
- Provide precomputed `DofToQuad`-style matrices (following MFEM pattern)

### 10.5 Parallel Scalability

**Finding:** MFEM's `ParFiniteElementSpace` uses collective MPI with a clean subclass design. hp3d uses collective MPI over global Fortran arrays. Camellia uses non-symmetric peer-to-peer MPI that is harder to scale.

**Recommendation:** Transition Camellia's parallel DOF numbering toward collective MPI patterns, following MFEM's `ParFiniteElementSpace` design.

---

## 11. Source File Reference

### 11.1 MFEM Source Files (Reference Implementation)

| File | Path | Size | Purpose |
| --- | --- | --- | --- |
| `fespace.hpp` | `reference-codes/mfem/fem/fespace.hpp` | 79,673 B | Core FE space, DOF management, conforming prolongation APIs |
| `pfespace.hpp` | `reference-codes/mfem/fem/pfespace.hpp` | 29,662 B | Parallel FE space, MPI DOF distribution |
| `ncmesh.hpp` | `reference-codes/mfem/mesh/ncmesh.hpp` | 57,167 B | Nonconforming mesh, hanging node constraints |
| `mesh.hpp` | `reference-codes/mfem/mesh/mesh.hpp` | 146,338 B | Core mesh class, refinement APIs |
| `nurbs.hpp` | `reference-codes/mfem/mesh/nurbs.hpp` | 59,108 B | NURBS geometry, NURBSExtension class |

### 11.2 hp3d Source Files (Reference Implementation)

| File | Path | Size | Purpose |
| --- | --- | --- | --- |
| `prolongation.F90` | `modules/prolongation.F90` | 43,313 B | hp-adaptive transfer operators (prolongation/restriction) |
| `coarse2macro.F90` | `prolongation/coarse2macro.F90` | 12,718 B | Coarse→macro mesh level mapping |
| `macro2coarse.F90` | `prolongation/macro2coarse.F90` | 13,061 B | Macro→coarse mesh level mapping (restriction) |
| `geom3D.F90` | `util/geom3D.F90` | 12,296 B | 3D geometry transformation routines |
| `geom_hp.F90` | `util/geom_hp.F90` | 2,732 B | hp-adaptive geometry utilities |
| `adaptivity_geom.F90` | `util/adaptivity_geom.F90` | 2,935 B | Geometry updates during refinement |

### 11.3 Camellia Source Files (Primary Analysis Target)

| File | Path | Size | Purpose |
| --- | --- | --- | --- |
| `GDAMinimumRule.h` | `src/include/GDAMinimumRule.h` | 14,165 B | DOF assignment via minimum rule; constraint computation |
| `GDAMinimumRule.cpp` | `src/Mesh/GDAMinimumRule.cpp` | 185,072 B | Implementation; 6 overlapping caches identified |
| `GlobalDofAssignment.h` | `src/include/GlobalDofAssignment.h` | 11,841 B | Abstract base for DOF assignment |
| `GlobalDofAssignment.cpp` | `src/Mesh/GlobalDofAssignment.cpp` | 49,137 B | DOF counting, partition management |
| `BasisReconciliation.h` | `src/include/BasisReconciliation.h` | 14,379 B | Sub-basis projection weights |
| `LocalDofMapper.h` | `src/include/LocalDofMapper.h` | 6,716 B | Local→global DOF mapping per cell |
| `MeshTopology.h` | `src/include/MeshTopology.h` | 21,808 B | Mesh connectivity, parent-child relationships |
| `MeshTopology.cpp` | `src/Mesh/MeshTopology.cpp` | 167,384 B | Implementation; god-object complexity |
| `GMGOperator.h` | `src/include/GMGOperator.h` | 19,384 B | GMG transfer operator |
| `Mesh.h` | `src/include/Mesh.h` | 14,535 B | High-level mesh API |
| `Solution.h` | `src/include/Solution.h` | 20,844 B | Solution management and DofInterpreter |
| `CamelliaCellTools.h` | `src/include/CamelliaCellTools.h` | 19,352 B | Reference↔physical frame mappings |

---

## 12. Cross-Reference to Analysis Documents

| Analysis Note | Status | Key MFEM/hp3d Comparisons |
| --- | --- | --- |
| `10-dof-numbering-and-constraints.md` | Active analysis | MFEM ConstraintMatrix vs Camellia CellConstraints; hp3d dense local matrices; Section 20 |
| `20-curvilinear-geometry.md` | Active analysis | MFEM GeometricFactors/NURBSExtension vs hp3d geom3D vs Camellia 2D-only; Section 4 |
| `30-gmg-hp3d-transfer-option.md` | Active analysis | hp3d 2-stage Coarse→Macro→Fine vs MFEM GetProlongationMatrix vs Camellia GMGOperator |
| `40-mesh-topology-simplification.md` | Active analysis | MFEM NCMesh minimal design vs Camellia MeshTopology god-object |
| `ANALYSIS-PLAN.md` | Reference | Overall comparison framework and phase definitions |
