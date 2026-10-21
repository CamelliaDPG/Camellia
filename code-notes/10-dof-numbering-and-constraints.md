# Global DOF Numbering and Adaptive Constraints Analysis

*Last updated: comprehensive analysis from source files (GDAMinimumRule.h ~14 KB, GDAMinimumRule.cpp ~185 KB, GlobalDofAssignment.h ~12 KB, GlobalDofAssignment.cpp ~49 KB, BasisReconciliation.h ~14 KB, LocalDofMapper.h ~7 KB)*

---

## 1. Subsystem Overview

Camellia manages global degree-of-freedom (DOF) assignment through a coordination among five classes:

| Class | Role |
| --- | --- |
| `GlobalDofAssignment` (GDA) | Abstract base: partitioning, element types, lookup lifecycle, `(cell, basis, local-dof)` → global index |
| `GDAMinimumRule` | Concrete GDA: implements the DPG minimum-rule constraint algorithm for hp-adaptive meshes |
| `BasisReconciliation` | Computes inter-cell reconciliation weights at non-conforming interfaces |
| `LocalDofMapper` | Wraps a `BasisMap` (collection of `SubBasisDofMapper`s) per variable per cell/side |
| `SubBasisDofMapper` | Atomic mapping: a set of fine basis-ordinals → a vector of global ordinals, with optional weight matrix |

The DPG minimum-rule says that a constrained DOF inherits its value from the lowest-dimension constraining ancestor subcell that has the required regularity, rather than the maximum-rule which always constrains to the coarser neighbor.

## 2. Key Data Structures

### 2.1 `AnnotatedEntity` (in `GDAMinimumRule.h`)

Records which cell and subcell owns/constrains a given DOF:

~~~cpp
struct AnnotatedEntity {
  GlobalIndexType cellID;
  IndexType       subcellOrdinal;  // ordinal within the cell
  unsigned        dimension;       // subcell dimension (redundant on purpose - see comment)
  // operator<, operator==, operator!= provided
};
~~~

Note: `dimension` is stored explicitly because subcells can be constrained by higher-dimensional subcells (e.g., a vertex may be constrained by an edge), making the dimension field non-redundant.

### 2.2 `OwnershipInfo` (in `GDAMinimumRule.h`)

~~~cpp
struct OwnershipInfo {
  GlobalIndexType cellID;
  unsigned        owningSubcellOrdinal; // ordinal within owning cell
  unsigned        dimension;            // dimension of owning subcell
};
~~~

### 2.3 `CellConstraints` (in `GDAMinimumRule.h`)

Per-cell constraint record:

~~~cpp
struct CellConstraints {
  // outer: subcell dim;  inner: subcell ordinal in cell
  vector< vector< AnnotatedEntity  > > subcellConstraints;
  vector< vector< OwnershipInfo    > > owningCellIDForSubcell;
};
~~~

### 2.4 `SubBasisReconciliationWeights` (in `BasisReconciliation.h`)

~~~cpp
struct SubBasisReconciliationWeights {
  FieldContainer<double> weights;       // (fine-ordinals x coarse-ordinals)
  vector<int>            fineOrdinals;
  vector<int>            coarseOrdinals;
};
~~~

The weight matrix `W` relates the fine-cell DOF coefficients to the coarse-cell DOF coefficients: `u_fine = W * u_coarse`.

### 2.5 `BasisMap` / `LocalDofMapper`

~~~cpp
typedef vector< SubBasisDofMapperPtr > BasisMap;
~~~

A `BasisMap` is a collection of `SubBasisDofMapper`s that together cover the full local basis for one variable. `SubBasisDofMapper` stores:
- `basisDofOrdinalFilter()` — the set of local basis ordinals it handles
- A list of global ordinals
- An optional dense weight matrix (present for constrained DOFs)

`` `LocalDofMapper` `` owns one volume `BasisMap` per variable (field) and one per (side, variable) pair (traces/fluxes).

## 3. Class Hierarchy and Inheritance

~~~
DofInterpreter (abstract)
  └── GlobalDofAssignment (abstract base)
        └── GDAMinimumRule (concrete)
              (GDAMaximumRule2D is a parallel concrete subclass, legacy/2D-only)
~~~

`` `GlobalDofAssignment` `` stores:
- `_meshTopology` (`MeshTopologyViewPtr`)
- `_varFactory` (`VarFactoryPtr`)
- `_dofOrderingFactory` (`DofOrderingFactoryPtr`)
- `_partitionPolicy` (`MeshPartitionPolicyPtr`)
- `_initialH1OrderTrial` (`vector<int>`)
- `_partitions` — vector of size `numRanks+1`; entry `[i]` is the global offset for rank `i`; `[numRanks]` is the total global DOF count
- `_partitionDofCount` — local DOF count for this rank
- `_partitionFieldDofCount`, `_partitionFluxDofCount`, `_partitionTraceDofCount` — breakdown by variable type
- `_constraintsCache` — `map<GlobalIndexType, CellConstraints>` — lazy per-cell constraint records
- `_dofMapperCache` — `map<GlobalIndexType, LocalDofMapperPtr>` — lazy per-cell full mappers
- `_dofMapperForVariableOnSideCache` — `map<cellID, map<side, map<varID, LocalDofMapperPtr>>>` — lazy per-cell/side/var mappers
- `_ownedGlobalDofIndicesCache` — `map<GlobalIndexType, SubcellDofIndices>` — owned subcell DOF index sets per cell
- `_globalDofIndicesForCellCache` — `map<GlobalIndexType, SubcellDofIndices>` — full (including shared) subcell DOF index sets per cell
- `_fittableGlobalIndicesCache` — `map<(cellID,(varID,sideOrdinal)), set<GlobalIndexType>>` — fittable DOF indices per cell/side/var

## 4. `rebuildLookups()` — The Central Rebuild Algorithm

`` `rebuildLookups()` `` is called by `GlobalDofAssignment` whenever the mesh topology changes (after h- or p-refinement, or unrefine). The call sequence in `GlobalDofAssignment.cpp` is:

1. `determineMinimumSubcellDimensionForContinuityEnforcement()` (MPI-collective)
2. `projectParentCoefficientsOntoUnsetChildren()`
3. `rebuildLookups()` (implemented in `GDAMinimumRule`)

### 4.1 `GDAMinimumRule::rebuildLookups()` steps (line ~3391 of GDAMinimumRule.cpp)

1. **`clearCaches()`** — clears all six caches:
   - `_constraintsCache`
   - `_dofMapperCache`
   - `_dofMapperForVariableOnSideCache`
   - `_ownedGlobalDofIndicesCache`
   - `_globalDofIndicesForCellCache`
   - `_fittableGlobalIndicesCache`

2. **`determineMinimumSubcellDimensionForContinuityEnforcement()`** — inspects all variable function spaces to determine the lowest topological dimension at which continuity must be enforced (0 for H1, 1 for H(div)/H(curl), etc.).

3. **Establish cell dimension / spaceDim** — reads `_meshTopology->getDimension()`.

4. **OWNERSHIP PHASE** — For each locally-owned active cell:
   - Calls `getOwnedGlobalDofIndices(cellID)` → computes owned subcell DOF indices.
   - Calls `getCellConstraints(cellID)` → retrieves (or computes and caches) the `CellConstraints`.
   - For each subcell (by dimension `d`, ordinal `scord`):
     - Resolves ownership via `constraints.owningCellIDForSubcell[d][scord]`.
     - Fetches the owning cell's `SubcellDofIndices` and copies over the DOF indices for constrained subcells.

5. **MPI EXCHANGE** — `distributeGlobalDofOwnership()` and `distributeCellGlobalDofs()` communicate owned DOF index ranges to all ranks that need them (for ghost cells).

6. **PARTITION OFFSET COMPUTATION** — Performs an MPI scan to build `_partitionDofOffsets`:
   - Entry `[rank]` = sum of DOF counts for ranks 0..rank-1
   - Entry `[numRanks]` = total global DOF count (returned by `globalDofCount()`)

7. **BASIS MAP CONSTRUCTION** — For each active cell, constructs `LocalDofMapper` via `getDofMapper()`, which internally calls `getBasisMap()` for each variable.

## 5. `getCellConstraints()` — Constraint Computation

`` `getCellConstraints(cellID)` `` is the lazily-evaluated, cached implementation that determines which cell/subcell owns each subcell of the queried cell. The result is cached in `_constraintsCache`.

### Algorithm sketch (from GDAMinimumRule.cpp ~line 2282):

1. If the cell is an active leaf cell, walk all subcells by dimension.
2. For each subcell, trace upward through the refinement tree to find the coarsest ancestor that still has an unconstrained side sharing that subcell — this is the **constraining entity** per the minimum-rule.
3. Determine the owning cell for that constraining entity. The owning cell is the active cell that claims that subcell for DOF ownership purposes (to avoid conflicts among siblings sharing the subcell). Return both the constraining entity and the `OwnershipInfo` for the owning cell.

The minimum-rule name derives from the fact that the constraint dimension is the *minimum* subcell dimension that can enforce the required continuity, as opposed to the maximum-rule which always constrains to the largest containing side.

## 6. `getConstrainingEntityInfo()` — Variable-Specific Constraint Resolution

~~~cpp
void getConstrainingEntityInfo(
  GlobalIndexType cellID, CellConstraints &cellConstraints,
  VarPtr var, int d, int scord,
  AnnotatedEntity* &constrainingInfo,
  OwnershipInfo* &ownershipInfo,
  bool &spaceOnlyConstraint);
~~~

This method handles the case where variable `var` has a **tag** (e.g., a Dirichlet or boundary-condition tag). If the variable has a tag, the constraints from `CellConstraints.subcellConstraints[d][scord]` are returned.

This allows boundary conditions and inter-element constraints to be handled through a unified constraint framework.

## 7. `getBasisMap()` — Building the DOF Mapping for a Variable

~~~cpp
BasisMap getBasisMap(GlobalIndexType cellID,
                     SubcellDofIndices& subcellDofIndices,
                     VarPtr var);
// side version:
BasisMap getBasisMap(GlobalIndexType cellID,
                     SubcellDofIndices& dofOwnershipInfo,
                     VarPtr var, int sideOrdinal);
~~~

Builds a `BasisMap` (vector of `SubBasisDofMapper`s) for variable `var` on the given cell/side.

### Volume variable algorithm (line ~826):

1. Iterate over all subcells (by dimension `d` and ordinal `scord`) that the volume basis has DOFs on.
2. For each subcell, call `getCellConstraints(cellID)` to get the `AnnotatedEntity subcellConstraint`.
3. If the subcell is **unconstrained** (owns its DOFs):
   - Look up the global DOF indices from `getOwnedGlobalDofIndices(owningCellID)`.
   - Create a permutation-only `SubBasisDofMapper` mapping local basis ordinals → global ordinals.
4. If the subcell is **constrained** (it defers to a coarser ancestor):
   - Determine the constraining cell and its basis from `elementType(subcellConstraint.cellID)->trialOrderPtr`.
   - Call `BasisReconciliation::weightsForSubcell()` (or `weightsForCoarseSubcell()`) to get the `SubBasisReconciliationWeights W`.
   - Look up the coarse (constraining) cell's global DOF indices.
   - Create a weight-matrix `SubBasisDofMapper` with `W` as the constraint map.
5. Also handles sub-sub-cell constraints (nested refinements) via recursive lookup through `getCellConstraints(subcellConstraint.cellID).subcellConstraints`.

### Side variable algorithm (line ~855+):

Similar to volume, but restricted to the side topology. Additionally:
- For p-constrained sides (where the neighbor cell has lower p), applies p-constraint weights.

### `getRestrictedBasisMap()`
Filters an existing `BasisMap` to only include specified `basisDofOrdinalRestriction`; used when restricting a volume basis map to a side.

## 8. `getOwnedGlobalDofIndices()` — DOF Ownership

~~~cpp
SubcellDofIndices & getOwnedGlobalDofIndices(GlobalIndexType cellID);
~~~

- Cached in `_ownedGlobalDofIndicesCache`.
- Returns a `SubcellDofIndices` that maps each subcell (by dimension, by ordinal) to a vector of global DOF indices that are **owned** by this cell.
- Only valid to call for locally-owned active cells; throws if called for a ghost or non-active cell.
- The owning cell for a subcell is determined by the ownership rule: among all active cells sharing a subcell, the owning cell is the one that wins the minimum-rule contest.

## 9. `getDofMapper()` — Building the LocalDofMapper

~~~cpp
LocalDofMapperPtr getDofMapper(GlobalIndexType cellID,
                               int varIDToMap = -1,
                               int sideOrdinalToMap = -1);
~~~

- Cached in `_dofMapperCache` (full cell) or `_dofMapperForVariableOnSideCache` (per-variable/side).
- Builds a `LocalDofMapper` for the given cell by assembling `BasisMap`s for all variables (or just `varIDToMap`) on all sides (or just `sideOrdinalToMap`).
- Internally calls `getBasisMap()` for each (variable, side) combination.
- The resulting `LocalDofMapper` is used by `Solution` to scatter local stiffness contributions into the global matrix.

## 10. `getFittableGlobalDofIndices()` — Boundary Fitting

~~~cpp
set<GlobalIndexType> getFittableGlobalDofIndices(
  GlobalIndexType cellID, int sideOrdinal, int varID = -1);
~~~

- Returns the set of global DOF indices for basis functions whose support intersects the given side with positive measure.
- Determined by taking the union of global DOF indices over all **constraining** (unconstrained) sides for the given side.
- Used during boundary condition application to determine which DOFs to fit against the boundary data.
- Cached in `_fittableGlobalIndicesCache` keyed by `(cellID, (varID, sideOrdinal))`.

## 11. `globalDofCount()` — Global DOF Count

~~~cpp
GlobalIndexType globalDofCount();
~~~

Returns `_partitionDofOffsets[numRanks]`. Guards against calling before `rebuildLookups()` sets up `_partitionDofOffsets`.

## 12. `globalDofIndicesForPartition(partitionNumber)` — Partition DOF Enumeration

Returns all global DOF indices belonging to a given partition by reading the offset range `[_partitionDofOffsets[p], _partitionDofOffsets[p+1])`.

## 13. `partitionForGlobalDofIndex(globalDofIndex)` — Reverse Lookup

Uses `std::upper_bound` on `_partitionDofOffsets` to find which partition owns a given global DOF index. O(log numRanks).

## 14. `BasisReconciliation` — Constraint Weight Computation

`` `BasisReconciliation` `` is a separate class (94 KB implementation) used internally by `GDAMinimumRule::getBasisMap()`. It caches and computes the reconciliation weight matrices that define how fine-mesh DOFs depend on coarse-mesh DOFs.

### Key caches (from BasisReconciliation.h):

| Cache | Key Type | Purpose |
|-------|----------|--------|
| `_simpleReconciliationWeights` | `(BasisPair, Permutation)` | p-refinement: simple basis-to-basis (no sides) |
| `_sideReconciliationWeights` | `((SideBasisRestriction×SideBasisRestriction), Permutation)` | p-refinement across conforming sides |
| `_simpleReconcilationWeights_h` | `(RefinedBasisPair, Permutation)` | h-refinement: coarse-to-fine, no sides |
| `_sideReconcilationWeights_h` | `(SideRefinedBasisPair, Permutation)` | h-refinement across sides |
| `_subcellReconcilationWeights` | `(SubcellRefinedBasisPair, Permutation)` | The **primary** cache; subcell-general h-reconciliation |
| `_termsTraced` | `TermTracedCacheKey` | Trace-to-field reconciliation weights per LinearTerm |

### `SubBasisReconciliationWeights` structure:
- `weights`: a `(fine-ordinals x coarse-ordinals)` dense matrix
- `fineOrdinals`: local ordinals on the fine (child) basis
- `coarseOrdinals`: local ordinals on the coarse (parent) basis

### Note on cache lifetime:
The comment in `BasisReconciliation.h` notes that `Function*` pointers in `_termsTraced` are not perfectly safe — a `BasisReconciliation` instance should not outlive the functions whose pointers are stored. This is a known technical debt.

### `weightsForCoarseSubcell()`
Filters `SubBasisReconciliationWeights` to only include DOFs interior to a given subcell on the coarse side (not on sub-subcells). Called in `getBasisMap()` at line ~957 to separate out the contribution from the constraining subcell's interior DOFs.

### `minimumSubcellDimension(basis)`
Static utility that returns the minimum subcell dimension on which the given basis has DOFs. Used to enforce continuity.

## 15. MPI Communication: `distributeGlobalDofOwnership()` and `distributeCellGlobalDofs()`

~~~cpp
void distributeGlobalDofOwnership(
  const map<GlobalIndexType, SubcellDofIndices> &myOwnedGlobalDofs,
  const set<GlobalIndexType> &remoteCellIDs);

void distributeCellGlobalDofs(
  const map<GlobalIndexType, SubcellDofIndices> &myCellGlobalDofs,
  const set<GlobalIndexType> &remoteCellIDs);
~~~

These are called during `rebuildLookups()` to communicate DOF ownership data to ranks that own ghost (remote) cells. The communication pattern is:

1. Each rank computes its owned subcell DOF indices for its local active cells.
2. Ranks identify which remote cells they need DOF data for (cells shared across partition boundaries).
3. `distributeGlobalDofOwnership()` sends each rank's owned DOF indices to the ranks that need them.
4. `distributeCellGlobalDofs()` sends the full cell DOF index arrays (including shared subcells) to remote ranks.

This follows a non-symmetric peer-to-peer pattern common in distributed FEM assembly.

## 16. `getConstraintPermutation()` — Orientation Handling

~~~cpp
unsigned getConstraintPermutation(GlobalIndexType cellID,
                                  unsigned subcdim, unsigned subcord);
~~~

Returns the permutation number that maps from the current cell's local view of the subcell to the constraining cell's local view. This is critical for bases that are orientation-sensitive (H(curl), H(div)) where the DOF ordering depends on the reference-to-physical orientation map.

Note: The two-argument overload `getConstraintPermutation(cellID, sideOrdinal, subcdim, subcord)` is **unimplemented** and throws `std::invalid_argument` — a known gap in the implementation.

## 17. `GDAMinimumRule` Factory Methods

~~~cpp
static Teuchos::RCP<GDAMinimumRule> create(
  MeshPtr mesh,
  VarFactoryPtr varFactory,
  DofOrderingFactoryPtr dofOrderingFactory,
  MeshPartitionPolicyPtr partitionPolicy,
  vector<int> initialH1OrderTrial,
  int testOrderEnhancement);
~~~

Constructs with `GDAMinimumRule(..., partitionPolicy, initialH1OrderTrial, testOrderEnhancement, false)`. The final `false` disables the max rule flag — this is the defining characteristic of the minimum-rule GDA.

## 18. `GDAMinimumRule` Internal State (Additional Fields)

From `GDAMinimumRule.h`:
- `_partitionDofOffsets` — size `numRanks+1`; cumulative DOF count per partition
- `_activeCellOffset` — global cell index offset for this partition's active cells
- `_localCellDofOffsets` — per-cell DOF count offset within the partition
- `_basisReconciliation` — shared `BasisReconciliation` instance
- `_spaceDim` — cached spatial dimension
- `_elementTypeCache` — `map<GlobalIndexType, ElementTypePtr>` — per-cell element type cache

## 19. Known Issues and Technical Debt

### 19.1 `_termsTraced` Cache Safety
The `_termsTraced` cache in `BasisReconciliation` stores raw `Function*` pointers, which are explicitly noted as not perfectly safe.

### 19.2 `getConstraintPermutation()` Overload Unimplemented
The overload `getConstraintPermutation(cellID, sideOrdinal, subcdim, subcord)` is declared but throws `std::invalid_argument` when called. This limits orientation handling for certain side-constrained H(curl)/H(div) configurations.

### 19.3 Cache Redundancy
The six overlapping caches should be redesigned as a unified `DofIndexRegistry` with clear invalidation semantics. The acknowledged overlap between `_ownedGlobalDofIndicesCache` and `_globalDofIndicesForCellCache` is a source of confusion.

### 19.4 Unimplemented Method
The overload `getConstraintPermutation(cellID, sideOrdinal, subcdim, subcord)` is declared but not implemented; it always throws.

## 20. Comparison with MFEM and hp3d Constraint Handling

### 20.1 MFEM

| Feature | Camellia | MFEM |
|---------|----------|------|
| **Constraint representation** | `CellConstraints` per cell | `ConstraintMatrix` (sparse, global) |
| **Orientation** | Per-subcell permutation | Orientation array per entity |
| **h-constraint weights** | `SubBasisReconciliationWeights` | `GetConformingInterpolation()` |
| **p-constraint** | Via `BasisReconciliation` | `BuildConformingInterpolation()` / `GetConformingProlongation()` |
| **Constraint representation** | Per-cell lazy evaluation | Global precomputed matrix |
| **MPI** | Non-symmetric peer-to-peer | Collective via `ParFiniteElementSpace` |

### 20.2 hp3d

hp3d uses a similar subcell-based ownership model, but constraint weights are stored as dense local matrices per element rather than per-subcell, and the ownership is resolved at assembly time rather than at DOF numbering time.

## 21. Summary: End-to-End DOF Mapping Flow

1. `GDAMinimumRule::rebuildLookups()` triggers the DOF rebuild.
2. `getCellConstraints(cellID)` computes (lazily) the per-cell constraint records.
3. `getOwnedGlobalDofIndices(cellID)` returns the DOF ranges owned by this cell.
4. `getBasisMap(cellID, ..., var)` assembles a `BasisMap` by:
   - **Unconstrained DOFs**: permutation-only mapper → global indices.
   - **Constrained DOFs (h-constraint)**: weight-matrix mapper → `W * u_coarse`.
   - **Constrained DOFs (p-constraint)**: weight-matrix mapper → similar but for polynomial-order mismatch.
5. `LocalDofMapper::mapLocalToGlobal()` applies the `BasisMap` to scatter local element vectors/matrices into the global system.