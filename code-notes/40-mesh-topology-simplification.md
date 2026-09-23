# MeshTopology Simplification Opportunities: Comprehensive Technical Analysis

## Executive Summary

Camellia's `MeshTopology` class (167 KB implementation, 21 KB header) maintains a complex hierarchy of nested containers to track mesh entities, topological relationships, refinement hierarchies, and boundary information. This analysis identifies five major simplification opportunities:

1. **`_boundarySides` elimination** - A TODO-flagged `set<IndexType>` that is semantically incorrect in distributed contexts; derivable on-demand from `_cellsForSideEntities`
2. **`_cells` container design** - Tradeoff between `map<GlobalIndexType,CellPtr>` (current) and a dense `vector<CellPtr>` (cache-friendly)
3. **`_knownEntities` lookup bottleneck** - `map<vector<IndexType>,IndexType>` imposes O(n log m) complexity; hash-based alternative exists
4. **`_activeCellsForEntities` design uncertainty** - Header comment explicitly flags unresolved indecision about whether ancestral cells are tracked
5. **`_parentEntities` / `_generalizedParentEntities` redundancy** - Two parent-tracking structures with overlapping semantic domains

---

## Current Data Structure Inventory

The `MeshTopology` class inherits from `MeshTopologyView` and contains approximately 15+ private data members.

### Private Members (from MeshTopology.h lines 65-90)

**`_entities`** - `vector< vector< vector<IndexType> > >`: Primary entity storage indexed by [dimension][entityIndex]. Innermost vector contains sorted vertex indices. Triple-nested structure incurs multiple allocation levels.

**`_knownEntities`** - `vector< map< vector<IndexType>, IndexType > >`: Inverse mapping from sorted vertex-index vectors to entity indices. MAJOR BOTTLENECK: O(n log m) combined complexity where n=entity vertex count, m=entity count at that dimension.

**`_canonicalEntityOrdering`** - `vector< vector< vector<IndexType> > >`: Canonical orderings of sub-entity vertices for conforming mesh orientation. Immutable after creation.

**`_activeCellsForEntities`** - `vector< vector< vector< pair<IndexType, unsigned> > > >`: Active cell membership per entity. DESIGN UNCERTAINTY: header comment states verbatim: "I'm vascillating on whether this should contain entries for active ancestral cells. Today, I think it should not."

**`_sidesForEntities`** - `vector< vector< vector<IndexType> > >`: Side entity indices for cells containing each entity, indexed by [dimension][entityIndex].

**`_cellsForSideEntities`** - `vector< pair< pair<IndexType,unsigned>, pair<IndexType,unsigned> > >`: Two adjacent cells per side entity as (cellIndex, sideOrdinal) pairs. Initialized with sentinel (-1,-1) for absent second cell. This is the ground-truth for boundary detection.

**`_boundarySides`** - `set<IndexType>`: Side entities on mesh boundary. TODO in MeshTopology.cpp:538: "Consider getting rid of _boundarySides altogether. With distributed MeshTopology, it contains sides of ghost cells that do not lie on the boundary. The right way: a side is on the boundary if it belongs to a cell we own AND that cell has no neighbor on that side."

**`_parentEntities`** - `vector< map< IndexType, vector< pair<IndexType,unsigned> > > >`: Multiple possible parent entities for refined entities.

**`_generalizedParentEntities`** - `vector< map<IndexType, pair<IndexType,unsigned> > >`: Single generalized parent per entity. Semantically overlaps with `_parentEntities`.

**`_entityCellTopologyKeys`** - `vector< map< Camellia::CellTopologyKey, RangeList<IndexType> > >`: Cell topology keys per entity.

**`_cells`** - `map<GlobalIndexType, CellPtr>`: Cells known on this MPI rank. Replaced an earlier `vector<CellPtr>` (commented out in header). Accompanied by `_validCells: RangeList<GlobalIndexType>` for faster range lookups.

**Periodic BC structures**: `_periodicBCIndicesMatchingNode`, `_equivalentNodeViaPeriodicBC`, `_canonicalVertexPeriodic` - track vertex equivalences under periodic boundary conditions.

---

## Key Simplification Opportunities

### A. Eliminate `_boundarySides`

The code's own TODO at MeshTopology.cpp:538 makes the case. In distributed meshes, ghost-cell sides contaminate `_boundarySides`. The authoritative source already exists:

    inline bool isBoundarySide(IndexType sideIndex) const {
      if (sideIndex >= _cellsForSideEntities.size()) return false;
      return _cellsForSideEntities[sideIndex].second.first == (IndexType)-1;
    }

Benefits: zero memory overhead, correct distributed semantics, O(1) query, no ghost-side erase operations needed.
Effort: Medium - audit ~20 `_boundarySides` access sites in MeshTopology.cpp.

### B. `_cells`: Map vs. Vector

The evolution from `vector<CellPtr>` to `map<GlobalIndexType,CellPtr>` suggests cell IDs became non-contiguous. The parallel `_validCells: RangeList<GlobalIndexType>` was added for faster lookups.

Tradeoffs:
- map: O(log n) lookup, cache-hostile iteration, handles sparse IDs
- vector: O(1) lookup, cache-friendly iteration, requires dense IDs

Recommendation: Profile cell-ID density after adaptive refinement. If >50% dense, consider `vector<unique_ptr<Cell>>` with skip-list for holes. If sparse, `absl::flat_hash_map` over `std::map` for better cache behavior.

### C. Hash-Based `_knownEntities`

Current: `map< vector<IndexType>, IndexType >` at O(n log m) per lookup.

Proposed: `unordered_map< vector<IndexType>, IndexType, VectorHash >` where VectorHash uses FNV-1a or similar over sorted vertex indices. Expected 2-5x speedup in mesh construction. Risk: hash collision pathology; test with degenerate meshes.

### D. Resolve `_activeCellsForEntities` Ambiguity

The header comment explicitly flags this as unresolved. Resolution:
1. Audit all read sites in GDAMinimumRule.cpp and MeshTopology.cpp
2. Determine which model each site assumes (direct vs. ancestral)
3. If both needed: introduce separate `_directActiveCellsForEntities` and `_ancestralActiveCellsForEntities`
4. If one needed: rename to express intent, remove the comment

### E. Unify `_parentEntities` and `_generalizedParentEntities`

These track overlapping parent-child relationships with different multiplicity. A unified approach:

    enum class ParentKind { Refinement, Generalized };
    struct ParentEntry { IndexType parentEntityIndex; unsigned subcordInParent; ParentKind kind; };
    vector< unordered_map<IndexType, vector<ParentEntry>> > _parentRelationships;

---

## Comparison to MFEM NCMesh

MFEM's `NCMesh` (ncmesh.hpp, 57 KB) takes a fundamentally different approach:

| Feature | Camellia MeshTopology | MFEM NCMesh |
| --- | --- | --- |
| Entity storage | Flat indexed arrays by dimension | Tree nodes in refinement hierarchy |
| Entity deduplication | map over vertex set | Containment-based (geometry defines entity) |
| Active-cell tracking | Explicit `_activeCellsForEntities` | On-demand tree traversal |
| Boundary detection | `set<IndexType> _boundarySides` | Computed from tree leaf nodes |
| Parent-child | Two overlapping maps | Explicit tree parent pointer |
| Cache locality | Poor (deep nested maps) | Better (space-filling curve ordering) |

Why Camellia differs: flat entity indexing integrates with DPG DOF assignment; explicit deduplication supports periodic BCs; design predates NCMesh maturity (~2010 era).

Hybrid recommendations: adopt lazy `_activeCellsForEntities` evaluation; use hash-based deduplication; consider space-filling curve entity ordering.

---

## Recommended Refactoring Plan

### Phase 1: Low-Risk, High-Impact (2-3 weeks)

1. **Eliminate `_boundarySides`**: Replace ~20 access sites with `isBoundarySide()` method
2. **Name the pair types**: Define `struct CellSideInfo { IndexType cellIndex; unsigned sideOrdinal; }` to replace nested `pair<IndexType,unsigned>` in `_cellsForSideEntities`
3. **Document parent structures**: Add header comments explaining `_parentEntities` vs. `_generalizedParentEntities` semantic difference

### Phase 2: Moderate-Risk, High-Benefit (4-6 weeks)

1. **Hash-based `_knownEntities`**: Implement `VectorHash`, replace `map` with `unordered_map`, benchmark on representative meshes
2. **Resolve `_activeCellsForEntities` ambiguity**: Audit call sites, decide on model, rename or split
3. **Profile and optimize `_cells`**: Measure cell-ID density, evaluate vector vs. flat_map migration

### Phase 3: Structural Refactoring (8-12 weeks)

1. **Unify parent structures**: Merge `_parentEntities` and `_generalizedParentEntities` into typed `_parentRelationships`
2. **Lazy topology evaluation**: Adopt MFEM NCMesh pattern for `_activeCellsForEntities`
3. **Space-filling curve ordering**: Improve iteration performance for large meshes

---

## Risk Assessment

| Change | Risk | Benefit | Priority |
| --- | --- | --- | --- |
| Eliminate `_boundarySides` | Low | Medium-High | P1 |
| Hash-based `_knownEntities` | Medium | High | P1 |
| `CellSideInfo` struct rename | Low | Low | P2 |
| `_activeCellsForEntities` clarification | Medium | Medium | P1 |
| Parent structure unification | High | Medium | P3 |
| `_cells` vector migration | Medium | Medium | P2 |
| Lazy topology evaluation | High | High | P3 |

**Key constraint:** MeshTopology is used across MPI-distributed contexts. Any change to distributed-facing members (`_cells`, `_ownedCellIndices`, `_boundarySides`) requires careful testing with distributed regression tests.
