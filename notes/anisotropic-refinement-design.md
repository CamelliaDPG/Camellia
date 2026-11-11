# Anisotropic refinement for high-dimensional hypercubes: design note

Status: prototype.  Nothing in the existing mesh, constraint or solver code has been changed.

- `src/include/BoxForest.h` (`unit_tests/BoxForestTests.cpp`): the mesh.
- `src/include/BoxDPG.h`, `src/Mesh/BoxDPG.cpp` (`unit_tests/BoxDPGTests.cpp`): a serial DPG discretization and solve
  on a `BoxForest`, for bilinear forms whose trial variables are L2 fields and L2 fluxes, using the library's own
  element matrices (`BF::stiffnessMatrix`, `IP::computeInnerProductMatrix`, `RHS`) without a `Mesh`.

## What the existing code would need

Findings from reading the branch (`316-tensor-native-mesh-for-planar-mitl-and-more`):

- `RefinementPattern::anisotropicRefinementPattern()` builds nothing for hypercubes beyond the 2D quad, and for tensor
  topologies handles only "all base dimensions or none".
- `MeshTopology` identifies entities by sorted vertex sets and records parent/child entities per refinement pattern.
  Two refinement histories that reach the same geometry (x then y, y then x, isotropic) produce different entity
  genealogies; the "related recipes" exist to reconcile them, are defined for the quad only, and are consumed in two
  places (`MeshTopology.cpp` ~3722, ~3864) with TODOs on both.  The number of recipes grows like the number of ordered
  set partitions of the refined directions, so this does not extend comfortably to 4D.
- `GDAMinimumRule` assumes the constraining entity belongs to an *active* cell (see the comments near lines 98-106,
  1511, 1848).  With anisotropic neighbours that is false: if one neighbour is bisected in y and the other in z, the
  minimum-rule trace on their interface is a single polynomial on the whole parent face, which is no active cell's
  face.  The comments suggest inducing a regular refinement there, which would propagate isotropy.
- The Vlasov phase-space formulation (`VlasovSpaceTimeFormulation`) has an L2 field `f` and a single L2 flux
  `t_n_hat`.  Its only inter-element coupling is therefore through (d-1)-dimensional faces.  Vertices, edges and
  2-faces of the 4D cells carry no degrees of freedom, yet they are most of what `MeshTopology` stores and most of
  what `GDAMinimumRule` reasons about.

Making the existing path anisotropic in 4D means finishing the patterns, generalising recipes, and relaxing the
active-constraining-cell assumption throughout `GDAMinimumRule`: work in the two largest and least-loved files
(4231 and 3610 lines), in the direction of more special cases.

## Proposal: cells are dyadic boxes

A hypercube cell obtained from a root cell by bisections is an axis-aligned box whose extent in each direction is a
dyadic sub-interval of the root's.  Identify the cell by that geometry: integer lower coordinates and a level per
direction (`BoxForest::Box`; this is `TensorMeshTopology::TensorCellTuple`, a tuple of `ComponentCellID1D`, in other
clothes, without the requirement that the active set be a full Cartesian product).  Then:

1. **No entity storage.**  A sub-entity of a cell is the same tuple with some intervals collapsed to points.
   Equality, containment and adjacency of cells and entities are integer comparisons.
2. **No refinement patterns, no recipes.**  Refining in a set of directions halves those intervals.  The result does
   not depend on the order of the cuts, because identity is geometric rather than genealogical.  The children of a
   candidate refinement are a pure function of the box (`Box::children(mask)`), available without touching the mesh,
   as the local refinement indicator needs.
3. **The refinement tree is only a spatial index.**  Each node records which directions were cut; a point or box
   query descends it.  Face neighbours are one such query.
4. **Minimum rule for L2 traces, in any dimension, without a regularity assumption.**  On an interface plane the
   cells on each side tile the plane with their faces.  A single-valued function that is a polynomial on every cell
   face is one polynomial on each connected set of mutually overlapping faces.  Call that set a *patch*; it has one
   block of trace degrees of freedom.  Patches are found with a union-find over faces (`BoxForest::skeleton()`).
   For isotropic 1-irregular meshes the patch is the coarse face, reproducing the present behaviour; for the y|z
   example above it is the whole parent face, with no special handling.
5. **Constraints are Kronecker products of 1D matrices.**  A cell face's trace coefficients are the patch polynomial
   restricted to the face.  In each tangential direction that is either the identity or the restriction of a 1D
   Legendre expansion to a sub-interval (`BoxForest::legendreRestriction`).  This replaces `BasisReconciliation` and
   the constraint tree for this case, and applies by sum factorisation.  p-anisotropy enters as a per-direction
   minimum degree over the patch.
6. **Transfer between any two nested meshes** (solution projection, GMG prolongation) is by containment: the coarse
   active cell containing a fine one, and again a Kronecker product of 1D restrictions.  No refinement history is
   consulted, so anisotropic refinement in the hierarchy is not a special case.

### What a mesh-regularity rule is still for

Nothing above needs 1-irregularity for correctness.  A rule is still wanted for approximation quality and conditioning:
with arbitrary refinement, chains of overlapping faces merge into large patches (in the prototype's unbalanced random
4D test, up to 96 faces in one patch, and 0.4% of patches are L-shaped unions rather than boxes).  The natural rule is
per direction: across any face, the levels of the two neighbours in each tangential direction differ by at most one.
A stricter optional rule forbids crossings (requires one side's face tiling to refine the other's), at the cost of
some isotropy propagating from where differently-oriented refinements meet.  Both are simple loops over
`faceNeighbors`.  Which to enforce is a decision to make with the indicator work, since it affects how local a
refinement stays.

### Continuous (H1) traces

The same construction extends to lower-dimensional entities, but the bookkeeping is no longer just faces: an edge of a
fine cell can lie inside a face or an edge of several coarser cells.  With the "no crossings" rule the constraining
entity is always an entity of an active cell and the constraint is again a Kronecker product of 1D restrictions; with
crossings allowed one needs the lower-dimensional analogue of patches.  This is not needed for the Vlasov phase-space
mesh.  The configuration-space (x,t) mesh that carries the Maxwell traces is a separate 2D mesh and can stay on
`MeshTopology` / `GDAMinimumRule`.  I would do H1 second, and only on evidence that it is wanted in d >= 3.

## Prototype results

`BoxForest.h` is about 350 lines, standard library only.  Checks (in `unit_tests/BoxForestTests.cpp`):

- 2D, neighbours bisected in x and in y; 3D, neighbours bisected in y and in z: the interface is one patch; the
  Kronecker constraint reproduces the patch polynomial on each cell face to 3e-15.
- x-then-y, y-then-x and isotropic refinement give identical active cells and patches.
- An L-shaped patch is detected as a non-box union.

Timing on one core (random refinement masks, levels up to 3, no balancing):

| d | active cells | refine | skeleton (all neighbours and patches) | patches | peak RSS |
|---|-------------:|-------:|--------------------------------------:|--------:|---------:|
| 3 | 101,935      | 0.06 s | 0.24 s                                | 231,341 | 74 MB    |
| 4 | 100,531      | 0.01 s | 0.22 s                                | 236,575 | 114 MB   |

72 bytes per tree node (about 9 MB for the 4D tree; most of the peak is the patch lists).

For scale, the existing `MeshTopology` on a 4D extruded mesh (interval x 3 extrusions, `Line_2 x Line_2 x Line_2 x
Line_2` cells, same machine, one core, topology only, no dof assignment):

| cells   | build  | RSS    | then refine (isotropic, the only kind available) |
|--------:|-------:|-------:|--------------------------------------------------|
| 10,000  | 3.5 s  | 90 MB  | 500 cells: 5.8 s (12 ms per cell)                |
| 104,976 | 80 s   | 849 MB | 2000 cells: 42 s (21 ms per cell), RSS 1.2 GB    |

The comparison is not like for like: `MeshTopology` stores every vertex, edge, 2-face and 3-face and their
relationships, which `GDAMinimumRule` needs for continuous variables, and `BoxForest` computes only what an L2 field
plus an L2 flux needs.  That difference is the point of the proposal, but it is also why the H1 case is the real test
of whether the approach stays small.

## DPG solves on box meshes (`BoxDPG`)

`BoxDPG` numbers field dofs per active cell and flux dofs per patch, maps each cell's side coefficients to its
patches' polynomials, assembles `B G^{-1} B^T` and solves with KLU2.  Cells are quads, hexes, and
`Line x Line x Line x Line` in 4D (what the extruded phase-space meshes use).  The side-to-patch map is computed as an
L2 projection with the side basis as Camellia provides it, so it is independent of that basis's ordering and scaling;
it is exact, and it is the place where the Kronecker form would go for speed.  The flux of a patch is that of the
cell below the patch's plane (side parity +1 there, -1 above).

Tests, with ultraweak steady convection (`-(u, beta . grad v) + <t_n, v>`, graph norm, `delta_k = 2`):

- A polynomial solution of the trial degree is reproduced to 1e-11 (observed about 1e-13) in 2D (p = 2), 3D (p = 2)
  and 4D (p = 1), on meshes with non-uniform roots, two levels of anisotropic refinement, and neighbours bisected in
  different directions (a patch that is no cell's face); energy errors vanish.
- The same on a mesh with an L-shaped patch, and on a 4D (x, t, u_x, u_z) mesh refined in the velocity directions
  only.
- For u = exp(2y)(1 + x) in 2D with p = 2, bisecting in y alone converges at rates 2.74, 3.01, 2.99.
- Global dof count against a hand count.
- Against `Mesh` / `GDAMinimumRule` / `Solution` on a mesh both can represent (2 x 2 quads, p = 2, before and after
  refining one cell isotropically, so with hanging nodes): the same number of global dofs (72, then 117), the same L2
  error to 1e-5 relative and the same energy error to 1e-7 relative, for a non-polynomial solution.

`BoxForest::balance()` enforces the per-direction rule (tangential levels of face neighbours differ by at most one) by
bisecting only in the offending direction; tested on a column refined three times in z, which induces z-only
bisections in its neighbours.

Cost, 4D with p = 1 (80 trial and 625 test dofs per cell), one core: about 35 ms per cell for the whole solve
(256 cells, 14,592 dofs: 9 s).  Nothing here is optimised: the optimal test functions and the side-to-patch
projections are computed with plain loops, cell by cell, and fields are not condensed out before the global solve.

Not done yet: parallel assembly; static condensation and iterative solvers; solution transfer between meshes.

## Scoping: a box mesh under `Mesh` / `Solution`, and under the Vlasov formulation

Question: what must a box-mesh topology, its cells and its dof assignment present so that the existing `Solution`,
and then `VlasovSpaceTimeFormulation`, run on it?  Method: a tally of every `MeshTopologyView` / `MeshTopology` /
`Cell` method called outside the core mesh classes, then a read of each consumer.  Line numbers are at 62d349c2.

### Shape of the adapter

There is a precedent: `SubExtrudedMeshTopology` subclasses `MeshTopologyView` with composite cell indices and its own
`SubExtrudedCell`, and has no base `MeshTopology`.  The box version is three classes:

- `BoxMeshTopology : MeshTopologyView`, wrapping a `BoxForest` and per-direction root vertices.  Cell ID = tree node
  index (dense, 32-bit `GlobalIndexType` is enough).  Replicated on every rank (`Comm()` null, `isDistributed()` false),
  which is also how the phase-space topology is used today (`allowPruning = false`, VlasovSpaceTimeFormulation.cpp:1682).
- `BoxCell : Cell`, overriding the neighbour, boundary, child and entity methods.
- `BoxGDA : GlobalDofAssignment`, holding what `BoxDPG` computes now (numbering, side-to-patch maps).

About 15 `MeshTopologyView` virtuals must be overridden because the defaults dereference the null base topology or
private counters (`getCell`, `getDimension`, `cellCount`, `activeCellCount`, `isValidCellIndex`, `getCellCentroid`,
`physicalCellNodesForCell`, `verticesForCell`, `getMyActiveCellIndices`, `getLocallyKnownActiveCellIndices`,
`getPeriodicBCs`, `Comm`, `isDistributed`, `isParent`, `getRootCellIndicesLocal`).  The dof assignment has 14 pure
virtuals plus 3 from `DofInterpreter`; `Solution` calls nothing that exists only on `GDAMinimumRule`.

### Stage 1: solve and evaluate on a fixed box mesh.  Small.

Caller changes needed (all verified in the source):

| Where | Problem | Change |
|---|---|---|
| Mesh.h:57, Mesh.cpp:55-190 | every public constructor does `new GDAMinimumRule`; the one taking a GDA is private and skips partitioning | a public constructor taking a GDA factory |
| GlobalDofAssignment.h:93, .cpp:155-282 | `assignParities` is non-virtual and uses a reciprocal peer neighbour, side entity indices, and the parent's `RefinementPattern`; it is called from the base constructor | make side parities virtual; box rule: +1 on the boundary and on upper faces, -1 on lower faces |
| Boundary.cpp:71-74 | `bcsToImpose` throws for any non-`MeshTopology`, even with no tag BCs | move the cast inside the tag-BC branch (or see stage 3) |
| Cell.cpp:867-876 | `boundarySides()` is non-virtual and reads stored neighbours, unlike the virtual `isBoundary()` | loop over `isBoundary()` |
| Cell.h | `getActiveNeighborIndices`, `getDescendantsForSide`, `refinementPattern`, `getParent`, `level` are non-virtual and read stored data | make the first two virtual; supply parent and level through `setParent` |

With those, assembly (`interpretLocalData`), legacy spatial-filter BCs, `importSolution`, energy error, `Function::solution`,
`l2norm`, point evaluation, Zoltan HSFC partitioning (centroids only) and `HDF5Exporter` work unchanged.  MPI comes
for free in replicated mode: `CellDataMigration` skips geometry packing when the topology is not distributed.

### Stage 2: refinement.  Moderate.

- `Mesh::hRefine` casts to `MeshTopology` and takes a `RefinementPattern` (Mesh.cpp:808).  Needs a new entry point,
  "refine these cells in these directions", and a virtual on the topology; `RefinementObserver` signatures carry
  `MeshTopologyPtr` and `RefinementPatternPtr`.
- Parent-to-child solution projection uses `parentCell->refinementPattern()->mapSubcellOrdinalFromChildToParent`
  (Solution.cpp:4811) and `GlobalDofAssignment::projectParentCoefficientsOntoUnsetChildren` (.cpp:1077-1137).  For
  boxes this is the Kronecker restriction; `BoxGDA` should own it.
- `RefinementStrategy` and `Mesh::enforceOneIrregularity` (Mesh.cpp:347-464) are entity- and pattern-based and choose
  the regular pattern.  Replace with a box-aware strategy plus `BoxForest::balance()`.  Energy and function error
  indicators work unchanged.  Unrefinement: leave unsupported at first.

### Stage 3: the Vlasov formulation.  The bulk of the work, and mostly not about `Solution`.

1. **Boundary tags.**  All Vlasov BCs are attached through entity sets of side entity indices
   (`getBoundarySidesThatMatch`, `createEntitySet`, `applyTag`; VlasovSpaceTimeFormulation.cpp:1817-1857), imposed in
   Boundary.cpp:71-130, and read again in `configureNextSlab` (:1387) and `Function::meshSideSetCharacteristic`.  A box
   mesh answers "is this side of this cell on root face d-min / d-max" by arithmetic.  Change: one virtual
   `sideMatchesTag(cellID, sideOrdinal, tag)` on the topology view, and route those three consumers through it.
   The inflow set is exact as "x-min faces with u_x >= 0, x-max faces with u_x <= 0" because the driver forces a vertex
   at u_x = 0.
2. **Phase-space to configuration-space maps.**  Ten `dynamic_cast<ExtrudedMeshTopology*>` sites (moments,
   `PhysicalSolutionExtrusion`, `TensorVlasovMomentSampling1D2V`, visualization) ask for the configuration cell under a
   phase cell and the cell's velocity extents.  Change: a small projection interface implemented by both
   `ExtrudedMeshTopology` and the box topology.  The velocity extents are the box's intervals.
3. **The configuration mesh is the phase mesh's base, and is refined in lockstep** (ExtrudedMeshTopology.cpp:203-219,
   `PhysicalMeshUpdater` :434-464).  The moment code needs each phase cell's (x,t) projection to be a node of the
   configuration refinement tree.  Consequences:
   - velocity-only refinement of phase cells needs nothing from the configuration mesh;
   - the moment code does not need a full tensor product of velocity cells over a configuration cell, so different
     velocity refinement at different x is fine for the paths the MITL driver uses
     (`RelativisticFirstMomentFunction`, not on that path, would double count);
   - refinement in x or t must be mirrored in the configuration `MeshTopology`, which stays pattern-based.  Two phase
     cells over one configuration cell cut in different orders (x then t, t then x) cannot both be tree nodes there.
     Either restrict (x,t) refinement of phase cells to what the configuration tree can mirror, or change the moment
     code to "configuration cells overlapping the projection, with an affine sub-box map".
4. **Slab-to-slab transfer** (`MeshTools::hRefineToMatchAtBoundary`, :92-301) matches vertices and side entities and
   refines isotropically until sides are one to one.  Replace with a match of t-max faces to t-min faces by their
   (x, u) intervals.
5. **Multigrid.**  The h-hierarchy (`getView`, `RefinementBranch`, `BasisReconciliation`; GMGOperator.cpp:1140-1737) must
   be replaced for box meshes; the Kronecker transfer is the replacement.  The p-only path needs little
   (`getActiveNeighborIndices`, parities, the interpret methods) if the caller supplies the mesh list.  The MITL
   driver turns GMG on above a dof threshold, so this is needed for production sizes.
6. Not needed for the MITL path: static condensation (never enabled), `MeshTransferFunction`, save/load of the mesh
   (must be replaced if wanted), phase-space visualization.

### Assessment

Stages 1 and 2 are contained: about a thousand lines of new code and half a dozen small edits to existing classes,
with `BoxDPG` as the reference to test against.  Stage 3 is where the cost is, and items 1-4 are changes to the Vlasov
code rather than to the mesh.  The first useful target that avoids the hard part of item 3 is **velocity-only
refinement of the phase-space mesh**: it needs stages 1-2 and items 1, 2, 4 and 5, and leaves the configuration mesh
and the Maxwell solve alone.

### What has been implemented (stages 1-3)

**Stage 1 and 2: `BoxMeshTopology`, `BoxCell`, `BoxGDA`.**  `BoxGDA::mesh(topology, bf, H1Order, delta_k)` gives a `Mesh`
on which the existing `Solution` runs.  Edits to existing classes (no change for `MeshTopology`-based meshes):

- `Mesh`: a public constructor taking a GDA factory; `hRefine(std::map<cellID, directionMask>)`; `hRefine(cellIDs)` and
  `enforceOneIrregularity()` work on topologies without `RefinementPattern`s (all directions; the box rule).
- `MeshTopologyView`: virtuals `getCellSideParities`, `getSidesMatchingTag`, `refineCellInDirections`,
  `cellsToRefineForRegularity`, each with a default.
- `GlobalDofAssignment`: parities from the topology when it supplies them; virtual
  `projectParentCoefficientsOntoChildren`, consulted by `Solution::projectOldCellOntoNewCells`.
- `Boundary` and `Function::meshSideSetCharacteristic`: tagged sides from the topology when there are no entity sets.
- `Cell`: `boundarySides()` through the virtual `isBoundary()`; virtual `getDescendantsForSide` and
  `getActiveNeighborIndices`; `setChildCells`.
- `RefinementObserver`: `didHRefineInDirections`.

Side parities on a box mesh: +1 on boundary sides and on upper faces, -1 on lower faces.  The topology is replicated
on every rank; a rank owns the field dofs of its cells and the flux dofs of the patches whose first face belongs to
one of its cells, numbered contiguously per rank without communication.  On refinement, fields are restricted
exactly to the children; fluxes are restricted on child sides that lie on a parent side and are zero on sides
interior to the parent.

Tests (`BoxMesh` group, 1 and 4 ranks): polynomial reproduction through `Solution` in 2D, 3D and 4D with tag BCs on
meshes with crossed anisotropic refinements; agreement with a `MeshTopology` mesh with hanging nodes (spatial-filter
BCs); exact projection through `Mesh::hRefine`; directional `enforceOneIrregularity`; the existing
`RefinementStrategy` driving a box mesh.

**Stage 3: the Vlasov formulation.**  `VlasovSpaceTimeFormulation::setUseBoxMeshForPhaseSpace(true)` makes formulations
constructed afterwards build their phase-space mesh on a `BoxMeshTopology` (a static switch, to avoid threading an
argument through three constructors; it could become one).  The configuration-space mesh is unchanged.

- Boundary tags (initial time, spatial inflow, velocity bounds) are side selectors on the box topology.
- `PhaseSpaceTopology` is the small interface the moment and field-extrusion code needs (velocity extents of a phase
  cell; the configuration cell beneath it).  `ExtrudedMeshTopology` and `BoxMeshTopology` implement it;
  `TensorVlasovMomentSampling1D2V`, `PhysicalSolutionExtrusion` and the `SolutionMoments` constructors use it.
- `BoxConfigurationMeshUpdater` keeps the configuration mesh at least as fine as the phase-space mesh: when phase
  cells are bisected in x and t, the configuration cells beneath them are refined (regular refinement), then
  one-irregularity of the configuration mesh is enforced.
- `BoxGDA::hRefineToMatchAtBoundary` replaces `MeshTools::hRefineToMatchAtBoundary` between time slabs: each side is
  bisected only in the directions in which it is coarser than its counterpart.
- Geometric multigrid works on box meshes (see below).
- `PlanarMITLRelativisticSpaceTime`: `--useBoxMesh`, and `--boxVelocityRefinements=N` with
  `--boxVelocityBandCenter` / `--boxVelocityBandHalfWidth` for u_x-only bisection of the first slab in a band.
  Unrefinement (`setMaxElements`) is skipped on box meshes.

Supported refinement of phase-space cells: any subset of directions, including x alone or t alone.  The rule is that
the configuration mesh is everywhere at least as fine as the phase-space mesh, so the projection of a phase-space
cell onto (x,t) is tiled by active configuration cells:

- `BoxConfigurationMeshUpdater` refines the active configuration cells under a refined phase-space cell until they are
  as fine as its children in every configuration direction.  The configuration mesh is refined regularly (in x and t
  together), so it can end up finer than the phase-space mesh in one direction; it keeps its existing
  `MeshTopology`, patterns and `GDAMinimumRule`.
- `PhaseSpaceTopology::phaseSpaceActiveConfigurationCells()` gives the configuration cells that tile a phase-space
  cell's projection; the moment code deposits on those.  `phaseSpaceConfigurationCellID()` returns the single matching
  configuration cell when there is one, and -1 otherwise (a cell at different levels in x and t).
- `PhysicalSolutionExtrusion` evaluates E and B on such cells point by point, in the configuration cell containing
  each point (`valuesForUnmatchedCells`).  This is a host implementation, and it builds a `BasisCache` per
  configuration cell on each call; cells that do match a configuration cell take the existing path.

Tests (`BoxVlasov` group, relativistic Vlasov-Maxwell 1D2V, 1 and 4 ranks), each against the extruded mesh:

- same root mesh: identical dof counts; ||f||, ||E_x|| and the Vlasov energy error agree to 1e-15 relative;
- every box cell bisected in u_x only, against the extruded mesh with twice as many u_x cells: agree to 1e-15, and the
  configuration mesh is untouched;
- one cell refined in all four directions (hanging faces everywhere, configuration mesh refined beneath): agree to
  1e-15, with and without charge;
- two time slabs, the second slab's mesh refined in u_x to match at the interface: agree;
- local u_x-only and u_y-only refinement (hanging and crossed faces): runs, fields stay zero for zero charge;
- bisecting in x and t refines the configuration mesh beneath;
- every cell bisected in x alone, and in t alone, with zero charge: agree to 1e-15 with the extruded mesh that has
  twice as many x (or t) cells;
- a configuration field that is a polynomial in (x,t), extruded to a phase-space mesh with cells bisected in x alone
  and in t alone: reproduced to 5e-15;
- moments of the same polynomial f on a mesh bisected in x alone and on one bisected in x and t (same configuration
  mesh): the fields they drive agree to 1e-9;
- the coupled solve with every cell bisected in x alone, in t alone, and in both: all run and land near each other.

The MITL driver itself, on a tiny case (2 x 1 x 4 x 4 cells, p = 1, two slabs, space-charge-limited emission, 35 and 9
fixed-point substeps): `--useBoxMesh` and `--useExtrudedMesh` print the same log, apart from the last digit of two
residuals.  With `--boxVelocityRefinements=2` and a band, the first slab goes from 32 to 64 phase-space cells with the
configuration mesh still at 2 cells, and the second slab is refined to match (11,176 dofs in both).

**Geometric multigrid on box meshes.**  The existing `GMGSolver` / `GMGOperator` cycle, smoothers and coarse solves are
used unchanged; box meshes supply the two things that were built from refinement patterns:

- *Hierarchy* (`BoxGDA::meshesForMultigrid`, reached through `GMGSolver::meshesForMultigrid`): as for other meshes,
  the root cells at order `kCoarse`, one more refinement generation per level, then the fine mesh.  Each coarse mesh
  has its own `BoxMeshTopology`, a copy of the fine one truncated at a generation
  (`BoxMeshTopology::copyTruncatedAtDepth`); cells correspond by geometry, not by ID.  A coarse cell is assigned to the
  rank owning the fine cell at its lower corner.
- *Prolongation* (`BoxGDA::prolongationOperator`, used by `GMGOperator::constructProlongationOperator`): exact, since
  the coarse space is a subspace of the fine one wherever both are defined.  Fields: the coarse cell's polynomial in
  the fine cell's basis.  Fluxes: the coarse patch's polynomial restricted to the fine patch, as a Kronecker product
  of 1D Legendre restrictions, where the fine patch lies on the coarse skeleton; zero where it lies inside a coarse
  cell.  (The existing multigrid does the same for fluxes without a `termTraced`, which is the Vlasov case.)

Static condensation with multigrid is not supported on box meshes.

Tests: the prolongation maps the coarse solution vector of a polynomial problem onto the fine one to 1e-13, for
coarse meshes at each generation and coarse orders 0 and 1, in 2D and 3D; multigrid-preconditioned CG converges on
anisotropically refined meshes in 2D, 3D and 4D (30, 70 and 51 iterations to 1e-10 for pure convection with the graph
norm, with 120, 864 and 1312 dofs); on a 2D mesh with hanging nodes that both topologies can represent, box multigrid
takes 162 iterations where the existing multigrid takes 156; the Vlasov solve on a box mesh with local u_x-only
refinement agrees with the direct solve to 1e-13 relative.  All on 1 and 4 ranks.

Not done: unrefinement;
p-refinement; serendipity bases (the flux space must be a full tensor product); mesh save/load; periodic directions; a spatial mesh that is not a 1D set of root cells; the legacy
`SolutionMoments` paths other than construction (the tensor-native moment path is the one that works).

### Serendipity, direction selection, and adaptive MITL runs

- **Flux spaces.**  `BoxGDA` finds the flux polynomial space numerically: which products of Legendre polynomials the
  side basis reproduces.  Any space whose multi-indices form a downward-closed set works, since such a space is
  preserved under restriction to a sub-box.  Camellia's serendipity L2 space of degree p is of this kind: the
  products with sum over the directions with index i >= 1 of (i + 1) at most p + 1 (13 functions on a 3D face at
  p = 3, against 64 for the full tensor space).
- **Direction selection.**  `BoxGDA::anisotropicRefinementMasks(mesh, f, cells, threshold)`: on each cell, f is sampled
  at tensor Gauss points and expanded in Legendre polynomials; eta_k is the L2 norm of the part in the highest modes of
  direction k (top two for p >= 3, top one below); direction k is bisected if eta_k >= threshold * max eta.  This is a
  cheap heuristic, not the DPG local-residual indicator.  `masksTangentialToTaggedSides` gives, for resolving boundary
  data, every direction but the normal to the tagged side.  `TRefinementStrategy::setCellRefiner` lets a strategy
  refine its marked cells through such a function.
- **Driver.**  `--boxAnisotropicAdaptivity` (default on with `--useBoxMesh --useAdaptivity`), `--boxAnisotropyThreshold`,
  `--maxRefinementsFirstSlab`.  The adaptive branch now writes the density, phase-space and ACCURACY diagnostics (they
  were written only on fixed meshes), and its final export no longer passes moment functions to the exporter, which
  cannot evaluate them (that threw, on either mesh type).
- `drivers/papers/RelativisticVlasovSpaceTime/production-adaptive-box-phase-space.sh` is an adaptive counterpart of the
  x32 production run, with a CHECK mode and a comparison of ACCURACY integrals against that run's log.

Tests: polynomial reproduction, exact projection and multigrid with serendipity bases in 3D and 4D; the Vlasov solve
with serendipity bases and hanging faces against the extruded mesh (agree); direction selection on functions with
known anisotropy; for a layer in y, anisotropic marking reaches the isotropic run's error with 68 cells where the
isotropic run has 408.

Observed in small adaptive MITL runs (4 ranks, p = 3, dp = 1, 440-cell base mesh):

- Multigrid-preconditioned CG takes 40-60 iterations on the base mesh and about 200 on meshes with cells bisected in
  t alone or x alone (condition estimates near 1400 against 200).  Not investigated.
- The energy error is concentrated in a few cells at the cathode, so Doerfler marking at 0.5 marks one cell per step.
- Cells bisected in x alone or t alone take `PhysicalSolutionExtrusion`'s point-by-point path.  It was the dominant
  cost on the rank owning them until the configuration-cell tiling and `BasisCache`s were cached per cell (one-slab run:
  the last refinement's solve went from 146 s to 86 s, and the imbalance between ranks disappeared).
- `plot_phase_space.py` runs on the adaptive output, but its "int n dx from samples" (7.82) disagreed with the driver's
  `phase_reductions` int_f (1.99) in one run.  Unresolved.

### Cost on the MITL driver

`PlanarMITLRelativisticSpaceTime` on 4 ranks, one slab, three fixed-point substeps, multigrid on (`--timeMonitorSummary`
prints the timers).  Times are seconds; "constraint work" is per rank and adds the side-to-patch maps
(`BoxGDA::cellMap`), their application (`interpretLocalData`, `interpretLocalBasisCoefficients`) and the multigrid
prolongation; "legacy equivalent" adds `GDAMinimumRule::getDofMapper`, `interpretLocalData`,
`interpretLocalBasisCoefficients` and `interpretGlobalCoefficients`.

| case             | dofs    | wall, extruded | wall, box | box constraint work | legacy equivalent |
|------------------|--------:|---------------:|----------:|--------------------:|------------------:|
| p = 1, 256 cells | 42,112  | 47             | 42        | 1.4                 | 4.4               |
| p = 1, 1024 cells| 164,992 | 202            | 180       | 6.2                 | (not captured)    |
| p = 1, 2304 cells| 368,640 | 708            | 562       | 12.2                | (not captured)    |
| p = 2, 256 cells | 110,848 | 221            | 202       | 3.0                 | 16                |

The two larger extruded runs finished their solves and then aborted in the HDF5 export, before printing timers:
OpenMPI's file layer overflows a buffer on a long output path, which is a property of the scratch path used, not of
the code.  The dense side-to-patch projection is 2-3% of the run and already cheaper than what it replaces, so it is
not the next thing to optimise.  Element matrices and the multigrid-preconditioned solve dominate, and the solve
grows fastest (14, 70, 324 s on the box mesh).

If it is optimised later: the map from Camellia's flux side basis to tensor Legendre on a face is a dense matrix, but
it is the same for every cell up to a scalar (the face measure on hexes; no scaling on the 4D tensor-topology cells).
So the constraint factors as one fixed matrix per side ordinal, a scalar per cell, and a Kronecker product of 1D
restrictions that is the identity on conforming faces.

### Kokkos

None of this is on device today, and neither is `MeshTopology`.  The box representation is the device-friendly one:
the active cells are flat arrays (lower coordinates and levels), the skeleton is a CSR list of faces per patch, and
the constraints are small 1D matrices indexed by (level difference, child position).  Building the tree and the
skeleton stays on the host; assembly and operator application would consume those arrays as Views.  The dense
side-to-patch projection in `BoxDPG` is the piece to replace by the Kronecker form before then.

## Getting from prototype to production

In increasing order of commitment; each step is useful on its own.

1. **Box core** (this prototype, hardened): periodic directions, physical coordinates from per-direction root vertex
   arrays, unrefinement, balance rules, a distributed version (space-filling-curve partition of the leaves; ghost
   layer by the same face query).  Roots are a Cartesian grid, which covers the phase-space meshes built from vertex
   arrays.  An unstructured base mesh times 1D components would need root-to-root adjacency supplied by the base mesh.
2. **A dof interpreter for L2 field + L2 flux** on a `BoxForest`: numbering (fields per cell, fluxes per patch),
   `interpretLocalData` / `interpretGlobalCoefficients` by Kronecker constraints.  `DofInterpreter` is already an
   abstract interface that `Solution` talks to.
3. **Use it under the existing `Mesh` / `Solution`** for the Vlasov phase-space mesh.  The entity-index API of
   `MeshTopologyView` is used almost only by `MeshTopology`, `GDAMinimumRule`, `Cell` and the view itself; the rest of
   the library mostly asks for cells, their vertices, their neighbours and their sides.  The open question is how
   much of `Cell` / `MeshTopologyView` an adapter must present for `Mesh`, `BasisCache`, the partition policy, and
   `ExtrudedMeshTopology`'s component maps (which become trivial: the component cell is the projection of the box).
   This step needs a scoping pass before committing to it.
4. **Tensor-native stack**: `TensorMeshTopology` already has a constructor taking an explicit list of active tuples.
   Backing it by a `BoxForest` gives the tensor-native operators adaptive, anisotropic meshes with the same patch
   and Kronecker-constraint structure, which is the form sum-factorised assembly wants.

What I would not do: finish `anisotropicRefinementPattern` and the recipes for hexes and tesseracts.
