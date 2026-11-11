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
