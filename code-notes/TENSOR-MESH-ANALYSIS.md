# Tensor Mesh Analysis

## Scope

This note assesses the tensor-oriented code currently on the shelf, with emphasis on whether it is ready to support high-order, performance-portable, sum-factorized assembly and matrix-free operators for the Planar MITL relativistic Vlasov target.

Files reviewed include the new tensor-oriented classes (`ComponentMeshTopology1D`, `TensorMeshTopology`, `TensorMesh`, `TensorVar`, `TensorVarFactory`, `TensorVarDiscretization`, `TensorDofLayout`, `TensorFunction`), the older `TensorBasis`, and the current Vlasov/MITL driver path (`VlasovSpaceTimeFormulation`, `RelativisticVlasovSpaceTimeFormulation`, `PlanarMITLRelativisticSpaceTime.cpp`).  Several shelved Trilinos/PAMatrix and matrix-free example files were present, but FileShelf reported UTF-8/text extraction failures for those and the PDFs in this run, so this analysis relies primarily on the readable Camellia sources plus the design intent stated in the task.

## High-level status

The current tensor code is a useful metadata/prototype layer, not yet an assembly/operator layer.  It can describe a mesh as a Cartesian product of 1D component meshes, register tensor-level variables, attach per-component polynomial-order deltas, compute tensor-product local DOF shapes, and represent some coefficient functions with dependency/coupling metadata.  It does not yet provide global numbering, distributed maps, boundary-condition enforcement, residual/Jacobian assembly, DPG Gram/B matrices, static condensation, sum-factorized kernels, matrix-free apply, or solver integration.

For the Planar MITL Vlasov target, the existing production path still builds a conventional extruded `MeshTopology` and `Mesh`, constructs a legacy `BF`, sets a legacy `IP`, and calls `Solution::solve()`.  `RelativisticVlasovSpaceTimeFormulation` explicitly disables sum-factorized Gram assembly.  The tensor-native layer is therefore not in the solve path yet.

## Current tensor topology model

### ComponentMeshTopology1D

`ComponentMeshTopology1D` represents one 1D component mesh.  It stores:

- coarse vertices in `std::vector<double>`;
- one `ActiveLeafSet1D` per root interval;
- active cells as `std::vector<ComponentCellID1D>`;
- per-cell p orders in `std::map<ComponentCellID1D,int>`;
- active-cell ordinals in `std::map<ComponentCellID1D,IndexType>`;
- Kokkos dual views for vertex coordinates, cell-to-vertex connectivity, and vertex-to-cell CSR-like incidence.

The class has a partial Kokkos orientation: topology coordinate/connectivity arrays are mirrored to device in `rebuildVertexIncidence()`.  However, the public API and most implementation choices are still host-centric.  Active cells, poly orders, lookup maps, and vertex incident cell descriptors are STL containers.  `containsCell()` recomputes active cells for a root and linearly scans them.  `activeCellsForVertex()` appears incorrect for per-vertex semantics: it validates the vertex index but returns `_vertexToActiveCellsHost` as a whole rather than the slice for the requested vertex.

This is acceptable for low-frequency mesh construction/refinement metadata, but not for inner assembly or matrix-free kernels.  Any hot path must use precomputed flat Kokkos views and ordinal-based IDs, not `std::map`, recursive construction, or `ComponentCellID1D` object lookups.

### TensorMeshTopology

`TensorMeshTopology` combines several `ComponentMeshTopology1D` objects.  It builds `_activeCellTuples` by recursively enumerating the Cartesian product of component active cells.  It supports side queries, boundary checks, slicing by fixed component vertices, and tensor skeleton enumeration.

Important limitations:

- Active tensor cells are stored as `std::vector<std::vector<ComponentCellID1D>>`; this is not device-ready and will be costly to traverse at scale.
- `tensorCellOrdinal()` linearly scans `_activeCellTuples`.  For matrix assembly and gather/scatter, ordinal lookup must be O(1) or avoided entirely through structured component ordinals.
- Boundary and skeleton methods perform host-side topology lookups.  That is fine for setup, but not for repeated integration kernels.
- The topology assumes all tensor factors are 1D.  This is consistent with the immediate goal.  Future products of a 2D/3D base mesh with 1D meshes are not represented except conceptually.
- Skeleton support is slice-based: for a fixed vertex in one component, it enumerates retained cell tuples in the other components.  This is a useful abstraction for trace variables on tensor-product sides, but it still needs orientation, side parity, neighbor ownership, and boundary-condition/tag integration before it can replace legacy traces.

## Tensor variable/discretization layer

### TensorVar

`TensorVar` is a tensor-level variable descriptor around per-component legacy `VarPtr`s.  It distinguishes only `TENSOR_FIELD` and `TENSOR_TRACE`.  Trace variables have one distinguished component ordinal, interpreted as the component orthogonal to the trace skeleton.  Flux-vs-value trace behavior is inferred from component `Var` types, with comments saying flux semantics should be recovered later by trace semantics and side-parity weighting.

This is a reasonable starting model for tensor-native DPG, but it is not yet rich enough for a robust Vlasov implementation.  Missing or underspecified items include:

- tensor-space function space per component (HGRAD/HVOL/L2/trace) as first-class data;
- strong mapping between tensor variables and legacy `Var` IDs used by `BF`, `IP`, `BC`, and `Solution`;
- side-orientation/parity conventions for flux traces;
- multi-component/vector-valued variables and component-block grouping;
- boundary-condition support and tagged sides.

### TensorVarDiscretization and TensorVarFactory

These are intentionally lightweight.  `TensorVarDiscretization` stores a `TensorVar`, optional basis-family override, and one p-order delta per component.  `TensorVarFactory` is a name-based registry.

These classes are adequate as metadata.  For performance, they should remain outside inner kernels.  Kernel code should consume compact integer IDs, Kokkos views, and prebuilt basis/evaluator descriptors.

## TensorDofLayout

`TensorDofLayout` constructs variable layouts and factor-space shapes.  It computes component cardinalities by asking the effective basis family for a line basis per component and multiplying cardinalities.  It can create volume and skeleton contexts and encode/decode local tensor-product DOF ordinals.

Strengths:

- It captures the key idea that a local tensor DOF is a tuple of 1D basis ordinals.
- It distinguishes volume support from trace-skeleton support.
- It can represent anisotropic per-component cardinalities and per-component p deltas.

Performance and correctness gaps:

- `cellLocalDofOrdinal()` and `decodeCellLocalDofOrdinal()` allocate/enumerate all component ordinal tuples on every call.  This is catastrophic in any hot path.  Both should be replaced with direct mixed-radix arithmetic or removed if the orderings are actually identical.
- All shapes are rebuilt from `std::vector`, map lookups, basis construction, and topology calls.  Assembly needs precomputed shape tables keyed by variable, space role, component cell ordinal, p order, and support kind.
- `buildAmbientShape()` handles trace shapes with a questionable placeholder: it initializes all component cells to the same incident orthogonal cell, then overwrites retained components.  That is probably only safe for tests and should not define production skeleton layout semantics.
- The layout is local only; it intentionally lacks global numbering/gather-scatter.  That must be added before tensor-native assembled matrices or matrix-free global operators can be used.
- Trial and test layouts are stored in `std::map` by name; this is fine for setup, not kernels.

## TensorFunction

`TensorFunction` is more Kokkos-aware than much of the mesh layer.  It defines device and host point/value views, has `values()` overloads, and tracks dependency scopes and separability/coupling metadata.  It supports constants, coordinate functions, products/sums, unary exp/sin/cos, derivatives, and 1D Maxwellians.

This is promising for tensorized coefficient evaluation because it records whether a coefficient is constant, single-component, separable, partially coupled, or fully coupled.  That metadata can drive fast assembly paths:

- fully separable coefficients can be integrated as Kronecker products of 1D weighted contractions;
- coefficients depending only on configuration dimensions, velocity dimensions, or one component can avoid full tensor quadrature materialization;
- fully coupled coefficients can fall back to generic tensor-point evaluation.

Current limitations:

- Composition allocates temporary `Kokkos::View<double*>` in each `values()` call.  That is acceptable for coarse prototype evaluation but should not be in assembly inner loops.
- The abstraction lacks a compiled device expression tree or reusable scratch interface.  Virtual calls and allocation in recursive function evaluation will be too expensive for high-order 4D kernels.
- `value(std::vector<double>)` copies to device and back for one point; this is only a convenience method, not suitable for repeated use.
- There is metadata for solution dependencies, but no evaluator bridge for solution-backed coefficients such as electromagnetic fields, moments, or fixed-point solution terms.

## TensorBasis

`TensorBasis` predates the new tensor mesh effort.  It supports a two-factor product, originally spatial-by-temporal, with one 1D temporal basis.  It exposes Intrepid2-style tensor points/basis values for OPERATOR_VALUE, and older `FieldContainer`-based tensor point/value utilities.

It is useful historical code and may be reusable for some basis-value plumbing, but it is not the general tensor-native basis abstraction needed for the current target:

- only two tensorial factors are represented explicitly;
- the implementation assumes a spatial basis plus temporal line basis, not an arbitrary product of 1D components;
- Intrepid2 tensor value support is currently limited to scalar VALUE operations in the shown path;
- older `FieldContainer` APIs coexist with Kokkos/Intrepid2 data structures.

For new work, it is better to define a compact N-factor basis/evaluator layer and reuse pieces of `TensorBasis` only where they align with Intrepid2 tensor data conventions.

## Current Vlasov/Planar MITL path

`PlanarMITLRelativisticSpaceTime.cpp` constructs a 1D spatial topology, temporal vertices, and two velocity-coordinate vertex arrays.  `getFormulation1D1T2V()` builds a `RelativisticVlasovSpaceTimeFormulation` with dimensions x, t, ux, uy.  The driver sets Maxwell boundary conditions, Vlasov initial/inflow data, optional adaptivity, iterative solver settings, and repeated time-slab solves.

Inside `VlasovSpaceTimeFormulation::initialize()`:

- configuration space is built as an extruded mesh over space and time;
- Vlasov phase-space-time is built as another extruded mesh over configuration space and velocity dimensions;
- legacy `VarFactory`, `BF`, `Mesh`, `Solution`, `RHS`, and `IP` objects are constructed;
- `_useSumFactorizedAssembly` is passed only to inner-product construction;
- global solve remains `Solution::solve()` with optional legacy geometric multigrid.

`RelativisticVlasovSpaceTimeFormulation` explicitly sets `_useSumFactorizedAssembly = false` and throws if asked for sum-factorized Gram assembly.  The bilinear form is the expected transport form:

- `-v[d] * f` paired with spatial derivatives of `w`;
- `-f` paired with time derivative of `w`;
- `-(q/m) a[vd] * f` paired with velocity derivatives of `w`;
- `t_n_hat` paired with `w` on the skeleton;
- optional positivity terms involving `psi`, `phi`, and `exp(psi_prev)`.

This structure is well suited to tensorization: most basis factors are 1D, many coefficients depend on a subset of dimensions, and derivative operators are one-dimensional actions tensored with values in the other dimensions.

## Main performance risks in the current tensor code

1. The new tensor mesh code is mostly host metadata and does not yet provide Kokkos views for kernel-ready tensor cells, DOF maps, sides, or tags.
2. The design uses `std::vector`, `std::map`, recursive enumeration, and per-call basis creation in places that must be setup-only or eliminated.
3. There is no tensor-native global DOF assignment, Tpetra map, import/export, graph construction, or boundary-condition path.
4. There is no tensorized `BF`/`IP` representation, so no way yet to translate Vlasov terms into sum-factorized kernels.
5. There is no implementation of B, G, K = B G^-1 B^T, mixed systems, or matrix-free apply using tensor-native data.
6. The old `TensorBasis` is not general enough for 4D 1D-by-1D-by-1D-by-1D Vlasov use without a new N-factor layer.
7. `TensorFunction` has useful separability metadata but evaluates through virtual calls and temporary allocations; this must be converted to reusable, kernel-friendly coefficient evaluators for production.
8. Existing Vlasov code interleaves formulation, mesh construction, solution setup, diagnostics, time-slab management, and solver selection.  A tensor-native fast path should be introduced alongside it rather than trying to mutate every legacy path at once.

## Recommended interpretation

Treat the current tensor-oriented code as a validated descriptive layer and test scaffold.  Keep it for topology/discretization setup and metadata.  Do not use it directly inside assembly kernels.  The next layer should compile this metadata into flattened Kokkos/Intrepid2 data structures, then execute sum-factorized kernels over those structures.

The fastest path to value is not a full replacement of Camellia `Mesh`/`Solution`, but a Vlasov-specific tensor operator stack that can be connected to a new Planar MITL driver while borrowing legacy formulation semantics and diagnostics where practical.