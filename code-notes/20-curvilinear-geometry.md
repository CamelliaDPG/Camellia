# Comprehensive Analysis: Curvilinear Geometry in Camellia

**Document:** `20-curvilinear-geometry.md`  
**Status:** Below budget (4% of $200 spent)  
**Generated:** Analysis of curvilinear geometry architecture, limitations, and proposed directions

---

## 1. Executive Summary

Camellia's curvilinear geometry support is currently **incomplete, tightly coupled, and 2D-limited**. The system handles curved edges via `ParametricCurve` abstractions and a separate `MeshTransformationFunction`, but lacks:

- **Generality:** No support for arbitrary high-order isoparametric mappings, NURBS, or parametric surfaces (3D).
- **Separation of concerns:** Geometric mapping logic intertwines with `CamelliaCellTools`, `MeshTopology`, and `Mesh`.
- **Consistent API:** Jacobian, normal, and reference-frame mapping queries lack unified interface.
- **Extensibility:** Adding new geometric primitives (e.g., NURBS patches, polynomial mappings) requires deep surgery.

**Impact on RLM:** Users cannot easily:
- Solve PDEs on domains with curved boundaries beyond simple circular/polynomial edges.
- Leverage high-order geometric representations (common in CAD/CAM).
- Combine adaptive refinement with complex geometries reliably.
- Port NURBS-based models from other systems (e.g., MFEM, Gmsh).

**Recommended direction:** Extract geometry into a dedicated **GeometryMapping** interface (similar to MFEM's `GeometricFactors`), decouple from cell tools, and enable pluggable geometric transformations.

---

## 2. Current Camellia Curvilinear Geometry Implementation

### 2.1 Architecture Overview

The curvilinear geometry system comprises four principal layers:

| Layer | Component | Role |
|-------|-----------|------|
| **User API** | `Mesh::initializeTransformationFunction()` | Entry point for curve registration |
| **Topology Layer** | `MeshTopology`, `_edgeToCurveMap` | Stores curve metadata; 2D only |
| **Geometry Layer** | `MeshTransformationFunction` | Applies curves to mesh coordinates |
| **Integration Layer** | `CamelliaCellTools::mapToPhysicalFrame()`, etc. | Maps between reference and physical frames |

### 2.2 Data Flow: Curve Registration to Integration

~~~markdown
User Code
  |
  +-> Mesh::addEdgeCurve(edge, ParametricCurvePtr)
  |        |
  |        v
  +-> MeshTopology::addEdgeCurve(edge, curve)
  |        |
  |        +-> Stores in _edgeToCurveMap[edge pair]
  |        +-> Marks cells as _cellIDsWithCurves (2D only)
  |        +-> Propagates to reverse edge
  |        +-> Does NOT update MeshTransformationFunction
  |
  +-> Mesh::initializeTransformationFunction()
       |
       +-> MeshTopology::initializeTransformationFunction(MeshPtr)
            |
            +-> Creates new MeshTransformationFunction (if needed)
            +-> Stores in Teuchos::RCP<MeshTransformationFunction> _transformationFunction
            |
            v
        Integration Phase:
         |
         +-> CamelliaCellTools::mapToPhysicalFrame(physPoints, refPoints, cellWorkset, ...)
         |    |
         |    +-> For non-tensorial topologies: call Intrepid2::CellTools<double>
         |    +-> Query MeshTransformationFunction for curved cells
         |
         +-> CamelliaCellTools::mapToReferenceFrame(refPoints, physPoints, meshTopo, cellID, cubatureDegree)
              |
              +-> MeshTopology::mapToReferenceFrame() (line 1057)
              +-> Uses Newton iteration with cubatureDegree for inverse map
~~~

**Source Evidence:**

From `MeshTopology.h` (line 87-90):
~~~c++
// these guys presently only support 2D:
set<IndexType> _cellIDsWithCurves;
map<pair<IndexType,IndexType>, ParametricCurvePtr> _edgeToCurveMap;
Teuchos::RCP<MeshTransformationFunction> _transformationFunction;
~~~

From `MeshTopology.cpp` (line 1057-1090), `mapToReferenceFrame()` uses Newton iteration:
~~~c++
bool MeshTopology::mapToReferenceFrame(FieldContainer<double> &refCoords, const FieldContainer<double> &physCoords,
                                        const shards::CellTopology &cellTopo, GlobalIndexType cellID, int cubatureDegree) {
  // Newton iteration to invert the map
}
~~~

### 2.3 Key APIs in `CamelliaCellTools.h`

**Forward mapping (physical frame):**
~~~c++
static void mapToPhysicalFrame(
  FieldContainer<double> &physPoints,
  const FieldContainer<double> &refPoints,
  const FieldContainer<double> &cellWorkset,
  CellTopoPtr cellTopo,
  int whichCell=-1
)
// Calls Intrepid's CellTools when cellTopo is non-tensorial
// Specialized handling for curvilinear geometry via cellWorkset
~~~

**Jacobian:**
~~~c++
static void setJacobian(
  FieldContainer<double> &jacobian,
  const FieldContainer<double> &points,
  const FieldContainer<double> &cellWorkset,
  CellTopoPtr cellTopo,
  int whichCell=-1
)
// Wraps Intrepid2::CellTools::setJacobian
// Computes Jacobian matrix DF of reference-to-physical map
~~~

### 2.4 Inverse Map (Reference Frame)

**From `CamelliaCellTools.h` (line 293-295):**
~~~c++
static void mapToReferenceFrame(
  Kokkos::DynRankView<double, PHX::Device> &refPoints,
  const Kokkos::DynRankView<double, PHX::Device> &physPoints,
  const Kokkos::DynRankView<double, PHX::Device> &cellWorkset,
  CellTopoPtr cellTopo,
  int whichCell=-1
)
~~~

### 2.5 Inverse Mapping (Physical to Reference)

**From CamelliaCellTools.h (line 293-295):**
~~~c++
static void mapToReferenceFrame(
  Kokkos::DynRankView<double, PHX::Device> &refPoints,
  const Kokkos::DynRankView<double, PHX::Device> &physPoints,
  const Kokkos::DynRankView<double, PHX::Device> &cellWorkset,
  CellTopoPtr cellTopo,
  int whichCell=-1
)
~~~

The function uses Newton iteration internally (via Intrepid2) to invert the forward map. For space-time elements, it currently throws `std::invalid_argument`.

### 2.6 Side Normal Computation

**From `CamelliaCellTools.h`:**
~~~c++
static void getPhysicalSideNormals(
  Kokkos::DynRankView<double, PHX::Device> &sideNormals,
  const Kokkos::DynRankView<double, PHX::Device> &inCellJacobian,
  CellTopoPtr parentCell
)
// Computes side normals from Jacobian
// Uses CamelliaCellTopology (supports 1D and 4D, unlike Intrepid)
~~~

**Issues:**
- For curvilinear elements, normals must be computed at each quadrature point (not just corners).
- `CamelliaCellTools::getPhysicalSideNormals()` delegates to `Intrepid2`, which may not handle curvilinear cases correctly.

---

## 3. Known Limitations and Gaps

### 3.1 2D-Only

From `MeshTopology.h` (line 87):
~~~c++
// these guys presently only support 2D:
set<IndexType> _cellIDsWithCurves;
~~~

All edge-curve handling (`_edgeToCurveMap`, `_cellIDsWithCurves`) is explicitly 2D-only. There is no mechanism for curved faces (3D) or curved volumes.

### 3.2 No NURBS or High-Order Geometry

The `ParametricCurve` class supports only piecewise-linear and low-degree polynomial curves. There is no infrastructure for:
- NURBS (Non-Uniform Rational B-Splines)
- Hermite or Bernstein patch representations
- Boundary representation (BREP) import

### 3.3 Tight Coupling

Curved geometry is embedded deeply in `MeshTopology` and `CamelliaCellTools`, making it difficult to add new geometry types without modifying core infrastructure.

### 3.4 Inconsistent API

The API for physical-reference-frame mapping is spread across `CamelliaCellTools`, `MeshTopology`, and `MeshTransformationFunction`, with no unified interface.

---

## 4. Comparison with MFEM and hp3d

### 4.1 MFEM

| Feature | Camellia | MFEM |
|---------|----------|------|
| **Geometry support** | 2D curved edges only | Full 3D, NURBS, high-order |
| **Reference mapping** | `CamelliaCellTools` | `GeometricFactors`, `ElementTransformation` |
| **High-order geometry** | Not supported | Full isoparametric support |
| **NURBS** | Not supported | `NURBSExtension` class |
| **Curved elements** | Limited 2D | Full 3D + hp-adaptive |
| **Geometry API** | Indirect via `MeshTransformationFunction` | Direct from NURBS evaluation |
| **Parametric integration** | Not supported | Full parametric quadrature |

### 4.2 hp3d

| Feature | Camellia | hp3d |
|---------|----------|------|
| **Geometry support** | 2D curved edges only | Full 3D geometry via HEXA/TETRA/PRISM |
| **Reference mapping** | `CamelliaCellTools` | Direct from geometry routines |
| **High-order geometry** | Not supported | Via `geom_hp.F90` |
| **NURBS** | Not supported | Not native (but adaptable) |
| **Curved elements** | Limited 2D | 3D with HEXA/PRISM mappings |
| **Geometry API** | Indirect | Direct |

---

## 5. Current API Summary

### 5.1 Key Entry Points

| Function | File | Purpose |
|----------|------|---------|
| `Mesh::initializeTransformationFunction()` | `Mesh.h` | Initializes curvilinear geometry for mesh |
| `MeshTopology::addEdgeCurve()` | `MeshTopology.h` | Adds a parametric curve to an edge |
| `CamelliaCellTools::mapToPhysicalFrame()` | `CamelliaCellTools.h` | Forward reference-to-physical map |
| `CamelliaCellTools::setJacobian()` | `CamelliaCellTools.h` | Compute Jacobian at quadrature points |
| `CamelliaCellTools::mapToReferenceFrame()` | `CamelliaCellTools.h` | Inverse physical-to-reference map |
| `CamelliaCellTools::getPhysicalSideNormals()` | `CamelliaCellTools.h` | Compute side normals |

### 5.2 Known Gaps

| Gap | Status | Note |
|-----|--------|------|
| 3D curved geometry | Missing | Only 2D edge curves exist |
| NURBS geometry | Missing | No infrastructure |
| High-order parametric maps | Missing | Only linear/low-degree |
| Space-time curvilinear | Throws | `mapToReferenceFrame` for space-time not implemented |
| Unified geometry API | Missing | Spread across 3+ classes |
| Geometry-adaptive refinement | Missing | No geometry-driven refinement |

---

## 6. Recommended Improvements

### 6.1 Introduce `GeometryMapping` Interface

Sequence:
  1. Mesh::construct()
  2. Mesh::addEdgeCurve(edge, curve)  <- Adds to MeshTopology
  3. Mesh::initializeTransformationFunction()  <- Creates mapper
  
Integration reads MeshTransformationFunction during assembly.

### 6.2 Decouple from `CamelliaCellTools`

Move geometry-sensitive operations out of static utility functions into the `GeometryMapping` interface. This allows:
- Pluggable geometry backends (polynomial, NURBS, CAD).
- Cleaner separation between topology (connectivity) and geometry (physical mapping).
- Testing without full `Mesh` or `MeshTopology` context.

### 6.3 Add NURBS Support

Leverage MFEM's `NURBSExtension` or OpenCASCADE bindings for NURBS-based geometry. Requires:
- New `NURBSGeometryMapping` implementing the `GeometryMapping` interface.
- Import/export routines for STEP/IGES formats.
- Integration with the adaptive refinement pipeline.

### 6.4 Extend to 3D

Replace 2D-only `_edgeToCurveMap` with a dimension-agnostic `_subcellToCurveMap`. Requires:
- New face-curve and volume-curve abstractions.
- 3D Newton iteration for inverse mapping.
- Updated `CamelliaCellTools` methods.

---

## 7. Summary Table

| Aspect | Current State | Recommended |
|--------|---------------|-------------|
| Geometry scope | 2D edges only | 3D faces + volumes |
| Geometry types | Polynomial only | NURBS, Hermite, CAD |
| API | Scattered | Unified `GeometryMapping` interface |
| Coupling | Tightly coupled | Loosely coupled via interface |
| Adaptivity | Not geometry-aware | Geometry-aware refinement |
| Testing | Implicit in mesh tests | Dedicated geometry unit tests |