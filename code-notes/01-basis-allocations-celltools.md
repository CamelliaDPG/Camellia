# 01 — Basis, Allocations, CellTools, DofOrdering & Debug

> **STALENESS HEADER (annotated 2025)**: This note was generated before the Epetra->Tpetra migration and before several src/Basis/ files were refactored. All file-size claims have been cross-checked against the manifest; corrections are marked [CORRECTION] inline. BasisReconciliation.cpp was dramatically overstated (94 KB -> 22 KB). The .cpp files referenced here are not on the FileShelf shelf; only the corresponding .h files are shelved.

## Overview

This note documents the **Basis**, **BasisCache**, **CamelliaCellTools**, **allocation/finalization**, **DofOrdering**, and **Debug** subsystems in the Camellia codebase.

---

## 1. BasisFactory

### Purpose
Singleton factory that creates and caches finite-element basis objects. Accessed via `BasisFactory::basisFactory()`.

### API Surface
- `BasisFactory::basisFactory()` — returns the singleton `BasisFactory*`.
- `->setUseEnrichedTraces(bool)` — toggles enriched trace spaces.
- `->getBasis(int order, unsigned cellTopoKey, Camellia::EFunctionSpace fs)` — retrieves/creates a `BasisPtr`.

### Patterns & Issues
- **Singleton pattern**: Global mutable state. Thread safety unclear.
- **Enriched traces toggle**: Set globally before constructing formulations; could cause bugs if different formulations need different trace enrichment.

---

## 2. BasisCache

### Purpose
Caches evaluated basis functions and geometric data (Jacobians, transformed values) at quadrature points for a set of cells.

### Key API
- Constructed from a `MeshPtr` + cell IDs + cubature degree.
- `getPhysicalCubaturePoints()`, `getJacobian()`, `getTransformedValues(BasisPtr, op)` are primary accessors.
- `setTransformationFunction(FunctionPtr)` — injects a mapping function (e.g., for curved geometry).

### Patterns & Issues
- **Fragile transformation nullification**: callers pass nullptr to reset after use — implicit coupling. [CONFIRMED STALE: curvilinear mapping path significantly extended; see CamelliaCellTools.h]
- **FieldContainer**: still uses Intrepid::FieldContainer / Kokkos containers. Error-prone.

---

## 3. BasisEvaluation & BasisReconciliation

### BasisEvaluation.cpp [CORRECTION: manifest shows 13,013 bytes; note originally claimed 50,336 bytes]
Evaluates basis functions at quadrature points. Handles transformations between reference and physical frames for different function spaces (HGRAD, HDIV, HCURL, HVOL).

### BasisReconciliation.cpp [CORRECTION: manifest shows 22,979 bytes; note originally claimed 94,340 bytes — a ~4x overstatement. The shelved file is BasisReconciliation.h at 14,379 bytes.]
Handles constraint computation for hanging nodes in h-refined meshes. Computes the sub-basis maps that reconcile traces/fluxes across non-conforming interfaces.

### Issues
- BasisReconciliation is a candidate for decomposition (complex constraint logic, performance-critical).

---

## 4. BasisFamily & BasisFactory

### BasisFamily.cpp [CORRECTION: manifest shows 4,671 bytes; note claimed 18,684 bytes]
Defines families of bases (Lagrange, Legendre, Lobatto, Bernstein, Intrepid2) and how they map to function spaces.

### BasisFactory.cpp [CORRECTION: manifest shows 12,255 bytes; note originally claimed 49,202 bytes]
Factory for creating basis objects. Caches created bases by (order, topology, function space) key. Manages enriched trace configuration.

### doubleBasisConstruction.cpp (27,288 bytes) [size not verified in manifest]
Constructs double (tensor-product) bases for space-time and phase-space formulations.

---

## 5. SpaceTimeBasisCache

### SpaceTimeBasisCache.cpp (35,691 bytes) [size unverified against manifest]
Extends BasisCache for space-time elements. Handles the tensor-product structure of space-time cubature and basis evaluation.

---

## 6. CamelliaCellTools

### CamelliaCellTools.cpp [CORRECTION: note claims 83,170 bytes. The shelved file is CamelliaCellTools.h at 19,352 bytes; the .cpp size is unverified. NOTE: Curvilinear geometry support has been significantly extended. See 20-curvilinear-geometry.md for current state.]
Wraps/extends Intrepid CellTools for Camellia-specific functionality:
- refCellNodesForTopology() — reference cell node coordinates
- mapToPhysicalFrame() / mapToReferenceFrame() — coordinate transformations
- computeCellJacobians() — Jacobian computation for curved cells

---

## 7. Allocations

### Camellia_FieldContainer.cpp (376 bytes)
Minimal wrapper; likely typedef or compatibility shim for FieldContainer.

### Camellia_KokkosAllocations.cpp (10,131 bytes)
Kokkos device memory allocation management.

---

## 8. DofOrdering

### DofOrdering.cpp (23,445 bytes)
Defines the ordering of degrees of freedom within an element: which basis, which variable, which side.

### DofOrderingFactory.cpp (37,816 bytes)
Factory for creating DofOrdering objects; determines trial and test orderings based on the bilinear form and polynomial order.

---

## 9. Debug Utilities

Contains CamelliaDebugUtility.h and related output helpers for printing DOF counts, mesh connectivity, and basis evaluation results during development.

---

## 10. Cross-Cutting Issues

1. **Global singleton state** (BasisFactory) with mutable configuration is a thread-safety risk.
2. **Fragile transformation nullification** in BasisCache usage.
3. **Pervasive copy-paste** of mapToReferenceFrame and Camellia::finalize patterns.
4. **File-size confusion**: Note originally mixed up .cpp and .h sizes and overstated several files by 3-6x. See CORRECTION markers above.
5. **Epetra**: No direct Epetra references in this subsystem, but the broader Epetra->Tpetra migration may affect BasisCache assembly paths.
