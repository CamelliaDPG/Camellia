# 01 — Basis, Allocations, CellTools, DofOrdering & Debug

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
Caches basis function evaluations, cubature points, Jacobians, and geometric quantities for cells. Central to numerical integration.

### API Surface
- `BasisCache::basisCacheForCell(MeshPtr mesh, GlobalIndexType cellID)` — static factory.
- `->getSideBasisCache(int sideOrdinal)` — returns side cache.
- `->setRefCellPoints(FieldContainer<double>)` — sets reference-space points.
- `->getPhysicalCubaturePoints()` — returns mapped physical-space points.
- `->getCellMeasures()` — returns cell areas/volumes.
- `->setTransformationFunction(FunctionPtr)` — overrides geometric transformation.

### Issues
- **Transformation nullification pattern**: Tests repeatedly call `setTransformationFunction(Function::null())` to prevent double-transformation. Fragile.
- **Manual FieldContainer sizing**: Users manually construct and resize containers. Error-prone.


---

## 3. BasisEvaluation & BasisReconciliation

### BasisEvaluation.cpp (50,336 bytes)
Evaluates basis functions at quadrature points. Handles transformations between reference and physical frames for different function spaces (HGRAD, HDIV, HCURL, HVOL).

### BasisReconciliation.cpp (94,340 bytes)
The largest file in src/Basis/. Handles constraint computation for hanging nodes in h-refined meshes. Computes the sub-basis maps that reconcile traces/fluxes across non-conforming interfaces.

### Issues
- BasisReconciliation at 94KB is a candidate for decomposition.
- Complex constraint logic is performance-critical and difficult to maintain.

---

## 4. BasisFamily & BasisFactory

### BasisFamily.cpp (18,684 bytes)
Defines families of bases (Lagrange, Legendre, Lobatto, Bernstein, Intrepid2) and how they map to function spaces.

### BasisFactory.cpp (49,202 bytes)
Factory for creating basis objects. Caches created bases by (order, topology, function space) key. Manages enriched trace configuration.

### doubleBasisConstruction.cpp (27,288 bytes)
Constructs "double" (tensor-product) bases for space-time and phase-space formulations.

---

## 5. SpaceTimeBasisCache

### SpaceTimeBasisCache.cpp (35,691 bytes)
Extends BasisCache for space-time elements. Handles the tensor-product structure of space-time cubature and basis evaluation.

---

## 6. CamelliaCellTools

### CamelliaCellTools.cpp (83,170 bytes)
Wraps/extends Intrepid CellTools for Camellia-specific functionality:
- `refCellNodesForTopology()` — reference cell node coordinates
- `permutationInverse()`, `permutationComposition()` — subcell permutations
- `mapToReferenceFrame()` — physical-to-reference coordinate mapping
- `mapToReferenceSubcell()` — subcell reference mapping

### Issues
- Duplicated `mapToReferenceFrame` call patterns across drivers (DRY violation).
- Some drivers use raw `Intrepid_CellTools` instead of `CamelliaCellTools` (inconsistent abstraction).

---

## 7. Allocations

### Camellia_FieldContainer.cpp (376 bytes)
Minimal wrapper; likely typedef or compatibility shim for FieldContainer.

### Camellia_KokkosAllocations.cpp (10,131 bytes)
Kokkos device memory allocation management.

### SharedScratchStorage*.cpp (~26KB total)
Scratch memory pool for temporary allocations during assembly:
- `SharedScratchStorage.cpp` — main pool manager
- `SharedScratchStorageDynRankView.cpp` — dynamic-rank Kokkos views
- `SharedScratchStorageFixedRankView.cpp` — fixed-rank Kokkos views

### Pattern
`Camellia::finalize(bool printAllocationReport)` called at program exit. Only experimental drivers call this; other drivers may miss allocation cleanup.

---

## 8. DofOrdering

### DofOrdering.cpp (23,445 bytes)
Defines the ordering of degrees of freedom within an element: which basis, which variable, which side.

### DofOrderingFactory.cpp (37,816 bytes)
Factory for creating DofOrdering objects; determines trial and test orderings based on the bilinear form and polynomial order.

---

## 9. Convergence Study & Debug

### HConvergenceStudy.cpp (29,622 bytes)
Automates h-convergence studies: solves on a sequence of refined meshes and computes convergence rates.

### CamelliaDebugUtility.cpp (12,438 bytes)
Debugging helpers: print FieldContainers, compare matrices, rank-conditional output.

### TimeLogger.cpp (2,464 bytes)
Simple timing utility for performance measurement.

---

## 10. Cross-Cutting Issues

1. **Global singleton state** (`BasisFactory`) with mutable configuration is a thread-safety risk.
2. **Fragile transformation nullification** in BasisCache usage.
3. **Pervasive copy-paste** of `mapToReferenceFrame` and `Camellia::finalize` patterns.
4. **Inconsistent abstraction levels**: Some files use `CamelliaCellTools`, others raw Intrepid.
5. **BasisReconciliation** is 94KB and should be decomposed.
6. **Incomplete finalization**: Only experimental drivers call `Camellia::finalize()`.
7. **Test sprawl**: CurvilinearMeshTests acknowledged as needing decomposition.
