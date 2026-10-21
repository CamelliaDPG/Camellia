# 70 — Implementation Roadmap

*Synthesized from completed analysis workstreams WS0–WS4 and supplementary documents 50–60. Generated after full execution of ANALYSIS-PLAN.md.*

---

## Overview

This document translates the Camellia modernization analysis into a concrete, phased implementation roadmap. It draws on:

- **WS0**: Subsystem inventory and staleness audit (`00-current-inventory.md`)
- **WS1**: DOF numbering and constraints analysis (`10-dof-numbering-and-constraints.md`)
- **WS2**: Curvilinear geometry analysis (`20-curvilinear-geometry.md`)
- **WS3**: GMG/hp3d transfer operator analysis (`30-gmg-hp3d-transfer-option.md`)
- **WS4**: MeshTopology simplification analysis (`40-mesh-topology-simplification.md`)
- **WS5**: Cross-workstream dependency analysis (`50-cross-workstream-dependencies.md`)
- **WS6**: MFEM/hp3d comparative source study (`60-mfem-hp3d-source-comparison.md`)

---

## Guiding Principles

1. **Topology first**: `MeshTopology` underpins `GDAMinimumRule`, `BasisReconciliation`, and `GMGOperator`. Stabilizing its interface before refactoring dependent subsystems prevents cascading API breakage.
2. **Geometry decoupled from topology**: The curvilinear geometry mapping layer must be independent of topology storage to allow each to evolve separately.
3. **Incremental compilation**: Each phase must leave the test suite green. No big-bang rewrites.
4. **Interface-first design**: Introduce abstract interfaces (e.g., `GeometryMapping`, `MultigridTransferBuilder`, `ConstraintAssembler`) before providing alternative implementations.
5. **MFEM/hp3d patterns as reference**: Where MFEM or hp3d implement a cleaner pattern, use it as a target design; do not blindly copy.

---

## Dependency Graph

~~~
Inventory (WS0)
    └─► DOF Numbering (WS1)
    └─► Curvilinear Geometry (WS2)
    └─► GMG Transfer (WS3)
    └─► MeshTopology (WS4)

MeshTopology (WS4)
    └─► DOF Numbering (WS1)   [GDA depends on topology queries]
    └─► GMG Transfer (WS3)    [GMGOperator walks topology]

Curvilinear Geometry (WS2)
    └─► MeshTopology (WS4)    [geometry decouple requires topology API stabilization]

DOF Numbering (WS1)
    └─► GMG Transfer (WS3)    [BasisReconciliation shared by both]
~~~

**Sequencing constraint**: WS4 → WS2 → WS1 → WS3 (with some WS1/WS2 parallelism after WS4 Phase 1).

---

## Phase 0 — Preparation and Baselines (0–2 weeks)

### Goal
Establish test coverage, API audit, and metrics baselines before any structural changes.

### Tasks

| ID | Task | Owner hint | Notes |
|----|------|------------|-------|
| P0.1 | Run full regression suite; record pass/fail baseline | CI | Required before any structural change |
| P0.2 | Add missing unit tests for `MeshTopology` query methods | Dev | Focus: `getEntitiesForSide`, `getConstrainedSideOrdinals` |
| P0.3 | Add missing unit tests for `GDAMinimumRule::rebuildLookups` | Dev | Needed to catch regressions in Phase 2 |
| P0.4 | Add missing unit tests for `CamelliaCellTools` geometry queries | Dev | Focus: `mapToReferenceFrame`, `getPhysicalCubaturePoints` |
| P0.5 | Profile `rebuildLookups` on a large h-refined mesh (≥10K elements) | Dev | Establish latency baseline |
| P0.6 | Profile `MeshTopology` memory footprint (size of redundant vectors) | Dev | Quantify savings opportunity |
| P0.7 | Document all public APIs that depend on `MeshTopology` internals | Dev | Identify which headers are "stable" vs "internal" |

### Exit criteria
- Regression suite passes at baseline
- Unit-test coverage ≥ 80% for targeted subsystems
- Performance baselines recorded

---

## Phase 1 — MeshTopology Quick Wins (2–6 weeks)

### Goal
Reduce `MeshTopology` size and coupling without breaking the public API. Yields immediate memory and complexity savings.

### Motivation
WS4 identified ~15 data members with redundant or derivable storage. Removing them reduces the surface area that WS1 (GDA) and WS3 (GMG) depend on.

### Tasks

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| P1.1 | Remove `_boundarySides` set — derive on demand from topology queries | High | Identified as highest-value quick win in WS4 |
| P1.2 | Remove `_sidesForCells` and `_cellsForSides` redundant bidirectional maps | High | Replace with O(1) computable accessors |
| P1.3 | Deprecate internal `_knownElements` usage outside constructors | Medium | Reduces stale-cache risk |
| P1.4 | Extract `RefinementTree` into a standalone class with its own unit tests | Medium | Enables independent testing of refinement logic |
| P1.5 | Introduce `MeshTopologyView` read-only adapter (interface extraction) | High | Required before WS1/WS2 can decouple from topology internals |
| P1.6 | Verify all removed/derived fields produce identical results to direct storage | High | Use regression suite + new unit tests from P0 |

### Exit criteria
- `MeshTopology.h` reduces from ~21KB to ≤15KB
- `MeshTopology.cpp` reduces from ~167KB by ≥15%
- `MeshTopologyView` interface published and used by at least `GDAMinimumRule` and `GMGOperator`
- All regression tests pass

---

## Phase 2 — Curvilinear Geometry Interface (4–10 weeks, overlaps Phase 1)

### Goal
Introduce a `GeometryMapping` abstract interface and decouple curvilinear geometry from topology storage.

### Motivation
WS2 found that `CamelliaCellTools` is 2D-only, tightly coupled to `MeshTopology`, and has no cache-invalidation protocol. The hp3d nodal-geometry-as-FE-field pattern and MFEM `GeometricFactors` both provide cleaner models.

### Tasks

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| P2.1 | Define `GeometryMapping` interface (reference-to-physical, Jacobian, inverse) | High | Must be topology-agnostic |
| P2.2 | Implement `AffineCellMapping` as simplest `GeometryMapping` (covers linear elements) | High | Replaces current default path in `CamelliaCellTools` |
| P2.3 | Implement `ParametricCurveMapping` wrapping existing `ParametricCurve`/`ParametricSurface` | Medium | Preserves existing 2D edge-curve support |
| P2.4 | Add `GeometryMapping` pointer to `Mesh` per cell (replacing inline CamelliaCellTools calls) | High | Central integration point |
| P2.5 | Implement `NURBSMapping` stub (interface only, deferred implementation) | Low | Future extensibility |
| P2.6 | Extend to 3D: audit all 3D paths in `CamelliaCellTools`, implement via `GeometryMapping` | High | WS2 identified 3D as entirely absent |
| P2.7 | Add cache-invalidation protocol to `GeometryMapping` (call on mesh refinement) | Medium | Prevents stale Jacobian caches |
| P2.8 | Remove geometry storage from `MeshTopology` (after P1.5 and P2.4) | Medium | Decoupling milestone |

### Dependency
- P1.5 (`MeshTopologyView`) must be available before P2.8
- P2.1–P2.4 can proceed in parallel with Phase 1

### Exit criteria
- `GeometryMapping` interface merged and documented
- At least `AffineCellMapping` and `ParametricCurveMapping` implemented and tested
- `CamelliaCellTools` calls routed through `GeometryMapping` for 2D cases
- Geometry data removed from `MeshTopology`

---

## Phase 3 — DOF Numbering Decomposition (8–16 weeks)

### Goal
Decompose the monolithic `GDAMinimumRule` and unify its 6+ overlapping caches into a coherent constraint-assembly pipeline.

### Motivation
WS1 identified `rebuildLookups()` as a 2000+ line monolithic function with 6+ cache structures that are rebuilt in an order-dependent way. MFEM's `BuildConformingInterpolation` provides a cleaner model.

### Tasks

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| P3.1 | Define `ConstraintAssembler` interface (topology query → constraint matrix) | High | MFEM `FiniteElementSpace` pattern |
| P3.2 | Extract constraint-discovery logic from `rebuildLookups` into `ConstraintDiscovery` class | High | First decomposition step |
| P3.3 | Extract DOF-index-assignment logic from `rebuildLookups` into `DofIndexer` class | High | Second decomposition step |
| P3.4 | Extract hanging-node constraint elimination into `HangingNodeResolver` | Medium | MFEM `BuildConformingInterpolation` analog |
| P3.5 | Unify 6+ cache maps into a single `GDALookupCache` struct with explicit invalidation | High | Eliminates stale-cache bugs |
| P3.6 | Add explicit `ConstraintMatrix` type (sparse) replacing ad-hoc maps | Medium | Enables MFEM-style true-DOF projection |
| P3.7 | Add `p`-adaptivity path: decouple `p`-order changes from full `rebuildLookups` | Low | Performance optimization |
| P3.8 | Benchmark new pipeline vs. old `rebuildLookups` on 10K-element mesh | High | Verify no regression |

### Dependency
- Phase 1 (`MeshTopologyView`) should be complete before P3.2–P3.4
- Phase 1 `RefinementTree` extraction (P1.4) simplifies P3.2

### Exit criteria
- `rebuildLookups()` replaced by `ConstraintAssembler::buildConstraints()` pipeline
- All 6+ cache maps unified
- Performance regression ≤ 5% vs. baseline (P0.5)
- All regression tests pass

---

## Phase 4 — GMG Transfer Operator Modernization (12–20 weeks)

### Goal
Introduce the hp3d/Petrides-style decomposed multigrid transfer pipeline as an option alongside the current `GMGOperator`.

### Motivation
WS3 found the current `GMGOperator` is a monolith that handles fields and traces in a single pass. The hp3d `Coarse2Macro` → `Macro2Fine` decomposed pipeline is more modular, enables Schur-complement extension/restriction, and is better suited to DPG trace handling.

### Tasks

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| P4.1 | Define `MultigridTransferBuilder` interface | High | Strategy pattern: allows old and new implementations |
| P4.2 | Implement `LegacyGMGTransfer` wrapping current `GMGOperator` logic | High | Ensures backward compatibility |
| P4.3 | Implement `Coarse2MacroTransfer`: coarse-grid DOF prolongation to macro level | High | Inspired by hp3d `coarse2macro.F90` |
| P4.4 | Implement `Macro2FineTransfer`: macro-to-fine element transfer with trace handling | High | Inspired by hp3d `macro2coarse.F90` (restriction direction) |
| P4.5 | Implement `DPGTraceExtension`: DPG trace Schur-complement extension operator | Medium | Key DPG-specific feature; Petrides et al. |
| P4.6 | Wire `MultigridTransferBuilder` into `GMGOperator` via strategy injection | High | Replaces direct transfer calls |
| P4.7 | Add switchable selection: `--gmg-transfer=legacy|hp3d` runtime flag | Low | Enables A/B comparison |
| P4.8 | Benchmark decomposed pipeline vs. legacy on adaptive DPG problem | High | Validate correctness and performance |

### Dependency
- Phase 3 (DOF numbering) must be stable before P4.3–P4.5
- `BasisReconciliation` shared with WS1; changes there affect WS3

### Exit criteria
- `MultigridTransferBuilder` interface merged
- `LegacyGMGTransfer` and `Coarse2MacroTransfer`+`Macro2FineTransfer` both passing regression tests
- `DPGTraceExtension` implemented and tested on at least one DPG problem
- Benchmark results documented

---

## Phase 5 — MeshTopology Structural Decomposition (16–24 weeks)

### Goal
Full structural decomposition of `MeshTopology` into `CombinatoricTopology` + `RefinementTree` + `NCConstraintManager` + `MeshTopologyAdapter`.

### Motivation
WS4 Phase 1 (above Phase 1) removes redundant storage; this phase performs the deeper structural split modeled on MFEM's `Mesh`/`NCMesh` separation.

### Tasks

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| P5.1 | Extract `CombinatoricTopology`: vertices, edges, faces, cells as flat arrays | High | Lean data model |
| P5.2 | Extract `RefinementTree`: refinement history, parent/child relationships | High | From P1.4 |
| P5.3 | Extract `NCConstraintManager`: hanging-node detection and constraint generation | High | MFEM `NCMesh` analog |
| P5.4 | Implement `MeshTopologyAdapter` composing the three components | High | Drop-in replacement for `MeshTopology` |
| P5.5 | Migrate `GDAMinimumRule` to use `NCConstraintManager` directly | Medium | Replaces topology-traversal constraint discovery |
| P5.6 | Migrate `GMGOperator` to use `RefinementTree` directly | Medium | Cleaner transfer logic |
| P5.7 | Retire `MeshTopology` god-object (deprecate, then delete) | Low | Final cleanup |

### Dependency
- All of Phase 1, 3 must be complete
- Phase 2 geometry decoupling must be complete (no geometry in topology)

### Exit criteria
- `MeshTopology` split into 3 focused classes
- `GDAMinimumRule` and `GMGOperator` fully migrated to new APIs
- Memory footprint reduced ≥ 30% vs. Phase 0 baseline
- All regression tests pass

---

## Phase 6 — Polish, Performance, and Documentation (ongoing after Phase 4)

### Goal
Address performance bottlenecks, code-smell items, and documentation gaps surfaced during analysis.

### Tasks

| ID | Task | Priority | Notes |
|----|------|----------|-------|
| P6.1 | Replace raw pointer usage with `shared_ptr`/`unique_ptr` throughout refactored code | Medium | Code modernization |
| P6.2 | Add Tpetra-aware DOF map export from `DofIndexer` | Medium | Post-Epetra cleanup |
| P6.3 | Remove residual Epetra includes from refactored headers | Medium | Identified in WS0 staleness audit |
| P6.4 | Add Doxygen documentation to all new interfaces | Low | |
| P6.5 | Update `code-notes/` analysis documents to reflect implemented state | Low | Keep notes in sync with code |
| P6.6 | Publish updated ANALYSIS-PLAN.md with final implementation status | Low | Closure document |

---

## Timeline Summary

| Phase | Duration | Parallel with | Key Milestone |
|-------|----------|---------------|---------------|
| P0: Preparation | 0–2 wk | — | Baseline established |
| P1: Topology Quick Wins | 2–6 wk | P2 start | `MeshTopologyView` published |
| P2: Geometry Interface | 4–10 wk | P1, P3 start | `GeometryMapping` merged |
| P3: DOF Decomposition | 8–16 wk | P2 tail | `rebuildLookups` replaced |
| P4: GMG Modernization | 12–20 wk | P3 tail | hp3d transfer option working |
| P5: Topology Full Split | 16–24 wk | P4 tail | `MeshTopology` retired |
| P6: Polish | ongoing | all | Code and docs clean |

Total elapsed calendar time (sequential critical path): **~24 weeks** assuming one focused developer.
With two developers working in parallel on Phase 2 + Phase 3: **~18 weeks**.

---

## Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| `rebuildLookups` decomposition breaks subtle ordering | High | High | Exhaustive unit tests before P3.2 |
| `BasisReconciliation` shared between WS1 and WS3 causes merge conflicts | Medium | Medium | Coordinate interface changes across phases |
| 3D geometry coverage gap deeper than anticipated | Medium | High | Audit 3D paths in P0 before committing P2 scope |
| Performance regression in DOF-numbering pipeline | Medium | Medium | P0.5 baseline + P3.8 benchmark gate |
| `MeshTopology` API surface larger than documented | Medium | High | P0.7 API audit before Phase 5 |
| hp3d transfer approach not DPG-compatible | Low | High | Prototype P4.3 early; compare to P4.2 baseline |

---

## Cross-Reference

| Document | Relation to Roadmap |
|----------|--------------------|
| `10-dof-numbering-and-constraints.md` | Detail for Phase 3 tasks |
| `20-curvilinear-geometry.md` | Detail for Phase 2 tasks |
| `30-gmg-hp3d-transfer-option.md` | Detail for Phase 4 tasks |
| `40-mesh-topology-simplification.md` | Detail for Phase 1 and Phase 5 tasks |
| `50-cross-workstream-dependencies.md` | Informs dependency graph and risk register |
| `60-mfem-hp3d-source-comparison.md` | Reference for MFEM/hp3d design patterns |
| `ANALYSIS-PLAN.md` | Parent plan; this document is its implementation translation |

---

*Last updated: after full ANALYSIS-PLAN.md execution. See ANALYSIS-PLAN.md for workstream execution status.*
