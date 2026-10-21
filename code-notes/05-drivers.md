# 05 — Drivers

> **STALENESS HEADER (annotated 2025)**: The key staleness issues in this note are: (1) drivers/pulse/ has an active Epetra dependency that is deprecated and being migrated to Tpetra; (2) 14+ experimental subdirectories have no status tracking and their continued relevance is unknown; (3) ConfusionProblemLegacy is confirmed dead code in drivers/DPGTests/. The overall structure of 8 subdirectories appears still accurate.

## Overview

The drivers/ directory contains ~8 subdirectories of executable programs that exercise the Camellia library for research, testing, and publication.

---

## 1. drivers/Poisson/
Simple Poisson driver programs for introductory/regression use.

## 2. drivers/Stokes/
Lid-driven cavity and similar Stokes benchmark drivers.

## 3. drivers/DPGTests/ (~63 files)

Legacy unit/integration test suite with custom TestSuite.h framework.

Test classes: ConfusionBilinearForm, CurvilinearMeshTests, DPGTests, ElementTests, FunctionTests, GDAMinimumRuleTests, GMGTests, HConvergenceStudyTests, IncompressibleFormulationsTests, LinearTermTests, LobattoBasisTests, MeshRefinementTests, MeshTestSuite, MeshTopologyTests, MPIWrapperTests, MultiBasisTests, ParametricCurveTests, PatchBasisTests, RHSTests, ScratchPadTests, SerialDenseMatrixUtilityTests, SolutionTests, VectorizedBasisTestSuite.

**Issues**: Custom framework (not GTest/Catch2). Monolithic runner. ConfusionProblemLegacy is dead code. Header-only test stubs inconsistent. [CONFIRMED STALE: should be migrated to standard test framework]

---

## 4. drivers/NavierStokes/
Nonlinear Navier-Stokes drivers using NavierStokesVGPFormulation.

## 5. drivers/papers/ (~27 files)

Publication-supporting drivers. May contain hard-coded paths or deprecated parameter-file formats.

## 6. drivers/preconditioning/

GMG and algebraic preconditioner evaluation drivers.
- **PreconditioningTestsDriver.cpp** (21,636 bytes per manifest)
- **StokesCavityFlowGMGDriver.cpp** (3,758 bytes per manifest)

## 7. drivers/pulse/ (~19 files)

BoBa external solver integration: DPG assembly, Vlasov system, compression, Epetra-to-BoBa conversion.

**[STALE — CRITICAL]: Epetra dependency in pulse/ is active and deprecated.** The Epetra-to-BoBa conversion layer uses Epetra_MultiVector and related types that must be migrated to Tpetra. This is a known open issue per AGENTS.md.

## 8. drivers/experimental/ (14+ subdirectories)

Research-stage drivers. No status tracking; many may be dead code.

---

## Cross-Cutting Issues

| Issue | Priority |
|-------|----------|
| No shared driver infrastructure/arg parsing | Medium |
| Naming inconsistencies (Driver vs Example, doubled names) | Low |
| Legacy/dead code (ConfusionProblemLegacy, bug repro drivers) | Medium |
| Tests in driver dirs with custom framework | Medium |
| Epetra dependency in pulse/ | Medium — active migration needed |
| Undocumented shell scripts | Low |
| 14+ experimental subdirs with no status tracking | Medium |
