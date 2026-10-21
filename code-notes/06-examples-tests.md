# 06 — Examples, Manual Examples, and Unit Tests

> **STALENESS HEADER (annotated 2025)**: Two key staleness issues: (1) Epetra_Operator_to_Epetra_MatrixTests.cpp is present in the unit test suite — this test file targets deprecated Epetra APIs and should be reviewed for removal or migration; (2) the test framework in drivers/DPGTests/ uses a custom bool-returning TestSuite pattern rather than GTest/Catch2. The example file inventory appears largely accurate but formulation sizes cited in examples may be stale (see 02-formulations.md corrections).

## Overview

Camellia ships three directories of executable code that demonstrate its API and verify correctness:

1. **examples/** — High-level API demos (Poisson, Stokes, Navier-Stokes, etc.)
2. **manual-tests/** — Semi-manual verification programs
3. **unit_tests/** — Automated unit tests (~90+ test files)

---

## Examples

| File | Program Name | Problem | Formulation Used | Level |
|------|-------------|---------|-----------------|-------|
| PoissonExample.cpp | Poisson | Standard Poisson | PoissonFormulation | Introductory |
| PoissonMatrixOutput.cpp | PoissonMatrixOutput | Poisson with matrix export | PoissonFormulation | Intermediate |
| PoissonPrimal.cpp | PoissonPrimal | Poisson in primal (H1) form | Standard H1 FEM | Introductory |
| StokesCavityFlowExample.cpp | StokesCavityFlow | Lid-driven cavity (Stokes) | StokesVGPFormulation | Intermediate |
| StokesBackwardFacingStepExample.cpp | StokesBackwardFacingStep | Backward-facing step | StokesVGPFormulation | Intermediate |
| StokesGramOutput.cpp | StokesCavityFlow | Gram matrix export for Stokes | StokesVGPFormulation | Debug/utility |
| NavierStokesCavityFlowExample.cpp | NavierStokesCavityFlow | Nonlinear cavity (Navier-Stokes) | NavierStokesVGPFormulation | Advanced |
| StokesCavityFlowSpaceTimeExample.cpp | StokesCavityFlow | Transient Stokes in space-time | SpaceTimeIncompressibleFormulation | Advanced |

---

## Unit Tests (unit_tests/)

### Basis Tests
| File | What It Tests |
|------|--|
| BasisCacheFactoryTests.cpp | BasisCache construction from topology |
| BasisCacheTests.cpp | Physical-to-reference mappings, Jacobians |
| BasisFactoryTests.cpp | Basis creation by type and order |
| BasisFamilyTests.cpp | Hierarchical basis family consistency |
| BasisReconciliationTests.cpp | Hanging-node constraint reconciliation |

### DOF Tests
| File | What It Tests |
|------|--|
| DofOrderingTests.cpp | DOF ordering within elements |
| DofOrderingFactoryTests.cpp | DOF ordering creation and consistency |

### Utility Tests
| File | What It Tests |
|------|--|
| FieldContainerTests.cpp | Intrepid FieldContainer wrappers |
| SerialDenseWrapperTests.cpp | Dense matrix operations |
| Epetra_Operator_to_Epetra_MatrixTests.cpp | Trilinos operator conversion [STALE: tests deprecated Epetra APIs] |
| MPIWrapperTests.cpp | MPI communication helpers |
| MPIGlobalMinMaxTests.cpp | Global min/max reductions |

---

## Relationship Between unit_tests/ and drivers/DPGTests/

- **unit_tests/**: Modern CMake-integrated test executables. Preferred for new tests.
- **drivers/DPGTests/**: Legacy custom TestSuite.h framework. Monolithic runner. [CONFIRMED STALE: not GTest/Catch2; should be migrated]

---

## Cross-Cutting Issues

1. **Epetra_Operator_to_Epetra_MatrixTests.cpp**: Tests deprecated APIs. Should be reviewed for removal or migration to Tpetra equivalents.
2. **Legacy test framework in DPGTests/**: The custom bool-returning TestSuite pattern should be migrated to GTest/Catch2 for CI integration.
3. **Experimental tests**: unit_tests/experimental/ contains DLSTests.cpp (Discrete Least Squares) and similar files with unclear maintenance status.
