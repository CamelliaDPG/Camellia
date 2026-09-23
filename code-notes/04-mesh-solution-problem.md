# 04 — Mesh, Solution, Problem, IO & MatrixUtilities

> **STALENESS HEADER (annotated 2025) — HIGH STALENESS**: The most critical error in this note is the GDAMinimumRule.cpp size claim. The note claims ~47KB; the actual shelf size is 185KB (185,072 bytes) — a nearly 4x growth indicating major logic additions since this note was written. The E_to_T.cpp / T_to_E.cpp Epetra converters are present but may be in the process of deprecation as part of the Epetra->Tpetra migration. Solution.cpp size (52KB / 52,491 bytes) is accurate.

## Overview

| Directory | Role |
|-----------|------|
| src/Mesh/ | Mesh topology, refinement, partitioning, GDA |
| src/Solution/ | Assembly, solve, GMG preconditioner |
| src/Problem/ | BCs, RHS, bilinear-form utilities, virtual DPG machinery |
| src/IO/ | HDF5 and GnuPlot output |
| src/MatrixUtilities/ | Dense-matrix helpers, Epetra/Tpetra conversion |

---

## Mesh Subsystem

### Key Files
- **MeshTopology.cpp** (167KB shelf-verified): Core topology data structure — cells, faces, vertices, refinement history.
- **GDAMinimumRule.cpp** [CORRECTION: shelf-verified at 185,072 bytes; note originally claimed ~47KB — a severe understatement]: Global DOF assignment via minimum-rule for h/p-adaptive meshes.
- **GlobalDofAssignment.cpp** (49KB shelf-verified): Base class for GDA hierarchy.
- **ZoltanMeshPartitionPolicy.cpp** (6KB): Zoltan-based parallel partitioning.

### Mesh Class Highlights
- Four constructor overloads accepting MeshTopologyViewPtr + VarFactoryPtr or TBFPtr
- Inherits DofInterpreter for local-to-global DOF translation

---

## Solution Subsystem

### Key Files
- **Solution.cpp** (52KB — ACCURATE per manifest: 52,491 bytes): TSolution template — assembles element stiffness/load, applies BCs, solves, distributes coefficients.
- **GMGOperator.cpp** (28KB): Geometric multigrid: prolongation, restriction, smoothing. [CROSS-REF: see 30-gmg-hp3d-transfer-option.md for current GMG analysis]
- **GMGSolver.cpp** (7KB): GMG preconditioned Krylov solver.

---

## Problem Subsystem

BCs, RHS, IP (inner product) and bilinear-form hierarchy. Provides the abstract interface used by all formulations.

---

## MatrixUtilities Subsystem

- **SerialDenseWrapper.cpp** (13KB): BLAS/LAPACK wrappers for dense element matrices.
- **E_to_T.cpp** / **T_to_E.cpp**: Epetra-to-Tpetra and Tpetra-to-Epetra converters. [NOTE: These may be deprecated once Epetra->Tpetra migration is complete. They remain as compatibility shims.]

---

## Cross-Cutting Issues

1. **GDAMinimumRule.cpp at 185KB**: Far larger than originally noted. This is the primary DOF-assignment engine and has grown significantly. Any refactoring must start here.
2. **Epetra->Tpetra migration**: E_to_T.cpp and T_to_E.cpp are active shims. When removed, all callers must use Tpetra types directly.
3. **Solution assembly**: TSolution<double> is the current concrete type. Any Epetra_Vector / Epetra_MultiVector usage in assembly paths is stale.
