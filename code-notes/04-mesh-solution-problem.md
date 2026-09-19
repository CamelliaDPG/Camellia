# 04 — Mesh, Solution, Problem, IO & MatrixUtilities

## Overview

| Directory | Role |
|-----------|------|
| src/Mesh/ | Computational mesh: topology, geometry, DOF assignment, partitioning, refinement |
| src/Solution/ | Global linear system assembly and solve, error estimation, GMG |
| src/Problem/ | BCs, RHS, bilinear-form utilities, virtual DPG machinery |
| src/IO/ | HDF5 and GnuPlot output |
| src/MatrixUtilities/ | Dense-matrix helpers, Epetra/Tpetra conversion |

---

## 1. src/Mesh/

### Key Files
- **MeshTopology.cpp** (42KB): Pure topological mesh — cells, vertices, entity sets, periodic BCs, refinement tree. No DOFs.
- **Mesh.cpp** (15KB): Central facade — owns MeshTopology + GlobalDofAssignment.
- **MeshFactory.cpp** (26KB): Static helpers: rectilinearMesh, quadMesh, triangleMesh, etc.
- **GDAMinimumRule.cpp** (47KB): Minimum-rule global DOF assignment — the production GDA for DPG.
- **GlobalDofAssignment.cpp** (13KB): Abstract GDA base.
- **Cell.cpp** (11KB): Per-cell geometry, parent/child relationships.
- **CellTopology.cpp** (11KB): Wraps Shards topologies with Camellia extensions.
- **RefinementPattern.cpp** (28KB): How cells split (regular, anisotropic).
- **LocalDofMapper.cpp** (15KB): Element-local to global DOF mapping.
- **ZoltanMeshPartitionPolicy.cpp** (6KB): Zoltan-based parallel partitioning.

### Mesh Class Highlights
- Four constructor overloads accepting MeshTopologyViewPtr + VarFactoryPtr or TBFPtr
- Inherits DofInterpreter for local-to-global DOF translation
- Registers solutions for refinement notification callbacks

---

## 2. src/Solution/

### Key Files
- **Solution.cpp** (52KB): TSolution template — assembles element stiffness/load, applies BCs, solves, distributes coefficients.
- **GMGOperator.cpp** (28KB): Geometric multigrid: prolongation, restriction, smoothing.
- **GMGSolver.cpp** (7KB): GMG preconditioned Krylov solver.
- **CondensedDofInterpreter.cpp** (14KB): Static condensation of interior DOFs.
- **RefinementStrategy.cpp** (8KB): Decides which cells to refine.
- **ErrorIndicator.cpp** (4KB): Base class for error indicators.
- **RieszRep.cpp** (3KB): Riesz representative computation.
- **Solver.cpp** (1KB): Solver interface (KLU, MUMPS, SuperLU, Amesos2).

### TSolution Highlights
- Templated on Scalar (double and complex double)
- solve() orchestrates: element stiffness assembly, global gather, BC imposition, linear solve, scatter
- Supports static condensation via CondensedDofInterpreter
- Registers with Mesh for refinement callbacks (solution projection)

---

## 3. src/Problem/

### Key Files
- **BC.cpp** (4.5KB): Boundary condition container. addDirichlet() with SpatialFilter.
- **RHS.cpp** (2KB): Right-hand-side container storing LinearTerm contributions.
- **Virtual.cpp** (7KB): Virtual DPG element — test-norm operators, field/trace associations.
- **TensorOperator.cpp** (12KB): Operators on tensor-product DOF layouts.
- **CompressibleProblems.cpp** (6KB): Pre-built compressible flow problems (Sod, Noh).

---

## 4. src/IO/

- **HDF5Exporter.cpp** (12KB): Solution to HDF5+XDMF for ParaView.
- **GnuPlotUtil.cpp** (5KB): Mesh skeleton and contour data for GnuPlot.

---

## 5. src/MatrixUtilities/

- **SerialDenseWrapper.cpp** (13KB): BLAS/LAPACK wrappers via Teuchos.
- **E_to_T.cpp** / **T_to_E.cpp**: Epetra-Tpetra format converters.
- **Epetra/Tpetra_Operator_to_Matrix**: Materialize operators into explicit matrices.

---

## 6. Design Patterns

1. **Facade**: Mesh wraps MeshTopology + GlobalDofAssignment
2. **Observer**: Solution registers with Mesh for refinement notification
3. **Strategy**: RefinementStrategy, MeshPartitionPolicy, Solver are pluggable
4. **Template method**: TSolution templated for real/complex
5. **Static condensation**: Schur-complement elimination via CondensedDofInterpreter
6. **Factory**: MeshFactory provides convenient mesh construction

---

## 7. Dependency Flow

MeshTopology → Mesh → GDAMinimumRule
Mesh → Solution → BasisCache → SerialDenseWrapper
Solution uses BC, RHS, BF, IP
Solution delegates to Solver (KLU/MUMPS/GMGSolver)
Output via HDF5Exporter, GnuPlotUtil
