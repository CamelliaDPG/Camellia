# Camellia Code Notes — Agent Guide (AGENTS.md)

## What is Camellia?

Camellia is a C++ finite element library implementing the Discontinuous Petrov-Galerkin (DPG) method. It is built on top of Trilinos (Intrepid2, Teuchos, Epetra/Tpetra, Belos, Amesos) and supports:
- Automatic optimal test functions for DPG
- h- and p-adaptive mesh refinement with hanging nodes
- Space-time formulations
- Phase-space (Vlasov/Boltzmann) formulations
- Geometric multigrid preconditioning
- Parallel execution via MPI

---

## Navigating the Code Notes

The `code-notes/` directory contains detailed documentation organized by subsystem:

| File | Contents |
|------|----------|
| `01-basis-allocations-celltools.md` | Basis functions, BasisCache, CamelliaCellTools, allocations, DofOrdering, debug utilities |
| `02-formulations.md` | All 34 DPG formulation classes organized by physics category |
| `03-functions.md` | Function class hierarchy, expression tree, operator overloading, evaluation paths |
| `04-mesh-solution-problem.md` | Mesh topology/factory, Solution assembly/solve, BCs, RHS, IO, matrix utilities |
| `05-drivers.md` | All driver programs: research, experimental, paper, pulse, preconditioning, tests |
| `06-examples-tests.md` | Examples, manual examples, and unit test inventory |
| `AGENTS.md` | This file — navigation guide and agent instructions |

---

## Project Directory Structure

| Directory | Purpose |
|-----------|---------|
| `src/Allocations/` | Memory allocation, Kokkos scratch storage |
| `src/Basis/` | Basis functions, evaluation, caching, reconciliation, cubature |
| `src/CellTools/` | Cell topology operations, reference-physical mappings |
| `src/ConvergenceStudy/` | h-convergence study automation |
| `src/Debug/` | Debug utilities, time logging |
| `src/DofOrdering/` | DOF ordering within elements |
| `src/Formulations/` | DPG formulation classes (34 files) |
| `src/Function/` | Function class hierarchy (38+ files) |
| `src/include/` | Header files for the entire library (~288 headers) |
| `src/IO/` | HDF5 and GnuPlot exporters |
| `src/MatrixUtilities/` | Dense matrix wrappers, Epetra/Tpetra converters |
| `src/Mesh/` | Mesh topology, factory, DOF assignment, refinement |
| `src/Problem/` | BCs, RHS, bilinear form utilities, DPG machinery |
| `src/Solution/` | Linear system assembly, solve, GMG, condensation |
| `src/TestingUtilities/` | Test support utilities |
| `drivers/` | Application driver programs |
| `examples/` | User-facing example programs |
| `manual-examples/` | Code examples tied to user manual chapters |
| `unit_tests/` | Modern unit tests (separate from legacy DPGTests) |

---

## Key Architectural Patterns

### 1. DPG Workflow
Every DPG computation follows: VarFactory → BF (bilinear form) → MeshFactory → BC/RHS → Solution → Solver → (optional refinement loop) → HDF5Exporter.

### 2. Smart Pointers
All major objects use `Teuchos::RCP` smart pointers. Typedefs: `MeshPtr`, `BFPtr`, `SolutionPtr`, `FunctionPtr`, `VarPtr`, `IPPtr`, `BCPtr`, `RHSPtr`.

### 3. Expression Trees
Functions are composed via operator overloading (+, -, *, /) into expression trees. Symbolic differentiation propagates through the tree.

### 4. Static Condensation
Interior (field) DOFs are eliminated locally; only skeleton (trace/flux) DOFs enter the global system. Controlled by CondensedDofInterpreter.

### 5. Mesh Registration
Solutions register with Mesh to receive refinement callbacks for automatic solution projection.

---

## Known Code Smells and Anti-Patterns

### High Priority
1. **Oversized files**: Function.cpp (166KB), VlasovSpaceTimeFormulation.cpp (136KB), BasisReconciliation.cpp (94KB), MeshTestSuite.cpp (110KB)
2. **Code duplication**: Boltzmann formulations and PhasePhysicalMeshMap copied between experimental/ and papers/ directories
3. **No common formulation base class**: Each formulation independently reimplements the same API pattern

### Medium Priority
4. **Global singleton state**: BasisFactory singleton with mutable configuration (thread-safety risk)
5. **Legacy test framework**: DPGTests uses custom bool-returning framework instead of standard (GTest/Catch2)
6. **Epetra dependency**: pulse/ drivers and some core code depend on deprecated Epetra (migration to Tpetra in progress)
7. **Dual evaluation paths**: Function base class has fragile mutual-recursion between host and device evaluation
8. **Missing test coverage**: No dedicated unit tests for Vlasov, MHD, acoustics, elasticity formulations

### Low Priority
9. **Commented-out code**: Throughout drivers and some core files
10. **Inconsistent naming**: Driver vs Example, year-stamped files, doubled names (ColdDiodeProblemProblem)
11. **Undocumented shell scripts**: In papers/ and Stefan/ directories

---

## Guidance for Future Agents

### Working with FileShelf JSON Change Sets

1. Use schema `fileshelf-change-set-v1` exactly.
2. Required top-level fields: `schema`, `fileshelf_document` (may be empty), `root_directory` (absolute path), `overall_summary`, `changes`.
3. The project root is `/Users/nvrober/development-2.0`.
4. For new files use `kind: create_file` with `contents` (not `content`).
5. For editing existing files use `kind: find_replace` (file must be on shelf) or `kind: replace_file`.
6. Do NOT use `kind: patch`.
7. For long files, break creation into chunks: create with CONTINUE_HERE marker, then find_replace to append.
8. Always `validate_json_change_set` before `autotest_json_change_set`.
9. Files created via autotest are on disk but NOT automatically on the shelf. Use `request_manifest_file` to add them to the shelf before editing.

### Code Modification Guidelines

1. **Do not modify code without understanding the DPG workflow** — changes to BF, IP, BC, or Solution can break the mathematical formulation.
2. **Test coverage is uneven** — verify changes against both unit_tests/ and drivers/DPGTests/.
3. **Static condensation is pervasive** — changes to DOF ordering or Solution assembly must account for CondensedDofInterpreter.
4. **Function expression trees** — operator overloads create composite objects; changes to Function subclasses must preserve derivative semantics.
5. **MPI awareness** — most operations are distributed; test with multiple ranks when modifying Mesh, Solution, or GMG code.
6. **Epetra vs Tpetra** — the codebase is transitioning from Epetra to Tpetra. New code should prefer Tpetra.

### RLM Workflow Tips

1. Use `rlm_query` for complex subtasks requiring iterative retrieval.
2. Decompose by subsystem, not by arbitrary file batches.
3. Child tasks should create their output files directly via change sets when possible.
4. Use `search_shelved_files` for targeted discovery; `read_shelved_file_lines` for focused reading.
5. For documentation tasks, the chunked creation pattern (create + find_replace CONTINUE_HERE) works well.

---

## Active Issues and Future Work

See `/Users/nvrober/work-sessions/camellia-future-work.md` for:
- Known test failures (issue #287)
- Epetra-to-Tpetra transition status
- Planned tensor bilinear-form abstractions
- Vlasov/Maxwell simulation campaigns
