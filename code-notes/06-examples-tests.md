# 06 - Examples, Manual Examples, and Unit Tests

## Overview

Camellia ships three directories of executable code that demonstrate its API and verify correctness:

| Directory | Purpose | File count | Framework |
|-----------|---------|------------|-----------|
| `examples/` | Standalone demo programs showing how to solve various PDEs | ~28 files across 16 subdirectories | `main()` drivers |
| `manual-examples/` | Companion code for the Camellia user manual | 5 files across 3 subdirectories | `main()` drivers |
| `unit_tests/` | Teuchos unit tests covering core library functionality | ~95 files (including 1 experimental subdir) | Teuchos UnitTestHarness |

---

## Examples Directory (`examples/`)

### Organization

Each subdirectory contains a single `*Example.cpp` (or similar) file that is a self-contained `main()` program.

### Catalog of Examples

#### Elliptic / Steady-State Problems

| File | Subdirectory | Physics | Formulation | Complexity |
|------|-------------|---------|-------------|------------|
| `Poisson1D.cpp` | Poisson1D | 1D Poisson | `PoissonFormulation` | Introductory |
| `Poisson1DMOAB.cpp` | Poisson1DMOAB | 1D Poisson with MOAB mesh | `PoissonFormulation` | Introductory |
| `Poisson2DTriangles.cpp` | Poisson2DTriangles | 2D Poisson on triangles | `PoissonFormulation` | Introductory |
| `PoissonMatrixOutput.cpp` | PoissonMatrixOutput | Poisson with matrix export | `PoissonFormulation` | Intermediate |
| `PoissonPrimal.cpp` | PoissonPrimal | Poisson in primal (H1) form | Standard H1 FEM | Introductory |
| `StokesCavityFlowExample.cpp` | StokesCavityFlow | Lid-driven cavity (Stokes) | `StokesVGPFormulation` | Intermediate |
| `StokesBackwardFacingStepExample.cpp` | StokesBackwardFacingStep | Backward-facing step | `StokesVGPFormulation` | Intermediate |
| `StokesGramOutput.cpp` | StokesCavityFlow | Gram matrix export for Stokes | `StokesVGPFormulation` | Debug/utility |
| `NavierStokesCavityFlowExample.cpp` | NavierStokesCavityFlow | Nonlinear cavity (Navier-Stokes) | `NavierStokesVGPFormulation` | Advanced |
| `IPWithTraceTerms.cpp` | IPWithTraceTerms | Interior penalty method demo | SIPG | Intermediate |

#### Hyperbolic / Advection Problems

| File | Subdirectory | Physics | Formulation | Complexity |
|------|-------------|---------|-------------|------------|
| `CGAdvectionExplicitExample.cpp` | CGAdvectionExplicit | CG advection, explicit RK | Standard CG | Intermediate |
| `DGAdvectionExample.cpp` | DGAdvection | DG advection, upwind flux | DG with numerical flux | Intermediate |
| `DPGAdvectionExample.cpp` | DPGAdvection | DPG advection, optimal test | Ultra-weak DPG | Intermediate |
| `ConvectingConeExample.cpp` | ConvectingCone | Rotating cone test | Advection | Intermediate |

#### Convection-Diffusion-Reaction

| File | Subdirectory | Physics | Notes |
|------|-------------|---------|-------|
| `ConvectionDiffusionExample.cpp` | ConvectionReactionDiffusion | Convection-dominated diffusion | `ConvectionDiffusionReactionFormulation`; handles boundary layers |
| `ConditionNumberMinimizationExperiment.cpp` | ConvectionReactionDiffusion | Condition number study | Experimental / research utility |
| `PoissonHessianGradientIndicatorExperiment.cpp` | ConvectionReactionDiffusion | Error indicator experiment | Tiny (~1 KB); research utility |
| `RebuildLookupsPerformanceTest.cpp` | ConvectionReactionDiffusion | Performance benchmark | Measures lookup rebuild time |

#### Compressible Flow / Shock Problems

| File | Subdirectory | Physics | Formulation |
|------|-------------|---------|-------------|
| `SodShockTubeExample.cpp` | SodShockTube | Compressible Euler (Sod) | `CompressibleNavierStokesConservationForm` |
| `StandingShockConservationFormExample.cpp` | SodShockTube | Standing shock (conservation) | Conservation form |
| `StandingShockExample.cpp` | SodShockTube | Standing shock (primitive) | Primitive variable form |
| `BrioWuShockTubeExample.cpp` | BrioWuShockTube | Ideal MHD Brio-Wu shock tube | `IdealMHDFormulation` |
| `CircularlyPolarizedAlfvenWaveExample.cpp` | CircularlyPolarizedAlfvenWave | MHD Alfven wave | `IdealMHDFormulation` |

#### Space-Time / Transient Problems

| File | Subdirectory | Physics | Formulation |
|------|-------------|---------|-------------|
| `HeatSpaceTimeExample.cpp` | HeatSpaceTime | Heat equation in space-time | `SpaceTimeHeatFormulation` |
| `StokesCavityFlowSpaceTimeExample.cpp` | StokesCavityFlow | Transient Stokes in space-time | `SpaceTimeIncompressibleFormulation` |

#### Mesh I/O

| File | Subdirectory | Notes |
|------|-------------|-------|
| `MOABReaderExample.cpp` | MOABReader | Reads `.h5m` (MOAB/HDF5) mesh files; ships with `quad_1000.h5m` |

### Common API Patterns in Examples

1. **Mesh creation** - `MeshFactory::rectilinearMesh(...)` or `MeshFactory::quadMesh(...)` for structured grids.
2. **Formulation instantiation** - e.g., `PoissonFormulation form(spaceDim, useConformingTraces)`.
3. **Boundary conditions** - `form.solution()->bc()->addDirichlet(...)`.
4. **Solve** - `form.solution()->solve()` or `GMGSolver` for multigrid.
5. **Adaptive refinement** - `RefinementStrategy::hRefine(energyError)` loop.
6. **Export** - `HDF5Exporter` for ParaView, or `GnuPlotUtil` for 1D.
7. **Nonlinear iteration** - Outer Newton/Picard loop calling `solve()` + `addSolution(backgroundFlow, 1.0)`.

---

## Manual Examples Directory (`manual-examples/`)

### Purpose

These files accompany specific chapters of the Camellia user manual. They are intentionally simple and pedagogical.

### Catalog

| File | Chapter | Physics | Formulation | Size |
|------|---------|---------|-------------|------|
| `ch03-StokesCavity.cpp` | Ch 3 | Stokes cavity flow | `StokesVGPFormulation` | ~4.4 KB |
| `ch04-NavierStokesCavity.cpp` | Ch 4 | Navier-Stokes cavity | `NavierStokesVGPFormulation` | ~6.5 KB |
| `ch05-NavierStokesCavity.cpp` | Ch 5 | Alternative NS formulation | `NavierStokesVGPFormulation` | ~3.2 KB |
| `ch05-PoissonHomogeneous.cpp` | Ch 5 | Homogeneous Poisson | `PoissonFormulation` | ~1.5 KB |
| `ch05-StokesCavity.cpp` | Ch 5 | Stokes (formulation focus) | `StokesVGPFormulation` | ~3.2 KB |

### Progression

- **Chapter 3**: Introduces basic DPG workflow: mesh, formulation, BCs, solve, export.
- **Chapter 4**: Adds nonlinearity (convective term), Newton iteration.
- **Chapter 5**: Compares formulation variants; shows how the same PDE can use different trial/test spaces.

### Notes

- Manual examples are minimal; they omit adaptive refinement, command-line parsing, and advanced solver options.
- They serve as copy-paste starting points for new problems.

---

## Unit Tests Directory (`unit_tests/`)

### Framework

All tests use the **Teuchos UnitTestHarness** (`TEUCHOS_UNIT_TEST` macro). They are compiled into a single test executable and run via CTest.

### Organization

Files follow a `<ClassName>Tests.cpp` naming convention. There is one `experimental/` subdirectory for work-in-progress tests.

### Complete Test File Inventory

#### Basis and Quadrature Tests
| File | What It Tests |
|------|--------------|
| `Basis1DIntegralCacheTests.cpp` | Caching of 1D basis integrals |
| `BasisCacheFactoryTests.cpp` | BasisCache construction from topology |
| `BasisCacheTests.cpp` | Physical-to-reference mappings, Jacobians |
| `BasisFactoryTests.cpp` | Basis creation by type and order |
| `BasisFamilyTests.cpp` | Hierarchical basis family consistency |
| `BasisReconciliationTests.cpp` | Hanging-node constraint reconciliation |
| `BasisSumFunctionTests.cpp` | Weighted sums of basis functions |
| `BasisTests.cpp` | Core basis evaluation correctness |
| `TensorBasisTests.cpp` | Tensor-product basis construction |
| `Intrepid2_HierarchicalBasisProlongatorTests.cpp` | Prolongation operators between levels |

#### Cell and Topology Tests
| File | What It Tests |
|------|--------------|
| `CamelliaCellToolsTests.cpp` | Cell geometry computations (Jacobians, normals) |
| `CellDataMigrationTests.cpp` | MPI cell data redistribution |
| `CellTests.cpp` | Cell entity queries, children, parents |
| `CellTopologyTests.cpp` | Topological queries (vertices, edges, faces, tensorial extrusion) |
| `SubcellDofIndicesTests.cpp` | DOF indexing on subcells |
| `ComponentCellID1DTests.cpp` | 1D component cell ID mapping |

#### Mesh Tests
| File | What It Tests |
|------|--------------|
| `MeshTests.cpp` | General mesh construction, queries, distributed mesh |
| `MeshToolsTests.cpp` | Mesh utility operations |
| `MeshTopologyTests.cpp` | Topology construction, refinement tracking, entity parents |
| `MeshTopologyViewTests.cpp` | Read-only topology views |
| `MeshRefinementTests.cpp` | h-refinement correctness |
| `MeshBasisIntegralTests.cpp` | Mesh-level integral caching |
| `MeshPolyOrderFunctionTests.cpp` | Polynomial order queries as functions |
| `MeshTransferFunctionTests.cpp` | Solution transfer between meshes |
| `MeshTransformationFunctionTests.cpp` | Curvilinear mesh transformations |
| `ExtrudedMeshTopologyTests.cpp` | Space-time mesh extrusion |
| `ComponentMeshTopology1DTests.cpp` | 1D component mesh topology |
| `ActiveLeafSet1DTests.cpp` | Active leaf cell tracking in 1D |

#### DOF Ordering Tests
| File | What It Tests |
|------|--------------|
| `DofOrderingTests.cpp` | DOF ordering within elements |
| `DofOrderingFactoryTests.cpp` | DOF ordering creation and consistency |
| `TensorDofLayoutTests.cpp` | Tensor-product DOF layouts |

#### Function Tests
| File | What It Tests |
|------|--------------|
| `FunctionTests.cpp` | Core Function evaluation, arithmetic, derivatives |
| `FunctionsTests.cpp` | Convenience function wrappers |
| `FunctionValuesTests.cpp` | Cached function value evaluation |
| `AbsFunctionTests.cpp` | Absolute value function |
| `ConstantScalarFunctionTests.cpp` | Constant scalar functions |
| `ConstantVectorFunctionTests.cpp` | Constant vector functions |
| `ExpFunctionTests.cpp` | Exponential function |
| `SqrtFunctionTests.cpp` | Square root function |
| `MinMaxFunctionsTests.cpp` | Min/max function combinations |
| `MonomialFunctionsTests.cpp` | Monomial functions (x^a * y^b * ...) |
| `ProductFunctionTests.cpp` | Product of two functions |
| `SumFunctionTests.cpp` | Sum of two functions |
| `TensorFunctionTests.cpp` | Tensor-valued functions |
| `PolarizedFunctionTests.cpp` | Polar/cylindrical coordinate functions |
| `BGKFunctionTests.cpp` | Bhatnagar-Gross-Krook collision model |
| `MLCollisionFunctionTests.cpp` | Machine-learning collision operator |
| `SimpleSolutionFunctionTests.cpp` | Simplified solution function wrappers |
| `SolutionMomentFunctionTests.cpp` | Moment computation from solutions |
| `SpaceTimeBasisCacheTests.cpp` | Basis cache for space-time elements |

#### Bilinear Form, Linear Term, and IP Tests
| File | What It Tests |
|------|--------------|
| `BFTests.cpp` | Bilinear form assembly |
| `LinearTermTests.cpp` | Linear term (RHS) evaluation |
| `IPTests.cpp` | Inner product definitions |
| `TensorBilinearFormTests.cpp` | Tensor-product bilinear forms |

#### Solution and Solver Tests
| File | What It Tests |
|------|--------------|
| `SolutionTests.cpp` | Solution solve, project, import/export |
| `BCTests.cpp` | Boundary condition application |
| `ConstraintTests.cpp` | Linear constraints (Lagrange multipliers) |
| `RieszRepTests.cpp` | Riesz representation computation |
| `ProjectorTests.cpp` | L2 and H1 projection |
| `ErrorIndicatorTests.cpp` | Energy error indicator |
| `RefinementPatternTests.cpp` | Refinement pattern geometry |
| `RefinementStrategyTests.cpp` | Adaptive refinement strategies |
| `SpatialIntegralOfSpaceTimeSolutionTests.cpp` | Spatial integration of space-time solutions |

#### Preconditioner and Solver Tests
| File | What It Tests |
|------|--------------|
| `GMGOperatorTests.cpp` | Geometric multigrid operator |
| `GMGSolverTests.cpp` | Geometric multigrid solver convergence |
| `AdditiveSchwarzTests.cpp` | Additive Schwarz domain decomposition |
| `OverlappingRowMatrixTests.cpp` | Overlapping row matrix for DD |
| `PreconditioningTests.cpp` | General preconditioner interface |
| `ComponentTransferOperator1DTests.cpp` | 1D transfer operators for multigrid |

#### Formulation-Specific Tests
| File | What It Tests |
|------|--------------|
| `ConvectionDiffusionReactionFormulationTests.cpp` | CDR formulation correctness |
| `CompressibleNavierStokesConservationFormulationTests.cpp` | Compressible NS (conservation) |
| `CompressibleNavierStokesFormulationTests.cpp` | Compressible NS (primitive) |
| `ConfigurationSpaceFormulationMaxwell1D2VTests.cpp` | Maxwell in config space |
| `ConfigurationSpaceFormulationPoissonTests.cpp` | Poisson in config space |
| `HeatFormulationTests.cpp` | Heat equation formulation |
| `IdealMHDFormulationTests.cpp` | Ideal MHD formulation |
| `NavierStokesVGPFormulationTests.cpp` | Navier-Stokes VGP |
| `StokesVGPFormulationTests.cpp` | Stokes VGP |
| `SpaceTimeHeatDivFormulationTests.cpp` | Space-time heat (div form) |
| `SpaceTimeHeatFormulationTests.cpp` | Space-time heat |
| `SpaceTimeTests.cpp` | General space-time infrastructure |
| `MimeticVlasovSpaceTimeFormulationTests.hpp` | Mimetic Vlasov (header-only tests) |
| `GoalOrientedTests.cpp` | Goal-oriented adaptivity |

#### GDA (Global DOF Assembly) Tests
| File | What It Tests |
|------|--------------|
| `GDAMinimumRuleTests.cpp` | Minimum-rule global assembly, hanging nodes |

#### Utility and Infrastructure Tests
| File | What It Tests |
|------|--------------|
| `FieldContainerTests.cpp` | Intrepid FieldContainer wrappers |
| `SerialDenseWrapperTests.cpp` | Dense matrix operations |
| `Epetra_Operator_to_Epetra_MatrixTests.cpp` | Trilinos operator conversion |
| `MPIWrapperTests.cpp` | MPI communication helpers |
| `MPIGlobalMinMaxTests.cpp` | Global min/max reductions |
| `RangeListTests.cpp` | Integer range list utilities |
| `ERKTableauTests.cpp` | Explicit Runge-Kutta tableaux |

#### Integration / Smoke Tests
| File | What It Tests |
|------|--------------|
| `ExampleTests.cpp` | Runs selected examples as integration tests |

#### Experimental Tests
| File | What It Tests |
|------|--------------|
| `experimental/DLSTests.cpp` | Discrete Least Squares (experimental) |

### Relationship Between `unit_tests/` and `drivers/DPGTests/`

Camellia contains **two** test suites:

1. **`unit_tests/`** - Modern Teuchos-based unit tests. Each file tests one class or feature in isolation. This is the active, maintained test suite.
2. **`drivers/DPGTests/`** - Legacy test suite with a custom `TestSuite` harness (`TestSuite.h`). Files like `DPGTests.cpp`, `SolutionTests.cpp`, and `MeshTestSuite.cpp` contain large monolithic test collections. **This suite appears partially abandoned**; many tests duplicate `unit_tests/` coverage with older APIs.

### Test Coverage Observations

**Well-covered areas:**
- Basis evaluation and caching
- Mesh topology and refinement
- Function arithmetic and evaluation
- Core formulations (Poisson, Stokes, Navier-Stokes, Heat, Compressible NS, MHD)
- Multigrid (GMG operator and solver)
- Space-time infrastructure

**Gaps or thin coverage:**
- `VlasovSpaceTimeFormulation` and `VlasovBackwardEulerFormulation` have no dedicated unit tests (tested only through drivers)
- `OldroydBFormulation*` (viscoelastic) has no unit tests
- `InviscidBurgersFormulation` has no unit tests
- `LinearElasticityFormulation` has no unit tests
- `PressurelessStokesFormulation` has no unit tests
- `RelativisticVlasovSpaceTimeFormulation` has no unit tests
- Several `Function` subclasses lack dedicated tests: `BCFunction`, `EnergyErrorFunction`, `hFunction`, `MaxwellianFunction`, `MeshPolyOrderFunction` (only minimal), `NeighborFunction`, `ParameterFunction`, `ParametricCurve`, `ParametricSurface`, `PreviousSolutionFunction`, `QuotientFunction`, `SideParityFunction`, `SpatiallyFilteredFunction`, `TimeRampFunction`
- `DofOrderingFactory` has tests but `DofOrdering` tests are basic
- `HConvergenceStudy` tested only in legacy suite

---

## Code Smells and Anti-Patterns

### In Examples

1. **Monolithic main() functions** - Most examples place all logic in a single `main()`, making them hard to reuse as library code. Some (e.g., `ConvectionDiffusionExample.cpp` at ~46 KB) are extremely long.
2. **Duplicated boilerplate** - Command-line parsing, mesh creation, and export code is copy-pasted across examples rather than factored into shared utilities.
3. **Mixed concerns** - Some files in `examples/ConvectionReactionDiffusion/` are research experiments (`ConditionNumberMinimizationExperiment.cpp`, `PoissonHessianGradientIndicatorExperiment.cpp`) or performance benchmarks (`RebuildLookupsPerformanceTest.cpp`) rather than user-facing examples.
4. **Inconsistent naming** - Some files end with `Example.cpp`, others with `Driver.cpp`, and some have no suffix convention.

### In Manual Examples

5. **Unclear formulation versioning** - `ch05-NavierStokesCavity.cpp` vs `ch04-NavierStokesCavity.cpp` solve the same problem with subtle differences that are not documented in code comments.

### In Unit Tests

6. **Legacy test suite coexistence** - `drivers/DPGTests/` duplicates significant coverage with `unit_tests/`, creating maintenance burden and confusion about which suite is authoritative.
7. **Header-only test file** - `MimeticVlasovSpaceTimeFormulationTests.hpp` is a `.hpp` in a directory of `.cpp` files, suggesting it may be included from elsewhere rather than compiled directly.
8. **Missing formulation tests** - Several formulations in `src/Formulations/` have no corresponding test file (see gaps listed above).
9. **Large test files** - `SolutionMomentFunctionTests.cpp` (~32 KB), `GDAMinimumRuleTests.cpp` (~27 KB), and `MeshTests.cpp` (~27 KB) are very large; consider splitting by feature.

### Potentially Abandoned Code

- **`drivers/DPGTests/`**: Legacy test framework with custom `TestSuite.h`. Many files have not been updated to use Teuchos macros. Likely candidates for removal or migration.
- **`examples/MOABReader/`**: MOAB support may be a compile-time option that is rarely used.
- **`examples/ConvectionReactionDiffusion/` experiments**: The `ConditionNumberMinimization` and `PoissonHessianGradientIndicator` files look like one-off research experiments.

---

## Cross-References

- Formulation implementations: see `code-notes/03-formulations.md`
- Function hierarchy: see `code-notes/04-functions.md`
- Mesh and topology: see `code-notes/02-mesh-topology.md`
- Solver infrastructure: see `code-notes/05-solution-solver.md`
- Driver programs (research): see `code-notes/07-drivers.md`
