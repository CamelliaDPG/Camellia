# 05 — Driver Programs

Catalogs all driver/example/test programs, organized by subdirectory.

---

## 1. drivers/Brendan/ (9 files)

Research drivers by Brendan Keith: acoustics, viscoelasticity, goal-oriented adaptivity, elasticity, GMG.
- AcousticsDriver, AcousticsTomographyDriver, GoalOrientedDPGDriver
- GoalOrientedOldroydBDriver, OldroydBDriver
- LinearElasticity2D/3D/3DGMG, StVenantKirchhoffElasticity3D

**Issues**: LinearElasticity 2D/3D/GMG share large blocks of near-identical code. Hardcoded paths/parameters.

---

## 2. drivers/Convection/ (2 files)

Convecting-cone benchmark and space-time prototype variant.

---

## 3. drivers/DPGTests/ (~63 files)

Legacy unit/integration test suite with custom TestSuite.h framework.

Test classes: ConfusionBilinearForm, CurvilinearMeshTests, DPGTests, ElementTests, FunctionTests, GDAMinimumRuleTests, GMGTests, HConvergenceStudyTests, IncompressibleFormulationsTests, LinearTermTests, LobattoBasisTests, MeshRefinementTests, MeshTestSuite, MeshTopologyTests, MPIWrapperTests, MultiBasisTests, ParametricCurveTests, PatchBasisTests, RHSTests, ScratchPadTests, SerialDenseMatrixUtilityTests, SolutionTests, VectorizedBasisTestSuite.

**Issues**: Custom framework (not GTest/Catch2). Monolithic runner. ConfusionProblemLegacy is dead code. Header-only test stubs inconsistent.

---

## 4. drivers/experimental/ (~40 files across 14 subdirs)

### Subgroups
- **B-dot, Heat, Ions**: Magnetic induction, heat equation, ion transport (IonsDriverForRefinementBug created to reproduce a bug)
- **mlc, NonconformingPoisson, ProximalGalerkin**: One-off experiments
- **ProtoBrioWu, ShearAlfvenWave**: MHD benchmarks
- **QuickTests**: Lightweight test subset
- **SpaceTimeMixed***: Space-time mixed Poisson and Vlasov
- **TensorCompression**: Tensor-train verification
- **TwoStreamInstability** (largest group, ~26 files): Boltzmann/Vlasov formulations (1D1V, 1D3V), ColdDiode, LandauDamping, TwoStreamInstability, PhasePhysicalMeshMap

**Issues**: ColdDiodeProblemProblem.cpp (doubled name). IonsDriverForRefinementBug should be regression test. Code duplicated with papers/. Many appear abandoned.

---

## 5. drivers/papers/ (~27 files)

Publication-supporting drivers:
- **RelativisticVlasovSpaceTime**: ColdDiode, Landau, PlanarMITL relativistic drivers
- **VlasovDiode**: Duplicates Boltzmann1D1VFormulation, PhasePhysicalMeshMap from experimental/
- **VlasovPoissonBackwardEuler**: BumpOnTail, LandauDamping, TwoStream + shell scripts
- **VlasovPoissonSpaceTime**: ColdDiode, Landau, SpaceTimeVlasovPoisson

**Issues**: Massive duplication with experimental/. Common code should be in src/. Shell scripts undocumented.

---

## 6. drivers/Preconditioning/ (6 files)

Condition number tests, multigrid preconditioning, Stokes/NS cavity flow with GMG.
Includes year-stamped PreconditioningTests2025Driver alongside undated variant.

---

## 7. drivers/pulse/ (~19 files)

BoBa external solver integration: DPG assembly, Vlasov system, compression, Epetra-to-BoBa conversion.

**Issues**: Local utility code in driver dir instead of src/. Epetra dependency (deprecated).

---

## 8. drivers/Stefan/ (3 files)

Heat steady/transient drivers with undocumented run.sh.

---

## 9. drivers/performance/ (1 file)

StokesAssemblyPerformanceTest.cpp. Maintenance status unclear.

---

## 10. Cross-Cutting Patterns

1. Every driver is a standalone main() program
2. Common flow: VarFactory → BF → MeshFactory → BC/RHS → Solution → Solver → HDF5Exporter
3. MPI init via Teuchos::GlobalMPISession
4. Manual adaptive refinement loops with energy-error marking

---

## 11. Cross-Cutting Issues

| Issue | Severity |
|-------|----------|
| Massive code duplication (experimental/ vs papers/) | High |
| No shared driver infrastructure/arg parsing | Medium |
| Naming inconsistencies (Driver vs Example, doubled names) | Low |
| Legacy/dead code (ConfusionProblemLegacy, bug repro drivers) | Medium |
| Tests in driver dirs with custom framework | Medium |
| Epetra dependency in pulse/ | Medium |
| Undocumented shell scripts | Low |
| 14+ experimental subdirs with no status tracking | Medium |
