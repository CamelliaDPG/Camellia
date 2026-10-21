# GMG Transfer and hp3d/Petrides-Style Options Analysis

*Updated analysis based on detailed inspection of Camellia GMGOperator.h and hp3d coarse2macro.F90, macro2coarse.F90, prolongation.F90, and related modules. Supersedes previous stub.*

## 1. Current State: Camellia's GMGOperator

### 1.1 Class Overview

GMGOperator (src/include/GMGOperator.h, ~19 KB; GMGOperator.cpp, ~28 KB) inherits from TOperator<double> and Narrator. It implements the full geometric multigrid cycle as a Trilinos/Tpetra operator: prolongation construction, smoothing, restriction, coarse-level solve, and multi-level recursion.

### 1.2 Multigrid Strategy Enum

    enum MultigridStrategy {
      TWO_LEVEL,        // aka ADDITIVE: fine smoother + coarse correction added
      V_CYCLE,          // standard V-cycle
      W_CYCLE,
      FULL_MULTIGRID_V,
      FULL_MULTIGRID_W
    };

For TWO_LEVEL, the smoother is applied once. For V_CYCLE/W_CYCLE, the smoother is applied `getSmootherApplicationCount()` times before and after each recursive coarse-level application.

### 1.3 Smoother Options

    enum SmootherChoice {
      POINT_JACOBI,
      POINT_SYMMETRIC_GAUSS_SEIDEL,
      BLOCK_JACOBI,
      BLOCK_SYMMETRIC_GAUSS_SEIDEL,
      IFPACK_ADDITIVE_SCHWARZ,
      CAMELLIA_ADDITIVE_SCHWARZ,   // cells as Schwarz blocks; predictable MPI scaling
      CHEBYSHEV,
      NONE
    };

Key distinction: CAMELLIA_ADDITIVE_SCHWARZ defines cells as Schwarz blocks, giving more predictable MPI scaling than Ifpack rank-level blocks. Schwarz weighting options include diagonal weight and scaling-weight variants.

### 1.4 Key Private Members

| Member | Type | Purpose |
|--------|------|----------|
| `_coarseSolution` | TSolutionPtr<Scalar> | Solution on coarse grid |
| `_coarseOperator` | RCP<GMGOperator> | Recursive coarse GMGOperator |
| `_coarseSolver` | RCP<Solver> | Direct solver at coarsest level |
| `_fineStiffnessMatrix` | TMatrixPtr<Scalar> | Fine-level stiffness matrix |
| `_P` | TMatrixPtr<double> | Prolongation operator (Tpetra CrsMatrix) |
| `_br` | RCP<BasisReconciliation> | Shared basis reconciliation |
| `_localCoefficientMap` | map<pair<pair<int,int>,RefinementBranch>,LocalDofMapperPtr> | Per-cell coefficient maps |
| `_fluxDuplicationMap` | map<DofOrdering*,RCP<SerialDenseMatrix>> | Flux duplication for h-refinement + static condensation |
| `_multigridStrategy` | MultigridStrategy | Active cycle type |
| `_smootherType` | SmootherChoice | Active smoother |
| `_isFinest` | bool | Whether this is the finest level |
| `_useStaticCondensation` | bool | Applies to both coarse and fine solves |
| `_haveSolvedOnCoarseMesh` | bool | Enables resolve() reuse |

### 1.5 Constructor Signatures

Two primary constructors accept MultigridStrategy:

    // With explicit coarse solver:
    GMGOperator(MultigridStrategy, BCPtr zeroBCs,
                MeshPtr coarseMesh, IPPtr coarseIP,
                MeshPtr fineMesh, RCP<DofInterpreter> fineDofInterpreter,
                TMapPtr finePartitionMap, RCP<Solver> coarseSolver,
                bool useStaticCondensation,
                RCP<BasisReconciliation> sharedBasisReconciliation);

    // Nested recursion (no explicit solver):
    GMGOperator(MultigridStrategy, BCPtr zeroBCs,
                MeshPtr coarseMesh, IPPtr coarseIP,
                MeshPtr fineMesh, RCP<DofInterpreter> fineDofInterpreter,
                TMapPtr finePartitionMap,
                bool useStaticCondensation,
                RCP<BasisReconciliation> sharedBasisReconciliation);

    // Legacy coarsest-level 2-level operator:
    GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                MeshPtr fineMesh, RCP<DofInterpreter> fineDofInterpreter,
                TMapPtr finePartitionMap, bool useStaticCondensation,
                RCP<BasisReconciliation> sharedBasisReconciliation);

All constructors take sharedBasisReconciliation (shared across levels) and useStaticCondensation.

### 1.6 Key Public Methods

| Method | Description |
|--------|-------------|
| `constructProlongationOperator()` | Builds and caches _P (fine rows, coarse cols) |
| `constructLocalCoefficientMaps()` | Eagerly builds per-cell coefficient maps (isolates timing) |
| `computeCoarseStiffnessMatrix(A_f)` | Computes A_c = P^T * A_f * P |
| `setFineStiffnessMatrix(A)` | Triggers computeCoarseStiffnessMatrix() and setUpSmoother() |
| `getLocalCoefficientMap(fineCellID)` | Returns LocalDofMapperPtr for a fine cell (public for testing) |
| `getCoarseCellID(fineCellID)` | Maps fine cell to its coarse parent (public for testing) |
| `getProlongationOperator()` | Returns cached _P Tpetra CrsMatrix |
| `getCoarseStiffnessMatrix()` | Returns Galerkin-coarsened stiffness |
| `getFineMesh()` | Returns fine mesh ptr |
| `apply(X, Y, mode, alpha, beta)` | Tpetra operator application: full MG cycle |
| `computeResidual(Y, res, A_Y)` | Computes residual for cycle logic |
| `setMultigridStrategy(choice)` | Changes cycle type |
| `setSmootherType(choice)` | Changes smoother |
| `setFineCoarseRolesSwapped(value)` | Transposes P construction and use |
| `getOperatorLevel()` | Returns level index from 0 (coarsest) |

### 1.7 Transfer Construction: The getLocalCoefficientMap Mechanism

The heart of prolongation construction is the **local coefficient map** mechanism:

1. For each fine cell, `getCoarseCellID()` finds the parent coarse cell via mesh topology ancestor query.
2. `getLocalCoefficientMap(fineCellID)` returns a LocalDofMapperPtr mapping coarse-cell local DOFs to fine-cell local DOFs. Maps are cached by (fineH1Order, coarseH1Order, RefinementBranch).
3. Maps are built via BasisReconciliation, which computes sub-basis projection weights between the coarse basis and the fine sub-basis restricted to the fine element domain.
4. The global prolongation matrix _P is assembled by iterating over fine cells and scattering local coefficient maps into the global Tpetra CrsMatrix.

For **static condensation + h-refinement**, _fluxDuplicationMap is additionally needed because DPG trace conformity requires duplicating flux entries for each fine sub-face of a subdivided coarse face.

### 1.8 Timing Instrumentation

GMGOperator profiles extensively via mutable double fields:
- _timeMapFineToCoarse, _timeMapCoarseToFine
- _timeCoarseImport, _timeConstruction, _timeCoarseSolve
- _timeLocalCoefficientMapConstruction
- _timeComputeCoarseStiffnessMatrix
- _timeProlongationOperatorConstruction
- _timeSetUpSmoother, _timeUpdateCoarseOperator
- _timeApplyFineStiffness, _timeApplySmoother

### 1.9 Design Strengths

- Recursive multi-level via _coarseOperator chain
- Galerkin coarsening ensures automatic coarse-level consistency
- DPG-aware trace handling via flux duplication maps under h-refinement + static condensation
- BasisReconciliation sharing across levels reduces construction overhead
- Extensive timing instrumentation for performance analysis
- V/W-cycle support via MultigridStrategy enum

### 1.10 Current Design Limitations

- **Tightly coupled transfer**: constructProlongationOperator() and getLocalCoefficientMap() are deeply embedded in GMGOperator. No strategy interface exists for swapping transfer approaches.
- **Single-step coarse-to-fine**: Direct Galerkin projection; no intermediate macro-element level as in hp3d.
- **h-refinement assumption**: Transfer logic assumes parent-child cell relationships; pure p-refinement between levels is not separately handled.
- **Static condensation complexity**: _fluxDuplicationMap is ad hoc and tightly coupled to the DPG trace path.
- **No explicit trace/field separation**: Fields and traces are handled together in the same coefficient map pass.

---

## 2. hp3d's Decomposed Transfer Approach

### 2.1 Architecture Overview

hp3d's MGCG (Multigrid Conjugate Gradient) solver implements a **decomposed two-stage transfer pipeline** under src/solver/MGCG/prolongation/:

| File | Role |
|------|------|
| modules/prolongation.F90 (43 KB) | Orchestrating module: loops over elements, calls coarse2macro/macro2coarse |
| prolongation/coarse2macro.F90 (13 KB) | Stage 1: Prolongation coarse to macro-element level |
| prolongation/macro2coarse.F90 (13 KB) | Stage 1 adjoint: Restriction macro to coarse |
| prolongation/schur_compl_extension.F90 (1.5 KB) | Stage 2a: Trace/interface DOF extension |
| prolongation/schur_compl_restriction.F90 (1.4 KB) | Stage 2b: Trace/interface DOF restriction |
| solve/prolong_solution.F90 | Applies full pipeline to a solution vector |

### 2.2 Stage 1: coarse2macro (Prolongation)

Signature:

    subroutine coarse2macro(Igrid, Iel, Zu_c, Nrdof_c, Zu_m, Nrdof_m)
      use data_structure3D, only: NODES, get_index
      integer, intent(in)  :: Igrid, Iel, Nrdof_c, Nrdof_m
      VTYPE,   intent(in)  :: Zu_c(nrdof_c)   ! coarse grid solution
      VTYPE,   intent(out) :: Zu_m(nrdof_m)   ! macro-element solution
    end subroutine

**Semantics**: Given coarse grid solution `Zu_c` for element `Iel` on grid `Igrid`, prolongs coarse DOFs to the **macro-element level**, producing `Zu_m`. The macro-element is the intermediate level corresponding to the coarse element support resolved at the next finer subdivision. Uses `NODES` from `data_structure3D` to traverse the element node tree and compute basis function expansions.

Key properties:
- Operates element-by-element (`Iel` index in orchestrating loop)
- `Igrid` identifies the grid level in the multi-level hierarchy
- Output `Zu_m` has the same element support as `Zu_c` but at macro-element resolution

### 2.3 Stage 1 Adjoint: macro2coarse (Restriction)

Signature:

    subroutine macro2coarse(Igrid, Iel, Zu_m, Nrdof_m, Zu_c, Nrdof_c)
      use data_structure3D, only: NODES, get_index
      integer, intent(in)  :: Igrid, Iel, Nrdof_m, Nrdof_c
      VTYPE,   intent(in)  :: Zu_m(nrdof_m)   ! macro-element residual
      VTYPE,   intent(out) :: Zu_c(nrdof_c)   ! coarse grid residual
    end subroutine  ! ~448 lines

**Semantics**: Given residual `Zu_m` at the macro-element level, restricts to coarse grid `Zu_c`. This is the adjoint (transpose) of coarse2macro. The ~448-line body handles different variable types (H1, HDiv, HCurl, L2) and their conformity constraints at macro-element interfaces.

### 2.4 Stage 2: Schur-Complement Extension/Restriction

The Schur complement modules handle **trace/interface unknowns** as a second distinct stage:

| Module | Size | Role |
|--------|------|------|
| schur_compl_extension.F90 | ~1550 tokens | Extends skeleton (trace) DOFs from macro to fine element |
| schur_compl_restriction.F90 | ~1368 tokens | Restricts skeleton DOFs from fine back to coarse |

By separating field and trace DOF transfer, hp3d can:
1. Handle trace conformity constraints explicitly
2. Apply different prolongation logic to fields vs. traces
3. Support DPG-style ultra-weak formulations where trace DOFs live only on element boundaries

### 2.5 Two-Stage Pipeline

Full prolongation (coarse to fine):

    coarse → [coarse2macro] → macro level → [schur_compl_extension] → fine grid

Full restriction (fine to coarse):

    fine grid → [schur_compl_restriction] → macro level → [macro2coarse] → coarse

This decomposition separates:
- **Basis projection** (coarse2macro/macro2coarse): maps DOF coefficients via basis expansion, uniform across variable types
- **Conformity extension** (Schur complement modules): handles additional trace/skeleton conformity constraints

### 2.6 The data_structure3D Substrate

Both coarse2macro and macro2coarse use `data_structure3D%NODES` and `get_index`. hp3d's NODES structure stores geometry, DOF counts/offsets per variable type (H1, HDiv, HCurl, L2), and parent-child relationships. This is roughly analogous to Camellia's MeshTopology + GlobalDofAssignment + GDAMinimumRule combined.

---

## 3. Detailed Comparison

### 3.1 Terminology Mapping

| hp3d Concept | Camellia Equivalent |
|-------------|---------------------|
| Igrid (grid level) | getOperatorLevel() |
| Iel (element natural order) | cellID (GlobalIndexType) |
| data_structure3D%NODES | MeshTopology + GlobalDofAssignment + GDAMinimumRule |
| Macro-element level | Parent cell level (h-refinement ancestry) |
| Zu_c / Zu_m | Column / row slices of _P prolongation matrix |
| coarse2macro + schur_compl_extension | constructProlongationOperator() (combined) |
| macro2coarse + schur_compl_restriction | P^T (restriction via transpose) |
| prolongation.F90 orchestrator | GMGOperator::constructProlongationOperator() |
| Schur complement trace extension | _fluxDuplicationMap + BasisReconciliation trace weights |

### 3.2 Structural Differences

#### Two-Stage vs. One-Stage Transfer

hp3d **explicitly separates** transfer into:
1. Basis projection (coarse2macro/macro2coarse) — pure polynomial basis expansion, all DOF types
2. Conformity extension (schur_compl_*) — trace/skeleton conformity as distinct post-processing

Camellia **combines both** into getLocalCoefficientMap(), with trace conformity handled implicitly via _fluxDuplicationMap.

**Implication**: In hp3d, the Schur complement modules can be replaced independently. In Camellia, changing trace treatment requires modifying getLocalCoefficientMap() logic.

#### Element-Local vs. Global Operator Assembly

hp3d operates **element-by-element** on local vectors (Zu_c, Zu_m); global assembly is done externally in the orchestrating module.

Camellia constructs the **global _P matrix** (Tpetra CrsMatrix) directly, scattering per-cell entries. This integrates directly with the Trilinos Krylov/preconditioner stack.

#### Static Condensation Interaction

Camellia's _P operates on the **condensed system** (skeleton DOFs only when useStaticCondensation=true). The _fluxDuplicationMap enforces conformity of trace DOFs across h-refined faces.

hp3d's Schur complement extension/restriction handles the skeleton-level transfer as an explicit, separate step after field-level transfer — more modular for the DPG case.

#### Multi-Level Recursion

Camellia: **Recursive** — _coarseOperator is another GMGOperator; terminates at _coarseSolver (KLU/MUMPS).

hp3d: **External loop** over Igrid in the MGCG driver; subroutines are level-agnostic.

### 3.3 Mathematical Differences

**Galerkin coarsening**: Both use A_c = P^T * A_f * P. Camellia computes this explicitly in computeCoarseStiffnessMatrix().

**DPG trace prolongation under h-refinement**: A coarse trace DOF on a subdivided face must be restricted to each fine sub-face. Camellia uses BasisReconciliation sub-basis weights + _fluxDuplicationMap; hp3d uses the dedicated schur_compl_extension module. For standard DPG h-refined problems, results should be equivalent; the hp3d approach may be cleaner for hp-refinement.

### 3.4 Summary Comparison Table

| Aspect | Camellia GMGOperator | hp3d MGCG |
|--------|----------------------|-----------|
| Transfer stages | Single (BasisReconciliation) | Two (coarse2macro + schur_compl) |
| Trace/field separation | Implicit (flux duplication) | Explicit (separate modules) |
| Global matrix | Yes (Tpetra CrsMatrix _P) | No (element-local vectors) |
| Multi-level | Recursive GMGOperator chain | External Igrid loop |
| Coarsening | Galerkin (P^T A_f P) | Likely Galerkin (macro level) |
| Strategy interface | None (monolithic) | Partially modular (F90 subroutines) |
| Static condensation | Integrated flag | Schur complement separate module |
| p-refinement support | Via BasisReconciliation | Via NODES DOF arrays |

---

## 4. Proposed Implementation Directions

### 4.1 MultigridTransferBuilder Strategy Interface

Extract transfer construction from GMGOperator into a strategy interface:

    class MultigridTransferBuilder {
    public:
      virtual ~MultigridTransferBuilder() = default;
      virtual TMatrixPtr<double> buildProlongationOperator(
          MeshPtr coarseMesh, MeshPtr fineMesh,
          RCP<DofInterpreter> coarseDofInterpreter,
          RCP<DofInterpreter> fineDofInterpreter,
          bool useStaticCondensation,
          RCP<BasisReconciliation> br) = 0;
    };

    class ExistingTransferBuilder : public MultigridTransferBuilder { ... };
    class DecomposedTransferBuilder : public MultigridTransferBuilder { ... };

GMGOperator would delegate to the configured MultigridTransferBuilder strategy.

### 4.2 Decomposed Pipeline Implementation

DecomposedTransferBuilder mimics hp3d's two-stage approach:

1. **Stage 1: Basis projection** — for each fine cell, compute coarse basis expansion at macro-element domain using BasisReconciliation for field (volume) DOFs only.
2. **Stage 2: Trace/conformity extension** — for each fine skeleton DOF, compute coarse skeleton DOF restriction handling conformity constraints explicitly (replaces ad hoc _fluxDuplicationMap).

### 4.3 Trace/Field Variable Separation

| Variable Type | Current Camellia | Proposed Decomposed |
|---------------|-----------------|---------------------|
| Interior (field) DOFs | BasisReconciliation (combined) | Stage 1: basis projection |
| Trace DOFs (skeleton) | BasisReconciliation + flux duplication | Stage 2: explicit Schur extension |

### 4.4 Verification Suite

- **1D scalar test**: Verify P * u_coarse = u_fine exactly for polynomials of degree <= p_coarse
- **DPG trace test**: Verify prolonged trace DOFs satisfy conformity on all fine faces
- **Convergence test**: Verify V-cycle converges at expected rate with decomposed builder

Existing GMGOperatorTests.cpp and ComponentTransferOperator1DTests.cpp provide starting points.

### 4.5 Staged Rollout

1. **Phase 1**: Refactor GMGOperator to accept a MultigridTransferBuilder. Wrap existing logic in ExistingTransferBuilder. No behavior change.
2. **Phase 2**: Implement DecomposedTransferBuilder for scalar DPG with h-refinement only.
3. **Phase 3**: Extend to vector variables, fluxes, p-refinement, mixed hp-refinement.

---

## 5. Primary Open Questions

1. **Current field/trace transfer**: How does GMGOperator transfer field and trace DOFs between levels?
   - *Partial answer*: Both handled in getLocalCoefficientMap() via BasisReconciliation; trace conformity via _fluxDuplicationMap under h-refinement + static condensation.

2. **DPG-specific assumptions**: Which assumptions are specific to DPG traces or h-refinement?
   - *Assessment*: _fluxDuplicationMap is DPG/trace-specific. BasisReconciliation is general but uses h-refinement parent-child topology. Pure p-refinement between levels is not separately represented.

3. **Mathematical difference**: What differs between current Camellia and hp3d/Petrides approach?
   - *Assessment*: Both use projection-based prolongation. Key difference is hp3d separates field and trace projection as explicit stages. For purely h-refined DPG problems, results should be equivalent. The hp3d approach may be better for hp-refinement.

4. **Strategy interface feasibility**: Can hp3d-style approach be added behind a strategy interface?
   - *Answer*: Yes. The private constructProlongationOperator(coarseDofInterpreter, fineDofInterpreter, useStaticCondensation, ...) overload shows the internal wiring already partially supports this factoring.

5. **Data requirements for transfer builder**: What data is needed?
   - *From source*: Coarse and fine Mesh objects, DofInterpreter for both levels, shared BasisReconciliation, useStaticCondensation flag, RefinementBranch per fine cell (from mesh topology). No extra geometry data needed.

---

## 6. Implementation Dependencies

| Required Component | Camellia Provider | Status |
|--------------------|-------------------|--------|
| Fine/coarse cell parent-child relationships | MeshTopology::getParentCellEntry() | Available |
| Per-cell DOF orderings | GlobalDofAssignment::getDofOrdering() | Available |
| Sub-basis projection weights | BasisReconciliation | Available (shared) |
| Global DOF numbering | GlobalDofAssignment::globalDofIndicesForCell() | Available |
| Trace conformity constraints | GDAMinimumRule constraint data | Available |
| Static condensation DofInterpreter | CondensedDofInterpreter | Available |

No new external components are needed. The work is primarily in factoring the existing logic in GMGOperator into separate, testable modules.

---

## 7. Key Source File References

### Camellia

| File | Location | Role |
|------|----------|------|
| GMGOperator.h | src/include/GMGOperator.h (19 KB) | Class definition: enums, members, constructors, public API |
| GMGOperator.cpp | (28 KB; not shelved) | Prolongation construction, cycle application, smoother setup |
| BasisReconciliation.h | src/include/BasisReconciliation.h (14 KB) | Sub-basis projection between coarse/fine elements |
| LocalDofMapper.h | src/include/LocalDofMapper.h (7 KB) | Maps local element DOFs; used as per-cell coefficient map |
| GDAMinimumRule.h/.cpp | src/include/, src/Mesh/ | DOF conformity and minimum rule for hp-meshes |
| GMGOperatorTests.cpp | (not shelved) | Tests for GMG operator correctness |
| ComponentTransferOperator1DTests.cpp | (not shelved) | 1D transfer operator tests |

### hp3d

| File | Location | Role |
|------|----------|------|
| prolongation.F90 | modules/ (43 KB) | Orchestrating module for full pipeline |
| coarse2macro.F90 | prolongation/ (13 KB) | Stage 1 prolongation (coarse to macro) |
| macro2coarse.F90 | prolongation/ (13 KB) | Stage 1 restriction (macro to coarse) |
| schur_compl_extension.F90 | prolongation/ (1.5 KB) | Stage 2a: trace extension |
| schur_compl_restriction.F90 | prolongation/ (1.4 KB) | Stage 2b: trace restriction |
| prolong_solution.F90 | solve/ | Applies full pipeline to a solution vector |
