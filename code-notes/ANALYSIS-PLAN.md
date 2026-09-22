# Camellia Analysis Plan

## Purpose

This document defines a staged analysis effort for Camellia's current architecture and for planning improvements in four priority areas:

1. Global degree-of-freedom numbering and adaptive mesh constraints.
2. Curvilinear geometry support.
3. Multigrid treatment of DPG traces and hp-style transfer options.
4. MeshTopology simplification and modernization.

It also defines how to update the existing notes in `code-notes/`, which were generated before major infrastructure changes such as the Epetra-to-Tpetra transition.

The intended output of the analysis is not a single monolithic rewrite plan. The intended output is a set of discrete, reviewable implementation plans, each with scope, dependencies, risks, validation strategy, and interaction with the other plans.

## Guiding Questions

For each subsystem, answer the following:

- What responsibilities does the subsystem currently own?
- Which responsibilities are essential, accidental, duplicated, or stale?
- Which code paths are central and which are legacy compatibility paths?
- What invariants does the subsystem assume but not document?
- Which interfaces are stable enough to build on, and which should be isolated behind adapters before further work?
- What existing tests cover the behavior, and where are tests missing?
- What changes are logically prior to other proposed changes?

## Comparative Source Study: MFEM and hp3d

Because both MFEM and hp3d are available in the source manifest, the analysis should include direct source-level comparisons rather than only conceptual comparisons. The goal is not to clone either project, but to identify mature implementation patterns that could inform Camellia's modernization.

### MFEM Comparison Targets

Use MFEM as the primary comparison point for general finite element infrastructure, especially separation of mesh topology, finite element spaces, nonconforming constraints, and geometry evaluation. Initial files to inspect include:

- `/Users/nvrober/reference-codes/mfem/fem/fespace.hpp`
- `/Users/nvrober/reference-codes/mfem/mesh/mesh.hpp`
- `/Users/nvrober/reference-codes/mfem/mesh/ncmesh.hpp`
- `/Users/nvrober/reference-codes/mfem/mesh/nurbs.hpp`

Specific MFEM concepts to compare against Camellia:

1. **DOF numbering and constraints:** `FiniteElementSpace` is documented as responsible for the FEM view of the mesh and for managing degrees of freedom. Compare this separation with Camellia's coupling among `Mesh`, `GlobalDofAssignment`, `GDAMinimumRule`, `BasisReconciliation`, and `LocalDofMapper`. Pay particular attention to MFEM's conforming prolongation/restriction APIs, true DOF handling, and nonconforming interpolation support such as `BuildConformingInterpolation()`, `GetConformingProlongation()`, `GetProlongationMatrix()`, and `GetRestrictionMatrix()`.
2. **Nonconforming topology:** compare Camellia's `MeshTopology` with MFEM's split between `Mesh` and optional `NCMesh` support. MFEM's `Mesh` explicitly carries optional nonconforming mesh support and initialization paths from `NCMesh`; this should be used as a reference point for whether Camellia should separate active mesh entities from refinement forest data.
3. **Curvilinear and high-order geometry:** compare Camellia's `CamelliaCellTools`, parametric curve/surface support, and mesh transformation functions with MFEM's optional `NURBSExtension`, mesh nodes, element transformations, `GeometricFactors`, and `FaceGeometricFactors`. The analysis should ask which geometric data are persistent mesh state, which are finite-element coordinate fields, and which are cached evaluation products.
4. **Topology/geometry separation:** identify how MFEM keeps mesh connectivity, nonconforming refinement, NURBS/high-order geometry, and geometric factor caches as separable concerns. Use this to evaluate whether Camellia's `MeshTopology` should become a narrower combinatorial topology object with geometry moved elsewhere.

### hp3d Comparison Targets

Use hp3d primarily as a comparison point for hp-adaptive multigrid transfer, trace/field relationships, and the Petrides-style approach mentioned in the modernization goal. Initial files to inspect include:

- `/Users/nvrober/reference-codes/hp3d/trunk/src/solver/MGCG/modules/prolongation.F90`
- `/Users/nvrober/reference-codes/hp3d/trunk/src/solver/MGCG/prolongation/coarse2macro.F90`
- `/Users/nvrober/reference-codes/hp3d/trunk/src/solver/MGCG/prolongation/macro2coarse.F90`
- `/Users/nvrober/reference-codes/hp3d/trunk/src/solver/MGCG/prolongation/schur_compl_extension.F90`
- `/Users/nvrober/reference-codes/hp3d/trunk/src/solver/MGCG/prolongation/schur_compl_restriction.F90`
- `/Users/nvrober/reference-codes/hp3d/trunk/src/solver/MGCG/solve/prolong_solution.F90`

Specific hp3d concepts to compare against Camellia:

1. **Macro/coarse transfer decomposition:** study how hp3d decomposes transfer into modules such as `coarse2macro` and `macro2coarse`, and compare that with Camellia's current `GMGOperator` transfer construction.
2. **Schur-complement trace handling:** inspect hp3d's Schur-complement extension and restriction modules to understand how trace/interface unknowns are transferred relative to element-interior fields. Compare this explicitly with Camellia's DPG trace treatment under h-refinement.
3. **Strategy extraction:** determine whether hp3d's organization suggests a clean `MultigridTransferBuilder` or equivalent strategy interface for Camellia, allowing the current transfer and a Petrides-style transfer to coexist.
4. **Constraint interaction:** identify what constraint, topology, and finite-element-space metadata hp3d's transfer needs, then compare that with what Camellia's `GDAMinimumRule`, `BasisReconciliation`, and `MeshTopology` can currently expose.

### Comparative Deliverables

Each focused analysis document should include a short comparison table with these columns:

- Camellia current mechanism
- MFEM equivalent or contrast, when applicable
- hp3d equivalent or contrast, when applicable
- Design lesson for Camellia
- Migration risk

The comparison should be evidence-based: cite file names, classes, routines, and relevant APIs. Where the external code has no direct analogue, say so explicitly rather than forcing a comparison.

## Phase 0: Repository and Note Refresh

### Goals

- Establish an accurate inventory of current architecture after the Tpetra and related conversions.
- Identify stale claims in existing `code-notes` documents.
- Produce an updated note map that distinguishes current facts from historical observations.

### Files and Areas to Review

- `code-notes/01-basis-allocations-celltools.md`
- `code-notes/02-formulations.md`
- `code-notes/03-functions.md`
- `code-notes/04-mesh-solution-problem.md`
- `code-notes/05-drivers.md`
- `code-notes/06-examples-tests.md`
- `code-notes/AGENTS.md`
- `src/include/*`
- `src/Mesh/*`
- Current build, examples, and tests that exercise mesh, dof, geometry, and GMG behavior.

### Work Products

- A stale-note audit listing statements that are wrong, obsolete, or require confirmation.
- A current subsystem map showing major classes, ownership boundaries, and dependency direction.
- A glossary of core terms: mesh entity, cell topology, side, subcell, global dof, local dof, trace, constraint, refinement, reconciliation, geometry map, and prolongation/restriction.

### Validation

- Cross-check notes against current headers, implementations, and tests.
- Mark uncertain claims explicitly instead of turning them into design assumptions.

## Phase 1: Global DOF Numbering and Adaptive Constraints

### Motivation

The global degree-of-freedom path appears to involve several tightly coupled classes and concepts, including `GlobalDofAssignment`, `GDAMinimumRule`, `BasisReconciliation`, and `LocalDofMapper`. This area is central to correctness under h- and p-refinement, and it is likely to constrain later work on geometry and multigrid.

### Primary Questions

- Where is the authoritative mapping from local element dofs to global dofs established?
- How are constraints represented, minimized, cached, and applied?
- What are the exact responsibilities of `GDAMinimumRule` versus `BasisReconciliation`?
- Which algorithms assume affine geometry, regular refinement, or specific basis families?
- Where do hanging-node, side, edge, vertex, and trace constraints enter the assignment process?
- How are local-to-global maps updated after refinement or repartitioning?
- Which parts are serial assumptions, and which are safe under distributed Tpetra execution?

### Files to Inspect First

- `src/include/GlobalDofAssignment.h`
- `src/Mesh/GlobalDofAssignment.cpp`
- `src/include/GDAMinimumRule.h`
- `src/Mesh/GDAMinimumRule.cpp`
- `src/include/BasisReconciliation.h`
- `src/include/LocalDofMapper.h`
- Mesh and solution classes that consume global dof maps.

### Analysis Steps

1. Trace dof assignment from mesh construction through solution assembly.
2. Document data structures used for local dofs, global dofs, constraints, and ownership.
3. Identify implicit invariants, especially around side orientation, basis order, and refined neighbors.
4. Build a call graph for refinement-triggered updates.
5. Compare constraint handling with modern finite element library patterns, including separation of topology, finite element space, constraints, and linear algebra maps.
6. Identify minimal refactors that reduce coupling without changing behavior.

### Expected Implementation Plans

- DOF ownership and constraint representation cleanup.
- Improved tests for refinement and p-order mismatch cases.
- Optional abstraction of finite element space behavior away from mesh topology details.
- Performance plan for caching, invalidation, and avoiding redundant reconciliation work.

## Phase 2: Curvilinear Geometry Modernization

### Motivation

Camellia's curvilinear geometry support is limited and non-standard. A modern design should support high-order geometry, robust geometry mapping, efficient evaluation, and clear separation between mesh topology and geometric realization.

### Primary Questions

- How does Camellia currently represent geometric nodes, vertices, parametric coordinates, and physical coordinates?
- Which code assumes straight-sided or affine elements?
- Where does `CamelliaCellTools` encode geometry-specific behavior?
- How are Jacobians, normals, side maps, orientations, and physical measures computed?
- How would high-order or curved geometry interact with basis evaluation, quadrature, dof assignment, and refinement?

### Files to Inspect First

- `src/include/CamelliaCellTools.h`
- `src/include/MeshTopology.h`
- `src/Mesh/MeshTopology.cpp`
- Mesh construction and element geometry code reachable from `Mesh.h`.
- Tests for curvilinear meshes and cell tools.

### Analysis Steps

1. Inventory all geometry and topology data structures.
2. Separate topological entities from geometric mappings in the analysis model.
3. Identify affine assumptions in quadrature, normals, side transforms, and element/side orientation handling.
4. Compare options from MFEM-style mesh/element transformations and hp3d-style geometry treatment.
5. Define requirements for curved cells, curved boundaries, geometry order, and adaptive refinement.
6. Propose migration paths that can coexist with current meshes.

### Expected Implementation Plans

- Geometry abstraction layer for element and side transformations.
- Curved element representation with explicit geometry order and control data.
- Normal/Jacobian evaluation API cleanup.
- Incremental compatibility path for existing affine meshes.
- Tests comparing affine behavior, curved manufactured geometries, and refinement cases.

## Phase 3: GMGOperator and DPG Trace Transfer Options

### Motivation

Camellia's `GMGOperator` appears to have effective existing treatment of DPG traces under h-refinement. hp3d, as described by Petrides et al., implements a more sophisticated approach that may be valuable as an optional mode rather than a replacement for the current path.

### Primary Questions

- How does `GMGOperator` currently transfer field and trace unknowns between levels?
- Which assumptions are specific to DPG traces, h-refinement, or Camellia's present basis reconciliation logic?
- What is the mathematical difference between the current transfer and the hp3d/Petrides approach?
- Can the hp3d-style approach be added behind a strategy interface?
- What data from DOF assignment, constraints, topology, geometry, and basis metadata must a transfer builder consume?

### Files to Inspect First

- `src/include/GMGOperator.h`
- the corresponding `GMGOperator.cpp` implementation
- `src/include/Solution.h`
- GDA and mesh-refinement code used during hierarchy construction
- GMG tests and examples

### Analysis Steps

1. Trace hierarchy construction and identify all objects shared between levels.
2. Document the existing prolongation and restriction construction.
3. Separate transfer strategy from operator application, smoothing, and coarse solve concerns.
4. Determine how constrained dofs are handled in the hierarchy.
5. Write a concept mapping from Petrides/hp3d terminology to Camellia classes.
6. Identify an initial supported subset for an optional hp3d-style transfer path.

### Expected Implementation Plans

- Diagnostics for transfer matrices and convergence behavior on small meshes.
- `MultigridTransferBuilder` strategy interface.
- Existing-transfer adapter to preserve current behavior.
- hp3d/Petrides prototype option for a limited scalar/trace case.
- Later extension to vector variables, fluxes, p-refinement, and distributed execution.

## Phase 4: MeshTopology Simplification

### Motivation

`MeshTopology` appears to be complex and broad, with redundant storage of mesh entities and relationships. Because topology is consumed by DOF assignment, constraints, geometry, and GMG, any simplification must be careful and staged.

### Primary Questions

- Which entities are primary and which are derived: vertices, edges, faces, cells, sides, subcells, ancestors, descendants, and neighbor relations?
- Which relationships are stored redundantly in both directions?
- Which data are truly topological, and which are geometric, discretization-specific, or cache data?
- What mutation paths must maintain invariants?
- Which APIs expose internal representation and therefore make refactoring difficult?

### Files to Inspect First

- `src/include/MeshTopology.h`
- `src/Mesh/MeshTopology.cpp`
- `src/include/Mesh.h`
- mesh topology, mesh refinement, component topology, and extruded topology tests

### Analysis Steps

1. Catalog every stored container and classify each as primary data, derived cache, compatibility facade, or redundant state.
2. Trace mutation paths and cache invalidation requirements.
3. Identify consumers of topology queries in GDA, geometry, basis cache, and GMG.
4. Compare with MFEM's split among `Mesh`, `NCMesh`, element transformations, and finite element spaces.
5. Propose a narrow read-only topology query interface to shield consumers from representation changes.
6. Design a canonical storage model and migration path only after the adapter and tests exist.

### Expected Implementation Plans

- MeshTopology invariant tests and memory/call-site audit.
- Read-only topology query adapter.
- Separation of topology from geometry and discretization metadata.
- Gradual conversion from redundant eager storage to canonical storage plus derived caches.

## Phase 5: Updating Existing `code-notes`

### Motivation

The existing notes are useful as a map of prior AI observations, but they are stale and should not be treated as authoritative. They need review against current source, especially after linear algebra and infrastructure conversions.

### Tasks

1. Add an explicit review status to each note: current, partially stale, stale, or superseded.
2. Replace brittle file-size statements with source-date or generated-inventory references.
3. Remove or annotate Epetra-era assumptions and update them for the current Tpetra-based state.
4. Split facts from recommendations.
5. Link each note to the focused analysis documents produced by Phases 1-4.
6. Create `code-notes/NOTES-INDEX.md` as an entry point for future readers.

### Expected Implementation Plans

- A notes-index creation plan.
- Per-note stale-claim patches.
- A convention for future notes: review date, source revision, confidence level, and evidence links.

## Cross-Plan Dependencies and Ordering

Recommended order:

1. Refresh the notes and produce an accurate subsystem map.
2. Analyze and document `MeshTopology` invariants, because topology is consumed by geometry, DOF assignment, and GMG.
3. Analyze DOF numbering and constraints, because GMG transfer and adaptive behavior depend on them.
4. Define geometry abstraction requirements, because curved geometry should not be bolted onto redundant topology state.
5. Design GMG transfer strategy options, using clarified topology, geometry, and dof-space interfaces.

### Important Interactions

- MeshTopology simplification can break DOF assignment if adjacency and entity identity semantics are changed without a compatibility layer.
- Curvilinear geometry should not be implemented by adding more topology-specific special cases; it should introduce explicit geometry mapping concepts.
- hp3d-style GMG transfer may need more information from finite element spaces and constraints than the current `GMGOperator` interface exposes.
- DOF constraint refactoring may simplify GMG transfer, but large simultaneous changes would be risky.
- Updating stale notes first reduces the chance of designing around obsolete Epetra-era assumptions.

## Deliverables

The analysis should produce the following documents or sections in updated notes:

1. Current-state architecture summary.
2. Stale-note audit and corrected note set.
3. DOF and adaptive constraint implementation plan.
4. Curvilinear geometry implementation plan.
5. GMG hp3d/Petrides option implementation plan.
6. MeshTopology simplification implementation plan.
7. Cross-plan dependency and risk matrix.
8. Test and benchmark plan.

## Suggested Analysis Schedule

### Pass 1: Fact Gathering

Read headers, implementations, tests, and examples. Record only observed facts and open questions.

### Pass 2: Execution Tracing

Trace representative workflows:

- Mesh construction.
- Uniform and adaptive refinement.
- Global dof assignment.
- Assembly and solution setup.
- GMG hierarchy construction.
- Geometry evaluation for affine and curved cases.

### Pass 3: Design Alternatives

For each priority area, write two or three viable implementation strategies with tradeoffs.

### Pass 4: Plan Selection

Select recommended plans based on risk, dependency order, testability, and compatibility with Camellia's DPG flexibility.

### Pass 5: Implementation-Ready Breakdown

Break each selected plan into small steps suitable for separate reviewable changes.

## Definition of Done

This analysis effort is complete when:

- Existing code notes have been audited and updated or clearly marked stale.
- The current architecture is described accurately enough for a new contributor to follow major control and data flows.
- Each priority area has a scoped implementation plan with dependencies, risks, tests, and migration steps.
- Interactions among plans are explicitly documented.
- The team can choose an implementation sequence without first needing another broad exploratory pass.
