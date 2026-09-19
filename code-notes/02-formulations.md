# 02 — Formulations (src/Formulations/)

## Overview

34 source files implementing DPG and related variational formulations for a wide range of PDEs.

---

## Formulation Categories

### 1. Incompressible Flow
- **StokesVGPFormulation.cpp** (49KB): VGP formulation for Stokes. Steady/transient, 2D/3D, conforming/non-conforming traces.
- **NavierStokesVGPFormulation.cpp** (51KB): Extends Stokes to Navier-Stokes via Newton linearization.
- **PressurelessStokesFormulation.cpp** (11KB): Stokes without pressure variable.
- **SpaceTimeIncompressibleFormulation.cpp** (22KB): Space-time transient incompressible flow.

### 2. Compressible Flow
- **CompressibleNavierStokesFormulation.cpp** (50KB): Primitive-variable compressible NS.
- **CompressibleNavierStokesConservationForm.cpp** (58KB): Conservation-form variant.
- **CompressibleNavierStokesProblem.cpp/.hpp** (5KB): Problem parameter helper.
- **SpaceTimeCompressibleFormulation.cpp** (51KB): Space-time compressible flow.

### 3. Viscoelastic Flow
- **OldroydBFormulationUW.cpp** (75KB): Ultraweak Oldroyd-B viscoelastic model.
- **OldroydBFormulationUWReduced.cpp** (56KB): Reduced variant. Likely significant duplication with UW.

### 4. Elasticity
- **LinearElasticityFormulation.cpp** (13KB): 2D/3D linear elasticity.

### 5. Scalar PDEs
- **PoissonFormulation.cpp** (6KB): Canonical Poisson/Laplacian. 1D/2D/3D.
- **H1ProjectionFormulation.cpp** (5KB): L2 projection into H1.
- **ConvectionFormulation.cpp** (2KB): Pure transport.
- **ConvectionDiffusionFormulation.cpp** (5KB): Convection-diffusion.
- **ConvectionDiffusionReactionFormulation.cpp** (12KB): With reaction term.
- **InviscidBurgersFormulation.cpp** (12KB): Nonlinear scalar conservation law.

### 6. Heat/Diffusion
- **HeatFormulation.cpp** (5KB): Transient heat equation.
- **SpaceTimeHeatFormulation.cpp** (8KB): Space-time heat.
- **SpaceTimeHeatDivFormulation.cpp** (8KB): Div-conforming variant.
- **SpaceTimeConvectionDiffusionFormulation.cpp** (13KB): Space-time conv-diff.

### 7. Acoustics
- **AcousticsFormulation.cpp** (8KB): Acoustic wave propagation.
- **AcousticsTomographyFormulation.cpp** (7KB): Acoustics tomography (inverse problem).

### 8. Kinetic Theory / Vlasov-Poisson
- **VlasovSpaceTimeFormulation.cpp** (136KB): BY FAR THE LARGEST. Phase-space + time.
- **VlasovBackwardEulerFormulation.cpp** (75KB): Implicit time integration.
- **MimeticVlasovSpaceTimeFormulation.cpp** (15KB): Mimetic discretization variant.
- **RelativisticVlasovSpaceTimeFormulation.cpp** (14KB): Relativistic extension.
- **ConfigurationSpaceFormulation.cpp** (9KB): Generic config-space framework.
- **ConfigurationSpaceFormulationMaxwell.cpp** (19KB): Maxwell in config space.
- **ConfigurationSpaceFormulationMaxwell1D2V.cpp** (17KB): 1D2V Maxwell variant.
- **ConfigurationSpaceFormulationPoisson.cpp** (14KB): Poisson in config space.
- **ConfigurationSpaceFormulationTrivial.cpp** (5KB): Test/trivial variant.

### 9. Ideal MHD
- **IdealMHDFormulation.cpp** (65KB): Coupled fluid + electromagnetic MHD system.

---

## Common API Pattern

All formulations follow:
1. Constructor: spaceDim, physical params, discretization options
2. `bf()` returning BFPtr (bilinear form)
3. Variable accessors: `u()`, `sigma()`, `p()`, etc. returning VarPtr
4. BC helpers: `addWallCondition()`, `addInflowCondition()`, etc.
5. Graph norm: `graphNorm()` or similar
6. Solution/mesh factory methods

---

## Code Smells and Issues

1. **Oversized files**: VlasovSpaceTimeFormulation (136KB), VlasovBackwardEuler (75KB), OldroydBUW (75KB), IdealMHD (65KB), CompressibleNSConservation (58KB)
2. **Code duplication**: OldroydB UW vs UWReduced; CompressibleNS primitive vs conservation; multiple ConfigurationSpace variants; Vlasov formulations overlap
3. **Inconsistent naming**: VGP suffix used inconsistently; SpaceTime prefix inconsistent; UW suffix not universal
4. **No common base class**: Each formulation independently implements the pattern above
5. **Missing unit tests**: No dedicated tests for compressible NS, Vlasov, MHD, acoustics, or elasticity formulations
6. **Potential dead code**: ConfigurationSpaceFormulationTrivial may be test-only; PressurelessStokes usage unclear
