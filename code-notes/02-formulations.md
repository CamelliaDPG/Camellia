# 02 — Formulations (src/Formulations/)

> **STALENESS HEADER (annotated 2025)**: File-size claims cross-checked against manifest. The original note overstated several formulation file sizes by 3-4x. Corrections are marked [CORRECTION] inline. The 34-file count appears accurate per manifest. Interface signatures may have changed with the Epetra->Tpetra migration.

## Overview

34 source files implementing DPG and related variational formulations for a wide range of PDEs.

---

## Formulation Categories

### 1. Incompressible Flow
- **StokesVGPFormulation.cpp** [CORRECTION: manifest shows 12,188 bytes; note claimed 49KB]: VGP formulation for Stokes. Steady/transient, 2D/3D, conforming/non-conforming traces.
- **NavierStokesVGPFormulation.cpp** [CORRECTION: manifest shows 12,688 bytes; note claimed 51KB]: Extends Stokes to Navier-Stokes via Newton linearization.
- **PressurelessStokesFormulation.cpp** (11KB): Stokes without pressure variable. [size unverified in manifest]
- **SpaceTimeIncompressibleFormulation.cpp** [CORRECTION: manifest shows 5,457 bytes; note claimed 22KB]: Space-time transient incompressible flow.

### 2. Compressible Flow
- **CompressibleNavierStokesFormulation.cpp** (50KB): Primitive-variable compressible NS. [size unverified against current manifest]

### 3. Convection-Diffusion and Scalar PDEs
- ConvectionDiffusionFormulation.cpp, PoissonFormulation.cpp, EricksonFormulation.cpp, etc.

### 4. Space-Time Formulations
- **SpaceTimeHeatFormulation.cpp** [manifest: 1,974 bytes]
- **SpaceTimeIncompressibleFormulation.cpp** — see section 1 above

### 5. Vlasov / Kinetic
- **VlasovBackwardEulerFormulation.cpp** (manifest: 18,662 bytes)
- **VlasovSpaceTimeFormulation.cpp** (manifest: 34,045 bytes)
- **MimeticVlasovSpaceTimeFormulation.cpp** (manifest: 3,872 bytes)

### 6. Structural / Visco-elastic
- **OldroydBFormulationUW.cpp** (manifest: 18,657 bytes)
- **OldroydBFormulationUWReduced.cpp** (manifest: 13,917 bytes)
- **LinearElasticityFormulation.cpp** (manifest: 3,319 bytes)

---

## Cross-Cutting Issues

1. **File-size inflation**: The original note inflated sizes 3-4x. Do not use original size claims for architectural planning.
2. **Epetra->Tpetra**: Any formulation using Epetra_MultiVector, Epetra_CrsMatrix, etc. must be migrated. TSolution<double> is the Tpetra-native assembly class.
3. **34-file count**: Confirmed accurate per manifest directory listing.
