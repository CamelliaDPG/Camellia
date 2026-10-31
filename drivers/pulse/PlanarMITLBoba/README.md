# Tensorized planar MITL driver

`PlanarMITLBobaDriver` is the prescribed-field assembly and algebraic solver
prototype for the plan in `mitl_theory`. The conventional
`PlanarMITLRelativisticSpaceTime` driver remains the coupled physical reference.

The prototype assembles the mixed DPG system directly in BoBa tensor trains:

- Four modes: `(x cell/basis, t cell/basis, ux cell/basis, uz cell/basis)`.
- Proper velocity `u=gamma*v`, with a quadrature-weighted SVD of
  `1/gamma = 1/sqrt(1+(ux*ux+uz*uz)/(c*c))`.
- Prescribed constant `Ex` and `By`, with the electron sign `q/m=-1` by default.
- Full graph Gram matrix `G=L^T W L + alpha*M`, including all cross terms.
- Volume block `B_f=-L^T W F` and separate shared interior normal fluxes.
- Reference order convention: field degree `p-1`, flux degree `p`, test degree `p+dp`.
- Uniform or graded product meshes; one-cell directions omit empty flux blocks.

`--validate` (the default) compares the TT volume and Gram matrices with
independent full quadrature using Camellia's 1D bases and exact Lorentz values.
It checks flux indexing and interface signs, then solves the small dense mixed
system. This is not yet a comparison with the conventional driver's assembled
global matrices and boundary DOF maps. Dense validation is size-limited.

`--solve` runs block AMEn from a reproducible, nonzero rank-one guess. The RHS
is manufactured algebraically from trial coefficients with zero test residual;
the trial RHS blocks therefore vanish. This fixture checks the solver, not
physical cathode emission or continuous-solution accuracy. Reported residuals
include every block, using a denominator floor of `1e-30`; a missed global
residual target produces a nonzero exit status. With validation enabled, a
second residual is evaluated against the uncompressed quadrature operator.

The current default is 100 sweeps with enrichment rank 4 and residual target
`1e-8`. The small case with interior x and ux fluxes passes this target, but
the indefinite mixed solve is sensitive to rounding and can lose accuracy in
later sweeps. A tighter `1e-10` target with `1e-14` rounding did not converge
in the tested 100-sweep run; its residual remained around `1e-9`. Stronger
solver tolerances need further investigation. The driver reports failure when
its final independent residual misses the requested target.
The next solver checks are the coupled stopping rule, preservation of the best
global iterate, and the effect of truncating the nearly zero test-residual block.

Small examples:

```sh
PlanarMITLBobaDriver --validate --solve
PlanarMITLBobaDriver --validate --solve --nx=2 --nux=2
PlanarMITLBobaDriver --validate --Ex=0 --By=0 --nt=2 --nuz=2
```

The prototype defaults (`p=1`, `dp=1`, `betaMax=0.5`, `alpha=1`) keep algebra
checks small. Physical comparisons must explicitly match the reference
parameters, especially its current defaults `beta0=-0.995`, `betaMax=0.995`,
and `alpha=1e6`. The prototype defaults to `beta0=0`; negative values include
returning electrons in the velocity domain.

Shared code:

- `drivers/PlanarMITL/PlanarMITLCommon.hpp`: product mesh helpers and relativistic
  coordinate conventions; also used by the conventional driver.
- `drivers/pulse/common/BobaTypes.hpp`: CPU tensor aliases and zero constructors;
  the existing DPG example includes this through its compatibility header.
- `drivers/pulse/common/BlockAmenSolver.hpp`: rank-one initialization, coupled
  AMEn configuration, rounding, and residual reporting.

The next steps are physical inflow/outflow boundary lifting and a conventional
Camellia comparison with explicit basis/DOF maps, followed by TT moment
contraction, Maxwell fixed-point coupling, and temporal flux transfer. MPI
distribution, local AMR, and positivity enforcement are not implemented here.
The coefficient SVD controls sampled quadrature error; off-grid coefficient and
conservation studies remain necessary before physical use.

The target is available through `BUILD_PULSE_DRIVERS`, or individually through
`BUILD_PLANAR_MITL_BOBA_DRIVER` with the `BOBA` CMake package available.
