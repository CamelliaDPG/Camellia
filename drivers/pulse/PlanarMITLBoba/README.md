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

Both executables now accept `--comparison` for a matched continuous
manufactured problem. This mode uses a product of quadratic bubbles,
`s=beta.grad(f_exact)`, prescribed fields, and zero exterior normal flux.
The vanilla executable uses its actual relativistic formulation, hierarchical
bases, standard local DPG assembly and sparse direct solver. The tensor
executable uses nodal bases spanning the same spaces and block AMEn.
`--comparison` automatically solves and uses its own defaults: `p=2`, `dp=1`,
quadrature degree 20, error sampling degree 10, and the same fields/domain in
both drivers. It is separate from the default algebraic fixture.

```sh
PlanarMITLRelativisticSpaceTime --comparison --p=3 --outputPrefix=vanilla
PlanarMITLBobaDriver --comparison --p=3 --outputPrefix=tensor
python3 mitl_theory/experiments/compare_drivers.py \
  --vanilla /path/to/PlanarMITLRelativisticSpaceTime \
  --tensor /path/to/PlanarMITLBobaDriver \
  --output /path/to/results --cases p1 p2 p3 x2_u2_p2 --repeats 3
```

The runner accepts shell-quoted commands (including environment wrappers),
checks matching metadata, coordinates and weights, and calculates physical
relative L2 truncation errors from the exported samples. It retains logs, samples, JSON
metrics, and a CSV table. Numerical timings exclude sampling, validation,
output and process startup. Storage counts distinguish returned field cores,
mixed operator cores, and conventional sparse matrix values; they do not
measure peak memory. A separate field TT-SVD at physical L2 tolerance `1e-8`
is a compressibility diagnostic, not the AMEn driver's storage. A failed
tensor residual target retains diagnostics and returns status 2; the runner
also fails if any relative L2 truncation error exceeds `1e-7`.

Current measured results are in `mitl_theory/driver_comparison.tex` and
`mitl_theory/experiments/driver_comparison/`. The first four cases agree to
`1.4e-9` or better, but the four-cell AMEn solve is slower than conventional
DPG, and returned field cores use more scalars than full field coefficients.

The fixed-order refinement experiment is driven by:

```sh
python3 mitl_theory/experiments/refine_drivers.py \
  --vanilla /path/to/PlanarMITLRelativisticSpaceTime \
  --tensor /path/to/PlanarMITLBobaDriver \
  --output /path/to/refinement --p 3 --n 1 2 4 8 16 \
  --directions x --timeout 600
python3 mitl_theory/experiments/plot_refinement.py /path/to/refinement
```

`--directions all` refines all four coordinates together; `xuxuz` holds
time fixed. `--resume` reuses completed or timed-out attempts with identical
commands. Every run has a hard process-group timeout of at most 600 seconds.
Timeouts and failed residual tests remain explicit in the data and figures.
The three panels show numerical runtime, returned-field/operator storage
ratios with `CR=1`, and the relative L2 truncation error. All figures included in
`mitl_theory` are now generated as native TikZ/PGFPlots `.tex` sources.

For the matched `p=3`, `(nx,nt,nux,nuz)=(n,1,1,1)` study, use the tested
`lu10` tuning preset. It sets AMEn tolerance `1e-3`, tensor SVD tolerance
`1e-5`, inverse-Lorentz SVD tolerance `1e-6`, and a ten-sweep cap with the
new BoBa partial-pivot LU local solver (`--localSolver=2`). Rank cap 64,
enrichment rank 4, and seed 7 remain fixed. The strict algebraic-fixture
defaults are retained.

```sh
python3 mitl_theory/experiments/tune_tolerances.py \
  --tensor /path/to/PlanarMITLBobaDriver \
  --executable /path/to/PlanarMITLBobaDriver \
  --reference mitl_theory/experiments/refinement_p3_x \
  --output /path/to/tuned_results --profiles lu10 --n 4 8 16 \
  --timeout 500 --require-accuracy
```

The extended full study drops `n=2` from plots, tables, and runtime fits.
It retains the matching `n=4,8,16` baseline and runs `n=32,48,64,...`
in steps of 16. From the workspace root, reproduce it with:

```sh
python3 Camellia/mitl_theory/experiments/extend_refinement.py \
  --vanilla 'bash build-Camellia/run-planar-mitl.sh' \
  --tensor 'bash build-Camellia-boba/run.sh' \
  --vanilla-executable build-Camellia/drivers/papers/RelativisticVlasovSpaceTime/PlanarMITLRelativisticSpaceTime \
  --tensor-executable build-Camellia-boba/drivers/pulse/PlanarMITLBoba/PlanarMITLBobaDriver \
  --baseline Camellia/mitl_theory/experiments/tolerance_new_boba/results.json \
  --reference Camellia/mitl_theory/experiments/refinement_p3_x \
  --output /path/to/extended_results --timeout 500
python3 Camellia/mitl_theory/experiments/plot_tolerance_study.py \
  /path/to/extended_results/results.json --profile lu10 \
  --output /path/to/extended_results/tuned_refinement.tex
```

Each new driver run has a 500-second process-group timeout; both drivers
are attempted sequentially at each size before stopping at the first size
with a timeout. A completed iterate that fails an accuracy check remains in
the data and does not end refinement. `stop.json` records the stopping
reason. `--resume` requires identical commands, timeout, solver settings,
and binary hashes. Historical raw measurements, including `n=2`, are retained.
The paper uses `mitl_theory/experiments/refinement_p3_extended/`.

The tensor command may include an environment wrapper; the executable
arguments record the actual binary hashes. The full-study figure has log–log runtime with
`a n^b` fits for both methods, compression including `10^0`, and relative
TT truncation error versus total element count `n`. Fits minimize squared
log-runtime residuals; full-precision parameters and input timings are saved
in `runtime_fits.json`. The paper caption defines all plotted notation.
The reference directory must contain the matching conventional samples.
`--resume` checks commands, BoBa revision, and binary
hash before reusing runs. The runner distinguishes driver residual status
from the physical criterion `||f_tensor-f_conventional|| < ||f_conventional-f_exact||`.
The reported relative truncation error is
`e_trunc = ||f_tensor-f_conventional|| / ||f_conventional||`; it is distinct
from discretization error. The retained `n=4,8,16` cases have relative
truncation errors `9.60e-4`, `6.97e-4`, and `1.91e-3`; their separate
truncation/discretization budget ratios are 0.752, 0.342, and 0.818.
Both errors use the conventional field norm when forming this ratio.
At `n=32`, TT takes 80.59 seconds versus 85.14 conventionally, with
relative truncation error `1.12e-3`, budget ratio 0.582, and field
compression 1.63. At `n=48`, TT takes 229.42 seconds versus 130.19,
and fails the physical budget: relative truncation error `8.45e-3`
versus relative discretization error `1.77e-3` (ratio 4.79).
At `n=64`, TT takes 462.89 seconds versus 169.31 conventionally,
with relative truncation error `3.26e-2` versus discretization error
`1.70e-3` (ratio 19.24), and field compression 1.37.
The sequence stops at `n=80`: conventional completes in 212.98 seconds,
while TT reaches the 500-second wall timeout without a returned solution.
The figure marks this timeout and omits its numerical runtime, error, and
compression from measurements and fits. A timeout at one mesh size does
not prove that all larger sizes would time out.
Completed inaccurate iterates remain in the runtime fit and are marked
with red crosses in the relative-error panel. All these TT runs still
miss the mixed-residual target and return driver status 2;
`--require-accuracy` in the tuning runner checks the separate physical
criterion. The small-mesh tuning does not establish accuracy on finer
meshes, flux accuracy, or reliability in a coupled discharge.

`--localSolver` selects COD (0), pivoted QR (1), LU (2), assembled Eigen
GMRES (3), or matrix-free BoBa GMRES (4). The iterative paths also use
`--localTolerance` and `--maxLocalIterations`; no matrix entries are dropped
in the assembled GMRES path. The five-sweep GMRES screening settings did
not meet the physical error budget. Full tolerance and solver comparisons
are in `mitl_theory/tolerance_study.tex`, with saved measurements in
`mitl_theory/experiments/tolerance_new_boba/`.

The expanded matrix-free sweep study uses presets `matrixfree10`,
`matrixfree20`, `matrixfree40`, and `matrixfree100`, with the other
settings held at AMEn `1e-3`, SVD `1e-5`, coefficient SVD `1e-6`, and
local GMRES `1e-5`/300 iterations. Use these profiles with `--n 2 4`
in the tuning command above and a separate output directory. All budgets
up to 100 were tested with a 600-second timeout. At 100 sweeps, relative
truncation errors are 23.5% and 367% for two and four elements; increasing
the outer budget alone did not meet the physical error criterion.
Measurements are in `mitl_theory/experiments/amen_sweep_budget/`, and the
paper's native TikZ sweep figure is generated by `plot_amen_sweeps.py`.

The next steps are improved coupled AMEn convergence and field compression,
physical inflow/outflow boundary lifting, followed by TT moment
contraction, Maxwell fixed-point coupling, and temporal flux transfer. MPI
distribution, local AMR, and positivity enforcement are not implemented here.
The coefficient SVD controls sampled quadrature error; off-grid coefficient and
conservation studies remain necessary before physical use.

The target is available through `BUILD_PULSE_DRIVERS`, or individually through
`BUILD_PLANAR_MITL_BOBA_DRIVER` with the `BOBA` CMake package available.
