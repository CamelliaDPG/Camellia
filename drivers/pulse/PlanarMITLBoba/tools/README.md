# PlanarMITLBoba analysis tools

Python scripts used for the Part 2 study of block AMEn on the mixed DPG system (report:
`overleaf/Planar-MITL-DPG-BoBa/Part-2/main.tex`; raw notes: `agent-notes-uur/boba-mitl-amen-results.md`).
They need numpy and scipy (a venv is fine: `python3 -m venv venv && venv/bin/pip install numpy scipy`).
Dense linear algebra on the dumped site systems is heavy; run single-threaded
(`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`) if other jobs share the machine.

## Producing the data

The tensor driver and BoBa have debug hooks, all off by default:

| environment variable | effect |
|---|---|
| `BOBA_AMEN_LOG_SITE_SIZES=1` | BoBa logs each coupled site's size, per-block `rank x mode x rank`, and sparse-LU fill |
| `BOBA_AMEN_DUMP_DIR=<dir>` | BoBa writes scaled dense site systems: `site<i>_sweep<s>_n<N>_{A,b,phiL,Acore,phiR}.bin` + `_meta.txt` |
| `BOBA_AMEN_DUMP_MIN_SIZE=<N>` | only dump sites with at least N unknowns (default 0) |
| `BOBA_AMEN_DUMP_MAX=<k>` | at most k dumps per run (default 4) |
| `PLANAR_MITL_DUMP_GLOBAL=<prefix>` | the driver writes the decompressed global blocks `<prefix>_<i>_<j>.bin` + `<prefix>_blocks.txt` |
| `PLANAR_MITL_CHECK_SPLIT=1` | with `--splitX`/`--splitAxes`, check that the mode split is exact |

Binary files are raw little-endian doubles in column-major (Fortran) order; sizes are in the text files.  Dumps are
large (a 16k-unknown site is 2.2 GB): use `BOBA_AMEN_DUMP_MIN_SIZE` and `BOBA_AMEN_DUMP_MAX`.

Example (velocity-resolved case at the production slab width, alpha_K, inf-sup enrichment, sparse-LU site solves):

```sh
BOBA_AMEN_DUMP_DIR=dumps BOBA_AMEN_DUMP_MIN_SIZE=5000 BOBA_AMEN_DUMP_MAX=2 \
PlanarMITLBobaDriver --comparison --p=3 --tWidth=0.1 --alpha=1e2 --alphaMeshScale=1.6 --nx=16 --nux=4 \
  --tolerance=1e-3 --svdTolerance=1e-5 --coefficientTolerance=1e-6 --maxSweeps=5 --verbose \
  --infSupEnrichment --localSolver=5 --outputPrefix=run
```

## Scripts

- `summarize_runs.py t-<tag>:v-<tag> ...` -- one line per tensor/conventional pair: e_h, rho, mixed residual, sweeps,
  times, final TT ranks.  Uses Part 1's `compare_drivers.py` (from `overleaf/Planar-MITL-DPG-BoBa/experiments`; set
  `MITL_EXPERIMENTS` if elsewhere).
- `site_preconditioners.py <dump dir>` -- for each dumped site: checks the Kronecker reconstruction of the Riesz block,
  then GMRES iteration counts with block-LU / block-diagonal preconditioners built from the exact Riesz block or its
  dominant Kronecker term.
- `site_inner_outer.py <dump base>` (base = path without `_A.bin`) -- Riesz-block spectrum; inner CG counts (none,
  Jacobi, dominant Kronecker term); outer GMRES with inexact inner PCG and exact or approximate Schur complement.
- `site_direct_cost.py <dump base>` -- wall time of dense LU of the whole site vs Cholesky of the Riesz block plus an
  exact Schur complement.
- `basis_conditioning.py <dir>` -- condition numbers of G, the mixed matrix, the DPG matrix K = B^T G^{-1} B, and the
  Jacobi-scaled K, for the eight basis-family combinations dumped as `<dir>/{N,H}{N,H}{N,H}_*` (test, field, trace;
  N = nodal, H = hierarchical).  Produce them with `PLANAR_MITL_DUMP_GLOBAL=<dir>/<tag>` and
  `--testBasis/--fieldBasis/--traceBasis`, e.g. on `--nx=4 --nux=2 --maxSweeps=1`.
- `element_gram_inverse_rank.py` -- single-element study of whether G^{-1} = M^{-1}/alpha + (low rank): (K, M)
  spectrum, rank of the correction, and Kronecker ranks of G^{-1} across (t,x)|(u_x,u_z).  Imports
  `element_stability.py` (drivers/papers/RelativisticVlasovSpaceTime, branch 316); set `ELEMENT_STABILITY_DIR`
  if it is not next to this tree.  Slow (minutes per case).

## Main findings (details in the report)

- Riesz block of the velocity sites: SPD; PCG with the dominant Kronecker term converges in ~80-95 iterations.  The
  trial-block Schur complement is the expensive part; exact local solves cost about the same as dense LU.
- Nodal test / hierarchical field and trace bases give the best-conditioned G and DPG matrix; block AMEn is insensitive.
- G^{-1} is not M^{-1}/alpha plus low rank at the accurate alpha (spectral gap pi^2 |beta|^2 / h^2).
