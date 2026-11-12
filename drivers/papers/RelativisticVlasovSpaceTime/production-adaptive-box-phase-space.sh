#!/bin/bash
# Planar MITL on an ADAPTIVE, ANISOTROPICALLY REFINED phase-space mesh: the adaptive counterpart of
# production-11-09-26-x32-phase-space.sh (the regular mesh with 32 graded x cells, 28,160 phase-space cells per slab,
# 3.40 M dofs, 926 s/slab on 16 ranks, 113.5 GB peak summed footprint).
#
# Same physics, discretization and diagnostics as that run:
#   transport-scale graph test norm, alpha_K = 1e2 + 1.6 [(v_x,max/h_x)^2 + 1/h_t^2 + (a_x/h_ux)^2 + (a_z/h_uz)^2],
#   (a_x, a_z) = (1.8, 0.65); p = 4, dp = 2; serendipity basis; SCL emission; GMG with a SuperLU_DIST coarse solve;
#   --accuracyDiagnostics, --densityDiagnostic, --phaseSpaceDiagnostic; 16 ranks.
# What differs:
#   - The phase-space mesh is a box mesh (--useBoxMesh: BoxMeshTopology + BoxGDA), which can bisect a cell in any subset
#     of (x, t, u_x, u_z).  The configuration-space (x,t) mesh is the usual one, kept at least as fine as the phase mesh.
#   - The base mesh is the production run's base mesh WITHOUT its refinements: 8 graded x cells, 1 t cell per slab,
#     11 x 10 graded velocity cells = 880 cells per slab (the production run bisected x once more and then refined
#     uniformly once: 32 x 2 x 22 x 20).  Everything finer than that is added where the solver asks for it.
#   - --useAdaptivity: each slab is solved, the cells carrying the most energy error are marked, and each marked cell is
#     bisected only in the directions in which f is least resolved on it (--boxAnisotropicAdaptivity).  To resolve the
#     inflow and initial data, cells are bisected only tangentially to the boundary carrying the data.
#
# REQUIRES the latest Camellia branch anisotropic-refinement (github.com/CamelliaDPG/Camellia), the
# Intrepid2 of trilinos/Trilinos PR 15871 (b93efd4b798), a Trilinos with SuperLU_DIST (for --coarseSolver), and a
# python3 with numpy and matplotlib.  Run from the Camellia build directory.  The branch is 316-tensor-native-mesh-for-planar-mitl-and-more
# plus the box-mesh work; it does NOT have n5-separable-test-norm's additions, so:
#   --numSpatialRefinements is not available (not needed here: x is refined adaptively);
#   CAMELLIA_RELEASE_FREED_MEMORY has no effect and the sum_footprint_gb field is absent (memory is reported as RSS).
#
# READ BEFORE THE FULL RUN.  This configuration has been run only at small scale (4 ranks, p = 3, dp = 1, a 440-cell
# base mesh, 3 slabs).  At that scale every option below works, but none of the settings are tuned for this problem:
#   1. Cost.  Adaptivity re-solves the slab after every refinement step: REF0+1 solves of slab 0, REF+1 of each later
#      slab.  The early solves are on small meshes (880 cells is 1/32 of the production mesh) and use coarse solver
#      tolerances, but the count matters.  A later slab starts from the base mesh refined to match the previous slab
#      at their interface, so it inherits the previous slab's refinement and needs few steps of its own.
#   2. Refinement only accumulates.  Box meshes do not unrefine yet, and each slab inherits the previous one's mesh at
#      the interface, so cells refined for a transient stay refined in every later slab.  Watch vlasov_elements grow
#      from slab to slab (printed below); if it approaches the production run's 28,160 the run will not be cheaper.
#   3. Multigrid iterations.  In the small runs the V-cycle-preconditioned CG took 40-60 iterations on the base mesh and
#      about 200 on meshes with cells refined in t alone or x alone (the production run's median was 199, maximum 302).
#      If iteration counts climb well beyond that, lower ANISO: at 0 every marked cell is bisected in all four
#      directions, which is kinder to the multigrid and costs more cells.
#   4. The direction indicator is a heuristic (the energy in f's highest Legendre modes in each direction), not the
#      DPG local-residual indicator.  ANISO sets how selective it is.
#   5. The marking.  In the small runs the energy error was concentrated in a handful of cells at the cathode, so a
#      Doerfler (bulk) criterion marked one cell per step.  The default here is the greedy criterion with a low
#      threshold (THRESH), which marks every cell within a factor of the worst.
#   6. plot_phase_space.py runs on the adaptive output and its portraits look right, but in the small run its own
#      "int n dx from samples" (7.82) disagreed with the driver's phase_reductions int_f (1.99).  I have not found which
#      is at fault; treat that script's integrated quantities with suspicion on adaptive meshes.  The ACCURACY lines
#      and phase_reductions.csv are computed by the driver cell by cell.
# So: run CHECK=1 first (3 slabs).  It prints, per slab, the element and dof counts, the relative energy error after
# each refinement step, the wall time of each solve, the GMG iteration range and the peak summed RSS, and compares the
# ACCURACY integrals with the production run's (PROD below).  Then choose REF0, REF, THRESH and ANISO.
#
# Knobs (environment variables):
#   REF0    refinement steps on slab 0                              (default 8)
#   REF     refinement steps on every later slab                    (default 2)
#   THRESH  greedy marking: refine cells with at least this fraction of the largest cell error (default 0.3; the
#           driver's default is 0.8, which marks very few cells).  Smaller marks more cells per step.
#   DOERFLER  if set (e.g. 0.5), use Doerfler marking with this target instead of THRESH
#   ANISO   direction selection: bisect in direction k if its indicator is at least this fraction of the largest
#           direction's (default 0.5).  0 bisects every marked cell in all four directions; near 1, only in the worst.
#   MAXDOFS a slab stops refining once its phase-space mesh has at least this many dofs (default -1: no limit).  The
#           solve on the mesh that reaches it is the slab's last.  The last step can nearly double the mesh (421 k ->
#           783 k dofs on slab 0 of the first full run), so set it below the size you can afford.
#   TARGET  relative energy error at which a slab stops refining    (default 0.02; in the small runs the relative
#           energy error in this test norm was 30-55%, so the step counts above are what ends refinement)
#   BCTOL   relative L2 error to which the inflow/initial data are resolved before each slab (default 0.1, the driver's)
#   DP      test enrichment                                          (default 2, as production; 1 roughly halves storage)
#   NP      MPI ranks                                                (default 16)
#   SLABS, SAMPLED  as in the production script                      (default 60; CHECK=1 gives 3)
#   PROD    console.log of the production run, for the ACCURACY comparison
#
# Usage (from the Camellia build directory):
#   CHECK=1 ./production-adaptive-box-phase-space.sh      then      ./production-adaptive-box-phase-space.sh

REF0=${REF0:-8}
REF=${REF:-2}
THRESH=${THRESH:-0.3}
ANISO=${ANISO:-0.5}
TARGET=${TARGET:-0.02}
MAXDOFS=${MAXDOFS:--1}
BCTOL=${BCTOL:-0.1}
DP=${DP:-2}
NP=${NP:-16}
SLABS=${SLABS:-60}; SAMPLED=${SAMPLED:-0,1,2,3,5,7,10,15,20,30,40,50,59}
PROD=${PROD:-/private/viz/nov2026/mitl/production-scl-graph-transport-a1.8-x32-dp2-phase/console.log}
OUT=/private/viz/nov2026/mitl/adaptive-box-scl-graph-transport-a1.8-dp$DP-ref$REF0-$REF-phase
if [ "${CHECK:-0}" = 1 ]; then SLABS=3; SAMPLED=0,2; OUT=$OUT-check; fi
X=./drivers/papers/RelativisticVlasovSpaceTime/PlanarMITLRelativisticSpaceTime
SRC=$(grep -m1 "^CMAKE_HOME_DIRECTORY" CMakeCache.txt | cut -d= -f2)
PLOT=$SRC/drivers/papers/RelativisticVlasovSpaceTime/plot_phase_space.py

# (the driver prints its help to stderr, and exits nonzero after printing it)
if [ ! -x $X ]; then echo "$X not found: run this from the Camellia build directory."; exit 1; fi
if ! $X --help 2>&1 | grep -q -- "--maxVlasovDofs"; then
  echo "This build of PlanarMITLRelativisticSpaceTime has no --maxVlasovDofs: rebuild it from the latest anisotropic-refinement branch."; exit 1
fi

if [ -n "$DOERFLER" ]; then MARKING="--doerflerTarget=$DOERFLER"; else MARKING="--refinementThreshold=$THRESH"; fi

mkdir -p $OUT

CAMELLIA_MEMORY_PRESSURE_ABORT=critical OMP_NUM_THREADS=1 mpirun -np $NP $X \
  --gradedVelocityMesh --velocityCoreWidthFactor=2 --velocityCoreHalfWidthFactor=2 --velocityGrowthRatio=3 \
  --xElements=8 --tElements=1 --numUniformRefinements=0 --maxDofsForDirectSolve=0 --p=4 --dp=$DP \
  --tWidth=1e-1 --numTimeSlabs=$SLABS --ratio=1.5 --minSubsteps=2 --maxSubsteps=10 --fixedPointTol=1e-4 --J=5e8 \
  --useSerendipityBasis --coarseSolver=SuperLUDist --accuracyDiagnostics --densityDiagnostic \
  --useGraphTestNorm --L2WeightGraph=1e2 --L2WeightGraphTransportScale=1.6 --L2WeightGraphAccelerationBounds=1.8,0.65 \
  --phaseSpaceDiagnostic --phaseSpaceDiagnosticSlabs=$SAMPLED \
  --useBoxMesh --useAdaptivity --boxAnisotropicAdaptivity --boxAnisotropyThreshold=$ANISO \
  --maxRefinementsFirstSlab=$REF0 --maxRefinements=$REF --maxRefinementsFinalSlab=$REF \
  --relErrorTarget=$TARGET --maxVlasovDofs=$MAXDOFS $MARKING --bcResolutionTolerance=$BCTOL \
  --outputDir=$OUT --phase0Instrumentation --phase0TimingFile=$OUT/timing.txt --timeMonitorSummary \
  > >(tee -a $OUT/console.log) 2>&1

LOG=$OUT/console.log
echo
echo "================================ summary ($(date)) ================================"
echo "settings: REF0=$REF0 REF=$REF marking=[$MARKING] ANISO=$ANISO TARGET=$TARGET MAXDOFS=$MAXDOFS BCTOL=$BCTOL DP=$DP NP=$NP"
echo
echo "per solve: slab, refinement step, phase-space elements, dofs, configuration elements, wall seconds"
grep "event=adaptive_slab_solve" $LOG | sed -E 's/.*seconds=([0-9.e+-]+).*vlasov_dofs=([0-9]+) vlasov_elements=([0-9]+).*config_elements=([0-9]+) slab=([0-9]+) refinement=([0-9]+).*/\5 \6 \3 \2 \4 \1/' \
  | awk '{ printf "  slab %3d  step %2d  %7d elements  %9d dofs  %5d config elements  %9.1f s\n", $1, $2, $3, $4, $5, $6; t[$1] += $6; e[$1] = $3 }
         END { print ""; print "per slab: final elements, total solve seconds (production: 28160 elements, 926 s/slab mean)";
               for (s=0; s in t; s++) { printf "  slab %3d  %7d elements  %9.1f s\n", s, e[s], t[s]; total += t[s] }
               printf "  all slabs: %.0f s = %.2f h of slab solves\n", total, total/3600 }'
echo
echo "relative energy error after each refinement step:"
grep -E "solution after refinement [0-9]+: relative error|After re-solve, relative error" $LOG | sed 's/^/  /'
echo
ITS=$(grep "event=vlasov_substep" $LOG | grep -o "iterations=[0-9]*" | cut -d= -f2 | sort -n)
if [ -n "$ITS" ]; then
  echo "GMG iterations per substep: min $(echo "$ITS" | head -1), median $(echo "$ITS" | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'), max $(echo "$ITS" | tail -1)   (production: 23 / 199 / 302)"
fi
PEAK=$(grep -o "sum_rss_gb=[0-9.e+-]*" $LOG | cut -d= -f2 | sort -g | tail -1)
RANK=$(grep -o "max_hwm_gb=[0-9.e+-]*" $LOG | cut -d= -f2 | sort -g | tail -1)
echo "peak summed RSS $PEAK GB, max per-rank high-water mark $RANK GB   (production: 150.5 GB summed RSS, 113.5 GB footprint)"
TOTAL=$(grep -o "event=total seconds=[0-9.e+-]*" $LOG | cut -d= -f3)
[ -n "$TOTAL" ] && echo "total wall time $(awk -v t=$TOTAL 'BEGIN { printf "%.0f s = %.2f h", t, t/3600 }')   (production: 68980 s = 19.16 h for 60 slabs)"
echo
if [ -f "$PROD" ]; then
  echo "ACCURACY integrals against the production run ($PROD): relative differences"
  awk '
    function field(line, name,   m) { if (match(line, name "=[-0-9.e+]+")) { m = substr(line, RSTART, RLENGTH); sub(/^[^=]*=/, "", m); return m } return "" }
    FNR == NR { if ($1 == "ACCURACY") { s = field($0, "slab"); pf[s] = field($0, "int_f"); pe[s] = field($0, "int_E2"); pj[s] = field($0, "int_Jx") } next }
    $1 == "ACCURACY" { s = field($0, "slab"); if (!(s in pf)) next
      f = field($0, "int_f"); e = field($0, "int_E2"); j = field($0, "int_Jx")
      printf "  slab %3d  int_f %+.3e (%.6f vs %.6f)  int_E2 %+.3e  int_Jx %+.3e   %s elements\n", s, (f-pf[s])/pf[s], f, pf[s], (e-pe[s])/pe[s], (j-pj[s])/(pj[s] != 0 ? pj[s] : 1), field($0, "vlasov_elements") }
  ' "$PROD" $LOG
  echo "  (the production run is itself an approximation: differences that shrink as REF0/REF grow are the adaptive mesh"
  echo "   converging; differences that settle at a nonzero value may be either run's error)"
else
  echo "ACCURACY lines (no production log at $PROD to compare with):"
  grep "^ACCURACY" $LOG | sed 's/^/  /'
fi
echo
if [ "${CHECK:-0}" = 1 ]; then
  echo "(check run.  Before the full run, look at: elements per slab against 28160 and whether they keep growing; seconds"
  echo " per slab against 926; GMG iterations; and the ACCURACY differences.  More resolution: raise REF0/REF or lower"
  echo " THRESH.  Too many cells: raise THRESH or ANISO.)"
  exit 0
fi
echo "plotting"
python3 $PLOT $OUT --average 20 || echo "plot_phase_space.py failed; the run's output is complete in $OUT"
