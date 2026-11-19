# Transport sweep for the matrix-free planar MITL DPG solver: update since the PDF

This update follows `static-condensation-trace-structure.pdf` (the 11-page version with Section 9, "First
measurements"). Notation is the PDF's:
- S is the global ultraweak DPG operator with the separable graph norm. It is matrix-free, uncondensed: fields and
  face fluxes.
- P = R^{-1} W^{-1} R^{-T} is the transport-sweep preconditioner of its Proposition 10. Each cell does its own local
  DPG solve for its fields and outflow fluxes, given its inflow fluxes. The cells are visited in upwind order.
- The identity S = R^T W R + Q holds exactly, with Q >= 0. So lambda_min(PS) >= 1, and kappa(PS) <= 1 + rho.

**Headline.** Used as the smoother inside the existing GMG h-hierarchy, the transport sweep gives h-independent
convergence under uniform refinement: 7 iterations and kappa about 1.10 at both 128 and 2,048 cells. Element Schwarz in
the same place gives 9 -> 20 iterations, and production GMG gives 47 -> 88. **Caveat (Section 3a):** on the deeply
graded cathode mesh, with cell Courant numbers up to about 15,000, the sweep smoother is much worse than element
Schwarz (107 against 17 iterations).

## Setting for every number below

| Setting | Value |
|---|---|
| Orders | H^1 order p = 2, Delta p = 2 |
| Spaces | Serendipity trial, full tensor-product test |
| Norm | Separable graph norm (`--L2WeightGraph=100 --L2WeightGraphTransportScale=1.6 --L2WeightGraphAccelerationBounds=1.8,0.65`) |
| Hardware | One core |
| Solver | PCG to a relative residual of 1e-10 |
| Operator | The first time slab, before its solve |

The fields are zero in that slab, so beta_u = 0. Transport is only in x and t, the u-faces are characteristic
(beta . n = 0), and the upwind graph has no cycles. **Nothing here has been tested with nonzero fields yet**; that is the
next run.

Coordinates are (x, t, u_x, u_y). A mesh label "x × t × v" means that many cells in x, t and each of u_x and u_y.
Entries are "iterations (kappa)".

## 1. The sweep as a smoother: h-independent

The smoother is Chebyshev of degree 2 or 4 applied to P S, with lambda_max from a Lanczos (CG) estimate times 1.1, and
the interval [lambda_max/20, lambda_max]. The coarse correction is the Galerkin coarse operator at p = 1 on the same
mesh, then Camellia's GMG on the p = 1 h-hierarchy down to the base mesh. The base mesh is 2 × 1 × 2 and r is the number
of uniform refinements.

| Smoother inside the full GMG hierarchy | r = 1 (128 cells, 5,344 dofs) | r = 2 (2,048 cells, 76,544 dofs) |
|---|---|---|
| **Transport sweep, Chebyshev degree 4** | **7 (1.108)** | **7 (1.104)** |
| Transport sweep, Chebyshev degree 2 | 12 (1.83) | 11 (1.84) |
| Element Schwarz, Chebyshev degree 4 | 9 (1.49) | 20 (4.86) |
| Production GMG (assembled, Serendipity test) | 47 | 88 |

At 2,048 cells the sweep-smoothed solve takes 74 s, against 176 s with element Schwarz. Setup is about 105 s in both
cases. In both, the prolongation (31 s) and the coarse GMG hierarchy (24 s) are the large items; the sweep's own blocks
take 21 s.

On a mesh with the spatial direction refined more (4 × 2 × 4 base, every x-element bisected twice, 512 cells), with an
exact p = 1 coarse solve:
- sweep smoother, degree 4: 6 iterations (kappa 1.10);
- element Schwarz smoother, degree 4: 16 iterations (kappa 4.23).

**Interpretation.** The sweep handles the error components that run along the characteristics. Those are what local
smoothers could not remove, and they are what made kappa grow like h^{-2} before. What the sweep leaves is smooth enough
for the h-coarse space. This mechanism is a hypothesis; only the iteration counts are measured.

## 2. The sweep alone: what controls it

As a one-level preconditioner, rho grows. Under uniform refinement, kappa(PS) goes 2.9 -> 4.5 -> 7.1 (PDF, Section 9).
The new finding is what drives that growth.

**The cell Courant number at the cathode.** The spatial mesh is hard-coded as geometrically graded toward x = 0:
`MeshFactory::geometricGradedIntervalMeshTopology`, so `xElements = N` gives h_min = 2^{-(N-1)}. Time cells are 0.5 or 1
wide. The cell Courant number is nu = |v_x| Δt / h_x, with |v_x| <= 0.92 here. At the smallest cell:

| Mesh | nu at the smallest cell | kappa(PS), sweep alone |
|---|---|---|
| 4 × 2 × 4 | about 4 | 5.2 |
| x bisected once (`--numSpatialRefinements=1`) | about 7 | 7.4 |
| x bisected twice | about 15 | 13.3 |
| 16 graded x-elements, slab length 1/8 | about 1,900 | 1,059 |
| 16 graded x-elements, slab length 1/4 | about 3,800 | 2,063 |
| 16 graded x-elements, slab length 1/2 | about 7,500 | 3,790 |
| 16 graded x-elements, slab length 1 | about 15,000 | 6,036 |

kappa tracks the largest nu. In a cell with nu >> 1, almost every characteristic enters through the x-face. The bottom
t-face is then nearly redundant inflow data that the low-order local polynomial cannot reconcile, and the local inflow
Schur complement Sigma_K (part of Q) becomes large. The power iteration on the 16-element mesh agrees: once the
characteristic faces are taken out of the sweep, the top eigenvector of PS is 99.99% t-face fluxes.

**Correction to Section 9 of the PDF.** I previously read "x refinement" as the driver, and claimed a
"fixed Courant number" test ruled nu out. Both came from raising `xElements`, which deepens the grading rather than
refining, so nu at the smallest cell was not held fixed. Proper x-refinement (bisection, which keeps the grading)
degrades the sweep only mildly, as in the table above. Refining u does not affect it; refining t raises kappa mildly.

## 3. Characteristic faces: separate treatment does not help

On faces with beta . n = 0 the inflow/outflow split is undefined. In the field-free slab these are all the u-faces. I
classified them by the cell averages of beta (|beta . n| <= 1e-8 max |beta|) and took them out of the sweep, so they get
no ownership and no dependency. They were then preconditioned by their own face blocks of S, either additively or by a
symmetric product (face blocks, sweep, face blocks).

| Mesh | Plain sweep | + face blocks, additive | + face blocks, multiplicative |
|---|---|---|---|
| 4 × 2 × 4 (128 cells) | 22 (5.2) | 31 (8.8) | 25 (6.2) |
| x bisected twice (512 cells) | 32 (13.3) | 46 (20.5) | 38 (14.7) |
| 16 graded x-elements (512 cells) | 236 (6036) | 320 (9704) | 310 (6979) |

This is slightly worse everywhere. The plain sweep's top mode was on u_y-face fluxes, but removing that class just
exposes the t-face modes behind it. So the characteristic faces are not what limits the sweep; the high-nu cells are.
The option stays in the code (`upwindCellOrder(..., characteristicFace, tolerance)` and
`transportSweep(..., characteristicFace, multiplicative)`), but it is not used by default.

## 3a. On the high-Courant-number cathode mesh the sweep smoother is worse (added after the first version)

Two-level runs with an exact p = 1 coarse solve, Chebyshev degree 4, on the 16 graded x-element mesh (512 cells,
nu up to about 15,000):

| Smoother | Iterations (kappa) | Solve |
|---|---|---|
| Transport sweep | 107 (196) | 327 s |
| Element Schwarz | **17 (11.1)** | 45 s |

There, P S has spectrum [1, about 6,640]. Chebyshev with eigenvalue ratio 20 smooths only [332, 6,640], so everything
in [1, 332] is left to the coarse space, which cannot handle it. On that mesh the sweep is a poor smoother and element
Schwarz is clearly better.

Section 1's h-independence therefore holds on meshes with moderate cell Courant numbers (up to about 15 here). It does
not hold on the deeply graded cathode mesh.

What to try next:
- a hybrid smoother: the sweep where nu is moderate, element (or column) blocks in the high-nu cells;
- better shaped time cells near the cathode;
- a Chebyshev interval adapted to the spectrum of P S.

## 4. Relevance to the tensor-train side (my reading)

- **Structure.** The useful structure is causal: inflow-to-outflow transport, in upwind order. The time direction is
  always upwind (beta_t = 1). Spatial upwinding is by the sign of v_x(u), which does not depend on x or t.
- **What a TT preconditioner should capture.** Something that resolves transport along characteristics plus a coarse
  correction. A purely local or additive smoother did not scale here.
- **Courant number in the discretization.** Cells with |v_x| Δt / h_x >> 1 (thin cells at the cathode against long time
  cells) are where the sweep degrades, alone and as a smoother. This is a property of the DPG discretization on those meshes, not of the matrix-free
  implementation, so it presumably affects any solver.
- **Not yet tested:** nonzero fields. Those turn the u-faces into genuine inflow/outflow faces, and the v × B rotation
  creates cyclic dependencies between cells of one time layer, which the sweep lags. That is the first thing to check.

## Status and next runs

The two-level run on the 16 graded x-element mesh is done (Section 3a): the coarse correction does not absorb the
high-nu cathode cells when the sweep is the smoother.

Next:
1. The same checks on a later slab with nonzero fields (`--matrixFreeCheckSlab=N`).
2. p = 4.
3. Cost: setup is dominated by the prolongation and the coarse hierarchy.

Code is on branch `separable-gram-fast-diagonalization` (latest commit 13c8b4c). The pieces are in
`src/ScratchPad/MatrixFreeDPGOperator.cpp`:
- `transportSweep()`;
- `upwindCellOrder()`;
- `ElementMultiplicativeSchwarz`.

The smoother option is `MatrixFreeDPGTwoLevel::Options::transportSweep`. The driver flags are:
- `--matrixFreeSweepSmoother`
- `--matrixFreeMultiplicativeSchwarz`
- `--matrixFreeSweepOnly`
- `--matrixFreeSweepDiagnose`
- `--matrixFreeCharacteristicTolerance`
- `--matrixFreeCheckSlab`
