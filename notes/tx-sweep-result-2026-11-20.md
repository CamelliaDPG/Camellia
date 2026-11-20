# Reply: κ with a t–x-only sweep (cloud side, 2026-11-20)

**Short answer: no.** The t–x-only sweep does not capture the near-kernel. With the actual fields it is worse than point
Jacobi: κ ≈ 1,113 against 425. With the fields frozen at zero in P it is much worse: κ ≈ 21,000. By your criterion this
says the velocity-direction coupling dominates. The partial sweep is not worth building in TT.

## Setting

As requested:
- Production L0 (880 cells, 37,146 dofs; 9,282 Dirichlet).
- Fields on: the operator after slab 0's solve, 2 substeps.
- Today's norm: α = 100, C = 1.6, a = (1.8, 0.65), transport-scaled α_K.
- p = 2, Δp = 2, separable graph norm, full-tensor test space, Serendipity trial.
- One-level PCG to 1e-10. κ and the Ritz bounds come from the CG Lanczos tridiagonal.
- All rows below use the same operator, in one run.

## What P is

**Upwind order.** Cells are ordered by their t and x faces only:
- t: the bottom face is the inflow;
- x: the left face is the inflow for u_x > 0, the right face for u_x < 0.

The velocity faces are dropped from the graph. The ordering then has 0 cycle breaks and 0 lagged faces.

**Cell problems.** Each cell's local DPG problem is solved for its fields and its t/x outflow fluxes, from its t/x
inflow data.

**Velocity-face traces.** There are 13,496 such flux dofs, 36% of the unknowns. They are not swept: each face's dofs form
a block of S, which is Cholesky-factored.
- **additive:** P = (t–x sweep on the other dofs) ⊕ (velocity-face blocks);
- **symmetric multiplicative:** face blocks, then the sweep, then face blocks, with residual updates by the true S.

**Variant (b), frozen fields.** P is built from the operator with the acceleration terms dropped. PCG still runs on the
true S. The flux–flux blocks do not depend on the fields, so the velocity-face blocks are the same in (a) and (b). Only
the cell solves and the transport maps differ.

## Results

| P | iterations | κ(PS) | λ range |
|---|---|---|---|
| point Jacobi | 157 | 425 | [0.0077, 3.28] |
| trace-block Jacobi | 157 | 417 | [0.0077, 3.22] |
| full sweep, weighted cycle breaking (all directions) | 70 | 109 | [0.55, 59.4] |
| **t–x sweep, actual fields, additive velocity-face blocks** | 224 | **1,113** | [0.0100, 11.1] |
| **t–x sweep, fields frozen in P, additive velocity-face blocks** | 773 | **21,061** | [0.0185, 390] |
| t–x sweep, either variant, symmetric multiplicative | 344 / 1,366 | — | (Lanczos estimate failed) |

The multiplicative composition with approximate blocks is not guaranteed to be positive definite. The CG Lanczos
estimate failed there, so treat those rows as invalid.

## What this says

1. **The near-kernel is not in the t–x transport alone.** Look at λ_min. The t–x sweep (a) leaves it at 0.0100, against
   Jacobi's 0.0077: the small eigenvalues barely move. A preconditioner that captured the near-kernel would lift them,
   as the full sweep does (λ_min = 0.55). So the slow modes involve the velocity directions. This matches our earlier
   diagnosis: with the fields on, the full sweep's extreme modes sat on u_x-face fluxes in the cathode cells. There,
   |β_{u_x}| = (q/m)|a_x| ≈ 4–6 dominates |β_x| = |v_x| ≤ 1 in cells of comparable size.
2. **Freezing the fields in P is harmful.** λ_max rises from 11 to 390. A field-free P transports along the wrong
   characteristics wherever the acceleration matters, so its cell solves disagree badly with S.
3. **Even the full sweep, with velocity transport, gains only a factor of about 4 over Jacobi at one level** (109
   against 425). It needs lagging on the cycles (0.5% of the coupling weight), and its limiting modes come from the
   domination failure of a = (1.8, 0.65). With a = (6.11, 4.15) the full sweep reaches κ = 43.

## Caveats

- p = 2 and L0 only. Your measurements are at p = 4 and L1. The structure of the conclusion (λ_min not lifted) should
  carry over, but it is not measured there.
- One slab, after 2 substeps. Fields are different in later slabs.
- The velocity-face blocks are block-diagonal per face, as you specified. A t–x sweep combined with something stronger
  in the velocity directions (e.g. a velocity-direction sweep or line solves) might do better. This experiment does not
  test that.

## Code

Branch `separable-gram-fast-diagonalization`, commit 8a03fbe:
- the driver option `--matrixFreeTxSweep`;
- `MatrixFreeDPGOperator::upwindCellOrder(..., sweepDirections)`;
- `BoxCellDPGOperator::setDroppedTestDirections()`;
- `transportSweep(..., residualOperator)`.
