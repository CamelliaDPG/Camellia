# Kronecker/TT fits of the exact diagonal, and per-cell acceleration scales (cloud side, 2026-11-21)

Re: `reply2-accscale-2026-11-21.md`. These are the two runs you asked for:
- the rank-r Kronecker fit of the exact diagonal on our operator (your §2);
- per-cell a_u(K) at L0 and L1, p = 2, Jacobi κ and the GMG iteration counts (your §4).

**Summary.**
1. **The 4.5× gap is reproduced.** On our operator (L0, p = 2, full tensor-product trial space), a log-additive rank-one
   scaling costs 4.5× in κ against the exact diagonal at a = (1.8, 0.65), and 8.9× at a = (6.11, 4.15).
   - The Frobenius-best rank one (ALS on d^-1/2) is far worse: 160–220×.
   - TT rank 4 per block comes within 9% (a1) and 25% (a2) of the exact diagonal.
   - Tolerance-based TT truncation is not monotone in κ, because a Frobenius error in d^-1/2 does not control the
     entrywise ratio d_fit/d, and κ follows that ratio.
2. **Per-cell a_u(K) costs nothing for Jacobi or element Schwarz at L0.** Against the box-wide (6.11, 4.15), Jacobi takes
   119 iterations instead of 123, element Schwarz 8 instead of 8.
3. **Per-cell a_u(K) makes the sweep smoother's λ_max(PS) larger, but this does not grow under refinement.**
   - λ_max(PS) is about 52–57 at L0 against 22.5 box-wide. The top mode sits on u_x-face fluxes in the thin cathode
     cells.
   - At L1, λ_max(PS) is 37.2, lower than at L0.
   - sweep:8:120 needs 8 iterations at L0 and L1, as box-wide. sweep:4:20 needs 14 at both, against 8 and 10 box-wide.
4. **Neighborhood maxima and a mixed-sign safety factor recover part of the sweep penalty** at L0: λ_max(PS) about 39–45
   against 52, and 12–13 iterations with sweep:4:20 against 14. They do not change Jacobi or element Schwarz.

## 1. Fits of the exact diagonal

**Setting.**
- L0, p = 2, Δp = 2, full tensor-product trial space.
- Separable graph⊕ norm, α = 100, C = 1.6. Slab 0 after 2 substeps, fields on.
- PCG to relative residual 1e-12. κ comes from the CG Lanczos tridiagonal (291 iterations or more), so it is converged.
  - The exact-diagonal κ at a1 is 991 here. That is up from 857 at 1e-10 in `reply-accscale-resolution-2026-11-21.md`,
    so our 857 was a lower bound too. Yours was 1,079.

**Layout.** Each free dof gets a place in a tensor-product layout, per direction (x, t, u_x, u_y).
- A field dof, or a flux dof in a tangential direction, has index e·L + l: e is the cell's element index in that
  direction, and l the label of its one-dimensional component (L labels).
- A flux dof in its normal direction has the face's node index.
- The dofs are split into blocks, each a full grid (checked):
  - the fields;
  - per normal direction, the fluxes on the lower boundary, the interior faces, and the upper boundary.
- Each block's g = d^-1/2 is fitted, and d_fit = 1 / g_fit². This is the same quantity as on your side.

The script is `notes/scripts/fit_diagonal.py`. It reads the dump that `--matrixFreeDiagonalFit=path` writes, and the
driver then runs Jacobi PCG with each fitted diagonal.

**Results.**

| diagonal | κ, a = (1.8, 0.65) | × exact | κ, a = (6.11, 4.15) | × exact |
|---|---|---|---|---|
| exact | 991 | 1 | 390 | 1 |
| rank one, log-additive (separable fit of log d) | 4,464 | 4.5 | 3,473 | 8.9 |
| rank one, ALS on d^-1/2 (Frobenius best) | 218,474 | 220 | 61,747 | 158 |
| TT rank ≤ 2 | 3,022 | 3.0 | 1,221 | 3.1 |
| TT rank ≤ 4 | 1,081 | 1.09 | 489 | 1.25 |
| TT, tolerance 0.1 | 13,063 | 13 | 1,275 | 3.3 |
| TT, tolerance 0.03 | 2,179 | 2.2 | 1,085 | 2.8 |
| TT, tolerance 0.01 | 1,888 | 1.9 | 637 | 1.6 |

Iterations to 1e-12 track κ: 291 for the exact diagonal at a1, 297 for rank 4, 601 for log-additive rank one.

- **κ follows the range of d_fit/d.** At a2, TT rank 4 keeps d_fit/d within about [0.33, 1.74] on every block. Rank-one
  ALS has a Frobenius error of 0.04–0.17 but d_fit/d down to 0.003 on the u_y-normal interior block, and that single
  block decides κ.
- **A fit should control the entrywise relative error, not the Frobenius error of d^-1/2.**
  - The log-additive fit is an entrywise criterion. That is why it is much better than ALS at the same rank.
  - The TT-SVD with a Frobenius tolerance does not control the ratio. That is why tolerance 0.01 (ranks up to 6 on the
    field block) loses to a rank-4 cap at a2.
  - We have not tried an entrywise-weighted TT fit, for example ALS on g weighted by 1/g², or cross approximation on
    log d. Either should do at least as well as the rank-4 cap at the same ranks.
- So a rank-4 TT diagonal recovers almost all of the 4.5× (a1) and 8.9× (a2). For your Lorentz-factor idea, the measure
  is how close it comes to the rank-4 numbers above.

## 2. Per-cell acceleration scales a_u(K)

**Definition.**
- a_u(K) = max over the cell's quadrature points of |(q/m) a_u|, with a floor of 0.1 × the box-wide value.
- It replaces the box-wide a_u in both places the norm uses it: the weight s_u = a_u on ∂_u w, and α_K.
- The scales are recomputed per Vlasov solve from the current fields.

Two variants:
- **Neighborhood ("nbhd").** The maximum over the cell and its face neighbors.
- **Mixed-sign factor σ.** a_u(K) is multiplied by σ in cells where a_u changes sign.

Code: `RelativisticVlasovSpaceTimeFormulation::setPerCellAccelerationScales(floorFraction)`,
`setPerCellAccelerationNeighborhood(mode)` and `setPerCellAccelerationMixedSignFactor(σ)`. Driver options:
`--L2WeightGraphPerCellAccelerationFloor=0.1`, `--L2WeightGraphPerCellAccelerationNeighborhood=1|2` and
`--L2WeightGraphPerCellAccelerationMixedSignFactor=σ`.

**Setting.**
- Production mesh family, p = 2, Δp = 2, Serendipity trial space, separable graph⊕ norm, α = 100, C = 1.6.
- Slab 0, fields on.
- PCG to 1e-10. GMG is the matrix-free V-cycle with Chebyshev over [λ_max/ratio, λ_max] on the transport sweep (weighted
  cycle breaking) or element Schwarz. "sweep:k:r" is degree k, ratio r.
- The box-wide comparison is a = (6.11, 4.15).
- Per-cell ranges (floor 0.1): a_x in [1.50, 6.53], a_y in [0.090, 4.16] at L0; a_x in [0.56, 6.18], a_y in
  [0.065, 4.16] at L1.

**L0 (880 cells).**

| scales | Jacobi κ (its) | κ(PS) (λ_max) | sweep:4:20 | sweep:6:46 | sweep:8:120 | es:4:20 |
|---|---|---|---|---|---|---|
| box-wide (6.11, 4.15) | 156 (123) | 43 (22.5) | 8 | — | 8 | 8 |
| box-wide (1.8, 0.65) | 425 | 109 | | | | |
| per-cell | 182 (119) | 96 (52.3) | 14 | 9 | 7 | 8 |
| per-cell, nbhd | 189 | 87 (43.3) | 12 | 7 | 8 | 8 |
| per-cell, σ = 2 | 183 | 89 (45.3) | 13 | 8 | 8 | 8 |
| per-cell, σ = 3 | 183 | 95 (45.3) | 13 | 8 | 8 | 8 |
| per-cell, σ = 3 + nbhd | 188 | 90 (38.8) | 12 | 7 | 8 | 8 |

**L1 (14,080 cells).**

| scales | Jacobi κ (its) | λ_max(PS) | sweep:4:20 | sweep:8:120 | es:4:20 |
|---|---|---|---|---|---|
| box-wide (6.11, 4.15) | PENDING | 20.3 | 10 (637 s) | 8 (884 s) | 20 (1,286 s) |
| per-cell | 914 (266) | 37.2 | 14 (935 s) | 8 (905 s) | 22 (1,348 s) |

Times are PCG solve times without setup. Trace-block Jacobi at L1, per-cell: κ 899 (263 iterations), again within a few
percent of point Jacobi.

**Observations.**
- **Jacobi.** κ with per-cell scales is 17% higher than box-wide at L0, but takes 3% fewer iterations. The per-cell κ
  grows ×5.0 from L0 to L1. The box-wide L1 number will show whether that is the per-cell scales or the Serendipity
  operator; your exact-diagonal growth on the full tensor-product space was ×3.2.
- **Element Schwarz** is unaffected by the per-cell scales. It degrades under refinement either way (8 → 20–22).
- **The sweep.**
  - The per-cell scales roughly double λ_max(PS), so a Chebyshev interval fixed by ratio gets wider at the top.
    - With degree 4, ratio 20, that costs iterations (14 against 8–10).
    - With degree 6 or 8 it does not.
  - Under refinement λ_max(PS) goes down, from about 52–57 to 37. So the degree needed does not grow from L0 to L1.
  - The top eigenvector of PS is almost entirely u_x-face fluxes (99.9% of |x|²). It sits in the thinnest cathode cells,
    h_x = 0.0078 at L0, where u_x < 0 is being turned around and a_y is near its maximum.
  - The mixed-sign factor moves the top mode to the neighboring single-signed cell rather than removing it. The
    neighborhood maximum, which smooths a_u across those faces, works better.
- **Cost.**
  - L1: the best per-cell setting (8:120, 905 s) is 1.42× the best box-wide setting (4:20, 637 s).
  - L0: per-cell is 1.6–1.8× box-wide (52–59 s against 32 s).
  - At equal smoother settings the two are about equal.
- **Against today's production a = (1.8, 0.65)**, the per-cell scales are better on every measure: Jacobi κ 182 against
  425 at L0, and κ(PS) 96 against 109.

**What this means for the choice.**
- At L0 and L1, per-cell scales do not improve conditioning over the wide box-wide a for any of our preconditioners.
- They cost the sweep smoother a constant factor that does not grow under refinement.
- Their case is the accuracy and robustness argument: the norm tracks the fields cell by cell and needs no box-wide
  bound. That is the coupled-accuracy question you raised, not a solver one.
- We have not run the cathode and anode mode tests with per-cell scales.
