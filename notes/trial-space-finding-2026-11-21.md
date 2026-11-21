# Trial space and the acceleration-scale κ disagreement (cloud side, 2026-11-21)

Re: the local reply, "Results (local, 2026-11-20 evening)". It proposed that the cloud isolate the trial space at L0.

**Summary.**
1. The cloud's Serendipity trial space at p = 2 is much smaller than the TT trial space: 61 against 232 trial dofs per
   cell.
2. With the full tensor-product trial space, the cloud's p = 2 baseline (κ 857 at L0) matches yours (780 at L1). So the
   trial space explains the baseline gap.
3. It does **not** explain the response to a. With the full tensor-product trial space, a = (6.11, 4.15) still lowers κ
   on the cloud side by 2.3× (857 → 372), against your −5% (780 → 743).
4. That leaves the **mesh** (L0 against L1). Your planned L0 run isolates it.

## 1. The trial spaces

Per cell, from the cloud's element type at L0, p = 2, Δp = 2:

| trial space | field f | flux t̂_n, per face | trial dofs per cell | test dofs per cell | global dofs at L0 (Dirichlet) |
|---|---|---|---|---|---|
| Serendipity (`--useSerendipityBasis`, also production's) | **5** (L² degree 1) | **7** (face degree 2) | 61 | 625 (Q₄) | 37,146 (9,282) |
| full tensor product (the TT discretization) | 16 (Q₁) | 27 (Q₂) | 232 | 625 (Q₄) | 140,386 (35,802) |

- The fluxes have face degree 2, enriched by 1 over their natural order, as expected.
- Camellia's Serendipity field space at degree 1 is the 5 functions of total degree ≤ 1, not Q₁.
- Its degree-2 face space has 7 of the 27 Q₂ functions.
- Free dofs: 27,864 against 104,584.
- The full tensor-product count decomposes exactly: 140,386 = 880 × 16 + 4,678 × 27 (880 cells, 4,678 faces).

## 2. κ with the full tensor-product trial space (cloud, L0, p = 2)

**Setting.**
- Production L0 (880 cells), fields on: after slab 0's solve, 2 substeps.
- Separable graph⊕ norm, α = 100, C = 1.6, full tensor-product test space.
- One-level PCG, κ from the CG Lanczos tridiagonal (a lower bound).

| trial space | point-Jacobi κ, a = (1.8, 0.65) | point-Jacobi κ, a = (6.11, 4.15) | change | λ_min | sweep κ(PS) |
|---|---|---|---|---|---|
| Serendipity (earlier) | 425 | 156 | −63% | 0.0077 → 0.0209 | 109 → 43 |
| **full tensor product** | **857** | **372** | **−57%** | **0.0047 → 0.0103** | 198 → 66 |
| *local TT: L1, p = 2, slab 0 after 2 substeps* | *780* | *743* | *−5%* | | |

Trace-block Jacobi agrees with point Jacobi: 805 → 353 with the full tensor-product trial space.

## 3. What this says

- **The trial space accounts for the baseline.** At p = 2 and today's a:
  - cloud with Serendipity, L0: 425;
  - cloud with full tensor product, L0: 857;
  - TT, L1: 780.

  So with the same trial space the two codes agree to 10%. (At p = 4 they already agreed: cloud Serendipity 1,141 at L0
  against TT 1,294 at L1.)
- **The trial space does not account for the response.** With the TT trial space, the cloud still sees a 2.3× drop from
  the wider a, again entirely at the low end: λ_min roughly doubles.
- **Ruled out between us:** p, the field state, the κ estimator, the boundary treatment, how a enters the norm, and now
  the trial space.
- **Left: the mesh (L0 against L1).** Your TT discretization on L0, p = 2, slab 0, both a, decides it.
  - If you see the 2.3× drop at L0, the effect of a is mesh-dependent. It would then shrink from L0 to L1, which would be
    worth understanding: for example, whether the low-end mode that a fixes is a coarse-mesh phenomenon in the cathode
    cells.
  - If you don't, something in the two implementations differs that we have not yet named.

## 4. Pending on the cloud side

- **p = 4 with the full tensor-product trial space at L0, both a.** Queued until our L1 multigrid run finishes; memory
  does not allow both at once.
- **L1 multigrid diagnostics** (corrected sweep smoother, Chebyshev-interval variants, slow-mode analysis). Running.

## Aside, for production

At p = 2, production's Serendipity trial space has only P1 fields (5 per cell) and 7 flux functions per face. It is
much smaller than the full tensor-product space. That may be intended. It is mentioned here because it is a large
difference between the conventional and TT discretizations.
