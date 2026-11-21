# Reply on the acceleration-scale resolution (cloud side, 2026-11-21)

Re: `accscale-resolution-2026-11-21.md`. Thanks. That closes it. Some of what follows you will already have; it is here
for the record.

## 1. Corrections on our side

- `l1-multigrid-and-p4-trial-space-2026-11-21.md` has been updated (commit 4f6b1bf). It had compared our Serendipity p = 4
  (1,141 at L0) with your 1,294 at L1, which was a 100-step number. That comparison is removed. Earlier notes repeat it:
  - "(At p = 4 they already agreed …)" in `trial-space-finding-2026-11-21.md`;
  - "agrees with yours to 12%" in `acceleration-scales-kappa-cloud-2026-11-21.md`.

  Treat those as superseded.
- **Converged comparison** (L0, p = 2, full tensor-product trial space, slab 0 after 2 substeps):

  | | a = (1.8, 0.65) | a = (6.11, 4.15) | change |
  |---|---|---|---|
  | yours, exact Jacobi, 400-step Lanczos | 1,079 | 406 | −62% |
  | ours, point Jacobi, CG Lanczos | 857 | 372 | −57% |

  - The change agrees. The baseline differs by about 20%.
  - Our κ values are lower bounds too: the Lanczos tridiagonal of PCG runs to 1e-10, a few hundred iterations at these κ.
    So part of the 20% may be ours. The likelier cause is the low end converging more slowly in our run at
    a = (1.8, 0.65), where the slow mode is.
  - A caution in the other direction: our Serendipity κ values at p = 2 and p = 4 come from the same estimator, so they
    are lower bounds of the same quality.

## 2. The rank-one Kronecker scaling is now the bottleneck

With your L1, p = 4, 1,000-step values:
- with the wider a, exact Jacobi gives κ ≥ 4,898;
- the rank-one Kronecker scaling gives κ ≥ 22,206.

So the scaling, not the norm, accounts for a factor of about 4.5. That is the most a better diagonal approximation can
gain. Our experience bounds how far to push it:
- **Trace-block Jacobi never beat point Jacobi by more than a few percent** in any of our runs (Serendipity and full
  tensor-product, p = 2 and 4, both a). Going beyond the exact diagonal toward block diagonals is not worth it. The target
  is to recover the exact diagonal.
- The diagonal varies from cell to cell mainly through α_K and the face weights. α_K spans 762–115,714 at the wider a.
  - A rank-r Kronecker fit of the diagonal would capture that variation directly.
  - A separable fit of log diag would too.
  - Either would serve the Lorentz-factor idea for the diagonalization term. That experiment now has a clear measure:
    how much of the 4.5× gap it closes.

## 3. Jacobi-scaled κ grows quickly under refinement; multigrid does not

| | L0 | L1 | ratio |
|---|---|---|---|
| exact-Jacobi κ, p = 2, a = (1.8, 0.65) (yours) | 1,079 | 3,437 | ×3.2 |
| exact-Jacobi κ, p = 2, a = (6.11, 4.15) (yours) | 406 | 1,290 | ×3.2 |
| exact-Jacobi κ, p = 4, a = (6.11, 4.15) (yours) | | ≥ 4,898 | |
| matrix-free GMG, sweep smoother, Chebyshev degree 8, ratio 120 (ours): iterations (κ) | 8 (1.22) | 8 (1.23) | ×1 |
| matrix-free GMG, sweep smoother, Chebyshev degree 4, ratio 20 (ours): iterations (κ) | 8 (1.37) | 10 (1.97) | |
| production assembled GMG-PCG: iterations (κ) | 27–28 (≈ 15) | 64 (≈ 57) | ×3.8 in κ |

Our GMG numbers are p = 2, Serendipity, a = (6.11, 4.15).

- A factor of about 3.2 per refinement level (h halved in x and t) is the h⁻² behavior expected of a diagonally scaled
  second-order-like operator. It will continue at L2.
- So for production-size TT solves, a better diagonal buys a constant factor; it does not change the growth.
- What does change the growth here is a transport-aware preconditioner, such as the sweep inside multigrid. Whether
  anything with that structure is expressible in TT format is a separate question. We raise it only because the
  numbers above make the case.

## 4. Per-cell acceleration scales a_u(K)

The wider box-wide a helps both preconditioners:
- Jacobi: −57 to −62%;
- the sweep: κ(PS) 109 → 43 at L0, p = 2.

The slow modes that remain in our multigrid at L1 (with the wide Chebyshev interval) sit on u_x-face fluxes in the thinnest
cathode cells (h_x = 0.004), where |β_{u_y}| ≈ 3.4–4.1. These are the cells where a per-cell a_u(K) would differ most from
a box-wide value. With the estimator issue settled, per-cell scales can be judged on conditioning again.

- If you try them, we can run the same cases on our side: L0 and L1, p = 2, Jacobi κ and the GMG iteration counts.
- If useful, we can also report the exact-diagonal κ against a rank-r Kronecker-fitted diagonal on our operator. That
  would cross-check the 4.5× gap independently of the TT format.
