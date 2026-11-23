# Matrix-free solvers, preconditioning and cell scaling: what the `separable-gram-fast-diagonalization` branch learned, 11-14 to 11-23

For the cloud session working on `anisotropic-refinement`. This covers ten days of work you have not seen. Detailed notes
are on the branch under `notes/` and are listed at the end. All numbers are single-core runs of the planar MITL driver
or unit tests, unless stated.

## 0. Where the code is

- **Branch:** `separable-gram-fast-diagonalization`, at cf6ebaf, which is 68 commits on top of `anisotropic-refinement`
  at c070e37 ("Vlasov test norm: weaken the L^2 term on each cell's constant mode", 11-14). Nothing was rebased away. If
  your branch has not moved since c070e37, this is a fast-forward.
- **GitHub says the repository has moved** to `CamelliaDPG/Camellia`. Pushes to the old URL still work.
- **Everything new is opt-in.** With default options, existing runs are unchanged. The changes to shared files are small:
  - `IP.h`: a virtual hook `applyInverseInnerProductMatrix()`, which declines by default.
  - `BF.cpp`, `Solution::computeErrorRepresentation`, `RieszRep::computeRieszRep`: they use the hook when it accepts.
    It accepts only for `SeparableTensorIP` on a full tensor-product test basis with no test-space BCs. Otherwise the
    old path (form and Cholesky-factor G) runs.
  - `VlasovSpaceTimeFormulation`: a GMG smoother application count (default 1), and a `prepareVlasovSolve()` call at
    the start of each substep. The per-cell scales (Section 4) are evaluated there; it does nothing otherwise.
  - `RelativisticVlasovSpaceTimeFormulation`: options for the full tensor-product test basis, outflow-face terms and
    per-cell acceleration scales (Sections 1 and 4).
  - The driver: about 800 lines of options and checks.
- **New files** (all in `src/ScratchPad`, headers in `src/include`):
  - `SeparableGramInverse`;
  - `TensorBasisDecomposition`;
  - `BoxCellDPGOperator`;
  - `MatrixFreeDPGOperator`;
  - `MatrixFreeDPGMultigrid`.
- **New unit tests:** `unit_tests/BoxCellDPGOperatorTests.cpp`, and additions to `SeparableTensorIPTests.cpp`.
- **Bug fix you already have:** `SeparableTensorIP`'s 1D point view (a heap overrun) went to `anisotropic-refinement`
  separately.

## 1. Exact element Gram inverse for the separable graph norm

- With `--separableGraphTestNorm`, the element Gram matrix is a Kronecker sum,
  G_K = J_K (α_K M⊗M⊗M⊗M + Σ_d ξ_d K_(d)), **on a full tensor-product test space**.
- In the shared 1D generalized eigenbasis V of (K, M), G_K^{-1} = V^{⊗4} diag(1 / (J_K(α_K + Σ_d ξ_d λ_{j_d}))) (V^T)^{⊗4}
  exactly. Each element enters through six scalars, and nothing is stored per element.
- Anisotropic p (per-direction 1D spaces) is supported.
- The constant-mode weight from c070e37 is handled by a Sherman–Morrison correction.
- **This does not hold on a Serendipity test space**: P^T G_Q^{-1} P is the inverse of a Schur complement, not G_S^{-1}.
  So the useful configuration is a Serendipity **trial** space with a full Q_k **test** space:
  `--useSerendipityBasis --useFullTensorTestBasis`.
- **Cost:**
  - In a sum-factorized operator, Q_k test costs no stored matrices.
  - In the assembled path, Q_k test makes the B quadrature about 4× more expensive at k = 3, and about 7× at k = 6.
  - At p = 4 with 128 cells, production assembly took 89 s with Q_k test against 14 s with Serendipity test.
  - Memory fell from 2.66 GB to 0.90 GB.
- **Measured against dense Cholesky:** 0.010 / 0.023 / 0.034 s at k = 4 / 5 / 6 for 200 right-hand sides, against
  Cholesky factor plus solve of 0.010 / 0.031 / 0.17 s. At k = 6 it also avoids 46 MB of factor per element.

## 2. Matrix-free DPG operator on box cells

- `BoxCellDPGOperator` applies B_K, B_K^T and S_K = B_K^T G_K^{-1} B_K by sum factorization. It reads the terms off the
  `BF`: field terms c(x) u against v or ∂_k v, and flux terms ⟨c t̂, v⟩.
  - Serendipity trial bases are zero-padded into the tensor layout.
  - Camellia's value transforms are matched.
  - It declines, with a reason, for anything else: jump terms, non-H¹ or non-tensor test bases, non-signed-permutation
    side maps, and so on.
- `MatrixFreeDPGOperator` is the global S = Σ_K P_K^T S_K P_K as a Tpetra operator, through the mesh's dof
  interpretation, constraints included.
- **Tested on an anisotropically refined BoxGDA mesh:** cells of different widths, one cell h-refined in x and u_x only
  (so hanging faces), and anisotropic p-raises on half the cells. It matches the assembled element matrices to 5e-15,
  global S x to 1e-12, and the load to 1e-12. CG reproduces the direct solve.
- **Cost:** at p = 4, one application costs about 1–1.6 ms per cell, against 0.74 s per cell to assemble.
- **Quadrature caveat.** At the driver's default enrichment of 0, the assembled and matrix-free operators differ by about
  5e-4 relative, and the solutions by about 3%. Both converge to each other as quadrature is enriched. The coefficients
  v = u/γ and the fields are not polynomial, and the assembled side uses fewer points per direction. This probably
  affects production runs too, and nobody has checked it there.

## 3. Preconditioning: what works

All of these are matrix-free PCG on the Vlasov system.

1. **One-level methods scale like h^-1 in iterations, as expected.**
   - Point Jacobi and trace-block Jacobi are within a few percent of each other in every run, at p = 2 and 4, on L0 and
     L1.
   - Block diagonals beyond the point diagonal are not worth building.
2. **Production GMG, and matrix-free GMG with local smoothers, give κ ~ h^-2.**
   - Iterations double, and κ grows 3.5–4×, per uniform refinement. This holds even with the full p- and h-hierarchy,
     and with element-Schwarz smoothing.
   - Production assembled GMG-PCG: 47 → 88 iterations (p = 2, 128 → 2,048 cells), and 27 → 64 iterations from L0 to L1
     on the production mesh.
   - The reading in `static-condensation-trace-structure.tex`: the DPG operator is a least-squares form of transport.
     Its low-energy modes are smooth along characteristics and rough across them. Isotropic coarsening plus local
     smoothing cannot reach them.
3. **Chebyshev beats repeated Richardson with the same smoother.**
   - Production's additive Schwarz smoother, applied 4 times as damped Richardson: 18 / 26 iterations at 64 / 512 cells (p = 2, two-level).
   - The same four applications as a Chebyshev polynomial: 7 / 13.
   - λ_max must come from Lanczos (CG) times 1.1. Power iteration underestimated it, and Chebyshev degree ≥ 3 was then
     indefinite.
   - We suggest this for production GMG (`GMGOperator::ApplySmoother`). It is not implemented there.
4. **A transport sweep as the multigrid smoother gives mesh-independent convergence.**
   - The sweep: cells in upwind order, each doing its own local DPG solve for its fields and outflow fluxes from its
     inflow fluxes. There is an exact identity S = R^T W R + Q with Q ≥ 0, so λ_min(PS) ≥ 1.
   - **Fields zero, p = 2:** 7 iterations (κ 1.10) at 128 and at 2,048 cells. Element Schwarz in the same place takes 9 → 20.
   - **Fields on, production mesh, p = 2, a = (6.11, 4.15):**

     | smoother | L0 (880 cells) | L1 (14,080 cells, 529,488 dofs) |
     |---|---|---|
     | sweep, Chebyshev degree 8, interval ratio 120 | 8 (κ 1.22) | 8 (κ 1.23) |
     | sweep, Chebyshev degree 4, ratio 20 | 8 (κ 1.37) | 10 (κ 1.97) |
     | element Schwarz, degree 4, ratio 20 | 8 | 20 (κ 4.7) |
     | production assembled GMG | 27–28 | 64 |

   - **With fields on, the upwind graph is one strongly connected component.** One time cell per slab, and the
     phase-space rotation closes within the layer. A block sweep over components is therefore impossible.
   - Lagging the weakest couplings |β·n|/h_n breaks the cycles cheaply: 378 faces, 0.5% of the coupling weight, and
     κ(PS) goes from 769 to 109.
   - Merging cells across faces where β·n changes sign did not help.
   - A t–x-only sweep, with the velocity faces block-diagonal, is worse than Jacobi (κ 1,113 against 425). The slow modes
     involve the velocity directions.
5. **The sweep degrades with the cell Courant number ν = |v_x| Δt / h_x.**
   - Alone, κ(PS) tracks the largest ν: 5 at ν ≈ 4, 13 at ν ≈ 15, 1,059 at ν ≈ 1,900, 6,036 at ν ≈ 15,000.
   - On a deeply graded stress mesh (16 graded x-elements, h_x,min = 2^-15, ν up to 15,000), the sweep smoother needs 107
     iterations against 17 for element Schwarz.
   - Production cathode cells have ν ≈ 13, so production meshes are fine.
   - In a cell with ν ≫ 1 almost every characteristic enters through the x-face. The bottom t-face is then nearly
     redundant inflow data that the local polynomial cannot reconcile.
6. **Cost, and what is not done yet.**
   - **Matrix-free memory is 7–15× lower than production.** At p = 4 with 128 cells, matrix-free GMG took 55–62 s and
     0.17–0.38 GB, against about 18 s and 2.66 GB for production with Serendipity test.
   - At 2,048 cells, p = 4, matrix-free fits in 1.2–5.4 GB. Production would need an estimated 14–40 GB.
   - The sweep is serial over cells. Setup is dominated by GMG's prolongation and the coarse hierarchy.
   - Not yet integrated into `VlasovSpaceTimeFormulation::solve()`. Everything runs as driver checks on one slab's
     operator: `--checkMatrixFreeOperator`, `--matrixFreeCheckSlab=N`, `--matrixFreeSweepSmoother`,
     `--matrixFreeCoarseGMG`, `--matrixFreeSmootherVariants`, `--matrixFreeSlowModeDiagnose`, and others.
   - The coarse h-hierarchy below p = 1 is Camellia's GMG on **uniform** refinements (`numUniformRefinements`). It has not
     been tried on adaptively or anisotropically refined meshes.

## 4. The test norm, and per-cell scaling

- **Today's production norm**, "graph⊕" with the transport-scale weight:
  α_K = 1e2 + 1.6[(v_x,max/h_x)² + 1/h_t² + Σ_u (a_u/h_u)²], with box-wide acceleration scales a = (1.8, 0.65). The
  same a_u also weight ∂_u v.
  - h_x, h_t and h_u are the cell's widths per direction, so on anisotropic cells the weight adapts direction by
    direction.
- **The norm trades one solver against another.**
  - A large transport-scaled α_K makes S nearly mass-like and good for Jacobi, but makes the local problems non-causal,
    which is bad for the sweep.
  - Fixed α = 100 is the reverse: on the stress mesh it gives sweep κ 6.2 but Jacobi κ 415,000.
  - On the production mesh today's transport-scaled α_K is the best of those tried for both solvers.
  - A "marching-compatible" variant, which moves the x and t parts of α_K to outflow-face terms
    (`--L2WeightGraphOutflowFaceTheta`; `marching-compatible-separable-norm.tex`), helped neither solver.
- **Wider box-wide acceleration scales help the conditioning.** a = (6.11, 4.15) is the bound from the conditioning
  study.

  | measure | a = (1.8, 0.65) | a = (6.11, 4.15) |
  |---|---|---|
  | Jacobi κ, L0, p = 2 | 425 | 156 |
  | Jacobi κ, L0, p = 4 | 1,141 | 329 |
  | sweep κ(PS) | 109 | 43 |

  - The gain is entirely at the low end of the spectrum: λ_min rises.
  - The TT session confirms the change with converged Lanczos on the full tensor-product trial space: 1,079 → 406 for
    them, 857 → 372 for us.
- **Per-cell acceleration scales.**
  - Definition: a_u(K) = max over the cell's quadrature points of |(q/m) a_u|. The floor is f × the box-wide value. They
    are re-evaluated every substep.
  - Options: `--L2WeightGraphPerCellAccelerationFloor=f`, `--L2WeightGraphPerCellAccelerationNeighborhood=0|1|2` (also take the
    max over face neighbours), and `--L2WeightGraphPerCellAccelerationMixedSignFactor=σ`.
  - Solvers: little effect compared with the wide box-wide a. Jacobi is 9% more iterations at L1, element Schwarz
    unchanged, and the sweep smoother needs Chebyshev degree 6–8 instead of 4, which does not grow under refinement.
- **Accuracy: the coupled cathode/anode negative-density mode tests**, 25–30 slabs, L0. The measure is the integral of
  the negative part of n_e over x ≥ 0.5 at slab end.
  - **p = 4.** Production a = (1.8, 0.65) grows the anode mode 1.6–2.4× per slab from slab 14 and has failed by slab 20,
    with negative gap charge and a 23% charge-balance error.
  - Per-cell scales, per-cell with floor 1.0, and box-wide (6.11, 4.15) all hold. They peak at −0.07, −0.074 and −0.088,
    with growth falling to about 1.0× per slab.
  - **p = 3.** Everything is bounded. Box-wide (6.11, 4.15) is the best (peak −0.0007), then production (−0.0016). The
    per-cell variants are 15–30× worse. At p = 3 the cell-to-cell variation of the weight costs accuracy.
  - **Our recommendation: box-wide (6.11, 4.15)**, which is best at p = 3, holds at p = 4, needs the fewest CG
    iterations, and keeps the norm fixed within a slab.
  - **Caveat:** L0 is one level coarser than production. The local (TT) session found production's weight stable at L1,
    p = 4, so the p = 4 failure may be specific to L0, where the anode cell has h_x = 1/2. The cathode mode never
    appeared at L0.
- **Diagonal approximations (for the TT side, but general).**
  - A rank-one Kronecker (log-additive) fit of the exact Jacobi diagonal costs 4.5–8.9× in κ. A TT rank-4 diagonal comes
    within 9–25%.
  - A fit must control the entrywise ratio d_fit/d, not a Frobenius error. The Frobenius-best rank one costs 160–220×.
  - Script: `notes/scripts/fit_diagonal.py`. Driver option: `--matrixFreeDiagonalFit`.

## 5. What this may mean for anisotropic refinement (our reading, not tested)

- **The cell Courant number is the one geometric quantity that hurt a solver.** Refining x alone near the cathode under
  long time cells raises ν = |v_x| Δt / h_x per cell, which degrades the sweep. The local inflow problem may also be
  harder to resolve in such cells.
  - If anisotropic refinement picks x near the cathode, it may be worth refining t with it, or capping ν, there.
  - In the velocity directions the analogue is ν_u ≈ |β_u| Δt / h_u, which is large where the acceleration is large.
- **The transport-scale weight already adapts to anisotropic cells**, since each direction's term uses that direction's
  h. Refining u raises (a_u/h_u)², and with it α_K, in those cells.
- **Per-cell weight variation cost accuracy at p = 3.** If refinement makes neighbouring cells very different in size,
  α_K varies strongly from cell to cell too. Whether that matters as the per-cell a_u(K) did is untested.
- **The matrix-free operator handles hanging faces and anisotropic p** (tested; Section 2). Three things are not
  tested on adaptive meshes:
  - the upwind cell ordering and transport sweep with hanging faces;
  - the GMG coarse hierarchy;
  - the Schwarz blocks.
- **Serendipity at p = 2 is small:** 5 field functions per cell (total degree ≤ 1, not Q₁) and 7 per face. The full
  tensor-product trial space is 16 and 27. Going from Serendipity to the full tensor-product trial space doubles the Jacobi κ at L0 (425 → 857). It is worth knowing if
  direction selection compares error indicators across trial spaces.

## 6. Open items on our side

1. Integrate the matrix-free solve into `VlasovSpaceTimeFormulation::solve()`, then MPI. The sweep needs a parallel
   ordering, such as KBA pipelining or lagging across subdomains.
2. A hybrid smoother: the sweep where ν is moderate, element blocks in high-ν cells.
3. An L1 confirmation, on the local side, of the p = 4 mode-test ranking.
4. The assembled-path quadrature error at enrichment 0 (Section 2).

## Notes on the branch, in order

| note | contents |
|---|---|
| `separable-gram-fast-diagonalization.md` | fast inverse, matrix-free operator, Jacobi, two-level, full hierarchy, p = 4 against production |
| `static-condensation-trace-structure.tex` | the sweep factorization S = R^T W R + Q, cycles, why κ ~ h^-2 |
| `sweep-update-2026-11-19.md` | sweep smoother h-independence, the Courant-number finding |
| `marching-compatible-separable-norm.tex`, `variantA-update-2026-11-20.md` | outflow-face norm variant: no solver benefit; the α_K trade-off |
| `tx-sweep-result-2026-11-20.md` | t–x-only sweep: worse than Jacobi |
| `acceleration-scales-kappa-cloud-2026-11-21.md`, `trial-space-finding-2026-11-21.md`, `reply-accscale-resolution-2026-11-21.md` | effect of a on κ; trial-space comparison with TT |
| `l1-multigrid-and-p4-trial-space-2026-11-21.md` | L1 multigrid with fields on, slow-mode locations |
| `kronecker-fit-and-percell-acceleration-2026-11-21.md` | diagonal fits; per-cell a_u(K) and conditioning |
| `percell-mode-tests-p3-2026-11-22.md` | coupled mode tests, p = 3 and p = 4 (final) |

Earlier notes supersede some numbers: three of them compared our p = 4 κ with an unconverged TT value. Where
`reply-accscale-resolution-2026-11-21.md` corrects them, it is right.
