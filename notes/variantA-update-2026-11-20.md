# Variant A of the marching-compatible norm: implementation and first measurements (cloud side, 2026-11-20)

This update follows `marching-compatible-separable-norm.pdf` (version 2, with the review corrections). It covers what was
built and what the first measurements show. **Headline:** at the review's weight μ = θ s²/h with θ = 1.6, the
outflow-face terms make conditioning worse on the deep-cathode test mesh. Shrinking θ helps the transport sweep but
hurts κ(S) with Jacobi, which is the number that matters for AMEn. So far, nothing beats simply removing the x and t parts
of α_K.

## 1. What is implemented (branch `separable-gram-fast-diagonalization`, commit a4906b2)

**`SeparableTensorIP::addOutflowFaceTerm(direction, weight, side)`**
- It adds weight(K) ∫_F v², where F is the face of K normal to `direction`. The upper face is used where the cell average
  of `side` is positive, the lower face where it is negative.
- `weight` must be constant on each cell.
- It is assembled from 1D factors: the endpoint mass matrix E_± in that direction, and the scaled mass matrices in the
  other directions. It works for full-tensor and Serendipity test bases.
- Where the 1D path doesn't apply, `computeInnerProductMatrix` throws rather than silently dropping the term.

**Fast inverse**
- The stiffness in that direction becomes K̂ + r E_±, with r = (weight/(h/2)) / b and b the direction's derivative weight.
- One `SeparableGramInverse` is cached per distinct pattern of (side, r), with r rounded to 12 digits.
- With weight = θ s²/h, r = θ/2 in every cell, so there are exactly two eigenbases: x-outflow up and x-outflow down.
  This is as the review predicted.
- The Sherman–Morrison constant-mode path now uses the per-cell inverse.

**`RelativisticVlasovSpaceTimeFormulation(..., testSpaceOutflowFaceTheta)`, driver flag `--L2WeightGraphOutflowFaceTheta=θ`**
- Requires the separable norm and the transport-scale weight. θ = 0 (the default) leaves today's norm unchanged.
- The L² weight becomes α'_K = α + C Σ_u (a_u/h_u)². The x- and t-terms are removed.
- Outflow-face terms are added with weight θ s_d²/h_d:
  - t: always the upper face;
  - x_d: the face that the sign of u_d points to, with s_x = v_x,max(K).
- Variant B (u-direction outflow terms) is not implemented.

**Unit test `SeparableTensorIP.OutflowFaceTerms`**
- The 1D-factor Gram matrix matches an independent reference to 2.5e-15. The reference is an ordinary IP whose boundary
  term is restricted to the same faces by an indicator weight on the side caches, integrated by quadrature.
- It is tested on hierarchical and Serendipity bases, with isotropic and anisotropic orders.
- The face terms make up to 9% of the largest Gram entry.
- The fast inverse has backward error 7e-16 on every cell.
- The other 8 separable and matrix-free tests still pass.

## 2. First measurements: deep-cathode mesh, fields zero

**Setup.**
- The mesh has 16 geometrically graded x-elements (h_x,min = 2^-15), t × u_x × u_y = 2 × 4 × 4, and 512 cells.
- p = 2, Δp = 2, Serendipity trial, full-tensor test.
- α = 100, C = 1.6, a = (1.8, 0.65).
- The operator is the first slab before its solve, so the fields are zero.
- One-level PCG to 1e-10 tolerance, with κ from Lanczos.
- This is the mesh where the transport-scaled α_K broke the sweep. It is not a production mesh: production cathode cells
  have ν ≈ 13, not ~15,000.

| Test norm | Sweep κ(PS) | Point-Jacobi κ | Trace-block-Jacobi κ |
|---|---|---|---|
| Current: transport-scaled α_K | 6,036 | 3,089 | 3,050 |
| Variant A, θ = 1.6 | 34,871 | 4,275 | 3,507 |
| Variant A, θ = 0.1 | 2,576 | 7,069 | 7,042 |
| Variant A, θ = 0.01 | 287 | 9,486 | 9,500 |
| Variant A, θ = 0.001 | 32.8 | 4,736 | 4,743 |
| **Fixed α = 100**, no x/t transport terms, no outflow terms | **6.2** | **415,010** | **415,581** |

On this mesh α'_K ≈ 100, because the (a_u/h_u)² terms are small with 4 wide velocity cells. So variant A differs from the
fixed-α run only by the outflow-face terms.

**What this says**
1. **At θ = 1.6 the outflow terms are harmful.** They raise the sweep's κ about 6× and Jacobi's somewhat. The gain in the
   fixed-α run came from *removing* the x and t parts of α_K, not from anything the outflow terms add.
2. **θ pulls the two measures in opposite directions.**
   - Smaller θ brings the sweep toward the fixed-α result (κ 34,871 → 2,576 → 287).
   - Smaller θ makes κ(S) with Jacobi worse (κ 4,275 → 7,069 → 9,486).
   - My reading: with little L² weight in x and t and a weak outflow term, the constant and outflow-trace modes of the
     test space are weakly controlled. That shows up as small eigenvalues of S. A sweep, built from exact local solves,
     absorbs them; Jacobi does not.
3. **Why a strong outflow term hurts (hypothesis).** A large μ pins the discrete test functions near zero on the outflow
   faces. Outflow fluxes then barely register in the residual: their dual-norm energy scales like 1/μ. This is severe in
   thin cells, where μ = θ s_x²/h_x is about 1e9 θ.
   - The review's point that the weight is free holds for the ideal test space, not the discrete one.
   - (1.1)'s own boundary term is small relative to its derivative term (ratio ~ h/|b|²). That is closer to the original
     μ ≈ |β| than to θ s²/h.
   - So the dimensional correction is right, but μ ~ s²/h appears to set the strength too high.
4. **For the TT/AMEn question, the relevant column is the Jacobi one.** No variant-A θ tried improves κ(S) with Jacobi on
   this mesh.

**Update: the fixed-α baseline is in, and it reverses the picture for Jacobi.** The norm the sweep likes best
(fixed α = 100, κ(PS) = 6.2) gives κ(S) ≈ 415,000 with point or trace-block Jacobi. That is 134× worse than today's
transport-scaled α_K (3,089). So there is a clear trade-off:
- **A large, transport-scaled L² weight** (α_K ~ C(|β|/h)²) makes S nearly mass-like and well scaled from cell to cell,
  which is good for Jacobi and AMEn. It also makes the local problems non-causal, which is bad for the sweep.
- **A small L² weight** leaves S as the normal equations of transport. Its smallest eigenvalues belong to modes smooth
  along characteristics, which the sweep removes exactly and Jacobi cannot.
- Variant A at the θ values tried sits in between and is not better than today's norm for either solver.

**Consequence for the TT side:** for a Jacobi-preconditioned TT solve, today's transport-scaled α_K is the better norm of
those tried here. Weakening the x and t parts of α_K, by moving them to faces or removing them, would make κ(S̃) worse,
not better. Variant A should not be expected to help the p = 4 AMEn problem. If anything, the measurements argue for
keeping or strengthening α_K's scaling there. Caveat: this is the deep-cathode stress mesh with zero fields. The
production L0 version of this table is next.

## 3. Other results since the proposal (production mesh family, p = 2)

These are L0 runs (880 cells), with fields on: the check runs after slab 0's solve, with 2 substeps.

| | Sweep κ (alone) | Two-level iterations (κ), sweep smoother | Two-level iterations (κ), element-Schwarz smoother |
|---|---|---|---|
| a = (1.8, 0.65), plain sweep | 769 | 23 (9.7) | 10 (2.4) |
| a = (1.8, 0.65), weighted cycle breaking | 109 | 15 (2.8) | 10 (2.4) |
| a = (6.11, 4.15), weighted cycle breaking | 43 | **8 (1.37)** | **8 (1.42)** |

- **With the fields on, the upwind graph is one strongly connected component** (all 880 cells: one time cell per slab, so
  the phase-space circulation closes within the layer).
  - A block sweep over components is therefore impossible.
  - Lagging the weakest couplings |β·n|/h_n helps a lot: 378 faces are lagged, but only 0.5% of the coupling weight.
- **Merging cells across faces where β·n changes sign does not help.** The top mode is unchanged, and the extra lagging
  hurts λ_min.
- **The remaining top mode at a = (1.8, 0.65)** sits on u_x-face fluxes in the cathode cells, where |β_{u_x}| = 4–6
  exceeds the norm's scale of 1.8. The box-wide bracket removes it, so per-cell a_u(K) should too.
- **L1 (14,080 cells) with fields on and a = (6.11, 4.15)** is running. It is the mesh-independence check for both
  smoothers.

## 4. Next on the cloud side

1. θ = 0.001; κ(S) with Jacobi for the fixed-α norm on the same mesh.
2. The same table at production L0 with the fields on, with a = (6.11, 4.15). This is where the norm would actually be used.
3. L1 results when they finish.

Suggestions for the local side:
- If small θ is to be tried in coupled runs, it is the outflow term's stabilizing role, replacing the x and t parts of
  α_K, that needs testing. On these numbers the solver gives no reason to prefer θ s²/h with θ = O(1).
