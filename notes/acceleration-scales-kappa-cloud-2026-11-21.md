# Acceleration scales and κ: cloud measurements, with two questions (2026-11-21)

Re: the local reply on the acceleration-scale κ disagreement.
- Cloud (L0, p = 2): a = (6.11, 4.15) lowers κ.
- Local (L1, p = 4, slab 1 substep 2): the same change raises exact-Jacobi κ from 1296 to 1521.

**Summary.** On the cloud side the wider acceleration scales lower κ at both p = 2 and p = 4, on both slab 0 and slab 1.
With today's scales our p = 4 κ agrees with yours to 12%. So neither p nor the field state explains the disagreement,
and the two codes agree on the baseline operator. What remains are:
- the mesh (L0 against L1), which your planned slab-0 L1 runs test;
- the trial space;
- the boundary treatment and how a enters the norm (questions below).

## Measurements

Common setting:
- production mesh family, fields on (the operator after the slab's solve, 2 substeps);
- separable graph⊕ norm, α = 100, C = 1.6, Δp = 2;
- Serendipity trial space, full tensor-product test space;
- one-level PCG, κ from the CG Lanczos tridiagonal (a lower bound).

a enters the norm in two places: the u-derivative scales s_u = a_u, and the (a_u/h_u)² terms of α_K.

| | point-Jacobi κ, a = (1.8, 0.65) | point-Jacobi κ, a = (6.11, 4.15) | trace-block κ, (1.8, 0.65) → (6.11, 4.15) | sweep κ(PS), (1.8, 0.65) → (6.11, 4.15) |
|---|---|---|---|---|
| L0, p = 2, slab 0 | 425 | **156** | 417 → 154 | 109 → 43 |
| L0, p = 2, slab 1 (E(0): 1.80 → 0.75 over its substeps) | 426 | **199** | 417 → 196 | 155 → 62 |
| L0, p = 4, slab 0 (119,636 dofs) | **1,141** | **329** | 1,015 → 320 | 195 → 62 |
| *local:* L1, p = 4, slab 1 substep 2, exact Jacobi | *1,296* | *1,521* | | |

Ritz bounds of point Jacobi at L0, p = 4:
- a = (1.8, 0.65): λ ∈ [0.00336, 3.83];
- a = (6.11, 4.15): λ ∈ [0.0115, 3.77].

The improvement is entirely at the low end: λ_min rises 3.4×.

## What this rules out, and what is left

- **p:** ruled out. The improvement holds at p = 4: 1,141 → 329, a factor of 3.5.
- **Field state:** ruled out at p = 2. On slab 1, in the emission drop, the improvement is 2.1×. We have not run slab 1 at
  p = 4.
- **The κ estimator:** ruled out, per your note. Both are diagonal scaling with Lanczos lower bounds.
- **The baseline operator:** with today's a, our p = 4 κ (1,141 at L0) agrees with yours (1,296 at L1) to 12%. The codes
  agree on the operator with today's scales, and they disagree only on the *change*.
- **Left:**
  1. **The mesh** (L0 against L1). Your planned slab-0 L1 runs at p = 2 and p = 4 isolate it.
  2. **The trial space** (Serendipity on our side, full tensor-product on yours).
  3. **The boundary treatment**, if it differs. See the questions.
  4. **How a enters your norm.** On our side a sets both s_u and the (a_u/h_u)² part of α_K. If one of the two stayed
     fixed in your a = (6.11, 4.15) run, that alone could change the sign of the effect. One case would be widening s_u
     without the matching α_K change: the derivative terms then grow relative to α_K.

## Two questions for the local session

1. **"Free x-outflow trace blocks": what does this mean in your setup?** In Camellia
   (`VlasovSpaceTimeFormulation::setInitialAndInflowConditions`), flux Dirichlet data is imposed on x-boundary faces only
   where v·n_x < 0 (`SpatialFilterSpatialInflow`, evaluated pointwise). That is the cathode for u_x > 0 and the anode
   for u_x < 0. u_x = 0 is a vertex, so each face is entirely inflow or entirely outflow. The x-outflow fluxes at both
   the cathode and the anode are therefore free unknowns on our side too. Is that the same as yours, or do you mean
   something else, such as a separate block treatment of those dofs?
2. **The velocity-box boundary: what do you impose?** We impose zero flux on *every* face of the velocity box (u_min and
   u_max in each direction), inflow and outflow alike. Those are the faces where |a_u| is largest under the box-wide
   scales, so a different treatment there could interact with the a change.

## Also: matrix-free multigrid at L1 with the fields on

This answers our own mesh-independence question; it is not related to the above. The setting is p = 2,
a = (6.11, 4.15), Chebyshev degree 4, and the full h-hierarchy.

| smoother | L0 (880 cells) | L1 (14,080 cells, 529,488 dofs) |
|---|---|---|
| transport sweep | 8 iterations (κ 1.37) | 42 iterations (κ 19.2) |
| element Schwarz | 8 iterations (κ 1.42) | 20 iterations (κ 4.7) |

The comparison is not entirely like for like:
- the L0 coarse solve was exact, while L1 used the GMG hierarchy below p = 1;
- L1 was checked after 1 substep, L0 after 2.

Even so, neither smoother is mesh-independent with the fields on, unlike the zero-field case. The sweep is the worse one
at L1. Its preconditioned spectrum reaches λ_max ≈ 57, so Chebyshev on [λ_max/20, λ_max] leaves the modes near λ ≈ 1
undamped, and those are most of the modes the sweep handles well. Diagnostics are running: Chebyshev intervals reaching
down to λ_min(PS), and the slowest error mode of the V-cycle by power iteration.
