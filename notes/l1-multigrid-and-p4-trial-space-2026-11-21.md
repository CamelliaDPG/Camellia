# L1 multigrid with the fields on, and the p = 4 trial-space run (cloud side, 2026-11-21)

Follows `trial-space-finding-2026-11-21.md`, Section 4 ("Pending on the cloud side").

**Summary.**
1. **p = 4 with the full tensor-product trial space at L0 does not fit on the cloud side.** Both runs (a = (1.8, 0.65) and
   (6.11, 4.15)) were killed for memory 40 s in, at 14 GB RSS of 15 GB. The setup needs the assembled production solve
   first, and at p = 4 that is about 810,000 dofs with 1,256 trial dofs per cell. So the trial-space question stands on
   the p = 2 result: the trial space explains the baseline, not the response to a. The mesh is still the open
   difference, and your L0 run decides it.
2. **The transport sweep as a multigrid smoother is close to mesh-independent with the fields on.**
   - Chebyshev degree 4 over [λ_max/20, λ_max]: 8 iterations at L0 and 10 at L1.
   - Degree 8 over [λ_max/120, λ_max]: 8 at both.
   - Element Schwarz (degree 4, ratio 20) goes from 8 to 20.

   The 42 iterations for the sweep at L1 in `acceleration-scales-kappa-cloud-2026-11-21.md` came from the merged-block
   bug, as already corrected there.

## 1. Multigrid iterations, L0 against L1

**Setting.**
- Production mesh family, p = 2, Δp = 2, Serendipity trial space, full tensor-product test space.
- Separable graph⊕ norm, α = 100, C = 1.6, a = (6.11, 4.15). Fields on: the operator after the slab-0 solve.
- Matrix-free GMG V-cycle as a PCG preconditioner, relative residual 1e-10.
- The smoother is Chebyshev over [λ_max/ratio, λ_max], applied to either:
  - the plain cell-by-cell transport sweep with weighted cycle breaking; or
  - element Schwarz.
- κ comes from the CG Lanczos tridiagonal.
- The slowest error mode is from power iteration on I − M⁻¹S. Its reduction is per V-cycle.

| smoother (kind:degree:ratio) | L0 (880 cells, 37,146 dofs): iterations (κ), slowest-mode reduction | L1 (14,080 cells, 529,488 dofs): iterations (κ), slowest-mode reduction |
|---|---|---|
| sweep:4:20 | 8 (1.37), 0.36 | **10 (1.97), 0.53** |
| sweep:4:120 | 13 (2.39), 0.58 | 13 (2.42), 0.58 |
| sweep:8:120 | 8 (1.22), 0.19 | **8 (1.23), 0.19** |
| element Schwarz, es:4:20 | 8 (1.41), 0.32 | 20 (4.7), from an earlier run |
| *production assembled GMG-PCG (for reference)* | *27–28 (κ ≈ 15)* | *64 (κ ≈ 57)* |

**Caveats.**
- L0 was checked after 2 substeps and L1 after 1. The L1 element-Schwarz number is from an earlier run in the same state
  as the L1 sweep runs.
- L1 refines t as well: there are 2 time cells per slab.
- The coarse solve was exact at L0. At L1 there are two coarse GMG levels below p = 1.

**Cost per run** (at L1, setup + solve):
- sweep:4:20: 784 s + 637 s;
- sweep:8:120: 561 s + 884 s.

Each single-threaded sweep is serial over 14,080 cells, so this does not yet say anything about parallel cost.

## 2. Where the slowest mode lives

Each mode is split by dof class (|x|²).

| variant | fields | t-faces | u_x-faces | u_y-faces | x-faces | dominant cells |
|---|---|---|---|---|---|---|
| sweep:4:20, L1 | 0 | 48% | 22% | 29% | 2% | spread out (no cell above 13%): slow particles (\|v_x\| ≲ 0.1–0.7) at x ≈ 0.03–0.3, mostly in the upper time cell |
| sweep:4:120, L1 | 0 | 1% | **74%** | 25% | 0% | the thinnest cathode cells (h_x = 0.004), \|β_{u_y}\| ≈ 2–4 |
| sweep:8:120, L1 | 0 | 12% | **72%** | 16% | 0% | the same cathode cells, at \|u_x\| ≈ 2–8 and \|β_{u_y}\| ≈ 3.4–4.1 |

**Reading.**
- With the wide Chebyshev interval (ratio 120), what is left is the acceleration-dominated cathode family: u_x-face
  fluxes in cells where the transverse acceleration β_{u_y} is large. This is the same family that limited the sweep at
  L0 with a = (1.8, 0.65), now weaker.
- With the narrow interval (ratio 20), the cathode mode is damped. What remains is a smooth, spread-out error along t in
  the slow-particle region. Its reduction got worse from L0 (0.36) to L1 (0.53), probably because L1 has two time cells
  per slab and the coarse correction carries less of that error.
- Raising the Chebyshev degree to 8 over the wide interval damps both, at about 40% more solve time.

## 3. What this means for the TT side

- **No new data on the κ disagreement beyond p = 2.** The full tensor-product p = 4 point needs more memory than the
  cloud container has. Your converged-Lanczos note (`accscale-resolution-2026-11-21`) has since resolved the
  disagreement, so this point is no longer needed.
  - The comparison this section originally made, "Serendipity p = 4 at L0 (1,141) agreed with your 1,294 at L1", used
    your unconverged 100-step number. Drop it.
  - At converged Lanczos, your full tensor-product L0, p = 2 values (1,079 → 406) and ours (857 → 372) agree to about
    20% on the baseline and closely on the change.
- **For any per-cell a_u(K) work:** the slow modes that remain sit on u_x-face fluxes in the thinnest cathode cells,
  where |β_{u_y}| ≈ 4. Per-cell acceleration scales would act directly on those cells. Their effect on the
  Jacobi-scaled κ is still the open question, and your L0 run will show whether it depends on the mesh.
