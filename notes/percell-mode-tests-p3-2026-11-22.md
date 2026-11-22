# Per-cell acceleration scales: anode and cathode mode tests at L0, p = 3 (cloud side, 2026-11-22)

Re: `answers-mode-tests-2026-11-22.md`. This covers the p = 3 pair. The p = 4 pair is still running (see the end).

**Summary.**
- **Cathode mode: none in either run.** ∫n⁻ over the cathode cells is −0.02 at slabs 1–2 (start-up), then zero from slab 4
  to slab 29. As you predicted, L0 is probably too coarse to show it, so this says little.
- **Anode side: per-cell passes your criterion but is clearly worse than box-wide.** In both runs a negative density
  forms near the anode from slab 18–21, peaks at slabs 25–26, and decays. With per-cell scales it starts three slabs
  earlier and its peak is about 24× larger (∫n⁻ −0.039 against −0.0016; min n over the slab −0.17 against −0.03).
- Growth per slab with per-cell scales: 9×, 2.9×, 1.9×, 1.5×, then down to 1.0× by slab 26. That is not sustained 3–4×
  growth, so it is bounded. Its peak is of the size of your L1 p = 3 baselines at slab 19 (graph −0.024, separable
  −0.030).
- The charge-balance error is 5× larger with per-cell scales in slabs 18–29 (0.003–0.0075 against 0.0003–0.0015).
- Emission, E(0) and f min are the same in both runs. ∫f⁻ levels off in both, at −0.59 per-cell against −0.52 base at slab 29.

## Setting

- L0 mesh: `--xElements=8 --numUniformRefinements=0`, h_x from 1/128 at the cathode to 1/2 at the anode (the last
  cell is x ∈ [0.5, 1]), 11 × 10 graded velocity cells, 880 cells per slab, h_t = 0.1.
  - It is one level coarser than your L1 mode tests. An L1 run does not fit in our container: the first substep of L1,
    p = 3 on 2 ranks did not finish in 40 min.
- p = 3, Δp = 2, Serendipity, 30 slabs of 0.1, 2–10 substeps, fixed-point tolerance 1e-4, J = 5e8, SCL emission. One
  rank per run.
- Graph norm, α = 1e2 + 1.6[(v_x,max/h_x)² + 1/h_t² + Σ_u (a_u/h_u)²].
  - base: a = (1.8, 0.65) box-wide.
  - per-cell: a_u(K) = max(max over the cell's quadrature points of |(q/m) a_u|, 0.1 × (1.8, 0.65)), re-evaluated
    every substep. Overall factor 1.0, as you suggested.
- The measure is yours: the slab-end ∫n⁻ over x, by region (trapezoid rule on each cell's density-diagnostic points).
  - Cathode cells: x < 0.0234. Anode side: x ≥ 0.5.
  - Script: `notes/scripts/mode_summary.py`.

## Anode side (x ≥ 0.5), slab end

| slab | base ∫n⁻ | base min n (whole gap) | per-cell ∫n⁻ | per-cell min n (whole gap) | per-cell growth of ∫n⁻ |
|---|---|---|---|---|---|
| 3 | −0.0001 | −2.4 | −0.0021 | −2.1 | |
| 4–9 | 0 | > 0 | −0.0019 to −0.0043 | −0.017 to −0.029 | (start-up transient) |
| 10 | 0 | 0.047 | −0.0003 | −0.0075 | |
| 11–17 | 0 | 0.05–0.08 | 0 | 0.006–0.026 | |
| 18 | 0 | 0.064 | −0.00027 | −0.012 | |
| 19 | 0 | 0.041 | −0.0025 | −0.035 | 9.2× |
| 20 | 0 | 0.015 | −0.0073 | −0.058 | 2.9× |
| 21 | −0.00015 | −0.009 | −0.0142 | −0.076 | 1.9× |
| 22 | −0.00065 | −0.024 | −0.0218 | −0.095 | 1.5× |
| 23 | −0.0012 | −0.029 | −0.0288 | −0.115 | 1.3× |
| 24 | −0.0016 | −0.023 | −0.0344 | −0.133 | 1.2× |
| 25 | −0.0016 | −0.016 | −0.0378 | −0.148 | 1.1× |
| 26 | −0.0016 | −0.016 | **−0.0386** | −0.156 | 1.02× |
| 27 | −0.0016 | −0.016 | −0.0370 | −0.155 | 0.96× |
| 28 | −0.0010 | −0.013 | −0.0332 | −0.146 | 0.90× |
| 29 | −0.00007 | −0.002 | −0.0281 | −0.131 | 0.85× |

- The minimum of n over the slab sits at x ≈ 0.83–1.0 from slab 21 on in both runs, in the last (largest) cell next to
  the anode. That is where your anode mode lived (at L1 the cells x = 0.5–1.0).
- The gap integral ∫n is the same to 3% in both runs (2.32 per-cell against 2.37 base at slab 29).

## Other diagnostics, slabs 18–29

| | base | per-cell |
|---|---|---|
| emitted density / n0 | 0.024–0.031 | 0.021–0.031 |
| E(0)/E0 | within ±0.0008 | within ±0.0009 |
| f min | −87 to −110 | −78 to −110 |
| ∫f⁻ | −0.62 → −0.52 | −0.74 → −0.59 |
| charge balance, relative | 0.0001–0.0015 | 0.0026–0.0075 |

## Reading

- By your criterion (bounded or decaying, no sustained 3–4×) per-cell scales pass at L0, p = 3.
- At the same p, though, they make the late anode-side negative density much larger than the production box-wide
  weight does. That matches your observation that every stability fix came from more weight.
- **Our guess at the cause, which we have not checked:** in the large anode cells the per-cell a_u(K) is below the
  box-wide (1.8, 0.65), so α_K there is smaller than in production.
  - At slab 0, L0 (in the p = 2 conditioning study), a_y(K) ranged down to 0.09, against 0.65 box-wide. We have not
    checked where those cells are.
  - That would make factor 0.5 worse, not better. We have not run 0.5.
- **To test the guess, we are running two variants** (L0, p = 3, 30 slabs):
  - **floor 1.0:** a_u(K) = max(own maximum, the box-wide bound). No cell gets less weight than in production; cells
    with stronger fields get more.
  - **neighborhood maximum** (`--L2WeightGraphPerCellAccelerationNeighborhood=2`, floor 0.1): the maximum over the cell
    and all its face neighbors.
  - If floor 1.0 removes the difference, the per-cell weight should never fall below the box-wide one. Then the TT side
    would need a box-wide floor as well.
- **For your L1 p = 4 per-cell run on the Mac,** the anode-side ∫n⁻ in slabs 11–25 against the 11-06 run is the number
  to watch. That run peaked at 0.02 at slab 9 and decayed to 0.002.

## Operational notes for anyone running these on few cores

- `mpirun -np 1` binds the rank to core 0 by default. Several concurrent one-rank runs then share one core: two runs get
  50% each, four get 25%.
  - Use `--bind-to none`, or pin each run with `taskset`.
  - Our earlier per-slab times (11.5 min at L0, p = 3 with two runs) were half speed for this reason. With one core
    each: about 118 s per substep at p = 3 and about 320 s at p = 4.
- `--vlasovCellBatchSize=2` keeps a run under 2 GB (L0, p = 3 and 4), against about 6 GB with the automatic batch.
  - Before commit 8adfdb7 it applied only to slab 0, because `nextTimeSlab()` builds new solutions.

## Still running

- **p = 4 pair** (L0, 25 slabs, base and per-cell floor 0.1): at slab 6 of 25, about 43 min per slab. Base and per-cell
  agree so far.
- **The two p = 3 variants** above: about 8 hours.
