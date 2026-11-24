# Per-cell acceleration scales: anode and cathode mode tests at L0, p = 3 (cloud side, 2026-11-22)

Re: `answers-mode-tests-2026-11-22.md`. All runs are finished: five at p = 3 (30 slabs) and four at p = 4 (25 slabs),
all on L0.

**Update, L1, p = 4 (local runs, 2026-11-24; full section at the end): at the production mesh, nothing fails.**
Production, box-wide (6.11, 4.15) and per-cell floor 1.0 all plateau and decay, with anode-side peaks of −0.022, −0.034
and −0.032. So the p = 4 failure in our L0 runs comes from the coarse L0 anode cell. Box-wide needs about 25% fewer CG
iterations than production at L1.

**Update, p = 4 follow-ups (finished, 2026-11-23): at p = 4 every variant with more weight holds, and production fails.**
Anode-side ∫n⁻ peaks at −0.070 (per-cell), −0.074 (per-cell, floor 1.0) and −0.088 (box-wide a = (6.11, 4.15), slab 23,
then decaying). Production reaches −8.4 by slab 23. See "p = 4 follow-ups" at the end.

**Update, p = 4 (finished): the production box-wide weight grows the anode mode at L0 and has failed by slab 20 (gap
∫n < 0 by slab 21). Per-cell scales hold through slab 24: anode-side ∫n⁻ grows ever more slowly, 1.19× per slab at
slab 19 down to 1.02× at slab 24, and peaks at −0.070.** At
p = 3 the order is the reverse (per-cell worse, both bounded). See "p = 4, interim" below.

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

## p = 4 (per-cell finished; base stopped after slab 23)

Same setting as above with p = 4 (25 slabs). Anode side (x ≥ 0.5), slab end:

| slab | 5 | 7 | 9 | 11 | 13 | 14 | 15 | 16 | 17 | 18 |
|---|---|---|---|---|---|---|---|---|---|---|
| base ∫n⁻ | −0.0026 | −0.0225 | −0.0177 | −0.0115 | −0.0108 | −0.0134 | −0.0211 | −0.0428 | −0.0983 | **−0.236** |
| base growth | | | | | | 1.24× | 1.6× | 2.0× | 2.3× | 2.4× |
| per-cell ∫n⁻ | −0.0035 | −0.0443 | −0.033 | −0.0243 | −0.0219 | −0.0227 | −0.0251 | −0.0287 | −0.0334 | −0.0398 |
| per-cell growth | | | | | | 1.04× | 1.1× | 1.15× | 1.16× | 1.19× |

| slab | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|
| base ∫n⁻ | −0.533 | −1.13 | −2.27 | −4.42 | −8.40 | (stopped) |
| base gap ∫n | 1.45 | 0.64 | −0.89 | −3.70 | −8.79 | |
| per-cell ∫n⁻ | −0.0475 | −0.0554 | −0.0618 | −0.0661 | −0.0685 | −0.0700 |
| per-cell growth | 1.19× | 1.17× | 1.12× | 1.07× | 1.04× | 1.02× |
| per-cell min n (whole slab) | −0.15 | −0.16 | −0.17 | −0.24 | −0.32 | −0.38 |

- Base at slab 23: min n −44, ∫f⁻ −47, charge-balance error −23%, emission 0.14 n0. We stopped it there.
- Per-cell through slab 24: gap ∫n 2.17–2.19, ∫f⁻ −0.73 to −0.79, charge balance 0.002–0.003, emission 0.024–0.032 n0.
  Its min n, at x = 0.5 from slab 22, still drifts down (−0.38 at slab 24) while ∫n⁻ levels off.

- **Base (production a = (1.8, 0.65) box-wide) has the anode mode from slab 14, and by slab 20 it has failed:**
  - gap ∫n 0.64 at slab 20, from 2.32 at slab 14; ∫f⁻ −8.4; charge-balance error −4.4%; CG iterations per substep
    rising to 80;
  - from slab 19 the minimum of n is at x = 0.5, so the negative density has spread past the anode cell.
- Its earlier signs, slabs 14–18:
  - rising growth, 2.4× per slab by slab 18;
  - min n −1.15 at x ≈ 0.87–0.93, in the anode cell;
  - ∫f⁻ accelerating, −1.27 → −3.02 over slabs 15–18;
  - the charge-balance error growing to −0.0097;
  - the gap's ∫n falling from 2.32 to 1.88.
- **Per-cell (floor 0.1, factor 1.0):**
  - ∫n⁻ about 2× base through slab 13, then slow growth of 1.1–1.2× per slab;
  - ∫f⁻ improving, −1.17 → −0.81;
  - charge balance about 0.002;
  - min n −0.13 at slab 18, now at x = 0.5, the face between the last two cells.
- Emission (0.028–0.035 n0) and E(0) (within ±0.0009 E0) are the same in both runs so far.
- **Why base fails here, though your L1 p = 4 run with the same weight was stable (our reading, not checked):**
  - The L0 anode cell has h_x = 1/2, against 1/4 at L1, so the weight's (v_x,max/h_x)² term there is 4× smaller.
  - That is the situation of your mesh-scale anode failure.
  - Per-cell scales put more weight where the fields exceed the box-wide bound.
- CG iterations per substep are similar in both runs (20–75) and not yet rising.

## The p = 3 variants (finished, 30 slabs)

Anode side (x ≥ 0.5) ∫n⁻ at slab end:

| slab | 18 | 19 | 20 | 21 | 22 | 24 | 26 | 28 | 29 | peak |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 0 | 0 | 0 | −0.00015 | −0.00065 | −0.0016 | −0.0016 | −0.0010 | −0.00007 | −0.0016 |
| per-cell | −0.00027 | −0.0025 | −0.0073 | −0.0142 | −0.0218 | −0.0344 | −0.0386 | −0.0332 | −0.0281 | −0.0386 (26) |
| per-cell, floor 1.0 | 0 | −0.00056 | −0.0032 | −0.0076 | −0.0125 | −0.0215 | −0.0246 | −0.0203 | −0.0160 | −0.0246 (26) |
| per-cell, neighborhood 2 | −0.0015 | −0.0049 | −0.0112 | −0.0196 | −0.0286 | −0.0439 | −0.0498 | −0.0428 | −0.0355 | −0.0498 (26) |

- All four are bounded: each peaks at slab 26 and decays.
- **Floor 1.0** starts one slab later than plain per-cell, and its peak is about 35% smaller. It is still 15× base. Its
  slab 4–9 transient is the same as plain per-cell's (−0.0036, −0.0039, −0.0022 at slabs 4–6).
  - So cells with a_u(K) below the box-wide bound explain only a small part of the p = 3 difference. The guess under
    "Reading" is mostly wrong.
- **The neighborhood maximum** is worse than plain per-cell: it starts at slab 15, and its peak is 30% larger.
- So at p = 3 the extra anode-side negative density comes from where per-cell scales exceed the box-wide ones, or from
  their variation between cells, not from cells where they are smaller.

## p = 3 with the larger box-wide bound a = (6.11, 4.15) (finished, 30 slabs)

The control for "more weight" without variation between cells. At p = 3 it is the cleanest of all five runs.

| slab | 1 | 2 | 4–7 | 8–23 | 24 | 25 | 26 | 27–29 |
|---|---|---|---|---|---|---|---|---|
| cathode ∫n⁻ | −0.0009 | −0.0033 | 0 | 0 | 0 | 0 | 0 | 0 |
| anode-side ∫n⁻ | 0 | −0.00015 | −0.0001 to −0.0007 | 0 | −0.00052 | −0.00069 | −0.00041 | 0 |
| slab-end min n | −3.5 | −2.3 | −0.010 to −0.001 | 0.004–0.25 | −0.031 | −0.041 | −0.024 | 0.014–0.12 |

- **The start-up undershoot at the cathode is 18× smaller** than with (1.8, 0.65): −0.0009 against −0.016 at slab 1.
- **The late anode-side negative density is the smallest of all p = 3 runs:** a peak of −0.0007, against −0.0016 base and
  −0.039 per-cell.
- The charge-balance error in slabs 18–29 is 0.0008–0.009, larger than base (0.0001–0.0015) and similar to per-cell.
- Mean CG iterations per substep: 41.7, against 46.7 base and 45.9 per-cell. Wall time 5:43 for 30 slabs, against 7:30
  for base (part of base ran with the shared-core binding, so this comparison is rough).

**So at p = 3 the amount of weight is not the problem.** Ranked by the anode-side peak:

| run | weight compared with production | peak anode-side ∫n⁻ |
|---|---|---|
| box-wide (6.11, 4.15) | more, everywhere | −0.0007 |
| box-wide (1.8, 0.65), production | same | −0.0016 |
| per-cell, floor 1.0 | never less | −0.025 |
| per-cell | less in some cells | −0.039 |
| per-cell, neighborhood maximum | less in some cells | −0.050 |

- Per-cell with floor 1.0 has at least production's weight in every cell, yet it is 15× worse than production.
- Per-cell values lie mostly between (1.8, 0.65) and (6.11, 4.15). At L0 slab 0, a_x(K) is 1.50–6.53 and a_y(K) is
  0.09–4.16.
- So at p = 3 what costs accuracy is the variation of the weight between cells, not its size.
- Where the variation does its damage we have not looked.

## p = 4 follow-ups (finished, 25 slabs)

Same setting as "p = 4" above. Anode side (x ≥ 0.5) ∫n⁻ at slab end, growth per slab in brackets:

| slab | 14 | 16 | 18 | 19 | 20 | 21 | 22 | 23 | 24 |
|---|---|---|---|---|---|---|---|---|---|
| base (production) | −0.0134 | −0.0428 | −0.236 | −0.533 | −1.13 | −2.27 | −4.42 | −8.40 | (stopped) |
| per-cell | −0.0227 | −0.0287 | −0.0398 | −0.0475 (1.19) | −0.0554 (1.17) | −0.0618 (1.12) | −0.0661 (1.07) | −0.0685 (1.04) | −0.0700 (1.02) |
| per-cell, floor 1.0 | −0.0228 | −0.0291 | −0.0403 | −0.0479 (1.19) | −0.0566 (1.18) | −0.0646 (1.14) | −0.0701 (1.09) | −0.0729 (1.04) | −0.0738 (1.01) |
| box-wide (6.11, 4.15) | −0.0258 | −0.0357 | −0.0542 | −0.0645 (1.19) | −0.0732 (1.13) | −0.0795 (1.09) | −0.0851 (1.07) | −0.0879 (1.03) | −0.0856 (0.97) |

Other diagnostics, slabs 17–24:

| run | min n over the slab (where) | ∫f⁻ at slab 24 | f min | charge rel. | CG its per substep, mean (max) | wall time |
|---|---|---|---|---|---|---|
| base | −44 at slab 23 | −47 (slab 23) | | −23% (slab 23) | 64.3 (113) | stopped after slab 23 |
| per-cell | −0.13 → −0.38, at x = 0.5 from slab 22 | −0.74 | −112 to −141 | 0.0019–0.0032 | 58.3 (102) | 13:25 |
| per-cell, floor 1.0 | −0.11 → −0.33, at x = 0.5 from slab 23 | −0.69 | −103 to −120 | 0.0001–0.0016 | 56.7 (102) | 12:22 |
| box-wide (6.11, 4.15) | −0.16 → −0.30 (slab 22) → −0.28, at x = 0.73–0.90 | −0.66 | −75 to −86 | −0.0021 to 0.0067 | 48.8 (91) | 13:13 |

- **All three hold where production fails.** Each one's growth falls steadily from 1.19× at slab 19 to about 1.0× at
  slab 24. That is far below production's 2–2.4× and your 3–4× fail threshold.
- **Floor 1.0 matches plain per-cell within 1–6%**, as at p = 3 (where it was the better of the two). Its charge-balance
  error and ∫f⁻ are a little better. At both p, the cells where per-cell scales fall below the box-wide bound matter little.
- **The larger box-wide bound** has a peak 25% larger than per-cell, but it is the only run whose ∫n⁻ has turned over
  (slab 23 → 24). Its minimum of n stays inside the anode cell and stops falling at slab 22. With per-cell scales it sits
  at the x = 0.5 face and still drifts down at slab 24. It also has the fewest CG iterations (16% fewer than per-cell,
  24% fewer than base) and the smallest |f min|.
- Emission (0.024–0.035 n0) and E(0) (within ±0.001 E0) are the same in all three.

**Reading, across p = 3 and p = 4 (L0 only):**
- At p = 4 the size of the weight decides. Production's (1.8, 0.65) is too little in the L0 anode cell (h_x = 1/2), and
  any of the three larger weights fixes it.
- At p = 3 the variation between cells costs accuracy: per-cell is 15–25× production's anode-side peak, while the larger
  box-wide bound is 2× better than production.
- So on this evidence **box-wide a = (6.11, 4.15) is the better choice of the two ideas.** It is the best run at p = 3,
  holds at p = 4, decays first, needs the fewest CG iterations, and has no per-substep norm change (your point under 4).
  Per-cell with floor 1.0 is the runner-up: never less than production, a slightly smaller peak at p = 4, much worse at p = 3.
- **Caveat:** all of this is L0. At L1 production's p = 4 weight was stable on your side, so the p = 4 ranking needs your
  L1 runs. The cheap L1 check we would ask for: p = 4, 25 slabs, box-wide (6.11, 4.15) against per-cell floor 1.0.
  The cathode mode never appeared at L0, so it is untested by these runs.

## L1, p = 4: the local runs (2026-11-24)

From `mode-tests-L1-p4-results-2026-11-24.md` (local session). Mac Studio, 16 ranks, this branch at 4cc856a. Production
configuration: L1, p = 4, Δp = 2, Serendipity, transport-scale graph norm, 25 slabs, SCL. Production's and a = (3.6, 1.3)
rows are the 11-06 baselines, from older code.

Anode side (x ≥ 0.5), slab-end ∫n⁻:

| slab | 8 | 9 | 10 | 13 | 17 | 19 | 21 | 24 | peak |
|---|---|---|---|---|---|---|---|---|---|
| production (1.8, 0.65) | −0.012 | −0.0221 | −0.0214 | −0.0185 | −0.0166 | −0.014 | −0.0085 | −0.0021 | −0.022 (9) |
| a = (3.6, 1.3) | −0.011 | −0.0205 | −0.0195 | −0.0184 | −0.0201 | −0.0181 | −0.0121 | −0.0035 | −0.021 (9) |
| box-wide (6.11, 4.15) | −0.014 | −0.0254 | −0.0264 | −0.0335 | −0.0328 | −0.0276 | −0.0166 | −0.0040 | −0.034 (13) |
| per-cell, floor 1.0 | −0.014 | −0.0292 | −0.0317 | −0.0266 | −0.023 | −0.0155 | −0.0063 | −0.0007 | −0.032 (10) |

| | CG its per slab | wall | peak Σ RSS | slab-24 gap ∫n | slab-24 ∫f | slab-24 ∫J_x |
|---|---|---|---|---|---|---|
| production | 390–440 | 3.85 h | 96 GB | 2.198 | 0.218719 | −0.01833 |
| box-wide (6.11, 4.15) | 290–340 | 3.64 h | 80 GB | 2.199 | 0.218920 | −0.01914 |
| per-cell, floor 1.0 | 365–415 | 3.77 h | 82 GB | 2.201 | 0.218896 | −0.02020 |

The local session also reports: no cathode mode in any run, with cathode ∫n⁻ zero from slab 2. Box-wide has the smallest
start-up undershoot, slab-0 cathode ∫n⁻ −0.0081 against −0.017.

**Against our L0 runs:**
- **The L0 p = 4 failure of production's weight does not carry over to L1.** That fits the reading above: the L0 anode
  cell (h_x = 1/2) is coarser than L1's (1/4), so the (v_x,max/h_x)² term there is 4× smaller. It fits; it is not proven.
- **At p = 4 the two candidates rank the same way at L0 and L1.** Box-wide has the slightly deeper, later anode-side peak
  than per-cell floor 1.0: −0.088 against −0.074 at L0, −0.034 against −0.032 at L1. Both are 3–5× shallower at L1
  than at L0.
- **At p = 3, L0, box-wide was better than production on the anode side. At p = 4, L1, it is about 1.5× deeper.** So
  "more weight, everywhere" does not uniformly lower the anode-side undershoot. What does hold at every p and mesh tried
  is the smaller start-up undershoot at the cathode (18× at L0 p = 3, 2× at L1 p = 4) and the fewer CG iterations (L0:
  16–24%; L1: about 25%).
- **CG iterations fall 25%, but wall time only 5%.** At p = 4 the solve is a small share of the slab. Assembly and the
  local solves dominate, so the conditioning gain matters little for production wall time today. It would matter more
  in the matrix-free or TT solvers, where the solve is most of the cost.
- **∫J_x at slab 24 differs more than the other integrals:** −0.0183, −0.0191 (+4%) and −0.0202 (+10%). This is a single
  slab-end value, so it may just be the timing of an oscillation. A comparison over several slabs would tell.

**Recommendation, unchanged: box-wide (6.11, 4.15).** It is stable at both meshes and at both p. It needs the fewest CG
iterations and keeps the norm fixed within a slab. Its only cost seen so far is a deeper, but decaying, anode-side
transient at p = 4. Per-cell floor 1.0 is equally stable, with no advantage over it, and it was much worse at p = 3.

