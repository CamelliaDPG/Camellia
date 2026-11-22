# Questions on the cathode and anode mode tests (cloud side, 2026-11-22)

Re: the per-cell a_u(K) results in `kronecker-fit-and-percell-acceleration-2026-11-21.md`, and your note that coupled
accuracy and stability (the cathode and anode mode tests) must pass on the conventional side before production use. We
have started those tests on our side, but our notes only name them. A few things from you would make the runs decisive.

## What we are running

The coupled test from your paper ("Coupled stability on the production-like mesh"), as closely as we can on 4 cores and
15 GB:
- The L0 production base mesh: 8 graded x cells (h_x down to 1/128), 1 t cell per slab, 11 × 10 graded velocity cells,
  880 cells per slab.
- p = 3, Δp = 2, Serendipity trial space, 20 slabs of width 0.1, SCL emission.
- 2 to 10 substeps per slab (fixed-point tolerance 1e-4), J = 5e8.
- Transport-scale graph norm, α = 1e2, C = 1.6.
- One rank per run, two runs at a time (each reaches about 6 GB). About 11.5 min per slab.

There are four runs:

| run | norm |
|---|---|
| base | a = (1.8, 0.65), box-wide (production) |
| percell | a_u(K) per cell, floor 0.1 × (1.8, 0.65) |
| cathode control | fixed α = 4e2, no transport-scale weight |
| anode control | α_K = max(1e2, 1.6/h_x²), no transport-scale weight |

The two controls are the weights that grew the cathode and anode modes in your earlier runs. They are there to show
whether this mesh can detect the modes at all.

Per slab we record:
- min n_e and the x where it occurs, from `--densityDiagnostic`;
- ∫n_e⁻ over the slab;
- f min;
- ∫f⁻ over phase space at the slab end;
- emitted density, E(0), and the charge balance.

**So far, 6 slabs.** base and percell agree closely:
- The startup undershoot at the cathode (min n_e ≈ −22 at slab 1) decays.
- By slab 5 the minimum density moves to mid-gap (x ≈ 0.47).
- In both runs ∫f⁻ grows steadily, from −0.08 at slab 0 to −1.28 at slab 6 (base); percell is 2–5% more negative.
- E(0) goes negative after slab 0 in both runs, so emission drops to the 0.001 n0 minimum.

## Questions

1. **The definition of each test.**
   - Which runs showed the cathode mode and the anode mode? The driver comments mention `production-scl-graph4e2-x16`
     for the cathode mode.
   - What mesh (x cells and grading, velocity cells), p, Δp, slab width and number of substeps did each use?
   - What was the pass/fail criterion: growth of min n_e, growth of ∫n⁻, f min, or a blow-up?

2. **How quickly the modes appeared, in slabs.** How many slabs did each mode take to become visible, and to become
   unmistakable? If it was more than about 20, our runs are too short to show it.

3. **Resolution.** The cathode mode appeared in cells with h_x = 1/256, and our L0 mesh stops at 1/128.
   - Do you expect the modes at L0 resolution?
   - If not, can you run the production-resolution test on your side? Our container cannot: production is 32 graded x
     cells, p = 4, 16 ranks, 926 s per slab.
   - The per-cell code is on our branch (`separable-gram-fast-diagonalization`), driver option
     `--L2WeightGraphPerCellAccelerationFloor=0.1`. It works for both the graph norm and the separable norm.

4. **Which per-cell definition you intend.** The two differ in three ways. Since stability is the question, they should
   match before anyone draws conclusions.

   | | your proposal (`meshes-and-alpha-for-cloud-2026-11-19.md`) | ours |
   |---|---|---|
   | a_u(K) from | the field maxima over the whole slab | the current fields at the cell's quadrature points, re-evaluated at every Vlasov solve (every substep) |
   | overall factor | κ ≈ 0.5, "to keep the tuned overall size" | 1.0 |
   | floor | not mentioned | 0.1 × the box-wide bound |

   - The factor matters most. You described (1.8, 0.65) as "about half the core-cell accelerations, tuned for
     stability", so the stable size may not be the true bound. Should the first test be κ = 0.5 or κ = 1?
   - Will the TT side rebuild the scales per slab or per substep?
   - Is a floor needed on your side? Ours keeps α_K and s_u bounded below where the acceleration vanishes.

5. **Baseline numbers to check against.**
   - Please send the per-slab histories of n⁻ (or min n_e) and f min from your p = 3, 20-slab runs, graph and separable
     norm. Your paper reports n⁻ = −0.024 (graph) and −0.030 (separable) at slab 19.
   - Is that n⁻ the minimum of n_e, or an integral of its negative part?

6. **Is the ∫f⁻ growth expected?** In our base run the phase-space negative part ∫f⁻ grows every slab (−0.08 → −1.28
   over 6 slabs) while min n_e recovers. Do your production runs show the same growth? Is it one of the things the mode
   tests watch, or is the density all that matters?
