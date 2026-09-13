# Fresh Tensor Implementation Plan, Part 2

## Purpose of this document

This is the fresh continuation plan for the tensor-native PAMatrix work.  It summarizes the work already completed under the first plan, records what the latest diagnostics say, and lays out the remaining work in the order that now seems most useful.

The previous plan should be treated as the archived Part 1 after this file is added.  This document should be the active plan going forward.

## Current status summary

The project has moved from a general implementation sketch to a working direct-PAMatrix experimental path for tensor-product HGRAD Gram operators.  The important pieces now exist in runnable diagnostic form:

- a direct `Intrepid2::PAMatrix` HGRAD Gram driver path in `drivers/performance/TensorPoissonPAMatrixDriver.cpp`;
- real HGRAD tensor basis mode for 3D runs;
- synthetic tensor basis mode for 4D scaling experiments;
- matrix-free apply diagnostics;
- standard PAMatrix `assemble()` diagnostics;
- experimental column-apply-style `assemble2()` diagnostics;
- term-separated timing for mass and stiffness components;
- PAMatrix diagonal extraction and inverse-diagonal diagnostics;
- local flat Tpetra `Operator` and `RowMatrix` adapters for smoke testing;
- Ifpack2 Chebyshev smoke diagnostics against a host-side reference recurrence;
- lower-spectral-estimate diagnostics based on assembled-reference inverse iteration for diagnostic runs.

The code is still an experiment and not yet a production global FE solver or preconditioner path.  In particular, the current Tpetra adapters are explicitly local-flat/nonproduction adapters with identical local domain/range/row/column maps and no global FE ownership semantics.  That limitation is acceptable for the current diagnostics, but it must not be confused with a finished distributed matrix-free solver integration.

## Diagnostics rerun for this planning turn

The latest requested diagnostic outputs are:

- `runTensorH1GramPAMatrixLowerSpectralChebyshevSweep-out.txt`
- `runTensorH1GramPAMatrixIfpack2PowerNoReferenceSmoke-out.txt`
- `runTensorH1GramPAMatrixIfpack2ChebyshevSmoke-out.txt`
- `runTensorH1GramPAMatrixTpetraRowMatrixSmoke-out.txt`

Together with the previous assembly sweeps, these show the following.

### 3D standard assembly versus assemble2

For the direct 3D HGRAD runs with `p = 8..12` and one cell:

- Standard `assemble()` times were approximately 0.116, 0.245, 0.513, 1.042, 1.796 seconds.
- Current `assemble2()` times were approximately 0.283, 0.510, 0.865, 1.882, 2.444 seconds.
- The reported `assemble2` slope was about 6.13, while the expected column-apply assembly slope label was 7.
- The reported standard `assemble()` slope was about 7.55, with term-separated slopes about 6.41 for mass and 8.09 for stiffness.

The most important practical observation is that current `assemble2()` is not the faster replacement for standard `assemble()` in these measurements.  It is often slower, despite a somewhat lower fitted slope over this small range.  This supports the suspicion that the current `useUnitColumnPath == true` branch is not implementing the intended staged unit-column optimization.

### 4D synthetic assembly versus assemble2

For synthetic 4D HGRAD runs with `p = 4..8` and one cell:

- Current `assemble2()` times were approximately 0.052, 0.193, 0.584, 1.380, 4.136 seconds.
- The reported `assemble2` slope was about 7.31, while the expected column-apply assembly slope label was 9.

These 4D numbers are encouraging in that they do not look like dense fallback behavior, but they do not by themselves validate the intended sparse-propagation unit-column algorithm.  They should be rerun after the `assemble2()` unit-column implementation is corrected and instrumented.

### Tpetra RowMatrix smoke path

The RowMatrix smoke diagnostics show that the current local-flat RowMatrix adapter can reproduce local PAMatrix results in the tested tiny cases:

- operator apply max absolute/relative errors were reported as zero in the visible smoke cases;
- diagonal errors were reported as zero;
- row copy and row view errors were around roundoff, e.g. `1.1e-16` to `3.3e-16`;
- the diagnostic clearly labels the path as `local_flat_map_nonproduction_rowmatrix` with `globalDofSemantics=not_supported` and `ifpack2RowMatrix=local_diag_copy_only_no_global_fe_semantics`.

This is the right status for now: useful smoke coverage for Ifpack2/RowMatrix interfaces, not a production distributed FE RowMatrix.

### Ifpack2 Chebyshev smoke path

The Ifpack2 Chebyshev smoke outputs show that the Chebyshev recurrence can be exercised against the local-flat diagnostic operator and compared to a matching host-side recurrence.  The visible smoke lines report errors at or near roundoff in the direct recurrence comparison, e.g. `maxAbsError` around `1e-14` or smaller depending on the diagnostic, with explicit diagnostic notes that this is not yet a production global solver path.

The lower-spectral diagnostic run also records an assembled-reference inverse-iteration lower estimate, a Gershgorin-style preconditioned upper estimate, and a successful Ifpack2 smoke comparison against the matching recurrence.  The separate inverse diagnostic may still show nonconvergence under a fixed iteration limit for the approximate inverse application; that should be interpreted as a diagnostic of the current inverse/preconditioner parameter choices, not as a failure of the basic Chebyshev recurrence smoke comparison.

## Current `assemble2()` unit-column branch

The current `assemble2()` implementation selects `useUnitColumnPath` when orientations are absent and when each component integral has compatible left/right tensor-op structure, scalar point-data spans, and matching product extents.  This is a reasonable guard for an optimized path.

However, the true branch currently does the first right-side contraction by launching over all cells, all quadrature point ordinals, and all block columns.  For each point and each requested unit column, it decodes the column ordinal and point ordinal, then multiplies the relevant 1D right-operator entries across every tensor component:

- for a global unit column, compute every tensor quadrature value in the full `P = product(M_r)` point space;
- store that dense point vector into `pointDataIn`;
- multiply by point data;
- run the left-side tensor contractions with GEMM calls;
- accumulate into output columns.

This avoids materializing the full dense input multivector, but it does not implement the staged sparse propagation optimization intended for a unit column.

The intended algorithm is different:

1. Start from a unit tensor-product coefficient column.  It has exactly one nonzero, equal to one.
2. In the first right-apply, multiply a 1D `(p x p)` basis-values matrix by a logically `(p x p^{d-1})` coefficient tensor with one nonzero.  The result is just one copied/scaled 1D basis column: `p` values copied into the appropriate slice.  This is memory movement of about `p` values, not a full dense point-tensor evaluation.
3. After the first right-apply, the intermediate has about `p` nonzeros.
4. The second right-apply costs about `p^2` multiply-add work.
5. The third right-apply costs about `p^3`, and so on.
6. For dimension `d`, the full right-side unit-column contraction should cost a geometric sum on the order of `p + p^2 + ... + p^d`, rather than repeatedly treating the input as dense from the start.

Asymptotically in `d`, this saves about half of the right-apply work for column assembly.  The expected savings are substantial enough to pursue: roughly 33% for 3D and roughly 44% for 4D in the user's estimate, matching the observed gap between the generic `assemble2()` path and the hand-crafted 3D `assemble()`-like implementation.

## Plan for remaining work

### Phase A: document and instrument the current `assemble2()` behavior

1. Add explicit diagnostic labels for the active assembly path:
   - `assemble2Path=unit_column_dense_point_eval` for the current true branch;
   - `assemble2Path=apply_unit_multivector_fallback` for the false branch;
   - later, `assemble2Path=staged_unit_column_sparse_right_apply` for the corrected path.
2. Add counters or estimates for the right-side unit-column setup work:
   - number of point entries written before point-data multiplication;
   - estimated multiplies in the right-side unit-column evaluation;
   - estimated entries present after each staged right apply once the new path exists.
3. Keep the current path available behind a diagnostic option until the new path has been verified, so that performance and correctness can be compared directly.

Acceptance criteria:

- Logs make clear which `assemble2()` path was used.
- The current path is no longer described as the optimized staged sparse unit-column path.
- Existing correctness comparisons still pass.

### Phase B: implement staged sparse unit-column right-apply

Implement the optimized `useUnitColumnPath == true` branch as a staged tensor contraction from a canonical unit column.

Required design points:

1. Decode each global column into per-dimension field ordinals once per cell/block column.
2. Choose a deterministic tensor-dimension ordering for the staged right apply.  Start with the natural storage order unless measurement shows a better choice.
3. First dimension:
   - copy the selected 1D basis column into the appropriate intermediate slice;
   - write only `M_0` values, not all `P` values.
4. Subsequent dimensions:
   - apply the next 1D right operator only over the currently active tensor slab;
   - grow the active extent from `product_{j<r} M_j` to `product_{j<=r} M_j`;
   - avoid looping over inactive entries.
5. After all right dimensions are applied, the result is dense over quadrature points, as before, and can be passed to the existing point-data multiply and left-side contractions.
6. Preserve support for a block of unit columns.  The first implementation may keep the current `blockSize <= 64` structure, but the staged kernels should avoid doing dense inactive work for each block column.
7. Keep the guard conservative.  If offsets/spans, point-data spans, orientations, or op extents do not match the assumptions, fall back to the existing safe path.

Acceptance criteria:

- For all existing small correctness tests, staged `assemble2()` agrees with standard `assemble()` to roundoff or to the current expected tolerance.
- The implementation is Kokkos-compatible and does not introduce host-only work in the performance path.
- The diagnostic path label confirms that the staged sparse right-apply branch is active for the HGRAD Gram cases.

### Phase C: verification matrix for the new `assemble2()` path

Add or extend tests so the optimized branch is verified across the cases that could break tensor indexing.

Minimum cases:

- dimensions 1, 2, 3, and synthetic 4;
- polynomial orders at least 1 through 5 for automated tests;
- mass-only, stiffness-only by component, and full H1 Gram;
- both constant and per-cell point data where supported by the optimized assumptions;
- multiple cells if the data structures allow it without changing the current local diagnostic focus;
- nonzero left/right ordinal offsets or mixed component spans should either be supported and tested or explicitly rejected by the guard and covered by fallback tests.

Correctness comparisons:

- optimized `assemble2()` versus standard `assemble()`;
- optimized `assemble2()` versus matrix-free apply on random vectors for small cases;
- diagonal extracted from optimized/full assembly versus `extractDiagonal()`;
- symmetry checks for H1 Gram assembled matrices.

Acceptance criteria:

- The test suite catches dimension-ordering mistakes, column ordinal decoding mistakes, and left/right offset mistakes.
- The fallback path remains covered.
- No diagnostic-only shortcut is required for correctness.

### Phase D: performance reruns after the staged path lands

Rerun the same diagnostics so Part 2 can be closed with direct before/after evidence.

Required reruns:

- 3D HGRAD `assemble()` p-sweep;
- 3D HGRAD `assemble2()` p-sweep with staged path active;
- 3D HGRAD assembly compare sweep;
- synthetic 4D `assemble()` p-sweep;
- synthetic 4D `assemble2()` p-sweep with staged path active;
- synthetic 4D assembly compare sweep;
- diagonal extraction sweep if diagonal code is touched;
- Tpetra RowMatrix smoke;
- Ifpack2 Chebyshev smoke;
- lower-spectral Chebyshev diagnostic.

Acceptance criteria:

- 3D optimized `assemble2()` shows a meaningful improvement consistent with the expected roughly one-third savings for the relevant assembly portion, or the discrepancy is explained by memory bandwidth, GEMM overhead, or point-data/left-apply dominance.
- 4D optimized `assemble2()` shows a stronger improvement, expected to be closer to the roughly 44% right-apply-side savings if the right-apply work dominates.
- Correctness errors remain at roundoff/small tolerance levels.
- Tpetra and Ifpack2 smoke diagnostics still pass.

### Phase E: clean up the diagnostic Tpetra/Ifpack2 story

The current Tpetra and Ifpack2 work is useful, but it must be kept accurately scoped.

Tasks:

1. Keep all local-flat adapter output labels explicit: nonproduction, local-only, no global FE semantics.
2. Separate recurrence verification from solver/preconditioner claims.
3. Keep power-method and spectral-estimate diagnostics opt-in and clearly label the source of each estimate:
   - ratio-based lower bound;
   - assembled-reference inverse-iteration lower bound;
   - assembled-reference Gershgorin upper bound;
   - power-method upper estimate when available without assembled reference.
4. Decide what the first production-like global integration target should be:
   - a true distributed Tpetra operator that applies the tensor local operator with gather/scatter; or
   - a production RowMatrix materialization path for assembled local matrices; or
   - a matrix-free preconditioner interface that does not pretend to be a RowMatrix.

Acceptance criteria:

- The diagnostic smoke tests remain useful but cannot be mistaken for production global FE capability.
- The next production integration target is explicitly chosen before large-scale solver work begins.

### Phase F: connect back to the original tensor-native objective

Once the PAMatrix HGRAD Gram kernels have the expected correctness and scaling, return to the broader tensor-native Camellia goals.

Tasks:

1. Identify the minimum DPG/Vlasov operator terms that can be expressed with the same PAMatrix component-integral model.
2. Determine where trial/test asymmetry, traces, graph norms, and Schur complement structure require extensions beyond the current HGRAD Gram machinery.
3. Reuse the verified tensor apply/assembly/diagonal kernels instead of creating a second independent tensor contraction implementation.
4. Add guardrails so dense local Schur complement paths remain opt-in baselines for small problems only.

Acceptance criteria:

- The HGRAD Gram experiment has a clear performance conclusion.
- The next DPG/Vlasov step is based on measured PAMatrix kernel behavior, not on assumptions.

## Immediate next task

The next implementation turn should focus narrowly on `assemble2()`:

1. rename or label the current true branch as dense point evaluation from unit columns;
2. add instrumentation sufficient to prove which path ran;
3. implement the staged sparse unit-column right-apply under the existing conservative guard;
4. compare optimized `assemble2()` against the existing current branch, standard `assemble()`, and matrix-free apply for small cases;
5. rerun the 3D and 4D p-sweeps.

Do not move on to production Tpetra/Ifpack2 integration until this assembly-path question is resolved.  The current timings indicate that the main missing optimization is still inside the local tensor assembly kernel, and fixing that has the best chance of changing the conclusions of the performance study.

## Proposed commit message

Add fresh tensor implementation plan part 2

- summarize completed PAMatrix HGRAD Gram, Tpetra, Ifpack2, and spectral diagnostics
- record current interpretation of assemble2 unit-column behavior
- plan staged sparse unit-column right-apply optimization
- define verification and performance rerun criteria for the remaining tensor work
