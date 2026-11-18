# Separable graph norm: exact element inverse, and the path to matrix-free planar MITL

Status: the fast-diagonalization inverse is implemented, tested and used by the assembled solves.  The matrix-free
operator is not started yet (see "Next steps").

## Why

With the graph norm, every element Gram matrix G_K differs (the fields vary), so a matrix-free DPG operator
S u = sum_K B_K^T G_K^{-1} B_K u_K would either factor every G_K on every application or store a factor per element.
The direction-separable graph norm (`--separableGraphTestNorm`; see the separable-graph-norm notes) has

    G_K = J_K ( alpha_K M^{(x)4} + sum_d xi_d K_(d) ),

a Kronecker sum, *on a full tensor-product test space*.  In the 1D generalized eigenbasis V of (K, M), shared by
all elements,

    G_K^{-1} = V^{(x)4} diag( 1 / (J_K (alpha_K + sum_d xi_d lambda_{j_d})) ) (V^T)^{(x)4},

exactly.  The element enters through six scalars (J_K, alpha_K, xi_d), so nothing is stored per element.

## Why a full tensor test space (and not serendipity)

The identity needs the full tensor space.  Restricting it to a serendipity test space S does not give G_S^{-1}:
P^T G_Q^{-1} P is the inverse of the Schur complement of G_Q onto S.  How close the two are depends on the 1D basis
that defines the complement.  Sampled over the compact set of element parameters that the transport-scale weight
allows (r_d = xi_d / alpha_K >= 0, sum_d r_d <= 4/C), in 4D:

| k | dim S | dim Q_k | kappa, hierarchical (current) | kappa, Legendre |
|---|-------|---------|-------------------------------|-----------------|
| 3 | 80    | 256     | 3.8e3                         | 1.17            |
| 4 | 136   | 625     | 3.1e5                         | 1.41            |
| 5 | 216   | 1296    | 3.0e5                         | 1.65            |
| 6 | 328   | 2401    | 5.8e6                         | 1.90            |

In a matrix-free operator, though, the test space costs no stored matrices.  Sum-factorized kernels work on the full
(k+1)^4 tensor anyway, padding serendipity with zeros.  So serendipity's memory advantage on the test side
disappears.  We therefore keep serendipity for the **trial** space, which sets the global unknowns, and use the full
Q_k **test** space: `--useSerendipityBasis --useFullTensorTestBasis`.  The larger test space can only help
stability, and the separable-norm equivalence results are stated for Q_k.

## What is implemented

- `SeparableGramInverse` (`src/include/SeparableGramInverse.h`, `src/ScratchPad/SeparableGramInverse.cpp`): applies
  G and G^{-1} by mode products, with per-direction 1D spaces (anisotropic p allowed).  It has no Trilinos dependence,
  so it can serve the matrix-free operator as is.
- `TIP::applyInverseInnerProductMatrix()`: a virtual hook that declines by default.  `SeparableTensorIP` implements it
  when its 1D assembly applies, the test basis is a full tensor product, and the value weight is positive.  On
  serendipity it declines.
- The hook is used by `BF::localStiffnessMatrixAndRHS` (factored-Cholesky path),
  `Solution::computeErrorRepresentation` (energy error) and `RieszRep::computeRieszRep`.  Each falls back to forming
  and factoring G when the hook declines or test-space BCs are imposed.
- `RelativisticVlasovSpaceTimeFormulation(..., fullTensorTestBasis)` and the driver option `--useFullTensorTestBasis`
  (default off: existing runs are unchanged).
- Fixed a heap overrun in `SeparableTensorIP`'s 1D matrices.  It allocated Intrepid2's LayoutStride `PointViewType`
  from extents alone.

## Measurements

Unit tests (`unit_tests/SeparableTensorIPTests.cpp`, all pass):

- The kernel matches dense Kronecker sums, with backward error <= 3e-17.
- Lambda_k reproduces Table 1 of the notes, k = 1..8.
- On box meshes with anisotropic p, the IP inverse has backward error 6e-16 and declines on serendipity.
- The BF fast path matches the Cholesky path to 2.3e-13.

Kernel against dense LAPACK Cholesky (OpenBLAS, one core, 200 right-hand sides), all agreeing to 1e-11:

| k | N    | fast diag | Cholesky factor | Cholesky solve |
|---|------|-----------|-----------------|----------------|
| 4 | 625  | 0.010 s   | 0.003 s         | 0.007 s        |
| 5 | 1296 | 0.023 s   | 0.016 s         | 0.015 s        |
| 6 | 2401 | 0.034 s   | 0.116 s         | 0.056 s        |

The mode products are plain loops (about 0.8 GFLOP/s).  Recasting them as batched GEMMs should gain about 10x.  Even
as they are, they avoid the per-element factorization, and its storage: 46 MB per element at k = 6.

PlanarMITL driver smoke test: p = 2, Delta p = 2 (k = 3), 4 x 1 x 4 x 4 cells, one slab, three substeps, separable
norm, serendipity trial.

|                         | serendipity test | full tensor test |
|-------------------------|------------------|------------------|
| local stiffness + load  | 1.4 s            | 5.6 s            |
| of which B (quadrature) | 1.7 s            | 6.7 s            |
| Cholesky solves         | 208              | 16               |

The 16 remaining Cholesky solves are presumably the configuration-space Maxwell solves.  f min/max change by 2–4%
from the test-space change.  In the **assembled** path, then, the cost of Q_k is the quadrature assembly of B and of
the load against the larger test space, not G^{-1}.  At k = 6 that is about 7x the serendipity test space.

## Next steps (toward a matrix-free planar MITL operator)

1. **Sum-factorized B_K and B_K^T** with in-element varying coefficients: v_x(u), and a(u, x, t) from the fields.
   Use quadrature on the tensor grid with 1D interpolation and derivative matrices: O(q^5) per term, field
   coefficients sampled at the points.  `LinearTermSumFactorized` / Intrepid2 PAMatrix are the place to start, or a
   hand-written kernel for the Vlasov form's terms.  This removes the assembled path's Q_k penalty measured above.
2. **Operator** S u = sum_K B_K^T G_K^{-1} B_K u_K as a Tpetra operator.  G_K^{-1} comes from
   `SeparableGramInverse` with six scalars per element; trial serendipity coefficients are zero-padded into the
   tensor layout.  Validate against the assembled operator on small meshes.
3. **Condensation and preconditioning**: a per-element factor of the field block (small, serendipity-sized) if field
   dofs are condensed, and diag(S) or block diagonals for the matrix-free GMG smoothers
   (`MatrixFreeGMGPreconditioner`).
4. **Kernel performance**: express the mode products as GEMMs, and apply them to several vectors at once.
