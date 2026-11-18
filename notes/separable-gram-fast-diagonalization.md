# Separable graph norm: exact element inverse, and the path to matrix-free planar MITL

Status: the fast-diagonalization inverse is implemented, tested and used by the assembled solves.  The matrix-free
operator (sum-factorized B, element and global Tpetra operators) is implemented and validated; preconditioning and
integration into the Vlasov solve are next (see "Next steps").

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

## Matrix-free operator (implemented)

- `TensorBasisDecomposition` (`src/include/TensorBasisDecomposition.h`) splits a tensor-product or serendipity basis
  into its 1D components and gives their value and derivative tables.  It drops the zero-dimensional point components
  that bases on the sides normal to the last tensor direction carry.
- `BoxCellDPGOperator` (`src/include/BoxCellDPGOperator.h`) applies B_K, B_K^T, S_K = B_K^T G_K^{-1} B_K and the load
  B_K^T G_K^{-1} l_K on box cells.  It reads them off the `BF`'s own terms:
  - field terms (c(x) u, v or d_k v), with c sampled at tensor Gauss points (an `Intrepid2::TensorPoints`, as
    `PhysicalSolutionExtrusion` requires);
  - flux terms <c t, v>, side by side, using signed-permutation side maps and side parities.
  B is applied by sum factorization (BLAS mode products), with serendipity trial bases zero-padded into the tensor
  layout.  G^{-1} is `SeparableTensorIP::cellInverses()`, cached per cell.  Camellia's value transforms are matched:
  L^2 fields are divided by det J; for fluxes the transform is measured per cell and side from the side `BasisCache`.
  On BoxGDA's 4D cells it is 1, i.e. Camellia does not divide those fluxes by the normal scaling.
- `MatrixFreeDPGOperator` (`src/include/MatrixFreeDPGOperator.h`) is the global S = sum_K P_K^T S_K P_K as a Tpetra
  operator, through the mesh's dof interpretation (constraints included).  Dirichlet dofs are identity rows and
  columns, and it provides the matching right-hand side.

Tests (`unit_tests/BoxCellDPGOperatorTests.cpp`, all pass), on a 4D box mesh with refinement and anisotropic p,
polynomial coefficients, a flux term, and tensor and serendipity trial bases:

- B, B^T and S_K match the assembled element matrices to 3e-15, 3e-15 and 5e-15 (2.7e-12 with the constant-mode
  weight).
- Global S x matches Solution's assembled matrix to 1.1e-12, and the load to 1.4e-12.
- CG on the matrix-free system with Dirichlet fluxes reproduces the direct solve to 1.3e-8.

PlanarMITL driver check (`--checkMatrixFreeOperator`, with `--separableGraphTestNorm --useSerendipityBasis
--useFullTensorTestBasis`).  It runs after the first slab, so the fields are real.  It compares S x with a freshly
assembled stiffness matrix on a random x:

| p | cells | dofs | quadrature enrichment | \|S x - A x\| / \|A x\| | assembly | MF setup + 1st apply | MF apply |
|---|-------|------|-----------------------|-------------------------|----------|----------------------|----------|
| 2 | 64    | 2896 | 0                     | 5.4e-4                  | 1.7 s    | 1.4 s                | 0.020 s  |
| 2 | 64    | 2896 | 2                     | 6.4e-5                  | 4.1 s    | 1.6 s                | 0.023 s  |
| 2 | 64    | 2896 | 4                     | 4.4e-5                  | 9.4 s    | 1.7 s                | 0.025 s  |
| 2 | 64    | 2896 | 8                     | 1.6e-6                  | 29 s     | 2.4 s                | 0.035 s  |
| 3 | 8     | 748  | 0                     | 2.2e-4                  | 1.3 s    | 0.22 s               | 0.005 s  |
| 4 | 8     | 1296 | 0                     | 2.8e-4                  | 6.0 s    | 0.27 s               | 0.008 s  |

The two operators converge to each other as both quadratures are enriched.  The difference is quadrature error from
the non-polynomial coefficients (v = u / gamma, the fields), mostly on the assembled side at the driver's default
enrichment 0; the matrix-free rule uses n_k (+ enrichment/2) points per direction.  At p = 4 one application costs about
1 ms per cell, against 0.74 s per cell to assemble.  One core, -O1 build.

## Jacobi preconditioning in the shared eigenbasis (implemented)

The diagonal of the global operator is computed without forming anything.  For each cell K and each global dof g
touching it, take u = P_K e_g, the cell's coefficients of global basis function g, from the dof interpretation
(constraints included).  Then

    (P_K e_g)^T S_K (P_K e_g) = sum_i ((V^T B_K u)_i)^2 / D_i   (+ c (z^T B_K u)^2 / (1 - c m^T z) with the constant-mode weight),

which costs one sum-factorized B and half a fast inverse (V^T only) per (cell, dof).  This is
`MatrixFreeDPGOperator::diagonal()`, with `BoxCellDPGOperator::schurQuadraticForms()` and
`SeparableTensorIP::inverseQuadraticForms()` underneath.  `DiagonalScalingOperator` is the Jacobi preconditioner.

- Unit test: diag(S) matches the assembled diagonal to 1.3e-12.  Jacobi PCG takes 72 iterations against 1028 for CG
  and matches the direct solve to 7.9e-11.
- Driver check (`--checkMatrixFreeOperator`): the first slab's system, with its BCs, load and current fields, is
  solved by matrix-free Jacobi PCG to relative residual 1e-10 and compared with a direct solve of the assembled system
  (p = 2, Delta p = 2, serendipity trial, full tensor test, one core):

| cells (x t ux uz) | dofs  | enrichment | PCG iterations | PCG time (setup) | direct (with assembly) | \|u_PCG - u_direct\| / \|u_direct\| |
|-------------------|-------|------------|----------------|------------------|------------------------|-------------------------------------|
| 4 1 4 4           | 2896  | 0          | 92             | 1.8 s (0.66 s)   | 2.1 s                  | 3.1e-2                              |
| 4 1 4 4           | 2896  | 4          | 92             | 2.4 s (0.82 s)   | 10.4 s                 | 1.3e-3                              |
| 4 1 4 4           | 2896  | 8          | 92             | 3.4 s (1.2 s)    | 35 s                   | 4.6e-5                              |
| 8 1 8 8           | 21824 | 4          | 170            | 37 s (6.6 s)     | 91 s                   | 1.1e-4                              |

  The PCG/direct difference tracks the operators' quadrature difference (5e-4, 4e-5, 1.6e-6 on 64 cells; 1.7e-6 on
  512 cells at enrichment 4), so it is quadrature error, amplified by the conditioning, and not a solver error.
  Iteration counts do not depend on the quadrature.  They grow like h^-1 (92 -> 170 when x, u_x and u_z are halved),
  as expected for a diagonal preconditioner.
- Note: at the driver's default quadrature (enrichment 0) the assembled and matrix-free solutions differ by 3%.  Both
  converge to each other under enrichment, so at least one of them, probably the assembled one, which uses fewer points
  per direction, under-integrates the non-polynomial coefficients v = u / gamma and a(u) by about that much.  This is
  worth checking against the production runs.

## Next steps

1. **A preconditioner that does not degrade with h.**  Jacobi's iterations grow like h^-1.  Options:
   - Chebyshev-Jacobi smoothing within `MatrixFreeGMGPreconditioner`, on the existing GMG hierarchy, with assembled
     coarse levels;
   - trace-block Jacobi, i.e. per-face blocks of S, from the same eigenbasis quadratic forms with off-diagonal pairs.
2. **Solve integration:** a matrix-free option in `VlasovSpaceTimeFormulation::solve()` (Belos CG on
   `MatrixFreeDPGOperator` with `DiagonalScalingOperator`, Dirichlet data from `dirichletData()`), validated slab by
   slab against the direct solve.
3. **Static condensation** of the field dofs: a per-element factor of the (serendipity-sized) field-field block.
4. **Performance:**
   - coefficient sampling at setup dominates the first application: exploit the separable structure of the Vlasov
     coefficients;
   - batch cells of equal shape through the BLAS mode products;
   - test MPI (overlap-map import/export is in place; only one rank has been run).
