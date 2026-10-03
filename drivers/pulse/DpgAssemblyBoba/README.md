# DPG Assembly BoBa driver

This directory contains the original space--time Vlasov DPG assembly driver
and shared finite-element assembly support. The seven-dimensional driver has
moved to the sibling [DpgAssemblyBoba_7d](../DpgAssemblyBoba_7d) directory.

## L2 Schur solve

From the build directory, assemble the L2 Camellia/BoBa block matrices and
solve the compressed Schur system with:

```text
./VlasovAssemblyBobaDriver \
  --nx=2 --nt=2 --nv=2 --p=4 --dp=2 --test_space_norm=L2 \
  --solve_schur --schur_method=block_amen --schur_tolerance=2e-8 \
  --amen_max_sweeps=20 --amen_minimum_sweeps=1 --amen_kickrank=4 \
  --svd_relative_tolerance=1e-12 --svd_absolute_tolerance=1e-12
```

The corewise Gram-inverse path assumes a rank-one separable $L^2$ Gram matrix.
Graph norms or nonseparable Gram blocks require an operator solve or a
compressed approximate inverse.
