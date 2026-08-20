# Next Items

- [x] Add an integrated compressed Schur solve path to `VlasovAssemblyBobaDriver`
- [ ] Estimate error in (F)GMRES computed solution
- [ ] Impose non-zero Dirichlet boundary conditions
- [ ] Convert naive loops into `boba`/`Kokkos` loops
- [ ] Move data from `Kokkos` to `boba` directly?
- [ ] Tensorize the $x$, $t$ and $v$ dimensions
- [ ] Poisson solver, coupled solver
- [ ] Tucker

# L2 Schur Solve

Assemble the L2 Camellia/BOBA block matrices and solve the in-memory compressed
Schur system from this build directory:

```text
./VlasovAssemblyBobaDriver \
  --nx=2 --nt=2 --nv=2 --p=4 --dp=2 --test_space_norm=L2 \
  --solve_schur --schur_method=block_amen --schur_tolerance=2e-8 \
  --amen_max_sweeps=20 --amen_minimum_sweeps=1 --amen_kickrank=4 \
  --svd_relative_tolerance=1e-12 --svd_absolute_tolerance=1e-12
```

The current Gram inverse path is `--gram_inverse=corewise`, which assumes the
L2 Gram block is a rank-one separable tensor-train matrix. This is appropriate
for the current L2 study. A more general graph-norm or nonseparable Gram block
should use an operator solve or a compressed approximate inverse instead of
explicit corewise inversion.
