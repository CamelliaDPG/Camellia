# Seven-dimensional DPG Vlasov driver

`Vlasov3X3VAssemblyBobaDriver` solves the manufactured constant-acceleration
Vlasov problem with three spatial coordinates, one time coordinate, and three
velocity coordinates. Its source and seven-mode AMEN implementation live in
this directory. It does not depend on sources from the sibling
`DpgAssemblyBoba` driver.

The default tensor-train order is `(x1,x2,x3,t,v1,v2,v3)`. Use
`--interleaved_order` for `(x1,v1,x2,v2,x3,v3,t)`.

From the build directory, a small solve is:

```text
./Vlasov3X3VAssemblyBobaDriver \
  --nx=2 --nt=2 --nv=2 --p=1 --dp=1 \
  --solve_schur --no_write_system --interleaved_order \
  --amen_max_sweeps=2 --amen_minimum_sweeps=1 --amen_kickrank=2 \
  --amen_exact_initial_guess --schur_tolerance=1e-3 \
  --svd_relative_tolerance=1e-10 --svd_absolute_tolerance=1e-10
```

The driver reports Schur and reconstructed-system residuals, TT ranks and
storage, solver time, and errors relative to both the represented exact field
and the continuous manufactured solution. `plot_vlasov_spectra.py` and
`study_mass_weighted_rounding.py` are retained here as seven-dimensional
analysis utilities.
