#include "vlasov_system.hpp"

#include "reference_integrals_1d.hpp"
#include "global_assembly.hpp"

// basis keys: w = test, f = field, t = trace
boba::BlockOperator<host_tt_matrix<3>>
construct_vlasov_matrices(Camellia::BasisPtr w_basis, Camellia::BasisPtr t_basis, Camellia::BasisPtr f_basis,
                          const double &x_min, const double &x_max, const size_t &nx,
                          const double &t_min, const double &t_max, const size_t &nt,
                          const double &v_min, const double &v_max, const size_t &nv,
                          const size_t &p, const size_t &dp) {
  boba::TicToc<::boba::tictoc_units::milliseconds> timer;

  //
  // integrals
  //

  timer.tic();

  // reference integrals (w = test_value, df = field_derivative, etc.)

  auto ref_ints_1d = reference_integrals_1d();

  auto ref_int_w_f    = ref_ints_1d.discretization(w_basis, false, f_basis, false);
  auto ref_int_w_t    = ref_ints_1d.discretization(w_basis, false, t_basis, false);
  auto ref_int_w_w    = ref_ints_1d.discretization(w_basis, false, w_basis, false);
  auto ref_int_dw_f   = ref_ints_1d.discretization(w_basis, true,  f_basis, false);
  auto ref_int_dw_w   = ref_ints_1d.discretization(w_basis, true,  w_basis, false);
  auto ref_int_dw_dw  = ref_ints_1d.discretization(w_basis, true,  w_basis, true );

  auto ref_int_w_vf   = ref_ints_1d.discretization(w_basis, false, f_basis, false, 1);
  auto ref_int_vw_w   = ref_ints_1d.discretization(w_basis, false, w_basis, false, 1);
  auto ref_int_dw_vw  = ref_ints_1d.discretization(w_basis, true,  w_basis, false, 1);

  auto ref_int_vw_vw  = ref_ints_1d.discretization(w_basis, false, w_basis, false, 2);

  auto ref_int_w_w_f  = ref_ints_1d.discretization(w_basis, false, w_basis, false, f_basis, false);
  auto ref_int_w_w_w  = ref_ints_1d.discretization(w_basis, false, w_basis, false, w_basis, false);
  auto ref_int_dw_w_w = ref_ints_1d.discretization(w_basis, true,  w_basis, false, w_basis, false);

  // reference to physical space

  auto int_x_w_f   = global_field_assembly_1d(ref_int_w_f,                              nx, x_min, x_max);
  auto int_t_w_f   = global_field_assembly_1d(ref_int_w_f,                              nt, t_min, t_max);
  auto int_v_w_f   = global_field_assembly_1d(ref_int_w_f,                              nv, v_min, v_max);
  auto int_v_w_vf  = global_field_assembly_1d(ref_int_w_f, ref_int_w_vf,                nv, v_min, v_max);

  auto int_x_w_t   = global_field_assembly_1d(ref_int_w_t,                              nx, x_min, x_max);
  auto int_t_w_t   = global_field_assembly_1d(ref_int_w_t,                              nt, t_min, t_max);
  auto int_v_w_t   = global_field_assembly_1d(ref_int_w_t,                              nv, v_min, v_max);

  auto int_x_w_w   = global_field_assembly_1d(ref_int_w_w,                              nx, x_min, x_max);
  auto int_t_w_w   = global_field_assembly_1d(ref_int_w_w,                              nt, t_min, t_max);
  auto int_v_w_w   = global_field_assembly_1d(ref_int_w_w,                              nv, v_min, v_max);
  auto int_v_w_vw  = global_field_assembly_1d(ref_int_w_w, ref_int_vw_w,                nv, v_min, v_max);
  auto int_v_vw_vw = global_field_assembly_1d(ref_int_w_w, ref_int_vw_w, ref_int_vw_vw, nv, v_min, v_max);

  auto int_x_dw_f  = global_field_assembly_1d(ref_int_dw_f,                             nx, x_min, x_max, 1);
  auto int_t_dw_f  = global_field_assembly_1d(ref_int_dw_f,                             nt, t_min, t_max, 1);
  auto int_v_dw_f  = global_field_assembly_1d(ref_int_dw_f,                             nv, v_min, v_max, 1);

  auto int_x_dw_w  = global_field_assembly_1d(ref_int_dw_w,                             nx, x_min, x_max, 1);
  auto int_t_dw_w  = global_field_assembly_1d(ref_int_dw_w,                             nt, t_min, t_max, 1);
  auto int_v_dw_w  = global_field_assembly_1d(ref_int_dw_w,                             nv, v_min, v_max, 1);
  auto int_v_dw_vw = global_field_assembly_1d(ref_int_dw_w, ref_int_dw_vw,              nv, v_min, v_max, 1);

  auto int_x_dw_dw = global_field_assembly_1d(ref_int_dw_dw,                            nx, x_min, x_max, 2);
  auto int_t_dw_dw = global_field_assembly_1d(ref_int_dw_dw,                            nt, t_min, t_max, 2);
  auto int_v_dw_dw = global_field_assembly_1d(ref_int_dw_dw,                            nv, v_min, v_max, 2);

  auto timing_integrals = timer.toc();

  //
  // bilinear field
  //

  timer.tic();

  auto bilinear_f =
    boba::make_ttm_from_matrices<3, host_space, double>({-int_x_w_f,  int_t_dw_f, int_v_w_f }) +
    boba::make_ttm_from_matrices<3, host_space, double>({-int_x_dw_f, int_t_w_f,  int_v_w_vf}) +
    boba::make_ttm_from_matrices<3, host_space, double>({-int_x_w_f,  int_t_w_f,  int_v_dw_f});
  bilinear_f.round();

  auto timing_field = timer.toc();

  //
  // bilinear trace
  //

  timer.tic();

  auto eval_x_w = global_trace_assembly_1d(w_basis->getCardinality(), nx);
  auto eval_t_w = global_trace_assembly_1d(w_basis->getCardinality(), nt);
  auto eval_v_w = global_trace_assembly_1d(w_basis->getCardinality(), nv);

  auto bilinear_x = boba::make_ttm_from_matrices<3, host_space, double>({ eval_x_w, int_t_w_t, int_v_w_t});
  auto bilinear_t = boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_t,  eval_t_w, int_v_w_t});
  auto bilinear_v = boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_t, int_t_w_t,  eval_v_w});

  auto timing_trace = timer.toc();

  //
  // gram
  //

  timer.tic();

  auto int_x_w_dw  = int_x_dw_w.transpose();
  auto int_t_w_dw  = int_t_dw_w.transpose();
  auto int_v_w_dw  = int_v_dw_w.transpose();
  auto int_v_vw_w  = int_v_w_vw.transpose();
  auto int_v_vw_dw = int_v_dw_vw.transpose();

  auto gram =
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_w,   int_t_dw_dw, int_v_w_w  }) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_dw,  int_t_dw_w,  int_v_w_vw }) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_w,   int_t_dw_w,  int_v_w_dw }) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_dw_w,  int_t_w_dw,  int_v_vw_w }) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_dw_dw, int_t_w_w,   int_v_vw_vw}) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_dw_w,  int_t_w_w,   int_v_vw_dw}) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_w,   int_t_w_dw,  int_t_dw_w }) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_dw,  int_t_w_w,   int_v_dw_vw}) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_w,   int_t_w_w,   int_v_dw_dw}) +
    boba::make_ttm_from_matrices<3, host_space, double>({int_x_w_w,   int_t_w_w,   int_v_w_w  });
  gram.round();

  auto timing_gram = timer.toc();

  //
  // stiffness matrix
  //

  timer.tic();

  boba::BlockOperator<host_tt_matrix<3>> stiffness(vlasov_vars::size, vlasov_vars::size);

  stiffness({vlasov_vars::test,    vlasov_vars::test   }) = gram;
  stiffness({vlasov_vars::test,    vlasov_vars::trace_x}) = bilinear_x;
  stiffness({vlasov_vars::test,    vlasov_vars::trace_t}) = bilinear_t;
  stiffness({vlasov_vars::test,    vlasov_vars::trace_v}) = bilinear_v;
  stiffness({vlasov_vars::test,    vlasov_vars::field  }) = bilinear_f;

  stiffness({vlasov_vars::trace_x, vlasov_vars::test   }) = bilinear_x.transpose();
  stiffness({vlasov_vars::trace_x, vlasov_vars::trace_x}) = zero_tt_matrix<3>(bilinear_x.core_cols(), bilinear_x.core_cols());
  stiffness({vlasov_vars::trace_x, vlasov_vars::trace_t}) = zero_tt_matrix<3>(bilinear_x.core_cols(), bilinear_t.core_cols());
  stiffness({vlasov_vars::trace_x, vlasov_vars::trace_v}) = zero_tt_matrix<3>(bilinear_x.core_cols(), bilinear_v.core_cols());
  stiffness({vlasov_vars::trace_x, vlasov_vars::field  }) = zero_tt_matrix<3>(bilinear_x.core_cols(), bilinear_f.core_cols());

  stiffness({vlasov_vars::trace_t, vlasov_vars::test   }) = bilinear_t.transpose();
  stiffness({vlasov_vars::trace_t, vlasov_vars::trace_x}) = zero_tt_matrix<3>(bilinear_t.core_cols(), bilinear_x.core_cols());
  stiffness({vlasov_vars::trace_t, vlasov_vars::trace_t}) = zero_tt_matrix<3>(bilinear_t.core_cols(), bilinear_t.core_cols());
  stiffness({vlasov_vars::trace_t, vlasov_vars::trace_v}) = zero_tt_matrix<3>(bilinear_t.core_cols(), bilinear_v.core_cols());
  stiffness({vlasov_vars::trace_t, vlasov_vars::field  }) = zero_tt_matrix<3>(bilinear_t.core_cols(), bilinear_f.core_cols());

  stiffness({vlasov_vars::trace_v, vlasov_vars::test   }) = bilinear_v.transpose();
  stiffness({vlasov_vars::trace_v, vlasov_vars::trace_x}) = zero_tt_matrix<3>(bilinear_v.core_cols(), bilinear_x.core_cols());
  stiffness({vlasov_vars::trace_v, vlasov_vars::trace_t}) = zero_tt_matrix<3>(bilinear_v.core_cols(), bilinear_t.core_cols());
  stiffness({vlasov_vars::trace_v, vlasov_vars::trace_v}) = zero_tt_matrix<3>(bilinear_v.core_cols(), bilinear_v.core_cols());
  stiffness({vlasov_vars::trace_v, vlasov_vars::field  }) = zero_tt_matrix<3>(bilinear_v.core_cols(), bilinear_f.core_cols());

  stiffness({vlasov_vars::field,   vlasov_vars::test   }) = bilinear_f.transpose();
  stiffness({vlasov_vars::field,   vlasov_vars::trace_x}) = zero_tt_matrix<3>(bilinear_f.core_cols(), bilinear_x.core_cols());
  stiffness({vlasov_vars::field,   vlasov_vars::trace_t}) = zero_tt_matrix<3>(bilinear_f.core_cols(), bilinear_t.core_cols());
  stiffness({vlasov_vars::field,   vlasov_vars::trace_v}) = zero_tt_matrix<3>(bilinear_f.core_cols(), bilinear_v.core_cols());
  stiffness({vlasov_vars::field,   vlasov_vars::field  }) = zero_tt_matrix<3>(bilinear_f.core_cols(), bilinear_f.core_cols());

  auto timing_stiffness = timer.toc();

  //
  // timing report
  //

  std::cout << "integrals" << std::endl
            << "  time = " << timing_integrals << " " << timer.units_string << std::endl << std::endl;

  std::cout << "bilinear_field" << std::endl
            << "  rows  = " << bilinear_f.core_rows() << " => " << bilinear_f.rows() << std::endl
            << "  cols  = " << bilinear_f.core_cols() << " => " << bilinear_f.cols() << std::endl
            << "  ranks = " << bilinear_f.ranks_string() << std::endl
            << "  time = " << timing_field << " " << timer.units_string << std::endl << std::endl;

  std::cout << "bilinear_trace" << std::endl
            << "  normal_x" << std::endl
            << "    rows  = " << bilinear_x.core_rows() << " => " << bilinear_x.rows() << std::endl
            << "    cols  = " << bilinear_x.core_cols() << " => " << bilinear_x.cols() << std::endl
            << "    ranks = " << bilinear_x.ranks_string() << std::endl
            << "  normal_t" << std::endl
            << "    rows  = " << bilinear_t.core_rows() << " => " << bilinear_t.rows() << std::endl
            << "    cols  = " << bilinear_t.core_cols() << " => " << bilinear_t.cols() << std::endl
            << "    ranks = " << bilinear_t.ranks_string() << std::endl
            << "  normal_v" << std::endl
            << "    rows  = " << bilinear_v.core_rows() << " => " << bilinear_v.rows() << std::endl
            << "    cols  = " << bilinear_v.core_cols() << " => " << bilinear_v.cols() << std::endl
            << "    ranks = " << bilinear_v.ranks_string() << std::endl
            << "  time = " << timing_trace << " " << timer.units_string << std::endl << std::endl;

  std::cout << "gram" << std::endl
            << "  rows  = " << gram.core_rows() << " => " << gram.rows() << std::endl
            << "  cols  = " << gram.core_cols() << " => " << gram.cols() << std::endl
            << "  ranks = " << gram.ranks_string() << std::endl
            << "  time = " << timing_gram << " " << timer.units_string << std::endl << std::endl;

  std::cout << "stiffness " << std::endl
            << "  rows  = " << stiffness.rows() << std::endl
            << "  cols  = " << stiffness.cols() << std::endl
            << "  time = " << timing_stiffness << " " << timer.units_string << std::endl << std::endl;

  std::cout << "total_timing " << std::endl
            << "  time = " << timing_integrals + timing_field + timing_trace + timing_gram + timing_stiffness << " " << timer.units_string << std::endl << std::endl;

  return stiffness;
}
