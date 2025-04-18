#include "global_assembly.hpp"

host_matrix
global_field_assembly_1d(const host_tensor<2> &input_x0,
                         const size_t &nx, const double &x_min, const double &x_max,
                         const size_t &num_derivative) {
  const double dx = (x_max - x_min) / nx;
  const double scale = 0.5 * dx / std::pow(0.5 * dx, num_derivative);

  const size_t nrow = input_x0.sizes(0);
  const size_t ncol = input_x0.sizes(1);

  host_matrix output({nx * nrow, nx * ncol});
  output.fill_with_zeros();

  const auto input_x0_view = input_x0.view();
  auto output_view = output.view();
  for (size_t icopy = 0; icopy < nx; ++icopy) {
    for (size_t icol = 0; icol < ncol; ++icol) {
      for (size_t irow = 0; irow < nrow; ++irow) {
        output_view({icopy * nrow + irow, icopy * ncol + icol}) = scale * input_x0_view({irow, icol});
      }
    }
  }

  return output;
}

host_matrix
global_field_assembly_1d(const host_tensor<2> &input_x0,
                         const host_tensor<2> &input_x1,
                         const size_t &nx, const double &x_min, const double &x_max,
                         const size_t &num_derivative) {
  const double dx = (x_max - x_min) / nx;
  const double scale = 0.5 * dx / std::pow(0.5 * dx, num_derivative);

  const size_t nrow = input_x0.sizes(0);
  const size_t ncol = input_x0.sizes(1);

  host_matrix output({nx * nrow, nx * ncol});
  output.fill_with_zeros();

  const auto input_x0_xiew = input_x0.view();
  const auto input_x1_view = input_x1.view();

  auto output_view = output.view();
  for (size_t iv = 0; iv < nx; ++iv) {
    const double xmid = x_min + (iv + 0.5) * dx;
    for (size_t icol = 0; icol < ncol; ++icol) {
      for (size_t irow = 0; irow < nrow; ++irow) {
        output_view({iv * nrow + irow, iv * ncol + icol}) = scale * (0.5 *   dx * input_x1_view({irow, icol}) +
                                                                           xmid * input_x0_xiew({irow, icol}));
      }
    }
  }

  return output;
}

host_matrix
global_field_assembly_1d(const host_tensor<2> &input_x0,
                         const host_tensor<2> &input_x1,
                         const host_tensor<2> &input_x2,
                         const size_t &nx, const double &x_min, const double &x_max,
                         const size_t &num_derivative) {
  const double dx = (x_max - x_min) / nx;
  const double scale = 0.5 * dx / std::pow(0.5 * dx, num_derivative);

  const size_t nrow = input_x0.sizes(0);
  const size_t ncol = input_x0.sizes(1);

  host_matrix output({nx * nrow, nx * ncol});
  output.fill_with_zeros();

  const auto input_x0_view = input_x0.view();
  const auto input_x1_view = input_x1.view();
  const auto input_x2_view = input_x2.view();

  auto output_view = output.view();
  for (size_t iv = 0; iv < nx; ++iv) {
    const double xmid = x_min + (iv + 0.5) * dx;
    for (size_t icol = 0; icol < ncol; ++icol) {
      for (size_t irow = 0; irow < nrow; ++irow) {
        output_view({iv * nrow + irow, iv * ncol + icol}) = scale * (0.25 *   dx *   dx * input_x2_view({irow, icol}) +
                                                                              dx * xmid * input_x1_view({irow, icol}) +
                                                                            xmid * xmid * input_x0_view({irow, icol}));
      }
    }
  }

  return output;
}

host_matrix
global_trace_assembly_1d(const size_t &nb, const size_t &nx, const bc_kind_1d &bc) {
  const bool truncate_left  = (bc == bc_kind_1d::both || bc == bc_kind_1d::left );
  const bool truncate_right = (bc == bc_kind_1d::both || bc == bc_kind_1d::right);

  const size_t start = truncate_left  ?      1 :  0;
  const size_t stop  = truncate_right ? nx - 1 : nx;

  host_matrix output({nx * nb, stop + 1 - start});
  output.fill_with_zeros();

  auto output_view = output.view();
  for (size_t ix = 0; ix < nx; ++ix) {
    if (!(truncate_left && ix == 0)) {
      output_view({ix * nb, ix - start}) = -1.0;
    }
    if (!(truncate_right && ix + 1 == nx)) {
      output_view({ix * nb + nb - 1, ix + 1 - start}) =  1.0;
    }
  }

  return output;
}
