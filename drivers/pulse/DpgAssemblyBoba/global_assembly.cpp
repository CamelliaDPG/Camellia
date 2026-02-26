#include "global_assembly.hpp"

#include "Teuchos_TestForException.hpp"

namespace {
  std::pair<int,int> lineEndpointDofOrdinals(Camellia::BasisPtr basis)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(basis == Teuchos::null, std::invalid_argument,
                               "lineEndpointDofOrdinals: basis is null");

    auto i2Basis = basis->intrepid2Basis();
    TEUCHOS_TEST_FOR_EXCEPTION(i2Basis == Teuchos::null, std::invalid_argument,
                               "lineEndpointDofOrdinals: intrepid2Basis() returned null");

    const int N = i2Basis->getCardinality();

    int left  = -1; // vertex 0
    int right = -1; // vertex 1

    for (int bfOrdinal = 0; bfOrdinal < N; ++bfOrdinal)
    {
      // Tag layout: [subcellDim, subcellOrd, dofOrdOnSubcell, totalDofsOnSubcell]
      auto tag = i2Basis->getDofTag(bfOrdinal);

      const int subcellDim       = tag[0];
      const int subcellOrd       = tag[1];
      const int dofOrdOnSubcell  = tag[2];

      if ((subcellDim == 0) && (dofOrdOnSubcell == 0))
      {
        if (subcellOrd == 0) left  = bfOrdinal;
        if (subcellOrd == 1) right = bfOrdinal;
      }
    }

    return {left, right};
  }
}

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

  const auto input_x0_view = input_x0.view();
  const auto input_x1_view = input_x1.view();

  auto output_view = output.view();
  for (size_t iv = 0; iv < nx; ++iv) {
    const double xmid = x_min + (iv + 0.5) * dx;
    for (size_t icol = 0; icol < ncol; ++icol) {
      for (size_t irow = 0; irow < nrow; ++irow) {
        output_view({iv * nrow + irow, iv * ncol + icol}) = scale * (0.5 *   dx * input_x1_view({irow, icol}) +
                                                                           xmid * input_x0_view({irow, icol}));
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
global_trace_assembly_1d(Camellia::BasisPtr w_basis, const size_t &nx, const bc_kind_1d &bc)
{
  TEUCHOS_TEST_FOR_EXCEPTION(w_basis == Teuchos::null, std::invalid_argument,
                             "global_trace_assembly_1d: w_basis is null");

  const size_t nb = w_basis->getCardinality();

  const bool truncate_left  = (bc == bc_kind_1d::both || bc == bc_kind_1d::left );
  const bool truncate_right = (bc == bc_kind_1d::both || bc == bc_kind_1d::right);

  const size_t start = truncate_left  ?      1 :  0;
  const size_t stop  = truncate_right ? nx - 1 : nx;

  const size_t numTraces = stop + 1 - start;
  const size_t numCols   = std::max<size_t>(1, numTraces);

  // Determine which basis ordinals correspond to the left/right endpoints on the line
  auto [leftOrd, rightOrd] = lineEndpointDofOrdinals(w_basis);

  TEUCHOS_TEST_FOR_EXCEPTION(leftOrd < 0 || rightOrd < 0, std::invalid_argument,
                             "global_trace_assembly_1d: could not find endpoint (vertex) dof ordinals "
                             "for the provided basis. This routine requires a basis with vertex dofs "
                             "(e.g., HGRAD on a line).");

  TEUCHOS_TEST_FOR_EXCEPTION((size_t)leftOrd >= nb || (size_t)rightOrd >= nb, std::logic_error,
                             "global_trace_assembly_1d: endpoint dof ordinal out of range.");

  host_matrix output({nx * nb, numCols});
  output.fill_with_zeros();

  auto output_view = output.view();
  for (size_t ix = 0; ix < nx; ++ix)
  {
    if (!(truncate_left && ix == 0))
    {
      output_view({ix * nb + (size_t)leftOrd, ix - start}) = -1.0;
    }
    if (!(truncate_right && ix + 1 == nx))
    {
      output_view({ix * nb + (size_t)rightOrd, ix + 1 - start}) =  1.0;
    }
  }

  return output;
}
