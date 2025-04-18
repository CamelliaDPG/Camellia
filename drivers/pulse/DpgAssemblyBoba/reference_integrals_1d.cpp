#include "reference_integrals_1d.hpp"

host_tensor<2>
reference_integrals_1d::discretization(Camellia::BasisPtr basis1, bool derivative1,
                                       Camellia::BasisPtr basis2, bool derivative2,
                                       int monomial_degree) {
  const std::size_t num_dofs1 = basis1->getCardinality();
  const std::size_t num_dofs2 = basis2->getCardinality();

  auto integral = integral_cache_.getReferenceSpaceIntegral(basis1, derivative1, basis2, derivative2,
                                                            monomial_degree, quadrature_degree_, layout_left_);
  auto integral_data = integral.data();

  host_tensor<2> integral_values({num_dofs1, num_dofs2});
  auto integral_values_view = integral_values.view();
  for (std::size_t j = 0; j < num_dofs2; ++j) {
    for (std::size_t i = 0; i < num_dofs1; ++i) {
      integral_values_view({i, j}) = integral_data[i * num_dofs2 + j];
    }
  }

  return integral_values;
}

host_tensor<3>
reference_integrals_1d::discretization(Camellia::BasisPtr basis1, bool derivative1,
                                       Camellia::BasisPtr basis2, bool derivative2,
                                       Camellia::BasisPtr basis3, bool derivative3,
                                       int monomial_degree) {
  const std::size_t num_dofs1 = basis1->getCardinality();
  const std::size_t num_dofs2 = basis2->getCardinality();
  const std::size_t num_dofs3 = basis3->getCardinality();

  auto integral = integral_cache_.getReferenceSpaceIntegral(basis1, derivative1, basis2, derivative2,
                                                            basis3, derivative3,
                                                            monomial_degree, quadrature_degree_, layout_left_);
  auto integral_data = integral.data();

  host_tensor<3> integral_values({num_dofs1, num_dofs2, num_dofs3});
  auto integral_values_view = integral_values.view();
  for (std::size_t k = 0; k < num_dofs3; ++k) {
    for (std::size_t j = 0; j < num_dofs2; ++j) {
      for (std::size_t i = 0; i < num_dofs1; ++i) {
        integral_values_view({i, j, k}) = integral_data[i * num_dofs2 * num_dofs3 + j * num_dofs3 + k];
      }
    }
  }

  return integral_values;
}

host_tensor<4>
reference_integrals_1d::discretization(Camellia::BasisPtr basis1, bool derivative1,
                                       Camellia::BasisPtr basis2, bool derivative2,
                                       Camellia::BasisPtr basis3, bool derivative3,
                                       Camellia::BasisPtr basis4, bool derivative4,
                                       int monomial_degree) {
  const std::size_t num_dofs1 = basis1->getCardinality();
  const std::size_t num_dofs2 = basis2->getCardinality();
  const std::size_t num_dofs3 = basis3->getCardinality();
  const std::size_t num_dofs4 = basis4->getCardinality();

  auto integral = integral_cache_.getReferenceSpaceIntegral(basis1, derivative1, basis2, derivative2,
                                                            basis3, derivative3, basis4, derivative4,
                                                            monomial_degree, quadrature_degree_, layout_left_);
  auto integral_data = integral.data();

  host_tensor<4> integral_values({num_dofs1, num_dofs2, num_dofs3, num_dofs4});
  auto integral_values_view = integral_values.view();
  for (std::size_t l = 0; l < num_dofs4; ++l) {
    for (std::size_t k = 0; k < num_dofs3; ++k) {
      for (std::size_t j = 0; j < num_dofs2; ++j) {
        for (std::size_t i = 0; i < num_dofs1; ++i) {
          integral_values_view({i, j, k, l}) = integral_data[i * num_dofs2 * num_dofs3 * num_dofs4 +
                                                             j * num_dofs3 * num_dofs4 + k * num_dofs4 + l];
        }
      }
    }
  }

  return integral_values;
}
