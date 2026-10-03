#ifndef REFERENCE_INTEGRALS_1D_HPP
#define REFERENCE_INTEGRALS_1D_HPP

#include "Basis1DIntegralCache.hpp"
#include "Camellia.h"

#include "boba_types.hpp"

//
// Basis function integrals over 1D reference elements
// A thin wrapper around Camellia's Basis1DIntegralCache
//
class reference_integrals_1d {
public:
  reference_integrals_1d() : integral_cache_(), quadrature_degree_(-1), layout_left_(false) {}

  ~reference_integrals_1d() = default;

  // Discretization of int x^m f1(x) f2(x) dx into 2D tensor
  host_tensor<2>
  discretization(Camellia::BasisPtr basis1, bool derivative1,
                 Camellia::BasisPtr basis2, bool derivative2,
                 int monomial_degree = 0);

  // Discretization of int x^m f1(x) f2(x) f3(x) dx into 3D tensor
  host_tensor<3>
  discretization(Camellia::BasisPtr basis1, bool derivative1,
                 Camellia::BasisPtr basis2, bool derivative2,
                 Camellia::BasisPtr basis3, bool derivative3,
                 int monomial_degree = 0);

  // Discretization of int x^m f1(x) f2(x) f3(x) f4(x) dx into 4D tensor
  host_tensor<4>
  discretization(Camellia::BasisPtr basis1, bool derivative1,
                 Camellia::BasisPtr basis2, bool derivative2,
                 Camellia::BasisPtr basis3, bool derivative3,
                 Camellia::BasisPtr basis4, bool derivative4,
                 int monomial_degree = 0);

private:
  Camellia::Basis1DIntegralCache<double> integral_cache_;
  int quadrature_degree_;
  bool layout_left_;
};

#endif
