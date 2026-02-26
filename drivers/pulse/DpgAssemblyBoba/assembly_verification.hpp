#ifndef ASSEMBLY_VERIFICATION_HPP
#define ASSEMBLY_VERIFICATION_HPP

#include "Camellia.h"
#include "boba_types.hpp"
#include <functional>

host_matrix assemble_mass_bruteforce(Camellia::BasisPtr basis,
                                     std::size_t ne,
                                     double x_min,
                                     double x_max,
                                     int cubatureDegree = -1);

host_matrix
assemble_v_weighted_bruteforce(Camellia::BasisPtr w_basis,
                               Camellia::BasisPtr f_basis,
                               std::size_t nv,
                               double v_min,
                               double v_max,
                               int cubatureDegree = -1);

void
test_global_field_assembly_v_weighted(Camellia::BasisPtr w_basis,
                                      Camellia::BasisPtr f_basis,
                                      std::size_t nv,
                                      double v_min,
                                      double v_max);

void test_1d_mass_consistency(Camellia::BasisPtr basis,
                              std::size_t ne,
                              double x_min,
                              double x_max);

host_vector
assemble_mixed_rhs_bruteforce(Camellia::BasisPtr w_basis,
                              std::size_t ne,
                              double x_min,
                              double x_max,
                              const std::function<double(double)>& g,
                              int cubatureDegree = -1);

host_vector assemble_rhs_bruteforce(Camellia::BasisPtr basis,
                                   std::size_t ne,
                                   double x_min,
                                   double x_max,
                                   const std::function<double(double)>& f,
                                   int cubatureDegree = -1);

host_vector
matvec_dense(const host_matrix& A, const host_vector& x);

void test_1d_moments_vs_coeffs(Camellia::BasisPtr basis);

void
test_1d_trace_coupling_consistency(Camellia::BasisPtr w_basis,
                                  Camellia::BasisPtr t_basis,
                                  std::size_t ne,
                                  double x_min,
                                  double x_max);

void test_boba_tt_vector_ordering();
void test_boba_ttm_mode_action();

#endif
