#ifndef DPG_BOBA_VLASOV_SYSTEM_HPP
#define DPG_BOBA_VLASOV_SYSTEM_HPP

#include "TypeDefs.h"

#include "boba_types.hpp"

struct vlasov_vars {
  static constexpr size_t test    = 0;
  static constexpr size_t trace_x = 1;
  static constexpr size_t trace_t = 2;
  static constexpr size_t trace_v = 3;
  static constexpr size_t field   = 4;
  static constexpr size_t size    = 5;
};


//
// Assemble the vlasov system for E = 1 on a uniform mesh over Cartesian x-t-v domain
//
// {test,trace,field}_basis            1D basis function families for different variables
// {x,t,v}_min, {x,t,v}_max, n{x,t,v}  Extents of the Cartesian domain
// p                                   Polynomial order for field variables
// dp                                  Polynomial order enrichment for test variables
//
boba::BlockOperator<host_tt_matrix<3>>
construct_vlasov_matrices(Camellia::BasisPtr test_basis, Camellia::BasisPtr trace_basis, Camellia::BasisPtr field_basis,
                          const double &x_min, const double &x_max, const size_t &nx,
                          const double &t_min, const double &t_max, const size_t &nt,
                          const double &v_min, const double &v_max, const size_t &nv,
                          const size_t &p, const size_t &dp);

#endif
