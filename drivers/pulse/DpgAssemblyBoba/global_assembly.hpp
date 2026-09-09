#ifndef DPG_BOBA_GLOBAL_ASSEMBLY_HPP
#define DPG_BOBA_GLOBAL_ASSEMBLY_HPP

#include "boba_types.hpp"

enum class bc_kind_1d {
  none,
  left,
  right,
  both
};

//
// Global assembly of (w, f)-type terms over 1D, where f is a field variable
// defined over the element
//
// input_x0          Discretization of (w, f) over reference element
// x_min, x_max, nx  Parameters for 1D uniform mesh
// num_derivative    Total number of derivatives over w and f terms, e.g.
//                   - (w', f) and (w, f') have num_derivative = 1
//                   - (w'', f) and (w', f') have num_derivative = 2
//
host_matrix
global_field_assembly_1d(const host_tensor<2> &input_x0,
                         const size_t &nx, const double &x_min, const double &x_max,
                         const size_t &num_derivative = 0);

//
// Global assembly of (w, x f)-type terms over 1D, where f is a field variable
// defined over the element
//
// input_x0          Discretization of (w, f) over reference element
// input_x1          Discretization of (w, x f) over reference element
// x_min, x_max, nx  Parameters for 1D uniform mesh
// num_derivative    Total number of derivatives over w and f terms, e.g.
//                   - (w', f) and (w, f') have num_derivative = 1
//                   - (w'', f) and (w', f') have num_derivative = 2
//
host_matrix
global_field_assembly_1d(const host_tensor<2> &input_x0,
                         const host_tensor<2> &input_x1,
                         const size_t &nx, const double &x_min, const double &x_max,
                         const size_t &num_derivative = 0);

//
// Global assembly of (w, x^2 f)-type terms over 1D, where f is a field variable
// defined over the element
//
// input_x0          Discretization of (w, f) over reference element
// input_x1          Discretization of (w, x f) over reference element
// input_x2          Discretization of (w, x^2 f) over reference element
// x_min, x_max, nx  Parameters for 1D uniform mesh
// num_derivative    Total number of derivatives over w and f terms, e.g.
//                   - (w', f) and (w, f') have num_derivative = 1
//                   - (w'', f) and (w', f') have num_derivative = 2
//
host_matrix
global_field_assembly_1d(const host_tensor<2> &input_x0,
                         const host_tensor<2> &input_x1,
                         const host_tensor<2> &input_x2,
                         const size_t &nx, const double &x_min, const double &x_max,
                         const size_t &num_derivative = 0);

//
// Global assembly of (w, f_hat)-type terms over 1D, where f_hat is a trace
// variable defined over element boundary points
//
// nb  Number of basis functions for the test variable w
// nx  Number of elements
// bc  Dirichlet boundary condition type to apply; for now, basically drops the
//     corresponding DOFs
//
host_matrix
global_trace_assembly_1d(const size_t &nb, const size_t &nx, const bc_kind_1d &bc = bc_kind_1d::both);

#endif
