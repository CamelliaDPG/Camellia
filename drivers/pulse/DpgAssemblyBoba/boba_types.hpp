#ifndef BOBA_TYPES_HPP
#define BOBA_TYPES_HPP

#include "BOBA/boba.hpp"

constexpr auto host_space = ::boba::execution_space::CPU;

using host_vector = ::boba::vector<host_space, double, std::size_t>;

using host_matrix = ::boba::matrix<host_space, double, std::size_t>;

template <std::size_t dimension>
using host_tensor = ::boba::tensor<dimension, host_space, double, std::size_t>;

template <std::size_t dimension>
using host_tt_vector = ::boba::tensor_train<dimension, host_space, double, std::size_t>;

template <std::size_t dimension>
using host_tt_matrix = ::boba::tensor_train_matrix<dimension, host_space, double, std::size_t>;

template <std::size_t dimension>
host_tt_vector<dimension>
zero_tt_vector(::boba::Array<size_t, dimension> sizes) {
  host_tt_vector<dimension> vector(sizes);
  vector.fill_with_zeros();
  return vector;
}

template <std::size_t dimension>
host_tt_matrix<dimension>
zero_tt_matrix(::boba::Array<size_t, dimension> rows, ::boba::Array<size_t, dimension> cols) {
  host_tt_matrix<dimension> matrix(rows, cols);
  matrix.fill_with_zeros();
  return matrix;
}

#endif
