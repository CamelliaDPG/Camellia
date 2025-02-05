#ifndef SPACE_TIME_MIXED_VLASOV_COMPRESS_HPP
#define SPACE_TIME_MIXED_VLASOV_COMPRESS_HPP

#include "BOBA/boba.hpp"
#include "SpaceTimeMixedVlasovSetup.hpp"

constexpr auto host_space = boba::execution_space::CPU;

using host_matrix = boba::matrix<host_space, double, size_t>;
using host_vector = boba::vector<host_space, double, size_t>;

using host_tt_matrix = boba::tensor_train_matrix<6, host_space, double, size_t>;
using host_tt_vector = boba::tensor_train<6, host_space, double, size_t>;

std::tuple<boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size>,
           boba::block_vector<host_tt_vector, subvectors::size>,
           boba::block_vector<host_tt_vector, subvectors::size>>
compressBlockSystem(const boba::block_operator<host_matrix, subvectors::size, subvectors::size> &block_matrix,
                    const boba::block_vector<host_vector, subvectors::size> &block_rhs,
                    const boba::block_vector<host_vector, subvectors::size> &block_lhs,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute = false,
                    const double &svd_relative_tol = 1.0e-12, const double &svd_absolute_tol = 1.0e-12);

std::tuple<boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size>,
           boba::block_vector<host_tt_vector, subvectors::size>>
compressBlockSystem(const boba::block_operator<host_matrix, subvectors::size, subvectors::size> &block_matrix,
                    const boba::block_vector<host_vector, subvectors::size> &block_rhs,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute = false,
                    const double &svd_relative_tol = 1.0e-12, const double &svd_absolute_tol = 1.0e-12);

boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size>
compressBlockSystem(const boba::block_operator<host_matrix, subvectors::size, subvectors::size> &block_matrix,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute = false,
                    const double &svd_relative_tol = 1.0e-12, const double &svd_absolute_tol = 1.0e-12);

#endif
