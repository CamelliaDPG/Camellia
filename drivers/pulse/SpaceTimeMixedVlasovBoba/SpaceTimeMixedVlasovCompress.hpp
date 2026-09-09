#ifndef SPACE_TIME_MIXED_VLASOV_COMPRESS_HPP
#define SPACE_TIME_MIXED_VLASOV_COMPRESS_HPP

#include "BOBA/boba.hpp"
#include "SpaceTimeMixedVlasovSetup.hpp"

constexpr auto host_space = boba::host_space;

using host_matrix = boba::Matrix<host_space, double>;
using host_vector = boba::Vector<host_space, double>;

using host_tt_matrix = boba::TensorTrainMatrix<6, host_space, double>;
using host_tt_vector = boba::TensorTrain<6, host_space, double>;

std::tuple<boba::BlockOperator<host_tt_matrix>,
           boba::BlockVector<host_tt_vector>,
           boba::BlockVector<host_tt_vector>>
compressBlockSystem(const boba::BlockOperator<host_matrix> &block_matrix,
                    const boba::BlockVector<host_vector> &block_rhs,
                    const boba::BlockVector<host_vector> &block_lhs,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute = false,
                    const double &svd_relative_tol = 1.0e-12, const double &svd_absolute_tol = 1.0e-12);

std::tuple<boba::BlockOperator<host_tt_matrix>,
           boba::BlockVector<host_tt_vector>>
compressBlockSystem(const boba::BlockOperator<host_matrix> &block_matrix,
                    const boba::BlockVector<host_vector> &block_rhs,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute = false,
                    const double &svd_relative_tol = 1.0e-12, const double &svd_absolute_tol = 1.0e-12);

boba::BlockOperator<host_tt_matrix>
compressBlockSystem(const boba::BlockOperator<host_matrix> &block_matrix,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute = false,
                    const double &svd_relative_tol = 1.0e-12, const double &svd_absolute_tol = 1.0e-12);

#endif
