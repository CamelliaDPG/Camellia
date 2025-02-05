#include "SpaceTimeMixedVlasovCompress.hpp"

using host_tensor_6d = boba::tensor<6, host_space, double, size_t>;
using host_tensor_12d = boba::tensor<12, host_space, double, size_t>;

std::tuple<boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size>,
           boba::block_vector<host_tt_vector, subvectors::size>,
           boba::block_vector<host_tt_vector, subvectors::size>>
compressBlockSystem(const boba::block_operator<host_matrix, subvectors::size, subvectors::size> &block_matrix,
                    const boba::block_vector<host_vector, subvectors::size> &block_rhs,
                    const boba::block_vector<host_vector, subvectors::size> &block_lhs,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute,
                    const double &svd_relative_tol, const double &svd_absolute_tol) {
  const size_t numFieldNode = polyOrder;
  const size_t numTraceNode = polyOrder + 1;
  const size_t numTestNode = polyOrder + polyOrderEnrich + 1;

  boba::Array<boba::Array<size_t, 6>, subvectors::size> original_sizes;
  //                                             0             1             2                3                4                5
  original_sizes[subvectors::epsw] = { numTestNode,  numTestNode,  numTestNode,     numElementX,     numElementT,     numElementV};
  original_sizes[subvectors::tnx]  = {           1, numTraceNode, numTraceNode, numElementX + 1,     numElementT,     numElementV};
  original_sizes[subvectors::tnt]  = {numTraceNode,            1, numTraceNode,     numElementX, numElementT + 1,     numElementV};
  original_sizes[subvectors::tnv]  = {numTraceNode, numTraceNode,            1,     numElementX,     numElementT, numElementV + 1};
  original_sizes[subvectors::f]    = {numFieldNode, numFieldNode, numFieldNode,     numElementX,     numElementT,     numElementV};

  boba::Array<boba::Array<size_t, 6>, subvectors::size> permuted_sizes;
  for (size_t row = 0; row < subvectors::size; ++row) {
    if (!permute) {
      permuted_sizes[row] = original_sizes[row];
    } else {
      permuted_sizes[row][0] = original_sizes[row][0];
      permuted_sizes[row][1] = original_sizes[row][3];
      permuted_sizes[row][2] = original_sizes[row][1];
      permuted_sizes[row][3] = original_sizes[row][4];
      permuted_sizes[row][4] = original_sizes[row][2];
      permuted_sizes[row][5] = original_sizes[row][5];
    }
  }

  // compress rhs blocks using temporary copies
  boba::block_vector<host_tt_vector, subvectors::size> block_tt_rhs;
  for(size_t row = 0; row < subvectors::size; row++) {
    host_tensor_6d temp;
    temp.resize(original_sizes[row]);
    temp.reshape(block_rhs(row));

    if (permute) {
      boba::permute(temp, {0, 3, 1, 4, 2, 5});
    }

    block_tt_rhs(row).setup(permuted_sizes[row]);
    block_tt_rhs(row).svd_tolerance_relative = svd_relative_tol;
    block_tt_rhs(row).svd_tolerance_absolute = svd_absolute_tol;
    block_tt_rhs(row).compress(temp);

    std::cout << "block_tt_rhs[" << row << "] CR " << block_tt_rhs(row).compression_rate() << " rank " << block_tt_rhs(row).ranks_string() << std::endl;
  }

  // compress lhs blocks using temporary copies
  boba::block_vector<host_tt_vector, subvectors::size> block_tt_lhs;
  for(size_t row = 0; row < subvectors::size; row++) {
    host_tensor_6d temp;
    temp.resize(original_sizes[row]);
    temp.reshape(block_lhs(row));

    if (permute) {
      boba::permute(temp, {0, 3, 1, 4, 2, 5});
    }

    block_tt_lhs(row).setup(permuted_sizes[row]);
    block_tt_lhs(row).svd_tolerance_relative = svd_relative_tol;
    block_tt_lhs(row).svd_tolerance_absolute = svd_absolute_tol;
    block_tt_lhs(row).compress(temp);

    std::cout << "block_tt_lhs[" << row << "] CR " << block_tt_lhs(row).compression_rate() << " rank " << block_tt_lhs(row).ranks_string() << std::endl;
  }

  // matrix blocks are [rhs, lhs] order
  boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size> block_tt_matrix;
  for (size_t row = 0; row < subvectors::size; ++row) {
    for (size_t col = 0; col < subvectors::size; ++col) {
      host_matrix temp;

      if (!permute) {
        temp = block_matrix({row, col});
      } else {
        boba::Array<size_t, 12> temp_size;
        for (size_t i = 0; i < 6; ++i) {
          temp_size[i] = original_sizes[row][i];
          temp_size[i + 6] = original_sizes[col][i];
        }

        host_tensor_12d temp1(temp_size);
        temp1.reshape(block_matrix({row, col}));
        boba::permute(temp1, {0, 3, 1, 4, 2, 5, 6 + 0, 6 + 3, 6 + 1, 6 + 4, 6 + 2, 6 + 5});

        temp.resize(block_matrix({row, col}).sizes());
        temp.reshape(temp1);
      }

      block_tt_matrix({row, col}).setup(permuted_sizes[row], permuted_sizes[col]);
      block_tt_matrix({row, col}).svd_tolerance_relative = svd_relative_tol;
      block_tt_matrix({row, col}).svd_tolerance_absolute = svd_absolute_tol;
      block_tt_matrix({row, col}).compress(temp);

      std::cout << "block_tt_matrix[" << row << "," << col << "] CR " << block_tt_matrix({row, col}).compression_rate() << " rank " << block_tt_matrix({row, col}).ranks_string() << std::endl;
    }
  }

  return {block_tt_matrix, block_tt_rhs, block_tt_lhs};
}

std::tuple<boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size>,
           boba::block_vector<host_tt_vector, subvectors::size>>
compressBlockSystem(const boba::block_operator<host_matrix, subvectors::size, subvectors::size> &block_matrix,
                    const boba::block_vector<host_vector, subvectors::size> &block_rhs,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute,
                    const double &svd_relative_tol, const double &svd_absolute_tol) {
  const size_t numFieldNode = polyOrder;
  const size_t numTraceNode = polyOrder + 1;
  const size_t numTestNode = polyOrder + polyOrderEnrich + 1;

  boba::Array<boba::Array<size_t, 6>, subvectors::size> original_sizes;
  //                                             0             1             2                3                4                5
  original_sizes[subvectors::epsw] = { numTestNode,  numTestNode,  numTestNode,     numElementX,     numElementT,     numElementV};
  original_sizes[subvectors::tnx]  = {           1, numTraceNode, numTraceNode, numElementX + 1,     numElementT,     numElementV};
  original_sizes[subvectors::tnt]  = {numTraceNode,            1, numTraceNode,     numElementX, numElementT + 1,     numElementV};
  original_sizes[subvectors::tnv]  = {numTraceNode, numTraceNode,            1,     numElementX,     numElementT, numElementV + 1};
  original_sizes[subvectors::f]    = {numFieldNode, numFieldNode, numFieldNode,     numElementX,     numElementT,     numElementV};

  boba::Array<boba::Array<size_t, 6>, subvectors::size> permuted_sizes;
  for (size_t row = 0; row < subvectors::size; ++row) {
    if (!permute) {
      permuted_sizes[row] = original_sizes[row];
    } else {
      permuted_sizes[row][0] = original_sizes[row][0];
      permuted_sizes[row][1] = original_sizes[row][3];
      permuted_sizes[row][2] = original_sizes[row][1];
      permuted_sizes[row][3] = original_sizes[row][4];
      permuted_sizes[row][4] = original_sizes[row][2];
      permuted_sizes[row][5] = original_sizes[row][5];
    }
  }

  // compress rhs blocks using temporary copies
  boba::block_vector<host_tt_vector, subvectors::size> block_tt_rhs;
  for(size_t row = 0; row < subvectors::size; row++) {
    host_tensor_6d temp;
    temp.resize(original_sizes[row]);
    temp.reshape(block_rhs(row));

    if (permute) {
      boba::permute(temp, {0, 3, 1, 4, 2, 5});
    }

    block_tt_rhs(row).setup(permuted_sizes[row]);
    block_tt_rhs(row).svd_tolerance_relative = svd_relative_tol;
    block_tt_rhs(row).svd_tolerance_absolute = svd_absolute_tol;
    block_tt_rhs(row).compress(temp);

    std::cout << "block_tt_rhs[" << row << "] CR " << block_tt_rhs(row).compression_rate() << " rank " << block_tt_rhs(row).ranks_string() << std::endl;
  }

  // matrix blocks are [rhs, lhs] order
  boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size> block_tt_matrix;
  for (size_t row = 0; row < subvectors::size; ++row) {
    for (size_t col = 0; col < subvectors::size; ++col) {
      host_matrix temp;

      if (!permute) {
        temp = block_matrix({row, col});
      } else {
        boba::Array<size_t, 12> temp_size;
        for (size_t i = 0; i < 6; ++i) {
          temp_size[i] = original_sizes[row][i];
          temp_size[i + 6] = original_sizes[col][i];
        }

        host_tensor_12d temp1(temp_size);
        temp1.reshape(block_matrix({row, col}));
        boba::permute(temp1, {0, 3, 1, 4, 2, 5, 6 + 0, 6 + 3, 6 + 1, 6 + 4, 6 + 2, 6 + 5});

        temp.resize(block_matrix({row, col}).sizes());
        temp.reshape(temp1);
      }

      block_tt_matrix({row, col}).setup(permuted_sizes[row], permuted_sizes[col]);
      block_tt_matrix({row, col}).svd_tolerance_relative = svd_relative_tol;
      block_tt_matrix({row, col}).svd_tolerance_absolute = svd_absolute_tol;
      block_tt_matrix({row, col}).compress(temp);

      std::cout << "block_tt_matrix[" << row << "," << col << "] CR " << block_tt_matrix({row, col}).compression_rate() << " rank " << block_tt_matrix({row, col}).ranks_string() << std::endl;
    }
  }

  return {block_tt_matrix, block_tt_rhs};
}

boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size>
compressBlockSystem(const boba::block_operator<host_matrix, subvectors::size, subvectors::size> &block_matrix,
                    const size_t &numElementX, const size_t &numElementT, const size_t &numElementV,
                    const size_t &polyOrder, const size_t &polyOrderEnrich,
                    const bool &permute,
                    const double &svd_relative_tol, const double &svd_absolute_tol) {
  const size_t numFieldNode = polyOrder;
  const size_t numTraceNode = polyOrder + 1;
  const size_t numTestNode = polyOrder + polyOrderEnrich + 1;

  boba::Array<boba::Array<size_t, 6>, subvectors::size> original_sizes;
  //                                             0             1             2                3                4                5
  original_sizes[subvectors::epsw] = { numTestNode,  numTestNode,  numTestNode,     numElementX,     numElementT,     numElementV};
  original_sizes[subvectors::tnx]  = {           1, numTraceNode, numTraceNode, numElementX + 1,     numElementT,     numElementV};
  original_sizes[subvectors::tnt]  = {numTraceNode,            1, numTraceNode,     numElementX, numElementT + 1,     numElementV};
  original_sizes[subvectors::tnv]  = {numTraceNode, numTraceNode,            1,     numElementX,     numElementT, numElementV + 1};
  original_sizes[subvectors::f]    = {numFieldNode, numFieldNode, numFieldNode,     numElementX,     numElementT,     numElementV};

  boba::Array<boba::Array<size_t, 6>, subvectors::size> permuted_sizes;
  for (size_t row = 0; row < subvectors::size; ++row) {
    if (!permute) {
      permuted_sizes[row] = original_sizes[row];
    } else {
      permuted_sizes[row][0] = original_sizes[row][0];
      permuted_sizes[row][1] = original_sizes[row][3];
      permuted_sizes[row][2] = original_sizes[row][1];
      permuted_sizes[row][3] = original_sizes[row][4];
      permuted_sizes[row][4] = original_sizes[row][2];
      permuted_sizes[row][5] = original_sizes[row][5];
    }
  }

  // matrix blocks are [rhs, lhs] order
  boba::block_operator<host_tt_matrix, subvectors::size, subvectors::size> block_tt_matrix;
  for (size_t row = 0; row < subvectors::size; ++row) {
    for (size_t col = 0; col < subvectors::size; ++col) {
      host_matrix temp;

      if (!permute) {
        temp = block_matrix({row, col});
      } else {
        boba::Array<size_t, 12> temp_size;
        for (size_t i = 0; i < 6; ++i) {
          temp_size[i] = original_sizes[row][i];
          temp_size[i + 6] = original_sizes[col][i];
        }

        host_tensor_12d temp1(temp_size);
        temp1.reshape(block_matrix({row, col}));
        boba::permute(temp1, {0, 3, 1, 4, 2, 5, 6 + 0, 6 + 3, 6 + 1, 6 + 4, 6 + 2, 6 + 5});

        temp.resize(block_matrix({row, col}).sizes());
        temp.reshape(temp1);
      }

      block_tt_matrix({row, col}).setup(permuted_sizes[row], permuted_sizes[col]);
      block_tt_matrix({row, col}).svd_tolerance_relative = svd_relative_tol;
      block_tt_matrix({row, col}).svd_tolerance_absolute = svd_absolute_tol;
      block_tt_matrix({row, col}).compress(temp);

      std::cout << "block_tt_matrix[" << row << "," << col << "] CR " << block_tt_matrix({row, col}).compression_rate() << " rank " << block_tt_matrix({row, col}).ranks_string() << std::endl;
    }
  }

  return block_tt_matrix;
}
