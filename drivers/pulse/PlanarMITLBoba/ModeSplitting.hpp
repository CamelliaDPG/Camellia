#pragma once

#include "TensorProductAssembly.hpp"
#include <Eigen/Dense>

namespace PlanarMITLBoba
{
/// Split each (cell*basis) mode into a basis mode and a cell mode, basis first, so that the linear index of every
/// entry is unchanged.  The four-mode assembly orders each axis as cell*nb + b (basis fastest); the split modes are
/// (x_b, x_c, t_b, t_c, ux_b, ux_c, uz_b, uz_c).  Each core is factored by a truncated SVD across the basis/cell cut.
/// Local block-AMEn systems then scale with the number of cells or the number of basis functions in one direction,
/// not with their product.
using SplitTTMatrix = host_tt_matrix<8>;
using SplitTTVector = host_tt_vector<8>;
using SplitBlockMatrix = boba::BlockOperator<SplitTTMatrix>;
using SplitBlockVector = boba::BlockVector<SplitTTVector>;

/// Per-axis local basis count of each block: the mode size divided by the number of cells.
struct ModeSplit
{
  std::vector<std::array<size_t, 4>> basisSizes; // [block][axis]
};

namespace detail
{
/// Truncated SVD, M ~ U S V^T, keeping singular values above tolerance * s_max (at least one).
inline void truncatedSVD(const Eigen::MatrixXd &M, double tolerance, Eigen::MatrixXd &left, Eigen::MatrixXd &right)
{
  Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto &s = svd.singularValues();
  Eigen::Index rank = 0;
  while (rank < s.size() && s(rank) > tolerance * s(0)) { ++rank; }
  rank = std::max<Eigen::Index>(rank, 1);
  left = svd.matrixU().leftCols(rank) * s.head(rank).asDiagonal();
  right = svd.matrixV().leftCols(rank).transpose();
}
} // namespace detail

/// Split a four-mode TT matrix whose rows/cols per axis are (cells * rowBasis[d], cells * colBasis[d]).
inline SplitTTMatrix splitMatrix(const TTMatrix &matrix, const std::array<size_t, 4> &rowBasis,
                                 const std::array<size_t, 4> &colBasis, double tolerance)
{
  boba::Array<size_t, 8> rows, cols;
  std::array<Eigen::MatrixXd, 8> cores; // flattened (rL * n * m, rR) in BoBa's column-major core order
  std::array<size_t, 9> ranks;
  ranks[0] = 1;
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &core = matrix.cores[d];
    const size_t rL = core.sizes(0), n = core.sizes(1), m = core.sizes(2), rR = core.sizes(3);
    const size_t nb = rowBasis[d], mb = colBasis[d];
    const size_t nc = n / nb, mc = m / mb;
    if (nb * nc != n || mb * mc != m)
    {
      throw std::invalid_argument("splitMatrix: mode size is not cells * basis");
    }
    auto view = core.const_view();
    // Rows (rL, b, b'), columns (c, c', rR): entry C[rL, b + nb*c, b' + mb*c', rR].
    Eigen::MatrixXd M(rL * nb * mb, nc * mc * rR);
    for (size_t r = 0; r < rR; ++r)
      for (size_t cp = 0; cp < mc; ++cp)
        for (size_t c = 0; c < nc; ++c)
          for (size_t bp = 0; bp < mb; ++bp)
            for (size_t b = 0; b < nb; ++b)
              for (size_t l = 0; l < rL; ++l)
              {
                M(l + rL * (b + nb * bp), c + nc * (cp + mc * r)) = view({l, b + nb * c, bp + mb * cp, r});
              }
    Eigen::MatrixXd left, right;
    detail::truncatedSVD(M, tolerance, left, right);
    const size_t k = left.cols();
    rows[2 * d] = nb; cols[2 * d] = mb;
    rows[2 * d + 1] = nc; cols[2 * d + 1] = mc;
    ranks[2 * d + 1] = k;
    ranks[2 * d + 2] = rR;
    cores[2 * d] = left;   // (rL, b, b') x k
    cores[2 * d + 1] = right; // k x (c, c', rR)
  }
  SplitTTMatrix result(rows, cols);
  for (size_t e = 0; e < 8; ++e)
  {
    const size_t rL = ranks[e], n = rows[e], m = cols[e], rR = ranks[e + 1];
    result.cores[e].resize({rL, n, m, rR});
    auto out = result.cores[e].view();
    const auto &F = cores[e];
    for (size_t r = 0; r < rR; ++r)
      for (size_t j = 0; j < m; ++j)
        for (size_t i = 0; i < n; ++i)
          for (size_t l = 0; l < rL; ++l)
          {
            // even cores: row index (l, i, j), column r;  odd cores: row l, column (i, j, r)
            out({l, i, j, r}) = (e % 2 == 0) ? F(l + rL * (i + n * j), r) : F(l, i + n * (j + m * r));
          }
  }
  return result;
}

/// Split a four-mode TT vector whose per-axis size is cells * basis[d].
inline SplitTTVector splitVector(const TTVector &vector, const std::array<size_t, 4> &basis, double tolerance)
{
  boba::Array<size_t, 8> sizes;
  std::array<Eigen::MatrixXd, 8> cores;
  std::array<size_t, 9> ranks;
  ranks[0] = 1;
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &core = vector.cores[d];
    const size_t rL = core.sizes(0), n = core.sizes(1), rR = core.sizes(2);
    const size_t nb = basis[d], nc = n / nb;
    if (nb * nc != n) { throw std::invalid_argument("splitVector: mode size is not cells * basis"); }
    auto view = core.const_view();
    Eigen::MatrixXd M(rL * nb, nc * rR);
    for (size_t r = 0; r < rR; ++r)
      for (size_t c = 0; c < nc; ++c)
        for (size_t b = 0; b < nb; ++b)
          for (size_t l = 0; l < rL; ++l)
          {
            M(l + rL * b, c + nc * r) = view({l, b + nb * c, r});
          }
    Eigen::MatrixXd left, right;
    detail::truncatedSVD(M, tolerance, left, right);
    sizes[2 * d] = nb; sizes[2 * d + 1] = nc;
    ranks[2 * d + 1] = left.cols(); ranks[2 * d + 2] = rR;
    cores[2 * d] = left; cores[2 * d + 1] = right;
  }
  SplitTTVector result(sizes);
  for (size_t e = 0; e < 8; ++e)
  {
    const size_t rL = ranks[e], n = sizes[e], rR = ranks[e + 1];
    result.cores[e].resize({rL, n, rR});
    auto out = result.cores[e].view();
    const auto &F = cores[e];
    for (size_t r = 0; r < rR; ++r)
      for (size_t i = 0; i < n; ++i)
        for (size_t l = 0; l < rL; ++l)
        {
          out({l, i, r}) = (e % 2 == 0) ? F(l + rL * i, r) : F(l, i + n * r);
        }
  }
  return result;
}

/// Merge (basis, cell) mode pairs back into four (cell*basis) modes, exactly.
inline TTVector mergeVector(const SplitTTVector &vector)
{
  Sizes sizes;
  for (size_t d = 0; d < 4; ++d) { sizes[d] = vector.sizes(2 * d) * vector.sizes(2 * d + 1); }
  TTVector result(sizes);
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &A = vector.cores[2 * d], &B = vector.cores[2 * d + 1];
    const size_t rL = A.sizes(0), nb = A.sizes(1), k = A.sizes(2), nc = B.sizes(1), rR = B.sizes(2);
    auto a = A.const_view();
    auto b = B.const_view();
    result.cores[d].resize({rL, nb * nc, rR});
    auto out = result.cores[d].view();
    for (size_t r = 0; r < rR; ++r)
      for (size_t c = 0; c < nc; ++c)
        for (size_t i = 0; i < nb; ++i)
          for (size_t l = 0; l < rL; ++l)
          {
            double sum = 0.0;
            for (size_t q = 0; q < k; ++q) { sum += a({l, i, q}) * b({q, c, r}); }
            out({l, i + nb * c, r}) = sum;
          }
  }
  return result;
}

/// Local basis counts per axis: mode size / cells.
inline std::array<size_t, 4> basisCounts(const Sizes &sizes, const std::vector<Axis> &axes)
{
  std::array<size_t, 4> result;
  for (size_t d = 0; d < 4; ++d)
  {
    const size_t cells = axes[d].vertices.size() - 1;
    // A one-cell direction's flux block has no faces in that direction and is absent; other flux blocks have
    // (cells - 1) faces along their normal, which is not cells * basis.  Treat the face mode as "cells" faces of one
    // basis function each when it does not divide.
    result[d] = (sizes[d] % cells == 0) ? sizes[d] / cells : 1;
  }
  return result;
}

/// Split every block of the mixed system and the RHS.  Flux blocks along their normal have (cells - 1) faces: that
/// mode is split as (1 basis) x (faces).
inline void splitSystem(const System &system, const BlockVector &rhs, const std::vector<Axis> &axes, double tolerance,
                        SplitBlockMatrix &matrix, SplitBlockVector &splitRHS, ModeSplit &split)
{
  const size_t blocks = system.sizes.size();
  split.basisSizes.resize(blocks);
  for (size_t b = 0; b < blocks; ++b) { split.basisSizes[b] = basisCounts(system.sizes[b], axes); }
  matrix = SplitBlockMatrix(blocks, blocks);
  for (size_t i = 0; i < blocks; ++i)
  {
    for (size_t j = 0; j < blocks; ++j)
    {
      const auto &block = system.matrix({i, j});
      if (block.get_number_elements() == 0)
      {
        boba::Array<size_t, 8> rows, cols;
        for (size_t d = 0; d < 4; ++d)
        {
          rows[2 * d] = split.basisSizes[i][d]; rows[2 * d + 1] = system.sizes[i][d] / split.basisSizes[i][d];
          cols[2 * d] = split.basisSizes[j][d]; cols[2 * d + 1] = system.sizes[j][d] / split.basisSizes[j][d];
        }
        matrix({i, j}) = zero_tt_matrix<8>(rows, cols);
        continue;
      }
      matrix({i, j}) = splitMatrix(block, split.basisSizes[i], split.basisSizes[j], tolerance);
    }
  }
  splitRHS = SplitBlockVector(blocks);
  for (size_t b = 0; b < blocks; ++b) { splitRHS(b) = splitVector(rhs(b), split.basisSizes[b], tolerance); }
}

inline BlockVector mergeSolution(const SplitBlockVector &solution)
{
  BlockVector result(solution.block_size);
  for (size_t b = 0; b < solution.block_size; ++b) { result(b) = mergeVector(solution(b)); }
  return result;
}
} // namespace PlanarMITLBoba

namespace PlanarMITLBoba
{
/// Split only the axes flagged in `split` (basis mode, then cell mode); unsplit axes stay one TT mode each.  D is the
/// resulting number of modes, 4 + (number of split axes).  With one cell in a direction an extra cell mode would be
/// trivial (size 1), which made block AMEn diverge in the all-axes split, so only multi-cell axes should be split.
/// Built from the eight-mode split, contracting each unsplit axis's (mode, size-1) pair back into one core.
template <size_t D>
host_tt_matrix<D> splitMatrixAxes(const TTMatrix &matrix, const std::array<bool, 4> &split,
                                  const std::array<size_t, 4> &rowBasis, const std::array<size_t, 4> &colBasis,
                                  double tolerance)
{
  std::array<size_t, 4> rb, cb;
  for (size_t d = 0; d < 4; ++d)
  {
    rb[d] = split[d] ? rowBasis[d] : matrix.core_rows(d);
    cb[d] = split[d] ? colBasis[d] : matrix.core_cols(d);
  }
  auto eight = splitMatrix(matrix, rb, cb, tolerance);
  boba::Array<size_t, D> rows, cols;
  std::vector<boba::Tensor<4, host_space, double>> cores;
  for (size_t d = 0; d < 4; ++d)
  {
    if (split[d])
    {
      cores.push_back(eight.cores[2 * d]);
      cores.push_back(eight.cores[2 * d + 1]);
      continue;
    }
    const auto &A = eight.cores[2 * d], &B = eight.cores[2 * d + 1];
    const size_t rL = A.sizes(0), n = A.sizes(1), m = A.sizes(2), k = A.sizes(3), rR = B.sizes(3);
    auto a = A.const_view(); auto b = B.const_view();
    boba::Tensor<4, host_space, double> core({rL, n, m, rR});
    auto out = core.view();
    for (size_t r = 0; r < rR; ++r)
      for (size_t j = 0; j < m; ++j)
        for (size_t i = 0; i < n; ++i)
          for (size_t l = 0; l < rL; ++l)
          {
            double sum = 0.0;
            for (size_t q = 0; q < k; ++q) { sum += a({l, i, j, q}) * b({q, 0, 0, r}); }
            out({l, i, j, r}) = sum;
          }
    cores.push_back(core);
  }
  if (cores.size() != D) { throw std::invalid_argument("splitMatrixAxes: D does not match the split mask"); }
  for (size_t e = 0; e < D; ++e) { rows[e] = cores[e].sizes(1); cols[e] = cores[e].sizes(2); }
  host_tt_matrix<D> result(rows, cols);
  for (size_t e = 0; e < D; ++e) { result.cores[e] = cores[e]; }
  return result;
}

template <size_t D>
host_tt_vector<D> splitVectorAxes(const TTVector &vector, const std::array<bool, 4> &split,
                                  const std::array<size_t, 4> &basis, double tolerance)
{
  std::array<size_t, 4> nb;
  for (size_t d = 0; d < 4; ++d) { nb[d] = split[d] ? basis[d] : vector.sizes(d); }
  auto eight = splitVector(vector, nb, tolerance);
  std::vector<boba::Tensor<3, host_space, double>> cores;
  for (size_t d = 0; d < 4; ++d)
  {
    if (split[d])
    {
      cores.push_back(eight.cores[2 * d]);
      cores.push_back(eight.cores[2 * d + 1]);
      continue;
    }
    const auto &A = eight.cores[2 * d], &B = eight.cores[2 * d + 1];
    const size_t rL = A.sizes(0), n = A.sizes(1), k = A.sizes(2), rR = B.sizes(2);
    auto a = A.const_view(); auto b = B.const_view();
    boba::Tensor<3, host_space, double> core({rL, n, rR});
    auto out = core.view();
    for (size_t r = 0; r < rR; ++r)
      for (size_t i = 0; i < n; ++i)
        for (size_t l = 0; l < rL; ++l)
        {
          double sum = 0.0;
          for (size_t q = 0; q < k; ++q) { sum += a({l, i, q}) * b({q, 0, r}); }
          out({l, i, r}) = sum;
        }
    cores.push_back(core);
  }
  if (cores.size() != D) { throw std::invalid_argument("splitVectorAxes: D does not match the split mask"); }
  boba::Array<size_t, D> sizes;
  for (size_t e = 0; e < D; ++e) { sizes[e] = cores[e].sizes(1); }
  host_tt_vector<D> result(sizes);
  for (size_t e = 0; e < D; ++e) { result.cores[e] = cores[e]; }
  return result;
}

/// Merge each split axis's (basis, cell) pair back into one (cell*basis) mode, exactly.
template <size_t D>
TTVector mergeVectorAxes(const host_tt_vector<D> &vector, const std::array<bool, 4> &split)
{
  std::vector<boba::Tensor<3, host_space, double>> cores;
  size_t e = 0;
  for (size_t d = 0; d < 4; ++d)
  {
    if (!split[d]) { cores.push_back(vector.cores[e++]); continue; }
    const auto &A = vector.cores[e], &B = vector.cores[e + 1];
    e += 2;
    const size_t rL = A.sizes(0), nb = A.sizes(1), k = A.sizes(2), nc = B.sizes(1), rR = B.sizes(2);
    auto a = A.const_view(); auto b = B.const_view();
    boba::Tensor<3, host_space, double> core({rL, nb * nc, rR});
    auto out = core.view();
    for (size_t r = 0; r < rR; ++r)
      for (size_t c = 0; c < nc; ++c)
        for (size_t i = 0; i < nb; ++i)
          for (size_t l = 0; l < rL; ++l)
          {
            double sum = 0.0;
            for (size_t q = 0; q < k; ++q) { sum += a({l, i, q}) * b({q, c, r}); }
            out({l, i + nb * c, r}) = sum;
          }
    cores.push_back(core);
  }
  Sizes sizes;
  for (size_t d = 0; d < 4; ++d) { sizes[d] = cores[d].sizes(1); }
  TTVector result(sizes);
  for (size_t d = 0; d < 4; ++d) { result.cores[d] = cores[d]; }
  return result;
}

/// Per-block basis counts along each axis: mode size / cells; a flux block's normal mode holds (cells - 1) faces of
/// one "basis function" each.
inline std::vector<std::array<size_t, 4>> axisBasisCounts(const System &system, const std::vector<Axis> &axes)
{
  std::vector<std::array<size_t, 4>> result(system.sizes.size());
  for (size_t b = 0; b < system.sizes.size(); ++b)
    for (size_t d = 0; d < 4; ++d)
    {
      const size_t cells = axes[d].vertices.size() - 1, n = system.sizes[b][d];
      result[b][d] = (n % cells == 0) ? n / cells : 1;
    }
  return result;
}

/// Split the flagged axes of every block, solve with block AMEn in D modes, and merge back to four modes.
template <size_t D>
BlockVector solveSplitAxes(const System &system, const BlockVector &rhs, const std::vector<Axis> &axes,
                           const std::array<bool, 4> &split, double tolerance, const DpgBoba::AmenOptions &options,
                           bool check)
{
  const size_t blocks = system.sizes.size();
  const auto basis = axisBasisCounts(system, axes);
  boba::BlockOperator<host_tt_matrix<D>> matrix(blocks, blocks);
  for (size_t i = 0; i < blocks; ++i)
    for (size_t j = 0; j < blocks; ++j)
      matrix({i, j}) = splitMatrixAxes<D>(system.matrix({i, j}), split, basis[i], basis[j], tolerance);
  boba::BlockVector<host_tt_vector<D>> splitRHS(blocks);
  for (size_t b = 0; b < blocks; ++b) { splitRHS(b) = splitVectorAxes<D>(rhs(b), split, basis[b], tolerance); }
  for (size_t b = 0; b < blocks; ++b)
    std::cout << "split block=" << system.names[b] << " sizes=" << splitRHS(b).sizes()
              << " test_row_ranks=" << matrix({0, b}).ranks_string() << '\n';
  if (check)
  {
    // (split A)(split x), merged, must equal A x for a random four-mode x.
    auto x = DpgBoba::initialGuess(rhs, 3);
    boba::BlockVector<host_tt_vector<D>> sx(blocks);
    for (size_t b = 0; b < blocks; ++b) { sx(b) = splitVectorAxes<D>(x(b), split, basis[b], 1e-14); }
    for (size_t i = 0; i < blocks; ++i)
    {
      auto roundtrip = mergeVectorAxes<D>(sx(i), split);
      roundtrip -= x(i);
      auto Ax = system.matrix({i, 0}) * x(0);
      auto SAx = matrix({i, 0}) * sx(0);
      for (size_t j = 1; j < blocks; ++j) { Ax += system.matrix({i, j}) * x(j); SAx += matrix({i, j}) * sx(j); }
      auto difference = mergeVectorAxes<D>(SAx, split);
      difference -= Ax;
      std::cout << "split_check block=" << system.names[i]
                << " vector_roundtrip=" << boba::norm_frobenius(roundtrip) / boba::norm_frobenius(x(i))
                << " apply_difference=" << boba::norm_frobenius(difference) / (boba::norm_frobenius(Ax) + 1e-300) << '\n';
    }
  }
  auto solution = DpgBoba::solveAmen(matrix, splitRHS, options);
  BlockVector result(blocks);
  for (size_t b = 0; b < blocks; ++b) { result(b) = mergeVectorAxes<D>(solution(b), split); }
  return result;
}

/// Dispatch on the number of split axes.
inline BlockVector solveSplitAxes(const System &system, const BlockVector &rhs, const std::vector<Axis> &axes,
                                  const std::array<bool, 4> &split, double tolerance,
                                  const DpgBoba::AmenOptions &options, bool check)
{
  const int count = int(split[0]) + int(split[1]) + int(split[2]) + int(split[3]);
  switch (count)
  {
  case 1: return solveSplitAxes<5>(system, rhs, axes, split, tolerance, options, check);
  case 2: return solveSplitAxes<6>(system, rhs, axes, split, tolerance, options, check);
  case 3: return solveSplitAxes<7>(system, rhs, axes, split, tolerance, options, check);
  case 4: return solveSplitAxes<8>(system, rhs, axes, split, tolerance, options, check);
  default: throw std::invalid_argument("solveSplitAxes: split at least one axis");
  }
}
} // namespace PlanarMITLBoba
