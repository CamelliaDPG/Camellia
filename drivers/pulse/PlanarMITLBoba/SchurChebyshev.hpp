#pragma once
// Tensor-train solver for the DPG Schur complement S = B^T G^{-1} B (trial unknowns only), by Chebyshev
// semi-iteration that applies S only to iterates:
//   r_k = c - B^T G^{-1} B x_k,   x_{k+1} = x_k + Chebyshev update of M^{-1} r_k,   c = B^T G^{-1} l.
// Krylov search directions are nearly full rank in TT form once mapped to the test space (G^{-1} B p); iterates are
// not (G^{-1} B x has the rank of G^{-1} l), and Chebyshev needs no inner products with S p.
//
// Everything is in an eight-mode TT layout with the cell indices first:
//   (c_x, c_t, c_ux, c_uz, b_x, b_t, b_ux, b_uz)   (b: local basis index within the cell; flux normal axes: c = face, b = 1)
// In that layout G^{-1} is applied exactly without visiting elements: G is block diagonal over elements and, on meshes
// uniform in x and t, the element block depends only on the velocity cell v, so for z with rank r_cb at the cells|basis
// cut, y = G^{-1} z = sum_v [cell cores of z restricted to v] (x) [G_v^{-1} Z], Z the r_cb local basis vectors of z.
// Cost: n_v Cholesky solves with r_cb right-hand sides plus TT rounding; independent of the number of x and t cells.
//
// The operators are assembled directly in this layout from the same one-dimensional factors as assemble(): each
// cell-blocked 1D factor is split as sum_s (cell matrix) (x) (local basis matrix) by an SVD, and each Kronecker term is
// rewritten with its cell factors first.  M^{-1} is a rank-one Kronecker (per mode, per trial block) approximation of the
// Jacobi diagonal, read from a file (tools/schur_chebyshev.py --prec kron8 --write-kron); the spectrum bounds of M^{-1} S
// are inputs as well (same tool).
#include "TensorProductAssembly.hpp"
#include <Eigen/Cholesky>
#include <Eigen/SVD>
#include <chrono>
#include <fstream>
#include <functional>
#include <map>
#include <algorithm>
#include <memory>
#include <sstream>

namespace PlanarMITLBoba
{
using TTMatrix8 = host_tt_matrix<8>;
using TTVector8 = host_tt_vector<8>;
using Size8 = boba::Array<size_t, 8>;

struct CellBasisSplit
{
  std::vector<Eigen::MatrixXd> cell, basis;
};

/// A (rows cellsR*nbR, cols cellsC*nbC, cell-major: index = cell*nb + local) = sum_s cell_s (x) basis_s.
inline CellBasisSplit splitCellBasis(const Eigen::MatrixXd &A, Eigen::Index cR, Eigen::Index nbR, Eigen::Index cC,
                                     Eigen::Index nbC, double tolerance = 1e-14)
{
  if (A.rows() != cR * nbR || A.cols() != cC * nbC)
    throw std::invalid_argument("splitCellBasis: factor is not cell-blocked as described");
  Eigen::MatrixXd M(cR * cC, nbR * nbC);
  for (Eigen::Index cc = 0; cc < cC; ++cc)
    for (Eigen::Index bc = 0; bc < nbC; ++bc)
      for (Eigen::Index cr = 0; cr < cR; ++cr)
        for (Eigen::Index br = 0; br < nbR; ++br)
          M(cr + cR * cc, br + nbR * bc) = A(cr * nbR + br, cc * nbC + bc);
  Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto s = svd.singularValues();
  CellBasisSplit out;
  for (Eigen::Index k = 0; k < s.size(); ++k)
  {
    if (!(s(k) > tolerance * s(0))) break;
    Eigen::VectorXd u = svd.matrixU().col(k) * s(k), v = svd.matrixV().col(k);
    out.cell.push_back(Eigen::Map<Eigen::MatrixXd>(u.data(), cR, cC));
    out.basis.push_back(Eigen::Map<Eigen::MatrixXd>(v.data(), nbR, nbC));
  }
  if (out.cell.empty()) { out.cell.push_back(Eigen::MatrixXd::Zero(cR, cC)); out.basis.push_back(Eigen::MatrixXd::Zero(nbR, nbC)); }
  return out;
}

/// The Kronecker product of four split factors as an eight-mode TT matrix with the cell modes first.
inline TTMatrix8 cellsFirstMatrix(const std::array<CellBasisSplit, 4> &f)
{
  Size8 rows, cols;
  std::array<size_t, 4> S;
  for (size_t d = 0; d < 4; ++d)
  {
    rows[d] = f[d].cell[0].rows(); cols[d] = f[d].cell[0].cols();
    rows[4 + d] = f[d].basis[0].rows(); cols[4 + d] = f[d].basis[0].cols();
    S[d] = f[d].cell.size();
  }
  TTMatrix8 M(rows, cols);
  size_t R = 1;
  for (size_t d = 0; d < 4; ++d)   // cell core d: rank index a = (s_0, ..., s_{d-1}) -> a + R s_d
  {
    auto &core = M.cores[d];
    core.resize({R, rows[d], cols[d], R * S[d]}); core.fill_with_zeros();
    auto v = core.view();
    for (size_t s = 0; s < S[d]; ++s)
      for (size_t j = 0; j < cols[d]; ++j)
        for (size_t i = 0; i < rows[d]; ++i)
        {
          const double value = f[d].cell[s](i, j);
          if (value != 0.0)
            for (size_t a = 0; a < R; ++a) v({a, i, j, a + R * s}) = value;
        }
    R *= S[d];
  }
  for (size_t d = 0; d < 4; ++d)   // basis core d consumes s_d, the fastest remaining index
  {
    const size_t Rout = R / S[d];
    auto &core = M.cores[4 + d];
    core.resize({R, rows[4 + d], cols[4 + d], Rout}); core.fill_with_zeros();
    auto v = core.view();
    for (size_t s = 0; s < S[d]; ++s)
      for (size_t j = 0; j < cols[4 + d]; ++j)
        for (size_t i = 0; i < rows[4 + d]; ++i)
        {
          const double value = f[d].basis[s](i, j);
          if (value != 0.0)
            for (size_t rest = 0; rest < Rout; ++rest) v({s + S[d] * rest, i, j, rest}) = value;
        }
    R = Rout;
  }
  return M;
}

struct CellsFirstSystem
{
  std::vector<TTMatrix8> B, BT;              // trial blocks 1..: B_j (test x trial_j) and its transpose
  // the field block in factored form, B_f = -T^T F (T: transport, quadrature x test; F: field values, rank one):
  // applying the factors keeps intermediate ranks at rank(T) * rank(vector) instead of rank(B_f) * rank(vector)
  TTMatrix8 T, TT, F, FT;
  std::vector<Size8> trialSizes;
  Size8 testSizes;
  std::array<size_t, 4> cells, nbTest;
};

inline void accumulate(TTMatrix8 &sum, bool &first, const TTMatrix8 &term, double tolerance)
{
  if (first) { sum = term; first = false; }
  else { sum += term; DpgBoba::round(sum, tolerance); }
}

/// G and the B_j in the cells-first layout, from the same 1D factors and terms as assemble().
inline CellsFirstSystem assembleCellsFirst(const std::vector<Axis> &axes, const LorentzFactors &h, double qOverM,
                                           double Ex, double By, double tolerance,
                                           const std::vector<std::string> &names)
{
  CellsFirstSystem sys;
  std::array<Eigen::Index, 4> cells, nq, nbT, nbF, nbJ;
  for (size_t d = 0; d < 4; ++d)
  {
    cells[d] = Eigen::Index(axes[d].vertices.size()) - 1; nq[d] = axes[d].points.size() / cells[d];
    nbT[d] = axes[d].test.cols() / cells[d]; nbF[d] = axes[d].field.cols() / cells[d]; nbJ[d] = axes[d].trace.cols() / cells[d];
    sys.cells[d] = cells[d]; sys.nbTest[d] = nbT[d];
    sys.testSizes[d] = cells[d]; sys.testSizes[4 + d] = nbT[d];
  }
  auto evaluated = [&](size_t d, const Eigen::MatrixXd &values, const Eigen::VectorXd &weight, Eigen::Index nb) {
    Eigen::VectorXd w = axes[d].weights.array().sqrt() * weight.array();
    return splitCellBasis(w.asDiagonal() * values, cells[d], nq[d], cells[d], nb);
  };
  // transport (rows: quadrature points, cols: test), as in assemble()
  TTMatrix8 transport; bool first = true;
  auto term = [&](size_t derivativeAxis, const Eigen::VectorXd &ux, const Eigen::VectorXd &uz, double coefficient) {
    std::array<CellBasisSplit, 4> f;
    for (size_t d = 0; d < 4; ++d)
    {
      Eigen::VectorXd weight = Eigen::VectorXd::Ones(axes[d].points.size());
      if (d == 2) weight = ux;
      if (d == 3) weight = uz;
      if (d == 0) weight *= coefficient;
      f[d] = evaluated(d, d == derivativeAxis ? axes[d].derivative : axes[d].test, weight, nbT[d]);
    }
    accumulate(transport, first, cellsFirstMatrix(f), tolerance);
  };
  const Eigen::VectorXd onesX = Eigen::VectorXd::Ones(axes[2].points.size());
  const Eigen::VectorXd onesZ = Eigen::VectorXd::Ones(axes[3].points.size());
  term(1, onesX, onesZ, 1.0);
  if (Ex != 0.0) term(2, onesX, onesZ, qOverM * Ex);
  for (Eigen::Index k = 0; k < h.left.cols(); ++k)
  {
    Eigen::VectorXd vx = h.left.col(k).array() * axes[2].points.array();
    Eigen::VectorXd vz = h.right.col(k).array() * axes[3].points.array();
    term(0, vx, h.right.col(k), 1.0);
    if (By != 0.0) { term(2, h.left.col(k), vz, -qOverM * By); term(3, vx, h.right.col(k), qOverM * By); }
  }
  // field block: B_f = -T^T F
  {
    std::array<CellBasisSplit, 4> f;
    for (size_t d = 0; d < 4; ++d) f[d] = evaluated(d, axes[d].field, Eigen::VectorXd::Ones(axes[d].points.size()), nbF[d]);
    sys.T = transport; sys.TT = transport.transpose(); sys.F = cellsFirstMatrix(f); sys.FT = sys.F.transpose();
    TTMatrix8 Bf = -1.0 * (sys.TT * sys.F);
    DpgBoba::round(Bf, tolerance);
    sys.B.push_back(Bf);
    Size8 sz; for (size_t d = 0; d < 4; ++d) { sz[d] = cells[d]; sz[4 + d] = nbF[d]; }
    sys.trialSizes.push_back(sz);
  }
  // flux blocks, in the order of names[2..] (flux_<direction>)
  const std::array<std::string, 4> directions = {"x", "t", "ux", "uz"};
  for (size_t b = 2; b < names.size(); ++b)
  {
    size_t normal = 4;
    for (size_t d = 0; d < 4; ++d) if (names[b] == "flux_" + directions[d]) normal = d;
    if (normal == 4) throw std::invalid_argument("assembleCellsFirst: unexpected block " + names[b]);
    std::array<CellBasisSplit, 4> f; Size8 sz;
    for (size_t d = 0; d < 4; ++d)
    {
      if (d == normal)
      {
        f[d] = splitCellBasis(axes[d].faces, cells[d], nbT[d], cells[d] - 1, 1);
        sz[d] = cells[d] - 1; sz[4 + d] = 1;
      }
      else
      {
        f[d] = splitCellBasis(axes[d].test.transpose() * axes[d].weights.asDiagonal() * axes[d].trace,
                              cells[d], nbT[d], cells[d], nbJ[d]);
        sz[d] = cells[d]; sz[4 + d] = nbJ[d];
      }
    }
    sys.B.push_back(cellsFirstMatrix(f));
    sys.trialSizes.push_back(sz);
  }
  for (const auto &Bj : sys.B) sys.BT.push_back(Bj.transpose());
  return sys;
}

/// Four-mode (cell*basis per axis) TT vector <-> eight-mode cells-first TT vector, through the full tensor (setup and
/// output only).
inline TTVector8 toCellsFirst(const host_tt_vector<4> &v, const Size8 &sizes8, double tolerance)
{
  auto full = v.decompress(); auto in = full.const_view();
  host_tensor<8> t(sizes8); auto out = t.view();
  std::array<size_t, 4> c, b, n4;
  for (size_t d = 0; d < 4; ++d) n4[d] = sizes8[d] * sizes8[4 + d];
  for (size_t i3 = 0; i3 < n4[3]; ++i3)
    for (size_t i2 = 0; i2 < n4[2]; ++i2)
      for (size_t i1 = 0; i1 < n4[1]; ++i1)
        for (size_t i0 = 0; i0 < n4[0]; ++i0)
        {
          const std::array<size_t, 4> i{i0, i1, i2, i3};
          for (size_t d = 0; d < 4; ++d) { c[d] = i[d] / sizes8[4 + d]; b[d] = i[d] % sizes8[4 + d]; }
          out({c[0], c[1], c[2], c[3], b[0], b[1], b[2], b[3]}) = in({i0, i1, i2, i3});
        }
  TTVector8 r(sizes8); r.svd_tolerance_relative = tolerance; r.svd_tolerance_absolute = 0.0;
  r.compress(t);
  return r;
}

inline host_tt_vector<4> toFourMode(const TTVector8 &v, double tolerance)
{
  auto full = v.decompress(); auto in = full.const_view();
  boba::Array<size_t, 4> n4; for (size_t d = 0; d < 4; ++d) n4[d] = v.sizes(d) * v.sizes(4 + d);
  host_tensor<4> t(n4); auto out = t.view();
  for (size_t i3 = 0; i3 < n4[3]; ++i3)
    for (size_t i2 = 0; i2 < n4[2]; ++i2)
      for (size_t i1 = 0; i1 < n4[1]; ++i1)
        for (size_t i0 = 0; i0 < n4[0]; ++i0)
        {
          const std::array<size_t, 4> i{i0, i1, i2, i3};
          std::array<size_t, 4> c, b;
          for (size_t d = 0; d < 4; ++d) { c[d] = i[d] / v.sizes(4 + d); b[d] = i[d] % v.sizes(4 + d); }
          out({i0, i1, i2, i3}) = in({c[0], c[1], c[2], c[3], b[0], b[1], b[2], b[3]});
        }
  host_tt_vector<4> r(n4); r.svd_tolerance_relative = tolerance; r.svd_tolerance_absolute = 0.0;
  r.compress(t);
  return r;
}

/// The element block of a cell-blocked four-mode TT matrix at cell c (rows nbR, cols nbC per axis; first axis fastest).
inline Eigen::MatrixXd elementBlock(const TTMatrix &matrix, const std::array<size_t, 4> &nbR, const std::array<size_t, 4> &nbC,
                                    const std::array<size_t, 4> &c)
{
  std::vector<Eigen::MatrixXd> partial(1, Eigen::MatrixXd::Ones(1, 1));
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &core = matrix.cores[d]; auto v = core.const_view();
    const size_t rl = core.sizes(0), rr = core.sizes(3), nr = nbR[d], nc = nbC[d];
    std::vector<Eigen::MatrixXd> next(rr, Eigen::MatrixXd::Zero(partial[0].rows() * nr, partial[0].cols() * nc));
    for (size_t b = 0; b < rr; ++b)
      for (size_t a = 0; a < rl; ++a)
      {
        const auto &P = partial[a];
        for (size_t j = 0; j < nc; ++j)
          for (size_t i = 0; i < nr; ++i)
          {
            const double value = v({a, c[d] * nr + i, c[d] * nc + j, b});
            if (value != 0.0) next[b].block(i * P.rows(), j * P.cols(), P.rows(), P.cols()) += value * P;
          }
      }
    partial = std::move(next);
  }
  return partial[0];
}

/// A block-diagonal (over elements) operator whose element block depends only on the velocity cell, applied exactly to
/// cells-first TT vectors: one local map per velocity cell, one SVD of the stacked local results (their span has
/// dimension <= the local output dimension), one TT rounding.  Nothing is evaluated per element.
struct StructuredLocalOperator
{
  std::array<size_t, 4> cells, nbIn, nbOut;
  // local map for element class v = c + nClass (i + n_ux j): c the x-width class, (i, j) the velocity cell
  std::function<Eigen::MatrixXd(size_t v, const Eigen::MatrixXd &rhs)> local;   // rhs: nlocIn x m
  std::vector<size_t> xClass;   // class of each x cell (empty: one class)
  size_t nClass = 1;

  TTVector8 apply(const TTVector8 &z, double tolerance) const
  {
    const auto &zc = z.cores;
    const size_t rcb = zc[3].sizes(2), nlocIn = nbIn[0] * nbIn[1] * nbIn[2] * nbIn[3], nlocOut = nbOut[0] * nbOut[1] * nbOut[2] * nbOut[3];
    // Z(a, local) from the basis cores (a: rank index at the cells|basis cut; local: b_x fastest)
    Eigen::MatrixXd T = Eigen::MatrixXd::Identity(rcb, rcb);
    for (size_t m = 4; m < 8; ++m)
    {
      auto v = zc[m].const_view(); const size_t r = zc[m].sizes(0), n = zc[m].sizes(1), rn = zc[m].sizes(2);
      Eigen::MatrixXd next(T.rows() * n, rn);
      for (size_t i = 0; i < n; ++i)
      {
        Eigen::MatrixXd core(r, rn);
        for (size_t q = 0; q < rn; ++q) for (size_t p = 0; p < r; ++p) core(p, q) = v({p, i, q});
        next.block(i * T.rows(), 0, T.rows(), rn) = T * core;
      }
      T = std::move(next);
    }
    Eigen::MatrixXd Z = Eigen::Map<Eigen::MatrixXd>(T.data(), rcb, nlocIn);
    const size_t nux = cells[2], nuz = cells[3], nC = nClass, nv = nC * nux * nuz;
    auto classOf = [&](size_t cx) { return xClass.empty() ? size_t(0) : xClass[cx]; };
    // all local results W_v (nlocOut x rcb) for every (x class, velocity cell), stacked; their span has dimension <= nlocOut
    Eigen::MatrixXd Wall(nlocOut, nv * rcb);
    for (size_t v = 0; v < nv; ++v) Wall.middleCols(v * rcb, rcb) = local(v, Z.transpose());
    Eigen::BDCSVD<Eigen::MatrixXd> svd(Wall, Eigen::ComputeThinU);
    const auto sv = svd.singularValues();
    Eigen::Index k = 0;
    {
      // drop singular values below tolerance/10 of the total (the final rounding applies the tolerance itself)
      const double total = sv.squaredNorm(); double tail = total;
      while (k < sv.size() && tail > std::pow(0.1 * tolerance, 2) * total) { tail -= sv(k) * sv(k); ++k; }
      k = std::max<Eigen::Index>(k, 1);
    }
    const Eigen::MatrixXd U = svd.matrixU().leftCols(k);                        // nloc x k, orthonormal
    // basis cores: TT of U^T (k x nloc) over the local modes, the k index as the left rank
    std::array<Eigen::MatrixXd, 4> basisCores; std::array<size_t, 4> kin{};
    {
      auto keep = [](const Eigen::VectorXd &s) { Eigen::Index r = 0; while (r < s.size() && s(r) > 1e-14 * s(0)) ++r; return std::max<Eigen::Index>(r, 1); };
      Eigen::MatrixXd rem = U.transpose();
      size_t K = k, restSize = nlocOut;
      for (size_t m = 0; m < 4; ++m)
      {
        const size_t n = nbOut[m]; restSize /= n;
        Eigen::MatrixXd M(K * n, restSize);                                       // rows kk + K b_m
        for (size_t rest = 0; rest < restSize; ++rest)
          for (size_t bm = 0; bm < n; ++bm)
            for (size_t kk = 0; kk < K; ++kk) M(kk + K * bm, rest) = rem(kk, bm + n * rest);
        kin[m] = K;
        if (m == 3) { basisCores[m] = M; break; }
        Eigen::JacobiSVD<Eigen::MatrixXd> s2(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::Index rk = keep(s2.singularValues());
        basisCores[m] = s2.matrixU().leftCols(rk);
        rem = s2.singularValues().head(rk).asDiagonal() * s2.matrixV().leftCols(rk).transpose();
        K = rk;
      }
    }
    // cell cores: c_ux carries i in the rank; c_uz slice j for block i is zc3[:, j, :] C_(i,j)^T, C_v = U^T W_v
    Size8 outSizes = z.sizes(); for (size_t m = 0; m < 4; ++m) outSizes[4 + m] = nbOut[m];
    TTVector8 y(outSizes);
    // cell cores: c_x routes each cell to its class block; c_t and c_ux carry (class, i) in the rank; the c_uz slice j
    // of block (c, i) is zc3[:, j, :] C_(c,i,j)^T with C_v = U^T W_v
    const size_t nx = cells[0], nt = cells[1];
    const size_t r1 = zc[0].sizes(2), r2 = zc[1].sizes(2), r3 = zc[2].sizes(2);
    y.cores[0].resize({1, nx, nC * r1}); y.cores[0].fill_with_zeros();
    {
      auto out = y.cores[0].view(); auto in = zc[0].const_view();
      for (size_t cx = 0; cx < nx; ++cx)
        for (size_t q = 0; q < r1; ++q) out({0, cx, classOf(cx) * r1 + q}) = in({0, cx, q});
    }
    y.cores[1].resize({nC * r1, nt, nC * r2}); y.cores[1].fill_with_zeros();
    {
      auto out = y.cores[1].view(); auto in = zc[1].const_view();
      for (size_t c = 0; c < nC; ++c)
        for (size_t ct = 0; ct < nt; ++ct)
          for (size_t q = 0; q < r2; ++q) for (size_t p = 0; p < r1; ++p) out({c * r1 + p, ct, c * r2 + q}) = in({p, ct, q});
    }
    y.cores[2].resize({nC * r2, nux, nC * nux * r3}); y.cores[2].fill_with_zeros();
    {
      auto out = y.cores[2].view(); auto in = zc[2].const_view();
      for (size_t c = 0; c < nC; ++c)
        for (size_t i = 0; i < nux; ++i)
          for (size_t q = 0; q < r3; ++q) for (size_t p = 0; p < r2; ++p) out({c * r2 + p, i, (c + nC * i) * r3 + q}) = in({p, i, q});
    }
    y.cores[3].resize({nC * nux * r3, nuz, size_t(k)}); y.cores[3].fill_with_zeros();
    {
      auto out = y.cores[3].view(); auto in = zc[3].const_view();
      Eigen::MatrixXd slice(r3, rcb);
      for (size_t j = 0; j < nuz; ++j)
      {
        for (size_t q = 0; q < rcb; ++q) for (size_t p = 0; p < r3; ++p) slice(p, q) = in({p, j, q});
        for (size_t i = 0; i < nux; ++i)
          for (size_t c = 0; c < nC; ++c)
          {
            const size_t v = c + nC * (i + nux * j), row0 = (c + nC * i) * r3;
            const Eigen::MatrixXd block = slice * (U.transpose() * Wall.middleCols(v * rcb, rcb)).transpose();  // r3 x k
            for (Eigen::Index q = 0; q < k; ++q) for (size_t p = 0; p < r3; ++p) out({row0 + p, j, size_t(q)}) = block(p, q);
          }
      }
    }
    for (size_t m = 0; m < 4; ++m)
    {
      const size_t n = nbOut[m], Kin = kin[m], Kout = basisCores[m].cols();
      auto &core = y.cores[4 + m]; core.resize({Kin, n, Kout}); auto out = core.view();
      for (size_t q = 0; q < Kout; ++q)
        for (size_t bm = 0; bm < n; ++bm)
          for (size_t kk = 0; kk < Kin; ++kk) out({kk, bm, q}) = basisCores[m](kk + Kin * bm, q);
    }
    DpgBoba::round(y, tolerance);
    return y;
  }
};

/// A dense block of a four-mode TT matrix: rows rowIdx[d], cols colIdx[d] in each mode (first mode fastest).
inline Eigen::MatrixXd tensorBlock(const TTMatrix &matrix, const std::array<std::vector<size_t>, 4> &rowIdx,
                                   const std::array<std::vector<size_t>, 4> &colIdx)
{
  std::vector<Eigen::MatrixXd> partial(1, Eigen::MatrixXd::Ones(1, 1));
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &core = matrix.cores[d]; auto v = core.const_view();
    const size_t rl = core.sizes(0), rr = core.sizes(3), nr = rowIdx[d].size(), nc = colIdx[d].size();
    std::vector<Eigen::MatrixXd> next(rr, Eigen::MatrixXd::Zero(partial[0].rows() * nr, partial[0].cols() * nc));
    for (size_t b = 0; b < rr; ++b)
      for (size_t a = 0; a < rl; ++a)
      {
        const auto &P = partial[a];
        for (size_t j = 0; j < nc; ++j)
          for (size_t i = 0; i < nr; ++i)
          {
            const double value = v({a, rowIdx[d][i], colIdx[d][j], b});
            if (value != 0.0) next[b].block(i * P.rows(), j * P.cols(), P.rows(), P.cols()) += value * P;
          }
      }
    partial = std::move(next);
  }
  return partial[0];
}

/// x-width classes of the x cells (cells of equal width share element blocks); t must be uniform.
struct ElementClasses
{
  std::vector<size_t> xClass, representative;   // class of each x cell; first x cell of each class
  size_t nClass = 0;
  explicit ElementClasses(const std::vector<Axis> &axes)
  {
    const auto &xv = axes[0].vertices; std::vector<double> widths;
    for (size_t c = 0; c + 1 < xv.size(); ++c)
    {
      const double h = xv[c + 1] - xv[c]; size_t k = 0;
      while (k < widths.size() && std::abs(widths[k] - h) > 1e-12 * h) ++k;
      if (k == widths.size()) { widths.push_back(h); representative.push_back(c); }
      xClass.push_back(k);
    }
    nClass = widths.size();
    const auto &tv = axes[1].vertices;
    for (size_t c = 0; c + 1 < tv.size(); ++c)
      if (std::abs((tv[c + 1] - tv[c]) - (tv[1] - tv[0])) > 1e-12 * (tv[1] - tv[0]))
        throw std::invalid_argument("Schur-Chebyshev: the t mesh must be uniform");
  }
};

/// Element blocks (rows nbR, cols nbC per axis, cell-blocked) for every (x class, velocity cell), index
/// c + nClass (i + n_ux j); checked against another element of the same class (last x cell of the class, last t cell).
inline std::vector<Eigen::MatrixXd> classBlocks(const TTMatrix &M, const std::array<size_t, 4> &cells, const ElementClasses &ec,
                                                const std::array<size_t, 4> &nbR, const std::array<size_t, 4> &nbC,
                                                const std::string &name)
{
  auto cellIdx = [](size_t c, size_t nb) { std::vector<size_t> v(nb); for (size_t b = 0; b < nb; ++b) v[b] = c * nb + b; return v; };
  auto block = [&](const std::array<size_t, 4> &c) {
    std::array<std::vector<size_t>, 4> r, k;
    for (size_t d = 0; d < 4; ++d) { r[d] = cellIdx(c[d], nbR[d]); k[d] = cellIdx(c[d], nbC[d]); }
    return tensorBlock(M, r, k);
  };
  std::vector<Eigen::MatrixXd> blocks; double worst = 0.0;
  for (size_t j = 0; j < cells[3]; ++j)
    for (size_t i = 0; i < cells[2]; ++i)
      for (size_t c = 0; c < ec.nClass; ++c)
      {
        blocks.push_back(block({ec.representative[c], 0, i, j}));
        size_t last = ec.representative[c];
        for (size_t cx = 0; cx < cells[0]; ++cx) if (ec.xClass[cx] == c) last = cx;
        Eigen::MatrixXd other = block({last, cells[1] - 1, i, j});
        worst = std::max(worst, (other - blocks.back()).norm() / blocks.back().norm());
      }
  std::cout << "structured " << name << " x_classes=" << ec.nClass << " velocity_cells=" << cells[2] * cells[3]
            << " block=" << blocks[0].rows() << "x" << blocks[0].cols() << " max_within_class_variation=" << worst << '\n';
  if (worst > 1e-8) throw std::invalid_argument("structured " + name + ": element blocks vary within an x-width class");
  return blocks;
}

inline StructuredLocalOperator structuredGramInverse(const TTMatrix &G4, const std::array<size_t, 4> &cells, const ElementClasses &ec,
                                                     const std::array<size_t, 4> &nb,
                                                     std::shared_ptr<std::vector<Eigen::LLT<Eigen::MatrixXd>>> &factorsOut)
{
  auto blocks = classBlocks(G4, cells, ec, nb, nb, "G^-1");
  auto factors = std::make_shared<std::vector<Eigen::LLT<Eigen::MatrixXd>>>();
  for (const auto &Gv : blocks)
  {
    factors->emplace_back(0.5 * (Gv + Gv.transpose()));
    if (factors->back().info() != Eigen::Success) throw std::runtime_error("structured G^-1: G_v not SPD");
  }
  factorsOut = factors;
  StructuredLocalOperator op{cells, nb, nb, [factors](size_t v, const Eigen::MatrixXd &rhs) { return Eigen::MatrixXd((*factors)[v].solve(rhs)); }};
  op.xClass = ec.xClass; op.nClass = ec.nClass;
  return op;
}

/// B (or B^T) for an element-local block (the field block): local matrices per element class.
inline StructuredLocalOperator structuredBlock(const TTMatrix &B4, const std::array<size_t, 4> &cells, const ElementClasses &ec,
                                               const std::array<size_t, 4> &nbRow, const std::array<size_t, 4> &nbCol, bool transpose)
{
  auto blocks = std::make_shared<std::vector<Eigen::MatrixXd>>(classBlocks(B4, cells, ec, nbRow, nbCol, transpose ? "B_f^T" : "B_f"));
  StructuredLocalOperator op = transpose
    ? StructuredLocalOperator{cells, nbRow, nbCol, [blocks](size_t v, const Eigen::MatrixXd &rhs) { return Eigen::MatrixXd((*blocks)[v].transpose() * rhs); }}
    : StructuredLocalOperator{cells, nbCol, nbRow, [blocks](size_t v, const Eigen::MatrixXd &rhs) { return Eigen::MatrixXd((*blocks)[v] * rhs); }};
  op.xClass = ec.xClass; op.nClass = ec.nClass;
  return op;
}

/// Jacobi diagonal of S = B^T G^{-1} B for each trial block, as full vectors in the cells-first layout (setup only).
/// A trial dof's diagonal entry is the sum over the elements it touches of e^T B_K^T G_K^{-1} B_K e, which depends on K
/// only through its class, its velocity cell, and which of its faces exist; one local computation per distinct key.
inline std::vector<Eigen::VectorXd> jacobiDiagonal(const System &system, const std::vector<Size8> &trialSizes,
                                                   const std::array<size_t, 4> &cells, const std::array<size_t, 4> &nbT,
                                                   const ElementClasses &ec,
                                                   const std::vector<Eigen::LLT<Eigen::MatrixXd>> &Gfactors)
{
  const size_t nt = trialSizes.size();
  std::vector<Eigen::VectorXd> diag(nt);
  for (size_t b = 0; b < nt; ++b)
  {
    const Size8 &sz = trialSizes[b];
    size_t total = 1; for (size_t m = 0; m < 8; ++m) total *= sz[m];
    diag[b] = Eigen::VectorXd::Zero(total);
    std::array<size_t, 8> stride; stride[0] = 1; for (size_t m = 1; m < 8; ++m) stride[m] = stride[m - 1] * sz[m - 1];
    std::array<bool, 4> face;
    for (size_t d = 0; d < 4; ++d) face[d] = sz[4 + d] == 1 && cells[d] > 1 && sz[d] == cells[d] - 1;
    std::map<std::vector<size_t>, Eigen::VectorXd> cache;
    for (size_t cz = 0; cz < cells[3]; ++cz)
      for (size_t cu = 0; cu < cells[2]; ++cu)
        for (size_t ct = 0; ct < cells[1]; ++ct)
          for (size_t cx = 0; cx < cells[0]; ++cx)
          {
            const std::array<size_t, 4> c{cx, ct, cu, cz};
            // local trial entries per mode: (cell-or-face index, basis index)
            std::array<std::vector<std::pair<size_t, size_t>>, 4> loc;
            std::vector<size_t> key{ec.xClass[cx], cu, cz};
            for (size_t d = 0; d < 4; ++d)
            {
              if (face[d])
              {
                const bool left = c[d] > 0, right = c[d] + 1 < cells[d];
                if (left) loc[d].push_back({c[d] - 1, 0});
                if (right) loc[d].push_back({c[d], 0});
                key.push_back(size_t(left) + 2 * size_t(right));
              }
              else for (size_t bb = 0; bb < sz[4 + d]; ++bb) loc[d].push_back({c[d], bb});
            }
            auto it = cache.find(key);
            if (it == cache.end())
            {
              std::array<std::vector<size_t>, 4> rows, cols;
              for (size_t d = 0; d < 4; ++d)
              {
                for (size_t bb = 0; bb < nbT[d]; ++bb) rows[d].push_back(c[d] * nbT[d] + bb);
                for (const auto &e : loc[d]) cols[d].push_back(face[d] ? e.first : e.first * sz[4 + d] + e.second);
              }
              const Eigen::MatrixXd E = tensorBlock(system.matrix({0, b + 1}), rows, cols);
              const size_t v = ec.xClass[cx] + ec.nClass * (cu + cells[2] * cz);
              const Eigen::MatrixXd GE = Gfactors[v].solve(E);
              it = cache.emplace(key, (E.array() * GE.array()).colwise().sum().transpose()).first;
            }
            const Eigen::VectorXd &local = it->second;
            size_t q = 0;
            for (const auto &e3 : loc[3]) for (const auto &e2 : loc[2]) for (const auto &e1 : loc[1]) for (const auto &e0 : loc[0])
            {
              const size_t g = e0.first * stride[0] + e1.first * stride[1] + e2.first * stride[2] + e3.first * stride[3] +
                               e0.second * stride[4] + e1.second * stride[5] + e2.second * stride[6] + e3.second * stride[7];
              diag[b](g) += local(q++);
            }
          }
  }
  return diag;
}

/// Rank-one Kronecker approximation of a diagonal (cells-first layout): log d fitted by main effects over the 8 modes.
inline TTMatrix8 kroneckerInverseDiagonal(const Eigen::VectorXd &d, const Size8 &sz, double &medianRelErr, double &maxRelErr)
{
  std::array<size_t, 8> stride; stride[0] = 1; for (size_t m = 1; m < 8; ++m) stride[m] = stride[m - 1] * sz[m - 1];
  const Eigen::ArrayXd L = d.array().log(); const double mean = L.mean();
  std::array<Eigen::VectorXd, 8> eff;
  for (size_t m = 0; m < 8; ++m) eff[m] = Eigen::VectorXd::Zero(sz[m]);
  for (Eigen::Index g = 0; g < L.size(); ++g)
    for (size_t m = 0; m < 8; ++m) eff[m]((g / stride[m]) % sz[m]) += L(g);
  for (size_t m = 0; m < 8; ++m) eff[m] = eff[m] / double(L.size() / sz[m]) - Eigen::VectorXd::Constant(sz[m], mean);
  std::vector<double> rel(L.size());
  for (Eigen::Index g = 0; g < L.size(); ++g)
  {
    double fit = mean; for (size_t m = 0; m < 8; ++m) fit += eff[m]((g / stride[m]) % sz[m]);
    rel[g] = std::abs(std::exp(fit) / d(g) - 1.0);
  }
  std::nth_element(rel.begin(), rel.begin() + rel.size() / 2, rel.end()); medianRelErr = rel[rel.size() / 2];
  maxRelErr = *std::max_element(rel.begin(), rel.end());
  boba::Array<host_matrix, 8> factors;
  for (size_t m = 0; m < 8; ++m)
  {
    factors[m] = host_matrix({sz[m], sz[m]}); factors[m].fill_with_zeros(); auto v = factors[m].view();
    for (size_t i = 0; i < sz[m]; ++i) v({i, i}) = std::exp(-(mean / 8.0 + eff[m](i)));
  }
  return boba::make_ttm_from_matrices<8, host_space, double>(factors);
}

struct ChebyshevOptions
{
  double lmin = 0, lmax = 0, tolerance = 1e-5, operatorTolerance = 1e-12, rtol = 0.0;
  bool factoredField = false;   // apply B_f as -T^T F (pays off when rank(B_f) >> rank(T))
  bool structuredField = true;  // apply B_f and B_f^T element-locally per velocity cell (no rank multiplication)
  int boundsSteps = 30;          // TT Lanczos steps for the spectrum bounds when lmin/lmax are not given
  double boundsTolerance = 1e-4, lminSafety = 0.8;
  int iterations = 200, every = 10;
  std::string kroneckerFile;
};

/// Rank-one Kronecker diagonal M^{-1} per trial block from "block mode n values..." lines (cells-first modes).
inline std::vector<TTMatrix8> readKroneckerJacobi(const std::string &file, const std::vector<Size8> &sizes)
{
  std::vector<boba::Array<host_matrix, 8>> factors(sizes.size());
  std::vector<std::array<bool, 8>> seen(sizes.size(), std::array<bool, 8>{});
  std::ifstream in(file); std::string line;
  if (!in) throw std::invalid_argument("cannot read Kronecker-Jacobi file " + file);
  while (std::getline(in, line))
  {
    std::istringstream s(line); size_t b, m, n; s >> b >> m >> n;
    if (b >= sizes.size() || m >= 8 || n != sizes[b][m]) throw std::invalid_argument("Kronecker-Jacobi file does not match the layout: " + line);
    factors[b][m] = host_matrix({n, n}); factors[b][m].fill_with_zeros(); auto v = factors[b][m].view();
    for (size_t i = 0; i < n; ++i) { double value; s >> value; v({i, i}) = 1.0 / value; }
    seen[b][m] = true;
  }
  std::vector<TTMatrix8> M;
  for (size_t b = 0; b < sizes.size(); ++b)
  {
    for (size_t m = 0; m < 8; ++m) if (!seen[b][m]) throw std::invalid_argument("Kronecker-Jacobi file misses a factor");
    M.push_back(boba::make_ttm_from_matrices<8, host_space, double>(factors[b]));
  }
  return M;
}

inline std::string maxRank(const TTVector8 &v) { size_t r = 0; for (size_t d = 1; d < 8; ++d) r = std::max(r, v.ranks(d)); return std::to_string(r); }

/// Chebyshev on S = B^T G^{-1} B with S applied only to iterates.  Returns the four-mode block solution
/// (block 0: eps = G^{-1}(l - B x); blocks 1..: trial).
inline BlockVector solveSchurChebyshev(const System &system, const BlockVector &rhs, const std::vector<Axis> &axes,
                                       const LorentzFactors &h, double qOverM, double Ex, double By,
                                       const ChebyshevOptions &opt)
{
  using clock = std::chrono::steady_clock;
  auto seconds = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };
  auto t0 = clock::now();
  auto sys = assembleCellsFirst(axes, h, qOverM, Ex, By, opt.operatorTolerance, system.names);
  std::cout << "cells_first";
  for (size_t j = 0; j < sys.B.size(); ++j) std::cout << " B_" << system.names[j + 1] << " ranks=" << sys.B[j].ranks_string();
  std::cout << " assembly_seconds=" << seconds(t0) << '\n';
  std::array<size_t, 4> nbT = sys.nbTest;
  const ElementClasses ec(axes);
  std::shared_ptr<std::vector<Eigen::LLT<Eigen::MatrixXd>>> Gfactors;
  auto Ginv = structuredGramInverse(system.matrix({0, 0}), sys.cells, ec, nbT, Gfactors);
  std::array<size_t, 4> nbF; for (size_t d = 0; d < 4; ++d) nbF[d] = sys.trialSizes[0][4 + d];
  auto Bf = structuredBlock(system.matrix({0, 1}), sys.cells, ec, nbT, nbF, false);
  auto BfT = structuredBlock(system.matrix({0, 1}), sys.cells, ec, nbT, nbF, true);
  std::vector<TTMatrix8> Minv;
  if (!opt.kroneckerFile.empty()) Minv = readKroneckerJacobi(opt.kroneckerFile, sys.trialSizes);
  else
  {
    const auto tj = clock::now();
    const auto diag = jacobiDiagonal(system, sys.trialSizes, sys.cells, nbT, ec, *Gfactors);
    for (size_t j = 0; j < diag.size(); ++j)
    {
      double med, mx; Minv.push_back(kroneckerInverseDiagonal(diag[j], sys.trialSizes[j], med, mx));
      std::cout << "kronecker_jacobi block=" << system.names[j + 1] << " |d_K/d - 1| median=" << med << " max=" << mx << '\n';
    }
    std::cout << "kronecker_jacobi seconds=" << seconds(tj) << '\n';
  }
  const double tol = opt.tolerance;
  auto l = toCellsFirst(rhs(0), sys.testSizes, opt.operatorTolerance);
  auto e = Ginv.apply(l, opt.operatorTolerance);                 // G^{-1} l
  const size_t nt = sys.B.size();
  std::vector<TTVector8> c(nt), x(nt), r(nt), d(nt);
  double cNorm2 = 0;
  // B_f x = -T^T (F x),  B_f^T y = -F^T (T y)
  auto applyB = [&](size_t j, const TTVector8 &v, double t) {
    if (j == 0 && opt.structuredField) return Bf.apply(v, t);
    if (j > 0 || !opt.factoredField) { TTVector8 out = sys.B[j] * v; DpgBoba::round(out, t); return out; }
    TTVector8 fx = sys.F * v; DpgBoba::round(fx, t);
    TTVector8 out = sys.TT * fx; out *= -1.0; DpgBoba::round(out, t); return out;
  };
  auto applyBT = [&](size_t j, const TTVector8 &v, double t) {
    if (j == 0 && opt.structuredField) return BfT.apply(v, t);
    if (j > 0 || !opt.factoredField) { TTVector8 out = sys.BT[j] * v; DpgBoba::round(out, t); return out; }
    TTVector8 ty = sys.T * v; DpgBoba::round(ty, t);
    TTVector8 out = sys.FT * ty; out *= -1.0; DpgBoba::round(out, t); return out;
  };
  std::cout << "transport T ranks=" << sys.T.ranks_string() << " field F ranks=" << sys.F.ranks_string() << '\n';
  for (size_t j = 0; j < nt; ++j)
  {
    c[j] = applyBT(j, e, opt.operatorTolerance);
    cNorm2 += std::pow(boba::norm_frobenius(c[j]), 2);
    x[j] = zero_tt_vector<8>(sys.trialSizes[j]);
  }
  const double cNorm = std::sqrt(cNorm2);
  // S applied to a trial block vector: out_j = B_j^T G^{-1} sum_i B_i v_i
  auto applyS = [&](const std::vector<TTVector8> &v, double t) {
    TTVector8 z = applyB(0, v[0], t);
    for (size_t j = 1; j < nt; ++j) { z += applyB(j, v[j], t); DpgBoba::round(z, t); }
    const TTVector8 y = Ginv.apply(z, t);
    std::vector<TTVector8> out(nt);
    for (size_t j = 0; j < nt; ++j) out[j] = applyBT(j, y, t);
    return out;
  };
  auto dot = [&](const std::vector<TTVector8> &a, const std::vector<TTVector8> &b) {
    double sum = 0; for (size_t j = 0; j < nt; ++j) sum += a[j].inner_product(b[j]); return sum;
  };
  double lmin = opt.lmin, lmax = opt.lmax;
  if (!(lmin > 0) || !(lmax > lmin))
  {
    // spectrum of M^{-1} S from the coefficients of a short TT PCG run (Lanczos), at a loose rounding tolerance
    const auto tb = clock::now(); const double bt = opt.boundsTolerance;
    std::vector<TTVector8> rr = c, zz(nt), pp(nt);
    for (size_t j = 0; j < nt; ++j) { zz[j] = Minv[j] * rr[j]; DpgBoba::round(zz[j], bt); }
    pp = zz; double rz = dot(rr, zz), alphaPrev = 0, betaPrev = 0;
    std::vector<double> dd, ee;
    for (int k = 0; k < opt.boundsSteps; ++k)
    {
      auto q = applyS(pp, bt);
      const double alpha = rz / dot(pp, q);
      for (size_t j = 0; j < nt; ++j) { auto aq = q[j]; aq *= -alpha; rr[j] += aq; DpgBoba::round(rr[j], bt); zz[j] = Minv[j] * rr[j]; DpgBoba::round(zz[j], bt); }
      const double rzNew = dot(rr, zz), beta = rzNew / rz;
      dd.push_back(1.0 / alpha + (k > 0 ? betaPrev / alphaPrev : 0.0)); ee.push_back(std::sqrt(std::abs(beta)) / alpha);
      for (size_t j = 0; j < nt; ++j) { pp[j] *= beta; pp[j] += zz[j]; DpgBoba::round(pp[j], bt); }
      rz = rzNew; alphaPrev = alpha; betaPrev = beta;
    }
    const size_t m = dd.size();
    Eigen::MatrixXd Tm = Eigen::MatrixXd::Zero(m, m);
    for (size_t i = 0; i < m; ++i) { Tm(i, i) = dd[i]; if (i + 1 < m) Tm(i, i + 1) = Tm(i + 1, i) = ee[i]; }
    const Eigen::VectorXd ritz = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(Tm).eigenvalues();
    lmin = ritz(0) * opt.lminSafety; lmax = ritz(m - 1) * 1.05;
    std::cout << "spectrum Ritz [" << ritz(0) << ", " << ritz(m - 1) << "] from " << m << " TT Lanczos steps (tol " << bt
              << ", " << seconds(tb) << " s); using [" << lmin << ", " << lmax << "] kappa " << lmax / lmin << '\n';
  }
  std::cout << "schur_chebyshev setup_seconds=" << seconds(t0) << " G^-1 l ranks=" << e.ranks_string()
            << " lmin=" << lmin << " lmax=" << lmax << " tol=" << tol << '\n';
  const double theta = 0.5 * (lmax + lmin), delta = 0.5 * (lmax - lmin), sigma = theta / delta;
  double rho = 1.0 / sigma;
  for (size_t j = 0; j < nt; ++j) { r[j] = c[j]; d[j] = Minv[j] * r[j]; d[j] *= 1.0 / theta; DpgBoba::round(d[j], tol); }
  TTVector8 y;
  double tBx = 0, tG = 0, tBT = 0, tUp = 0;
  std::vector<double> history;
  const auto loopStart = clock::now();
  for (int k = 1; k <= opt.iterations; ++k)
  {
    auto ta = clock::now();
    for (size_t j = 0; j < nt; ++j) { x[j] += d[j]; DpgBoba::round(x[j], tol); }
    TTVector8 z = applyB(0, x[0], tol);
    for (size_t j = 1; j < nt; ++j) { z += applyB(j, x[j], tol); DpgBoba::round(z, tol); }
    tBx += seconds(ta); ta = clock::now();
    y = Ginv.apply(z, tol);
    tG += seconds(ta); ta = clock::now();
    double rNorm2 = 0;
    for (size_t j = 0; j < nt; ++j)
    {
      r[j] = c[j]; r[j] -= applyBT(j, y, tol); DpgBoba::round(r[j], tol);
      rNorm2 += std::pow(boba::norm_frobenius(r[j]), 2);
    }
    tBT += seconds(ta); ta = clock::now();
    const double rhoNew = 1.0 / (2.0 * sigma - rho);
    for (size_t j = 0; j < nt; ++j)
    {
      d[j] *= rhoNew * rho;
      auto mr = Minv[j] * r[j]; mr *= 2.0 * rhoNew / delta;
      d[j] += mr; DpgBoba::round(d[j], tol);
    }
    rho = rhoNew;
    tUp += seconds(ta);
    history.push_back(std::sqrt(rNorm2) / cNorm);
    // converged: below rtol, or stagnated at the rounding floor (no 10% decrease over the last 25 iterations)
    const bool stagnated = k >= 50 && history.back() > 0.9 * history[history.size() - 26];
    const bool converged = history.back() < opt.rtol || stagnated;
    if (k % opt.every == 0 || k == 1 || k == opt.iterations || converged)
    {
      std::cout << "chebyshev it=" << k << " |r|/|c|=" << std::sqrt(rNorm2) / cNorm << " rank(x)=";
      for (size_t j = 0; j < nt; ++j) std::cout << (j ? "," : "") << maxRank(x[j]);
      std::cout << " rank(Bx)=" << maxRank(z) << " rank(G^-1Bx)=" << maxRank(y) << " rank(r_f)=" << maxRank(r[0])
                << " seconds=" << seconds(loopStart) << " [Bx " << tBx << " G^-1 " << tG << " B^T " << tBT << " update " << tUp << "]"
                << std::endl;
    }
    if (converged) { std::cout << "chebyshev " << (stagnated ? "stagnated" : "converged") << " it=" << k << '\n'; break; }
  }
  // epsilon = G^{-1} l - G^{-1} B x (with the last y), and the four-mode solution
  BlockVector solution(system.names.size());
  auto eps = e; eps -= y; DpgBoba::round(eps, opt.operatorTolerance);
  solution(0) = toFourMode(eps, opt.operatorTolerance);
  for (size_t j = 0; j < nt; ++j) solution(j + 1) = toFourMode(x[j], opt.operatorTolerance);
  std::cout << "schur_chebyshev total_seconds=" << seconds(t0) << '\n';
  return solution;
}
} // namespace PlanarMITLBoba
