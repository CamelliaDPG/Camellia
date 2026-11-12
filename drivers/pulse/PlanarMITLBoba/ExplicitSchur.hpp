#pragma once
// Explicit TT Schur complement for the N5 (direction-separable) test norm, and Chebyshev on the trial space only.
//
// With an exact element inverse (fast diagonalization), S = sum_K B_K^T G_K^{-1} B_K is formed once:
//   1. per element, B_K (test x [field, flux_x, flux_t, flux_ux, flux_uz]) from the same 1D factors as the TT assembly,
//      G_K^{-1} B_K by fast diagonalization, S_K = B_K^T G_K^{-1} B_K;
//   2. per trial block pair (a, b), the element tensor T_ab(c, i, j) with one mode per direction holding
//      (cell, row-local, col-local) -- the four-mode (cell x basis per direction) ordering in which S has low rank --
//      compressed by TT-SVD, and its cores expanded to an assembled four-mode TT matrix (traces: (cell, face slot) ->
//      mesh face; exterior faces dropped, as in the TT system);
//   3. the exact Jacobi diagonal from the S_K; a rank-one Kronecker fit P ~ D^{-1/2} per block; S~ = P S P;
//   4. c = B^T G^{-1} l element-locally; Chebyshev on S~ y = P c (bounds from a short TT Lanczos run), x = P y;
//      eps = G^{-1}(l - B x) element-locally.
// Everything is in the four-mode layout of System (trial blocks: field cell*nbF; flux normal: interior faces; flux
// other directions: cell*nbJ), so the result plugs into the existing residual check and output.
#include "SchurChebyshev.hpp"

namespace PlanarMITLBoba
{
struct ExplicitSchurOptions
{
  double sTolerance = 1e-6;       // TT-SVD / rounding tolerance of the S blocks
  double tolerance = 1e-5;        // iterate rounding tolerance
  double rtol = 0.0;
  int iterations = 400, every = 10;
  int boundsSteps = 20; double boundsTolerance = 1e-2, lminSafety = 0.8;
  bool zipUp = true;
  // lambda_max: "power" (Ifpack2-style deterministic power iteration on S~ = P S P, times boostFactor), "bound"
  // (rigorous element bound, O(n^3) per element), or "lanczos" (the Lanczos Ritz max x 1.05)
  std::string lambdaMaxMode = "power";
  int powerIterations = 20; double powerTolerance = 1e-6, powerStop = 1e-3, boostFactor = 1.1;
  int lambdaMinDiagnosticIterations = 0;   // > 0: shifted power iteration on (lambda_max I - S~) for lambda_min
};

namespace ExplicitSchurDetail
{
using Mat = Eigen::MatrixXd;

inline Mat kron(const Mat &outer, const Mat &inner)
{
  Mat K(outer.rows() * inner.rows(), outer.cols() * inner.cols());
  for (Eigen::Index a = 0; a < outer.rows(); ++a)
    for (Eigen::Index b = 0; b < outer.cols(); ++b)
      K.block(a * inner.rows(), b * inner.cols(), inner.rows(), inner.cols()) = outer(a, b) * inner;
  return K;
}

/// A trial block in the four-mode layout: normal = -1 for the field, else the flux's normal direction.
struct TrialBlock
{
  int normal = -1;
  std::array<size_t, 4> nLocal{}, global{};
  size_t localSize() const { return nLocal[0] * nLocal[1] * nLocal[2] * nLocal[3]; }
  /// global four-mode index in direction d of local index i in cell c (-1: exterior face)
  long index(size_t d, size_t c, size_t i, size_t cells) const
  {
    if (int(d) == normal) { const long f = long(c) - 1 + long(i); return (f < 0 || f > long(cells) - 2) ? -1 : f; }
    return long(c * nLocal[d] + i);
  }
};

/// A TT core (r0, n, r1), first index fastest.
struct Core3
{
  size_t r0 = 1, n = 1, r1 = 1; std::vector<double> v;
  Core3(size_t a, size_t b, size_t c) : r0(a), n(b), r1(c), v(a * b * c, 0.0) {}
  double &operator()(size_t a, size_t i, size_t b) { return v[a + r0 * (i + n * b)]; }
  double operator()(size_t a, size_t i, size_t b) const { return v[a + r0 * (i + n * b)]; }
};

/// TT-SVD of a full tensor T (mode 0 fastest) with sizes n, relative tolerance tol (T is consumed).  Each unfolding is
/// truncated through the smaller of its two Gram matrices.
inline std::vector<Core3> ttSvd(std::vector<double> &T, const std::array<size_t, 4> &n, double tol)
{
  double norm2 = 0; for (double v : T) norm2 += v * v;
  const double delta2 = tol * tol * norm2 / 3.0;
  std::vector<Core3> cores;
  size_t r = 1, rest = T.size();
  Mat C = Eigen::Map<Mat>(T.data(), Eigen::Index(n[0]), Eigen::Index(rest / n[0]));
  { std::vector<double>().swap(T); }
  for (size_t d = 0; d < 3; ++d)
  {
    rest /= n[d];
    const Mat M = Eigen::Map<const Mat>(C.data(), Eigen::Index(r * n[d]), Eigen::Index(rest));
    Mat U; Eigen::VectorXd w; Mat nextC;
    if (M.rows() <= M.cols())
    {
      Eigen::SelfAdjointEigenSolver<Mat> es(Mat(M * M.transpose()));
      w = es.eigenvalues().reverse().cwiseMax(0.0); U = es.eigenvectors().rowwise().reverse();
    }
    else
    {
      Eigen::SelfAdjointEigenSolver<Mat> es(Mat(M.transpose() * M));
      w = es.eigenvalues().reverse().cwiseMax(0.0); const Mat V = es.eigenvectors().rowwise().reverse();
      U = M * V;   // columns scaled by sigma; normalized below for the kept ones
      for (Eigen::Index q = 0; q < U.cols(); ++q) if (w(q) > 0) U.col(q) /= std::sqrt(w(q));
      nextC = w.cwiseSqrt().asDiagonal() * V.transpose();
    }
    double tail = w.sum(); Eigen::Index k = 0;
    while (k < w.size() && tail > delta2) { tail -= w(k); ++k; }
    k = std::max<Eigen::Index>(k, 1);
    Core3 core(r, n[d], size_t(k));
    for (Eigen::Index q = 0; q < k; ++q) for (size_t i = 0; i < n[d]; ++i) for (size_t a = 0; a < r; ++a)
      core(a, i, size_t(q)) = U(Eigen::Index(a + r * i), q);
    cores.push_back(core);
    C = nextC.size() ? Mat(nextC.topRows(k)) : Mat(U.leftCols(k).transpose() * M);    // k x rest
    r = size_t(k);
  }
  Core3 last(r, n[3], 1);
  for (size_t i = 0; i < n[3]; ++i) for (size_t a = 0; a < r; ++a) last(a, i, 0) = C(Eigen::Index(a), Eigen::Index(i));
  cores.push_back(last);
  return cores;
}
} // namespace ExplicitSchurDetail

inline BlockVector solveExplicitSchur(const System &system, const BlockVector &rhs, const std::vector<Axis> &axes,
                                      const LorentzFactors &h, double qOverM, double Ex, double By,
                                      const ExplicitSchurOptions &opt, const TestNormWeight &norm)
{
  using namespace ExplicitSchurDetail;
  using clock = std::chrono::steady_clock;
  auto seconds = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };
  const auto t0 = clock::now();
  if (!norm.separable) throw std::invalid_argument("explicit Schur complement: requires the separable (N5) test norm");
  std::array<size_t, 4> cells, nbT, nbF, nbJ, nq;
  for (size_t d = 0; d < 4; ++d)
  {
    cells[d] = axes[d].vertices.size() - 1; nq[d] = axes[d].points.size() / cells[d];
    nbT[d] = axes[d].test.cols() / cells[d]; nbF[d] = axes[d].field.cols() / cells[d]; nbJ[d] = axes[d].trace.cols() / cells[d];
  }
  const size_t nlocT = nbT[0] * nbT[1] * nbT[2] * nbT[3];
  const ElementClasses ec(axes);
  auto sep = separableGramInverseData(axes, cells, ec, nbT, norm);
  // trial blocks, in the order of system.names[1..]
  const std::array<std::string, 4> dirs = {"x", "t", "ux", "uz"};
  std::vector<TrialBlock> blocks;
  for (size_t b = 1; b < system.names.size(); ++b)
  {
    TrialBlock tb;
    if (system.names[b] == "f") tb.normal = -1;
    else { for (size_t d = 0; d < 4; ++d) if (system.names[b] == "flux_" + dirs[d]) tb.normal = int(d); }
    for (size_t d = 0; d < 4; ++d)
    {
      if (tb.normal < 0) { tb.nLocal[d] = nbF[d]; tb.global[d] = cells[d] * nbF[d]; }
      else if (int(d) == tb.normal) { tb.nLocal[d] = 2; tb.global[d] = cells[d] - 1; }
      else { tb.nLocal[d] = nbJ[d]; tb.global[d] = cells[d] * nbJ[d]; }
      if (tb.global[d] != system.sizes[b][d]) throw std::runtime_error("explicit Schur: trial layout mismatch for " + system.names[b]);
    }
    blocks.push_back(tb);
  }
  const size_t nt = blocks.size();
  std::vector<size_t> off(nt + 1, 0); for (size_t a = 0; a < nt; ++a) off[a + 1] = off[a] + blocks[a].localSize();
  const size_t nlocTrial = off[nt];

  // --- element B_K: field from the grouped transport factors (as structuredVaryingFieldBlock), traces from faces/mass
  const auto &profile = fieldProfile();
  auto factor = [&](size_t d, bool derivative, const Eigen::VectorXd &w) {
    const Mat &A = derivative ? axes[d].derivative : axes[d].test;
    return Mat(A.transpose() * (axes[d].weights.array() * w.array()).matrix().asDiagonal() * axes[d].field);
  };
  auto cellBlock = [&](const Mat &Q, size_t d, size_t c, size_t nr, size_t nc) { return Mat(Q.block(c * nr, c * nc, nr, nc)); };
  struct VTerm { double coefficient; Mat Q2, Q3; };
  struct Group { Mat Q0, Q1; std::vector<VTerm> terms; };
  std::vector<Group> groups(4);
  {
    const Eigen::VectorXd oX = Eigen::VectorXd::Ones(axes[0].points.size()), oT = Eigen::VectorXd::Ones(axes[1].points.size());
    const Eigen::VectorXd oUx = Eigen::VectorXd::Ones(axes[2].points.size()), oUz = Eigen::VectorXd::Ones(axes[3].points.size());
    auto prof = [&](size_t d, int field) { return profile.varying() ? profile.values(axes[d].points, field, int(d)) : Eigen::VectorXd(d == 0 ? oX : oT); };
    groups[0] = {factor(0, false, oX), factor(1, true, oT), {{1.0, factor(2, false, oUx), factor(3, false, oUz)}}};
    groups[1].Q0 = factor(0, true, oX); groups[1].Q1 = factor(1, false, oT);
    groups[2].Q0 = factor(0, false, prof(0, 0)); groups[2].Q1 = factor(1, false, prof(1, 0));
    if (Ex != 0.0) groups[2].terms.push_back({qOverM * Ex, factor(2, true, oUx), factor(3, false, oUz)});
    groups[3].Q0 = factor(0, false, prof(0, 1)); groups[3].Q1 = factor(1, false, prof(1, 1));
    for (Eigen::Index k = 0; k < h.left.cols(); ++k)
    {
      const Eigen::VectorXd vx = h.left.col(k).array() * axes[2].points.array(), vz = h.right.col(k).array() * axes[3].points.array();
      groups[1].terms.push_back({1.0, factor(2, false, vx), factor(3, false, h.right.col(k))});
      if (By != 0.0)
      {
        groups[3].terms.push_back({-qOverM * By, factor(2, true, h.left.col(k)), factor(3, false, vz)});
        groups[3].terms.push_back({qOverM * By, factor(2, false, vx), factor(3, true, h.right.col(k))});
      }
    }
  }
  std::array<Mat, 4> traceMass;
  for (size_t d = 0; d < 4; ++d) traceMass[d] = axes[d].test.transpose() * axes[d].weights.asDiagonal() * axes[d].trace;
  // columns of the selected trial blocks (all if sel is empty), concatenated in block order
  auto elementBsel = [&](const std::array<size_t, 4> &c, const std::vector<bool> &sel) {
    std::vector<size_t> soff(nt + 1, 0);
    for (size_t a = 0; a < nt; ++a) soff[a + 1] = soff[a] + (sel[a] ? blocks[a].localSize() : 0);
    Mat B = Mat::Zero(nlocT, soff[nt]);
    // field: B_f = -T^T F
    Mat Bf = Mat::Zero(nlocT, nbF[0] * nbF[1] * nbF[2] * nbF[3]);
    bool needField = false; for (size_t a = 0; a < nt; ++a) if (sel[a] && blocks[a].normal < 0) needField = true;
    if (needField)
    for (const auto &g : groups)
    {
      if (g.terms.empty()) continue;
      Mat V;
      for (const auto &t : g.terms)
      {
        Mat term = (-t.coefficient) * kron(cellBlock(t.Q3, 3, c[3], nbT[3], nbF[3]), cellBlock(t.Q2, 2, c[2], nbT[2], nbF[2]));
        if (V.size() == 0) V = term; else V += term;
      }
      Bf += kron(V, kron(cellBlock(g.Q1, 1, c[1], nbT[1], nbF[1]), cellBlock(g.Q0, 0, c[0], nbT[0], nbF[0])));
    }
    for (size_t a = 0; a < nt; ++a)
    {
      if (!sel[a]) continue;
      if (blocks[a].normal < 0) { B.middleCols(soff[a], blocks[a].localSize()) = Bf; continue; }
      const size_t nd = size_t(blocks[a].normal);
      std::array<Mat, 4> f;
      for (size_t d = 0; d < 4; ++d)
      {
        if (d == nd)
        {
          // slot 0: left face (vertex c), slot 1: right face (vertex c + 1), values as in Axis::faces
          f[d] = Mat::Zero(nbT[d], 2);
          const Mat &F = axes[d].faces;
          if (c[d] >= 1) f[d].col(0) = F.block(c[d] * nbT[d], c[d] - 1, nbT[d], 1);
          if (c[d] + 1 <= cells[d] - 1) f[d].col(1) = F.block(c[d] * nbT[d], c[d], nbT[d], 1);
        }
        else f[d] = cellBlock(traceMass[d], d, c[d], nbT[d], nbJ[d]);
      }
      B.middleCols(soff[a], blocks[a].localSize()) = kron(f[3], kron(f[2], kron(f[1], f[0])));
    }
    return B;
  };
  const std::vector<bool> allBlocks(nt, true);
  auto elementB = [&](const std::array<size_t, 4> &c) { return elementBsel(c, allBlocks); };
  auto classOf = [&](const std::array<size_t, 4> &c) { return ec.xClass[c[0]] + ec.nClass * (c[2] + cells[2] * c[3]); };
  auto forEachElement = [&](const std::function<void(const std::array<size_t, 4> &)> &f) {
    for (size_t c3 = 0; c3 < cells[3]; ++c3) for (size_t c2 = 0; c2 < cells[2]; ++c2)
      for (size_t c1 = 0; c1 < cells[1]; ++c1) for (size_t c0 = 0; c0 < cells[0]; ++c0) f({c0, c1, c2, c3});
  };
  auto localIndex = [](const std::array<size_t, 4> &n, size_t lin) { std::array<size_t, 4> i; for (size_t d = 0; d < 4; ++d) { i[d] = lin % n[d]; lin /= n[d]; } return i; };
  const size_t nel = cells[0] * cells[1] * cells[2] * cells[3];

  // --- S blocks, in passes over block pairs (memory), with the exact diagonal on the way
  std::vector<std::vector<TTMatrix>> S(nt, std::vector<TTMatrix>(nt));
  std::vector<Eigen::VectorXd> diag(nt);
  std::vector<size_t> fullSize(nt);
  for (size_t a = 0; a < nt; ++a) { fullSize[a] = 1; for (size_t d = 0; d < 4; ++d) fullSize[a] *= blocks[a].global[d]; diag[a] = Eigen::VectorXd::Zero(fullSize[a]); }
  auto globalLinear = [&](size_t a, const std::array<size_t, 4> &c, const std::array<size_t, 4> &i) -> long {
    long g = 0, stride = 1;
    for (size_t d = 0; d < 4; ++d) { const long k = blocks[a].index(d, c[d], i[d], cells[d]); if (k < 0) return -1; g += k * stride; stride *= long(blocks[a].global[d]); }
    return g;
  };

  // --- full vectors <-> four-mode TT
  auto toFull = [&](const TTVector &v) {
    auto full = v.decompress(); auto in = full.const_view();
    const size_t n0 = v.sizes(0), n1 = v.sizes(1), n2 = v.sizes(2), n3 = v.sizes(3);
    Eigen::VectorXd out(n0 * n1 * n2 * n3);
    for (size_t i3 = 0; i3 < n3; ++i3) for (size_t i2 = 0; i2 < n2; ++i2) for (size_t i1 = 0; i1 < n1; ++i1) for (size_t i0 = 0; i0 < n0; ++i0)
      out(i0 + n0 * (i1 + n1 * (i2 + n2 * i3))) = in({i0, i1, i2, i3});
    return out;
  };
  auto toTT = [&](const Eigen::VectorXd &x, const Sizes &n, double tol) {
    host_tensor<4> t(n); auto out = t.view();
    for (size_t i3 = 0; i3 < n[3]; ++i3) for (size_t i2 = 0; i2 < n[2]; ++i2) for (size_t i1 = 0; i1 < n[1]; ++i1) for (size_t i0 = 0; i0 < n[0]; ++i0)
      out({i0, i1, i2, i3}) = x(i0 + n[0] * (i1 + n[1] * (i2 + n[2] * i3)));
    TTVector r(n); r.svd_tolerance_relative = tol; r.svd_tolerance_absolute = 0.0; r.compress(t); return r;
  };
  // element-local gather / scatter of test and trial vectors (four-mode linear index, mode 0 fastest)
  auto testIndex = [&](const std::array<size_t, 4> &c, size_t il) {
    const auto i = localIndex(nbT, il); size_t g = 0, stride = 1;
    for (size_t d = 0; d < 4; ++d) { g += (c[d] * nbT[d] + i[d]) * stride; stride *= cells[d] * nbT[d]; }
    return g;
  };
  const Eigen::VectorXd lFull = toFull(rhs(0));
  // c = B^T G^{-1} l and the exact Jacobi diagonal of S, element-locally
  std::vector<Eigen::VectorXd> cFull(nt);
  for (size_t a = 0; a < nt; ++a) cFull[a] = Eigen::VectorXd::Zero(fullSize[a]);
  forEachElement([&](const std::array<size_t, 4> &c) {
    Eigen::VectorXd lK(nlocT); for (size_t il = 0; il < nlocT; ++il) lK(il) = lFull(testIndex(c, il));
    const Mat B = elementB(c);
    Mat BL(nlocT, nlocTrial + 1); BL << B, lK;
    const Mat X = separableGramSolve(*sep, classOf(c), BL);          // [G^-1 B, G^-1 l]
    const Eigen::VectorXd cK = B.transpose() * X.col(nlocTrial);
    const Eigen::VectorXd dK = (B.array() * X.leftCols(nlocTrial).array()).colwise().sum().transpose();
    for (size_t a = 0; a < nt; ++a)
      for (size_t il = 0; il < blocks[a].localSize(); ++il)
      {
        const long g = globalLinear(a, c, localIndex(blocks[a].nLocal, il));
        if (g >= 0) { cFull[a](g) += cK(off[a] + il); diag[a](g) += dK(off[a] + il); }
      }
  });
  // --- scaling P ~ D^{-1/2} (rank-one Kronecker fit of log d over the four modes), S~ = P S P
  std::vector<std::array<Eigen::VectorXd, 4>> pf(nt);
  for (size_t a = 0; a < nt; ++a)
  {
    std::array<size_t, 4> n = blocks[a].global; std::array<size_t, 4> stride{1, n[0], n[0] * n[1], n[0] * n[1] * n[2]};
    const Eigen::ArrayXd L = diag[a].array().max(1e-300).log(); const double mean = L.mean();
    for (size_t d = 0; d < 4; ++d)
    {
      Eigen::VectorXd e = Eigen::VectorXd::Zero(n[d]);
      for (Eigen::Index g = 0; g < L.size(); ++g) e((size_t(g) / stride[d]) % n[d]) += L(g);
      e = e / double(L.size() / n[d]) - Eigen::VectorXd::Constant(n[d], mean);
      pf[a][d] = (-0.5 * (e.array() + mean / 4.0)).exp().matrix();
    }
  }
  auto scaleMatrix = [&](const TTMatrix &M, size_t a, size_t b) {
    TTMatrix R = M;
    for (size_t d = 0; d < 4; ++d)
    {
      auto v = R.cores[d].view(); const size_t r0 = R.cores[d].sizes(0), nr = R.cores[d].sizes(1), nc = R.cores[d].sizes(2), r1 = R.cores[d].sizes(3);
      for (size_t r = 0; r < r1; ++r) for (size_t j = 0; j < nc; ++j) for (size_t i = 0; i < nr; ++i) for (size_t l = 0; l < r0; ++l)
        v({l, i, j, r}) *= pf[a][d](i) * pf[b][d](j);
    }
    return R;
  };
  auto scaleVector = [&](const TTVector &x, size_t a) {
    TTVector y = x;
    for (size_t d = 0; d < 4; ++d)
    {
      auto v = y.cores[d].view(); const size_t r0 = y.cores[d].sizes(0), n = y.cores[d].sizes(1), r1 = y.cores[d].sizes(2);
      for (size_t r = 0; r < r1; ++r) for (size_t i = 0; i < n; ++i) for (size_t l = 0; l < r0; ++l) v({l, i, r}) *= pf[a][d](i);
    }
    return y;
  };
  std::vector<std::pair<size_t, size_t>> pairs;
  for (size_t a = 0; a < nt; ++a) for (size_t b = a; b < nt; ++b) pairs.push_back({a, b});
  const double passBytes = 4.0e9;
  size_t pi = 0; int pass = 0;
  double tElements = 0, tSvd = 0;
  while (pi < pairs.size())
  {
    std::vector<std::pair<size_t, size_t>> ps; double bytes = 0;
    while (pi < pairs.size())
    {
      const double bb = double(nel) * blocks[pairs[pi].first].localSize() * blocks[pairs[pi].second].localSize() * 8.0;
      if (!ps.empty() && bytes + bb > passBytes) break;
      ps.push_back(pairs[pi]); bytes += bb; ++pi;
    }
    // element tensors T_ab: mode d index m_d = c_d + cells_d (i_d + nA_d j_d), mode 0 fastest
    std::vector<std::vector<double>> T(ps.size());
    std::vector<std::array<size_t, 4>> tn(ps.size());
    for (size_t q = 0; q < ps.size(); ++q)
    {
      size_t total = 1;
      for (size_t d = 0; d < 4; ++d) { tn[q][d] = cells[d] * blocks[ps[q].first].nLocal[d] * blocks[ps[q].second].nLocal[d]; total *= tn[q][d]; }
      T[q].assign(total, 0.0);
    }
    const auto te = clock::now();
    // only the columns of the blocks in this pass
    std::vector<bool> sel(nt, false); for (const auto &pr : ps) { sel[pr.first] = true; sel[pr.second] = true; }
    std::vector<size_t> soff(nt + 1, 0); for (size_t a = 0; a < nt; ++a) soff[a + 1] = soff[a] + (sel[a] ? blocks[a].localSize() : 0);
    forEachElement([&](const std::array<size_t, 4> &c) {
      const Mat B = elementBsel(c, sel);
      const Mat X = separableGramSolve(*sep, classOf(c), B);
      // P_K S_K P_K: the TT-SVD then sees O(1) entries on every cell (graded meshes: unscaled entries span orders of
      // magnitude, and truncation relative to the largest makes the scaled operator indefinite)
      Eigen::VectorXd pK(soff[nt]);
      for (size_t a = 0; a < nt; ++a)
        if (sel[a])
          for (size_t il = 0; il < blocks[a].localSize(); ++il)
          {
            const auto i = localIndex(blocks[a].nLocal, il); double v = 1.0;
            for (size_t d = 0; d < 4; ++d) { const long k = blocks[a].index(d, c[d], i[d], cells[d]); v = k < 0 ? 0.0 : v * pf[a][d](size_t(k)); }
            pK(soff[a] + il) = v;
          }
      const Mat SK = pK.asDiagonal() * (B.transpose() * X) * pK.asDiagonal();
      for (size_t q = 0; q < ps.size(); ++q)
      {
        const auto &A = blocks[ps[q].first], &Bb = blocks[ps[q].second];
        for (size_t jl = 0; jl < Bb.localSize(); ++jl)
        {
          const auto j = localIndex(Bb.nLocal, jl);
          for (size_t il = 0; il < A.localSize(); ++il)
          {
            const auto i = localIndex(A.nLocal, il);
            size_t lin = 0, stride = 1;
            for (size_t d = 0; d < 4; ++d) { lin += (c[d] + cells[d] * (i[d] + A.nLocal[d] * j[d])) * stride; stride *= tn[q][d]; }
            T[q][lin] = SK(soff[ps[q].first] + il, soff[ps[q].second] + jl);
          }
        }
      }
    });
    tElements += seconds(te);
    const auto ts = clock::now();
    for (size_t q = 0; q < ps.size(); ++q)
    {
      const size_t a = ps[q].first, b = ps[q].second;
      const auto &A = blocks[a], &Bb = blocks[b];
      auto cores = ttSvd(T[q], tn[q], opt.sTolerance);
      Sizes rows, cols; for (size_t d = 0; d < 4; ++d) { rows[d] = A.global[d]; cols[d] = Bb.global[d]; }
      TTMatrix M(rows, cols);
      for (size_t d = 0; d < 4; ++d)
      {
        const auto &G = cores[d]; const size_t r0 = G.r0, r1 = G.r1;
        M.cores[d].resize({r0, rows[d], cols[d], r1}); M.cores[d].fill_with_zeros(); auto v = M.cores[d].view();
        for (size_t c = 0; c < cells[d]; ++c)
          for (size_t i = 0; i < A.nLocal[d]; ++i)
          {
            const long row = A.index(d, c, i, cells[d]); if (row < 0) continue;
            for (size_t j = 0; j < Bb.nLocal[d]; ++j)
            {
              const long col = Bb.index(d, c, j, cells[d]); if (col < 0) continue;
              const size_t m = c + cells[d] * (i + A.nLocal[d] * j);
              for (size_t r = 0; r < r1; ++r) for (size_t l = 0; l < r0; ++l)
                v({l, size_t(row), size_t(col), r}) += G(l, m, r);
            }
          }
      }
      DpgBoba::round(M, opt.sTolerance);
      std::cout << "explicit_S block " << system.names[a + 1] << "," << system.names[b + 1] << " ranks=" << M.ranks_string() << std::endl;
      S[a][b] = M;
      if (a != b) S[b][a] = M.transpose();
    }
    tSvd += seconds(ts);
    ++pass;
  }
  std::cout << "explicit_S assembled: " << pass << " passes, element blocks " << tElements << " s, TT-SVD + expansion " << tSvd << " s" << std::endl;
  // S holds S~ = P S P already
  std::vector<std::vector<TTMatrix>> &St = S;
  // rigorous upper bound: S~ = sum_K R_K^T (P_K S_K P_K) R_K and a trial dof lies in at most m elements (m = 1 for the
  // field, 2 for a trace), so lambda_max(S~) <= m_max max_K lambda_max(P_K S_K P_K) (exterior trace dofs: P_K = 0)
  double lmaxBound = 0;
  if (opt.lambdaMaxMode == "bound")
  {
    const auto tbnd = clock::now();
    forEachElement([&](const std::array<size_t, 4> &c) {
      const Mat B = elementB(c);
      const Mat X = separableGramSolve(*sep, classOf(c), B);
      Eigen::VectorXd pK = Eigen::VectorXd::Zero(nlocTrial);
      for (size_t a = 0; a < nt; ++a)
        for (size_t il = 0; il < blocks[a].localSize(); ++il)
        {
          const auto i = localIndex(blocks[a].nLocal, il);
          double pv = 1.0; bool interior = true;
          for (size_t d = 0; d < 4; ++d) { const long k = blocks[a].index(d, c[d], i[d], cells[d]); if (k < 0) { interior = false; break; } pv *= pf[a][d](size_t(k)); }
          pK(off[a] + il) = interior ? pv : 0.0;
        }
      const Mat SK = pK.asDiagonal() * (B.transpose() * X) * pK.asDiagonal();
      const double lK = Eigen::SelfAdjointEigenSolver<Mat>(0.5 * (SK + SK.transpose()), Eigen::EigenvaluesOnly).eigenvalues().maxCoeff();
      lmaxBound = std::max(lmaxBound, lK);
    });
    size_t mMax = 1; for (size_t a = 0; a < nt; ++a) if (blocks[a].normal >= 0) mMax = 2;
    lmaxBound *= double(mMax);
    std::cout << "explicit_S lambda_max bound " << lmaxBound << " (" << seconds(tbnd) << " s)" << std::endl;
  }

  std::vector<TTVector> ct(nt), x(nt), r(nt), dlt(nt);
  double cNorm2 = 0;
  for (size_t a = 0; a < nt; ++a)
  {
    ct[a] = scaleVector(toTT(cFull[a], system.sizes[a + 1], 1e-12), a);
    cNorm2 += std::pow(boba::norm_frobenius(ct[a]), 2);
    x[a] = zero_tt_vector<4>(system.sizes[a + 1]);
  }
  const double cNorm = std::sqrt(cNorm2);
  std::cout << "explicit_S setup_seconds=" << seconds(t0) << std::endl;

  // --- S~ applied to a block vector
  auto applyS = [&](const std::vector<TTVector> &v, double t) {
    std::vector<TTVector> out(nt);
    for (size_t a = 0; a < nt; ++a)
    {
      bool first = true;
      for (size_t b = 0; b < nt; ++b)
      {
        TTVector y = opt.zipUp ? zipUpApply(St[a][b], v[b], 0.1 * t) : St[a][b] * v[b];
        DpgBoba::round(y, t);
        if (first) { out[a] = y; first = false; } else { out[a] += y; DpgBoba::round(out[a], t); }
      }
    }
    return out;
  };
  auto dot = [&](const std::vector<TTVector> &u, const std::vector<TTVector> &v) { double s = 0; for (size_t a = 0; a < nt; ++a) s += u[a].inner_product(v[a]); return s; };
  auto maxRank4 = [](const TTVector &v) { size_t m = 0; for (size_t d = 1; d < 4; ++d) m = std::max(m, v.ranks(d)); return m; };

  // --- spectrum bounds by TT Lanczos (CG coefficients)
  double lmin, lmax, ritzMax = 0;
  {
    const auto tb = clock::now(); const double bt = opt.boundsTolerance;
    std::vector<TTVector> rr = ct, pp = ct; double rz = dot(rr, rr), alphaPrev = 0, betaPrev = 0;
    std::vector<double> dd, ee;
    for (int k = 0; k < opt.boundsSteps; ++k)
    {
      auto q = applyS(pp, bt); const double alpha = rz / dot(pp, q);
      for (size_t a = 0; a < nt; ++a) { auto aq = q[a]; aq *= -alpha; rr[a] += aq; DpgBoba::round(rr[a], bt); }
      const double rzNew = dot(rr, rr), beta = rzNew / rz;
      dd.push_back(1.0 / alpha + (k > 0 ? betaPrev / alphaPrev : 0.0)); ee.push_back(std::sqrt(std::abs(beta)) / alpha);
      for (size_t a = 0; a < nt; ++a) { pp[a] *= beta; pp[a] += rr[a]; DpgBoba::round(pp[a], bt); }
      rz = rzNew; alphaPrev = alpha; betaPrev = beta;
    }
    const size_t m = dd.size(); Mat Tm = Mat::Zero(m, m);
    for (size_t i = 0; i < m; ++i) { Tm(i, i) = dd[i]; if (i + 1 < m) Tm(i, i + 1) = Tm(i + 1, i) = ee[i]; }
    const Eigen::VectorXd ritz = Eigen::SelfAdjointEigenSolver<Mat>(Tm).eigenvalues();
    lmin = ritz(0) * opt.lminSafety;
    lmax = opt.lambdaMaxMode == "bound" ? lmaxBound : ritz(m - 1) * 1.05;
    ritzMax = ritz(m - 1);
    std::cout << "explicit_S spectrum Ritz [" << ritz(0) << ", " << ritz(m - 1) << "] (" << seconds(tb) << " s); using ["
              << lmin << ", " << lmax << "] (upper: element bound; Ritz max / bound " << ritz(m - 1) / lmaxBound << ") kappa "
              << lmax / lmin << std::endl;
  }

  // Ifpack2-style deterministic power iteration (TensorTpetraRowMatrixDiagnostics.hpp): start 1 + (global index mod 97),
  // normalize every step, stop when the estimate changes by less than powerStop; lambda_max = boostFactor x estimate.
  // Rounded at powerTolerance (tight: the top eigenvectors are high-rank).
  auto patternStart = [&]() {
    std::vector<TTVector> v(nt); size_t g0 = 0;
    for (size_t a = 0; a < nt; ++a)
    {
      Eigen::VectorXd full(fullSize[a]);
      for (size_t g = 0; g < fullSize[a]; ++g) full(g) = 1.0 + double((g0 + g) % 97);
      g0 += fullSize[a]; v[a] = toTT(full, system.sizes[a + 1], 1e-12);
    }
    return v;
  };
  auto normOf = [&](const std::vector<TTVector> &v) { return std::sqrt(dot(v, v)); };
  if (opt.lambdaMaxMode == "power")
  {
    const auto tp = clock::now(); const double pt = opt.powerTolerance;
    auto v = patternStart(); double nv = normOf(v); for (auto &b : v) b *= 1.0 / nv;
    double est = 0, old = 0; int it = 0;
    for (it = 1; it <= opt.powerIterations; ++it)
    {
      v = applyS(v, pt); nv = normOf(v); old = est; est = nv;
      for (auto &b : v) b *= 1.0 / nv;
      if (it > 1 && std::abs(est - old) < opt.powerStop * est) break;
    }
    size_t rk = 0; for (const auto &b : v) rk = std::max(rk, maxRank4(b));
    lmax = opt.boostFactor * est;
    std::cout << "explicit_S power iteration lambda_max ~ " << est << " after " << std::min(it, opt.powerIterations)
              << " its (tol " << pt << ", rank " << rk << ", " << seconds(tp) << " s); Lanczos Ritz max " << ritzMax
              << "; using lambda_max = " << opt.boostFactor << " x " << est << " = " << lmax << ", kappa " << lmax / lmin << std::endl;
  }
  if (opt.lambdaMinDiagnosticIterations > 0)
  {
    // shifted power iteration: the top eigenvalue of (sigma I - S~), sigma = lmax, is sigma - lambda_min(S~)
    const auto tm = clock::now(); const double pt = opt.powerTolerance, shift = lmax;
    auto v = patternStart(); double nv = normOf(v); for (auto &b : v) b *= 1.0 / nv;
    double est = 0;
    for (int it = 1; it <= opt.lambdaMinDiagnosticIterations; ++it)
    {
      auto Sv = applyS(v, pt);
      for (size_t a = 0; a < nt; ++a) { auto w = v[a]; w *= shift; Sv[a] *= -1.0; Sv[a] += w; DpgBoba::round(Sv[a], pt); }
      const double rq = dot(v, Sv);            // Rayleigh quotient of (shift I - S~)
      est = shift - rq;
      nv = normOf(Sv); for (size_t a = 0; a < nt; ++a) { v[a] = Sv[a]; v[a] *= 1.0 / nv; }
      if (it % 10 == 0 || it == opt.lambdaMinDiagnosticIterations)
        std::cout << "explicit_S lambda_min diagnostic it=" << it << " Rayleigh estimate " << est << " (Lanczos Ritz min "
                  << lmin / opt.lminSafety << ")" << std::endl;
    }
    std::cout << "explicit_S lambda_min diagnostic: " << est << (est < 0 ? "  NEGATIVE: S~ (as rounded) is indefinite" : "")
              << " (" << seconds(tm) << " s)" << std::endl;
  }

  // --- Chebyshev on S~ y = c~
  const double tol = opt.tolerance, theta = 0.5 * (lmax + lmin), delta = 0.5 * (lmax - lmin), sigma = theta / delta;
  double rho = 1.0 / sigma;
  for (size_t a = 0; a < nt; ++a) { r[a] = ct[a]; dlt[a] = r[a]; dlt[a] *= 1.0 / theta; DpgBoba::round(dlt[a], tol); }
  std::vector<double> history; const auto loop = clock::now(); double tS = 0;
  std::vector<TTVector> xBest = x; double best = 1e300; int bestIt = 0; bool diverged = false;
  for (int k = 1; k <= opt.iterations; ++k)
  {
    for (size_t a = 0; a < nt; ++a) { x[a] += dlt[a]; DpgBoba::round(x[a], tol); }
    const auto ta = clock::now(); auto Sx = applyS(x, tol); tS += seconds(ta);
    double rn2 = 0;
    for (size_t a = 0; a < nt; ++a) { r[a] = ct[a]; r[a] -= Sx[a]; DpgBoba::round(r[a], tol); rn2 += std::pow(boba::norm_frobenius(r[a]), 2); }
    const double rhoNew = 1.0 / (2.0 * sigma - rho);
    for (size_t a = 0; a < nt; ++a) { dlt[a] *= rhoNew * rho; auto mr = r[a]; mr *= 2.0 * rhoNew / delta; dlt[a] += mr; DpgBoba::round(dlt[a], tol); }
    rho = rhoNew;
    history.push_back(std::sqrt(rn2) / cNorm);
    // x_k's residual is r_k (computed from the iterate just updated)
    if (history.back() < best) { best = history.back(); bestIt = k; xBest = x; }
    diverged = history.back() > 10.0 * best;
    bool stagnated = false;
    if (k >= 50) { const auto split = history.end() - 25; stagnated = *std::min_element(split, history.end()) > 0.9 * *std::min_element(history.begin(), split); }
    const bool converged = history.back() < opt.rtol || stagnated || diverged;
    if (k % opt.every == 0 || k == 1 || converged || k == opt.iterations)
    {
      std::cout << "explicit_S chebyshev it=" << k << " |r|/|c|=" << history.back() << " rank(x)=";
      for (size_t a = 0; a < nt; ++a) std::cout << (a ? "," : "") << maxRank4(x[a]);
      std::cout << " rank(Sx)=" << maxRank4(Sx[0]) << " seconds=" << seconds(loop) << " [S " << tS << "]" << std::endl;
    }
    if (converged)
    {
      std::cout << "explicit_S chebyshev " << (diverged ? "diverging" : stagnated ? "stagnated" : "converged") << " it=" << k
                << " (best |r|/|c|=" << best << " at it " << bestIt << ")" << std::endl;
      break;
    }
  }
  x = xBest;   // the best iterate (a stagnating or diverging tail does not improve it)

  // --- solution: x = P y; eps = G^{-1}(l - B x) element-locally
  BlockVector solution(system.names.size());
  std::vector<Eigen::VectorXd> xFull(nt);
  for (size_t a = 0; a < nt; ++a) { solution(a + 1) = scaleVector(x[a], a); DpgBoba::round(solution(a + 1), 1e-12); xFull[a] = toFull(solution(a + 1)); }
  Eigen::VectorXd eps = Eigen::VectorXd::Zero(lFull.size());
  forEachElement([&](const std::array<size_t, 4> &c) {
    Eigen::VectorXd xK = Eigen::VectorXd::Zero(nlocTrial);
    for (size_t a = 0; a < nt; ++a)
      for (size_t il = 0; il < blocks[a].localSize(); ++il)
      {
        const long g = globalLinear(a, c, localIndex(blocks[a].nLocal, il));
        if (g >= 0) xK(off[a] + il) = xFull[a](g);
      }
    Eigen::VectorXd lK(nlocT); for (size_t il = 0; il < nlocT; ++il) lK(il) = lFull(testIndex(c, il));
    const Mat B = elementB(c);
    const Eigen::VectorXd eK = separableGramSolve(*sep, classOf(c), Eigen::VectorXd(lK - B * xK));
    for (size_t il = 0; il < nlocT; ++il) eps(testIndex(c, il)) = eK(il);
  });
  solution(0) = toTT(eps, system.sizes[0], 1e-10);
  std::cout << "explicit_S total_seconds=" << seconds(t0) << std::endl;
  return solution;
}
} // namespace PlanarMITLBoba
