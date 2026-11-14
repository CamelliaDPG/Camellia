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
#include <fstream>
#include <atomic>
#include <thread>

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
  std::string lambdaMaxMode = "fullpower";
  // Defaults (11-13): 10 iterations (Ifpack2's default) from a seeded rank-one start, rounded at 1e-4, boost 1.2.  The
  // Ifpack2 pattern start (1 + index mod 97) is high-rank in TT form (~97), which made 20 iterations at 1e-6 cost
  // ~1000-2000 s at production shape; 10 iterations reach ~0.9 x the converged lambda_max, hence the larger boost.
  int powerIterations = 10; double powerTolerance = 1e-4, powerStop = 1e-3, boostFactor = 1.2;
  std::string powerStart = "random";        // "random" (seeded rank one) or "pattern" (1 + index mod 97)
  int lambdaMinDiagnosticIterations = 0;   // > 0: shifted power iteration on (lambda_max I - S~) for lambda_min
  // solver for S~ y = c~: "chebyshev" (above) or "amen" (block AMEn on the SPD trial-space system, optionally with the
  // (basis, cell) mode split of solveSplitAxes)
  std::string solver = "chebyshev";
  DpgBoba::AmenOptions amen;
  std::array<bool, 4> split{false, false, false, false};
  // threads for the setup (element loops, per-pass TT-SVDs); 0: CAMELLIA_EXPLICIT_S_THREADS, else 1.  std::thread
  // (the build has no OpenMP); each thread works on whole elements, so results are identical to the serial loops.
  int threads = 0;
};

namespace ExplicitSchurDetail
{
using Mat = Eigen::MatrixXd;

/// current and peak resident set size (GB) of this process, from /proc/self/status (0 if unavailable)
inline std::pair<double, double> rssGB()
{
  std::ifstream in("/proc/self/status"); std::string key; double now = 0, peak = 0;
  while (in >> key)
  {
    if (key == "VmRSS:") { in >> now; now /= 1048576.0; }
    else if (key == "VmHWM:") { in >> peak; peak /= 1048576.0; }
  }
  return {now, peak};
}
inline void reportMemory(const std::string &stage)
{
  const auto m = rssGB();
  std::cout << "explicit_S memory " << stage << ": rss " << m.first << " GB, peak " << m.second << " GB" << std::endl;
}

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

/// Run f(begin, end, part) on `parts` contiguous ranges of [0, n), one std::thread each (part 0 on the caller).
inline void splitRange(Eigen::Index n, int parts, const std::function<void(Eigen::Index, Eigen::Index, int)> &f)
{
  parts = int(std::max<Eigen::Index>(1, std::min<Eigen::Index>(parts, n)));
  if (parts == 1) { f(0, n, 0); return; }
  std::vector<std::thread> pool; std::vector<std::exception_ptr> err(parts);
  auto run = [&](int t) { try { f(n * t / parts, n * (t + 1) / parts, t); } catch (...) { err[t] = std::current_exception(); } };
  for (int t = 1; t < parts; ++t) pool.emplace_back(run, t);
  run(0); for (auto &th : pool) th.join();
  for (auto &e : err) if (e) std::rethrow_exception(e);
}

/// Threaded products for the TT-SVD unfoldings M (m x k, column-major): M M^T (split over columns), M^T M (split over
/// rows), M V (split over rows) and W^T M (split over columns).  Partial Grams are summed in a fixed order.
inline Mat gramRows(const Eigen::Map<const Mat> &M, int threads)
{
  std::vector<Mat> part(std::max(1, threads));
  splitRange(M.cols(), threads, [&](Eigen::Index b, Eigen::Index e, int t) {
    part[t] = Mat::Zero(M.rows(), M.rows()); part[t].selfadjointView<Eigen::Lower>().rankUpdate(M.middleCols(b, e - b)); });
  Mat G = Mat::Zero(M.rows(), M.rows()); for (const auto &P : part) if (P.size()) G += P;
  G.triangularView<Eigen::StrictlyUpper>() = G.transpose(); return G;
}
inline Mat gramCols(const Eigen::Map<const Mat> &M, int threads)
{
  std::vector<Mat> part(std::max(1, threads));
  splitRange(M.rows(), threads, [&](Eigen::Index b, Eigen::Index e, int t) {
    part[t] = Mat::Zero(M.cols(), M.cols()); part[t].selfadjointView<Eigen::Lower>().rankUpdate(M.middleRows(b, e - b).transpose()); });
  Mat G = Mat::Zero(M.cols(), M.cols()); for (const auto &P : part) if (P.size()) G += P;
  G.triangularView<Eigen::StrictlyUpper>() = G.transpose(); return G;
}

/// TT-SVD of a full tensor T (mode 0 fastest) with sizes n, relative tolerance tol (T is consumed).  Each unfolding is
/// truncated through the smaller of its two Gram matrices.  threads > 1: the large products are split over threads.
inline std::vector<Core3> ttSvd(std::vector<double> &T, const std::array<size_t, 4> &n, double tol, int threads = 1)
{
  double norm2 = 0; for (double v : T) norm2 += v * v;
  const double delta2 = tol * tol * norm2 / 3.0;
  std::vector<Core3> cores;
  size_t r = 1, rest = T.size();
  // the unfoldings are mapped in place (no copies of the element tensor): T for the first, then C
  std::vector<double> buffer; buffer.swap(T);
  Mat C; const double *data = buffer.data();
  for (size_t d = 0; d < 3; ++d)
  {
    rest /= n[d];
    const Eigen::Map<const Mat> M(data, Eigen::Index(r * n[d]), Eigen::Index(rest));
    Mat U; Eigen::VectorXd w; Mat nextC;
    if (M.rows() <= M.cols())
    {
      Eigen::SelfAdjointEigenSolver<Mat> es(gramRows(M, threads));
      w = es.eigenvalues().reverse().cwiseMax(0.0); U = es.eigenvectors().rowwise().reverse();
    }
    else
    {
      Eigen::SelfAdjointEigenSolver<Mat> es(gramCols(M, threads));
      w = es.eigenvalues().reverse().cwiseMax(0.0); const Mat V = es.eigenvectors().rowwise().reverse();
      U.resize(M.rows(), V.cols());
      splitRange(M.rows(), threads, [&](Eigen::Index b, Eigen::Index e, int) { U.middleRows(b, e - b).noalias() = M.middleRows(b, e - b) * V; });   // columns scaled by sigma; normalized below for the kept ones
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
    Mat Cnew;    // k x rest
    if (nextC.size()) Cnew = nextC.topRows(k);
    else
    {
      Cnew.resize(k, M.cols()); const Mat Wt = U.leftCols(k).transpose();
      splitRange(M.cols(), threads, [&](Eigen::Index b, Eigen::Index e, int) { Cnew.middleCols(b, e - b).noalias() = Wt * M.middleCols(b, e - b); });
    }
    C.swap(Cnew); data = C.data();
    if (d == 0) std::vector<double>().swap(buffer);
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
  reportMemory("after N5 inverse data");
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
  // half: V^T B in factored form (V the fast-diagonalization basis shared by all elements, V^T applied to each 1D factor)
  auto elementBsel = [&](const std::array<size_t, 4> &c, const std::vector<bool> &sel, bool half = false) {
    auto T1 = [&](size_t d, Mat M) { return half ? Mat(sep->VT[d] * M) : M; };
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
        Mat term = (-t.coefficient) * kron(T1(3, cellBlock(t.Q3, 3, c[3], nbT[3], nbF[3])), T1(2, cellBlock(t.Q2, 2, c[2], nbT[2], nbF[2])));
        if (V.size() == 0) V = term; else V += term;
      }
      Bf += kron(V, kron(T1(1, cellBlock(g.Q1, 1, c[1], nbT[1], nbF[1])), T1(0, cellBlock(g.Q0, 0, c[0], nbT[0], nbF[0]))));
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
        f[d] = T1(d, f[d]);
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
  const int nThreads = std::max(1, opt.threads > 0 ? opt.threads
                                   : (std::getenv("CAMELLIA_EXPLICIT_S_THREADS") ? std::atoi(std::getenv("CAMELLIA_EXPLICIT_S_THREADS")) : 1));
  std::cout << "explicit_S setup threads " << nThreads << std::endl;
  // run f(i, thread) for i in [0, n) on nThreads threads (dynamic chunks of `chunk`); exceptions rethrown
  auto parallelFor = [&](size_t n, size_t chunk, const std::function<void(size_t, int)> &f) {
    if (nThreads == 1 || n <= 1) { for (size_t i = 0; i < n; ++i) f(i, 0); return; }
    std::atomic<size_t> next{0}; std::vector<std::exception_ptr> err(nThreads);
    auto work = [&](int t) {
      try { for (size_t i0; (i0 = next.fetch_add(chunk)) < n;) for (size_t i = i0; i < std::min(n, i0 + chunk); ++i) f(i, t); }
      catch (...) { err[t] = std::current_exception(); next = n; }
    };
    std::vector<std::thread> pool; for (int t = 1; t < nThreads; ++t) pool.emplace_back(work, t);
    work(0); for (auto &th : pool) th.join();
    for (auto &e : err) if (e) std::rethrow_exception(e);
  };
  auto elementOf = [&](size_t lin) { std::array<size_t, 4> c; for (size_t d = 0; d < 4; ++d) { c[d] = lin % cells[d]; lin /= cells[d]; } return c; };
  auto parallelForEachElement = [&](const std::function<void(const std::array<size_t, 4> &, int)> &f) {
    parallelFor(cells[0] * cells[1] * cells[2] * cells[3], 8, [&](size_t lin, int t) { f(elementOf(lin), t); });
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
  const auto tRhs = clock::now();
  std::cout << "explicit_S setup before the element loops " << seconds(t0) << " s" << std::endl;
  for (size_t a = 0; a < nt; ++a) cFull[a] = Eigen::VectorXd::Zero(fullSize[a]);
  // trace dofs are shared by neighbouring elements: per-thread accumulators (thread 0 writes cFull, diag directly)
  std::vector<std::vector<Eigen::VectorXd>> cT(nThreads), dT(nThreads);
  for (int t = 1; t < nThreads; ++t)
    for (size_t a = 0; a < nt; ++a) { cT[t].push_back(Eigen::VectorXd::Zero(fullSize[a])); dT[t].push_back(Eigen::VectorXd::Zero(fullSize[a])); }
  parallelForEachElement([&](const std::array<size_t, 4> &c, int t) {
    auto &cAcc = t == 0 ? cFull : cT[t]; auto &dAcc = t == 0 ? diag : dT[t];
    Eigen::VectorXd lK(nlocT); for (size_t il = 0; il < nlocT; ++il) lK(il) = lFull(testIndex(c, il));
    // G_K^-1 = V D V^T: with Z = D^1/2 V^T B and z = D^1/2 V^T l, c_K = Z^T z and diag(S_K) = |Z e_j|^2
    const Eigen::ArrayXd sqrtD = sep->diagonal[classOf(c)].array().sqrt();
    const Mat Z = sqrtD.matrix().asDiagonal() * elementBsel(c, allBlocks, true);
    const Eigen::VectorXd z = sqrtD.matrix().asDiagonal() * kroneckerApply(sep->VT, Mat(lK));
    const Eigen::VectorXd cK = Z.transpose() * z;
    const Eigen::VectorXd dK = Z.colwise().squaredNorm().transpose();
    for (size_t a = 0; a < nt; ++a)
      for (size_t il = 0; il < blocks[a].localSize(); ++il)
      {
        const long g = globalLinear(a, c, localIndex(blocks[a].nLocal, il));
        if (g >= 0) { cAcc[a](g) += cK(off[a] + il); dAcc[a](g) += dK(off[a] + il); }
      }
  });
  for (int t = 1; t < nThreads; ++t) for (size_t a = 0; a < nt; ++a) { cFull[a] += cT[t][a]; diag[a] += dT[t][a]; }
  std::vector<std::vector<Eigen::VectorXd>>().swap(cT); std::vector<std::vector<Eigen::VectorXd>>().swap(dT);
  std::cout << "explicit_S rhs + jacobi element loop " << seconds(tRhs) << " s" << std::endl;
  reportMemory("after rhs + jacobi");
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
  double tElements = 0, tSvd = 0, tB = 0, tG = 0, tS = 0, tScatter = 0, tTtSvd = 0;
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
    reportMemory("pass " + std::to_string(pass) + " element tensors allocated (" + std::to_string(bytes / 1e9) + " GB)");
    // only the columns of the blocks in this pass
    std::vector<bool> sel(nt, false); for (const auto &pr : ps) { sel[pr.first] = true; sel[pr.second] = true; }
    std::vector<size_t> soff(nt + 1, 0); for (size_t a = 0; a < nt; ++a) soff[a + 1] = soff[a] + (sel[a] ? blocks[a].localSize() : 0);
    // per pair: the local part of the element-tensor index of entry (il, jl); the cell part is added per element
    std::vector<std::vector<size_t>> locIdx(ps.size());
    for (size_t q = 0; q < ps.size(); ++q)
    {
      const auto &A = blocks[ps[q].first], &Bb = blocks[ps[q].second];
      locIdx[q].resize(A.localSize() * Bb.localSize());
      for (size_t jl = 0; jl < Bb.localSize(); ++jl)
      {
        const auto j = localIndex(Bb.nLocal, jl);
        for (size_t il = 0; il < A.localSize(); ++il)
        {
          const auto i = localIndex(A.nLocal, il);
          size_t lin = 0, stride = 1;
          for (size_t d = 0; d < 4; ++d) { lin += cells[d] * (i[d] + A.nLocal[d] * j[d]) * stride; stride *= tn[q][d]; }
          locIdx[q][il + A.localSize() * jl] = lin;
        }
      }
    }
    // each element writes disjoint entries of the T[q]; per-thread timers (summed: thread-seconds)
    std::vector<std::array<double, 4>> tt(nThreads, std::array<double, 4>{0, 0, 0, 0});
    parallelForEachElement([&](const std::array<size_t, 4> &c, int th) {
      double &tB = tt[th][0], &tG = tt[th][1], &tS = tt[th][2], &tScatter = tt[th][3];
      auto tq = clock::now();
      const Mat VtB = elementBsel(c, sel, true);
      tB += seconds(tq); tq = clock::now();
      // P_K S_K P_K = Z^T Z with Z = D^{1/2} V^T B P_K (fast diagonalization G_K^-1 = V D V^T; V^T B formed factor by
      // factor in elementBsel); only the blocks of this
      // pass's pairs are formed.  The scaling lets the TT-SVD see O(1) entries on every cell (graded meshes: unscaled
      // entries span orders of magnitude, and truncation relative to the largest makes the scaled operator indefinite).
      Eigen::VectorXd pK(soff[nt]);
      for (size_t a = 0; a < nt; ++a)
        if (sel[a])
          for (size_t il = 0; il < blocks[a].localSize(); ++il)
          {
            const auto i = localIndex(blocks[a].nLocal, il); double v = 1.0;
            for (size_t d = 0; d < 4; ++d) { const long k = blocks[a].index(d, c[d], i[d], cells[d]); v = k < 0 ? 0.0 : v * pf[a][d](size_t(k)); }
            pK(soff[a] + il) = v;
          }
      const Mat Z = sep->diagonal[classOf(c)].array().sqrt().matrix().asDiagonal() * VtB * pK.asDiagonal();
      tG += seconds(tq);
      for (size_t q = 0; q < ps.size(); ++q)
      {
        tq = clock::now();
        const size_t a = ps[q].first, b = ps[q].second, na = blocks[a].localSize(), nb = blocks[b].localSize();
        Mat SKq;
        if (a == b)
        {
          SKq = Mat::Zero(na, na);
          SKq.selfadjointView<Eigen::Lower>().rankUpdate(Z.middleCols(soff[a], na).transpose());
          SKq.triangularView<Eigen::StrictlyUpper>() = SKq.transpose();
        }
        else SKq.noalias() = Z.middleCols(soff[a], na).transpose() * Z.middleCols(soff[b], nb);
        tS += seconds(tq); tq = clock::now();
        size_t base = 0, stride = 1;
        for (size_t d = 0; d < 4; ++d) { base += c[d] * stride; stride *= tn[q][d]; }
        const size_t *loc = locIdx[q].data(); const double *sk = SKq.data(); double *Tq = T[q].data();
        for (size_t k = 0; k < na * nb; ++k) Tq[base + loc[k]] = sk[k];
        tScatter += seconds(tq);
      }
    });
    for (const auto &x : tt) { tB += x[0]; tG += x[1]; tS += x[2]; tScatter += x[3]; }
    tElements += seconds(te);
    const auto ts = clock::now();
    // the TT-SVDs of the pass's pairs are independent (each consumes its own T[q])
    std::vector<std::vector<Core3>> passCores(ps.size());
    {
      auto tq = clock::now();
      if (ps.size() == 1) passCores[0] = ttSvd(T[0], tn[0], opt.sTolerance, nThreads);
      else parallelFor(ps.size(), 1, [&](size_t q, int) { passCores[q] = ttSvd(T[q], tn[q], opt.sTolerance); });
      tTtSvd += seconds(tq);
    }
    for (size_t q = 0; q < ps.size(); ++q)
    {
      const size_t a = ps[q].first, b = ps[q].second;
      const auto &A = blocks[a], &Bb = blocks[b];
      std::vector<Core3> cores; cores.swap(passCores[q]);
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
    reportMemory("after S pass " + std::to_string(pass));
    ++pass;
  }
  std::cout << "explicit_S assembled: " << pass << " passes, element blocks " << tElements << " s (thread-seconds: V^T B_K " << tB << ", Z = D^1/2 V^T B P " << tG << ", S_K blocks " << tS
            << ", scatter " << tScatter << "), TT-SVD + expansion " << tSvd << " s (TT-SVD " << tTtSvd << ")" << std::endl;
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
  const auto tCt = clock::now();
  for (size_t a = 0; a < nt; ++a)
  {
    ct[a] = scaleVector(toTT(cFull[a], system.sizes[a + 1], 1e-12), a);
    std::cout << "explicit_S rhs block " << system.names[a + 1] << " ranks (1e-12)=" << ct[a].ranks_string();
    { TTVector cr = ct[a]; DpgBoba::round(cr, 1e-6); std::cout << " (1e-6)=" << cr.ranks_string(); }
    { TTVector cr = ct[a]; DpgBoba::round(cr, 1e-4); std::cout << " (1e-4)=" << cr.ranks_string() << std::endl; }
    cNorm2 += std::pow(boba::norm_frobenius(ct[a]), 2);
    x[a] = zero_tt_vector<4>(system.sizes[a + 1]);
  }
  const double cNorm = std::sqrt(cNorm2);
  std::cout << "explicit_S rhs to TT " << seconds(tCt) << " s" << std::endl;
  reportMemory("end of setup");
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

  // Matrix-free S~ on FULL trial vectors (Ifpack2-style power iteration, 11-13): S~ v = sum_K R_K^T P_K B_K^T G_K^-1 B_K
  // P_K R_K v, element by element, with B_K applied in factored form (field: per group the x/t cell factor and the
  // pre-summed velocity matrix of the velocity cell, two small GEMMs; traces: Kronecker products of the 1D face and
  // trace-mass factors) and G_K^-1 by fast diagonalization.  Nothing per element is stored; memory is a few full
  // vectors plus the per-(group, velocity cell) velocity matrices.
    // velocity matrices V_g(i, j) = sum_tau -c_tau Q3_tau(j) (x) Q2_tau(i), per group and velocity cell
    std::vector<std::vector<Mat>> Vg(groups.size());
    for (size_t g = 0; g < groups.size(); ++g)
    {
      if (groups[g].terms.empty()) continue;
      Vg[g].resize(cells[2] * cells[3]);
      for (size_t j = 0; j < cells[3]; ++j)
        for (size_t i = 0; i < cells[2]; ++i)
        {
          Mat V;
          for (const auto &t : groups[g].terms)
          {
            Mat term = (-t.coefficient) * kron(cellBlock(t.Q3, 3, j, nbT[3], nbF[3]), cellBlock(t.Q2, 2, i, nbT[2], nbF[2]));
            if (V.size() == 0) V = term; else V += term;
          }
          Vg[g][i + cells[2] * j] = V;
        }
    }
    const Eigen::Index nT01 = Eigen::Index(nbT[0] * nbT[1]), nT23 = Eigen::Index(nbT[2] * nbT[3]);
    const Eigen::Index nF01 = Eigen::Index(nbF[0] * nbF[1]), nF23 = Eigen::Index(nbF[2] * nbF[3]);
    auto traceFactors = [&](size_t a, const std::array<size_t, 4> &c) {
      const size_t nd = size_t(blocks[a].normal); std::array<Mat, 4> f;
      for (size_t d = 0; d < 4; ++d)
      {
        if (d == nd)
        {
          f[d] = Mat::Zero(nbT[d], 2); const Mat &F = axes[d].faces;
          if (c[d] >= 1) f[d].col(0) = F.block(c[d] * nbT[d], c[d] - 1, nbT[d], 1);
          if (c[d] + 1 <= cells[d] - 1) f[d].col(1) = F.block(c[d] * nbT[d], c[d], nbT[d], 1);
        }
        else f[d] = cellBlock(traceMass[d], d, c[d], nbT[d], nbJ[d]);
      }
      return f;
    };
    auto applyFull = [&](const std::vector<Eigen::VectorXd> &v) {
      std::vector<Eigen::VectorXd> out(nt);
      for (size_t a = 0; a < nt; ++a) out[a] = Eigen::VectorXd::Zero(fullSize[a]);
      std::vector<std::vector<long>> gidx(nt); std::vector<Eigen::VectorXd> pK(nt), vK(nt);
      forEachElement([&](const std::array<size_t, 4> &c) {
        Eigen::VectorXd t = Eigen::VectorXd::Zero(Eigen::Index(nlocT));
        std::array<Mat, 4> fx;   // per-group x/t factors are formed per element below
        for (size_t a = 0; a < nt; ++a)
        {
          const size_t n = blocks[a].localSize(); gidx[a].resize(n); pK[a].resize(Eigen::Index(n)); vK[a].resize(Eigen::Index(n));
          for (size_t il = 0; il < n; ++il)
          {
            const auto i = localIndex(blocks[a].nLocal, il); double pv = 1.0; long g = 0, stride = 1; bool ok = true;
            for (size_t d = 0; d < 4; ++d)
            {
              const long k = blocks[a].index(d, c[d], i[d], cells[d]);
              if (k < 0) { ok = false; break; }
              pv *= pf[a][d](size_t(k)); g += k * stride; stride *= long(blocks[a].global[d]);
            }
            gidx[a][il] = ok ? g : -1; pK[a](Eigen::Index(il)) = ok ? pv : 0.0; vK[a](Eigen::Index(il)) = ok ? pv * v[a](g) : 0.0;
          }
          if (blocks[a].normal < 0)
          {
            Eigen::Map<const Mat> U(vK[a].data(), nF01, nF23);
            for (size_t g = 0; g < groups.size(); ++g)
            {
              if (groups[g].terms.empty()) continue;
              const Mat XT = kron(cellBlock(groups[g].Q1, 1, c[1], nbT[1], nbF[1]), cellBlock(groups[g].Q0, 0, c[0], nbT[0], nbF[0]));
              Eigen::Map<Mat> T(t.data(), nT01, nT23);
              T.noalias() += XT * U * Vg[g][c[2] + cells[2] * c[3]].transpose();
            }
          }
          else t += kroneckerApply(traceFactors(a, c), Mat(vK[a]));
        }
        const Eigen::VectorXd w = separableGramSolve(*sep, classOf(c), Mat(t));
        for (size_t a = 0; a < nt; ++a)
        {
          Eigen::VectorXd y;
          if (blocks[a].normal < 0)
          {
            Eigen::Map<const Mat> W(w.data(), nT01, nT23); Mat Y = Mat::Zero(nF01, nF23);
            for (size_t g = 0; g < groups.size(); ++g)
            {
              if (groups[g].terms.empty()) continue;
              const Mat XT = kron(cellBlock(groups[g].Q1, 1, c[1], nbT[1], nbF[1]), cellBlock(groups[g].Q0, 0, c[0], nbT[0], nbF[0]));
              Y.noalias() += XT.transpose() * W * Vg[g][c[2] + cells[2] * c[3]];
            }
            y = Eigen::Map<const Eigen::VectorXd>(Y.data(), Y.size());
          }
          else
          {
            auto f = traceFactors(a, c); for (auto &m : f) m.transposeInPlace();
            y = kroneckerApply(f, Mat(w));
          }
          for (size_t il = 0; il < blocks[a].localSize(); ++il)
            if (gidx[a][il] >= 0) out[a](gidx[a][il]) += pK[a](Eigen::Index(il)) * y(Eigen::Index(il));
        }
      });
      return out;
    };
  if (opt.solver == "amen")
  {
    // block AMEn on S~ y = c~ (SPD, trial blocks only)
    const auto ta = clock::now();
    System sysS;
    sysS.matrix = BlockMatrix(nt, nt);
    for (size_t a = 0; a < nt; ++a) for (size_t b = 0; b < nt; ++b) sysS.matrix({a, b}) = St[a][b];
    for (size_t a = 0; a < nt; ++a) { sysS.names.push_back(system.names[a + 1]); sysS.sizes.push_back(system.sizes[a + 1]); }
    BlockVector rhsS(nt); for (size_t a = 0; a < nt; ++a) { rhsS(a) = ct[a]; DpgBoba::round(rhsS(a), 0.1 * opt.amen.roundingTolerance); }
    // AMEn's convergence certificate by the matrix-free apply on full vectors (the TT product S~ y has ranks
    // rank(S~) * rank(y) and took ~34 s of a 45 s level-1 solve)
    const auto residualS = [&](const BlockVector &y) {
      std::vector<Eigen::VectorXd> yFull(nt); double num = 0, den = 0;
      for (size_t a = 0; a < nt; ++a) yFull[a] = toFull(y(a));
      const auto Sy = applyFull(yFull);
      for (size_t a = 0; a < nt; ++a) { const Eigen::VectorXd ca = toFull(rhsS(a)); num += (ca - Sy[a]).squaredNorm(); den += ca.squaredNorm(); }
      return std::sqrt(num / den);
    };
    const bool anySplit = opt.split[0] || opt.split[1] || opt.split[2] || opt.split[3];
    BlockVector yS = anySplit ? solveSplitAxes(sysS, rhsS, axes, opt.split, 1e-14, opt.amen, false)
                              : DpgBoba::solveAmen(sysS.matrix, rhsS, opt.amen, std::function<double(const BlockVector &)>(residualS));
    const double amenSeconds = seconds(ta);
    const auto tr = clock::now();
    double rn2 = 0;
    {
      // residual by the matrix-free full-vector apply (exact; the TT St*y product was ~40 s at level 1)
      std::vector<Eigen::VectorXd> yFull(nt), cFullS(nt);
      for (size_t a = 0; a < nt; ++a) { x[a] = yS(a); yFull[a] = toFull(yS(a)); cFullS[a] = toFull(ct[a]); }
      const auto Sy = applyFull(yFull);
      for (size_t a = 0; a < nt; ++a) rn2 += (cFullS[a] - Sy[a]).squaredNorm();
    }
    std::cout << "explicit_S amen |r~|/|c~|=" << std::sqrt(rn2) / cNorm << " rank(y)=";
    for (size_t a = 0; a < nt; ++a) std::cout << (a ? "," : "") << maxRank4(x[a]);
    std::cout << " split=" << opt.split[0] << opt.split[1] << opt.split[2] << opt.split[3] << " seconds=" << seconds(ta) << " (solve " << amenSeconds << ", residual " << seconds(tr) << ")" << std::endl;
  }
  else
  {
  auto fullBounds = [&]() {
    const auto tf = clock::now();
    // Lanczos (CG coefficients) on full vectors from c~, for the Ritz values (lambda_min); exact arithmetic
    std::array<double, 3> result{0, 0, 0};
    {
      const auto tl = clock::now();
      std::vector<Eigen::VectorXd> rr(nt), pp(nt);
      for (size_t a = 0; a < nt; ++a) { rr[a] = toFull(ct[a]); pp[a] = rr[a]; }
      auto fdot = [&](const std::vector<Eigen::VectorXd> &u, const std::vector<Eigen::VectorXd> &w) { double d = 0; for (size_t a = 0; a < nt; ++a) d += u[a].dot(w[a]); return d; };
      double rz = fdot(rr, rr), alphaPrev = 0, betaPrev = 0; std::vector<double> dd, ee;
      for (int k = 0; k < opt.boundsSteps; ++k)
      {
        const auto q = applyFull(pp); const double alpha = rz / fdot(pp, q);
        for (size_t a = 0; a < nt; ++a) rr[a] -= alpha * q[a];
        const double rzNew = fdot(rr, rr), beta = rzNew / rz;
        dd.push_back(1.0 / alpha + (k > 0 ? betaPrev / alphaPrev : 0.0)); ee.push_back(std::sqrt(std::abs(beta)) / alpha);
        for (size_t a = 0; a < nt; ++a) pp[a] = rr[a] + beta * pp[a];
        rz = rzNew; alphaPrev = alpha; betaPrev = beta;
      }
      const size_t m = dd.size(); Mat Tm = Mat::Zero(m, m);
      for (size_t i = 0; i < m; ++i) { Tm(i, i) = dd[i]; if (i + 1 < m) Tm(i, i + 1) = Tm(i + 1, i) = ee[i]; }
      const Eigen::VectorXd ritz = Eigen::SelfAdjointEigenSolver<Mat>(Tm).eigenvalues();
      result[0] = ritz(0); result[1] = ritz(m - 1);
      std::cout << "explicit_S matrix-free Lanczos Ritz [" << ritz(0) << ", " << ritz(m - 1) << "] from " << m << " steps ("
                << seconds(tl) << " s)" << std::endl;
    }
    // Ifpack2-style deterministic power iteration on full vectors
    std::vector<Eigen::VectorXd> v(nt); size_t g0 = 0; double nrm2 = 0;
    for (size_t a = 0; a < nt; ++a)
    {
      v[a].resize(Eigen::Index(fullSize[a]));
      for (size_t g = 0; g < fullSize[a]; ++g) v[a](Eigen::Index(g)) = 1.0 + double((g0 + g) % 97);
      g0 += fullSize[a]; nrm2 += v[a].squaredNorm();
    }
    for (auto &b : v) b /= std::sqrt(nrm2);
    double est = 0, old = 0; int its = 0; double tApply = 0;
    for (its = 1; its <= opt.powerIterations; ++its)
    {
      const auto ta = clock::now(); v = applyFull(v); tApply += seconds(ta);
      double n2 = 0; for (const auto &b : v) n2 += b.squaredNorm();
      old = est; est = std::sqrt(n2); for (auto &b : v) b /= est;
      if (its > 1 && std::abs(est - old) < opt.powerStop * est) break;
    }
    its = std::min(its, opt.powerIterations);
    std::cout << "explicit_S matrix-free power iteration lambda_max ~ " << est << " after " << its << " its ("
              << tApply / its << " s per full-vector apply, " << seconds(tf) << " s in all)" << std::endl;
    reportMemory("after matrix-free bounds");
    result[2] = est;
    return result;
  };
  // --- spectrum bounds by TT Lanczos (CG coefficients)
  double lmin, lmax, ritzMax = 0;
  std::array<double, 3> fullBoundValues{0, 0, 0};
  if (opt.lambdaMaxMode == "fullpower")
  {
    // all bounds matrix-free on full vectors: Lanczos for lambda_min, power iteration for lambda_max (no TT work)
    fullBoundValues = fullBounds();
    lmin = fullBoundValues[0] * opt.lminSafety; ritzMax = fullBoundValues[1]; lmax = ritzMax * 1.05;
  }
  else
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
              << lmin << ", " << lmax << "] (upper: " << (opt.lambdaMaxMode == "bound" ? "element bound" : (opt.lambdaMaxMode == "power" || opt.lambdaMaxMode == "fullpower") ? "power iteration, below" : "Ritz max x 1.05")
              << ") kappa "
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
  auto powerStartVector = [&]() {
    if (opt.powerStart == "pattern") return patternStart();
    BlockVector shape(nt); for (size_t a = 0; a < nt; ++a) shape(a) = ct[a];
    const auto guess = DpgBoba::initialGuess(shape, 11);   // seeded rank-one TT per block
    std::vector<TTVector> v(nt); for (size_t a = 0; a < nt; ++a) v[a] = guess(a);
    return v;
  };
  if (opt.lambdaMaxMode == "fullpower")
  {
    const double est = fullBoundValues[2];
    // the exact Lanczos Ritz max is also a lower bound on lambda_max, and after 20 steps usually the better one
    lmax = std::max(opt.boostFactor * est, 1.05 * ritzMax);
    std::cout << "explicit_S using lambda_max = max(" << opt.boostFactor << " x " << est << ", 1.05 x Ritz max " << ritzMax << ") = "
              << lmax << ", lambda_min = " << opt.lminSafety << " x Ritz min = " << lmin << ", kappa " << lmax / lmin << std::endl;
  }
  if (opt.lambdaMaxMode == "power")
  {
    const auto tp = clock::now(); const double pt = opt.powerTolerance;
    auto v = powerStartVector(); double nv = normOf(v); for (auto &b : v) b *= 1.0 / nv;
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

  }

  reportMemory("after solve");
  // --- solution: x = P y; eps = G^{-1}(l - B x) element-locally
  BlockVector solution(system.names.size());
  std::vector<Eigen::VectorXd> xFull(nt);
  for (size_t a = 0; a < nt; ++a) { solution(a + 1) = scaleVector(x[a], a); DpgBoba::round(solution(a + 1), 1e-12); xFull[a] = toFull(solution(a + 1)); }
  const auto tEps = clock::now();
  Eigen::VectorXd eps = Eigen::VectorXd::Zero(lFull.size());
  // test dofs are element-local (discontinuous test space): disjoint writes
  parallelForEachElement([&](const std::array<size_t, 4> &c, int) {
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
  std::cout << "explicit_S eps = G^-1 (l - B x) element loop " << seconds(tEps) << " s" << std::endl;
  const auto tEpsTT = clock::now();
  {
    // threaded TT-SVD (same truncation rule as toTT: relative Frobenius tolerance)
    const Sizes &n = system.sizes[0];
    std::vector<double> T(eps.data(), eps.data() + eps.size()); Eigen::VectorXd().swap(eps);
    auto cores = ttSvd(T, {n[0], n[1], n[2], n[3]}, 1e-10, nThreads);
    TTVector r(n);
    for (size_t d = 0; d < 4; ++d)
    {
      const auto &G = cores[d]; r.cores[d].resize({G.r0, G.n, G.r1}); auto v = r.cores[d].view();
      for (size_t q = 0; q < G.r1; ++q) for (size_t i = 0; i < G.n; ++i) for (size_t l = 0; l < G.r0; ++l) v({l, i, q}) = G(l, i, q);
    }
    solution(0) = r;
  }
  std::cout << "explicit_S eps to TT " << seconds(tEpsTT) << " s" << std::endl;
  std::cout << "explicit_S total_seconds=" << seconds(t0) << std::endl;
  return solution;
}
} // namespace PlanarMITLBoba
