#pragma once
// One space-time slab of the planar MITL Vlasov problem solved by the TT backend (explicit Schur complement + block
// AMEn), with the interfaces the coupled Vlasov-Maxwell fixed-point loop needs:
//   - fields E_x(x, t), B_y(x, t) from values on the slab's (x, t) quadrature grid (low rank by SVD);
//   - the cathode inflow density (the inflow load is linear in it);
//   - t = t_n data: the previous slab's final-time flux trace (t^ = f on the face t = t_n);
//   - solve(): warm-started from the previous substep's iterate; returns the relative L2 change in f;
//   - moments n, int w_x f, int w_z f at arbitrary (x, t) points (w = u / gamma or u);
//   - the final-time trace for the next slab.
//
// Exterior faces other than t = T are prescribed: the cathode inflow data at x = x0, zero elsewhere (the outflow parts
// of the anode and velocity faces included; see plan-tt-coupled-production.md).
#include "TensorComparison.hpp"

namespace PlanarMITLBoba
{
namespace SlabDetail
{
using Mat = Eigen::MatrixXd;
/// core d of a TT vector as the (r0 * n) x r1 matrix (r0 fastest)
inline Mat coreMatrix(const TTVector &v, size_t d)
{
  const auto &c = v.cores[d]; auto view = c.const_view();
  const size_t r0 = c.sizes(0), n = c.sizes(1), r1 = c.sizes(2);
  Mat out(r0 * n, r1);
  for (size_t b = 0; b < r1; ++b) for (size_t i = 0; i < n; ++i) for (size_t a = 0; a < r0; ++a) out(a + r0 * i, b) = view({a, i, b});
  return out;
}
/// a^T (M_0 x M_1 x M_2 x M_3) b for TT vectors a, b and per-mode matrices M_d (n_d x n_d)
inline double weightedInner(const TTVector &a, const TTVector &b, const std::array<Mat, 4> &M)
{
  Mat E = Mat::Ones(1, 1);   // (rank of a) x (rank of b)
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &ca = a.cores[d], &cb = b.cores[d];
    const size_t ra0 = ca.sizes(0), n = ca.sizes(1), ra1 = ca.sizes(2), rb0 = cb.sizes(0), rb1 = cb.sizes(2);
    auto va = ca.const_view(), vb = cb.const_view();
    // T(i, beta0, alpha1) = sum_alpha0 E(alpha0, beta0) a(alpha0, i, alpha1)
    Mat next = Mat::Zero(ra1, rb1);
    for (size_t i = 0; i < n; ++i)
    {
      Mat Ai(ra0, ra1); for (size_t x = 0; x < ra1; ++x) for (size_t y = 0; y < ra0; ++y) Ai(y, x) = va({y, i, x});
      const Mat EA = E.transpose() * Ai;   // rb0 x ra1
      for (size_t j = 0; j < n; ++j)
      {
        const double m = M[d](Eigen::Index(i), Eigen::Index(j));
        if (m == 0.0) continue;
        Mat Bj(rb0, rb1); for (size_t x = 0; x < rb1; ++x) for (size_t y = 0; y < rb0; ++y) Bj(y, x) = vb({y, j, x});
        next.noalias() += m * EA.transpose() * Bj;
      }
    }
    E = next;
  }
  return E(0, 0);
}
} // namespace SlabDetail

class TensorVlasovSlab
{
public:
  using Mat = Eigen::MatrixXd;

  /// problem: the TT settings (p, dp, quadratureDegree, c, qOverM, test norm, solver options, inflow profile);
  /// vertices: the slab's x, t, u_x, u_z mesh vertices.  Ex, By in problem are ignored: the fields are set by setFields.
  TensorVlasovSlab(const PlanarMITL::ComparisonProblem &problem, const std::vector<std::vector<double>> &vertices)
      : _problem(problem)
  {
    if (vertices.size() != 4) throw std::invalid_argument("TensorVlasovSlab: four axes of vertices");
    basisChoice() = BasisChoice{problem.testBasis, problem.fieldBasis, problem.traceBasis};
    for (const auto &mesh : vertices) _axes.emplace_back(mesh, problem.p, problem.dp, problem.quadratureDegree);
    _axes[1].freeUpperFace();   // t = T: the outflow face whose trace is the next slab's initial data
    _h = std::make_unique<LorentzFactors>(_axes[2], _axes[3], problem.c, problem.coefficientTolerance);
    _norm.alpha = problem.alpha; _norm.meshScale = problem.alphaMeshScale; _norm.transportScale = problem.alphaTransportScale;
    _norm.c = problem.c; _norm.accelerationBounds = problem.accelerationBoundValues(); _norm.separable = problem.separableTestNorm;
    _problem.Ex = 1.0; _problem.By = 1.0;   // the profile carries the fields
    _fields.x0 = vertices[0].front(); _fields.x1 = vertices[0].back();
    _fields.t0 = vertices[1].front(); _fields.t1 = vertices[1].back();
    setFields(Mat::Zero(xPoints().size(), tPoints().size()), Mat::Zero(xPoints().size(), tPoints().size()));
  }

  const std::vector<Axis> &axes() const { return _axes; }
  /// the (x, t) quadrature points at which setFields takes the field values
  const Eigen::VectorXd &xPoints() const { return _axes[0].points; }
  const Eigen::VectorXd &tPoints() const { return _axes[1].points; }

  /// E_x and B_y at (xPoints()(i), tPoints()(j)); compressed by an SVD (relative tolerance fieldTolerance)
  void setFields(const Mat &E, const Mat &B, double fieldTolerance = 1e-10)
  {
    _fieldRanks[0] = _fields.fromGrid(0, E, fieldTolerance);
    _fieldRanks[1] = _fields.fromGrid(1, B, fieldTolerance);
    _fields.variation = 0.0;
  }
  std::array<int, 2> fieldRanks() const { return _fieldRanks; }
  /// cathode inflow density n (the Maxwellian's density; its shape is problem.inflowU0, inflowSigma)
  void setInflowDensity(double n) { _inflowDensity = n; }
  /// t = t_n data from the previous slab (same x and velocity meshes): its final-time trace
  void setInitialTrace(const TTVector &previousFluxT, size_t previousFinalFace)
  {
    // load += (test_d^T W trace_d for d != t) x (lowerEnd_t e_last^T) applied to the previous flux_t trace:
    // B's trace term at the lower exterior face is -t^ w with t^ = f(t_n), moved to the right-hand side
    boba::Array<host_matrix, 4> factors;
    for (size_t d = 0; d < 4; ++d)
    {
      Mat m;
      if (d == 1)
      {
        m = Mat::Zero(_axes[1].test.cols(), Eigen::Index(previousFluxT.sizes()[1]));
        m.col(Eigen::Index(previousFinalFace)) = _axes[1].lowerEnd;
      }
      else
      {
        m = _axes[d].test.transpose() * _axes[d].weights.asDiagonal() * _axes[d].trace;
        if (m.cols() != Eigen::Index(previousFluxT.sizes()[d]))
          throw std::invalid_argument("TensorVlasovSlab::setInitialTrace: the previous slab's mesh differs in direction " + std::to_string(d));
      }
      factors[d] = hostMatrix(m);
    }
    _initialLoad = boba::make_ttm_from_matrices<4, host_space, double>(factors) * previousFluxT;
    DpgBoba::round(_initialLoad, _problem.svdTolerance);
    _hasInitial = true;
  }
  void setInitialTrace(const TensorVlasovSlab &previous) { setInitialTrace(previous.finalTrace(), previous.finalFace()); }

  /// solve with the current fields and data; returns |f - f_previous| / |f| in L2 (1 on the first solve)
  double solve()
  {
    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();
    fieldProfile() = _fields;
    _problem.inflowDensity = _inflowDensity;
    _problem.mitlInflow = true;
    auto system = assemble(_axes, *_h, _problem.qOverM, _problem.Ex, _problem.By, _norm, _problem.svdTolerance);
    auto rhs = inflowRHS(system, _axes, *_h, _problem);
    if (_hasInitial) { rhs(0) += _initialLoad; DpgBoba::round(rhs(0), _problem.svdTolerance); }
    _names = system.names;
    auto ex = explicitSchurOptions(_problem);
    ex.solver = "amen";
    // the (basis, cell) mode split, as --splitX / --splitAxes (default: none, the measured-fastest choice)
    if (_problem.splitX) ex.split[0] = true;
    if (_problem.splitAxes == "auto") for (size_t d = 0; d < 4; ++d) ex.split[d] = _axes[d].vertices.size() > 2;
    else if (_problem.splitAxes.size() == 4) for (size_t d = 0; d < 4; ++d) ex.split[d] = _problem.splitAxes[d] == '1';
    ex.computeErrorRepresentation = false;
    ex.iterate = &_iterate;
    const double assembly = std::chrono::duration<double>(clock::now() - t0).count();
    auto solution = solveExplicitSchur(system, rhs, _axes, *_h, _problem.qOverM, _problem.Ex, _problem.By, ex, _norm);
    const double total = std::chrono::duration<double>(clock::now() - t0).count();
    // relative change in f, in L2 (field mass on each axis)
    std::array<Mat, 4> M;
    for (size_t d = 0; d < 4; ++d) M[d] = _axes[d].field.transpose() * _axes[d].weights.asDiagonal() * _axes[d].field;
    const TTVector &f = solution(1);
    const double ff = SlabDetail::weightedInner(f, f, M);
    double change = 1.0;
    if (_solved)
    {
      const double gg = SlabDetail::weightedInner(_solution(1), _solution(1), M), fg = SlabDetail::weightedInner(f, _solution(1), M);
      change = std::sqrt(std::max(0.0, ff - 2 * fg + gg) / ff);
    }
    _solution = solution; _solved = true;
    std::cout << "tt_slab solve: fields rank " << _fieldRanks[0] << "," << _fieldRanks[1] << " inflow n " << _inflowDensity
              << " initial " << (_hasInitial ? "trace" : "zero") << " |f| " << std::sqrt(ff) << " rel_change " << change
              << " assembly " << assembly << " s total " << total << " s" << std::endl;
    return change;
  }

  bool solved() const { return _solved; }
  const BlockVector &solution() const { return _solution; }
  const TTVector &f() const { return _solution(1); }
  /// the flux_t trace block (its t mode: interior faces, then the final face t = T)
  const TTVector &finalTrace() const { return _solution(block("flux_t")); }
  size_t finalFace() const { return size_t(_axes[1].faces.cols()) - 1; }

  /// moments at the points (x(i), t(i)): density int f du, and int w_x f du, int w_z f du with w = u / gamma
  /// (relativistic) or w = u
  struct Moments { Eigen::VectorXd density, x, z; };
  Moments moments(const Eigen::VectorXd &x, const Eigen::VectorXd &t, bool relativistic) const
  {
    if (!_solved) throw std::logic_error("TensorVlasovSlab::moments before solve");
    const auto &ux = _axes[2], &uz = _axes[3];
    // velocity weights as sums of separable terms (u_x factor, u_z factor) at the velocity quadrature points
    using Terms = std::vector<std::pair<Eigen::VectorXd, Eigen::VectorXd>>;
    const Eigen::VectorXd onesX = Eigen::VectorXd::Ones(ux.points.size()), onesZ = Eigen::VectorXd::Ones(uz.points.size());
    Terms density{{onesX, onesZ}}, wx, wz;
    if (relativistic)
      for (Eigen::Index k = 0; k < _h->left.cols(); ++k)
      {
        wx.push_back({ux.points.cwiseProduct(_h->left.col(k)), _h->right.col(k)});
        wz.push_back({_h->left.col(k), uz.points.cwiseProduct(_h->right.col(k))});
      }
    else { wx.push_back({ux.points, onesZ}); wz.push_back({onesX, uz.points}); }
    const TTVector &fv = f();
    const Mat X = SlabDetail::coreMatrix(fv, 0);                       // (nx) x r1
    const Mat Tc = SlabDetail::coreMatrix(fv, 1), Uc = SlabDetail::coreMatrix(fv, 2), Zc = SlabDetail::coreMatrix(fv, 3);
    const size_t r1 = fv.cores[1].sizes(0), nt = fv.cores[1].sizes(1), r2 = fv.cores[1].sizes(2);
    const size_t nu = fv.cores[2].sizes(1), r3 = fv.cores[2].sizes(2), nz = fv.cores[3].sizes(1);
    const Mat Px = _axes[0].fieldValuesAt(x), Pt = _axes[1].fieldValuesAt(t);
    const Mat PX = Px * X;                                              // points x r1
    auto evaluate = [&](const Terms &terms) {
      Eigen::VectorXd v2 = Eigen::VectorXd::Zero(Eigen::Index(r2));
      for (const auto &term : terms)
      {
        const Eigen::VectorXd gx = ux.field.transpose() * ux.weights.cwiseProduct(term.first);
        const Eigen::VectorXd gz = uz.field.transpose() * uz.weights.cwiseProduct(term.second);
        Eigen::VectorXd v3 = Eigen::VectorXd::Zero(Eigen::Index(r3));   // sum_j Z(:, j, 0) gz(j)
        for (size_t j = 0; j < nz; ++j) v3 += Zc.middleRows(Eigen::Index(j * r3), Eigen::Index(r3)) * gz(Eigen::Index(j));
        for (size_t i = 0; i < nu; ++i) v2 += gx(Eigen::Index(i)) * (Uc.middleRows(Eigen::Index(i * r2), Eigen::Index(r2)) * v3);
      }
      Mat Y = Mat::Zero(Eigen::Index(nt), Eigen::Index(r1));                       // Y(j, a) = sum_b T(a, j, b) v2(b)
      const Eigen::VectorXd tv = Tc * v2;                                // (r1 * nt)
      for (size_t j = 0; j < nt; ++j) Y.row(Eigen::Index(j)) = tv.segment(Eigen::Index(j * r1), Eigen::Index(r1)).transpose();
      return Eigen::VectorXd((PX.array() * (Pt * Y).array()).rowwise().sum());
    };
    return {evaluate(density), evaluate(wx), evaluate(wz)};
  }

private:
  size_t block(const std::string &name) const
  {
    for (size_t b = 0; b < _names.size(); ++b) if (_names[b] == name) return b;
    throw std::logic_error("TensorVlasovSlab: no block " + name);
  }
  PlanarMITL::ComparisonProblem _problem;
  std::vector<Axis> _axes;
  std::unique_ptr<LorentzFactors> _h;
  TestNormWeight _norm;
  FieldProfile _fields;
  std::array<int, 2> _fieldRanks{0, 0};
  double _inflowDensity = 0;
  TTVector _initialLoad; bool _hasInitial = false;
  BlockVector _iterate, _solution; bool _solved = false;
  std::vector<std::string> _names;
};
} // namespace PlanarMITLBoba
