#pragma once

#include "Basis.h"
#include "BasisFamily.h"
#include "CubatureFactory.h"
#include "../common/BlockAmenSolver.hpp"
#include "../../PlanarMITL/PlanarMITLCommon.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <array>
#include <string>
#include <vector>

namespace PlanarMITLBoba
{
static_assert(Kokkos::SpaceAccessibility<Kokkos::HostSpace, Camellia::DeviceMemorySpace>::accessible,
              "PlanarMITLBobaDriver currently requires host-accessible Camellia basis data");
using TTMatrix = host_tt_matrix<4>;
using TTVector = host_tt_vector<4>;
using BlockMatrix = boba::BlockOperator<TTMatrix>;
using BlockVector = boba::BlockVector<TTVector>;
using Sizes = boba::Array<size_t, 4>;

inline host_matrix hostMatrix(const Eigen::MatrixXd &matrix)
{
  host_matrix result({size_t(matrix.rows()), size_t(matrix.cols())});
  auto values = result.view();
  for (Eigen::Index j = 0; j < matrix.cols(); ++j)
  {
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
    {
      values({size_t(i), size_t(j)}) = matrix(i, j);
    }
  }
  return result;
}

inline Eigen::MatrixXd denseMatrix(const TTMatrix &matrix)
{
  auto full = matrix.decompress();
  auto values = full.const_view();
  Eigen::MatrixXd result(matrix.rows(), matrix.cols());
  for (Eigen::Index j = 0; j < result.cols(); ++j)
  {
    for (Eigen::Index i = 0; i < result.rows(); ++i)
    {
      result(i, j) = values({size_t(i), size_t(j)});
    }
  }
  return result;
}

/// 1D basis families for the test, field, and trace spaces ("nodal" or "hierarchical"; default nodal).  These span
/// the same spaces; the choice changes only the conditioning of the assembled blocks.
struct BasisChoice
{
  std::string test = "nodal", field = "nodal", trace = "nodal";
  static Camellia::BasisFamilyPtr family(const std::string &name)
  {
    if (name == "nodal") return Camellia::BasisFamily::nodalBasisFamily();
    if (name == "hierarchical") return Camellia::BasisFamily::hierarchicalBasisFamily();
    throw std::invalid_argument("basis family must be nodal or hierarchical, got " + name);
  }
};
inline BasisChoice &basisChoice() { static BasisChoice choice; return choice; }

/// Broken 1D bases on arbitrary increasing vertices; physical derivatives and weights.
struct Axis
{
  std::vector<double> vertices;
  Eigen::VectorXd points, weights;
  Eigen::MatrixXd test, derivative, field, trace, faces;

  Axis(const std::vector<double> &mesh, int p, int dp, int quadratureDegree) : vertices(mesh)
  {
    auto line = Camellia::CellTopology::line();
    const auto &choice = basisChoice();
    auto w = BasisChoice::family(choice.test)->getBasis(line, Camellia::efsForSpace(Camellia::HGRAD_DISC), p + dp);
    // BasisFamily takes H1 order, including for L2 bases (whose degree is one less).
    auto f = BasisChoice::family(choice.field)->getBasis(line, Camellia::efsForSpace(Camellia::L2), p);
    auto j = BasisChoice::family(choice.trace)->getBasis(line, Camellia::efsForSpace(Camellia::L2), p + 1);
    if (w->getCardinality() != p + dp + 1 || f->getCardinality() != p || j->getCardinality() != p + 1)
    {
      throw std::runtime_error("Unexpected Camellia one-dimensional basis order convention");
    }
    auto cubature = Camellia::CubatureFactory().create(line, quadratureDegree);
    auto q = cubature->allocateCubaturePoints();
    auto weightsRef = cubature->allocateCubatureWeights();
    cubature->getCubature(q, weightsRef);
    const int nq = q.extent_int(0), cells = int(vertices.size()) - 1;
    points.resize(cells * nq);
    weights.resize(cells * nq);
    for (int cell = 0; cell < cells; ++cell)
    {
      const double jac = (vertices[cell + 1] - vertices[cell]) / 2.0;
      if (!(jac > 0.0))
      {
        throw std::invalid_argument("Mesh vertices must be strictly increasing");
      }
      for (int k = 0; k < nq; ++k)
      {
        points(cell * nq + k) = vertices[cell] + jac * (1.0 + q(k, 0));
        weights(cell * nq + k) = jac * weightsRef(k);
      }
    }
    auto evaluate = [&](Camellia::BasisPtr basis, bool grad) {
      auto op = grad ? Intrepid2::OPERATOR_GRAD : Intrepid2::OPERATOR_VALUE;
      auto values = basis->allocateBasisValues(q, op);
      basis->getValues(values, q, op);
      const int nb = basis->getCardinality();
      Eigen::MatrixXd result = Eigen::MatrixXd::Zero(cells * nq, cells * nb);
      for (int cell = 0; cell < cells; ++cell)
      {
        const double jac = (vertices[cell + 1] - vertices[cell]) / 2.0;
        for (int k = 0; k < nq; ++k)
        {
          for (int b = 0; b < nb; ++b)
          {
            result(cell * nq + k, cell * nb + b) = grad ? values(b, k, 0) / jac : values(b, k);
          }
        }
      }
      return result;
    };
    test = evaluate(w, false);
    derivative = evaluate(w, true);
    field = evaluate(f, false);
    trace = evaluate(j, false);
    Kokkos::DynRankView<double, Camellia::DeviceType> endpointStorage("MITL endpoints", 2, 1);
    Intrepid2::TensorPoints<double, Camellia::DeviceType> endpoints(endpointStorage);
    endpoints(0, 0) = -1.0;
    endpoints(1, 0) = 1.0;
    auto endpointValues = w->allocateBasisValues(endpoints, Intrepid2::OPERATOR_VALUE);
    w->getValues(endpointValues, endpoints, Intrepid2::OPERATOR_VALUE);
    const int nb = w->getCardinality();
    faces = Eigen::MatrixXd::Zero(cells * nb, cells - 1);
    for (int face = 1; face < cells; ++face)
    {
      for (int b = 0; b < nb; ++b)
      {
        faces((face - 1) * nb + b, face - 1) = endpointValues(b, 1);
        faces(face * nb + b, face - 1) = -endpointValues(b, 0);
      }
    }
  }
};

/// Weighted SVD of h(ux,uz); its factors are values on physical quadrature points.
struct LorentzFactors
{
  Eigen::MatrixXd left, right;
  double relativeError;

  LorentzFactors(const Axis &ux, const Axis &uz, double c, double tolerance)
  {
    Eigen::MatrixXd weighted(ux.points.size(), uz.points.size());
    for (Eigen::Index i = 0; i < weighted.rows(); ++i)
    {
      for (Eigen::Index j = 0; j < weighted.cols(); ++j)
      {
        weighted(i, j) = std::sqrt(ux.weights(i) * uz.weights(j)) *
                         PlanarMITL::inverseGamma(ux.points(i), uz.points(j), c);
      }
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(weighted, Eigen::ComputeThinU | Eigen::ComputeThinV);
    const auto singular = svd.singularValues();
    int rank = int(singular.size());
    double discarded = 0.0;
    while (rank > 1 && discarded + singular(rank - 1) * singular(rank - 1) <=
                         tolerance * tolerance * singular.squaredNorm())
    {
      discarded += singular(rank - 1) * singular(rank - 1);
      --rank;
    }
    left = ux.weights.array().sqrt().inverse().matrix().asDiagonal() *
           svd.matrixU().leftCols(rank) * singular.head(rank).asDiagonal();
    right = uz.weights.array().sqrt().inverse().matrix().asDiagonal() * svd.matrixV().leftCols(rank);
    relativeError = std::sqrt(discarded / singular.squaredNorm());
    std::cout << "inverse_gamma_rank=" << rank << " weighted_relative_error=" << relativeError << '\n';
  }
};

/// (x, t) profiles of the prescribed fields: E_x(x, t) = Ex eX(x) eT(t), B_y(x, t) = By bX(x) bT(t), each rank one in
/// (x, t); with s, tau the positions scaled to [0, 1] on the x and t axes and v = variation:
///   eX = 1 + v cos(pi s), eT = 1 + v tau, bX = 1 + v (2 s - 1), bT = 1 - 0.6 v tau.
/// variation = 0 (the default): constant fields.  Set by the comparison driver; read by the assembly routines.
struct FieldProfile
{
  double variation = 0, x0 = 0, x1 = 1, t0 = 0, t1 = 1;
  double eX(double x) const { return 1 + variation * std::cos(M_PI * (x - x0) / (x1 - x0)); }
  double eT(double t) const { return 1 + variation * (t - t0) / (t1 - t0); }
  double bX(double x) const { return 1 + variation * (2 * (x - x0) / (x1 - x0) - 1); }
  double bT(double t) const { return 1 - 0.6 * variation * (t - t0) / (t1 - t0); }
  bool varying() const { return variation != 0.0; }
  /// profile values at an axis's quadrature points: field 0 = E, 1 = B; axis 0 = x, 1 = t
  Eigen::VectorXd values(const Eigen::VectorXd &points, int field, int axis) const
  {
    Eigen::VectorXd v(points.size());
    for (Eigen::Index i = 0; i < points.size(); ++i)
      v(i) = field == 0 ? (axis == 0 ? eX(points(i)) : eT(points(i))) : (axis == 0 ? bX(points(i)) : bT(points(i)));
    return v;
  }
  /// max over the domain of E_x / Ex and B_y / By (variation >= 0)
  double maxE() const { return std::pow(1 + std::abs(variation), 2); }
  double maxB() const { return 1 + std::abs(variation); }
};
inline FieldProfile &fieldProfile() { static FieldProfile profile; return profile; }

struct System
{
  BlockMatrix matrix;
  TTMatrix transport, fieldValues;
  std::vector<std::string> names;
  std::vector<Sizes> sizes;
};

/// Build sqrt(W)*L, then B=-L^T W F and G=L^T W L+alpha*M.
/// The four TT modes are (cell*basis) for x,t,ux,uz, with x fastest.
/// Graph-norm L^2 weight options.  meshScale > 0: alpha_K = max(alpha, meshScale / h_x^2).  transportScale > 0 (the
/// production weight, Camellia aa406de0): alpha_K = alpha + C [ (v_x,max(K) / h_x)^2 + 1 / h_t^2 + sum_u (a_u / h_u)^2 ],
/// v_x,max(K) = max |u_x| / gamma over K's velocity box, a_u fixed bounds on the acceleration per velocity direction.
struct TestNormWeight
{
  double alpha = 1, meshScale = 0, transportScale = 0, c = 1;
  std::array<double, 2> accelerationBounds = {0, 0}; // (u_x, u_z)
  /// N5: direction-separable graph norm alpha_K |w|^2 + sum_d s_d(K)^2 |d_d w|^2, s = (v_x,max(K), 1, a_x, a_z) (the
  /// acceleration bounds), instead of |beta.grad w|^2 (Camellia --separableGraphTestNorm).  Field-independent.
  bool separable = false;
};

/// v_x,max(K)^2 over the (u_x cell, u_z cell) pairs, split exactly as sum_k left_k(i) right_k(j) by an SVD.
inline std::vector<std::pair<std::vector<double>, std::vector<double>>> vxMaxSquaredTerms(const std::vector<Axis> &axes, double c,
                                                                                          Eigen::MatrixXd *matrix = nullptr)
{
  const size_t nux = axes[2].vertices.size() - 1, nuz = axes[3].vertices.size() - 1;
  Eigen::MatrixXd vmax2(nux, nuz);
  for (size_t i = 0; i < nux; ++i)
  {
    const double ux = std::max(std::abs(axes[2].vertices[i]), std::abs(axes[2].vertices[i + 1]));
    for (size_t j = 0; j < nuz; ++j)
    {
      const double z0 = axes[3].vertices[j], z1 = axes[3].vertices[j + 1];
      const double uz = (z0 <= 0.0 && z1 >= 0.0) ? 0.0 : std::min(std::abs(z0), std::abs(z1));
      const double v = ux * PlanarMITL::inverseGamma(ux, uz, c);
      vmax2(i, j) = v * v;
    }
  }
  if (matrix) *matrix = vmax2;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(vmax2, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const auto sigma = svd.singularValues();
  std::vector<std::pair<std::vector<double>, std::vector<double>>> terms;
  for (Eigen::Index k = 0; k < sigma.size(); ++k)
  {
    if (sigma(k) <= 1e-14 * sigma(0)) break;
    std::vector<double> left(nux), right(nuz);
    for (size_t i = 0; i < nux; ++i) left[i] = sigma(k) * svd.matrixU()(i, k);
    for (size_t j = 0; j < nuz; ++j) right[j] = svd.matrixV()(j, k);
    terms.push_back({left, right});
  }
  return terms;
}

/// Values per cell, expanded to the axis's quadrature points.
inline Eigen::VectorXd cellValues(const Axis &axis, const std::vector<double> &perCell)
{
  const int cells = int(axis.vertices.size()) - 1, nq = int(axis.points.size()) / cells;
  Eigen::VectorXd values(axis.points.size());
  for (int cell = 0; cell < cells; ++cell) values.segment(cell * nq, nq).setConstant(perCell[cell]);
  return values;
}

/// Weighted test mass matrix: a sum of Kronecker terms, each a product of per-axis weights (given at quadrature points).
inline TTMatrix weightedTestMass(const std::vector<Axis> &axes, const std::vector<std::array<Eigen::VectorXd, 4>> &terms,
                                 double tolerance)
{
  TTMatrix mass;
  for (size_t k = 0; k < terms.size(); ++k)
  {
    boba::Array<host_matrix, 4> factors;
    for (size_t d = 0; d < 4; ++d)
    {
      const Eigen::VectorXd w = axes[d].weights.array() * terms[k][d].array();
      factors[d] = hostMatrix(axes[d].test.transpose() * w.asDiagonal() * axes[d].test);
    }
    auto term = boba::make_ttm_from_matrices<4, host_space, double>(factors);
    if (k == 0) mass = term; else { mass += term; DpgBoba::round(mass, tolerance); }
  }
  return mass;
}

/// The Kronecker terms of the L^2 weight alpha_K (see TestNormWeight).
inline std::vector<std::array<Eigen::VectorXd, 4>> testNormWeightTerms(const std::vector<Axis> &axes,
                                                                       const TestNormWeight &norm)
{
  std::array<Eigen::VectorXd, 4> ones;
  std::array<std::vector<double>, 4> widths;
  for (size_t d = 0; d < 4; ++d)
  {
    ones[d] = Eigen::VectorXd::Ones(axes[d].points.size());
    for (size_t i = 0; i + 1 < axes[d].vertices.size(); ++i) widths[d].push_back(axes[d].vertices[i + 1] - axes[d].vertices[i]);
  }
  auto inverseSquares = [&](size_t d, double scale) {
    std::vector<double> values;
    for (double h : widths[d]) values.push_back(scale / (h * h));
    return cellValues(axes[d], values);
  };
  std::vector<std::array<Eigen::VectorXd, 4>> terms;
  if (norm.transportScale > 0.0)
  {
    const double C = norm.transportScale;
    terms.push_back(ones); terms.back()[0] *= norm.alpha;
    terms.push_back(ones); terms.back()[1] = inverseSquares(1, C);
    for (int u = 0; u < 2; ++u)
    {
      const double a = norm.accelerationBounds[u];
      if (a == 0.0) continue;
      terms.push_back(ones); terms.back()[2 + u] = inverseSquares(2 + u, C * a * a);
    }
    // (v_x,max(K) / h_x)^2: v_x,max depends on the (u_x, u_z) cell pair; split that small matrix exactly by SVD.
    const size_t nux = widths[2].size(), nuz = widths[3].size();
    Eigen::MatrixXd vmax2;
    for (const auto &lr : vxMaxSquaredTerms(axes, norm.c, &vmax2))
    {
      terms.push_back(ones);
      terms.back()[0] = inverseSquares(0, C);
      terms.back()[2] = cellValues(axes[2], lr.first);
      terms.back()[3] = cellValues(axes[3], lr.second);
    }
    double lo = 1e300, hi = 0;
    for (size_t ix = 0; ix < widths[0].size(); ++ix)
      for (size_t it = 0; it < widths[1].size(); ++it)
        for (size_t i = 0; i < nux; ++i)
          for (size_t j = 0; j < nuz; ++j)
          {
            const double w = norm.alpha + C * (vmax2(i, j) / std::pow(widths[0][ix], 2) + 1.0 / std::pow(widths[1][it], 2) +
                             std::pow(norm.accelerationBounds[0] / widths[2][i], 2) +
                             std::pow(norm.accelerationBounds[1] / widths[3][j], 2));
            lo = std::min(lo, w); hi = std::max(hi, w);
          }
    std::cout << "transport_scale_weight terms=" << terms.size() << " alpha_K range " << lo << " .. " << hi << '\n';
  }
  else
  {
    std::vector<double> values;
    for (double h : widths[0]) values.push_back(norm.meshScale > 0.0 ? std::max(norm.alpha, norm.meshScale / (h * h)) : norm.alpha);
    terms.push_back(ones); terms.back()[0] = cellValues(axes[0], values);
  }
  return terms;
}

/// The derivative terms of the N5 norm, sum_d s_d(K)^2 int d_d w d_d v, as Kronecker terms: (derivative axis, per-axis
/// weights at quadrature points).  s = (v_x,max(K), 1, a_x, a_z), the a_u the fixed acceleration bounds.
inline std::vector<std::pair<size_t, std::array<Eigen::VectorXd, 4>>> separableStiffnessTerms(const std::vector<Axis> &axes,
                                                                                               const TestNormWeight &norm)
{
  std::array<Eigen::VectorXd, 4> ones;
  for (size_t d = 0; d < 4; ++d) ones[d] = Eigen::VectorXd::Ones(axes[d].points.size());
  std::vector<std::pair<size_t, std::array<Eigen::VectorXd, 4>>> terms;
  for (const auto &lr : vxMaxSquaredTerms(axes, norm.c))
  {
    auto w = ones; w[2] = cellValues(axes[2], lr.first); w[3] = cellValues(axes[3], lr.second);
    terms.push_back({0, w});
  }
  terms.push_back({1, ones});
  for (int u = 0; u < 2; ++u)
  {
    const double a = norm.accelerationBounds[u];
    if (!(a > 0.0)) throw std::invalid_argument("separable test norm: acceleration bounds must be positive");
    auto w = ones; w[0] *= a * a;
    terms.push_back({size_t(2 + u), w});
  }
  return terms;
}

inline TTMatrix separableStiffness(const std::vector<Axis> &axes, const TestNormWeight &norm, double tolerance)
{
  TTMatrix sum; bool first = true;
  for (const auto &term : separableStiffnessTerms(axes, norm))
  {
    boba::Array<host_matrix, 4> factors;
    for (size_t d = 0; d < 4; ++d)
    {
      const Eigen::MatrixXd &A = d == term.first ? axes[d].derivative : axes[d].test;
      const Eigen::VectorXd w = axes[d].weights.array() * term.second[d].array();
      factors[d] = hostMatrix(A.transpose() * w.asDiagonal() * A);
    }
    auto t = boba::make_ttm_from_matrices<4, host_space, double>(factors);
    if (first) { sum = t; first = false; } else { sum += t; DpgBoba::round(sum, tolerance); }
  }
  return sum;
}

inline System assemble(const std::vector<Axis> &axes, const LorentzFactors &h,
                       double qOverM, double Ex, double By, const TestNormWeight &norm, double tolerance)
{
  boba::Array<host_matrix, 4> evaluation, fieldEvaluation;
  Sizes testSizes, fieldSizes;
  for (size_t d = 0; d < 4; ++d)
  {
    const Eigen::VectorXd sqrtW = axes[d].weights.array().sqrt();
    evaluation[d] = hostMatrix(sqrtW.asDiagonal() * axes[d].test);
    fieldEvaluation[d] = hostMatrix(sqrtW.asDiagonal() * axes[d].field);
    testSizes[d] = axes[d].test.cols();
    fieldSizes[d] = axes[d].field.cols();
  }
  // Graph-norm L^2 term: a weighted test mass matrix, alpha_K constant on each element (TestNormWeight).
  auto mass = weightedTestMass(axes, testNormWeightTerms(axes, norm), tolerance);
  auto fieldValues = boba::make_ttm_from_matrices<4, host_space, double>(fieldEvaluation);
  // field: -1 none, 0 E_x, 1 B_y: the (x, t) profile (FieldProfile) multiplies the x and t weights
  const auto &profile = fieldProfile();
  auto term = [&](size_t derivativeAxis, const Eigen::VectorXd &ux,
                  const Eigen::VectorXd &uz, double coefficient, int field = -1) {
    auto factors = evaluation;
    for (size_t d = 0; d < 4; ++d)
    {
      Eigen::VectorXd weight = axes[d].weights.array().sqrt();
      if (d == 2) { weight.array() *= ux.array(); }
      if (d == 3) { weight.array() *= uz.array(); }
      if (field >= 0 && profile.varying() && d < 2) { weight.array() *= profile.values(axes[d].points, field, int(d)).array(); }
      factors[d] = hostMatrix(weight.asDiagonal() *
                              (d == derivativeAxis ? axes[d].derivative : axes[d].test));
    }
    return coefficient * boba::make_ttm_from_matrices<4, host_space, double>(factors);
  };
  const Eigen::VectorXd onesX = Eigen::VectorXd::Ones(axes[2].points.size());
  const Eigen::VectorXd onesZ = Eigen::VectorXd::Ones(axes[3].points.size());
  auto transport = term(1, onesX, onesZ, 1.0);
  if (Ex != 0.0) { transport += term(2, onesX, onesZ, qOverM * Ex, 0); }
  for (Eigen::Index k = 0; k < h.left.cols(); ++k)
  {
    Eigen::VectorXd vx = h.left.col(k).array() * axes[2].points.array();
    Eigen::VectorXd vz = h.right.col(k).array() * axes[3].points.array();
    transport += term(0, vx, h.right.col(k), 1.0);
    if (By != 0.0)
    {
      transport += term(2, h.left.col(k), vz, -qOverM * By, 1);
      transport += term(3, vx, h.right.col(k), qOverM * By, 1);
    }
    DpgBoba::round(transport, tolerance);
  }
  TTMatrix gram;
  if (norm.separable) gram = mass + separableStiffness(axes, norm, tolerance);
  else gram = transport.transpose() * transport + mass;
  DpgBoba::round(gram, tolerance);
  auto volume = -1.0 * (transport.transpose() * fieldValues);
  DpgBoba::round(volume, tolerance);
  std::vector<TTMatrix> bilinear = {volume};
  std::vector<std::string> names = {"test", "f"};
  std::vector<Sizes> sizes = {testSizes, fieldSizes};
  const std::array<std::string, 4> directions = {"x", "t", "ux", "uz"};
  for (size_t normal = 0; normal < 4; ++normal)
  {
    // Homogeneous exterior normal flux; a one-cell direction has no free faces.
    if (axes[normal].faces.cols() == 0) { continue; }
    boba::Array<host_matrix, 4> factors;
    Sizes traceSizes;
    for (size_t d = 0; d < 4; ++d)
    {
      Eigen::MatrixXd matrix;
      if (d == normal) { matrix = axes[d].faces; }
      else { matrix = axes[d].test.transpose() * axes[d].weights.asDiagonal() * axes[d].trace; }
      factors[d] = hostMatrix(matrix);
      traceSizes[d] = matrix.cols();
    }
    bilinear.push_back(boba::make_ttm_from_matrices<4, host_space, double>(factors));
    names.push_back("flux_" + directions[normal]);
    sizes.push_back(traceSizes);
  }
  BlockMatrix matrix(names.size(), names.size());
  for (size_t i = 0; i < names.size(); ++i)
  {
    for (size_t j = 0; j < names.size(); ++j)
    {
      matrix({i, j}) = zero_tt_matrix<4>(sizes[i], sizes[j]);
    }
  }
  matrix({0, 0}) = gram;
  for (size_t j = 1; j < names.size(); ++j)
  {
    matrix({0, j}) = bilinear[j - 1];
    matrix({j, 0}) = bilinear[j - 1].transpose();
  }
  return {matrix, transport, fieldValues, names, sizes};
}
} // namespace PlanarMITLBoba
