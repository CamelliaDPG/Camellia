#pragma once

#include "Basis.h"
#include "BasisFamily.h"
#include "CubatureFactory.h"
#include "../common/BlockAmenSolver.hpp"
#include "../../PlanarMITL/PlanarMITLCommon.hpp"
#include <Eigen/Dense>
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

/// Broken 1D bases on arbitrary increasing vertices; physical derivatives and weights.
struct Axis
{
  std::vector<double> vertices;
  Eigen::VectorXd points, weights;
  Eigen::MatrixXd test, derivative, field, trace, faces;

  Axis(const std::vector<double> &mesh, int p, int dp, int quadratureDegree) : vertices(mesh)
  {
    auto family = Camellia::BasisFamily::nodalBasisFamily();
    auto line = Camellia::CellTopology::line();
    auto w = family->getBasis(line, Camellia::efsForSpace(Camellia::HGRAD_DISC), p + dp);
    // BasisFamily takes H1 order, including for L2 bases (whose degree is one less).
    auto f = family->getBasis(line, Camellia::efsForSpace(Camellia::L2), p);
    auto j = family->getBasis(line, Camellia::efsForSpace(Camellia::L2), p + 1);
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

struct System
{
  BlockMatrix matrix;
  TTMatrix transport, fieldValues;
  std::vector<std::string> names;
  std::vector<Sizes> sizes;
};

/// Build sqrt(W)*L, then B=-L^T W F and G=L^T W L+alpha*M.
/// The four TT modes are (cell*basis) for x,t,ux,uz, with x fastest.
inline System assemble(const std::vector<Axis> &axes, const LorentzFactors &h,
                       double qOverM, double Ex, double By, double alpha, double tolerance)
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
  auto massEvaluation = boba::make_ttm_from_matrices<4, host_space, double>(evaluation);
  auto fieldValues = boba::make_ttm_from_matrices<4, host_space, double>(fieldEvaluation);
  auto term = [&](size_t derivativeAxis, const Eigen::VectorXd &ux,
                  const Eigen::VectorXd &uz, double coefficient) {
    auto factors = evaluation;
    for (size_t d = 0; d < 4; ++d)
    {
      Eigen::VectorXd weight = axes[d].weights.array().sqrt();
      if (d == 2) { weight.array() *= ux.array(); }
      if (d == 3) { weight.array() *= uz.array(); }
      factors[d] = hostMatrix(weight.asDiagonal() *
                              (d == derivativeAxis ? axes[d].derivative : axes[d].test));
    }
    return coefficient * boba::make_ttm_from_matrices<4, host_space, double>(factors);
  };
  const Eigen::VectorXd onesX = Eigen::VectorXd::Ones(axes[2].points.size());
  const Eigen::VectorXd onesZ = Eigen::VectorXd::Ones(axes[3].points.size());
  auto transport = term(1, onesX, onesZ, 1.0);
  if (Ex != 0.0) { transport += term(2, onesX, onesZ, qOverM * Ex); }
  for (Eigen::Index k = 0; k < h.left.cols(); ++k)
  {
    Eigen::VectorXd vx = h.left.col(k).array() * axes[2].points.array();
    Eigen::VectorXd vz = h.right.col(k).array() * axes[3].points.array();
    transport += term(0, vx, h.right.col(k), 1.0);
    if (By != 0.0)
    {
      transport += term(2, h.left.col(k), vz, -qOverM * By);
      transport += term(3, vx, h.right.col(k), qOverM * By);
    }
    DpgBoba::round(transport, tolerance);
  }
  auto gram = transport.transpose() * transport + alpha * (massEvaluation.transpose() * massEvaluation);
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
