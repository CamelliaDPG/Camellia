#include "Camellia.h"
#include "TensorProductAssembly.hpp"
#include "TensorComparison.hpp"
#include <iomanip>

using namespace PlanarMITLBoba;

namespace
{
std::array<size_t, 4> multiindex(size_t ordinal, const Sizes &sizes)
{
  std::array<size_t, 4> result;
  for (size_t d = 0; d < 4; ++d)
  {
    result[d] = ordinal % sizes[d];
    ordinal /= sizes[d];
  }
  return result;
}

size_t product(const Sizes &sizes)
{
  size_t count = 1;
  for (size_t d = 0; d < 4; ++d) { count *= sizes[d]; }
  return count;
}

Eigen::VectorXd denseVector(const BlockVector &vector)
{
  size_t n = 0;
  for (size_t b = 0; b < vector.block_size; ++b) { n += product(vector(b).sizes()); }
  Eigen::VectorXd result(n);
  size_t offset = 0;
  for (size_t b = 0; b < vector.block_size; ++b)
  {
    auto tensor = vector(b).decompress();
    auto values = tensor.const_view();
    const size_t count = product(vector(b).sizes());
    for (size_t i = 0; i < count; ++i) { result(offset + i) = values(i); }
    offset += count;
  }
  return result;
}

/// Independent full quadrature evaluation using Camellia bases, without Lorentz SVD.
Eigen::MatrixXd validate(const System &system, const std::vector<Axis> &axes,
                         double c, double qOverM, double Ex, double By, double alpha,
                         double tolerance, int denseLimit)
{
  size_t total = 0;
  for (const auto &size : system.sizes) { total += product(size); }
  Sizes quadratureSizes;
  for (size_t d = 0; d < 4; ++d) { quadratureSizes[d] = axes[d].points.size(); }
  const size_t nq = product(quadratureSizes), nw = product(system.sizes[0]);
  const size_t nf = product(system.sizes[1]);
  if (total > size_t(denseLimit) || double(nq) * double(nw + nf) > 2e7)
  {
    throw std::invalid_argument("Dense validation size limit exceeded; use a smaller case or --no-validate");
  }
  Eigen::MatrixXd L(nq, nw), W(nq, nw), F(nq, nf);
  for (size_t k = 0; k < nq; ++k)
  {
    const auto q = multiindex(k, quadratureSizes);
    auto beta = PlanarMITL::transport(axes[2].points(q[2]), axes[3].points(q[3]), c, qOverM, Ex, By);
    double weight = 1.0;
    for (size_t d = 0; d < 4; ++d) { weight *= axes[d].weights(q[d]); }
    const double sqrtWeight = std::sqrt(weight);
    for (size_t i = 0; i < nw; ++i)
    {
      const auto index = multiindex(i, system.sizes[0]);
      double value = 1.0, transport = 0.0;
      for (size_t d = 0; d < 4; ++d) { value *= axes[d].test(q[d], index[d]); }
      for (size_t derivative = 0; derivative < 4; ++derivative)
      {
        double term = beta[derivative];
        for (size_t d = 0; d < 4; ++d)
        {
          term *= d == derivative ? axes[d].derivative(q[d], index[d]) : axes[d].test(q[d], index[d]);
        }
        transport += term;
      }
      L(k, i) = sqrtWeight * transport;
      W(k, i) = sqrtWeight * value;
    }
    for (size_t i = 0; i < nf; ++i)
    {
      const auto index = multiindex(i, system.sizes[1]);
      double value = sqrtWeight;
      for (size_t d = 0; d < 4; ++d) { value *= axes[d].field(q[d], index[d]); }
      F(k, i) = value;
    }
  }
  auto check = [&](const std::string &name, const Eigen::MatrixXd &actual, const Eigen::MatrixXd &reference) {
    const double error = (actual - reference).norm() / std::max(reference.norm(), 1e-30);
    std::cout << name << "_relative_error=" << error << '\n';
    if (!std::isfinite(error) || error > tolerance)
    {
      throw std::runtime_error(name + " failed independent quadrature comparison");
    }
  };
  check("transport", denseMatrix(system.transport), L);
  const Eigen::MatrixXd gram = L.transpose() * L + alpha * (W.transpose() * W);
  const Eigen::MatrixXd volume = -L.transpose() * F;
  check("gram", denseMatrix(system.matrix({0, 0})), gram);
  check("volume", denseMatrix(system.matrix({0, 1})), volume);
  Eigen::MatrixXd full = Eigen::MatrixXd::Zero(total, total);
  size_t rowOffset = 0;
  for (size_t i = 0; i < system.sizes.size(); ++i)
  {
    size_t colOffset = 0;
    for (size_t j = 0; j < system.sizes.size(); ++j)
    {
      const auto block = denseMatrix(system.matrix({i, j}));
      full.block(rowOffset, colOffset, block.rows(), block.cols()) = block;
      colOffset += block.cols();
    }
    rowOffset += product(system.sizes[i]);
  }
  const double symmetryError = (full - full.transpose()).norm() / full.norm();
  std::cout << "mixed_symmetry_error=" << symmetryError << '\n';
  if (!std::isfinite(symmetryError) || symmetryError > tolerance)
  {
    throw std::runtime_error("Mixed operator lost symmetry");
  }
  // Residual validation uses the uncompressed coefficients, including G's cross terms.
  full.topLeftCorner(nw, nw) = gram;
  full.block(0, nw, nw, nf) = volume;
  full.block(nw, 0, nf, nw) = volume.transpose();
  size_t traceOffset = nw + nf, traceBlock = 2;
  for (size_t normal = 0; normal < 4; ++normal)
  {
    if (axes[normal].faces.cols() == 0) { continue; }
    const size_t nj = product(system.sizes[traceBlock]);
    Eigen::MatrixXd flux(nw, nj);
    for (size_t j = 0; j < nj; ++j)
    {
      const auto column = multiindex(j, system.sizes[traceBlock]);
      for (size_t i = 0; i < nw; ++i)
      {
        const auto row = multiindex(i, system.sizes[0]);
        double value = axes[normal].faces(row[normal], column[normal]);
        for (size_t d = 0; d < 4; ++d)
        {
          if (d == normal) { continue; }
          double integral = 0.0;
          for (Eigen::Index k = 0; k < axes[d].points.size(); ++k)
          {
            integral += axes[d].weights(k) * axes[d].test(k, row[d]) * axes[d].trace(k, column[d]);
          }
          value *= integral;
        }
        flux(i, j) = value;
      }
    }
    check(system.names[traceBlock], denseMatrix(system.matrix({0, traceBlock})), flux);
    if (axes[normal].faces.colwise().sum().norm() > 1e-12)
    {
      throw std::runtime_error("Interior flux orientations do not cancel for constant tests");
    }
    full.block(0, traceOffset, nw, nj) = flux;
    full.block(traceOffset, 0, nj, nw) = flux.transpose();
    traceOffset += nj;
    ++traceBlock;
  }
  return full;
}

int run(int argc, char **argv)
{
  int nx = 1, nt = 1, nux = 1, nuz = 1, p = 1, dp = 1, quadratureDegree = -1, denseLimit = 2000;
  double beta0 = 0.0, betaMax = 0.5, c = 1.0, qOverM = -1.0, Ex = -1.0, By = 0.5;
  double xWidth = 1.0, tWidth = 1.0, alpha = 1.0, coefficientTolerance = 1e-12;
  bool doSolve = false, doValidate = true, graded = false;
  double coreCenter = 0.198801, sigma = 0.0497003;
  double coreWidthFactor = 4.0, coreHalfWidthFactor = 2.0, growthRatio = 8.0;
  DpgBoba::AmenOptions options;
  Teuchos::CommandLineProcessor cmd(false, true);
  cmd.setDocString("Prescribed-field tensorized planar MITL assembly and algebraic AMEn validation.\n"
                  "Uses homogeneous exterior flux and a manufactured discrete RHS, not cathode injection.\n"
                  "Order p matches the reference: field p-1, trace p, test p+dp.\n");
  cmd.setOption("nx", &nx); cmd.setOption("nt", &nt);
  cmd.setOption("nux", &nux); cmd.setOption("nuz", &nuz);
  cmd.setOption("p", &p); cmd.setOption("dp", &dp);
  cmd.setOption("quadratureDegree", &quadratureDegree);
  cmd.setOption("beta0", &beta0); cmd.setOption("betaMax", &betaMax); cmd.setOption("c", &c);
  cmd.setOption("qOverM", &qOverM); cmd.setOption("Ex", &Ex); cmd.setOption("By", &By);
  cmd.setOption("xWidth", &xWidth); cmd.setOption("tWidth", &tWidth);
  cmd.setOption("alpha", &alpha, "Graph norm mass weight; the reference MITL default is 1e6.");
  cmd.setOption("coefficientTolerance", &coefficientTolerance);
  cmd.setOption("svdTolerance", &options.roundingTolerance);
  cmd.setOption("tolerance", &options.tolerance);
  cmd.setOption("maxSweeps", &options.maxSweeps);
  cmd.setOption("kickrank", &options.kickrank); cmd.setOption("maxRank", &options.maxRank);
  cmd.setOption("localSolver", &options.localSolver, "0=COD, 1=QR, 2=LU, 3=Eigen GMRES, 4=matrix-free BoBa GMRES");
  cmd.setOption("localTolerance", &options.localTolerance);
  cmd.setOption("maxLocalIterations", &options.maxLocalIterations);
  cmd.setOption("verbose", "quiet", &options.verbose);
  cmd.setOption("solve", "no-solve", &doSolve);
  cmd.setOption("validate", "no-validate", &doValidate);
  cmd.setOption("denseLimit", &denseLimit);
  cmd.setOption("gradedVelocityMesh", "uniformVelocityMesh", &graded);
  cmd.setOption("coreCenter", &coreCenter); cmd.setOption("sigma", &sigma);
  cmd.setOption("velocityCoreWidthFactor", &coreWidthFactor);
  cmd.setOption("velocityCoreHalfWidthFactor", &coreHalfWidthFactor);
  cmd.setOption("velocityGrowthRatio", &growthRatio);
  const auto parsed = cmd.parse(argc, argv);
  if (parsed == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) { return 0; }
  if (parsed != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) { return 1; }
  if (nx < 1 || nt < 1 || nux < 1 || nuz < 1 || p < 1 || dp < 0 ||
      !(betaMax > 0.0) || !(xWidth > 0.0) || !(tWidth > 0.0) || !(alpha > 0.0) ||
      !(coefficientTolerance > 0.0 && coefficientTolerance < 1.0) ||
      !(options.roundingTolerance > 0.0 && options.roundingTolerance < 1.0) || denseLimit < 1)
  {
    throw std::invalid_argument("Invalid mesh, order, physical parameters, or tolerances");
  }
  const double uMax = PlanarMITL::properVelocityFromBeta(betaMax, c);
  const double uMin = PlanarMITL::properVelocityFromBeta(beta0, c);
  if (!(uMin < uMax)) { throw std::invalid_argument("Expected beta0 < betaMax"); }
  if (quadratureDegree < 0) { quadratureDegree = 2 * (p + dp) + 4; }
  if (quadratureDegree < 2 * (p + dp)) { throw std::invalid_argument("Quadrature underintegrates the test mass"); }
  std::vector<std::vector<double>> vertices = {
    PlanarMITL::uniformVertices(0, xWidth, nx), PlanarMITL::uniformVertices(0, tWidth, nt),
    PlanarMITL::uniformVertices(uMin, uMax - uMin, nux), PlanarMITL::uniformVertices(-uMax, 2 * uMax, nuz)};
  if (graded)
  {
    if (!(sigma > 0.0) || !(coreHalfWidthFactor > 0.0) || !(coreCenter >= uMin && coreCenter <= uMax))
    {
      throw std::invalid_argument("Invalid velocity core parameters");
    }
    vertices[2] = PlanarMITL::gradedVertices(uMin, uMax, coreCenter, coreWidthFactor * sigma, coreHalfWidthFactor * sigma, growthRatio);
    vertices[3] = PlanarMITL::gradedVertices(-uMax, uMax, 0, coreWidthFactor * sigma, coreHalfWidthFactor * sigma, growthRatio);
  }
  std::cout << std::setprecision(12)
            << "Prescribed-field MITL algebraic validation (no Maxwell coupling or emission)\n"
            << "mode_order=x,t,ux,uz beta0=" << beta0 << " betaMax=" << betaMax << " uMin=" << uMin << " uMax=" << uMax
            << " Ex=" << Ex << " By=" << By << " alpha=" << alpha << '\n';
  std::vector<Axis> axes;
  for (const auto &mesh : vertices) { axes.emplace_back(mesh, p, dp, quadratureDegree); }
  LorentzFactors h(axes[2], axes[3], c, coefficientTolerance);
  auto system = assemble(axes, h, qOverM, Ex, By, alpha, options.roundingTolerance);
  for (size_t b = 0; b < system.names.size(); ++b)
  {
    std::cout << "block=" << b << " name=" << system.names[b] << " sizes=" << system.sizes[b]
              << " test_row_ranks=" << system.matrix({0, b}).ranks_string() << '\n';
  }
  // An algebraic fixture with r=0 gives zero trial RHS blocks as in mixed DPG.
  BlockVector shape(system.sizes.size());
  for (size_t b = 0; b < shape.block_size; ++b) { shape(b) = zero_tt_vector<4>(system.sizes[b]); }
  auto exact = DpgBoba::initialGuess(shape, 19);
  exact(0).fill_with_zeros();
  auto rhs = DpgBoba::apply(system.matrix, exact, options.roundingTolerance);
  Eigen::MatrixXd dense;
  Eigen::VectorXd reference;
  if (doValidate)
  {
    dense = validate(system, axes, c, qOverM, Ex, By, alpha,
                     100 * (coefficientTolerance + options.roundingTolerance), denseLimit);
    auto factorization = dense.completeOrthogonalDecomposition();
    reference = factorization.solve(denseVector(rhs));
    const double residual = (dense * reference - denseVector(rhs)).norm() / denseVector(rhs).norm();
    std::cout << "dense_rank=" << factorization.rank() << '/' << dense.rows()
              << " dense_relative_residual=" << residual << '\n';
    if (!std::isfinite(residual) || residual > 1e-9) { throw std::runtime_error("Dense mixed solve failed"); }
  }
  if (doSolve)
  {
    auto solution = DpgBoba::solveAmen(system.matrix, rhs, options);
    const double residual = DpgBoba::reportResidual(system.matrix, rhs, solution,
                                                   std::min(1e-14, options.roundingTolerance * 0.01));
    if (doValidate)
    {
      const auto vector = denseVector(solution);
      const double denseResidual = (dense * vector - denseVector(rhs)).norm() / denseVector(rhs).norm();
      std::cout << "dense_evaluated_amen_residual=" << denseResidual
                << " amen_dense_solution_difference=" << (vector - reference).norm() / reference.norm() << '\n';
      if (!std::isfinite(denseResidual) || denseResidual > options.tolerance)
      {
        throw std::runtime_error("AMEn failed the independently evaluated residual target");
      }
    }
    if (!std::isfinite(residual) || residual > options.tolerance)
    {
      throw std::runtime_error("AMEn did not reach the mixed residual target");
    }
  }
  return 0;
}
} // namespace

int main(int argc, char **argv)
{
  Teuchos::GlobalMPISession mpi(&argc, &argv, nullptr);
  if (mpi.getNProc() != 1)
  {
    if (mpi.getRank() == 0) { std::cerr << "PlanarMITLBobaDriver currently requires one MPI rank\n"; }
    return 1;
  }
  Kokkos::initialize(argc, argv);
  boba::init();
  int status = 0;
  try { status = PlanarMITL::comparisonRequested(argc, argv) ? runTensorComparison(argc, argv) : run(argc, argv); }
  catch (const std::exception &error) { std::cerr << "PlanarMITLBobaDriver: " << error.what() << '\n'; status = 1; }
  boba::finalize();
  Camellia::finalize(false);
  return status;
}
