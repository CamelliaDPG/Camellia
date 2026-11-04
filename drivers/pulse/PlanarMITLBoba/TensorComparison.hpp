#pragma once

#include "ModeSplitting.hpp"
#include <cstdlib>
#include "TensorProductAssembly.hpp"
#include "../../PlanarMITL/ComparisonProblem.hpp"

namespace PlanarMITLBoba
{
/// Integrate s=beta.grad(product of bubbles), including the continuous internal flux.
/// This is a continuum manufactured RHS, independently of the trial basis.
inline BlockVector comparisonRHS(const System &system, const std::vector<Axis> &axes,
                                const LorentzFactors &h, const PlanarMITL::ComparisonProblem &problem)
{
  std::array<Eigen::VectorXd,4> value, derivative;
  for (size_t d = 0; d < 4; ++d)
  {
    const auto &axis = axes[d];
    value[d].resize(axis.points.size()); derivative[d].resize(axis.points.size());
    for (Eigen::Index i = 0; i < axis.points.size(); ++i)
    {
      value[d](i) = problem.bubble(axis.points(i),axis.vertices.front(),axis.vertices.back());
      derivative[d](i) = problem.bubbleDerivative(axis.points(i),axis.vertices.front(),axis.vertices.back());
    }
  }
  auto term = [&](size_t direction, const Eigen::VectorXd &ux,
                  const Eigen::VectorXd &uz, double coefficient) {
    boba::Array<host_vector,4> factors;
    for (size_t d = 0; d < 4; ++d)
    {
      Eigen::VectorXd weighted = axes[d].weights.array()*(d == direction ? derivative[d] : value[d]).array();
      if (d == 2) weighted.array() *= ux.array();
      if (d == 3) weighted.array() *= uz.array();
      Eigen::VectorXd integral = axes[d].test.transpose()*weighted;
      factors[d] = host_vector({size_t(integral.size())});
      auto view = factors[d].view();
      for (Eigen::Index i = 0; i < integral.size(); ++i) view(i) = integral(i);
    }
    return coefficient*boba::make_tt_from_vectors<4,host_space,double>(factors);
  };
  const Eigen::VectorXd onesX = Eigen::VectorXd::Ones(axes[2].points.size());
  const Eigen::VectorXd onesZ = Eigen::VectorXd::Ones(axes[3].points.size());
  auto source = term(1,onesX,onesZ,1);
  if (problem.Ex != 0) source += term(2,onesX,onesZ,problem.qOverM*problem.Ex);
  for (Eigen::Index k = 0; k < h.left.cols(); ++k)
  {
    Eigen::VectorXd vx = h.left.col(k).array()*axes[2].points.array();
    Eigen::VectorXd vz = h.right.col(k).array()*axes[3].points.array();
    source += term(0,vx,h.right.col(k),1);
    if (problem.By != 0)
    {
      source += term(2,h.left.col(k),vz,-problem.qOverM*problem.By);
      source += term(3,vx,h.right.col(k),problem.qOverM*problem.By);
    }
    DpgBoba::round(source,problem.svdTolerance);
  }
  BlockVector rhs(system.sizes.size());
  for (size_t b = 0; b < rhs.block_size; ++b) rhs(b) = zero_tt_vector<4>(system.sizes[b]);
  rhs(0) = source;
  return rhs;
}

inline int runTensorComparison(int argc, char **argv)
{
  using namespace PlanarMITL;
  ComparisonProblem problem;
  const int parsed = problem.parse(argc,argv,"tensor");
  if (parsed) return parsed == 2 ? 0 : parsed;
  DpgBoba::AmenOptions options;
  options.tolerance = problem.tolerance; options.roundingTolerance = problem.svdTolerance;
  options.maxSweeps = problem.maxSweeps; options.kickrank = problem.kickrank;
  options.maxRank = problem.maxRank; options.verbose = problem.verbose;
  options.localSolver = problem.localSolver;
  options.localTolerance = problem.localTolerance;
  options.maxLocalIterations = problem.maxLocalIterations;
  options.infSupEnrichment = problem.infSupEnrichment;
  options.rieszTolerance = problem.rieszSvdTolerance;
  const auto start = ComparisonClock::now();
  const auto vertices = problem.vertices();
  std::vector<Axis> axes;
  for (const auto &mesh : vertices) axes.emplace_back(mesh,problem.p,problem.dp,problem.quadratureDegree);
  LorentzFactors h(axes[2],axes[3],problem.c,problem.coefficientTolerance);
  auto system = assemble(axes,h,problem.qOverM,problem.Ex,problem.By,problem.alpha,problem.svdTolerance,
                         problem.alphaMeshScale);
  auto rhs = comparisonRHS(system,axes,h,problem);
  const double assemblySeconds = elapsed(start);
  const auto solveStart = ComparisonClock::now();
  BlockVector solution;
  std::array<bool,4> splitMask{false,false,false,false};
  if (problem.splitX) splitMask[0] = true;
  if (problem.splitAxes == "auto")
    for (size_t d = 0; d < 4; ++d) splitMask[d] = axes[d].vertices.size() > 2;
  else if (!problem.splitAxes.empty())
  {
    if (problem.splitAxes.size() != 4) throw std::invalid_argument("--splitAxes takes a 4-digit mask or auto");
    for (size_t d = 0; d < 4; ++d) splitMask[d] = problem.splitAxes[d] == '1';
  }
  if (splitMask[0] || splitMask[1] || splitMask[2] || splitMask[3])
  {
    std::cout << "split_axes=" << splitMask[0] << splitMask[1] << splitMask[2] << splitMask[3] << '\n';
    solution = solveSplitAxes(system,rhs,axes,splitMask,problem.svdTolerance,options,
                              std::getenv("PLANAR_MITL_CHECK_SPLIT") != nullptr);
  }
  else if (problem.splitModes)
  {
    SplitBlockMatrix splitMatrix; SplitBlockVector splitRHS; ModeSplit split;
    splitSystem(system,rhs,axes,problem.svdTolerance,splitMatrix,splitRHS,split);
    for (size_t b = 0; b < system.names.size(); ++b)
      std::cout << "split block=" << system.names[b] << " sizes=" << splitRHS(b).sizes()
                << " test_row_ranks=" << splitMatrix({0,b}).ranks_string() << '\n';
    if (std::getenv("PLANAR_MITL_CHECK_SPLIT"))
    {
      // Consistency check: (split A)(split x) merged must equal A x for a random four-mode x.
      auto x = DpgBoba::initialGuess(rhs,3);
      SplitBlockVector sx(x.block_size);
      for (size_t b = 0; b < x.block_size; ++b) sx(b) = splitVector(x(b),split.basisSizes[b],1e-14);
      auto merged = mergeSolution(sx);
      for (size_t i = 0; i < x.block_size; ++i)
      {
        auto d = merged(i); d -= x(i);
        auto Ax = system.matrix({i,0})*x(0); auto SAx = splitMatrix({i,0})*sx(0);
        for (size_t j = 1; j < x.block_size; ++j) { Ax += system.matrix({i,j})*x(j); SAx += splitMatrix({i,j})*sx(j); }
        BlockVector one(1); SplitBlockVector sone(1); sone(0) = SAx; one = mergeSolution(sone);
        auto e = one(0); e -= Ax;
        std::cout << "split_check block=" << system.names[i] << " vector_roundtrip=" << boba::norm_frobenius(d)/boba::norm_frobenius(x(i))
                  << " apply_difference=" << boba::norm_frobenius(e)/(boba::norm_frobenius(Ax)+1e-300) << '\n';
      }
    }
    solution = mergeSolution(DpgBoba::solveAmen(splitMatrix,splitRHS,options));
  }
  else solution = DpgBoba::solveAmen(system.matrix,rhs,options);
  const double solveSeconds = elapsed(solveStart);
  const double totalSeconds = elapsed(start);

  // Residual evaluation, storage accounting, and output are outside the timed solve.
  auto metrics = problem.metadata();
  metrics["assembly_seconds"] = assemblySeconds;
  metrics["linear_solve_seconds"] = solveSeconds;
  metrics["total_seconds"] = totalSeconds;
  metrics["inverse_gamma_rank"] = h.left.cols();
  metrics["inverse_gamma_svd_error"] = h.relativeError;
  metrics["mixed_residual"] = DpgBoba::reportResidual(system.matrix,rhs,solution,
                                                    std::min(1e-14,problem.svdTolerance*0.01));
  double storedMatrix = 0, storedSolution = 0, storedRHS = 0, fullTrial = 0, fullMixed = 0;
  for (size_t b = 0; b < solution.block_size; ++b)
  {
    const double full = solution(b).get_full_size();
    storedSolution += solution(b).get_number_elements();
    storedRHS += rhs(b).get_number_elements();
    fullMixed += full;
    if (b) fullTrial += full;
    std::cout << "comparison block=" << system.names[b] << " solution_ranks=" << solution(b).ranks_string()
              << " scalars=" << solution(b).get_number_elements() << " full=" << full << '\n';
    for (size_t j = 0; j < solution.block_size; ++j)
      storedMatrix += system.matrix({b,j}).get_number_elements();
  }
  metrics["tt_matrix_scalars"] = storedMatrix;
  metrics["tt_solution_scalars"] = storedSolution;
  metrics["tt_rhs_scalars"] = storedRHS;
  metrics["full_trial_dofs"] = fullTrial;
  metrics["full_mixed_dofs"] = fullMixed;
  metrics["field_dofs"] = solution(1).get_full_size();
  metrics["tt_field_scalars"] = solution(1).get_number_elements();
  metrics["field_compression"] = solution(1).get_full_size()/solution(1).get_number_elements();
  metrics["mixed_vector_compression"] = fullMixed/storedSolution;
  const double nw = solution(0).get_full_size();
  metrics["dense_active_operator_scalars"] = nw*nw+2*nw*fullTrial;
  metrics["dense_operator_compression"] = metrics["dense_active_operator_scalars"]/storedMatrix;

  std::vector<Axis> samples;
  boba::Array<host_matrix,4> evaluation;
  Sizes sampleSizes;
  for (size_t d = 0; d < 4; ++d)
  {
    samples.emplace_back(vertices[d],problem.p,problem.dp,problem.sampleDegree);
    evaluation[d] = hostMatrix(samples[d].field);
    sampleSizes[d] = samples[d].points.size();
  }
  auto sampled = boba::make_ttm_from_matrices<4,host_space,double>(evaluation)*solution(1);
  auto full = sampled.decompress();
  auto values = full.const_view();
  std::ofstream out(problem.outputPrefix+"_samples.csv");
  if (!out) throw std::runtime_error("Cannot write comparison samples");
  out << std::setprecision(17) << "x,t,ux,uz,weight,f,exact\n";
  size_t count = 1;
  for (size_t d = 0; d < 4; ++d) count *= sampleSizes[d];
  double norm2 = 0, error2 = 0;
  for (size_t k = 0; k < count; ++k)
  {
    size_t ordinal = k;
    double weight = 1, exact = 1;
    for (size_t d = 0; d < 4; ++d)
    {
      const size_t i = ordinal%sampleSizes[d]; ordinal /= sampleSizes[d];
      const double point = samples[d].points(i);
      out << point << ',';
      weight *= samples[d].weights(i);
      exact *= problem.bubble(point,vertices[d].front(),vertices[d].back());
    }
    const double value = values(k);
    out << weight << ',' << value << ',' << exact << '\n';
    norm2 += weight*exact*exact;
    error2 += weight*(value-exact)*(value-exact);
  }
  metrics["relative_exact_l2_error"] = std::sqrt(error2/norm2);
  const bool passed = std::isfinite(metrics["mixed_residual"]) && metrics["mixed_residual"] <= problem.tolerance;
  metrics["solve_passed"] = passed ? 1 : 0;
  writeMetrics(problem.outputPrefix+"_metrics.json",metrics);
  std::cout << std::setprecision(12) << "tensor total_seconds=" << totalSeconds
            << " relative_exact_l2_error=" << metrics["relative_exact_l2_error"]
            << " field_compression=" << metrics["field_compression"] << '\n';
  return passed ? 0 : 2; // preserve failed-run diagnostics for the comparison table
}
} // namespace PlanarMITLBoba
