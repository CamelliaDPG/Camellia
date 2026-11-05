#pragma once
#include <set>

#include "PlanarMITLCommon.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>

namespace PlanarMITL
{
using ComparisonClock = std::chrono::steady_clock;
inline double elapsed(ComparisonClock::time_point start)
{
  return std::chrono::duration<double>(ComparisonClock::now() - start).count();
}

inline bool comparisonRequested(int argc, char **argv)
{
  for (int i = 1; i < argc; ++i)
    if (std::string(argv[i]) == "--comparison") return true;
  return false;
}

/// Identical continuum problem and discretization controls for both entry points.
struct ComparisonProblem
{
  int nx = 1, nt = 1, nux = 1, nuz = 1, p = 2, dp = 1;
  int quadratureDegree = 20, sampleDegree = 10;
  double beta0 = 0, betaMax = 0.5, c = 1, qOverM = -1, Ex = -1, By = 0.5;
  double xWidth = 1, tWidth = 1, alpha = 1, alphaMeshScale = 0;
  double tolerance = 1e-8, svdTolerance = 1e-12, coefficientTolerance = 1e-12;
  int maxSweeps = 100, kickrank = 4, maxRank = 64;
  int localSolver = 0, maxLocalIterations = 300;
  double localTolerance = 1e-6, rieszSvdTolerance = 0;
  bool verbose = false, comparison = true, splitModes = false, splitX = false, infSupEnrichment = false, gradedX = false;
  int xRefinements = 0;
  std::string splitAxes = "";
  std::string outputPrefix;

  int parse(int argc, char **argv, const std::string &backend)
  {
    outputPrefix = "mitl_" + backend;
    Teuchos::CommandLineProcessor cmd(false, true);
    cmd.setDocString("Matched prescribed-field MITL comparison: quadratic bubble product, s=beta.grad(f), zero exterior flux.\n");
    cmd.setOption("comparison", "no-comparison", &comparison);
    cmd.setOption("nx", &nx); cmd.setOption("nt", &nt);
    cmd.setOption("nux", &nux); cmd.setOption("nuz", &nuz);
    cmd.setOption("p", &p); cmd.setOption("dp", &dp);
    cmd.setOption("quadratureDegree", &quadratureDegree);
    cmd.setOption("sampleDegree", &sampleDegree);
    cmd.setOption("beta0", &beta0); cmd.setOption("betaMax", &betaMax);
    cmd.setOption("c", &c); cmd.setOption("qOverM", &qOverM);
    cmd.setOption("Ex", &Ex); cmd.setOption("By", &By);
    cmd.setOption("xWidth", &xWidth); cmd.setOption("tWidth", &tWidth);
    cmd.setOption("gradedX", "uniformX", &gradedX,
                  "Geometric x grading toward x=0 by bisection (as MeshFactory::geometricGradedIntervalMeshTopology): nx base cells");
    cmd.setOption("xRefinements", &xRefinements, "Uniform bisections of every x cell after building the base mesh");
    cmd.setOption("alpha", &alpha);
    cmd.setOption("alphaMeshScale", &alphaMeshScale,
                  "If positive, the graph-norm L^2 weight on x-cell K is max(alpha, alphaMeshScale / h_x(K)^2)");
    cmd.setOption("tolerance", &tolerance); cmd.setOption("svdTolerance", &svdTolerance);
    cmd.setOption("coefficientTolerance", &coefficientTolerance);
    cmd.setOption("maxSweeps", &maxSweeps); cmd.setOption("kickrank", &kickrank);
    cmd.setOption("maxRank", &maxRank); cmd.setOption("verbose", "quiet", &verbose);
    cmd.setOption("localSolver", &localSolver, "Tensor coupled site solver: 0=COD, 1=QR, 2=LU, 3=Eigen GMRES, 4=matrix-free BoBa GMRES, 5=sparse LU (dense LU if dense), 6=sparse LU (COD if dense)");
    cmd.setOption("localTolerance", &localTolerance);
    cmd.setOption("maxLocalIterations", &maxLocalIterations);
    cmd.setOption("splitModes", "no-splitModes", &splitModes,
                  "Tensor backend: split each (cell*basis) mode into basis and cell modes (eight TT modes)");
    cmd.setOption("splitX", "no-splitX", &splitX,
                  "Tensor backend: split only the x (cell*basis) mode into x-basis and x-cell modes (five TT modes)");
    cmd.setOption("splitAxes", &splitAxes,
                  "Tensor backend: split (basis, cell) modes of these axes: a 4-digit mask in (x,t,ux,uz), or \"auto\" for every multi-cell axis");
    cmd.setOption("infSupEnrichment", "no-infSupEnrichment", &infSupEnrichment,
                  "Tensor backend: kick the Riesz block with the projected image of the trial blocks");
    cmd.setOption("rieszSvdTolerance", &rieszSvdTolerance,
                  "Tensor backend: relative SVD tolerance for the Riesz block's local splits (<= 0: svdTolerance)");
    cmd.setOption("outputPrefix", &outputPrefix);
    auto result = cmd.parse(argc, argv);
    if (result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) return 2;
    if (result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) return 1;
    if (nx < 1 || nt < 1 || nux < 1 || nuz < 1 || p < 1 || dp < 0 ||
        !(c > 0) || !(betaMax > 0 && betaMax < 1) || !(beta0 > -1 && beta0 < betaMax) ||
        !(xWidth > 0) || !(tWidth > 0) || !(alpha > 0) ||
        quadratureDegree < 2*(p+dp) || sampleDegree < 2*std::max(2,p-1) ||
        !(tolerance > 0) || !(svdTolerance > 0) || !(coefficientTolerance > 0))
      throw std::invalid_argument("Invalid comparison parameters or underintegrated mass/error quadrature");
    return 0;
  }

  std::array<int,4> counts() const { return {int(xVertices().size()) - 1, nt, nux, nuz}; }
  //! x vertices: uniform, or bisection-graded toward x = 0 (widths 1/2, 1/4, ..., two cells of 2^{1-nx}), then
  //! xRefinements uniform bisections of every cell.
  std::vector<double> xVertices() const
  {
    std::vector<double> v;
    if (!gradedX || nx == 1) v = uniformVertices(0,xWidth,nx);
    else
    {
      std::set<double> s {0.0, xWidth};
      double right = xWidth;
      for (int i = 2; i <= nx; ++i) { right /= 2; s.insert(right); }
      v.assign(s.begin(), s.end());
    }
    for (int r = 0; r < xRefinements; ++r)
    {
      std::vector<double> refined {v.front()};
      for (size_t i = 1; i < v.size(); ++i) { refined.push_back(0.5*(v[i-1]+v[i])); refined.push_back(v[i]); }
      v = refined;
    }
    return v;
  }
  std::vector<std::vector<double>> vertices() const
  {
    double uMax = properVelocityFromBeta(betaMax,c), uMin = properVelocityFromBeta(beta0,c);
    return {xVertices(), uniformVertices(0,tWidth,nt),
            uniformVertices(uMin,uMax-uMin,nux), uniformVertices(-uMax,2*uMax,nuz)};
  }
  static double bubble(double y, double a, double b)
  {
    const double z = (y-a)/(b-a);
    return 4*z*(1-z);
  }
  static double bubbleDerivative(double y, double a, double b)
  {
    return 4*(1-2*(y-a)/(b-a))/(b-a);
  }
  std::map<std::string,double> metadata() const
  {
    return {{"nx",int(xVertices().size()) - 1},{"graded_x",double(gradedX)},{"x_refinements",double(xRefinements)},{"nt",nt},{"nux",nux},{"nuz",nuz},{"p",p},{"dp",dp},
            {"quadrature_degree",quadratureDegree},{"sample_degree",sampleDegree},
            {"beta0",beta0},{"betaMax",betaMax},{"c",c},{"q_over_m",qOverM},
            {"Ex",Ex},{"By",By},{"x_width",xWidth},{"t_width",tWidth},{"alpha",alpha},{"alpha_mesh_scale",alphaMeshScale},
            {"target",tolerance},{"svd_tolerance",svdTolerance},{"coefficient_tolerance",coefficientTolerance},
            {"max_sweeps",maxSweeps},{"kickrank",kickrank},{"max_rank",maxRank},
            {"local_solver",localSolver},{"local_tolerance",localTolerance},{"max_local_iterations",maxLocalIterations},{"split_modes",double(splitModes)},{"split_x",double(splitX)},{"inf_sup_enrichment",double(infSupEnrichment)},{"riesz_svd_tolerance",rieszSvdTolerance}};
  }
};

inline void writeMetrics(const std::string &path, const std::map<std::string,double> &metrics)
{
  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot write " + path);
  out << std::setprecision(17) << "{\n";
  size_t n = 0;
  for (const auto &entry : metrics)
  {
    out << "  \"" << entry.first << "\": ";
    if (std::isfinite(entry.second)) out << entry.second; else out << "null";
    out << (++n < metrics.size() ? ",\n" : "\n");
  }
  out << "}\n";
}
} // namespace PlanarMITL
