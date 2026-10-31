#pragma once

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
  double xWidth = 1, tWidth = 1, alpha = 1;
  double tolerance = 1e-8, svdTolerance = 1e-12, coefficientTolerance = 1e-12;
  int maxSweeps = 100, kickrank = 4, maxRank = 64;
  bool verbose = false, comparison = true;
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
    cmd.setOption("alpha", &alpha);
    cmd.setOption("tolerance", &tolerance); cmd.setOption("svdTolerance", &svdTolerance);
    cmd.setOption("coefficientTolerance", &coefficientTolerance);
    cmd.setOption("maxSweeps", &maxSweeps); cmd.setOption("kickrank", &kickrank);
    cmd.setOption("maxRank", &maxRank); cmd.setOption("verbose", "quiet", &verbose);
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

  std::array<int,4> counts() const { return {nx, nt, nux, nuz}; }
  std::vector<std::vector<double>> vertices() const
  {
    double uMax = properVelocityFromBeta(betaMax,c), uMin = properVelocityFromBeta(beta0,c);
    return {uniformVertices(0,xWidth,nx), uniformVertices(0,tWidth,nt),
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
    return {{"nx",nx},{"nt",nt},{"nux",nux},{"nuz",nuz},{"p",p},{"dp",dp},
            {"quadrature_degree",quadratureDegree},{"sample_degree",sampleDegree},
            {"beta0",beta0},{"betaMax",betaMax},{"c",c},{"q_over_m",qOverM},
            {"Ex",Ex},{"By",By},{"x_width",xWidth},{"t_width",tWidth},{"alpha",alpha},
            {"target",tolerance},{"svd_tolerance",svdTolerance},{"coefficient_tolerance",coefficientTolerance}};
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
