#pragma once
#include <set>

#include "PlanarMITLCommon.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
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
  double xWidth = 1, tWidth = 1, alpha = 1, alphaMeshScale = 0, alphaTransportScale = 0;
  std::string accelerationBounds = "auto";
  // (x, t)-varying prescribed fields (both backends; PlanarMITLBoba::FieldProfile): E_x = Ex (1 + v cos(pi x/L)) (1 + v t/T),
  // B_y = By (1 + v (2x/L - 1)) (1 - 0.6 v t/T); 0: constant
  double fieldVariation = 0;
  // MITL first-substep mode: no manufactured source; the production inflow Maxwellian at the cathode (u_x > 0),
  // n / (2 pi sigma^2) exp(-((u_x - u0)^2 + u_z^2) / (2 sigma^2)); zero data at t = 0 and on the velocity boundary; the
  // final-time face is a free (outflow) trace.  Defaults: the production run's scaled n0, u0, sigma.
  bool mitlInflow = false;
  bool freeOutflowX = false;   // tensor backend: x-face outflow parts are trial unknowns (cathode u_x < 0, anode u_x > 0)
  double inflowDensity = 185.422, inflowU0 = 0.198801, inflowSigma = 0.0497003;
  bool separableTestNorm = false;   // separable graph norm (graph-oplus), both backends
  bool schurChebyshev = false, chebFactoredField = false;
  double chebLmin = 0, chebLmax = 0, chebTolerance = 1e-5, chebRtol = 0, chebBoundsTolerance = 1e-4, chebLminSafety = 0.8;
  int chebBoundsSteps = 30;
  double chebJacobiTolerance = 0;
  bool chebScaled = true, chebZipUp = false, chebStructuredField = true;
  bool explicitSchur = false; double explicitSchurTolerance = 1e-6;
  std::string explicitSchurLambdaMax = "fullpower"; int explicitSchurPowerIterations = 10, explicitSchurLambdaMinIterations = 0;
  double explicitSchurPowerTolerance = 1e-4, explicitSchurBoost = 1.2;
  std::string explicitSchurPowerStart = "random";
  std::string explicitSchurSolver = "chebyshev";
  int explicitSchurRefinements = 0; double explicitSchurRefineRhsTolerance = 3e-2, explicitSchurRefineRounding = 0;
  double explicitSchurTheta = 0, explicitSchurInnerReduction = 0.1; int explicitSchurEnergyLanczosSteps = 0;
  int explicitSchurEnergyCGSteps = 200; double explicitSchurEnergyCGTolerance = 1e-3, explicitSchurRefineGuessTolerance = 0.3;
  int chebIterations = 200, chebEvery = 10;
  std::string chebKronecker;
  std::string conventionalSolver = "KLU";
  bool conventionalSerendipity = false;   // conventional backend: Serendipity bases, as in production
  double gmgTolerance = 1e-9; int gmgMaxIterations = 5000;
  // production-like meshes: graded velocity (PlanarMITL::gradedVertices, production parameters by default) and uniform
  // refinement of the whole base mesh (as production's --numUniformRefinements; conventional GMG uses it as an h-level)
  bool gradedVelocity = false;
  double velocityCenter = 0.198801, velocitySigma = 0.0497003, velocityCoreWidthFactor = 2, velocityCoreHalfWidthFactor = 2,
         velocityGrowthRatio = 3;
  int uniformRefinements = 0;
  double tolerance = 1e-8, svdTolerance = 1e-12, coefficientTolerance = 1e-12;
  int maxSweeps = 100, kickrank = 4, maxRank = 64;
  double amenStagnation = -1;   // AMEn with a matrix-free residual: stop when it falls by less than this fraction over two sweeps (< 0: path default)
  int localSolver = 0, maxLocalIterations = 300;
  double localTolerance = 1e-6, rieszSvdTolerance = 0, denseFillThreshold = 1.1;
  bool verbose = false, comparison = true, splitModes = false, splitX = false, infSupEnrichment = false, gradedX = false;
  int xRefinements = 0;
  std::string splitAxes = "";
  std::string testBasis = "nodal", fieldBasis = "nodal", traceBasis = "nodal";
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
    cmd.setOption("alphaTransportScale", &alphaTransportScale,
                  "If positive, the production transport-scale weight alpha_K = alpha + C [ (v_x,max(K)/h_x)^2 "
                  "+ 1/h_t^2 + (a_x/h_ux)^2 + (a_z/h_uz)^2 ] (Camellia --L2WeightGraphTransportScale); exclusive with alphaMeshScale");
    cmd.setOption("accelerationBounds", &accelerationBounds,
                  "Bounds a_x,a_z on |acceleration| for alphaTransportScale, or \"auto\": |q/m| (|Ex| + v_max |By|), |q/m| v_max |By|");
    cmd.setOption("fieldVariation", &fieldVariation,
                  "Relative (x, t) variation v of the prescribed fields, E_x = Ex (1 + v cos(pi x/L))(1 + v t/T), "
                  "B_y = By (1 + v (2x/L - 1))(1 - 0.6 v t/T); 0: constant fields");
    cmd.setOption("separableTestNorm", "graphTestNorm", &separableTestNorm,
                  "Separable graph norm (graph-oplus) alpha_K |w|^2 + sum_d s_d^2 |d_d w|^2, s = (v_x,max, 1, a_x, a_z) "
                  "(the acceleration bounds; needs alphaTransportScale)");
    cmd.setOption("tolerance", &tolerance); cmd.setOption("svdTolerance", &svdTolerance);
    cmd.setOption("coefficientTolerance", &coefficientTolerance);
    cmd.setOption("maxSweeps", &maxSweeps); cmd.setOption("kickrank", &kickrank);
    cmd.setOption("amenStagnation", &amenStagnation, "AMEn with a matrix-free residual: stop when it falls by less than this fraction over two sweeps (0 off; < 0 path default, 0.1 for explicit S)");
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
    cmd.setOption("denseFillThreshold", &denseFillThreshold,
                  "Tensor backend, localSolver 5/6: site systems above this fill use a dense solve (> 1: never)");
    cmd.setOption("testBasis", &testBasis, "Tensor backend: 1D test basis family, nodal or hierarchical");
    cmd.setOption("fieldBasis", &fieldBasis, "Tensor backend: 1D field (f) basis family, nodal or hierarchical");
    cmd.setOption("traceBasis", &traceBasis, "Tensor backend: 1D trace basis family, nodal or hierarchical");
    cmd.setOption("schurChebyshev", "no-schurChebyshev", &schurChebyshev,
                  "Tensor backend: Chebyshev on the Schur complement with S applied only to iterates, cells-first TT layout and "
                  "structured G^-1 (uniform x and t; hierarchical trial bases recommended; use a small --svdTolerance)");
    cmd.setOption("chebLmin", &chebLmin, "Chebyshev: lower spectrum bound of M^-1 S");
    cmd.setOption("chebLmax", &chebLmax, "Chebyshev: upper spectrum bound of M^-1 S");
    cmd.setOption("chebTolerance", &chebTolerance, "Chebyshev: TT rounding tolerance for iterates and intermediates");
    cmd.setOption("chebRtol", &chebRtol, "Chebyshev: stop when the TT residual |r|/|c| falls below this");
    cmd.setOption("chebFactoredField", "no-chebFactoredField", &chebFactoredField, "Chebyshev: apply B_f as -T^T F");
    cmd.setOption("chebBoundsTolerance", &chebBoundsTolerance, "Chebyshev: TT rounding tolerance of the Lanczos bound estimate");
    cmd.setOption("chebBoundsSteps", &chebBoundsSteps, "Chebyshev: Lanczos steps for the bound estimate (without --chebLmin/--chebLmax)");
    cmd.setOption("chebJacobiTolerance", &chebJacobiTolerance, "Chebyshev: > 0: Jacobi diagonal as a TT at this tolerance (else rank-one Kronecker)");
    cmd.setOption("chebScaled", "no-chebScaled", &chebScaled, "Chebyshev: scaled rounding (D^{1/2} x, D^{-1/2} r, test vectors / |K|)");
    cmd.setOption("chebLminSafety", &chebLminSafety, "Chebyshev: factor on the smallest Ritz value");
    cmd.setOption("chebStructuredField", "no-chebStructuredField", &chebStructuredField,
                  "Chebyshev: apply B_f element-locally (varying fields: with (x, t) cell weights); else as a TT matrix");
    cmd.setOption("explicitSchur", "no-explicitSchur", &explicitSchur,
                  "Tensor backend (N5 only): form S = B^T G^-1 B once as a four-mode TT matrix and run Chebyshev on the trial "
                  "space (chebTolerance: iterate rounding; chebRtol, chebIterations, chebEvery, chebBounds*, chebLminSafety)");
    cmd.setOption("explicitSchurTolerance", &explicitSchurTolerance, "Explicit Schur: TT-SVD / rounding tolerance of the S blocks");
    cmd.setOption("explicitSchurLambdaMax", &explicitSchurLambdaMax, "Explicit Schur lambda_max: fullpower (matrix-free Ifpack2-style power iteration on full vectors; default), power (TT), bound, or lanczos");
    cmd.setOption("explicitSchurPowerIterations", &explicitSchurPowerIterations, "Explicit Schur: power iterations for lambda_max");
    cmd.setOption("explicitSchurPowerTolerance", &explicitSchurPowerTolerance, "Explicit Schur: TT rounding tolerance of the power iteration");
    cmd.setOption("explicitSchurBoost", &explicitSchurBoost, "Explicit Schur: lambda_max boost factor (Ifpack2 default 1.1; 1.2 for 10 TT iterations)");
    cmd.setOption("explicitSchurPowerStart", &explicitSchurPowerStart, "Explicit Schur: power start vector, random (rank one) or pattern (1 + index mod 97)");
    cmd.setOption("explicitSchurTheta", &explicitSchurTheta, "Explicit Schur amen, staged: stop refinement passes once e_s <= theta eta (error representation); also sets each pass's rounding (0: off)");
    cmd.setOption("explicitSchurInnerReduction", &explicitSchurInnerReduction, "Explicit Schur staged: residual reduction per correction solve");
    cmd.setOption("explicitSchurEnergyCGSteps", &explicitSchurEnergyCGSteps, "Explicit Schur: maximum CG steps for the e_s estimate r^T S^-1 r");
    cmd.setOption("explicitSchurEnergyCGTolerance", &explicitSchurEnergyCGTolerance, "Explicit Schur: CG e_s estimate stops when the last 4 terms add less than this fraction");
    cmd.setOption("explicitSchurRefineGuessTolerance", &explicitSchurRefineGuessTolerance, "Explicit Schur staged: relative truncation of the residual used as the correction's starting guess");
    cmd.setOption("explicitSchurEnergyLanczosSteps", &explicitSchurEnergyLanczosSteps, "Explicit Schur: separate Lanczos steps for lambda_min/max (0: from the CG of the e_s estimate)");
    cmd.setOption("explicitSchurRefinements", &explicitSchurRefinements, "Explicit Schur amen: iterative refinement passes (exact matrix-free residual, correction solved with the compressed S~)");
    cmd.setOption("explicitSchurRefineRhsTolerance", &explicitSchurRefineRhsTolerance, "Explicit Schur refinement: relative TT tolerance of the residual right-hand side");
    cmd.setOption("explicitSchurRefineRounding", &explicitSchurRefineRounding, "Explicit Schur refinement: relative rounding of y after each correction (<= 0: 1e-8)");
    cmd.setOption("explicitSchurSolver", &explicitSchurSolver,
                  "Explicit Schur solver: chebyshev, or amen (block AMEn on S~ with --tolerance, --chebTolerance as the local "
                  "SVD tolerance, --kickrank, --maxRank, --maxSweeps, --localSolver; --splitAxes for the (basis, cell) split)");
    cmd.setOption("explicitSchurLambdaMinIterations", &explicitSchurLambdaMinIterations,
                  "Explicit Schur: > 0: diagnostic shifted power iteration for lambda_min of the rounded S~");
    cmd.setOption("chebZipUp", "no-chebZipUp", &chebZipUp, "Chebyshev: apply the TT blocks B_j, B_j^T by zip-up (core-by-core truncation)");
    cmd.setOption("chebIterations", &chebIterations); cmd.setOption("chebEvery", &chebEvery);
    cmd.setOption("chebKronecker", &chebKronecker, "Chebyshev: rank-one Kronecker Jacobi factors (tools/schur_chebyshev.py --prec kron8 --write-kron)");
    cmd.setOption("conventionalSolver", &conventionalSolver,
                  "Conventional backend: KLU or SuperLUDist (direct), or GMG (production-style GMG-preconditioned CG on the condensed system)");
    cmd.setOption("mitlInflow", "manufacturedSolution", &mitlInflow,
                  "MITL first-substep problem instead of the manufactured bubble: no source, the production inflow Maxwellian at "
                  "x = 0 for u_x > 0, f = 0 at t = 0, zero on the velocity boundary, free final-time trace (tensor backend: the "
                  "other exterior faces are held at zero, which is exact while no electrons reach them)");
    cmd.setOption("freeOutflowX", "prescribedOutflowX", &freeOutflowX,
                  "tensor backend: the outflow parts of the x faces (cathode u_x < 0, anode u_x > 0) are free traces, as production; default: prescribed (zero)");
    cmd.setOption("inflowDensity", &inflowDensity, "mitlInflow: n (scaled; production n0 = 185.422)");
    cmd.setOption("inflowU0", &inflowU0, "mitlInflow: Maxwellian mean u_x (production u0 = 0.198801)");
    cmd.setOption("inflowSigma", &inflowSigma, "mitlInflow: Maxwellian width (production sigma = 0.0497003)");
    cmd.setOption("conventionalSerendipity", "conventionalFullTensor", &conventionalSerendipity,
                  "Conventional backend: Serendipity bases, as the production runs use (a different discrete space from the tensor backend's full tensor-product bases)");
    cmd.setOption("gmgTolerance", &gmgTolerance, "Conventional GMG: CG tolerance (production 1e-9)");
    cmd.setOption("gmgMaxIterations", &gmgMaxIterations);
    cmd.setOption("gradedVelocity", "uniformVelocity", &gradedVelocity,
                  "Production graded velocity mesh (PlanarMITL::gradedVertices; nux/nuz ignored): u_x centered at velocityCenter, "
                  "u_z at 0, core width/half-width in units of velocitySigma, geometric growth; vertex at 0 ensured");
    cmd.setOption("velocityCenter", &velocityCenter); cmd.setOption("velocitySigma", &velocitySigma);
    cmd.setOption("velocityCoreWidthFactor", &velocityCoreWidthFactor);
    cmd.setOption("velocityCoreHalfWidthFactor", &velocityCoreHalfWidthFactor);
    cmd.setOption("velocityGrowthRatio", &velocityGrowthRatio);
    cmd.setOption("uniformRefinements", &uniformRefinements,
                  "Uniform bisections of every cell in all four directions after building the base mesh (tensor backend: the "
                  "refined vertices; conventional: the formulation's numUniformRefinements, so GMG has an h-level)");
    cmd.setOption("outputPrefix", &outputPrefix);
    auto result = cmd.parse(argc, argv);
    if (result == Teuchos::CommandLineProcessor::PARSE_HELP_PRINTED) return 2;
    if (result != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) return 1;
    if (nx < 1 || nt < 1 || nux < 1 || nuz < 1 || p < 1 || dp < 0 ||
        !(c > 0) || !(betaMax > 0 && betaMax < 1) || !(beta0 > -1 && beta0 < betaMax) ||
        !(xWidth > 0) || !(tWidth > 0) || !(alpha > 0) || (alphaMeshScale > 0 && alphaTransportScale > 0) ||
        quadratureDegree < 2*(p+dp) || sampleDegree < 2*std::max(2,p-1) ||
        !(tolerance > 0) || !(svdTolerance > 0) || !(coefficientTolerance > 0) ||
        fieldVariation < 0 || fieldVariation >= 1 || (separableTestNorm && !(alphaTransportScale > 0)))
      throw std::invalid_argument("Invalid comparison parameters or underintegrated mass/error quadrature");
    return 0;
  }

  //! Acceleration bounds (u_x, u_z) for the transport-scale weight: parsed, or bounds of |q/m| (E + v x B) over the
  //! velocity box (|v| <= betaMax c).
  std::array<double,2> accelerationBoundValues() const
  {
    if (accelerationBounds != "auto")
    {
      std::array<double,2> a {0,0};
      std::stringstream in(accelerationBounds); std::string item;
      for (int i = 0; i < 2 && std::getline(in, item, ','); ++i) a[i] = std::stod(item);
      return a;
    }
    const double vMax = betaMax * c, eMax = std::pow(1 + fieldVariation, 2), bMax = 1 + fieldVariation;
    return {std::abs(qOverM) * (std::abs(Ex) * eMax + vMax * std::abs(By) * bMax), std::abs(qOverM) * vMax * std::abs(By) * bMax};
  }
  std::array<int,4> counts() const
  {
    const auto v = vertices();
    return {int(v[0].size()) - 1, int(v[1].size()) - 1, int(v[2].size()) - 1, int(v[3].size()) - 1};
  }
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
  //! Base mesh vertices (before --uniformRefinements).
  std::vector<std::vector<double>> baseVertices() const
  {
    double uMax = properVelocityFromBeta(betaMax,c), uMin = properVelocityFromBeta(beta0,c);
    if (!gradedVelocity)
      return {xVertices(), uniformVertices(0,tWidth,nt),
              uniformVertices(uMin,uMax-uMin,nux), uniformVertices(-uMax,2*uMax,nuz)};
    auto ux = gradedVertices(uMin, uMax, velocityCenter, velocityCoreWidthFactor * velocitySigma,
                             velocityCoreHalfWidthFactor * velocitySigma, velocityGrowthRatio);
    auto uz = gradedVertices(-uMax, uMax, 0.0, velocityCoreWidthFactor * velocitySigma,
                             velocityCoreHalfWidthFactor * velocitySigma, velocityGrowthRatio);
    ensureVertexAt(ux, 0.0); ensureVertexAt(uz, 0.0);
    return {xVertices(), uniformVertices(0,tWidth,nt), ux, uz};
  }
  //! Mesh vertices: the base mesh with every cell bisected uniformRefinements times in every direction.
  std::vector<std::vector<double>> vertices() const
  {
    auto v = baseVertices();
    for (int r = 0; r < uniformRefinements; ++r)
      for (auto &axis : v)
      {
        std::vector<double> refined {axis.front()};
        for (size_t i = 1; i < axis.size(); ++i) { refined.push_back(0.5*(axis[i-1]+axis[i])); refined.push_back(axis[i]); }
        axis = refined;
      }
    return v;
  }
  //! the inflow Maxwellian (mitlInflow), restricted to u_x > 0
  double inflowMaxwellian(double ux, double uz) const
  {
    if (!(ux > 0)) return 0.0;
    const double s2 = inflowSigma * inflowSigma;
    return inflowDensity / (2 * M_PI * s2) * std::exp(-0.5 * ((ux - inflowU0) * (ux - inflowU0) + uz * uz) / s2);
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
    const auto n = counts();
    return {{"nx",n[0]},{"graded_x",double(gradedX)},{"x_refinements",double(xRefinements)},{"nt",n[1]},{"nux",n[2]},{"nuz",n[3]},
            {"graded_velocity",double(gradedVelocity)},{"uniform_refinements",double(uniformRefinements)},{"p",p},{"dp",dp},
            {"quadrature_degree",quadratureDegree},{"sample_degree",sampleDegree},
            {"beta0",beta0},{"betaMax",betaMax},{"c",c},{"q_over_m",qOverM},
            {"Ex",Ex},{"By",By},{"x_width",xWidth},{"t_width",tWidth},{"alpha",alpha},{"alpha_mesh_scale",alphaMeshScale},{"alpha_transport_scale",alphaTransportScale},{"field_variation",fieldVariation},{"conventional_serendipity",double(conventionalSerendipity)},{"mitl_inflow",double(mitlInflow)},{"inflow_density",inflowDensity},{"inflow_u0",inflowU0},{"inflow_sigma",inflowSigma},{"separable_test_norm",double(separableTestNorm)},
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
