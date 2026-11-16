#pragma once

#include "GMGSolver.h"
#include "ComparisonProblem.hpp"
#include "Camellia.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "RelativisticVlasovSpaceTimeFormulation.h"
#include "MonomialFunctions.h"
#include "CubatureFactory.h"

namespace PlanarMITL
{
/// (x, t) profile of a prescribed field on phase space (x, t, u_x, u_z), the tensor backend's FieldProfile
/// (PlanarMITLBoba/TensorProductAssembly.hpp): with s, tau the positions scaled to [0, 1] on the x and t axes and
/// v = variation,
///   E_x = Ex (1 + v cos(pi s)) (1 + v tau),   B_y = By (1 + v (2 s - 1)) (1 - 0.6 v tau).
class PrescribedFieldProfile : public Camellia::TFunction<double>
{
  bool _electric;
  double _amplitude, _variation, _x0, _x1, _t0, _t1;
public:
  PrescribedFieldProfile(bool electric, double amplitude, double variation, double x0, double x1, double t0, double t1)
    : _electric(electric), _amplitude(amplitude), _variation(variation), _x0(x0), _x1(x1), _t0(t0), _t1(t1) {}
  double value(double x, double t) const
  {
    const double s = (x - _x0) / (_x1 - _x0), tau = (t - _t0) / (_t1 - _t0), v = _variation;
    return _electric ? _amplitude * (1 + v * std::cos(M_PI * s)) * (1 + v * tau)
                     : _amplitude * (1 + v * (2 * s - 1)) * (1 - 0.6 * v * tau);
  }
  void values(Camellia::FieldContainer<double> &values, Camellia::BasisCachePtr basisCache) override
  {
    const auto &points = basisCache->getPhysicalCubaturePointsFC();
    for (int cell = 0; cell < points.dimension(0); ++cell)
      for (int pt = 0; pt < points.dimension(1); ++pt)
        values(cell, pt) = value(points(cell, pt, 0), points(cell, pt, 1));
  }
};

/// The inflow Maxwellian of ComparisonProblem (mitlInflow) on phase space (x, t, u_x, u_z)
class InflowMaxwellianFunction : public Camellia::TFunction<double>
{
  const ComparisonProblem &_problem;
public:
  explicit InflowMaxwellianFunction(const ComparisonProblem &problem) : _problem(problem) {}
  void values(Camellia::FieldContainer<double> &values, Camellia::BasisCachePtr basisCache) override
  {
    const auto &points = basisCache->getPhysicalCubaturePointsFC();
    for (int cell = 0; cell < points.dimension(0); ++cell)
      for (int pt = 0; pt < points.dimension(1); ++pt)
        values(cell, pt) = _problem.inflowMaxwellian(points(cell, pt, 2), points(cell, pt, 3));
  }
};

/// Diagnostic (PLANAR_MITL_CONV_TT_BOUNDARY): the tensor backend's exterior-trace data on phase space -- t_n = v.n f_in
/// on the cathode face x = x0 (outward n = -e_x, so -v_x f_in), zero on every other exterior face.
class TensorBoundaryTraceData : public Camellia::TFunction<double>
{
  const ComparisonProblem &_problem; double _x0;
public:
  TensorBoundaryTraceData(const ComparisonProblem &problem, double x0) : _problem(problem), _x0(x0) {}
  void values(Camellia::FieldContainer<double> &values, Camellia::BasisCachePtr basisCache) override
  {
    const auto &points = basisCache->getPhysicalCubaturePointsFC();
    for (int cell = 0; cell < points.dimension(0); ++cell)
      for (int pt = 0; pt < points.dimension(1); ++pt)
      {
        const double x = points(cell, pt, 0), ux = points(cell, pt, 2), uz = points(cell, pt, 3);
        if (std::abs(x - _x0) > 1e-10) { values(cell, pt) = 0.0; continue; }
        const double c = _problem.c, vx = ux / std::sqrt(1 + (ux * ux + uz * uz) / (c * c));
        values(cell, pt) = -vx * _problem.inflowMaxwellian(ux, uz);
      }
  }
};

/// Keep the reference formulation and freeze only the electromagnetic fields: constant, or the tensor backend's
/// (x, t)-varying profile (--fieldVariation).
class PrescribedFields : public Camellia::ConfigurationSpaceFormulationMaxwell1D2V
{
  Camellia::FunctionPtr _E, _B;
public:
  PrescribedFields(const ComparisonProblem &problem, const std::vector<std::vector<double>> &vertices)
    : ConfigurationSpaceFormulationMaxwell1D2V(problem.qOverM,1)
  {
    if (problem.fieldVariation == 0)
    {
      _E = Camellia::Function::constant(problem.Ex);
      _B = Camellia::Function::constant(problem.By);
    }
    else
    {
      const double x0 = vertices[0].front(), x1 = vertices[0].back(), t0 = vertices[1].front(), t1 = vertices[1].back();
      _E = Teuchos::rcp(new PrescribedFieldProfile(true, problem.Ex, problem.fieldVariation, x0, x1, t0, t1));
      _B = Teuchos::rcp(new PrescribedFieldProfile(false, problem.By, problem.fieldVariation, x0, x1, t0, t1));
    }
  }
  Camellia::FunctionPtr E_i(int i) override { return i == 0 ? _E : Camellia::Function::zero(); }
  Camellia::FunctionPtr B_i(int i) override { return i == 1 ? _B : Camellia::Function::zero(); }
};

inline int runConventionalComparison(int argc, char **argv)
{
  using namespace Camellia;
  ComparisonProblem problem;
  int parsed = problem.parse(argc, argv, "vanilla");
  if (parsed) return parsed == 2 ? 0 : parsed;
  if (Teuchos::GlobalMPISession::getNProc() != 1)
    throw std::invalid_argument("Matched comparison currently requires one MPI rank");
  const auto start = ComparisonClock::now();
  const auto vertices = problem.vertices();            // final mesh (for the exact solution's domain)
  const auto base = problem.baseVertices();            // refined uniformRefinements times by the formulation
  auto spatial = MeshFactory::intervalMeshTopology(base[0]);
  auto fields = Teuchos::rcp(new PrescribedFields(problem, vertices));
  const auto bounds = problem.accelerationBoundValues();
  RelativisticVlasovSpaceTimeFormulation form(problem.c,problem.qOverM,1,fields,spatial,
      base[1],base[2],base[3],{},problem.p,problem.dp,problem.uniformRefinements,
      RelativisticVlasovSpaceTimeFormulation::GraphTestSpaceInnerProduct,problem.alpha,
      problem.conventionalSerendipity ? BASIS_FAMILY_SERENDIPITY : BASIS_FAMILY_HIERARCHICAL,0,false,
      problem.alphaMeshScale,{},problem.alphaTransportScale,
      std::vector<double>(bounds.begin(), bounds.end()),problem.separableTestNorm);
  std::cout << "conventional fields: Ex=" << problem.Ex << " By=" << problem.By << " variation=" << problem.fieldVariation
            << " test_norm=" << (problem.separableTestNorm ? "separable (graph-oplus)" : "graph")
            << " acceleration_bounds=" << bounds[0] << "," << bounds[1]
            << " basis=" << (problem.conventionalSerendipity ? "serendipity" : "full tensor") << '\n';
  auto solution = form.vlasovSolution();
  auto mesh = solution->mesh();
  if (std::getenv("PLANAR_MITL_TRACE_BASIS_INFO"))
  {
    // which branch of Projector::projectFunctionOntoBasisInterpolating a Dirichlet trace condition takes
    const auto cellID = *mesh->getActiveCellIDsGlobal().begin();
    auto trialOrder = mesh->getElementType(cellID)->trialOrderPtr;
    const int numSides = mesh->getElementType(cellID)->cellTopoPtr->getSideCount();
    for (int side = 0; side < numSides; ++side)
    {
      auto basis = trialOrder->getBasis(form.t_n_hat()->ID(), side);
      const bool wrapsHGRAD = (basis->intrepid2Basis() != Teuchos::null) &&
                              (basis->intrepid2Basis()->getFunctionSpace() == Intrepid2::FUNCTION_SPACE_HGRAD);
      std::cout << "t_n_hat side " << side << ": cardinality " << basis->getCardinality() << " degree " << basis->getDegree()
                << " camellia fs " << int(basis->functionSpace()) << " (HVOL=" << int(FUNCTION_SPACE_HVOL) << ")"
                << " broken " << Camellia::functionSpaceIsBroken(basis->functionSpace())
                << " discontinuous " << Camellia::functionSpaceIsDiscontinuous(basis->functionSpace())
                << " intrepid2 " << (basis->intrepid2Basis() != Teuchos::null)
                << " wrapsHGRAD " << wrapsHGRAD << " domain dim " << basis->domainTopology()->getDimension() << '\n';
    }
    auto imposition = BasisCache::basisCacheForCell(mesh, cellID); // as Boundary::bcsToImpose builds it
    std::cout << "BC imposition cache: volume cubature degree " << imposition->cubatureDegree()
              << ", side 0 cubature degree " << imposition->getSideBasisCache(0)->cubatureDegree() << '\n';
  }
  if (problem.mitlInflow && std::getenv("PLANAR_MITL_CONV_TT_BOUNDARY"))
  {
    // the tensor backend's boundary treatment: every exterior face prescribed (cathode inflow data, zero elsewhere,
    // outflow parts included) except the final-time face
    std::cout << "conventional: tensor-backend boundary treatment (all exterior faces but t = T prescribed)\n";
    auto bc = BC::bc();
    auto notFinalTime = SpatialFilter::negatedFilter(SpatialFilter::matchingComponent(1, vertices[1].back()));
    bc->addDirichlet(form.t_n_hat(), notFinalTime, Teuchos::rcp(new TensorBoundaryTraceData(problem, vertices[0].front())));
    solution->setBC(bc);
  }
  else if (problem.mitlInflow)
    form.setInitialAndInflowConditions(Function::zero(), Teuchos::rcp(new InflowMaxwellianFunction(problem)));
  else
  {
    // PLANAR_MITL_FREE_FINAL_TIME (a test): leave the final-time face free, as the tensor backend can
    auto bc = BC::bc();
    auto filter = std::getenv("PLANAR_MITL_FREE_FINAL_TIME") ?
        SpatialFilter::negatedFilter(SpatialFilter::matchingComponent(1, vertices[1].back())) : SpatialFilter::allSpace();
    bc->addDirichlet(form.t_n_hat(), filter, Function::zero());
    solution->setBC(bc);
  }
  std::array<FunctionPtr,4> bubble, derivative;
  for (int d = 0; d < 4; ++d)
  {
    const double a = vertices[d].front(), width = vertices[d].back()-a;
    auto z = (MonomialFunction::xi_n(d,1)-a)/width;
    bubble[d] = 4*z*(1-z);
    derivative[d] = 4*(1-2*z)/width;
  }
  std::array<FunctionPtr,4> beta = {form.v_i(0),Function::constant(1),form.a()[0],form.a()[1]};
  FunctionPtr source = Function::zero();
  for (int d = 0; d < 4; ++d)
  {
    FunctionPtr term = beta[d];
    for (int e = 0; e < 4; ++e) term = term*(e == d ? derivative[e] : bubble[e]);
    source = source+term;
  }
  auto rhs = RHS::rhs();
  if (!problem.mitlInflow) rhs->addTerm(source*form.w());
  solution->setRHS(rhs);
  solution->setUseCondensedSolve(false);
  const auto cell = *mesh->getActiveCellIDsGlobal().begin();
  const int baseDegree = BasisCache::basisCacheForCell(mesh,cell,false)->cubatureDegree();
  const int enrichment = std::max(0,problem.quadratureDegree-baseDegree);
  solution->setCubatureEnrichmentDegree(enrichment);
  const double setupSeconds = elapsed(start);
  const auto solveStart = ComparisonClock::now();
  // conventional local Riesz elimination + sparse direct solve, or production-style GMG-preconditioned CG
  int status = 0, gmgIterations = -1;
  if (problem.conventionalSolver == "GMG")
  {
    // as VlasovSpaceTimeFormulation::constructGMGSolver: h-levels from the refinement history, p-coarsening to order 0,
    // V-cycle, additive Schwarz smoothing, direct coarse solve, CG on the statically condensed system
    solution->setUseCondensedSolve(true);
    auto meshes = GMGSolver::meshesForMultigrid(mesh, 0, 1);
    auto gmg = Teuchos::rcp(new GMGSolver(solution, meshes, problem.gmgMaxIterations, problem.gmgTolerance,
                                          GMGOperator::V_CYCLE, GMGOperator::DEFAULT_ADDITIVE_SCHWARZ,
                                          Solver::getDirectSolver(true), true));
    gmg->setSmootherType(GMGOperator::DEFAULT_ADDITIVE_SCHWARZ);
    std::cout << "conventional GMG levels=" << meshes.size() << " tolerance=" << problem.gmgTolerance << '\n';
    status = solution->solve(gmg);
    gmgIterations = gmg->iterationCount();
    std::cout << "conventional GMG iterations=" << gmgIterations << '\n';
  }
  else if (problem.conventionalSolver == "KLU" || problem.conventionalSolver == "SuperLUDist")
    status = solution->solve(Solver::getSolver(problem.conventionalSolver == "KLU" ? Solver::KLU : Solver::SuperLUDist, false));
  else throw std::invalid_argument("--conventionalSolver must be KLU, SuperLUDist, or GMG");
  const double assemblySolveSeconds = elapsed(solveStart);
  const double totalSeconds = elapsed(start);
  if (status) throw std::runtime_error("Conventional MITL comparison solve failed");

  // Everything below is validation/output and is excluded from the numerical timings.
  auto metrics = problem.metadata();
  auto matrix = solution->getStiffnessMatrix();
  metrics["setup_seconds"] = setupSeconds;
  metrics["assembly_solve_seconds"] = assemblySolveSeconds;
  metrics["total_seconds"] = totalSeconds;
  metrics["local_stiffness_seconds"] = solution->totalTimeLocalStiffness();
  metrics["global_assembly_seconds"] = solution->totalTimeGlobalAssembly();
  metrics["bc_seconds"] = solution->totalTimeBCImposition();
  metrics["linear_solve_seconds"] = solution->totalTimeSolve();
  metrics["gmg_iterations"] = gmgIterations;
  metrics["trial_dofs_with_boundary"] = mesh->numGlobalDofs();
  metrics["boundary_dofs"] = solution->getBCDofCount();
  metrics["sparse_rows"] = matrix->getGlobalNumRows();
  metrics["sparse_nnz"] = matrix->getGlobalNumEntries();
  metrics["bilinear_quadrature_degree"] = baseDegree+enrichment;
  metrics["gram_quadrature_degree"] = BasisCache::basisCacheForCell(mesh,cell,true,enrichment)->cubatureDegree();

  auto cubature = CubatureFactory().create(CellTopology::line(),problem.sampleDegree);
  auto q = cubature->allocateCubaturePoints();
  auto w = cubature->allocateCubatureWeights();
  cubature->getCubature(q,w);
  int n = q.extent_int(0), nq = n*n*n*n;
  FieldContainer<double> refPoints(nq,4);
  std::vector<double> weights(nq);
  // Reference weights; each cell's Jacobian (product of half-widths) is applied below, so graded meshes are handled.
  for (int k = 0; k < nq; ++k)
  {
    int ordinal = k;
    weights[k] = 1;
    for (int d = 0; d < 4; ++d)
    {
      int i = ordinal%n; ordinal /= n;
      refPoints(k,d) = q(i,0); weights[k] *= w(i);
    }
  }
  std::ofstream out(problem.outputPrefix+"_samples.csv");
  if (!out) throw std::runtime_error("Cannot write comparison samples");
  out << std::setprecision(17) << "x,t,ux,uz,weight,f,exact\n";
  auto f = Function::solution(form.f(),solution);
  double norm2 = 0, error2 = 0, fNorm2 = 0, fIntegral = 0;
  for (const auto cellID : mesh->getActiveCellIDsGlobal())
  {
    auto cache = BasisCache::basisCacheForCell(mesh,cellID);
    cache->setRefCellPointsFC(refPoints);
    const auto &nodes = cache->getPhysicalCellNodes();
    double jacobian = 1;
    for (int d = 0; d < 4; ++d)
    {
      double lo = nodes(0,0,d), hi = nodes(0,0,d);
      for (int node = 1; node < nodes.dimension(1); ++node) { lo = std::min(lo,nodes(0,node,d)); hi = std::max(hi,nodes(0,node,d)); }
      jacobian *= (hi-lo)/2;
    }
    const auto &points = cache->getPhysicalCubaturePointsFC();
    FieldContainer<double> values(1,nq);
    f->values(values,cache);
    for (int k = 0; k < nq; ++k)
    {
      double exact = 1;
      for (int d = 0; d < 4; ++d)
      {
        out << points(0,k,d) << ',';
        exact *= problem.mitlInflow ? 0.0 : ComparisonProblem::bubble(points(0,k,d),vertices[d].front(),vertices[d].back());
      }
      const double value = values(0,k);
      const double weight = jacobian*weights[k];
      out << weight << ',' << value << ',' << exact << '\n';
      norm2 += weight*exact*exact;
      error2 += weight*(value-exact)*(value-exact);
      fNorm2 += weight*value*value; fIntegral += weight*value;
    }
  }
  metrics["relative_exact_l2_error"] = problem.mitlInflow ? std::nan("") : std::sqrt(error2/norm2);
  metrics["f_l2_norm"] = std::sqrt(fNorm2); metrics["f_integral"] = fIntegral;
  metrics["solve_passed"] = 1;
  writeMetrics(problem.outputPrefix+"_metrics.json",metrics);
  std::cout << std::setprecision(12) << "vanilla total_seconds=" << totalSeconds
            << " relative_exact_l2_error=" << metrics["relative_exact_l2_error"]
            << " f_l2_norm=" << metrics["f_l2_norm"] << " f_integral=" << metrics["f_integral"] << '\n';
  return 0;
}
} // namespace PlanarMITL
