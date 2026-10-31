#pragma once

#include "ComparisonProblem.hpp"
#include "Camellia.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "RelativisticVlasovSpaceTimeFormulation.h"
#include "MonomialFunctions.h"
#include "CubatureFactory.h"

namespace PlanarMITL
{
/// Keep the reference formulation and freeze only the electromagnetic fields.
class PrescribedFields : public Camellia::ConfigurationSpaceFormulationMaxwell1D2V
{
  double _Ex, _By;
public:
  explicit PrescribedFields(const ComparisonProblem &problem)
    : ConfigurationSpaceFormulationMaxwell1D2V(problem.qOverM,1), _Ex(problem.Ex), _By(problem.By) {}
  Camellia::FunctionPtr E_i(int i) override { return Camellia::Function::constant(i == 0 ? _Ex : 0); }
  Camellia::FunctionPtr B_i(int i) override { return Camellia::Function::constant(i == 1 ? _By : 0); }
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
  const auto vertices = problem.vertices();
  auto spatial = MeshFactory::intervalMeshTopology(0,problem.xWidth,problem.nx);
  auto fields = Teuchos::rcp(new PrescribedFields(problem));
  RelativisticVlasovSpaceTimeFormulation form(problem.c,problem.qOverM,1,fields,spatial,
      vertices[1],vertices[2],vertices[3],{},problem.p,problem.dp,0,
      RelativisticVlasovSpaceTimeFormulation::GraphTestSpaceInnerProduct,problem.alpha,
      BASIS_FAMILY_HIERARCHICAL,0,false,false);
  auto solution = form.vlasovSolution();
  auto mesh = solution->mesh();
  auto bc = BC::bc();
  bc->addDirichlet(form.t_n_hat(), SpatialFilter::allSpace(), Function::zero());
  solution->setBC(bc);
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
  rhs->addTerm(source*form.w());
  solution->setRHS(rhs);
  solution->setUseCondensedSolve(false);
  const auto cell = *mesh->getActiveCellIDsGlobal().begin();
  const int baseDegree = BasisCache::basisCacheForCell(mesh,cell,false)->cubatureDegree();
  const int enrichment = std::max(0,problem.quadratureDegree-baseDegree);
  solution->setCubatureEnrichmentDegree(enrichment);
  const double setupSeconds = elapsed(start);
  const auto solveStart = ComparisonClock::now();
  const int status = solution->solve(); // conventional local Riesz elimination + sparse direct solve
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
  double jacobian = 1;
  for (int d = 0; d < 4; ++d) jacobian *= (vertices[d][1]-vertices[d][0])/2;
  for (int k = 0; k < nq; ++k)
  {
    int ordinal = k;
    weights[k] = jacobian;
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
  double norm2 = 0, error2 = 0;
  for (const auto cellID : mesh->getActiveCellIDsGlobal())
  {
    auto cache = BasisCache::basisCacheForCell(mesh,cellID);
    cache->setRefCellPointsFC(refPoints);
    const auto &points = cache->getPhysicalCubaturePointsFC();
    FieldContainer<double> values(1,nq);
    f->values(values,cache);
    for (int k = 0; k < nq; ++k)
    {
      double exact = 1;
      for (int d = 0; d < 4; ++d)
      {
        out << points(0,k,d) << ',';
        exact *= ComparisonProblem::bubble(points(0,k,d),vertices[d].front(),vertices[d].back());
      }
      const double value = values(0,k);
      out << weights[k] << ',' << value << ',' << exact << '\n';
      norm2 += weights[k]*exact*exact;
      error2 += weights[k]*(value-exact)*(value-exact);
    }
  }
  metrics["relative_exact_l2_error"] = std::sqrt(error2/norm2);
  metrics["solve_passed"] = 1;
  writeMetrics(problem.outputPrefix+"_metrics.json",metrics);
  std::cout << std::setprecision(12) << "vanilla total_seconds=" << totalSeconds
            << " relative_exact_l2_error=" << metrics["relative_exact_l2_error"] << '\n';
  return 0;
}
} // namespace PlanarMITL
