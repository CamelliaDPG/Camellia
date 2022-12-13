//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  VlasovSpaceTimeFormulationTests.cpp
//
//  Created by Nate Roberts on 11/15/2022.
//
//

#include "VlasovSpaceTimeFormulation.h"
#include "ConfigurationSpaceFormulationPoisson.h"

#include "BF.h"
#include "MonomialFunctions.h"
#include "PhaseSpaceVisualization.h"
#include "RHS.h"
#include "Solution.h"

#include "Teuchos_UnitTestHarness.hpp"

#include <Intrepid2_SerendipityBasis.hpp>

using namespace Camellia;

namespace {
  const double TEST_X0     = 0.0;
  const double TEST_KX     = 1.0; // wavelength -- DEBUGGING: set to to 1.0 to reduce roundoff/scaling errors (was 0.2)
  const double TEST_XWIDTH = 2 * M_PI / TEST_KX;
  const double TEST_T0     = 0.0;
  const double TEST_TWIDTH = 1.0;
  const double TEST_V0     = -10.0;
  const double TEST_VWIDTH = 20.0;

  const double TEST_Q      = 1.0;
  const double TEST_M      = 1.0;
  const double TEST_EPS0   = 0.5; // non-zero, but let's keep it close to 1 and assume that client codes also nondimensionalize

  LinearTermPtr vlasovRHSForExactSoln(VlasovSpaceTimeFormulation form, FunctionPtr f_exact, FunctionPtr E_exact, const double q)
  {
    // want an RHS so that f_exact is the exact solution
    auto w = form.w(); // test function for Vlasov RHS
    const int spaceDim              = form.configurationSpaceMesh()->getDimension() - 1; // space, not space-time, dim
    const int velocityDim           = form.vlasovMesh()->getDimension() - spaceDim - 1;
    const int timeCompIndexOneBased = spaceDim + 1;
    const int vxCompIndexOneBased   = spaceDim + 2;
    FunctionPtr lhsAtExactSoln      = Function::zero();
    // spatial derivatives:
    for (int i=0; i<spaceDim; i++)
    {
      auto vComp     = vxCompIndexOneBased + i;
      auto vi        = MonomialFunction::xi_n(vComp-1,1); // Annoyingly, confusingly, MonomialFunction's xi is zero-based.  TODO: make di and xi_n match in terms of one- or zero-basedness.
      lhsAtExactSoln = lhsAtExactSoln + (f_exact * vi)->di(i+1);
    }
    // temporal derivative:
    lhsAtExactSoln   = lhsAtExactSoln + f_exact->di(timeCompIndexOneBased);
    // velocity derivatives:
    vector<FunctionPtr> E_soln_vector {E_exact};
    for (int i=0; i<velocityDim; i++)
    {
      auto vComp     = vxCompIndexOneBased + i;
      lhsAtExactSoln = lhsAtExactSoln + ((q / TEST_M) * E_soln_vector[i] * f_exact)->di(vComp);
    }
    return lhsAtExactSoln * w;
  }

  VlasovSpaceTimeFormulation getFormulation1D1T1V(const int xElements, const int tElements, const int vElements, const int H1Order, const bool usePeriodicBCs, double q, const bool useSerendipity)
  {
    // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
    double eps0   = TEST_EPS0;
    double m      = TEST_M;
    double x0     = TEST_X0;
    double xWidth = TEST_XWIDTH;
    double t0     = TEST_T0;
    double v0     = TEST_V0;
    double vWidth = TEST_VWIDTH;
    
    const int delta_k = 2;
    const int numUniformRefinements = 0;
    
    Teuchos::RCP<ConfigurationSpaceFormulationPoisson> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q, eps0, SPACE_TIME) );
    
    MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
    
    std::vector<double> temporalVertices(tElements + 1);
    for (int i=0; i<=tElements; i++)
    {
      const double t = double(i) / double(tElements) * TEST_TWIDTH + t0;
      temporalVertices[i] = t;
    }
    
    std::vector<double> velocitySpaceVertices_x(vElements+1);
    for (int i=0; i<=vElements; i++)
    {
      const double v = double(i) / double(vElements) * vWidth + v0;
      velocitySpaceVertices_x[i] = v;
    }
    
    std::vector<double> velocitySpaceVertices_y; // empty
    std::vector<double> velocitySpaceVertices_z; // empty
    
    VlasovSpaceTimeFormulation form(q, m, useSerendipity, configForm, spatialMeshTopo,
                                    temporalVertices,
                                    velocitySpaceVertices_x,
                                    velocitySpaceVertices_y,
                                    velocitySpaceVertices_z,
                                    H1Order, delta_k,
                                    numUniformRefinements);
    
    
    return form;
  }

  class Maxwellian1T1V : public TFunction<double>
  {
    double _u;
    double _sigma;
    double _n;
  public:
    Maxwellian1T1V(double u, double sigma, double n)
    :
    _u(u),
    _sigma(sigma),
    _n(n)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(sigma <= 0.0, std::invalid_argument, "sigma must be positive");
    }
    
    using TFunction<double>::values;
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
      const int numCells  = phaseSpacePoints.dimension(0);
      const int numPoints = phaseSpacePoints.dimension(1);
      
      const double weight = _n / (sqrt(2. * M_PI) * _sigma);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,2); // vx is the third dimension for 1D1T1V
          const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
          values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
        }
      }
    }
    
    TFunctionPtr<double> di(int i) override // 1 for dx(), 2 for dt(), 3 for dv()
    {
      if (i==1) // dx
      {
        return Function::zero();
      }
      else if (i==2) // dt
      {
        return Function::zero();
      }
      else // dv
      {
        const int vx_component_zero_based = 2;
        FunctionPtr v = MonomialFunction::xi_n(vx_component_zero_based, 1);
        FunctionPtr weight = - 1. / ( _sigma * _sigma) * (_u - v);
        FunctionPtr maxwellian = Teuchos::rcp( new Maxwellian1T1V(_u,_sigma,_n) );
        return weight * maxwellian;
      }
    }
  };

  FunctionPtr getVx(const int spaceDim, const bool spaceTimeMesh)
  {
    // the vx component is the first one after the dimension of the physical mesh
    const int temporalDimensions = spaceTimeMesh ? 1 : 0;
    const int vx_component_zero_based = spaceDim + temporalDimensions;
    return MonomialFunction::xi_n(vx_component_zero_based, 1);
  }

void patchTestVlasovPoisson(FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                            const double q,
                            const int H1Order, const int xElements, const int tElements, const int vElements,
                            std::vector<std::vector<GlobalIndexType> > refinementSequence,
                            const bool overrideVlasovBCs,
                            const bool useSerendipity,
                            const double tol,
                            Teuchos::FancyOStream &out, bool &success,
                            const bool outputViz, const string vizPrefix)
{
  const bool usePeriodicBCs = false;
  
  auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, useSerendipity);
  
  form.setUseGeometricMultigrid(true);
  form.setIterativeSolveTolerance(1e-15);
//    form.setIterativeSolveOutputLevel(10);
  form.setFixedPointMaxSubsteps(5);
  form.setFixedPointMinSubsteps(5);
  
  FunctionPtr vx = form.v_i(0);

  auto vlasovSoln = form.vlasovSolution();
  
  auto poissonForm = form.configurationSpaceFormulation();
  auto poissonSoln = form.configurationSpaceSolution();
  FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
  FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
  
  vlasovSoln->rhs()->addTerm(vlasovRHSForExactSoln(form, f_exact, E_exact, q));
  
  form.setInitialAndInflowConditions(f_exact, f_exact);
  
  SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
  SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
  
  const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
  auto xLeftRightTag = form.createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
  auto VE_traceVar = form.VE_configurationSpaceTrace();
  form.addConfigurationSpaceBC(xLeftRightTag, VE_traceVar, VE_exact);
  
  if (overrideVlasovBCs)
  {
    auto vlasovBC = BC::bc();
    VarPtr traceVar = form.t_n_hat();
    
    const int initialTimeTag = form.tagVlasovInitialTime();
    
    vlasovBC->addDirichlet(traceVar, initialTimeTag, -f_exact); // minus sign thanks to outward normal on temporal interface
    
    const int VX_COMPONENT = 0;
    
    const int minVelocityTag = form.tagVlasovMinVelocity(VX_COMPONENT);
    const int maxVelocityTag = form.tagVlasovMaxVelocity(VX_COMPONENT);
    
    vlasovBC->addDirichlet(traceVar, minVelocityTag, -f_exact * (q/TEST_M) * E_exact); // outward normal is negative
    vlasovBC->addDirichlet(traceVar, maxVelocityTag,  f_exact * (q/TEST_M) * E_exact); // outward normal is positive
    
    const int inflowTag = form.tagVlasovSpatialInflow();
    
    FunctionPtr n = Function::normal();
    
    FunctionPtr traceValue = f_exact * vx * n->x();
    vlasovBC->addDirichlet(traceVar, inflowTag,  traceValue);
    vlasovSoln->setBC(vlasovBC);
  }
  
  auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
  auto vlasovMesh  = vlasovSoln->mesh();
  auto poissonMesh = poissonSoln->mesh();
  for (const auto &cellsToRefine : refinementSequence)
  {
    auto refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
    vlasovMesh->hRefine(cellsToRefine, refPattern, false); // false: don't rebuild yet (wait for 1-irregularity enforcement)
    vlasovMesh->enforceOneIrregularity(true);
    
    poissonMesh->repartitionAndRebuild();
  }
  
  const double m    = TEST_M;
  
  FunctionPtr sgn = Function::sideParity();
  FunctionPtr n   = Function::normal();
  FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn;
  
  // to do the "pure" Poisson solve, we do need to have an f solution defined, since that enters the RHS
  std::map<int, FunctionPtr> initialGuess_Vlasov; // project the exact field solution onto the mesh
  initialGuess_Vlasov[form.f()->ID()] = f_exact;
  initialGuess_Vlasov[form.t_n_hat()->ID()] = tn_0;
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
  
  // check that tn_0 can be represented on the mesh, at least to within tol
  FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
  FunctionPtr tn_err = tn_0 - tn_soln;
  double tn_projection_err = tn_err->l2norm(vlasovSoln->mesh());
  double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
  out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;
  TEST_COMPARE(tn_projection_err, <=, tol * tn_soln_magnitude);
  
  // now, project again, this time with a zero t_n
  //  {
  //    // DEBUGGING: do we move from correct solution if we start there?
  //    out << "NOTE: projecting exact tn_hat onto solution -- for iterative solver, this is cheating.\n";
  //    initialGuess_Vlasov[form.t_n_hat()->ID()] = tn_0;
  //  }
  initialGuess_Vlasov[form.t_n_hat()->ID()] = Function::zero();
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);

//  form.setExportViz(true); // DEBUGGING
//  form.configurationSpaceFormulation()->setExportViz(true); // DEBUGGING
  form.configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.
  poissonSoln->solve();

  out << "Checking E_soln after pure Poisson solve.\n";
  double E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
  double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);

  out << "Checking VE_soln after pure Poisson solve.\n";
  
  FunctionPtr VE_soln = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
  double VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
  double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(VE_err_l2, <=, tol * VE_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_PurePoisson";
    
    auto poissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
  }

  form.solve();

//    if (outputViz)
//    {
//      std::ostringstream vizPath;
//      vizPath << vizPrefix << "_Vlasov";
//      auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vizPath.str(), "/tmp"));
//      vlasovViz->exportSolution(form.vlasovSolution());
//    }
  
  out << "Comparing f0 with f_soln after Vlasov-Poisson solve.\n";
  double f_soln_magnitude = f_soln->l2norm(vlasovSoln->mesh());
  double f_err_l2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(f_err_l2, <=, tol * f_soln_magnitude);
  
  out << "Checking E_soln after Vlasov-Poisson solve.\n";
  E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
  E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());
  
  TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);
  
  out << "Checking VE_soln after Vlasov-Poisson solve.\n";
  VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
  VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());
  
  TEST_COMPARE(VE_err_l2, <=, tol * VE_soln_magnitude);

  out << "Comparing tn_0 with t_n_soln after Vlasov-Poisson solve.\n";

  const int time_comp_zero_based = form.configurationSpaceMesh()->getDimension() - 1; // time is last dimension of Poisson mesh
  SpatialFilterPtr initialTimeFilter = SpatialFilter::matchingComponent(time_comp_zero_based, TEST_T0);

  double tn_err_overall = tn_err->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(tn_err_overall, <=, tol * tn_soln_magnitude);

  out << "Comparing initial-time BC with solution (checking initial time imposition).\n";
  auto initialTime_tn_err = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction(tn_err, initialTimeFilter);
  double timeBC_err = initialTime_tn_err->l2norm(vlasovSoln->mesh());
  auto initialTime_tn_soln = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction(tn_soln, initialTimeFilter);
  double initialTime_tn_magnitude = initialTime_tn_soln->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(timeBC_err, <=, tol * initialTime_tn_magnitude);
  
  TEST_ASSERT(tn_0->boundaryValueOnly() || tn_0->isZero()); // if zero, we allow it not to claim to be boundary-only.  (This is an artifact of the logic regarding Function products: zero times anything is zero, and we don't retain the boundary-valuedness in the zero result.  We could fix this, but it's not a high priority.)
  TEST_ASSERT(tn_soln->boundaryValueOnly());

  tn_soln_magnitude = tn_0->l2norm(vlasovSoln->mesh());
  double tn_err_l2 = (tn_soln-tn_0)->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(tn_err_l2, <=, tol * tn_soln_magnitude);
  
  out << "Checking f_soln.\n";
  f_soln_magnitude = f_exact->l2norm(vlasovSoln->mesh());
  f_err_l2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(f_err_l2, <=, tol * f_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream vlasovVizPath;
    vlasovVizPath << vizPrefix << "_Vlasov";
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});
    
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_Poisson";
    
    auto poissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
  }
}

void patchTestVlasovPoisson(FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                            const double q,
                            const int H1Order, const int xElements, const int tElements, const int vElements,
                            const bool overrideVlasovBCs,
                            const bool useSerendipity,
                            const double tol,
                            Teuchos::FancyOStream &out, bool &success,
                            const bool outputViz=false, const string vizPrefix="")
{
  std::vector<std::vector<GlobalIndexType> > refinementSequence;
  patchTestVlasovPoisson(f_exact, E_exact, VE_exact,
                         q, H1Order, xElements, tElements, vElements, refinementSequence, overrideVlasovBCs, useSerendipity, tol, out, success, outputViz, vizPrefix);
}

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, NextTimeSlab )
  {
    // sets up a formulation like the PoissonTwoStreamSteadyState test below, but does not solve.  Just checks that the new slab
    // is created, and has the right number of elements.  We might want to add further checks in the future.
    double n0 = 1.;
    double sigma = 1;
    double u0 = 3.0;
    FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, 0.5 * n0) );
    FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1T1V(-u0, sigma, 0.5 * n0) );
    FunctionPtr f0           = fM_u0 + fM_minus_u0;
    
    const double q = 1.;
    
    const int xElements = 2;
    const int tElements = 1;
    const int vElements = 8; // we need a somewhat fine mesh in v to get relatively good resolution
    const int H1Order   = 4;
    
    const bool useSerendipity    = false;
    
    FunctionPtr  f_exact = f0;
    FunctionPtr VE_exact = Function::zero();
    FunctionPtr  E_exact = Function::zero();
        
    const bool usePeriodicBCs = true;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, useSerendipity);
   
    form.setInitialAndInflowConditions(f_exact, f_exact);
    
    bool useGeometricMultigrid = true;
    bool cgTol = 1e-6 / 100.0;
    int maxSubsteps = 3;
    int minSubsteps = 1;
    
    std::vector<double> cgTols {1,2,3};
    
    bool enforceQuasiNeutrality = true;
    int azOutputLevel = 5;
    int maxIters = 1000;
    bool exportViz = true;
    std::string vizDir = "/some/random/dir";
    
    form.setUseGeometricMultigrid(useGeometricMultigrid);
    form.setIterativeSolveTolerance(cgTol);
    form.setIterativeSolveTolerances(cgTols);
    form.setFixedPointMaxSubsteps(maxSubsteps);
    form.setFixedPointMinSubsteps(minSubsteps);
    form.setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    form.setIterativeSolveOutputLevel(azOutputLevel);
    form.setIterativeSolveMaxIterations(maxIters);
    form.setExportViz(exportViz, vizDir);
    
    auto nextSlab = form.nextTimeSlab(TEST_TWIDTH, tElements);
    
    int numElementsExpected = xElements * tElements * vElements;
    int numElementsActual   = nextSlab->vlasovMesh()->numActiveElements();
    TEST_EQUALITY(numElementsExpected, numElementsActual);
    
    TEST_EQUALITY(useGeometricMultigrid,  nextSlab->useGeometricMultigrid());
    TEST_EQUALITY(cgTol,                  nextSlab->iterativeSolveTolerance());
    TEST_EQUALITY(maxSubsteps,            nextSlab->fixedPointMaxSubsteps());
    TEST_EQUALITY(minSubsteps,            nextSlab->fixedPointMinSubsteps());
    TEST_EQUALITY(enforceQuasiNeutrality, nextSlab->enforceQuasiNeutrality());
    TEST_EQUALITY(azOutputLevel,          nextSlab->iterativeSolveOutputLevel());
    TEST_EQUALITY(maxIters,               nextSlab->iterativeSolveMaxIterations());
    TEST_EQUALITY(exportViz,              nextSlab->exportViz());
    TEST_EQUALITY(vizDir,                 nextSlab->vizOutputDir());
    
    auto nextSlabCGTols = nextSlab->iterativeSolveTolerances();
    TEST_EQUALITY(cgTols.size(), nextSlabCGTols.size());
    if (cgTols.size() == nextSlabCGTols.size())
    {
      for (int i=0; i<int(cgTols.size()); i++)
      {
        TEST_EQUALITY(cgTols[i], nextSlabCGTols[i]);
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonTwoStreamSteadyState )
  {
    // like the two-stream instability problem, but no perturbation.  Should stay fixed.
    double n0 = 1.;
    double sigma = 1;
    double u0 = 3.0;
    FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1T1V( u0, sigma, 0.5 * n0) );
    FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1T1V(-u0, sigma, 0.5 * n0) );
    FunctionPtr f0           = fM_u0 + fM_minus_u0;
    
    const double q = 1.;
    
    const int xElements = 2;
    const int tElements = 1;
    const int vElements = 8; // we need a somewhat fine mesh in v to get relatively good resolution
    const int H1Order   = 4;
    
    const bool useSerendipity    = false;
    const double tolRelativeToProjectionError = 1.1; // we allow 10% error relative to the best approximation for f, tn
    const double absTolPoisson = 1e-4; // Poisson solutions should be 0, so we use an absolute tol for these
    
    FunctionPtr  f_exact = f0;
    FunctionPtr VE_exact = Function::zero();
    FunctionPtr  E_exact = Function::zero();
    
    const bool outputViz = false;
    const std::string vizPrefix = "TwoStreamTest";
    
    const bool usePeriodicBCs = true;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, useSerendipity);
    
    form.setUseGeometricMultigrid(true);
    form.setIterativeSolveTolerance(1e-8);
    form.setFixedPointMaxSubsteps(1);
    form.setFixedPointMinSubsteps(1);
    
    FunctionPtr vx = form.v_i(0);

    auto vlasovSoln = form.vlasovSolution();
    
    auto poissonForm = form.configurationSpaceFormulation();
    auto poissonSoln = form.configurationSpaceSolution();
    FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
    FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
        
    form.setInitialAndInflowConditions(f_exact, f_exact);
    
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
    
    const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
    auto xLeftTag = form.createConfigurationMeshSideSetTag(xLeftFilter, restrictSideSetsToBoundary);
    auto VE_traceVar = form.VE_configurationSpaceTrace();
    form.addConfigurationSpaceBC(xLeftTag, VE_traceVar, VE_exact);
    
//    auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
//    auto vlasovMesh  = vlasovSoln->mesh();
//    auto poissonMesh = poissonSoln->mesh();
//    for (const auto &cellsToRefine : refinementSequence)
//    {
//      auto refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
//      vlasovMesh->hRefine(cellsToRefine, refPattern, false); // false: don't rebuild yet (wait for 1-irregularity enforcement)
//      vlasovMesh->enforceOneIrregularity(true);
//
//      poissonMesh->repartitionAndRebuild();
//    }
    
    const double m    = TEST_M;
    
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr n   = Function::normal();
    FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn;
    
    // to do the "pure" Poisson solve, we do need to have an f solution defined, since that enters the RHS
    std::map<int, FunctionPtr> initialGuess_Vlasov; // project the exact field solution onto the mesh
    initialGuess_Vlasov[form.f()->ID()] = f_exact;
    vlasovSoln->projectOntoMesh(initialGuess_Vlasov);

    double f_magnitude = f_exact->l2norm(vlasovSoln->mesh());
    double f_proj_err_l2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
    out << "f_proj_err_l2: " << f_proj_err_l2 / f_magnitude << std::endl;
    
    double relTol_f = f_proj_err_l2 / f_magnitude * tolRelativeToProjectionError;
    
//    form.setExportViz(true); // DEBUGGING
//    form.configurationSpaceFormulation()->setExportViz(true); // DEBUGGING
    form.configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.
    poissonSoln->solve();

    out << "Checking E_soln after pure Poisson solve.\n";
    double soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
    double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

    TEST_COMPARE(E_err_l2, <=, absTolPoisson);

    out << "Checking VE_soln after pure Poisson solve.\n";
    
    FunctionPtr VE_soln = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
    soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
    double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

    TEST_COMPARE(VE_err_l2, <=, absTolPoisson);
    
    if (outputViz)
    {
      std::ostringstream poissonVizPath;
      poissonVizPath << vizPrefix << "_PurePoisson";

      auto poissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
      poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
    }

    form.solve();

    if (outputViz)
    {
      std::ostringstream vizPath;
      vizPath << vizPrefix << "_Vlasov";
      auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vizPath.str(), "/tmp"));
      vlasovViz->exportSolution(form.vlasovSolution());
    }
    
    out << "Comparing f0 with f_soln after Vlasov-Poisson solve.\n";
    soln_magnitude = f_soln->l2norm(vlasovSoln->mesh());
    double f_err_l2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(f_err_l2, <=, relTol_f * soln_magnitude);
    
    out << "Checking E_soln after Vlasov-Poisson solve.\n";
    soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
    E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());
    
    TEST_COMPARE(E_err_l2, <=, absTolPoisson);
    
    out << "Checking VE_soln after Vlasov-Poisson solve.\n";
    soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
    VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());
    
    TEST_COMPARE(VE_err_l2, <=, absTolPoisson);

    out << "Comparing tn_0 with t_n_soln after Vlasov-Poisson solve.\n";
    
    FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
    double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
    out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;

    const int time_comp_zero_based = form.configurationSpaceMesh()->getDimension() - 1; // time is last dimension of Poisson mesh
    SpatialFilterPtr initialTimeFilter = SpatialFilter::matchingComponent(time_comp_zero_based, TEST_T0);

    FunctionPtr tn_err = tn_0 - tn_soln;

    double tn_err_overall = tn_err->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(tn_err_overall, <=, relTol_f * tn_soln_magnitude);

    out << "Comparing initial-time BC with solution (checking initial time imposition).\n";
    auto initialTime_tn_err = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction(tn_err, initialTimeFilter);
    auto initialTime_tn_soln = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction(tn_soln, initialTimeFilter);
    double initialTime_tn_magnitude = initialTime_tn_soln->l2norm(vlasovSoln->mesh());
    double timeBC_err = initialTime_tn_err->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(timeBC_err, <=, relTol_f * initialTime_tn_magnitude);
    
    TEST_ASSERT(tn_0->boundaryValueOnly() || tn_0->isZero()); // if zero, we allow it not to claim to be boundary-only.  (This is an artifact of the logic regarding Function products: zero times anything is zero, and we don't retain the boundary-valuedness in the zero result.  We could fix this, but it's not a high priority.)
    TEST_ASSERT(tn_soln->boundaryValueOnly());

    tn_soln_magnitude = tn_0->l2norm(vlasovSoln->mesh());
    double tn_err_l2 = (tn_soln-tn_0)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(tn_err_l2, <=, relTol_f * tn_soln_magnitude);
    
    out << "Checking f_soln.\n";
    soln_magnitude = f_exact->l2norm(vlasovSoln->mesh());
    f_err_l2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(f_err_l2, <=, relTol_f * soln_magnitude);
    
    if (outputViz)
    {
      std::ostringstream vlasovVizPath;
      vlasovVizPath << vizPrefix << "_Vlasov";
      auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovVizPath.str(), "/tmp"));
      vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});

      std::ostringstream poissonVizPath;
      poissonVizPath << vizPrefix << "_Poisson";

      auto poissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
      poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
    }
    
    out << "Continuing on to next time slab.\n";
    auto nextSlab = form.nextTimeSlab(TEST_TWIDTH, tElements);
    
    nextSlab->solve();
    
    FunctionPtr f_soln_nextSlab = Function::solution(nextSlab->f(), nextSlab->vlasovSolution());
    soln_magnitude = f_exact->l2norm(vlasovSoln->mesh());
    f_err_l2 = (f_soln_nextSlab-f_exact)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(f_err_l2, <=, relTol_f * soln_magnitude);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonPatchTest_f_xv2_E_zero )
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int tElements = 1;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
    
    const int spaceDim = 1;
    const bool spaceTimeMesh = true;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = getVx(spaceDim, spaceTimeMesh);
    FunctionPtr v2 = v * v;
    FunctionPtr f_exact = x * v2;
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();
    
    const bool overrideVlasovBCs = false;
    const bool useSerendipity    = false;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonPatchTest_f_xv2_E_zero_Serendipity )
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int tElements = 1;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 5; // quadratic L^2 functions; allows exact representation of analytic solution trace
    
    const int spaceDim = 1;
    const bool spaceTimeMesh = true;
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = getVx(spaceDim, spaceTimeMesh);
    FunctionPtr v2 = v * v;
    FunctionPtr f_exact = x * v2;
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();
    
    const bool overrideVlasovBCs = false;
    const bool useSerendipity    = true;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_zero_E_1 )
  {
    const double q = TEST_Q;
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr f_exact   = Function::constant(0.0);
    FunctionPtr E_exact = Function::constant(1.0);
    FunctionPtr V_E_exact = -x;
    
    const int xElements = 3;
    const int tElements = 2;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 2; // linear L^2 functions (allows exact VE)
    
    const bool overrideVlasovBCs = false;
    const bool useSerendipity    = false;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, V_E_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_zero_E_1_Serendipity )
  {
    const double q = TEST_Q;
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr f_exact   = Function::constant(0.0);
    FunctionPtr E_exact = Function::constant(1.0);
    FunctionPtr V_E_exact = -x;
    
    const int xElements = 3;
    const int tElements = 2;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 2; // linear L^2 functions (allows exact VE)
    
    const bool overrideVlasovBCs = false;
    const bool useSerendipity    = true;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, V_E_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonPatchTest_f_constant_E_linear )
  {
    const double q      = TEST_Q;
    
    const int spaceDim = 1;
    const bool spaceTimeMesh = true;
    auto vx = getVx(spaceDim, spaceTimeMesh);
    
    FunctionPtr x       = Function::xn(1);
    FunctionPtr f_exact = Function::constant(1.0);
    FunctionPtr E_exact =   (q / TEST_EPS0 * TEST_VWIDTH) * x;
    FunctionPtr V_E     = - (q / TEST_EPS0 * TEST_VWIDTH) / 2.0 * x * x;

    const int xElements = 3;
    const int tElements = 1;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const double tol = 1e-12;
    
    const bool outputViz = false;
    const std::string vizPrefix = "PatchTest_f_constant_E_linear";
    const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
    const bool useSerendipity    = false;
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol, out, success, outputViz, vizPrefix);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonPatchTest_f_constant_E_linear_Serendipity )
  {
    const double q      = TEST_Q;
    
    const int spaceDim = 1;
    const bool spaceTimeMesh = true;
    auto vx = getVx(spaceDim, spaceTimeMesh);
    
    FunctionPtr x       = Function::xn(1);
    FunctionPtr f_exact = Function::constant(1.0);
    FunctionPtr E_exact =   (q / TEST_EPS0 * TEST_VWIDTH) * x;
    FunctionPtr V_E     = - (q / TEST_EPS0 * TEST_VWIDTH) / 2.0 * x * x;

    const int xElements = 3;
    const int tElements = 1;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const double tol = 1e-12;
    
    const bool outputViz = false;
    const std::string vizPrefix = "PatchTest_f_constant_E_linear";
    const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
    const bool useSerendipity    = true;
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol, out, success, outputViz, vizPrefix);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic )
  {
    const double q      = TEST_Q;
    const double eps0   = TEST_EPS0;
    
    FunctionPtr x = Function::xn(1);
    const double vWidth = TEST_VWIDTH;
    FunctionPtr f_exact = x/vWidth - (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the phase-space domain
    
    FunctionPtr E_exact = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x);
    FunctionPtr V_E = -(q/eps0) * (x * x * x / 6.0 - ((TEST_X0 + TEST_XWIDTH) / 4.0) * x * x);

    const int xElements = 3;
    const int tElements = 1;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const double tol = 1e-12;
    
    const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
    const bool useSerendipity    = false;
    
    const bool outputViz = true;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic";
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol,
                           out, success, outputViz, testPrefix);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic_Serendipity )
  {
    const double q      = TEST_Q;
    const double eps0   = TEST_EPS0;
    
    FunctionPtr x = Function::xn(1);
    const double vWidth = TEST_VWIDTH;
    FunctionPtr f_exact = x/vWidth - (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the phase-space domain
    
    FunctionPtr E_exact = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x);
    FunctionPtr V_E = -(q/eps0) * (x * x * x / 6.0 - ((TEST_X0 + TEST_XWIDTH) / 4.0) * x * x);

    const int xElements = 3;
    const int tElements = 1;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const double tol = 1e-12;
    
    const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
    const bool useSerendipity    = true;
    
    const bool outputViz = true;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic";
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, useSerendipity, tol,
                           out, success, outputViz, testPrefix);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, SerendipityIsNestedWithinTensor )
  {
    bool disableTest = true;
    if (disableTest)
    {
      std::cout << "\n\n******************************************* WARNING: Test Disabled ********************************************\n";
      std::cout << "* This test is disabled because of failures; it should be investigated, fixed, and re-enabled.  See Issue #134. *\n";
      std::cout << "***************************************************************************************************************\n\n";
      
      return;
    }
    
    // create Vlasov formulation (H^1 order 3), with Serendipity and without.
    
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 1;
    const int tElements = 1;
    const int vElements = 1;
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
    const bool usePeriodicBCs = false;
    
    const double tol = 1e-13;
    
    auto serendipityForm = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, true);  // true: use Serendipity
    auto fullTensorForm  = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, false); // false: don't use Serendipity
    
    auto serendipityIP = serendipityForm.vlasovSolution()->ip();
    auto fullTensorIP  = fullTensorForm.vlasovSolution()->ip();
    
    auto serendipityBF = serendipityForm.vlasovSolution()->bf();
    auto fullTensorBF  = fullTensorForm.vlasovSolution()->bf();
    
    auto serendipityMesh = serendipityForm.vlasovMesh();
    auto fullTensorMesh  = fullTensorForm.vlasovMesh();
    
    // Compute IP for each.  Test that the Serendipity basis IP maps onto the correct subset of the tensor basis IP.
    if (serendipityMesh->myCellsInclude(0))
    {
      const GlobalIndexType cellID = 0;
      auto serendipityTestOrdering = serendipityMesh->getElementType(cellID)->testOrderPtr;
      auto fullTensorTestOrdering  = fullTensorMesh->getElementType(cellID)->testOrderPtr;
      
      auto serendipityTrialOrdering = serendipityMesh->getElementType(cellID)->trialOrderPtr;
      auto fullTensorTrialOrdering  = fullTensorMesh->getElementType(cellID)->trialOrderPtr;
      
      auto serendipityBasisCache = BasisCache::basisCacheForCell(serendipityMesh, cellID);
      auto fullTensorBasisCache  = BasisCache::basisCacheForCell(fullTensorMesh,  cellID);
      
      auto serendipityTestDofCount = serendipityTestOrdering->totalDofs();
      auto fullTensorTestDofCount  = fullTensorTestOrdering->totalDofs();
      
      auto serendipityTrialDofCount = serendipityTrialOrdering->totalDofs();
      auto fullTensorTrialDofCount  = fullTensorTrialOrdering->totalDofs();
      
      map<int,int> testDofOrderingMap, trialDofOrderingMap;
      
      auto testVarIDs = serendipityTestOrdering->getVarIDs();
      for (const auto &varID : testVarIDs)
      {
        auto sides = serendipityTestOrdering->getSidesForVarID(varID);
        for (const auto & side : sides)
        {
          BasisPtr basis = serendipityTestOrdering->getBasis(varID, side);
          
          // try Intrepid2 TensorBasis
          using Intrepid2Basis = Intrepid2::Basis<DeviceType,double,double>;
          
          using SerendipityBasis = Intrepid2::SerendipityBasis<Intrepid2Basis>;
          
          SerendipityBasis* serendipityBasis = dynamic_cast<SerendipityBasis*>(basis->intrepid2Basis().get());
          
          TEUCHOS_TEST_FOR_EXCEPTION(serendipityBasis == NULL, std::invalid_argument, "Serendipity formulation has a non-Serendipity basis.");
          
          auto ordinalMap = serendipityBasis->ordinalMap();
          auto ordinalMapHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ordinalMap);
          
          for (int i=0; i<ordinalMapHost.extent_int(0); i++)
          {
            int iTensor = ordinalMapHost(i);
            int serendipityDofOrdinal = serendipityTestOrdering->getDofIndex(varID,      i, side);
            int fullTensorDofOrdinal  = fullTensorTestOrdering->getDofIndex(varID, iTensor, side);
            
            testDofOrderingMap[serendipityDofOrdinal] = fullTensorDofOrdinal;
          }
          
          // while we're here, let's also check the basis values:
          auto fullTensorBasisIntrepid2 = serendipityBasis->getUnderlyingBasis();
          auto fullTensorBasis = Teuchos::rcp(new Intrepid2BasisWrapper<>(fullTensorBasisIntrepid2, basis->rangeDimension(), basis->rangeRank(), basis->functionSpace()));
          
          Teuchos::RCP< const FieldContainer<double> > fullTensorValues, serendipityValues;
          if (side == -1)
          {
            fullTensorValues  = fullTensorBasisCache->getTransformedValues(fullTensorBasis, OP_VALUE);
            serendipityValues = serendipityBasisCache->getTransformedValues(basis,          OP_VALUE);
          }
          else
          {
            fullTensorValues  =  fullTensorBasisCache->getSideBasisCache(side)->getTransformedValues(fullTensorBasis, OP_VALUE);
            serendipityValues = serendipityBasisCache->getSideBasisCache(side)->getTransformedValues(basis,           OP_VALUE);
          }
          
          const int cellOrdinal = 0;
          const int numPoints   = fullTensorValues->dimension(2);
          TEUCHOS_TEST_FOR_EXCEPTION(serendipityValues->dimension(2) != numPoints, std::invalid_argument, "numPoints do not match");
          for (int i=0; i<ordinalMapHost.extent_int(0); i++)
          {
            int iTensor = ordinalMapHost(i);
            for (int l=0; l<numPoints; l++)
            {
              const double diff = std::abs((*serendipityValues)(cellOrdinal,i,l) - (*fullTensorValues)(cellOrdinal,iTensor,l));
              if (diff > tol)
              {
                out << "(" << i << "," << l << ")\n";
                TEST_FLOATING_EQUALITY((*serendipityValues)(cellOrdinal,i,l), (*fullTensorValues)(cellOrdinal,iTensor,l), tol);
              }
            }
          }
        }
      }
      
      out << "testDofOrderingMap:\n";
      for (auto entry : testDofOrderingMap)
      {
        out << entry.first << " --> " << entry.second << std::endl;
      }
      
      auto trialVarIDs = serendipityTrialOrdering->getVarIDs();
      for (const auto &varID : trialVarIDs)
      {
        auto sides = serendipityTrialOrdering->getSidesForVarID(varID);
        for (const auto & side : sides)
        {
          BasisPtr basis = serendipityTrialOrdering->getBasis(varID, side);
          
          // try Intrepid2 TensorBasis
          using Intrepid2Basis = Intrepid2::Basis<DeviceType,double,double>;
          
          using SerendipityBasis = Intrepid2::SerendipityBasis<Intrepid2Basis>;
          
          SerendipityBasis* serendipityBasis = dynamic_cast<SerendipityBasis*>(basis->intrepid2Basis().get());
          
          TEUCHOS_TEST_FOR_EXCEPTION(serendipityBasis == NULL, std::invalid_argument, "Serendipity formulation has a non-Serendipity basis.");
          
          auto ordinalMap = serendipityBasis->ordinalMap();
          auto ordinalMapHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ordinalMap);
          
          for (int i=0; i<ordinalMapHost.extent_int(0); i++)
          {
            int iTensor = ordinalMapHost(i);
            int serendipityDofOrdinal = serendipityTrialOrdering->getDofIndex(varID,      i, side);
            int fullTensorDofOrdinal  = fullTensorTrialOrdering->getDofIndex(varID, iTensor, side);
            
            trialDofOrderingMap[serendipityDofOrdinal] = fullTensorDofOrdinal;
          }
          
          // while we're here, let's also check the basis values:
          auto fullTensorBasisIntrepid2 = serendipityBasis->getUnderlyingBasis();
          auto fullTensorBasis = Teuchos::rcp(new Intrepid2BasisWrapper<>(fullTensorBasisIntrepid2, basis->rangeDimension(), basis->rangeRank(), basis->functionSpace()));
          
          Teuchos::RCP< const FieldContainer<double> > fullTensorValues, serendipityValues;
          if (side == -1)
          {
            fullTensorValues  = fullTensorBasisCache->getTransformedValues(fullTensorBasis, OP_VALUE);
            serendipityValues = serendipityBasisCache->getTransformedValues(basis,          OP_VALUE);
          }
          else
          {
            fullTensorValues  =  fullTensorBasisCache->getSideBasisCache(side)->getTransformedValues(fullTensorBasis, OP_VALUE);
            serendipityValues = serendipityBasisCache->getSideBasisCache(side)->getTransformedValues(basis,           OP_VALUE);
          }
          
          const int cellOrdinal = 0;
          const int numPoints   = fullTensorValues->dimension(2);
          TEUCHOS_TEST_FOR_EXCEPTION(serendipityValues->dimension(2) != numPoints, std::invalid_argument, "numPoints do not match");
          for (int i=0; i<ordinalMapHost.extent_int(0); i++)
          {
            int iTensor = ordinalMapHost(i);
            for (int l=0; l<numPoints; l++)
            {
              const double diff = std::abs((*serendipityValues)(cellOrdinal,i,l) - (*fullTensorValues)(cellOrdinal,iTensor,l));
              if (diff > tol)
              {
                out << "(" << i << "," << l << ")\n";
                TEST_FLOATING_EQUALITY((*serendipityValues)(cellOrdinal,i,l), (*fullTensorValues)(cellOrdinal,iTensor,l), tol);
              }
            }
          }
        }
      }
      
      out << "trialDofOrderingMap:\n";
      for (auto entry : trialDofOrderingMap)
      {
        out << entry.first << " --> " << entry.second << std::endl;
      }
      
      const int numCells = 1;
      FieldContainer<double> serendipityGramMatrix(numCells,serendipityTestDofCount,serendipityTestDofCount);
      FieldContainer<double> fullTensorGramMatrix (numCells, fullTensorTestDofCount, fullTensorTestDofCount);
      
      serendipityIP->computeInnerProductMatrix(serendipityGramMatrix, serendipityTestOrdering, serendipityBasisCache);
      fullTensorIP->computeInnerProductMatrix( fullTensorGramMatrix,  fullTensorTestOrdering,  fullTensorBasisCache);
      
      const int cellOrdinal = 0;
      for (int i=0; i<serendipityTestDofCount; i++)
      {
        const int iTensor = testDofOrderingMap[i];
        for (int j=0; j<serendipityTestDofCount; j++)
        {
          const int jTensor = testDofOrderingMap[j];
          
          const double diff = std::abs(serendipityGramMatrix(cellOrdinal,i,j) - fullTensorGramMatrix(cellOrdinal,iTensor,jTensor));
          
          // we really expect exact zero difference, on current implementations.
          if (diff > tol)
          {
            out << "(" << i << "," << j << ")\n";
            TEST_FLOATING_EQUALITY(serendipityGramMatrix(cellOrdinal,i,j), fullTensorGramMatrix(cellOrdinal,iTensor,jTensor), tol);
          }
        }
      }
    
      // Do the same for the rectangular BF.  (The square BF will not precisely nest: the discrete test space differs...)
      FieldContainer<double> serendipityStiffnessMatrix(numCells,serendipityTestDofCount,serendipityTrialDofCount);
      FieldContainer<double> fullTensorStiffnessMatrix (numCells, fullTensorTestDofCount, fullTensorTrialDofCount);
      
      const int numElements = 1;
      auto elemTypeSerendipity = serendipityMesh->getElementType(cellID);
      auto elemTypeFullTensor  = fullTensorMesh-> getElementType(cellID);
      FieldContainer<double> K_serendipity(numElements,serendipityTestDofCount,serendipityTrialDofCount);
      FieldContainer<double> K_fullTensor (numElements, fullTensorTestDofCount, fullTensorTrialDofCount);
      BasisCachePtr basisCacheSerendipity = BasisCache::basisCacheForCell(serendipityMesh, cellID);
      BasisCachePtr basisCacheFullTensor  = BasisCache::basisCacheForCell( fullTensorMesh, cellID);
      serendipityBF->stiffnessMatrix(K_serendipity, elemTypeSerendipity, basisCacheSerendipity->getCellSideParities(), basisCacheSerendipity);
      fullTensorBF ->stiffnessMatrix(K_fullTensor,  elemTypeFullTensor,  basisCacheFullTensor->getCellSideParities(),  basisCacheFullTensor );
      
      for (int i=0; i<serendipityTestDofCount; i++)
      {
        const int iTensor = testDofOrderingMap[i];
        for (int j=0; j<serendipityTrialDofCount; j++)
        {
          const int jTensor = trialDofOrderingMap[j];
          
          const double diff = std::abs(K_serendipity(cellOrdinal,i,j) - K_fullTensor(cellOrdinal,iTensor,jTensor));
          
          // we really expect exact zero difference, on current implementations.
          if (diff > tol)
          {
            out << "(" << i << "," << j << ")\n";
            TEST_FLOATING_EQUALITY(K_serendipity(cellOrdinal,i,j), K_fullTensor(cellOrdinal,iTensor,jTensor), tol);
          }
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, SideSetInitialTime )
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int tElements = 1;
    const int vElements = 4;
    const int H1Order   = 3;
    const bool usePeriodicBCs = false;
    const bool useSerendipity = false; // should not affect test in any way
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, useSerendipity);

    auto vlasovSoln = form.vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int initialTimeTag = form.tagVlasovInitialTime();

    auto actualSideSets = vlasovMeshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, initialTimeTag);
    
    std::set<IndexType> sideIndices;
    
    const int phaseSpaceDim = vlasovMeshTopo->getDimension();
    const int sideDim       = phaseSpaceDim - 1;
    for (auto sideSet : actualSideSets)
    {
      auto entityIndices = sideSet->entitiesOfDimension(sideDim);
      sideIndices.insert(entityIndices.begin(), entityIndices.end());
      
      int maxEntityDimension = sideSet->maxEntityDimension();
      
      // check that there aren't any non-side entities in the set (that would be weird)
      for (int d=0; d<=maxEntityDimension; d++)
      {
        if (d==sideDim) continue;
        TEST_EQUALITY(sideSet->entitiesOfDimension(d).size(), 0);
      }
    }
    
    int expectedSideCount = xElements * vElements;
    int actualSideCount   = sideIndices.size();
    
    TEST_EQUALITY(expectedSideCount, actualSideCount);
    
    const int TIME_COMPONENT_INDEX_ZERO_BASED = 1;
    
    // next, check that the sides that did match each have t=0.
    for (auto sideIndex : sideIndices)
    {
      auto vertexIndices = vlasovMeshTopo->getEntityVertexIndices(sideDim, sideIndex);
      for (auto vertexIndex : vertexIndices)
      {
        const auto & vertex = vlasovMeshTopo->getVertex(vertexIndex);
        TEST_EQUALITY(vertex[TIME_COMPONENT_INDEX_ZERO_BASED], 0.0);
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, SideSetInflow )
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int tElements = 1;
    const int vElements = 4;
    const int H1Order   = 3;
    const bool usePeriodicBCs = false;
    const bool useSerendipity = false; // should not affect test in any way
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, useSerendipity);

    auto vlasovSoln = form.vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int inflowTag = form.tagVlasovSpatialInflow();

    auto actualSideSets = vlasovMeshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, inflowTag);
    
    std::set<IndexType> sideIndices;
    
    const int phaseSpaceDim = vlasovMeshTopo->getDimension();
    const int sideDim       = phaseSpaceDim - 1;
    for (auto sideSet : actualSideSets)
    {
      int maxEntityDimension = sideSet->maxEntityDimension();
      TEST_EQUALITY(maxEntityDimension, sideDim);
      
      auto entityIndices = sideSet->entitiesOfDimension(sideDim);
      sideIndices.insert(entityIndices.begin(), entityIndices.end());
            
      // check that there aren't any non-side entities in the set (that would be weird)
      for (int d=0; d<sideDim; d++)
      {
        TEST_EQUALITY(sideSet->entitiesOfDimension(d).size(), 0);
      }
    }
    
    int expectedSideCount = vElements * tElements; // upper left plus lower right
    int actualSideCount   = sideIndices.size();
    
    TEST_EQUALITY(expectedSideCount, actualSideCount);
    
    const int SPACE_COMPONENT_INDEX_ZERO_BASED    = 0;
    const int VELOCITY_COMPONENT_INDEX_ZERO_BASED = 2;
    
    const auto TEST_X_MAX = TEST_X0 + TEST_XWIDTH;
    // next, check that the sides that did match each have (v>=0 and x==0) or (v <= 0 and x==X_MAX).
    for (auto sideIndex : sideIndices)
    {
      auto vertexIndices = vlasovMeshTopo->getEntityVertexIndices(sideDim, sideIndex);
      for (auto vertexIndex : vertexIndices)
      {
        const auto & vertex = vlasovMeshTopo->getVertex(vertexIndex);
        const auto & x      = vertex[SPACE_COMPONENT_INDEX_ZERO_BASED];
        const auto & v      = vertex[VELOCITY_COMPONENT_INDEX_ZERO_BASED];
        const bool  leftInflow = (x == TEST_X0)    && (v >= 0.0);
        const bool rightInflow = (x == TEST_X_MAX) && (v <= 0.0);
        
        TEST_ASSERT( leftInflow || rightInflow );
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, SideSetMaxVelocity )
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int tElements = 1;
    const int vElements = 4;
    const int H1Order   = 3;
    const bool usePeriodicBCs = false;
    const bool useSerendipity = false; // should not affect test in any way
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, useSerendipity);

    auto vlasovSoln = form.vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int maxVelocityTag = form.tagVlasovMaxVelocity(0);

    auto actualSideSets = vlasovMeshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, maxVelocityTag);
    
    std::set<IndexType> sideIndices;
    
    const int phaseSpaceDim = vlasovMeshTopo->getDimension();
    const int sideDim       = phaseSpaceDim - 1;
    for (auto sideSet : actualSideSets)
    {
      int maxEntityDimension = sideSet->maxEntityDimension();
      TEST_EQUALITY(maxEntityDimension, sideDim);
      
      auto entityIndices = sideSet->entitiesOfDimension(sideDim);
      sideIndices.insert(entityIndices.begin(), entityIndices.end());
            
      // check that there aren't any non-side entities in the set (that would be weird)
      for (int d=0; d<sideDim; d++)
      {
        TEST_EQUALITY(sideSet->entitiesOfDimension(d).size(), 0);
      }
    }
    
    int expectedSideCount = xElements * tElements; // slice with v=vmax
    int actualSideCount   = sideIndices.size();
    
    TEST_EQUALITY(expectedSideCount, actualSideCount);
    
    const int VELOCITY_COMPONENT_INDEX_ZERO_BASED = 2;
    
    // next, check that the sides that did match each have v==vmax
    for (auto sideIndex : sideIndices)
    {
      auto vertexIndices = vlasovMeshTopo->getEntityVertexIndices(sideDim, sideIndex);
      for (auto vertexIndex : vertexIndices)
      {
        const auto & vertex = vlasovMeshTopo->getVertex(vertexIndex);
        TEST_COMPARE(vertex[VELOCITY_COMPONENT_INDEX_ZERO_BASED], ==, TEST_V0 + TEST_VWIDTH);
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, SideSetMinVelocity )
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int tElements = 1;
    const int vElements = 4;
    const int H1Order   = 3;
    const bool usePeriodicBCs = false;
    const bool useSerendipity = false; // should not affect test in any way
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, usePeriodicBCs, q, useSerendipity);

    auto vlasovSoln = form.vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int minVelocityTag = form.tagVlasovMinVelocity(0);

    auto actualSideSets = vlasovMeshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, minVelocityTag);
    
    std::set<IndexType> sideIndices;
    
    const int phaseSpaceDim = vlasovMeshTopo->getDimension();
    const int sideDim       = phaseSpaceDim - 1;
    for (auto sideSet : actualSideSets)
    {
      int maxEntityDimension = sideSet->maxEntityDimension();
      TEST_EQUALITY(maxEntityDimension, sideDim);
      
      auto entityIndices = sideSet->entitiesOfDimension(sideDim);
      sideIndices.insert(entityIndices.begin(), entityIndices.end());
            
      // check that there aren't any non-side entities in the set (that would be weird)
      for (int d=0; d<sideDim; d++)
      {
        TEST_EQUALITY(sideSet->entitiesOfDimension(d).size(), 0);
      }
    }
    
    int expectedSideCount = xElements * tElements; // slice with v=vmin
    int actualSideCount   = sideIndices.size();
    
    TEST_EQUALITY(expectedSideCount, actualSideCount);
    
    const int VELOCITY_COMPONENT_INDEX_ZERO_BASED = 2;
    
    // next, check that the sides that did match each have v==vmin
    for (auto sideIndex : sideIndices)
    {
      auto vertexIndices = vlasovMeshTopo->getEntityVertexIndices(sideDim, sideIndex);
      for (auto vertexIndex : vertexIndices)
      {
        const auto & vertex = vlasovMeshTopo->getVertex(vertexIndex);
        TEST_COMPARE(vertex[VELOCITY_COMPONENT_INDEX_ZERO_BASED], ==, TEST_V0);
      }
    }
  }
} // end anonymous namespace
