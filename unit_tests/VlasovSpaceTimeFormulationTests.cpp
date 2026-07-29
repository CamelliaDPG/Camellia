//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  VlasovSpaceTimeFormulationTests.cpp
//
//  Created by Nate Roberts on 11/15/2022.
//
//

#include "MimeticVlasovSpaceTimeFormulation.h"
#include "VlasovSpaceTimeFormulation.h"
#include "RelativisticVlasovSpaceTimeFormulation.h"
#include "ConfigurationSpaceFormulationPoisson.h"
#include "ConfigurationSpaceFormulationMaxwell1D2V.h"
#include "MeshTools.h"

#include "BF.h"
#include "MaxwellianFunction.h"
#include "MonomialFunctions.h"
#include "PhaseSpaceVisualization.h"
#include "RHS.h"
#include "Solution.h"

#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaTestingHelpers.h"

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

  LinearTermPtr vlasovRHSForExactSoln(Teuchos::RCP<VlasovSpaceTimeFormulation> form, FunctionPtr f_exact, FunctionPtr E_exact, const double q)
  {
    // want an RHS so that f_exact is the exact solution
    auto w = form->w(); // test function for Vlasov RHS
    const int spaceDim              = form->configurationSpaceMesh()->getDimension() - 1; // space, not space-time, dim
    const int velocityDim           = form->vlasovMesh()->getDimension() - spaceDim - 1;
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
      if (i < E_soln_vector.size()) // if 1D2V, e.g., modeling assumption is that E_y = 0.
      {
        lhsAtExactSoln = lhsAtExactSoln + ((q / TEST_M) * E_soln_vector[i] * f_exact)->di(vComp);
      }
    }
    return lhsAtExactSoln * w;
  }

  Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T1V(const int xElements, const int tElements, const int vElements,
                                                                const int H1Order, const bool useGradedSpatialMesh,
                                                                const bool usePeriodicBCs, double q,
                                                                const EBasisFamily &basisFamily, const bool useMimeticDPG,
                                                                const int delta_k = 2,
                                                                const bool enforcePositivity = false)
  {
    // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
    double eps0   = TEST_EPS0;
    double m      = TEST_M;
    double x0     = TEST_X0;
    double xWidth = TEST_XWIDTH;
    double t0     = TEST_T0;
    double v0     = TEST_V0;
    double vWidth = TEST_VWIDTH;
    
    const int numUniformRefinements = 0;
    
    Teuchos::RCP<ConfigurationSpaceFormulationPoisson> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q, eps0, SPACE_TIME) );
    
    MeshTopologyPtr spatialMeshTopo;
    if (useGradedSpatialMesh)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(usePeriodicBCs, std::invalid_argument, "periodic BCs not supported for geometric graded meshes");
      spatialMeshTopo = MeshFactory::geometricGradedIntervalMeshTopology(x0, x0+xWidth, xElements);
    }
    else
    {
      spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
    }
    
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
    
    Teuchos::RCP<VlasovSpaceTimeFormulation> form;
    if (!useMimeticDPG)
    {
      form = Teuchos::rcp(new VlasovSpaceTimeFormulation(q, m, configForm, spatialMeshTopo,
                                                         temporalVertices,
                                                         velocitySpaceVertices_x,
                                                         velocitySpaceVertices_y,
                                                         velocitySpaceVertices_z,
                                                         H1Order, delta_k,
                                                         numUniformRefinements,
                                                         basisFamily,
                                                         0, false, enforcePositivity) );
    }
    else
    {
      form = Teuchos::rcp(new MimeticVlasovSpaceTimeFormulation(q, m, configForm, spatialMeshTopo,
                                                                temporalVertices,
                                                                velocitySpaceVertices_x,
                                                                velocitySpaceVertices_y,
                                                                velocitySpaceVertices_z,
                                                                H1Order, delta_k,
                                                                numUniformRefinements,
                                                                basisFamily) );
    }
    return form;
  }

  Teuchos::RCP<VlasovSpaceTimeFormulation> getFormulation1D1T2V(const int xElements,  const int tElements,
                                                                const int vxElements, const int vyElements,
                                                                const int H1Order, const bool useGradedSpatialMesh,
                                                                const bool usePeriodicBCs, double q,
                                                                const EBasisFamily &basisFamily, const bool useMimeticDPG,
                                                                const int delta_k = 2,
                                                                const bool enforcePositivity = false)
  {
    double eps0    = TEST_EPS0;
    double m       = TEST_M;
    double x0      = TEST_X0;
    double xWidth  = TEST_XWIDTH;
    double t0      = TEST_T0;
    double tWidth  = TEST_TWIDTH;
    double vx0     = TEST_V0;
    double vxWidth = TEST_VWIDTH;
    double vy0     = TEST_V0;
    double vyWidth = TEST_VWIDTH;
    
    const int numUniformRefinements = 0;
    
    Teuchos::RCP<ConfigurationSpaceFormulationPoisson> configForm = Teuchos::rcp(new ConfigurationSpaceFormulationPoisson(q, eps0, SPACE_TIME) );
    
    MeshTopologyPtr spatialMeshTopo;
    if (useGradedSpatialMesh)
    {
      spatialMeshTopo = MeshFactory::geometricGradedIntervalMeshTopology(x0, x0+xWidth, xElements);
    }
    else
    {
      spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0+xWidth, xElements, usePeriodicBCs);
    }
    
    std::vector<double> temporalVertices(tElements + 1);
    for (int i=0; i<=tElements; i++)
    {
      const double t = double(i) / double(tElements) * tWidth + t0;
      temporalVertices[i] = t;
    }
    
    std::vector<double> velocitySpaceVertices_x(vxElements+1);
    for (int i=0; i<=vxElements; i++)
    {
      const double vx = double(i) / double(vxElements) * vxWidth + vx0;
      velocitySpaceVertices_x[i] = vx;
    }
    
    std::vector<double> velocitySpaceVertices_y(vyElements+1);
    for (int i=0; i<=vyElements; i++)
    {
      const double vy = double(i) / double(vyElements) * vyWidth + vy0;
      velocitySpaceVertices_y[i] = vy;
    }
    
    std::vector<double> velocitySpaceVertices_z; // empty -- 1D2V
    
    Teuchos::RCP<VlasovSpaceTimeFormulation> form;
    if (!useMimeticDPG)
    {
      form = Teuchos::rcp(new VlasovSpaceTimeFormulation(q, m, configForm, spatialMeshTopo,
                                                         temporalVertices,
                                                         velocitySpaceVertices_x,
                                                         velocitySpaceVertices_y,
                                                         velocitySpaceVertices_z,
                                                         H1Order, delta_k,
                                                         numUniformRefinements,
                                                         basisFamily,
                                                         0, false, enforcePositivity) );
    }
    else
    {
      form = Teuchos::rcp(new MimeticVlasovSpaceTimeFormulation(q, m, configForm, spatialMeshTopo,
                                                                temporalVertices,
                                                                velocitySpaceVertices_x,
                                                                velocitySpaceVertices_y,
                                                                velocitySpaceVertices_z,
                                                                H1Order, delta_k,
                                                                numUniformRefinements,
                                                                basisFamily) );
    }
    return form;
  }

  class MeshDestructorObserver : public RefinementObserver
  {
    bool _meshDestructorCalled = false;
    
  public:
    MeshDestructorObserver(MeshPtr mesh)
    {
      auto weakThisPtr = Teuchos::rcp(this, false); // false: weak ptr
      mesh->registerObserver(weakThisPtr);
    }
    
    void meshDestructorCalled() override
    {
      _meshDestructorCalled = true;
    }
    
    bool meshDestructorHasBeenCalled()
    {
      return _meshDestructorCalled;
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
                            const bool useGradedSpatialMesh,
                            const bool overrideVlasovBCs,
                            const EBasisFamily &basisFamily,
                            const bool useMimeticDPG,
                            const double tol,
                            Teuchos::FancyOStream &out, bool &success,
                            const bool outputViz, const string vizPrefix)
{
  const bool usePeriodicBCs = false;
  
  auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);
  
  form->setUseGeometricMultigrid(true);
  form->setIterativeSolveTolerance(1e-15);
//    form->setIterativeSolveOutputLevel(10);
  form->setFixedPointMaxSubsteps(5);
  form->setFixedPointMinSubsteps(2);
  
  FunctionPtr vx = form->v_i(0);

  auto vlasovSoln = form->vlasovSolution();
  
  auto poissonForm = form->configurationSpaceFormulation();
  auto poissonSoln = form->configurationSpaceSolution();
  FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
  FunctionPtr f_soln = Function::solution(form->f(), vlasovSoln);
  
  vlasovSoln->rhs()->addTerm(vlasovRHSForExactSoln(form, f_exact, E_exact, q));
  
  form->setInitialAndInflowConditions(f_exact, f_exact);
  
  SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
  SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
  
  const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
  auto xLeftRightTag = form->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
  auto VE_traceVar = form->VE_configurationSpaceTrace();
  form->addConfigurationSpaceBC(xLeftRightTag, VE_traceVar, VE_exact);
  
  if (overrideVlasovBCs)
  {
    auto vlasovBC = BC::bc();
    VarPtr traceVar = form->t_n_hat();
    
    const int initialTimeTag = form->tagVlasovInitialTime();
    
    vlasovBC->addDirichlet(traceVar, initialTimeTag, -f_exact); // minus sign thanks to outward normal on temporal interface
    
    const int VX_COMPONENT = 0;
    
    const int minVelocityTag = form->tagVlasovMinVelocity(VX_COMPONENT);
    const int maxVelocityTag = form->tagVlasovMaxVelocity(VX_COMPONENT);
    
    vlasovBC->addDirichlet(traceVar, minVelocityTag, -f_exact * (q/TEST_M) * E_exact); // outward normal is negative
    vlasovBC->addDirichlet(traceVar, maxVelocityTag,  f_exact * (q/TEST_M) * E_exact); // outward normal is positive
    
    const int inflowTag = form->tagVlasovSpatialInflow();
    
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
  initialGuess_Vlasov[form->f()->ID()] = f_exact;
  initialGuess_Vlasov[form->t_n_hat()->ID()] = tn_0;
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
  
  // check that tn_0 can be represented on the mesh, at least to within tol
  FunctionPtr tn_soln = Function::solution(form->t_n_hat(), vlasovSoln, false);
  FunctionPtr tn_err = tn_0 - tn_soln;
  double tn_projection_err = tn_err->l2norm(vlasovSoln->mesh());
  double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
  out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;
  TEST_COMPARE(tn_projection_err, <=, tol * tn_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream vlasovVizPath;
    vlasovVizPath << vizPrefix << "_VlasovProjection";
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form->vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});
  }
  
  // now, project again, this time with a zero t_n
  //  {
  //    // DEBUGGING: do we move from correct solution if we start there?
  //    out << "NOTE: projecting exact tn_hat onto solution -- for iterative solver, this is cheating.\n";
  //    initialGuess_Vlasov[form->t_n_hat()->ID()] = tn_0;
  //  }
  initialGuess_Vlasov[form->t_n_hat()->ID()] = Function::zero();
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);

//  form->setExportViz(true); // DEBUGGING
//  form->configurationSpaceFormulation()->setExportViz(true); // DEBUGGING
  form->configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.
  poissonSoln->solve();

  out << "Checking E_soln after pure Poisson solve.\n";
  double E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
  double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);

  out << "Checking VE_soln after pure Poisson solve.\n";
  
  FunctionPtr VE_soln = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
  double VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
  double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(VE_err_l2, <=, tol * VE_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_PurePoisson";
    
    auto density = form->solutionMoments(0)->solutionMomentFunction(0);
    
    auto poissonViz = Teuchos::rcp(new HDF5Exporter(form->configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    poissonViz->exportSolution(form->configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact, density}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact", "density"});
  }

  form->solve();

//    if (outputViz)
//    {
//      std::ostringstream vizPath;
//      vizPath << vizPrefix << "_Vlasov";
//      auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vizPath.str(), "/tmp"));
//      vlasovViz->exportSolution(form->vlasovSolution());
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

  const int time_comp_zero_based = form->configurationSpaceMesh()->getDimension() - 1; // time is last dimension of Poisson mesh
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
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form->vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});
    
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_Poisson";
    
    auto poissonViz = Teuchos::rcp(new HDF5Exporter(form->configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    poissonViz->exportSolution(form->configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
  }
}

void patchTestVlasovPoisson(FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                            const double q,
                            const int H1Order, const int xElements, const int tElements, const int vElements,
                            const bool useGradedSpatialMesh,
                            const bool overrideVlasovBCs,
                            const EBasisFamily &basisFamily,
                            const bool useMimeticDPG,
                            const double tol,
                            Teuchos::FancyOStream &out, bool &success,
                            const bool outputViz=false, const string vizPrefix="")
{
  std::vector<std::vector<GlobalIndexType> > refinementSequence;
  patchTestVlasovPoisson(f_exact, E_exact, VE_exact,
                         q, H1Order, xElements, tElements, vElements, refinementSequence,
                         useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG,
                         tol, out, success,
                         outputViz, vizPrefix);
}

void patchTestVlasovPoisson1D2V(FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                                const double q,
                                const int H1Order, const int xElements, const int tElements,
                                const int vxElements, const int vyElements,
                                std::vector<std::vector<GlobalIndexType> > refinementSequence,
                                const bool useGradedSpatialMesh,
                                const bool overrideVlasovBCs,
                                const EBasisFamily &basisFamily,
                                const bool useMimeticDPG,
                                const double tol,
                                Teuchos::FancyOStream &out, bool &success,
                                const bool outputViz, const string vizPrefix)
{
  const bool usePeriodicBCs = false;
  
  auto form = getFormulation1D1T2V(xElements, tElements, vxElements, vyElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);
  
  form->setUseGeometricMultigrid(true);
  form->setIterativeSolveTolerance(1e-15);
//    form->setIterativeSolveOutputLevel(10);
  form->setFixedPointMaxSubsteps(5);
  form->setFixedPointMinSubsteps(2);
  
  FunctionPtr vx = form->v_i(0);
  
  auto vlasovSoln = form->vlasovSolution();
  
  auto poissonForm = form->configurationSpaceFormulation();
  auto poissonSoln = form->configurationSpaceSolution();
  FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
  FunctionPtr f_soln = Function::solution(form->f(), vlasovSoln);
  
  vlasovSoln->rhs()->addTerm(vlasovRHSForExactSoln(form, f_exact, E_exact, q));
  
  form->setInitialAndInflowConditions(f_exact, f_exact);
  
  SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
  SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
  
  const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
  auto xLeftRightTag = form->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
  auto VE_traceVar = form->VE_configurationSpaceTrace();
  form->addConfigurationSpaceBC(xLeftRightTag, VE_traceVar, VE_exact);
  
  if (overrideVlasovBCs)
  {
    auto vlasovBC = BC::bc();
    VarPtr traceVar = form->t_n_hat();
    
    const int initialTimeTag = form->tagVlasovInitialTime();
    
    vlasovBC->addDirichlet(traceVar, initialTimeTag, -f_exact); // minus sign thanks to outward normal on temporal interface
    
    const int VX_COMPONENT = 0, VY_COMPONENT = 1;
    
    const int minxVelocityTag = form->tagVlasovMinVelocity(VX_COMPONENT);
    const int maxxVelocityTag = form->tagVlasovMaxVelocity(VX_COMPONENT);
    
    const int minyVelocityTag = form->tagVlasovMinVelocity(VY_COMPONENT);
    const int maxyVelocityTag = form->tagVlasovMaxVelocity(VY_COMPONENT);
    
    vlasovBC->addDirichlet(traceVar, minxVelocityTag, -f_exact * (q/TEST_M) * E_exact); // outward normal is negative
    vlasovBC->addDirichlet(traceVar, maxxVelocityTag,  f_exact * (q/TEST_M) * E_exact); // outward normal is positive
    
    auto zero = Function::zero();
    vlasovBC->addDirichlet(traceVar, minyVelocityTag, zero); // outward normal is negative
    vlasovBC->addDirichlet(traceVar, maxyVelocityTag, zero); // outward normal is positive
    
    const int inflowTag = form->tagVlasovSpatialInflow();
    
    FunctionPtr n = Function::normal();
    
    FunctionPtr traceValue = f_exact * vx * n->x(); // 1D inflow -> only see the x normal part; n_y is 0 everywhere on the mesh.
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
  FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn; // n->y() means n_t
  
  // to do the "pure" Poisson solve, we do need to have an f solution defined, since that enters the RHS
  std::map<int, FunctionPtr> initialGuess_Vlasov; // project the exact field solution onto the mesh
  initialGuess_Vlasov[form->f()->ID()] = f_exact;
  initialGuess_Vlasov[form->t_n_hat()->ID()] = tn_0;
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
  
  // check that tn_0 can be represented on the mesh, at least to within tol
  FunctionPtr tn_soln = Function::solution(form->t_n_hat(), vlasovSoln, false);
  FunctionPtr tn_err = tn_0 - tn_soln;
  double tn_projection_err = tn_err->l2norm(vlasovSoln->mesh());
  double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
  out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;
  TEST_COMPARE(tn_projection_err, <=, tol * tn_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream vlasovVizPath;
    vlasovVizPath << vizPrefix << "_VlasovProjection";
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form->vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});
  }
  
  // now, project again, this time with a zero t_n
  //  {
  //    // DEBUGGING: do we move from correct solution if we start there?
  //    out << "NOTE: projecting exact tn_hat onto solution -- for iterative solver, this is cheating.\n";
  //    initialGuess_Vlasov[form->t_n_hat()->ID()] = tn_0;
  //  }
  initialGuess_Vlasov[form->t_n_hat()->ID()] = Function::zero();
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);

//  form->setExportViz(true); // DEBUGGING
//  form->configurationSpaceFormulation()->setExportViz(true); // DEBUGGING
  form->configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.

  out << "Checking E_soln after pure Poisson solve.\n";
  double E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
  double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);
  
  out << "Checking VE_soln after pure Poisson solve.\n";
  
  FunctionPtr VE_soln = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
  double VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
  double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(VE_err_l2, <=, tol * VE_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_PurePoisson";
    
    auto density = form->solutionMoments(0)->solutionMomentFunction(0);
    
    auto poissonViz = Teuchos::rcp(new HDF5Exporter(form->configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    poissonViz->exportSolution(form->configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact, density}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact", "density"});
  }

  form->solve();

//    if (outputViz)
//    {
//      std::ostringstream vizPath;
//      vizPath << vizPrefix << "_Vlasov";
//      auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vizPath.str(), "/tmp"));
//      vlasovViz->exportSolution(form->vlasovSolution());
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

  const int time_comp_zero_based = form->configurationSpaceMesh()->getDimension() - 1; // time is last dimension of Poisson mesh
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
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form->vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});
    
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_Poisson";
    
    auto poissonViz = Teuchos::rcp(new HDF5Exporter(form->configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    poissonViz->exportSolution(form->configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
  }
}

void patchTestVlasovPoisson1D2V(FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                                const double q,
                                const int H1Order, const int xElements, const int tElements,
                                const int vxElements, const int vyElements,
                                const bool useGradedSpatialMesh,
                                const bool overrideVlasovBCs,
                                const EBasisFamily &basisFamily,
                                const bool useMimeticDPG,
                                const double tol,
                                Teuchos::FancyOStream &out, bool &success,
                                const bool outputViz=false, const string vizPrefix="")
{
  std::vector<std::vector<GlobalIndexType> > refinementSequence;
  patchTestVlasovPoisson1D2V(f_exact, E_exact, VE_exact,
                             q, H1Order, xElements, tElements, vxElements, vyElements, refinementSequence,
                             useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG,
                             tol, out, success,
                             outputViz, vizPrefix);
}

void patchTestTimeSlab(const bool refineOnSecondSlab,
                       FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                       const double q,
                       const int H1Order, const int xElements, const int tElements, const int vElements,
                       const bool useGradedSpatialMesh,
                       const bool overrideVlasovBCs,
                       const EBasisFamily &basisFamily,
                       const bool useMimeticDPG,
                       const double tol,
                       Teuchos::FancyOStream &out, bool &success,
                       const bool outputViz=false, const string vizPrefix="")
{
  const bool usePeriodicBCs = false;
  
  auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);
  
  form->setUseGeometricMultigrid(true);
  form->setIterativeSolveTolerance(1e-15);
//    form->setIterativeSolveOutputLevel(10);
  form->setFixedPointMaxSubsteps(5);
  form->setFixedPointMinSubsteps(2);
  
  FunctionPtr vx = form->v_i(0);

  auto vlasovSoln = form->vlasovSolution();
  
  auto poissonForm = form->configurationSpaceFormulation();
  auto poissonSoln = form->configurationSpaceSolution();
  FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
  FunctionPtr f_soln = Function::solution(form->f(), vlasovSoln);
  
  vlasovSoln->rhs()->addTerm(vlasovRHSForExactSoln(form, f_exact, E_exact, q));
  
  form->setInitialAndInflowConditions(f_exact, f_exact);
  
  SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
  SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
  
  const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
  auto xLeftRightTag = form->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
  auto VE_traceVar = form->VE_configurationSpaceTrace();
  form->addConfigurationSpaceBC(xLeftRightTag, VE_traceVar, VE_exact);
  
  if (overrideVlasovBCs)
  {
    auto vlasovBC = BC::bc();
    VarPtr traceVar = form->t_n_hat();
    
    const int initialTimeTag = form->tagVlasovInitialTime();
    
    vlasovBC->addDirichlet(traceVar, initialTimeTag, -f_exact); // minus sign thanks to outward normal on temporal interface
    
    const int VX_COMPONENT = 0;
    
    const int minVelocityTag = form->tagVlasovMinVelocity(VX_COMPONENT);
    const int maxVelocityTag = form->tagVlasovMaxVelocity(VX_COMPONENT);
    
    vlasovBC->addDirichlet(traceVar, minVelocityTag, -f_exact * (q/TEST_M) * E_exact); // outward normal is negative
    vlasovBC->addDirichlet(traceVar, maxVelocityTag,  f_exact * (q/TEST_M) * E_exact); // outward normal is positive
    
    const int inflowTag = form->tagVlasovSpatialInflow();
    
    FunctionPtr n = Function::normal();
    
    FunctionPtr traceValue = f_exact * vx * n->x();
    vlasovBC->addDirichlet(traceVar, inflowTag,  traceValue);
    vlasovSoln->setBC(vlasovBC);
  }
  
  auto cellTopo = CellTopology::cellTopology(CellTopology::line(), 2);
  auto vlasovMesh  = vlasovSoln->mesh();
  auto poissonMesh = poissonSoln->mesh();
  
  const double m    = TEST_M;
  
  FunctionPtr sgn = Function::sideParity();
  FunctionPtr n   = Function::normal();
  FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn;
  
  // to do the "pure" Poisson solve, we do need to have an f solution defined, since that enters the RHS
  std::map<int, FunctionPtr> initialGuess_Vlasov; // project the exact field solution onto the mesh
  initialGuess_Vlasov[form->f()->ID()] = f_exact;
  initialGuess_Vlasov[form->t_n_hat()->ID()] = tn_0;
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
  
  // check that tn_0 can be represented on the mesh, at least to within tol
  FunctionPtr tn_soln = Function::solution(form->t_n_hat(), vlasovSoln, false);
  FunctionPtr tn_err = tn_0 - tn_soln;
  double tn_projection_err = tn_err->l2norm(vlasovSoln->mesh());
  double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
  out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;
  TEST_COMPARE(tn_projection_err, <=, tol * tn_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream vlasovVizPath;
    vlasovVizPath << vizPrefix << "_VlasovProjection";
    auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovVizPath.str(), "/tmp"));
    vlasovViz->exportSolution(form->vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});
  }
  
  // now, project again, this time with a zero t_n
  initialGuess_Vlasov[form->t_n_hat()->ID()] = Function::zero();
  vlasovSoln->projectOntoMesh(initialGuess_Vlasov);

  form->configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.
  poissonSoln->solve();

  out << "Checking E_soln after pure Poisson solve.\n";
  double E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
  double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);

  out << "Checking VE_soln after pure Poisson solve.\n";
  
  FunctionPtr VE_soln = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
  double VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
  double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

  TEST_COMPARE(VE_err_l2, <=, tol * VE_soln_magnitude);
  
  if (outputViz)
  {
    std::ostringstream poissonVizPath;
    poissonVizPath << vizPrefix << "_PurePoisson";
    
    auto poissonViz = Teuchos::rcp(new HDF5Exporter(form->configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    poissonViz->exportSolution(form->configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
  }

  form->solve();

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

  const int time_comp_zero_based = form->configurationSpaceMesh()->getDimension() - 1; // time is last dimension of Poisson mesh
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
  
  auto nextSlab = form->nextTimeSlab(TEST_TWIDTH, tElements);
  form = nextSlab;
  vlasovSoln  = form->vlasovSolution();
  poissonForm = form->configurationSpaceFormulation();
  poissonSoln = form->configurationSpaceSolution();
  tn_soln = Function::solution(form->t_n_hat(), vlasovSoln, false);
  tn_err = tn_0 - tn_soln;
  vlasovSoln->rhs()->addTerm(vlasovRHSForExactSoln(form, f_exact, E_exact, q));
  
  if (refineOnSecondSlab)
  {
    // find elements at the temporal interface:
    auto initialTimeTag = form->tagVlasovInitialTime();
    auto meshTopo = Teuchos::rcp(form->vlasovMesh()->getTopology()->baseMeshTopology(), false);
    auto entitySets = meshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, initialTimeTag);
    std::set<IndexType> allCells = form->vlasovMesh()->getActiveCellIDsGlobal();
    
    std::set<IndexType> cellsToRefine;
    for (auto entitySet : entitySets)
    {
      std::set<IndexType> matchingCells = entitySet->cellIDsThatMatch(meshTopo, allCells);
      cellsToRefine.insert(matchingCells.begin(),matchingCells.end());
    }
    
    out << "Refining cells at initial time interface on second slab.\n";
    form->vlasovMesh()->hRefine(cellsToRefine);
  }
  
  out << "Solving on second slab.\n";
  form->solve();
  
  initialTimeFilter = SpatialFilter::matchingComponent(time_comp_zero_based, TEST_T0 + TEST_TWIDTH);

  out << "Comparing tn_0 with tn_soln on second time slab.\n";
  tn_soln_magnitude =   tn_0->l2norm(vlasovSoln->mesh());
  tn_err_overall    = tn_err->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(tn_err_overall, <=, tol * tn_soln_magnitude);

  out << "Comparing initial-time BC with solution (checking initial time imposition) on second slab.\n";
  initialTime_tn_err = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction(tn_err, initialTimeFilter);
  timeBC_err = initialTime_tn_err->l2norm(vlasovSoln->mesh());
  initialTime_tn_soln = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction(tn_soln, initialTimeFilter);
  initialTime_tn_magnitude = initialTime_tn_soln->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(timeBC_err, <=, tol * initialTime_tn_magnitude);
  
  E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
  f_soln = Function::solution(form->f(), vlasovSoln);
  
  out << "Comparing f0 with f_soln after Vlasov-Poisson solve on second time slab.\n";
  f_soln_magnitude = f_soln->l2norm(vlasovSoln->mesh());
  f_err_l2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
  TEST_COMPARE(f_err_l2, <=, tol * f_soln_magnitude);
  
  out << "Checking E_soln after Vlasov-Poisson solve on second time slab.\n";
  E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
  E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());
  
  TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);
}

void patchTestRefinedTimeSlab(FunctionPtr f_exact, FunctionPtr E_exact, FunctionPtr VE_exact,
                              const double q,
                              const int H1Order, const int xElements, const int tElements, const int vElements,
                              const bool overrideVlasovBCs,
                              const EBasisFamily &basisFamily,
                              const bool useMimeticDPG,
                              const double tol,
                              Teuchos::FancyOStream &out, bool &success,
                              const bool outputViz=false, const string vizPrefix="")
{
  bool refineOnSecondSlab = true;
  const bool useGradedSpatialMesh = false;
  patchTestTimeSlab(refineOnSecondSlab, f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
}


  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, ConservationDiagnostics )
  {
    FunctionPtr  f_exact;
    {
      FunctionPtr t = MonomialFunction::xi_n(1, 1); // first argument is the coordinate dimension (corresponds to time); second is the power.
      f_exact = t;
    }
    
    const double q = 1.;
    
    const int xElements = 2;
    const int tElements = 1;
    const int vElements = 4;
    const int H1Order   = 3;
    
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG       = false;
    
    const bool usePeriodicBCs = true;
    const bool useGradedSpatialMesh = false;
    
    auto initialSlab = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);
   
    auto initialVlasovSoln = initialSlab->vlasovSolution();
    
    map<int, FunctionPtr> projectionMap;
    projectionMap[initialSlab->f()->ID()] = f_exact;
    
    initialSlab->setMaxMomentForConservationDiagnostics(2);
    
    int numTemporalPointsPerSlab = 11;
    double dt = TEST_TWIDTH / (numTemporalPointsPerSlab - 1);
    
    int numTimeSlabs = 3;
    vector<pair<double,double>> expectedMassSeries(numTemporalPointsPerSlab);
    vector<pair<double,double>> expectedMomentumSeries(numTemporalPointsPerSlab);
    vector<pair<double,double>> expectedEnergySeries(numTemporalPointsPerSlab);
    double meshMeasure         = TEST_VWIDTH * TEST_XWIDTH; // measure in dimensions excluding time
    double vMin                = TEST_V0;
    double vMax                = TEST_V0 + TEST_VWIDTH;
    double meshMomentumMeasure = TEST_XWIDTH * (vMax * vMax - vMin * vMin) / 2.0; // integral of v over x,v dimensions
    double meshEnergyMeasure   = TEST_XWIDTH * (vMax * vMax * vMax - vMin * vMin * vMin) / 3.0; // integral of v^2 over x,v dimensions
    
    double slabStart = TEST_T0;
    auto currentSlab = initialSlab;
    for (int slabNumber=0; slabNumber<numTimeSlabs; slabNumber++)
    {
      auto vlasovSoln = currentSlab->vlasovSolution();
      vlasovSoln->projectOntoMesh(projectionMap);
      currentSlab->computeConservationDiagnostics(true); // true: do update solution moments; they haven't been computed yet
      
      double t = slabStart;
      for (int i=0; i<numTemporalPointsPerSlab; i++)
      {
        double massIntegralValue = meshMeasure * t;
        expectedMassSeries[i] = {t,massIntegralValue};
        double momentumIntegralValue = meshMomentumMeasure * t;
        expectedMomentumSeries[i] = {t,momentumIntegralValue};
        double energyIntegralValue = meshEnergyMeasure * t;
        expectedEnergySeries[i] = {t, energyIntegralValue};
        t += dt;
      }
      auto actualMassSeries = currentSlab->totalMassTimeSeries(slabStart, dt, numTemporalPointsPerSlab);
      TEST_EQUALITY(expectedMassSeries.size(), actualMassSeries.size());
      auto actualMomentumSeries = currentSlab->totalXMomentumTimeSeries(slabStart, dt, numTemporalPointsPerSlab);
      TEST_EQUALITY(expectedMomentumSeries.size(), actualMomentumSeries.size());
      auto actualEnergySeries = currentSlab->totalEnergyTimeSeries(slabStart, dt, numTemporalPointsPerSlab);
      
      auto compareSeries = [&] (const vector<pair<double,double>> &expected,
                                const vector<pair<double,double>> &actual) -> void
      {
        const double absTol = 1e-11;
        const double relTol = 1e-13;
        for (int i=0; i<numTemporalPointsPerSlab; i++)
        {
          const double t_expected = expected[i].first;
          const double t_actual   = actual[i].first;
          const double integral_expected = expected[i].second;
          const double integral_actual   = actual[i].second;
          
          TEST_FLOATING_EQUALITY(t_expected, t_actual, relTol);
          if (integral_expected < absTol)
          {
            TEST_COMPARE(std::abs(integral_expected-integral_actual), <, absTol);
          }
          else
          {
            TEST_FLOATING_EQUALITY(integral_expected, integral_actual, relTol);
          }
        }
      };
      
      compareSeries(expectedMassSeries,     actualMassSeries);
      compareSeries(expectedMomentumSeries, actualMomentumSeries);
      compareSeries(expectedEnergySeries,   actualEnergySeries);
      
      currentSlab = currentSlab->nextTimeSlab(TEST_TWIDTH, tElements);
      slabStart += TEST_TWIDTH;
    }
  }

  void testNextTimeSlab(const bool useMimeticDPG, Teuchos::FancyOStream &out, bool &success,
                        const bool enforcePositivity = false)
  {
    // sets up a formulation, but does not solve.  Checks that the new slab is created, has the right number
    // of elements, and that it copies settings (intended to include all the settings we want to be be copied).
    
    const double q = 1.;
    
    const int xElements = 1;
    const int tElements = 1;
    const int vElements = 2;
    const int H1Order   = 2;
    
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    
    FunctionPtr  f_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();
    FunctionPtr  E_exact = Function::zero();
        
    const bool usePeriodicBCs = true;
    const bool useGradedSpatialMesh = false;
    
    const int delta_k = 2;
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG,
                                      delta_k, enforcePositivity);
   
    TEST_EQUALITY(enforcePositivity, form->enforcePositivity());
    form->setInitialAndInflowConditions(f_exact, f_exact);
    if (enforcePositivity)
    {
      form->setProximalGalerkinParameters(2.0, 1.0);
    }
    if (enforcePositivity)
    {
      form->setProximalGalerkinParameters(2.0, 1.0);
    }
    
    bool useGeometricMultigrid = true;
    double cgTol = 1e-6 / 100.0;
    double fixedPointTol = 3e-8;
    int maxSubsteps = 3;
    int minSubsteps = 1;
    
    std::vector<double> cgTols {1,2,3};
    
    bool enforceQuasiNeutrality = true;
    int azOutputLevel = 5;
    int maxIters = 1000;
    bool exportViz = true;
    std::string vizDir = "/some/random/dir";
    
    form->setUseGeometricMultigrid(useGeometricMultigrid);
    form->setIterativeSolveTolerance(cgTol);
    form->setIterativeSolveTolerances(cgTols);
    form->setFixedPointMaxSubsteps(maxSubsteps);
    form->setFixedPointMinSubsteps(minSubsteps);
    form->setFixedPointTolerance(fixedPointTol);
    form->setEnforceQuasiNeutrality(enforceQuasiNeutrality);
    form->setIterativeSolveOutputLevel(azOutputLevel);
    form->setIterativeSolveMaxIterations(maxIters);
    form->setExportViz(exportViz, vizDir);
    
    auto trialVars = form->vlasovBF()->varFactory()->trialVars();
    auto testVars  = form->vlasovBF()->varFactory()->testVars();
    
    // to avoid console warnings about cell data not being present, project onto the first slab's solution to set
    // the tn values at the temporal outflow.
    auto tn = form->t_n_hat();
    std::map<int,FunctionPtr> tn_map;
    tn_map[tn->ID()] = Function::zero();
    form->vlasovSolution()->projectOntoMesh(tn_map);
    
    auto nextSlab = form->nextTimeSlab(TEST_TWIDTH, tElements);
    
    int numElementsExpected = xElements * tElements * vElements;
    int numElementsActual   = nextSlab->vlasovMesh()->numActiveElements();
    TEST_EQUALITY(numElementsExpected, numElementsActual);
    
    TEST_EQUALITY(useGeometricMultigrid,  nextSlab->useGeometricMultigrid());
    TEST_EQUALITY(cgTol,                  nextSlab->iterativeSolveTolerance());
    TEST_EQUALITY(maxSubsteps,            nextSlab->fixedPointMaxSubsteps());
    TEST_EQUALITY(minSubsteps,            nextSlab->fixedPointMinSubsteps());
    TEST_EQUALITY(fixedPointTol,          nextSlab->fixedPointTolerance());
    TEST_EQUALITY(enforceQuasiNeutrality, nextSlab->enforceQuasiNeutrality());
    TEST_EQUALITY(azOutputLevel,          nextSlab->iterativeSolveOutputLevel());
    TEST_EQUALITY(maxIters,               nextSlab->iterativeSolveMaxIterations());
    TEST_EQUALITY(exportViz,              nextSlab->exportViz());
    TEST_EQUALITY(vizDir,                 nextSlab->vizOutputDir());
    TEST_EQUALITY(enforcePositivity,      nextSlab->enforcePositivity());
    if (enforcePositivity)
    {
      TEST_ASSERT(nextSlab->psi() != Teuchos::null);
      TEST_ASSERT(nextSlab->phi() != Teuchos::null);
      TEST_ASSERT(nextSlab->proximalGalerkinAlpha() != Teuchos::null);
    }
    
    auto nextSlabCGTols = nextSlab->iterativeSolveTolerances();
    TEST_EQUALITY(cgTols.size(), nextSlabCGTols.size());
    if (cgTols.size() == nextSlabCGTols.size())
    {
      for (int i=0; i<int(cgTols.size()); i++)
      {
        TEST_EQUALITY(cgTols[i], nextSlabCGTols[i]);
      }
    }
    
    auto nextTrialVars = nextSlab->vlasovBF()->varFactory()->trialVars();
    auto nextTestVars  = nextSlab->vlasovBF()->varFactory()->testVars();
    
    TEST_EQUALITY(trialVars.size(), nextTrialVars.size());
    TEST_EQUALITY( testVars.size(), nextTestVars.size());
  }

  void testPoissonTwoStreamSteadyState(const double tolRelativeToProjectionError,
                                       const bool useMimeticDPG, Teuchos::FancyOStream &out, bool &success)
  {
    // like the two-stream instability problem, but no perturbation.  Should stay fixed.
    double n0 = 1.;
    double sigma = 1;
    double u0 = 3.0;
    
    // 1D1T1V
    const int firstVelocityDimension = 2; // zero-based
    const int numVelocityDimensions  = 1;
    FunctionPtr fM_u0        = MaxwellianFunction::maxwellian(firstVelocityDimension, numVelocityDimensions, { u0}, sigma * sigma, 0.5 * n0);
    FunctionPtr fM_minus_u0  = MaxwellianFunction::maxwellian(firstVelocityDimension, numVelocityDimensions, {-u0}, sigma * sigma, 0.5 * n0);
    FunctionPtr f0           = fM_u0 + fM_minus_u0;
    
    const double q = 1.;
    
    const int xElements = 2;
    const int tElements = 1;
    const int vElements = 8; // we need a somewhat fine mesh in v to get relatively good resolution
    const int H1Order   = 4;
    
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const double poissonTolRelaxation = 1.0 + tolRelativeToProjectionError; // if we're tolerating looseness in f, we'd better tolerate in poisson, too.
    const double absTolPoisson = 1e-4 * poissonTolRelaxation; // Poisson solutions should be 0, so we use an absolute tol for these
    
    FunctionPtr  f_exact = f0;
    FunctionPtr VE_exact = Function::zero();
    FunctionPtr  E_exact = Function::zero();
    
    const bool outputViz = false;
    const std::string vizPrefix = "TwoStreamTest";
    
    const bool usePeriodicBCs = true;
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);
    
    form->setUseGeometricMultigrid(true);
    form->setIterativeSolveTolerance(1e-8);
    form->setFixedPointMaxSubsteps(1);
    form->setFixedPointMinSubsteps(1);
    
    FunctionPtr vx = form->v_i(0);

    auto vlasovSoln = form->vlasovSolution();
    
    auto poissonForm = form->configurationSpaceFormulation();
    auto poissonSoln = form->configurationSpaceSolution();
    FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
    FunctionPtr f_soln = Function::solution(form->f(), vlasovSoln);
        
    form->setInitialAndInflowConditions(f_exact, f_exact);
    
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
    
    const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
    auto xLeftTag = form->createConfigurationMeshSideSetTag(xLeftFilter, restrictSideSetsToBoundary);
    auto VE_traceVar = form->VE_configurationSpaceTrace();
    form->addConfigurationSpaceBC(xLeftTag, VE_traceVar, VE_exact);
    
    const double m    = TEST_M;
    
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr n   = Function::normal();
    FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn;
    
    // to do the "pure" Poisson solve, we do need to have an f solution defined, since that enters the RHS
    std::map<int, FunctionPtr> initialGuess_Vlasov; // project the exact field solution onto the mesh
    initialGuess_Vlasov[form->f()->ID()] = f_exact;
    vlasovSoln->projectOntoMesh(initialGuess_Vlasov);

    double f_magnitude = f_exact->l2norm(vlasovSoln->mesh());
    double f_proj_err_l2 = (f_soln-f_exact)->l2norm(vlasovSoln->mesh());
    out << "f_proj_err_l2: " << f_proj_err_l2 / f_magnitude << std::endl;
    
    double relTol_f = f_proj_err_l2 / f_magnitude * (1. + tolRelativeToProjectionError);
    
//    form->setExportViz(true); // DEBUGGING
//    form->configurationSpaceFormulation()->setExportViz(true); // DEBUGGING
    form->configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.
    poissonSoln->solve();

    out << "Checking E_soln after pure Poisson solve.\n";
    double soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
    double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

    TEST_COMPARE(E_err_l2, <=, absTolPoisson);

    out << "Checking VE_soln after pure Poisson solve.\n";
    
    FunctionPtr VE_soln = Function::solution(form->configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
    soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
    double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

    TEST_COMPARE(VE_err_l2, <=, absTolPoisson);
    
    if (outputViz)
    {
      std::ostringstream poissonVizPath;
      poissonVizPath << vizPrefix << "_PurePoisson";

      auto poissonViz = Teuchos::rcp(new HDF5Exporter(form->configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
      poissonViz->exportSolution(form->configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
    }

    form->solve();

    if (outputViz)
    {
      std::ostringstream vizPath;
      vizPath << vizPrefix << "_Vlasov";
      auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vizPath.str(), "/tmp"));
      vlasovViz->exportSolution(form->vlasovSolution());
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
    
    FunctionPtr tn_soln = Function::solution(form->t_n_hat(), vlasovSoln, false);
    double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
    out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;

    const int time_comp_zero_based = form->configurationSpaceMesh()->getDimension() - 1; // time is last dimension of Poisson mesh
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
      auto vlasovViz = Teuchos::rcp(new HDF5Exporter(form->vlasovMesh(), vlasovVizPath.str(), "/tmp"));
      vlasovViz->exportSolution(form->vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_exact, f_exact, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "t_n err", "t_n exact"});

      std::ostringstream poissonVizPath;
      poissonVizPath << vizPrefix << "_Poisson";

      auto poissonViz = Teuchos::rcp(new HDF5Exporter(form->configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
      poissonViz->exportSolution(form->configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
    }
    
    out << "Continuing on to next time slab.\n";
    auto nextSlab = form->nextTimeSlab(TEST_TWIDTH, tElements);
    
    nextSlab->solve();
    
    FunctionPtr f_soln_nextSlab = Function::solution(nextSlab->f(), nextSlab->vlasovSolution());
    soln_magnitude = f_exact->l2norm(vlasovSoln->mesh());
    f_err_l2 = (f_soln_nextSlab-f_exact)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(f_err_l2, <=, relTol_f * soln_magnitude);
  }

  void testSumFactorizedGramAssembly(Teuchos::FancyOStream &out, bool &success)
  {
    const double q = 1.;
    
    const int xElements = 1;
    const int tElements = 1;
    const int vElements = 1;
    const int H1Order   = 3;
    const int delta_k   = 2;
    
    const bool usePeriodicBCs = true;
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG  = false;
    
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG, delta_k);
    
    auto sfIP = form->constructVlasovInnerProduct(true);
    auto ip   = form->constructVlasovInnerProduct(false);
    
    auto myCells = form->vlasovMesh()->cellIDsInPartition();
    
    const double absTol = 1e-12;
    const double relTol = 1e-14;
    for (auto cellID : myCells)
    {
      const bool testVsTest = true;
      auto basisCache   = BasisCache::basisCacheForCell(form->vlasovMesh(), cellID, testVsTest);
      auto testOrdering = form->vlasovMesh()->getElementType(cellID)->testOrderPtr;
      const int numCells    = 1;
      const int numTestDofs = testOrdering->totalDofs();
      FieldContainer<double> sfIPMatrix(numCells,numTestDofs,numTestDofs);
      FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
      
      sfIP->computeInnerProductMatrix(sfIPMatrix, testOrdering, basisCache);
      ip->computeInnerProductMatrix(    ipMatrix, testOrdering, basisCache);
      
      bool localSuccess = compareFloatingArraysCamellia(sfIPMatrix, "sum-factorized IP matrix", ipMatrix, "expected IP matrix", relTol, absTol, out);
      
      if (! localSuccess)
      {
        out << "IP comparison failed on cell " << cellID << std::endl;
        success = false;
        
        out << "sum-factorized-assembly matrix: " << sfIPMatrix << std::endl;
        out << "standard-assembly matrix: " << ipMatrix << std::endl;
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, NextTimeSlab )
  {
    bool useMimeticDPG = false;
    testNextTimeSlab(useMimeticDPG, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PositivityPreservationNextTimeSlab )
  {
    bool useMimeticDPG = false;
    bool enforcePositivity = true;
    testNextTimeSlab(useMimeticDPG, out, success, enforcePositivity);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PositivityPreservationAPI )
  {
    const int xElements = 1;
    const int tElements = 1;
    const int vElements = 2;
    const int H1Order   = 2;
    const int delta_k   = 1;
    
    const bool usePeriodicBCs = true;
    const bool useGradedSpatialMesh = false;
    const bool useMimeticDPG = false;
    const double q = 1.0;
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    
    auto defaultForm = getFormulation1D1T1V(xElements, tElements, vElements, H1Order,
                                            useGradedSpatialMesh, usePeriodicBCs, q,
                                            basisFamily, useMimeticDPG, delta_k);
    TEST_EQUALITY(false, defaultForm->enforcePositivity());
    
    const bool enforcePositivity = true;
    auto positiveForm = getFormulation1D1T1V(xElements, tElements, vElements, H1Order,
                                             useGradedSpatialMesh, usePeriodicBCs, q,
                                             basisFamily, useMimeticDPG, delta_k,
                                             enforcePositivity);
    TEST_EQUALITY(true, positiveForm->enforcePositivity());
    
    TEST_ASSERT(positiveForm->psi() != Teuchos::null);
    TEST_ASSERT(positiveForm->phi() != Teuchos::null);
    TEST_ASSERT(positiveForm->proximalGalerkinAlpha() != Teuchos::null);
    
    positiveForm->setProximalGalerkinParameters(2.0, 1.0);
    TEST_ASSERT(positiveForm->proximalGalerkinAlpha() != Teuchos::null);
    
    auto trialVars = positiveForm->vlasovBF()->varFactory()->trialVars();
    auto testVars  = positiveForm->vlasovBF()->varFactory()->testVars();
    TEST_ASSERT(trialVars.find(positiveForm->psi()->ID()) != trialVars.end());
    TEST_ASSERT(testVars.find(positiveForm->phi()->ID()) != testVars.end());
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PositivityPreservationTinySolve )
  {
    const int xElements = 1;
    const int tElements = 1;
    const int vElements = 2;
    const int H1Order   = 2;
    const int delta_k   = 1;
    
    const bool usePeriodicBCs = true;
    const bool useGradedSpatialMesh = false;
    const bool useMimeticDPG = false;
    const double q = 1.0;
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool enforcePositivity = true;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order,
                                      useGradedSpatialMesh, usePeriodicBCs, q,
                                      basisFamily, useMimeticDPG, delta_k,
                                      enforcePositivity);
    
    form->setFixedPointMinSubsteps(1);
    form->setFixedPointMaxSubsteps(1);
    form->setFixedPointTolerance(1.0);
    form->setProximalGalerkinParameters(1.0, 1.0);
    
    FunctionPtr f0 = Function::constant(1.0);
    form->setInitialAndInflowConditions(f0, f0);
    
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
    const bool restrictSideSetsToBoundary = !usePeriodicBCs; // for periodic BCs, the config mesh region will *not* be on a mesh boundary
    auto xLeftRightTag = form->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
    auto VE_traceVar = form->VE_configurationSpaceTrace();
    form->addConfigurationSpaceBC(xLeftRightTag, VE_traceVar, Function::zero());
    
    TEST_NOTHROW(form->solve());
    
    TEST_COMPARE(form->vlasovMesh()->getActiveCellIDsGlobal().size(), >, 0);
    auto myCells = form->vlasovMesh()->cellIDsInPartition();
    // In MPI runs, a rank may legitimately own no active Vlasov cells.  Such ranks simply
    // have no local limiter diagnostics to check.
    for (auto cellID : myCells)
    {
      const double theta = form->fluxLimitingWeightForCell(cellID, 0);
      const double expPsi = form->expPsiForCell(cellID, 0);
      const double thetaSpecies0Compatibility = form->fluxLimitingWeightForCell(cellID);
      const double expPsiSpecies0Compatibility = form->expPsiForCell(cellID);
      
      TEST_ASSERT(theta == theta);   // not NaN
      TEST_ASSERT(expPsi == expPsi); // not NaN
      TEST_COMPARE(theta, >=, 0.0);
      TEST_COMPARE(theta, <=, 1.0);
      TEST_COMPARE(expPsi, >, 0.0);
      TEST_EQUALITY(theta, thetaSpecies0Compatibility);
      TEST_EQUALITY(expPsi, expPsiSpecies0Compatibility);
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PositivityPreservationLimiterRegression )
  {
    const int xElements = 1;
    const int tElements = 1;
    const int vElements = 2;
    const int H1Order   = 2;
    const int delta_k   = 1;
    
    const bool usePeriodicBCs = true;
    const bool useGradedSpatialMesh = false;
    const bool useMimeticDPG = false;
    const double q = 1.0;
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool enforcePositivity = true;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order,
                                      useGradedSpatialMesh, usePeriodicBCs, q,
                                      basisFamily, useMimeticDPG, delta_k,
                                      enforcePositivity);
    
    FunctionPtr f0 = Function::constant(1.0);
    form->setInitialAndInflowConditions(f0, f0);
    
    std::map<int,FunctionPtr> projectionMap;
    projectionMap[form->f()->ID()]   = Function::constant(-0.5);
    projectionMap[form->psi()->ID()] = Function::zero(); // exp(psi) = 1
    form->vlasovSolution()->projectOntoMesh(projectionMap);
    
    TEST_NOTHROW(form->applyPositivityLimiter());
    
    TEST_COMPARE(form->vlasovMesh()->getActiveCellIDsGlobal().size(), >, 0);
    auto myCells = form->vlasovMesh()->cellIDsInPartition();
    // In MPI runs, a rank may legitimately own no active Vlasov cells.  Such ranks simply
    // have no local limiter diagnostics to check.
    for (auto cellID : myCells)
    {
      const double theta = form->fluxLimitingWeightForCell(cellID, 0);
      const double expPsi = form->expPsiForCell(cellID, 0);
      const double thetaSpecies0Compatibility = form->fluxLimitingWeightForCell(cellID);
      const double expPsiSpecies0Compatibility = form->expPsiForCell(cellID);
      
      TEST_ASSERT(theta == theta);   // not NaN
      TEST_ASSERT(expPsi == expPsi); // not NaN
      TEST_COMPARE(theta, >=, 0.0);
      TEST_COMPARE(theta, <, 1.0);
      TEST_COMPARE(expPsi, >, 0.0);
      TEST_EQUALITY(theta, thetaSpecies0Compatibility);
      TEST_EQUALITY(expPsi, expPsiSpecies0Compatibility);
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, LocalConservationDiagnostics )
  {
    const int xElements = 5;
    const int tElements = 1;
    const int vElements = 4;
    const int H1Order   = 3;
    
    const bool usePeriodicBCs = true;
    const double q = 1.0;
    const double m = 1.0;
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG  = false;
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);
    
    auto vx = form->v_i(0);
    // note that E_exact and f_exact are *not* compatible solutions; E is not a solution to Poisson loaded with the zeroth moment of f.
    // we are just exercising the local (mass) conservation computation right now.
    FunctionPtr E_exact = Function::constant(1.0);
    FunctionPtr f_exact = Function::constant(1.0);
    
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr n   = Function::normal();
    FunctionPtr tn_0 = f_exact * (vx * n->x() + n->y() + (q/m) * E_exact * n->z()) * sgn;
    
    {
      // project our synthetic (f,tn) solution onto the Vlasov mesh
      std::map<int, FunctionPtr> solnMap;
      solnMap[form->f()->ID()] = f_exact;
      solnMap[form->t_n_hat()->ID()] = tn_0;
      form->vlasovSolution()->projectOntoMesh(solnMap);
    }
    
    {
      // project E_exact onto Poisson mesh:
      std::map<int, FunctionPtr> solnMap;
      VarPtr E = form->configurationSpaceFormulation()->E_fieldVar(0);
      solnMap[E->ID()] = E_exact;
      form->configurationSpaceSolution()->projectOntoMesh(solnMap);
      
      // ensure that the Poisson solution is available locally for the residual computation below.
      form->configurationSpaceFormulation()->updateConfigurationSpaceFunctions();
    }
    
    form->computeConservationDiagnostics(false);
    
    auto & localMassConservationDiagnostic = form->localMassConservationDiagnostics(0);
    
    const double maxElementwiseFailure = localMassConservationDiagnostic.maxElementConservationFailure;
    const double netFailure            = std::abs(localMassConservationDiagnostic.netGlobalConservationFailure);
    
    // we expect the failures to be pretty small: exactly representable function with exact net flux of 0
    double tol = 1e-14;
    TEST_COMPARE(maxElementwiseFailure, <, tol);
    TEST_COMPARE(netFailure, <, tol);
    
    // relax the tolerance a few orders for the residual norm, which involves some higher-order functions
    tol = 1e-11;
    const double w1_residualNorm = form->conservationLawResidualNorm(0);
    TEST_COMPARE(w1_residualNorm, <, tol);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, MultiSlabSequenceAllowsInitialMeshToBeDestroyed )
  {
    // sets up a formulation like the PoissonTwoStreamSteadyState test below, and then creates two subsequent slabs,
    // nullifying the first.  Checks that the mesh belonging to the first has its destructor called.
    double n0 = 1.;
    double sigma = 1;
    double u0 = 3.0;
    
    // 1D1T1V
    const double q = 1.;
    
    const int xElements = 2;
    const int tElements = 1;
    const int vElements = 8; // we need a somewhat fine mesh in v to get relatively good resolution
    const int H1Order   = 4;
    
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG  = false;
    
    FunctionPtr  f_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();
    FunctionPtr  E_exact = Function::zero();
        
    const bool usePeriodicBCs = true;
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);
   
    form->setInitialAndInflowConditions(f_exact, f_exact);
    
    MeshDestructorObserver dtorObserver(form->vlasovMesh());
    
    // to avoid console warnings about cell data not being present, project onto the first slab's solution to set
    // the tn values at the temporal outflow.
    auto tn = form->t_n_hat();
    std::map<int,FunctionPtr> tn_map;
    tn_map[tn->ID()] = Function::zero();
    form->vlasovSolution()->projectOntoMesh(tn_map);
    
    TEST_EQUALITY(false, dtorObserver.meshDestructorHasBeenCalled());
    
    auto secondSlab = form->nextTimeSlab(TEST_TWIDTH, tElements);
    secondSlab->vlasovSolution()->projectOntoMesh(tn_map);
    auto thirdSlab  = secondSlab->nextTimeSlab(TEST_TWIDTH, tElements);
    
    // still should not have been destroyed; we still have a reference, through form:
    TEST_EQUALITY(false, dtorObserver.meshDestructorHasBeenCalled());
    
    form = Teuchos::null;
    
    // we used to have references one deep with temporal BCs referring to the immediately prior slab;
    // on the new (directly set) BC implementation, there are no back-references (a better design).
    // So on the new design, mesh dtor should have been called just now.
    TEST_EQUALITY(true, dtorObserver.meshDestructorHasBeenCalled());
    
    secondSlab = Teuchos::null;
    TEST_EQUALITY(true, dtorObserver.meshDestructorHasBeenCalled());
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonTwoStreamSteadyState )
  {
    const double tolRelativeToProjectionError = 0.1; // we allow 10% error relative to the best approximation for f, tn
    const bool useMimeticDPG = false;
    testPoissonTwoStreamSteadyState(tolRelativeToProjectionError, useMimeticDPG, out, success);
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonPatchTestTimeSlab_f_xv2_E_zero )
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    const double tol = 1e-12;
    
    bool refineSecondSlab = false;
    
    patchTestTimeSlab(refineSecondSlab, f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PoissonPatchTestRefinedTimeSlab_f_xv2_E_zero )
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const double tol = 1e-12;
    
    patchTestRefinedTimeSlab(f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, VE_exact, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, V_E_exact, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
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
    const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    const double tol = 1e-12;
    
    patchTestVlasovPoisson(f_exact, E_exact, V_E_exact, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success);
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success, outputViz, vizPrefix);
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
    const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol, out, success, outputViz, vizPrefix);
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    
    const bool outputViz = true;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic";
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
                           out, success, outputViz, testPrefix);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic_UseGradedSpatialMesh )
  {
    const double q      = TEST_Q;
    const double eps0   = TEST_EPS0;
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr t = Function::yn(1); // time is second dimension --> y

    const double vWidth = TEST_VWIDTH;
    FunctionPtr f_exact = t * x/vWidth - t * (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the phase-space domain
    
    FunctionPtr E_exact = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x) * t;
    FunctionPtr V_E = -(q/eps0) * (x * x * x / 6.0 - ((TEST_X0 + TEST_XWIDTH) / 4.0) * x * x) * t;

    const int xElements = 3;
    const int tElements = 1;
    const int vElements = 2; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const double tol = 1e-12;
    
    const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = true;
    
    const bool outputViz = true;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic_gradedSpatial";
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
                           out, success, outputViz, testPrefix);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic_UseGradedSpatialMesh_1D2V )
  {
    const double q      = TEST_Q;
    const double eps0   = TEST_EPS0;
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr t = Function::yn(1); // time is second dimension --> y
    const double vWidth2 = TEST_VWIDTH * TEST_VWIDTH; // vWidth squared because 2V
    FunctionPtr f_exact = t * x/vWidth2 - t * (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth2); // chosen to have zero integral on the phase-space domain
    
    FunctionPtr E_exact = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x) * t;
    FunctionPtr V_E = -(q/eps0) * (x * x * x / 6.0 - ((TEST_X0 + TEST_XWIDTH) / 4.0) * x * x) * t;

    const int xElements = 3;
    const int tElements = 1;
    const int vxElements = 2; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int vyElements = 2; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const double tol = 1e-12;
    
    const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = true;
    
    const bool outputViz = false;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic_gradedSpatial";
    patchTestVlasovPoisson1D2V(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vxElements, vyElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
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
    const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    
    const bool outputViz = true;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic";
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
                           out, success, outputViz, testPrefix);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic_Serendipity_UseGradedSpatialMesh )
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
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = true;
    
    const EBasisFamily basisFamily = BASIS_FAMILY_SERENDIPITY;
    
    const bool outputViz = true;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic";
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
                           out, success, outputViz, testPrefix);
  }


  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, PatchTest_f_linear_E_quadratic_refined )
  {
    const double q      = TEST_Q;
    const double eps0   = TEST_EPS0;
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr v = Function::zn(1);
    const double vWidth = TEST_VWIDTH;
//    const double C = (TEST_X0 + TEST_XWIDTH / 2.) * (TEST_V0 + TEST_VWIDTH / 2.); // + TEST_V0 * TEST_VWIDTH * TEST_VWIDTH + TEST_V0 * TEST_V0 * TEST_VWIDTH + TEST_VWIDTH * TEST_VWIDTH * TEST_VWIDTH / 3.;
//    FunctionPtr f_exact = x * v - C;
//
//    FunctionPtr E_exact = (q/eps0) * (x * x / 4. * (2 * TEST_V0 * TEST_VWIDTH + TEST_VWIDTH * TEST_VWIDTH) - C * TEST_VWIDTH * x);
//    FunctionPtr V_E = -(q/eps0) * (x * x * x / 12. * (2 * TEST_V0 * TEST_VWIDTH + TEST_VWIDTH * TEST_VWIDTH) - C * TEST_VWIDTH * x * x / 2.);
    
    FunctionPtr f_exact = x/vWidth + v - (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the phase-space domain
    
    FunctionPtr E_exact = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x);
    FunctionPtr V_E = -(q/eps0) * (x * x * x / 6.0 - ((TEST_X0 + TEST_XWIDTH) / 4.0) * x * x);

    const int xElements = 4;
    const int tElements = 1;
    const int vElements = 8; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 4; // quadratic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const double tol = 1e-11;
    
    const bool overrideVlasovBCs = true; // required because this patch test does not have zero t_n at the velocity extrema
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    
    const bool outputViz = true;
    const std::string testPrefix = "PatchTest_f_linear_E_quadratic_refined";
    
    std::vector<std::vector<GlobalIndexType>> refinementSequence;
    refinementSequence.push_back({16});
//    refinementSequence.push_back({16,17,18,19});
//    refinementSequence.push_back({32,33,34,35,40,41,42,43,48,49,50,51,56,57,58,59});
//    refinementSequence.push_back({64,65,66,67,72,73,74,75,80,81,82,83,88,89,90,91,96,97,98,99,104,105,106,107,112,113,114,115,120,121,122,123,128,129,130,131,136,137,138,139,144,145,146,147,152,153,154,155,160,161,162,163,168,169,170,171,176,177,178,179,184,185,186,187});
//    refinementSequence.push_back({224,225,226,227,232,233,234,235,240,241,242,243,248,249,250,251,256,257,258,259,264,265,266,267,272,273,274,275,280,281,282,283,288,289,290,291,296,297,298,299,304,305,306,307,312,313,314,315,320,321,322,323,328,329,330,331,336,337,338,339,344,345,346,347,352,353,354,355,360,361,362,363,368,369,370,371,376,377,378,379,384,385,386,387,392,393,394,395,400,401,402,403,408,409,410,411,416,417,418,419,424,425,426,427,432,433,434,435,440,441,442,443,448,449,450,451,456,457,458,459,464,465,466,467,472,473,474,475,480,481,482,483,488,489,490,491,496,497,498,499,504,505,506,507,512,513,514,515,520,521,522,523,528,529,530,531,536,537,538,539,544,545,546,547,552,553,554,555,560,561,562,563,568,569,570,571,576,577,578,579,584,585,586,587,592,593,594,595,600,601,602,603,608,609,610,611,616,617,618,619,624,625,626,627,632,633,634,635,640,641,642,643,648,649,650,651,656,657,658,659,664,665,666,667,672,673,674,675,680,681,682,683,688,689,690,691,696,697,698,699,704,705,706,707,712,713,714,715,720,721,722,723,728,729,730,731});
//    refinementSequence.push_back({880,882,928,930,944,946,1120,1122,1124,1126,1136,1138,1140,1142,1184,1186,1188,1190,1200,1202,1204,1206,2932,2934,2980,2982,2996});
//    refinementSequence.push_back({866,948,950,2912,2914,2928,3424,3426,3432,3434,3438,3440,3442,3444,3446,3448,3450,3452,3454,3456,3458,3460,3462,3464,3466,3468,3470,3472,3474,3476,3478,3480,3482,3484,3486,3488,3490,3496,3498,3504,3506,3508,3510,3512,3514,3516,3518,3520,3522,3528,3530,3536,3538,3540,3542,3544,3546,3548,3550,3552,3554,3560,3562,3568,3570,3572,3574,3576,3578,3580,3582,3584,3586,3592,3594,3598,3600,3602,3604,3606,3608,3610,3612,3614,3616,3618,3620,3622,3628,3630,3636,3638,3682,3684,3686,3688,3690,3692,3694});
//    refinementSequence.push_back({3428,3430,3436,3534,3556,3558,3564,3566,3588,3590,3596,3624,3680,3752,3754,3758,3760,3762,3768,3770,3776,3778,3780,3782,3788,3790,3796,3798,3804,3806,3812,3814,3820,3822,3828,3830,3832,3834,3840,3842,3844,3846,3848,3850,3852,3854,3856,3858,3862,3864,3866,3868,3870,3872,3874,3876,3878,3880,3882,3884,3886,3888,3890,3892,3894,3896,3898,3900,3902,3904,3906,3908,3910,3912,3914,3916,3918,3920,3922,3924,3926,3928,3930,3932,3934,3936,3938,3940,3942,3944,3946,3948,3950,3952,3954,3956,3958,3960,3962,3964,3966,3968,3970,3972,3974,3976,3978,3980,3982,3984,3986,3988,3990,3992,3994,3996,3998,4000,4002,4004,4006,4008,4010,4012,4014,4016,4018,4020,4022,4024,4026,4028,4030,4032,4034,4040,4042,4048,4050,4056,4058,4062,4064,4066,4068,4070,4072,4074,4076,4078,4080,4082,4084,4086,4088,4090,4092,4094,4096,4098,4100,4102,4104,4106,4108,4110,4112,4114,4116,4118,4120,4122,4124,4126,4128,4130,4132,4134,4136,4138,4140,4142,4144,4146,4148,4150,4152,4154,4156,4158,4160,4162,4164,4166,4168,4170,4172,4174,4176,4178,4180,4182,4184,4186,4188,4190,4192,4194,4196,4198,4200,4202,4204,4206,4208,4210,4212,4214,4216,4218,4220,4222,4224,4226,4228,4230,4232,4234,4236,4238,4240,4242,4244,4246,4248,4250,4252,4254,4256,4258,4260,4262,4264,4266,4268,4270,4272,4274,4276,4278,4280,4282,4284,4286,4288,4290,4292,4294,4296,4298,4300,4302,4304,4306,4308,4310,4312,4314,4316,4318,4320,4322,4324,4326,4328,4330,4332,4334,4336,4338,4340,4342,4344,4346,4348,4350,4352,4354,4356,4360,4362,4364,4366,4368,4370,4372,4374,4376,4378,4380,4382,4384,4386,4388,4390,4392,4394,4396,4398,4400,4402,4404,4406,4408,4410,4412,4414,4416,4418,4420,4422,4428,4430,4436,4438,4440,4442,4444,4446,4448,4450,4452,4454,4456,4458,4460,4462,4464,4466,4468,4470,4472,4474,4476,4478,4484,4486,4488,4490,4502,4506,4508,4512,4514,4516,4518,4520,4522,4524,4526,4528,4530,4532,4534,4536,4538,4540,4542,4584,4586,4628,4630});
    
    const bool useGradedSpatialMesh = false;
    
    patchTestVlasovPoisson(f_exact, E_exact, V_E, q, H1Order, xElements, tElements, vElements, refinementSequence, useGradedSpatialMesh, overrideVlasovBCs, basisFamily, useMimeticDPG, tol,
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
    const bool useMimeticDPG  = false;
    
    const double tol = 1e-13;
    const bool useGradedSpatialMesh = false;
    
    auto serendipityForm = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, BASIS_FAMILY_SERENDIPITY,  useMimeticDPG);
    auto fullTensorForm  = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, BASIS_FAMILY_HIERARCHICAL, useMimeticDPG);
    
    auto serendipityIP = serendipityForm->vlasovSolution()->ip();
    auto fullTensorIP  = fullTensorForm->vlasovSolution()->ip();
    
    auto serendipityBF = serendipityForm->vlasovSolution()->bf();
    auto fullTensorBF  = fullTensorForm->vlasovSolution()->bf();
    
    auto serendipityMesh = serendipityForm->vlasovMesh();
    auto fullTensorMesh  = fullTensorForm->vlasovMesh();
    
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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL;
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);

    auto vlasovSoln = form->vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int initialTimeTag = form->tagVlasovInitialTime();

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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL; // should not affect test in any way
    const bool useMimeticDPG     = false;
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);

    auto vlasovSoln = form->vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int inflowTag = form->tagVlasovSpatialInflow();

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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL; // should not affect test in any way
    const bool useMimeticDPG  = false;
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);

    auto vlasovSoln = form->vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int maxVelocityTag = form->tagVlasovMaxVelocity(0);

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
    const EBasisFamily basisFamily = BASIS_FAMILY_HIERARCHICAL; // should not affect test in any way
    const bool useMimeticDPG  = false;
    const bool useGradedSpatialMesh = false;
    
    auto form = getFormulation1D1T1V(xElements, tElements, vElements, H1Order, useGradedSpatialMesh, usePeriodicBCs, q, basisFamily, useMimeticDPG);

    auto vlasovSoln = form->vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int minVelocityTag = form->tagVlasovMinVelocity(0);

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

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, SumFactorizedGramAssembly )
  {
    testSumFactorizedGramAssembly(out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, RelativisticMaxwell1D2V_ZeroCurrentSymmetry )
  {
    const double q = TEST_Q;
    const double m = TEST_M;
    const double c = 2.0;
    const int xElements = 1;
    const int tElements = 1;
    const int vxElements = 1;
    const int vyElements = 1;
    const int H1Order = 2;
    const int delta_k = 1;
    const int numUniformRefinements = 0;
    const bool usePeriodicBCs = false;
    const bool useGradedSpatialMesh = false;

    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(TEST_X0, TEST_X0 + TEST_XWIDTH, xElements, usePeriodicBCs);
    std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(TEST_T0, TEST_TWIDTH, tElements);
    std::vector<double> velocitySpaceVertices_x = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vxElements);
    std::vector<double> velocitySpaceVertices_y = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vyElements);
    std::vector<double> velocitySpaceVertices_z;

    auto configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, TEST_EPS0));
    auto relForm = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, q, m,
                                                                           configForm,
                                                                           spatialMeshTopo,
                                                                           temporalVertices,
                                                                           velocitySpaceVertices_x,
                                                                           velocitySpaceVertices_y,
                                                                           velocitySpaceVertices_z,
                                                                           H1Order, delta_k,
                                                                           numUniformRefinements));

    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
    const bool restrictSideSetsToBoundary = true;
    auto xLeftRightTag = relForm->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
    relForm->addConfigurationSpaceBC(xLeftRightTag, relForm->VE_configurationSpaceTrace(), Function::zero());
    relForm->addConfigurationSpaceBC(xLeftRightTag, configForm->A_hat(1), Function::zero());
    relForm->addConfigurationSpaceBC(relForm->tagConfigurationInitialTime(), configForm->E_hat(0), Function::zero());

    FunctionPtr x = Function::xn(1);
    FunctionPtr t = Function::yn(1);
    FunctionPtr ux = MonomialFunction::xi_n(2, 1); // 1D1T2V => coords are x,t,ux,uy
    FunctionPtr uy = MonomialFunction::xi_n(3, 1);
    FunctionPtr f_exact = x * t * ux * uy;

    relForm->setInitialAndInflowConditions(f_exact, f_exact);

    std::map<int,FunctionPtr> projectionMap;
    projectionMap[relForm->f()->ID()] = f_exact;
    relForm->vlasovSolution()->projectOntoMesh(projectionMap);

    relForm->configurationSpaceSolve();

    auto configSoln = relForm->configurationSpaceSolution();
    auto configMesh = relForm->configurationSpaceMesh();
    FunctionPtr Ex = Function::solution(configForm->E_fieldVar(0), configSoln);
    FunctionPtr By = Function::solution(configForm->B_fieldVar(1), configSoln);
    FunctionPtr phi = Function::solution(configForm->VE_fieldVar(), configSoln);
    FunctionPtr A = Function::solution(configForm->A_fieldVar(1), configSoln);

    const double tol = 1e-10;
    TEST_COMPARE(Ex->l2norm(configMesh), <=, tol);
    TEST_COMPARE(By->l2norm(configMesh), <=, tol);
    TEST_COMPARE(phi->l2norm(configMesh), <=, tol);
    TEST_COMPARE(A->l2norm(configMesh), <=, tol);
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, RelativisticPositivityPreservationAPI )
  {
    const double q = TEST_Q;
    const double m = TEST_M;
    const double c = 2.0;
    const int xElements = 1;
    const int tElements = 1;
    const int vxElements = 1;
    const int vyElements = 1;
    const int H1Order = 2;
    const int delta_k = 1;
    const int numUniformRefinements = 0;
    const bool usePeriodicBCs = false;
    const bool enforcePositivity = true;

    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(TEST_X0, TEST_X0 + TEST_XWIDTH, xElements, usePeriodicBCs);
    std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(TEST_T0, TEST_TWIDTH, tElements);
    std::vector<double> velocitySpaceVertices_x = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vxElements);
    std::vector<double> velocitySpaceVertices_y = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vyElements);
    std::vector<double> velocitySpaceVertices_z;

    auto configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, TEST_EPS0));
    auto L2TestSpaceInnerProduct = RelativisticVlasovSpaceTimeFormulation::L2TestSpaceInnerProduct;
    auto relForm = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, q, m,
                                                                           configForm,
                                                                           spatialMeshTopo,
                                                                           temporalVertices,
                                                                           velocitySpaceVertices_x,
                                                                           velocitySpaceVertices_y,
                                                                           velocitySpaceVertices_z,
                                                                           H1Order, delta_k,
                                                                           numUniformRefinements,
                                                                           L2TestSpaceInnerProduct, 1.0, BASIS_FAMILY_HIERARCHICAL,
                                                                           0, enforcePositivity));
    
    TEST_EQUALITY(true, relForm->enforcePositivity());
    TEST_ASSERT(relForm->psi() != Teuchos::null);
    TEST_ASSERT(relForm->phi() != Teuchos::null);
    TEST_ASSERT(relForm->proximalGalerkinAlpha() != Teuchos::null);
    
    auto trialVars = relForm->vlasovBF()->varFactory()->trialVars();
    auto testVars  = relForm->vlasovBF()->varFactory()->testVars();
    TEST_ASSERT(trialVars.find(relForm->psi()->ID()) != trialVars.end());
    TEST_ASSERT(testVars.find(relForm->phi()->ID()) != testVars.end());
    
    // This is intentionally only an API/variable-construction smoke test.  nextTimeSlab()
    // transfers temporal-trace coefficients from a populated previous slab; calling it here before
    // a solve or trace initialization can fail for reasons unrelated to positivity-preservation setup.
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, RelativisticPositivityPreservationTinySolve )
  {
    const double q = TEST_Q;
    const double m = TEST_M;
    const double c = 2.0;
    const int xElements = 1;
    const int tElements = 1;
    const int vxElements = 1;
    const int vyElements = 1;
    const int H1Order = 2;
    const int delta_k = 1;
    const int numUniformRefinements = 0;
    const bool usePeriodicBCs = false;
    const bool enforcePositivity = true;

    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(TEST_X0, TEST_X0 + TEST_XWIDTH, xElements, usePeriodicBCs);
    std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(TEST_T0, TEST_TWIDTH, tElements);
    std::vector<double> velocitySpaceVertices_x = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vxElements);
    std::vector<double> velocitySpaceVertices_y = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vyElements);
    std::vector<double> velocitySpaceVertices_z;

    auto configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, TEST_EPS0));
    auto L2TestSpaceInnerProduct = RelativisticVlasovSpaceTimeFormulation::L2TestSpaceInnerProduct;
    auto relForm = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, q, m,
                                                                           configForm,
                                                                           spatialMeshTopo,
                                                                           temporalVertices,
                                                                           velocitySpaceVertices_x,
                                                                           velocitySpaceVertices_y,
                                                                           velocitySpaceVertices_z,
                                                                           H1Order, delta_k,
                                                                           numUniformRefinements,
                                                                           L2TestSpaceInnerProduct, 1.0, BASIS_FAMILY_HIERARCHICAL,
                                                                           0, enforcePositivity));
    
    relForm->setFixedPointMinSubsteps(1);
    relForm->setFixedPointMaxSubsteps(1);
    relForm->setFixedPointTolerance(1.0);
    relForm->setProximalGalerkinParameters(1.0, 1.0);
    
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
    const bool restrictSideSetsToBoundary = true;
    auto xLeftRightTag = relForm->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
    relForm->addConfigurationSpaceBC(xLeftRightTag, relForm->VE_configurationSpaceTrace(), Function::zero());
    relForm->addConfigurationSpaceBC(xLeftRightTag, configForm->A_hat(1), Function::zero());
    relForm->addConfigurationSpaceBC(relForm->tagConfigurationInitialTime(), configForm->E_hat(0), Function::zero());
    
    FunctionPtr f0 = Function::constant(1.0);
    relForm->setInitialAndInflowConditions(f0, f0);
    
    TEST_NOTHROW(relForm->solve());
    
    TEST_COMPARE(relForm->vlasovMesh()->getActiveCellIDsGlobal().size(), >, 0);
    auto myCells = relForm->vlasovMesh()->cellIDsInPartition();
    // In MPI runs, a rank may legitimately own no active Vlasov cells.  Such ranks simply
    // have no local limiter diagnostics to check.
    for (auto cellID : myCells)
    {
      const double theta = relForm->fluxLimitingWeightForCell(cellID, 0);
      const double expPsi = relForm->expPsiForCell(cellID, 0);
      const double thetaSpecies0Compatibility = relForm->fluxLimitingWeightForCell(cellID);
      const double expPsiSpecies0Compatibility = relForm->expPsiForCell(cellID);
      
      TEST_ASSERT(theta == theta);   // not NaN
      TEST_ASSERT(expPsi == expPsi); // not NaN
      TEST_COMPARE(theta, >=, 0.0);
      TEST_COMPARE(theta, <=, 1.0);
      TEST_COMPARE(expPsi, >, 0.0);
      TEST_EQUALITY(theta, thetaSpecies0Compatibility);
      TEST_EQUALITY(expPsi, expPsiSpecies0Compatibility);
    }
  }

  TEUCHOS_UNIT_TEST( VlasovSpaceTimeFormulation, RelativisticMaxwell1D2V_ZeroCurrentSymmetry_HangingNodeRefined )
  {
    const double q = TEST_Q;
    const double m = TEST_M;
    const double c = 2.0;
    const int xElements = 1;
    const int tElements = 1;
    const int vxElements = 1;
    const int vyElements = 1;
    const int H1Order = 2;
    const int delta_k = 1;
    const int numUniformRefinements = 0;
    const bool usePeriodicBCs = false;

    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(TEST_X0, TEST_X0 + TEST_XWIDTH, xElements, usePeriodicBCs);
    std::vector<double> temporalVertices = MeshTools::verticesFor1DIntervalMesh(TEST_T0, TEST_TWIDTH, tElements);
    std::vector<double> velocitySpaceVertices_x = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vxElements);
    std::vector<double> velocitySpaceVertices_y = MeshTools::verticesFor1DIntervalMesh(TEST_V0, TEST_VWIDTH, vyElements);
    std::vector<double> velocitySpaceVertices_z;

    auto configForm = Teuchos::rcp(new ConfigurationSpaceFormulationMaxwell1D2V(q, TEST_EPS0));
    auto relForm = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, q, m,
                                                                           configForm,
                                                                           spatialMeshTopo,
                                                                           temporalVertices,
                                                                           velocitySpaceVertices_x,
                                                                           velocitySpaceVertices_y,
                                                                           velocitySpaceVertices_z,
                                                                           H1Order, delta_k,
                                                                           numUniformRefinements));

    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
    const bool restrictSideSetsToBoundary = true;
    auto xLeftRightTag = relForm->createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter, restrictSideSetsToBoundary);
    relForm->addConfigurationSpaceBC(xLeftRightTag, relForm->VE_configurationSpaceTrace(), Function::zero());
    relForm->addConfigurationSpaceBC(xLeftRightTag, configForm->A_hat(1), Function::zero());
    relForm->addConfigurationSpaceBC(relForm->tagConfigurationInitialTime(), configForm->E_hat(0), Function::zero());

    FunctionPtr x = Function::xn(1);
    FunctionPtr t = Function::yn(1);
    FunctionPtr ux = MonomialFunction::xi_n(2, 1);
    FunctionPtr uy = MonomialFunction::xi_n(3, 1);
    FunctionPtr f_exact = x * t * ux * uy;

    relForm->setInitialAndInflowConditions(f_exact, f_exact);

    std::map<int,FunctionPtr> projectionMap;
    projectionMap[relForm->f()->ID()] = f_exact;
    relForm->vlasovSolution()->projectOntoMesh(projectionMap);

    std::set<GlobalIndexType> oneCell{*relForm->vlasovMesh()->getActiveCellIDsGlobal().begin()};
    relForm->vlasovMesh()->hRefine(oneCell);

    // Reproject after refinement so the refined active mesh still represents the intended analytic phase-space function.
    relForm->vlasovSolution()->projectOntoMesh(projectionMap);

    relForm->configurationSpaceSolve();

    auto configSoln = relForm->configurationSpaceSolution();
    auto configMesh = relForm->configurationSpaceMesh();
    FunctionPtr Ex = Function::solution(configForm->E_fieldVar(0), configSoln);
    FunctionPtr By = Function::solution(configForm->B_fieldVar(1), configSoln);
    FunctionPtr phi = Function::solution(configForm->VE_fieldVar(), configSoln);
    FunctionPtr A = Function::solution(configForm->A_fieldVar(1), configSoln);

    const double tol = 1e-10;
    TEST_COMPARE(Ex->l2norm(configMesh), <=, tol);
    TEST_COMPARE(By->l2norm(configMesh), <=, tol);
    TEST_COMPARE(phi->l2norm(configMesh), <=, tol);
    TEST_COMPARE(A->l2norm(configMesh), <=, tol);
  }

// include tests against MimeticVlasovSpaceTimeFormulation in this compilation block:
#include "MimeticVlasovSpaceTimeFormulationTests.hpp"
} // end anonymous namespace
