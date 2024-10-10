//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  VlasovBackwardEulerFormulationTests.cpp
//
//  Created by Nate Roberts on 3/4/2020.
//
//

#include "VlasovBackwardEulerFormulation.h"

#include "BF.h"
#include "ExpFunction.h"
#include "MaxwellianFunction.h"
#include "MonomialFunctions.h"
#include "PhaseSpaceVisualization.h"
#include "RHS.h"
#include "Solution.h"
#include "TimeRampFunction.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

namespace {
  const double TEST_X0     = 0.0;
  const double TEST_KX     = 0.2; // wavelength
  const double TEST_XWIDTH = 2 * M_PI / TEST_KX;
  const double TEST_V0     = -10.0;
  const double TEST_VWIDTH = 20.0;
  const double TEST_DT     = 0.1;

  const double TEST_Q      = 1.0;
  const double TEST_M      = 1.0;
  const double TEST_EPS0   = 1e-4;

  VlasovBackwardEulerFormulation getFormulation(const int xElements, const int vElements, const int H1Order,
                                                const bool usePeriodicBCs, double q = TEST_Q,
                                                bool homogeneousPotentialBCs = true, bool useCrankNicolson = false,
                                                const double x0 = TEST_X0, const double xWidth = TEST_XWIDTH,
                                                const double v0 = TEST_V0, const double vWidth = TEST_VWIDTH,
                                                const double dt = TEST_DT)
  {
    double eps0   = TEST_EPS0;
    double m      = TEST_M;
    
    const int delta_k = 2;
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    const bool useSerendipity = false;
    const int maxMomentNumber = 0;
    VlasovBackwardEulerFormulation form(q, m, useSerendipity, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k, maxMomentNumber,
                                        useCrankNicolson);
    form.setRunQuietly(true);
    auto poissonMesh = poissonConfig->configurationSpaceMesh();
    
    auto poissonBC = BC::bc();
    if (homogeneousPotentialBCs)
    {
      auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
      
      auto poissonMeshTopo = form.configurationSpaceFormulation()->configurationSpaceMeshTopology();
      
      // 1D mesh entity sets:
      // create an empty entity set.  In higher dimensions, this could be an edge or face; on refinement,
      // the entity set contains the descendants of the original.
      EntitySetPtr x0EntitySet = poissonMeshTopo->createEntitySet();
      EntitySetPtr x1EntitySet = poissonMeshTopo->createEntitySet();
      
      // vertex lookup.  There are quite a few ways to look up entities in the MeshTopology.
      IndexType x0Vertex;
      bool x0Found = poissonMeshTopo->getVertexIndex({x0}, x0Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
      
      IndexType x1Vertex;
      bool x1Found = poissonMeshTopo->getVertexIndex({x0 + xWidth}, x1Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x1Found, std::invalid_argument, "x1 not found");
      
      x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
      x1EntitySet->addEntity(0, x1Vertex); // add entity of dimension 0 with ID x1Vertex to the x1 entity set.
      
      static const int POISSON_TAG_X0 = 0; // left  (x=x0)
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, x0EntitySet);
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
      
      if (!usePeriodicBCs)
      {
        static const int POISSON_TAG_X1 = 1; // right (x=x0 + xWidth)
        poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X1, x1EntitySet);
        poissonBC->addDirichlet(VE_hat, POISSON_TAG_X1, Function::constant(0.0)); // pin the potential to 0 at x0+xWidth
      }
    }
    
    poissonConfig->solution()->setBC(poissonBC);
    
    return form;
  }

  void patchTestVlasovPoisson(std::vector<FunctionPtr> f, // f exact at first n time steps (0 being the initial state)
                              FunctionPtr f_inflow,
                              FunctionPtr E_exact, FunctionPtr VE_exact,
                              std::vector<FunctionPtr> rhs_extra_load, // difference between the LHS and RHS of the temporally discretized strong form, evaluated at the exact solution at time k+1 (on the left) and at time k (on the right), for k = 0 to n-1
                              const double q,
                              VlasovBackwardEulerFormulation &form,
                              const bool overrideVlasovBCs,
                              const bool useCrankNicolson,
                              const bool enforceInflowBCsWeakly,
                              const double tol,
                              Teuchos::FancyOStream &out, bool &success,
                              const bool outputViz=false, const string vizPrefix="")
  {
    form.setUseGeometricMultigrid(true);
    form.setIterativeSolveTolerance(1e-15);
  //    form.setIterativeSolveOutputLevel(10);
    form.setFixedPointMaxSubsteps(5);
    form.setFixedPointMinSubsteps(2);
    
    FunctionPtr vx = form.v_i(0);

    auto vlasovSoln = form.vlasovSolution();
    
    auto poissonForm = form.configurationSpaceFormulation();
    auto poissonSoln = form.configurationSpaceSolution();
    FunctionPtr E_soln = Function::solution(poissonForm->E_fieldVar(0), poissonSoln);
    FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
    
    SpatialFilterPtr xLeftFilter  = SpatialFilter::matchingX(TEST_X0);
    SpatialFilterPtr xRightFilter = SpatialFilter::matchingX(TEST_X0 + TEST_XWIDTH);
    
    auto xLeftRightTag = form.createConfigurationMeshSideSetTag(xLeftFilter | xRightFilter);
    auto VE_traceVar = form.VE_configurationSpaceTrace();
    form.addConfigurationSpaceBC(xLeftRightTag, VE_traceVar, VE_exact);
    
    // to do the "pure" Poisson solve, we do need to have an f solution defined, since that enters the RHS
    std::map<int, FunctionPtr> initialGuess_Vlasov; // project the exact field solution onto the mesh
    initialGuess_Vlasov[form.f()->ID()] = f[1];
//    initialGuess_Vlasov[form.t_n_hat()->ID()] = tn_0;
    vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
    
//    const double m    = TEST_M;
//
//    FunctionPtr sgn = Function::sideParity();
//    FunctionPtr n   = Function::normal();
//    FunctionPtr tn_0 = f_BC * (vx * n->x() + (q/m) * E_exact * n->y()) * sgn;
//    // check that tn_0 can be represented on the mesh, at least to within tol
//    FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
//    FunctionPtr tn_err = tn_0 - tn_soln;
//    double tn_projection_err = tn_err->l2norm(vlasovSoln->mesh());
//    double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());
//    out << "tn_soln_magnitude = " << tn_soln_magnitude << std::endl;
//    TEST_COMPARE(tn_projection_err, <=, tol * tn_soln_magnitude);
    
    form.configurationSpaceFormulation()->solve(true); // true: ensure that SolutionMomentFunction is properly set with the value that we just projected.
//    poissonSoln->solve();

    out << "Checking E_soln after pure Poisson solve.\n";
    double E_soln_magnitude = E_soln->l2norm(poissonSoln->mesh());
    double E_err_l2 = (E_soln-E_exact)->l2norm(poissonSoln->mesh());

    TEST_COMPARE(E_err_l2, <=, tol * E_soln_magnitude);

    out << "Checking VE_soln after pure Poisson solve.\n";
    
    FunctionPtr VE_soln = Function::solution(form.configurationSpaceFormulation()->VE_fieldVar(), poissonSoln);
    double VE_soln_magnitude = VE_soln->l2norm(poissonSoln->mesh());
    double VE_err_l2 = (VE_soln-VE_exact)->l2norm(poissonSoln->mesh());

    TEST_COMPARE(VE_err_l2, <=, tol * VE_soln_magnitude);
    
    // initialize viz outputs, if using
    Teuchos::RCP<HDF5Exporter> vlasovViz, poissonViz, purePoissonViz, vlasovPriorTimeViz;
    if (outputViz)
    {
      std::ostringstream purePoissonVizPath;
      purePoissonVizPath << vizPrefix << "_PurePoisson";
      
      purePoissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), purePoissonVizPath.str(), "/tmp"));
      
      std::ostringstream vlasovVizPath;
      vlasovVizPath << vizPrefix << "_Vlasov";
      vlasovViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovVizPath.str(), "/tmp"));
      
      std::ostringstream vlasovPriorTimeVizPath;
      vlasovPriorTimeVizPath << vizPrefix << "_VlasovPriorTime";
      vlasovPriorTimeViz = Teuchos::rcp(new HDF5Exporter(form.vlasovMesh(), vlasovPriorTimeVizPath.str(), "/tmp"));
      
      std::ostringstream poissonVizPath;
      poissonVizPath << vizPrefix << "_Poisson";
      
      poissonViz = Teuchos::rcp(new HDF5Exporter(form.configurationSpaceMesh(), poissonVizPath.str(), "/tmp"));
    }
    
    if (outputViz)
    {
      purePoissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"});
    }

    // reset Vlasov solution:
    initialGuess_Vlasov[form.f()->ID()]       = Function::zero();
    initialGuess_Vlasov[form.t_n_hat()->ID()] = Function::zero();
    vlasovSoln->projectOntoMesh(initialGuess_Vlasov);
        
    out << "Setting initial condition: f = " << f[0]->displayString() << std::endl;
    out << "Setting inflow  condition: f = " << f_inflow->displayString() << std::endl;
    const double timeRamp = 0;
    form.setInitialAndInflowConditions(f[0], f_inflow, timeRamp, enforceInflowBCsWeakly);
    
    if (outputViz)
    {
      vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f[0], f[0], f_inflow, Function::zero(), Function::zero()}, std::vector<string>{"f err", "f exact", "f inflow", "t_n err", "t_n exact"}, 0);
      
      poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"}, 0);
    }
    
    auto w = form.w(); // test function for Vlasov RHS
    for (int k=0; k<int(f.size()) - 1; k++) // k: previous time step number (k+1 is current time step number)
    {
      FunctionPtr f_k   = f[k];
      FunctionPtr f_kp1 = f[k+1];
      
      FunctionPtr rhs_extra_load_k = (k >= 1) ? rhs_extra_load[k] - rhs_extra_load[k-1] : rhs_extra_load[k];
      
//      std::cout << "rhs_extra_load[" << k << "]: " << rhs_extra_load_k->displayString() << std::endl;
      
      vlasovSoln->rhs()->addTerm(rhs_extra_load_k * w);
      
      FunctionPtr f_BC = f_kp1;
      if (enforceInflowBCsWeakly && useCrankNicolson)
      {
        // then for reasons I don't entirely understand, it appears the right thing for f_BC is this:
        f_BC = f_kp1 + f_k;
      }
      
      if (overrideVlasovBCs)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(enforceInflowBCsWeakly, std::invalid_argument, "When using weak BC enforcement, this test does not support overriding Vlasov BCs");
        
        auto vlasovBC = BC::bc();
        VarPtr traceVar = form.t_n_hat();
        
        const int VX_COMPONENT = 0;
        
        const int minVelocityTag = form.tagVlasovMinVelocity(VX_COMPONENT);
        const int maxVelocityTag = form.tagVlasovMaxVelocity(VX_COMPONENT);
        
        vlasovBC->addDirichlet(traceVar, minVelocityTag, -f_BC * (q/TEST_M) * E_exact); // outward normal is negative
        vlasovBC->addDirichlet(traceVar, maxVelocityTag,  f_BC * (q/TEST_M) * E_exact); // outward normal is positive
        
        const int inflowTag = form.tagVlasovSpatialInflow();
        
        FunctionPtr n = Function::normal();
        
        FunctionPtr traceValue = f_BC * vx * n->x();
        vlasovBC->addDirichlet(traceVar, inflowTag,  traceValue);
        vlasovSoln->setBC(vlasovBC);
      }
      
      form.takeTimeStep();
      
      out << "Comparing f[" << k+1 << "] with f_soln after Vlasov-Poisson solve.\n";
      double f_soln_magnitude = f_soln->l2norm(vlasovSoln->mesh());
      double f_err_l2 = (f_soln-f_kp1)->l2norm(vlasovSoln->mesh());
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

      const double m    = TEST_M;

      FunctionPtr sgn = Function::sideParity();
      FunctionPtr n   = Function::normal();
      FunctionPtr tn_0 = f_BC * (vx * n->x() + (q/m) * E_exact * n->y()) * sgn;
      
      if (enforceInflowBCsWeakly)
      {
        // then we expect t_n to be 0 (due to homogeneous BCs) at the inflow boundary
        
        FunctionPtr inflow_characteristic = Function::meshSideSetCharacteristic(DIRICHLET_SET_TAG_NAME, form.tagVlasovSpatialInflow());
        tn_0 = tn_0 - inflow_characteristic * tn_0;
      }
      out << "tn_0 = " << tn_0->displayString() << std::endl;
      
      // check that tn_0 can be represented on the mesh, at least to within tol
      FunctionPtr tn_soln = Function::solution(form.t_n_hat(), vlasovSoln, false);
      double tn_soln_magnitude = tn_soln->l2norm(vlasovSoln->mesh());

      FunctionPtr tn_err = tn_0 - tn_soln;
      double tn_err_overall = tn_err->l2norm(vlasovSoln->mesh());
      TEST_COMPARE(tn_err_overall, <=, tol * tn_soln_magnitude);

      TEST_ASSERT(tn_0->boundaryValueOnly() || tn_0->isZero()); // if zero, we allow it not to claim to be boundary-only.  (This is an artifact of the logic regarding Function products: zero times anything is zero, and we don't retain the boundary-valuedness in the zero result.  We could fix this, but it's not a high priority.)
      TEST_ASSERT(tn_soln->boundaryValueOnly());

      tn_soln_magnitude = tn_0->l2norm(vlasovSoln->mesh());
      double tn_err_l2 = (tn_soln-tn_0)->l2norm(vlasovSoln->mesh());
      TEST_COMPARE(tn_err_l2, <=, tol * tn_soln_magnitude);
      
      if (outputViz)
      {
        double dt = form.timeStep();
        vlasovViz->exportSolution(form.vlasovSolution(), std::vector<FunctionPtr> {f_soln-f_kp1, f_kp1, f_inflow, tn_soln-tn_0, tn_0}, std::vector<string>{"f err", "f exact", "f inflow", "t_n err", "t_n exact"}, (k+1) * dt);
        
        vlasovPriorTimeViz->exportSolution(form.vlasovSolutionPriorTime(), k * dt); // output such that prior-time and current-time viz *should* agree at the same time; facilitates comparison in ParaView
        
        poissonViz->exportSolution(form.configurationSpaceSolution(), std::vector<FunctionPtr> {E_soln-E_exact, E_exact, VE_soln-VE_exact, VE_exact}, std::vector<string>{"E err", "E exact", "V_E err", "V_E exact"}, (k+1) * dt);
      }
    }
  }

  class Maxwellian1V : public TFunction<double>
  {
    double _u;
    double _sigma;
    double _n;
  public:
    Maxwellian1V(double u, double sigma, double n)
    :
    _u(u),
    _sigma(sigma),
    _n(n)
    {}
    
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
          const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V
          const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
          values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
        }
      }
    }
  };

  void testInflowConditionsSteadyState(const bool useCrankNicolson, const bool enforceWeakly,
                                       Teuchos::FancyOStream &out, bool &success)
  {
    if (enforceWeakly)
      out << "Enforcing inflow BCs weakly.\n";
    else
      out << "Enforcing inflow BCs strongly.\n";
    
    const double q    = 0; // 0 charge --> zero electric field
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr v2 = Function::yn(2);
    FunctionPtr v3 = Function::yn(3);
    FunctionPtr f0 = x * v2;
    
    const int xElements = 4;
    const int vElements = 4;
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
    
    const bool usePeriodicBCs = false;
    const bool useHomogeneousPotentialBCs = true;

    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q, useHomogeneousPotentialBCs, useCrankNicolson);
    
    auto vlasovSoln = form.vlasovSolution();
    
    // want to set up RHS/BCs so that f0 is a stationary solution
    auto w = form.w(); // test function for Boltzmann RHS
    vlasovSoln->rhs()->addTerm(w * v3); // this is the result of evaluating the LHS at f0
    
    double timeRamp = 0.0;
    form.setInitialAndInflowConditions(f0, f0, timeRamp, enforceWeakly);
    
//    std::cout << "Vlasov BF:\n" << vlasovSoln->bf()->displayString() << std::endl;
//    std::cout << "Vlasov RHS:\n" << vlasovSoln->rhs()->linearTerm()->displayString() << std::endl;
//    std::cout << "Vlasov IP:\n" << vlasovSoln->ip()->displayString() << std::endl;
        
    const int X_COMP = 0; // zero-based comp for E_fieldVar() call below
    auto poissonSoln = form.configurationSpaceSolution();
    FunctionPtr E_soln = Function::solution(form.configurationSpaceFormulation()->E_fieldVar(X_COMP), poissonSoln);
    
    FunctionPtr E0 = Function::zero();
    
    double err_l2 = (E_soln-E0)->l2norm(poissonSoln->mesh());
    
    double tol = 1e-13;
    TEST_COMPARE(err_l2, <=, tol);
        
    auto poissonBC = BC::bc();
    // set 0 potential BCs at both ends of mesh
    {
      auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
      
      auto poissonMeshTopo = form.configurationSpaceFormulation()->configurationSpaceMeshTopology();
      
      // 1D mesh entity sets:
      // create an empty entity set.  In higher dimensions, this could be an edge or face; on refinement,
      // the entity set contains the descendants of the original.
      EntitySetPtr x0EntitySet = poissonMeshTopo->createEntitySet();
      EntitySetPtr x1EntitySet = poissonMeshTopo->createEntitySet();
      
      // vertex lookup.  There are quite a few ways to look up entities in the MeshTopology.
      IndexType x0Vertex;
      bool x0Found = poissonMeshTopo->getVertexIndex({TEST_X0}, x0Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
      
      IndexType x1Vertex;
      bool x1Found = poissonMeshTopo->getVertexIndex({TEST_X0+TEST_XWIDTH}, x1Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x1Found, std::invalid_argument, "x1 not found");
      
      x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
      x1EntitySet->addEntity(0, x1Vertex); // add entity of dimension 0 with ID x1Vertex to the x1 entity set.
      
      static const int POISSON_TAG_X0 = 0; // left  (x=TEST_X0)
      static const int POISSON_TAG_X1 = 1; // right (x=TEST_X0 + TEST_XWIDTH)
      
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, x0EntitySet);
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X1, x1EntitySet);
      
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X1, Function::constant(0.0)); // pin the potential to 0 at x0+xWidth
    }
    
    poissonSoln->setBC(poissonBC);
//    Teuchos::RCP<HDF5Exporter> vlasovViz = Teuchos::rcp( new HDF5Exporter (vlasovSoln->mesh(), "VlasovTests1D1V_Stationary", "."));
//    vlasovViz->exportSolution(vlasovSoln, 0);
    
    FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
    out << "Comparing f0 with f_soln after setting initial conditions.\n";
    double soln_magnitude = f0->l2norm(vlasovSoln->mesh());
    err_l2 = (f_soln-f0)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);

//    Teuchos::RCP<HDF5Exporter> vlasovViz = Teuchos::rcp( new HDF5Exporter (vlasovSoln->mesh(), "VlasovBackwardEulerFormulation_InflowConditions", "/tmp"));
//    vlasovViz->exportSolution(vlasovSoln, form.t());
    
    form.takeTimeStep();
//    vlasovViz->exportSolution(vlasovSoln, 0.1);
    
    out << "Comparing f0 with f_soln after taking time step.\n";
    soln_magnitude = f0->l2norm(vlasovSoln->mesh());
    err_l2 = (f_soln-f0)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
    
    form.takeTimeStep();
    out << "Comparing f0 with f_soln after taking a second time step.\n";
    soln_magnitude = f0->l2norm(vlasovSoln->mesh());
    err_l2 = (f_soln-f0)->l2norm(vlasovSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
  }

  void testPoissonPatchTest_f_xv2_tp1_E_zero(const bool useCrankNicolson, const bool enforceInflowBCsWeakly, Teuchos::FancyOStream &out, bool &success)
  {
    if (enforceInflowBCsWeakly)
      out << "Enforcing inflow BCs weakly.\n";
    else
      out << "Enforcing inflow BCs strongly.\n";
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
        
    double theta = useCrankNicolson ? 0.5 : 1.0;
    
    const bool usePeriodicBCs = false;
    
    bool homogeneousPotentialBCs = true;
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q, homogeneousPotentialBCs, useCrankNicolson);
    
    // exact solution: x * v2 * (1+t)
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = Function::yn(1);
    FunctionPtr v2 = v * v;
    FunctionPtr v3 = v * v * v;
    FunctionPtr t  = form.t();
    
    FunctionPtr f_inflow  = x * v2 * (1 + t + TEST_DT); // f evaluated at t + Delta t
    
    std::vector<FunctionPtr> f;
    std::vector<FunctionPtr> rhs_extra_load;
    int numTimeSteps = 5;
    for (int k=0; k<=numTimeSteps; k++)
    {
      double t_k = TEST_DT * k;
      FunctionPtr f_k = x * v2 * (1 + t_k);
      f.push_back(f_k);
      if (k<numTimeSteps)
      {
        FunctionPtr rhs = x * v2 + (1 + t_k + theta * TEST_DT) * v3;
        rhs_extra_load.push_back(rhs);
      }
    }
    
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();
    
    const bool overrideVlasovBCs = false;
    const double tol = 1e-12;
    
    // DEBUGGING
    bool outputViz = false;
    std::string vizPrefix = useCrankNicolson ? "Patch_CN" : "Patch_BE";

    patchTestVlasovPoisson(f, f_inflow, E_exact, VE_exact, rhs_extra_load, q,
                           form, overrideVlasovBCs, useCrankNicolson, enforceInflowBCsWeakly, tol, out, success, outputViz, vizPrefix);
  }


  void testPoissonPatchTest_f_xp1v2_t_E_zero(const bool useCrankNicolson, bool enforceInflowBCsWeakly,
                                             Teuchos::FancyOStream &out, bool &success)
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
        
    double theta = useCrankNicolson ? 0.5 : 1.0;
    
    const bool usePeriodicBCs = false;
    
    bool homogeneousPotentialBCs = true;
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q, homogeneousPotentialBCs, useCrankNicolson);
    
    // exact solution: (x+1) * v2 * t
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = Function::yn(1);
    FunctionPtr v2 = v * v;
    FunctionPtr v3 = v * v * v;
    FunctionPtr t  = form.t();
    
    FunctionPtr f_inflow  = (x+1) * v2 * (t + TEST_DT); // f evaluated at t + Delta t
    
    std::vector<FunctionPtr> f;
    std::vector<FunctionPtr> rhs_extra_load;
    int numTimeSteps = 5;
    for (int k=0; k<=numTimeSteps; k++)
    {
      double t_k   = TEST_DT * k;
      double t_kp1 = TEST_DT * (k+1);
      FunctionPtr f_k = (x+1) * v2 * t_k;
      f.push_back(f_k);
      if (k<numTimeSteps)
      {
        FunctionPtr rhs = (x+1) * v2 + (theta * t_kp1 + (1-theta) * t_k) * v3;
        rhs_extra_load.push_back(rhs);
      }
    }
    
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();
    
    const bool overrideVlasovBCs = false;
    const double tol = 1e-12;
    
    // DEBUGGING
    bool outputViz = false;
    std::string vizPrefix = useCrankNicolson ? "Patch_CN" : "Patch_BE";

    if (enforceInflowBCsWeakly)
      out << "Enforcing inflow BCs weakly.\n";
    else
      out << "Enforcing inflow BCs strongly.\n";
    
    patchTestVlasovPoisson(f, f_inflow, E_exact, VE_exact, rhs_extra_load, q,
                           form, overrideVlasovBCs, useCrankNicolson, enforceInflowBCsWeakly,
                           tol, out, success, outputViz, vizPrefix);
  }

  void testPoissonPatchTest_f_xp1v2_tp1_E_zero(const bool useCrankNicolson, bool enforceInflowBCsWeakly,
                                               Teuchos::FancyOStream &out, bool &success)
  {
    const double q    = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
        
    double theta = useCrankNicolson ? 0.5 : 1.0;
    
    const bool usePeriodicBCs = false;
    
    bool homogeneousPotentialBCs = true;
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q, homogeneousPotentialBCs, useCrankNicolson);
    
    // exact solution: (x+1) * v2 * (t + 1)
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = Function::yn(1);
    FunctionPtr v2 = v * v;
    FunctionPtr v3 = v * v * v;
    FunctionPtr t  = form.t();
    
    FunctionPtr f_inflow  = (x+1) * v2 * (t + TEST_DT + 1); // f evaluated at t + Delta t
    
    std::vector<FunctionPtr> f;
    std::vector<FunctionPtr> rhs_extra_load;
    int numTimeSteps = 5;
    for (int k=0; k<=numTimeSteps; k++)
    {
      double t_k   = TEST_DT * k;
      double t_kp1 = TEST_DT * (k+1);
      FunctionPtr f_k = (x+1) * v2 * (t_k + 1);
      f.push_back(f_k);
      if (k<numTimeSteps)
      {
        FunctionPtr rhs = (x+1) * v2 + (theta * t_kp1 + (1-theta) * t_k + 1) * v3;
        rhs_extra_load.push_back(rhs);
      }
    }
    
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();
    
    const bool overrideVlasovBCs = false;
    const double tol = 1e-12;
    
    // DEBUGGING
    bool outputViz = false;
    std::string vizPrefix = useCrankNicolson ? "Patch_CN" : "Patch_BE";

    if (enforceInflowBCsWeakly)
      out << "Enforcing inflow BCs weakly.\n";
    else
      out << "Enforcing inflow BCs strongly.\n";
    
    patchTestVlasovPoisson(f, f_inflow, E_exact, VE_exact, rhs_extra_load, q,
                           form, overrideVlasovBCs, useCrankNicolson, enforceInflowBCsWeakly,
                           tol, out, success, outputViz, vizPrefix);
  }

  void testPoissonPatchTest_f_xv2_t_E_zero(const bool useCrankNicolson, bool enforceInflowBCsWeakly,
                                           Teuchos::FancyOStream &out, bool &success)
  {
    const double q    = 0; // 0 charge --> zero electric field

    const int xElements = 4;
    const int vElements = 4; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace

    double theta = useCrankNicolson ? 0.5 : 1.0;
    
    const bool usePeriodicBCs = false;
    
    bool homogeneousPotentialBCs = true;
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q, homogeneousPotentialBCs, useCrankNicolson);

    // exact solution: x * v2 * t

    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = Function::yn(1);
    FunctionPtr v2 = v * v;
    FunctionPtr v3 = v * v * v;
    FunctionPtr t  = form.t();
    
    FunctionPtr f_inflow  = x * v2 * (t + TEST_DT); // f evaluated at t + Delta t
    
    std::vector<FunctionPtr> f;
    std::vector<FunctionPtr> rhs_extra_load;
    int numTimeSteps = 5;
    for (int k=0; k<=numTimeSteps; k++)
    {
      double t_k   = TEST_DT * k;
      double t_kp1 = TEST_DT * (k+1);
      FunctionPtr f_k = x * v2 * t_k;
      f.push_back(f_k);
      if (k<numTimeSteps)
      {
        FunctionPtr rhs = x * v2 + (theta * t_kp1 + (1-theta) * t_k) * v3;
        rhs_extra_load.push_back(rhs);
      }
    }
    
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();

    const bool overrideVlasovBCs = false;
    const double tol = 1e-12;
    
    // DEBUGGING
    bool outputViz = false;
    std::string vizPrefix = useCrankNicolson ? "Patch_CN" : "Patch_BE";
    
    patchTestVlasovPoisson(f, f_inflow, E_exact, VE_exact, rhs_extra_load, q,
                           form, overrideVlasovBCs, useCrankNicolson, enforceInflowBCsWeakly,
                           tol, out, success, outputViz, vizPrefix);
  }

  void testMaxwellianInflow_E_zero(const bool useCrankNicolson, Teuchos::FancyOStream &out, bool &success)
  {
    // DEBUGGING TEST: try imposing a maxwellian at the inflow.  For now, we'll just compare to f_inflow at each time step; the errors will be large, but I just want to see the viz.
    const double q    = 0; // 0 charge --> zero electric field

    const int xElements = 200;
    const int vElements = 40; // it's important that v have a cell interface at v=0; otherwise, BCs will not be correctly applied…
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
    
    const bool usePeriodicBCs = false;
    
    bool homogeneousPotentialBCs = true;
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q, homogeneousPotentialBCs, useCrankNicolson);

    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = Function::yn(1);
    FunctionPtr v2 = v * v;
    FunctionPtr v3 = v * v * v;
    FunctionPtr t  = form.t();
    
    FunctionPtr f_inflow  = MaxwellianFunction::maxwellian(1, 1, {1.0}, 0.01, 1.0);
    
    std::vector<FunctionPtr> f;
    std::vector<FunctionPtr> rhs_extra_load;
    int numTimeSteps = 50;
    for (int k=0; k<=numTimeSteps; k++)
    {
      FunctionPtr f_k = (k==0) ? Function::zero() : f_inflow;
      f.push_back(f_k);
      if (k<numTimeSteps)
      {
        FunctionPtr rhs = Function::zero();
        rhs_extra_load.push_back(rhs);
      }
    }
    
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();

    const bool overrideVlasovBCs = false;
    const double tol = 1e-12;
    
    // DEBUGGING
    bool outputViz = false;
    std::string vizPrefix = useCrankNicolson ? "Patch_CN" : "Patch_BE";

    std::vector<bool> enforceBCsWeaklyChoices {false, true};
    for (auto enforceBCsWeakly : enforceBCsWeaklyChoices)
    {
      if (enforceBCsWeakly)
        out << "Enforcing inflow BCs weakly.\n";
      else
        out << "Enforcing inflow BCs strongly.\n";
      
      patchTestVlasovPoisson(f, f_inflow, E_exact, VE_exact, rhs_extra_load, q,
                             form, overrideVlasovBCs, useCrankNicolson, enforceBCsWeakly, tol, out, success, outputViz, vizPrefix);
    }
  }

  void testConstantInflow_E_zero(const bool useCrankNicolson, Teuchos::FancyOStream &out, bool &success)
  {
    // DEBUGGING TEST: try imposing a discontinuous constant inflow.  For now, we'll just compare to f_inflow at each time step; the errors will be large, but I just want to see the viz.
    const double q    = 0; // 0 charge --> zero electric field

    const int xElements = 10; // 10
    const int vElements = 40; // 40
    const int H1Order   = 3; // quadratic L^2 functions; allows exact representation of analytic solution trace
    
    const bool usePeriodicBCs = false;
    const double x0     = 0;
    const double xWidth = 1;
    const double v0     = 0;
    const double vWidth = 4;
    const double dt     = 0.05; // 0.01
    
    bool homogeneousPotentialBCs = true;
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q, homogeneousPotentialBCs, useCrankNicolson,
                               x0, xWidth, v0, vWidth, dt);

    FunctionPtr x  = Function::xn(1);
    FunctionPtr v  = Function::yn(1);
    FunctionPtr v2 = v * v;
    ParameterFunctionPtr t  = form.t();
    
    double transitTimeAcrossGap = 1.0; // for the lower part of the distribution; for upper, it's half this
    double timeRampWidth        = transitTimeAcrossGap * 3.0;
    FunctionPtr timeRamp        = TimeRampFunction::timeRamp(t, timeRampWidth); // timeRamp evaluated at current time
    
    FunctionPtr heaviside_v1 = Function::heavisideY(1.0); // value is 1 for v > 1
    FunctionPtr heaviside_v2 = Function::heavisideY(2.0); // value is 1 for v > 2
    FunctionPtr f_inflow  = timeRamp * (heaviside_v1 - heaviside_v2);  // value is 1 for 1 < v < 2; 0 elsewhere
    
    std::vector<FunctionPtr> f;
    std::vector<FunctionPtr> rhs_extra_load;
    int numTimeSteps = (timeRampWidth + transitTimeAcrossGap) / dt; // give the true BC value time to get across the gap
    for (int k=0; k<=numTimeSteps; k++)
    {
      FunctionPtr f_k = (k==0) ? Function::zero() : f_inflow;
      f.push_back(f_k);
      if (k<numTimeSteps)
      {
        FunctionPtr rhs = Function::zero();
        rhs_extra_load.push_back(rhs);
      }
    }
    
    FunctionPtr E_exact = Function::zero();
    FunctionPtr VE_exact = Function::zero();

    const bool overrideVlasovBCs = false;
    const double tol = 1e-12;
    
    // DEBUGGING
    bool outputViz = false;
    std::string vizPrefix = useCrankNicolson ? "Constant_CN" : "Constant_BE";

    std::vector<bool> enforceBCsWeaklyChoices {false, true};
    for (auto enforceBCsWeakly : enforceBCsWeaklyChoices)
    {
      if (enforceBCsWeakly)
        out << "Enforcing inflow BCs weakly.\n";
      else
        out << "Enforcing inflow BCs strongly.\n";
      
      patchTestVlasovPoisson(f, f_inflow, E_exact, VE_exact, rhs_extra_load, q,
                             form, overrideVlasovBCs, useCrankNicolson, enforceBCsWeakly, tol, out, success, outputViz, vizPrefix);
    }
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, InflowConditionsBE )
  {
    const bool useCrankNicolson = false; // use backward euler
    bool enforceBCsWeakly = false;
    testInflowConditionsSteadyState(useCrankNicolson, enforceBCsWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, InflowConditionsBE_Weak )
  {
    const bool useCrankNicolson = false; // use backward euler
    bool enforceBCsWeakly = true;
    testInflowConditionsSteadyState(useCrankNicolson, enforceBCsWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, InflowConditionsCN )
  {
    const bool useCrankNicolson = true;
    bool enforceBCsWeakly = false;
    testInflowConditionsSteadyState(useCrankNicolson, enforceBCsWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, InflowConditionsCN_Weak )
  {
    const bool useCrankNicolson = true;
    bool enforceBCsWeakly = true;
    testInflowConditionsSteadyState(useCrankNicolson, enforceBCsWeakly, out, success);
  }

//  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, MaxwellianDEBUG_BackwardEuler )
//  {
//    const bool useCrankNicolson = false;
//    testMaxwellianInflow_E_zero(useCrankNicolson, out, success);
//  }
//
//  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, MaxwellianDEBUG_CrankNicolson )
//  {
//    const bool useCrankNicolson = true;
//    testMaxwellianInflow_E_zero(useCrankNicolson, out, success);
//  }
//
//  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, ConstantDEBUG_BackwardEuler )
//  {
//    const bool useCrankNicolson = false;
//    testConstantInflow_E_zero(useCrankNicolson, out, success);
//  }
//
//  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, ConstantDEBUG_CrankNicolson )
//  {
//    const bool useCrankNicolson = true;
//    testConstantInflow_E_zero(useCrankNicolson, out, success);
//  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_t_E_zero_BackwardEuler )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowBCsWeakly = false;
    testPoissonPatchTest_f_xv2_t_E_zero(useCrankNicolson, enforceInflowBCsWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_t_E_zero_BackwardEuler_WeakInflowBC )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowBCsWeakly = true;
    testPoissonPatchTest_f_xv2_t_E_zero(useCrankNicolson, enforceInflowBCsWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_t_E_zero_CrankNicolson )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowBCsWeakly = false;
    testPoissonPatchTest_f_xv2_t_E_zero(useCrankNicolson, enforceInflowBCsWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_t_E_zero_CrankNicolson_WeakInflowBC )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowBCsWeakly = true;
    testPoissonPatchTest_f_xv2_t_E_zero(useCrankNicolson, enforceInflowBCsWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_t_E_zero_BackwardEuler )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowWeakly = false;
    testPoissonPatchTest_f_xp1v2_t_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_t_E_zero_BackwardEuler_WeakInflowBC )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowWeakly = true;
    testPoissonPatchTest_f_xp1v2_t_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_t_E_zero_CrankNicolson )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowWeakly = false;
    testPoissonPatchTest_f_xp1v2_t_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_t_E_zero_CrankNicolson_WeakInflowBC )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowWeakly = true;
    testPoissonPatchTest_f_xp1v2_t_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }
  
  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_tp1_E_zero_BackwardEuler )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowWeakly = false;
    testPoissonPatchTest_f_xp1v2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }
  
  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_tp1_E_zero_BackwardEuler_WeakInflowBC )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowWeakly = true;
    testPoissonPatchTest_f_xp1v2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_tp1_E_zero_CrankNicolson )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowWeakly = false;
    testPoissonPatchTest_f_xp1v2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xp1v2_tp1_E_zero_CrankNicolson_WeakInflowBC )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowWeakly = true;
    testPoissonPatchTest_f_xp1v2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_tp1_E_zero_BackwardEuler )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowWeakly = false;
    testPoissonPatchTest_f_xv2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_tp1_E_zero_BackwardEuler_WeakInflowBC )
  {
    const bool useCrankNicolson = false;
    const bool enforceInflowWeakly = true;
    testPoissonPatchTest_f_xv2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_tp1_E_zero_CrankNicolson )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowWeakly = false;
    testPoissonPatchTest_f_xv2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, PoissonPatchTest_f_xv2_tp1_E_zero_CrankNicolson_WeakInflowBC )
  {
    const bool useCrankNicolson = true;
    const bool enforceInflowWeakly = true;
    testPoissonPatchTest_f_xv2_tp1_E_zero(useCrankNicolson, enforceInflowWeakly, out, success);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, RefineUniformly )
  {
    const double q      = 1.0;
    const int xElements = 3;
    const int vElements = 4;
    const int H1Order   = 1;
    const bool usePeriodicBCs = false;
    
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q);
    
    int expectedActiveElementCountVlasov = xElements * vElements;
    int actualActiveElementCountVlasov   = form.vlasovMesh()->numActiveElements();
    
    TEST_EQUALITY(expectedActiveElementCountVlasov, actualActiveElementCountVlasov);
    
    int expectedActiveElementCountPoisson = xElements;
    int actualActiveElementCountPoisson   = form.configurationSpaceMesh()->numActiveElements();
    
    TEST_EQUALITY(expectedActiveElementCountPoisson, actualActiveElementCountPoisson);
    
    int numUniformRefinements = 1;
    form.refineUniformly(numUniformRefinements);
    
    int vlasovCountMultiplier  = 4;
    int poissonCountMultiplier = 2;
    
    expectedActiveElementCountVlasov  *= vlasovCountMultiplier;
    expectedActiveElementCountPoisson *= poissonCountMultiplier;
    
    actualActiveElementCountVlasov  = form.vlasovMesh()->numActiveElements();
    actualActiveElementCountPoisson = form.configurationSpaceMesh()->numActiveElements();
    TEST_EQUALITY(expectedActiveElementCountVlasov, actualActiveElementCountVlasov);
    TEST_EQUALITY(expectedActiveElementCountPoisson, actualActiveElementCountPoisson);
    
    numUniformRefinements = 2;
    form.refineUniformly(numUniformRefinements);
    
    vlasovCountMultiplier  = 4*4;
    poissonCountMultiplier = 2*2;
    
    expectedActiveElementCountVlasov  *= vlasovCountMultiplier;
    expectedActiveElementCountPoisson *= poissonCountMultiplier;
    
    actualActiveElementCountVlasov  = form.vlasovMesh()->numActiveElements();
    actualActiveElementCountPoisson = form.configurationSpaceMesh()->numActiveElements();
    TEST_EQUALITY(expectedActiveElementCountVlasov, actualActiveElementCountVlasov);
    TEST_EQUALITY(expectedActiveElementCountPoisson, actualActiveElementCountPoisson);
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, SideSetInflow )
  {
    const double q = 0; // 0 charge --> zero electric field
    
    const int xElements = 4;
    const int vElements = 4;
    const int H1Order   = 3;
    const bool usePeriodicBCs = false;
    
    auto form = getFormulation(xElements, vElements, H1Order, usePeriodicBCs, q);

    auto vlasovSoln = form.vlasovSolution();
    auto vlasovMeshTopo = vlasovSoln->mesh()->getTopology()->baseMeshTopology();

    int spatialInflowTag = form.tagVlasovSpatialInflow();

    auto actualSideSets = vlasovMeshTopo->getEntitySetsForTagID(DIRICHLET_SET_TAG_NAME, spatialInflowTag);
    
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
    
    int expectedSideCount = vElements; // vElements/2 from upper left, vElements/2 from lower right
    int actualSideCount   = sideIndices.size();
    
    TEST_EQUALITY(expectedSideCount, actualSideCount);
    
    const int SPACE_COMPONENT_INDEX_ZERO_BASED    = 0;
    const int VELOCITY_COMPONENT_INDEX_ZERO_BASED = 1;
    
    // next, check that the sides that did match each have v>=0, and x==0.
    for (auto sideIndex : sideIndices)
    {
      auto vertexIndices = vlasovMeshTopo->getEntityVertexIndices(sideDim, sideIndex);
      for (auto vertexIndex : vertexIndices)
      {
        const auto & vertex = vlasovMeshTopo->getVertex(vertexIndex);
        const bool leftVertex  = vertex[SPACE_COMPONENT_INDEX_ZERO_BASED] == TEST_X0;
        const bool rightVertex = vertex[SPACE_COMPONENT_INDEX_ZERO_BASED] == TEST_X0 + TEST_XWIDTH;
        const bool leftInflow  = leftVertex  && (vertex[VELOCITY_COMPONENT_INDEX_ZERO_BASED] >= 0);
        const bool rightInflow = rightVertex && (vertex[VELOCITY_COMPONENT_INDEX_ZERO_BASED] <= 0);
        TEST_ASSERT(leftInflow || rightInflow);
      }
    }
  }

  TEUCHOS_UNIT_TEST( VlasovBackwardEulerFormulation, TimeStepStabilityUltraweakImplicit )
  {
    // confirm, for an example problem, that || u^{n+1} || ≤ || u^n ||
    // we set everything up as in two-stream instability, and use a coarse (2 x 2) high-order (H1Order = 4) mesh.
    // We run with both our usual test space norm, and the automatic graph norm for the bilinear form.
    // We use dt's that go from 1e0 to 1e-5.
    // run test suite with --details=ALL argument to see numerical details.
    
    // set things up as in two-stream
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    double v0     = -10.0;
    double vWidth =  20.0;
    
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    double n = 0.5 * n0;
    FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1V( u0, sigma, n) );
    FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1V(-u0, sigma, n) );
    FunctionPtr f0           = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
    
    const int xElements = 2;
    const int vElements = 2;
    const int H1Order   = 4; // cubic L^2 functions
    
    const int delta_k = 2;
    const double dt = 1.0;
    const bool usePeriodicBCs = true;
    
    auto poissonConfig = ConfigurationSpaceFormulation::poissonFormulation(q, eps0, TIME_MARCHING);
    auto spatialMeshTopo = MeshFactory::intervalMeshTopology(x0, x0 + xWidth, xElements, usePeriodicBCs);
    std::vector<double> vxVertices;
    double vx = v0;
    for (int i=0; i<=vElements; i++)
    {
      vx = v0 + (i * vWidth) / vElements;
      vxVertices.push_back(vx);
    }
    const bool useSerendipity = true;
    VlasovBackwardEulerFormulation form(q, m, useSerendipity, poissonConfig, spatialMeshTopo, dt, vxVertices, H1Order, delta_k);
    form.setRunQuietly(true);
    auto poissonMesh = poissonConfig->configurationSpaceMesh();
    // set a zero BC on Poisson at x0
    auto poissonBC = BC::bc();
    {
      auto VE_hat = form.configurationSpaceFormulation()->VE_hat();
      
      auto poissonMeshTopo = form.configurationSpaceFormulation()->configurationSpaceMeshTopology();
      
      // 1D mesh entity sets:
      // create an empty entity set.  In higher dimensions, this could be an edge or face; on refinement,
      // the entity set contains the descendants of the original.
      EntitySetPtr x0EntitySet = poissonMeshTopo->createEntitySet();
      
      // vertex lookup.  There are quite a few ways to look up entities in the MeshTopology.
      IndexType x0Vertex;
      bool x0Found = poissonMeshTopo->getVertexIndex({TEST_X0}, x0Vertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!x0Found, std::invalid_argument, "x0 not found");
      
      x0EntitySet->addEntity(0, x0Vertex); // add entity of dimension 0 with ID x0Vertex to the x0 entity set.
      
      static const int POISSON_TAG_X0 = 0; // left  (x=TEST_X0)
      poissonMeshTopo->applyTag(DIRICHLET_SET_TAG_NAME, POISSON_TAG_X0, x0EntitySet);
      poissonBC->addDirichlet(VE_hat, POISSON_TAG_X0, Function::constant(0.0)); // pin the potential to 0 at x0
    }
    poissonConfig->solution()->setBC(poissonBC);
    
    form.setEnforceQuasiNeutrality(true);
    form.setInitialAndInflowConditions(f0, f0);
    
    auto vlasovSoln = form.vlasovSolution();
    IPPtr defaultIP = vlasovSoln->ip(); // our default inner product for Boltzmann; at present, this is one inspired by Führer et al.
    IPPtr graphNorm = vlasovSoln->bf()->graphNorm(); // the usual default choice for generic DPG discretization (has unit weight on the extra L^2 terms)
    
    std::vector<IPPtr> ipsToTest {defaultIP, graphNorm};
    
    std::vector<double> dtsToTest {1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    
    for (const auto & ipToTest : ipsToTest)
    {
      out << "************************************************************\n";
      out << "testing IP: " << ipToTest->displayString() << std::endl;
      vlasovSoln->setIP(ipToTest);
      
      for (const auto & dt : dtsToTest)
      {
        out << "testing dt: " << dt << std::endl;
        form.setTimeStep(dt);
        
        bool imposeInflowWeakly = false;
        form.setInitialConditions(f0, imposeInflowWeakly);
        
        // we should measure the projected f0 rather than the exact.
        
        FunctionPtr f_soln = Function::solution(form.f(), vlasovSoln);
        const double f0_l2 = f_soln->l2norm(form.vlasovMesh());
        out << "f0_l2: " << f0_l2 << std::endl;

        form.takeTimeStep();
        
        const double f1_l2 = f_soln->l2norm(form.vlasovMesh());
        
        out << "f1_l2: " << f1_l2 << std::endl;
        out << "f1_l2 / f0_l2: " << f1_l2 / f0_l2 << std::endl;
        out << "1 - f1_l2 / f0_l2 = " << 1 - f1_l2 / f0_l2 << std::endl;
        
        TEST_COMPARE(f1_l2, <=, f0_l2);
      }
    }
  }
} // end anonymous namespace
