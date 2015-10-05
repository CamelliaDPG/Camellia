//
//  NavierStokesVGPFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"

using namespace Camellia;
using namespace Intrepid;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, Consistency_Steady_2D )
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 1.0e2;
    int fieldPolyOrder = 3, delta_k = 1;

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    //    FunctionPtr u1 = x;
    //    FunctionPtr u2 = -y; // divergence 0
    //    FunctionPtr p = y * y * y; // zero average
    FunctionPtr u1 = x * x * y;
    FunctionPtr u2 = -x * y * y;
    FunctionPtr p = y * y * y;

    bool useConformingTraces = true;
    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
//    cout << "forcingFunction: " << forcingFunction->displayString() << endl;
    form.setForcingFunction(forcingFunction);

    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();

    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u(1)->ID()] = u1;
    exactMap[form.u(2)->ID()] = u2;
    exactMap[form.p()->ID() ] =  p;
    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
    exactMap[form.sigma(2,2)->ID()] = sigma2->y();

    // fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
    exactMap[form.tn_hat(1)->ID()] = t1_n;
    exactMap[form.tn_hat(2)->ID()] = t2_n;

    // traces:
    exactMap[form.u_hat(1)->ID()] = u1;
    exactMap[form.u_hat(2)->ID()] = u2;

    map<int, FunctionPtr> zeroMap;
    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++)
    {
      zeroMap[exactMapIt->first] = Function::zero(exactMapIt->second->rank());
    }

    form.solution()->projectOntoMesh(exactMap);
    form.solutionIncrement()->projectOntoMesh(zeroMap);

    RHSPtr rhs = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly
    form.solutionIncrement()->setRHS(rhs);

//    cout << "rhs: " << rhs->linearTerm()->displayString() << endl;

    double energyError = form.solutionIncrement()->energyErrorTotal();

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, Consistency_SteadyConservation_2D )
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 1.0e2;
    int fieldPolyOrder = 3, delta_k = 1;

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    //    FunctionPtr u1 = x;
    //    FunctionPtr u2 = -y; // divergence 0
    //    FunctionPtr p = y * y * y; // zero average
    FunctionPtr u1 = x * x * y;
    FunctionPtr u2 = -x * y * y;
    FunctionPtr p = y * y * y;

    bool useConformingTraces = true;
    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyConservationFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
    // NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
//    cout << "forcingFunction: " << forcingFunction->displayString() << endl;
    form.setForcingFunction(forcingFunction);

    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();

    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u(1)->ID()] = u1;
    exactMap[form.u(2)->ID()] = u2;
    exactMap[form.p()->ID() ] =  p;
    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
    exactMap[form.sigma(2,2)->ID()] = sigma2->y();

    // fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr n_x = Function::normal();
    FunctionPtr n_x_parity = n_x * TFunction<double>::sideParity();
    FunctionPtr t1_n = u1*u1*n_x_parity->x() + u1*u2*n_x_parity->y() - sigma1*n_x_parity + p*n_x_parity->x();
    FunctionPtr t2_n = u2*u1*n_x_parity->x() + u2*u2*n_x_parity->y() - sigma2*n_x_parity + p*n_x_parity->y();
    // FunctionPtr t1_n = t1_n_lt->evaluate(exactMap) + u1*u1*n_x_parity->x() + u1*u2*n_x_parity->y();
    // FunctionPtr t2_n = t2_n_lt->evaluate(exactMap) + u2*u1*n_x_parity->x() + u2*u2*n_x_parity->y();
    // FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
    // FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
    exactMap[form.tn_hat(1)->ID()] = t1_n;
    exactMap[form.tn_hat(2)->ID()] = t2_n;

    // traces:
    exactMap[form.u_hat(1)->ID()] = u1;
    exactMap[form.u_hat(2)->ID()] = u2;

    map<int, FunctionPtr> zeroMap;
    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++)
    {
      zeroMap[exactMapIt->first] = Function::zero(exactMapIt->second->rank());
    }

    form.solution()->projectOntoMesh(exactMap);
    form.solutionIncrement()->projectOntoMesh(zeroMap);

    RHSPtr rhs = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly
    form.solutionIncrement()->setRHS(rhs);

//    cout << "rhs: " << rhs->linearTerm()->displayString() << endl;

    double energyError = form.solutionIncrement()->energyErrorTotal();

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, ExactSolution_Steady_2D_Slow )
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 1.0;
    int fieldPolyOrder = 3, delta_k = 1;

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);

    FunctionPtr u1 = x * x * y;
    FunctionPtr u2 = -x * y * y;
    FunctionPtr p = y;

    bool useConformingTraces = true;
    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
    form.setForcingFunction(forcingFunction);
    RHSPtr rhsForSolve = form.solutionIncrement()->rhs();

//    cout << "bf for Navier-Stokes:\n";
//    form.bf()->printTrialTestInteractions();

//    cout << "rhs for Navier-Stokes solve:\n" << rhsForSolve->linearTerm()->displayString();

    form.addInflowCondition(SpatialFilter::allSpace(), Function::vectorize(u1, u2));
    form.addZeroMeanPressureCondition();

    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();

    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u(1)->ID()] = u1;
    exactMap[form.u(2)->ID()] = u2;
    exactMap[form.p()->ID() ] =  p;
    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
    exactMap[form.sigma(2,2)->ID()] = sigma2->y();

    // fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
    exactMap[form.tn_hat(1)->ID()] = t1_n;
    exactMap[form.tn_hat(2)->ID()] = t2_n;

    // traces:
    exactMap[form.u_hat(1)->ID()] = u1;
    exactMap[form.u_hat(2)->ID()] = u2;

    map<int, FunctionPtr> zeroMap;
    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++)
    {
      VarPtr trialVar = form.bf()->varFactory()->trial(exactMapIt->first);
      FunctionPtr zero = Function::zero();
      for (int i=0; i<trialVar->rank(); i++)
      {
        if (spaceDim == 2)
          zero = Function::vectorize(zero, zero);
        else if (spaceDim == 3)
          zero = Function::vectorize(zero, zero, zero);
      }
      zeroMap[exactMapIt->first] = zero;
    }

    RHSPtr rhsWithBoundaryTerms = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly

    double tol = 1e-12;

    // sanity/consistency check: is the energy error for a zero solutionIncrement zero?
    form.solutionIncrement()->projectOntoMesh(zeroMap);
    form.solution()->projectOntoMesh(exactMap);
    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
    double energyError = form.solutionIncrement()->energyErrorTotal();

    TEST_COMPARE(energyError, <, tol);

    // change RHS back for solve below:
    form.solutionIncrement()->setRHS(rhsForSolve);

    // first real test: with exact background flow, if we solve, do we maintain zero energy error?
    form.solveAndAccumulate();
    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
    form.solutionIncrement()->projectOntoMesh(zeroMap); // zero out since we've accumulated
    energyError = form.solutionIncrement()->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);

    // change RHS back for solve below:
    form.solutionIncrement()->setRHS(rhsForSolve);

    // next test: try starting from a zero initial guess
    form.solution()->projectOntoMesh(zeroMap);

    SolutionPtr solnIncrement = form.solutionIncrement();

    FunctionPtr u1_incr = Function::solution(form.u(1), solnIncrement);
    FunctionPtr u2_incr = Function::solution(form.u(2), solnIncrement);
    FunctionPtr p_incr = Function::solution(form.p(), solnIncrement);

    FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr;

    double l2_norm_incr = 0.0;
    double nonlinearTol = 1e-12;
    int maxIters = 10;
    do
    {
      form.solveAndAccumulate();
      l2_norm_incr = sqrt(l2_incr->integrate(solnIncrement->mesh()));
      out << "iteration " << form.nonlinearIterationCount() << ", L^2 norm of increment: " << l2_norm_incr << endl;
    }
    while ((l2_norm_incr > nonlinearTol) && (form.nonlinearIterationCount() < maxIters));

    form.solutionIncrement()->setRHS(rhsWithBoundaryTerms);
    form.solutionIncrement()->projectOntoMesh(zeroMap); // zero out since we've accumulated
    energyError = form.solutionIncrement()->energyErrorTotal();
    TEST_COMPARE(energyError, <, tol);

    //    if (energyError >= tol) {
    //      HDF5Exporter::exportSolution("/tmp", "NSVGP_background_flow",form.solution());
    //      HDF5Exporter::exportSolution("/tmp", "NSVGP_soln_increment",form.solutionIncrement());
    //    }
  }

  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, ForcingFunction_Steady_2D)
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 1.0e2;
    int fieldPolyOrder = 3, delta_k = 1;

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);

    FunctionPtr u1 = x * x * y;
    FunctionPtr u2 = -x * y * y;
    FunctionPtr p = y * y * y;

    bool useConformingTraces = true;
    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
    FunctionPtr forcingFunction = form.forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
    
    FunctionPtr expectedForcingFunction_x = p->dx() - (1.0 / Re) * (u1->dx()->dx() + u1->dy()->dy()) + u1 * u1->dx() + u2 * u1->dy();
    FunctionPtr expectedForcingFunction_y = p->dy() - (1.0 / Re) * (u2->dx()->dx() + u2->dy()->dy()) + u1 * u2->dx() + u2 * u2->dy();
    
    double err_x = (expectedForcingFunction_x - forcingFunction->x())->l2norm(form.solution()->mesh());
    double err_y = (expectedForcingFunction_y - forcingFunction->y())->l2norm(form.solution()->mesh());

    double tol = 1e-12;
    TEST_COMPARE(err_x, <, tol);
    TEST_COMPARE(err_y, <, tol);
//    cout << forcingFunction->displayString();
  }

  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, StokesConsistency_Steady_2D )
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 10.0;
    int fieldPolyOrder = 2, delta_k = 1;

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    //    FunctionPtr u1 = x;
    //    FunctionPtr u2 = -y; // divergence 0
    //    FunctionPtr p = y * y * y; // zero average
    FunctionPtr u1 = x;
    FunctionPtr u2 = -y;
    FunctionPtr p = y;

    FunctionPtr forcingFunction_x = p->dx() - (1.0/Re) * (u1->dx()->dx() + u1->dy()->dy());
    FunctionPtr forcingFunction_y = p->dy() - (1.0/Re) * (u2->dx()->dx() + u2->dy()->dy());
    FunctionPtr forcingFunction = Function::vectorize(forcingFunction_x, forcingFunction_y);

    bool useConformingTraces = true;
    NavierStokesVGPFormulation form = NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, useConformingTraces, meshTopo, fieldPolyOrder, delta_k);
    form.setForcingFunction(forcingFunction);

    BFPtr stokesBF = form.stokesBF();

    MeshPtr stokesMesh = Teuchos::rcp( new Mesh(meshTopo,stokesBF,fieldPolyOrder+1, delta_k) );

    SolutionPtr stokesSolution = Solution::solution(stokesMesh);
    stokesSolution->setIP(stokesBF->graphNorm());
    RHSPtr rhs = RHS::rhs();
    rhs->addTerm(forcingFunction_x * form.v(1));
    rhs->addTerm(forcingFunction_y * form.v(2));

    stokesSolution->setRHS(rhs);

    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();

    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u(1)->ID()] = u1;
    exactMap[form.u(2)->ID()] = u2;
    exactMap[form.p()->ID() ] =  p;
    exactMap[form.sigma(1,1)->ID()] = sigma1->x();
    exactMap[form.sigma(1,2)->ID()] = sigma1->y();
    exactMap[form.sigma(2,1)->ID()] = sigma2->x();
    exactMap[form.sigma(2,2)->ID()] = sigma2->y();

    // fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
    exactMap[form.tn_hat(1)->ID()] = t1_n;
    exactMap[form.tn_hat(2)->ID()] = t2_n;

    // traces:
    exactMap[form.u_hat(1)->ID()] = u1;
    exactMap[form.u_hat(2)->ID()] = u2;

    stokesSolution->projectOntoMesh(exactMap);

    double energyError = stokesSolution->energyErrorTotal();

    double tol = 1e-14;
    TEST_COMPARE(energyError, <, tol);
  }
} // namespace
