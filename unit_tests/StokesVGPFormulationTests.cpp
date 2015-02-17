//
//  StokesVGPFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#include "StokesVGPFormulation.h"
#include "MeshFactory.h"
#include "Solution.h"
#include "HDF5Exporter.h"

#include "Teuchos_UnitTestHarness.hpp"
namespace {
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_2D_Steady )
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
    FunctionPtr u1 = x;
    FunctionPtr u2 = -y; // divergence 0
    FunctionPtr p = y * y * y; // zero average
    //    FunctionPtr u1 = Function::constant(1.0);
    //    FunctionPtr u2 = Function::constant(1.0);
    //    FunctionPtr p = Function::zero();
    
    FunctionPtr forcingFunction_x = p->dx() - (1.0/Re) * (u1->dx()->dx() + u1->dy()->dy());
    FunctionPtr forcingFunction_y = p->dy() - (1.0/Re) * (u2->dx()->dx() + u2->dy()->dy());
    FunctionPtr forcingFunction = Function::vectorize(forcingFunction_x, forcingFunction_y);
    bool useConformingTraces = true;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re);
    
    BFPtr stokesBF = form.bf();
    
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
    exactMap[form.sigma(1)->ID()] = sigma1;
    exactMap[form.sigma(2)->ID()] = sigma2;
    
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
    
    TEST_EQUALITY(exactMap.size(), 9); // 5 fields, two fluxes, two traces
    
    FunctionPtr pSoln = Function::solution(form.p(), stokesSolution);
    
    stokesSolution->clearComputedResiduals();
    
    double energyError = stokesSolution->energyErrorTotal();
    
    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_2D_Transient )
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 1.0;
    int fieldPolyOrder = 3, delta_k = 1;

    // testing transient formulation consistency goes much as with the steady state;
    // if we project a steady solution onto the previous solution as well as the current solution,
    // we should have a zero residual
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr u1 = x;
    FunctionPtr u2 = -y; // divergence 0
    FunctionPtr p = y * y * y; // zero average
    
    FunctionPtr forcingFunction_x = p->dx() - (1.0/Re) * (u1->dx()->dx() + u1->dy()->dy());
    FunctionPtr forcingFunction_y = p->dy() - (1.0/Re) * (u2->dx()->dx() + u2->dy()->dy());
    FunctionPtr forcingFunction = Function::vectorize(forcingFunction_x, forcingFunction_y);
    bool useConformingTraces = true;
    bool transient = true;
    double dt = 1.0;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re, transient, dt);
    
    form.initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction);
  
    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();
    
    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u(1)->ID()] = u1;
    exactMap[form.u(2)->ID()] = u2;
    exactMap[form.p()->ID() ] =  p;
    exactMap[form.sigma(1)->ID()] = sigma1;
    exactMap[form.sigma(2)->ID()] = sigma2;
    
    // fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    exactMap[form.tn_hat(1)->ID()] = form.tn_hat(1)->termTraced()->evaluate(exactMap);
    exactMap[form.tn_hat(2)->ID()] = form.tn_hat(2)->termTraced()->evaluate(exactMap);
    
    // traces:
    exactMap[form.u_hat(1)->ID()] = form.u_hat(1)->termTraced()->evaluate(exactMap);
    exactMap[form.u_hat(2)->ID()] = form.u_hat(2)->termTraced()->evaluate(exactMap);
    
    form.solution()->projectOntoMesh(exactMap);
    form.solutionPreviousTimeStep()->projectOntoMesh(exactMap);
    
    TEST_EQUALITY(exactMap.size(), 9); // 5 fields, two fluxes, two traces
    
    FunctionPtr pSoln = Function::solution(form.p(), form.solution());
    
    form.solution()->clearComputedResiduals();
    
    double energyError = form.solution()->energyErrorTotal();
    
    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, ForcingFunction_2D) {
    double Re = 10.0;

    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    bool useConformingTraces = true;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re);
    int fieldPolyOrder = 1;
    int delta_k = 1;
    MeshPtr stokesMesh = Teuchos::rcp( new Mesh(meshTopo,form.bf(),fieldPolyOrder+1, delta_k) );

    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr u1 = x;
    FunctionPtr u2 = -y; // divergence 0
    //    FunctionPtr p = y * y * y; // zero average
    //    FunctionPtr u1 = Function::constant(1.0);
    //    FunctionPtr u2 = Function::constant(1.0);
    FunctionPtr p = x + y;
    
    FunctionPtr forcingFunction_x = p->dx() - (1.0/Re) * (u1->dx()->dx() + u1->dy()->dy());
    FunctionPtr forcingFunction_y = p->dy() - (1.0/Re) * (u2->dx()->dx() + u2->dy()->dy());
    FunctionPtr forcingFunctionExpected = Function::vectorize(forcingFunction_x, forcingFunction_y);

    FunctionPtr forcingFunctionActual = StokesVGPFormulation::forcingFunction(spaceDim, 1.0 / Re, Function::vectorize(u1, u2), p);
    
    double tol = 1e-13;
    double err = (forcingFunctionExpected - forcingFunctionActual)->l2norm(stokesMesh);
    TEST_COMPARE(err, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( StokesVGPFormulation, Projection_2D )
  {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 1.0;
    int fieldPolyOrder = 1, delta_k = 1;
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr u1 = x;
    FunctionPtr u2 = -y; // divergence 0
    FunctionPtr u = Function::vectorize(u1, u2);
//    FunctionPtr p = y * y * y; // zero average
//    FunctionPtr u1 = Function::constant(1.0);
//    FunctionPtr u2 = Function::constant(1.0);
    FunctionPtr p = x + y;
    
    FunctionPtr forcingFunction = StokesVGPFormulation::forcingFunction(spaceDim, 1.0 / Re, u, p);
    
    bool useConformingTraces = true;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re);
    
    form.initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction);
    
    MeshPtr stokesMesh = form.solution()->mesh();
    
    BFPtr stokesBF = form.bf();
    
    //uniform h-refinement:
    stokesMesh->hRefine(stokesMesh->getActiveCellIDs());
    
    form.addZeroMeanPressureCondition();
    form.addInflowCondition(SpatialFilter::allSpace(), u);
    
    form.solve();
    
    FunctionPtr sigma1 = (1.0 / Re) * u1->grad();
    FunctionPtr sigma2 = (1.0 / Re) * u2->grad();
    
    LinearTermPtr t1_n_lt = form.tn_hat(1)->termTraced();
    LinearTermPtr t2_n_lt = form.tn_hat(2)->termTraced();
    
    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u(1)->ID()] = u1;
    exactMap[form.u(2)->ID()] = u2;
    exactMap[form.p()->ID() ] =  p;
    exactMap[form.sigma(1)->ID()] = sigma1;
    exactMap[form.sigma(2)->ID()] = sigma2;
    
    // fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
    exactMap[form.tn_hat(1)->ID()] = t1_n;
    exactMap[form.tn_hat(2)->ID()] = t2_n;
    
    // traces:
    exactMap[form.u_hat(1)->ID()] = u1;
    exactMap[form.u_hat(2)->ID()] = u2;
    
    SolutionPtr stokesProjection = Solution::solution(stokesMesh);
    stokesProjection->projectOntoMesh(exactMap);
    
    TEST_EQUALITY(exactMap.size(), 9); // 5 fields, two fluxes, two traces
    
    SolutionPtr stokesSolution = form.solution();
    stokesSolution->addSolution(stokesProjection, -1);
    
    FunctionPtr u1_diff = Function::solution(form.u(1), stokesSolution);
    FunctionPtr u2_diff = Function::solution(form.u(2), stokesSolution);
    FunctionPtr sigma1_diff = Function::solution(form.sigma(1), stokesSolution);
    FunctionPtr sigma2_diff = Function::solution(form.sigma(2), stokesSolution);
    FunctionPtr p_diff = Function::solution(form.p(), stokesSolution);
    
    double p_diff_l2 = p_diff->l2norm(stokesMesh);
    double u1_diff_l2 = u1_diff->l2norm(stokesMesh);
    double u2_diff_l2 = u2_diff->l2norm(stokesMesh);
    double sigma1_diff_l2 = sigma1_diff->l2norm(stokesMesh);
    double sigma2_diff_l2 = sigma2_diff->l2norm(stokesMesh);
    
    double tol = 1e-13;
    TEST_COMPARE(p_diff_l2, <, tol);
    TEST_COMPARE(u1_diff_l2, <, tol);
    TEST_COMPARE(u2_diff_l2, <, tol);
    TEST_COMPARE(sigma1_diff_l2, <, tol);
    TEST_COMPARE(sigma2_diff_l2, <, tol);
  }
  
} // namespace