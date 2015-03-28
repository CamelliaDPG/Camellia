//
//  NavierStokesVGPFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#include "NavierStokesVGPFormulation.h"

#include "MeshFactory.h"

#include "Teuchos_UnitTestHarness.hpp"
namespace {
  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, Consistency_2D )
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
    //    FunctionPtr u1 = x;
    //    FunctionPtr u2 = -y; // divergence 0
    //    FunctionPtr p = y * y * y; // zero average
    FunctionPtr u1 = x * x * y;
    FunctionPtr u2 = -x * y * y;
    FunctionPtr p = y * y * y;
    
    FunctionPtr forcingFunction = NavierStokesVGPFormulation::forcingFunction(spaceDim, Re, Function::vectorize(u1,u2), p);
    NavierStokesVGPFormulation form(meshTopo, Re, fieldPolyOrder, delta_k, forcingFunction);
    
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
    
    map<int, FunctionPtr> zeroMap;
    for (map<int, FunctionPtr>::iterator exactMapIt = exactMap.begin(); exactMapIt != exactMap.end(); exactMapIt++) {
      zeroMap[exactMapIt->first] = Function::zero(exactMapIt->second->rank());
    }
    
    form.solution()->projectOntoMesh(exactMap);
    form.solutionIncrement()->projectOntoMesh(zeroMap);
    
    RHSPtr rhs = form.rhs(forcingFunction, false); // false: *include* boundary terms in the RHS -- important for computing energy error correctly
    form.solutionIncrement()->setRHS(rhs);
    
    double energyError = form.solutionIncrement()->energyErrorTotal();
    
    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }
  
  TEUCHOS_UNIT_TEST( NavierStokesVGPFormulation, StokesConsistency_2D )
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
    NavierStokesVGPFormulation form(meshTopo, Re, fieldPolyOrder, delta_k, forcingFunction);
    
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
    
    double energyError = stokesSolution->energyErrorTotal();
    
    double tol = 1e-14;
    TEST_COMPARE(energyError, <, tol);
  }
} // namespace