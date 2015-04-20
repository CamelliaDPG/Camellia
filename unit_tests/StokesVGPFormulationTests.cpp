//
//  StokesVGPFormulationTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/5/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "StokesVGPFormulation.h"
#include "GDAMinimumRule.h"
#include "MeshFactory.h"
#include "Solution.h"
#include "HDF5Exporter.h"

using namespace Camellia;
using namespace Intrepid;

namespace {
  void projectExactSolution(StokesVGPFormulation &form, SolutionPtr<double> stokesSolution, FunctionPtr u, FunctionPtr p) {
    double mu = form.mu();

    FunctionPtr u1, u2, u3, sigma1, sigma2, sigma3;
    int spaceDim = stokesSolution->mesh()->getDimension();

    u1 = u->x();
    u2 = u->y();
    sigma1 = mu * u1->grad();
    sigma2 = mu * u2->grad();
    if (spaceDim==3) {
      u3 = u->z();
      sigma3 = mu * u3->grad();
    }

    LinearTermPtr t1_n_lt, t2_n_lt, t3_n_lt;
    t1_n_lt = form.tn_hat(1)->termTraced();
    t2_n_lt = form.tn_hat(2)->termTraced();
    if (spaceDim==3) {
      t3_n_lt = form.tn_hat(3)->termTraced();
    }

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.u(1)->ID()] = u1;
    exactMap[form.u(2)->ID()] = u2;
    exactMap[form.p()->ID() ] =  p;
    exactMap[form.sigma(1)->ID()] = sigma1;
    exactMap[form.sigma(2)->ID()] = sigma2;

    if (spaceDim==3) {
      exactMap[form.u(3)->ID()] = u3;
      exactMap[form.sigma(3)->ID()] = sigma3;
    }

    // fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr t1_n = t1_n_lt->evaluate(exactMap);
    FunctionPtr t2_n = t2_n_lt->evaluate(exactMap);
    exactMap[form.tn_hat(1)->ID()] = t1_n;
    exactMap[form.tn_hat(2)->ID()] = t2_n;

    // traces:
    exactMap[form.u_hat(1)->ID()] = u1;
    exactMap[form.u_hat(2)->ID()] = u2;

    if (spaceDim==3) {
      FunctionPtr t3_n = t3_n_lt->evaluate(exactMap);
      exactMap[form.tn_hat(3)->ID()] = t3_n;
      exactMap[form.u_hat(3)->ID()] = u3;
    }

    stokesSolution->projectOntoMesh(exactMap);
  }

  void setupExactSolution(StokesVGPFormulation &form, FunctionPtr u, FunctionPtr p,
                          MeshTopologyPtr meshTopo, int fieldPolyOrder, int delta_k) {
    int spaceDim = meshTopo->getSpaceDim();
    double mu = form.mu();

    FunctionPtr forcingFunction = StokesVGPFormulation::forcingFunction(spaceDim, mu, u, p);

    form.initializeSolution(meshTopo, fieldPolyOrder, delta_k, forcingFunction);

    form.addZeroMeanPressureCondition();
    form.addInflowCondition(SpatialFilter::allSpace(), u);
  }

  void testStokesConsistencySteady(int spaceDim, Teuchos::FancyOStream &out, bool &success) {
    vector<double> dimensions(spaceDim,2.0); // 2x2 square domain
    vector<int> elementCounts(spaceDim,1); // 1 x 1 mesh
    vector<double> x0(spaceDim,-1.0);
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
    double Re = 1.0;
    int fieldPolyOrder = 2, delta_k = 1;

    FunctionPtr u, p;
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    if (spaceDim == 2) {
      FunctionPtr u1 = x;
      FunctionPtr u2 = -y; // divergence 0
      u = Function::vectorize(u1,u2);
      p = x + 2. * y; // zero average
    } else if (spaceDim == 3) {
      FunctionPtr u1 = 2. * x;
      FunctionPtr u2 = -y; // divergence 0
      FunctionPtr u3 = -z;
      u = Function::vectorize(u1,u2,u3);
      p = x + 2. * y + 3. * z; // zero average
    }

    bool useConformingTraces = true;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re);

    setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);
    projectExactSolution(form, form.solution(), u, p);

    FunctionPtr pSoln = Function::solution(form.p(), form.solution());

    form.solution()->clearComputedResiduals();

    double energyError = form.solution()->energyErrorTotal();

    double tol = 1e-13;
    TEST_COMPARE(energyError, <, tol);
  }

  TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_2D_Steady )
  {
    int spaceDim = 2;
    testStokesConsistencySteady(spaceDim,out,success);
  }

  TEUCHOS_UNIT_TEST( StokesVGPFormulation, Consistency_3D_Steady )
  {
    int spaceDim = 3;
    testStokesConsistencySteady(spaceDim,out,success);
  }

  TEUCHOS_UNIT_TEST( StokesVGPFormulation, StreamFormulationConsistency ) {
    /*
      The stream formulation's psi function should be (-u2, u1).  Here, we project that solution
      onto the stream formulation, and test that the residual is 0.
     */
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
    FunctionPtr u = Function::vectorize(u1,u2);
    FunctionPtr p = y * y * y; // zero average

    bool useConformingTraces = true;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re);

    setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);

    SolutionPtr<double> streamSoln = form.streamSolution();

    // to determine phi_exact, we solve the problem:
    //   d/dx phi = -u2
    //   d/dy phi = +u1
    // subject to the constraint that its integral on the domain is zero.
    // Here, phi = xy solves it.

    FunctionPtr phi_exact = x * y;
    FunctionPtr psi_exact = Function::vectorize(-u2, u1);

    map<int, FunctionPtr> exactMap;
    // fields:
    exactMap[form.streamFormulation().phi()->ID()] = phi_exact;
    exactMap[form.streamFormulation().psi()->ID()] = psi_exact;

    VarPtr phi_hat = form.streamFormulation().phi_hat();
    VarPtr psi_n_hat = form.streamFormulation().psi_n_hat();

    // traces and fluxes:
    // use the exact field variable solution together with the termTraced to determine the flux traced
    FunctionPtr phi_hat_exact = phi_hat->termTraced()->evaluate(exactMap);
    FunctionPtr psi_n_hat_exact = psi_n_hat->termTraced()->evaluate(exactMap);
    exactMap[phi_hat->ID()] = phi_hat_exact;
    exactMap[psi_n_hat->ID()] = psi_n_hat_exact;

    streamSoln->projectOntoMesh(exactMap);

    double energyError = streamSoln->energyErrorTotal();

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
    FunctionPtr u = Function::vectorize(u1,u2);
    FunctionPtr p = y * y * y; // zero average

    bool useConformingTraces = true;
    bool transient = true;
    double dt = 1.0;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re, transient, dt);

    setupExactSolution(form, u, p, meshTopo, fieldPolyOrder, delta_k);
    projectExactSolution(form, form.solution(), u, p);
    projectExactSolution(form, form.solutionPreviousTimeStep(), u, p);

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
    FunctionPtr u = Function::vectorize(u1,u2);

//    FunctionPtr p = y * y * y; // zero average
//    FunctionPtr u1 = Function::constant(1.0);
//    FunctionPtr u2 = Function::constant(1.0);
    FunctionPtr p = x + y;

    bool useConformingTraces = true;
    StokesVGPFormulation form(spaceDim, useConformingTraces, 1.0 / Re);
    setupExactSolution(form,u,p,meshTopo,fieldPolyOrder,delta_k);

    MeshPtr stokesMesh = form.solution()->mesh();

    BFPtr stokesBF = form.bf();

    //uniform h-refinement:
    stokesMesh->hRefine(stokesMesh->getActiveCellIDs());

    form.solve();

    SolutionPtr<double> stokesProjection = Solution<double>::solution(stokesMesh);
    projectExactSolution(form, stokesProjection, u, p);

    SolutionPtr<double> stokesSolution = form.solution();
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

  TEUCHOS_UNIT_TEST( StokesVGPFormulation, SaveAndLoad ) {
    int spaceDim = 2;
    vector<double> dimensions(spaceDim); // 1x2 domain
    dimensions[0] = 1.0;
    dimensions[1] = 2.0;

    vector<int> elementCounts(spaceDim); // 3 x 2 mesh
    elementCounts[0] = 3;
    elementCounts[1] = 2;
    vector<double> x0(spaceDim,0.0);

    double mu = 1.0;
    bool useConformingTraces = true;
    StokesVGPFormulation form(spaceDim, useConformingTraces, mu);

    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

    int fieldPolyOrder = 1, delta_k = 1;

    form.initializeSolution(meshTopo, fieldPolyOrder, delta_k);

    set<GlobalIndexType> cellsToRefine;
    cellsToRefine.insert(0);
    form.solution()->mesh()->pRefine(cellsToRefine);

    string savePrefix = "stokesBackstep";
    form.save(savePrefix);

    StokesVGPFormulation loadedForm(spaceDim,useConformingTraces, mu);
    loadedForm.initializeSolution(savePrefix,fieldPolyOrder,delta_k);

    GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*> (loadedForm.solution()->mesh()->globalDofAssignment().get());

    minRule->printConstraintInfo(0);

    loadedForm.solution()->mesh()->pRefine(cellsToRefine);
  }

} // namespace
