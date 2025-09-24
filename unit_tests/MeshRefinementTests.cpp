//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  MeshRefinementTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/29/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BC.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "Mesh.h"
#include "MeshFactory.h"
#include "MPIWrapper.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( MeshRefinement, TraceTermProjection )
{
  MPIWrapper::CommWorld()->Barrier();
  int spaceDim = 2;

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  FunctionPtr u_exact; // want an exactly representable solution with non-trivial sigma_n (i.e. grad u dot n should be non-trivial)
  // for now, we just go very simple.  Linear in x,y,z.
  switch (spaceDim)
  {
  case 1:
    u_exact = x;
    break;
  case 2:
    u_exact = x + y;
    break;
  case 3:
    u_exact = x + y + z;
    break;
  default:
    cout << "MeshRefinementTests::testTraceTermProjection(): unhandled space dimension.\n";
    break;
  }

  //  int H1Order = 5; // debugging
  int H1Order = 2; // so field order is linear

  bool useConformingTraces = true;
  PoissonFormulation pf(spaceDim,useConformingTraces);

  BFPtr bf = pf.bf();

  // fields
  VarPtr u = pf.u();
  VarPtr sigma = pf.sigma();

  // traces
  VarPtr u_hat = pf.u_hat();
  VarPtr sigma_n = pf.sigma_n_hat();

  // tests
  VarPtr tau = pf.tau();
  VarPtr v = pf.v();

  int testSpaceEnrichment = 1; //
  //  double width = 1.0, height = 1.0, depth = 1.0;

  vector<double> dimensions;
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(1.0);
  }

  //  cout << "dimensions[0] = " << dimensions[0] << "; dimensions[1] = " << dimensions[1] << endl;
  //  cout << "numCells[0] = " << numCells[0] << "; numCells[1] = " << numCells[1] << endl;

  vector<int> numCells(spaceDim, 1); // one element in each spatial direction

  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, numCells, H1Order, testSpaceEnrichment);

  // rhs = f * v, where f = \Delta u
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f;
  switch (spaceDim)
  {
  case 1:
    f = u_exact->dx()->dx();
    break;
  case 2:
    f = u_exact->dx()->dx() + u_exact->dy()->dy();
    break;
  case 3:
    f = u_exact->dx()->dx() + u_exact->dy()->dy() + u_exact->dz()->dz();
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled spaceDim");
  }
  rhs->addTerm(f * v);

  IPPtr graphNorm = bf->graphNorm();

  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  SolutionPtr solution;

  bc->addDirichlet(u_hat, boundary, u_exact);
  solution = Solution::solution(mesh, bc, rhs, graphNorm);

  solution->solve();

  FunctionPtr sigma_exact = (spaceDim > 1) ? u_exact->grad() : u_exact->dx();

  FunctionPtr traceFxn = Function::meshSkeletonCharacteristic();
  map<int, FunctionPtr> sigmaMap;
  sigmaMap[sigma->ID()] = sigma_exact;

  FunctionPtr sigma_n_exact = sigma_n->termTraced()->evaluate(sigmaMap);

  FunctionPtr sigma_soln = Function::solution(sigma, solution);
  FunctionPtr sigma_n_soln = Function::solution(sigma_n, solution, false); // false: don't weight fluxes by parity
  FunctionPtr u_hat_soln = Function::solution(u_hat, solution);

  FunctionPtr sigma_err = sigma_exact - sigma_soln;
  FunctionPtr sigma_n_err = sigma_n_exact - sigma_n_soln;
  FunctionPtr u_hat_err = u_exact * traceFxn - u_hat_soln;

  double err_L2 = sigma_err->l2norm(mesh);

  double tol = 1e-12;

  // SANITY CHECKS ON INITIAL SOLUTION
  // sigma error first
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection error: sigma in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;

    double soln_l2 = sigma_soln->l2norm(mesh);
    double exact_l2 = sigma_exact->l2norm(mesh);

    cout << "L^2 norm of exact solution: " << exact_l2 << ", versus " << soln_l2 << " for initial solution\n";
  }

  // sigma_n error:
  err_L2 = sigma_n_err->l2norm(mesh);
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection error: sigma_n in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  // u_hat error:
  err_L2 = u_hat_err->l2norm(mesh);
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection error: u_hat in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  // do a uniform refinement, then check that sigma_n_soln and u_hat_soln match the exact
  mesh->registerSolution(solution); // this way, solution will get the memo to project
  mesh->hRefine(mesh->getActiveCellIDsGlobal());

  err_L2 = u_hat_err->l2norm(mesh);
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection failure: projected u_hat differs from exact by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  err_L2 = sigma_n_err->l2norm(mesh);

  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection failure: projected sigma_n differs from exact by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  if (success==false)   // then export
  {
    HDF5Exporter solnExporter(mesh, "soln", "/tmp");
    VarFactoryPtr vf = bf->varFactory();
    solnExporter.exportSolution(solution, 0, 10);

    HDF5Exporter fxnExporter(mesh, "fxn");
    vector<string> fxnNames;
    vector< FunctionPtr > fxns;
    // fields:
    fxnNames.push_back("sigma_exact");
    fxns.push_back(sigma_exact);
    fxnNames.push_back("sigma_soln");
    fxns.push_back(sigma_soln);

    fxnNames.push_back("sigma_exact_x");
    fxns.push_back(sigma_exact->x());
    fxnNames.push_back("sigma_soln_x");
    fxns.push_back(sigma_soln->x());

    fxnNames.push_back("sigma_exact_y");
    fxns.push_back(sigma_exact->y());
    fxnNames.push_back("sigma_soln_y");
    fxns.push_back(sigma_soln->y());
    fxnExporter.exportFunction(fxns, fxnNames, 0, 10);

    // traces:
    fxnNames.clear();
    fxns.clear();
    fxnNames.push_back("sigma_n_exact");
    fxns.push_back(sigma_n_exact);
    fxnNames.push_back("sigma_n_soln");
    fxns.push_back(sigma_n_soln);
    fxnNames.push_back("sigma_n_err");
    fxns.push_back(sigma_n_err);
    fxnExporter.exportFunction(fxns, fxnNames, 0, 10);
  }
}
} // namespace
