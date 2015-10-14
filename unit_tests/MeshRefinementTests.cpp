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
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
TEUCHOS_UNIT_TEST( MeshRefinement, TraceTermProjection )
{
  int spaceDim = 2;

  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  FunctionPtr phi_exact; // want an exactly representable solution with non-trivial psi_n (i.e. grad phi dot n should be non-trivial)
  // for now, we just go very simple.  Linear in x,y,z.
  switch (spaceDim)
  {
  case 1:
    phi_exact = x;
    break;
  case 2:
    phi_exact = x + y;
    break;
  case 3:
    phi_exact = x + y + z;
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
  VarPtr phi = pf.phi();
  VarPtr psi = pf.psi();

  // traces
  VarPtr phi_hat = pf.phi_hat();
  VarPtr psi_n = pf.psi_n_hat();

  // tests
  VarPtr tau = pf.tau();
  VarPtr q = pf.q();

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

  // rhs = f * q, where f = \Delta phi
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f;
  switch (spaceDim)
  {
  case 1:
    f = phi_exact->dx()->dx();
    break;
  case 2:
    f = phi_exact->dx()->dx() + phi_exact->dy()->dy();
    break;
  case 3:
    f = phi_exact->dx()->dx() + phi_exact->dy()->dy() + phi_exact->dz()->dz();
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled spaceDim");
    break;
  }
  rhs->addTerm(f * q);

  IPPtr graphNorm = bf->graphNorm();

  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  SolutionPtr solution;

  bc->addDirichlet(phi_hat, boundary, phi_exact);
  solution = Solution::solution(mesh, bc, rhs, graphNorm);

  // DEBUGGING:
//    solution->setWriteMatrixToMatrixMarketFile(true, "/tmp/traceTermProjectionMatrix.dat");
//    solution->setWriteRHSToMatrixMarketFile(true, "/tmp/traceTermProjection_RHS.dat");
//
//    SolverPtr superLUSolver = Solver::getSolver(Solver::SuperLUDist, false);

  solution->solve();

  FunctionPtr psi_exact = (spaceDim > 1) ? phi_exact->grad() : phi_exact->dx();

  map<int, FunctionPtr> psiMap;
  psiMap[psi->ID()] = psi_exact;

  FunctionPtr psi_n_exact = psi_n->termTraced()->evaluate(psiMap);

  FunctionPtr psi_soln = Function::solution(psi, solution);
  FunctionPtr psi_n_soln = Function::solution(psi_n, solution, false); // false: don't weight fluxes by parity
  FunctionPtr phi_hat_soln = Function::solution(phi_hat, solution);

  FunctionPtr psi_err = psi_exact - psi_soln;
  FunctionPtr psi_n_err = psi_n_exact - psi_n_soln;
  FunctionPtr phi_hat_err = phi_exact - phi_hat_soln;

  double err_L2 = psi_err->l2norm(mesh);

  double tol = 1e-12;

  // SANITY CHECKS ON INITIAL SOLUTION
  // psi error first
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection error: psi in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;

    double soln_l2 = psi_soln->l2norm(mesh);
    double exact_l2 = psi_exact->l2norm(mesh);

    cout << "L^2 norm of exact solution: " << exact_l2 << ", versus " << soln_l2 << " for initial solution\n";
  }

  // psi_n error:
  err_L2 = psi_n_err->l2norm(mesh);
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection error: psi_n in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  // phi_hat error:
  err_L2 = phi_hat_err->l2norm(mesh);
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection error: phi_hat in initial solution (prior to projection) differs from exact solution by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  // do a uniform refinement, then check that psi_n_soln and phi_hat_soln match the exact
  CellPtr sampleCell = mesh->getTopology()->getCell(0);
  CellTopoPtr cellTopo = sampleCell->topology();
  RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo->getKey());

  mesh->registerSolution(solution); // this way, solution will get the memo to project
  mesh->hRefine(mesh->getActiveCellIDs(), refPattern);

  err_L2 = phi_hat_err->l2norm(mesh);
  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection failure: projected phi_hat differs from exact by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  err_L2 = psi_n_err->l2norm(mesh);

  if (err_L2 > tol)
  {
    cout << "testTraceTermProjection failure: projected psi_n differs from exact by " << err_L2 << " in L^2 norm.\n";
    success = false;
  }

  if (success==false)   // then export
  {
#ifdef HAVE_EPETRAEXT_HDF5
    HDF5Exporter solnExporter(mesh, "soln", "/tmp");
    VarFactoryPtr vf = bf->varFactory();
    solnExporter.exportSolution(solution, 0, 10);

    HDF5Exporter fxnExporter(mesh, "fxn");
    vector<string> fxnNames;
    vector< FunctionPtr > fxns;
    // fields:
    fxnNames.push_back("psi_exact");
    fxns.push_back(psi_exact);
    fxnNames.push_back("psi_soln");
    fxns.push_back(psi_soln);

    fxnNames.push_back("psi_exact_x");
    fxns.push_back(psi_exact->x());
    fxnNames.push_back("psi_soln_x");
    fxns.push_back(psi_soln->x());

    fxnNames.push_back("psi_exact_y");
    fxns.push_back(psi_exact->y());
    fxnNames.push_back("psi_soln_y");
    fxns.push_back(psi_soln->y());
    fxnExporter.exportFunction(fxns, fxnNames, 0, 10);

    // traces:
    fxnNames.clear();
    fxns.clear();
    fxnNames.push_back("psi_n_exact");
    fxns.push_back(psi_n_exact);
    fxnNames.push_back("psi_n_soln");
    fxns.push_back(psi_n_soln);
    fxnNames.push_back("psi_n_err");
    fxns.push_back(psi_n_err);
    fxnExporter.exportFunction(fxns, fxnNames, 0, 10);
#endif
  }
}
} // namespace
