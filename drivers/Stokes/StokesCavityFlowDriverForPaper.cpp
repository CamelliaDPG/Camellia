#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

#include "CGSolver.h"
#include "MLSolver.h"

#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
#include <xmmintrin.h>
#endif

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif


class TopBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    return (abs(y-1.0) < tol);
  }
  
  bool matchesPoint(double x, double y, double z) {
    return matchesPoint(x, y);
  }
};

class RampBoundaryFunction_U1 : public SimpleFunction {
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    if ( (abs(x) < _eps) ) { // top left
      return x / _eps;
    } else if ( abs(1.0-x) < _eps) { // top right
      return (1.0-x) / _eps;
    } else { // top middle
      return 1;
    }
  }
  double value(double x, double y, double z) {
    // bilinear interpolation with ramp of width _eps around top edges
    double xFactor = 1.0;
    double zFactor = 1.0;
    if ( (abs(x) < _eps) ) { // top left
      xFactor = x / _eps;
    } else if ( abs(1.0-x) < _eps) { // top right
      xFactor = (1.0-x) / _eps;
    }
    if ( (abs(z) < _eps) ) { // top back
      zFactor = z / _eps;
    } else if ( abs(1.0-z) < _eps) { // top front
      zFactor = (1.0-z) / _eps;
    }
    return xFactor * zFactor;
  }
};

int main(int argc, char *argv[]) {
#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();
  
  bool use3D = false;
  int refCount = 10;
  
  int k = 4; // poly order for field variables
  int H1Order = k + 1;
  int delta_k = use3D ? 3 : 2;   // test space enrichment
  
  bool useMinRule = false;
  bool useMumps = false;
  bool useCGSolver = false;
  bool useMLSolver = true;
  bool clearSolution = false;
  
  bool enforceOneIrregularity = true;
  
  bool conformingTraces = true;
  
  bool testing_2_irregularity = false;
  
  VarFactory varFactory;
  // fields:
  VarPtr u1 = varFactory.fieldVar("u_1", L2);
  VarPtr u2 = varFactory.fieldVar("u_2", L2);
  VarPtr u3;
  if (use3D) u3 = varFactory.fieldVar("u_3", L2);
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2", VECTOR_L2);
  VarPtr sigma3;
  if (use3D) sigma3 = varFactory.fieldVar("\\sigma_3", VECTOR_L2);
  VarPtr p = varFactory.fieldVar("p");
  
  FunctionPtr n = Function::normal();
  // traces:
  VarPtr u1hat, u2hat, u3hat;
  
  if (conformingTraces) {
    u1hat = varFactory.traceVar("\\widehat{u}_1", u1);
    u2hat = varFactory.traceVar("\\widehat{u}_2", u2);
    
    if (use3D) {
      u3hat = varFactory.traceVar("\\widehat{u}_3", u3);
    }
  } else {
    cout << "Note: using non-conforming traces.\n";
    u1hat = varFactory.traceVar("\\widehat{u}_1", u1, L2);
    u2hat = varFactory.traceVar("\\widehat{u}_2", u2, L2);
    
    if (use3D) {
      u3hat = varFactory.traceVar("\\widehat{u}_3", u3, L2);
    }
  }
  //  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1", L2); // switched to L2 just to isolate/debug
  //  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2", L2);
  //  cout << "WARNING/NOTE: for debugging purposes, temporarily switching traces to use L^2 discretizations (i.e. they are not conforming, and they are of lower order than they should be).\n";
  VarPtr t1_n = varFactory.fluxVar("\\widehat{t}_{1n}", sigma1 * n + p * n->x());
  VarPtr t2_n = varFactory.fluxVar("\\widehat{t}_{2n}", sigma2 * n + p * n->y());
  VarPtr t3_n;
  if (use3D) {
    t3_n = varFactory.fluxVar("\\widehat{t}_{3n}", sigma3 * n + p * n->z());
  }
  
  // test functions:
  VarPtr tau1 = varFactory.testVar("\\tau_1", HDIV);  // tau_1
  VarPtr tau2 = varFactory.testVar("\\tau_2", HDIV);  // tau_2
  VarPtr tau3;
  if (use3D) tau3 = varFactory.testVar("\\tau_3", HDIV);  // tau_3
  VarPtr v1 = varFactory.testVar("v1", HGRAD);        // v_1
  VarPtr v2 = varFactory.testVar("v2", HGRAD);        // v_2
  VarPtr v3;
  if (use3D) v3 = varFactory.testVar("v3", HGRAD);
  VarPtr q = varFactory.testVar("q", HGRAD);          // q
  
  BFPtr stokesBF = Teuchos::rcp( new BF(varFactory) );
  double mu = 1.0; // viscosity
  // tau1 terms:
  stokesBF->addTerm(u1, tau1->div());
  stokesBF->addTerm(sigma1, tau1); // (sigma1, tau1)
  stokesBF->addTerm(-u1hat, tau1->dot_normal());
  
  // tau2 terms:
  stokesBF->addTerm(u2, tau2->div());
  stokesBF->addTerm(sigma2, tau2);
  stokesBF->addTerm(-u2hat, tau2->dot_normal());
  
  // tau3:
  if (use3D) {
    stokesBF->addTerm(u3, tau3->div());
    stokesBF->addTerm(sigma3, tau3);
    stokesBF->addTerm(-u3hat, tau3->dot_normal());
  }
  
  // v1:
  stokesBF->addTerm(mu * sigma1, v1->grad()); // (mu sigma1, grad v1)
  stokesBF->addTerm( - p, v1->dx() );
  stokesBF->addTerm( t1_n, v1);
  
  // v2:
  stokesBF->addTerm(mu * sigma2, v2->grad()); // (mu sigma2, grad v2)
  stokesBF->addTerm( - p, v2->dy());
  stokesBF->addTerm( t2_n, v2);
  
  // v3:
  if (use3D) {
    stokesBF->addTerm(mu * sigma3, v3->grad()); // (mu sigma3, grad v3)
    stokesBF->addTerm( - p, v3->dz());
    stokesBF->addTerm( t3_n, v3);
  }
  
  // q:
  stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBF->addTerm(-u2,q->dy());
  if (use3D) stokesBF->addTerm(-u3, q->dz());
  if (!use3D) stokesBF->addTerm(u1hat * n->x() + u2hat * n->y(), q);
  else stokesBF->addTerm(u1hat * n->x() + u2hat * n->y() + u3hat * n->z(), q);
  
  double width = 1.0, height = 1.0, depth = 1.0;
  int horizontalCells = 2, verticalCells = 2, depthCells = 2;
  
  vector<double> domainDimensions;
  domainDimensions.push_back(width);
  domainDimensions.push_back(height);
  
  vector<int> elementCounts;
  elementCounts.push_back(horizontalCells);
  elementCounts.push_back(verticalCells);
  
  if (use3D) {
    domainDimensions.push_back(depth);
    elementCounts.push_back(depthCells);
  }
  
  MeshPtr mesh;
  
  if (!use3D) {
    mesh = useMinRule ? MeshFactory::quadMeshMinRule(stokesBF, H1Order, delta_k, width, height,
                                                     horizontalCells, verticalCells)
                            : MeshFactory::quadMesh(stokesBF, H1Order, delta_k, width, height,
                                                    horizontalCells, verticalCells);
  } else {
    mesh = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, H1Order, delta_k);
  }
  
  RHSPtr rhs = RHS::rhs(); // zero
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr topBoundary = Teuchos::rcp( new TopBoundary );
  SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);
  
  // top boundary:
  FunctionPtr u1_bc_fxn = Teuchos::rcp( new RampBoundaryFunction_U1(1.0/64.0) );
  FunctionPtr zero = Function::zero();
  bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
  bc->addDirichlet(u2hat, topBoundary, zero);
  if (use3D) bc->addDirichlet(u3hat, topBoundary, zero);
  
  // everywhere else:
  bc->addDirichlet(u1hat, otherBoundary, zero);
  bc->addDirichlet(u2hat, otherBoundary, zero);
  if (use3D) bc->addDirichlet(u3hat, otherBoundary, zero);
  
  bc->addZeroMeanConstraint(p);

  IPPtr graphNorm = stokesBF->graphNorm();
  
  if (testing_2_irregularity) {
    enforceOneIrregularity = false;
    set<unsigned> cellsToRefine;
    cellsToRefine.insert(1);
    cellsToRefine.insert(3);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    cellsToRefine.clear();
    cellsToRefine.insert(7);
    cellsToRefine.insert(10);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    cellsToRefine.clear();
    cellsToRefine.insert(14);
    cellsToRefine.insert(15);
    cellsToRefine.insert(18);
    cellsToRefine.insert(19);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    refCount = 0;
  }
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  mesh->registerSolution(solution); // sign up for projection of old solution onto refined cells.
  
  bool debuggingCrashInMultiBasis = false;
  if (debuggingCrashInMultiBasis) {
    // some initial refinements to get us to the issue in the debugger faster:
    vector<GlobalIndexType> cellsToRefine;
    cellsToRefine.push_back(1);
    cellsToRefine.push_back(3);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    cellsToRefine.clear();
    cellsToRefine.push_back(7);
    cellsToRefine.push_back(10);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    cellsToRefine.clear();
    cellsToRefine.push_back(15);
    cellsToRefine.push_back(18);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    cellsToRefine.clear();
    cellsToRefine.push_back(23);
    cellsToRefine.push_back(26);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    cellsToRefine.clear();
    cellsToRefine.push_back(31);
    cellsToRefine.push_back(34);
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
//    // set up zero solution data (to trigger bona fide projection on refinement)
//    set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
//    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
//      GlobalIndexType cellID = *cellIDIt;
//      DofOrderingPtr trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
//      FieldContainer<double> zeroCoefficients(trialOrdering->totalDofs());
//      solution->setSolnCoeffsForCellID(zeroCoefficients, cellID);
//    }
//    
//    cellsToRefine.clear();
//    cellsToRefine.push_back(38);
//    cellsToRefine.push_back(39);
//    cellsToRefine.push_back(42);
//    cellsToRefine.push_back(43);
//    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
  }
  
  
  double energyThreshold = 0.2;
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  refinementStrategy.setReportPerCellErrors(true);
  refinementStrategy.setEnforceOneIrregularity(enforceOneIrregularity);
  
  Teuchos::RCP<Solver> coarseSolver, fineSolver;
  if (useMumps) {
#ifdef USE_MUMPS
    coarseSolver = Teuchos::rcp( new MumpsSolver );
#else
    cout << "useMumps=true, but MUMPS is not available!\n";
    exit(0);
#endif
  } else {
    coarseSolver = Teuchos::rcp( new KluSolver );
  }
  if (useCGSolver) {
    int maxIters = 80000;
    double tol = 1e-6;
    fineSolver = Teuchos::rcp( new CGSolver(maxIters, tol) );
  } else if (useMLSolver) {
    int maxIters = 80000;
    double tol = 1e-6;
    fineSolver = Teuchos::rcp( new MLSolver(tol, maxIters) );
  } else {
    fineSolver = coarseSolver;
  }
  solution->solve(coarseSolver);
  solution->reportTimings();
  for (int refIndex=0; refIndex < refCount; refIndex++) {
    double energyError = solution->energyErrorTotal();
    if (rank==0) {
      cout << "Before refinement " << refIndex << ", energy error = " << energyError;
      cout << " (using " << mesh->numFluxDofs() << " trace degrees of freedom)." << endl;
    }
    refinementStrategy.refine(rank==0);
    
#ifdef HAVE_EPETRAEXT_HDF5
    ostringstream dir_name;
    dir_name << "stokesCavityFlow_k" << k << "_after_ref" << refIndex << "_projection";
    bool deleteOldFiles = false;
    HDF5Exporter exporter(mesh,dir_name.str(),deleteOldFiles);
    exporter.exportSolution(solution,varFactory,0);
#endif
    
    if (!use3D) {
      GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/stokesMesh", mesh, true); // true: label cells
    }
    
//    solution->setWriteMatrixToFile(true, "/tmp/stiffness.dat");
//    solution->setWriteRHSToMatrixMarketFile(true, "/tmp/rhs.dat");
    
    if (clearSolution) solution->clear();
    
    solution->solve(fineSolver);
    solution->reportTimings();
#ifdef HAVE_EPETRAEXT_HDF5
    dir_name.clear();
    dir_name << "stokesCavityFlow_k" << k << "_ref" << refIndex;
    HDF5Exporter exporter2(mesh,dir_name.str(),deleteOldFiles);
    exporter2.exportSolution(solution,varFactory,0);
#endif
  }
  double energyErrorTotal = solution->energyErrorTotal();
  
//  cout << "Final Mesh, entities report:\n";
//  solution->mesh()->getTopology()->printAllEntities();
  
  FunctionPtr massFlux;
  if (!use3D) massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat * n->x() + u2hat * n->y()) );
  else massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat * n->x() + u2hat * n->y() + u3hat * n->z()) );
  double netMassFlux = massFlux->integrate(mesh); // integrate over the mesh skeleton

  if (rank==0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numFluxDofs() << " trace dofs (";
    cout << mesh->numGlobalDofs() << " total dofs, including fields).\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "Net mass flux: " << netMassFlux << endl;
  }
  
#ifdef HAVE_EPETRAEXT_HDF5
  ostringstream dir_name;
  dir_name << "stokesCavityFlow_k" << k << "_final";
  bool deleteOldFiles = false;
  HDF5Exporter exporter(mesh,dir_name.str(),deleteOldFiles);
  exporter.exportSolution(solution,varFactory,0);
#endif
  
  if (!use3D) {
#ifdef USE_VTK
    VTKExporter solnExporter(solution,mesh,varFactory);
    solnExporter.exportSolution("stokesCavityFlowSolution");
#endif
    
    GnuPlotUtil::writeComputationalMeshSkeleton("cavityFlowRefinedMesh", mesh);
  } else {
#ifdef USE_VTK
    NewVTKExporter exporter(mesh->getTopology());
    FunctionPtr u1_soln = Function::solution(u1, solution);
    FunctionPtr u2_soln = Function::solution(u2, solution);
    FunctionPtr u3_soln = Function::solution(u3, solution);
    exporter.exportFunction(u1_soln, "u1_soln");
    exporter.exportFunction(u2_soln, "u2_soln");
    exporter.exportFunction(u3_soln, "u3_soln");
#endif
  }
  
  return 0;
}
