#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "CGSolver.h"
#include "MLSolver.h"
#include "GMGSolver.h"

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
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int refCount = 1;
  
  int k = 1; // poly order for field variables
  int delta_k = 3;   // test space enrichment
  
  bool useMumps = true;
  bool useCGSolver = false;
  bool useMLSolver = false;
  bool useGMGSolver = false;
  bool clearSolution = false;
  
  bool enforceOneIrregularity = true;
  
  bool conformingTraces = true;
  
  double width = 1.0, height = 1.0, depth = 1.0;
  int horizontalCells = 2, verticalCells = 2, depthCells = 2;
  
  int numCells = -1;
  
  double eps = 1.0/64.0;
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numCells",&numCells,"number of cells in x/y/z directions");
  cmdp.setOption("useMumps", "useKLU", &useMumps, "use MUMPS (if available)");
  cmdp.setOption("numRefs",&refCount,"number of refinements");
  cmdp.setOption("eps", &eps, "ramp width (set to 0 for no ramp in BCs)");
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  int H1Order = k + 1;

  if (numCells != -1) {
    horizontalCells = numCells;
    verticalCells = numCells;
    depthCells = numCells;
  }
  
  VarFactory varFactory;
  // fields:
  VarPtr u1 = varFactory.fieldVar("u_1", L2);
  VarPtr u2 = varFactory.fieldVar("u_2", L2);
  VarPtr u3 = varFactory.fieldVar("u_3", L2);
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2", VECTOR_L2);
  VarPtr sigma3 = varFactory.fieldVar("\\sigma_3", VECTOR_L2);
  VarPtr p = varFactory.fieldVar("p");
  
  FunctionPtr n = Function::normal();
  // traces:
  VarPtr u1hat, u2hat, u3hat;
  
  if (conformingTraces) {
    u1hat = varFactory.traceVar("\\widehat{u}_1", u1);
    u2hat = varFactory.traceVar("\\widehat{u}_2", u2);
    u3hat = varFactory.traceVar("\\widehat{u}_3", u3);
  } else {
    cout << "Note: using non-conforming traces.\n";
    u1hat = varFactory.traceVar("\\widehat{u}_1", u1, L2);
    u2hat = varFactory.traceVar("\\widehat{u}_2", u2, L2);
    u3hat = varFactory.traceVar("\\widehat{u}_3", u3, L2);
  }
  //  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1", L2); // switched to L2 just to isolate/debug
  //  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2", L2);
  //  cout << "WARNING/NOTE: for debugging purposes, temporarily switching traces to use L^2 discretizations (i.e. they are not conforming, and they are of lower order than they should be).\n";
  VarPtr t1_n = varFactory.fluxVar("\\widehat{t}_{1n}", sigma1 * n + p * n->x());
  VarPtr t2_n = varFactory.fluxVar("\\widehat{t}_{2n}", sigma2 * n + p * n->y());
  VarPtr t3_n = varFactory.fluxVar("\\widehat{t}_{3n}", sigma3 * n + p * n->z());
  
  // test functions:
  VarPtr tau1 = varFactory.testVar("\\tau_1", HDIV);  // tau_1
  VarPtr tau2 = varFactory.testVar("\\tau_2", HDIV);  // tau_2
  VarPtr tau3 = varFactory.testVar("\\tau_3", HDIV);  // tau_3
  VarPtr v1 = varFactory.testVar("v1", HGRAD);        // v_1
  VarPtr v2 = varFactory.testVar("v2", HGRAD);        // v_2
  VarPtr v3 = varFactory.testVar("v3", HGRAD);
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
  stokesBF->addTerm(u3, tau3->div());
  stokesBF->addTerm(sigma3, tau3);
  stokesBF->addTerm(-u3hat, tau3->dot_normal());
  
  // v1:
  stokesBF->addTerm(mu * sigma1, v1->grad()); // (mu sigma1, grad v1)
  stokesBF->addTerm( - p, v1->dx() );
  stokesBF->addTerm( t1_n, v1);
  
  // v2:
  stokesBF->addTerm(mu * sigma2, v2->grad()); // (mu sigma2, grad v2)
  stokesBF->addTerm( - p, v2->dy());
  stokesBF->addTerm( t2_n, v2);
  
  // v3:
  stokesBF->addTerm(mu * sigma3, v3->grad()); // (mu sigma3, grad v3)
  stokesBF->addTerm( - p, v3->dz());
  stokesBF->addTerm( t3_n, v3);
  
  // q:
  stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBF->addTerm(-u2,q->dy());
  stokesBF->addTerm(-u3, q->dz());
  stokesBF->addTerm(u1hat * n->x() + u2hat * n->y() + u3hat * n->z(), q);
  
  vector<double> domainDimensions;
  domainDimensions.push_back(width);
  domainDimensions.push_back(height);
  domainDimensions.push_back(depth);
  
  vector<int> elementCounts;
  elementCounts.push_back(horizontalCells);
  elementCounts.push_back(verticalCells);
  elementCounts.push_back(depthCells);
  
  MeshPtr mesh, coarseMesh;
  
  Epetra_Time timer(Comm);
  mesh = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, H1Order, delta_k);
  coarseMesh = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, H1Order, delta_k);
  double meshConstructionTime = timer.ElapsedTime();
  cout << "On rank " << rank << ", mesh construction time: " << meshConstructionTime << endl;
  
  RHSPtr rhs = RHS::rhs(); // zero
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr topBoundary = Teuchos::rcp( new TopBoundary );
  SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);
  
  // top boundary:
  FunctionPtr u1_bc_fxn = Teuchos::rcp( new RampBoundaryFunction_U1(eps) );
  FunctionPtr zero = Function::zero();
  bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
  bc->addDirichlet(u2hat, topBoundary, zero);
  bc->addDirichlet(u3hat, topBoundary, zero);
  
  // everywhere else:
  bc->addDirichlet(u1hat, otherBoundary, zero);
  bc->addDirichlet(u2hat, otherBoundary, zero);
  bc->addDirichlet(u3hat, otherBoundary, zero);
  
  bc->addSinglePointBC(p->ID(), Function::zero());

  IPPtr graphNorm = stokesBF->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  mesh->registerSolution(solution); // sign up for projection of old solution onto refined cells.
  
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
  } else if (useGMGSolver) {
    double tol = 1e-6;
    int maxIters = 80000;
    BCPtr zeroBCs = bc->copyImposingZero();
    // as a test, do "multi" grid between fine and fine meshes.
    fineSolver = Teuchos::rcp( new GMGSolver(zeroBCs, mesh, graphNorm, mesh,
                                             solution->getPartitionMap(), maxIters, tol, coarseSolver) );

//    fineSolver = Teuchos::rcp( new GMGSolver(zeroBCs, coarseMesh, graphNorm, mesh,
//                                             solution->getPartitionMap(), maxIters, tol, coarseSolver) );

  } else {
    fineSolver = coarseSolver;
  }
  if (rank==0) cout << "About to start solve.\n";
  timer.ResetStartTime();
  solution->solve(coarseSolver);
  double totalSolveTime = timer.ElapsedTime();
  if (rank==0) cout << "total solve time (as seen by rank 0) " << totalSolveTime << " seconds.\n";
  solution->reportTimings();
  
#ifdef HAVE_EPETRAEXT_HDF5
  if (rank==0) cout << "Beginning export of initial solution.\n";
  ostringstream dir_name;
  dir_name << "stokesCavityFlow3D_k" << k << "_ref" << 0;
  HDF5Exporter exporter2(mesh,dir_name.str());
  exporter2.exportSolution(solution,varFactory,0);
  if (rank==0) cout << "...completed.  Beginning export of initial mesh.\n";
  // straight-line mesh
  dir_name.str("");
  dir_name << "stokesCavityFlow3D_k" << k << "_ref" << 0 << "_mesh";
  HDF5Exporter meshExporter(mesh,dir_name.str());
  meshExporter.exportFunction(Function::normal(),"mesh",0,2);
  if (rank==0) cout << "...completed.\n";
#endif
  
  for (int refIndex=0; refIndex < refCount; refIndex++) {
    double energyError = solution->energyErrorTotal();
    int numFluxDofs = mesh->numFluxDofs();
    if (rank==0) {
      cout << "Before refinement " << refIndex << ", energy error = " << energyError;
      cout << " (using " << numFluxDofs << " trace degrees of freedom)." << endl;
    }
    refinementStrategy.refine(rank==0);
    
    if (clearSolution) solution->clear();
    
    solution->solve(fineSolver);
    solution->reportTimings();
#ifdef HAVE_EPETRAEXT_HDF5
    if (rank==0) cout << "Beginning export of refinement " << refIndex << " solution.\n";
    dir_name.str("");
    dir_name << "stokesCavityFlow3D_k" << k << "_ref" << refIndex;
    HDF5Exporter exporter2(mesh,dir_name.str());
    exporter2.exportSolution(solution,varFactory,0);
    if (rank==0) cout << "...completed.  Beginning export of refinement " << refIndex << " mesh.\n";
    // straight-line mesh
    dir_name.str("");
    dir_name << "stokesCavityFlow3D_k" << k << "_ref" << refIndex << "_mesh";
    HDF5Exporter meshExporter(mesh,dir_name.str());
    meshExporter.exportFunction(Function::normal(),"mesh",0,2);
    if (rank==0) cout << "Finished export of refinement " << refIndex << " solution.\n";
#endif
  }
  if (rank==0) cout << "Beginning computation of energy error.\n";
  double energyErrorTotal = solution->energyErrorTotal();
  if (rank==0) cout << "...completed.\n";
  
//  cout << "Final Mesh, entities report:\n";
//  solution->mesh()->getTopology()->printAllEntities();
  
  if (rank==0) cout << "Beginning computation of mass flux.\n";
  FunctionPtr massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat * n->x() + u2hat * n->y() + u3hat * n->z()) );
  double netMassFlux = massFlux->integrate(mesh); // integrate over the mesh skeleton
  if (rank==0) cout << "...completed.\n";

  int numFluxDofs = mesh->numFluxDofs();
  if (rank==0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << mesh->numGlobalDofs() << " total dofs, including fields).\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "Net mass flux: " << netMassFlux << endl;
  }
  
#ifdef HAVE_EPETRAEXT_HDF5
  if (rank==0) cout << "Beginning export of final solution.\n";
  dir_name.str("");
  dir_name << "stokesCavityFlow3D_k" << k << "_final";
  HDF5Exporter exporter(mesh,dir_name.str());
  exporter.exportSolution(solution,varFactory,0);
  // straight-line mesh
  if (rank==0) cout << "...completed.  Beginning export of final mesh.\n";
  dir_name.str("");
  dir_name << "stokesCavityFlow3D_k" << k << "_finalMesh";
  HDF5Exporter meshExporter2(mesh,dir_name.str());
  meshExporter2.exportFunction(Function::normal(),"mesh",0,2);
  if (rank==0) cout << "...completed.\n";
#endif
  
//#ifdef USE_VTK
//    NewVTKExporter vtkExporter(mesh->getTopology());
//    FunctionPtr u1_soln = Function::solution(u1, solution);
//    FunctionPtr u2_soln = Function::solution(u2, solution);
//    FunctionPtr u3_soln = Function::solution(u3, solution);
//    vtkExporter.exportFunction(u1_soln, "u1_soln");
//    vtkExporter.exportFunction(u2_soln, "u2_soln");
//    vtkExporter.exportFunction(u3_soln, "u3_soln");
//#endif
  
  return 0;
}
