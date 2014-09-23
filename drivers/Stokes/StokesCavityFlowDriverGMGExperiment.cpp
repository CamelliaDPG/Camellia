#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

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

#include "GlobalDofAssignment.h"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

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
  
  bool use3D = false;
  int refCount = 10;
  
  int k = 4; // poly order for field variables
  int delta_k = use3D ? 3 : 2;   // test space enrichment
  int k_coarse = 0;
  
  bool useMumps = true;
  bool useGMGSolver = true;
  
  bool enforceOneIrregularity = true;
  bool conformingTraces = false;
  bool applyDiagonalSmoothing = true;
  
  bool printRefinementDetails = false;
  
  bool useWeightedGraphNorm = true; // graph norm scaled according to units, more or less
  
  int numCells = 2;
  
  double eps = 1.0 / 64.0;
  
  int AztecOutputLevel = 1;
  int gmgMaxIterations = 200;
  double gmgTolerance = 1e-6;
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("k_coarse", &k_coarse, "polynomial order for field variables on coarse mesh");
  cmdp.setOption("numRefs",&refCount,"number of refinements");
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("enforceOneIrregularity", "dontEnforceOneIrregularity", &enforceOneIrregularity);
  cmdp.setOption("useSmoothing", "useNoSmoothing", &applyDiagonalSmoothing);
  cmdp.setOption("printRefinementDetails", "dontPrintRefinementDetails", &printRefinementDetails);
  cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
  cmdp.setOption("numCells", &numCells, "number of cells in the initial mesh");
  cmdp.setOption("eps", &eps, "ramp width");
  cmdp.setOption("useScaledGraphNorm", "dontUseScaledGraphNorm", &useWeightedGraphNorm);
  cmdp.setOption("gmgTol", &gmgTolerance, "tolerance for GMG convergence");
  cmdp.setOption("gmgMaxIterations", &gmgMaxIterations, "tolerance for GMG convergence");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
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
  int horizontalCells = numCells, verticalCells = numCells, depthCells = numCells;
  
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
  
  MeshPtr mesh, k0Mesh;
  
  int H1Order = k + 1;
  int H1Order_coarse = k_coarse + 1;
  if (!use3D) {
    mesh = MeshFactory::quadMeshMinRule(stokesBF, H1Order, delta_k, width, height,
                                        horizontalCells, verticalCells);
    k0Mesh = MeshFactory::quadMeshMinRule(stokesBF, H1Order_coarse, delta_k, width, height,
                                              horizontalCells, verticalCells);
  } else {
    mesh = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, H1Order, delta_k);
    k0Mesh = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, H1Order_coarse, delta_k);
  }
  
//  k0Mesh->setPartitionPolicy(Teuchos::rcp( new MeshPartitionPolicy ));
//  mesh->setPartitionPolicy(Teuchos::rcp( new MeshPartitionPolicy ));
  
  mesh->registerObserver(k0Mesh); // ensure that the k0 mesh refinements track those of the solution mesh

  
  RHSPtr rhs = RHS::rhs(); // zero
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr topBoundary = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);
  
  // top boundary:
  FunctionPtr u1_bc_fxn = Teuchos::rcp( new RampBoundaryFunction_U1(eps) );
  FunctionPtr zero = Function::zero();
  bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
  bc->addDirichlet(u2hat, topBoundary, zero);
  if (use3D) bc->addDirichlet(u3hat, topBoundary, zero);
  
  // everywhere else:
  bc->addDirichlet(u1hat, otherBoundary, zero);
  bc->addDirichlet(u2hat, otherBoundary, zero);
  if (use3D) bc->addDirichlet(u3hat, otherBoundary, zero);
  
//  bc->addZeroMeanConstraint(p);
  bc->addSinglePointBC(p->ID(), zero);

  IPPtr graphNorm;
  
  FunctionPtr h = Teuchos::rcp( new hFunction() );
  
  if (useWeightedGraphNorm) {
    graphNorm = IP::ip();
    graphNorm->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
    graphNorm->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
    graphNorm->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
    graphNorm->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
    graphNorm->addTerm( v1->dx() + v2->dy() );       // pressure
    graphNorm->addTerm( h * tau1->div() - h * q->dx() ); // u1
    graphNorm->addTerm( h * tau2->div() - h * q->dy() ); // u2
    graphNorm->addTerm( (mu / h) * v1 );
    graphNorm->addTerm( (mu / h) * v2 );
    graphNorm->addTerm(  q );
    graphNorm->addTerm( tau1 );
    graphNorm->addTerm( tau2 );
  } else {
     graphNorm = stokesBF->graphNorm();
  }
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  mesh->registerSolution(solution); // sign up for projection of old solution onto refined cells.
  
  double energyThreshold = 0.2;
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  refinementStrategy.setReportPerCellErrors(true);
  refinementStrategy.setEnforceOneIrregularity(enforceOneIrregularity);
  
  Teuchos::RCP<Solver> coarseSolver, fineSolver;
  if (useMumps) {
#ifdef USE_MUMPS
    coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
#else
    cout << "useMumps=true, but MUMPS is not available!\n";
    exit(0);
#endif
  } else {
    coarseSolver = Teuchos::rcp( new KluSolver );
  }
  GMGSolver* gmgSolver;
  
  if (useGMGSolver) {
    double tol = gmgTolerance;
    int maxIters = gmgMaxIterations;
    BCPtr zeroBCs = bc->copyImposingZero();
    gmgSolver = new GMGSolver(zeroBCs, k0Mesh, graphNorm, mesh,
                                         solution->getPartitionMap(), maxIters, tol, coarseSolver);
    gmgSolver->setAztecOutput(AztecOutputLevel);
    gmgSolver->setApplySmoothingOperator(applyDiagonalSmoothing);
    fineSolver = Teuchos::rcp( gmgSolver );
  } else {
    fineSolver = coarseSolver;
  }
  
//  if (rank==0) cout << "experimentally starting by solving with MUMPS on the fine mesh.\n";
//  solution->solve( Teuchos::rcp( new MumpsSolver) );
  
  solution->solve(fineSolver);
  
  
#ifdef HAVE_EPETRAEXT_HDF5
  ostringstream dir_name;
  dir_name << "stokesCavityFlow_k" << k;
  HDF5Exporter exporter(mesh,dir_name.str());
  exporter.exportSolution(solution,varFactory,0);
#endif
  
#ifdef USE_MUMPS
  if (useMumps) coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
#endif
  
  solution->reportTimings();
  if (useGMGSolver) gmgSolver->gmgOperator().reportTimings();
  for (int refIndex=0; refIndex < refCount; refIndex++) {
    double energyError = solution->energyErrorTotal();
    GlobalIndexType numFluxDofs = mesh->numFluxDofs();
    if (rank==0) {
      cout << "Before refinement " << refIndex << ", energy error = " << energyError;
      cout << " (using " << numFluxDofs << " trace degrees of freedom)." << endl;
    }
    bool printToConsole = printRefinementDetails && (rank==0);
    refinementStrategy.refine(printToConsole);
    
    GlobalIndexType fineDofs = mesh->globalDofCount();
    GlobalIndexType coarseDofs = k0Mesh->globalDofCount();
    if (rank==0) {
      cout << "After refinement, coarse mesh has " << k0Mesh->numActiveElements() << " elements and " << coarseDofs << " dofs.\n";
      cout << "  Fine mesh has " << mesh->numActiveElements() << " elements and " << fineDofs << " dofs.\n";
    }
    
    if (!use3D) {
      ostringstream fineMeshLocation, coarseMeshLocation;
      fineMeshLocation << "stokesFineMesh_k" << k << "_ref" << refIndex;
      GnuPlotUtil::writeComputationalMeshSkeleton(fineMeshLocation.str(), mesh, true); // true: label cells
      coarseMeshLocation << "stokesCoarseMesh_k" << k << "_ref" << refIndex;
      GnuPlotUtil::writeComputationalMeshSkeleton(coarseMeshLocation.str(), k0Mesh, true); // true: label cells
    }
//
//    cout << "Fine Mesh entities:\n";
//    mesh->getTopology()->printAllEntities();
//
//    cout << "Coarse Mesh entities:\n";
//    k0Mesh->getTopology()->printAllEntities();
    
//    if (refIndex >= 3) {
//      set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
//      cout << "Coarse mesh parities:\n";
//      for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
//        GlobalIndexType cellID = *cellIDIt;
//        cout << cellID << ":\n" << k0Mesh->globalDofAssignment()->cellSideParitiesForCell(cellID);
//      }
//      cout << "Fine mesh parities:\n";
//      for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
//        GlobalIndexType cellID = *cellIDIt;
//        cout << cellID << ":\n" << mesh->globalDofAssignment()->cellSideParitiesForCell(cellID);
//      }
//    }
    
    if (useGMGSolver) { // create fresh fineSolver now that the meshes have changed:
#ifdef USE_MUMPS
      if (useMumps) coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
#endif
      double tol = gmgTolerance;
      int maxIters = gmgMaxIterations;
      BCPtr zeroBCs = bc->copyImposingZero();
      gmgSolver = new GMGSolver(zeroBCs, k0Mesh, graphNorm, mesh,
                                           solution->getPartitionMap(), maxIters, tol, coarseSolver);
      gmgSolver->setAztecOutput(AztecOutputLevel);
      gmgSolver->setApplySmoothingOperator(applyDiagonalSmoothing);
      fineSolver = Teuchos::rcp( gmgSolver );
    }
    
    solution->solve(fineSolver);
    solution->reportTimings();
    if (useGMGSolver) gmgSolver->gmgOperator().reportTimings();
    
#ifdef HAVE_EPETRAEXT_HDF5
    exporter.exportSolution(solution,varFactory,refIndex+1);
#endif
  }
  double energyErrorTotal = solution->energyErrorTotal();
  
//  cout << "Final Mesh, entities report:\n";
//  solution->mesh()->getTopology()->printAllEntities();
  
  FunctionPtr massFlux;
  if (!use3D) massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat * n->x() + u2hat * n->y()) );
  else massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat * n->x() + u2hat * n->y() + u3hat * n->z()) );
  double netMassFlux = massFlux->integrate(mesh); // integrate over the mesh skeleton

  GlobalIndexType numFluxDofs = mesh->numFluxDofs();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  if (rank==0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << numGlobalDofs << " total dofs, including fields).\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "Net mass flux: " << netMassFlux << endl;
  }
  
#ifdef HAVE_EPETRAEXT_HDF5
  exporter.exportSolution(solution,varFactory,0);
#endif
  
  if (!use3D) {
    GnuPlotUtil::writeComputationalMeshSkeleton("cavityFlowRefinedMesh", mesh, true);
  }
  
  return 0;
}
