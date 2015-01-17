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

#include "IndexType.h"

#include "CondensedDofInterpreter.h"

#ifdef WATCH_BGQ_FLOP_COUNTERS

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

enum SolverChoice {
  MUMPS, KLU, SLU, UNKNOWN
};

SolverChoice getSolverChoice(string solverString) {
  if (solverString=="MUMPS") return MUMPS;
  if (solverString=="KLU") return KLU;
  if (solverString=="SLU") return SLU;
  return UNKNOWN;
}

int main(int argc, char *argv[]) {
#ifdef ENABLE_INTEL_FLOATING_POINT_EXCEPTIONS
  cout << "NOTE: enabling floating point exceptions for divide by zero.\n";
  _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
#endif
  
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();
  int numRanks = Teuchos::GlobalMPISession::getNProc();
  
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
  int delta_k = 1;   // test space enrichment (1 suffices)
  
  bool useCondensedSolve = true;
//  bool useSuperLUDist = true;
//  bool useMumps = false;
//  bool useCGSolver = false;
//  bool useMLSolver = false;
  bool useGMGSolver = true;
  bool clearSolution = false;
  
  string solverString = "MUMPS";
  
  bool enforceOneIrregularity = true;
  
  bool conformingTraces = false;
  
  bool bugTestRefs = false;
  
  double width = 1.0, height = 1.0, depth = 1.0;
  int horizontalCells = 2, verticalCells = 2, depthCells = 2;
  
  int numCells = -1;
  
  double eps = 1.0/64.0;
  int mumpsMaxMemoryMB = 768;
  
  double energyThreshold = 0.2;
  
  double coarseSolveTolerance = 1e-4;
  
  double relativeTol = 1e-6;
  
  int coarseMesh_k = 0;
  
  bool pMultiGridOnly = true;
  bool useWeightedGraphNorm = false;
  
  string meshLoadName = "", coarseMeshLoadName_p = "", coarseMeshLoadName_h = ""; // file(s) to load mesh from
  int startingRefinementNumber = 0;
  int maxCellsPerRank = 300;
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numCells",&numCells,"number of cells in x/y/z directions");
  cmdp.setOption("numRefs",&refCount,"number of refinements");
  cmdp.setOption("eps", &eps, "ramp width (set to 0 for no ramp in BCs)");
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("globalSolver", &solverString, "global solver choice -- MUMPS, KLU, GMG, or SLU");
  cmdp.setOption("useGMG", "useDirect", &useGMGSolver, "use geometric multi-grid");
  cmdp.setOption("relativeTol", &relativeTol, "Energy error-relative tolerance for iterative solver.");
  cmdp.setOption("coarseMesh_k", &coarseMesh_k, "field order for coarse mesh in GMG solve.");
//  cmdp.setOption("useMumps", "useKLU", &useMumps, "use MUMPS (if available)");
  cmdp.setOption("mumpsMaxMemoryMB", &mumpsMaxMemoryMB, "max allocation size MUMPS is allowed to make, in MB");
  cmdp.setOption("refinementThreshold", &energyThreshold, "relative energy threshold for refinements");
  cmdp.setOption("meshLoadName", &meshLoadName, "file to load initial mesh from");
  cmdp.setOption("coarseMeshLoadName_h", &coarseMeshLoadName_h, "file to load coarse (h) mesh from");
  cmdp.setOption("coarseMeshLoadName_p", &coarseMeshLoadName_p, "file to load coarse (p) mesh from");
  cmdp.setOption("startingRefNumber", &startingRefinementNumber, "where to start counting refinements (useful for restart)");
  cmdp.setOption("maxCellsPerRank", &maxCellsPerRank, "max cells per rank (will quit refining once this is reached)");
  cmdp.setOption("usePMultigrid", "useHPMultigrid", &pMultiGridOnly);
  cmdp.setOption("useScaledGraphNorm", "dontUseScaledGraphNorm", &useWeightedGraphNorm);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  int H1Order = k + 1;

  SolverChoice solverChoice = getSolverChoice(solverString);

  if (solverChoice==UNKNOWN) {
    if (rank==0) cout << "Unrecognized global solver choice.  Exiting\n";
    return 1;
  }
  
  if (useCondensedSolve) {
    if (rank==0) cout << "Using condensed solve for global problem.\n";
  } else {
    if (rank==0) cout << "Using standard solve for global problem.\n";
  }
  
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
    if (rank==0) cout << "Note: using non-conforming traces.\n";
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
  
  MeshPtr mesh, coarseMesh_p, coarseMesh_h;
  
  Epetra_Time timer(Comm);
  
  if (meshLoadName.length() == 0) {
    mesh = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, H1Order, delta_k);
    coarseMesh_p = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, coarseMesh_k + 1, delta_k);
    coarseMesh_h = MeshFactory::rectilinearMesh(stokesBF, domainDimensions, elementCounts, coarseMesh_k + 1, delta_k);
  } else {
    mesh = MeshFactory::loadFromHDF5(stokesBF, meshLoadName);
    coarseMesh_p = MeshFactory::loadFromHDF5(stokesBF, coarseMeshLoadName_p);
    coarseMesh_h = MeshFactory::loadFromHDF5(stokesBF, coarseMeshLoadName_h);
  }
  
  mesh->registerObserver(coarseMesh_p);
  double meshConstructionTime = timer.ElapsedTime();
  if (rank==0) cout << "On rank " << rank << ", mesh construction time: " << meshConstructionTime << endl;
  int elementCount = mesh->getActiveCellIDs().size();
  int dofCount = mesh->numGlobalDofs();
  if (rank==0) cout << "Initial mesh has " << elementCount << " elements and " << dofCount << " global dofs.\n";
  
  int maxCells = (maxCellsPerRank != INT_MAX) ? maxCellsPerRank * numRanks : INT_MAX;
  
  if (rank==0) cout << "maxCellsPerRank is " << maxCellsPerRank << "; will stop if mesh exceeds " << maxCells << " elements.\n";
  if (rank==0) cout << "energyThreshold is " << energyThreshold << endl;
  
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
  
  bc->addSinglePointBC(p->ID(), 0);
//  bc->addZeroMeanConstraint(p);

  IPPtr graphNorm;
  if (! useWeightedGraphNorm) {
    graphNorm = stokesBF->graphNorm();
  } else {
    FunctionPtr h = Teuchos::rcp( new hFunction() );
    graphNorm = IP::ip();
    graphNorm->addTerm( mu * v1->grad() + tau1 ); // sigma1
    graphNorm->addTerm( mu * v2->grad() + tau2 ); // sigma2
    graphNorm->addTerm( mu * v3->grad() + tau3 ); // sigma3
    graphNorm->addTerm( v1->dx() + v2->dy() + v3->dz() );       // pressure
    graphNorm->addTerm( h * tau1->div() - h * q->dx() ); // u1
    graphNorm->addTerm( h * tau2->div() - h * q->dy() ); // u2
    graphNorm->addTerm( h * tau3->div() - h * q->dz() ); // u3
    graphNorm->addTerm( (mu / h) * v1 );
    graphNorm->addTerm( (mu / h) * v2 );
    graphNorm->addTerm( (mu / h) * v3 );
    graphNorm->addTerm(  q );
    graphNorm->addTerm( tau1 );
    graphNorm->addTerm( tau2 );
    graphNorm->addTerm( tau3 );
  }
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  SolutionPtr solution_p = Solution::solution(coarseMesh_p, bc, rhs, graphNorm); // solution on constant mesh
  
  mesh->registerSolution(solution); // sign up for projection of old solution onto refined cells.
  
  if (bugTestRefs) {
    // manually refine according to a series of refinements that caused a crash on Vesta earlier
    
    set<GlobalIndexType> cellIDs;
    // first refinement:
    cellIDs.insert(2); cellIDs.insert(3); cellIDs.insert(6); cellIDs.insert(7);
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());
    
    // second refinement:
    cellIDs.clear();
    cellIDs.insert(10); cellIDs.insert(11); cellIDs.insert(14); cellIDs.insert(18); cellIDs.insert(22); cellIDs.insert(23); cellIDs.insert(26); cellIDs.insert(27); cellIDs.insert(31); cellIDs.insert(35); cellIDs.insert(38); cellIDs.insert(39);
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());
    
    // third refinement:
    cellIDs.clear();
    cellIDs.insert(1); cellIDs.insert(4); cellIDs.insert(5); cellIDs.insert(42); cellIDs.insert(43); cellIDs.insert(46); cellIDs.insert(50); cellIDs.insert(51); cellIDs.insert(58); cellIDs.insert(62); cellIDs.insert(63); cellIDs.insert(66); cellIDs.insert(67); cellIDs.insert(70); cellIDs.insert(71); cellIDs.insert(74); cellIDs.insert(75); cellIDs.insert(78); cellIDs.insert(79); cellIDs.insert(86); cellIDs.insert(87); cellIDs.insert(90); cellIDs.insert(91); cellIDs.insert(98); cellIDs.insert(99); cellIDs.insert(103); cellIDs.insert(107); cellIDs.insert(111); cellIDs.insert(115); cellIDs.insert(119); cellIDs.insert(126); cellIDs.insert(127); cellIDs.insert(131); cellIDs.insert(134); cellIDs.insert(135);
    mesh->hRefine(cellIDs, RefinementPattern::regularRefinementPatternHexahedron());
  }
  
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  refinementStrategy.setReportPerCellErrors(true);
  refinementStrategy.setEnforceOneIrregularity(enforceOneIrregularity);
  
  if ((solverChoice == MUMPS) && (Teuchos::GlobalMPISession::getNProc()==1)) {
    cout << "solverChoice == MUMPS, but only running on one proc.  There are issues with Amesos_Mumps::Destroy() in this case.  Therefore, overriding solverChoice = KLU.\n";
    solverChoice = KLU;
  }
  
  Teuchos::RCP<Solver> coarsestSolver, fineSolver, intermediateSolver;
  switch(solverChoice) {
      case MUMPS:
#ifdef HAVE_AMESOS_MUMPS
      coarsestSolver = Teuchos::rcp( new MumpsSolver(mumpsMaxMemoryMB, true) );
#else
      cout << "useMumps=true, but MUMPS is not available!\n";
      exit(1);
#endif
      break;
    case KLU:
      coarsestSolver = Teuchos::rcp( new KluSolver );
      break;
    case SLU:
      coarsestSolver = Teuchos::rcp( new SuperLUDistSolver(true) ); // true: save factorization
      break;
    case UNKNOWN:
      // should be unreachable
      break;
  }
  
  solution->setUseCondensedSolve(useCondensedSolve);
  solution_p->setUseCondensedSolve(useCondensedSolve);

  BCPtr zeroBCs = bc->copyImposingZero();
  int maxIters = 80000;
  
  if (!pMultiGridOnly) {
    GMGSolver* intermediateSolverGMG = new GMGSolver(zeroBCs, coarseMesh_h, graphNorm, coarseMesh_p, solution_p->getDofInterpreter(),
                                                     solution_p->getPartitionMap(), maxIters, coarseSolveTolerance, coarsestSolver, useCondensedSolve);
    intermediateSolverGMG->setAztecOutput(0); // suppress output for nested solver
    intermediateSolver = Teuchos::rcp( intermediateSolverGMG );
  }
  
  GMGSolver* gmgSolver = NULL;
  
  if (useGMGSolver) {
    double tol = relativeTol;

    if (pMultiGridOnly) {
      gmgSolver = new GMGSolver(zeroBCs, coarseMesh_p, graphNorm, mesh, solution->getDofInterpreter(),
                                solution->getPartitionMap(), maxIters, tol, coarsestSolver, useCondensedSolve);
    } else {
      gmgSolver = new GMGSolver(zeroBCs, coarseMesh_p, graphNorm, mesh, solution->getDofInterpreter(),
                                solution->getPartitionMap(), maxIters, tol, intermediateSolver, useCondensedSolve);
    }
    gmgSolver->setAztecOutput(100); // print residual every 100 iterations;
//    gmgSolver->gmgOperator().constructLocalCoefficientMaps(); // for separating out the timings
    fineSolver = Teuchos::rcp( gmgSolver );
  } else {
    fineSolver = coarsestSolver;
  }
  if (elementCount > maxCells) {
    if (rank==0) cout << "Initial mesh size exceeds maxCells; exiting.\n";
    return 0;
  }
  if (rank==0) cout << "About to start solve.\n";
  timer.ResetStartTime();
  
  if (useCondensedSolve) {
    solution->solve(fineSolver);
  } else {
    solution->solve(fineSolver);
  }
  
  double totalSolveTime = timer.ElapsedTime();
  if (rank==0) cout << "total solve time (as seen by rank 0) " << totalSolveTime << " seconds.\n";
  solution->reportTimings();
  
  if (useGMGSolver) {
    gmgSolver->gmgOperator().reportTimings();
    gmgSolver->gmgOperator().clearTimings();
  }
  
#ifdef HAVE_EPETRAEXT_HDF5
  ostringstream dir_name;
  dir_name << "stokesCavityFlow3D_k" << k;
  HDF5Exporter exporter(mesh,dir_name.str());
  exporter.exportSolution(solution,varFactory,startingRefinementNumber);
  if (rank==0) cout << "...completed.\n";
  ostringstream meshFileName, coarseMeshFileName_h, coarseMeshFileName_p;
  meshFileName << "stokesCavityFlow3D_k" << k << "_ref" << startingRefinementNumber << ".mesh";
  mesh->saveToHDF5(meshFileName.str());
#endif
  
  for (int refIndex=startingRefinementNumber; refIndex < refCount+startingRefinementNumber; refIndex++) {
    if (rank==0) cout << "About to start refinement " << refIndex + 1 << ".\n";
    timer.ResetStartTime();
    double energyError = solution->energyErrorTotal();
    int numFluxDofs = mesh->numFluxDofs();
    if (rank==0) {
      cout << "Before refinement " << refIndex + 1 << ", energy error = " << energyError;
      cout << " (using " << numFluxDofs << " trace degrees of freedom)." << endl;
    }
#ifdef HAVE_AMESOS_MUMPS
    // recreate coarsest solver prior to refinement (true means it keeps a factorization, which is unsafe when the coarse mesh is being refined...)
    if (solverChoice==MUMPS) coarsestSolver = Teuchos::rcp( new MumpsSolver(mumpsMaxMemoryMB, true) );
#endif
    if (solverChoice==SLU)   coarsestSolver = Teuchos::rcp( new SuperLUDistSolver(true) ); // true: save factorization

    refinementStrategy.refine(rank==0);
    double refinementTime = timer.ElapsedTime();
    if (rank==0) cout << "refinement time (as seen by rank 0) " << refinementTime << " seconds.\n";

#ifdef HAVE_EPETRAEXT_HDF5
    if (rank==0) cout << "Beginning export of refinement " << refIndex+1 << " mesh.\n";
    ostringstream meshFileName;
    meshFileName << "stokesCavityFlow3D_k" << k << "_ref" << refIndex+1 << ".mesh";
    mesh->saveToHDF5(meshFileName.str());
    coarseMeshFileName_h.str("");
    coarseMeshFileName_h << "stokesCavityFlow3D_k" << k << "_ref" << refIndex+1 << "_coarse_h.mesh";
    coarseMesh_h->saveToHDF5(coarseMeshFileName_h.str());
    coarseMeshFileName_p.str("");
    coarseMeshFileName_p << "stokesCavityFlow3D_k" << k << "_ref" << refIndex+1 << "_coarse_p.mesh";
    coarseMesh_p->saveToHDF5(coarseMeshFileName_p.str());
    if (rank==0) cout << "Refined mesh saved to " << meshFileName.str() << endl;
    if (rank==0) cout << "(Use --meshLoadName=\"" << meshFileName.str() << "\" --coarseMeshLoadName_h=\"" << coarseMeshFileName_h.str() << "\" --coarseMeshLoadName_p=\"" << coarseMeshFileName_p.str() << "\" --startingRefNumber=" << refIndex + 1 << ")\n";
#endif
    
    int elementCount = mesh->getActiveCellIDs().size();
    if (elementCount > maxCells) {
      if (rank==0) cout << "Cell count (" << elementCount << ") exceeds maxCells; exiting.\n";
      return 0;
    }
    
    if (clearSolution) solution->clear();
    
    if (useGMGSolver) { // recreate fineSolver...
      if (!pMultiGridOnly) {
        if (useCondensedSolve) {
          CondensedDofInterpreter* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter*>(solution_p->getDofInterpreter().get());
          if (condensedDofInterpreter != NULL) {
            condensedDofInterpreter->reinitialize();
          }
          condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter*>(solution->getDofInterpreter().get());
          if (condensedDofInterpreter != NULL) {
            condensedDofInterpreter->reinitialize();
          }
        }
        GMGSolver* intermediateSolverGMG = new GMGSolver(zeroBCs, coarseMesh_h, graphNorm, coarseMesh_p, solution_p->getDofInterpreter(),
                                                         solution_p->getPartitionMap(), maxIters, coarseSolveTolerance, coarsestSolver, useCondensedSolve);
        intermediateSolverGMG->setAztecOutput(0); // suppress output for nested solver
        intermediateSolver = Teuchos::rcp( intermediateSolverGMG );
      }
      
      gmgSolver = NULL;
      
      if (useGMGSolver) {
        double tol = relativeTol * energyError;
        if (pMultiGridOnly) {
          gmgSolver = new GMGSolver(zeroBCs, coarseMesh_p, graphNorm, mesh, solution->getDofInterpreter(),
                                    solution->getPartitionMap(), maxIters, tol, coarsestSolver, useCondensedSolve);
        } else {
          gmgSolver = new GMGSolver(zeroBCs, coarseMesh_p, graphNorm, mesh, solution->getDofInterpreter(),
                                    solution->getPartitionMap(), maxIters, tol, intermediateSolver, useCondensedSolve);
        }
        gmgSolver->setAztecOutput(100); // print residual every 100 iterations;
        //    gmgSolver->gmgOperator().constructLocalCoefficientMaps(); // for separating out the timings
        fineSolver = Teuchos::rcp( gmgSolver );
      } else {
        fineSolver = coarsestSolver;
      }
    }
    
    timer.ResetStartTime();

    solution->solve(fineSolver);
    double totalSolveTime = timer.ElapsedTime();
    if (rank==0) cout << "total solve time (as seen by rank 0) " << totalSolveTime << " seconds.\n";
    solution->reportTimings();
    if (useGMGSolver) {
      gmgSolver->gmgOperator().reportTimings();
      gmgSolver->gmgOperator().clearTimings();
    }
    
#ifdef HAVE_EPETRAEXT_HDF5
    if (rank==0) cout << "Beginning export of refinement " << refIndex+1 << " solution.\n";
//    dir_name.str("");
//    dir_name << "stokesCavityFlow3D_k" << k << "_ref" << refIndex;
//    HDF5Exporter exporter2(mesh,dir_name.str());
    exporter.exportSolution(solution,varFactory,refIndex+1);
    if (rank==0) cout << "...completed.\n"; //  Beginning export of refinement " << refIndex << " mesh.\n";
#endif
  }
  if (rank==0) cout << "Beginning computation of energy error, and final refinement.\n";
  double energyErrorTotal = solution->energyErrorTotal();
  refinementStrategy.refine(rank==0);
  if (rank==0) cout << "...completed.\n";
  
#ifdef HAVE_EPETRAEXT_HDF5
  if (rank==0) cout << "Beginning export of refinement " << refCount + startingRefinementNumber + 1 << " mesh.\n";
  meshFileName.str("");
  meshFileName << "stokesCavityFlow3D_k" << k << "_ref" << refCount + startingRefinementNumber + 1 << ".mesh";
  mesh->saveToHDF5(meshFileName.str());
  coarseMeshFileName_h.str("");
  coarseMeshFileName_h << "stokesCavityFlow3D_k" << k << "_ref" << refCount + startingRefinementNumber+1 << "_coarse_h.mesh";
  coarseMesh_h->saveToHDF5(coarseMeshFileName_h.str());
  coarseMeshFileName_p.str("");
  coarseMeshFileName_p << "stokesCavityFlow3D_k" << k << "_ref" << refCount + startingRefinementNumber+1 << "_coarse_p.mesh";
  coarseMesh_p->saveToHDF5(coarseMeshFileName_p.str());
  if (rank==0) cout << "Refined mesh saved to " << meshFileName.str() << endl;
  if (rank==0) cout << "(Use --meshLoadName=\"" << meshFileName.str() << "\" --coarseMeshLoadName_h=\"" << coarseMeshFileName_h.str() << "\" --coarseMeshLoadName_p=\"" << coarseMeshFileName_p.str() << "\" --startingRefNumber=" << refCount + startingRefinementNumber + 1 << ")\n";

#endif
  
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
  
  coarsestSolver = Teuchos::rcp((Solver*) NULL); // without this when useMumps = true and running on one rank, we see a crash on exit, which may have to do with MPI being finalized before coarseSolver is deleted.
  
//#ifdef HAVE_EPETRAEXT_HDF5
//  if (rank==0) cout << "Beginning export of final solution.\n";
//  dir_name.str("");
//  dir_name << "stokesCavityFlow3D_k" << k << "_final";
//  HDF5Exporter exporter(mesh,dir_name.str());
//  exporter.exportSolution(solution,varFactory,0);
//  // straight-line mesh
//  if (rank==0) cout << "...completed.  Beginning export of final mesh.\n";
//  dir_name.str("");
//  dir_name << "stokesCavityFlow3D_k" << k << "_finalMesh";
//  HDF5Exporter meshExporter2(mesh,dir_name.str());
//  meshExporter2.exportFunction(Function::normal(),"mesh",0,2);
//  if (rank==0) cout << "...completed.\n";
//#endif
  
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
