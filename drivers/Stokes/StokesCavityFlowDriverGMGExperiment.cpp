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

#include "CondensedDofInterpreter.h"

#include "PressurelessStokesFormulation.h"

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
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  bool use3D = true;
  int refCount = 10;
  
  int k = 3; // poly order for field variables
  int delta_k = use3D ? 3 : 2;   // test space enrichment
  int k_coarse = 0;
  
  bool useGMGSolver = true;
  bool useCG = true;
  
  bool enforceOneIrregularity = true;
  bool conformingTraces = false;
  bool applyDiagonalSmoothing = true;
  
  bool useDiagonalScaling = false; // of the global stiffness matrix in GMGSolver
  
  bool printRefinementDetails = false;
  
  bool useWeightedGraphNorm = false; // graph norm scaled according to units, more or less
  bool useStaticCondensation = true;
  
  int mumpsMaxMemoryMB = 768;
  double energyThreshold = 0.2;
  
  string solverString = "MUMPS";
  
  int numCells = 2;
  
  double eps = 1.0 / 64.0;
  
  int AztecOutputLevel = 1;
  int gmgMaxIterations = 200;
  double relativeTol = 1e-6;
  double minTol = 1e-11; // sorta unreasonable to ask for tighter tolerance than this
  int smootherOverlap = 0;
  double graphNormL2TermWeight = 1.0;
  string aztecConvergenceString = "AZ_rhs";
  
  cmdp.setOption("useGMG", "useDirect", &useGMGSolver, "use GMG solver (otherwise, use direct solver--option for testing)");
  cmdp.setOption("use3D", "use2D", &use3D);
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("k_coarse", &k_coarse, "polynomial order for field variables on coarse mesh");
  cmdp.setOption("numRefs",&refCount,"number of refinements");
  cmdp.setOption("refinementThreshold", &energyThreshold, "relative energy threshold for refinements");
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("enforceOneIrregularity", "dontEnforceOneIrregularity", &enforceOneIrregularity);
  cmdp.setOption("useSmoothing", "useNoSmoothing", &applyDiagonalSmoothing);
  cmdp.setOption("globalSolver", &solverString, "global solver choice -- MUMPS, KLU, GMG, or SLU");
  cmdp.setOption("l2WeightForGraphNorm", &graphNormL2TermWeight, "a.k.a. 'beta' weight");
  cmdp.setOption("mumpsMaxMemoryMB", &mumpsMaxMemoryMB, "max allocation size MUMPS is allowed to make, in MB");
  cmdp.setOption("smootherOverlap", &smootherOverlap, "overlap for smoother");
  cmdp.setOption("printRefinementDetails", "dontPrintRefinementDetails", &printRefinementDetails);
  cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
  cmdp.setOption("azConv", &aztecConvergenceString, "Aztec convergence criterion");
  cmdp.setOption("numCells", &numCells, "number of cells in the initial mesh");
  cmdp.setOption("eps", &eps, "ramp width");
  cmdp.setOption("useScaledGraphNorm", "dontUseScaledGraphNorm", &useWeightedGraphNorm);
  cmdp.setOption("useDiagonalScaling", "dontUseDiagonalScaling", &useDiagonalScaling);
  cmdp.setOption("useStaticCondensation", "dontUseStaticCondensation", &useStaticCondensation);
  cmdp.setOption("useCG", "useGMRES", &useCG, "use conjugate gradient or GMRES with multi-grid preconditioner");
  
  //  cmdp.setOption("gmgTol", &gmgTolerance, "tolerance for GMG convergence");
  cmdp.setOption("relativeTol", &relativeTol, "Energy error-relative tolerance for iterative solver.");
  cmdp.setOption("minTol", &minTol, "Minimum tolerance for iterative solver.");
  cmdp.setOption("gmgMaxIterations", &gmgMaxIterations, "tolerance for GMG convergence");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  SolverChoice solverChoice = getSolverChoice(solverString);
  
  if (solverChoice==UNKNOWN) {
    if (rank==0) cout << "Unrecognized global solver choice.  Exiting\n";
    return 1;
  }

  int azConv;
  if (aztecConvergenceString=="AZ_rhs") {
    azConv = AZ_rhs;
  } else if (aztecConvergenceString=="AZ_noscaled") {
    azConv = AZ_noscaled;
  } else if (aztecConvergenceString=="AZ_r0") {
    azConv = AZ_r0;
  } else {
    if (rank==0) cout << "Unrecognized Aztec convergence criterion " << aztecConvergenceString << endl;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unrecognized Aztec convergence criterion");
  }
  
  bool usePressurelessFormulation = false; // VGP otherwise
  
  BFPtr stokesBF;
  
  VarPtr u1hat, u2hat, u3hat, p;
  VarPtr t1n;
  
  FunctionPtr n = Function::normal();
  
  double mu = 1.0; // viscosity
  
  IPPtr weightedGraphNorm; // only valid for VGP
  
  if (usePressurelessFormulation) {
    int spaceDim = use3D ? 3 : 2;
    PressurelessStokesFormulation stokesForm(spaceDim);
    stokesBF = stokesForm.bf();
    u1hat = stokesForm.u_hat(1);
    u2hat = stokesForm.u_hat(2);
    if (use3D) u3hat = stokesForm.u_hat(3);
    t1n = stokesForm.tn_hat(1);
    
    if (rank==0) cout << "Using pressure-free Stokes formulation.\n";
  } else { // VGP formulation
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
    p = varFactory.fieldVar("p");
    
    if (conformingTraces) {
      u1hat = varFactory.traceVar("\\widehat{u}_1", u1);
      u2hat = varFactory.traceVar("\\widehat{u}_2", u2);
      
      if (use3D) {
        u3hat = varFactory.traceVar("\\widehat{u}_3", u3);
      }
    } else {
      if (rank==0) cout << "Note: using non-conforming traces.\n";
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
    
    stokesBF = Teuchos::rcp( new BF(varFactory) );
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
    
    FunctionPtr h = Teuchos::rcp( new hFunction() );
    
    weightedGraphNorm = IP::ip();
    {
      weightedGraphNorm->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
      weightedGraphNorm->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
      weightedGraphNorm->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
      weightedGraphNorm->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
      weightedGraphNorm->addTerm( v1->dx() + v2->dy() );       // pressure
      weightedGraphNorm->addTerm( h * tau1->div() - h * q->dx() ); // u1
      weightedGraphNorm->addTerm( h * tau2->div() - h * q->dy() ); // u2
      weightedGraphNorm->addTerm( (mu / h) * v1 );
      weightedGraphNorm->addTerm( (mu / h) * v2 );
      weightedGraphNorm->addTerm(  q );
      weightedGraphNorm->addTerm( tau1 );
      weightedGraphNorm->addTerm( tau2 );
    }
  }
  
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
  
  if (!usePressurelessFormulation) {
//    bc->addSinglePointBC(p->ID(), zero);
//    if (rank==0) cout << "using single-point BC for pressure.\n";
    bc->addZeroMeanConstraint(p);
  } else {
    // need to do something to take care of the extra mode
    
  }

  IPPtr fineIP, coarseIP;
  
  IPPtr standardGraphNorm = stokesBF->graphNorm(graphNormL2TermWeight);
  
  if (useWeightedGraphNorm) {
    if (rank==0) cout << "Using weighted graph norm for fine and coarse solves.\n";
    fineIP = weightedGraphNorm;
    coarseIP = weightedGraphNorm;
  } else {
    if (rank==0) cout << "Using standard graph norm for both fine and coarse solves.\n";
    fineIP = standardGraphNorm;
    coarseIP = standardGraphNorm;
  }
  
  VarFactory varFactory = stokesBF->varFactory();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, fineIP);
  
  solution->setUseCondensedSolve(useStaticCondensation);
  
  mesh->registerSolution(solution); // sign up for projection of old solution onto refined cells.
  
  LinearTermPtr residual = stokesBF->testFunctional(solution);
  
//  RefinementStrategy refinementStrategy( mesh, residual, standardGraphNorm, energyThreshold); // even when we use the weighted graph norm for solving, we should use the standard one for refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );

  double energyErrorForZeroSolution = 0;
  if (refCount > 0) {
    if (minTol < relativeTol) {
      if (rank==0) cout << "Computing energy error for 0 solution on coarse mesh.\n";
      solution->initializeStiffnessAndLoad();
      solution->initializeLHSVector();
      solution->populateStiffnessAndLoad();
      solution->imposeBCs();
      solution->importSolution();
      energyErrorForZeroSolution = solution->energyErrorTotal();
      if (rank==0) cout << "Energy error for 0 solution on coarse mesh: " << energyErrorForZeroSolution << "\n";
    } else {
      // take minTol >= relativeTol as an indication that Aztec tolerance should not be energy error relative
      energyErrorForZeroSolution = 1.0;
    }
  }
  
  refinementStrategy.setReportPerCellErrors(true);
  refinementStrategy.setEnforceOneIrregularity(enforceOneIrregularity);
  
  Teuchos::RCP<Solver> coarseSolver, fineSolver;
  if ((solverChoice == MUMPS) && (Teuchos::GlobalMPISession::getNProc()==1)) {
    cout << "solverChoice == MUMPS, but only running on one proc.  There are issues with Amesos_Mumps::Destroy() in this case.  Therefore, overriding solverChoice = KLU.\n";
    solverChoice = KLU;
  }
  
  switch(solverChoice) {
    case MUMPS:
#ifdef USE_MUMPS
      coarseSolver = Teuchos::rcp( new MumpsSolver(mumpsMaxMemoryMB, true) );
#else
      cout << "useMumps=true, but MUMPS is not available!\n";
      exit(1);
#endif
      break;
    case KLU:
      coarseSolver = Teuchos::rcp( new KluSolver );
      break;
    case SLU:
      coarseSolver = Teuchos::rcp( new SuperLUDistSolver(true) ); // true: save factorization
      break;
    case UNKNOWN:
      // should be unreachable
      break;
  }
  
  vector<double> condestForRefinement;
  vector<int> iterationsForRefinement;
  vector<double> tolForRefinement;
  
  GMGSolver* gmgSolver;
  
  if (useGMGSolver) {
    double tol = relativeTol;
    tolForRefinement.push_back(tol);
    if (rank==0) cout << "Initial iterative solve tolerance: " << tol << endl;
    int maxIters = gmgMaxIterations;
    BCPtr zeroBCs = bc->copyImposingZero();
//    GMGSolver(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter,
//              Epetra_Map finePartitionMap, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation);

//    if (useStaticCondensation) {
//      solution->initializeLHSVector();
//      solution->initializeStiffnessAndLoad();
//      solution->populateStiffnessAndLoad();
//    }
    
    gmgSolver = new GMGSolver(zeroBCs, k0Mesh, coarseIP, mesh, solution->getDofInterpreter(),
                              solution->getPartitionMap(), maxIters, tol, coarseSolver, useStaticCondensation);
    gmgSolver->setAztecConvergenceOption(azConv);
    gmgSolver->setAztecOutput(AztecOutputLevel);
    gmgSolver->setApplySmoothingOperator(applyDiagonalSmoothing);
    gmgSolver->setUseConjugateGradient(useCG);
    gmgSolver->setUseDiagonalScaling(useDiagonalScaling);
    gmgSolver->gmgOperator().setSmootherType(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    gmgSolver->gmgOperator().setSmootherOverlap(smootherOverlap);
    fineSolver = Teuchos::rcp( gmgSolver );
  } else {
    // otherwise, make a new Solver of the same type as coarseSolver
    switch(solverChoice) {
      case MUMPS:
#ifdef USE_MUMPS
        coarseSolver = Teuchos::rcp( new MumpsSolver(mumpsMaxMemoryMB, true) ); // true: save factorization
#else
        cout << "useMumps=true, but MUMPS is not available!\n";
        exit(1);
#endif
        break;
      case KLU:
        coarseSolver = Teuchos::rcp( new KluSolver );
        break;
      case SLU:
        coarseSolver = Teuchos::rcp( new SuperLUDistSolver(true) ); // true: save factorization
        break;
      case UNKNOWN:
        // should be unreachable
        break;
    }
    fineSolver = Teuchos::rcp( new MumpsSolver(512, false) ); // false: don't save factorization for fine solve
  }
  
//  if (rank==0) cout << "experimentally starting by solving with MUMPS on the fine mesh.\n";
//  solution->solve( Teuchos::rcp( new MumpsSolver) );
  
  solution->setWriteMatrixToFile(true, "/tmp/A_stokes.dat");
  
  solution->solve(fineSolver);
  
#ifdef HAVE_EPETRAEXT_HDF5
  ostringstream dir_name;
  dir_name << "stokesCavityFlow_k" << k;
  HDF5Exporter exporter(mesh,dir_name.str());
  exporter.exportSolution(solution,varFactory,0);
#endif
  
//#ifdef USE_MUMPS
//  if (useMumps) coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
//#endif
  
  solution->reportTimings();
  if (useGMGSolver) {
    condestForRefinement.push_back(gmgSolver->condest());
    iterationsForRefinement.push_back(gmgSolver->iterationCount());
    gmgSolver->gmgOperator().reportTimings();
  }
  for (int refIndex=0; refIndex < refCount; refIndex++) {
    GlobalIndexType numFluxDofs = mesh->numFluxDofs();
    bool printToConsole = printRefinementDetails && (rank==0);
    refinementStrategy.refine(printToConsole);
    double energyError = refinementStrategy.getEnergyError(refIndex);
    if (rank==0) {
      cout << "Before refinement " << refIndex << ", energy error was " << energyError;
      cout << " (using " << numFluxDofs << " trace degrees of freedom)." << endl;
    }
    
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
    
    if (useGMGSolver) { // create fresh fineSolver now that the meshes have changed:
//#ifdef USE_MUMPS
//      if (useMumps) coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
//#endif
      
      if (useStaticCondensation) {
        CondensedDofInterpreter* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter*>(solution->getDofInterpreter().get());
        if (condensedDofInterpreter != NULL) {
          condensedDofInterpreter->reinitialize();
        }
      }
      
      double tol = max(relativeTol * energyError / energyErrorForZeroSolution, minTol);
      tolForRefinement.push_back(tol);
      if (rank==0) cout << "Setting iterative solve tolerance to " << tol << endl;
      int maxIters = gmgMaxIterations;
      BCPtr zeroBCs = bc->copyImposingZero();
      gmgSolver = new GMGSolver(zeroBCs, k0Mesh, coarseIP, mesh, solution->getDofInterpreter(),
                                solution->getPartitionMap(), maxIters, tol, coarseSolver, useStaticCondensation);
      gmgSolver->setAztecConvergenceOption(azConv);
      gmgSolver->setAztecOutput(AztecOutputLevel);
      gmgSolver->setApplySmoothingOperator(applyDiagonalSmoothing);
      gmgSolver->setUseConjugateGradient(useCG);
      gmgSolver->setUseDiagonalScaling(useDiagonalScaling);
      gmgSolver->gmgOperator().setSmootherType(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
      gmgSolver->gmgOperator().setSmootherOverlap(smootherOverlap);
      fineSolver = Teuchos::rcp( gmgSolver );
    }
    
    solution->solve(fineSolver);
    solution->reportTimings();

    if (useGMGSolver) {
      condestForRefinement.push_back(gmgSolver->condest());
      iterationsForRefinement.push_back(gmgSolver->iterationCount());
      
      gmgSolver->gmgOperator().reportTimings();
    }
    
#ifdef HAVE_EPETRAEXT_HDF5
    exporter.exportSolution(solution,varFactory,refIndex+1);
#endif
  }
  solution->setIP(standardGraphNorm); // since we won't be solving again, and want to measure the energy error
  double energyErrorTotal = solution->energyErrorTotal();
  
//  { // DEBUGGING
//    double tol = 1e-14;
//    set<GlobalIndexType> cells = mesh->cellIDsInPartition();
//    for (set<GlobalIndexType>::iterator cellIt = cells.begin(); cellIt != cells.end(); cellIt++) {
//      GlobalIndexType cellID = *cellIt;
//      FieldContainer<double> solnCoeffs = solution->allCoefficientsForCellID(cellID);
//      FieldContainer<double> solnDirectCoeffs = solutionDirect->allCoefficientsForCellID(cellID);
//      for (int i=0; i<solnCoeffs.size(); i++) {
//        double diff = abs(solnCoeffs[i] - solnDirectCoeffs[i]);
//        if (diff > tol) {
//          cout << "cell ID " << cellID << ", soln coefficient " << i << " differs by " << diff << endl;
//        }
//      }
//    }
//  }
  
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
  
  if (!use3D) {
    GnuPlotUtil::writeComputationalMeshSkeleton("cavityFlowRefinedMesh", mesh, true);
  }

  coarseSolver = Teuchos::rcp((Solver*) NULL); // without this when useMumps = true and running on one rank, we see a crash on exit, which may have to do with MPI being finalized before coarseSolver is deleted.

  int col1 = 7, col2 = 8, col3 = 15, col4 = 12, col5 = 15, col6 = 12;
  if (rank==0) {
    cout << "Refinement history:\n";
    if (useGMGSolver)
      cout << setw(col1) << "# elems" << setw(col2) << "# dofs" << setw(col3) << "energy error" << setw(col4) << "condest" << setw(col5) << "iter. count" << setw(col6) << "epsilon" << endl;
    else
      cout << setw(col1) << "# elems" << setw(col2) << "# dofs" << setw(col3) << "energy error" << endl;
  }
  for (int refIndex=0; refIndex < refCount; refIndex++) {
    GlobalIndexType numElements = refinementStrategy.getNumElements(refIndex);
    GlobalIndexType numDofs = refinementStrategy.getNumDofs(refIndex);
    double energyError = refinementStrategy.getEnergyError(refIndex);
    double condest = useGMGSolver ? condestForRefinement[refIndex] : -1;
    int iterCount = useGMGSolver ? iterationsForRefinement[refIndex] : -1;
    double epsilon = useGMGSolver ? tolForRefinement[refIndex] : 0;
    if (rank==0) {
      if (useGMGSolver) {
        cout << setw(col1) << numElements << setw(col2) << numDofs << setw(col3) << setprecision(2) << scientific << energyError;
        cout << setw(col4) << condest << setw(col5) << iterCount << setw(col6) << epsilon << endl;
      } else {
        cout << setw(col1) << numElements << setw(col2) << numDofs << setw(col3) << setprecision(2) << scientific << energyError << endl;
      }
    }
  }
  int numElements = mesh->numActiveElements();
  
  if (rank==0) {
    if (useGMGSolver) {
      double epsilon = useGMGSolver ? tolForRefinement[refCount] : 0;
      cout << setw(col1) << numElements << setw(col2) << numGlobalDofs << setw(col3) << setprecision(2) << scientific << energyErrorTotal;
      cout << setw(col4) << condestForRefinement[refCount] << setw(col5) << iterationsForRefinement[refCount] << setw(col6) << epsilon << endl;
    } else {
      cout << setw(col1) << numElements << setw(col2) << numGlobalDofs << setw(col3) << setprecision(2) << scientific << energyErrorTotal << endl;
    }
  }
  
  return 0;
}
