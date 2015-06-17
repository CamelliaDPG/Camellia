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

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

static const double PI  = 3.141592653589793238462;

int main(int argc, char *argv[])
{
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

  double minTol = 1e-8;

  bool use3D = false;
  int refCount = 10;

  int k = 4; // poly order for field variables
  int delta_k = use3D ? 3 : 2;   // test space enrichment
  int k_coarse = 0;

  bool useMumps = true;
  bool useGMGSolver = true;

  bool enforceOneIrregularity = true;
  bool useStaticCondensation = false;
  bool conformingTraces = false;
  bool applySmoothing = true;
  bool useDiagonalScaling = false; // of the global stiffness matrix in GMGSolver

  bool printRefinementDetails = false;

  bool useWeightedGraphNorm = true; // graph norm scaled according to units, more or less

  int numCells = 2;

  int AztecOutputLevel = 1;
  int gmgMaxIterations = 10000;
  int smootherOverlap = 0;
  double relativeTol = 1e-6;
  double D = 1.0; // characteristic length scale

  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("k_coarse", &k_coarse, "polynomial order for field variables on coarse mesh");
  cmdp.setOption("numRefs",&refCount,"number of refinements");
  cmdp.setOption("D", &D, "domain dimension");
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("enforceOneIrregularity", "dontEnforceOneIrregularity", &enforceOneIrregularity);
  cmdp.setOption("useSmoothing", "useNoSmoothing", &applySmoothing);

  cmdp.setOption("smootherOverlap", &smootherOverlap, "overlap for smoother");

  cmdp.setOption("printRefinementDetails", "dontPrintRefinementDetails", &printRefinementDetails);
  cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
  cmdp.setOption("numCells", &numCells, "number of cells in the initial mesh");
  cmdp.setOption("useScaledGraphNorm", "dontUseScaledGraphNorm", &useWeightedGraphNorm);
//  cmdp.setOption("gmgTol", &gmgTolerance, "tolerance for GMG convergence");
  cmdp.setOption("relativeTol", &relativeTol, "Energy error-relative tolerance for iterative solver.");
  cmdp.setOption("gmgMaxIterations", &gmgMaxIterations, "tolerance for GMG convergence");

  bool enhanceUField = false;
  cmdp.setOption("enhanceUField", "dontEnhanceUField", &enhanceUField);
  cmdp.setOption("useStaticCondensation", "dontUseStaticCondensation", &useStaticCondensation);

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  double width = D, height = D, depth = D;

  VarFactory varFactory;
  // fields:
  VarPtr u = varFactory.fieldVar("u", L2);
  VarPtr sigma = varFactory.fieldVar("\\sigma", VECTOR_L2);

  FunctionPtr n = Function::normal();
  // traces:
  VarPtr u_hat;

  if (conformingTraces)
  {
    u_hat = varFactory.traceVar("\\widehat{u}", u);
  }
  else
  {
    cout << "Note: using non-conforming traces.\n";
    u_hat = varFactory.traceVar("\\widehat{u}", u, L2);
  }
  VarPtr sigma_n_hat = varFactory.fluxVar("\\widehat{\\sigma}_{n}", sigma * n);

  // test functions:
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  BFPtr poissonBF = Teuchos::rcp( new BF(varFactory) );
  FunctionPtr alpha = Function::constant(1); // viscosity

  // tau terms:
  poissonBF->addTerm(sigma / alpha, tau);
  poissonBF->addTerm(-u, tau->div()); // (sigma1, tau1)
  poissonBF->addTerm(u_hat, tau * n);

  // v terms:
  poissonBF->addTerm(- sigma, v->grad()); // (mu sigma1, grad v1)
  poissonBF->addTerm( sigma_n_hat, v);

  int horizontalCells = numCells, verticalCells = numCells, depthCells = numCells;

  vector<double> domainDimensions;
  domainDimensions.push_back(width);
  domainDimensions.push_back(height);

  vector<int> elementCounts;
  elementCounts.push_back(horizontalCells);
  elementCounts.push_back(verticalCells);

  if (use3D)
  {
    domainDimensions.push_back(depth);
    elementCounts.push_back(depthCells);
  }

  MeshPtr mesh, k0Mesh;

  int H1Order = k + 1;
  int H1Order_coarse = k_coarse + 1;
  if (!use3D)
  {
    Teuchos::ParameterList pl;

    map<int,int> trialOrderEnhancements;

    if (enhanceUField)
    {
      trialOrderEnhancements[u->ID()] = 1;
    }

    BFPtr poissonBilinearForm = poissonBF;

    pl.set("useMinRule", true);
    pl.set("bf",poissonBilinearForm);
    pl.set("H1Order", H1Order);
    pl.set("delta_k", delta_k);
    pl.set("horizontalElements", horizontalCells);
    pl.set("verticalElements", verticalCells);
    pl.set("divideIntoTriangles", false);
    pl.set("useConformingTraces", conformingTraces);
    pl.set("trialOrderEnhancements", &trialOrderEnhancements);
    pl.set("x0",(double)0);
    pl.set("y0",(double)0);
    pl.set("width", width);
    pl.set("height",height);

    mesh = MeshFactory::quadMesh(pl);

    pl.set("H1Order", H1Order_coarse);
    k0Mesh = MeshFactory::quadMesh(pl);

  }
  else
  {
    mesh = MeshFactory::rectilinearMesh(poissonBF, domainDimensions, elementCounts, H1Order, delta_k);
    k0Mesh = MeshFactory::rectilinearMesh(poissonBF, domainDimensions, elementCounts, H1Order_coarse, delta_k);
  }

  mesh->registerObserver(k0Mesh); // ensure that the k0 mesh refinements track those of the solution mesh

  RHSPtr rhs = RHS::rhs(); // zero
  FunctionPtr sin_pi_x = Teuchos::rcp( new Sin_ax(PI/D) );
  FunctionPtr sin_pi_y = Teuchos::rcp( new Sin_ay(PI/D) );
  FunctionPtr u_exact = sin_pi_x * sin_pi_y;
  FunctionPtr f = -(2.0 * PI * PI / (D * D)) * sin_pi_x * sin_pi_y;
  rhs->addTerm( f * v );

  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();

  bc->addDirichlet(u_hat, boundary, u_exact);

  IPPtr graphNorm;

  FunctionPtr h = Teuchos::rcp( new hFunction() );

  if (useWeightedGraphNorm)
  {
    graphNorm = IP::ip();
    graphNorm->addTerm( tau->div() ); // u
    graphNorm->addTerm( (h / alpha) * tau - h * v->grad() ); // sigma
    graphNorm->addTerm( v ); // boundary term (adjoint to u)
    graphNorm->addTerm( h * tau );

//    // new effort, with the idea that the test norm should be considered in reference space, basically
//    graphNorm = IP::ip();
//    graphNorm->addTerm( tau->div() ); // u
//    graphNorm->addTerm( tau / h - v->grad() ); // sigma
//    graphNorm->addTerm( v / h ); // boundary term (adjoint to u)
//    graphNorm->addTerm( tau / h );
  }
  else
  {
    map<int, double> trialWeights; // on the squared terms in the trial space norm
    trialWeights[u->ID()] = 1.0 / (D * D);
    trialWeights[sigma->ID()] = 1.0;
    graphNorm = poissonBF->graphNorm(trialWeights, 1.0); // 1.0: weight on the L^2 terms
  }

  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  solution->setUseCondensedSolve(useStaticCondensation);

  mesh->registerSolution(solution); // sign up for projection of old solution onto refined cells.

  double energyThreshold = 0.2;
  RefinementStrategy refinementStrategy( solution, energyThreshold );

  refinementStrategy.setReportPerCellErrors(true);
  refinementStrategy.setEnforceOneIrregularity(enforceOneIrregularity);

  Teuchos::RCP<Solver> coarseSolver, fineSolver;
  if (useMumps)
  {
#ifdef HAVE_AMESOS_MUMPS
    coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
#else
    cout << "useMumps=true, but MUMPS is not available!\n";
    exit(0);
#endif
  }
  else
  {
    coarseSolver = Teuchos::rcp( new KluSolver );
  }
  GMGSolver* gmgSolver;

  if (useGMGSolver)
  {
    double tol = relativeTol;
    int maxIters = gmgMaxIterations;
    BCPtr zeroBCs = bc->copyImposingZero();
    gmgSolver = new GMGSolver(zeroBCs, k0Mesh, graphNorm, mesh, solution->getDofInterpreter(),
                              solution->getPartitionMap(), maxIters, tol, coarseSolver,
                              useStaticCondensation);
    gmgSolver->setAztecOutput(AztecOutputLevel);
    gmgSolver->setApplySmoothingOperator(applySmoothing);

    gmgSolver->setAztecOutput(AztecOutputLevel);
    gmgSolver->setUseConjugateGradient(true);
    gmgSolver->gmgOperator()->setSmootherType(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    gmgSolver->gmgOperator()->setSmootherOverlap(smootherOverlap);

    fineSolver = Teuchos::rcp( gmgSolver );
  }
  else
  {
    fineSolver = coarseSolver;
  }

//  if (rank==0) cout << "experimentally starting by solving with MUMPS on the fine mesh.\n";
//  solution->solve( Teuchos::rcp( new MumpsSolver) );

  solution->solve(fineSolver);

#ifdef HAVE_EPETRAEXT_HDF5
  ostringstream dir_name;
  dir_name << "poissonCavityFlow_k" << k;
  HDF5Exporter exporter(mesh,dir_name.str());
  exporter.exportSolution(solution,varFactory,0);
#endif

#ifdef HAVE_AMESOS_MUMPS
  if (useMumps) coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
#endif

  solution->reportTimings();
  if (useGMGSolver) gmgSolver->gmgOperator()->reportTimings();
  for (int refIndex=0; refIndex < refCount; refIndex++)
  {
    double energyError = solution->energyErrorTotal();
    GlobalIndexType numFluxDofs = mesh->numFluxDofs();
    if (rank==0)
    {
      cout << "Before refinement " << refIndex << ", energy error = " << energyError;
      cout << " (using " << numFluxDofs << " trace degrees of freedom)." << endl;
    }
    bool printToConsole = printRefinementDetails && (rank==0);
    refinementStrategy.refine(printToConsole);

    if (useStaticCondensation)
    {
      CondensedDofInterpreter* condensedDofInterpreter = dynamic_cast<CondensedDofInterpreter*>(solution->getDofInterpreter().get());
      if (condensedDofInterpreter != NULL)
      {
        condensedDofInterpreter->reinitialize();
      }
    }

    GlobalIndexType fineDofs = mesh->globalDofCount();
    GlobalIndexType coarseDofs = k0Mesh->globalDofCount();
    if (rank==0)
    {
      cout << "After refinement, coarse mesh has " << k0Mesh->numActiveElements() << " elements and " << coarseDofs << " dofs.\n";
      cout << "  Fine mesh has " << mesh->numActiveElements() << " elements and " << fineDofs << " dofs.\n";
    }

    if (!use3D)
    {
      ostringstream fineMeshLocation, coarseMeshLocation;
      fineMeshLocation << "poissonFineMesh_k" << k << "_ref" << refIndex;
      GnuPlotUtil::writeComputationalMeshSkeleton(fineMeshLocation.str(), mesh, true); // true: label cells
      coarseMeshLocation << "poissonCoarseMesh_k" << k << "_ref" << refIndex;
      GnuPlotUtil::writeComputationalMeshSkeleton(coarseMeshLocation.str(), k0Mesh, true); // true: label cells
    }

    if (useGMGSolver)   // create fresh fineSolver now that the meshes have changed:
    {
#ifdef HAVE_AMESOS_MUMPS
      if (useMumps) coarseSolver = Teuchos::rcp( new MumpsSolver(512, true) );
#endif
      double tol = max(relativeTol * energyError, minTol);
      int maxIters = gmgMaxIterations;
      BCPtr zeroBCs = bc->copyImposingZero();
      gmgSolver = new GMGSolver(zeroBCs, k0Mesh, graphNorm, mesh, solution->getDofInterpreter(),
                                solution->getPartitionMap(), maxIters, tol, coarseSolver, useStaticCondensation);
      gmgSolver->setAztecOutput(AztecOutputLevel);
      gmgSolver->setApplySmoothingOperator(applySmoothing);
      gmgSolver->setUseDiagonalScaling(useDiagonalScaling);
      fineSolver = Teuchos::rcp( gmgSolver );
    }

    solution->solve(fineSolver);
    solution->reportTimings();
    if (useGMGSolver) gmgSolver->gmgOperator()->reportTimings();

#ifdef HAVE_EPETRAEXT_HDF5
    exporter.exportSolution(solution,varFactory,refIndex+1);
#endif
  }
  double energyErrorTotal = solution->energyErrorTotal();

  GlobalIndexType numFluxDofs = mesh->numFluxDofs();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  if (rank==0)
  {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << numGlobalDofs << " total dofs, including fields).\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
  }

#ifdef HAVE_EPETRAEXT_HDF5
  exporter.exportSolution(solution,varFactory,0);
#endif

  if (!use3D)
  {
    GnuPlotUtil::writeComputationalMeshSkeleton("poissonRefinedMesh", mesh, true);
  }

  coarseSolver = Teuchos::rcp((Solver*) NULL); // without this when useMumps = true and running on one rank, we see a crash on exit, which may have to do with MPI being finalized before coarseSolver is deleted.

  return 0;
}
