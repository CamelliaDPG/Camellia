#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

#include "Solver.h"
#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"

Teuchos::RCP<Epetra_Operator> additiveSchwarzPreconditioner(Epetra_RowMatrix* A, int overlapLevel) {
  Teuchos::RCP<Ifpack_Preconditioner> preconditioner = Teuchos::rcp(new Ifpack_AdditiveSchwarz<Ifpack_Amesos>(A, overlapLevel) );
  
  Teuchos::ParameterList List;
  
  List.set("schwarz: combine mode", "Add"); // The PDF doc says to use "Insert" to maintain symmetry, but the HTML docs (which are more recent) say to use "Add".  http://trilinos.org/docs/r11.10/packages/ifpack/doc/html/index.html
  int err = preconditioner->SetParameters(List);
  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->SetParameters() returned with err " << err << endl;
  }
  
  
  err = preconditioner->Initialize();
  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Initialize() returned with err " << err << endl;
  }
  err = preconditioner->Compute();
  
  if (err != 0) {
    cout << "WARNING: In additiveSchwarzPreconditioner, preconditioner->Compute() returned with err = " << err << endl;
  }
  
  return preconditioner;
}

class AztecSolver : public Solver {
  int _maxIters;
  double _tol;
  int _schwarzOverlap;
  bool _useSchwarzPreconditioner;
  
  int _azOutputLevel;
public:
  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner) {
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _azOutputLevel = 1;
  }
  int solve() {
    AztecOO solver(problem());
    
    solver.SetAztecOption(AZ_solver, AZ_cg_condnum);
    
    Epetra_RowMatrix *A = problem().GetMatrix();
    
    Teuchos::RCP<Epetra_Operator> preconditioner = additiveSchwarzPreconditioner(A, _schwarzOverlap);
    
    if (_useSchwarzPreconditioner) {
      solver.SetPrecOperator(preconditioner.get());
      solver.SetAztecOption(AZ_precond, AZ_user_precond);
    } else {
      solver.SetAztecOption(AZ_precond, AZ_none);
    }
    
    solver.SetAztecOption(AZ_output, _azOutputLevel);
    
    solver.SetAztecOption(AZ_conv, AZ_r0); // convergence is relative to the initial residual
    
    int solveResult = solver.Iterate(_maxIters,_tol);
    
    const double* status = solver.GetAztecStatus();
    int whyTerminated = status[AZ_why];
    switch (whyTerminated) {
      case AZ_normal:
        cout << "whyTerminated: AZ_normal " << endl;
        break;
      case AZ_param:
        cout << "whyTerminated: AZ_param " << endl;
        break;
      case AZ_breakdown:
        cout << "whyTerminated: AZ_breakdown " << endl;
        break;
      case AZ_loss:
        cout << "whyTerminated: AZ_loss " << endl;
        break;
      case AZ_ill_cond:
        cout << "whyTerminated: AZ_ill_cond " << endl;
        break;
      case AZ_maxits:
        cout << "whyTerminated: AZ_maxits " << endl;
        break;
      default:
        break;
    }
    
    return solveResult;
  }
};

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

#include "PoissonFormulation.h"

#include "GMGSolver.h"

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
  
  int k = 1; // poly order for field variables
  int delta_k = 2;   // test space enrichment
  
  bool conformingTraces = true;
  bool precondition = true;
  
  int numCells = 2;
  
  int AztecOutputLevel = 1;
  int cgMaxIterations = 10000;
  int schwarzOverlap = 0;
  
  bool schwarzOnly = true;
  
  bool useStaticCondensation = false;
  
  double cgTol = 1e-10;
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  
  cmdp.setOption("useSchwarzPreconditioner", "useGMGPreconditioner", &schwarzOnly);
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("precondition", "dontPrecondition", &precondition);
  
  cmdp.setOption("useStaticCondensation", "dontUseStaticCondensation", &useStaticCondensation);
  
  cmdp.setOption("overlap", &schwarzOverlap, "Schwarz overlap level");
  
  cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
  cmdp.setOption("numCells", &numCells, "number of cells in the initial mesh");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  int horizontalCells = numCells, verticalCells = numCells;
  double width = 1.0, height = 1.0;
  
  int spaceDim = 2;
  PoissonFormulation formulation(spaceDim, conformingTraces);
  
  BFPtr poissonBF = formulation.bf();
  
  VarPtr phi_hat = formulation.phi_hat();
  
  MeshPtr mesh;
  
  int H1Order = k + 1;
  
  Teuchos::ParameterList pl;
  
  map<int,int> trialOrderEnhancements;
  Teuchos::RCP<BilinearForm> poissonBilinearForm = poissonBF;
  
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
  
  // coarse mesh:
  int H1Order_coarse = 0 + 1;
  pl.set("H1Order", H1Order_coarse);
  MeshPtr k0Mesh = MeshFactory::quadMesh(pl);
  
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = Function::constant(1.0);
  
  VarPtr q = formulation.q();
  rhs->addTerm( f * q );
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  bc->addDirichlet(phi_hat, boundary, Function::zero());
  
  IPPtr graphNorm = poissonBF->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  solution->setUseCondensedSolve(useStaticCondensation);
  
  Teuchos::RCP<Solver> solver;
  if (schwarzOnly) {
    solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition) );
  } else {
    BCPtr zeroBCs = bc->copyImposingZero();
    Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new KluSolver );
    GMGSolver* gmgSolver = new GMGSolver(zeroBCs, k0Mesh, graphNorm, mesh, solution->getDofInterpreter(),
                              solution->getPartitionMap(), cgMaxIterations, cgTol, coarseSolver,
                              useStaticCondensation);
    gmgSolver->setAztecOutput(AztecOutputLevel);
    
    gmgSolver->setUseConjugateGradient(true);
    gmgSolver->gmgOperator().setSmootherType(GMGOperator::ADDITIVE_SCHWARZ);
    gmgSolver->gmgOperator().setSmootherOverlap(schwarzOverlap);
    solver = Teuchos::rcp( gmgSolver ); // we use "new" above, so we can let this RCP own the memory
  }
  
  solution->setWriteMatrixToFile(true, "/tmp/A_poisson.dat");
  solution->solve(solver);
  
  solution->reportTimings();
  double energyErrorTotal = solution->energyErrorTotal();
  
  GlobalIndexType numFluxDofs = mesh->numFluxDofs();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  if (rank==0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << numGlobalDofs << " total dofs, including fields).\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
  }
  
  return 0;
}