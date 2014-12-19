#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

#include "Solver.h"
#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"

#include "AdditiveSchwarz.h"

#include "Teuchos_GlobalMPISession.hpp"

#include "Epetra_Operator_to_Epetra_Matrix.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"

using namespace Camellia;

Teuchos::RCP<Epetra_Operator> CamelliaAdditiveSchwarzPreconditioner(Epetra_RowMatrix* A, int overlapLevel, MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter) {
  Teuchos::RCP<Ifpack_Preconditioner> preconditioner = Teuchos::rcp(new AdditiveSchwarz<Ifpack_Amesos>(A, overlapLevel, mesh, dofInterpreter) );
  
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

Teuchos::RCP<Epetra_Operator> IfPackAdditiveSchwarzPreconditioner(Epetra_RowMatrix* A, int overlapLevel) {
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
  
  MeshPtr _mesh;
  Teuchos::RCP<DofInterpreter> _dofInterpreter;
  GMGOperator::FactorType _schwarzBlockFactorization;
public:
  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner, GMGOperator::FactorType schwarzBlockFactorization) {
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _schwarzBlockFactorization = schwarzBlockFactorization;
    _azOutputLevel = 1;
  }
  
  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner,
              MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter, GMGOperator::FactorType schwarzBlockFactorization) {
    _mesh = mesh;
    _dofInterpreter = dofInterpreter;
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _schwarzBlockFactorization = schwarzBlockFactorization;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _azOutputLevel = 1;
  }
  int solve() {
    AztecOO solver(problem());
    
    solver.SetAztecOption(AZ_solver, AZ_cg);
    
    Epetra_RowMatrix *A = problem().GetMatrix();
    
    Teuchos::RCP<Epetra_Operator> preconditioner;
    if (_mesh != Teuchos::null) {
      preconditioner = CamelliaAdditiveSchwarzPreconditioner(A, _schwarzOverlap, _mesh, _dofInterpreter, _schwarzBlockFactorization);
      
//      Teuchos::RCP< Epetra_CrsMatrix > M;
//      M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, A->RowMatrixRowMap());
//
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "writing preconditioner to /tmp/preconditioner.dat.\n";
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/preconditioner.dat",*M, NULL, NULL, false);
      
    } else {
      preconditioner = IfPackAdditiveSchwarzPreconditioner(A, _schwarzOverlap, _schwarzBlockFactorization);
    }
    
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
    int rank = Teuchos::GlobalMPISession::getRank();
    
    if (rank==0) {
      switch (whyTerminated) {
        case AZ_normal:
  //        cout << "whyTerminated: AZ_normal " << endl;
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

static const double PI  = 3.141592653589793238462;

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
  
  bool useCamelliaAdditiveSchwarz = true;
  
  int numCells = 2;
  
  int AztecOutputLevel = 1;
  int cgMaxIterations = 10000;
  int schwarzOverlap = 0;
  
  double cgTol = 1e-10;
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("useCamelliaAdditiveSchwarz", "useIfPackAdditiveSchwarz", &useCamelliaAdditiveSchwarz);
  cmdp.setOption("precondition", "dontPrecondition", &precondition);

  cmdp.setOption("overlap", &schwarzOverlap, "Schwarz overlap level");
  
  cmdp.setOption("maxIterations", &cgMaxIterations, "Max iterations for CG solve.");
  
  cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
  cmdp.setOption("numCells", &numCells, "number of cells in the initial mesh");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  double width = 1.0, height = 1.0;
  
  PoissonFormulation formulation(2, conformingTraces);
  
  BFPtr poissonBF = formulation.bf();
  
  VarPtr phi_hat = formulation.phi_hat();
  
  int horizontalCells = numCells, verticalCells = numCells;
  
  vector<double> domainDimensions;
  domainDimensions.push_back(width);
  domainDimensions.push_back(height);
  
  vector<int> elementCounts;
  elementCounts.push_back(horizontalCells);
  elementCounts.push_back(verticalCells);
  
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
  
  RHSPtr rhs = RHS::rhs(); // zero
  FunctionPtr sin_pi_x = Teuchos::rcp( new Sin_ax(PI) );
  FunctionPtr sin_pi_y = Teuchos::rcp( new Sin_ay(PI) );
  FunctionPtr phi_exact = sin_pi_x * sin_pi_y;
  FunctionPtr f = phi_exact->dx()->dx() + phi_exact->dy()->dy(); // - (2.0 * PI * PI ) * sin_pi_x * sin_pi_y;
  
  VarPtr q = formulation.q();
  rhs->addTerm( f * q );
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  bc->addDirichlet(phi_hat, boundary, phi_exact);

  IPPtr graphNorm = poissonBF->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  Teuchos::RCP<Solver> solver;
  if (useCamelliaAdditiveSchwarz) {
    solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition,mesh, solution->getDofInterpreter(), schwarzBlockFactorization) );
  } else {
    solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition, schwarzBlockFactorization) );
  }
  
//  solution->setWriteMatrixToFile(true, "/tmp/A_barker.dat");
  
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
