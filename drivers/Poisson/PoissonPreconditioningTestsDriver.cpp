#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshFactory.h"
#include "SolutionExporter.h"
#include <Teuchos_GlobalMPISession.hpp>
#include "GnuPlotUtil.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "EpetraExt_MatrixMatrix.h"

#include "Solver.h"
#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "AdditiveSchwarz.h"
#include "MeshFactory.h"

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
  
  int _iterationCount;
  
  int _azOutputLevel;
  
  MeshPtr _mesh;
  Teuchos::RCP<DofInterpreter> _dofInterpreter;
public:
  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner) {
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _azOutputLevel = 1;
  }
  
  AztecSolver(int maxIters, double tol, int schwarzOverlapLevel, bool useSchwarzPreconditioner,
              MeshPtr mesh, Teuchos::RCP<DofInterpreter> dofInterpreter) {
    _mesh = mesh;
    _dofInterpreter = dofInterpreter;
    _maxIters = maxIters;
    _tol = tol;
    _schwarzOverlap = schwarzOverlapLevel;
    _useSchwarzPreconditioner = useSchwarzPreconditioner;
    _azOutputLevel = 1;
  }
  void setAztecOutputLevel(int AztecOutputLevel) {
    _azOutputLevel = AztecOutputLevel;
  }
  
  int solve() {
    AztecOO solver(problem());
    
    solver.SetAztecOption(AZ_solver, AZ_cg);
    
    Epetra_RowMatrix *A = problem().GetMatrix();
    
    Teuchos::RCP<Epetra_Operator> preconditioner;
    if (_mesh != Teuchos::null) {
      preconditioner = CamelliaAdditiveSchwarzPreconditioner(A, _schwarzOverlap, _mesh, _dofInterpreter);
      
//      Teuchos::RCP< Epetra_CrsMatrix > M;
//      M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, A->RowMatrixRowMap());
//      
//      int rank = Teuchos::GlobalMPISession::getRank();
//      if (rank==0) cout << "writing preconditioner to /tmp/preconditioner.dat.\n";
//      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/preconditioner.dat",*M, NULL, NULL, false);
      
    } else {
      preconditioner = IfPackAdditiveSchwarzPreconditioner(A, _schwarzOverlap);
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
    
    _iterationCount = status[AZ_its];
    
    return solveResult;
  }
  Teuchos::RCP< Epetra_CrsMatrix > getPreconditionerMatrix(const Epetra_Map &map) {
    Epetra_RowMatrix *A = problem().GetMatrix();
    Teuchos::RCP<Epetra_Operator> preconditioner;
    if (_mesh != Teuchos::null) {
      preconditioner = CamelliaAdditiveSchwarzPreconditioner(A, _schwarzOverlap, _mesh, _dofInterpreter);
      
      Teuchos::RCP< Epetra_CrsMatrix > M;
      M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, A->RowMatrixRowMap());
      
      int rank = Teuchos::GlobalMPISession::getRank();
      if (rank==0) cout << "writing preconditioner to /tmp/preconditioner.dat.\n";
      EpetraExt::RowMatrixToMatrixMarketFile("/tmp/preconditioner.dat",*M, NULL, NULL, false);
      
    } else {
      preconditioner = IfPackAdditiveSchwarzPreconditioner(A, _schwarzOverlap);
    }
    
    return Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*preconditioner, map);
  }
  int iterationCount() {
    return _iterationCount;
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

void run(int &iterationCount, int spaceDim, int numCells, int k, int delta_k, bool conformingTraces, bool precondition,
         bool schwarzOnly, bool useCamelliaAdditiveSchwarz, int schwarzOverlap, double cgTol,
         int cgMaxIterations, int AztecOutputLevel, bool reportTimings, bool reportEnergyError) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  double width = 1.0; // in each dimension
  
  PoissonFormulation formulation(spaceDim, conformingTraces);
  
  BFPtr poissonBF = formulation.bf();
  
  VarPtr phi_hat = formulation.phi_hat();
  
  MeshPtr mesh;
  
  int H1Order = k + 1;
  
  Teuchos::RCP<BilinearForm> poissonBilinearForm = poissonBF;
  
  vector<double> dimensions;
  vector<int> elementCounts;
  for (int d=0; d<spaceDim; d++) {
    dimensions.push_back(width);
    elementCounts.push_back(numCells);
  }
  mesh = MeshFactory::rectilinearMesh(poissonBF, dimensions, elementCounts, H1Order, delta_k);
  
  int H1Order_coarse = 0 + 1;
  MeshPtr k0Mesh = MeshFactory::rectilinearMesh(poissonBF, dimensions, elementCounts, H1Order_coarse, delta_k);
  
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = Function::constant(1.0);
  
  VarPtr q = formulation.q();
  rhs->addTerm( f * q );
  
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = SpatialFilter::allSpace();
  bc->addDirichlet(phi_hat, boundary, Function::zero());
  
  IPPtr graphNorm = poissonBF->graphNorm();
  
  SolutionPtr solution = Solution::solution(mesh, bc, rhs, graphNorm);
  
  Teuchos::RCP<Solver> solver;
  if (!precondition) {
    solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition) );
    ((AztecSolver*) solver.get())->setAztecOutputLevel(AztecOutputLevel);
  } else if (schwarzOnly) {
    if (useCamelliaAdditiveSchwarz) {
      solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition,mesh, solution->getDofInterpreter()) );
    } else {
      solver = Teuchos::rcp( new AztecSolver(cgMaxIterations,cgTol,schwarzOverlap,precondition) );
    }
    ((AztecSolver*) solver.get())->setAztecOutputLevel(AztecOutputLevel);
  } else {
    BCPtr zeroBCs = bc->copyImposingZero();
    bool saveFactorization = true;
    Teuchos::RCP<Solver> coarseSolver = Teuchos::rcp( new SuperLUDistSolver(saveFactorization) );
    bool useStaticCondensation = false;
    GMGSolver* gmgSolver = new GMGSolver(zeroBCs, k0Mesh, graphNorm, mesh, solution->getDofInterpreter(),
                                         solution->getPartitionMap(), cgMaxIterations, cgTol, coarseSolver,
                                         useStaticCondensation);
    gmgSolver->setAztecOutput(AztecOutputLevel);
    
    gmgSolver->setUseConjugateGradient(true);
    if (useCamelliaAdditiveSchwarz) {
      gmgSolver->gmgOperator().setSmootherType(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    } else {
      gmgSolver->gmgOperator().setSmootherType(GMGOperator::IFPACK_ADDITIVE_SCHWARZ);
    }
    gmgSolver->gmgOperator().setSmootherOverlap(schwarzOverlap);
    solver = Teuchos::rcp( gmgSolver ); // we use "new" above, so we can let this RCP own the memory
  }
  
  //  solution->setWriteMatrixToFile(true, "/tmp/A_poisson.dat");
  
  solution->solve(solver);
  
  if (!precondition) {
    iterationCount = ((AztecSolver *) solver.get())->iterationCount();
  } else if (schwarzOnly) {
    iterationCount = ((AztecSolver *) solver.get())->iterationCount();
  } else {
    iterationCount = ((GMGSolver *) solver.get())->iterationCount();
  }
  
  if (reportTimings) solution->reportTimings();
  double energyErrorTotal = solution->energyErrorTotal();
  
  //  Teuchos::RCP< Epetra_CrsMatrix > A = solution->getStiffnessMatrix();
  //  Teuchos::RCP< Epetra_CrsMatrix > M;
  //  if (schwarzOnly) {
  //    M = ((AztecSolver*)solver.get())->getPreconditionerMatrix(A->DomainMap());
  //  } else {
  //    GMGOperator* op = &((GMGSolver*)solver.get())->gmgOperator();
  //    M = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*op, A->DomainMap());
  //    Teuchos::RCP< Epetra_CrsMatrix > A_coarse = op->getCoarseStiffnessMatrix();
  //    Teuchos::RCP< Epetra_CrsMatrix > A_coarse_inverse = Epetra_Operator_to_Epetra_Matrix::constructInverseMatrix(*A_coarse, A_coarse->DomainMap());
  //    if (rank==0) cout << "writing A_coarse to /tmp/A_coarse_poisson.dat.\n";
  //    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/A_coarse_poisson.dat",*A_coarse, NULL, NULL, false);
  //    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/A_coarse_inv_poisson.dat",*A_coarse_inverse, NULL, NULL, false);
  
  //    Teuchos::RCP< Epetra_CrsMatrix > S = op->getSmootherAsMatrix();
  //    EpetraExt::RowMatrixToMatrixMarketFile("/tmp/S.dat",*S, NULL, NULL, false);
  //  }
  
  //  if (rank==0) cout << "writing M (preconditioner) to /tmp/M_poisson.dat.\n";
  //  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/M_poisson.dat",*M, NULL, NULL, false);
  
  //  Epetra_CrsMatrix AM(::Copy, A->DomainMap(), 0);
  //  int err = EpetraExt::MatrixMatrix::Multiply(*A, false, *M, false, AM);
  //
  //  AM.FillComplete();
  //
  //  EpetraExt::RowMatrixToMatrixMarketFile("/tmp/AM_poisson.dat",AM, NULL, NULL, false);
  
  GlobalIndexType numFluxDofs = mesh->numFluxDofs();
  GlobalIndexType numGlobalDofs = mesh->numGlobalDofs();
  if ((rank==0) && reportEnergyError) {
    cout << "Mesh has " << mesh->numActiveElements() << " elements and " << numFluxDofs << " trace dofs (";
    cout << numGlobalDofs << " total dofs, including fields).\n";
    cout << "Energy error: " << energyErrorTotal << endl;
  }
}

void runMany(int spaceDim, int delta_k, bool conformingTraces, double cgTol, int cgMaxIterations, int aztecOutputLevel) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  vector<bool> preconditionValues;
  preconditionValues.push_back(false);
  preconditionValues.push_back(true);
  
  vector<int> kValues;
  kValues.push_back(1);
  kValues.push_back(2);
  kValues.push_back(4);
  if (spaceDim < 3) kValues.push_back(8);
  if (spaceDim < 2) kValues.push_back(16);
  
  vector<int> numCellsValues;
  int numCells = 2;
  while (pow((double)numCells,spaceDim) <= Teuchos::GlobalMPISession::getNProc()) {
    // want to do as many as we can with just one cell per processor
    numCellsValues.push_back(numCells);
    numCells *= 2;
  }
  
  vector<int> overlapValues;
  
  ostringstream results;
  results << "Preconditioner\tSmoother\tOverlap\tnum_cells\th\tk\tIterations\n";
  
  for (vector<bool>::iterator preconditionChoiceIt = preconditionValues.begin(); preconditionChoiceIt != preconditionValues.end(); preconditionChoiceIt++) {
    bool precondition = *preconditionChoiceIt;
    
    vector<bool> schwarzOnlyValues, useCamelliaSchwarzValues;
    if (precondition) {
      schwarzOnlyValues.push_back(false);
      schwarzOnlyValues.push_back(true);
      useCamelliaSchwarzValues.push_back(false);
      useCamelliaSchwarzValues.push_back(true);
      overlapValues.push_back(0);
      overlapValues.push_back(1);
      if (spaceDim < 3) overlapValues.push_back(2);
    } else {
      // schwarzOnly and useCamelliaSchwarz ignored; just use one of them
      schwarzOnlyValues.push_back(false);
      useCamelliaSchwarzValues.push_back(false);
      overlapValues.push_back(0);
    }
    for (vector<bool>::iterator schwarzOnlyChoiceIt = schwarzOnlyValues.begin(); schwarzOnlyChoiceIt != schwarzOnlyValues.end(); schwarzOnlyChoiceIt++) {
      bool schwarzOnly = *schwarzOnlyChoiceIt;
      for (vector<bool>::iterator useCamelliaSchwarzChoiceIt = useCamelliaSchwarzValues.begin(); useCamelliaSchwarzChoiceIt != useCamelliaSchwarzValues.end(); useCamelliaSchwarzChoiceIt++) {
        bool useCamelliaSchwarz = *useCamelliaSchwarzChoiceIt;
        
        string S_str; // smoother choice description
        if (!precondition) {
          S_str = "None";
        } else {
          if (useCamelliaSchwarz) {
            S_str = "Schwarz (geometric)";
          } else {
            S_str = "Schwarz (algebraic)";
          }
        }
        
        string M_str; // preconditioner descriptor for output
        if (!precondition) {
          M_str = "None";
        } else {
          if (schwarzOnly) {
            M_str = S_str;
            S_str = "-"; // no smoother
          } else {
            M_str = "GMG";
          }
        }
        
        for (vector<int>::iterator overlapValueIt = overlapValues.begin(); overlapValueIt != overlapValues.end(); overlapValueIt++) {
          int overlapValue = *overlapValueIt;
          for (vector<int>::iterator numCellsValueIt = numCellsValues.begin(); numCellsValueIt != numCellsValues.end(); numCellsValueIt++) {
            int numCells1D = *numCellsValueIt;
            for (vector<int>::iterator kValueIt = kValues.begin(); kValueIt != kValues.end(); kValueIt++) {
              int k = *kValueIt;
              
              int iterationCount;
              bool reportTimings = false;
              bool reportEnergyError = false;
              run(iterationCount, spaceDim, numCells1D, k, delta_k, conformingTraces,
                  precondition, schwarzOnly, useCamelliaSchwarz, overlapValue,
                  cgTol, cgMaxIterations, aztecOutputLevel, reportTimings, reportEnergyError);
              
              double h = 1.0 / numCells1D;
              int numCells = pow((double)numCells1D, spaceDim);
              
              results << M_str << "\t" << S_str << "\t" << overlapValue << "\t" << numCells << "\t";
              results << h << "\t" << k << "\t" << iterationCount << endl;
            }
          }
        }
      }
    }
    if (rank==0) cout << results.str(); // output results so far
  }
  
  if (rank == 0) {
    ostringstream filename;
    filename << "PoissonDriver" << spaceDim << "D_results.dat";
    ofstream fout(filename.str().c_str());
    fout << results.str();
    fout.close();
    cout << "Wrote results to " << filename.str() << ".\n";
  }
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
  
  int k = 1; // poly order for field variables
  int delta_k = -1;   // test space enrichment; -1 for default detection (defaults to spaceDim)
  
  bool conformingTraces = true;
  bool precondition = true;
  
  int numCells = 2;
  
  int AztecOutputLevel = 1;
  int cgMaxIterations = 10000;
  int schwarzOverlap = 0;
  
  int spaceDim = 1;
  
  bool useCamelliaAdditiveSchwarz = true;
  
  bool schwarzOnly = true;
  
  double cgTol = 1e-10;
  
  bool runAutomatic = false;
  
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");

  cmdp.setOption("useSchwarzPreconditioner", "useGMGPreconditioner", &schwarzOnly);
  cmdp.setOption("useCamelliaAdditiveSchwarz", "useIfPackAdditiveSchwarz", &useCamelliaAdditiveSchwarz);

  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("precondition", "dontPrecondition", &precondition);
  
  cmdp.setOption("overlap", &schwarzOverlap, "Schwarz overlap level");

  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (1 to 3)");
  
  cmdp.setOption("azOutput", &AztecOutputLevel, "Aztec output level");
  cmdp.setOption("numCells", &numCells, "number of cells in the initial mesh");
  
  cmdp.setOption("runMany", "runOne", &runAutomatic, "Run in automatic mode (ignores several input parameters)");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  if (delta_k==-1) delta_k = spaceDim;
  
  if (! runAutomatic) {
    int iterationCount;
    bool reportTimings = true, reportEnergyError = true;
    
    run(iterationCount, spaceDim, numCells, k, delta_k, conformingTraces,
        precondition, schwarzOnly, useCamelliaAdditiveSchwarz, schwarzOverlap,
        cgTol, cgMaxIterations, AztecOutputLevel, reportTimings, reportEnergyError);
    
    if (rank==0) cout << "Iteration count: " << iterationCount << endl;
  } else {
    if (rank==0) {
      cout << "Running in automatic mode, with spaceDim " << spaceDim;
      cout << ", delta_k = " << delta_k << ", ";
      if (conformingTraces)
        cout << "conforming traces, ";
      else
        cout << "non-conforming traces, ";
      cout << "CG tolerance = " << cgTol << ", max iterations = " << cgMaxIterations << endl;
    }
    
    runMany(spaceDim, delta_k, conformingTraces, cgTol, cgMaxIterations, AztecOutputLevel);
  }
  return 0;
}