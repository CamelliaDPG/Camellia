//
//  GMGSolver.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/7/14.
//
//

#include "GMGSolver.h"
#include "MPIWrapper.h"

#include "AztecOO.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

using namespace Camellia;

GMGSolver::GMGSolver(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
                     MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
                     int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
  Narrator("GMGSolver"),
  _finePartitionMap(finePartitionMap)
{
  _gmgOperator = Teuchos::rcp(new GMGOperator(zeroBCs,coarseMesh,coarseIP,fineMesh,fineDofInterpreter,
                                              finePartitionMap,coarseSolver, useStaticCondensation));
  
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;

  _computeCondest = false;
  _azOutput = AZ_warnings;

  _useCG = true;
  _azConvergenceOption = AZ_rhs;
}

GMGSolver::GMGSolver(TSolutionPtr<double> fineSolution, MeshPtr coarseMesh, int maxIters, double tol,
                     Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
  Narrator("GMGSolver"),
  _finePartitionMap(fineSolution->getPartitionMap())
{
  _gmgOperator = Teuchos::rcp(new GMGOperator(fineSolution->bc()->copyImposingZero(),coarseMesh,
                                              fineSolution->ip(), fineSolution->mesh(), fineSolution->getDofInterpreter(),
                                              _finePartitionMap, coarseSolver, useStaticCondensation));
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;

  _computeCondest = true;
  _azOutput = AZ_warnings;

  _useCG = true;
  _azConvergenceOption = AZ_rhs;
}

GMGSolver::GMGSolver(TSolutionPtr<double> fineSolution, int maxIters, double tol, int H1OrderCoarse,
                     Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
Narrator("GMGSolver"),
_finePartitionMap(fineSolution->getPartitionMap())
{
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
  
  _computeCondest = true;
  _azOutput = AZ_warnings;
  
  _useCG = true;
  _azConvergenceOption = AZ_rhs;
  
  // notion here is that we build a hierarchy of meshes in some intelligent way
  // for now, we jump in p from whatever it is on the fine mesh to H1OrderCoarse, and then
  // do single h-coarsening steps until we reach the coarsest topology.
  
  vector<MeshPtr> meshesCoarseToFine;
  MeshPtr fineMesh = fineSolution->mesh();
  meshesCoarseToFine.push_back(fineMesh);
  
  VarFactoryPtr vf = fineMesh->bilinearForm()->varFactory();
  int delta_k = 1; // this shouldn't matter for meshes outside the finest
  MeshPtr mesh_pCoarsened = Teuchos::rcp( new Mesh(fineMesh->getTopology(), vf, H1OrderCoarse, delta_k) );
  
  meshesCoarseToFine.insert(meshesCoarseToFine.begin(), mesh_pCoarsened);
  
  
  
  // TODO: finish implementing this
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "This GMGSolver constructor not yet completed!");
  
  // once we have a list of meshes, use gmgOperatorFromMeshSequence to build the operator
}

GMGSolver::GMGSolver(TSolutionPtr<double> fineSolution, const std::vector<MeshPtr> &meshesCoarseToFine, int maxIters,
                     double tol, GMGOperator::MultigridStrategy multigridStrategy, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation) :
Narrator("GMGSolver"),
_finePartitionMap(fineSolution->getPartitionMap())
{
  _maxIters = maxIters;
  _printToConsole = false;
  _tol = tol;
  
  _computeCondest = true;
  _azOutput = AZ_warnings;
  
  if ((multigridStrategy == GMGOperator::FULL_MULTIGRID_V) && (multigridStrategy == GMGOperator::FULL_MULTIGRID_W))
    _useCG = false; // Full multigrid is not symmetric
  else
    _useCG = true;
  _azConvergenceOption = AZ_rhs;
  
  _gmgOperator = gmgOperatorFromMeshSequence(meshesCoarseToFine, fineSolution, multigridStrategy, coarseSolver, useStaticCondensation);
}

double GMGSolver::condest()
{
  return _condest;
}

vector<int> GMGSolver::getIterationCountLog()
{
  return _iterationCountLog;
}

int GMGSolver::iterationCount()
{
  return _iterationCount;
}

Teuchos::RCP<GMGOperator> GMGSolver::gmgOperatorFromMeshSequence(const std::vector<MeshPtr> &meshesCoarseToFine, SolutionPtr fineSolution,
                                                                 GMGOperator::MultigridStrategy multigridStrategy,
                                                                 SolverPtr coarseSolver, bool useStaticCondensationInCoarseSolve)
{
  TEUCHOS_TEST_FOR_EXCEPTION(meshesCoarseToFine.size() < 2, std::invalid_argument, "meshesCoarseToFine must have at least two meshes");
  Teuchos::RCP<GMGOperator> coarseOperator = Teuchos::null, finerOperator = Teuchos::null, finestOperator = Teuchos::null;
  
  Teuchos::RCP<DofInterpreter> fineDofInterpreter = fineSolution->getDofInterpreter();
  IPPtr ip = fineSolution->ip();
  BCPtr zeroBCs = fineSolution->bc()->copyImposingZero();
  Epetra_Map finePartitionMap = fineSolution->getPartitionMap();
  
  for (int i=meshesCoarseToFine.size()-1; i>0; i--)
  {
    MeshPtr fineMesh = meshesCoarseToFine[i];
    MeshPtr coarseMesh = meshesCoarseToFine[i-1];
    if (i>1)
    {
      coarseOperator = Teuchos::rcp(new GMGOperator(zeroBCs, coarseMesh, ip, fineMesh, fineDofInterpreter, finePartitionMap,
                                                    useStaticCondensationInCoarseSolve));
    }
    else
    {
      coarseOperator = Teuchos::rcp(new GMGOperator(zeroBCs, coarseMesh, ip, fineMesh, fineDofInterpreter, finePartitionMap,
                                                    coarseSolver, useStaticCondensationInCoarseSolve));
    }
    coarseOperator->setSmootherType(GMGOperator::CAMELLIA_ADDITIVE_SCHWARZ);
    coarseOperator->setUseSchwarzScalingWeight(true);
    coarseOperator->setMultigridStrategy(multigridStrategy);
    bool hRefined = fineMesh->numActiveElements() > coarseMesh->numActiveElements();
    coarseOperator->setUseHierarchicalNeighborsForSchwarz(hRefined);
    if (hRefined)
    {
      coarseOperator->setSmootherOverlap(1);
      coarseOperator->setUseSchwarzDiagonalWeight(false); // empirically, this doesn't work well for h-multigrid
    }
    else
    {
      coarseOperator->setUseSchwarzDiagonalWeight(false); // not sure which is better; use false for now
    }
    

    if (finerOperator != Teuchos::null)
    {
      finerOperator->setCoarseOperator(coarseOperator);
    }
    else
    {
      finestOperator = coarseOperator;
    }
    
    finerOperator = coarseOperator;
    finePartitionMap = finerOperator->getCoarseSolution()->getPartitionMap();
    fineDofInterpreter = finerOperator->getCoarseSolution()->getDofInterpreter();
  }
  return finestOperator;
}

void GMGSolver::setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap)
{
  _gmgOperator->setFineMesh(fineMesh, finePartitionMap);
}

void GMGSolver::setPrintToConsole(bool printToConsole)
{
  _printToConsole = printToConsole;
}

void GMGSolver::setTolerance(double tol)
{
  _tol = tol;
}

int GMGSolver::resolve()
{
  bool buildCoarseStiffness = false; // won't have changed since solve() was called
  return solve(buildCoarseStiffness);
}

int GMGSolver::solve()
{
  bool buildCoarseStiffness = true;
  return solve(buildCoarseStiffness);
}

int GMGSolver::solve(bool buildCoarseStiffness)
{
  int rank = Teuchos::GlobalMPISession::getRank();

  Epetra_LinearProblem problem(_stiffnessMatrix.get(), _lhs.get(), _rhs.get());
  AztecOO solver(problem);

  Epetra_CrsMatrix *A = dynamic_cast<Epetra_CrsMatrix *>( problem.GetMatrix() );

  if (A == NULL)
  {
    cout << "Error: GMGSolver requires an Epetra_CrsMatrix.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: GMGSolver requires an Epetra_CrsMatrix.\n");
  }

  //  EpetraExt::RowMatrixToMatlabFile("/tmp/A_pre_scaling.dat",*A);

  //  Epetra_MultiVector *b = problem().GetRHS();
  //  EpetraExt::MultiVectorToMatlabFile("/tmp/b_pre_scaling.dat",*b);

  //  Epetra_MultiVector *x = problem().GetLHS();
  //  EpetraExt::MultiVectorToMatlabFile("/tmp/x_initial_guess.dat",*x);

//  const Epetra_Map* map = &A->RowMatrixRowMap();

  if (buildCoarseStiffness)
  {
    _gmgOperator->setFineStiffnessMatrix(A);
  }

  solver.SetAztecOption(AZ_scaling, AZ_none);
  if (_useCG)
  {
    if (_computeCondest)
    {
      solver.SetAztecOption(AZ_solver, AZ_cg_condnum);
    }
    else
    {
      solver.SetAztecOption(AZ_solver, AZ_cg);
    }
  }
  else
  {
    solver.SetAztecOption(AZ_kspace, 200); // default is 30
    if (_computeCondest)
    {
      solver.SetAztecOption(AZ_solver, AZ_gmres_condnum);
    }
    else
    {
      solver.SetAztecOption(AZ_solver, AZ_gmres);
    }
  }

  solver.SetPrecOperator(_gmgOperator.get());
  //  solver.SetAztecOption(AZ_precond, AZ_none);
  solver.SetAztecOption(AZ_precond, AZ_user_precond);
  solver.SetAztecOption(AZ_conv, _azConvergenceOption);
  //  solver.SetAztecOption(AZ_output, AZ_last);
  solver.SetAztecOption(AZ_output, _azOutput);

  int solveResult = solver.Iterate(_maxIters,_tol);

  const double* status = solver.GetAztecStatus();
  int remainingIters = _maxIters;

  int whyTerminated = status[AZ_why];
  int maxRestarts = 1;
  int numRestarts = 0;
  while ((whyTerminated==AZ_loss) && (numRestarts < maxRestarts))
  {
    remainingIters -= status[AZ_its];
    if (rank==0) cout << "Aztec warned that the recursive residual indicates convergence even though the true residual is too large.  Restarting with the new solution as initial guess, with maxIters = " << remainingIters << endl;
    solveResult = solver.Iterate(remainingIters,_tol);
    whyTerminated = status[AZ_why];
    numRestarts++;
  }
  remainingIters -= status[AZ_its];
  _iterationCount = _maxIters - remainingIters;
  _condest = solver.Condest(); // will be -1 if running without condest

  if (rank==0)
  {
    switch (whyTerminated)
    {
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
  
  _iterationCountLog.push_back(_iterationCount);

  return solveResult;
}

void GMGSolver::setAztecConvergenceOption(int value)
{
  _azConvergenceOption = value;
}

void GMGSolver::setAztecOutput(int value)
{
  _azOutput = value;
}

void GMGSolver::setComputeConditionNumberEstimate(bool value)
{
  _computeCondest = value;
}

void GMGSolver::setPrintIterationCount(bool value)
{
  _printIterationCountIfNoAzOutput = value;
}

void GMGSolver::setUseConjugateGradient(bool value)
{
  _useCG = value;
}
