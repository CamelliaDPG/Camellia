//
//  GMGSolver.h
//  Camellia
//
//  Created by Nate Roberts on 7/7/14.
//
//

#ifndef __Camellia_debug__GMGSolver__
#define __Camellia_debug__GMGSolver__

#include "Solver.h"
#include "Mesh.h"
#include "GMGOperator.h"
#include "Narrator.h"

#include "Teuchos_ParameterList.hpp"

namespace Camellia
{
class GMGSolver : public Solver, public Narrator
{
  int _maxIters;
  bool _printToConsole;
  double _tol;

  Epetra_Map _finePartitionMap;

  Teuchos::RCP<GMGOperator> _gmgOperator;

  bool _computeCondest;

  int _azOutput;

  bool _useCG; // otherwise, will use GMRES

  // info about the last call to solve()
  double _condest; // -1 if none exists
  int _iterationCount;
  
  int _azConvergenceOption; // defaults to AZ_rhs

  bool _printIterationCountIfNoAzOutput;

  std::vector< int > _iterationCountLog; // each time solve() is called, we push_back the number of iterations we run

  int solve(bool rebuildCoarseStiffness);
  
  static Teuchos::RCP<GMGOperator> gmgOperatorFromMeshSequence(const std::vector<MeshPtr> &meshesCoarseToFine, SolutionPtr fineSolution,
                                                               GMGOperator::MultigridStrategy multigridStrategy, SolverPtr coarseSolver,
                                                               bool useStaticCondensationInCoarseSolve, bool useDiagonalSchwarzWeighting);
public:
  GMGSolver(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter,
            Epetra_Map finePartitionMap, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation);
  GMGSolver(TSolutionPtr<double> fineSolution, MeshPtr coarseMesh, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation);
  GMGSolver(TSolutionPtr<double> fineSolution, int maxIters, double tol, int H1OrderCoarse = 1,
            Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true), bool useStaticCondensation = false);
  GMGSolver(TSolutionPtr<double> fineSolution, const std::vector<MeshPtr> &meshesCoarseToFine, int maxIters, double tol,
            GMGOperator::MultigridStrategy multigridStrategy = GMGOperator::W_CYCLE,
            Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true), bool useStaticCondensation = false,
            bool useDiagonalSchwarzWeighting = false);

  double condest();

  int iterationCount();

  void setPrintToConsole(bool printToConsole);

  int resolve();
  int solve();

  void setComputeConditionNumberEstimate(bool value);

  void setTolerance(double tol);

  Teuchos::RCP<GMGOperator> gmgOperator()
  {
    return _gmgOperator;
  }

  void setAztecConvergenceOption(int value);
  void setAztecOutput(int value);

  void setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap);

  void setUseConjugateGradient(bool value); // otherwise will use GMRES

  void setPrintIterationCount(bool value);

  vector<int> getIterationCountLog();
  
  static std::vector<MeshPtr> meshesForMultigrid(MeshPtr fineMesh, int kCoarse, int delta_k);
  
  // ! "kCoarse", "delta_k", "jumpToCoarsePolyOrder"
  static std::vector<MeshPtr> meshesForMultigrid(MeshPtr fineMesh, Teuchos::ParameterList &parameters);
};
}

#endif /* defined(__Camellia_debug__GMGSolver__) */
