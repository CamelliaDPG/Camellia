// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
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
  
  bool _exportFullOperators = false; // for debugging/analysis.  Will be slow
  string _pathForExport = "./";
  bool _returnErrorIfMaxItersReached = false; // default to false.

  TMapPtr _finePartitionMap;

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
  
  static Teuchos::RCP<GMGOperator> gmgOperatorFromMeshSequence(const std::vector<MeshPtr> &meshesCoarseToFine,
                                                               SolutionPtr fineSolution,
                                                               GMGOperator::MultigridStrategy multigridStrategy,
                                                               GMGOperator::SmootherChoice smoother,
                                                               SolverPtr coarseSolver,
                                                               bool useStaticCondensationInCoarseSolve,
                                                               bool useDiagonalSchwarzWeighting,
                                                               Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation);
public:
  GMGSolver(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter,
            TMapPtr finePartitionMap, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation, int cubatureEnrichmentDegree=0);
  GMGSolver(TSolutionPtr<double> fineSolution, MeshPtr coarseMesh, int maxIters, double tol, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation);
  GMGSolver(TSolutionPtr<double> fineSolution, int maxIters, double tol, int H1OrderCoarse = 1,
            Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true), bool useStaticCondensation = false);
  GMGSolver(TSolutionPtr<double> fineSolution, const std::vector<MeshPtr> &meshesCoarseToFine, int maxIters, double tol,
            GMGOperator::MultigridStrategy multigridStrategy = GMGOperator::V_CYCLE,
            GMGOperator::SmootherChoice smoother = GMGOperator::DEFAULT_ADDITIVE_SCHWARZ,
            Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true), bool useStaticCondensation = false,
            bool useDiagonalSchwarzWeighting = false);

  double condest();

  vector<Teuchos::RCP<GMGOperator>> getOperatorStack(bool coarseToFine);
  
  // ! returns number of grid levels
  int getGridLevelCount() const;
  
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
  
  // ! If true, will export "A.dat" (system matrix) and "M.dat" (preconditioner) during the solve.  Intended for debugging and analysis, not for production.
  void setExportFullOperatorsOnSolve(bool exportOnSolve, string path = "./");

  void setUseConjugateGradient(bool value); // otherwise will use GMRES

  void setPrintIterationCount(bool value);
  
  void setReturnErrorIfMaxItersReached(bool value);

  // ! Sets the number of times the smoother will be applied when it is applied at each grid level.  (For multigrid strategies other than TWO_LEVEL, it is applied this many times before and after each application of the coarse operator.)
  void setSmootherApplicationCount(int count);
  
  void setSmootherType(GMGOperator::SmootherChoice smootherType);
  
  vector<int> getIterationCountLog();
  
  // ! delta_k = -1: use fineMesh's delta_k
  static std::vector<MeshPtr> meshesForMultigrid(MeshPtr fineMesh, int kCoarse, int delta_k = -1);
  
  // ! "kCoarse", "delta_k", "jumpToCoarsePolyOrder"
  static std::vector<MeshPtr> meshesForMultigrid(MeshPtr fineMesh, Teuchos::ParameterList &parameters);
  
  static Teuchos::RCP<GMGSolver> cgSolver(SolutionPtr soln, double cgTol, int maxIters);
  static Teuchos::RCP<GMGSolver> gmgSolver(SolutionPtr soln, bool useCG, double tol, int maxIters);
  static Teuchos::RCP<GMGSolver> gmresSolver(SolutionPtr soln, double gmresTol, int maxIters);
  
  static Teuchos::RCP<GMGSolver> gmgSolver(TSolutionPtr<double> fineSolution, bool useConjugateGradient,
                                           const std::vector<MeshPtr> &meshesCoarseToFine, int maxIters, double tol,
                                           GMGOperator::MultigridStrategy multigridStrategy = GMGOperator::V_CYCLE,
                                           Teuchos::RCP<Solver> coarseSolver = Solver::getDirectSolver(true));
};
}

#endif /* defined(__Camellia_debug__GMGSolver__) */
