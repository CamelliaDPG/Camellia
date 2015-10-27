//
//  Solver.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/24/14.
//
//

#include "Solver.h"

#include "GMGSolver.h"
#include "Mesh.h"
#include "Solution.h"
#include "SuperLUDistSolver.h"

using namespace Camellia;

template <typename Scalar>
void TSolver<Scalar>::printAvailableSolversReport()
{
  cout << "Available solvers:\n";
  cout << solverChoiceString(KLU) << endl;
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  cout << solverChoiceString(SuperLUDist) << endl;
#endif
#ifdef HAVE_AMESOS_MUMPS
  cout << solverChoiceString(MUMPS) << endl;
#endif
//  cout << solverChoiceString(GMG) << endl;
}

template <typename Scalar>
TSolverPtr<Scalar> TSolver<Scalar>::getSolver(SolverChoice choice, bool saveFactorization,
    double residualTolerance, int maxIterations,
    TSolutionPtr<double> fineSolution, TSolverPtr<Scalar> coarseSolver)
{
  switch (choice)
  {
  case KLU:
    return Teuchos::rcp( new TAmesos2Solver<Scalar>(saveFactorization, "klu") );
    break;
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  case SuperLUDist:
    return Teuchos::rcp( new SuperLUDistSolver(saveFactorization) );
//    return Teuchos::rcp( new TAmesos2Solver<Scalar>(saveFactorization, "superlu_dist") );
#endif
#ifdef HAVE_AMESOS_MUMPS
  case MUMPS:
    return Teuchos::rcp( new MumpsSolver(saveFactorization) );
#endif
      
  case GMG:
  {
    Teuchos::ParameterList pl;
    pl.set("kCoarse", 0);
    pl.set("delta_k", 1); // this should not really matter in this context
    pl.set("jumpToCoarsePolyOrder", false); //, k_coarse == 1); // due to an apparent issue in meshesForMultigrid, "jump" from 1 to 0
    vector<MeshPtr> meshesCoarseToFine = GMGSolver::meshesForMultigrid(fineSolution->mesh(), pl);
    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(fineSolution, meshesCoarseToFine, maxIterations, residualTolerance, GMGOperator::V_CYCLE,
                                                                   coarseSolver, fineSolution->usesCondensedSolve(), false));
    
    gmgSolver->setAztecOutput(0);
    gmgSolver->setComputeConditionNumberEstimate(false);

    return gmgSolver;
  }
  default:
    cout << "Solver choice " << solverChoiceString(choice) << " not recognized.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Solver choice not recognized!");
  }
}

template <typename Scalar>
TSolverPtr<Scalar> TSolver<Scalar>::getDirectSolver(bool saveFactorization) {
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  return TSolver<Scalar>::getSolver(TSolver<Scalar>::SuperLUDist, saveFactorization);
#elif defined(HAVE_AMESOS_MUMPS)
  return TSolver<Scalar>::getSolver(TSolver<Scalar>::MUMPS, saveFactorization);
#else
  return TSolver<Scalar>::getSolver(TSolver<Scalar>::KLU, saveFactorization);
#endif
}

namespace Camellia
{
template class TSolver<double>;
template class TAmesos2Solver<double>;
}
