//
//  Solver.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/24/14.
//
//

#include "Solver.h"

#include "SimpleMLSolver.h"

#include "GMGSolver.h"

#include "Solution.h"
#include "Mesh.h"

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
  cout << solverChoiceString(SimpleML) << endl;
  cout << solverChoiceString(GMGSolver_1_Level_h) << endl;
}

template <typename Scalar>
TSolverPtr<Scalar> TSolver<Scalar>::getSolver(SolverChoice choice, bool saveFactorization,
    double residualTolerance, int maxIterations,
    TSolutionPtr<double> fineSolution, MeshPtr coarseMesh,
    TSolverPtr<Scalar> coarseSolver)
{
  switch (choice)
  {
  case KLU:
    return Teuchos::rcp( new TAmesos2Solver<Scalar>(saveFactorization, "klu") );
    break;
#if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
  case SuperLUDist:
    return Teuchos::rcp( new TAmesos2Solver<Scalar>(saveFactorization, "superlu_dist") );
#endif
#ifdef HAVE_AMESOS_MUMPS
  case MUMPS:
    return Teuchos::rcp( new MumpsSolver(saveFactorization) );
#endif
  case SimpleML:
    return Teuchos::rcp( new SimpleMLSolver(saveFactorization, residualTolerance, maxIterations) );

  case GMGSolver_1_Level_h:
  {
    // false below: don't use condensed solve...
    bool useCondensedSolve = false;
    GMGSolver* gmgSolver = new GMGSolver(fineSolution, coarseMesh, maxIterations, residualTolerance, coarseSolver, useCondensedSolve);

    gmgSolver->setComputeConditionNumberEstimate(false); // faster if we don't compute it

    // testing:
//      gmgSolver->setAztecOutput(100);

    // testing:
//      gmgSolver->setApplySmoothingOperator(false);

    return Teuchos::rcp(gmgSolver);
  }
  default:
    cout << "Solver choice " << solverChoiceString(choice) << " not recognized.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Solver choice not recognized!");
  }
}

template <typename Scalar>
TSolverPtr<Scalar> TSolver<Scalar>::getDirectSolver(bool saveFactorization) {
#if defined(HAVE_AMESOS_SUPERLUDIST)
  return getSolver(TSolver<Scalar>::SuperLUDist, saveFactorization);
#elif defined(HAVE_AMESOS_MUMPS)
  return getSolver(TSolver<Scalar>::MUMPS, saveFactorization);
#else
  return getSolver(TSolver<Scalar>::KLU, saveFactorization);
#endif
}

namespace Camellia
{
template class TSolver<double>;
template class TAmesos2Solver<double>;
}
