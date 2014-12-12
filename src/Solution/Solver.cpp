//
//  Solver.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 11/24/14.
//
//

#include "Solver.h"

#include "SimpleMLSolver.h"

Teuchos::RCP<Solver> Solver::getSolver(SolverChoice choice, bool saveFactorization,
                                       double residualTolerance, int maxIterations) {
  switch (choice) {
    case KLU:
      return Teuchos::rcp( new KluSolver(saveFactorization) );
      break;
#ifdef HAVE_AMESOS_SUPERLUDIST
    case SuperLUDist:
      return Teuchos::rcp( new SuperLUDistSolver(saveFactorization) );
#endif
#ifdef USE_MUMPS
    case MUMPS:
      return Teuchos::rcp( new MumpsSolver(saveFactorization) );
#endif
    case SimpleML:
      return Teuchos::rcp( new SimpleMLSolver(saveFactorization, residualTolerance, maxIterations) );
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Solver choice not recognized!");
  }
}
