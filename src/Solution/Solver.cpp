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

Teuchos::RCP<Solver> Solver::getSolver(SolverChoice choice, bool saveFactorization,
                                       double residualTolerance, int maxIterations,
                                       SolutionPtr fineSolution, MeshPtr coarseMesh,
                                       SolverPtr coarseSolver) {
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
    case GMGSolver_1_Level_h:
      // false below: don't use condensed solve...
      return Teuchos::rcp( new GMGSolver(fineSolution, coarseMesh, maxIterations, residualTolerance, coarseSolver, false) );
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Solver choice not recognized!");
  }
}
