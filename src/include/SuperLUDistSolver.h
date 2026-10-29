// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  SuperLUDistSolver.h
//  Camellia
//
//  Created by Nate Roberts on 6/5/15.
//
//

#ifndef Camellia_SuperLUDistSolver_h
#define Camellia_SuperLUDistSolver_h

#include "Solver.h"

#include "Teuchos_CommHelpers.hpp"

#if defined(HAVE_AMESOS2_SUPERLUDIST)
namespace Camellia {

  //! SuperLU_DIST through Amesos2 (ParMETIS fill-reducing ordering by default), distributed over all ranks of the matrix's
  //! communicator.  Amesos2's defaults use static pivoting without a row permutation or iterative refinement, so systems
  //! with zero diagonal entries (e.g., Lagrange-multiplier constraints) are solved only approximately; Amesos2's
  //! LargeDiag_MC64 row permutation crashes (in pddistribute) with SuperLU_DIST 9.2.  Suitable for preconditioning,
  //! e.g. GMG coarse solves; Solver::getDirectSolver() returns KLU2 for exact direct solves.
  class SuperLUDistSolver : public TAmesos2Solver<double>
  {
  public:
    SuperLUDistSolver(bool saveFactorization)
    : TAmesos2Solver<double>(saveFactorization, "superlu_dist") {}

    //! ParMETIS (SuperLU_DIST's default column ordering) fails with a floating-point exception when a rank owns no
    //! rows, as can happen for small coarse matrices; such matrices are solved with KLU2 instead.
    int solve() override
    {
      auto comm = this->_stiffnessMatrix->getComm();
      const int myRowCount = (this->_stiffnessMatrix->getLocalNumRows() > 0) ? 1 : 0;
      int everyRankHasRows = 0;
      Teuchos::reduceAll(*comm, Teuchos::REDUCE_MIN, 1, &myRowCount, &everyRankHasRows);
      this->setSolverString(everyRankHasRows ? "superlu_dist" : "klu");
      return TAmesos2Solver<double>::solve();
    }

    //! Retained for interface compatibility; Amesos2 uses a process grid over all ranks of the matrix's communicator.
    void setMaxProcsToUse(int maxProcs) { (void) maxProcs; }

    //! Retained for interface compatibility; Amesos2's SuperLU_DIST interface does not print by default.
    void setRunSilent(bool value) { (void) value; }
  };
}
#endif

#endif
