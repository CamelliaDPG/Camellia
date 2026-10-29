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

#if defined(HAVE_AMESOS2_SUPERLUDIST)
namespace Camellia {

  //! SuperLU_DIST through Amesos2 (ParMETIS fill-reducing ordering by default), distributed over all ranks of the matrix's
  //! communicator.
  class SuperLUDistSolver : public TAmesos2Solver<double>
  {
  public:
    SuperLUDistSolver(bool saveFactorization)
    : TAmesos2Solver<double>(saveFactorization, "superlu_dist") {}

    //! Retained for interface compatibility; Amesos2 uses a process grid over all ranks of the matrix's communicator.
    void setMaxProcsToUse(int maxProcs) { (void) maxProcs; }

    //! Retained for interface compatibility; Amesos2's SuperLU_DIST interface does not print by default.
    void setRunSilent(bool value) { (void) value; }
  };
}
#endif

#endif
