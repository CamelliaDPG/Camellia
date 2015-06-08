//
//  SuperLUDistSolver.h
//  Camellia
//
//  Created by Nate Roberts on 6/5/15.
//
//

#ifndef Camellia_SuperLUDistSolver_h
#define Camellia_SuperLUDistSolver_h

#include "Amesos_config.h"
#include "Amesos.h"
#ifdef HAVE_AMESOS_SUPERLUDIST
#include "Amesos_Superludist.h"

namespace Camellia {

  class SuperLUDistSolver : public Solver
  {
    Teuchos::RCP<Amesos_Superludist> _savedSolver;
    Teuchos::RCP<Epetra_LinearProblem> _savedProblem;
    bool _saveFactorization;
  public:
    SuperLUDistSolver(bool saveFactorization)
    {
      _saveFactorization = saveFactorization;
    }
    
    int solve() {
      if (!_saveFactorization)
      {
        Epetra_LinearProblem problem(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get());
        Amesos_Superludist slu(problem);
        return slu.Solve();
      }
      else
      {
        _savedProblem = Teuchos::rcp( new Epetra_LinearProblem(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get()) );
        _savedSolver = Teuchos::rcp( new Amesos_Superludist(*_savedProblem) );
        return _savedSolver->Solve();
      }
    }
    
    int resolve() {
      if (_savedSolver.get() != NULL)
      {
        return _savedSolver->Solve();
      }
      else
      {
        return solve();
      }
    }
  };
}
#endif

#endif
