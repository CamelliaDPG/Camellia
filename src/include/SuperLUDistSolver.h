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

#include "MPIWrapper.h"

namespace Camellia {

  class SuperLUDistSolver : public Solver
  {
    Teuchos::RCP<Amesos_Superludist> _savedSolver;
    Teuchos::RCP<Epetra_LinearProblem> _savedProblem;
    bool _saveFactorization;
    bool _havePrintedStatus;
  public:
    SuperLUDistSolver(bool saveFactorization)
    {
      _saveFactorization = saveFactorization;
      _havePrintedStatus = false;
    }
    
    int solve() {
      Teuchos::ParameterList paramList;
      
      if (!_saveFactorization)
      {
        Epetra_LinearProblem problem(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get());
        Amesos_Superludist slu(problem);
        slu.SetParameters(paramList);
        return slu.Solve();
      }
      else
      {
        _savedProblem = Teuchos::rcp( new Epetra_LinearProblem(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get()) );
        _savedSolver = Teuchos::rcp( new Amesos_Superludist(*_savedProblem) );
        
        paramList.set("ReuseSymbolic",true);
        paramList.set("Fact", "SamePattern_SameRowPerm");
        _savedSolver->SetParameters(paramList);
        
        int err = _savedSolver->Solve();
        
        if (!_havePrintedStatus)
        { // print status for diagnostic purposes
          if (MPIWrapper::rank() == 0) cout << "SuperLUDistSolver.h : SuperLU_Dist factorization reuse requested.  Details of solver:\n";
          _savedSolver->PrintStatus();
          _havePrintedStatus = true;
        }
        
        return err;
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
    virtual void stiffnessMatrixChanged()
    {
      _savedSolver = Teuchos::null;
      _savedProblem = Teuchos::null;
      _havePrintedStatus = false;
    }
  };
}
#endif

#endif
