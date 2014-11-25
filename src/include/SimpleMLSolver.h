//
//  SimpleMLSolver.h
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SimpleMLSolver_h
#define Camellia_SimpleMLSolver_h

// Trilinos/ML includes
#include "ml_include.h"
#include "ml_MultiLevelPreconditioner.h"
#include "ml_epetra_utils.h"

#include "Teuchos_RCP.hpp"

#include "Solver.h"

class SimpleMLSolver : public Solver {
  bool _saveFactorization;
  Teuchos::RCP< AztecOO > _savedSolver;
  Teuchos::RCP< ML_Epetra::MultiLevelPreconditioner > _savedPreconditioner;
  
  double _resTol;
  int _maxIters;
public:
  SimpleMLSolver(bool saveFactorization, double residualTolerance, int maxIterations) {
    _saveFactorization = saveFactorization;
    _resTol = residualTolerance;
    _maxIters = maxIterations;

  }
  int solve() {
    AztecOO *solver = new AztecOO( problem() );
    // create a parameter list for ML options
    ParameterList MLList;
    
    // Sets default parameters for classic smoothed aggregation. After this
    // call, MLList contains the default values for the ML parameters,
    // as required by typical smoothed aggregation for symmetric systems.
    // Other sets of parameters are available for non-symmetric systems
    // ("DD" and "DD-ML"), and for the Maxwell equations ("maxwell").
    
    int maxLevels = 8;
    char parameter[80];
    
    ML_Epetra::MultiLevelPreconditioner* MLPrec;

    ML_Epetra::SetDefaults("SA",MLList);
  
    Epetra_RowMatrix *A = problem().GetMatrix();
    
    MLPrec = new ML_Epetra::MultiLevelPreconditioner(*A, MLList);

    solver->SetPrecOperator(MLPrec);
    //    solver.SetAztecOption(AZ_solver, AZ_GMRESR); // could do AZ_cg -- we are SPD (but so far it seems cg takes a bit longer...)
    solver->SetAztecOption(AZ_output, 0);
    int result = solver->Iterate(_maxIters, _resTol);
    
    if (! _saveFactorization) {
      // destroy the preconditioner
      delete MLPrec;
      // destroy the solver object
      delete solver;
    } else {
      _savedSolver = Teuchos::rcp( solver );
      _savedPreconditioner = Teuchos::rcp( MLPrec );
    }
    
    return result;
  }
  int resolve() {
    if (_savedSolver.get() != NULL) {
      return _savedSolver->Iterate(_maxIters, _resTol);;
    } else {
      return solve();
    }
  }
};

#endif