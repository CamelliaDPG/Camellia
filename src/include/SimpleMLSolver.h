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
#include "AztecOO.h"

#include "Solver.h"

class SimpleMLSolver : public Solver {
  bool _saveFactorization;
  Teuchos::RCP< AztecOO > _savedSolver;
  Teuchos::RCP< ML_Epetra::MultiLevelPreconditioner > _savedPreconditioner;
  
  double _resTol;
  int _maxIters;
public:
  SimpleMLSolver(bool saveFactorization, double residualTolerance, int maxIterations);
  int solve();
  int resolve();
  // void setProblem(Teuchos::RCP< Epetra_LinearProblem > problem);
};

#endif