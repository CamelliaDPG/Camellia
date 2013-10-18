//
//  Solver.h
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Solver_h
#define Camellia_Solver_h

#include "Epetra_LinearProblem.h"
#include "Amesos_Klu.h"
#include "AztecOO.h"

#include "CamelliaConfig.h"

// abstract class for solving Epetra_LinearProblem problems
class Solver {
private:
  Teuchos::RCP< Epetra_LinearProblem > _problem;
public:
  virtual Epetra_LinearProblem & problem() { return *(_problem.get()); }
  virtual void setProblem(Teuchos::RCP< Epetra_LinearProblem > problem) {
    _problem = problem;
  }
  virtual int solve() = 0; // solve with an error code response
};

// some concrete implementations follow…
class KluSolver : public Solver {
public:
  int solve() {
    Amesos_Klu klu(problem());
    return klu.Solve();
  }
};

using namespace std;

// only use MUMPS when we have MPI
#ifdef USE_MUMPS
#ifdef HAVE_MPI
#include "Amesos_Mumps.h"
class MumpsSolver : public Solver {
  int _maxMemoryPerCoreMB;
public:
  MumpsSolver(int maxMemoryPerCoreMB = 768) {
    // maximum amount of memory MUMPS may allocate per core.
    _maxMemoryPerCoreMB = maxMemoryPerCoreMB;
  }
  
  int solve() {
    Amesos_Mumps mumps(problem());
    mumps.SymbolicFactorization();
    mumps.NumericFactorization();
    int relaxationParam = 0; // the default
    int* info = mumps.GetINFO();
    int numProcs=1;
    int rank=0;
    
#ifdef HAVE_MPI
    rank     = Teuchos::GlobalMPISession::getRank();
    numProcs = Teuchos::GlobalMPISession::getNProc();
#else
#endif
    int previousSize = 0;
    int numErrors = 0;
    while (info[0] < 0) { // error occurred
      
      numErrors++;
      if (rank == 0) {
        int* infog = mumps.GetINFOG();
        if (infog[0] == -9) {
          int minSize = infog[26-1];
          // want to set ICNTL 23 to a size "significantly larger" than minSize
          int sizeToSet = max(2 * minSize, previousSize*2);
          sizeToSet = min(sizeToSet, _maxMemoryPerCoreMB);
          mumps.SetICNTL(23, sizeToSet);
          cout << "MUMPS memory allocation too small.  Resetting to: " << sizeToSet << endl;
          previousSize = sizeToSet;
        } else if (infog[0]==-13) {
          if (previousSize > 0) {
            int sizeToSet = 3 * previousSize / 4; // reduce size by 25%
            mumps.SetICNTL(23, sizeToSet);
            cout << "MUMPS memory allocation error -13.  Resetting to: " << sizeToSet << endl;
          } else {
            cout << "MUMPS memory allocation error -13, but previousSize was 0.  (Unhandled case)." << endl;
          }
        } else {
          cout << "MUMPS encountered unhandled error code " << infog[0] << endl;
        }
      }
      mumps.SymbolicFactorization();
      mumps.NumericFactorization();
      if (numErrors > 200) {
        if (rank==0) cout << "Too many errors during MUMPS factorization.  Quitting.\n";
        break;
      }
    }
    return mumps.Solve();
  }
};
#else
class MumpsSolver : public Solver {
public:
  int solve() {
    cout << "ERROR: no MUMPS support for non-MPI runs yet (because Nate hasn't built MUMPS for his serial-debug Trilinos).\n";
    return -1;
  }
};
#endif

#endif

#endif
