//
//  Solver.h
//  Camellia
//
//  Created by Nathan Roberts on 3/3/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_Solver_h
#define Camellia_Solver_h

#include "TypeDefs.h"

#include <Teuchos_RCP.hpp>
#include <Epetra_LinearProblem.h>
#include <Epetra_Time.h>
#include <Epetra_CrsMatrix.h>
#include <Epetra_MultiVector.h>

// Might need these later
// #include <Teuchos_ScalarTraits.hpp>
// #include <Teuchos_RCP.hpp>
// #include <Teuchos_GlobalMPISession.hpp>
// #include <Teuchos_oblackholestream.hpp>
// #include <Teuchos_Tuple.hpp>
// #include <Teuchos_VerboseObject.hpp>
// #include <Tpetra_DefaultPlatform.hpp>
// #include <Tpetra_Map.hpp>
// #include <Tpetra_MultiVector.hpp>
// #include <Tpetra_CrsMatrix.hpp>
#include "Amesos2.hpp"

using namespace std;

namespace Camellia {
  // abstract class for solving Epetra_LinearProblem problems
  template <typename Scalar>
  class TSolver {
  protected:
    Teuchos::RCP<Epetra_CrsMatrix> _stiffnessMatrix;
    Teuchos::RCP<Epetra_MultiVector> _lhs;
    Teuchos::RCP<Epetra_MultiVector> _rhs;

    TMatrixPtr<Scalar> _stiffnessMatrix2;
    TVectorPtr<Scalar> _lhs2;
    TVectorPtr<Scalar> _rhs2;
  public:
    virtual ~TSolver() {}
    virtual void setProblem(Teuchos::RCP<Epetra_CrsMatrix> stiffnessMatrix, Teuchos::RCP<Epetra_MultiVector> lhs, Teuchos::RCP<Epetra_MultiVector> rhs) {
      _stiffnessMatrix = stiffnessMatrix;
      _lhs = lhs;
      _rhs = rhs;
    }
    virtual void setProblem(TMatrixPtr<Scalar> stiffnessMatrix, TVectorPtr<Scalar> lhs, TVectorPtr<Scalar> rhs) {
      _stiffnessMatrix2 = stiffnessMatrix;
      _lhs2 = lhs;
      _rhs2 = rhs;
    }
    virtual void setStiffnessMatrix(Teuchos::RCP<Epetra_CrsMatrix> stiffnessMatrix) {
      _stiffnessMatrix = stiffnessMatrix;
    }
    virtual void setStiffnessMatrix(TMatrixPtr<Scalar> stiffnessMatrix) {
      _stiffnessMatrix2 = stiffnessMatrix;
    }
    virtual void setLHS(Teuchos::RCP<Epetra_MultiVector> lhs) {
      _lhs = lhs;
    }
    virtual void setLHS2(TVectorPtr<Scalar> lhs) {
      _lhs2 = lhs;
    }
    virtual void setRHS(Teuchos::RCP<Epetra_MultiVector> rhs) {
      _rhs = rhs;
    }
    virtual void setRHS2(TVectorPtr<Scalar> rhs) {
      _rhs2 = rhs;
    }
    virtual int solve() = 0; // solve with an error code response
    virtual int resolve() {
      // must be preceded by a call to solve(); caller attests that the system matrix has not been altered since last call to solve()
      // subclasses may override to reuse factorization information
      return solve();
    }

    enum SolverChoice {
      KLU,
      SuperLUDist,
      MUMPS,
      SimpleML,
      GMGSolver_1_Level_h
    };

    static TSolverPtr<Scalar> getSolver(SolverChoice choice, bool saveFactorization,
                               double residualTolerance = 1e-12, int maxIterations = 50000,
                               TSolutionPtr<double> fineSolution = Teuchos::null, MeshPtr coarseMesh = Teuchos::null,
                               TSolverPtr<Scalar> coarseSolver = Teuchos::null);

    static TSolverPtr<Scalar> getDirectSolver(bool saveFactorization=false);

    static SolverChoice solverChoiceFromString(std::string choiceString) {
      if (choiceString=="KLU") return KLU;
      if (choiceString=="SuperLUDist") return SuperLUDist;
      if (choiceString=="MUMPS") return MUMPS;
      if (choiceString=="SimpleML") return SimpleML;
      if (choiceString=="GMGSolver_1_Level_h") return GMGSolver_1_Level_h;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "choiceString not recognized!");
    }
    static std::string solverChoiceString(SolverChoice choice) {
      if (choice==KLU) return "KLU";
      if (choice==SuperLUDist) return "SuperLUDist";
      if (choice==MUMPS) return "MUMPS";
      if (choice==SimpleML) return "SimpleML";
      if (choice==GMGSolver_1_Level_h) return "GMGSolver_1_Level_h";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "choice not recognized!");
    }
  };


  // some concrete implementations follow…
  template <typename Scalar>
  class TAmesos2Solver : public TSolver<Scalar> {
    bool _saveFactorization;
    std::string _solverString;
    Teuchos::RCP<Amesos2::Solver<Epetra_CrsMatrix,Epetra_MultiVector> > _savedSolver;
  public:
    // Solver options available: klu superlu_dist superlu_mt superlu pardisomkl lapack amesos2_cholmod
    // May require specific instructions while building Trilinos
    TAmesos2Solver(bool saveFactorization = false, std::string solverString="klu"): _saveFactorization(saveFactorization), _solverString(solverString) {}
    int solve() {
      TEUCHOS_TEST_FOR_EXCEPTION(this->_stiffnessMatrix.get() == NULL, std::invalid_argument, "stiffness matrix is unset.");
      TEUCHOS_TEST_FOR_EXCEPTION(this->_lhs.get() == NULL, std::invalid_argument, "lhs is unset.");
      TEUCHOS_TEST_FOR_EXCEPTION(this->_rhs.get() == NULL, std::invalid_argument, "rhs is unset.");
      if (!_saveFactorization) {
        Teuchos::RCP<Amesos2::Solver<Epetra_CrsMatrix,Epetra_MultiVector> > solver
          = Amesos2::create<Epetra_CrsMatrix,Epetra_MultiVector>(_solverString, this->_stiffnessMatrix, this->_lhs, this->_rhs);
        solver->solve();
      } else {
        _savedSolver = Amesos2::create<Epetra_CrsMatrix,Epetra_MultiVector>(_solverString, this->_stiffnessMatrix, this->_lhs, this->_rhs);
        _savedSolver->solve();
      }
      return 0;
    }
    int resolve() {
      if (_savedSolver.get() != NULL) {
        _savedSolver->solve();
      } else {
        solve();
      }
      return 0;
    }
  };

  extern template class TSolver<double>;
  extern template class TAmesos2Solver<double>;
} // namespace

 // only use MUMPS when we have MPI
 #ifdef HAVE_AMESOS_MUMPS
 #ifdef HAVE_MPI
 #include "Amesos_Mumps.h"
namespace Camellia {
 class MumpsSolver : public TSolver<double> {
   int _maxMemoryPerCoreMB;
   bool _saveFactorization;
   Teuchos::RCP<Amesos_Mumps> _savedSolver;
 // protected:
 //   Teuchos::RCP< Epetra_LinearProblem > _problem;
 public:
   // Epetra_LinearProblem & problem() { return *(_problem.get()); }
   MumpsSolver(int maxMemoryPerCoreMB = 512, bool saveFactorization = false) {
     // maximum amount of memory MUMPS may allocate per core.
     _maxMemoryPerCoreMB = maxMemoryPerCoreMB;
     _saveFactorization = saveFactorization;
   }
   // void setProblem(Teuchos::RCP< Epetra_LinearProblem > problem) {
   //   _savedSolver = Teuchos::rcp((Amesos_Mumps*)NULL);
   //   this->_problem = problem;
   // }
   int solve() {
     Epetra_LinearProblem problem(this->_stiffnessMatrix.get(), this->_lhs.get(), this->_rhs.get());
     Teuchos::RCP<Amesos_Mumps> mumps = Teuchos::rcp(new Amesos_Mumps(problem));
     int numProcs=1;
     int rank=0;
     int previousSize = 0;
     rank     = Teuchos::GlobalMPISession::getRank();
     numProcs = Teuchos::GlobalMPISession::getNProc();
     mumps->SetICNTL(28, 0); // 0: automatic choice between parallel and sequential analysis
 //    mumps->SetICNTL(28, 2); // 2: parallel analysis
 //    mumps->SetICNTL(29, 2); // 2: use PARMETIS; 1: use PT-SCOTCH
 //    int minSize = max(infog[26-1], infog[16-1]);
 //    // want to set ICNTL 23 to a size "significantly larger" than minSize
 //    int sizeToSet = max(2 * minSize, previousSize*2);
 //    sizeToSet = min(sizeToSet, _maxMemoryPerCoreMB);
 //    previousSize = sizeToSet;
     //    mumps->SetICNTL(23, sizeToSet);
     // not sure why we shouldn't just do this: (I don't think MUMPS will allocate as much as we allow it, unless it thinks it needs it)
     mumps->SetICNTL(1,6); // set output stream for errors (this is supposed to be the default, but maybe Amesos clobbers it?)
 //    int sizeToSet = _maxMemoryPerCoreMB;
 //    cout << "setting ICNTL 23 to " << sizeToSet << endl;
 //    mumps->SetICNTL(23, sizeToSet);
     mumps->SymbolicFactorization();
     mumps->NumericFactorization();
     int relaxationParam = 0; // the default
     int* info = mumps->GetINFO();
     int* infog = mumps->GetINFOG();
     int numErrors = 0;
     while (info[0] < 0) { // error occurred
       info = mumps->GetINFO(); // not sure if these can change locations between invocations -- just in case...
       infog = mumps->GetINFOG();
       numErrors++;
       if (rank == 0) {
         if (infog[0] == -9) {
           int minSize = infog[26-1];
           // want to set ICNTL 23 to a size "significantly larger" than minSize
           int sizeToSet = max(2 * minSize, previousSize*2);
           sizeToSet = min(sizeToSet, _maxMemoryPerCoreMB);
           mumps->SetICNTL(23, sizeToSet);
           cout << "\nMUMPS memory allocation too small.  Setting to: " << sizeToSet << " MB/core." << endl;
           previousSize = sizeToSet;
         } else if (infog[0] == -7) {
           // some error related to an integer array allocation.
           // since I'm not sure how to determine how much we previously had, we'll just try again with the max
           int sizeToSet = _maxMemoryPerCoreMB;
           cout << "\nMUMPS encountered an error allocating an integer workspace of size " << infog[2-1] << " (bytes, I think).\n";
           cout << "-- perhaps it's running into our allocation limit?? Setting the allocation limit to ";
           cout << sizeToSet << " MB/core." << endl;
           mumps->SetICNTL(23, sizeToSet);
         } else if (infog[0]==-13) { // error during a Fortran ALLOCATE statement
           int infog_sizeRequested = infog[2-1];
           long long sizeRequested;
           if (infog_sizeRequested > 0) {
             sizeRequested = infog_sizeRequested;
           } else {
             sizeRequested = -infog_sizeRequested * 1e6;
           }
           if (previousSize > 0) {
             int sizeToSet = 3 * previousSize / 4; // reduce size by 25%
             mumps->SetICNTL(23, sizeToSet);
             cout << "MUMPS memory allocation error -13 while requesting allocation of size " << sizeRequested;
             cout << " (bytes?); likely indicates we're out of memory.  Reducing by 25%; setting to: " << sizeToSet << " MB/core." << endl;
           } else {
             int sizeToSet = ((3 * sizeRequested) / 4) / 1e6; // reduce by 25% and convert to MB
             mumps->SetICNTL(23, sizeToSet);
             cout << "MUMPS memory allocation error -13 while requesting allocation of size " << sizeRequested;
             cout << " (bytes?); likely indicates we're out of memory.  Setting ICNTL 23 to" << sizeToSet << " MB/core." << endl;
           }
         } else {
           cout << "MUMPS encountered unhandled error code " << infog[0] << endl;
           for (int i=0; i<40; i++) {
             cout << "infog[" << setw(2) << i+1 << "] = " << infog[i] << endl; // i+1 because 1-based indices are used in MUMPS manual
           }
           TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled MUMPS error code");
         }
       }
       mumps->SymbolicFactorization();
       mumps->NumericFactorization();
       if (numErrors > 20) {
         if (rank==0) cout << "Too many errors during MUMPS factorization.  Quitting.\n";
         break;
       }
     }
     if (_saveFactorization) _savedSolver = mumps;
     return mumps->Solve();
   }
   int resolve() {
 //    if (_savedSolver.get() == NULL) {
 //      cout << "You must call solve() before calling resolve().  Also, _saveFactorization must be true.\n";
 //      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must call solve() before calling resolve().  Also, _saveFactorization must be true.");
 //    }
 //    return _savedSolver->Solve();
     if (_savedSolver.get() != NULL) {
       return _savedSolver->Solve();
     } else {
       return solve();
     }
   }
 };
} // namespace
 #else
namespace Camellia {
 class MumpsSolver : public TSolver<double> {
 public:
   int solve() {
     cout << "ERROR: no MUMPS support for non-MPI runs yet (because Nate hasn't built MUMPS for his serial-debug Trilinos).\n";
     return -1;
   }
 };
} // namespace
#endif

#endif

#endif
