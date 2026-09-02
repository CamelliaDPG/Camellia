// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

#include "Amesos_config.h"
#include "Amesos2.hpp"

using namespace std;

namespace Camellia
{
// abstract class for solving Tpetra linear systems
template <typename Scalar>
class TSolver
{
protected:
  TMatrixPtr<Scalar> _stiffnessMatrix;
  TVectorPtr<Scalar> _lhs;
  TVectorPtr<Scalar> _rhs;
public:
  virtual ~TSolver() {}
  
  TVectorPtr<Scalar> lhs()
  {
    return _lhs;
  }
  virtual void setProblem(TMatrixPtr<Scalar> stiffnessMatrix, TVectorPtr<Scalar> lhs, TVectorPtr<Scalar> rhs)
  {
    _stiffnessMatrix = stiffnessMatrix;
    _lhs = lhs;
    _rhs = rhs;
    stiffnessMatrixChanged();
  }
  virtual void setStiffnessMatrix(TMatrixPtr<Scalar> stiffnessMatrix)
  {
    _stiffnessMatrix = stiffnessMatrix;
    stiffnessMatrixChanged();
  }
  virtual void setLHS2(TVectorPtr<Scalar> lhs)
  {
    _lhs = lhs;
  }
  virtual void setRHS2(TVectorPtr<Scalar> rhs)
  {
    _rhs = rhs;
  }
  virtual void stiffnessMatrixChanged()
  {
    
  }
  virtual int solve() = 0; // solve with an error code response
  virtual int resolve()
  {
    // must be preceded by a call to solve(); caller attests that the system matrix has not been altered since last call to solve()
    // subclasses may override to reuse factorization information
    return solve();
  }

  enum SolverChoice
  {
    KLU,
    SuperLUDist,
    MUMPS,
    Pardiso,
    GMG
  };
  
  static TSolverPtr<Scalar> getDirectSolver(bool saveFactorization=false);
  
  static TSolverPtr<Scalar> getSolver(SolverChoice choice, bool saveFactorization,
                                      double residualTolerance = 1e-12, int maxIterations = 50000,
                                      TSolutionPtr<double> fineSolution = Teuchos::null,
                                      TSolverPtr<Scalar> coarseSolver = Teuchos::null);
  
  static void printAvailableSolversReport();
  
  static SolverChoice solverChoiceFromString(std::string choiceString)
  {
    if (choiceString=="KLU") return KLU;
    if (choiceString=="SuperLUDist") return SuperLUDist;
    if (choiceString=="MUMPS") return MUMPS;
    if (choiceString=="Pardiso") return Pardiso;
    if (choiceString=="GMG") return GMG;
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "choiceString not recognized!");
  }
  static std::string solverChoiceString(SolverChoice choice)
  {
    if (choice==KLU) return "KLU";
    if (choice==SuperLUDist) return "SuperLUDist";
    if (choice==MUMPS) return "MUMPS";
    if (choice==Pardiso) return "Pardiso";
    if (choice==GMG) return "GMG";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "choice not recognized!");
  }
};


// some concrete implementations follow…
template <typename Scalar>
class TAmesos2Solver : public TSolver<Scalar>
{
  bool _saveFactorization;
  std::string _solverString;
  Teuchos::RCP<Amesos2::Solver< TMatrix<Scalar>,TVector<Scalar> > > _savedSolver;
public:
  // Solver options available: klu superlu_dist superlu_mt superlu pardisomkl lapack amesos2_cholmod
  // May require specific instructions while building Trilinos
  TAmesos2Solver(bool saveFactorization = false, std::string solverString="klu"): _saveFactorization(saveFactorization), _solverString(solverString) {}
  int solve()
  {
    TEUCHOS_TEST_FOR_EXCEPTION(this->_stiffnessMatrix.get() == NULL, std::invalid_argument, "stiffness matrix is unset.");
    TEUCHOS_TEST_FOR_EXCEPTION(this->_lhs.get() == NULL, std::invalid_argument, "lhs is unset.");
    TEUCHOS_TEST_FOR_EXCEPTION(this->_rhs.get() == NULL, std::invalid_argument, "rhs is unset.");
    if (!_saveFactorization)
    {
      Teuchos::RCP<Amesos2::Solver<TMatrix<Scalar>,TVector<Scalar> > > solver
        = Amesos2::create< TMatrix<Scalar>,TVector<Scalar> >(_solverString, this->_stiffnessMatrix, this->_lhs, this->_rhs);
      solver->solve();
    }
    else
    {
      _savedSolver = Amesos2::create< TMatrix<Scalar>,TVector<Scalar> >(_solverString, this->_stiffnessMatrix, this->_lhs, this->_rhs);
      _savedSolver->solve();
    }
    return 0;
  }
  int resolve()
  {
    if (_savedSolver.get() != NULL)
    {
      _savedSolver->solve();
    }
    else
    {
      solve();
    }
    return 0;
  }
  virtual void stiffnessMatrixChanged()
  {
    _savedSolver = Teuchos::null;
  }
};

extern template class TSolver<double>;
extern template class TAmesos2Solver<double>;
} // namespace

namespace Camellia
{
#ifdef HAVE_AMESOS_MUMPS
class MumpsSolver : public TAmesos2Solver<double>
{
public:
  MumpsSolver(int maxMemoryPerCoreMB = 512, bool saveFactorization = false)
  : TAmesos2Solver<double>(saveFactorization, "mumps")
  {
    // Amesos2 manages MUMPS memory options through its parameter list.
    (void) maxMemoryPerCoreMB;
  }
};
#else
class MumpsSolver : public TSolver<double>
{
public:
  int solve()
  {
    cout << "ERROR: Amesos2 MUMPS support is not available.\n";
    return -1;
  }
};
#endif
} // namespace

#endif
