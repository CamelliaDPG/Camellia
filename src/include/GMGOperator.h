//
//  GMGOperator.h
//  Camellia-debug
//
//  Created by Nate Roberts on 7/3/14.
//
//
#ifndef __Camellia_debug__GMGOperator__
#define __Camellia_debug__GMGOperator__

#include "Epetra_Operator.h"

#include "IP.h"
#include "Mesh.h"
#include "RefinementPattern.h"
#include "Solution.h"

#include "BasisReconciliation.h"
#include "LocalDofMapper.h"

#include "Ifpack_Preconditioner.h"

#include "Solver.h"

#include <map>

using namespace std;

struct TimeStatistics {
  double min;
  double max;
  double mean;
  double sum;
};

class GMGOperator : public Epetra_Operator {
  SolutionPtr _coarseSolution;
  
  bool _useStaticCondensation; // for both coarse and fine solves
  Teuchos::RCP<DofInterpreter> _fineDofInterpreter;
  
  MeshPtr _fineMesh, _coarseMesh;
  Epetra_Map _finePartitionMap;
  
  bool _fineSolverUsesDiagonalScaling;
  bool _applySmoothingOperator; // almost always true; false for some tests
  BCPtr _bc;

  TimeStatistics getStatistics(double timeValue) const;
  
  Teuchos::RCP<Solver> _coarseSolver;
  
  mutable BasisReconciliation _br;
  mutable map< pair< pair<int,int>, RefinementBranch >, LocalDofMapperPtr > _localCoefficientMap; // pair(fineH1Order,coarseH1Order)
  
  Teuchos::RCP<Epetra_MultiVector> _diag; // diagonal of the fine (global) stiffness matrix
  Teuchos::RCP<Epetra_MultiVector> _diag_sqrt; // square root of the diagonal of the fine (global) stiffness matrix
  Teuchos::RCP<Epetra_MultiVector> _diag_inv; // inverse of the diagonal
  
  mutable double _timeMapFineToCoarse, _timeMapCoarseToFine, _timeCoarseImport, _timeConstruction, _timeCoarseSolve, _timeLocalCoefficientMapConstruction;  // totals over the life of the object
  
  mutable bool _haveSolvedOnCoarseMesh; // if this is true, then we can call resolve() instead of solve().
  
  Teuchos::RCP<Epetra_CrsMatrix> _P; // prolongation operator
  
  Teuchos::RCP<Epetra_Operator> _smoother;
public: // promoted these two to public for testing purposes:
  LocalDofMapperPtr getLocalCoefficientMap(GlobalIndexType fineCellID) const;
  GlobalIndexType getCoarseCellID(GlobalIndexType fineCellID) const;

  set<GlobalIndexTypeToCast> setCoarseRHSVector(const Epetra_MultiVector &X, Epetra_FEVector &coarseRHSVector) const;
  
  void setUpSmoother(Epetra_CrsMatrix *fineStiffnessMatrix);
public:
  //! @name Destructor
  //@{
  //! Destructor
  ~GMGOperator() {}
  //@}
  
  //! @name Constructor
  //@{
  //! Constructor
  GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap, Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation, bool fineSolverUsesDiagonalScaling = true);
  //@}
  
  //! @name Attribute set methods
  //@{
  
  //! If set true, transpose of this operator will be applied.
  /*! This flag allows the transpose of the given operator to be used implicitly.  Setting this flag
   affects only the Apply() and ApplyInverse() methods.  If the implementation of this interface
   does not support transpose use, this method should return a value of -1.
   
   \param In
   UseTranspose -If true, multiply by the transpose of operator, otherwise just use operator.
   
   \return Integer error code, set to 0 if successful.  Set to -1 if this implementation does not support transpose.
   */
  int SetUseTranspose(bool UseTranspose);
  //@}
  
  //! Diagonal of the stiffness matrix
  /*!
   
   \param In
   diagonal - diagonal of the stiffness matrix
   
   */
  void setStiffnessDiagonal(Teuchos::RCP<Epetra_MultiVector> diagonal);
  
  //! Set new fine mesh
  /*!
   
   \param In
   fineMesh - new fine mesh
   
   
   \param In
   finePartitionMap - partition map for the new fine mesh
   
   */
  void setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap);

  void clearTimings();
  void reportTimings() const;
  
  void constructLocalCoefficientMaps(); // we'll do this lazily if this is not called; this is mostly a way to separate out the time costs
  
  void computeCoarseStiffnessMatrix(Epetra_CrsMatrix *fineStiffnessMatrix);
  
  Teuchos::RCP<Epetra_CrsMatrix> constructProlongationOperator(); // rows belong to the fine grid, columns to the coarse
  
  //! @name Mathematical functions
  //@{
  
  //! Returns the result of a Epetra_Operator applied to a Epetra_MultiVector X in Y.
  /*!
   \param In
   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
   \param Out
   Y -A Epetra_MultiVector of dimension NumVectors containing result.
   
   \return Integer error code, set to 0 if successful.
   */
  int Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;
  
  //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
  /*!
   \param In
   X - A Epetra_MultiVector of dimension NumVectors to solve for.
   \param Out
   Y -A Epetra_MultiVector of dimension NumVectors containing result.
   
   \return Integer error code, set to 0 if successful.
   
   \warning In order to work with AztecOO, any implementation of this method must
   support the case where X and Y are the same object.
   */
  int ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;
  
  //! Returns the infinity norm of the global matrix.
  /* Returns the quantity \f$ \| A \|_\infty\f$ such that
   \f[\| A \|_\infty = \max_{1\lei\lem} \sum_{j=1}^n |a_{ij}| \f].
   
   \warning This method must not be called unless HasNormInf() returns true.
   */
  double NormInf() const;
  //@}

  //! @name Attribute access functions
  //@{
  
  //! Returns a character string describing the operator
  const char * Label() const;
  
  //! Returns the current UseTranspose setting.
  bool UseTranspose() const;
  
  //! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
  bool HasNormInf() const;
  
  //! Returns a pointer to the Epetra_Comm communicator associated with this operator.
  const Epetra_Comm & Comm() const;
  
  //! Returns the Epetra_Map object associated with the domain of this operator.
  const Epetra_Map & OperatorDomainMap() const;
  
  //! Returns the Epetra_Map object associated with the range of this operator.
  const Epetra_Map & OperatorRangeMap() const;
  
  void setApplyDiagonalSmoothing(bool value);
  
  //! factorization choices for Schwarz blocks, when a Schwarz smoother is used.
  enum FactorType {
    Direct,
    ILU,
    IC
  };
  
  void setSchwarzFactorizationType(FactorType choice);
  
  enum SmootherChoice {
    POINT_JACOBI,
    POINT_SYMMETRIC_GAUSS_SEIDEL,
    BLOCK_JACOBI,
    BLOCK_SYMMETRIC_GAUSS_SEIDEL,
    IFPACK_ADDITIVE_SCHWARZ,
    CAMELLIA_ADDITIVE_SCHWARZ
  };
  
  void setSmootherType(SmootherChoice smootherType);
  void setSmootherOverlap(int overlap);
  
  void setFineSolverUsesDiagonalScaling(bool value);
  //@}

  //! Returns the prolongation operator (an Epetra_CrsMatrix).
  Teuchos::RCP<Epetra_CrsMatrix> getProlongationOperator(); // prolongation operator
  
  //! Constructs and returns an Epetra_CrsMatrix for the smoother.
  Teuchos::RCP<Epetra_CrsMatrix> getSmootherAsMatrix();

  //! Returns the coarse stiffness matrix (an Epetra_CrsMatrix).
  Teuchos::RCP<Epetra_CrsMatrix> getCoarseStiffnessMatrix();
private:
  SmootherChoice _smootherType;
  int _smootherOverlap;
  
  FactorType _schwarzBlockFactorizationType;
};



#endif /* defined(__Camellia_debug__GMGOperator__) */
