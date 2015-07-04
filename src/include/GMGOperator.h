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

#include "BasisReconciliation.h"
#include "IP.h"
#include "LocalDofMapper.h"
#include "Mesh.h"
#include "RefinementPattern.h"
#include "Solution.h"
#include "Solver.h"

#include "Ifpack_Preconditioner.h"
#include <map>

using namespace std;

namespace Camellia
{
struct TimeStatistics
{
  double min;
  double max;
  double mean;
  double sum;
};
  enum StatisticChoice {
    ALL, MIN, MAX, MEAN, SUM
  };

class GMGOperator : public Epetra_Operator, public Narrator
{
public:
  enum SmootherChoice
  {
    POINT_JACOBI,
    POINT_SYMMETRIC_GAUSS_SEIDEL,
    BLOCK_JACOBI,
    BLOCK_SYMMETRIC_GAUSS_SEIDEL,
    IFPACK_ADDITIVE_SCHWARZ,
    CAMELLIA_ADDITIVE_SCHWARZ,
    NONE
  };
  
  enum SmootherApplicationType
  {
    ADDITIVE,
    MULTIPLICATIVE
  };
private:
  bool _debugMode; // in debug mode, output verbose info about what we're doing on rank 0

  bool _hierarchicalNeighborsForSchwarz; // Applies only to Camellia Additive Schwarz

  TSolutionPtr<double> _coarseSolution;

  bool _useStaticCondensation; // for both coarse and fine solves
  Teuchos::RCP<DofInterpreter> _fineDofInterpreter;

  SmootherApplicationType _smootherApplicationType;
  
  MeshPtr _fineMesh, _coarseMesh;
  Epetra_Map _finePartitionMap;

  BCPtr _bc;

  TimeStatistics getStatistics(double timeValue) const;

  Teuchos::RCP<Solver> _coarseSolver;
  Teuchos::RCP<GMGOperator> _coarseOperator;

  mutable BasisReconciliation _br;
  mutable map< pair< pair<int,int>, RefinementBranch >, LocalDofMapperPtr > _localCoefficientMap; // pair(fineH1Order,coarseH1Order)

  Epetra_CrsMatrix* _fineStiffnessMatrix;
  
  mutable double _timeMapFineToCoarse, _timeMapCoarseToFine, _timeCoarseImport, _timeConstruction, _timeCoarseSolve, _timeLocalCoefficientMapConstruction, _timeComputeCoarseStiffnessMatrix, _timeProlongationOperatorConstruction,
      _timeSetUpSmoother, _timeUpdateCoarseOperator, _timeApplyFineStiffness, _timeApplySmoother; // totals over the life of the object

  mutable bool _haveSolvedOnCoarseMesh; // if this is true, then we can call resolve() instead of solve().

  Teuchos::RCP<Epetra_CrsMatrix> _P; // prolongation operator

  Teuchos::RCP<Epetra_Operator> _smoother;
  double _smootherWeight;
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

  //! @name Constructor (for coarsest grid operator)
  //@{
  //! Constructor
  /*! This constructor is intended for the topmost (coarsest) level of multigrid operators.
   \param coarseIP - May be null if useStaticCondensation is false
   */
  GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh,
              Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
              Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation);
  //@}
  
  //! @name Constructor (for all other operators)
  //@{
  //! Constructor
  /*! This constructor is intended for any multigrid operators finer than the coarsest mesh.
   */
  GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap);
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

  //! Set new fine mesh
  /*!

   \param In
   fineMesh - new fine mesh


   \param In
   finePartitionMap - partition map for the new fine mesh

   */
  void setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap);

  void clearTimings();
  void reportTimings(StatisticChoice stat = ALL) const;

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
  
  //! Returns the result of a the coarse operator (either the coarse solver or the coarse GMGOperator) applied to a Epetra_MultiVector X in Y.
  /*!
   \param In
   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
   \param Out
   Y -A Epetra_MultiVector of dimension NumVectors containing result.
   
   \return Integer error code, set to 0 if successful.
   */
  int ApplyInverseCoarseOperator(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const;

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

  //! factorization choices for Schwarz blocks, when a Schwarz smoother is used.
  enum FactorType
  {
    Direct,
    ILU,
    IC
  };
  
  //! set the coarse Operator
  void setCoarseOperator(Teuchos::RCP<GMGOperator> coarseOperator);
  
  //! set the coarse Solver
  void setCoarseSolver(SolverPtr coarseSolver);
  
  //! sets debug mode for verbose console output on rank 0.
  void setDebugMode(bool value);

  void setSchwarzFactorizationType(FactorType choice);
  
  // ! When set to MULTIPLICATIVE, will compute new residuals before and after the coarse solve.  Done in such a way as to preserve symmetry.
  void setSmootherApplicationType(SmootherApplicationType value);
  
  SmootherChoice getSmootherType();
  
  void setSmootherType(SmootherChoice smootherType);
  void setSmootherOverlap(int overlap);

  // ! smoother weight is applied to each application of the smoother. Default = 1.0
  double getSmootherWeight();
  // ! smoother weight is applied to each application of the smoother. Default = 1.0
  void setSmootherWeight(double weight);

  void setLevelOfFill(int fillLevel);
  void setFillRatio(double fillRatio);
  
  static std::string smootherString(SmootherChoice choice);

  //! If true, use sibling/cousin relationships to define neighborhoods for Schwarz blocks.  Requires CAMELLIA_ADDITIVE_SCHWARZ as the smoother choice.
  void setUseHierarchicalNeighborsForSchwarz(bool value);
  //@}

  //! Returns the fine stiffness matrix
  Epetra_CrsMatrix* getFineStiffnessMatrix();
  
  //! Computes an Epetra_CrsMatrix representation of this operator.  Note that this can be an expensive operation, and is primarily intended for testing.
  Teuchos::RCP<Epetra_CrsMatrix> getMatrixRepresentation();
  
  //! Returns the prolongation operator (an Epetra_CrsMatrix).
  Teuchos::RCP<Epetra_CrsMatrix> getProlongationOperator(); // prolongation operator

  //! Constructs and returns an Epetra_CrsMatrix for the smoother.  Note that this can be an expensive operation.  Primarily intended for testing.
  Teuchos::RCP<Epetra_CrsMatrix> getSmootherAsMatrix();

  //! Returns the coarse stiffness matrix (an Epetra_CrsMatrix).
  Teuchos::RCP<Epetra_CrsMatrix> getCoarseStiffnessMatrix();

  //! Set the fine stiffness matrix; calls computeCoarseStiffnessMatrix() and setUpSmoother()
  void setFineStiffnessMatrix(Epetra_CrsMatrix* fineStiffnessMatrix);

  //! Returns the coarse operator applied in the coarse solve.
  Teuchos::RCP<GMGOperator> getCoarseOperator();
  
  //! Returns the Solver used in the coarse solve.
  SolverPtr getCoarseSolver();

  //! Returns the Solution object used in the coarse solve.
  TSolutionPtr<double> getCoarseSolution();
  
  //! Returns the fine dof interpreter
  Teuchos::RCP<DofInterpreter> getFineDofInterpreter();
private:
  SmootherChoice _smootherType;
  int _smootherOverlap;

  FactorType _schwarzBlockFactorizationType;
  int _levelOfFill;
  double _fillRatio;
};
}



#endif /* defined(__Camellia_debug__GMGOperator__) */
