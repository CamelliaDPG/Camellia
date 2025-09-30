// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  GMGOperator.h
//  Camellia-debug
//
//  Created by Nate Roberts on 7/3/14.
//
//
#ifndef __Camellia_debug__GMGOperator__
#define __Camellia_debug__GMGOperator__

#include "BasisReconciliation.h"
#include "HDF5Exporter.h"
#include "IP.h"
#include "LocalDofMapper.h"
#include "Mesh.h"
#include "RefinementPattern.h"
#include "Solution.h"
#include "Solver.h"

#include "Ifpack_Preconditioner.h"
#include "Tpetra_Operator.hpp"
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

class GMGOperator : public TOperator<double>, public Narrator
{
  using Scalar = double;
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
  
  enum MultigridStrategy
  {
    TWO_LEVEL, // aka ADDITIVE
    V_CYCLE,
    W_CYCLE,
    FULL_MULTIGRID_V,
    FULL_MULTIGRID_W,
    SMOOTHER_ONLY // No coarse solve -- just one application of the smoother per "multigrid" cycle
  };
private:
  bool _debugMode; // in debug mode, output verbose info about what we're doing on rank 0

  bool _hierarchicalNeighborsForSchwarz; // Applies only to Camellia Additive Schwarz
  int _dimensionForSchwarzNeighborRelationship; // Applies only to Camellia Additive Schwarz
  
  bool _isFinest = true;
  bool _clearFinestCondensedDofInterpreterAfterProlongation = false;

  TSolutionPtr<Scalar> _coarseSolution;

  bool _useStaticCondensation; // for both coarse and fine solves
  Teuchos::RCP<DofInterpreter> _fineDofInterpreter;

  SmootherApplicationType _smootherApplicationType;
  
  MeshPtr _fineMesh, _coarseMesh;
  Epetra_Map _finePartitionMap;

  BCPtr _bc;

  TimeStatistics getStatistics(double timeValue) const;

  Teuchos::RCP<Solver> _coarseSolver;
  Teuchos::RCP<GMGOperator> _coarseOperator;
  
  // map from trial ordering to a matrix that will "duplicate" the fluxes as dictated by conformity,
  // required for h-refinements in context of static condensation.
  // see comment in getLocalCoefficientMap() implementation
  mutable map<DofOrdering*, Teuchos::RCP<Epetra_SerialDenseMatrix>> _fluxDuplicationMap;
  Teuchos::RCP<Epetra_SerialDenseMatrix> fluxDuplicationMapForCoarseCell(GlobalIndexType coarseCellID) const;

  mutable Teuchos::RCP<BasisReconciliation> _br;
  mutable map< pair< pair<int,int>, RefinementBranch >, LocalDofMapperPtr > _localCoefficientMap; // pair(fineH1Order,coarseH1Order)

  TMatrixPtr<Scalar> _fineStiffnessMatrix;
  Teuchos::RCP<Epetra_CrsMatrix> _eFineStiffnessMatrix;
  
  mutable double _timeMapFineToCoarse, _timeMapCoarseToFine, _timeCoarseImport, _timeConstruction, _timeCoarseSolve, _timeLocalCoefficientMapConstruction, _timeComputeCoarseStiffnessMatrix, _timeProlongationOperatorConstruction,
      _timeSetUpSmoother, _timeUpdateCoarseOperator, _timeApplyFineStiffness, _timeApplySmoother; // totals over the life of the object

  mutable bool _haveSolvedOnCoarseMesh; // if this is true, then we can call resolve() instead of solve().

  MultigridStrategy _multigridStrategy;
  TMatrixPtr<double> _P; // prolongation operator

  Teuchos::RCP<Epetra_Operator> _smoother;
  double _smootherWeight;
  int _smootherApplicationCount; // default to 1, but 2 may often be a better choice (especially when doing more than 2 levels)
  Teuchos::RCP<Epetra_MultiVector> _smootherDiagonalWeight;
  bool _useSchwarzDiagonalWeight, _useSchwarzScalingWeight; // when true, will set _smootherWeight_sqrt and _smootherWeight during setUpSmoother()
  
  void reportTimings(StatisticChoice whichStat, bool sumAllOperators) const;
  
  // ! private method; allows us to swap the fine and coarse roles in certain circumstances.
  Teuchos::RCP<Epetra_FECrsMatrix> constructProlongationOperator(Teuchos::RCP<DofInterpreter> coarseDofInterpreter,
                                                                 Teuchos::RCP<DofInterpreter> fineDofInterpreter,
                                                                 bool useStaticCondensation,
                                                                 Epetra_Map &coarseMap, Epetra_Map &fineMap,
                                                                 MeshPtr coarseMesh, MeshPtr fineMesh);
  
  int prolongationRowCount() const;
  int prolongationColCount() const;
  
  Teuchos::RCP<HDF5Exporter> _functionExporter;
  FunctionPtr _functionToExport;
  std::string _functionToExportName;
  int _exportTimeStep;
  
  void exportFunction();
public: // promoted these two to public for testing purposes:
  LocalDofMapperPtr getLocalCoefficientMap(GlobalIndexType fineCellID) const;
  GlobalIndexType getCoarseCellID(GlobalIndexType fineCellID) const;

  void setUpSmoother(Epetra_CrsMatrix* fineStiffnessMatrix);
public:
  //! @name Destructor
  //@{
  //! Destructor
  ~GMGOperator() {}
  //@}
  
  GMGOperator(GMGOperator::MultigridStrategy multigridStrategy, BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
              MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
              Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation, Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation);
  
  GMGOperator(GMGOperator::MultigridStrategy multigridStrategy, BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP,
              MeshPtr fineMesh, Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
              bool useStaticCondensation, Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation);

  //! @name Constructor (for coarsest grid operator)
  //@{
  //! Constructor
  /*! This constructor is intended for the topmost (coarsest) level of multigrid operators.
   \param coarseIP - May be null if useStaticCondensation is false
   */
  GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh,
              Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap,
              Teuchos::RCP<Solver> coarseSolver, bool useStaticCondensation,
              Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation);
  //@}
  
  //! @name Constructor (for all other operators)
  //@{
  //! Constructor
  /*! This constructor is intended for any multigrid operators finer than the coarsest mesh.
   */
  GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh,
              Teuchos::RCP<DofInterpreter> fineDofInterpreter, Epetra_Map finePartitionMap, bool useStaticCondensation,
              Teuchos::RCP<BasisReconciliation> sharedBasisReconciliation);
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

  // ! When set to true, will clear the stored local stiffness matrices and load vectors after prolongation operator is constructed.  This may save a good deal of memory, but at the expense of further computation during interpretation of the condensed solution.
  void setClearFinestCondensedDofInterpreterAfterProlongation(bool value);
  
  void clearTimings();
  void reportTimings(StatisticChoice stat = ALL) const;
  void reportTimingsSumOfOperators(StatisticChoice whichStat) const;
  std::map<string, double> timingReport() const;
  std::map<string, double> timingReportSumOfOperators() const;
  
  // ! res should hold the RHS on entry, Y the current solution.  On exit, res will have the updated residual, and A_Y will have A * Y
  void computeResidual(const TVector<Scalar>& Y, TVector<Scalar>& res, TVector<Scalar>& A_Y) const;
  
  void constructLocalCoefficientMaps(); // we'll do this lazily if this is not called; this is mostly a way to separate out the time costs

  void computeCoarseStiffnessMatrix(TMatrixPtr<Scalar> fineStiffnessMatrix);

  TMatrixPtr<Scalar> constructProlongationOperator(); // rows belong to the fine grid, columns to the coarse
  
  //! \brief Computes the operator-multivector application.
  /*! Loosely, performs \f$Y = \alpha \cdot A^{\textrm{mode}} \cdot X + \beta \cdot Y\f$. However, the details of operation
      vary according to the values of \c alpha and \c beta. Specifically
      - if <tt>beta == 0</tt>, apply() <b>must</b> overwrite \c Y, so that any values in \c Y (including NaNs) are ignored.
      - if <tt>alpha == 0</tt>, apply() <b>may</b> short-circuit the operator, so that any values in \c X (including NaNs) are ignored.
   */
  void apply(const TVector<Scalar> &X,
             TVector<Scalar> &Y,
             Teuchos::ETransp mode = Teuchos::NO_TRANS,
             Scalar alpha          = Teuchos::ScalarTraits<Scalar>::one(),
             Scalar beta           = Teuchos::ScalarTraits<Scalar>::zero()) const override;
  
  //! Returns the result of a the coarse operator (either the coarse solver or the coarse GMGOperator) applied to a Epetra_MultiVector X in Y.
  /*!
   \param In
   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
   \param Out
   Y -A Epetra_MultiVector of dimension NumVectors containing result.
   
   \return Integer error code, set to 0 if successful.
   */
  int ApplyInverseCoarseOperator(const TVector<Scalar> &X, TVector<Scalar> &Y) const;

  //! Returns the result of a Epetra_Operator inverse applied to an Epetra_MultiVector X in Y.
  /*!
   \param In
   X - A Epetra_MultiVector of dimension NumVectors to solve for.
   \param Out
   Y -A Epetra_MultiVector of dimension NumVectors containing result.

   \return Integer error code, set to 0 if successful.

   \warning Implementations of this method must support the case where X and Y are the same object.
   */
  int ApplyInverse(const TVector<Scalar> &X, TVector<Scalar> &Y) const;

  //! Returns the result of the smoother, using any scalar or vector weights that are set, applied to a Epetra_MultiVector X in Y.
  /*!
   \param In
   X - A Epetra_MultiVector of dimension NumVectors to multiply with matrix.
   \param Out
   Y -A Epetra_MultiVector of dimension NumVectors containing result.
   
   \return Integer error code, set to 0 if successful.
   */
  int ApplySmoother(const TVector<Scalar> &X, TVector<Scalar> &Y, bool weightOnLeft) const;
  
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

  //! Returns a pointer to the Teuchos_Comm communicator associated with this operator.
  const Teuchos_CommPtr Comm() const;

  //! Returns the Tpetra Map object associated with the domain of this operator.
  TMapConstPtr getDomainMap() const override;

  //! Returns the Tpetra Map object associated with the range of this operator.
  TMapConstPtr getRangeMap() const override;

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

  //! Dimension that will be used for neighbor relationships for Schwarz blocks, if CAMELLIA_ADDITIVE_SCHWARZ is the smoother choice.
  int getDimensionForSchwarzNeighborRelationship();

  //! Set dimension to use for neighbor relationships for Schwarz blocks.  Requires CAMELLIA_ADDITIVE_SCHWARZ as the smoother choice.
  void setDimensionForSchwarzNeighborRelationship(int value);
  
  void setSchwarzFactorizationType(FactorType choice);

  // ! Sets the number of times the smoother will be applied when it is applied.  (For multigrid strategies other than TWO_LEVEL, it is applied this many times before and after each application of the coarse operator.)
  void setSmootherApplicationCount(int count);
  
  // ! When set to MULTIPLICATIVE, will compute new residuals before and after the coarse solve.  Done in such a way as to preserve symmetry.
  void setSmootherApplicationType(SmootherApplicationType value);
  
  // ! When set to true, will weight (symmetrically) according to the inverse of the number of Schwarz blocks each dof participates in.  Currently only supported for CAMELLIA_ADDITIVE_SCHWARZ smoothers.
  void setUseSchwarzDiagonalWeight(bool value);
  
  // ! When set to true, will scale using the inverse of the maximum eigenvalue of the Schwarz smoother times the fine matrix.  Currently only supported for CAMELLIA_ADDITIVE_SCHWARZ smoothers.
  void setUseSchwarzScalingWeight(bool value);
  
  Teuchos::RCP<Epetra_Operator> getSmoother() const;
  
  SmootherChoice getSmootherType();
  
  void setSmootherType(SmootherChoice smootherType);
  int getSmootherOverlap() const;
  void setSmootherOverlap(int overlap);

  // ! Computed as 1/(1+N), where N = max #neighbors of any cell's overlap region.
  double computeSchwarzSmootherWeight();
  // ! smoother weight is applied to each application of the smoother. Default = 1.0
  double getSmootherWeight();
  // ! smoother weight is applied to each application of the smoother. Default = 1.0
  void setSmootherWeight(double weight);

  // ! smoother weight vector (used for Camellia additive Schwarz; may be null in other cases)
  Teuchos::RCP<Epetra_MultiVector> getSmootherWeightVector();
  
  void setLevelOfFill(int fillLevel);
  void setFillRatio(double fillRatio);
  
  //! Set the multigrid strategy: two-level (additive), V-cycle, W-cycle, or full multigrid.
  void setMultigridStrategy(MultigridStrategy choice);
  
  static std::string smootherString(SmootherChoice choice);

  //! If true, use sibling/cousin relationships to define neighborhoods for Schwarz blocks.  Requires CAMELLIA_ADDITIVE_SCHWARZ as the smoother choice.
  void setUseHierarchicalNeighborsForSchwarz(bool value);
  //@}

  //! Returns the fine mesh
  MeshPtr getFineMesh() const;
  
  //! Returns the fine stiffness matrix
  TMatrixPtr<Scalar> getFineStiffnessMatrix();
  
  //! Computes a Tpetra_CrsMatrix representation of this operator.  Note that this can be an expensive operation, and is primarily intended for testing.
  TMatrixPtr<Scalar> getMatrixRepresentation();
  
  //! Returns the prolongation operator (a Tpetra_CrsMatrix).
  TMatrixPtr<Scalar> getProlongationOperator(); // prolongation operator

  //! Constructs and returns an Tpetra_CrsMatrix for the smoother.  Note that this can be an expensive operation.  Primarily intended for testing.
  TMatrixPtr<Scalar> getSmootherAsMatrix();
  
  //! Returns the number of times the smoother will be applied when it is applied.
  int getSmootherApplicationCount() const;

  //! Returns the coarse stiffness matrix (a Tpetra_CrsMatrix).
  TMatrixPtr<Scalar> getCoarseStiffnessMatrix();

  //! Set the fine stiffness matrix; calls computeCoarseStiffnessMatrix() and setUpSmoother()
  void setFineStiffnessMatrix(TMatrixPtr<Scalar> fineStiffnessMatrix);

  //! Returns the coarse operator applied in the coarse solve.
  Teuchos::RCP<GMGOperator> getCoarseOperator();
  
  //! Returns the Solver used in the coarse solve.
  SolverPtr getCoarseSolver();

  //! Returns the Solution object used in the coarse solve.
  TSolutionPtr<Scalar> getCoarseSolution();
  
  //! returns the operator level, counting from 0 (the coarsest level, where the coarse solver is defined).
  int getOperatorLevel() const;
  
  //! Returns the fine dof interpreter
  Teuchos::RCP<DofInterpreter> getFineDofInterpreter();

  //! Set an exporter for typically an error function, to allow visualization of various smoother/coarse solve steps.
  void setFunctionExporter(Teuchos::RCP<HDF5Exporter> exporter, FunctionPtr function, std::string functionName="error");
  
  //! Informs the GMGOperator whether it is the finest one.  (If not, can freely discard local stiffness matrices stored in CondensedDofInterpreter, e.g.).  Default value is true.
  void setIsFinest(bool value);
  
  //! When true, the roles of fine and coarse are swapped during prolongation operator construction,
  //! and the transpose of the prolongation operator is used.
  bool getFineCoarseRolesSwapped() const;
  
  //! When true, the roles of fine and coarse are swapped during prolongation operator construction,
  //! and the transpose of the prolongation operator is used.
  void setFineCoarseRolesSwapped(bool value);
private:
  SmootherChoice _smootherType;
  int _smootherOverlap;

  FactorType _schwarzBlockFactorizationType;
  int _levelOfFill;
  double _fillRatio;
};
}



#endif /* defined(__Camellia_debug__GMGOperator__) */
