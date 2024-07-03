// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BF.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//

#ifndef Camellia_BF_h
#define Camellia_BF_h

#include "TypeDefs.h"

#include "LinearTerm.h"

#include "VarFactory.h"

#include "IP.h"

namespace Camellia
{
  const static string BF_FACTORED_CHOLESKY_SOLVE_TIMER_STRING = "BF factored Cholesky solve";
  const static string BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING = "BF local stiffness and RHS assembly"; //  (includes optimal test solve, if applicable)
  
template <typename Scalar>
class TBF
{
public:
  enum OptimalTestSolver
  {
    CHOLESKY,
    FACTORED_CHOLESKY,
    LU,
    QR
  };
private:
  vector< TBilinearTerm<Scalar> > _terms;
  vector< TBilinearTerm<Scalar> > _jumpTerms; // DG-style jump terms
  
  VarFactoryPtr _varFactory;

  std::function<void(int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType)> _optimalTestTimingCallback;
  std::function<void(int numElements, double timeRHS, ElementTypePtr elemType)> _rhsTimingCallback;
  
  bool _isLegacySubclass;
  //members that used to be part of BilinearForm:
protected:
  vector< int > _trialIDs, _testIDs;
  static set<int> _normalOperators;
  
  OptimalTestSolver _optimalTestSolver = FACTORED_CHOLESKY;
  
  bool _useIterativeRefinementsWithSPDSolve = false;
  bool _warnAboutZeroRowsAndColumns = true;
  bool _useSubgridMeshForOptimalTestSolve = false;
  bool _printTermWiseIntegrationOutput = false;
  
  TBFPtr<Scalar> _bfForOptimalTestSolve; // if not null, then _bfForOptimalTestSolve will be used as the RHS in the test solve, instead of "this"
  
  bool checkSymmetry(Camellia::FieldContainer<Scalar> &innerProductMatrix);
public:
  TBF( bool isLegacySubclass ); // legacy version; new code should use a VarFactory version of the constructor

  TBF( VarFactoryPtr varFactory ); // copies (note that external changes in VarFactory won't be registered by TBF)
  TBF( VarFactoryPtr varFactory, VarFactory::BubnovChoice choice);

  void addTerm( TLinearTermPtr<Scalar> trialTerm, TLinearTermPtr<Scalar> testTerm );
  void addTerm( VarPtr trialVar, TLinearTermPtr<Scalar> testTerm );
  void addTerm( VarPtr trialVar, VarPtr testVar );
  void addTerm( TLinearTermPtr<Scalar> trialTerm, VarPtr testVar);
  
  // ! Add a DG-style jump term.  Only applicable when running in Bubnov-Galerkin mode (i.e. a null IP in Solution).
  void addJumpTerm( TLinearTermPtr<Scalar> trialTerm, TLinearTermPtr<Scalar> testTerm );

  // applyBilinearFormData() methods are all legacy methods
  virtual void applyBilinearFormData(int trialID, int testID,
                                     Camellia::FieldContainer<Scalar> &trialValues, Camellia::FieldContainer<Scalar> &testValues,
                                     const Camellia::FieldContainer<double> &points)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override some version of applyBilinearFormData!");
  }

  virtual void applyBilinearFormData(Camellia::FieldContainer<Scalar> &trialValues, Camellia::FieldContainer<Scalar> &testValues,
                                     int trialID, int testID, int operatorIndex,
                                     const Camellia::FieldContainer<double> &points); // default implementation calls operatorIndex-less version

  virtual void applyBilinearFormData(Camellia::FieldContainer<Scalar> &trialValues, Camellia::FieldContainer<Scalar> &testValues,
                                     int trialID, int testID, int operatorIndex,
                                     BasisCachePtr basisCache);
  // default implementation calls BasisCache-less version

  TBFPtr<Scalar> copy() const;
  
  // BilinearForm implementation:
  virtual const string & testName(int testID);
  virtual const string & trialName(int trialID);

  virtual Camellia::EFunctionSpace functionSpaceForTest(int testID);
  virtual Camellia::EFunctionSpace functionSpaceForTrial(int trialID);

  virtual bool isFluxOrTrace(int trialID);

  TIPPtr<Scalar> graphNorm(double weightForL2TestTerms = 1.0);
  TIPPtr<Scalar> graphNorm(const map<int, double> &varWeights, double weightForL2TestTerms = 1.0);
  TIPPtr<Scalar> graphNorm(const map<int, double> &trialVarWeights, const map<int, double> &testVarL2TermWeights);
  TIPPtr<Scalar> l2Norm();
  TIPPtr<Scalar> naiveNorm(int spaceDim);

  string displayString();
  
  const std::vector< TBilinearTerm<Scalar> > & getTerms() const;
  const std::vector< TBilinearTerm<Scalar> > & getJumpTerms() const;

  static int factoredCholeskySolve(Camellia::FieldContainer<Scalar> &ipMatrix, Camellia::FieldContainer<Scalar> &stiffnessEnriched,
                                   Camellia::FieldContainer<Scalar> &rhsEnriched, Camellia::FieldContainer<Scalar> &stiffness,
                                   Camellia::FieldContainer<Scalar> &rhs);
  
  virtual void localStiffnessMatrixAndRHS(Camellia::FieldContainer<Scalar> &localStiffness,
                                          Camellia::FieldContainer<Scalar> &rhsVector,
                                          TIPPtr<Scalar> ip, BasisCachePtr ipBasisCache,
                                          TRHSPtr<Scalar> rhs,  BasisCachePtr basisCache);

  // ! returns a list of test variables from VarFactory that do not enter the bilinear form
  std::vector<VarPtr> missingTestVars();
  
  // ! returns a list of trial variables from VarFactory that do not enter the bilinear form
  std::vector<VarPtr> missingTrialVars();
  
  // ! returns the number of potential nonzeros for the given trial ordering and test ordering
  int nonZeroEntryCount(DofOrderingPtr trialOrdering, DofOrderingPtr testOrdering);

  // ! computes both the optimal test weights and the (square) stiffness matrix generated by testing with these
  virtual int optimalTestWeightsAndStiffness(Camellia::FieldContainer<Scalar> &optimalTestWeights, Camellia::FieldContainer<Scalar> &stiffnessMatrix,
                                             ElementTypePtr elemType, const Camellia::FieldContainer<double> &cellSideParities,
                                             BasisCachePtr stiffnessBasisCache,
                                             IPPtr ip, BasisCachePtr ipBasisCache);

  void printTrialTestInteractions();
  
  // ! Debugging facility: "narrates" as bilinear form is accumulated.  Particularly useful if you are seeing "zero row" warnings (best set to true on at most one MPI rank; otherwise, the output will be confusing...)
  void setPrintTermWiseIntegrationOutput(bool value);

  virtual void stiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                               const Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);
  virtual void stiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                               const Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache,
                               bool rowMajor, bool checkForZeroCols);

  // legacy version of stiffnessMatrix():
  virtual void stiffnessMatrix(Camellia::FieldContainer<Scalar> &stiffness, DofOrderingPtr trialOrdering,
                               DofOrderingPtr testOrdering, Camellia::FieldContainer<double> &cellSideParities,
                               BasisCachePtr basisCache);

  void bubnovStiffness(Camellia::FieldContainer<Scalar> &stiffness, Teuchos::RCP<ElementType> elemType,
                       const Camellia::FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);

  TLinearTermPtr<Scalar> testFunctional(TSolutionPtr<Scalar> trialSolution, bool excludeBoundaryTerms=false,
                                        SolutionComponent solutionComponent=STANDARD_SOLUTION, int solutionStage=0,
                                        std::set<int> testIDsToExclude=std::set<int>());
  //! takes as argument a map that has trialID keys and FunctionPtr values.  Omitted trialIDs are taken to be zero.
  TLinearTermPtr<Scalar> testFunctional(const std::map<int,FunctionPtr> &solnMap, std::set<int> testIDsToExclude=std::set<int>());

  //! takes as argument a map that has testID keys and FunctionPtr values.  Omitted testIDs are taken to be zero.
  TLinearTermPtr<Scalar> trialFunctional(const std::map<int,FunctionPtr> &testMap);

  map<int, TFunctionPtr<Scalar> > applyAdjointOperatorDPGstar(TRieszRepPtr<double> dualSolution);

  virtual bool trialTestOperator(int trialID, int testID,
                                 Camellia::EOperator &trialOperator,
                                 Camellia::EOperator &testOperator)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override either trialTestOperator or trialTestOperators!");
  }; // specifies differential operators to apply to trial and test (bool = false if no test-trial term)

  virtual void trialTestOperators(int trialID, int testID,
                                  vector<Camellia::EOperator> &trialOps,

                                  vector<Camellia::EOperator> &testOps); // default implementation calls trialTestOperator

  virtual VarFactoryPtr varFactory();
  
  // ! If not null, the provided BF will be used to compute the optimal test functions, instead of "this".
  // ! In this case, "this" is integrated against the optimal test functions thus computed during "stiffness" matrix computations.
  void setBFForOptimalTestSolve(TBFPtr<Scalar> bf);
  
  void setOptimalTestTimingCallback(std::function<void(int numElements, double timeG, double timeB, double timeT, double timeK, ElementTypePtr elemType)> &optimalTestTimingCallback);
  void setRHSTimingCallback(std::function<void(int numElements, double timeRHS, ElementTypePtr elemType)> &rhsTimingCallback);

  OptimalTestSolver optimalTestSolver() const;
  void setOptimalTestSolver(OptimalTestSolver choice);
  void setUseIterativeRefinementsWithSPDSolve(bool value);
  void setUseExtendedPrecisionSolveForOptimalTestFunctions(bool value);
  void setUseSubgridMeshForOptimalTestFunctions(bool value);
  void setWarnAboutZeroRowsAndColumns(bool value);

  // ! Returns a map from trial ID to a function corresponding to that trial ID.  Fluxes are weighted by side parity (so that they are uniquely valued).
  // ! the solutionIdentifierExponent will be used in string representations (if "", an overbar is used; otherwise "^k" will be used for "k" argument here)
  std::map<int,FunctionPtr> solutionMap(SolutionPtr soln, const std::string &solutionIdentifierExponent="");
  
  const vector< int > & trialIDs();
  const vector< int > & testIDs();

  vector<int> trialVolumeIDs();
  vector<int> trialBoundaryIDs();

  virtual ~TBF() {}

  static TBFPtr<Scalar> bf(VarFactoryPtr &vf);
};

extern template class TBF<double>;
}

#endif
