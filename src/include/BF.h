//
//  BF.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_BF_h
#define Camellia_BF_h

#include "LinearTerm.h"

#include "VarFactory.h"

#include "IP.h"

class RHS;
typedef Teuchos::RCP<RHS> RHSPtr;

class ElementType;
typedef Teuchos::RCP<ElementType> ElementTypePtr;

class BF {
  typedef pair< LinearTermPtr, LinearTermPtr > BilinearTerm;
  vector< BilinearTerm > _terms;
  VarFactory _varFactory;
  
  bool _isLegacySubclass;
  //members that used to be part of BilinearForm:
protected:
  vector< int > _trialIDs, _testIDs;
  static set<int> _normalOperators;
  bool _useSPDSolveForOptimalTestFunctions, _useIterativeRefinementsWithSPDSolve;
  bool _useQRSolveForOptimalTestFunctions;
  bool _warnAboutZeroRowsAndColumns;
  
  bool checkSymmetry(FieldContainer<double> &innerProductMatrix);
public:
  BF( bool isLegacySubclass ); // legacy version; new code should use a VarFactory version of the constructor
  
  BF( VarFactory varFactory ); // copies (note that external changes in VarFactory won't be registered by BF)
  BF( VarFactory varFactory, VarFactory::BubnovChoice choice);

  void addTerm( LinearTermPtr trialTerm, LinearTermPtr testTerm );
  void addTerm( VarPtr trialVar, LinearTermPtr testTerm );
  void addTerm( VarPtr trialVar, VarPtr testVar );
  void addTerm( LinearTermPtr trialTerm, VarPtr testVar);
  
  // applyBilinearFormData() methods are all legacy methods
  virtual void applyBilinearFormData(int trialID, int testID,
                                     FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                     const FieldContainer<double> &points) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override either some version of applyBilinearFormData!");
  }
  
  virtual void applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                     int trialID, int testID, int operatorIndex,
                                     const FieldContainer<double> &points); // default implementation calls operatorIndex-less version
  
  virtual void applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                                     int trialID, int testID, int operatorIndex,
                                     BasisCachePtr basisCache);
  // default implementation calls BasisCache-less version
  
  // BilinearForm implementation:
  virtual const string & testName(int testID);
  virtual const string & trialName(int trialID);
  
  virtual Camellia::EFunctionSpace functionSpaceForTest(int testID);
  virtual Camellia::EFunctionSpace functionSpaceForTrial(int trialID);
  
  virtual bool isFluxOrTrace(int trialID);
  
  IPPtr graphNorm(double weightForL2TestTerms = 1.0);
  IPPtr graphNorm(const map<int, double> &varWeights, double weightForL2TestTerms = 1.0);
  IPPtr l2Norm();
  IPPtr naiveNorm(int spaceDim);
  
  string displayString();
  
  virtual void localStiffnessMatrixAndRHS(FieldContainer<double> &localStiffness, FieldContainer<double> &rhsVector,
                                          IPPtr ip, BasisCachePtr ipBasisCache,
                                          RHSPtr rhs,  BasisCachePtr basisCache);
  
  virtual int optimalTestWeights(FieldContainer<double> &optimalTestWeights, FieldContainer<double> &innerProductMatrix,
                                 ElementTypePtr elemType, FieldContainer<double> &cellSideParities,
                                 BasisCachePtr stiffnessBasisCache);
  
  void printTrialTestInteractions();
  
  void stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
                       FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);
  void stiffnessMatrix(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
		       FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache,
		       bool checkForZeroCols);
  
  // legacy version of stiffnessMatrix():
  virtual void stiffnessMatrix(FieldContainer<double> &stiffness, DofOrderingPtr trialOrdering,
                               DofOrderingPtr testOrdering, FieldContainer<double> &cellSideParities,
                               BasisCachePtr basisCache);
  
  void bubnovStiffness(FieldContainer<double> &stiffness, Teuchos::RCP<ElementType> elemType,
		       FieldContainer<double> &cellSideParities, Teuchos::RCP<BasisCache> basisCache);
  
  LinearTermPtr testFunctional(SolutionPtr trialSolution, bool excludeBoundaryTerms=false, bool overrideMeshCheck=false);
  
  virtual bool trialTestOperator(int trialID, int testID,
                                 Camellia::EOperator &trialOperator,
                                 Camellia::EOperator &testOperator) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override either trialTestOperator or trialTestOperators!");
    return false;
  }; // specifies differential operators to apply to trial and test (bool = false if no test-trial term)
  
  virtual void trialTestOperators(int trialID, int testID,
                                  vector<Camellia::EOperator> &trialOps,
                                  
                                  vector<Camellia::EOperator> &testOps); // default implementation calls trialTestOperator
  
  virtual VarFactory varFactory();
  
  // non-virtual methods (originally from BilinearForm):
  void setUseSPDSolveForOptimalTestFunctions(bool value);
  void setUseIterativeRefinementsWithSPDSolve(bool value);
  void setUseExtendedPrecisionSolveForOptimalTestFunctions(bool value);
  void setWarnAboutZeroRowsAndColumns(bool value);
  
  const vector< int > & trialIDs();
  const vector< int > & testIDs();
  
  vector<int> trialVolumeIDs();
  vector<int> trialBoundaryIDs();

  virtual ~BF() {}
  
  static Teuchos::RCP<BF> bf(VarFactory &vf);
};

typedef Teuchos::RCP<BF> BFPtr;

#endif
