// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER 

#ifndef BILINEAR_FORM_SPECIFICATION
#define BILINEAR_FORM_SPECIFICATION

#include "Intrepid_Types.hpp"
#include "Intrepid_FieldContainer.hpp"

#include "CamelliaIntrepidExtendedTypes.h" // defined by us

#include "DofOrdering.h"

#include "IP.h"

class BasisCache;
class ElementType;
class VarFactory;
class RHS;
typedef Teuchos::RCP< BasisCache > BasisCachePtr;
typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< RHS > RHSPtr;

using namespace std;
using namespace Intrepid;

using namespace Camellia;

class BilinearForm {
public:
  BilinearForm();
  virtual bool trialTestOperator(int trialID, int testID, 
                                 Camellia::EOperator &trialOperator,
                                 Camellia::EOperator &testOperator) { 
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "You must override either trialTestOperator or trialTestOperators!");
    return false;
  }; // specifies differential operators to apply to trial and test (bool = false if no test-trial term)
  
  virtual void trialTestOperators(int trialID, int testID, 
                                  vector<Camellia::EOperator> &trialOps,
                                  
                                  vector<Camellia::EOperator> &testOps); // default implementation calls trialTestOperator
  
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
  
  virtual void localStiffnessMatrixAndRHS(FieldContainer<double> &localStiffness, FieldContainer<double> &rhsVector,
                                          IPPtr ip, BasisCachePtr ipBasisCache,
                                          RHSPtr rhs,  BasisCachePtr basisCache);
  
  virtual int optimalTestWeights(FieldContainer<double> &optimalTestWeights, FieldContainer<double> &innerProductMatrix,
                                 ElementTypePtr elemType, FieldContainer<double> &cellSideParities,
                                 BasisCachePtr stiffnessBasisCache);
  
  virtual void stiffnessMatrix(FieldContainer<double> &stiffness, ElementTypePtr elemType,
                               FieldContainer<double> &cellSideParities, BasisCachePtr basisCache);
  
  virtual void stiffnessMatrix(FieldContainer<double> &stiffness, DofOrderingPtr trialOrdering, 
                               DofOrderingPtr testOrdering, FieldContainer<double> &cellSideParities,
                               BasisCachePtr basisCache);
                           
  const vector< int > & trialIDs();
  const vector< int > & testIDs();
  
  virtual const string & testName(int testID) = 0;
  virtual const string & trialName(int trialID) = 0;
  
  virtual Camellia::EFunctionSpace functionSpaceForTest(int testID) = 0;
  virtual Camellia::EFunctionSpace functionSpaceForTrial(int trialID) = 0;
  
  virtual bool isFluxOrTrace(int trialID) = 0;
  
  static const string & operatorName(Camellia::EOperator op);
  static int operatorRank(Camellia::EOperator op,
                          Camellia::EFunctionSpace fs);
  vector<int> trialVolumeIDs();
  vector<int> trialBoundaryIDs();
  
  virtual void printTrialTestInteractions();
  
  void setUseSPDSolveForOptimalTestFunctions(bool value);
  void setUseIterativeRefinementsWithSPDSolve(bool value);
  void setUseExtendedPrecisionSolveForOptimalTestFunctions(bool value);
  void setWarnAboutZeroRowsAndColumns(bool value);
  
  virtual VarFactory varFactory();
  
  virtual ~BilinearForm() {}
protected:
 
  vector< int > _trialIDs, _testIDs;
  static set<int> _normalOperators;
  bool _useSPDSolveForOptimalTestFunctions, _useIterativeRefinementsWithSPDSolve;
  bool _useQRSolveForOptimalTestFunctions;
  bool _warnAboutZeroRowsAndColumns;
};

typedef Teuchos::RCP<BilinearForm> BilinearFormPtr;
#endif
