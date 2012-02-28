//
//  TimeMarchingProblem.h
//  Camellia
//
//  Created by Nathan Roberts on 2/27/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_TimeMarchingProblem_h
#define Camellia_TimeMarchingProblem_h

#include "BilinearForm.h"

class TimeMarchingProblem : public BilinearForm, public RHS {
  Teuchos::RCP<BilinearForm> _bilinearForm;
  Teuchos::RCP<RHS> _rhs;
  double _dt;
public:
  TimeMarchingProblem(Teuchos::RCP<BilinearForm> bilinearForm, Teuchos::RCP<RHS> rhs);
  
  // BILINEAR FORM:
  virtual void trialTestOperators(int trialID, int testID, 
                                  vector<EOperatorExtended> &trialOps,
                                  vector<EOperatorExtended> &testOps);
  
  virtual void applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                                     int trialID, int testID, int operatorIndex,
                                     Teuchos::RCP<BasisCache> basisCache);
  
  virtual EFunctionSpaceExtended functionSpaceForTest(int testID) = 0;
  virtual EFunctionSpaceExtended functionSpaceForTrial(int trialID) = 0; 
  
  virtual bool isFluxOrTrace(int trialID) = 0;
  
  // RHS:
  bool nonZeroRHS(int testVarID);
  vector<EOperatorExtended> operatorsForTestID(int testID);
  virtual void rhs(int testVarID, int operatorIndex, Teuchos::RCP<BasisCache> basisCache, 
                   FieldContainer<double> &values);
  
  // methods that belong to the TimeMarchingProblem:
  virtual void timeLHS(FieldContainer<double> trialValues, int trialID);
  
  void setTimeStepSize(double dt);
  double timeStepSize();
  
};

#endif
