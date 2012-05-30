#ifndef STOKES_MATH_BILINEAR_FORM
#define STOKES_MATH_BILINEAR_FORM

/*
 *  StokesMathBilinearForm.h
 *
 *  Created by Nathan Roberts on 3/22/12.
 *
 */

#include "BilinearForm.h"

class StokesMathBilinearForm : public BilinearForm {
  double _mu; // viscosity (constant)
public:
  StokesMathBilinearForm(double mu);
    
  // implement the virtual methods declared in super:
  const string & testName(int testID);
  const string & trialName(int trialID);
  
  bool trialTestOperator(int trialID, int testID, 
                         EOperatorExtended &trialOperator, EOperatorExtended &testOperator);
  
  void applyBilinearFormData(int trialID, int testID,
                             FieldContainer<double> &trialValues, FieldContainer<double> &testValues,
                             const FieldContainer<double> &points);
  
  virtual EFunctionSpaceExtended functionSpaceForTest(int testID);
  virtual EFunctionSpaceExtended functionSpaceForTrial(int trialID);
  
  bool isFluxOrTrace(int trialID);
  
  enum ETestIDs {
    Q_1 = 0, // 0
    Q_2,     // 1
    V_1,     // 2
    V_2,     // 3
    V_3      // 4
  };
  
  enum ETrialIDs {
    U1_HAT = 0,     // 0
    U2_HAT,         // 1
    SIGMA1_N_HAT,   // 2
    SIGMA2_N_HAT,   // 3
    U1,             // 4
    U2,             // 5
    SIGMA_11,       // 6
    SIGMA_12,       // 7
    SIGMA_21,       // 8
    SIGMA_22,       // 9
    P               // 10
  };


};

#endif
