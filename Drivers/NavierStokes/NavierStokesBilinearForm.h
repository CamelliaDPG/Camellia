#ifndef NAVIER_STOKES_BILINEAR_FORM
#define NAVIER_STOKES_BILINEAR_FORM

/*
 *  NavierStokesBilinearForm.h
 *
 *  Created by Nathan Roberts on 3/6/12.
 *
 */

#include "BilinearForm.h"

class NavierStokesBilinearForm : public BilinearForm {
  double _Reyn, _Mach; // Reynolds number (constant)

  double _gamma,_cv,_Pr, _mu, _eta, _kappa;
  FieldContainer<int> _EulerianMatrixEntries;
  map< pair<int, int>, vector< int > > _backFlowInteractions_x; // map from (trial,test) --> trialID involved from background flow
  map< pair<int, int>, vector< int > > _backFlowInteractions_y; // map from (trial,test) --> trialID involved from background flow
  Teuchos::RCP<Solution> _backgroundFlow;
  FieldContainer<double> getValuesToDot(int trialID, int testID, Teuchos::RCP<BasisCache> basisCache);
public:
  NavierStokesBilinearForm(double Reyn, double Mach);
    
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
    // TODO: edit these
    TAU_1 = 0,
    TAU_2,
    TAU_3,
    V_1,
    V_2,
    V_3,
    V_4
  };
  
  enum ETrialIDs {
    // traces
    U1_HAT = 0,
    U2_HAT,
    T_HAT,
    // fluxes
    F1_N,
    F2_N,
    F3_N,
    F4_N,
    // field variables
    U1,  // x velocity
    U2,  // y velocity
    RHO, // density
    T,   // temperature
    SIGMA_11, // stress tensor
    SIGMA_21,
    SIGMA_22,
    Q_1,    // heat flux (x component)
    Q_2,    // heat flux (y component)
    OMEGA   // infinitesimal strain
  };

  bool isViscousStressVariable(int trialID);  
  bool isHeatStressVariable(int trialID);  
  bool isEulerianVariable(int trialID);
  
}
 
};

#endif
