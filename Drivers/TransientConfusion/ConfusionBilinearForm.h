#ifndef DPG_CONFUSION_BILINEAR_FORM
#define DPG_CONFUSION_BILINEAR_FORM

#include "BilinearForm.h"

class ConfusionBilinearForm : public BilinearForm {
private:
  double _epsilon, _beta_x, _beta_y, _dt, _T;
  vector<EOperatorExtended> _uvTestOperators;
public:
  ConfusionBilinearForm(double epsilon, double beta_x, double beta_y, double dt);
  
  // implement the virtual methods declared in super:
  const string & testName(int testID);
  const string & trialName(int trialID);
  
  void trialTestOperators(int trialID, int testID, 
			  vector<EOperatorExtended> &trialOperators,
			  vector<EOperatorExtended> &testOperators);
  
  void applyBilinearFormData(FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
						    int trialID, int testID, int operatorIndex,
						    FieldContainer<double> &points);

  void set_dt(double new_dt);
  double get_dt();
  double increment_T();
  double get_T();
  int get_transient_trialID();
  
  virtual EFunctionSpaceExtended functionSpaceForTest(int testID);
  virtual EFunctionSpaceExtended functionSpaceForTrial(int trialID);
  
  bool isFluxOrTrace(int trialID);
  
  enum ETestIDs {
    TAU = 0,
    V
  };
  
  enum ETrialIDs {
    U_HAT = 0,
    BETA_N_U_MINUS_SIGMA_HAT,
    U,
    SIGMA_1,
    SIGMA_2
  };
};

#endif
