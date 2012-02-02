#ifndef DPG_CONFUSION_BILINEAR_FORM
#define DPG_CONFUSION_BILINEAR_FORM

#include "BilinearForm.h"

class ConfusionBilinearForm : public BilinearForm {
private:
  double _epsilon, _beta_x, _beta_y;
  bool _useConstBeta;
public:
  ConfusionBilinearForm(double epsilon, double beta_x, double beta_y);
  ConfusionBilinearForm(double epsilon);
  
  // implement the virtual methods declared in super:
  const string & testName(int testID);
  const string & trialName(int trialID);
  
  bool trialTestOperator(int trialID, int testID, 
                         EOperatorExtended &trialOperator, EOperatorExtended &testOperator);
  
  void applyBilinearFormData(int trialID, int testID,
                             FieldContainer<double> &trialValues, FieldContainer<double> &testValues, 
                             FieldContainer<double> &points);
  
  virtual EFunctionSpaceExtended functionSpaceForTest(int testID);
  virtual EFunctionSpaceExtended functionSpaceForTrial(int trialID);

  double getEpsilon();
  void setEpsilon(double newEpsilon);
  vector<double> getBeta(double x, double y);
  
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
