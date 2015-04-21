#ifndef DPG_CONFUSION_BILINEAR_FORM
#define DPG_CONFUSION_BILINEAR_FORM

#include "BF.h"

using namespace Camellia;

class ConfusionBilinearForm {
private:
  double _epsilon, _beta_x, _beta_y;
  bool _useConstBeta;
public:
  // trial variable names:
  static const string S_SIGMA_1, S_SIGMA_2, S_U, S_U_HAT, S_BETA_N_U_MINUS_SIGMA_HAT;
  // test variable names:
  static const string S_V, S_TAU;
  
  static BFPtr confusionBF(double epsilon, double beta_x, double beta_y);
  static BFPtr confusionBF(double epsilon, FunctionPtr beta);
  
  static int U_ID, V_ID, SIGMA_1_ID, SIGMA_2_ID;
};

#endif
