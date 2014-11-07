//
//  StokesVGPFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/29/14.
//
//

#ifndef Camellia_StokesVGPFormulation_h
#define Camellia_StokesVGPFormulation_h

#include "VarFactory.h"
#include "BF.h"

class StokesVGPFormulation {
  BFPtr _stokesBF;
  int _spaceDim;
  bool _useConformingTraces;
  double _mu;

  static const string S_U1, S_U2, S_U3;
  static const string S_P;
  static const string S_SIGMA1, S_SIGMA2, S_SIGMA3;

  static const string S_U1_HAT, S_U2_HAT, S_U3_HAT;
  static const string S_TN1_HAT, S_TN2_HAT, S_TN3_HAT;

  static const string S_V1, S_V2, S_V3;
  static const string S_Q;
  static const string S_TAU1, S_TAU2, S_TAU3;
public:
  StokesVGPFormulation(int spaceDim, bool useConformingTraces, double mu = 1.0);
  
  BFPtr bf();
  
  // field variables:
  VarPtr sigma(int i);
  VarPtr u(int i);
  VarPtr p();
  
  // traces:
  VarPtr tn_hat(int i);
  VarPtr u_hat(int i);
  
  // test variables:
  VarPtr tau(int i);
  VarPtr v(int i);
};

#endif