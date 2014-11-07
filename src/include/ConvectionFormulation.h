//
//  ConvectionFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/16/14.
//
//

#ifndef Camellia_ConvectionFormulation_h
#define Camellia_ConvectionFormulation_h

#include "VarFactory.h"
#include "BF.h"

class ConvectionFormulation {
  BFPtr _convectionBF;
  int _spaceDim;

  static const string S_U;

  static const string S_Q_N_HAT;

  static const string S_V;
public:
  ConvectionFormulation(int spaceDim, FunctionPtr convectiveFunction);
  
  BFPtr bf();
  
  // field variables:
  VarPtr u();
  
  // traces:
  VarPtr q_n_hat();
  
  // test variables:
  VarPtr v();
};

#endif