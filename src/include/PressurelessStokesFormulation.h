//
//  PressurelessStokesFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/9/14.
//
//

#ifndef Camellia_PressurelessStokesFormulation_h
#define Camellia_PressurelessStokesFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"

namespace Camellia {
  class PressurelessStokesFormulation {
    BFPtr _stokesBF;
    int _spaceDim;

    static const string S_U1, S_U2, S_U3;
    static const string S_SIGMA11, S_SIGMA12, S_SIGMA13;
    static const string S_SIGMA22, S_SIGMA23;
    static const string S_SIGMA33;

    static const string S_U1_HAT, S_U2_HAT, S_U3_HAT;
    static const string S_TN1_HAT, S_TN2_HAT, S_TN3_HAT;

    static const string S_V1, S_V2, S_V3;
    static const string S_TAU11, S_TAU12, S_TAU13;
    static const string S_TAU22, S_TAU23;
    static const string S_TAU33;
  public:
    PressurelessStokesFormulation(int spaceDim);
    
    BFPtr bf();
    
    // field variables:
    VarPtr sigma(int i, int j);
    VarPtr u(int i);
    
    // traces:
    VarPtr tn_hat(int i);
    VarPtr u_hat(int i);
    
    // test variables:
    VarPtr tau(int i, int j);
    VarPtr v(int i);
    
    LinearTermPtr p(); // pressure (defined in terms of sigma trace)
  };
}

#endif