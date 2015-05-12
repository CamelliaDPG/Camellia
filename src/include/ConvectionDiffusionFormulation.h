//
//  ConvectionDiffusionFormulation.h
//  Camellia
//
//  Created by Nate Roberts on 10/16/14.
//
//

#ifndef Camellia_ConvectionDiffusionFormulation_h
#define Camellia_ConvectionDiffusionFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"

namespace Camellia {
  class ConvectionDiffusionFormulation {
    BFPtr _convectionDiffusionBF;
    int _spaceDim;
    double _epsilon;
    FunctionPtr _beta;

    static const string s_u;
    static const string s_sigma;

    static const string s_uhat;
    static const string s_tc;

    static const string s_v;
    static const string s_tau;
  public:
    ConvectionDiffusionFormulation(int spaceDim, bool useConformingTraces, FunctionPtr beta, double epsilon=1e-2);

    BFPtr bf();

    // field variables:
    VarPtr u();
    VarPtr sigma();

    // traces:
    VarPtr tc();
    VarPtr uhat();

    // test variables:
    VarPtr v();
    VarPtr tau();
  };
}

#endif
