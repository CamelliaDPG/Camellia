// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  HeatFormulation.h
//  Camellia
//
//  Created by Stefan Henneking on 11/14/19.
//
//

#ifndef Camellia_HeatFormulation_h
#define Camellia_HeatFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"

namespace Camellia
{
class HeatFormulation
{
public:
  enum HeatFormulationChoice
  {
    CONTINUOUS_GALERKIN,
    PRIMAL,
    ULTRAWEAK
  };
private:
  BFPtr _heatBF;
  int _spaceDim;

  static const string S_U;
  static const string S_SIGMA;

  static const string S_U_HAT;
  static const string S_SIGMA_N_HAT;

  static const string S_V;
  static const string S_TAU;
public:
  HeatFormulation(int spaceDim, bool useConformingTraces, double timeStep, double theta=1.0, HeatFormulationChoice formulationChoice=ULTRAWEAK, bool fieldImplicit=false);

  BFPtr bf();
  
  RHSPtr rhs(FunctionPtr forcingFunction);

  // field variables:
  VarPtr u();
  VarPtr sigma();

  // traces:
  VarPtr sigma_n_hat();
  VarPtr u_hat();

  // test variables:
  VarPtr v();
  VarPtr tau();
};
}

#endif
