//
//  H1ProjectionFormulation.h
//  Camellia
//
//  Created by Brendan Keith 08/16.
//
//

#ifndef Camellia_H1ProjectionFormulation_h
#define Camellia_H1ProjectionFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"

namespace Camellia
{
class H1ProjectionFormulation
{
public:
  enum H1ProjectionFormulationChoice
  {
    CONTINUOUS_GALERKIN,
    PRIMAL,
    ULTRAWEAK
  };
private:
  BFPtr _H1ProjectionBF;
  int _spaceDim;

  static const string S_PHI;
  static const string S_PSI;

  static const string S_PHI_HAT;
  static const string S_PSI_N_HAT;

  static const string S_Q;
  static const string S_TAU;
public:
  H1ProjectionFormulation(int spaceDim, bool useConformingTraces, H1ProjectionFormulationChoice formulationChoice=ULTRAWEAK);

  BFPtr bf();

  // field variables:
  VarPtr phi();
  VarPtr psi();

  // traces:
  VarPtr psi_n_hat();
  VarPtr phi_hat();

  // test variables:
  VarPtr q();
  VarPtr tau();
};
}

#endif