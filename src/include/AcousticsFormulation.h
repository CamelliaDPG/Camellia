//
//  AcousticsFormulation.h
//  Camellia
//
//  Created by Brendan Keith 11/18.
//
//

#ifndef Camellia_AcousticsFormulation_h
#define Camellia_AcousticsFormulation_h

#include "TypeDefs.h"

#include "VarFactory.h"
#include "BF.h"
#include "RHS.h"
#include "Solution.h"
#include "SpatiallyFilteredFunction.h"

namespace Camellia
{
class AcousticsFormulation
{
public:
  enum AcousticsFormulationChoice
  {
    // CONTINUOUS_GALERKIN,
    // PRIMAL,
    ULTRAWEAK
  };
private:
  VarFactoryPtr _vf;
  BFPtr _acousticsBF;
  RHSPtr _acousticsRHS;
  int _spaceDim;
  double _omega;

  static const string S_P1;
  static const string S_P2;
  static const string S_Q1;
  static const string S_Q2;

  static const string S_U1;
  static const string S_U2;
  static const string S_V1;
  static const string S_V2;

  static const string S_P1_HAT;
  static const string S_P2_HAT;
  static const string S_U1_N_HAT;
  static const string S_U2_N_HAT;
public:
  AcousticsFormulation(int spaceDim, bool useConformingTraces, AcousticsFormulationChoice formulationChoice=ULTRAWEAK, double angularFrequency=1.0);

  BFPtr bf();
  RHSPtr rhs();

  void CHECK_VALID_COMPONENT(int i) const; // throws exception on bad component value (should be either 1 or 2)

  // field variables:
  VarPtr p(int i);
  VarPtr u(int i);

  // traces:
  VarPtr p_hat(int i);
  VarPtr u_n_hat(int i);

  // test variables:
  VarPtr q(int i);
  VarPtr v(int i);

  // impedance BCs
  void addImpedanceBCs(SpatialFilterPtr boundary, FunctionPtr g1=Function::zero(), FunctionPtr g2=Function::zero());
};
}
#endif