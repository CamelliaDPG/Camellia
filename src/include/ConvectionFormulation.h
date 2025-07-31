// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER
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
#include "BasisFamily.h"
#include "BF.h"

namespace Camellia
{
class ConvectionFormulation
{
  BFPtr _convectionBF;
  int _spaceDim;

  static const string S_U;

  static const string S_Q_N_HAT;

  static const string S_V;
public:
  ConvectionFormulation(int spaceDim, TFunctionPtr<double> convectiveFunction,
                        BasisFamilyPtr basisFamily = BasisFamily::hierarchicalBasisFamily());

  BFPtr bf();

  // field variables:
  VarPtr u();

  // traces:
  VarPtr q_n_hat();

  // test variables:
  VarPtr v();
};
}


#endif
