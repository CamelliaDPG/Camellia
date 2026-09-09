// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  LobattoHGRAD_LineBasis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LobattoHGRAD_LineBasis_h
#define Camellia_debug_LobattoHGRAD_LineBasis_h

#include "Basis.h"
#include "Camellia_FieldContainer.hpp"

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=Camellia::FieldContainer<Scalar> > class LobattoHGRAD_LineBasis;
template<class Scalar, class ArrayScalar> class LobattoHGRAD_LineBasis : public Basis<Scalar,ArrayScalar>
{
protected:
  void initializeTags() const;
  int _degree;
  bool _conforming;

  Camellia::FieldContainer<double> _lobattoL2norms;
  void initializeL2normValues();
public:
  LobattoHGRAD_LineBasis(int degree, bool conforming); // conforming means vertex dofs defined...

  using Basis<Scalar,ArrayScalar>::getValues;
  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Camellia::EOperator operatorType) const;

  bool isConforming() const;
};
}

#include "LobattoHGRAD_LineBasisDef.h"

#endif
