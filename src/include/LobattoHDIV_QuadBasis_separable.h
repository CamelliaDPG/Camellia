// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  LobattoHDIV_QuadBasis_separable.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LobattoHDIV_QuadBasis_separable_h
#define Camellia_debug_LobattoHDIV_QuadBasis_separable_h

#include "Basis.h"

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=Intrepid::FieldContainer<double> > class LobattoHDIV_QuadBasis_separable;
template<class Scalar, class ArrayScalar> class LobattoHDIV_QuadBasis_separable : public Basis<Scalar,ArrayScalar>
{
protected:
  void initializeTags() const;
  int _degree_x, _degree_y;
  bool _conforming;
  bool _onlyDivFreeFunctions;

  Camellia::FieldContainer<double> _legendreL2normsSquared, _lobattoL2normsSquared;
  void initializeL2normValues();
  int dofOrdinalMap(int xDofOrdinal, int yDofOrdinal, bool divFree) const;
public:
  LobattoHDIV_QuadBasis_separable(int degree, bool conforming = false, bool onlyDivFreeFunctions = false); // conforming means not strictly hierarchical, but has e.g. vertex dofs defined...
  LobattoHDIV_QuadBasis_separable(int degree_x, int degree_y, bool conforming = false, bool onlyDivFreeFunctions = false);

  using Basis<Scalar,ArrayScalar>::getValues;
  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Camellia::EOperator operatorType) const;
};
}

#include "LobattoHDIV_QuadBasis_separableDef.h"

#endif
