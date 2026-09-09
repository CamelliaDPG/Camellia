// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  LobattoHGRAD_QuadBasis.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/3/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_LobattoHGRAD_QuadBasis_h
#define Camellia_debug_LobattoHGRAD_QuadBasis_h

#include "Basis.h"
#include "Camellia_FieldContainer.hpp"

namespace Camellia
{
template<class Scalar=double, class ArrayScalar=Camellia::FieldContainer<Scalar> > class LobattoHGRAD_QuadBasis;
template<class Scalar, class ArrayScalar> class LobattoHGRAD_QuadBasis : public Basis<Scalar,ArrayScalar>
{
protected:
  void initializeTags() const;
  int _degree_x, _degree_y;
  bool _conforming;

  Camellia::FieldContainer<double> _legendreL2normsSquared, _lobattoL2normsSquared;
  void initializeL2normValues();
  int dofOrdinalMap(int xDofOrdinal, int yDofOrdinal) const;
public:
  LobattoHGRAD_QuadBasis(int degree, bool conforming = false); // conforming means not strictly hierarchical, but has e.g. vertex dofs defined...
  LobattoHGRAD_QuadBasis(int degree_x, int degree_y, bool conforming = false);

  using Basis<Scalar,ArrayScalar>::getValues;
  void getValues(ArrayScalar &values, const ArrayScalar &refPoints, Camellia::EOperator operatorType) const;
};
}

#include "LobattoHGRAD_QuadBasisDef.h"

#endif
