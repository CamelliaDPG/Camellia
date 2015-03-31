//
//  LagrangeConstraints.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/23/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//


#include "LagrangeConstraints.h"

#include "CamelliaCellTools.h"

using namespace Intrepid;
using namespace Camellia;

void LagrangeConstraints::addConstraint(const Constraint &c) {
  _constraints.push_back(c);
}

void LagrangeConstraints::addGlobalConstraint(const Constraint &c) {
  _globalConstraints.push_back(c);
}

void LagrangeConstraints::getCoefficients(FieldContainer<double> &lhs, FieldContainer<double> &rhs,
                                          int elemConstraintIndex, DofOrderingPtr trialOrdering, 
                                          BasisCachePtr basisCache) {
  LinearTermPtr lt = _constraints[elemConstraintIndex].linearTerm();
  FunctionPtr f = _constraints[elemConstraintIndex].f();
  lt->integrate(lhs, trialOrdering, basisCache);
  bool onBoundary = f->boundaryValueOnly();
  if ( !onBoundary ) {
    f->integrate(rhs, basisCache);
  } else {
    int numSides = basisCache->cellTopology()->getSideCount();
    rhs.initialize(0);
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      f->integrate(rhs, basisCache->getSideBasisCache(sideIndex), true); // true: sumInto
    }
  }
}

int LagrangeConstraints::numElementConstraints() {
  return _constraints.size();
}

int LagrangeConstraints::numGlobalConstraints() {
  return _globalConstraints.size();
}

void LagrangeConstraints::addConstraint(const Constraint &c, SpatialFilterPtr sf) {
  Constraint sfc = Constraint::spatiallyFilteredConstraint(c,sf);
  _constraints.push_back(sfc);
}

Constraint & LagrangeConstraints::getElementConstraint(int constraintOrdinal) {
  return _constraints[constraintOrdinal];
}

Constraint & LagrangeConstraints::getGlobalConstraint(int constraintOrdinal) {
  return _globalConstraints[constraintOrdinal];
}