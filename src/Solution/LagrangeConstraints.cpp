//
//  LagrangeConstraints.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/23/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "LagrangeConstraints.h"

void LagrangeConstraints::addConstraint(const Constraint &c) {
  _constraints.push_back(c);
}

void LagrangeConstraints::addGlobalConstraint(const Constraint &c) {
  _globalConstraints.push_back(c);
}

int LagrangeConstraints::numElementConstraints() {
  return _constraints.size();
}

int LagrangeConstraints::numGlobalConstraints() {
  return _globalConstraints.size();
}

// void LagrangeConstraints::addConstraint(const Constraint &c, SpatialFilterPtr sf) {
//  Constraint sfc = Constraint::spatiallyFilteredConstraint(c,sf);
//  _constraints.push_back(sfc);
// }