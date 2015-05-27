//
//  LagrangeConstraints.h
//  Camellia
//
//  Created by Nathan Roberts on 4/23/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_LagrangeConstraints_h
#define Camellia_LagrangeConstraints_h

#include "TypeDefs.h"

#include "SpatialFilter.h"
#include "Constraint.h"

namespace Camellia
{
class LagrangeConstraints
{
private:
  vector< Constraint > _constraints;
  vector< Constraint > _globalConstraints;
public:
  void addConstraint(const Constraint &c);
  void addConstraint(const Constraint &c, SpatialFilterPtr sf);
  void addGlobalConstraint(const Constraint &c);
  void getCoefficients(Intrepid::FieldContainer<double> &lhs, Intrepid::FieldContainer<double> &rhs,
                       int elemConstraintIndex, DofOrderingPtr trialOrdering,
                       BasisCachePtr basisCache);
  int numElementConstraints();
  int numGlobalConstraints();

  Constraint & getElementConstraint(int constraintOrdinal);
  Constraint & getGlobalConstraint(int constraintOrdinal);
};
}

#endif
