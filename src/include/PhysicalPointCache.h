// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  PhysicalPointCache.h
//  Camellia-debug
//
//  Created by Nate Roberts on 6/6/14.
//
//

#ifndef Camellia_debug_PhysicalPointCache_h
#define Camellia_debug_PhysicalPointCache_h

#include "BasisCache.h"

namespace Camellia
{
class PhysicalPointCache : public BasisCache
{
  Camellia::FieldContainer<double> _physCubPoints;
public:
  PhysicalPointCache(const Camellia::FieldContainer<double> &physCubPoints);
  const Camellia::FieldContainer<double> & getPhysicalCubaturePointsFC() override;
  Camellia::FieldContainer<double> & writablePhysicalCubaturePoints();
  int getSpaceDim() override; // overrides BasisCache::getSpaceDim();
};
}

#endif
