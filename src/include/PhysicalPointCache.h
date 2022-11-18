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
  Camellia::FieldContainer<double> _physCubPointsFC;
  PhysicalTensorPoints _physicalCubaturePoints;
public:
  PhysicalPointCache(const Camellia::FieldContainer<double> &physCubPoints);
  const PhysicalTensorPoints & getPhysicalCubaturePoints() override;
  const Camellia::FieldContainer<double> & getPhysicalCubaturePointsFC() override;
  int getSpaceDim() override; // overrides BasisCache::getSpaceDim();
};
}

#endif
