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

class PhysicalPointCache : public BasisCache {
  Intrepid::FieldContainer<double> _physCubPoints;
public:
  PhysicalPointCache(const Intrepid::FieldContainer<double> &physCubPoints);
  const Intrepid::FieldContainer<double> & getPhysicalCubaturePoints();
  Intrepid::FieldContainer<double> & writablePhysicalCubaturePoints();
  int getSpaceDim(); // overrides BasisCache::getSpaceDim();
};

#endif
