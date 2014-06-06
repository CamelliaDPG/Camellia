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
  FieldContainer<double> _physCubPoints;
public:
  PhysicalPointCache(const FieldContainer<double> &physCubPoints);
  const FieldContainer<double> & getPhysicalCubaturePoints();
  FieldContainer<double> & writablePhysicalCubaturePoints();
};

#endif
