//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  PhysicalPointCache.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 6/6/14.
//
//

#include "PhysicalPointCache.h"

using namespace Camellia;

PhysicalPointCache::PhysicalPointCache(const Camellia::FieldContainer<double> &physCubPoints) : BasisCache()
{
  _physCubPointsFC = physCubPoints;
  
  // copy the physical points into a single PhysicalTensorPoints component
  _numCells = _physCubPointsFC.dimension(0);
  const int numPoints    = _physCubPointsFC.dimension(1);
  const int numCoordDims = _physCubPointsFC.dimension(2);
  Kokkos::DynRankView<double, DeviceType> physPointsComponent("physical points component",_numCells,numPoints,numCoordDims);
  auto physCubPointsView = _physCubPointsFC.checkoutDeviceView();
  Kokkos::deep_copy(physPointsComponent, physCubPointsView);
  vector< Kokkos::DynRankView<double, DeviceType> > physicalTensorPointsComponents {physPointsComponent};
  _physCubPointsFC.checkinDeviceView(false);
  _physicalCubaturePoints = PhysicalTensorPoints(physicalTensorPointsComponents);
}

const PhysicalTensorPoints & PhysicalPointCache::getPhysicalCubaturePoints()
{
  return _physicalCubaturePoints;
}

const Camellia::FieldContainer<double> & PhysicalPointCache::getPhysicalCubaturePointsFC()   // overrides super
{
  return _physCubPointsFC;
}

int PhysicalPointCache::getSpaceDim()
{
  return _physCubPointsFC.dimension(2);
}
