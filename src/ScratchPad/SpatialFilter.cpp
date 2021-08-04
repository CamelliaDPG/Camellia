//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "SpatialFilter.h"

using namespace Camellia;

class SpatialFilterMatchingComponent : public SpatialFilter
{
  double _tol;
  int    _comp;
  double _compValueToMatch;
public:
  SpatialFilterMatchingComponent(int comp, double compValueToMatch, double tol=1e-14)
  :
  _tol(tol),
  _comp(comp),
  _compValueToMatch(compValueToMatch)
  {}

  using SpatialFilter::matchesPoints;
  virtual void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    int    comp             = _comp;
    double compValueToMatch = _compValueToMatch;
    double tol              = _tol;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & compValue = points(cellOrdinal,pointOrdinal,comp);
      pointsMatch(cellOrdinal,pointOrdinal) = (std::abs(compValue-compValueToMatch) < tol);
    });
  }
  
  using SpatialFilter::matchesPoint;
  virtual bool matchesPoint(double x) override
  {
    if (_comp == 0)
    {
      return (std::abs(x-_compValueToMatch) < _tol);
    }
    return false;
  }
  
  virtual bool matchesPoint(double x, double y) override
  {
    if (_comp == 1)
    {
      return (std::abs(y-_compValueToMatch) < _tol);
    }
    return matchesPoint(x);
  }
  
  virtual bool matchesPoint(double x, double y, double z) override
  {
    if (_comp == 2)
    {
      return (std::abs(z-_compValueToMatch) < _tol);
    }
    return matchesPoint(x,y);
  }
  
  virtual bool matchesPoint(double x, double y, double z, double t) override
  {
    if (_comp == 3)
    {
      return (std::abs(t-_compValueToMatch) < _tol);
    }
    return matchesPoint(x,y,z);
  }
  
};

class SpatialFilterInflow : public SpatialFilter
{
  std::vector<FunctionPtr> _convectionComponents;
public:
  SpatialFilterInflow(std::vector<FunctionPtr> convectionComponents)
  :
  _convectionComponents(convectionComponents)
  {}
  
  bool matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(basisCache->getSpaceDim() != _convectionComponents.size(), std::invalid_argument, "length of convectionComponents does not match spatial dimension")
    
    // TODO: Kokkos implementation
    const auto & normals = basisCache->getSideNormals(); // shape: C,P,D
    
    const int numCells  = normals.dimension(0);
    const int numPoints = normals.dimension(1);
    const int spaceDim  = normals.dimension(2);
    FieldContainer<double> convectionComponentValues(normals.dimension(0),normals.dimension(1)); // C,P
    FieldContainer<double> convectionDottedWithNormal(normals.dimension(0),normals.dimension(1)); // C,P
    for (int d=0; d<spaceDim; d++)
    {
      const auto & convectionComponent = _convectionComponents[d];
      convectionComponent->values(convectionComponentValues, basisCache);
      for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
        {
          const auto & convectionValue = convectionComponentValues(cellOrdinal,pointOrdinal);
          const auto & normalValue     = normals(cellOrdinal,pointOrdinal,d);
          convectionDottedWithNormal(cellOrdinal,pointOrdinal) += convectionValue * normalValue;
        }
      }
    }
    
    bool somePointMatches = false;
    for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal < numPoints; pointOrdinal++)
      {
        const auto & convectionDottedWithNormalValue = convectionDottedWithNormal(cellOrdinal,pointOrdinal);
        const bool matches = (convectionDottedWithNormalValue < 0);
        pointsMatch(cellOrdinal,pointOrdinal) = matches;
        if (matches) somePointMatches = true;
      }
    }
    return somePointMatches;
  }
  
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double>) override
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Not supported");
  }
};

class SpatialFilterMatchingT : public SpatialFilter
{
  double _tol;
  double _tToMatch;
  bool _matchSpatialSides;
public:
  SpatialFilterMatchingT(double tToMatch, bool matchSpatialSides, double tol=1e-14)
  {
    _tToMatch = tToMatch;
    _matchSpatialSides = matchSpatialSides;
    _tol = tol;
  }
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    double tToMatch  = _tToMatch;
    double tol       = _tol;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    const int tComp  = points.extent_int(3)-1;
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & t = points(cellOrdinal,pointOrdinal,tComp);
      pointsMatch(cellOrdinal,pointOrdinal) = (std::abs(t-tToMatch) < tol);
    });
  }
  using SpatialFilter::matchesPoint;
  bool matchesPoint(double x, double t) override
  {
    if (abs(t-_tToMatch)<_tol)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y, double t) override
  {
    if (abs(t-_tToMatch)<_tol)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y, double z, double t) override
  {
    if (abs(t-_tToMatch)<_tol)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesSpatialSides() override
  {
    return _matchSpatialSides;
  }
  bool matchesTemporalSides() override
  {
    return true;
  }
};

class SpatialFilterLessThanX : public SpatialFilter
{
  double _xToMatch;
public:
  SpatialFilterLessThanX(double xToMatch)
  {
    _xToMatch = xToMatch;
  }
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    double xToMatch = _xToMatch;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & x = points(cellOrdinal,pointOrdinal,0);
      pointsMatch(cellOrdinal,pointOrdinal) = (x<xToMatch);
    });
  }
  using SpatialFilter::matchesPoint;
  bool matchesPoint(double x) override
  {
    if (x<_xToMatch)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y) override
  {
    return matchesPoint(x);
  }
  bool matchesPoint(double x, double y, double z) override
  {
    return matchesPoint(x);
  }
};

class SpatialFilterLessThanY : public SpatialFilter
{
  double _yToMatch;
public:
  SpatialFilterLessThanY(double yToMatch)
  {
    _yToMatch = yToMatch;
  }
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    double yToMatch = _yToMatch;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & y = points(cellOrdinal,pointOrdinal,1);
      pointsMatch(cellOrdinal,pointOrdinal) = (y<yToMatch);
    });
  }
  using SpatialFilter::matchesPoint;
  bool matchesPoint(double x, double y) override
  {
    if (y<_yToMatch)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y, double z) override
  {
    return matchesPoint(x,y);
  }
};

class SpatialFilterLessThanZ : public SpatialFilter
{
  double _zToMatch;
public:
  SpatialFilterLessThanZ(double zToMatch)
  {
    _zToMatch = zToMatch;
  }
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    double zToMatch = _zToMatch;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & z = points(cellOrdinal,pointOrdinal,2);
      pointsMatch(cellOrdinal,pointOrdinal) = (z<zToMatch);
    });
  }
  using SpatialFilter::matchesPoint;
  bool matchesPoint(double x, double y, double z) override
  {
    if (z<_zToMatch)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

class SpatialFilterGreaterThanX : public SpatialFilter
{
  double _xToMatch;
public:
  SpatialFilterGreaterThanX(double xToMatch)
  {
    _xToMatch = xToMatch;
  }
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    double xToMatch = _xToMatch;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & x = points(cellOrdinal,pointOrdinal,0);
      pointsMatch(cellOrdinal,pointOrdinal) = (x>xToMatch);
    });
  }
  using SpatialFilter::matchesPoint;
  bool matchesPoint(double x) override
  {
    if (x>_xToMatch)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y) override
  {
    return matchesPoint(x);
  }
  bool matchesPoint(double x, double y, double z) override
  {
    return matchesPoint(x);
  }
};

class SpatialFilterGreaterThanY : public SpatialFilter
{
  double _yToMatch;
public:
  SpatialFilterGreaterThanY(double yToMatch)
  {
    _yToMatch = yToMatch;
  }
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    double yToMatch = _yToMatch;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & y = points(cellOrdinal,pointOrdinal,1);
      pointsMatch(cellOrdinal,pointOrdinal) = (y>yToMatch);
    });
  }
  using SpatialFilter::matchesPoint;
  bool matchesPoint(double x, double y) override
  {
    if (y>_yToMatch)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y, double z) override
  {
    return matchesPoint(x,y);
  }
};

class SpatialFilterGreaterThanZ : public SpatialFilter
{
  double _zToMatch;
public:
  SpatialFilterGreaterThanZ(double zToMatch)
  {
    _zToMatch = zToMatch;
  }
  using SpatialFilter::matchesPoints;
  void matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points) override
  {
    double zToMatch = _zToMatch;
    auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
    Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
    {
      const double & z = points(cellOrdinal,pointOrdinal,2);
      pointsMatch(cellOrdinal,pointOrdinal) = (z>zToMatch);
    });
  }
  using SpatialFilter::matchesPoint;
  bool matchesPoint(double x, double y, double z) override
  {
    if (z>_zToMatch)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
};

bool SpatialFilter::matchesSpatialSides()
{
  return true;
}

bool SpatialFilter::matchesTemporalSides()
{
  return false;
}

bool SpatialFilter::matchesPoint(double x)
{
  cout << "matchesPoint(x) unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x) unimplemented.");
//  return false;
}

bool SpatialFilter::matchesPoint(double x, double y)
{
  cout << "matchesPoint(x,y) unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y) unimplemented.");
//  return false;
}

bool SpatialFilter::matchesPoint(double x, double y, double z)
{
  cout << "matchesPoint(x,y,z) unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y,z) unimplemented.");
}

bool SpatialFilter::matchesPoint(double x, double y, double z, double t)
{
  cout << "matchesPoint(x,y,z,t) unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y,z,t) unimplemented.");
}

bool SpatialFilter::matchesPoint(double x0, double x1, double x2, double x3, double x4)
{
  cout << "matchesPoint() unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "5-argument matchesPoint() unimplemented.");
}

bool SpatialFilter::matchesPoint(double x0, double x1, double x2, double x3, double x4, double x5)
{
  cout << "matchesPoint() unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "6-argument matchesPoint() unimplemented.");
}

bool SpatialFilter::matchesPoint(double x0, double x1, double x2, double x3, double x4, double x5, double x6)
{
  cout << "matchesPoint() unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "7-argument matchesPoint() unimplemented.");
}

bool SpatialFilter::matchesPoint(const vector<double> &point)
{
  if (point.size() == 7)
  {
    return matchesPoint(point[0],point[1],point[2],point[3],point[4],point[5],point[6]);
  }
  else if (point.size() == 6)
  {
    return matchesPoint(point[0],point[1],point[2],point[3],point[4],point[5]);
  }
  else if (point.size() == 5)
  {
    return matchesPoint(point[0],point[1],point[2],point[3],point[4]);
  }
  else if (point.size() == 4)
  {
    return matchesPoint(point[0],point[1],point[2],point[3]);
  }
  else if (point.size() == 3)
  {
    return matchesPoint(point[0],point[1],point[2]);
  }
  else if (point.size() == 2)
  {
    return matchesPoint(point[0],point[1]);
  }
  else if (point.size() == 1)
  {
    return matchesPoint(point[0]);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "point is of unsupported dimension.");
//    return false;
  }
}

bool SpatialFilter::matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const Camellia::FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePointsFC());
  //    cout << "points:\n" << *points;
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  int spaceDim = points->dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(numCells != pointsMatch.dimension(0), std::invalid_argument, "numCells do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION(numPoints != pointsMatch.dimension(1), std::invalid_argument, "numPoints do not match.");
  pointsMatch.initialize(false);
  bool somePointMatches = false;
  
  if (basisCache->isSideCache())
  {
    // then check whether we have space-time cell topology
    // if so, make sure that the side type (temporal/spatial) is one we match
    if (basisCache->cellTopologyIsSpaceTime())
    {
      int sideOrdinal = basisCache->getSideIndex();
      int sideIsSpatial = basisCache->cellTopology()->sideIsSpatial(sideOrdinal);
      if (!matchesSpatialSides() && sideIsSpatial) return false; // no match
      if (!matchesTemporalSides() && !sideIsSpatial) return false; // no match
    }
  }
  
  vector<double> point(spaceDim);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        point[d] = (*points)(cellIndex,ptIndex,d);
      }
      if (matchesPoint(point))
      {
        somePointMatches = true;
        pointsMatch(cellIndex,ptIndex) = true;
      }
    }
  }
  return somePointMatches;
}

SpatialFilterPtr SpatialFilter::allSpace()
{
  return Teuchos::rcp( new SpatialFilterUnfiltered );
}

SpatialFilterPtr SpatialFilter::inflowFilter(std::vector<FunctionPtr> convectionComponents)
{
  return Teuchos::rcp( new SpatialFilterInflow(convectionComponents) );
}

SpatialFilterPtr SpatialFilter::unionFilter(SpatialFilterPtr a, SpatialFilterPtr b)
{
  return Teuchos::rcp( new SpatialFilterLogicalOr(a,b) );
}

SpatialFilterPtr SpatialFilter::intersectionFilter(SpatialFilterPtr a, SpatialFilterPtr b)
{
  return Teuchos::rcp( new SpatialFilterLogicalAnd(a,b) );
}

SpatialFilterPtr SpatialFilter::negatedFilter(SpatialFilterPtr filterToNegate)
{
  return Teuchos::rcp( new NegatedSpatialFilter(filterToNegate) );
}

void SpatialFilterUnfiltered::matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points)
{
  Kokkos::deep_copy(pointsMatch, true);
}

bool SpatialFilterUnfiltered::matchesPoint(double x)
{
  return true;
}

bool SpatialFilterUnfiltered::matchesPoint(double x, double y)
{
  return true;
}

bool SpatialFilterUnfiltered::matchesPoint(double x, double y, double z)
{
  return true;
}

bool SpatialFilterUnfiltered::matchesPoint(double x, double y, double z, double t)
{
  return true;
}

bool SpatialFilterUnfiltered::matchesPoint(const vector<double> &point)
{
  return true;
}

SpatialFilterPtr SpatialFilter::matchingComponent(int comp, double compValue)
{
  return Teuchos::rcp( new SpatialFilterMatchingComponent(comp,compValue) );
}

SpatialFilterPtr SpatialFilter::matchingX(double x)
{
  return Teuchos::rcp( new SpatialFilterMatchingComponent(0,x) );
}

SpatialFilterPtr SpatialFilter::matchingY(double y)
{
  return Teuchos::rcp( new SpatialFilterMatchingComponent(1,y) );
}

SpatialFilterPtr SpatialFilter::matchingZ(double z)
{
  return Teuchos::rcp( new SpatialFilterMatchingComponent(2,z) );
}

SpatialFilterPtr SpatialFilter::matchingT(double t, bool matchSpatialSides)
{
  return Teuchos::rcp( new SpatialFilterMatchingT(t, matchSpatialSides) );
}

SpatialFilterPtr SpatialFilter::lessThanOrEqualToX(double x)
{
  return SpatialFilter::lessThanX(x) | SpatialFilter::matchingX(x);
}

SpatialFilterPtr SpatialFilter::lessThanOrEqualToY(double y)
{
  return SpatialFilter::lessThanY(y) | SpatialFilter::matchingY(y);
}

SpatialFilterPtr SpatialFilter::lessThanOrEqualToZ(double z)
{
  return SpatialFilter::lessThanZ(z) | SpatialFilter::matchingZ(z);
}

SpatialFilterPtr SpatialFilter::lessThanX(double x)
{
  return Teuchos::rcp( new SpatialFilterLessThanX(x) );
}

SpatialFilterPtr SpatialFilter::lessThanY(double y)
{
  return Teuchos::rcp( new SpatialFilterLessThanY(y) );
}

SpatialFilterPtr SpatialFilter::lessThanZ(double z)
{
  return Teuchos::rcp( new SpatialFilterLessThanZ(z) );
}

SpatialFilterPtr SpatialFilter::greaterThanX(double x)
{
  return Teuchos::rcp( new SpatialFilterGreaterThanX(x) );
}

SpatialFilterPtr SpatialFilter::greaterThanY(double y)
{
  return Teuchos::rcp( new SpatialFilterGreaterThanY(y) );
}

SpatialFilterPtr SpatialFilter::greaterThanZ(double z)
{
  return Teuchos::rcp( new SpatialFilterGreaterThanZ(z) );
}

SpatialFilterPtr SpatialFilter::greaterThanOrEqualToX(double x)
{
  return SpatialFilter::greaterThanX(x) | SpatialFilter::matchingX(x);
}

SpatialFilterPtr SpatialFilter::greaterThanOrEqualToY(double y)
{
  return SpatialFilter::greaterThanY(y) | SpatialFilter::matchingY(y);
}

SpatialFilterPtr SpatialFilter::greaterThanOrEqualToZ(double z)
{
  return SpatialFilter::greaterThanZ(z) | SpatialFilter::matchingZ(z);
}

SpatialFilterLogicalOr::SpatialFilterLogicalOr(SpatialFilterPtr sf1, SpatialFilterPtr sf2)
{
  _sf1 = sf1;
  _sf2 = sf2;
}

bool SpatialFilterLogicalOr::matchesPoint(double x)
{
  return _sf1->matchesPoint(x) || _sf2->matchesPoint(x);
}

bool SpatialFilterLogicalOr::matchesPoint(double x, double y)
{
  return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
}

bool SpatialFilterLogicalOr::matchesPoint(double x, double y, double z)
{
  return _sf1->matchesPoint(x,y,z) || _sf2->matchesPoint(x,y,z);
}

bool SpatialFilterLogicalOr::matchesPoint(double x, double y, double z, double t)
{
  return _sf1->matchesPoint(x,y,z,t) || _sf2->matchesPoint(x,y,z,t);
}

void SpatialFilterLogicalOr::matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points)
{
  auto pointsMatch2 = SharedScratchStorage<bool>::sharedScratchStorage().checkoutScratchStorageMatchingDeviceView(pointsMatch, "SpatialFilterLogicalOr::matchesPoints: pointsMatch2");
  Kokkos::deep_copy(pointsMatch2,false);
  
  _sf1->matchesPoints(pointsMatch, points);
  _sf2->matchesPoints(pointsMatch2, points);
  
  Kokkos::fence();
  
  auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
  Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
  {
    pointsMatch(cellOrdinal,pointOrdinal) = pointsMatch(cellOrdinal,pointOrdinal) || pointsMatch2(cellOrdinal,pointOrdinal);
  });
  SharedScratchStorage<bool>::sharedScratchStorage().checkinScratchStorage(pointsMatch2);
}

bool SpatialFilterLogicalOr::matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const Camellia::FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePointsFC());
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  Camellia::FieldContainer<bool> pointsMatch2(pointsMatch);
  bool somePointMatches1 = _sf1->matchesPoints(pointsMatch,basisCache);
  bool somePointMatches2 = _sf2->matchesPoints(pointsMatch2,basisCache);
  if ( !somePointMatches2 )
  {
    // then what's in pointsMatch is exactly right
    return somePointMatches1;
  }
  else if ( !somePointMatches1 )
  {
    // then what's in pointsMatch2 is exactly right
    pointsMatch = pointsMatch2;
    return somePointMatches2;
  }
  else
  {
    // need to combine them:
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        pointsMatch(cellIndex,ptIndex) |= pointsMatch2(cellIndex,ptIndex);
      }
    }
    // if we're here, then some point matched: return true:
    return true;
  }
}

bool SpatialFilterLogicalOr::matchesSpatialSides()
{
  return _sf1->matchesSpatialSides() || _sf2->matchesSpatialSides();
}

bool SpatialFilterLogicalOr::matchesTemporalSides()
{
  return _sf1->matchesTemporalSides() || _sf2->matchesTemporalSides();
}

SpatialFilterLogicalAnd::SpatialFilterLogicalAnd(SpatialFilterPtr sf1, SpatialFilterPtr sf2)
{
  _sf1 = sf1;
  _sf2 = sf2;
}

void SpatialFilterLogicalAnd::matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points)
{
  auto pointsMatch2 = SharedScratchStorage<bool>::sharedScratchStorage().checkoutScratchStorageMatchingDeviceView(pointsMatch, "SpatialFilterLogicalAnd::matchesPoints(): pointsMatch2");
  Kokkos::deep_copy(pointsMatch2,false);
  
  _sf1->matchesPoints(pointsMatch, points);
  _sf2->matchesPoints(pointsMatch2, points);
  
  Kokkos::fence();
  
  auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
  Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
  {
    pointsMatch(cellOrdinal,pointOrdinal) = pointsMatch(cellOrdinal,pointOrdinal) && pointsMatch2(cellOrdinal,pointOrdinal);
  });
  SharedScratchStorage<bool>::sharedScratchStorage().checkinScratchStorage(pointsMatch2);
}

bool SpatialFilterLogicalAnd::matchesPoint(double x)
{
  return _sf1->matchesPoint(x) && _sf2->matchesPoint(x);
}

bool SpatialFilterLogicalAnd::matchesPoint(double x, double y)
{
  return _sf1->matchesPoint(x,y) && _sf2->matchesPoint(x,y);
}

bool SpatialFilterLogicalAnd::matchesPoint(double x, double y, double z)
{
  return _sf1->matchesPoint(x,y,z) && _sf2->matchesPoint(x,y,z);
}

bool SpatialFilterLogicalAnd::matchesPoint(double x, double y, double z, double t)
{
  return _sf1->matchesPoint(x,y,z,t) && _sf2->matchesPoint(x,y,z,t);
}

bool SpatialFilterLogicalAnd::matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const Camellia::FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePointsFC());
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  Camellia::FieldContainer<bool> pointsMatch1(pointsMatch);
  Camellia::FieldContainer<bool> pointsMatch2(pointsMatch);
  bool somePointMatches1 = _sf1->matchesPoints(pointsMatch1,basisCache);
  bool somePointMatches2 = _sf2->matchesPoints(pointsMatch2,basisCache);
  bool samePointsMatch = false;
  if (somePointMatches1 && somePointMatches2)
  {
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        if (pointsMatch1(cellIndex,ptIndex) && pointsMatch2(cellIndex,ptIndex))
        {
          pointsMatch(cellIndex,ptIndex) = true;
          samePointsMatch = true;
        }
        else
          pointsMatch(cellIndex,ptIndex) = false;
      }
    }
  }
  return samePointsMatch;
}

bool SpatialFilterLogicalAnd::matchesSpatialSides()
{
  return _sf1->matchesSpatialSides() && _sf2->matchesSpatialSides();
}

bool SpatialFilterLogicalAnd::matchesTemporalSides()
{
  return _sf1->matchesTemporalSides() && _sf2->matchesTemporalSides();
}

NegatedSpatialFilter::NegatedSpatialFilter(SpatialFilterPtr filterToNegate)
{
  _filterToNegate = filterToNegate;
}

void NegatedSpatialFilter::matchesPoints(DeviceView<bool> pointsMatch, DeviceViewConst<double> points)
{
  _filterToNegate->matchesPoints(pointsMatch, points);
  
  Kokkos::fence();
  
  auto rangePolicy = Camellia::FieldContainer<bool>::getMDRangePolicy2(pointsMatch);
  Kokkos::parallel_for(rangePolicy, KOKKOS_LAMBDA(const int cellOrdinal, const int pointOrdinal)
  {
    pointsMatch(cellOrdinal,pointOrdinal) = ! pointsMatch(cellOrdinal,pointOrdinal);
  });
}

bool NegatedSpatialFilter::matchesPoint(double x)
{
  return ! _filterToNegate->matchesPoint(x);
}

bool NegatedSpatialFilter::matchesPoint(double x, double y)
{
  return ! _filterToNegate->matchesPoint(x,y);
}

bool NegatedSpatialFilter::matchesPoint(double x, double y, double z)
{
  return ! _filterToNegate->matchesPoint(x,y,z);
}

bool NegatedSpatialFilter::matchesPoint(double x, double y, double z, double t)
{
  return ! _filterToNegate->matchesPoint(x,y,z,t);
}

bool NegatedSpatialFilter::matchesPoints(Camellia::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const Camellia::FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePointsFC());
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  _filterToNegate->matchesPoints(pointsMatch,basisCache);
  bool somePointMatches = false;
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      pointsMatch(cellIndex,ptIndex) = ! pointsMatch(cellIndex,ptIndex);
      somePointMatches |= pointsMatch(cellIndex,ptIndex);
    }
  }
  return somePointMatches;
}

bool NegatedSpatialFilter::matchesSpatialSides()
{
  return _filterToNegate->matchesSpatialSides();
}

bool NegatedSpatialFilter::matchesTemporalSides()
{
  return _filterToNegate->matchesTemporalSides();
}

namespace Camellia
{
SpatialFilterPtr operator!(SpatialFilterPtr sf)
{
  return SpatialFilter::negatedFilter(sf);
}

SpatialFilterPtr operator|(SpatialFilterPtr sf1, SpatialFilterPtr sf2)
{
  return SpatialFilter::unionFilter(sf1, sf2);
}

SpatialFilterPtr operator&(SpatialFilterPtr sf1, SpatialFilterPtr sf2)
{
  return SpatialFilter::intersectionFilter(sf1, sf2);
}
}
