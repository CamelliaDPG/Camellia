//
//  SpatialFilter.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/28/13.
//
//

#include "SpatialFilter.h"

using namespace Intrepid;
using namespace Camellia;

class SpatialFilterMatchingX : public SpatialFilter
{
  double _tol;
  double _xToMatch;
public:
  SpatialFilterMatchingX(double xToMatch, double tol=1e-14)
  {
    _xToMatch = xToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x)
  {
    if (abs(x-_xToMatch)<_tol)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y)
  {
    return matchesPoint(x);
  }
  bool matchesPoint(double x, double y, double z)
  {
    return matchesPoint(x);
  }
};

class SpatialFilterMatchingY : public SpatialFilter
{
  double _tol;
  double _yToMatch;
public:
  SpatialFilterMatchingY(double yToMatch, double tol=1e-14)
  {
    _yToMatch = yToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x, double y)
  {
    if (abs(y-_yToMatch)<_tol)
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  bool matchesPoint(double x, double y, double z)
  {
    return matchesPoint(x,y);
  }
};

class SpatialFilterMatchingZ : public SpatialFilter
{
  double _tol;
  double _zToMatch;
public:
  SpatialFilterMatchingZ(double zToMatch, double tol=1e-14)
  {
    _zToMatch = zToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x, double y, double z)
  {
    if (abs(z-_zToMatch)<_tol)
    {
      return true;
    }
    else
    {
      return false;
    }
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
  bool matchesPoint(double x, double t)
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
  bool matchesPoint(double x, double y, double t)
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
  bool matchesPoint(double x, double y, double z, double t)
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
  bool matchesSpatialSides()
  {
    return _matchSpatialSides;
  }
  bool matchesTemporalSides()
  {
    return true;
  }
};

class SpatialFilterLessThanX : public SpatialFilter
{
  double _tol;
  double _xToMatch;
public:
  SpatialFilterLessThanX(double xToMatch, double tol=1e-14)
  {
    _xToMatch = xToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x)
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
  bool matchesPoint(double x, double y)
  {
    return matchesPoint(x);
  }
  bool matchesPoint(double x, double y, double z)
  {
    return matchesPoint(x);
  }
};

class SpatialFilterLessThanY : public SpatialFilter
{
  double _tol;
  double _yToMatch;
public:
  SpatialFilterLessThanY(double yToMatch, double tol=1e-14)
  {
    _yToMatch = yToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x, double y)
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
  bool matchesPoint(double x, double y, double z)
  {
    return matchesPoint(x,y);
  }
};

class SpatialFilterLessThanZ : public SpatialFilter
{
  double _tol;
  double _zToMatch;
public:
  SpatialFilterLessThanZ(double zToMatch, double tol=1e-14)
  {
    _zToMatch = zToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x, double y, double z)
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
  double _tol;
  double _xToMatch;
public:
  SpatialFilterGreaterThanX(double xToMatch, double tol=1e-14)
  {
    _xToMatch = xToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x)
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
  bool matchesPoint(double x, double y)
  {
    return matchesPoint(x);
  }
  bool matchesPoint(double x, double y, double z)
  {
    return matchesPoint(x);
  }
};

class SpatialFilterGreaterThanY : public SpatialFilter
{
  double _tol;
  double _yToMatch;
public:
  SpatialFilterGreaterThanY(double yToMatch, double tol=1e-14)
  {
    _yToMatch = yToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x, double y)
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
  bool matchesPoint(double x, double y, double z)
  {
    return matchesPoint(x,y);
  }
};

class SpatialFilterGreaterThanZ : public SpatialFilter
{
  double _tol;
  double _zToMatch;
public:
  SpatialFilterGreaterThanZ(double zToMatch, double tol=1e-14)
  {
    _zToMatch = zToMatch;
    _tol = tol;
  }
  bool matchesPoint(double x, double y, double z)
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
  return false;
}

bool SpatialFilter::matchesPoint(double x, double y)
{
  cout << "matchesPoint(x,y) unimplemented.\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y) unimplemented.");
  return false;
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

bool SpatialFilter::matchesPoint(const vector<double> &point)
{
  if (point.size() == 4)
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
    return false;
  }
}

bool SpatialFilter::matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
  //    cout << "points:\n" << *points;
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  int spaceDim = points->dimension(2);
  TEUCHOS_TEST_FOR_EXCEPTION(numCells != pointsMatch.dimension(0), std::invalid_argument, "numCells do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION(numPoints != pointsMatch.dimension(1), std::invalid_argument, "numPoints do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > 3, std::invalid_argument, "matchesPoints supports 1D, 2D, and 3D only.");
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
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      vector<double> point;
      for (int d=0; d<spaceDim; d++)
      {
        point.push_back((*points)(cellIndex,ptIndex,d));
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

bool SpatialFilterUnfiltered::matchesPoint(vector<double> &point)
{
  return true;
}

SpatialFilterPtr SpatialFilter::matchingX(double x)
{
  return Teuchos::rcp( new SpatialFilterMatchingX(x) );
}

SpatialFilterPtr SpatialFilter::matchingY(double y)
{
  return Teuchos::rcp( new SpatialFilterMatchingY(y) );
}

SpatialFilterPtr SpatialFilter::matchingZ(double z)
{
  return Teuchos::rcp( new SpatialFilterMatchingZ(z) );
}

SpatialFilterPtr SpatialFilter::matchingT(double t, bool matchSpatialSides)
{
  return Teuchos::rcp( new SpatialFilterMatchingT(t, matchSpatialSides) );
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

bool SpatialFilterLogicalOr::matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  FieldContainer<bool> pointsMatch2(pointsMatch);
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

bool SpatialFilterLogicalAnd::matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
  int numCells = points->dimension(0);
  int numPoints = points->dimension(1);
  FieldContainer<bool> pointsMatch1(pointsMatch);
  FieldContainer<bool> pointsMatch2(pointsMatch);
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

bool NegatedSpatialFilter::matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache)
{
  const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
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