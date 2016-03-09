//
//  SpatialFilter.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SpatialFilter_h
#define Camellia_SpatialFilter_h

#include "TypeDefs.h"

#include "BasisCache.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "Intrepid_FieldContainer.hpp"

namespace Camellia
{
class SpatialFilter
{
public:
  virtual bool matchesPoint(double x);
  virtual bool matchesPoint(double x, double y);
  virtual bool matchesPoint(double x, double y, double z);
  virtual bool matchesPoint(double x, double y, double z, double t);
  virtual bool matchesPoint(const vector<double>&point);
  virtual bool matchesPoints(Intrepid::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);
  
  virtual bool matchesSpatialSides(); // default: true
  virtual bool matchesTemporalSides(); // default: false

  static SpatialFilterPtr allSpace();
  static SpatialFilterPtr unionFilter(SpatialFilterPtr a, SpatialFilterPtr b);
  static SpatialFilterPtr intersectionFilter(SpatialFilterPtr a, SpatialFilterPtr b);

  static SpatialFilterPtr negatedFilter(SpatialFilterPtr filterToNegate);

  static SpatialFilterPtr matchingX(double x);
  static SpatialFilterPtr matchingY(double y);
  static SpatialFilterPtr matchingZ(double z);
  
  static SpatialFilterPtr matchingT(double t, bool matchSpatialSides = false); // matchesSpatialSides() is false; matchesTemporalSides() is true

  static SpatialFilterPtr lessThanX(double x);
  static SpatialFilterPtr lessThanY(double y);
  static SpatialFilterPtr lessThanZ(double z);

  static SpatialFilterPtr greaterThanX(double x);
  static SpatialFilterPtr greaterThanY(double y);
  static SpatialFilterPtr greaterThanZ(double z);

  virtual ~SpatialFilter() {}
};

class SpatialFilterUnfiltered : public SpatialFilter
{
  bool matchesPoint(vector<double> &point);

  virtual bool matchesPoint(double x);
  virtual bool matchesPoint(double x, double y);
  virtual bool matchesPoint(double x, double y, double z);
  virtual bool matchesPoint(double x, double y, double z, double t);
};

class SpatialFilterLogicalOr : public SpatialFilter
{
  SpatialFilterPtr _sf1, _sf2;
public:
  SpatialFilterLogicalOr(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
  bool matchesPoints(Intrepid::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);

  //  bool matchesPoint( double x, double y ) {
  //    return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
  //  }
  virtual bool matchesPoint(double x);
  virtual bool matchesPoint(double x, double y);
  virtual bool matchesPoint(double x, double y, double z);
  virtual bool matchesPoint(double x, double y, double z, double t);
  
  virtual bool matchesSpatialSides();
  virtual bool matchesTemporalSides();
};

class SpatialFilterLogicalAnd : public SpatialFilter
{
  SpatialFilterPtr _sf1, _sf2;
public:
  SpatialFilterLogicalAnd(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
  bool matchesPoints(Intrepid::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);

  //  bool matchesPoint( double x, double y ) {
  //    return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
  //  }
  virtual bool matchesPoint(double x);
  virtual bool matchesPoint(double x, double y);
  virtual bool matchesPoint(double x, double y, double z);
  virtual bool matchesPoint(double x, double y, double z, double t);
  
  virtual bool matchesSpatialSides();
  virtual bool matchesTemporalSides();
};

class NegatedSpatialFilter : public SpatialFilter
{
  SpatialFilterPtr _filterToNegate;
public:
  NegatedSpatialFilter(SpatialFilterPtr FilterToNegate);
  bool matchesPoints(Intrepid::FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);

  virtual bool matchesPoint(double x);
  virtual bool matchesPoint(double x, double y);
  virtual bool matchesPoint(double x, double y, double z);
  virtual bool matchesPoint(double x, double y, double z, double t);
  
  // ! Returns the same result as _filterToNegate; the negation is understood to happen in space or time, not both
  virtual bool matchesSpatialSides();
  
  // ! Returns the same result as _filterToNegate; the negation is understood to happen in space or time, not both
  virtual bool matchesTemporalSides();
};

SpatialFilterPtr operator!(SpatialFilterPtr sf);
SpatialFilterPtr operator|(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
SpatialFilterPtr operator&(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
}

#endif
