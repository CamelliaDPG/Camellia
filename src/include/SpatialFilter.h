//
//  SpatialFilter.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SpatialFilter_h
#define Camellia_SpatialFilter_h

#include "BasisCache.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"
#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;

class SpatialFilter;
typedef Teuchos::RCP< SpatialFilter > SpatialFilterPtr;

class SpatialFilter {
public:
  // just 2D for now:
  virtual bool matchesPoint(double x);
  virtual bool matchesPoint(double x, double y);
  virtual bool matchesPoint(double x, double y, double z);
  virtual bool matchesPoint(vector<double>&point);
  virtual bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);

  static SpatialFilterPtr allSpace();
  static SpatialFilterPtr unionFilter(SpatialFilterPtr a, SpatialFilterPtr b);
  static SpatialFilterPtr intersectionFilter(SpatialFilterPtr a, SpatialFilterPtr b);

  static SpatialFilterPtr negatedFilter(SpatialFilterPtr filterToNegate);

  static SpatialFilterPtr matchingX(double x);
  static SpatialFilterPtr matchingY(double y);
  static SpatialFilterPtr matchingZ(double z);

  static SpatialFilterPtr lessThanX(double x);
  static SpatialFilterPtr lessThanY(double y);
  static SpatialFilterPtr lessThanZ(double z);

  static SpatialFilterPtr greaterThanX(double x);
  static SpatialFilterPtr greaterThanY(double y);
  static SpatialFilterPtr greaterThanZ(double z);

  virtual ~SpatialFilter() {}
};

class SpatialFilterUnfiltered : public SpatialFilter {
  bool matchesPoint(vector<double> &point);
};

class SpatialFilterLogicalOr : public SpatialFilter {
  SpatialFilterPtr _sf1, _sf2;
public:
  SpatialFilterLogicalOr(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);

//  bool matchesPoint( double x, double y ) {
//    return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
//  }
};

class SpatialFilterLogicalAnd : public SpatialFilter {
  SpatialFilterPtr _sf1, _sf2;
public:
  SpatialFilterLogicalAnd(SpatialFilterPtr sf1, SpatialFilterPtr sf2);
  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);

//  bool matchesPoint( double x, double y ) {
//    return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
//  }
};

class NegatedSpatialFilter : public SpatialFilter {
  SpatialFilterPtr _filterToNegate;
public:
  NegatedSpatialFilter(SpatialFilterPtr FilterToNegate);
  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache);
};
//
//SpatialFilterPtr operator!(SpatialFilterPtr sf) {
//  return Teuchos::rcp( new NegatedSpatialFilter(sf) );
//}


#endif
