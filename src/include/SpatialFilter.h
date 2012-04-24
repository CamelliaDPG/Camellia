//
//  SpatialFilter.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SpatialFilter_h
#define Camellia_SpatialFilter_h

// Teuchos includes
#include "Teuchos_RCP.hpp"

class SpatialFilter;
typedef Teuchos::RCP< SpatialFilter > SpatialFilterPtr;

class SpatialFilter {
public:
  // just 2D for now:
  virtual bool matchesPoint(double x, double y) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y) unimplemented.");
  }
  virtual bool matchesPoints(FieldContainer<bool> pointsMatch, BasisCachePtr basisCache) {
    const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
    int numCells = points->dimension(0);
    int numPoints = points->dimension(1);
    int spaceDim = points->dimension(2);
    TEST_FOR_EXCEPTION(numCells != pointsMatch.dimension(0), std::invalid_argument, "numCells do not match.");
    TEST_FOR_EXCEPTION(numPoints != pointsMatch.dimension(1), std::invalid_argument, "numPoints do not match.");
    TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "matchesPoints only supports 2D so far.");
    pointsMatch.initialize(false);
    bool somePointMatches = false;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numCells; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        if (matchesPoint(x,y)) {
          somePointMatches = true;
          pointsMatch(cellIndex,ptIndex) = true;
        }
      }
    }
    return somePointMatches;
  }
  //  bool matchesPoint(double x, double y, double z) {
  //    TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y,z) unimplemented.");
  //  }
};

class SpatialFilterLogicalOr : public SpatialFilter {
  SpatialFilterPtr _sf1, _sf2;
public:
  SpatialFilterLogicalOr(SpatialFilterPtr sf1, SpatialFilterPtr sf2) {
    _sf1 = sf1;
    _sf2 = sf2;
  }
  bool matchesPoints(FieldContainer<bool> pointsMatch, BasisCachePtr basisCache) {
    const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
    int numCells = points->dimension(0);
    int numPoints = points->dimension(1);
    int spaceDim = points->dimension(2);
    FieldContainer<bool> pointsMatch2(pointsMatch);
    bool somePointMatches1 = _sf1->matchesPoints(pointsMatch,basisCache);
    bool somePointMatches2 = _sf1->matchesPoints(pointsMatch2,basisCache);
    if ( !somePointMatches2 ) {
      // then what's in pointsMatch is exactly right
      return somePointMatches1;
    } else if ( !somePointMatches1 ) {
      // then what's in pointsMatch2 is exactly right
      pointsMatch = pointsMatch2;
      return somePointMatches2;
    } else {
      // need to combine them:
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          pointsMatch(cellIndex,ptIndex) |= pointsMatch2(cellIndex,ptIndex);
        }
      }
    }
  }
//  bool matchesPoint( double x, double y ) {
//    return _sf1->matchesPoint(x,y) || _sf2->matchesPoint(x,y);
//  }
};

class NegatedSpatialFilter : public SpatialFilter {
  SpatialFilterPtr _filterToNegate;
public:
  NegatedSpatialFilter(SpatialFilterPtr filterToNegate) {
    _filterToNegate = filterToNegate;
  }
  bool matchesPoints(FieldContainer<bool> pointsMatch, BasisCachePtr basisCache) {
    const FieldContainer<double>* points = &(basisCache->getPhysicalCubaturePoints());
    int numCells = points->dimension(0);
    int numPoints = points->dimension(1);
    int spaceDim = points->dimension(2);
    _filterToNegate->matchesPoints(pointsMatch,basisCache);
    bool somePointMatches = false;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        pointsMatch(cellIndex,ptIndex) = ! pointsMatch(cellIndex,ptIndex);
        somePointMatches |= pointsMatch(cellIndex,ptIndex);
      }
    }
  }
};
//
//SpatialFilterPtr operator!(SpatialFilterPtr sf) {
//  return Teuchos::rcp( new NegatedSpatialFilter(sf) );
//}

#endif
