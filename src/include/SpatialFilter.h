//
//  SpatialFilter.h
//  Camellia
//
//  Created by Nathan Roberts on 3/30/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SpatialFilter_h
#define Camellia_SpatialFilter_h

class SpatialFilter;
typedef Teuchos::RCP< SpatialFilter > SpatialFilterPtr;

class SpatialFilter {
public:
  // just 2D for now:
  virtual bool matchesPoint(double x, double y) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y) unimplemented.");
  }
  //  bool matchesPoint(double x, double y, double z) {
  //    TEST_FOR_EXCEPTION(true, std::invalid_argument, "matchesPoint(x,y,z) unimplemented.");
  //  }
};

class NegatedSpatialFilter : public SpatialFilter {
  SpatialFilterPtr _filterToNegate;
public:
  NegatedSpatialFilter(SpatialFilterPtr filterToNegate) {
    _filterToNegate = filterToNegate;
  }
  bool matchesPoint(double x, double y) {
    return ! _filterToNegate->matchesPoint(x,y);
  }
};
//
//SpatialFilterPtr operator!(SpatialFilterPtr sf) {
//  return Teuchos::rcp( new NegatedSpatialFilter(sf) );
//}

#endif
