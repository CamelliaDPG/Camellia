//
//  SpatialFilter.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/28/13.
//
//

#include "SpatialFilter.h"

SpatialFilterPtr SpatialFilter::allSpace() {
  return Teuchos::rcp( new SpatialFilterUnfiltered );
}

SpatialFilterPtr SpatialFilter::unionFilter(SpatialFilterPtr a, SpatialFilterPtr b) {
  return Teuchos::rcp( new SpatialFilterLogicalOr(a,b) );
}