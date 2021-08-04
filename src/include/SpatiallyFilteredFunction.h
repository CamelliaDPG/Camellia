// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//
//  SpatiallyFilteredFunction.h
//  Camellia
//
//  Created by Nathan Roberts on 4/4/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_SpatiallyFilteredFunction_h
#define Camellia_SpatiallyFilteredFunction_h

#include "Function.h"
#include "SpatialFilter.h"

namespace Camellia
{
template <typename Scalar>
class SpatiallyFilteredFunction : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _f;
  SpatialFilterPtr _sf;

public:
  SpatiallyFilteredFunction(TFunctionPtr<Scalar> f, SpatialFilterPtr sf);
  
  using TFunction<Scalar>::values;
  virtual void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache);
  void values(DeviceView<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache);
  
  bool boundaryValueOnly();

  TFunctionPtr<Scalar> curl();
  TFunctionPtr<Scalar> div();

  TFunctionPtr<Scalar> dx();
  TFunctionPtr<Scalar> dy();
  TFunctionPtr<Scalar> dz();
  
  std::vector< TFunctionPtr<Scalar> > memberFunctions();
  
  static TFunctionPtr<Scalar> spatiallyFilteredFunction(Scalar value, SpatialFilterPtr sf);
  static TFunctionPtr<Scalar> spatiallyFilteredFunction(TFunctionPtr<Scalar> f, SpatialFilterPtr sf);
};
}

#endif
