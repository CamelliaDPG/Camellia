// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

//  ExpFunction.h
//  Camellia
//
//  Created by Nate Roberts on 4/8/15.
//
//

#ifndef Camellia_MaxwellianFunction_h
#define Camellia_MaxwellianFunction_h

#include "Function.h"
#include "SimpleFunction.h"
#include "TypeDefs.h"

namespace Camellia
{
  class MaxwellianFunction : public TFunction<double>
  {
    int _firstVelocityDimension;
    int _numVelocityDims;
    
    Kokkos::Array<double,3> _u;
    double _sigmaSquared;
    double _n;
  public:
    MaxwellianFunction(const int firstVelocityDimensionZeroBased, const int numVelocityDims, std::vector<double> u, double sigmaSquared, double n);
    
    using TFunction<double>::values;
    void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
    
    static FunctionPtr maxwellian(const int firstVelocityDimensionZeroBased, const int numVelocityDims, std::vector<double> u, double sigmaSquared, double n);
  };
}
#endif
