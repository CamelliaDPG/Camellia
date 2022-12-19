//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "MaxwellianFunction.h"
#include "MonomialFunctions.h"

#include "BasisCache.h"

using namespace Camellia;

MaxwellianFunction::MaxwellianFunction(const int firstVelocityDimensionZeroBased, const int numVelocityDims, std::vector<double> u, double sigmaSquared, double n)
:
_firstVelocityDimension(firstVelocityDimensionZeroBased),
_numVelocityDims(numVelocityDims),
_sigmaSquared(sigmaSquared),
_n(n)
{
  TEUCHOS_TEST_FOR_EXCEPTION(int(u.size()) != numVelocityDims, std::invalid_argument, "u's length must match numVelocityDims");
  for (int d=0; d<numVelocityDims; d++)
  {
    _u[d] = u[d];
  }
  
  int variationDimensions = 0;
  for (int vd=0; vd<numVelocityDims; vd++)
  {
    variationDimensions |= (VARIES_IN_DIMENSION_0 << firstVelocityDimensionZeroBased + vd);
  }
  setVariationDimensions(variationDimensions);
}

FunctionPtr MaxwellianFunction::maxwellian(const int firstVelocityDimensionZeroBased, const int numVelocityDims, std::vector<double> u, double sigmaSquared, double n)
{
  return Teuchos::rcp( new MaxwellianFunction(firstVelocityDimensionZeroBased, numVelocityDims, u, sigmaSquared, n) );
}

void MaxwellianFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Maxwellian values()");

  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values);
  
  typename FieldContainer<double>::DeviceView valuesToAssign;
  
  if (modType == ASSIGN)
  {
    valuesToAssign = values;
  }
  else
  {
    valuesToAssign = basisCache->checkoutScratchStorage<double>(values);
  }
  
  auto points = basisCache->getPhysicalCubaturePoints();
  const int numCells  = points.extent_int(0);
  const int numPoints = points.extent_int(1);
  
  const double weight = _n / std::pow(2. * M_PI * _sigmaSquared, _numVelocityDims / 2.);
  
  const auto u = _u;
  const int vdimOffset = _firstVelocityDimension;
  
  const int numVelocityDims = _numVelocityDims;
  const double sigmaSquared = _sigmaSquared;
  
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      double v_minus_u_squared = 0.0;
      for (int vdim=0; vdim<numVelocityDims; vdim++)
      {
        const double v_d = points(cellOrdinal,pointOrdinal,vdim+vdimOffset);
        const double u_d = u[vdim];
        v_minus_u_squared += (v_d-u_d) * (v_d-u_d);
      }
      const double exponent = - 0.5 * v_minus_u_squared / sigmaSquared;
      valuesToAssign(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
    }
  }
  
  if (modType != ASSIGN)
  {
    assignValues<double>(values, modType, valuesToAssign);
    // we checked out scratch storage for valuesToAssign; check back in
    basisCache->checkinScratchStorage<double>(valuesToAssign);
  }
}

TFunctionPtr<double> MaxwellianFunction::di(int i)
{
  if ((i < _firstVelocityDimension + 1) || (i > _firstVelocityDimension + _numVelocityDims))
  {
    return Function::zero();
  }
  else
  {
    const int d = i - 1 - _firstVelocityDimension; // zero-based index into velocity dimensions (d==0 corresponds to v_x)
    auto v_d = MonomialFunction::xi_n(d, 1); // v_d^1
    std::vector<double> u(_numVelocityDims);
    for (int i=0; i<_numVelocityDims; i++)
    {
      u[i] = _u[i];
    }
    FunctionPtr f_M = Teuchos::rcp(new MaxwellianFunction(_firstVelocityDimension, _numVelocityDims, u, _sigmaSquared, _n) );
    
    return (u[d] - v_d) / _sigmaSquared * f_M;
  }
}
