//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "MaxwellianFunction.h"

#include "BasisCache.h"

using namespace Camellia;

MaxwellianFunction::MaxwellianFunction(const int firstVelocityDimensionZeroBased, const int numVelocityDims, std::vector<double> u, double sigmaSquared, double n)
:
_firstVelocityDimension(firstVelocityDimensionZeroBased),
_numVelocityDims(numVelocityDims),
_sigmaSquared(sigmaSquared),
_n(n)
{
  TEUCHOS_TEST_FOR_EXCEPTION(u.size() != numVelocityDims, std::invalid_argument, "u's length must match numVelocityDims");
  for (int d=0; d<numVelocityDims; d++)
  {
    _u[d] = u[d];
  }
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
  
  const double weight = _n / std::pow(2. * M_PI * _sigmaSquared, 1.5);
  
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
