//
//  BGKFunction.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 12/22/21.
//

#include "BGKFunction.h"

using namespace Camellia;

BGKFunction::BGKFunction(const double tau, FunctionPtr f_soln, Teuchos::RCP<SolutionMoments> f_solnMoments)
:
TFunction<double>("BGKFunction", 0)
{
  _tau           = tau;
  _f_soln        = f_soln;
  _f_solnMoments = f_solnMoments;
  
  const int numVelocityDims = f_solnMoments->numVelocityDims();
  if (numVelocityDims == 1)
  {
    _zeroMoment = f_solnMoments->solutionMomentFunction(0);
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(1));
    _secondMoment = f_solnMoments->solutionMomentFunction(2);
  }
  else if (numVelocityDims == 2)
  {
    _zeroMoment = f_solnMoments->solutionMomentFunction(0,0);
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(1,0));
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(0,1));
    _secondMoment = f_solnMoments->solutionMomentFunction(2,0) + f_solnMoments->solutionMomentFunction(0,2);
  }
  else if (numVelocityDims == 3)
  {
    _zeroMoment = f_solnMoments->solutionMomentFunction(0,0);
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(1,0,0));
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(0,1,0));
    _firstMoments.push_back(f_solnMoments->solutionMomentFunction(0,0,1));
    _secondMoment = f_solnMoments->solutionMomentFunction(2,0,0) + f_solnMoments->solutionMomentFunction(0,2,0) + f_solnMoments->solutionMomentFunction(0,0,2);
  }
}

void BGKFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("BGKFunction values()");
  
  using Data = Intrepid2::Data<double, Camellia::DeviceType>;
  
  auto points = basisCache->getPhysicalCubaturePoints();
  
  // checkout storage and store values:
  Data zeroMomentData = _zeroMoment->checkoutValuesData(basisCache);
  _zeroMoment->values(zeroMomentData, basisCache);
  Kokkos::Array<Data,3> firstMomentData;
  
  const int numVelocityDims = _firstMoments.size();
  const int numSpatialDims  = points.extent_int(2); // (C,P,D)
  for (int i=0; i<numVelocityDims; i++)
  {
    firstMomentData[i] = _firstMoments[i]->checkoutValuesData(basisCache);
    _firstMoments[i]->values(firstMomentData[i], basisCache);
  }
  Data secondMomentData = _secondMoment->checkoutValuesData(basisCache);
  _secondMoment->values(secondMomentData, basisCache);
  
  auto fValues = basisCache->checkoutScratchStorage<double>(values);

  _f_soln->values(fValues, ASSIGN, basisCache);
  
  typename FieldContainer<double>::DeviceView valuesToAssign;
  
  if (modType == ASSIGN)
  {
    valuesToAssign = values;
  }
  else
  {
    valuesToAssign = basisCache->checkoutScratchStorage<double>(values);
  }
    
  const auto tau = _tau; // avoid implicit reference to this in the KOKKOS_LAMBDA blocks below.
  
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values);
  Kokkos::fence();
  Kokkos::parallel_for("BGK_values", rangePolicy,
  KOKKOS_LAMBDA (const int i0, const int i1) {
    const double n = zeroMomentData(i0,i1);
    
    double sigmaSquared = (1. / n) * secondMomentData(i0,i1);
    double v_minus_u_squared = 0.0;
    for (int d=0; d<numVelocityDims; d++)
    {
      const double u_d = (1. / n) * firstMomentData[d](i0,i1);
      sigmaSquared -= u_d * u_d;
      const double v_d = points(i0,i1,d+numSpatialDims);
      v_minus_u_squared += (v_d - u_d) * (v_d - u_d);
    }
    
    sigmaSquared *= (1./numVelocityDims);
    
    const double weight   = - n / pow(2 * M_PI * sigmaSquared, 1.5);
    const double exponent = - 1. / (2. * sigmaSquared) * v_minus_u_squared;
    const double f_M = weight * std::exp(exponent);
    const double f = fValues(i0,i1);
    valuesToAssign(i0,i1) = (-1.0 / tau) * (f - f_M);
  });
  
  if (modType != ASSIGN)
  {
    assignValues<double>(values, modType, valuesToAssign);
  }
    
  if (modType != ASSIGN)
  {
    // then we checked out scratch storage for valuesToAssign
    basisCache->checkinScratchStorage<double>(valuesToAssign);
  }
  
  basisCache->checkinScratchStorage<double>(fValues);
  
  _zeroMoment->checkinValuesData(zeroMomentData);
  for (int i=0; i<numVelocityDims; i++)
  {
    _firstMoments[i]->checkinValuesData(firstMomentData[i]);
  }
  _secondMoment->checkinValuesData(secondMomentData);
}
