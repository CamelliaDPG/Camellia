//
//  Basis1DIntegralCache.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 1/28/25.
//

#include "Basis1DIntegralCache.hpp"

#include <Basis.h>
#include <CubatureFactory.h>

using namespace Camellia;

template<typename Scalar>
Kokkos::View<Scalar**, Camellia::DeviceType>
Basis1DIntegralCache<Scalar>::computeBasisValues(BasisPtr basis, bool differentiate,
                                                 Intrepid2::TensorPoints<double,DeviceType> points)
{
  int cardinality = basis->getCardinality();
  
  Intrepid2::EOperator op = differentiate ? Intrepid2::OPERATOR_GRAD : Intrepid2::OPERATOR_VALUE;
  
  auto basisValuesDevice = basis->allocateBasisValues(points, op);
  basis->getValues(basisValuesDevice,points,op);
  
  const int numPoints = points.extent_int(0);
  
  // for derivatives, create a view that does not have the extra derivative dimension
  // for both derivatives and values, copy into a regular 2D View instead of BasisValues object.
  View2D basisValuesFlattened("basisValuesFlattened",cardinality,numPoints);
  
  Kokkos::parallel_for("flatten basisValues", cardinality,
  KOKKOS_LAMBDA(const int &basisOrdinal)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      if (differentiate)
      {
        basisValuesFlattened(basisOrdinal,pointOrdinal) = basisValuesDevice(basisOrdinal,pointOrdinal,0);
      }
      else
      {
        basisValuesFlattened(basisOrdinal,pointOrdinal) = basisValuesDevice(basisOrdinal,pointOrdinal);
      }
    }
  });
  
  return basisValuesFlattened;
}

template<typename Scalar>
Kokkos::View<Scalar*, Camellia::DeviceType>
Basis1DIntegralCache<Scalar>::getReferenceSpaceIntegral(BasisPtr basis1, bool differentiateBasis1,
                                                        BasisPtr basis2, bool differentiateBasis2,
                                                        int monomialWeightDegree, int quadratureDegree,
                                                        bool layoutLeft)
{
  // determine key for cache lookup/storage
  BasisKey basisKey1 {basis1->intrepid2Basis()->getName(), differentiateBasis1, basis1->getDegree()};
  BasisKey basisKey2 {basis2->intrepid2Basis()->getName(), differentiateBasis2, basis2->getDegree()};
  std::vector<BasisKey> basisKeyList {basisKey1,basisKey2};
  RefIntegralKey integralKey {basisKeyList,monomialWeightDegree,layoutLeft};
  
  if (_refSpaceIntegrals.find(integralKey) == _refSpaceIntegrals.end())
  {
    if (quadratureDegree == -1)
    {
      quadratureDegree = monomialWeightDegree + basis1->getDegree() + basis2->getDegree();
    }
    
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), quadratureDegree);
    auto refPoints       = cubature->allocateCubaturePoints();
    auto cubatureWeights = cubature->allocateCubatureWeights();
    cubature->getCubature(refPoints, cubatureWeights);

#ifndef KOKKOS_ENABLE_HIP
    Intrepid2::TensorPoints<double,Kokkos::DefaultHostExecutionSpace::device_type> refPointsHost(refPoints);
#else
    Intrepid2::TensorPoints<double,Kokkos::HIPHostPinnedSpace::device_type> refPointsHost(refPoints);
#endif
    
    const int numPoints = refPoints.extent_int(0);
    
    using View2D     = Kokkos::View<Scalar**, Camellia::DeviceType>;
    using HostView2D = typename View2D::HostMirror;
    
    auto basisValues1 = computeBasisValues(basis1, differentiateBasis1, refPoints);
    auto basisValues2 = computeBasisValues(basis2, differentiateBasis2, refPoints);
    
    int basis1Cardinality = basis1->getCardinality();
    int basis2Cardinality = basis2->getCardinality();
    
    Kokkos::View<Scalar*, Camellia::DeviceType> integrals("Basis1DIntegralCache integrals", basis1Cardinality*basis2Cardinality);
    
    Kokkos::MDRangePolicy<DeviceExecutionSpace,Kokkos::Rank<2>> rangePolicy({0,0},{basis1Cardinality,basis2Cardinality});
    
    Kokkos::parallel_for(rangePolicy,
                         KOKKOS_LAMBDA(const int &i0, const int &i1)
                         {
      Scalar integralValue = 0.0;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double &x = refPointsHost(pointOrdinal,0);
        double xWeight = 1.0;
        for (int i=0; i<monomialWeightDegree; i++)
        {
          xWeight *= x;
        }
        const double &quadratureWeight = cubatureWeights(pointOrdinal);
        const double &value1 = basisValues1(i0,pointOrdinal);
        const double &value2 = basisValues2(i1,pointOrdinal);
        integralValue += quadratureWeight * xWeight * value1 * value2;
      }
      const int resultIndex = layoutLeft ? i0 + i1 * basis1Cardinality : i1 + i0 * basis2Cardinality;
      integrals(resultIndex) = integralValue;
    });
    _refSpaceIntegrals[integralKey] = integrals;
  }
  return _refSpaceIntegrals[integralKey];
} // getReferenceSpaceIntegral(): two-basis version

template<typename Scalar>
Kokkos::View<Scalar*, Camellia::DeviceType>
Basis1DIntegralCache<Scalar>::getReferenceSpaceIntegral(BasisPtr basis1, bool differentiateBasis1,
                                                        BasisPtr basis2, bool differentiateBasis2,
                                                        BasisPtr basis3, bool differentiateBasis3,
                                                        int monomialWeightDegree, int quadratureDegree,
                                                        bool layoutLeft)
{
  // determine key for cache lookup/storage
  BasisKey basisKey1 {basis1->intrepid2Basis()->getName(), differentiateBasis1, basis1->getDegree()};
  BasisKey basisKey2 {basis2->intrepid2Basis()->getName(), differentiateBasis2, basis2->getDegree()};
  BasisKey basisKey3 {basis3->intrepid2Basis()->getName(), differentiateBasis3, basis3->getDegree()};
  std::vector<BasisKey> basisKeyList {basisKey1,basisKey2,basisKey3};
  RefIntegralKey integralKey {basisKeyList,monomialWeightDegree,layoutLeft};
  
  if (_refSpaceIntegrals.find(integralKey) == _refSpaceIntegrals.end())
  {
    if (quadratureDegree == -1)
    {
      quadratureDegree = monomialWeightDegree + basis1->getDegree() + basis2->getDegree() + basis3->getDegree();
    }
    
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), quadratureDegree);
    auto refPoints       = cubature->allocateCubaturePoints();
    auto cubatureWeights = cubature->allocateCubatureWeights();
    cubature->getCubature(refPoints, cubatureWeights);

#ifndef KOKKOS_ENABLE_HIP
    Intrepid2::TensorPoints<double,Kokkos::DefaultHostExecutionSpace::device_type> refPointsHost(refPoints);
#else
    Intrepid2::TensorPoints<double,Kokkos::HIPHostPinnedSpace::device_type> refPointsHost(refPoints);
#endif
    
    const int numPoints = refPoints.extent_int(0);
    
    using View2D     = Kokkos::View<Scalar**, Camellia::DeviceType>;
    using HostView2D = typename View2D::HostMirror;
    
    auto basisValues1 = computeBasisValues(basis1, differentiateBasis1, refPoints);
    auto basisValues2 = computeBasisValues(basis2, differentiateBasis2, refPoints);
    auto basisValues3 = computeBasisValues(basis3, differentiateBasis3, refPoints);
    
    int basis1Cardinality = basis1->getCardinality();
    int basis2Cardinality = basis2->getCardinality();
    int basis3Cardinality = basis3->getCardinality();
    
    Kokkos::View<Scalar*, Camellia::DeviceType> integrals("Basis1DIntegralCache integrals", basis1Cardinality*basis2Cardinality*basis3Cardinality);
    
    Kokkos::MDRangePolicy<DeviceExecutionSpace,Kokkos::Rank<3>> rangePolicy({0,0,0},{basis1Cardinality,basis2Cardinality,basis3Cardinality});
    
    Kokkos::parallel_for(rangePolicy,
    KOKKOS_LAMBDA(const int &i1, const int &i2, const int &i3)
    {
      Scalar integralValue = 0.0;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double &x = refPointsHost(pointOrdinal,0);
        double xWeight = 1.0;
        for (int i=0; i<monomialWeightDegree; i++)
        {
          xWeight *= x;
        }
        const double &quadratureWeight = cubatureWeights(pointOrdinal);
        const double &value1 = basisValues1(i1,pointOrdinal);
        const double &value2 = basisValues2(i2,pointOrdinal);
        const double &value3 = basisValues3(i3,pointOrdinal);
        integralValue += quadratureWeight * xWeight * value1 * value2 * value3;
      }
      const int resultIndex = layoutLeft ? i1 + i2 * basis1Cardinality + i3 * basis1Cardinality * basis2Cardinality
                                         : i3 + i2 * basis3Cardinality + i1 * basis3Cardinality * basis2Cardinality;
      integrals(resultIndex) = integralValue;
    });
    _refSpaceIntegrals[integralKey] = integrals;
  }
  return _refSpaceIntegrals[integralKey];
} // getReferenceSpaceIntegral(): three-basis version

template<typename Scalar>
Kokkos::View<Scalar*, Camellia::DeviceType>
Basis1DIntegralCache<Scalar>::getReferenceSpaceIntegral(BasisPtr basis1, bool differentiateBasis1,
                                                        BasisPtr basis2, bool differentiateBasis2,
                                                        BasisPtr basis3, bool differentiateBasis3,
                                                        BasisPtr basis4, bool differentiateBasis4,
                                                        int monomialWeightDegree, int quadratureDegree,
                                                        bool layoutLeft)
{
  // determine key for cache lookup/storage
  BasisKey basisKey1 {basis1->intrepid2Basis()->getName(), differentiateBasis1, basis1->getDegree()};
  BasisKey basisKey2 {basis2->intrepid2Basis()->getName(), differentiateBasis2, basis2->getDegree()};
  BasisKey basisKey3 {basis3->intrepid2Basis()->getName(), differentiateBasis3, basis3->getDegree()};
  BasisKey basisKey4 {basis4->intrepid2Basis()->getName(), differentiateBasis4, basis4->getDegree()};
  std::vector<BasisKey> basisKeyList {basisKey1,basisKey2,basisKey3,basisKey4};
  RefIntegralKey integralKey {basisKeyList,monomialWeightDegree,layoutLeft};
  
  if (_refSpaceIntegrals.find(integralKey) == _refSpaceIntegrals.end())
  {
    if (quadratureDegree == -1)
    {
      quadratureDegree = monomialWeightDegree + basis1->getDegree() + basis2->getDegree() + basis3->getDegree() + basis4->getDegree();
    }
    
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), quadratureDegree);
    auto refPoints       = cubature->allocateCubaturePoints();
    auto cubatureWeights = cubature->allocateCubatureWeights();
    cubature->getCubature(refPoints, cubatureWeights);

#ifndef KOKKOS_ENABLE_HIP
    Intrepid2::TensorPoints<double,Kokkos::DefaultHostExecutionSpace::device_type> refPointsHost(refPoints);
#else
    Intrepid2::TensorPoints<double,Kokkos::HIPHostPinnedSpace::device_type> refPointsHost(refPoints);
#endif
    
    const int numPoints = refPoints.extent_int(0);
    
    using View2D     = Kokkos::View<Scalar**, Camellia::DeviceType>;
    using HostView2D = typename View2D::HostMirror;
    
    auto basisValues1 = computeBasisValues(basis1, differentiateBasis1, refPoints);
    auto basisValues2 = computeBasisValues(basis2, differentiateBasis2, refPoints);
    auto basisValues3 = computeBasisValues(basis3, differentiateBasis3, refPoints);
    auto basisValues4 = computeBasisValues(basis4, differentiateBasis4, refPoints);
    
    int basis1Cardinality = basis1->getCardinality();
    int basis2Cardinality = basis2->getCardinality();
    int basis3Cardinality = basis3->getCardinality();
    int basis4Cardinality = basis4->getCardinality();
    
    Kokkos::View<Scalar*, Camellia::DeviceType> integrals("Basis1DIntegralCache integrals", basis1Cardinality*basis2Cardinality*basis3Cardinality*basis4Cardinality);
    
    Kokkos::MDRangePolicy<DeviceExecutionSpace,Kokkos::Rank<4>> rangePolicy({0,0,0,0},{basis1Cardinality,basis2Cardinality,basis3Cardinality,basis4Cardinality});
    
    Kokkos::parallel_for(rangePolicy,
    KOKKOS_LAMBDA(const int &i1, const int &i2, const int &i3, const int &i4)
    {
      Scalar integralValue = 0.0;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        const double &x = refPointsHost(pointOrdinal,0);
        double xWeight = 1.0;
        for (int i=0; i<monomialWeightDegree; i++)
        {
          xWeight *= x;
        }
        const double &quadratureWeight = cubatureWeights(pointOrdinal);
        const double &value1 = basisValues1(i1,pointOrdinal);
        const double &value2 = basisValues2(i2,pointOrdinal);
        const double &value3 = basisValues3(i3,pointOrdinal);
        const double &value4 = basisValues4(i4,pointOrdinal);
        integralValue += quadratureWeight * xWeight * value1 * value2 * value3 * value4;
      }
      const int resultIndex = layoutLeft ? i1 + (i2 + (i3 + i4 * basis3Cardinality) * basis2Cardinality) * basis1Cardinality
                                         : i4 + (i3 + (i2 + i1 * basis2Cardinality) * basis3Cardinality) * basis4Cardinality;
      integrals(resultIndex) = integralValue;
    });
    _refSpaceIntegrals[integralKey] = integrals;
  }
  return _refSpaceIntegrals[integralKey];
} // getReferenceSpaceIntegral(): four-basis version

namespace Camellia
{
template class Basis1DIntegralCache<float>;
template class Basis1DIntegralCache<double>;
}
