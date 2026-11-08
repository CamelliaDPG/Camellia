#include "Teuchos_UnitTestHarness.hpp"

#include "TensorMomentReduction.h"

#include <Kokkos_Core.hpp>

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorMomentReduction, ReducesDensityAndProperVelocityMomentsFromVlasovCoefficients )
{
  TensorMomentReduction::HostMatrix f("f", 2, 3);
  f(0,0) = 1.0; f(0,1) = 2.0; f(0,2) = 3.0;
  f(1,0) = 4.0; f(1,1) = 5.0; f(1,2) = 6.0;

  TensorMomentReduction::HostMatrix velocityPoints("velocityPoints", 3, 2);
  velocityPoints(0,0) = -1.0; velocityPoints(0,1) = 0.0;
  velocityPoints(1,0) =  0.0; velocityPoints(1,1) = 2.0;
  velocityPoints(2,0) =  3.0; velocityPoints(2,1) = 4.0;

  TensorMomentReduction::HostVector velocityWeights("velocityWeights", 3);
  velocityWeights(0) = 0.5;
  velocityWeights(1) = 1.0;
  velocityWeights(2) = 1.5;

  TensorMomentReduction::HostVector density("density", 2);
  TensorMomentReduction::reduceHost(TensorMomentDescriptor::planar1D2VDensity(), f, velocityPoints, velocityWeights, density);
  TEST_FLOATING_EQUALITY(density(0), 1.0*0.5 + 2.0*1.0 + 3.0*1.5, 1e-14);
  TEST_FLOATING_EQUALITY(density(1), 4.0*0.5 + 5.0*1.0 + 6.0*1.5, 1e-14);

  TensorMomentReduction::HostVector uxMoment("uxMoment", 2);
  TensorMomentReduction::reduceHost(TensorMomentDescriptor::planar1D2VXProperVelocityMoment(), f, velocityPoints, velocityWeights, uxMoment);
  TEST_FLOATING_EQUALITY(uxMoment(0), 1.0*0.5*(-1.0) + 2.0*1.0*0.0 + 3.0*1.5*3.0, 1e-14);
  TEST_FLOATING_EQUALITY(uxMoment(1), 4.0*0.5*(-1.0) + 5.0*1.0*0.0 + 6.0*1.5*3.0, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorMomentReduction, RelativisticCurrentUsesCoupledVelocityMap )
{
  TensorMomentReduction::HostMatrix f("f", 1, 2);
  f(0,0) = 2.0;
  f(0,1) = 3.0;

  TensorMomentReduction::HostMatrix velocityPoints("velocityPoints", 2, 2);
  velocityPoints(0,0) = 3.0; velocityPoints(0,1) = 4.0;
  velocityPoints(1,0) = 0.0; velocityPoints(1,1) = 4.0;

  TensorMomentReduction::HostVector velocityWeights("velocityWeights", 2);
  velocityWeights(0) = 1.0;
  velocityWeights(1) = 2.0;

  TensorMomentReduction::HostVector jx("jx", 1);
  const double q = -2.0;
  const double c = 5.0;
  TensorMomentReduction::reduceHost(TensorMomentDescriptor::planar1D2VCurrentX(q), f, velocityPoints, velocityWeights, jx, c);

  const double gamma0 = std::sqrt(1.0 + (3.0*3.0 + 4.0*4.0) / (c*c));
  const double gamma1 = std::sqrt(1.0 + (0.0*0.0 + 4.0*4.0) / (c*c));
  const double expected = 2.0 * 1.0 * q * 3.0 / gamma0 + 3.0 * 2.0 * q * 0.0 / gamma1;
  TEST_FLOATING_EQUALITY(jx(0), expected, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorMomentReduction, DeviceKernelReducesPlanarMoments )
{
  TensorMomentReduction::HostMatrix fHost("fHost", 2, 3);
  fHost(0,0) = 1.0; fHost(0,1) = 2.0; fHost(0,2) = 3.0;
  fHost(1,0) = 4.0; fHost(1,1) = 5.0; fHost(1,2) = 6.0;

  TensorMomentReduction::HostMatrix velocityPointsHost("velocityPointsHost", 3, 2);
  velocityPointsHost(0,0) = -1.0; velocityPointsHost(0,1) = 0.0;
  velocityPointsHost(1,0) =  0.0; velocityPointsHost(1,1) = 2.0;
  velocityPointsHost(2,0) =  3.0; velocityPointsHost(2,1) = 4.0;

  TensorMomentReduction::HostVector velocityWeightsHost("velocityWeightsHost", 3);
  velocityWeightsHost(0) = 0.5;
  velocityWeightsHost(1) = 1.0;
  velocityWeightsHost(2) = 1.5;

  TensorMomentReduction::DeviceMatrix fDevice("fDevice", 2, 3);
  TensorMomentReduction::DeviceMatrix velocityPointsDevice("velocityPointsDevice", 3, 2);
  TensorMomentReduction::DeviceVector velocityWeightsDevice("velocityWeightsDevice", 3);
  Kokkos::deep_copy(fDevice, fHost);
  Kokkos::deep_copy(velocityPointsDevice, velocityPointsHost);
  Kokkos::deep_copy(velocityWeightsDevice, velocityWeightsHost);

  TensorMomentReduction::DeviceVector densityDevice("densityDevice", 2);
  TensorMomentReduction::reduceDevicePlanar1D2V(TensorMomentDescriptor::planar1D2VDensity(),
                                                fDevice,
                                                velocityPointsDevice,
                                                velocityWeightsDevice,
                                                densityDevice);
  auto densityHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), densityDevice);
  TEST_FLOATING_EQUALITY(densityHost(0), 1.0*0.5 + 2.0*1.0 + 3.0*1.5, 1e-14);
  TEST_FLOATING_EQUALITY(densityHost(1), 4.0*0.5 + 5.0*1.0 + 6.0*1.5, 1e-14);

  TensorMomentReduction::DeviceVector uxMomentDevice("uxMomentDevice", 2);
  TensorMomentReduction::reduceDevicePlanar1D2V(TensorMomentDescriptor::planar1D2VXProperVelocityMoment(),
                                                fDevice,
                                                velocityPointsDevice,
                                                velocityWeightsDevice,
                                                uxMomentDevice);
  auto uxMomentHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), uxMomentDevice);
  TEST_FLOATING_EQUALITY(uxMomentHost(0), 1.0*0.5*(-1.0) + 2.0*1.0*0.0 + 3.0*1.5*3.0, 1e-14);
  TEST_FLOATING_EQUALITY(uxMomentHost(1), 4.0*0.5*(-1.0) + 5.0*1.0*0.0 + 6.0*1.5*3.0, 1e-14);

  TensorMomentReduction::DeviceVector jxDevice("jxDevice", 2);
  const double q = -2.0;
  const double c = 5.0;
  TensorMomentReduction::reduceDevicePlanar1D2V(TensorMomentDescriptor::planar1D2VCurrentX(q),
                                                fDevice,
                                                velocityPointsDevice,
                                                velocityWeightsDevice,
                                                jxDevice,
                                                c);
  auto jxHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jxDevice);
  for (int point=0; point<2; point++)
  {
    double expected = 0.0;
    for (int v=0; v<3; v++)
    {
      const double ux = velocityPointsHost(v,0);
      const double uz = velocityPointsHost(v,1);
      const double gamma = std::sqrt(1.0 + (ux*ux + uz*uz) / (c*c));
      expected += fHost(point,v) * velocityWeightsHost(v) * q * ux / gamma;
    }
    TEST_FLOATING_EQUALITY(jxHost(point), expected, 1e-14);
  }
}
}
