#include "Teuchos_UnitTestHarness.hpp"

#include "TensorMomentDescriptor.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorMomentDescriptor, Planar1D2VMomentsMirrorLegacySolutionMomentArguments )
{
  auto density = TensorMomentDescriptor::planar1D2VDensity();
  TEST_EQUALITY(density.key(), "density");
  TEST_EQUALITY(density.velocityDim(), 2);
  TEST_ASSERT(density.isDensityMoment());
  TEST_EQUALITY(density.velocityComponentPower(0), 0);
  TEST_EQUALITY(density.velocityComponentPower(1), 0);
  TEST_ASSERT(density.requiresExactTensorQuadrature());

  auto uxMoment = TensorMomentDescriptor::planar1D2VXProperVelocityMoment();
  TEST_EQUALITY(uxMoment.key(), "ux_moment");
  TEST_EQUALITY(uxMoment.velocityComponentPower(0), 1);
  TEST_EQUALITY(uxMoment.velocityComponentPower(1), 0);
  TEST_EQUALITY(uxMoment.totalVelocityMomentOrder(), 1);
  TEST_ASSERT(!uxMoment.usesRelativisticVelocityMap());

  auto secondMoment = TensorMomentDescriptor::planar1D2VSecondProperVelocityMoment();
  TEST_EQUALITY(secondMoment.key(), "uy_or_uz_moment");
  TEST_EQUALITY(secondMoment.velocityComponentPower(0), 0);
  TEST_EQUALITY(secondMoment.velocityComponentPower(1), 1);
  TEST_EQUALITY(secondMoment.totalVelocityMomentOrder(), 1);
}

TEUCHOS_UNIT_TEST( TensorMomentDescriptor, RelativisticCurrentDescriptorsRecordVelocityMapAndCharge )
{
  const double q = -2.0;
  auto jx = TensorMomentDescriptor::planar1D2VCurrentX(q);
  TEST_EQUALITY(jx.key(), "Jx");
  TEST_ASSERT(jx.usesRelativisticVelocityMap());
  TEST_EQUALITY(jx.relativisticVelocityComponentOrdinal(), 0);
  TEST_FLOATING_EQUALITY(jx.chargeWeight(), q, 1e-15);
  TEST_EQUALITY(jx.velocityComponentPower(0), 1);
  TEST_EQUALITY(jx.velocityComponentPower(1), 0);

  auto jz = TensorMomentDescriptor::planar1D2VCurrentZ(q);
  TEST_EQUALITY(jz.key(), "Jz");
  TEST_ASSERT(jz.usesRelativisticVelocityMap());
  TEST_EQUALITY(jz.relativisticVelocityComponentOrdinal(), 1);
  TEST_EQUALITY(jz.velocityComponentPower(0), 0);
  TEST_EQUALITY(jz.velocityComponentPower(1), 1);
}

TEUCHOS_UNIT_TEST( TensorMomentDescriptor, RejectsInvalidMetadata )
{
  TEST_THROW(TensorMomentDescriptor::density(0), std::invalid_argument);
  TEST_THROW(TensorMomentDescriptor::properVelocityMoment("bad", {1, -1}), std::invalid_argument);
  TEST_THROW(TensorMomentDescriptor::properVelocityMoment("", {0, 0}), std::invalid_argument);
  TEST_THROW(TensorMomentDescriptor::relativisticCurrent("bad-current", {1, 0}, 2, 1.0), std::invalid_argument);
}
}
