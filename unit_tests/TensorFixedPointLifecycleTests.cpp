#include "Teuchos_UnitTestHarness.hpp"

#include "TensorFixedPointLifecycle.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorFixedPointLifecycle, PlanarMITLDefaultsMirrorDriverKnobs )
{
  auto lifecycle = TensorFixedPointLifecycle::planarMITLDefaults(5, 50, 1e-8);

  TEST_ASSERT(lifecycle.startsFromZeroFieldGuess());
  TEST_EQUALITY(lifecycle.initialGuessPolicy(), TENSOR_FIXED_POINT_INITIAL_GUESS_ZERO);
  TEST_EQUALITY(lifecycle.minSubsteps(), 5);
  TEST_EQUALITY(lifecycle.maxSubsteps(), 50);
  TEST_FLOATING_EQUALITY(lifecycle.relativeL2Tolerance(), 1e-8, 1e-15);
  TEST_ASSERT(lifecycle.cachesAreScopedToFixedPointSubstep());
  TEST_ASSERT(lifecycle.invalidateCoefficientCachesWhenConfigurationSolutionChanges());
}

TEUCHOS_UNIT_TEST( TensorFixedPointLifecycle, PreviousSlabPolicyCanBeRecordedButDoesNotChangeCacheContract )
{
  TensorFixedPointLifecycle lifecycle(TENSOR_FIXED_POINT_INITIAL_GUESS_PREVIOUS_SLAB,
                                      2,
                                      10,
                                      1e-2,
                                      TENSOR_COEFFICIENT_CACHE_PER_FIXED_POINT_SUBSTEP,
                                      true);

  TEST_ASSERT(!lifecycle.startsFromZeroFieldGuess());
  TEST_EQUALITY(lifecycle.initialGuessPolicy(), TENSOR_FIXED_POINT_INITIAL_GUESS_PREVIOUS_SLAB);
  TEST_ASSERT(lifecycle.cachesAreScopedToFixedPointSubstep());
  TEST_ASSERT(lifecycle.invalidateCoefficientCachesWhenConfigurationSolutionChanges());
}

TEUCHOS_UNIT_TEST( TensorFixedPointLifecycle, RejectsInvalidLifecycleMetadata )
{
  TEST_THROW(TensorFixedPointLifecycle::planarMITLDefaults(-1, 5, 1e-8), std::invalid_argument);
  TEST_THROW(TensorFixedPointLifecycle::planarMITLDefaults(6, 5, 1e-8), std::invalid_argument);
  TEST_THROW(TensorFixedPointLifecycle::planarMITLDefaults(1, 5, -1e-8), std::invalid_argument);
}
}
