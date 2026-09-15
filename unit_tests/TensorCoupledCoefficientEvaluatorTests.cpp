#include "Teuchos_UnitTestHarness.hpp"

#include "TensorCoupledCoefficientEvaluator.h"

#include <cmath>

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorCoupledCoefficientEvaluator, EvaluatesRelativisticGammaAndVelocityAsPointData )
{
  const double c = 5.0;
  auto gamma = TensorFunction::relativisticGamma({2, 3}, c);
  auto vx = TensorFunction::relativisticVelocityComponent(2, {2, 3}, c);

  TEST_ASSERT(TensorCoupledCoefficientEvaluator::requiresNonseparablePointData(gamma));
  TEST_ASSERT(TensorCoupledCoefficientEvaluator::requiresNonseparablePointData(vx));

  TensorFunction::PointViewHost points("points", 2, 4);
  points(0,0) = 0.0; points(0,1) = 0.0; points(0,2) = 3.0; points(0,3) = 4.0;
  points(1,0) = 0.0; points(1,1) = 0.0; points(1,2) = 0.0; points(1,3) = 4.0;

  TensorFunction::ValueViewHost gammaValues("gammaValues", 2);
  TensorFunction::ValueViewHost vxValues("vxValues", 2);
  TensorCoupledCoefficientEvaluator::values(gamma, points, gammaValues);
  TensorCoupledCoefficientEvaluator::values(vx, points, vxValues);

  const double gamma0 = std::sqrt(1.0 + (3.0*3.0 + 4.0*4.0) / (c*c));
  const double gamma1 = std::sqrt(1.0 + (0.0*0.0 + 4.0*4.0) / (c*c));
  TEST_FLOATING_EQUALITY(gammaValues(0), gamma0, 1e-14);
  TEST_FLOATING_EQUALITY(gammaValues(1), gamma1, 1e-14);
  TEST_FLOATING_EQUALITY(vxValues(0), 3.0 / gamma0, 1e-14);
  TEST_FLOATING_EQUALITY(vxValues(1), 0.0 / gamma1, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorCoupledCoefficientEvaluator, RejectsSeparableAndSolutionDependentFallbacks )
{
  TensorFunction::PointViewHost points("points", 1, 2);
  points(0,0) = 1.0;
  points(0,1) = 2.0;
  TensorFunction::ValueViewHost values("values", 1);

  TEST_THROW(TensorCoupledCoefficientEvaluator::values(TensorFunction::componentCoordinate(0), points, values), std::invalid_argument);

  auto source = TensorFunction::configurationSpaceSolutionCoefficient1D(0, "E_x");
  auto coupledSolutionCoefficient = source + source;
  TEST_THROW(TensorCoupledCoefficientEvaluator::values(coupledSolutionCoefficient, points, values), std::invalid_argument);
}
}
