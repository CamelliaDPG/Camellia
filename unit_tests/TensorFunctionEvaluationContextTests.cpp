#include "Teuchos_UnitTestHarness.hpp"

#include "TensorFunctionEvaluationContext.h"

#include "Function.h"

#include <cmath>

using namespace Camellia;

namespace
{
class CountingSolutionCoefficientEvaluator : public TensorSolutionCoefficientEvaluator
{
public:
  mutable int callCount = 0;

  void values(int componentOrdinal,
              TensorFunction::PointViewHost pointsIn,
              TensorFunction::ValueViewHost valuesOut) const override
  {
    callCount++;
    for (int p=0; p<static_cast<int>(pointsIn.extent(0)); p++)
    {
      valuesOut(p) = 100.0 + 10.0 * componentOrdinal + pointsIn(p,0);
    }
  }
};

TEUCHOS_UNIT_TEST( TensorFunctionEvaluationContext, DirectSolutionCoefficientUsesRegisteredEvaluatorAndCache )
{
  auto coefficient = TensorFunction::configurationSpaceSolutionCoefficient1D(0, "E_x");
  auto evaluator = Teuchos::rcp(new CountingSolutionCoefficientEvaluator);
  TensorFunctionEvaluationContext context(TensorFixedPointLifecycle::planarMITLDefaults(1, 3, 1e-8));
  context.registerSolutionCoefficientEvaluator(coefficient->dependencySources()[0], evaluator);

  TensorFunction::PointViewHost points("points", 2, 1);
  points(0,0) = 0.25;
  points(1,0) = 0.75;
  TensorFunction::ValueViewHost values("values", 2);

  context.values(coefficient, points, values);
  TEST_EQUALITY(evaluator->callCount, 1);
  TEST_FLOATING_EQUALITY(values(0), 100.25, 1e-14);
  TEST_FLOATING_EQUALITY(values(1), 100.75, 1e-14);
  TEST_EQUALITY(context.cachedCoefficientEntryCount(), 1);

  values(0) = 0.0;
  values(1) = 0.0;
  context.values(coefficient, points, values);
  TEST_EQUALITY(evaluator->callCount, 1);
  TEST_FLOATING_EQUALITY(values(0), 100.25, 1e-14);
  TEST_FLOATING_EQUALITY(values(1), 100.75, 1e-14);

  context.beginFixedPointSubstep(1);
  TEST_EQUALITY(context.cachedCoefficientEntryCount(), 0);
  context.values(coefficient, points, values);
  TEST_EQUALITY(evaluator->callCount, 2);

  context.configurationSolutionChanged();
  TEST_EQUALITY(context.configurationSolutionEpoch(), 1);
  TEST_EQUALITY(context.cachedCoefficientEntryCount(), 0);
}

TEUCHOS_UNIT_TEST( TensorFunctionEvaluationContext, LegacyFunctionEvaluatorUsesProductionFunctionValues )
{
  auto coefficient = TensorFunction::configurationSpaceSolutionCoefficient1D(0, "E_x");
  auto evaluator = Teuchos::rcp(new TensorLegacyFunctionSolutionCoefficientEvaluator(Function::xn(1), Teuchos::null, 1));
  TensorFunctionEvaluationContext context;
  context.registerSolutionCoefficientEvaluator(coefficient->dependencySources()[0], evaluator);

  TensorFunction::PointViewHost points("points", 2, 1);
  points(0,0) = 0.25;
  points(1,0) = 0.75;
  TensorFunction::ValueViewHost values("values", 2);

  context.values(coefficient, points, values);
  TEST_FLOATING_EQUALITY(values(0), 0.25, 1e-14);
  TEST_FLOATING_EQUALITY(values(1), 0.75, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorFunctionEvaluationContext, EvaluatesSolutionDependentExpressionTreeWithoutCoordinatePlaceholderFallback )
{
  auto coefficient = TensorFunction::configurationSpaceSolutionCoefficient1D(0, "E_x");
  auto composed = sin((coefficient + TensorFunction::constant(1.0)) * TensorFunction::componentCoordinate(0));
  auto evaluator = Teuchos::rcp(new CountingSolutionCoefficientEvaluator);
  TensorFunctionEvaluationContext context;
  context.registerSolutionCoefficientEvaluator(coefficient->dependencySources()[0], evaluator);

  TensorFunction::PointViewHost points("points", 2, 1);
  points(0,0) = 0.25;
  points(1,0) = 0.75;
  TensorFunction::ValueViewHost values("values", 2);

  context.values(composed, points, values);
  TEST_EQUALITY(evaluator->callCount, 1);
  TEST_FLOATING_EQUALITY(values(0), std::sin((100.25 + 1.0) * 0.25), 1e-14);
  TEST_FLOATING_EQUALITY(values(1), std::sin((100.75 + 1.0) * 0.75), 1e-14);
}
}
