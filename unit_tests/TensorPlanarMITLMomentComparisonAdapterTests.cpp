#include "Teuchos_UnitTestHarness.hpp"

#include "TensorPlanarMITLMomentComparisonAdapter.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorPlanarMITLMomentComparisonAdapter, ReportsFieldwiseMaxAbsoluteAndRelativeDifferences )
{
  TensorMaxwellRHSMomentData expected(1, 3);
  TensorMaxwellRHSMomentData actual(1, 3);

  for (int point=0; point<3; point++)
  {
    expected.rhoRHS(0, point) = 10.0 + point;
    expected.Jx(0, point) = 20.0 + point;
    expected.Jz(0, point) = 30.0 + point;
    actual.rhoRHS(0, point) = expected.rhoRHS(0, point);
    actual.Jx(0, point) = expected.Jx(0, point);
    actual.Jz(0, point) = expected.Jz(0, point);
  }

  actual.rhoRHS(0, 1) += 0.5;
  actual.Jx(0, 2) -= 2.0;
  actual.Jz(0, 0) += 0.03;

  auto comparison = TensorPlanarMITLMomentComparisonAdapter::compare(actual, expected);
  TEST_FLOATING_EQUALITY(comparison.rhoRHS.maxAbsDifference, 0.5, 1e-14);
  TEST_EQUALITY(comparison.rhoRHS.maxAbsEntry, 1);
  TEST_FLOATING_EQUALITY(comparison.rhoRHS.maxRelDifference, 0.5 / 11.0, 1e-14);
  TEST_EQUALITY(comparison.rhoRHS.maxRelEntry, 1);

  TEST_FLOATING_EQUALITY(comparison.Jx.maxAbsDifference, 2.0, 1e-14);
  TEST_EQUALITY(comparison.Jx.maxAbsEntry, 2);
  TEST_FLOATING_EQUALITY(comparison.Jx.maxRelDifference, 2.0 / 22.0, 1e-14);
  TEST_EQUALITY(comparison.Jx.maxRelEntry, 2);

  TEST_FLOATING_EQUALITY(comparison.Jz.maxAbsDifference, 0.03, 1e-12);
  TEST_EQUALITY(comparison.Jz.maxAbsEntry, 0);
  TEST_FLOATING_EQUALITY(comparison.Jz.maxRelDifference, 0.03 / 30.0, 1e-12);
  TEST_EQUALITY(comparison.Jz.maxRelEntry, 0);

  TEST_ASSERT(comparison.within(2.0, 0.1));
  TEST_ASSERT(!comparison.within(0.1, 0.1));
  TEST_ASSERT(!comparison.within(2.0, 0.01));
}

TEUCHOS_UNIT_TEST( TensorPlanarMITLMomentComparisonAdapter, UsesRelativeFloorNearZero )
{
  TensorMaxwellRHSMomentData expected(1, 1);
  TensorMaxwellRHSMomentData actual(1, 1);
  expected.rhoRHS(0, 0) = 0.0;
  actual.rhoRHS(0, 0) = 1e-12;

  auto comparison = TensorPlanarMITLMomentComparisonAdapter::compare(actual, expected, 1e-9);
  TEST_FLOATING_EQUALITY(comparison.rhoRHS.maxAbsDifference, 1e-12, 1e-14);
  TEST_FLOATING_EQUALITY(comparison.rhoRHS.maxRelDifference, 1e-3, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorPlanarMITLMomentComparisonAdapter, RejectsIncompatibleMomentData )
{
  TensorMaxwellRHSMomentData oneCell(1, 2);
  TensorMaxwellRHSMomentData twoCells(2, 2);
  TensorMaxwellRHSMomentData differentPointCount(1, 3);

  TEST_THROW(TensorPlanarMITLMomentComparisonAdapter::compare(oneCell, twoCells), std::invalid_argument);
  TEST_THROW(TensorPlanarMITLMomentComparisonAdapter::compare(oneCell, differentPointCount), std::invalid_argument);
  TEST_THROW(TensorPlanarMITLMomentComparisonAdapter::compare(oneCell, oneCell, 0.0), std::invalid_argument);
}
}
