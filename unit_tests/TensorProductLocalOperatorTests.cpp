#include "Teuchos_UnitTestHarness.hpp"

#include "TensorProductLocalOperator.h"

using namespace Camellia;

namespace
{
TensorFactorSpaceShape makeShape(const std::vector<int> &cardinalities)
{
  std::vector<IndexType> strides(cardinalities.size(), 1);
  IndexType totalDofs = 1;
  for (int d=0; d<static_cast<int>(cardinalities.size()); d++)
  {
    strides[d] = totalDofs;
    totalDofs *= cardinalities[d];
  }
  return TensorFactorSpaceShape(cardinalities, strides, totalDofs);
}

TEUCHOS_UNIT_TEST( TensorProductLocalOperator, EntryIsProductOfComponentEntries )
{
  TensorComponentLocalMatrix component0(2, 2, {1.0, 2.0,
                                               3.0, 4.0});
  TensorComponentLocalMatrix component1(3, 2, { 5.0,  6.0,
                                                7.0,  8.0,
                                                9.0, 10.0});
  TensorProductLocalOperator op(makeShape({2,2}), makeShape({2,3}), {component0, component1});

  // range ordinal 3 -> (1,1), domain ordinal 2 -> (0,1), lower component fastest.
  TEST_EQUALITY(op.entry(3, 2), component0.entry(1,0) * component1.entry(1,1));
}

TEUCHOS_UNIT_TEST( TensorProductLocalOperator, ApplyMatchesExplicitDenseProduct )
{
  TensorComponentLocalMatrix component0(2, 2, {1.0, 2.0,
                                               3.0, 4.0});
  TensorComponentLocalMatrix component1(2, 3, { 0.5,  1.0, -1.0,
                                                2.0, -0.5,  1.5});
  TensorProductLocalOperator op(makeShape({2,3}), makeShape({2,2}), {component0, component1});

  std::vector<double> input(op.domainSize(), 0.0);
  for (IndexType i=0; i<op.domainSize(); i++) input[i] = 0.25 * (i + 1);

  std::vector<double> actual;
  op.apply(input, actual);

  TEST_EQUALITY(actual.size(), op.rangeSize());
  for (IndexType row=0; row<op.rangeSize(); row++)
  {
    double expected = 0.0;
    for (IndexType col=0; col<op.domainSize(); col++)
    {
      expected += op.entry(row, col) * input[col];
    }
    TEST_FLOATING_EQUALITY(actual[row], expected, 1e-15);
  }
}

TEUCHOS_UNIT_TEST( TensorProductLocalOperator, RejectsMismatchedComponentMatrixShape )
{
  TensorComponentLocalMatrix badComponent(1, 2, {1.0, 2.0});
  TEST_THROW(TensorProductLocalOperator(makeShape({2}), makeShape({2}), {badComponent}), std::invalid_argument);
}
}
