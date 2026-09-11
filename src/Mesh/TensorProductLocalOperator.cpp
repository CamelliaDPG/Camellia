#include "TensorProductLocalOperator.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorComponentLocalMatrix::TensorComponentLocalMatrix(int rowCount,
                                                       int columnCount,
                                                       const std::vector<double> &entries)
: _rowCount(rowCount),
  _columnCount(columnCount),
  _entries(entries)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_rowCount < 0, std::invalid_argument,
                             "TensorComponentLocalMatrix requires nonnegative row count.");
  TEUCHOS_TEST_FOR_EXCEPTION(_columnCount < 0, std::invalid_argument,
                             "TensorComponentLocalMatrix requires nonnegative column count.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(_entries.size()) != _rowCount * _columnCount, std::invalid_argument,
                             "TensorComponentLocalMatrix entry count must match rowCount * columnCount.");
}

double TensorComponentLocalMatrix::entry(int rowOrdinal,
                                         int columnOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((rowOrdinal < 0) || (rowOrdinal >= _rowCount), std::invalid_argument,
                             "TensorComponentLocalMatrix::entry row ordinal out of range.");
  TEUCHOS_TEST_FOR_EXCEPTION((columnOrdinal < 0) || (columnOrdinal >= _columnCount), std::invalid_argument,
                             "TensorComponentLocalMatrix::entry column ordinal out of range.");
  return _entries[rowOrdinal * _columnCount + columnOrdinal];
}

TensorProductLocalOperator::TensorProductLocalOperator(const TensorFactorSpaceShape &domainShape,
                                                       const TensorFactorSpaceShape &rangeShape,
                                                       const std::vector<TensorComponentLocalMatrix> &componentMatrices)
: _domainShape(domainShape),
  _rangeShape(rangeShape),
  _componentMatrices(componentMatrices)
{
  validate();
}

void TensorProductLocalOperator::validate() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_domainShape.componentCount() != _rangeShape.componentCount(), std::invalid_argument,
                             "TensorProductLocalOperator requires domain/range shapes with the same tensor rank.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(_componentMatrices.size()) != _domainShape.componentCount(), std::invalid_argument,
                             "TensorProductLocalOperator requires one component matrix per tensor component.");
  for (int d=0; d<_domainShape.componentCount(); d++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_componentMatrices[d].columnCount() != _domainShape.componentCardinality(d), std::invalid_argument,
                               "TensorProductLocalOperator component matrix column count must match domain component cardinality.");
    TEUCHOS_TEST_FOR_EXCEPTION(_componentMatrices[d].rowCount() != _rangeShape.componentCardinality(d), std::invalid_argument,
                               "TensorProductLocalOperator component matrix row count must match range component cardinality.");
  }
}

void TensorProductLocalOperator::decodeOrdinal(const TensorFactorSpaceShape &shape,
                                               IndexType ordinal,
                                               std::vector<int> &componentOrdinals)
{
  TEUCHOS_TEST_FOR_EXCEPTION((ordinal < 0) || (ordinal >= shape.totalDofs()), std::invalid_argument,
                             "TensorProductLocalOperator::decodeOrdinal ordinal out of range.");
  componentOrdinals.assign(shape.componentCount(), 0);
  for (int d=0; d<shape.componentCount(); d++)
  {
    componentOrdinals[d] = static_cast<int>((ordinal / shape.componentStride(d)) % shape.componentCardinality(d));
  }
}

int TensorProductLocalOperator::componentCount() const
{
  return _domainShape.componentCount();
}

IndexType TensorProductLocalOperator::domainSize() const
{
  return _domainShape.totalDofs();
}

IndexType TensorProductLocalOperator::rangeSize() const
{
  return _rangeShape.totalDofs();
}

const TensorFactorSpaceShape &TensorProductLocalOperator::domainShape() const
{
  return _domainShape;
}

const TensorFactorSpaceShape &TensorProductLocalOperator::rangeShape() const
{
  return _rangeShape;
}

const TensorComponentLocalMatrix &TensorProductLocalOperator::componentMatrix(int componentOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= componentCount()), std::invalid_argument,
                             "TensorProductLocalOperator::componentMatrix component ordinal out of range.");
  return _componentMatrices[componentOrdinal];
}

double TensorProductLocalOperator::entry(IndexType rangeOrdinal,
                                         IndexType domainOrdinal) const
{
  std::vector<int> rangeComponentOrdinals;
  std::vector<int> domainComponentOrdinals;
  decodeOrdinal(_rangeShape, rangeOrdinal, rangeComponentOrdinals);
  decodeOrdinal(_domainShape, domainOrdinal, domainComponentOrdinals);

  double value = 1.0;
  for (int d=0; d<componentCount(); d++)
  {
    value *= _componentMatrices[d].entry(rangeComponentOrdinals[d], domainComponentOrdinals[d]);
  }
  return value;
}

void TensorProductLocalOperator::apply(const std::vector<double> &input,
                                       std::vector<double> &output) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != domainSize(), std::invalid_argument,
                             "TensorProductLocalOperator::apply input size must match domain size.");

  output.assign(rangeSize(), 0.0);
  for (IndexType rangeOrdinal=0; rangeOrdinal<rangeSize(); rangeOrdinal++)
  {
    double value = 0.0;
    for (IndexType domainOrdinal=0; domainOrdinal<domainSize(); domainOrdinal++)
    {
      value += entry(rangeOrdinal, domainOrdinal) * input[domainOrdinal];
    }
    output[rangeOrdinal] = value;
  }
}
