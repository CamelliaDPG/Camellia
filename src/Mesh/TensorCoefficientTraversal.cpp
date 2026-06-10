#include "TensorCoefficientTraversal.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorCoefficientTraversal::TensorCoefficientTraversal(TensorTraversalKind kind,
                                                       TensorSpaceRole role,
                                                       const std::string &varName,
                                                       TensorLocalOrdering ordering,
                                                       const TensorDofLayout::VariableLayout &variableLayout,
                                                       const TensorFactorSpaceShape &shape,
                                                       const std::vector<IndexType> &componentStrides,
                                                       const std::vector<IndexType> &componentFactorBaseCoordinates,
                                                       bool isTraceTraversal,
                                                       int traceComponentOrdinal,
                                                       IndexType traceSupportCoordinate)
: _kind(kind),
  _role(role),
  _varName(varName),
  _ordering(ordering),
  _variableLayout(variableLayout),
  _shape(shape),
  _componentStrides(componentStrides),
  _componentFactorBaseCoordinates(componentFactorBaseCoordinates),
  _isTraceTraversal(isTraceTraversal),
  _traceComponentOrdinal(traceComponentOrdinal),
  _traceSupportCoordinate(traceSupportCoordinate)
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(_componentStrides.size()) != _shape.componentCount(), std::invalid_argument,
                             "TensorCoefficientTraversal requires one global component stride per tensor component.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(_componentFactorBaseCoordinates.size()) != _shape.componentCount(), std::invalid_argument,
                             "TensorCoefficientTraversal requires one base coordinate per tensor component.");
  TEUCHOS_TEST_FOR_EXCEPTION(_shape.componentCount() != _variableLayout.componentCount(), std::invalid_argument,
                             "TensorCoefficientTraversal shape rank must match variable layout rank.");
  if (_isTraceTraversal)
  {
    TEUCHOS_TEST_FOR_EXCEPTION((_traceComponentOrdinal < 0) || (_traceComponentOrdinal >= _shape.componentCount()), std::invalid_argument,
                               "TensorCoefficientTraversal trace component ordinal out of range.");
    TEUCHOS_TEST_FOR_EXCEPTION(_traceSupportCoordinate < 0, std::invalid_argument,
                               "TensorCoefficientTraversal trace traversal requires a valid trace support coordinate.");
  }
}

TensorTraversalKind TensorCoefficientTraversal::kind() const
{
  return _kind;
}

TensorSpaceRole TensorCoefficientTraversal::role() const
{
  return _role;
}

const std::string &TensorCoefficientTraversal::varName() const
{
  return _varName;
}

TensorLocalOrdering TensorCoefficientTraversal::ordering() const
{
  return _ordering;
}

const TensorDofLayout::VariableLayout &TensorCoefficientTraversal::variableLayout() const
{
  return _variableLayout;
}

const TensorFactorSpaceShape &TensorCoefficientTraversal::shape() const
{
  return _shape;
}

bool TensorCoefficientTraversal::isTraceTraversal() const
{
  return _isTraceTraversal;
}

int TensorCoefficientTraversal::traceComponentOrdinal() const
{
  return _traceComponentOrdinal;
}

IndexType TensorCoefficientTraversal::traceSupportCoordinate() const
{
  return _traceSupportCoordinate;
}

IndexType TensorCoefficientTraversal::localDofCount() const
{
  return _shape.totalDofs();
}

IndexType TensorCoefficientTraversal::globalIndexForLocalOrdinal(IndexType localOrdinal,
                                                                 const TensorDofLayoutPtr &tensorDofLayout) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorCoefficientTraversal::globalIndexForLocalOrdinal requires a non-null TensorDofLayout.");
  TEUCHOS_TEST_FOR_EXCEPTION((localOrdinal < 0) || (localOrdinal >= localDofCount()), std::invalid_argument,
                             "TensorCoefficientTraversal::globalIndexForLocalOrdinal local ordinal out of range.");

  std::vector<int> componentBasisOrdinals;
  tensorDofLayout->decodeLocalDofOrdinal(_variableLayout, _shape, _ordering, localOrdinal, componentBasisOrdinals);

  IndexType globalOrdinal = 0;
  for (int d=0; d<_shape.componentCount(); d++)
  {
    IndexType componentCoordinate = _componentFactorBaseCoordinates[d] + componentBasisOrdinals[d];
    if (_isTraceTraversal && (d == _traceComponentOrdinal))
    {
      componentCoordinate = _traceSupportCoordinate;
    }
    globalOrdinal += componentCoordinate * _componentStrides[d];
  }
  return globalOrdinal;
}

std::vector<IndexType> TensorCoefficientTraversal::globalIndices(const TensorDofLayoutPtr &tensorDofLayout) const
{
  std::vector<IndexType> indices(localDofCount(), -1);
  for (IndexType localOrdinal=0; localOrdinal<localDofCount(); localOrdinal++)
  {
    indices[localOrdinal] = globalIndexForLocalOrdinal(localOrdinal, tensorDofLayout);
  }
  return indices;
}

bool TensorCoefficientTraversal::hasFastestComponentContiguousRuns() const
{
  return (_shape.componentCount() > 0) && (_componentStrides[0] == 1);
}

std::vector<TensorCoefficientBlock> TensorCoefficientTraversal::fastestComponentBlocks(const TensorDofLayoutPtr &tensorDofLayout) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorCoefficientTraversal::fastestComponentBlocks requires a non-null TensorDofLayout.");
  if (!hasFastestComponentContiguousRuns()) return {};
  if (_shape.componentCount() == 0) return {};

  const int fastestCount = _shape.componentCardinality(0);
  const IndexType outerCount = (fastestCount > 0) ? localDofCount() / fastestCount : 0;
  std::vector<TensorCoefficientBlock> blocks;
  blocks.reserve(static_cast<size_t>(outerCount));

  for (IndexType outerOrdinal=0; outerOrdinal<outerCount; outerOrdinal++)
  {
    IndexType localOrdinal = outerOrdinal * fastestCount;
    IndexType base = globalIndexForLocalOrdinal(localOrdinal, tensorDofLayout);
    blocks.push_back(TensorCoefficientBlock(base, 1, fastestCount));
  }
  return blocks;
}
