#include "TensorTransferOperator.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void TensorTransferOperator::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseMesh == Teuchos::null, std::invalid_argument,
                             "TensorTransferOperator requires a non-null coarse TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_fineMesh == Teuchos::null, std::invalid_argument,
                             "TensorTransferOperator requires a non-null fine TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseLayout == Teuchos::null, std::invalid_argument,
                             "TensorTransferOperator requires a non-null coarse TensorDofLayout.");
  TEUCHOS_TEST_FOR_EXCEPTION(_fineLayout == Teuchos::null, std::invalid_argument,
                             "TensorTransferOperator requires a non-null fine TensorDofLayout.");
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseMesh->topology()->componentCount() != _fineMesh->topology()->componentCount(), std::invalid_argument,
                             "TensorTransferOperator requires coarse and fine tensor meshes of the same rank.");
}

TensorTransferOperator::TensorTransferOperator(TensorMeshPtr coarseMesh,
                                               TensorMeshPtr fineMesh,
                                               TensorDofLayoutPtr coarseLayout,
                                               TensorDofLayoutPtr fineLayout,
                                               TensorSpaceRole role,
                                               const std::string &varName,
                                               ComponentTransferType transferType)
: _coarseMesh(coarseMesh), _fineMesh(fineMesh), _coarseLayout(coarseLayout), _fineLayout(fineLayout),
  _role(role), _varName(varName), _transferType(transferType)
{
  validateConstructorInputs();
  const int componentCount_ = _coarseMesh->topology()->componentCount();
  _componentTransfers.reserve(componentCount_);
  for (int d=0; d<componentCount_; d++)
  {
    _componentTransfers.push_back(Teuchos::rcp(new ComponentTransferOperator1D(_coarseMesh, _fineMesh, _coarseLayout, _fineLayout,
                                                                               _role, _varName, d, _transferType)));
  }
}

TensorMeshPtr TensorTransferOperator::coarseMesh() const
{
  return _coarseMesh;
}

TensorMeshPtr TensorTransferOperator::fineMesh() const
{
  return _fineMesh;
}

TensorSpaceRole TensorTransferOperator::role() const
{
  return _role;
}

const std::string &TensorTransferOperator::varName() const
{
  return _varName;
}

ComponentTransferType TensorTransferOperator::transferType() const
{
  return _transferType;
}

int TensorTransferOperator::componentCount() const
{
  return static_cast<int>(_componentTransfers.size());
}

ComponentTransferOperator1DPtr TensorTransferOperator::componentTransfer(int componentOrdinal) const
{
  return _componentTransfers[componentOrdinal];
}

IndexType TensorTransferOperator::rowCount() const
{
  const auto shape = _fineLayout->ambientShape(_varName, _role);
  return shape.totalDofs();
}

IndexType TensorTransferOperator::colCount() const
{
  const auto shape = _coarseLayout->ambientShape(_varName, _role);
  return shape.totalDofs();
}

void TensorTransferOperator::apply(const std::vector<double> &input,
                                   std::vector<double> &output) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != colCount(), std::invalid_argument,
                             "TensorTransferOperator::apply input vector size does not match operator column count.");

  output.assign(rowCount(), 0.0);

  const auto coarseShape = _coarseLayout->ambientShape(_varName, _role);
  const auto fineShape   = _fineLayout->ambientShape(_varName, _role);
  const int dims = componentCount();

  std::vector<IndexType> coarseStrides(dims,1), fineStrides(dims,1);
  for (int d=1; d<dims; d++)
  {
    coarseStrides[d] = coarseStrides[d-1] * coarseShape.componentCardinality(d-1);
    fineStrides[d]   = fineStrides[d-1]   * fineShape.componentCardinality(d-1);
  }

  for (IndexType fineOrdinal=0; fineOrdinal<rowCount(); fineOrdinal++)
  {
    std::vector<IndexType> fineComponentOrdinals(dims,0);
    IndexType remainder = fineOrdinal;
    for (int d=dims-1; d>=0; d--)
    {
      fineComponentOrdinals[d] = remainder / fineStrides[d];
      remainder %= fineStrides[d];
    }

    for (IndexType coarseOrdinal=0; coarseOrdinal<colCount(); coarseOrdinal++)
    {
      std::vector<IndexType> coarseComponentOrdinals(dims,0);
      IndexType coarseRemainder = coarseOrdinal;
      for (int d=dims-1; d>=0; d--)
      {
        coarseComponentOrdinals[d] = coarseRemainder / coarseStrides[d];
        coarseRemainder %= coarseStrides[d];
      }

      double weight = 1.0;
      for (int d=0; d<dims; d++)
      {
        weight *= _componentTransfers[d]->entry(fineComponentOrdinals[d], coarseComponentOrdinals[d]);
        if (weight == 0.0) break;
      }
      output[fineOrdinal] += weight * input[coarseOrdinal];
    }
  }
}
