#include "ComponentTransferOperator1D.h"

#include <algorithm>
#include <stdexcept>

#include "Basis.h"
#include "Var.h"
#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void ComponentTransferOperator1D::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseMesh == Teuchos::null, std::invalid_argument,
                             "ComponentTransferOperator1D requires a non-null coarse TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_fineMesh == Teuchos::null, std::invalid_argument,
                             "ComponentTransferOperator1D requires a non-null fine TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseLayout == Teuchos::null, std::invalid_argument,
                             "ComponentTransferOperator1D requires a non-null coarse TensorDofLayout.");
  TEUCHOS_TEST_FOR_EXCEPTION(_fineLayout == Teuchos::null, std::invalid_argument,
                             "ComponentTransferOperator1D requires a non-null fine TensorDofLayout.");
  TEUCHOS_TEST_FOR_EXCEPTION((_componentOrdinal < 0) || (_componentOrdinal >= _coarseMesh->topology()->componentCount()), std::invalid_argument,
                             "ComponentTransferOperator1D component ordinal out of range for coarse mesh.");
  TEUCHOS_TEST_FOR_EXCEPTION((_componentOrdinal < 0) || (_componentOrdinal >= _fineMesh->topology()->componentCount()), std::invalid_argument,
                             "ComponentTransferOperator1D component ordinal out of range for fine mesh.");
}

void ComponentTransferOperator1D::buildPTransferOperatorMatrix()
{
  const auto coarseShape = _coarseLayout->ambientShape(_varName, _role);
  const auto fineShape   = _fineLayout->ambientShape(_varName, _role);
  const IndexType coarseComponentCardinality = coarseShape.componentCardinality(_componentOrdinal);
  const IndexType fineComponentCardinality   = fineShape.componentCardinality(_componentOrdinal);

  _operatorMatrix.assign(fineComponentCardinality, std::vector<double>(coarseComponentCardinality, 0.0));

  const auto coarseDiscretization = (_role == TENSOR_TRIAL_SPACE)
      ? _coarseMesh->trialDiscretization(_varName)
      : _coarseMesh->testDiscretization(_varName);
  const auto fineDiscretization = (_role == TENSOR_TRIAL_SPACE)
      ? _fineMesh->trialDiscretization(_varName)
      : _fineMesh->testDiscretization(_varName);

  const auto coarseTopo = _coarseMesh->topology()->componentMeshTopology(_componentOrdinal);
  const auto fineTopo   = _fineMesh->topology()->componentMeshTopology(_componentOrdinal);

  const auto &coarseCells = coarseTopo->activeCells();
  const auto &fineCells   = fineTopo->activeCells();
  TEUCHOS_TEST_FOR_EXCEPTION(coarseCells.empty() || fineCells.empty(), std::invalid_argument,
                             "ComponentTransferOperator1D requires nonempty active-cell sets in the selected component.");

  const ComponentCellID1D coarseCell = coarseCells[0];
  const ComponentCellID1D fineCell   = fineCells[0];

  const bool isTest = (_role == TENSOR_TEST_SPACE);
  const int coarsePolyOrder = _coarseMesh->effectivePolyOrder(coarseDiscretization, coarseCell, _componentOrdinal, isTest);
  const int finePolyOrder   = _fineMesh->effectivePolyOrder(fineDiscretization,   fineCell,   _componentOrdinal, isTest);

  auto coarseBasisFamily = _coarseMesh->effectiveBasisFamily(coarseDiscretization);
  auto fineBasisFamily   = _fineMesh->effectiveBasisFamily(fineDiscretization);
  auto coarseVar = coarseDiscretization->tensorVar()->componentVar(_componentOrdinal);
  auto fineVar   = fineDiscretization->tensorVar()->componentVar(_componentOrdinal);

  auto coarseBasis = coarseBasisFamily->getBasis(CellTopology::line(), efsForSpace(coarseVar->space()), coarsePolyOrder);
  auto fineBasis   = fineBasisFamily->getBasis(CellTopology::line(), efsForSpace(fineVar->space()), finePolyOrder);

  (void) coarseBasis;
  (void) fineBasis;

  const IndexType diagCount = std::min(fineComponentCardinality, coarseComponentCardinality);
  for (IndexType i=0; i<diagCount; i++)
  {
    _operatorMatrix[i][i] = 1.0;
  }
}

void ComponentTransferOperator1D::buildHTransferOperatorMatrix()
{
  const auto coarseShape = _coarseLayout->ambientShape(_varName, _role);
  const auto fineShape   = _fineLayout->ambientShape(_varName, _role);
  const IndexType coarseComponentCardinality = coarseShape.componentCardinality(_componentOrdinal);
  const IndexType fineComponentCardinality   = fineShape.componentCardinality(_componentOrdinal);

  _operatorMatrix.assign(fineComponentCardinality, std::vector<double>(coarseComponentCardinality, 0.0));

  const IndexType diagCount = std::min(fineComponentCardinality, coarseComponentCardinality);
  for (IndexType i=0; i<diagCount; i++)
  {
    _operatorMatrix[i][i] = 1.0;
  }
}

void ComponentTransferOperator1D::buildOperatorMatrix()
{
  if (_transferType == COMPONENT_PROLONGATION)
  {
    buildPTransferOperatorMatrix();
  }
  else
  {
    buildHTransferOperatorMatrix();
  }
}

ComponentTransferOperator1D::ComponentTransferOperator1D(TensorMeshPtr coarseMesh,
                                                         TensorMeshPtr fineMesh,
                                                         TensorDofLayoutPtr coarseLayout,
                                                         TensorDofLayoutPtr fineLayout,
                                                         TensorSpaceRole role,
                                                         const std::string &varName,
                                                         int componentOrdinal,
                                                         ComponentTransferType transferType)
: _coarseMesh(coarseMesh),
  _fineMesh(fineMesh),
  _coarseLayout(coarseLayout),
  _fineLayout(fineLayout),
  _role(role),
  _varName(varName),
  _componentOrdinal(componentOrdinal),
  _transferType(transferType)
{
  validateConstructorInputs();
  buildOperatorMatrix();
}

TensorMeshPtr ComponentTransferOperator1D::coarseMesh() const
{
  return _coarseMesh;
}

TensorMeshPtr ComponentTransferOperator1D::fineMesh() const
{
  return _fineMesh;
}

TensorSpaceRole ComponentTransferOperator1D::role() const
{
  return _role;
}

const std::string &ComponentTransferOperator1D::varName() const
{
  return _varName;
}

int ComponentTransferOperator1D::componentOrdinal() const
{
  return _componentOrdinal;
}

ComponentTransferType ComponentTransferOperator1D::transferType() const
{
  return _transferType;
}

IndexType ComponentTransferOperator1D::rowCount() const
{
  return static_cast<IndexType>(_operatorMatrix.size());
}

IndexType ComponentTransferOperator1D::colCount() const
{
  return _operatorMatrix.empty() ? 0 : static_cast<IndexType>(_operatorMatrix[0].size());
}

double ComponentTransferOperator1D::entry(IndexType rowOrdinal, IndexType colOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((rowOrdinal < 0) || (rowOrdinal >= rowCount()), std::invalid_argument,
                             "ComponentTransferOperator1D::entry row ordinal out of range.");
  TEUCHOS_TEST_FOR_EXCEPTION((colOrdinal < 0) || (colOrdinal >= colCount()), std::invalid_argument,
                             "ComponentTransferOperator1D::entry column ordinal out of range.");
  return _operatorMatrix[rowOrdinal][colOrdinal];
}

const std::vector<std::vector<double>> &ComponentTransferOperator1D::operatorMatrix() const
{
  return _operatorMatrix;
}

void ComponentTransferOperator1D::apply(const std::vector<double> &input,
                                        std::vector<double> &output) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != colCount(), std::invalid_argument,
                             "ComponentTransferOperator1D::apply input vector size does not match column count.");

  output.assign(rowCount(), 0.0);
  for (IndexType i=0; i<rowCount(); i++)
  {
    for (IndexType j=0; j<colCount(); j++)
    {
      output[i] += entry(i,j) * input[j];
    }
  }
}
