#include "ComponentTransferOperator1D.h"

#include <stdexcept>

#include "Basis.h"
#include "BasisFamily.h"
#include "Intrepid2_HierarchicalBasisProlongator.hpp"
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
  TEUCHOS_TEST_FOR_EXCEPTION(_componentOrdinal < 0 || _componentOrdinal >= _coarseMesh->topology()->componentCount(), std::invalid_argument,
                             "ComponentTransferOperator1D component ordinal out of range for coarse mesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_componentOrdinal < 0 || _componentOrdinal >= _fineMesh->topology()->componentCount(), std::invalid_argument,
                             "ComponentTransferOperator1D component ordinal out of range for fine mesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(!_coarseLayout->hasLayout(_varName, _role), std::invalid_argument,
                             "ComponentTransferOperator1D requires the variable to be registered in the coarse layout for the selected role.");
  TEUCHOS_TEST_FOR_EXCEPTION(!_fineLayout->hasLayout(_varName, _role), std::invalid_argument,
                             "ComponentTransferOperator1D requires the variable to be registered in the fine layout for the selected role.");
}

ComponentTransferOperator1D::ComponentTransferOperator1D(TensorMeshPtr coarseMesh,
                                                         TensorMeshPtr fineMesh,
                                                         TensorDofLayoutPtr coarseLayout,
                                                         TensorDofLayoutPtr fineLayout,
                                                         TensorSpaceRole role,
                                                         const std::string &varName,
                                                         int componentOrdinal,
                                                         ComponentTransferType transferType)
: _coarseMesh(coarseMesh), _fineMesh(fineMesh), _coarseLayout(coarseLayout), _fineLayout(fineLayout),
  _role(role), _varName(varName), _componentOrdinal(componentOrdinal), _transferType(transferType)
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
  if (_operatorMatrix.empty()) return 0;
  return static_cast<IndexType>(_operatorMatrix[0].size());
}

double ComponentTransferOperator1D::entry(IndexType row, IndexType col) const
{
  return _operatorMatrix[row][col];
}

const std::vector<std::vector<double>> &ComponentTransferOperator1D::operatorMatrix() const
{
  return _operatorMatrix;
}

void ComponentTransferOperator1D::buildOperatorMatrix()
{
  const auto &coarseVarLayout = _coarseLayout->layout(_varName, _role);
  const auto &fineVarLayout   = _fineLayout->layout(_varName, _role);

  const int coarseComponentCardinality = coarseVarLayout.componentCardinality(_componentOrdinal);
  const int fineComponentCardinality   = fineVarLayout.componentCardinality(_componentOrdinal);

  bool sameTopology = false;
  {
    auto coarseComponentTopo = _coarseMesh->topology()->componentMeshTopology(_componentOrdinal);
    auto fineComponentTopo   = _fineMesh->topology()->componentMeshTopology(_componentOrdinal);
    if (coarseComponentTopo->activeCellCount() == fineComponentTopo->activeCellCount())
    {
      sameTopology = true;
      for (IndexType i=0; i<coarseComponentTopo->activeCellCount(); i++)
      {
        if (!(coarseComponentTopo->activeCells()[i] == fineComponentTopo->activeCells()[i]))
        {
          sameTopology = false;
          break;
        }
      }
    }
  }

  if (sameTopology && (coarseComponentCardinality != fineComponentCardinality))
  {
    buildPTransferOperatorMatrix();
  }
  else
  {
    buildHTransferOperatorMatrix();
  }
}

void ComponentTransferOperator1D::buildPTransferOperatorMatrix()
{
  const auto coarseDisc = (_role == TENSOR_TRIAL_SPACE) ? _coarseMesh->trialDiscretization(_varName) : _coarseMesh->testDiscretization(_varName);
  const auto fineDisc   = (_role == TENSOR_TRIAL_SPACE) ? _fineMesh->trialDiscretization(_varName) : _fineMesh->testDiscretization(_varName);

  const auto coarseComponentTopo = _coarseMesh->topology()->componentMeshTopology(_componentOrdinal);
  const auto fineComponentTopo   = _fineMesh->topology()->componentMeshTopology(_componentOrdinal);

  const auto &coarseVarLayout = _coarseLayout->layout(_varName, _role);
  const auto &fineVarLayout   = _fineLayout->layout(_varName, _role);

  const int coarseP = _coarseMesh->effectivePolyOrder(coarseDisc, coarseComponentTopo->activeCells()[0], _componentOrdinal, _role == TENSOR_TEST_SPACE);
  const int fineP   = _fineMesh->effectivePolyOrder(fineDisc, fineComponentTopo->activeCells()[0], _componentOrdinal, _role == TENSOR_TEST_SPACE);

  auto basisFamily = _coarseMesh->effectiveBasisFamily(coarseDisc);
  auto coarseVar = coarseDisc->tensorVar()->componentVar(_componentOrdinal);
  auto fineVar   = fineDisc->tensorVar()->componentVar(_componentOrdinal);

  auto coarseBasis = basisFamily->getBasis(CellTopology::line(), efsForSpace(coarseVar->space()), coarseP);
  auto fineBasis   = basisFamily->getBasis(CellTopology::line(), efsForSpace(fineVar->space()), fineP);

  using DeviceType = DefaultTestDeviceType;
  using Scalar = double;
  auto prolongator = Intrepid2::HierarchicalBasisProlongator<DeviceType, Scalar>(coarseBasis, fineBasis);

  const int coarseCardinality = coarseBasis->getCardinality();
  const int fineCardinality   = fineBasis->getCardinality();

  Kokkos::View<double**,DeviceType> coarseBasisVectors("coarseBasisVectors", coarseCardinality, coarseCardinality);
  Kokkos::View<double**,DeviceType> fineBasisVectors("fineBasisVectors", coarseCardinality, fineCardinality);
  Kokkos::deep_copy(coarseBasisVectors, 0.0);

  auto coarseBasisVectorsHost = Kokkos::create_mirror_view(coarseBasisVectors);
  for (int i=0; i<coarseCardinality; i++)
  {
    coarseBasisVectorsHost(i,i) = 1.0;
  }
  Kokkos::deep_copy(coarseBasisVectors, coarseBasisVectorsHost);

  prolongator.apply(fineBasisVectors, coarseBasisVectors, false);

  auto fineBasisVectorsHost = Kokkos::create_mirror_view(fineBasisVectors);
  Kokkos::deep_copy(fineBasisVectorsHost, fineBasisVectors);

  const int rowCount_ = (_transferType == COMPONENT_PROLONGATION) ? fineCardinality : coarseCardinality;
  const int colCount_ = (_transferType == COMPONENT_PROLONGATION) ? coarseCardinality : fineCardinality;
  _operatorMatrix.assign(rowCount_, std::vector<double>(colCount_, 0.0));

  if (_transferType == COMPONENT_PROLONGATION)
  {
    for (int coarseOrdinal=0; coarseOrdinal<coarseCardinality; coarseOrdinal++)
    {
      for (int fineOrdinal=0; fineOrdinal<fineCardinality; fineOrdinal++)
      {
        _operatorMatrix[fineOrdinal][coarseOrdinal] = fineBasisVectorsHost(coarseOrdinal, fineOrdinal);
      }
    }
  }
  else
  {
    for (int coarseOrdinal=0; coarseOrdinal<coarseCardinality; coarseOrdinal++)
    {
      for (int fineOrdinal=0; fineOrdinal<fineCardinality; fineOrdinal++)
      {
        _operatorMatrix[coarseOrdinal][fineOrdinal] = fineBasisVectorsHost(coarseOrdinal, fineOrdinal);
      }
    }
  }
}

void ComponentTransferOperator1D::buildHTransferOperatorMatrix()
{
  const auto coarseComponentTopo = _coarseMesh->topology()->componentMeshTopology(_componentOrdinal);
  const auto fineComponentTopo   = _fineMesh->topology()->componentMeshTopology(_componentOrdinal);

  const auto &coarseVarLayout = _coarseLayout->layout(_varName, _role);
  const auto &fineVarLayout   = _fineLayout->layout(_varName, _role);

  const int coarseCardinality = coarseVarLayout.componentCardinality(_componentOrdinal);
  const int fineCardinality   = fineVarLayout.componentCardinality(_componentOrdinal);

  const int rowCount_ = (_transferType == COMPONENT_PROLONGATION)
                      ? static_cast<int>(fineComponentTopo->activeCellCount()) * fineCardinality
                      : static_cast<int>(coarseComponentTopo->activeCellCount()) * coarseCardinality;
  const int colCount_ = (_transferType == COMPONENT_PROLONGATION)
                      ? static_cast<int>(coarseComponentTopo->activeCellCount()) * coarseCardinality
                      : static_cast<int>(fineComponentTopo->activeCellCount()) * fineCardinality;

  _operatorMatrix.assign(rowCount_, std::vector<double>(colCount_, 0.0));

  if (_transferType == COMPONENT_PROLONGATION)
  {
    for (IndexType fineCellOrdinal=0; fineCellOrdinal<fineComponentTopo->activeCellCount(); fineCellOrdinal++)
    {
      const auto &fineCell = fineComponentTopo->activeCells()[fineCellOrdinal];
      ComponentCellID1D coarseCell = fineCell;
      while (coarseCell.level() > 0 && !coarseComponentTopo->containsCell(coarseCell))
      {
        coarseCell = coarseCell.parent();
      }
      TEUCHOS_TEST_FOR_EXCEPTION(!coarseComponentTopo->containsCell(coarseCell), std::invalid_argument,
                                 "ComponentTransferOperator1D h-prolongation could not find coarse ancestor cell.");

      IndexType coarseCellOrdinal = -1;
      for (IndexType i=0; i<coarseComponentTopo->activeCellCount(); i++)
      {
        if (coarseComponentTopo->activeCells()[i] == coarseCell)
        {
          coarseCellOrdinal = i;
          break;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(coarseCellOrdinal == -1, std::invalid_argument,
                                 "ComponentTransferOperator1D h-prolongation could not identify coarse cell ordinal.");

      for (int basisOrdinal=0; basisOrdinal<std::min(coarseCardinality,fineCardinality); basisOrdinal++)
      {
        IndexType row = fineCellOrdinal * fineCardinality + basisOrdinal;
        IndexType col = coarseCellOrdinal * coarseCardinality + basisOrdinal;
        _operatorMatrix[row][col] = 1.0;
      }
    }
  }
  else
  {
    for (IndexType coarseCellOrdinal=0; coarseCellOrdinal<coarseComponentTopo->activeCellCount(); coarseCellOrdinal++)
    {
      const auto &coarseCell = coarseComponentTopo->activeCells()[coarseCellOrdinal];
      std::vector<IndexType> matchingFineCellOrdinals;
      for (IndexType fineCellOrdinal=0; fineCellOrdinal<fineComponentTopo->activeCellCount(); fineCellOrdinal++)
      {
        const auto &fineCell = fineComponentTopo->activeCells()[fineCellOrdinal];
        if (coarseCell.isAncestorOf(fineCell))
        {
          matchingFineCellOrdinals.push_back(fineCellOrdinal);
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(matchingFineCellOrdinals.empty(), std::invalid_argument,
                                 "ComponentTransferOperator1D h-restriction found no matching fine cells for coarse cell.");

      const double weight = 1.0 / matchingFineCellOrdinals.size();
      for (int basisOrdinal=0; basisOrdinal<std::min(coarseCardinality,fineCardinality); basisOrdinal++)
      {
        IndexType row = coarseCellOrdinal * coarseCardinality + basisOrdinal;
        for (IndexType fineCellOrdinal : matchingFineCellOrdinals)
        {
          IndexType col = fineCellOrdinal * fineCardinality + basisOrdinal;
          _operatorMatrix[row][col] = weight;
        }
      }
    }
  }
}

void ComponentTransferOperator1D::apply(const std::vector<double> &input,
                                        std::vector<double> &output) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(input.size()) != colCount(), std::invalid_argument,
                             "ComponentTransferOperator1D::apply input vector size does not match operator column count.");

  output.assign(rowCount(), 0.0);
  for (IndexType row=0; row<rowCount(); row++)
  {
    for (IndexType col=0; col<colCount(); col++)
    {
      output[row] += _operatorMatrix[row][col] * input[col];
    }
  }
}
