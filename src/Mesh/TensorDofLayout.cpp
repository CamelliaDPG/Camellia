#include "TensorDofLayout.h"
#include "Basis.h"
#include "Var.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

TensorDofLayout::VariableLayout TensorDofLayout::buildVariableLayout(const TensorVarDiscretizationPtr &discretization, TensorSpaceRole role) const
{
  const int componentCount = discretization->componentCount();
  std::vector<int> componentCardinalities(componentCount, 0);
  std::vector<IndexType> componentStrides(componentCount, 1);

  for (int d=0; d<componentCount; d++)
  {
    auto componentTopo = _tensorMesh->topology()->componentMeshTopology(d);
    const auto &activeCells = componentTopo->activeCells();
    TEUCHOS_TEST_FOR_EXCEPTION(activeCells.empty(), std::invalid_argument,
                               "TensorDofLayout requires at least one active cell in each component topology.");

    const ComponentCellID1D &sampleCell = activeCells[0];
    const int effectivePolyOrder = _tensorMesh->effectivePolyOrder(discretization, sampleCell, d, role == TENSOR_TEST_SPACE);
    const auto var = discretization->tensorVar()->componentVar(d);
    auto basisFamily = _tensorMesh->effectiveBasisFamily(discretization);
    auto basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(var->space()), effectivePolyOrder);
    componentCardinalities[d] = basis->getCardinality();
  }

  IndexType runningStride = 1;
  for (int d=0; d<componentCount; d++)
  {
    componentStrides[d] = runningStride;
    runningStride *= componentCardinalities[d];
  }

  return VariableLayout(discretization->tensorVar()->name(), role, componentCardinalities, componentStrides, runningStride);
}

TensorDofLayout::TensorDofLayout(TensorMeshPtr tensorMesh)
: _tensorMesh(tensorMesh)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorDofLayout requires a non-null TensorMesh.");

  for (const auto &trialDisc : _tensorMesh->trialDiscretizations())
  {
    _trialLayouts[trialDisc->tensorVar()->name()] = buildVariableLayout(trialDisc, TENSOR_TRIAL_SPACE);
  }
  for (const auto &testDisc : _tensorMesh->testDiscretizations())
  {
    _testLayouts[testDisc->tensorVar()->name()] = buildVariableLayout(testDisc, TENSOR_TEST_SPACE);
  }
}

TensorMeshPtr TensorDofLayout::tensorMesh() const
{
  return _tensorMesh;
}

bool TensorDofLayout::hasLayout(const std::string &name, TensorSpaceRole role) const
{
  if (role == TENSOR_TRIAL_SPACE)
  {
    return _trialLayouts.find(name) != _trialLayouts.end();
  }
  else
  {
    return _testLayouts.find(name) != _testLayouts.end();
  }
}

const TensorDofLayout::VariableLayout &TensorDofLayout::layout(const std::string &name, TensorSpaceRole role) const
{
  if (role == TENSOR_TRIAL_SPACE)
  {
    auto entry = _trialLayouts.find(name);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _trialLayouts.end(), std::invalid_argument,
                               "Unknown trial TensorDofLayout variable name.");
    return entry->second;
  }
  else
  {
    auto entry = _testLayouts.find(name);
    TEUCHOS_TEST_FOR_EXCEPTION(entry == _testLayouts.end(), std::invalid_argument,
                               "Unknown test TensorDofLayout variable name.");
    return entry->second;
  }
}

std::vector<TensorDofLayout::VariableLayout> TensorDofLayout::trialLayouts() const
{
  std::vector<VariableLayout> layouts;
  for (const auto &entry : _trialLayouts)
  {
    layouts.push_back(entry.second);
  }
  return layouts;
}

std::vector<TensorDofLayout::VariableLayout> TensorDofLayout::testLayouts() const
{
  std::vector<VariableLayout> layouts;
  for (const auto &entry : _testLayouts)
  {
    layouts.push_back(entry.second);
  }
  return layouts;
}

IndexType TensorDofLayout::localDofOrdinal(const std::string &name,
                                           TensorSpaceRole role,
                                           const std::vector<int> &componentBasisOrdinals) const
{
  const auto &varLayout = layout(name, role);
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentBasisOrdinals.size()) != varLayout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::localDofOrdinal requires one component basis ordinal per component.");

  IndexType ordinal = 0;
  for (int d=0; d<varLayout.componentCount(); d++)
  {
    const int basisOrdinal = componentBasisOrdinals[d];
    TEUCHOS_TEST_FOR_EXCEPTION((basisOrdinal < 0) || (basisOrdinal >= varLayout.componentCardinality(d)), std::invalid_argument,
                               "TensorDofLayout::localDofOrdinal basis ordinal out of range.");
    ordinal += basisOrdinal * varLayout.componentStride(d);
  }
  return ordinal;
}
