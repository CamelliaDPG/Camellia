#include "TensorDofLayout.h"
#include "Basis.h"
#include "Var.h"

#include <algorithm>
#include <functional>
#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  std::vector<std::vector<int>> enumerateComponentOrdinalsByCellOrdering(const TensorDofLayout::VariableLayout &layout)
  {
    const int componentCount = layout.componentCount();
    std::vector<std::vector<int>> tuples;
    std::vector<int> current(componentCount, 0);

    std::function<void(int)> recurse = [&](int componentOrdinal)
    {
      if (componentOrdinal < 0)
      {
        tuples.push_back(current);
        return;
      }
      for (int basisOrdinal=0; basisOrdinal<layout.componentCardinality(componentOrdinal); basisOrdinal++)
      {
        current[componentOrdinal] = basisOrdinal;
        recurse(componentOrdinal - 1);
      }
    };

    recurse(componentCount - 1);
    return tuples;
  }
}

TensorDofLayout::VariableLayout TensorDofLayout::buildVariableLayout(const TensorVarDiscretizationPtr &discretization, TensorSpaceRole role) const
{
  const int componentCount = discretization->componentCount();
  std::vector<int> componentCardinalities(componentCount, 0);
  std::vector<IndexType> componentStrides(componentCount, 1);
  std::vector<TensorComponentSupportKind> componentSupportKinds(componentCount, TENSOR_VOLUME_BASIS_SUPPORT);

  const auto tensorVar = discretization->tensorVar();
  const bool isTrace = tensorVar->isTrace();
  const int traceComponentOrdinal = tensorVar->hasTraceComponentOrdinal() ? tensorVar->traceComponentOrdinal() : -1;

  for (int d=0; d<componentCount; d++)
  {
    auto componentTopo = _tensorMesh->topology()->componentMeshTopology(d);
    const auto &activeCells = componentTopo->activeCells();
    TEUCHOS_TEST_FOR_EXCEPTION(activeCells.empty(), std::invalid_argument,
                               "TensorDofLayout requires at least one active cell in each component topology.");

    if (isTrace && (d == traceComponentOrdinal))
    {
      componentSupportKinds[d] = TENSOR_TRACE_SKELETON_SUPPORT;
      // For issue #300, the distinguished trace component contributes a skeleton-support extent
      // rather than a volume basis cardinality. In the present implementation we use the active
      // component-vertex extent as the trace-local support factor. This gives a coherent first
      // ordinal/layout interpretation while leaving room for later trace-basis and side-local
      // operator realization work to refine how those support entities are used numerically.
      componentCardinalities[d] = componentTopo->vertexCount();
    }
    else
    {
      const ComponentCellID1D &sampleCell = activeCells[0];
      const int effectivePolyOrder = _tensorMesh->effectivePolyOrder(discretization, sampleCell, d, role == TENSOR_TEST_SPACE);
      const auto var = tensorVar->componentVar(d);
      auto basisFamily = _tensorMesh->effectiveBasisFamily(discretization);
      auto basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(var->space()), effectivePolyOrder);
      componentCardinalities[d] = basis->getCardinality();
      componentSupportKinds[d] = TENSOR_VOLUME_BASIS_SUPPORT;
    }
  }

  IndexType runningStride = 1;
  for (int d=0; d<componentCount; d++)
  {
    componentStrides[d] = runningStride;
    runningStride *= componentCardinalities[d];
  }

  return VariableLayout(discretization->tensorVar()->name(), role, componentCardinalities, componentStrides,
                        componentSupportKinds, traceComponentOrdinal, runningStride);
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

IndexType TensorDofLayout::variableLocalDofOrdinal(const std::string &name,
                                                   TensorSpaceRole role,
                                                   const std::vector<int> &componentBasisOrdinals) const
{
  const auto &varLayout = layout(name, role);
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentBasisOrdinals.size()) != varLayout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::variableLocalDofOrdinal requires one component basis ordinal per component.");

  IndexType ordinal = 0;
  for (int d=0; d<varLayout.componentCount(); d++)
  {
    const int basisOrdinal = componentBasisOrdinals[d];
    TEUCHOS_TEST_FOR_EXCEPTION((basisOrdinal < 0) || (basisOrdinal >= varLayout.componentCardinality(d)), std::invalid_argument,
                               "TensorDofLayout::variableLocalDofOrdinal basis ordinal out of range.");
    ordinal += basisOrdinal * varLayout.componentStride(d);
  }
  return ordinal;
}

IndexType TensorDofLayout::cellLocalDofOrdinal(const std::string &name,
                                               TensorSpaceRole role,
                                               const std::vector<int> &componentBasisOrdinals) const
{
  const auto &varLayout = layout(name, role);
  const auto tuples = enumerateComponentOrdinalsByCellOrdering(varLayout);
  for (IndexType ordinal=0; ordinal<static_cast<IndexType>(tuples.size()); ordinal++)
  {
    if (tuples[ordinal] == componentBasisOrdinals) return ordinal;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                             "TensorDofLayout::cellLocalDofOrdinal basis ordinal tuple out of range.");
}

IndexType TensorDofLayout::localDofCount(const std::string &name,
                                         TensorSpaceRole role,
                                         TensorLocalOrdering ordering) const
{
  const auto &varLayout = layout(name, role);
  (void) ordering;
  return varLayout.totalDofs();
}

IndexType TensorDofLayout::localDofOrdinal(const std::string &name,
                                           TensorSpaceRole role,
                                           const std::vector<int> &componentBasisOrdinals) const
{
  return variableLocalDofOrdinal(name, role, componentBasisOrdinals);
}

IndexType TensorDofLayout::localDofOrdinal(const std::string &name,
                                           TensorSpaceRole role,
                                           TensorLocalOrdering ordering,
                                           const std::vector<int> &componentBasisOrdinals) const
{
  if (ordering == TENSOR_VARIABLE_ORDERING)
  {
    return variableLocalDofOrdinal(name, role, componentBasisOrdinals);
  }
  else
  {
    return cellLocalDofOrdinal(name, role, componentBasisOrdinals);
  }
}

void TensorDofLayout::decodeVariableLocalDofOrdinal(const std::string &name,
                                                    TensorSpaceRole role,
                                                    IndexType ordinal,
                                                    std::vector<int> &componentBasisOrdinals) const
{
  const auto &varLayout = layout(name, role);
  TEUCHOS_TEST_FOR_EXCEPTION((ordinal < 0) || (ordinal >= varLayout.totalDofs()), std::invalid_argument,
                             "TensorDofLayout::decodeVariableLocalDofOrdinal ordinal out of range.");

  componentBasisOrdinals.assign(varLayout.componentCount(), 0);
  for (int d=varLayout.componentCount()-1; d>=0; d--)
  {
    const IndexType stride = varLayout.componentStride(d);
    const int cardinality = varLayout.componentCardinality(d);
    componentBasisOrdinals[d] = static_cast<int>((ordinal / stride) % cardinality);
  }
}

void TensorDofLayout::decodeCellLocalDofOrdinal(const std::string &name,
                                                TensorSpaceRole role,
                                                IndexType ordinal,
                                                std::vector<int> &componentBasisOrdinals) const
{
  const auto &varLayout = layout(name, role);
  TEUCHOS_TEST_FOR_EXCEPTION((ordinal < 0) || (ordinal >= varLayout.totalDofs()), std::invalid_argument,
                             "TensorDofLayout::decodeCellLocalDofOrdinal ordinal out of range.");
  const auto tuples = enumerateComponentOrdinalsByCellOrdering(varLayout);
  componentBasisOrdinals = tuples[ordinal];
}

void TensorDofLayout::decodeLocalDofOrdinal(const std::string &name,
                                            TensorSpaceRole role,
                                            TensorLocalOrdering ordering,
                                            IndexType ordinal,
                                            std::vector<int> &componentBasisOrdinals) const
{
  if (ordering == TENSOR_VARIABLE_ORDERING)
  {
    decodeVariableLocalDofOrdinal(name, role, ordinal, componentBasisOrdinals);
  }
  else
  {
    decodeCellLocalDofOrdinal(name, role, ordinal, componentBasisOrdinals);
  }
}
