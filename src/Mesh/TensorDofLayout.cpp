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
  TensorFactorSpaceShape makeFactorSpaceShapeFromCardinalities(const std::vector<int> &componentCardinalities)
  {
    std::vector<IndexType> componentStrides(componentCardinalities.size(), 1);
    IndexType runningStride = 1;
    for (int d=0; d<static_cast<int>(componentCardinalities.size()); d++)
    {
      componentStrides[d] = runningStride;
      runningStride *= componentCardinalities[d];
    }
    return TensorFactorSpaceShape(componentCardinalities, componentStrides, runningStride);
  }

  std::vector<std::vector<int>> enumerateComponentOrdinalsByCellOrdering(const TensorFactorSpaceShape &shape)
  {
    const int componentCount = shape.componentCount();
    std::vector<std::vector<int>> tuples;
    std::vector<int> current(componentCount, 0);

    std::function<void(int)> recurse = [&](int componentOrdinal)
    {
      if (componentOrdinal < 0)
      {
        tuples.push_back(current);
        return;
      }
      for (int basisOrdinal=0; basisOrdinal<shape.componentCardinality(componentOrdinal); basisOrdinal++)
      {
        current[componentOrdinal] = basisOrdinal;
        recurse(componentOrdinal - 1);
      }
    };

    recurse(componentCount - 1);
    return tuples;
  }
}

TensorDofLayout::VariableLayout TensorDofLayout::buildVariableLayout(const TensorVarDiscretizationPtr &discretization,
                                                                     TensorSpaceRole role) const
{
  const int componentCount = discretization->componentCount();
  std::vector<TensorComponentSupportKind> componentSupportKinds(componentCount, TENSOR_VOLUME_BASIS_SUPPORT);

  const auto tensorVar = discretization->tensorVar();
  const bool isTrace = tensorVar->isTrace();
  const int traceComponentOrdinal = tensorVar->hasTraceComponentOrdinal() ? tensorVar->traceComponentOrdinal() : -1;

  for (int d=0; d<componentCount; d++)
  {
    if (isTrace && (d == traceComponentOrdinal))
    {
      componentSupportKinds[d] = TENSOR_TRACE_SKELETON_SUPPORT;
    }
    else
    {
      componentSupportKinds[d] = TENSOR_VOLUME_BASIS_SUPPORT;
    }
  }

  return VariableLayout(discretization->tensorVar()->name(),
                        role,
                        componentCount,
                        componentSupportKinds,
                        traceComponentOrdinal);
}

TensorFactorSpaceShape TensorDofLayout::buildFactorSpaceShape(const TensorVarDiscretizationPtr &discretization,
                                                              TensorSpaceRole role,
                                                              const std::vector<ComponentCellID1D> &componentCells,
                                                              int orthogonalTraceComponentOrdinal,
                                                              const std::vector<int> &traceSupportComponentOrdinals) const
{
  const int componentCount = discretization->componentCount();
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentCells.size()) != componentCount, std::invalid_argument,
                             "TensorDofLayout::buildFactorSpaceShape requires one component cell per tensor component.");

  const auto tensorVar = discretization->tensorVar();
  const bool isTrace = tensorVar->isTrace();
  const int traceComponentOrdinal = tensorVar->hasTraceComponentOrdinal() ? tensorVar->traceComponentOrdinal() : -1;

  std::vector<int> componentCardinalities(componentCount, 0);

  for (int d=0; d<componentCount; d++)
  {
    const bool traceSupportHere = std::find(traceSupportComponentOrdinals.begin(),
                                            traceSupportComponentOrdinals.end(),
                                            d) != traceSupportComponentOrdinals.end();

    if (traceSupportHere)
    {
      componentCardinalities[d] = 1;
      continue;
    }

    if (isTrace && (d == traceComponentOrdinal) && (orthogonalTraceComponentOrdinal < 0))
    {
      componentCardinalities[d] = 1;
      continue;
    }

    const int effectivePolyOrder = _tensorMesh->effectivePolyOrder(discretization,
                                                                   componentCells[d],
                                                                   d,
                                                                   role == TENSOR_TEST_SPACE);
    const auto var = tensorVar->componentVar(d);
    auto basisFamily = _tensorMesh->effectiveBasisFamily(discretization);
    auto basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(var->space()), effectivePolyOrder);
    componentCardinalities[d] = basis->getCardinality();
  }

  return makeFactorSpaceShapeFromCardinalities(componentCardinalities);
}

TensorFactorSpaceShape TensorDofLayout::buildAmbientShape(const std::string &name,
                                                          TensorSpaceRole role) const
{
  const auto &variableLayout = layout(name, role);
  const int componentCount = variableLayout.componentCount();
  std::vector<int> maxima(componentCount, 0);

  if (_tensorMesh->topology()->activeCellTuples().empty())
  {
    return makeFactorSpaceShapeFromCardinalities(std::vector<int>(componentCount, 0));
  }

  for (const auto &cellTuple : _tensorMesh->topology()->activeCellTuples())
  {
    auto shape = factorSpaceShape(name, role, cellTuple);
    for (int d=0; d<componentCount; d++)
    {
      maxima[d] = std::max(maxima[d], shape.componentCardinality(d));
    }
  }

  if (variableLayout.hasTraceComponentOrdinal())
  {
    maxima[variableLayout.traceComponentOrdinal()] = 1;

    auto skeleton = _tensorMesh->topology()->tensorSkeleton(variableLayout.traceComponentOrdinal());
    for (const auto &entity : skeleton.entities())
    {
      auto shape = factorSpaceShape(name,
                                    role,
                                    entity.retainedCellTuple().empty() ? std::vector<ComponentCellID1D>(componentCount) : [&]() {
                                      std::vector<ComponentCellID1D> componentCells(componentCount);
                                      auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(variableLayout.traceComponentOrdinal());
                                      const IndexType begin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
                                      const ComponentCellID1D &incidentOrthogonalCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(begin)];
                                      for (int d=0; d<componentCount; d++) componentCells[d] = incidentOrthogonalCell;
                                      const auto retainedOrdinals = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(variableLayout.traceComponentOrdinal());
                                      for (size_t r=0; r<retainedOrdinals.size(); r++) componentCells[retainedOrdinals[r]] = entity.retainedCellTuple()[r];
                                      return componentCells;
                                    }(),
                                    variableLayout.traceComponentOrdinal(),
                                    {variableLayout.traceComponentOrdinal()});
      for (int d=0; d<componentCount; d++)
      {
        maxima[d] = std::max(maxima[d], shape.componentCardinality(d));
      }
    }
  }

  return makeFactorSpaceShapeFromCardinalities(maxima);
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

TensorFactorSpaceShape TensorDofLayout::factorSpaceShape(const std::string &name,
                                                         TensorSpaceRole role,
                                                         const std::vector<ComponentCellID1D> &componentCells,
                                                         int orthogonalTraceComponentOrdinal,
                                                         const std::vector<int> &traceSupportComponentOrdinals) const
{
  const auto discretization = (role == TENSOR_TRIAL_SPACE)
      ? _tensorMesh->trialDiscretization(name)
      : _tensorMesh->testDiscretization(name);
  return buildFactorSpaceShape(discretization,
                               role,
                               componentCells,
                               orthogonalTraceComponentOrdinal,
                               traceSupportComponentOrdinals);
}

TensorFactorSpaceShape TensorDofLayout::ambientShape(const std::string &name,
                                                     TensorSpaceRole role) const
{
  return buildAmbientShape(name, role);
}

TensorLocalVolumeContext TensorDofLayout::volumeContext(const std::string &trialVarName,
                                                        const std::string &testVarName,
                                                        const TensorMeshTopology::TensorCellTuple &cellTuple) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_tensorMesh->topology()->containsTensorCell(cellTuple), std::invalid_argument,
                             "TensorDofLayout::volumeContext requires a tensor cell tuple in the topology.");

  auto trialShape = factorSpaceShape(trialVarName, TENSOR_TRIAL_SPACE, cellTuple);
  auto testShape  = factorSpaceShape(testVarName,  TENSOR_TEST_SPACE,  cellTuple);
  return TensorLocalVolumeContext(cellTuple, trialShape, testShape);
}

TensorLocalSkeletonContext TensorDofLayout::skeletonContext(const std::string &trialVarName,
                                                            const std::string &testVarName,
                                                            const TensorMeshTopology::TensorSkeletonEntity &entity) const
{
  const int componentCount = _tensorMesh->topology()->componentCount();
  std::vector<ComponentCellID1D> trialCells(componentCount);
  std::vector<ComponentCellID1D> testCells(componentCount);

  const int orthogonalComponentOrdinal = entity.orthogonalComponentOrdinal();
  const auto retainedOrdinals = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);

  auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
  const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
  const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
  TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                             "TensorDofLayout::skeletonContext expected at least one incident orthogonal-component cell.");
  const ComponentCellID1D &incidentOrthogonalCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];

  for (int d=0; d<componentCount; d++)
  {
    trialCells[d] = incidentOrthogonalCell;
    testCells[d]  = incidentOrthogonalCell;
  }
  for (size_t retainedOrdinal=0; retainedOrdinal<retainedOrdinals.size(); retainedOrdinal++)
  {
    const int d = retainedOrdinals[retainedOrdinal];
    trialCells[d] = entity.retainedCellTuple()[retainedOrdinal];
    testCells[d]  = entity.retainedCellTuple()[retainedOrdinal];
  }

  std::vector<int> trialTraceSupportOrdinals;
  std::vector<int> testTraceSupportOrdinals;
  auto trialLayout = layout(trialVarName, TENSOR_TRIAL_SPACE);
  auto testLayout  = layout(testVarName,  TENSOR_TEST_SPACE);
  if (trialLayout.hasTraceComponentOrdinal()) trialTraceSupportOrdinals.push_back(trialLayout.traceComponentOrdinal());
  if (testLayout.hasTraceComponentOrdinal())  testTraceSupportOrdinals.push_back(testLayout.traceComponentOrdinal());

  auto trialShape = factorSpaceShape(trialVarName,
                                     TENSOR_TRIAL_SPACE,
                                     trialCells,
                                     orthogonalComponentOrdinal,
                                     trialTraceSupportOrdinals);
  auto testShape  = factorSpaceShape(testVarName,
                                     TENSOR_TEST_SPACE,
                                     testCells,
                                     orthogonalComponentOrdinal,
                                     testTraceSupportOrdinals);

  return TensorLocalSkeletonContext(entity, trialShape, testShape);
}

IndexType TensorDofLayout::variableLocalDofOrdinal(const VariableLayout &layout,
                                                   const TensorFactorSpaceShape &shape,
                                                   const std::vector<int> &componentBasisOrdinals) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentBasisOrdinals.size()) != layout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::variableLocalDofOrdinal requires one component basis ordinal per component.");
  TEUCHOS_TEST_FOR_EXCEPTION(shape.componentCount() != layout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::variableLocalDofOrdinal shape rank must match layout rank.");

  IndexType ordinal = 0;
  for (int d=0; d<layout.componentCount(); d++)
  {
    const int basisOrdinal = componentBasisOrdinals[d];
    TEUCHOS_TEST_FOR_EXCEPTION((basisOrdinal < 0) || (basisOrdinal >= shape.componentCardinality(d)), std::invalid_argument,
                               "TensorDofLayout::variableLocalDofOrdinal basis ordinal out of range.");
    ordinal += basisOrdinal * shape.componentStride(d);
  }
  return ordinal;
}

IndexType TensorDofLayout::cellLocalDofOrdinal(const VariableLayout &layout,
                                               const TensorFactorSpaceShape &shape,
                                               const std::vector<int> &componentBasisOrdinals) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentBasisOrdinals.size()) != layout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::cellLocalDofOrdinal requires one component basis ordinal per component.");
  TEUCHOS_TEST_FOR_EXCEPTION(shape.componentCount() != layout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::cellLocalDofOrdinal shape rank must match layout rank.");

  const auto tuples = enumerateComponentOrdinalsByCellOrdering(shape);
  for (IndexType ordinal=0; ordinal<static_cast<IndexType>(tuples.size()); ordinal++)
  {
    if (tuples[ordinal] == componentBasisOrdinals) return ordinal;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                             "TensorDofLayout::cellLocalDofOrdinal basis ordinal tuple out of range.");
}

IndexType TensorDofLayout::localDofCount(const TensorFactorSpaceShape &shape,
                                         TensorLocalOrdering ordering) const
{
  (void) ordering;
  return shape.totalDofs();
}

IndexType TensorDofLayout::localDofOrdinal(const VariableLayout &layout,
                                           const TensorFactorSpaceShape &shape,
                                           const std::vector<int> &componentBasisOrdinals) const
{
  return variableLocalDofOrdinal(layout, shape, componentBasisOrdinals);
}

IndexType TensorDofLayout::localDofOrdinal(const VariableLayout &layout,
                                           const TensorFactorSpaceShape &shape,
                                           TensorLocalOrdering ordering,
                                           const std::vector<int> &componentBasisOrdinals) const
{
  if (ordering == TENSOR_VARIABLE_ORDERING)
  {
    return variableLocalDofOrdinal(layout, shape, componentBasisOrdinals);
  }
  else
  {
    return cellLocalDofOrdinal(layout, shape, componentBasisOrdinals);
  }
}

void TensorDofLayout::decodeVariableLocalDofOrdinal(const VariableLayout &layout,
                                                    const TensorFactorSpaceShape &shape,
                                                    IndexType ordinal,
                                                    std::vector<int> &componentBasisOrdinals) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(shape.componentCount() != layout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::decodeVariableLocalDofOrdinal shape rank must match layout rank.");
  TEUCHOS_TEST_FOR_EXCEPTION((ordinal < 0) || (ordinal >= shape.totalDofs()), std::invalid_argument,
                             "TensorDofLayout::decodeVariableLocalDofOrdinal ordinal out of range.");

  componentBasisOrdinals.assign(layout.componentCount(), 0);
  for (int d=layout.componentCount()-1; d>=0; d--)
  {
    const IndexType stride = shape.componentStride(d);
    const int cardinality = shape.componentCardinality(d);
    componentBasisOrdinals[d] = static_cast<int>((ordinal / stride) % cardinality);
  }
}

void TensorDofLayout::decodeCellLocalDofOrdinal(const VariableLayout &layout,
                                                const TensorFactorSpaceShape &shape,
                                                IndexType ordinal,
                                                std::vector<int> &componentBasisOrdinals) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(shape.componentCount() != layout.componentCount(), std::invalid_argument,
                             "TensorDofLayout::decodeCellLocalDofOrdinal shape rank must match layout rank.");
  TEUCHOS_TEST_FOR_EXCEPTION((ordinal < 0) || (ordinal >= shape.totalDofs()), std::invalid_argument,
                             "TensorDofLayout::decodeCellLocalDofOrdinal ordinal out of range.");
  const auto tuples = enumerateComponentOrdinalsByCellOrdering(shape);
  componentBasisOrdinals = tuples[ordinal];
}

void TensorDofLayout::decodeLocalDofOrdinal(const VariableLayout &layout,
                                            const TensorFactorSpaceShape &shape,
                                            TensorLocalOrdering ordering,
                                            IndexType ordinal,
                                            std::vector<int> &componentBasisOrdinals) const
{
  if (ordering == TENSOR_VARIABLE_ORDERING)
  {
    decodeVariableLocalDofOrdinal(layout, shape, ordinal, componentBasisOrdinals);
  }
  else
  {
    decodeCellLocalDofOrdinal(layout, shape, ordinal, componentBasisOrdinals);
  }
}
