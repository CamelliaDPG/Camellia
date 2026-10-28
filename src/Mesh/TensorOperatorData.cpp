#include "TensorOperatorData.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void TensorOperatorData::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorOperatorData requires a non-null TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorOperatorData requires a non-null TensorDofLayout.");
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorDofLayout->tensorMesh().get() != _tensorMesh.get(), std::invalid_argument,
                             "TensorOperatorData requires a TensorDofLayout built from the supplied TensorMesh.");
}

void TensorOperatorData::buildComponentData()
{
  const int componentCount = _tensorMesh->topology()->componentCount();
  _components.resize(componentCount);

  for (int d=0; d<componentCount; d++)
  {
    auto componentTopo = _tensorMesh->topology()->componentMeshTopology(d);
    ComponentData &data = _components[d];
    data._activeCellCount = componentTopo->activeCellCount();
    data._vertexCount = componentTopo->vertexCount();
    data._vertexCoordinates = componentTopo->activeVertexCoordinates();
    data._cellVertices = componentTopo->cellVertices();
    data._vertexToCellOffsets = componentTopo->vertexToCellOffsets();
    data._vertexToCellOrdinals = componentTopo->vertexToCellOrdinals();
    data._cellWidths = ComponentCellMetricDualView("tensor_operator_data_component_cell_widths", data._activeCellCount);
    data._cellJacobians = ComponentCellMetricDualView("tensor_operator_data_component_cell_jacobians", data._activeCellCount);
    data._cellPolyOrders = ComponentCellPolyOrderDualView("tensor_operator_data_component_cell_poly_orders", data._activeCellCount);

    for (IndexType cellOrdinal=0; cellOrdinal<data._activeCellCount; cellOrdinal++)
    {
      const auto &cellID = componentTopo->activeCells()[cellOrdinal];
      const double width = componentTopo->cellWidth(cellID);
      data._cellWidths.view_host()(cellOrdinal) = width;
      data._cellJacobians.view_host()(cellOrdinal) = 0.5 * width;
      data._cellPolyOrders.view_host()(cellOrdinal) = componentTopo->cellPolyOrder(cellID);
    }

    data._cellWidths.modify_host();
    data._cellWidths.sync_device();
    data._cellJacobians.modify_host();
    data._cellJacobians.sync_device();
    data._cellPolyOrders.modify_host();
    data._cellPolyOrders.sync_device();
  }
}

void TensorOperatorData::buildActiveTensorCellData()
{
  const IndexType cellCount = _tensorMesh->topology()->activeCellCount();
  const int componentCount = _tensorMesh->topology()->componentCount();
  _activeTensorCellComponentOrdinals = ActiveTensorCellComponentOrdinalsDualView("tensor_operator_data_active_tensor_cell_component_ordinals",
                                                                                cellCount,
                                                                                componentCount);

  for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<cellCount; tensorCellOrdinal++)
  {
    const auto cellTuple = _tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
    for (int d=0; d<componentCount; d++)
    {
      _activeTensorCellComponentOrdinals.view_host()(tensorCellOrdinal,d) = _tensorMesh->topology()->componentMeshTopology(d)->activeCellOrdinal(cellTuple[d]);
    }
  }

  _activeTensorCellComponentOrdinals.modify_host();
  _activeTensorCellComponentOrdinals.sync_device();
}

void TensorOperatorData::buildBoundarySideData()
{
  const IndexType cellCount = _tensorMesh->topology()->activeCellCount();
  const int componentCount = _tensorMesh->topology()->componentCount();
  _boundarySideFlags = BoundarySideFlagDualView("tensor_operator_data_boundary_side_flags",
                                                cellCount,
                                                2 * componentCount);

  for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<cellCount; tensorCellOrdinal++)
  {
    const auto cellTuple = _tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
    for (int d=0; d<componentCount; d++)
    {
      for (int sideOrdinalInComponent=0; sideOrdinalInComponent<2; sideOrdinalInComponent++)
      {
        const int tensorSideOrdinal = 2 * d + sideOrdinalInComponent;
        TensorMeshTopology::TensorSide side(cellTuple, d, sideOrdinalInComponent);
        _boundarySideFlags.view_host()(tensorCellOrdinal,tensorSideOrdinal) = _tensorMesh->topology()->isBoundarySide(side) ? 1 : 0;
      }
    }
  }

  _boundarySideFlags.modify_host();
  _boundarySideFlags.sync_device();
}

namespace
{
  // Returns the 1D outward-normal sign for the first incident active cell:
  // -1 on the lower/left endpoint and +1 on the upper/right endpoint.
  int firstIncidentCellSideParity(const Teuchos::RCP<ComponentMeshTopology1D> &componentTopo,
                                  IndexType vertexOrdinal)
  {
    const IndexType begin = componentTopo->vertexToCellOffsetHost(vertexOrdinal);
    const IndexType end   = componentTopo->vertexToCellOffsetHost(vertexOrdinal + 1);
    TEUCHOS_TEST_FOR_EXCEPTION(begin >= end, std::logic_error,
                               "TensorOperatorData expected at least one incident component cell for skeleton metadata.");

    const ComponentCellID1D &incidentCell = componentTopo->activeCells()[componentTopo->vertexToCellOrdinalHost(begin)];
    const IndexType leftVertex  = componentTopo->cellVertexIndex(incidentCell, 0);
    const IndexType rightVertex = componentTopo->cellVertexIndex(incidentCell, 1);
    if (vertexOrdinal == leftVertex) return -1;
    if (vertexOrdinal == rightVertex) return 1;

    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                               "TensorOperatorData found an incident-cell/vertex mismatch while building skeleton parity metadata.");
  }
}

void TensorOperatorData::buildVolumeLocalToGlobalData(VariableData &data,
                                                      const TensorCoefficientMap &coefficientMap) const
{
  if (data.isTraceVariable()) return;

  const IndexType cellCount = _tensorMesh->topology()->activeCellCount();
  std::vector<std::vector<IndexType>> indicesByCell(cellCount);
  data._maxVolumeLocalDofCount = 0;
  for (IndexType cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    indicesByCell[cellOrdinal] = coefficientMap.indicesForVolumeCell(_tensorMesh->topology()->tensorCellTuple(cellOrdinal));
    data._maxVolumeLocalDofCount = std::max(data._maxVolumeLocalDofCount,
                                            static_cast<IndexType>(indicesByCell[cellOrdinal].size()));
  }

  data._volumeLocalDofCounts = LocalDofCountDualView("tensor_operator_data_volume_local_dof_counts", cellCount);
  data._volumeLocalToGlobalIndices = LocalToGlobalIndexDualView("tensor_operator_data_volume_local_to_global_indices",
                                                               cellCount,
                                                               data._maxVolumeLocalDofCount);
  for (IndexType cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++)
  {
    data._volumeLocalDofCounts.view_host()(cellOrdinal) = static_cast<IndexType>(indicesByCell[cellOrdinal].size());
    for (IndexType localOrdinal=0; localOrdinal<data._maxVolumeLocalDofCount; localOrdinal++)
    {
      data._volumeLocalToGlobalIndices.view_host()(cellOrdinal,localOrdinal) = (localOrdinal < static_cast<IndexType>(indicesByCell[cellOrdinal].size()))
          ? indicesByCell[cellOrdinal][localOrdinal]
          : -1;
    }
  }
  data._volumeLocalDofCounts.modify_host();
  data._volumeLocalDofCounts.sync_device();
  data._volumeLocalToGlobalIndices.modify_host();
  data._volumeLocalToGlobalIndices.sync_device();
}

void TensorOperatorData::buildSkeletonLocalToGlobalData(VariableData &data,
                                                        const TensorCoefficientMap &coefficientMap) const
{
  if (!data.isTraceVariable()) return;

  auto skeleton = _tensorMesh->topology()->tensorSkeleton(data.traceComponentOrdinal());
  data._skeletonEntityCount = skeleton.entityCount();
  std::vector<std::vector<IndexType>> indicesByEntity(data._skeletonEntityCount);
  data._maxSkeletonLocalDofCount = 0;
  for (IndexType entityOrdinal=0; entityOrdinal<data._skeletonEntityCount; entityOrdinal++)
  {
    indicesByEntity[entityOrdinal] = coefficientMap.indicesForSkeletonEntity(skeleton.entity(entityOrdinal));
    data._maxSkeletonLocalDofCount = std::max(data._maxSkeletonLocalDofCount,
                                              static_cast<IndexType>(indicesByEntity[entityOrdinal].size()));
  }

  data._skeletonLocalDofCounts = LocalDofCountDualView("tensor_operator_data_skeleton_local_dof_counts",
                                                       data._skeletonEntityCount);
  data._skeletonLocalToGlobalIndices = LocalToGlobalIndexDualView("tensor_operator_data_skeleton_local_to_global_indices",
                                                                 data._skeletonEntityCount,
                                                                 data._maxSkeletonLocalDofCount);
  data._skeletonEntityVertexOrdinals = SkeletonEntityOrdinalDualView("tensor_operator_data_skeleton_entity_vertex_ordinals",
                                                                     data._skeletonEntityCount);
  data._skeletonEntitySideParities = SkeletonEntityParityDualView("tensor_operator_data_skeleton_entity_side_parities",
                                                                 data._skeletonEntityCount);

  auto componentTopo = _tensorMesh->topology()->componentMeshTopology(data.traceComponentOrdinal());
  for (IndexType entityOrdinal=0; entityOrdinal<data._skeletonEntityCount; entityOrdinal++)
  {
    const auto &entity = skeleton.entity(entityOrdinal);
    data._skeletonLocalDofCounts.view_host()(entityOrdinal) = static_cast<IndexType>(indicesByEntity[entityOrdinal].size());
    data._skeletonEntityVertexOrdinals.view_host()(entityOrdinal) = entity.vertexOrdinal();
    data._skeletonEntitySideParities.view_host()(entityOrdinal) = firstIncidentCellSideParity(componentTopo, entity.vertexOrdinal());
    for (IndexType localOrdinal=0; localOrdinal<data._maxSkeletonLocalDofCount; localOrdinal++)
    {
      data._skeletonLocalToGlobalIndices.view_host()(entityOrdinal,localOrdinal) = (localOrdinal < static_cast<IndexType>(indicesByEntity[entityOrdinal].size()))
          ? indicesByEntity[entityOrdinal][localOrdinal]
          : -1;
    }
  }

  data._skeletonLocalDofCounts.modify_host();
  data._skeletonLocalDofCounts.sync_device();
  data._skeletonLocalToGlobalIndices.modify_host();
  data._skeletonLocalToGlobalIndices.sync_device();
  data._skeletonEntityVertexOrdinals.modify_host();
  data._skeletonEntityVertexOrdinals.sync_device();
  data._skeletonEntitySideParities.modify_host();
  data._skeletonEntitySideParities.sync_device();
}

void TensorOperatorData::buildVariableData()
{
  _variables.clear();
  for (const auto &layout : _tensorDofLayout->trialLayouts())
  {
    TensorCoefficientMap coefficientMap(_tensorMesh, _tensorDofLayout, TENSOR_TRIAL_SPACE, layout.name());
    VariableData data;
    data._name = layout.name();
    data._role = TENSOR_TRIAL_SPACE;
    data._isTraceVariable = layout.hasTraceComponentOrdinal();
    data._traceComponentOrdinal = layout.hasTraceComponentOrdinal() ? layout.traceComponentOrdinal() : -1;
    data._coefficientCount = coefficientMap.coefficientCount();
    data._ambientShape = _tensorDofLayout->ambientShape(layout.name(), TENSOR_TRIAL_SPACE);
    buildVolumeLocalToGlobalData(data, coefficientMap);
    buildSkeletonLocalToGlobalData(data, coefficientMap);
    _variables.push_back(data);
  }
  for (const auto &layout : _tensorDofLayout->testLayouts())
  {
    TensorCoefficientMap coefficientMap(_tensorMesh, _tensorDofLayout, TENSOR_TEST_SPACE, layout.name());
    VariableData data;
    data._name = layout.name();
    data._role = TENSOR_TEST_SPACE;
    data._isTraceVariable = layout.hasTraceComponentOrdinal();
    data._traceComponentOrdinal = layout.hasTraceComponentOrdinal() ? layout.traceComponentOrdinal() : -1;
    data._coefficientCount = coefficientMap.coefficientCount();
    data._ambientShape = _tensorDofLayout->ambientShape(layout.name(), TENSOR_TEST_SPACE);
    buildVolumeLocalToGlobalData(data, coefficientMap);
    buildSkeletonLocalToGlobalData(data, coefficientMap);
    _variables.push_back(data);
  }
}

TensorOperatorData::TensorOperatorData(TensorMeshPtr tensorMesh,
                                       TensorDofLayoutPtr tensorDofLayout)
: _tensorMesh(tensorMesh),
  _tensorDofLayout(tensorDofLayout)
{
  validateConstructorInputs();
  buildComponentData();
  buildActiveTensorCellData();
  buildBoundarySideData();
  buildVariableData();
}

TensorMeshPtr TensorOperatorData::tensorMesh() const
{
  return _tensorMesh;
}

TensorDofLayoutPtr TensorOperatorData::tensorDofLayout() const
{
  return _tensorDofLayout;
}

int TensorOperatorData::componentCount() const
{
  return _tensorMesh->topology()->componentCount();
}

IndexType TensorOperatorData::activeCellCount() const
{
  return _tensorMesh->topology()->activeCellCount();
}

IndexType TensorOperatorData::tensorSideCount() const
{
  return 2 * componentCount();
}

const TensorOperatorData::ComponentData &TensorOperatorData::componentData(int componentOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= componentCount()), std::invalid_argument,
                             "TensorOperatorData::componentData component ordinal out of range.");
  return _components[componentOrdinal];
}

const TensorOperatorData::ActiveTensorCellComponentOrdinalsDualView &TensorOperatorData::activeTensorCellComponentOrdinals() const
{
  return _activeTensorCellComponentOrdinals;
}

IndexType TensorOperatorData::activeTensorCellComponentOrdinal(IndexType tensorCellOrdinal,
                                                               int componentOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((tensorCellOrdinal < 0) || (tensorCellOrdinal >= activeCellCount()), std::invalid_argument,
                             "TensorOperatorData::activeTensorCellComponentOrdinal tensor cell ordinal out of range.");
  TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= componentCount()), std::invalid_argument,
                             "TensorOperatorData::activeTensorCellComponentOrdinal component ordinal out of range.");
  return _activeTensorCellComponentOrdinals.view_host()(tensorCellOrdinal, componentOrdinal);
}

const TensorOperatorData::BoundarySideFlagDualView &TensorOperatorData::boundarySideFlags() const
{
  return _boundarySideFlags;
}

bool TensorOperatorData::isBoundarySide(IndexType tensorCellOrdinal,
                                        int tensorSideOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((tensorCellOrdinal < 0) || (tensorCellOrdinal >= activeCellCount()), std::invalid_argument,
                             "TensorOperatorData::isBoundarySide tensor cell ordinal out of range.");
  TEUCHOS_TEST_FOR_EXCEPTION((tensorSideOrdinal < 0) || (tensorSideOrdinal >= tensorSideCount()), std::invalid_argument,
                             "TensorOperatorData::isBoundarySide tensor side ordinal out of range.");
  return _boundarySideFlags.view_host()(tensorCellOrdinal, tensorSideOrdinal) != 0;
}

IndexType TensorOperatorData::variableCount() const
{
  return static_cast<IndexType>(_variables.size());
}

const TensorOperatorData::VariableData &TensorOperatorData::variableData(IndexType variableOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((variableOrdinal < 0) || (variableOrdinal >= static_cast<IndexType>(_variables.size())), std::invalid_argument,
                             "TensorOperatorData::variableData variable ordinal out of range.");
  return _variables[variableOrdinal];
}

IndexType TensorOperatorData::variableOrdinal(TensorSpaceRole role,
                                            const std::string &varName) const
{
  for (IndexType variableOrdinal=0; variableOrdinal<static_cast<IndexType>(_variables.size()); variableOrdinal++)
  {
    const auto &variable = _variables[variableOrdinal];
    if ((variable.role() == role) && (variable.name() == varName))
    {
      return variableOrdinal;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                             "TensorOperatorData::variableOrdinal variable/role pair not found.");
}

const TensorOperatorData::VariableData &TensorOperatorData::variableData(TensorSpaceRole role,
                                                                         const std::string &varName) const
{
  return variableData(variableOrdinal(role, varName));
}

IndexType TensorOperatorData::coefficientCount(TensorSpaceRole role,
                                               const std::string &varName) const
{
  return variableData(role, varName).coefficientCount();
}

IndexType TensorOperatorData::volumeLocalDofCount(TensorSpaceRole role,
                                                  const std::string &varName,
                                                  IndexType tensorCellOrdinal) const
{
  const auto &variable = variableData(role, varName);
  TEUCHOS_TEST_FOR_EXCEPTION(variable.isTraceVariable(), std::invalid_argument,
                             "TensorOperatorData::volumeLocalDofCount is only valid for volume variables.");
  TEUCHOS_TEST_FOR_EXCEPTION((tensorCellOrdinal < 0) || (tensorCellOrdinal >= activeCellCount()), std::invalid_argument,
                             "TensorOperatorData::volumeLocalDofCount tensor cell ordinal out of range.");
  return variable.volumeLocalDofCounts().view_host()(tensorCellOrdinal);
}

IndexType TensorOperatorData::volumeLocalToGlobalIndex(TensorSpaceRole role,
                                                       const std::string &varName,
                                                       IndexType tensorCellOrdinal,
                                                       IndexType localOrdinal) const
{
  const auto &variable = variableData(role, varName);
  const IndexType localDofCount = volumeLocalDofCount(role, varName, tensorCellOrdinal);
  TEUCHOS_TEST_FOR_EXCEPTION((localOrdinal < 0) || (localOrdinal >= localDofCount), std::invalid_argument,
                             "TensorOperatorData::volumeLocalToGlobalIndex local ordinal out of range.");
  return variable.volumeLocalToGlobalIndices().view_host()(tensorCellOrdinal, localOrdinal);
}

IndexType TensorOperatorData::skeletonEntityCount(TensorSpaceRole role,
                                                  const std::string &varName) const
{
  const auto &variable = variableData(role, varName);
  TEUCHOS_TEST_FOR_EXCEPTION(!variable.isTraceVariable(), std::invalid_argument,
                             "TensorOperatorData::skeletonEntityCount is only valid for trace variables.");
  return variable.skeletonEntityCount();
}

IndexType TensorOperatorData::skeletonLocalDofCount(TensorSpaceRole role,
                                                    const std::string &varName,
                                                    IndexType skeletonEntityOrdinal) const
{
  const auto &variable = variableData(role, varName);
  TEUCHOS_TEST_FOR_EXCEPTION(!variable.isTraceVariable(), std::invalid_argument,
                             "TensorOperatorData::skeletonLocalDofCount is only valid for trace variables.");
  TEUCHOS_TEST_FOR_EXCEPTION((skeletonEntityOrdinal < 0) || (skeletonEntityOrdinal >= variable.skeletonEntityCount()), std::invalid_argument,
                             "TensorOperatorData::skeletonLocalDofCount skeleton entity ordinal out of range.");
  return variable.skeletonLocalDofCounts().view_host()(skeletonEntityOrdinal);
}

IndexType TensorOperatorData::skeletonLocalToGlobalIndex(TensorSpaceRole role,
                                                         const std::string &varName,
                                                         IndexType skeletonEntityOrdinal,
                                                         IndexType localOrdinal) const
{
  const auto &variable = variableData(role, varName);
  const IndexType localDofCount = skeletonLocalDofCount(role, varName, skeletonEntityOrdinal);
  TEUCHOS_TEST_FOR_EXCEPTION((localOrdinal < 0) || (localOrdinal >= localDofCount), std::invalid_argument,
                             "TensorOperatorData::skeletonLocalToGlobalIndex local ordinal out of range.");
  return variable.skeletonLocalToGlobalIndices().view_host()(skeletonEntityOrdinal, localOrdinal);
}

int TensorOperatorData::skeletonEntitySideParity(TensorSpaceRole role,
                                                 const std::string &varName,
                                                 IndexType skeletonEntityOrdinal) const
{
  const auto &variable = variableData(role, varName);
  TEUCHOS_TEST_FOR_EXCEPTION(!variable.isTraceVariable(), std::invalid_argument,
                             "TensorOperatorData::skeletonEntitySideParity is only valid for trace variables.");
  TEUCHOS_TEST_FOR_EXCEPTION((skeletonEntityOrdinal < 0) || (skeletonEntityOrdinal >= variable.skeletonEntityCount()), std::invalid_argument,
                             "TensorOperatorData::skeletonEntitySideParity skeleton entity ordinal out of range.");
  return variable.skeletonEntitySideParities().view_host()(skeletonEntityOrdinal);
}

int TensorOperatorData::skeletonEntityOutwardNormalSign(TensorSpaceRole role,
                                                       const std::string &varName,
                                                       IndexType skeletonEntityOrdinal) const
{
  return skeletonEntitySideParity(role, varName, skeletonEntityOrdinal);
}
