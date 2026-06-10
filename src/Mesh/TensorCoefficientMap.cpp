#include "TensorCoefficientMap.h"

#include <stdexcept>

#include "Basis.h"
#include "Var.h"
#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void TensorCoefficientMap::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorCoefficientMap requires a non-null TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(_tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorCoefficientMap requires a non-null TensorDofLayout.");
  TEUCHOS_TEST_FOR_EXCEPTION(!_tensorDofLayout->hasLayout(_varName, _role), std::invalid_argument,
                             "TensorCoefficientMap requires a registered variable layout.");
}

bool TensorCoefficientMap::variableIsTraceLike() const
{
  return _variableLayout.hasTraceComponentOrdinal();
}

int TensorCoefficientMap::traceComponentOrdinalForVariable() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!variableIsTraceLike(), std::invalid_argument,
                             "TensorCoefficientMap::traceComponentOrdinalForVariable requires a trace variable.");
  return _variableLayout.traceComponentOrdinal();
}

std::string TensorCoefficientMap::compatibleOppositeRoleVarName(int traceComponentOrdinal) const
{
  const auto oppositeRoleLayouts = (_role == TENSOR_TRIAL_SPACE)
      ? _tensorDofLayout->testLayouts()
      : _tensorDofLayout->trialLayouts();

  for (const auto &layout : oppositeRoleLayouts)
  {
    if (layout.hasTraceComponentOrdinal() && (layout.traceComponentOrdinal() == traceComponentOrdinal))
    {
      return layout.name();
    }
  }

  for (const auto &layout : oppositeRoleLayouts)
  {
    if (!layout.hasTraceComponentOrdinal())
    {
      return layout.name();
    }
  }

  if (_tensorDofLayout->hasLayout(_varName, (_role == TENSOR_TRIAL_SPACE) ? TENSOR_TEST_SPACE : TENSOR_TRIAL_SPACE))
  {
    const auto &sameNameOppositeLayout = _tensorDofLayout->layout(_varName,
                                                                   (_role == TENSOR_TRIAL_SPACE) ? TENSOR_TEST_SPACE : TENSOR_TRIAL_SPACE);
    if ((!sameNameOppositeLayout.hasTraceComponentOrdinal()) ||
        (sameNameOppositeLayout.traceComponentOrdinal() == traceComponentOrdinal))
    {
      return _varName;
    }
  }

  return _varName;
}

std::string TensorCoefficientMap::compatibleSkeletonTrialVarName(int traceComponentOrdinal) const
{
  if ((_role == TENSOR_TRIAL_SPACE) && variableIsTraceLike())
  {
    return _varName;
  }
  return compatibleOppositeRoleVarName(traceComponentOrdinal);
}

std::string TensorCoefficientMap::compatibleSkeletonTestVarName(int traceComponentOrdinal) const
{
  if ((_role == TENSOR_TEST_SPACE) && variableIsTraceLike())
  {
    return _varName;
  }
  return compatibleOppositeRoleVarName(traceComponentOrdinal);
}

TensorCoefficientMap::ComponentFactorData TensorCoefficientMap::buildComponentFactorData(int componentOrdinal) const
{
  ComponentFactorData data;
  auto componentTopo = _tensorMesh->topology()->componentMeshTopology(componentOrdinal);
  data._activeCells = componentTopo->activeCells();
  data._cellBasisCardinalities.resize(data._activeCells.size(), 0);
  data._cellBasisOffsets.resize(data._activeCells.size(), 0);

  IndexType runningOffset = 0;
  for (IndexType cellOrdinal=0; cellOrdinal<static_cast<IndexType>(data._activeCells.size()); cellOrdinal++)
  {
    const auto &cellID = data._activeCells[cellOrdinal];
    data._cellBasisOffsets[cellOrdinal] = runningOffset;

    const auto discretization = (_role == TENSOR_TRIAL_SPACE)
        ? _tensorMesh->trialDiscretization(_varName)
        : _tensorMesh->testDiscretization(_varName);
    const int effectivePolyOrder = _tensorMesh->effectivePolyOrder(discretization,
                                                                   cellID,
                                                                   componentOrdinal,
                                                                   _role == TENSOR_TEST_SPACE);
    auto basisFamily = _tensorMesh->effectiveBasisFamily(discretization);
    auto var = discretization->tensorVar()->componentVar(componentOrdinal);
    auto basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(var->space()), effectivePolyOrder);
    data._cellBasisCardinalities[cellOrdinal] = basis->getCardinality();
    runningOffset += basis->getCardinality();
  }

  data._factorSize = runningOffset;
  return data;
}

TensorCoefficientMap::TraceSupportFactorData TensorCoefficientMap::buildTraceSupportFactorData(int traceComponentOrdinal) const
{
  TraceSupportFactorData data;
  auto componentTopo = _tensorMesh->topology()->componentMeshTopology(traceComponentOrdinal);
  data._vertexOrdinals.resize(componentTopo->vertexCount());
  data._vertexComponentOrdinals.resize(componentTopo->vertexCount());

  for (IndexType vertexOrdinal=0; vertexOrdinal<componentTopo->vertexCount(); vertexOrdinal++)
  {
    data._vertexOrdinals[vertexOrdinal] = vertexOrdinal;
    data._vertexComponentOrdinals[vertexOrdinal] = vertexOrdinal;
    data._vertexOrdinalsToFactorOrdinals[vertexOrdinal] = vertexOrdinal;
  }
  data._factorSize = componentTopo->vertexCount();
  return data;
}

void TensorCoefficientMap::buildComponentFactors()
{
  const int componentCount = _tensorMesh->topology()->componentCount();
  _componentFactors.resize(componentCount);
  for (int d=0; d<componentCount; d++)
  {
    _componentFactors[d] = buildComponentFactorData(d);
  }
}

void TensorCoefficientMap::buildVolumeMetadata()
{
  const int componentCount = _tensorMesh->topology()->componentCount();
  _componentStrides.assign(componentCount, 1);
  IndexType runningStride = 1;
  for (int d=0; d<componentCount; d++)
  {
    _componentStrides[d] = runningStride;
    runningStride *= _componentFactors[d]._factorSize;
  }
  _coefficientCount = runningStride;
}

void TensorCoefficientMap::buildTraceMetadata()
{
  const int componentCount = _tensorMesh->topology()->componentCount();
  _componentStrides.assign(componentCount, 1);
  IndexType runningStride = 1;
  for (int d=0; d<componentCount; d++)
  {
    _componentStrides[d] = runningStride;
    const IndexType factorSize = (d == _traceComponentOrdinal)
        ? _traceSupportFactor._factorSize
        : _componentFactors[d]._factorSize;
    runningStride *= factorSize;
  }
  _coefficientCount = runningStride;
}

std::vector<ComponentCellID1D> TensorCoefficientMap::componentCellsForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity) const
{
  const int componentCount = _tensorMesh->topology()->componentCount();
  std::vector<ComponentCellID1D> componentCells(componentCount);

  const int orthogonalComponentOrdinal = entity.orthogonalComponentOrdinal();
  const auto retainedOrdinals = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);
  auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);

  const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
  const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
  TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                             "TensorCoefficientMap expected at least one incident orthogonal-component cell for the requested skeleton vertex.");
  const ComponentCellID1D &incidentOrthogonalCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];

  for (int d=0; d<componentCount; d++)
  {
    componentCells[d] = incidentOrthogonalCell;
  }
  for (size_t r=0; r<retainedOrdinals.size(); r++)
  {
    componentCells[retainedOrdinals[r]] = entity.retainedCellTuple()[r];
  }
  return componentCells;
}

ComponentCellID1D TensorCoefficientMap::incidentOrthogonalCellForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                                                const TensorFactorSpaceShape &shape) const
{
  const int orthogonalComponentOrdinal = entity.orthogonalComponentOrdinal();
  auto orthogonalTopo = _tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
  const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
  const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
  TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                             "TensorCoefficientMap expected at least one incident orthogonal-component cell for the requested skeleton vertex.");

  ComponentCellID1D chosenOrthogonalCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
  for (IndexType i=incidentBegin; i<incidentEnd; i++)
  {
    const ComponentCellID1D &candidateCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(i)];
    auto candidateCells = componentCellsForSkeletonEntity(entity);
    candidateCells[orthogonalComponentOrdinal] = candidateCell;
    auto candidateShape = _tensorDofLayout->factorSpaceShape(_varName,
                                                             _role,
                                                             candidateCells,
                                                             orthogonalComponentOrdinal,
                                                             {orthogonalComponentOrdinal});
    if (candidateShape.componentCardinality(orthogonalComponentOrdinal) == shape.componentCardinality(orthogonalComponentOrdinal))
    {
      chosenOrthogonalCell = candidateCell;
      break;
    }
  }
  return chosenOrthogonalCell;
}

IndexType TensorCoefficientMap::componentCellBasisOffset(int componentOrdinal,
                                                        const ComponentCellID1D &cellID) const
{
  const auto &factorData = _componentFactors[componentOrdinal];
  auto componentTopo = _tensorMesh->topology()->componentMeshTopology(componentOrdinal);
  TEUCHOS_TEST_FOR_EXCEPTION(!componentTopo->containsCell(cellID), std::invalid_argument,
                             "TensorCoefficientMap::componentCellBasisOffset could not find the requested component cell.");
  const IndexType cellOrdinal = componentTopo->activeCellOrdinal(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION((cellOrdinal < 0) || (cellOrdinal >= static_cast<IndexType>(factorData._cellBasisOffsets.size())), std::invalid_argument,
                             "TensorCoefficientMap::componentCellBasisOffset active cell ordinal out of range.");
  return factorData._cellBasisOffsets[cellOrdinal];
}

IndexType TensorCoefficientMap::componentCoordinate(int componentOrdinal,
                                                    const ComponentCellID1D &cellID,
                                                    int basisOrdinal) const
{
  const auto &factorData = _componentFactors[componentOrdinal];
  auto componentTopo = _tensorMesh->topology()->componentMeshTopology(componentOrdinal);
  TEUCHOS_TEST_FOR_EXCEPTION(!componentTopo->containsCell(cellID), std::invalid_argument,
                             "TensorCoefficientMap::componentCoordinate could not find the requested component cell.");
  const IndexType cellOrdinal = componentTopo->activeCellOrdinal(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION((cellOrdinal < 0) || (cellOrdinal >= static_cast<IndexType>(factorData._cellBasisCardinalities.size())), std::invalid_argument,
                             "TensorCoefficientMap::componentCoordinate active cell ordinal out of range.");
  TEUCHOS_TEST_FOR_EXCEPTION((basisOrdinal < 0) || (basisOrdinal >= factorData._cellBasisCardinalities[cellOrdinal]), std::invalid_argument,
                             "TensorCoefficientMap::componentCoordinate basis ordinal out of range for the requested component cell.");
  return factorData._cellBasisOffsets[cellOrdinal] + basisOrdinal;
}

IndexType TensorCoefficientMap::traceSupportCoordinate(IndexType vertexOrdinal) const
{
  auto entry = _traceSupportFactor._vertexOrdinalsToFactorOrdinals.find(vertexOrdinal);
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _traceSupportFactor._vertexOrdinalsToFactorOrdinals.end(), std::invalid_argument,
                             "TensorCoefficientMap::traceSupportCoordinate could not find the requested trace-support vertex ordinal.");
  return entry->second;
}

IndexType TensorCoefficientMap::flattenCoordinates(const std::vector<IndexType> &componentCoordinates) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentCoordinates.size()) != _tensorMesh->topology()->componentCount(), std::invalid_argument,
                             "TensorCoefficientMap::flattenCoordinates requires one component coordinate per tensor component.");
  IndexType flatOrdinal = 0;
  for (int d=0; d<static_cast<int>(componentCoordinates.size()); d++)
  {
    flatOrdinal += componentCoordinates[d] * _componentStrides[d];
  }
  return flatOrdinal;
}

TensorCoefficientMap::TensorCoefficientMap(TensorMeshPtr tensorMesh,
                                           TensorDofLayoutPtr tensorDofLayout,
                                           TensorSpaceRole role,
                                           const std::string &varName,
                                           TensorLocalOrdering ordering)
: _tensorMesh(tensorMesh),
  _tensorDofLayout(tensorDofLayout),
  _role(role),
  _varName(varName),
  _ordering(ordering),
  _variableLayout(tensorDofLayout->layout(varName, role)),
  _coefficientCount(0),
  _isTraceVariable(false),
  _traceComponentOrdinal(-1)
{
  validateConstructorInputs();
  buildComponentFactors();

  if (_variableLayout.hasTraceComponentOrdinal())
  {
    _isTraceVariable = true;
    _traceComponentOrdinal = traceComponentOrdinalForVariable();
    _traceSupportFactor = buildTraceSupportFactorData(_traceComponentOrdinal);
    buildTraceMetadata();
  }
  else
  {
    buildVolumeMetadata();
  }
}

TensorMeshPtr TensorCoefficientMap::tensorMesh() const
{
  return _tensorMesh;
}

TensorDofLayoutPtr TensorCoefficientMap::tensorDofLayout() const
{
  return _tensorDofLayout;
}

TensorSpaceRole TensorCoefficientMap::role() const
{
  return _role;
}

const std::string &TensorCoefficientMap::varName() const
{
  return _varName;
}

TensorLocalOrdering TensorCoefficientMap::ordering() const
{
  return _ordering;
}

const TensorDofLayout::VariableLayout &TensorCoefficientMap::variableLayout() const
{
  return _variableLayout;
}

IndexType TensorCoefficientMap::coefficientCount() const
{
  return _coefficientCount;
}

TensorCoefficientTraversal TensorCoefficientMap::volumeTraversal(const TensorMeshTopology::TensorCellTuple &cellTuple) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_isTraceVariable, std::invalid_argument,
                             "TensorCoefficientMap::volumeTraversal is only valid for volume variables.");
  auto shape = _tensorDofLayout->factorSpaceShape(_varName, _role, cellTuple);

  std::vector<IndexType> componentBases(cellTuple.size(), 0);
  for (int d=0; d<static_cast<int>(cellTuple.size()); d++)
  {
    componentBases[d] = componentCellBasisOffset(d, cellTuple[d]);
  }

  return TensorCoefficientTraversal(TENSOR_VOLUME_TRAVERSAL,
                                    _role,
                                    _varName,
                                    _ordering,
                                    _variableLayout,
                                    shape,
                                    _componentStrides,
                                    componentBases,
                                    false,
                                    -1,
                                    -1);
}

TensorCoefficientTraversal TensorCoefficientMap::skeletonTraversal(const TensorMeshTopology::TensorSkeletonEntity &entity) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_isTraceVariable, std::invalid_argument,
                             "TensorCoefficientMap::skeletonTraversal is only valid for trace variables.");
  TEUCHOS_TEST_FOR_EXCEPTION(entity.orthogonalComponentOrdinal() != _traceComponentOrdinal, std::invalid_argument,
                             "TensorCoefficientMap::skeletonTraversal received an entity from a different tensor skeleton family than the trace variable expects.");

  const std::string trialVarNameForContext = compatibleSkeletonTrialVarName(_traceComponentOrdinal);
  const std::string testVarNameForContext  = compatibleSkeletonTestVarName(_traceComponentOrdinal);
  auto context = _tensorDofLayout->skeletonContext(trialVarNameForContext, testVarNameForContext, entity);
  auto shape = (_role == TENSOR_TRIAL_SPACE) ? context.trialShape() : context.testShape();

  std::vector<IndexType> componentBases(shape.componentCount(), 0);
  const ComponentCellID1D orthogonalCell = incidentOrthogonalCellForSkeletonEntity(entity, shape);
  const auto retainedOrdinals = _tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(entity.orthogonalComponentOrdinal());
  std::map<int, ComponentCellID1D> retainedCellMap;
  for (size_t r=0; r<retainedOrdinals.size(); r++)
  {
    retainedCellMap[retainedOrdinals[r]] = entity.retainedCellTuple()[r];
  }

  for (int d=0; d<shape.componentCount(); d++)
  {
    if (d == _traceComponentOrdinal)
    {
      componentBases[d] = 0;
    }
    else
    {
      const auto retainedEntry = retainedCellMap.find(d);
      const ComponentCellID1D &cellID = (retainedEntry != retainedCellMap.end()) ? retainedEntry->second : orthogonalCell;
      componentBases[d] = componentCellBasisOffset(d, cellID);
    }
  }

  return TensorCoefficientTraversal(TENSOR_SKELETON_TRAVERSAL,
                                    _role,
                                    _varName,
                                    _ordering,
                                    _variableLayout,
                                    shape,
                                    _componentStrides,
                                    componentBases,
                                    true,
                                    _traceComponentOrdinal,
                                    traceSupportCoordinate(entity.vertexOrdinal()));
}

IndexType TensorCoefficientMap::globalIndexForVolumeLocalOrdinal(const TensorMeshTopology::TensorCellTuple &cellTuple,
                                                                 IndexType localOrdinal) const
{
  return volumeTraversal(cellTuple).globalIndexForLocalOrdinal(localOrdinal, _tensorDofLayout);
}

IndexType TensorCoefficientMap::globalIndexForSkeletonLocalOrdinal(const TensorMeshTopology::TensorSkeletonEntity &entity,
                                                                   IndexType localOrdinal) const
{
  return skeletonTraversal(entity).globalIndexForLocalOrdinal(localOrdinal, _tensorDofLayout);
}

std::vector<IndexType> TensorCoefficientMap::indicesForVolumeCell(const TensorMeshTopology::TensorCellTuple &cellTuple) const
{
  return volumeTraversal(cellTuple).globalIndices(_tensorDofLayout);
}

std::vector<IndexType> TensorCoefficientMap::indicesForSkeletonEntity(const TensorMeshTopology::TensorSkeletonEntity &entity) const
{
  return skeletonTraversal(entity).globalIndices(_tensorDofLayout);
}
