#include "TensorMeshTopology.h"

#include <functional>
#include <set>
#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

void TensorMeshTopology::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_components.empty(), std::invalid_argument,
                             "TensorMeshTopology requires at least one component topology.");

  for (size_t i=0; i<_components.size(); i++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_components[i] == Teuchos::null, std::invalid_argument,
                               "TensorMeshTopology requires non-null component topologies.");
  }
}

void TensorMeshTopology::rebuildActiveCellTuples()
{
  _activeCellTuples.clear();

  TensorCellTuple currentTuple;
  currentTuple.resize(_components.size());

  std::function<void(int)> buildTuplesRecursively = [&](int componentOrdinal)
  {
    if (componentOrdinal < 0)
    {
      _activeCellTuples.push_back(currentTuple);
      return;
    }

    const auto &componentCells = _components[componentOrdinal]->activeCells();
    for (const auto &cellID : componentCells)
    {
      currentTuple[componentOrdinal] = cellID;
      buildTuplesRecursively(componentOrdinal - 1);
    }
  };

  buildTuplesRecursively(static_cast<int>(_components.size()) - 1);
}

TensorMeshTopology::TensorMeshTopology(const std::vector<Teuchos::RCP<ComponentMeshTopology1D>> &components)
: _components(components)
{
  validateConstructorInputs();
  _originalComponentOrdinals.resize(_components.size());
  for (size_t i=0; i<_components.size(); i++)
  {
    _originalComponentOrdinals[i] = static_cast<int>(i);
  }
  rebuildActiveCellTuples();
}

TensorMeshTopology::TensorMeshTopology(const std::vector<Teuchos::RCP<ComponentMeshTopology1D>> &components,
                                       const std::vector<TensorCellTuple> &activeCellTuples,
                                       const std::vector<int> &originalComponentOrdinals)
: _components(components), _activeCellTuples(activeCellTuples), _originalComponentOrdinals(originalComponentOrdinals)
{
  validateConstructorInputs();
  TEUCHOS_TEST_FOR_EXCEPTION(_originalComponentOrdinals.size() != _components.size(), std::invalid_argument,
                             "TensorMeshTopology explicit constructor requires one original-component entry per retained component.");
  for (const auto &cellTuple : _activeCellTuples)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(cellTuple.size() != _components.size(), std::invalid_argument,
                               "TensorMeshTopology explicit constructor received a tensor cell tuple with incorrect rank.");
  }
}

int TensorMeshTopology::componentCount() const
{
  return static_cast<int>(_components.size());
}

IndexType TensorMeshTopology::activeCellCount() const
{
  return static_cast<IndexType>(_activeCellTuples.size());
}

Teuchos::RCP<ComponentMeshTopology1D> TensorMeshTopology::componentMeshTopology(int componentOrdinal) const
{
  return _components[componentOrdinal];
}

int TensorMeshTopology::originalComponentOrdinal(int componentOrdinal) const
{
  return _originalComponentOrdinals[componentOrdinal];
}

const std::vector<TensorMeshTopology::TensorCellTuple> &TensorMeshTopology::activeCellTuples() const
{
  return _activeCellTuples;
}

bool TensorMeshTopology::containsTensorCell(const TensorCellTuple &cellTuple) const
{
  if (cellTuple.size() != _components.size()) return false;

  for (size_t d=0; d<_components.size(); d++)
  {
    if (!_components[d]->containsCell(cellTuple[d])) return false;
  }
  return true;
}

IndexType TensorMeshTopology::tensorCellOrdinal(const TensorCellTuple &cellTuple) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!containsTensorCell(cellTuple), std::invalid_argument,
                             "TensorMeshTopology::tensorCellOrdinal requires a tensor cell in the topology.");

  for (IndexType ordinal=0; ordinal<static_cast<IndexType>(_activeCellTuples.size()); ordinal++)
  {
    if (_activeCellTuples[ordinal] == cellTuple) return ordinal;
  }

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                             "TensorMeshTopology::tensorCellOrdinal internal error: tensor cell not found after validation.");
}

TensorMeshTopology::TensorCellTuple TensorMeshTopology::tensorCellTuple(IndexType ordinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((ordinal < 0) || (ordinal >= static_cast<IndexType>(_activeCellTuples.size())), std::invalid_argument,
                             "TensorMeshTopology::tensorCellTuple ordinal out of range.");
  return _activeCellTuples[ordinal];
}

std::vector<TensorMeshTopology::TensorSide> TensorMeshTopology::sidesForTensorCell(const TensorCellTuple &cellTuple) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!containsTensorCell(cellTuple), std::invalid_argument,
                             "TensorMeshTopology::sidesForTensorCell requires a tensor cell in the topology.");

  std::vector<TensorSide> sides;
  sides.reserve(2 * componentCount());
  for (int d=0; d<componentCount(); d++)
  {
    sides.push_back(TensorSide(cellTuple, d, 0));
    sides.push_back(TensorSide(cellTuple, d, 1));
  }
  return sides;
}

bool TensorMeshTopology::sideIsValid(const TensorSide &side) const
{
  if (side.componentOrdinal() < 0 || side.componentOrdinal() >= componentCount()) return false;
  if (side.sideOrdinalInComponent() < 0 || side.sideOrdinalInComponent() > 1) return false;
  return containsTensorCell(side.cellTuple());
}

bool TensorMeshTopology::isBoundarySide(const TensorSide &side) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!sideIsValid(side), std::invalid_argument,
                             "TensorMeshTopology::isBoundarySide requires a valid tensor side.");

  const int d = side.componentOrdinal();
  const auto &componentCell = side.cellTuple()[d];
  const auto componentTopo = _components[d];
  const IndexType vertexIndex = componentTopo->cellVertexIndex(componentCell, side.sideOrdinalInComponent());
  const IndexType begin = componentTopo->vertexToCellOffsetHost(vertexIndex);
  const IndexType end   = componentTopo->vertexToCellOffsetHost(vertexIndex + 1);

  bool sameRootNeighborFound = false;
  for (IndexType i=begin; i<end; i++)
  {
    const IndexType incidentCellOrdinal = componentTopo->vertexToCellOrdinalHost(i);
    const auto &incidentCell = componentTopo->activeCells()[incidentCellOrdinal];
    if (incidentCell == componentCell) continue;

    // same root and sharing the selected endpoint means this is a neighbor candidate across the side.
    if (incidentCell.rootOrdinal() == componentCell.rootOrdinal())
    {
      sameRootNeighborFound = true;
      break;
    }
  }

  if (sameRootNeighborFound) return false;

  // For nonperiodic topologies, the remaining case is a boundary endpoint.
  // For periodic topologies, current ComponentMeshTopology1D periodicity is state-only, so this remains a conservative topology-side interpretation for now.
  return true;
}

TensorMeshTopology TensorMeshTopology::slice(const std::map<int, IndexType> &fixedComponentVertices) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(fixedComponentVertices.empty(), std::invalid_argument,
                             "TensorMeshTopology::slice requires at least one fixed component vertex.");

  for (const auto &entry : fixedComponentVertices)
  {
    const int componentOrdinal = entry.first;
    const IndexType vertexIndex = entry.second;
    TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= componentCount()), std::invalid_argument,
                               "TensorMeshTopology::slice received an invalid component ordinal.");

    auto componentTopo = _components[componentOrdinal];
    TEUCHOS_TEST_FOR_EXCEPTION((vertexIndex < 0) || (vertexIndex >= componentTopo->vertexCount()), std::invalid_argument,
                               "TensorMeshTopology::slice received an invalid vertex index.");
  }

  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(fixedComponentVertices.size()) >= componentCount(), std::invalid_argument,
                             "TensorMeshTopology::slice does not yet support fixing all components.");

  std::vector<Teuchos::RCP<ComponentMeshTopology1D>> retainedComponents;
  std::vector<int> retainedOriginalOrdinals;
  for (int d=0; d<componentCount(); d++)
  {
    if (fixedComponentVertices.find(d) == fixedComponentVertices.end())
    {
      retainedComponents.push_back(_components[d]);
      retainedOriginalOrdinals.push_back(_originalComponentOrdinals[d]);
    }
  }

  std::map<int, std::set<ComponentCellID1D>> allowedCellsForFixedComponents;
  for (const auto &entry : fixedComponentVertices)
  {
    const int componentOrdinal = entry.first;
    const IndexType vertexIndex = entry.second;
    const auto componentTopo = _components[componentOrdinal];

    const IndexType begin = componentTopo->vertexToCellOffsetHost(vertexIndex);
    const IndexType end   = componentTopo->vertexToCellOffsetHost(vertexIndex + 1);
    std::set<ComponentCellID1D> allowedCells;
    for (IndexType i=begin; i<end; i++)
    {
      const IndexType cellOrdinal = componentTopo->vertexToCellOrdinalHost(i);
      allowedCells.insert(componentTopo->activeCells()[cellOrdinal]);
    }
    allowedCellsForFixedComponents[componentOrdinal] = allowedCells;
  }

  std::vector<TensorCellTuple> slicedTuples;
  for (const auto &fullTuple : _activeCellTuples)
  {
    bool matchesSlice = true;
    for (const auto &entry : fixedComponentVertices)
    {
      const int fixedComponent = entry.first;
      const auto &allowedCells = allowedCellsForFixedComponents[fixedComponent];
      if (allowedCells.find(fullTuple[fixedComponent]) == allowedCells.end())
      {
        matchesSlice = false;
        break;
      }
    }
    if (!matchesSlice) continue;

    TensorCellTuple slicedTuple;
    for (int d=0; d<componentCount(); d++)
    {
      if (fixedComponentVertices.find(d) == fixedComponentVertices.end())
      {
        slicedTuple.push_back(fullTuple[d]);
      }
    }
    slicedTuples.push_back(slicedTuple);
  }

  return TensorMeshTopology(retainedComponents, slicedTuples, retainedOriginalOrdinals);
}
