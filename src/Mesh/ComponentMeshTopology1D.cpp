#include "ComponentMeshTopology1D.h"

#include <algorithm>
#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  struct ActiveVertexDescriptor
  {
    IndexType _rootOrdinal;
    unsigned _level;
    unsigned long long _address;

    bool operator<(const ActiveVertexDescriptor &other) const
    {
      if (_rootOrdinal != other._rootOrdinal) return _rootOrdinal < other._rootOrdinal;

      const unsigned commonLevel = (_level > other._level) ? _level : other._level;
      const unsigned long long thisScaled  = _address << (commonLevel - _level);
      const unsigned long long otherScaled = other._address << (commonLevel - other._level);
      if (thisScaled != otherScaled) return thisScaled < otherScaled;

      if (_level != other._level) return _level < other._level;
      return _address < other._address;
    }
  };

  ActiveVertexDescriptor canonicalizeVertexDescriptor(ActiveVertexDescriptor vertex)
  {
    while ((vertex._level > 0) && ((vertex._address % 2ULL) == 0ULL))
    {
      vertex._address /= 2ULL;
      vertex._level--;
    }
    return vertex;
  }
}

void ComponentMeshTopology1D::validateConstructorInputs() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_coarseVertices.size() < 2, std::invalid_argument,
                             "ComponentMeshTopology1D requires at least two coarse vertices.");

  for (size_t i=1; i<_coarseVertices.size(); i++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!(_coarseVertices[i-1] < _coarseVertices[i]), std::invalid_argument,
                               "ComponentMeshTopology1D requires strictly increasing coarse vertices.");
  }

  TEUCHOS_TEST_FOR_EXCEPTION(_activeLeafSets.size() != _coarseVertices.size() - 1, std::invalid_argument,
                             "ComponentMeshTopology1D requires one ActiveLeafSet1D per root cell.");

  TEUCHOS_TEST_FOR_EXCEPTION(_defaultPolyOrder < 0, std::invalid_argument,
                             "ComponentMeshTopology1D requires nonnegative default polynomial order.");

  for (const auto &leafSet : _activeLeafSets)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(!leafSet.isValid(), std::invalid_argument,
                               "ComponentMeshTopology1D requires each ActiveLeafSet1D to be a valid exact cover.");
  }
}

void ComponentMeshTopology1D::rebuildActiveCells()
{
  _activeCells.clear();
  _activeCellOrdinals.clear();
  for (IndexType rootOrdinal=0; rootOrdinal<static_cast<IndexType>(_activeLeafSets.size()); rootOrdinal++)
  {
    auto rootCells = _activeLeafSets[rootOrdinal].cellIDsForRoot(rootOrdinal);
    for (const auto &cellID : rootCells)
    {
      _activeCellOrdinals[cellID] = static_cast<IndexType>(_activeCells.size());
      _activeCells.push_back(cellID);
    }
  }
}

void ComponentMeshTopology1D::initializePolyOrders()
{
  _cellPolyOrders.clear();
  for (const auto &cellID : _activeCells)
  {
    _cellPolyOrders[cellID] = _defaultPolyOrder;
  }
}

void ComponentMeshTopology1D::rebuildVertexIncidence()
{
  std::map<ActiveVertexDescriptor, IndexType> vertexOrdinalMap;
  std::vector<ActiveVertexDescriptor> orderedVertices;

  for (const auto &cellID : _activeCells)
  {
    ActiveVertexDescriptor left  = canonicalizeVertexDescriptor({cellID.rootOrdinal(), cellID.level(), cellID.address()});
    ActiveVertexDescriptor right = canonicalizeVertexDescriptor({cellID.rootOrdinal(), cellID.level(), cellID.address() + 1ULL});

    if (vertexOrdinalMap.find(left) == vertexOrdinalMap.end())
    {
      vertexOrdinalMap[left] = static_cast<IndexType>(orderedVertices.size());
      orderedVertices.push_back(left);
    }
    if (vertexOrdinalMap.find(right) == vertexOrdinalMap.end())
    {
      vertexOrdinalMap[right] = static_cast<IndexType>(orderedVertices.size());
      orderedVertices.push_back(right);
    }
  }

  const IndexType vertexCount_ = static_cast<IndexType>(orderedVertices.size());
  _activeVertexCoordinates = VertexCoordinateDualView("component_active_vertex_coordinates", vertexCount_);
  _cellVertices            = CellVerticesDualView("component_cell_vertices", activeCellCount(), 2);

  for (IndexType vertexOrdinal=0; vertexOrdinal<vertexCount_; vertexOrdinal++)
  {
    const auto &vertexDesc = orderedVertices[vertexOrdinal];
    const double x0 = rootCellLeft(vertexDesc._rootOrdinal);
    const double x1 = rootCellRight(vertexDesc._rootOrdinal);
    const double width = x1 - x0;
    const double denom = static_cast<double>(1ULL << vertexDesc._level);
    _activeVertexCoordinates.h_view(vertexOrdinal) = x0 + width * static_cast<double>(vertexDesc._address) / denom;
  }

  std::vector<std::vector<IndexType>> incidentCellOrdinals(vertexCount_);
  for (const auto &cellID : _activeCells)
  {
    const auto cellOrdinal = _activeCellOrdinals[cellID];
    ActiveVertexDescriptor left  = canonicalizeVertexDescriptor({cellID.rootOrdinal(), cellID.level(), cellID.address()});
    ActiveVertexDescriptor right = canonicalizeVertexDescriptor({cellID.rootOrdinal(), cellID.level(), cellID.address() + 1ULL});
    const IndexType leftVertexOrdinal  = vertexOrdinalMap[left];
    const IndexType rightVertexOrdinal = vertexOrdinalMap[right];
    _cellVertices.h_view(cellOrdinal,0) = leftVertexOrdinal;
    _cellVertices.h_view(cellOrdinal,1) = rightVertexOrdinal;
    incidentCellOrdinals[leftVertexOrdinal].push_back(cellOrdinal);
    incidentCellOrdinals[rightVertexOrdinal].push_back(cellOrdinal);
  }

  _vertexToActiveCellsHost.clear();
  _vertexToCellOffsets = VertexToCellOffsetsDualView("component_vertex_to_cell_offsets", vertexCount_ + 1);
  IndexType runningOffset = 0;
  for (IndexType vertexOrdinal=0; vertexOrdinal<vertexCount_; vertexOrdinal++)
  {
    _vertexToCellOffsets.h_view(vertexOrdinal) = runningOffset;
    runningOffset += static_cast<IndexType>(incidentCellOrdinals[vertexOrdinal].size());
  }
  _vertexToCellOffsets.h_view(vertexCount_) = runningOffset;

  _vertexToCellOrdinals = VertexToCellOrdinalsDualView("component_vertex_to_cell_ordinals", runningOffset);
  IndexType entryOrdinal = 0;
  for (IndexType vertexOrdinal=0; vertexOrdinal<vertexCount_; vertexOrdinal++)
  {
    for (const auto &cellOrdinal : incidentCellOrdinals[vertexOrdinal])
    {
      _vertexToCellOrdinals.h_view(entryOrdinal) = cellOrdinal;
      _vertexToActiveCellsHost.push_back(_activeCells[cellOrdinal]);
      entryOrdinal++;
    }
  }

  _activeVertexCoordinates.modify_host();
  _activeVertexCoordinates.sync_device();
  _cellVertices.modify_host();
  _cellVertices.sync_device();
  _vertexToCellOffsets.modify_host();
  _vertexToCellOffsets.sync_device();
  _vertexToCellOrdinals.modify_host();
  _vertexToCellOrdinals.sync_device();
}

ComponentMeshTopology1D::ComponentMeshTopology1D(const std::vector<double> &coarseVertices,
                                                 const std::vector<ActiveLeafSet1D> &activeLeafSets,
                                                 bool isPeriodic,
                                                 int defaultPolyOrder)
: _coarseVertices(coarseVertices), _activeLeafSets(activeLeafSets), _isPeriodic(isPeriodic), _defaultPolyOrder(defaultPolyOrder)
{
  validateConstructorInputs();
  rebuildActiveCells();
  initializePolyOrders();
  rebuildVertexIncidence();
}

IndexType ComponentMeshTopology1D::rootCellCount() const
{
  return static_cast<IndexType>(_coarseVertices.size() - 1);
}

IndexType ComponentMeshTopology1D::activeCellCount() const
{
  return static_cast<IndexType>(_activeCells.size());
}

IndexType ComponentMeshTopology1D::vertexCount() const
{
  return static_cast<IndexType>(_activeVertexCoordinates.extent(0));
}

bool ComponentMeshTopology1D::isPeriodic() const
{
  return _isPeriodic;
}

int ComponentMeshTopology1D::defaultPolyOrder() const
{
  return _defaultPolyOrder;
}

double ComponentMeshTopology1D::rootCellLeft(IndexType rootOrdinal) const
{
  return _coarseVertices[rootOrdinal];
}

double ComponentMeshTopology1D::rootCellRight(IndexType rootOrdinal) const
{
  return _coarseVertices[rootOrdinal + 1];
}

const ActiveLeafSet1D &ComponentMeshTopology1D::activeLeafSet(IndexType rootOrdinal) const
{
  return _activeLeafSets[rootOrdinal];
}

const std::vector<ComponentCellID1D> &ComponentMeshTopology1D::activeCells() const
{
  return _activeCells;
}

std::vector<ComponentCellID1D> ComponentMeshTopology1D::activeCellsForRoot(IndexType rootOrdinal) const
{
  return _activeLeafSets[rootOrdinal].cellIDsForRoot(rootOrdinal);
}

const std::vector<ComponentCellID1D> &ComponentMeshTopology1D::activeCellsForVertex(IndexType vertexIndex) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((vertexIndex < 0) || (vertexIndex >= vertexCount()), std::invalid_argument,
                             "ComponentMeshTopology1D::activeCellsForVertex vertex index out of range.");
  return _vertexToActiveCellsHost;
}

bool ComponentMeshTopology1D::containsCell(const ComponentCellID1D &cellID) const
{
  if (!cellID.isValid()) return false;
  if (cellID.rootOrdinal() < 0) return false;
  if (cellID.rootOrdinal() >= rootCellCount()) return false;

  auto rootCells = _activeLeafSets[cellID.rootOrdinal()].cellIDsForRoot(cellID.rootOrdinal());
  for (const auto &candidate : rootCells)
  {
    if (candidate == cellID) return true;
  }
  return false;
}

IndexType ComponentMeshTopology1D::activeCellOrdinal(const ComponentCellID1D &cellID) const
{
  auto entry = _activeCellOrdinals.find(cellID);
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _activeCellOrdinals.end(), std::invalid_argument,
                             "ComponentMeshTopology1D::activeCellOrdinal requires an active cell in the topology.");
  return entry->second;
}

int ComponentMeshTopology1D::cellPolyOrder(const ComponentCellID1D &cellID) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!containsCell(cellID), std::invalid_argument,
                             "ComponentMeshTopology1D::cellPolyOrder requires an active cell in the topology.");
  return _cellPolyOrders.find(cellID)->second;
}

void ComponentMeshTopology1D::setCellPolyOrder(const ComponentCellID1D &cellID, int polyOrder)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!containsCell(cellID), std::invalid_argument,
                             "ComponentMeshTopology1D::setCellPolyOrder requires an active cell in the topology.");
  TEUCHOS_TEST_FOR_EXCEPTION(polyOrder < 0, std::invalid_argument,
                             "ComponentMeshTopology1D::setCellPolyOrder requires nonnegative polynomial order.");
  _cellPolyOrders[cellID] = polyOrder;
}

void ComponentMeshTopology1D::refineCell(const ComponentCellID1D &cellID)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!containsCell(cellID), std::invalid_argument,
                             "ComponentMeshTopology1D::refineCell requires an active cell in the topology.");

  const IndexType rootOrdinal = cellID.rootOrdinal();
  auto entries = _activeLeafSets[rootOrdinal].entries();

  std::vector<ActiveLeafSet1D::Entry> newEntries;
  newEntries.reserve(entries.size() + 1);

  bool foundCell = false;
  for (const auto &entry : entries)
  {
    if ((entry.level() == cellID.level()) && (entry.address() == cellID.address()))
    {
      foundCell = true;
    }
    else
    {
      newEntries.push_back(entry);
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(!foundCell, std::invalid_argument,
                             "ComponentMeshTopology1D::refineCell could not locate the requested active leaf.");

  newEntries.push_back(ActiveLeafSet1D::Entry(cellID.level() + 1, 2ULL * cellID.address()));
  newEntries.push_back(ActiveLeafSet1D::Entry(cellID.level() + 1, 2ULL * cellID.address() + 1ULL));

  const int parentPolyOrder = cellPolyOrder(cellID);

  _activeLeafSets[rootOrdinal] = ActiveLeafSet1D(newEntries);
  rebuildActiveCells();
  rebuildVertexIncidence();

  _cellPolyOrders.erase(cellID);
  _cellPolyOrders[ComponentCellID1D(rootOrdinal, cellID.level() + 1, 2ULL * cellID.address())] = parentPolyOrder;
  _cellPolyOrders[ComponentCellID1D(rootOrdinal, cellID.level() + 1, 2ULL * cellID.address() + 1ULL)] = parentPolyOrder;
}

IndexType ComponentMeshTopology1D::cellVertexIndex(const ComponentCellID1D &cellID, int vertexOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!containsCell(cellID), std::invalid_argument,
                             "ComponentMeshTopology1D::cellVertexIndex requires an active cell in the topology.");
  TEUCHOS_TEST_FOR_EXCEPTION((vertexOrdinal < 0) || (vertexOrdinal > 1), std::invalid_argument,
                             "ComponentMeshTopology1D::cellVertexIndex vertex ordinal must be 0 or 1.");
  return _cellVertices.h_view(_activeCellOrdinals.find(cellID)->second, vertexOrdinal);
}

const ComponentMeshTopology1D::VertexCoordinateDualView &ComponentMeshTopology1D::activeVertexCoordinates() const
{
  return _activeVertexCoordinates;
}

const ComponentMeshTopology1D::CellVerticesDualView &ComponentMeshTopology1D::cellVertices() const
{
  return _cellVertices;
}

const ComponentMeshTopology1D::VertexToCellOffsetsDualView &ComponentMeshTopology1D::vertexToCellOffsets() const
{
  return _vertexToCellOffsets;
}

const ComponentMeshTopology1D::VertexToCellOrdinalsDualView &ComponentMeshTopology1D::vertexToCellOrdinals() const
{
  return _vertexToCellOrdinals;
}

IndexType ComponentMeshTopology1D::vertexToCellOffsetHost(IndexType vertexIndex) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((vertexIndex < 0) || (vertexIndex > vertexCount()), std::invalid_argument,
                             "ComponentMeshTopology1D::vertexToCellOffsetHost vertex index out of range.");
  return _vertexToCellOffsets.h_view(vertexIndex);
}

IndexType ComponentMeshTopology1D::vertexToCellOrdinalHost(IndexType entryOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((entryOrdinal < 0) || (entryOrdinal >= static_cast<IndexType>(_vertexToCellOrdinals.extent(0))), std::invalid_argument,
                             "ComponentMeshTopology1D::vertexToCellOrdinalHost entry ordinal out of range.");
  return _vertexToCellOrdinals.h_view(entryOrdinal);
}

double ComponentMeshTopology1D::cellLeft(const ComponentCellID1D &cellID) const
{
  const double x0 = rootCellLeft(cellID.rootOrdinal());
  const double x1 = rootCellRight(cellID.rootOrdinal());
  const double width = x1 - x0;
  const double scale = 1.0 / static_cast<double>(1ULL << cellID.level());
  return x0 + width * static_cast<double>(cellID.address()) * scale;
}

double ComponentMeshTopology1D::cellRight(const ComponentCellID1D &cellID) const
{
  const double x0 = rootCellLeft(cellID.rootOrdinal());
  const double x1 = rootCellRight(cellID.rootOrdinal());
  const double width = x1 - x0;
  const double scale = 1.0 / static_cast<double>(1ULL << cellID.level());
  return x0 + width * static_cast<double>(cellID.address() + 1ULL) * scale;
}

double ComponentMeshTopology1D::cellWidth(const ComponentCellID1D &cellID) const
{
  return cellRight(cellID) - cellLeft(cellID);
}
