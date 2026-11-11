//
//  BoxMeshTopology.cpp
//  Camellia
//
//  See BoxMeshTopology.h.
//

#include "BoxMeshTopology.h"

#include "CamelliaCellTools.h"
#include "MeshTopology.h"
#include "PeriodicBC.h"

#include <algorithm>
#include <cmath>

using namespace Camellia;
using namespace std;

// ---------------------------------------------------------------- BoxCell

BoxCell::BoxCell(CellTopoPtr cellTopo, const vector<IndexType> &vertices, const vector<vector<unsigned>> &subcellPermutations,
                 IndexType cellIndex, const BoxMeshTopology* boxTopology)
: Cell(cellTopo, vertices, subcellPermutations, cellIndex, NULL), _boxTopology(boxTopology)
{}

IndexType BoxCell::entityIndex(unsigned subcdim, unsigned subcord)
{
  const unsigned sideDim = topology()->getDimension() - 1;
  TEUCHOS_TEST_FOR_EXCEPTION(subcdim != sideDim, std::invalid_argument, "BoxCell: box meshes have numbered entities for sides only");
  return cellIndex() * IndexType(getSideCount()) + IndexType(subcord);
}

vector<IndexType> BoxCell::entitiesOnNeighborInterfaces(unsigned dimensionOfInterest, bool peersOnly, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxCell::entitiesOnNeighborInterfaces() is not supported: box meshes have no numbered entities");
}

vector<pair<IndexType,unsigned>> BoxCell::getDescendantsForSide(int sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity, bool leafNodesOnly)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!leafNodesOnly, std::invalid_argument, "BoxCell::getDescendantsForSide() supports leafNodesOnly = true only");
  const BoxForest &forest = _boxTopology->forest();
  const int dim = forest.dimension();
  const int face = _boxTopology->faceForSideOrdinal(sideOrdinal), k = face / 2;
  const BoxForest::Box &box = forest.node(cellIndex()).box;
  std::array<BoxForest::Coord,BoxForest::MAX_DIM> lo = box.lo, hi{};
  for (int j=0; j<dim; j++) hi[j] = box.hi(j);
  if (face % 2 == 1) lo[k] = hi[k] - 1; // the layer of finest-level units just inside the side
  else hi[k] = lo[k] + 1;
  vector<pair<IndexType,unsigned>> descendants;
  forest.forEachLeafOverlapping(lo, hi, [&](int node){ descendants.push_back({IndexType(node), unsigned(sideOrdinal)}); });
  return descendants;
}

set<GlobalIndexType> BoxCell::getActiveNeighborIndices(ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  set<GlobalIndexType> neighbors;
  for (unsigned sideOrdinal=0; sideOrdinal<getSideCount(); sideOrdinal++)
  {
    for (IndexType neighbor : _boxTopology->sideNeighbors(cellIndex(), sideOrdinal)) neighbors.insert(neighbor);
  }
  return neighbors;
}

pair<GlobalIndexType,unsigned> BoxCell::getNeighborInfo(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  vector<IndexType> neighbors = _boxTopology->sideNeighbors(cellIndex(), sideOrdinal);
  if (neighbors.size() == 0) return {-1, -1};
  const int face = _boxTopology->faceForSideOrdinal(sideOrdinal);
  const int oppositeFace = (face % 2 == 1) ? face - 1 : face + 1;
  return {neighbors[0], unsigned(_boxTopology->sideOrdinalForFace(oppositeFace))};
}

CellPtr BoxCell::getNeighbor(unsigned sideOrdinal, ConstMeshTopologyViewPtr meshTopoViewForCellValidity)
{
  GlobalIndexType neighborID = getNeighborInfo(sideOrdinal, meshTopoViewForCellValidity).first;
  if (neighborID == -1) return Teuchos::null;
  return _boxTopology->getCell(neighborID);
}

bool BoxCell::isBoundary(unsigned sideOrdinal)
{
  return _boxTopology->sideIsOnBoundary(cellIndex(), sideOrdinal);
}

RefinementBranch BoxCell::refinementBranch(IndexType ancestralCellID)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxCell::refinementBranch() is not supported: box meshes have no RefinementPatterns");
}

// ---------------------------------------------------------------- BoxMeshTopology

BoxMeshTopology::BoxMeshTopology(const vector<vector<double>> &rootVertices)
: _forest([&]{ vector<int> counts; for (const auto &v : rootVertices) counts.push_back(int(v.size()) - 1); return counts; }()),
  _rootVertices(rootVertices), _dim(int(rootVertices.size()))
{
  TEUCHOS_TEST_FOR_EXCEPTION(_dim < 2, std::invalid_argument, "BoxMeshTopology requires dimension >= 2");
  if (_dim == 2) _cellTopo = CellTopology::quad();
  else if (_dim == 3) _cellTopo = CellTopology::hexahedron();
  else _cellTopo = CellTopology::cellTopology(CellTopology::line(), _dim - 1);

  _refNodes.resize(_cellTopo->getNodeCount(), _dim);
  CamelliaCellTools::refCellNodesForTopology(_refNodes, _cellTopo);
  const int sideCount = _cellTopo->getSideCount();
  _sideForFace.assign(2 * _dim, -1);
  _faceForSide.assign(sideCount, -1);
  for (int side=0; side<sideCount; side++)
  {
    const int nodeCount = _cellTopo->getSide(side)->getNodeCount();
    for (int k=0; k<_dim; k++)
    {
      const double value = _refNodes(_cellTopo->getNodeMap(_dim-1, side, 0), k);
      bool constant = true;
      for (int n=1; n<nodeCount; n++)
      {
        if (std::abs(_refNodes(_cellTopo->getNodeMap(_dim-1, side, n), k) - value) > 1e-12) constant = false;
      }
      if (constant)
      {
        const int face = 2*k + (value > 0 ? 1 : 0);
        _sideForFace[face] = side;
        _faceForSide[side] = face;
      }
    }
  }
  for (int side : _sideForFace) TEUCHOS_TEST_FOR_EXCEPTION(side < 0, std::logic_error, "BoxMeshTopology: could not identify the sides of the cell topology");

  for (int node=0; node<_forest.nodeCount(); node++)
  {
    _cells.push_back(createCell(node));
    _rootCells.insert(node);
    _activeCells.insert(node);
  }
}

CellPtr BoxMeshTopology::createCell(IndexType cellIndex)
{
  const int nodeCount = _cellTopo->getNodeCount();
  vector<IndexType> vertices(nodeCount);
  for (int n=0; n<nodeCount; n++) vertices[n] = cellIndex * nodeCount + n; // synthetic: see getVertex()
  vector<vector<unsigned>> subcellPermutations(_dim + 1);
  for (int d=0; d<=_dim; d++) subcellPermutations[d].assign(_cellTopo->getSubcellCount(d), 0);
  return Teuchos::rcp(new BoxCell(_cellTopo, vertices, subcellPermutations, cellIndex, this));
}

const BoxForest::Skeleton &BoxMeshTopology::skeleton() const
{
  if (!_skeletonValid)
  {
    _skeleton = _forest.skeleton();
    _skeletonValid = true;
  }
  return _skeleton;
}

double BoxMeshTopology::coordinate(int k, BoxForest::Coord c) const
{
  const int i = int(c >> BoxForest::MAX_LEVEL);
  if (i == int(_rootVertices[k].size()) - 1) return _rootVertices[k][i];
  const double fraction = std::ldexp(double(c - (BoxForest::Coord(i) << BoxForest::MAX_LEVEL)), -BoxForest::MAX_LEVEL);
  return _rootVertices[k][i] + fraction * (_rootVertices[k][i+1] - _rootVertices[k][i]);
}

BoxMeshTopology::Extents BoxMeshTopology::cellExtents(IndexType cellIndex) const
{
  const BoxForest::Box &box = _forest.node(cellIndex).box;
  Extents extents(_dim);
  for (int k=0; k<_dim; k++) extents[k] = {coordinate(k, box.lo[k]), coordinate(k, box.hi(k))};
  return extents;
}

bool BoxMeshTopology::sideIsOnBoundary(IndexType cellIndex, unsigned sideOrdinal) const
{
  const int face = _faceForSide[sideOrdinal], k = face / 2;
  const BoxForest::Box &box = _forest.node(cellIndex).box;
  if (face % 2 == 0) return box.lo[k] == 0;
  return box.hi(k) == (BoxForest::Coord(_rootVertices[k].size() - 1) << BoxForest::MAX_LEVEL);
}

vector<IndexType> BoxMeshTopology::sideNeighbors(IndexType cellIndex, unsigned sideOrdinal) const
{
  vector<int> neighbors = _forest.faceNeighbors(cellIndex, _faceForSide[sideOrdinal]);
  return vector<IndexType>(neighbors.begin(), neighbors.end());
}

vector<IndexType> BoxMeshTopology::refineCell(IndexType cellIndex, unsigned directionMask)
{
  const int firstChild = _forest.refine(cellIndex, directionMask);
  const int childCount = _forest.nodeCount() - firstChild;
  vector<IndexType> childIndices;
  vector<CellPtr> children;
  for (int j=0; j<childCount; j++)
  {
    const IndexType childIndex = firstChild + j;
    _cells.push_back(createCell(childIndex));
    children.push_back(_cells[childIndex]);
    childIndices.push_back(childIndex);
    _activeCells.insert(childIndex);
  }
  _cells[cellIndex]->setChildCells(children);
  _activeCells.erase(cellIndex);
  _skeletonValid = false;
  return childIndices;
}

vector<IndexType> BoxMeshTopology::refineCellInDirections(IndexType cellIndex, unsigned directionMask)
{
  return refineCell(cellIndex, directionMask);
}

vector<pair<IndexType,unsigned>> BoxMeshTopology::cellsToRefineForBalance(int maxLevelDifference) const
{
  vector<pair<IndexType,unsigned>> result;
  for (int node : _forest.leaves())
  {
    unsigned mask = 0;
    for (int face=0; face<2*_dim; face++)
    {
      for (int neighbor : _forest.faceNeighbors(node, face))
      {
        for (int j=0; j<_dim; j++)
        {
          if (j == face / 2) continue;
          if (int(_forest.node(neighbor).box.level[j]) > int(_forest.node(node).box.level[j]) + maxLevelDifference) mask |= (1u << j);
        }
      }
    }
    if (mask != 0) result.push_back({IndexType(node), mask});
  }
  return result;
}

map<IndexType,unsigned> BoxMeshTopology::cellsToRefineForRegularity() const
{
  map<IndexType,unsigned> result;
  const unsigned configurationMask = (_configurationTopology == Teuchos::null) ? 0u : ((1u << _configurationDimensionCount) - 1);
  for (const auto &entry : cellsToRefineForBalance())
  {
    unsigned mask = entry.second;
    if (mask & configurationMask) mask |= configurationMask;
    result[entry.first] = mask;
  }
  return result;
}

void BoxMeshTopology::applyTag(const string &tagName, int tagID, SideSelector selector)
{
  _sideTags[{tagName, tagID}].push_back(selector);
}

vector<unsigned> BoxMeshTopology::getSidesMatchingTag(IndexType cellIndex, const string &tagName, int tagID) const
{
  vector<unsigned> sides;
  auto entry = _sideTags.find({tagName, tagID});
  if (entry == _sideTags.end()) return sides;
  Extents extents;
  for (unsigned side=0; side<_cellTopo->getSideCount(); side++)
  {
    if (!sideIsOnBoundary(cellIndex, side)) continue;
    if (extents.size() == 0) extents = cellExtents(cellIndex);
    const int face = _faceForSide[side];
    for (const SideSelector &selector : entry->second)
    {
      if (selector(face / 2, face % 2 == 1, extents))
      {
        sides.push_back(side);
        break;
      }
    }
  }
  return sides;
}

long long BoxMeshTopology::approximateMemoryFootprint() const
{
  return (long long)(_forest.nodeCount()) * (long long)(sizeof(BoxForest::Node) + sizeof(BoxCell) + sizeof(CellPtr));
}

vector<IndexType> BoxMeshTopology::cellIDsForPoints(const Camellia::FieldContainer<double> &physicalPoints) const
{
  const int numPoints = physicalPoints.dimension(0);
  vector<IndexType> cellIDs(numPoints, -1);
  for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
  {
    std::array<BoxForest::Coord,BoxForest::MAX_DIM> lo{}, hi{};
    bool inDomain = true;
    for (int k=0; k<_dim; k++)
    {
      const vector<double> &v = _rootVertices[k];
      const double x = physicalPoints(pointOrdinal, k);
      if ((x < v.front()) || (x > v.back())) { inDomain = false; break; }
      int i = int(std::upper_bound(v.begin(), v.end(), x) - v.begin()) - 1;
      i = std::min(i, int(v.size()) - 2);
      const double fraction = (x - v[i]) / (v[i+1] - v[i]);
      const BoxForest::Coord unitsPerRoot = BoxForest::Coord(1) << BoxForest::MAX_LEVEL;
      BoxForest::Coord offset = BoxForest::Coord(fraction * double(unitsPerRoot));
      if (offset >= unitsPerRoot) offset = unitsPerRoot - 1;
      lo[k] = (BoxForest::Coord(i) << BoxForest::MAX_LEVEL) + offset;
      hi[k] = lo[k] + 1;
    }
    if (!inDomain) continue;
    _forest.forEachLeafOverlapping(lo, hi, [&](int node){ cellIDs[pointOrdinal] = node; });
  }
  return cellIDs;
}

IndexType BoxMeshTopology::cellCount() const { return _forest.nodeCount(); }
IndexType BoxMeshTopology::activeCellCount() const { return IndexType(_activeCells.size()); }
Teuchos_CommPtr BoxMeshTopology::Comm() const { return Teuchos::null; } // replicated
bool BoxMeshTopology::isDistributed() const { return false; }
const MeshTopology* BoxMeshTopology::baseMeshTopology() const { return NULL; }
unsigned BoxMeshTopology::getDimension() const { return _dim; }
const set<IndexType> &BoxMeshTopology::getLocallyKnownActiveCellIndices() const { return _activeCells; }
const set<IndexType> &BoxMeshTopology::getMyActiveCellIndices() const { return _activeCells; }
const set<IndexType> &BoxMeshTopology::getRootCellIndicesLocal() const { return _rootCells; }
bool BoxMeshTopology::isParent(IndexType cellIndex) const { return !_forest.isLeaf(cellIndex); }
bool BoxMeshTopology::isValidCellIndex(IndexType cellIndex) const { return (cellIndex >= 0) && (cellIndex < _forest.nodeCount()); }

CellPtr BoxMeshTopology::getCell(IndexType cellIndex) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(!isValidCellIndex(cellIndex), std::invalid_argument, "BoxMeshTopology::getCell: invalid cell index");
  return _cells[cellIndex];
}

vector<double> BoxMeshTopology::getCellCentroid(IndexType cellIndex) const
{
  Extents extents = cellExtents(cellIndex);
  vector<double> centroid(_dim);
  for (int k=0; k<_dim; k++) centroid[k] = 0.5 * (extents[k].first + extents[k].second);
  return centroid;
}

bool BoxMeshTopology::getCellSideParities(IndexType cellIndex, vector<int> &parities) const
{
  // the flux on a face is that of the cell below it; on the boundary, the outward flux
  const int sideCount = _cellTopo->getSideCount();
  parities.resize(sideCount);
  for (int side=0; side<sideCount; side++)
  {
    const bool upper = (_faceForSide[side] % 2 == 1);
    parities[side] = (upper || sideIsOnBoundary(cellIndex, side)) ? 1 : -1;
  }
  return true;
}

const vector<double>& BoxMeshTopology::getVertex(IndexType vertexIndex) const
{
  auto entry = _vertexCache.find(vertexIndex);
  if (entry != _vertexCache.end()) return entry->second;
  const int nodeCount = _cellTopo->getNodeCount();
  const IndexType cellIndex = vertexIndex / nodeCount;
  const int node = vertexIndex % nodeCount;
  const BoxForest::Box &box = _forest.node(cellIndex).box;
  vector<double> vertex(_dim);
  for (int k=0; k<_dim; k++) vertex[k] = (_refNodes(node,k) > 0) ? coordinate(k, box.hi(k)) : coordinate(k, box.lo[k]);
  return _vertexCache[vertexIndex] = vertex;
}

void BoxMeshTopology::verticesForCell(Camellia::FieldContainer<double>& vertices, IndexType cellID) const
{
  const BoxForest::Box &box = _forest.node(cellID).box;
  const int nodeCount = _cellTopo->getNodeCount();
  vertices.resize(nodeCount, _dim);
  for (int n=0; n<nodeCount; n++)
    for (int k=0; k<_dim; k++)
      vertices(n,k) = (_refNodes(n,k) > 0) ? coordinate(k, box.hi(k)) : coordinate(k, box.lo[k]);
}

Camellia::FieldContainer<double> BoxMeshTopology::physicalCellNodesForCell(IndexType cellIndex, bool includeCellDimension) const
{
  Camellia::FieldContainer<double> nodes;
  verticesForCell(nodes, cellIndex);
  if (includeCellDimension) nodes.resize(1, nodes.dimension(0), nodes.dimension(1));
  return nodes;
}

Teuchos::RCP<MeshTopology> BoxMeshTopology::deepCopy() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxMeshTopology::deepCopy() is not supported");
}

MeshTopologyViewPtr BoxMeshTopology::getView(const set<IndexType> &activeCellIndices) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxMeshTopology::getView() is not supported");
}

MeshTopologyPtr BoxMeshTopology::getGatheredCopy() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxMeshTopology::getGatheredCopy() is not supported");
}

MeshTopologyPtr BoxMeshTopology::getGatheredCopy(const set<IndexType> &cellsToInclude) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BoxMeshTopology::getGatheredCopy() is not supported");
}

const vector< Teuchos::RCP<PeriodicBC> > & BoxMeshTopology::getPeriodicBCs() const
{
  static const vector< Teuchos::RCP<PeriodicBC> > noPeriodicBCs;
  return noPeriodicBCs;
}

// ---------------------------------------------------------------- phase space

void BoxMeshTopology::setConfigurationTopology(MeshTopologyViewPtr configurationTopology, int configurationDimensionCount)
{
  TEUCHOS_TEST_FOR_EXCEPTION((configurationDimensionCount < 1) || (configurationDimensionCount >= _dim), std::invalid_argument,
                             "invalid configurationDimensionCount");
  TEUCHOS_TEST_FOR_EXCEPTION(int(configurationTopology->getDimension()) != configurationDimensionCount, std::invalid_argument,
                             "configurationTopology dimension does not match configurationDimensionCount");
  _configurationTopology = configurationTopology;
  _configurationDimensionCount = configurationDimensionCount;
  _configurationCellForCell.clear();
}

int BoxMeshTopology::phaseSpaceVelocityDimensionCount() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_configurationTopology == Teuchos::null, std::invalid_argument, "BoxMeshTopology: no configuration topology has been set");
  return _dim - _configurationDimensionCount;
}

void BoxMeshTopology::phaseSpaceVelocityExtents(double &v0, double &v1, IndexType phaseSpaceCellID, int velocityDimension) const
{
  const int k = _configurationDimensionCount + velocityDimension;
  const BoxForest::Box &box = _forest.node(phaseSpaceCellID).box;
  v0 = coordinate(k, box.lo[k]);
  v1 = coordinate(k, box.hi(k));
}

IndexType BoxMeshTopology::phaseSpaceConfigurationCellID(IndexType phaseSpaceCellID) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(_configurationTopology == Teuchos::null, std::invalid_argument, "BoxMeshTopology: no configuration topology has been set");
  auto cached = _configurationCellForCell.find(phaseSpaceCellID);
  if (cached != _configurationCellForCell.end()) return cached->second;

  const BoxForest::Box &box = _forest.node(phaseSpaceCellID).box;
  const int level = box.level[0];
  for (int k=1; k<_configurationDimensionCount; k++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(int(box.level[k]) != level, std::invalid_argument,
                               "BoxMeshTopology: phase-space cells must be at the same refinement level in every configuration direction");
  }
  // the active configuration cell at the centroid of the projection, then up the tree to the projection's level
  Camellia::FieldContainer<double> centroid(1, _configurationDimensionCount);
  for (int k=0; k<_configurationDimensionCount; k++) centroid(0,k) = 0.5 * (coordinate(k, box.lo[k]) + coordinate(k, box.hi(k)));
  IndexType configurationCellID = _configurationTopology->cellIDsForPoints(centroid)[0];
  TEUCHOS_TEST_FOR_EXCEPTION(configurationCellID == -1, std::invalid_argument, "BoxMeshTopology: no configuration cell found under phase-space cell");
  CellPtr configurationCell = _configurationTopology->getCell(configurationCellID);
  TEUCHOS_TEST_FOR_EXCEPTION(configurationCell->level() < level, std::invalid_argument,
                             "BoxMeshTopology: the configuration mesh is coarser than the phase-space mesh");
  while (configurationCell->level() > level) configurationCell = configurationCell->getParent();
  return _configurationCellForCell[phaseSpaceCellID] = configurationCell->cellIndex();
}

// ---------------------------------------------------------------- copies for multigrid

int BoxMeshTopology::refinementDepth() const
{
  int depth = 0;
  for (IndexType cellIndex : _activeCells) depth = std::max(depth, _cells[cellIndex]->level());
  return depth;
}

Teuchos::RCP<BoxMeshTopology> BoxMeshTopology::copyTruncatedAtDepth(int maxDepth) const
{
  Teuchos::RCP<BoxMeshTopology> copy = Teuchos::rcp(new BoxMeshTopology(_rootVertices));
  copy->_sideTags = _sideTags;
  // children are created after their parents, so a pass in node order sees every parent's copy before its children
  vector<IndexType> copyIndex(_forest.nodeCount(), -1);
  for (IndexType root : _rootCells) copyIndex[root] = root; // roots are numbered identically
  for (int node=0; node<_forest.nodeCount(); node++)
  {
    if (_forest.isLeaf(node) || (copyIndex[node] == -1)) continue;
    if (_cells[node]->level() >= maxDepth) continue;
    const BoxForest::Node &forestNode = _forest.node(node);
    vector<IndexType> copyChildren = copy->refineCell(copyIndex[node], forestNode.mask);
    for (size_t j=0; j<copyChildren.size(); j++) copyIndex[forestNode.firstChild + int(j)] = copyChildren[j];
  }
  return copy;
}
