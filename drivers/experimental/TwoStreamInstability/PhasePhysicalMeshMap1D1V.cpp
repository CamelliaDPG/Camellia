//
//  PhasePhysicalMeshMap1D1V.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 5/13/20.
//

#include "PhasePhysicalMeshMap1D1V.h"

#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"

using namespace Camellia;

PhasePhysicalMeshMap1D1V::PhasePhysicalMeshMap1D1V( MeshPtr mesh_1D, MeshPtr mesh_2D )
:
_mesh_1D(mesh_1D),
_mesh_2D(mesh_2D)
{
  Teuchos::RCP<RefinementObserver> thisWeakPtr = Teuchos::rcp(this, false);
  _mesh_2D->registerObserver(thisWeakPtr);
  this->rebuildMap();
  
  _mesh_1D->globalDofAssignment()->setAllowMeshTopologyPruning(false); // want to keep all 1D geometry info available to all MPI ranks
}

PhasePhysicalPointIterator PhasePhysicalMeshMap1D1V::getPhysicalPointIterator(BasisCachePtr phaseSpaceBasisCache, GlobalIndexType phaseSpaceCellID,
                                                                              map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > &physicalBasisCachePairs) const
{
  const auto &physicalCellInfoVector = this->physicalCellInfo(phaseSpaceCellID);
  
  bool isSideCache = (phaseSpaceBasisCache->getSideIndex() != -1);
  int physicalSpaceSideOrdinal;
  if (!isSideCache)
  {
    physicalSpaceSideOrdinal = -1;
  }
  else // isSideCache == true
  {
    const int phaseSpaceSideOrdinal = phaseSpaceBasisCache->getSideIndex();
    // what follows assumes 1D1V with quad elements; let's throw an exception if we're not on a quad
    TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceBasisCache->cellTopology()->getKey() != CellTopology::quad()->getKey(), std::invalid_argument, "Only phase-space quads supported right now");
    if ((phaseSpaceSideOrdinal == 0) || (phaseSpaceSideOrdinal == 2))
    {
      // we're in the physical cell interior
      physicalSpaceSideOrdinal = -1;
    }
    else if (phaseSpaceSideOrdinal == 1)
    {
      physicalSpaceSideOrdinal = 1;
    }
    else
    {
      physicalSpaceSideOrdinal = 0;
    }
  }
  const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
  
  // if we are on the top side (sideOrdinal == 2) of the phase space cell, then because that side is oriented right to left, the
  // points in the phase space BasisCache will be reversed relative to the component physical points
  const bool reversePointOrder = (phaseSpaceBasisCache->getSideIndex() == 2);

  std::vector< std::pair<int,int> > pointStartEndPairs(physicalCellInfoVector.size());
  
  int numPhysicalPoints = 0;
  for (int physicalCellOrdinal = physicalCellInfoVector.size()-1; physicalCellOrdinal >= 0; physicalCellOrdinal--)
  {
    const auto & physicalCellInfo  = physicalCellInfoVector[physicalCellOrdinal];
    auto & physicalCellID          = physicalCellInfo.physicalCellID;
    auto & unrefinementLevel       = physicalCellInfo.unrefinementLevel;
    auto & offsetsInPhysicalValues = physicalBasisCachePairs[physicalCellID].second;
    
    int values1D_pointStart, values1D_pointEnd;
    if (inPhysicalVolume)
    {
      values1D_pointStart = offsetsInPhysicalValues[unrefinementLevel]  ;
      values1D_pointEnd   = offsetsInPhysicalValues[unrefinementLevel+1];
    }
    else
    {
      // if we're on a physical side (1D: a point), then there will be just one point value in values1D
      // the question is: does this physical cell contain that point?
      if ((physicalSpaceSideOrdinal == 0) && (physicalCellOrdinal == 0))
      {
        // the phase-space side abuts the leftmost physical cell, and we're on the leftmost of the physical cells
        values1D_pointStart = 0;
        values1D_pointEnd   = 1;
      }
      else if ((physicalSpaceSideOrdinal == 1) && (physicalCellOrdinal == physicalCellInfoVector.size()-1))
      {
        // the phase-space side abuts the rightmost physical cell, and we're on the rightmost of the physical cells
        values1D_pointStart = 0;
        values1D_pointEnd   = 1;
      }
      else
      {
        // we don't match; take no points
        values1D_pointStart = -1;
        values1D_pointEnd   = -1;
      }
    }
    pointStartEndPairs[physicalCellOrdinal] = {values1D_pointStart, values1D_pointEnd};
    numPhysicalPoints += values1D_pointEnd - values1D_pointStart;
  }
  
  return PhasePhysicalPointIterator(physicalCellInfoVector, pointStartEndPairs, reversePointOrder, numPhysicalPoints);
}

int PhasePhysicalMeshMap1D1V::maximumUnrefinementLevel() const
{
  int unrefinementLevel = 0; // 0 means each phase space cell is the same width as the sole corresponding physical cell
  for (const auto & cellIDToCellInfoVectorEntry : _phaseSpaceCellIDToPhysicalCellInfo)
  {
    const auto & cellInfoVector = cellIDToCellInfoVectorEntry.second;
    for (const auto & cellInfo : cellInfoVector)
    {
      unrefinementLevel = std::max(cellInfo.unrefinementLevel, unrefinementLevel);
    }
  }
  return unrefinementLevel;
}

MeshPtr PhasePhysicalMeshMap1D1V::mesh1D() const
{
  return _mesh_1D;
}

MeshPtr PhasePhysicalMeshMap1D1V::mesh2D() const
{
  return _mesh_2D;
}

const std::vector<PhysicalCellRelativeRefinementInfo> & PhasePhysicalMeshMap1D1V::physicalCellInfo(const GlobalIndexType &phaseSpaceCellID) const
{
  auto entry = _phaseSpaceCellIDToPhysicalCellInfo.find(phaseSpaceCellID);
  if (entry == _phaseSpaceCellIDToPhysicalCellInfo.end())
  {
    int myPID = _mesh_2D->Comm()->MyPID();
    std::cout << "phaseSpaceCellID " << phaseSpaceCellID << " not found on rank " << myPID << std::endl;
    std::cout << "valid phaseSpaceCellIDs on rank " << myPID << ": ";
    const auto & my2DCells = _mesh_2D->cellIDsInPartition();
    for (auto & my2DCell : my2DCells)
    {
      std::cout << my2DCell << " ";
    }
    std::cout << std::endl << std::flush;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _phaseSpaceCellIDToPhysicalCellInfo.end(), std::invalid_argument, "phaseSpaceCellID not found");
  return entry->second;
}

std::set<GlobalIndexType> PhasePhysicalMeshMap1D1V::physicalCellIDsForPhaseSpaceCells(const std::set<GlobalIndexType> &phaseSpaceCellIDs) const
{
  std::set<GlobalIndexType> physicalCellIDs;
  for (auto & phaseSpaceCellID : phaseSpaceCellIDs)
  {
    auto entryIt = _phaseSpaceCellIDToPhysicalCellInfo.find(phaseSpaceCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(entryIt == _phaseSpaceCellIDToPhysicalCellInfo.end(), std::invalid_argument, "phase space cellID not found");
    auto & physicalCellInfoVector = entryIt->second;
    for (auto & cellInfo : physicalCellInfoVector)
    {
      physicalCellIDs.insert(cellInfo.physicalCellID);
    }
  }
  return physicalCellIDs;
}

std::vector<GlobalIndexType> PhasePhysicalMeshMap1D1V::physicalCellIDsForPhaseSpaceCells(const std::vector<GlobalIndexType> &phaseSpaceCellIDs) const
{
  std::set<GlobalIndexType> physicalCellIDs;
  for (auto & phaseSpaceCellID : phaseSpaceCellIDs)
  {
    auto entryIt = _phaseSpaceCellIDToPhysicalCellInfo.find(phaseSpaceCellID);
    TEUCHOS_TEST_FOR_EXCEPTION(entryIt == _phaseSpaceCellIDToPhysicalCellInfo.end(), std::invalid_argument, "phase space cellID not found");
    auto & physicalCellInfoVector = entryIt->second;
    for (auto & cellInfo : physicalCellInfoVector)
    {
      physicalCellIDs.insert(cellInfo.physicalCellID);
    }
  }
  std::vector<GlobalIndexType> physicalCellIDsVector(physicalCellIDs.begin(),physicalCellIDs.end());
  return physicalCellIDsVector;
}

void PhasePhysicalMeshMap1D1V::rebuildMap()
{
  // Note that much of the logic here, and in this class more generally, does assume grid-aligned quad elements in 2D
  // We also assume that the 1D mesh is nowhere coarser than the 2D mesh
  auto & myCells_2D = _mesh_2D->cellIDsInPartition();
  std::vector<double> vertex_1D(1);
  _phaseSpaceCellIDToPhysicalCellInfo.clear();
  
  const int DIM_X = 0;
  
  for (auto cell_2D : myCells_2D)
  {
    auto meshTopo_2D = _mesh_2D->getTopology();
    auto vertexIndices = meshTopo_2D->getCell(cell_2D)->vertices();
    double xLeft = meshTopo_2D->getVertex(vertexIndices[0])[DIM_X];
    double width = meshTopo_2D->getVertex(vertexIndices[1])[DIM_X] - xLeft;
    // we're assuming here that the vertex indices are laid out in CCW order, and have cartesian structure
    // let's confirm that:
    {
      if (width <= 0)
      {
        std::cout << "pruning ordinal " << meshTopo_2D->baseMeshTopology()->pruningOrdinal() << std::endl;
        Camellia::print("non-positive-width cell, vertexIndices", vertexIndices);
        TEUCHOS_TEST_FOR_EXCEPTION(width <= 0, std::invalid_argument, "element width must be positive!");
      }
      double topWidth = meshTopo_2D->getVertex(vertexIndices[2])[DIM_X] - meshTopo_2D->getVertex(vertexIndices[3])[DIM_X];
      const double diff = std::fabs(topWidth-width);
      const double tol  = std::max(1e-12 * width, 1e-12);
      if (diff > tol)
      {
        using std::to_string;
        using std::cout;
        using std::endl;
        cout << "width at top and bottom of element are required to match! topWidth: " << topWidth << "; bottom width: " << width << " (diff: " << diff << ")\n";
        TEUCHOS_TEST_FOR_EXCEPTION( diff > tol , std::invalid_argument, "width at top and bottom of element are required to match! topWidth: " + to_string(topWidth) + "; bottom width: " + to_string(width) + " (diff: " + to_string(diff) + ")");
      }
    }
    vertex_1D[0] = xLeft;
    auto meshTopo_1D = _mesh_1D->getTopology();
    IndexType leftVertexIndex;
    const unsigned leftVertexOrdinal  = 0;
    const unsigned rightVertexOrdinal = 1;
    const unsigned vertexDim          = 0;
    bool vertexFound = meshTopo_1D->getVertexIndex(vertex_1D, leftVertexIndex);
    if (!vertexFound)
    {
      // for debugging, try the lookup again:
      meshTopo_1D->getVertexIndex(vertex_1D, leftVertexIndex);
      meshTopo_1D->printAllEntitiesInBaseMeshTopology();
      TEUCHOS_TEST_FOR_EXCEPTION(!vertexFound, std::invalid_argument, "vertex at coordinate " + std::to_string(xLeft) + " not found!")
    }
    std::vector<IndexType> vertexNodeVector {leftVertexIndex};
    IndexType leftVertexIndexForCellLookup = meshTopo_1D->baseMeshTopology()->getCanonicalEntityNodesViaPeriodicBCs(vertexDim, vertexNodeVector)[0];
    
    set< pair<IndexType, unsigned> > cellPairsWithVertex = meshTopo_1D->getCellsContainingEntity(0, leftVertexIndexForCellLookup);
    IndexType physicalCellIDLeft = -1;
    for (auto cellPair_1D : cellPairsWithVertex)
    {
      const unsigned & vertexOrdinalInCell = cellPair_1D.second;
      if (vertexOrdinalInCell == leftVertexOrdinal)
      {
        physicalCellIDLeft = cellPair_1D.first;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(physicalCellIDLeft == -1, std::invalid_argument, "Physical cell not found");
    CellPtr physicalCellLeft = meshTopo_1D->getCell(physicalCellIDLeft);
    
    IndexType rightVertexIndex;
    vertex_1D[0] = xLeft + width; // xRight
    vertexFound = meshTopo_1D->getVertexIndex(vertex_1D, rightVertexIndex);
    if (!vertexFound)
    {
      // for debugging, try the lookup again:
      meshTopo_1D->getVertexIndex(vertex_1D, leftVertexIndex);
      meshTopo_1D->printAllEntitiesInBaseMeshTopology();
      TEUCHOS_TEST_FOR_EXCEPTION(!vertexFound, std::invalid_argument, "vertex at coordinate " + std::to_string(xLeft) + " not found!")
    }
    vertexNodeVector[0] = rightVertexIndex;
    IndexType rightVertexIndexForCellLookup = meshTopo_1D->baseMeshTopology()->getCanonicalEntityNodesViaPeriodicBCs(vertexDim, vertexNodeVector)[0];
    cellPairsWithVertex = meshTopo_1D->getCellsContainingEntity(0, rightVertexIndexForCellLookup);
    IndexType physicalCellIDRight = -1;
    for (auto cellPair_1D : cellPairsWithVertex)
    {
      const unsigned & vertexOrdinalInCell = cellPair_1D.second;
      if (vertexOrdinalInCell == rightVertexOrdinal)
      {
        physicalCellIDRight = cellPair_1D.first;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(physicalCellIDRight == -1, std::invalid_argument, "Physical cell not found");
    
    // find the ancestor whose width corresponds with the phase space cell width
    GlobalIndexType ancestorCellID = physicalCellIDLeft;
    int numUnrefinementLevels = 0;
    if (physicalCellIDLeft != physicalCellIDRight)
    {
      auto ancestor = physicalCellLeft;
      auto ancestorRightVertexIndex = ancestor->vertices()[rightVertexOrdinal];
      while (ancestorRightVertexIndex != rightVertexIndex)
      {
        ancestor = ancestor->getParent();
        ancestorRightVertexIndex = ancestor->vertices()[rightVertexOrdinal];
        numUnrefinementLevels++;
      }
      ancestorCellID = ancestor->cellIndex();
    }
    
    // now, march along physicalCells from left until we get to right
    double refSpaceXCoordThusFar = -1.0;
    std::vector<PhysicalCellRelativeRefinementInfo> physicalCellInfoVector;
    PhysicalCellRelativeRefinementInfo physicalCellInfo;
    physicalCellInfo.physicalCellID = physicalCellIDLeft;
    physicalCellInfo.uniformDescendantNumber = 0;
    physicalCellInfo.unrefinementLevel = numUnrefinementLevels;
    physicalCellInfoVector.push_back(physicalCellInfo);
    
    if (numUnrefinementLevels > 0)
    {
      if (_physicalCellIDToRefSpaceXCoordInAncestor[physicalCellIDLeft].size() < numUnrefinementLevels)
      {
        _physicalCellIDToRefSpaceXCoordInAncestor[physicalCellIDLeft].resize(numUnrefinementLevels);
      }
      _physicalCellIDToRefSpaceXCoordInAncestor[physicalCellIDLeft][numUnrefinementLevels-1] = refSpaceXCoordThusFar;
    }
    
    GlobalIndexType rightmostPhysicalCellID = physicalCellIDLeft;
    while (rightmostPhysicalCellID != physicalCellIDRight)
    {
      CellPtr rightmostPhysicalCell = meshTopo_1D->getCell(rightmostPhysicalCellID);
      const double previousCellWidthInAncestor = 2.0 / (1 << numUnrefinementLevels);
      refSpaceXCoordThusFar += previousCellWidthInAncestor;
      const int previousUnrefinementLevel = physicalCellInfo.unrefinementLevel;
      const int descendantNumberInPreviousUnrefinementLevel = physicalCellInfo.uniformDescendantNumber + 1;
      auto neighbor = rightmostPhysicalCell->getNeighbor(rightVertexOrdinal, meshTopo_1D);
      physicalCellInfo.physicalCellID = neighbor->cellIndex();
      auto ancestor = neighbor;
      numUnrefinementLevels = 0;
      while (ancestor->cellIndex() != ancestorCellID)
      {
        ancestor = ancestor->getParent();
        numUnrefinementLevels++;
      }
      const int previousLevelTotalDescendantCount = 1 << previousUnrefinementLevel;
      const int totalDescendantCount = 1 << numUnrefinementLevels;
      
      physicalCellInfo.unrefinementLevel = numUnrefinementLevels;
      physicalCellInfo.uniformDescendantNumber = (descendantNumberInPreviousUnrefinementLevel * totalDescendantCount) / previousLevelTotalDescendantCount;
      physicalCellInfoVector.push_back(physicalCellInfo);
      
      if (numUnrefinementLevels > 0)
      {
        if (_physicalCellIDToRefSpaceXCoordInAncestor[physicalCellInfo.physicalCellID].size() < numUnrefinementLevels)
        {
          _physicalCellIDToRefSpaceXCoordInAncestor[physicalCellInfo.physicalCellID].resize(numUnrefinementLevels);
        }
        _physicalCellIDToRefSpaceXCoordInAncestor[physicalCellInfo.physicalCellID][numUnrefinementLevels-1] = refSpaceXCoordThusFar;
      }
      
      rightmostPhysicalCellID = neighbor->cellIndex();
    }
    _phaseSpaceCellIDToPhysicalCellInfo[cell_2D] = physicalCellInfoVector;
  }
}

std::set< std::pair<int,int> > PhasePhysicalMeshMap1D1V::referenceSpaceUnrefinementLevelDescendantNumberPairs() const
{
  std::set< std::pair<int,int> > levelNumberPairs;
  for (const auto & cellEntriesVectorPair : _phaseSpaceCellIDToPhysicalCellInfo)
  {
    const auto & entriesVector = cellEntriesVectorPair.second;
    for (const auto & entry : entriesVector)
    {
      const auto & unrefinementLevel = entry.unrefinementLevel;
      const auto & descendantNumber  = entry.uniformDescendantNumber;
      levelNumberPairs.insert({unrefinementLevel,descendantNumber});
    }
  }
  return levelNumberPairs;
}

//! Returns BasisCache, alongside a vector indicating point offsets for each ancestral level.
std::pair<BasisCachePtr, std::vector<int> > PhasePhysicalMeshMap1D1V::physicalCellBasisCache(const GlobalIndexType physicalCellID, bool testVsTest, int quadratureEnrichment) const
{
  auto elemType = _mesh_1D->getElementType(physicalCellID);
  BasisCachePtr basisCache =  BasisCache::basisCacheForCellType(_mesh_1D, elemType, testVsTest, quadratureEnrichment);
  auto refPointsStandard = basisCache->getRefCellPointsFC();
  std::vector<double> xCoordsInAncestors;
  auto physicalCellNodes = _mesh_1D->physicalCellNodesForCell(physicalCellID);
  auto xCoordsEntry = _physicalCellIDToRefSpaceXCoordInAncestor.find(physicalCellID);
  if (xCoordsEntry != _physicalCellIDToRefSpaceXCoordInAncestor.end())
  {
    xCoordsInAncestors = xCoordsEntry->second;
  }
  if (xCoordsInAncestors.size() == 0)
  {
    basisCache->setPhysicalCellNodes(physicalCellNodes, {physicalCellID}, true);
    const int pointStart = 0;
    const int pointEnd   = basisCache->getRefCellPointsFC().dimension(0);
    return {basisCache, std::vector<int>{pointStart,pointEnd} };
  }
  
  const int numAncestors = xCoordsInAncestors.size();
  std::vector< Camellia::FieldContainer<double> > refSpaceCoordsForLevel(numAncestors+1);
  refSpaceCoordsForLevel[0] = refPointsStandard;
  int totalPointCount = refPointsStandard.dimension(0);
  
  double width = 2.0;
  for (int ancestorLevel=0; ancestorLevel<numAncestors; ancestorLevel++)
  {
    width /= 2.0;
    int firstMatchingPointOrdinal = -1;
    int lastMatchingPointOrdinal = -2;
    const double & xCoordLeft = xCoordsInAncestors[ancestorLevel];
    
    auto pointMatches = [&](double pointValue) {
      // on the left, we match on equals; on the right, we do not match on equals:
      const bool matchesLeft  = pointValue >= xCoordLeft;
      const bool matchesRight = pointValue < xCoordLeft + width;
      return matchesLeft && matchesRight;
    };
    for (int pointOrdinal=0; pointOrdinal<refPointsStandard.dimension(0); pointOrdinal++)
    {
      // here, we assume that points are ordered, but do not assume whether ordered left to right or right to left
      // (it happens that Intrepid orders them right to left; I'm not sure which Intrepid2 does)
      if ( pointMatches(refPointsStandard(pointOrdinal,0)) )
      {
        if (firstMatchingPointOrdinal == -1)
        {
          firstMatchingPointOrdinal = pointOrdinal;
        }
        lastMatchingPointOrdinal  = pointOrdinal;
      }
    }
    const int pointCountForLevel = lastMatchingPointOrdinal-firstMatchingPointOrdinal+1;
    // now we know which refSpacePoints in ancestor match; need to map these to this cell
    // in ancestor, these are in [xCoordLeft,xCoordLeft+width]
    // in this cell, they are in [-1,1]
    // formula for point in this cell from x in ancestor is (xAncestor-xCoordLeft) * 2/width - 1
    Camellia::FieldContainer<double> refPointsFromAncestor(pointCountForLevel,1);
    for (int pointOrdinal=0; pointOrdinal < pointCountForLevel; pointOrdinal++)
    {
      double xAncestor = refPointsStandard(pointOrdinal + firstMatchingPointOrdinal,0);
      double x = (xAncestor - xCoordLeft) * 2.0 / width - 1.0;
      {
        if ((x < -1.0) || (x>1.0))
        {
          std::cout << "x (" << x << ") is out of bounds!\n";
        }
      }
      refPointsFromAncestor(pointOrdinal,0) = x;
    }
    refSpaceCoordsForLevel[ancestorLevel+1] = refPointsFromAncestor;
    totalPointCount += pointCountForLevel;
  }
  Camellia::FieldContainer<double> allRefPoints(totalPointCount,1);
  int pointOffset = 0;
  std::vector<int> pointOffsets(numAncestors+2);
  pointOffsets[0] = 0;
  for (int level=0; level<refSpaceCoordsForLevel.size(); level++)
  {
    const auto & levelCoords = refSpaceCoordsForLevel[level];
    const int pointCountForLevel = levelCoords.dimension(0);
    for (int pointOrdinal=0; pointOrdinal<pointCountForLevel; pointOrdinal++)
    {
      allRefPoints(pointOffset++,0) = levelCoords(pointOrdinal,0);
    }
    pointOffsets[level+1] = pointOffset;
  }
  basisCache->setRefCellPointsFC(allRefPoints);
  basisCache->setPhysicalCellNodes(physicalCellNodes, {physicalCellID}, true);
  return {basisCache,pointOffsets};
}
  
void PhasePhysicalMeshMap1D1V::didHRefine(MeshTopologyPtr phaseSpaceMesh, const set<GlobalIndexType> &phaseSpaceCellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool willRepartitionAndRebuild)
{
  // We need to maintain the rule that the 1D mesh is nowhere coarser than the 2D mesh
  // Any phase space cell that is being refined is right now strictly finer than the 1D mesh if and only if the length of
  // the corresponding vector in _phaseSpaceCellIDToPhysicalCellInfo is greater than 1 (i.e. there is more than one
  // physical cell corresponding to the phase space cell).
  std::set<GlobalIndexType> physicalCellsToRefineSet;
  auto meshTopo1D = _mesh_1D->getTopology();
  for (auto phaseSpaceCellID : phaseSpaceCellIDs)
  {
    if (_phaseSpaceCellIDToPhysicalCellInfo.find(phaseSpaceCellID) != _phaseSpaceCellIDToPhysicalCellInfo.end())
    {
      const int numPhysicalCells = _phaseSpaceCellIDToPhysicalCellInfo[phaseSpaceCellID].size();
      if (numPhysicalCells == 1)
      {
        const GlobalIndexType physicalCellID = _phaseSpaceCellIDToPhysicalCellInfo[phaseSpaceCellID][0].physicalCellID;
        
        // add this cell to refinement list if it has not already been refined by e.g. another PhasePhysicalMeshMap1D1V instance
        if (! meshTopo1D->getCell(physicalCellID)->isParent(meshTopo1D))
        {
          physicalCellsToRefineSet.insert(physicalCellID);
        }
      }
    }
  }
  std::vector<GlobalIndexType> allPhysicalCellIDsToRefine;
  std::vector<GlobalIndexType> myPhysicalCellIDsToRefine(physicalCellsToRefineSet.begin(),physicalCellsToRefineSet.end());
  std::vector<int> offsets;
  
  MPIWrapper::allGatherVariable<GlobalIndexType>(*_mesh_2D->Comm(), allPhysicalCellIDsToRefine, myPhysicalCellIDsToRefine, offsets);
  
  // allPhysicalCellIDsToRefine may have duplicate entries; use set to deduplicate these:
  physicalCellsToRefineSet.insert(allPhysicalCellIDsToRefine.begin(),allPhysicalCellIDsToRefine.end());
  
  _mesh_1D->hRefine(physicalCellsToRefineSet, RefinementPattern::regularRefinementPatternLine());
  
  // if we're going to repartition anyway, we'll rebuild the map then (see didRepartition() below), and we don't need it up to date between now and then
  if (!willRepartitionAndRebuild)
  {
    rebuildMap();
  }
}

void PhasePhysicalMeshMap1D1V::didHUnrefine(MeshTopologyPtr phaseSpaceMesh, const std::set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild)
{
  // since the 1D mesh is pretty cheap, we don't unrefine it here even if we could.
  // if we're going to repartition anyway, we'll rebuild the map then (see didRepartition() below), and we don't need it up to date between now and then
  if (!didRepartitionAndRebuild)
  {
    rebuildMap();
  }
}

void PhasePhysicalMeshMap1D1V::didRepartition(MeshTopologyPtr phaseSpaceMesh)
{
  rebuildMap();
}
