//
//  PhasePhysicalMeshMap.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 5/13/20.
//

#include "PhasePhysicalMeshMap.h"

#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"

using namespace Camellia;

PhasePhysicalMeshMap::PhasePhysicalMeshMap( MeshPtr physicalMesh, MeshPtr phaseSpaceMesh )
:
_physicalMesh(physicalMesh),
_phaseMesh(phaseSpaceMesh)
{
  _extrudedMeshTopology = dynamic_cast<ExtrudedMeshTopology*>(phaseSpaceMesh->getTopology().get());
  TEUCHOS_TEST_FOR_EXCEPTION(_extrudedMeshTopology == NULL, std::invalid_argument, "Phase-space MeshTopology must be an ExtrudedMeshTopology");
  
  Teuchos::RCP<RefinementObserver> thisWeakPtr = Teuchos::rcp(this, false);
  _phaseMesh->registerObserver(thisWeakPtr);
  this->rebuildMap();
  
  _physicalMesh->globalDofAssignment()->setAllowMeshTopologyPruning(false); // want to keep all physical geometry info available to all MPI ranks
}

PhasePhysicalMeshMap::~PhasePhysicalMeshMap()
{
  _phaseMesh->unregisterObserver(this);
}

PhasePhysicalPointIterator PhasePhysicalMeshMap::getPhysicalPointIterator(BasisCachePtr phaseSpaceBasisCache, GlobalIndexType phaseSpaceCellID,
                                                                          map< GlobalIndexType , pair<BasisCachePtr, vector<int> > > &physicalBasisCachePairs) const
{
  const auto &physicalCellInfoVector = this->physicalCellInfo(phaseSpaceCellID);
  
  const int numVelocityDimensions = _phaseMesh->getDimension() - _physicalMesh->getDimension();
  
  bool isSideCache = (phaseSpaceBasisCache->getSideIndex() != -1);
  int physicalSpaceSideOrdinal;
  if (!isSideCache)
  {
    physicalSpaceSideOrdinal = -1;
  }
  else // isSideCache == true
  {
    const int phaseSpaceSideOrdinal = phaseSpaceBasisCache->getSideIndex();
    // what follows assumes 1DnV with extruded line elements, 1 ≤ n ≤ 3; let's throw an exception if we're not on such an element
    auto phaseCellTopo    = phaseSpaceBasisCache->cellTopology();
    auto physicalCellTopo = CellTopology::cellTopology(phaseCellTopo->getShardsTopology(), phaseCellTopo->getTensorialDegree() - numVelocityDimensions);
    
    if (phaseCellTopo->getTensorialDegree() + _physicalMesh->getDimension() != _phaseMesh->getDimension())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "phaseCellTopo must be an orthogonal extrusion of the physical mesh");
    }
    
    /*
       subcell ordering in tensor cell topology is as follows.
       
       We take the one-lower-dimensional component topology T and make two copies, T0 and T1.
       
       The subcells of dimension d in the tensor topology are ordered as follows:
       - all subcells of dimension d from T0
       - all subcells of dimension d from T1
       - all (subcells of dimension (d-1) from T) x Line.
       
       Since our physical 1D cell is the first tensorial component, the phase space sides that correspond to physical sides are exactly
       the last two phase space sides.  (For higher physical dimensions where the physical cell has N sides, the phase
       space sides of interest will be the last N sides.)
       */
      
    const int numPhysicalSides = physicalCellTopo->getSideCount();
    
    if (phaseSpaceSideOrdinal < phaseCellTopo->getSideCount() - numPhysicalSides)
    {
      physicalSpaceSideOrdinal = -1;
    }
    else
    {
      physicalSpaceSideOrdinal = phaseSpaceSideOrdinal - (phaseCellTopo->getSideCount() - numPhysicalSides);
    }
  }
  const bool inPhysicalVolume = (physicalSpaceSideOrdinal == -1);
  
  // reversePointOrder is always false for a proper tensor-product mesh.  (1D1V was initially implemented with a 2D mesh, with shards quadrilaterals, which don't have a tensor-product structure.)
  const bool reversePointOrder = false;

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

int PhasePhysicalMeshMap::maximumUnrefinementLevel() const
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

MeshPtr PhasePhysicalMeshMap::physicalMesh() const
{
  return _physicalMesh;
}

MeshPtr PhasePhysicalMeshMap::phaseMesh() const
{
  return _phaseMesh;
}

const std::vector<PhysicalCellRelativeRefinementInfo> & PhasePhysicalMeshMap::physicalCellInfo(const GlobalIndexType &phaseSpaceCellID) const
{
  auto entry = _phaseSpaceCellIDToPhysicalCellInfo.find(phaseSpaceCellID);
  if (entry == _phaseSpaceCellIDToPhysicalCellInfo.end())
  {
    int myPID = _phaseMesh->Comm()->MyPID();
    std::cout << "phaseSpaceCellID " << phaseSpaceCellID << " not found on rank " << myPID << std::endl;
    std::cout << "valid phaseSpaceCellIDs on rank " << myPID << ": ";
    const auto & myPhaseSpaceCells = _phaseMesh->cellIDsInPartition();
    for (auto & myPhaseSpaceCell : myPhaseSpaceCells)
    {
      std::cout << myPhaseSpaceCell << " ";
    }
    std::cout << std::endl << std::flush;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(entry == _phaseSpaceCellIDToPhysicalCellInfo.end(), std::invalid_argument, "phaseSpaceCellID not found");
  return entry->second;
}

std::set<GlobalIndexType> PhasePhysicalMeshMap::physicalCellIDsForPhaseSpaceCells(const std::set<GlobalIndexType> &phaseSpaceCellIDs) const
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

std::vector<GlobalIndexType> PhasePhysicalMeshMap::physicalCellIDsForPhaseSpaceCells(const std::vector<GlobalIndexType> &phaseSpaceCellIDs) const
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

void PhasePhysicalMeshMap::rebuildMap()
{
  auto & myPhaseCells = _phaseMesh->cellIDsInPartition();
  _phaseSpaceCellIDToPhysicalCellInfo.clear();
    
  auto phaseMeshTopo = _phaseMesh->getTopology();
  auto meshTopo_1D = _physicalMesh->getTopology();
  
  for (auto phaseCellID : myPhaseCells)
  {
    auto physicalCellID = _extrudedMeshTopology->physicalCellIDForPhaseSpaceCellID(phaseCellID);
    CellPtr physicalCell = meshTopo_1D->getCell(physicalCellID);
    
    const int leftVertexOrdinal  = 0; // in 1D element, left vertex is the first one
    const int rightVertexOrdinal = 1; // in 1D, right vertex is the second one
    
    const bool leafNodesOnly = true;
    auto  leftDescendants = physicalCell->getDescendantsForSide( leftVertexOrdinal, meshTopo_1D, leafNodesOnly);
    auto rightDescendants = physicalCell->getDescendantsForSide(rightVertexOrdinal, meshTopo_1D, leafNodesOnly);
    
    const auto &  leftDescendantCellID =  leftDescendants[0].first;
    const auto & rightDescendantCellID = rightDescendants[0].first;
    
    CellPtr leftDescendant = meshTopo_1D->getCell(leftDescendantCellID);
    
    const int physicalCellLevel = physicalCell->level();
    const int leftDescendantLevel = leftDescendant->level();
    int numUnrefinementLevels = leftDescendantLevel - physicalCellLevel;
    
    // now, march along physicalCells from left until we get to right
    double refSpaceXCoordThusFar = -1.0;
    std::vector<PhysicalCellRelativeRefinementInfo> physicalCellInfoVector;
    PhysicalCellRelativeRefinementInfo physicalCellInfo;
    physicalCellInfo.physicalCellID = leftDescendantCellID; // this is the one that shares the left vertex with the physical ancestor corresponding to the phase space cell
    physicalCellInfo.uniformDescendantNumber = 0;
    physicalCellInfo.unrefinementLevel = numUnrefinementLevels;
    physicalCellInfoVector.push_back(physicalCellInfo);
    
    if (numUnrefinementLevels > 0)
    {
      if (_physicalCellIDToRefSpaceXCoordInAncestor[leftDescendantCellID].size() < numUnrefinementLevels)
      {
        _physicalCellIDToRefSpaceXCoordInAncestor[leftDescendantCellID].resize(numUnrefinementLevels);
      }
      _physicalCellIDToRefSpaceXCoordInAncestor[leftDescendantCellID][numUnrefinementLevels-1] = refSpaceXCoordThusFar;
    }
    
    GlobalIndexType rightmostPhysicalCellID = leftDescendantCellID;
    while (rightmostPhysicalCellID != rightDescendantCellID)
    {
      CellPtr rightmostPhysicalCell = meshTopo_1D->getCell(rightmostPhysicalCellID);
      const double previousCellWidthInAncestor = 2.0 / (1 << numUnrefinementLevels);
      refSpaceXCoordThusFar += previousCellWidthInAncestor;
      const int previousUnrefinementLevel = physicalCellInfo.unrefinementLevel;
      const int descendantNumberInPreviousUnrefinementLevel = physicalCellInfo.uniformDescendantNumber + 1;
      auto neighbor = rightmostPhysicalCell->getNeighbor(rightVertexOrdinal, meshTopo_1D);
      physicalCellInfo.physicalCellID = neighbor->cellIndex();
      auto ancestor = neighbor;
      numUnrefinementLevels = neighbor->level() - physicalCellLevel;
      
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
    _phaseSpaceCellIDToPhysicalCellInfo[phaseCellID] = physicalCellInfoVector;
  }
}

std::set< std::pair<int,int> > PhasePhysicalMeshMap::referenceSpaceUnrefinementLevelDescendantNumberPairs() const
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
std::pair<BasisCachePtr, std::vector<int> > PhasePhysicalMeshMap::physicalCellBasisCache(const GlobalIndexType physicalCellID, bool testVsTest, int quadratureEnrichment) const
{
  auto elemType = _physicalMesh->getElementType(physicalCellID);
  BasisCachePtr basisCache =  BasisCache::basisCacheForCellType(_physicalMesh, elemType, testVsTest, quadratureEnrichment);
  auto refPointsStandard = basisCache->getRefCellPointsFC();
  std::vector<double> xCoordsInAncestors;
  auto physicalCellNodes = _physicalMesh->physicalCellNodesForCell(physicalCellID);
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
  
void PhasePhysicalMeshMap::didHRefine(MeshTopologyPtr phaseSpaceMesh, const set<GlobalIndexType> &phaseSpaceCellIDs, Teuchos::RCP<RefinementPattern> refPattern, bool willRepartitionAndRebuild)
{
  // We need to maintain the rule that the 1D mesh is nowhere coarser than the 2D mesh
  // Any phase space cell that is being refined is right now strictly finer than the 1D mesh if and only if the length of
  // the corresponding vector in _phaseSpaceCellIDToPhysicalCellInfo is greater than 1 (i.e. there is more than one
  // physical cell corresponding to the phase space cell).
  std::set<GlobalIndexType> physicalCellsToRefineSet;
  auto meshTopo1D = _physicalMesh->getTopology();
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
  
  MPIWrapper::allGatherVariable<GlobalIndexType>(*_phaseMesh->Comm(), allPhysicalCellIDsToRefine, myPhysicalCellIDsToRefine, offsets);
  
  // allPhysicalCellIDsToRefine may have duplicate entries; use set to deduplicate these:
  physicalCellsToRefineSet.insert(allPhysicalCellIDsToRefine.begin(),allPhysicalCellIDsToRefine.end());
  
  _physicalMesh->hRefine(physicalCellsToRefineSet, RefinementPattern::regularRefinementPatternLine());
  
  // if we're going to repartition anyway, we'll rebuild the map then (see didRepartition() below), and we don't need it up to date between now and then
  if (!willRepartitionAndRebuild)
  {
    rebuildMap();
  }
}

void PhasePhysicalMeshMap::didHUnrefine(MeshTopologyPtr phaseSpaceMesh, const std::set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild)
{
  // since the 1D mesh is pretty cheap, we don't unrefine it here even if we could.
  // if we're going to repartition anyway, we'll rebuild the map then (see didRepartition() below), and we don't need it up to date between now and then
  if (!didRepartitionAndRebuild)
  {
    rebuildMap();
  }
}

void PhasePhysicalMeshMap::didRepartition(MeshTopologyPtr phaseSpaceMesh)
{
  rebuildMap();
}
