//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "NeighborFunction.h"

#include "BasisCache.h"
#include "CamelliaCellTools.h"
#include "GlobalDofAssignment.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
NeighborFunction<Scalar>::NeighborFunction(TFunctionPtr<Scalar> fxn, MeshPtr mesh)
: TFunction<Scalar>("NeighborFunction", fxn->rank()),
  _fxn(fxn),
  _mesh(mesh)
{}

template <typename Scalar>
bool NeighborFunction<Scalar>::boundaryValueOnly()
{
  return true;
}

template <typename Scalar>
string NeighborFunction<Scalar>::displayString()
{
  ostringstream str;
  str << "neighbor(" << _fxn->displayString() << ") ";
  return str.str();
}

template <typename Scalar>
size_t NeighborFunction<Scalar>::getCellDataSize(GlobalIndexType cellID)
{
  return _fxn->getCellDataSize(cellID);
}

template <typename Scalar>
void NeighborFunction<Scalar>::packCellData(GlobalIndexType cellID, char* cellData, size_t bufferLength)
{
  _fxn->packCellData(cellID, cellData, bufferLength);
}

template <typename Scalar>
size_t NeighborFunction<Scalar>::unpackCellData(GlobalIndexType cellID, const char* cellData, size_t bufferLength)
{
  return _fxn->unpackCellData(cellID, cellData, bufferLength);
}

template <typename Scalar>
void NeighborFunction<Scalar>::importCellData(std::vector<GlobalIndexType> cells)
{
  set<GlobalIndexType> cellSet(cells.begin(),cells.end());
  
  auto meshTopo = _mesh->getTopology();
  
  set<GlobalIndexType> neighbors;
  for (auto cellID : cellSet)
  {
    CellPtr cell = _mesh->getTopology()->getCell(cellID);
    int sideCount = cell->getSideCount();
    
    for (int sideOrdinal = 0; sideOrdinal < sideCount; sideOrdinal++)
    {
      pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
      GlobalIndexType neighborCellID = neighborInfo.first;
      if (neighborCellID != -1)
      {
        auto neighbor = meshTopo->getCell(neighborCellID);
        if (! neighbor->isParent(meshTopo))
        {
          // off-rank, active neighbor for which we're responsible: ask for import
          neighbors.insert(neighborCellID);
        }
        else
        {
          // get descendants on this side:
          int sideOrdinalInNeighbor = neighborInfo.second;
          auto childSidePairs = neighbor->getDescendantsForSide(sideOrdinalInNeighbor, meshTopo);
          for (auto childSidePair : childSidePairs)
          {
            // first in pair is child cell ID:
            auto childID = childSidePair.first;
            // this is an active neighbor: add it.
            neighbors.insert(childID);
          }
        }
      }
    }
  }
  cellSet.insert(neighbors.begin(),neighbors.end());
  
  std::vector<GlobalIndexType> cellsAndNeighbors(cellSet.begin(),cellSet.end());

  _fxn->importCellData(cellsAndNeighbors);
}

template <typename Scalar>
void NeighborFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(! basisCache->isSideCache(), std::invalid_argument, "NeighborFunction is only defined on mesh skeleton.");
  
  /*
   OUTLINE:
   Loop over each cell:
   1. Get reference-space side coordinates in neighbor cell's reference-space frame.
   2. Set up a neighbor basis cache (cached for reuse, preferably) with these side coordinates.
   3. Compute these; store in values.
   
   Note that particularly in the case of not being cached for reuse in Step 2, this is not a very efficient approach.
   The difficulty is that creating a new BasisCache forces us to recompute basis values, which may already have been
   computed previously.  The inefficiency is worst in the case of a uniform mesh (no hanging nodes), where the transformation
   of coordinates is just a permutation, and all basis values could in principle be reused.
   
   Possibly one approach for reducing the inefficiencies here (as well as reducing some code redundancies) would be to develop
   some geometric mapping facilities for moving from a cell to its neighbor.  We use some related facilities below, mapping from
   cell to ancestor, etc., but the parts that involve using the BasisCache for the purpose of e.g. permuting points could be
   replaced with more targeted facilities.  Such facilities could store maps for reuse, or even hard-code certain permutations.
   
   There is similar code elsewhere, including in Function::squaredL2NormOfJumps(), and Solution::applyDGJumpTerms().  If we can
   factor out that logic into a common facility, those implementations could be simplified.
   */
  
  // by default, populate values with _fxn on this cell -- for boundary cells, this is exactly what we want
  // we'll overwrite the rest…  (This is another example of inefficient computation that could be addressed when optimizing)
  _fxn->values(values, basisCache);
  auto &sideRefPoints = basisCache->getRefCellPointsFC();
  
  // allPhysicalPoints is used cell-wise below for a sanity check
  auto &allPhysicalPoints = basisCache->getPhysicalCubaturePointsFC();
  
  int numPoints = sideRefPoints.dimension(0);
  int spaceDim  = _mesh->getDimension();
  int sideDim   = spaceDim - 1;
  
  auto meshTopo = _mesh->getTopology();
  
  // placeholders for geometric computations below:
  Camellia::FieldContainer<double> emptyRefPointsVolume(0,spaceDim); // (P,D)
  Camellia::FieldContainer<double> emptyRefPointsSide(0,sideDim);    // (P,D)
  Camellia::FieldContainer<double> emptyQuadratureWeights(1,0);      // (C,P)
  const int dummyQuadratureOrder = 0;
  
  const int oneCell = 1;
  FieldContainer<double> valuesOneCell(oneCell,numPoints);
  if (_fxn->rank() > 0)
  {
    Teuchos::Array<int> dims(_fxn->rank() + 2);
    const int oneCell = 1;
    dims[0] = oneCell;
    dims[1] = numPoints;
    for (int i=2; i < dims.size(); i++)
    {
      dims[i] = spaceDim;
    }
    valuesOneCell = FieldContainer<double>(dims);
  }
  const int numValuesPerPoint = valuesOneCell.size() / numPoints;
  
  auto & cellIDs  = basisCache->cellIDs();
  int sideOrdinal = basisCache->getSideIndex();
  auto cellTopo   = basisCache->getVolumeBasisCache()->cellTopology();
  auto sideTopo   = cellTopo->getSide(sideOrdinal);
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForVolumeTopo;         // used for "my" cells
  map<CellTopologyKey,BasisCachePtr> basisCacheForNeighborVolumeTopo; // used for neighbor cells
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnVolumeTopo;
  map<pair<int,CellTopologyKey>,BasisCachePtr> basisCacheForSideOnNeighborVolumeTopo; // these can have permuted cubature points (i.e. we need to set them every time, so we can't share with basisCacheForSideOnVolumeTopo, which tries to avoid this)
  
  map<CellTopologyKey,BasisCachePtr> basisCacheForReferenceCellTopo;
  
  int cellOrdinal = -1; // cell's ordinal in the cellIDs container
  for (auto cellID : cellIDs)
  {
    cellOrdinal++;
    auto cell = meshTopo->getCell(cellID);
    auto mySideRefPoints = basisCache->getRefCellPointsFC();
    
    pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, meshTopo);
    GlobalIndexType neighborCellID = neighborInfo.first;
    auto mySideIndexInNeighbor = neighborInfo.second;
    
    if (neighborCellID == -1)
    {
      // boundary cell, no neighbor -- NeighborFunction is defined as _fxn evaluated locally
      // this is already filled in above
    }
    else
    {
      auto neighbor = meshTopo->getCell(neighborCellID);
      auto neighborTopo = neighbor->topology();
      pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(mySideIndexInNeighbor, meshTopo);
      bool neighborIsPeer = neighborNeighborInfo.first == cell->cellIndex();
      
      unsigned permutationFromMeToNeighbor;
      
      if (!neighborIsPeer) // then we have some refinements relative to neighbor
      {
        /*******   Map my ref points to my ancestor ******/
        pair<GlobalIndexType,unsigned> ancestorInfo = neighbor->getNeighborInfo(mySideIndexInNeighbor, meshTopo);
        GlobalIndexType ancestorCellIndex = ancestorInfo.first;
        unsigned ancestorSideOrdinal = ancestorInfo.second;
        
        RefinementBranch refinementBranch = cell->refinementBranchForSide(sideOrdinal, meshTopo);
        RefinementBranch sideRefinementBranch = RefinementPattern::sideRefinementBranch(refinementBranch, ancestorSideOrdinal);
        Camellia::FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(sideRefinementBranch);
        
        cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
        BasisCachePtr ancestralBasisCache = Teuchos::rcp(new BasisCache(cellNodes,sideTopo,dummyQuadratureOrder,false)); // false: don't create side cache too
        
        ancestralBasisCache->setRefCellPointsFC(mySideRefPoints, emptyQuadratureWeights, dummyQuadratureOrder, true);
        
        // now, the "physical" points in ancestral cache are the ones we want
        mySideRefPoints = ancestralBasisCache->getPhysicalCubaturePointsFC();
        mySideRefPoints.resize(numPoints,sideDim); // strip cell dimension
        
        /*******  Determine ancestor's permutation of the side relative to neighbor ******/
        CellPtr ancestor = meshTopo->getCell(ancestorCellIndex);
        vector<IndexType> ancestorSideNodes, neighborSideNodes; // this will list the indices as seen by MeshTopology
        
        // the following permutations map from canonical ordering to that seen by the cell
        // we want to construct a mapping that takes us from ancestor's ordering to neighbor's
        unsigned ancestorSidePermutation = ancestor->subcellPermutation(sideDim, sideOrdinal);
        unsigned neighborSidePermutation = neighbor->subcellPermutation(sideDim, mySideIndexInNeighbor);
        
        // invert ancestor permutation; this takes us from my ordering to canonical
        unsigned ancestorInverse = CamelliaCellTools::permutationInverse(sideTopo, ancestorSidePermutation);
        // compose this with neighbor permutation; this takes us from canonical to neighbor
        permutationFromMeToNeighbor = CamelliaCellTools::permutationComposition(sideTopo, ancestorInverse, neighborSidePermutation);
      }
      else
      {
        // neighbor is peer, but might not be active (i.e., might have descendants)
        
        // the following permutations map from canonical ordering to that seen by the cell
        // we want to construct a mapping that takes us from my ordering to neighbor's
        unsigned mySidePermutation       = cell->subcellPermutation(sideDim, sideOrdinal);
        unsigned neighborSidePermutation = neighbor->subcellPermutation(sideDim, mySideIndexInNeighbor);
        
        // invert my permutation; this takes us from my ordering to canonical
        unsigned myInverse = CamelliaCellTools::permutationInverse(sideTopo, mySidePermutation);
        // compose this with neighbor permutation; this takes us from canonical to neighbor
        permutationFromMeToNeighbor = CamelliaCellTools::permutationComposition(sideTopo, myInverse, neighborSidePermutation);
      }
      
      CellTopoPtr sideTopo = neighborTopo->getSide(mySideIndexInNeighbor); // for non-peers, this is my ancestor's cell topo
      int nodeCount = sideTopo->getNodeCount();
      
      Camellia::FieldContainer<double> permutedRefNodes(nodeCount,sideDim);
      CamelliaCellTools::refCellNodesForTopology(permutedRefNodes, sideTopo, permutationFromMeToNeighbor);
      permutedRefNodes.resize(1,nodeCount,sideDim); // add cell dimension to make this a "physical" node container
      if (basisCacheForReferenceCellTopo.find(sideTopo->getKey()) == basisCacheForReferenceCellTopo.end())
      {
        basisCacheForReferenceCellTopo[sideTopo->getKey()] = BasisCache::basisCacheForReferenceCell(sideTopo, -1);
      }
      BasisCachePtr referenceBasisCache = basisCacheForReferenceCellTopo[sideTopo->getKey()];
      referenceBasisCache->setRefCellPointsFC(mySideRefPoints,emptyQuadratureWeights,dummyQuadratureOrder,false);
      std::vector<GlobalIndexType> cellIDs = {0}; // unused
      referenceBasisCache->setPhysicalCellNodes(permutedRefNodes, cellIDs, false);
      // now, the "physical" points are the ones we should use as ref points for the neighbor
      Camellia::FieldContainer<double> neighborRefCellPoints = referenceBasisCache->getPhysicalCubaturePointsFC();
      neighborRefCellPoints.resize(numPoints,sideDim); // strip cell dimension to convert to a "reference" point container
      
      Camellia::FieldContainer<double> neighborCellNodes = _mesh->physicalCellNodesForCell(neighborCellID);
      if (basisCacheForNeighborVolumeTopo.find(neighborTopo->getKey()) == basisCacheForNeighborVolumeTopo.end())
      {
        basisCacheForNeighborVolumeTopo[neighborTopo->getKey()] = Teuchos::rcp( new BasisCache(neighborCellNodes, neighborTopo,
                                                                                               emptyRefPointsVolume, emptyQuadratureWeights) );
      }
      BasisCachePtr neighborVolumeCache = basisCacheForNeighborVolumeTopo[neighborTopo->getKey()];
      neighborVolumeCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
      
      pair<int,CellTopologyKey> neighborSideCacheKey{mySideIndexInNeighbor,neighborTopo->getKey()};
      if (basisCacheForSideOnNeighborVolumeTopo.find(neighborSideCacheKey) == basisCacheForSideOnNeighborVolumeTopo.end())
      {
        basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey]
        = Teuchos::rcp( new BasisCache(mySideIndexInNeighbor, neighborVolumeCache, emptyRefPointsSide, emptyQuadratureWeights, -1));
      }
      BasisCachePtr neighborSideCache = basisCacheForSideOnNeighborVolumeTopo[neighborSideCacheKey];
      neighborSideCache->setMesh(_mesh);
      neighborSideCache->setRefCellPointsFC(neighborRefCellPoints, emptyQuadratureWeights, dummyQuadratureOrder, false);
      neighborSideCache->setPhysicalCellNodes(neighborCellNodes, {neighborCellID}, false);
      
      // if there aren't periodic BCs, perform the sanity check.  If there are periodic BCs, we expect to fail the check...
      if (meshTopo->getPeriodicBCs().size() == 0)
      {
        // Sanity check that the physical points agree:
        double tol = 1e-8;
        
        Camellia::FieldContainer<double> neighborPhysicalPoints = neighborSideCache->getPhysicalCubaturePointsFC();
        Camellia::FieldContainer<double> myPhysicalPoints(numPoints,spaceDim);
        // once we move to Kokkos, "myPhysicalPoints" here could be defined as a subview of allPhysicalPoints
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          for (int d=0; d<spaceDim; d++)
          {
            myPhysicalPoints(pointOrdinal,d) = allPhysicalPoints(cellOrdinal,pointOrdinal,d);
          }
        }
        
        bool pointsMatch = (myPhysicalPoints.size() == neighborPhysicalPoints.size()); // true unless we find a point that doesn't match
        double maxDiff = 0.0;
        if (pointsMatch)
        {
          for (int i=0; i<myPhysicalPoints.size(); i++)
          {
            double diff = abs(myPhysicalPoints[i]-neighborPhysicalPoints[i]);
            if (diff > tol)
            {
              pointsMatch = false;
              maxDiff = std::max(diff,maxDiff);
            }
          }
        }
        
        if (!pointsMatch)
        {
          cout << "WARNING: pointsMatch is false; maxDiff = " << maxDiff << "\n";
//        cout << "myPhysicalPoints:\n" << myPhysicalPoints;
//        cout << "neighborPhysicalPoints:\n" << neighborPhysicalPoints;
        }
      }
      if (!neighbor->isParent(meshTopo))
      {
        _fxn->values(valuesOneCell, neighborSideCache);
      }
      else
      {
        // if neighbor is parent, we need to divide up the points among neighbor's children
        // again, this is not particularly efficient and could be refactored for efficiency
        std::vector<int> numRefSpaceChildrenForPoint(numPoints,0);
        
        auto sideRefPattern = neighbor->refinementPattern()->sideRefinementPatterns()[mySideIndexInNeighbor];
        std::vector<double> parentRefPointSide(sideDim);
        
        int numChildrenInSide = sideRefPattern->numChildren();
        
        vector< vector<int> > parentPointOrdinalsForChild(numChildrenInSide);
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          for (int d=0; d<sideDim; d++)
          {
            parentRefPointSide[d] = neighborRefCellPoints(pointOrdinal,d);
          }
          for (int childID=0; childID<numChildrenInSide; childID++)
          {
            auto childCellID = childID + 1; // 0 is the parent in refinement mesh topology
            int quadratureDegree = 1;
            if (sideRefPattern->refinementMeshTopology()->cellContainsPoint(childCellID, parentRefPointSide, quadratureDegree))
            {
              numRefSpaceChildrenForPoint[pointOrdinal]++;
              parentPointOrdinalsForChild[childID].push_back(pointOrdinal);
            }
          }
        }
      
        auto neighborRefPattern = neighbor->refinementPattern();
        for (int childOrdinalInSide=0; childOrdinalInSide<numChildrenInSide; childOrdinalInSide++)
        {
          auto & parentPointOrdinals = parentPointOrdinalsForChild[childOrdinalInSide];
          int numPoints = parentPointOrdinals.size();
          if (numPoints==0) continue; // no points for this child
          Camellia::FieldContainer<double> parentRefCellPoints(oneCell,numPoints,sideDim);
          Camellia::FieldContainer<double> childRefCellPoints(oneCell,numPoints,sideDim);
          
          for (int childPointOrdinal=0; childPointOrdinal<numPoints; childPointOrdinal++)
          {
            int parentPointOrdinal = parentPointOrdinals[childPointOrdinal];
            for (int d=0; d<sideDim; d++)
            {
              parentRefCellPoints(0,childPointOrdinal,d) = neighborRefCellPoints(parentPointOrdinal,d);
            }
          }
          
          const int childIDInRefinementMeshTopology = childOrdinalInSide + 1;
          sideRefPattern->mapPointsToChildRefCoordinates(parentRefCellPoints, childIDInRefinementMeshTopology, childRefCellPoints);
          childRefCellPoints.resize(numPoints,sideDim); // strip cell dimension
          
          unsigned childOrdinalVolume = neighborRefPattern->mapSideChildIndex(mySideIndexInNeighbor, childOrdinalInSide);
          unsigned childSideOrdinal = neighborRefPattern->mapSubcellFromParentToChild(childOrdinalVolume, sideDim, mySideIndexInNeighbor).second;
          
          auto child     = neighbor->children()[childOrdinalVolume];
          auto childTopo = child->topology();
          GlobalIndexType childCellID = child->cellIndex();
          
          Camellia::FieldContainer<double> childCellNodes = _mesh->physicalCellNodesForCell(childCellID);
          if (basisCacheForNeighborVolumeTopo.find(childTopo->getKey()) == basisCacheForNeighborVolumeTopo.end())
          {
            basisCacheForNeighborVolumeTopo[childTopo->getKey()] = Teuchos::rcp( new BasisCache(childCellNodes, childTopo,
                                                                                                emptyRefPointsVolume, emptyQuadratureWeights) );
          }
          BasisCachePtr childVolumeCache = basisCacheForNeighborVolumeTopo[childTopo->getKey()];
          childVolumeCache->setPhysicalCellNodes(childCellNodes, {childCellID}, false);
          
          pair<int,CellTopologyKey> childSideCacheKey{childSideOrdinal,childTopo->getKey()};
          if (basisCacheForSideOnNeighborVolumeTopo.find(childSideCacheKey) == basisCacheForSideOnNeighborVolumeTopo.end())
          {
            basisCacheForSideOnNeighborVolumeTopo[childSideCacheKey]
            = Teuchos::rcp( new BasisCache(childSideOrdinal, childVolumeCache, emptyRefPointsSide, emptyQuadratureWeights, -1));
          }
          BasisCachePtr childSideCache = basisCacheForSideOnNeighborVolumeTopo[childSideCacheKey];
          childSideCache->setMesh(_mesh);
          childSideCache->setRefCellPointsFC(childRefCellPoints, emptyQuadratureWeights, dummyQuadratureOrder, false);
          childSideCache->setPhysicalCellNodes(childCellNodes, {childCellID}, false);
          
          Teuchos::Array<int> childValueDims; // (C,P,...)
          valuesOneCell.dimensions(childValueDims); // valuesOneCell has the right dimensions except for the points dimension
          childValueDims[1] = numPoints;
          FieldContainer<double> childValues(childValueDims);
          
          _fxn->values(childValues, childSideCache);
          
          // now, sum into valuesOneCell
          Teuchos::Array<int> multiIndex(valuesOneCell.rank());
          multiIndex[0] = 0; // both childValues and valuesOneCell have span 1 in their first, cell dimension
          for (int childPointOrdinal=0; childPointOrdinal<numPoints; childPointOrdinal++)
          {
            const int parentPointOrdinal = parentPointOrdinals[childPointOrdinal];
            multiIndex[1] = parentPointOrdinal;
            int parentPointEnumerationIndex = valuesOneCell.getEnumeration(multiIndex);
            multiIndex[1] = childPointOrdinal;
            int childPointEnumerationIndex = childValues.getEnumeration(multiIndex);
            
            double weight = 1.0 / double(numRefSpaceChildrenForPoint[parentPointOrdinal]);
            
            for (int i=0; i<numValuesPerPoint; i++)
            {
              valuesOneCell[parentPointEnumerationIndex++] += weight * childValues[childPointEnumerationIndex++];
            }
          }
        }
      }
      Teuchos::Array<int> multiIndex(values.rank());
      multiIndex[0] = cellOrdinal;
      int fullEnumerationIndex = values.getEnumeration(multiIndex);
      int cellEnumerationIndex = 0;
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        for (int i=0; i<numValuesPerPoint; i++)
        {
          values[fullEnumerationIndex++] = valuesOneCell[cellEnumerationIndex++];
        }
      }
    }
  }
}

namespace Camellia
{
template class NeighborFunction<double>;
}

