#include "MeshTransferFunction.h"

#include "GlobalDofAssignment.h"

#include "CamelliaCellTools.h"

#include "RefinementPattern.h"

using namespace std;

MeshTransferFunction::MeshTransferFunction(FunctionPtr originalFunction, MeshPtr originalMesh,
                                           MeshPtr newMesh, double interface_t) : Function(originalFunction->rank()) {
  _originalFunction = originalFunction;
  _originalMesh = originalMesh;
  _newMesh = newMesh;
  
  MeshTopologyPtr newMeshTopology = newMesh->getTopology();
  
  int sideDim = newMeshTopology->getSpaceDim() - 1;
  int timeDimOrdinal = sideDim; // time is the last dimension
  
  // 1. Find (cellID, sideOrdinal) belonging to this rank that have interface_t values in newMesh.
  
  // initial strategy: examine the vertices that belong to cells owned by this rank. Sides that are wholly
  //                   comprised of vertices on the interface are the sides of interest
  
  set<GlobalIndexType> myNewMeshCells = newMesh->globalDofAssignment()->cellsInPartition(-1); // -1: this rank
  set<CellSide> newMeshCellSides;
  double tol = 1e-15; // for matching on the interface
  for (set<GlobalIndexType>::iterator cellIDIt = myNewMeshCells.begin(); cellIDIt != myNewMeshCells.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    CellPtr cell = newMeshTopology->getCell(cellID);
    vector<unsigned> boundarySideOrdinals = cell->boundarySides();
    for (int i=0; i<boundarySideOrdinals.size(); i++) {
      bool sideMatchesInterface = true; // will set to false if we find a vertex that does not match
      IndexType sideEntityIndex = cell->entityIndex(sideDim, boundarySideOrdinals[i]);
      vector<IndexType> vertexIndices = newMeshTopology->getEntityVertexIndices(sideDim, sideEntityIndex);
      for (int vertexOrdinal=0; vertexOrdinal<vertexIndices.size(); vertexOrdinal++) {
        vector<double> vertex = newMeshTopology->getVertex(vertexIndices[vertexOrdinal]);
        if (abs(vertex[timeDimOrdinal] - interface_t) > tol) {
          sideMatchesInterface = false;
        }
      }
      if (sideMatchesInterface) {
        CellSide cellSide = make_pair(cellID, boundarySideOrdinals[i]);
        newMeshCellSides.insert(cellSide);
      }
    }
  }
  
  // 2. Find corresponding cells in originalMesh.  During construction, we require that originalMesh be more refined
  //    than newMesh along the interface.  After construction, newMesh may be arbitrarily refined.
  // 3. Create a map in each direction.  This should be a bijection, but not all cells will be active;
  //    some will be inactive parent cells.
  
  MeshTopologyPtr originalMeshTopology = originalMesh->getTopology();
  
  _newToOriginalMap.clear();
  _originalToNewMap.clear();
  
  const set<GlobalIndexType>* originalMeshLocalCells = &originalMesh->globalDofAssignment()->cellsInPartition(-1);
  
  vector<GlobalIndexType> cellsToImport;
  
  for ( set<CellSide>::iterator newMeshEntryIt = newMeshCellSides.begin();
       newMeshEntryIt != newMeshCellSides.end(); newMeshEntryIt++) {
    CellSide newCellSide = *newMeshEntryIt;
    CellSide originalCellSide;
    
    map<IndexType,IndexType> originalVertexIndexToNewVertexIndex;
    
    bool notFound = true;
    while (notFound) {
      notFound = false;
      CellPtr newMeshCell = newMeshTopology->getCell(newCellSide.first);
      IndexType newMeshSideEntityIndex = newMeshCell->entityIndex(sideDim, newCellSide.second);
      vector<IndexType> newVertexIndices = newMeshTopology->getEntityVertexIndices(sideDim, newMeshSideEntityIndex);
      vector<IndexType> originalVertexIndices(newVertexIndices.size());
      for (int vertexOrdinal=0; vertexOrdinal<newVertexIndices.size(); vertexOrdinal++) {
        vector<double> newVertex = newMeshTopology->getVertex(newVertexIndices[vertexOrdinal]);
        
        if (! originalMeshTopology->getVertexIndex(newVertex, originalVertexIndices[vertexOrdinal])) {
          notFound = true;
          break;
        }
        originalVertexIndexToNewVertexIndex[originalVertexIndices[vertexOrdinal]] = newVertexIndices[vertexOrdinal];
      }
      if (notFound) { // missing at least one vertex; try the parent of newCellSide
        if (newMeshCell->getParent().get() == NULL) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellSide not found in originalMesh hierarchy");
        }
        CellPtr parent = newMeshCell->getParent();
        unsigned childOrdinalInParent = parent->childOrdinal(newMeshCell->cellIndex());
        
        // subcellInfo pair is (subcdim, subcord):
        pair<unsigned, unsigned> subcellInfo = parent->refinementPattern()->mapSubcellFromChildToParent(childOrdinalInParent, sideDim, newCellSide.second);
        
        unsigned parentSideOrdinal = subcellInfo.second;
        if (parentSideOrdinal == -1) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No matching side in parent");
        }
        
        newCellSide.first = parent->cellIndex();
        newCellSide.second = parentSideOrdinal;
      } else {
        set<IndexType> vertexSet(originalVertexIndices.begin(),originalVertexIndices.end());
        IndexType originalSideEntityIndex = originalMeshTopology->getEntityIndex(sideDim, vertexSet);
        if (originalSideEntityIndex == -1) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertices found in originalMesh, but no matching side!");
        }
        set< CellSide > cellSides = originalMeshTopology->getCellsContainingEntity(sideDim, originalSideEntityIndex);
        if (cellSides.size() == 0) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "side found, but no cells containing side found in originalMesh");
        }
        if (cellSides.size() != 1) {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "multiple cells containing side found in originalMesh");
        }
        originalCellSide = *cellSides.begin();
        _newToOriginalMap[newCellSide] = originalCellSide;
        _originalToNewMap[originalCellSide] = newCellSide;
        
        // determine permutation:
        vector<unsigned> originalOrder(originalVertexIndices.size()); // order in originalMesh
        vector<unsigned> permutedOrder = newVertexIndices; // order in the newMesh
        for (int vertexOrdinal=0; vertexOrdinal<originalVertexIndices.size(); vertexOrdinal++) {
          IndexType originalVertexIndex = originalVertexIndices[vertexOrdinal];
          originalOrder[vertexOrdinal] = originalVertexIndexToNewVertexIndex[originalVertexIndex];
        }
        const shards::CellTopology* sideTopo = &_originalMesh->getTopology()->getEntityTopology(sideDim, originalSideEntityIndex);
        _permutationForNewMeshCellSide[newCellSide] = CamelliaCellTools::permutationMatchingOrder(*sideTopo, originalOrder, permutedOrder);
        
        if (originalMeshLocalCells->find(originalCellSide.first) == originalMeshLocalCells->end()) {
          // off-rank, record for import:
          cellsToImport.push_back(originalCellSide.first);
        }
      }
    }
  }
  
  // 4. Register for refinement notifications on both meshes.  When a notification is received, update
  //    the map.  For simplicity in the initial implementation, it may be useful to make the assumption
  //    that originalMesh does not change after MeshTransferFunction is constructed.  This can be enforced
  //    by throwing an exception if originalMesh is refined.
  
  Teuchos::RCP<RefinementObserver> thisPtr = Teuchos::rcp(this, false);
  _originalMesh->registerObserver(thisPtr);
  _newMesh->registerObserver(thisPtr);
  
  // 5. import off-rank cell data according to current mesh partitioning:
  _originalFunction->importCellData(cellsToImport);
  
  // One question is how to deal with the fact that corresponding cells in original and new mesh
  // may belong to different ranks.  In such a case, values() will be called with data (that is, physical
  // points and cell IDs) corresponding to newMesh.  It seems that the most general approach will be to
  // do a round of MPI communication (via Epetra_MpiDistributor) to get appropriate data to the owner
  // of the corresponding cell on originalMesh.  Then each rank will translate the data to originalMesh
  // (possibly by appropriate permutations of the reference points), create a corresponding BasisCache,
  // and compute values.  Finally, another round of MPI communication will send the data back to the requestor.
  
  // The above is not a communications-optimized approach, in that cubature points, e.g., can be locally determined
  // without being sent.  Similarly, in the case of solution data, simply sending the solution coefficients to the
  // requesting MPI rank would give sufficient information that all necessary computation could be done locally.  However,
  // this is making an assumption about the nature of originalFunction.  It is worth noting that the communications
  // here typically only need to be done once per solve on newMesh, and this during the determination of boundary conditions.
}

bool MeshTransferFunction::boundaryValueOnly() {
  // This function is only valid on the interface (boundary) between the two meshes.
  return true;
}


void MeshTransferFunction::didHRefine(MeshTopologyPtr meshTopo, const set<GlobalIndexType> &cellIDs,
                                      Teuchos::RCP<RefinementPattern> refPattern) {
  cout << "WARNING: didHRefine not yet handled in MeshTransferFunction.\n";

//  vector<GlobalIndexType> cellsToImport;
//  _originalFunction->importCellData(cellsToImport);
}

void MeshTransferFunction::didHUnrefine(MeshTopologyPtr meshTopo, const set<GlobalIndexType> &cellIDs) {
  cout << "WARNING: didHUnrefine not yet handled in MeshTransferFunction.\n";
  //  vector<GlobalIndexType> cellsToImport;
  //  _originalFunction->importCellData(cellsToImport);
}

void MeshTransferFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  // TODO: implement this
  
  cout << "WARNING: MeshTransferFunction::values() not yet implemented.\n";
  
  // incoming basisCache should be defined on newMesh
  if (basisCache->mesh().get() != _newMesh.get()) {
    cout << "ERROR: MeshTransferFunction::values() requires incoming BasisCache to be defined on newMesh.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache not defined on newMesh");
  }
  
  // to be general and to keep this first version simple, we just iterate over the cells in basisCache,
  // creating BasisCache(s) on originalMesh
  
  FieldContainer<double> refCellPoints = basisCache->getRefCellPoints();
  
  Teuchos::Array<int> valuesLocation(values.rank());
  Teuchos::Array<int> valuesDimOneCell;
  values.dimensions(valuesDimOneCell);
  valuesDimOneCell[0] = 1; // one cell
  Teuchos::Array<int> valuesDimOneCellOnePoint = valuesDimOneCell;
  valuesDimOneCellOnePoint[1] = 1; // one point
  
  int numPoints = refCellPoints.dimension(0);
  
  vector<GlobalIndexType> newMeshCellIDs = basisCache->cellIDs();
  int cellOrdinal = 0;
  for (vector<GlobalIndexType>::iterator cellIDIt = newMeshCellIDs.begin(); cellIDIt != newMeshCellIDs.end(); cellIDIt++, cellOrdinal++) {
    valuesLocation[0] = cellOrdinal;
    
    GlobalIndexType newMeshCellID = *cellIDIt;
    unsigned newMeshCellSideOrdinal = basisCache->getSideIndex();
    CellSide newMeshCellSide = make_pair(newMeshCellID, newMeshCellSideOrdinal);
    
    FieldContainer<double> newMeshCellReferencePoints;
    
    // in newMesh, may have to map upward to an ancestor
    if (_activeSideToAncestralSideInNewMesh.find(newMeshCellSide) == _activeSideToAncestralSideInNewMesh.end()) {
      newMeshCellReferencePoints = refCellPoints;
    } else {
      CellSide ancestralSide = _activeSideToAncestralSideInNewMesh[newMeshCellSide];
      GlobalIndexType ancestralCellID = ancestralSide.first;
      CellPtr cell = _newMesh->getTopology()->getCell(ancestralCellID);
      RefinementBranch refBranchVolume;
      while (cell->cellIndex() != ancestralCellID) {
        CellPtr parent = cell->getParent();
        unsigned childOrdinal = parent->childOrdinal(cell->cellIndex());
        refBranchVolume.insert(refBranchVolume.end(), make_pair(parent->refinementPattern().get(),childOrdinal));
        cell = parent;
      }
      RefinementBranch refBranch = RefinementPattern::sideRefinementBranch(refBranchVolume, ancestralSide.second);
      RefinementPattern::mapRefCellPointsToAncestor(refBranch, refCellPoints, newMeshCellReferencePoints);
      newMeshCellSide = ancestralSide;
    }
    
    if (_newToOriginalMap.find(newMeshCellSide) == _newToOriginalMap.end()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CellSide not found!");
    }
    CellSide originalCellSide = _newToOriginalMap[newMeshCellSide];
    
    // permute newMeshCellReferencePoints according to _permutationForNewMeshCellSide
    unsigned permutation = _permutationForNewMeshCellSide[newMeshCellSide];
    
    unsigned sideDim = _originalMesh->getDimension() - 1;
    IndexType originalSideEntityIndex = _originalMesh->getTopology()->getCell(originalCellSide.first)->entityIndex(sideDim,originalCellSide.second);
    const shards::CellTopology* sideTopo = &_originalMesh->getTopology()->getEntityTopology(sideDim, originalSideEntityIndex);
    
    FieldContainer<double> originalMeshCellReferencePoints(newMeshCellReferencePoints.dimension(0), newMeshCellReferencePoints.dimension(1));
    CamelliaCellTools::permutedReferenceCellPoints(*sideTopo, permutation, newMeshCellReferencePoints, originalMeshCellReferencePoints);
    
    // in originalMesh, may have to map downward to descendants
    CellPtr cell = _originalMesh->getTopology()->getCell(originalCellSide.first);
    if (! cell->isParent()) {
      BasisCachePtr originalBasisCache = BasisCache::basisCacheForCell(_originalMesh, originalCellSide.first);
      BasisCachePtr originalBasisCacheSide = originalBasisCache->getSideBasisCache(originalCellSide.second);
      originalBasisCacheSide->setRefCellPoints(originalMeshCellReferencePoints);
      int enumeration = values.getEnumeration(valuesLocation);
      FieldContainer<double> cellValues(valuesDimOneCell, &values[enumeration]);
      _originalFunction->values(cellValues, originalBasisCacheSide);
    } else {
      // otherwise, we examine points individually -- a good deal of room for optimization here, though I'm not sure how
      // much this method will be used; should do a profile to see how expensive it really is
      
//      vector< vector<unsigned> > branchForPoint(numPoints); // list of child ordinals
//      vector< unsigned > sideOrdinalForPoint(numPoints); // side ordinal in descendant
//      vector< vector<double> > pointInDescendant(numPoints); // in ref space for descendant side
      
      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) {
        FieldContainer<double> parentPointFC(1,sideDim);
        vector<double> refPointParent(sideDim);
        for (int d=0; d<sideDim; d++) {
          parentPointFC(0,d) = originalMeshCellReferencePoints(pointOrdinal,d);
          refPointParent[d] = originalMeshCellReferencePoints(pointOrdinal,d);
        }
        
        CellPtr descendantCell = cell;
        vector<unsigned> branch;
        unsigned sideOrdinal;
        
        while (descendantCell->isParent()) {
          RefinementPatternPtr refPattern = descendantCell->refinementPattern();
          RefinementPatternPtr sideRefPattern = refPattern->sideRefinementPatterns()[originalCellSide.second];
          
          unsigned childOrdinalInSide = sideRefPattern->childOrdinalForPoint(refPointParent);
          unsigned childOrdinalVolume = refPattern->mapSideChildIndex(originalCellSide.second, childOrdinalInSide);
          unsigned childSideOrdinal = refPattern->mapSubcellFromParentToChild(childOrdinalVolume, sideDim, originalCellSide.second).second;
          
          branch.push_back(childOrdinalVolume);
          
          FieldContainer<double> childPoint(1,sideDim);
          sideRefPattern->mapPointsToChildRefCoordinates(parentPointFC, childOrdinalInSide, childPoint);
          parentPointFC = childPoint;
          
          for (int d=0; d<sideDim; d++) {
            refPointParent[d] = childPoint(0,d);
          }
          
          sideOrdinal = childSideOrdinal;
          descendantCell = descendantCell->children()[childOrdinalVolume];
        }
//        branchForPoint[pointOrdinal] = branch;
//        sideOrdinalForPoint[pointOrdinal] = sideOrdinal;
//        pointInDescendant[pointOrdinal] = refPointParent;
        
        valuesLocation[1] = pointOrdinal;
        int enumeration = values.getEnumeration(valuesLocation);
        valuesLocation[1] = 0; // clear

        FieldContainer<double> pointValues(valuesDimOneCellOnePoint, &values[enumeration]);
        BasisCachePtr originalMeshBasisCache = BasisCache::basisCacheForCell(_originalMesh, originalCellSide.first);
        BasisCachePtr originalMeshBasisCacheSide = basisCache->getSideBasisCache(sideOrdinal);
        originalMeshBasisCacheSide->setRefCellPoints(parentPointFC);
        
        _originalFunction->values(pointValues, originalMeshBasisCacheSide);
      }
      
    }
  }
  
  /*
   basisCache contains info on cells in newMesh.
   Basic idea: for each CellSide in basisCache, determine the corresponding CellSide in originalMesh.
   This may not be an active cell.  If it is not, we need to subdivide the physical points, assigning
   subsets to active cells.  We might do this by sending *all* the physical points to all the owners of
   the CellSide's descendants.  It would then be the owner's responsibility to determine which physical 
   points belong to them, and to communicate regarding these.  (It might be that this is done via an
   integer array indicating which point ordinals are being claimed followed by a double array indicating
   which values correspond.)
   */
  
}

MeshTransferFunction::~MeshTransferFunction() {
  _newMesh->unregisterObserver(this);
  _originalMesh->unregisterObserver(this);
}