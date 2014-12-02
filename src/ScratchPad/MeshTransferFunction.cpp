#include "MeshTransferFunction.h"

#include "GlobalDofAssignment.h"

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
  
  for ( set<CellSide>::iterator newMeshEntryIt = newMeshCellSides.begin();
       newMeshEntryIt != newMeshCellSides.end(); newMeshEntryIt++) {
    CellSide newCellSide = *newMeshEntryIt;
    CellSide originalCellSide;
    
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
}

void MeshTransferFunction::didHUnrefine(MeshTopologyPtr meshTopo, const set<GlobalIndexType> &cellIDs) {
  cout << "WARNING: didHUnrefine not yet handled in MeshTransferFunction.\n";
}

void MeshTransferFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  // TODO: implement this
  
  // MPI-communicating method.  Must be called on all ranks.
  
  cout << "WARNING: MeshTransferFunction::values() not yet implemented.\n";
  
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