#include "MeshTransferFunction.h"

#include "GlobalDofAssignment.h"

#include "CamelliaCellTools.h"

#include "RefinementPattern.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace std;

using namespace Intrepid;
using namespace Camellia;

MeshTransferFunction::MeshTransferFunction(TFunctionPtr<double> originalFunction, MeshPtr originalMesh,
    MeshPtr newMesh, double interface_t) : TFunction<double>(originalFunction->rank())
{
  _originalFunction = originalFunction;
  _originalMesh = originalMesh;
  _newMesh = newMesh;
  _interface_t = interface_t;

  // Register for refinement notifications on both meshes.  When a notification is received, update
  // the map.  For simplicity in the initial implementation, it may be useful to make the assumption
  // that originalMesh does not change after MeshTransferFunction is constructed.  This can be enforced
  // by throwing an exception if originalMesh is refined.

  Teuchos::RCP<RefinementObserver> thisPtr = Teuchos::rcp(this, false);
  _originalMesh->registerObserver(thisPtr);
  _newMesh->registerObserver(thisPtr);

  rebuildMaps();
}

bool MeshTransferFunction::boundaryValueOnly()
{
  // This function is only valid on the interface (boundary) between the two meshes.
  return true;
}


void MeshTransferFunction::didRepartition(MeshTopologyPtr meshTopo)
{
  if ((meshTopo.get() != _originalMesh->getTopology().get()) && (meshTopo.get() != _newMesh->getTopology().get()))
  {
    cout << "ERROR: MeshTransferFunction received didHRefine notification for unrecognized MeshTopology!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "MeshTransferFunction received didHRefine notification for unrecognized MeshTopology!");
  }

  rebuildMaps();
}

bool MeshTransferFunction::findAncestralPairForNewMeshCellSide(const CellSide &newMeshCellSide,
    CellSide &newMeshCellSideAncestor, CellSide &originalMeshCellSideAncestor,
    unsigned &newMeshCellSideAncestorPermutation)
{
  MeshTopologyPtr newMeshTopology = _newMesh->getTopology();
  MeshTopologyPtr originalMeshTopology = _originalMesh->getTopology();

  newMeshCellSideAncestor = newMeshCellSide;

  int sideDim = newMeshTopology->getDimension() - 1;

  bool notFound = true;
  while (notFound)
  {
    notFound = false;
    map<IndexType,IndexType> originalVertexIndexToNewVertexIndex;

    CellPtr newMeshCell = newMeshTopology->getCell(newMeshCellSideAncestor.first);
    IndexType newMeshActiveSideEntityIndex = newMeshCell->entityIndex(sideDim, newMeshCellSideAncestor.second);
    vector<IndexType> newVertexIndices = newMeshTopology->getEntityVertexIndices(sideDim, newMeshActiveSideEntityIndex);
    vector<IndexType> originalVertexIndices(newVertexIndices.size());
    for (int vertexOrdinal=0; vertexOrdinal<newVertexIndices.size(); vertexOrdinal++)
    {
      vector<double> newVertex = newMeshTopology->getVertex(newVertexIndices[vertexOrdinal]);

      if (! originalMeshTopology->getVertexIndex(newVertex, originalVertexIndices[vertexOrdinal]))
      {
        notFound = true;
        break;
      }
      originalVertexIndexToNewVertexIndex[originalVertexIndices[vertexOrdinal]] = newVertexIndices[vertexOrdinal];
    }
    if (notFound)   // missing at least one vertex; try the parent of newCellSide
    {
      if (newMeshCell->getParent().get() == NULL)
      {
        return false;
        // TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellSide not found in originalMesh hierarchy");
      }
      CellPtr parent = newMeshCell->getParent();
      unsigned childOrdinalInParent = parent->childOrdinal(newMeshCell->cellIndex());

      // subcellInfo pair is (subcdim, subcord):
      pair<unsigned, unsigned> subcellInfo = parent->refinementPattern()->mapSubcellFromChildToParent(childOrdinalInParent, sideDim, newMeshCellSideAncestor.second);

      unsigned parentSideOrdinal = subcellInfo.second;
      if ((parentSideOrdinal == -1) || (subcellInfo.first != sideDim))
      {
        return false;
      }

      newMeshCellSideAncestor.first = parent->cellIndex();
      newMeshCellSideAncestor.second = parentSideOrdinal;
    }
    else
    {
      // if we get here, found a matching cell side; now work out the details:
      set<IndexType> vertexSet(originalVertexIndices.begin(),originalVertexIndices.end());
      IndexType originalSideEntityIndex = originalMeshTopology->getEntityIndex(sideDim, vertexSet);
      if (originalSideEntityIndex == -1)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "vertices found in originalMesh, but no matching side!");
      }
      set< CellSide > cellSides = originalMeshTopology->getCellsContainingEntity(sideDim, originalSideEntityIndex);
      if (cellSides.size() == 0)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "side found, but no cells containing side found in originalMesh");
      }
      if (cellSides.size() != 1)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "multiple cells containing side found in originalMesh");
      }
      originalMeshCellSideAncestor = *cellSides.begin();

      // now, get originalVertexIndices in the order seen by the side in the original mesh
      CellPtr originalMeshCell = originalMeshTopology->getCell(originalMeshCellSideAncestor.first);
      IndexType originalMeshActiveSideEntityIndex = originalMeshCell->entityIndex(sideDim, originalMeshCellSideAncestor.second);
      originalVertexIndices = originalMeshTopology->getEntityVertexIndices(sideDim, originalMeshActiveSideEntityIndex);

      // determine permutation:
      vector<unsigned> originalOrder(originalVertexIndices.size()); // order in originalMesh
      vector<unsigned> permutedOrder = newVertexIndices; // order in the newMesh
      for (int vertexOrdinal=0; vertexOrdinal<originalVertexIndices.size(); vertexOrdinal++)
      {
        IndexType originalVertexIndex = originalVertexIndices[vertexOrdinal];
        originalOrder[vertexOrdinal] = originalVertexIndexToNewVertexIndex[originalVertexIndex];
      }
      CellTopoPtr sideTopo = _originalMesh->getTopology()->getEntityTopology(sideDim, originalSideEntityIndex);
      newMeshCellSideAncestorPermutation = CamelliaCellTools::permutationMatchingOrder(sideTopo, originalOrder, permutedOrder);
    }
  }
  return true;
}

void MeshTransferFunction::rebuildMaps()
{
  MeshTopologyPtr newMeshTopology = _newMesh->getTopology();

  int sideDim = newMeshTopology->getDimension() - 1;
  int timeDimOrdinal = sideDim; // time is the last dimension

  // 1. Find (cellID, sideOrdinal) belonging to this rank that have interface_t values in newMesh.

  // initial strategy: examine the vertices that belong to cells owned by this rank. Sides that are wholly
  //                   comprised of vertices on the interface are the sides of interest

  set<GlobalIndexType> myNewMeshCells = _newMesh->globalDofAssignment()->cellsInPartition(-1); // -1: this rank
  set<CellSide> newMeshActiveCellSides;
  double tol = 1e-15; // for matching on the interface
  for (set<GlobalIndexType>::iterator cellIDIt = myNewMeshCells.begin(); cellIDIt != myNewMeshCells.end(); cellIDIt++)
  {
    GlobalIndexType cellID = *cellIDIt;
    CellPtr cell = newMeshTopology->getCell(cellID);
    vector<unsigned> boundarySideOrdinals = cell->boundarySides();

    for (int i=0; i<boundarySideOrdinals.size(); i++)
    {
      bool sideMatchesInterface = true; // will set to false if we find a vertex that does not match
      IndexType sideEntityIndex = cell->entityIndex(sideDim, boundarySideOrdinals[i]);
      vector<IndexType> vertexIndices = newMeshTopology->getEntityVertexIndices(sideDim, sideEntityIndex);
      for (int vertexOrdinal=0; vertexOrdinal<vertexIndices.size(); vertexOrdinal++)
      {
        vector<double> vertex = newMeshTopology->getVertex(vertexIndices[vertexOrdinal]);
        if (abs(vertex[timeDimOrdinal] - _interface_t) > tol)
        {
          sideMatchesInterface = false;
        }
      }
      if (sideMatchesInterface)
      {
        CellSide cellSide = make_pair(cellID, boundarySideOrdinals[i]);
        newMeshActiveCellSides.insert(cellSide);
      }
    }
  }

//  {// DEBUGGING:
//    int rank = Teuchos::GlobalMPISession::getRank();
//    if (rank==1) {
//      cout << "On rank 1, newMeshActiveCellSides has " << newMeshActiveCellSides.size() << " entries.\n";
//      if (newMeshActiveCellSides.size() > 0) {
//        cout << "On rank 1, newMeshActiveCellSides[0] = (" << (*newMeshActiveCellSides.begin()).first << ").\n";
//      }
//      Camellia::print("Rank 1 myNewMeshCells", myNewMeshCells);
//    } else {
//      Camellia::print("Rank other than 1 myNewMeshCells", myNewMeshCells);
//    }
//  }

  // 2. Find corresponding cells in originalMesh.  During construction, we require that originalMesh be more refined
  //    than newMesh along the interface.  After construction, newMesh may be arbitrarily refined.
  // 3. Create a map in each direction.  This should be a bijection, but not all cells will be active;
  //    some will be inactive parent cells.

  MeshTopologyPtr originalMeshTopology = _originalMesh->getTopology();

  _newToOriginalMap.clear();
  _originalToNewMap.clear();

  const set<GlobalIndexType>* originalMeshActiveRankLocalCells = &_originalMesh->globalDofAssignment()->cellsInPartition(-1);

  vector<GlobalIndexType> cellsToImport;

  for ( set<CellSide>::iterator newMeshEntryIt = newMeshActiveCellSides.begin();
        newMeshEntryIt != newMeshActiveCellSides.end(); newMeshEntryIt++)
  {
    CellSide newActiveCellSide = *newMeshEntryIt;
    CellSide originalCellSide;

    CellSide newCellSide;

    unsigned permutation;
    bool matchFound = findAncestralPairForNewMeshCellSide(newActiveCellSide, newCellSide, originalCellSide, permutation);
    if (!matchFound)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no match found during rebuildMaps()");
    }

    _newToOriginalMap[newCellSide] = originalCellSide;
    _originalToNewMap[originalCellSide] = newCellSide;

    _permutationForNewMeshCellSide[newCellSide] = permutation;

    if (originalMeshActiveRankLocalCells->find(originalCellSide.first) == originalMeshActiveRankLocalCells->end())
    {
      // off-rank or parent; record for import:
      CellPtr originalCell = _originalMesh->getTopology()->getCell(originalCellSide.first);
      if (!originalCell->isParent())
      {
        cellsToImport.push_back(originalCellSide.first);
      }
      else
      {
        vector< pair< GlobalIndexType, unsigned> > descendants = originalCell->getDescendantsForSide(originalCellSide.second);
        for (vector< pair< GlobalIndexType, unsigned> >::iterator entryIt = descendants.begin(); entryIt != descendants.end(); entryIt++)
        {
          cellsToImport.push_back(entryIt->first);
        }
      }
    }
  }

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

void MeshTransferFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache)
{
  // incoming basisCache should be defined on newMesh
  if (basisCache->mesh().get() != _newMesh.get())
  {
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
  for (vector<GlobalIndexType>::iterator cellIDIt = newMeshCellIDs.begin(); cellIDIt != newMeshCellIDs.end(); cellIDIt++, cellOrdinal++)
  {
    valuesLocation[0] = cellOrdinal;

    GlobalIndexType newMeshCellID = *cellIDIt;
    unsigned newMeshCellSideOrdinal = basisCache->getSideIndex();
    CellSide newMeshActiveCellSide = make_pair(newMeshCellID, newMeshCellSideOrdinal);

    CellSide newMeshAncestralCellSide, originalMeshAncestralCellSide;

    unsigned newMeshAncestralCellSidePermutation;
    bool matchFound = findAncestralPairForNewMeshCellSide(newMeshActiveCellSide, newMeshAncestralCellSide, originalMeshAncestralCellSide, newMeshAncestralCellSidePermutation);

    if (!matchFound)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CellSide not found!");
    }

    FieldContainer<double> newMeshCellReferencePoints;

    // in newMesh, may have to map upward to an ancestor
    if (newMeshActiveCellSide == newMeshAncestralCellSide)
    {
      newMeshCellReferencePoints = refCellPoints;
    }
    else
    {
      GlobalIndexType ancestralCellID = newMeshAncestralCellSide.first;
      CellPtr cell = _newMesh->getTopology()->getCell(newMeshCellID);
      RefinementBranch refBranchVolume;
      while (cell->cellIndex() != ancestralCellID)
      {
        CellPtr parent = cell->getParent();
        unsigned childOrdinal = parent->childOrdinal(cell->cellIndex());
        refBranchVolume.insert(refBranchVolume.end(), make_pair(parent->refinementPattern().get(),childOrdinal));
        cell = parent;
      }
      RefinementBranch refBranch = RefinementPattern::sideRefinementBranch(refBranchVolume, newMeshAncestralCellSide.second);
      RefinementPattern::mapRefCellPointsToAncestor(refBranch, refCellPoints, newMeshCellReferencePoints);
    }

    {
      // DEBUGGING
      BasisCachePtr ancestorBasisCache = BasisCache::basisCacheForCell(_newMesh, newMeshAncestralCellSide.first);
      BasisCachePtr ancestorSideBasisCache = ancestorBasisCache->getSideBasisCache(newMeshAncestralCellSide.second);
      ancestorSideBasisCache->setRefCellPoints(newMeshCellReferencePoints);

      FieldContainer<double> originalPhysicalPoints(1,newMeshCellReferencePoints.dimension(0), _newMesh->getDimension());

      for (int pointOrdinal=0; pointOrdinal<newMeshCellReferencePoints.dimension(0); pointOrdinal++)
      {
        for (int d=0; d<_newMesh->getDimension(); d++)
        {
          originalPhysicalPoints(0,pointOrdinal,d) = basisCache->getPhysicalCubaturePoints()(cellOrdinal,pointOrdinal,d);
        }
      }

      FieldContainer<double> ancestorPhysicalPoints = ancestorSideBasisCache->getPhysicalCubaturePoints();
      double tol = 1e-15;
      double maxDiff = 0;
      for (int pointOrdinal=0; pointOrdinal<newMeshCellReferencePoints.dimension(0); pointOrdinal++)
      {
        for (int d=0; d<_newMesh->getDimension(); d++)
        {
          double diff = abs( originalPhysicalPoints(0,pointOrdinal,d) - ancestorPhysicalPoints(0,pointOrdinal,d) );
          if (diff==0) continue;
          double maxVal = std::max(abs(originalPhysicalPoints(0,pointOrdinal,d)), abs(ancestorPhysicalPoints(0,pointOrdinal,d)));
          double relDiff = diff / maxVal;
          maxDiff = std::max(maxDiff,relDiff);
        }
      }
      if (maxDiff > tol)
      {
        cout << "ERROR: ancestorPhysicalPoints and originalPhysicalPoints differ.\n";
        cout << "originalPhysicalPoints:\n" << originalPhysicalPoints;
        cout << "ancestorPhysicalPoints:\n" << ancestorPhysicalPoints;
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestorPhysicalPoints and originalPhysicalPoints differ");
      }
    }

    // permute newMeshCellReferencePoints according to newMeshAncestralCellSidePermutation

    unsigned sideDim = _originalMesh->getDimension() - 1;
    IndexType originalSideEntityIndex = _originalMesh->getTopology()->getCell(originalMeshAncestralCellSide.first)->entityIndex(sideDim,originalMeshAncestralCellSide.second);
    CellTopoPtr sideTopo = _originalMesh->getTopology()->getEntityTopology(sideDim, originalSideEntityIndex);

    FieldContainer<double> originalMeshCellReferencePoints(newMeshCellReferencePoints.dimension(0), newMeshCellReferencePoints.dimension(1));
    CamelliaCellTools::permutedReferenceCellPoints(sideTopo, newMeshAncestralCellSidePermutation, newMeshCellReferencePoints, originalMeshCellReferencePoints);

    // in originalMesh, may have to map downward to descendants
    CellPtr cell = _originalMesh->getTopology()->getCell(originalMeshAncestralCellSide.first);
    if (! cell->isParent())
    {
      BasisCachePtr originalBasisCache = BasisCache::basisCacheForCell(_originalMesh, originalMeshAncestralCellSide.first);
      BasisCachePtr originalBasisCacheSide = originalBasisCache->getSideBasisCache(originalMeshAncestralCellSide.second);
      originalBasisCacheSide->setRefCellPoints(originalMeshCellReferencePoints);
      int enumeration = values.getEnumeration(valuesLocation);
      FieldContainer<double> cellValues(valuesDimOneCell, &values[enumeration]);
      _originalFunction->values(cellValues, originalBasisCacheSide);
    }
    else
    {
      // otherwise, we examine points individually -- a good deal of room for optimization here, though I'm not sure how
      // much this method will be used; should do a profile to see how expensive it really is

//      vector< vector<unsigned> > branchForPoint(numPoints); // list of child ordinals
//      vector< unsigned > sideOrdinalForPoint(numPoints); // side ordinal in descendant
//      vector< vector<double> > pointInDescendant(numPoints); // in ref space for descendant side

      for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
      {
        FieldContainer<double> parentPointFC(1,sideDim);
        vector<double> refPointParent(sideDim);
        for (int d=0; d<sideDim; d++)
        {
          parentPointFC(0,d) = originalMeshCellReferencePoints(pointOrdinal,d);
          refPointParent[d] = originalMeshCellReferencePoints(pointOrdinal,d);
        }

        CellPtr descendantCell = cell;
        vector<unsigned> branch;
        unsigned sideOrdinal = originalMeshAncestralCellSide.second;

        while (descendantCell->isParent())
        {
          RefinementPatternPtr refPattern = descendantCell->refinementPattern();
          RefinementPatternPtr sideRefPattern = refPattern->sideRefinementPatterns()[sideOrdinal];

          unsigned childOrdinalInSide = sideRefPattern->childOrdinalForPoint(refPointParent);
          unsigned childOrdinalVolume = refPattern->mapSideChildIndex(sideOrdinal, childOrdinalInSide);
          unsigned childSideOrdinal = refPattern->mapSubcellFromParentToChild(childOrdinalVolume, sideDim, sideOrdinal).second;

          branch.push_back(childOrdinalVolume);

          FieldContainer<double> childPoint(1,sideDim);
          sideRefPattern->mapPointsToChildRefCoordinates(parentPointFC, childOrdinalInSide, childPoint);
          parentPointFC = childPoint;

          for (int d=0; d<sideDim; d++)
          {
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
        BasisCachePtr originalMeshBasisCache = BasisCache::basisCacheForCell(_originalMesh, originalMeshAncestralCellSide.first);
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

MeshTransferFunction::~MeshTransferFunction()
{
  _newMesh->unregisterObserver(this);
  _originalMesh->unregisterObserver(this);
}
