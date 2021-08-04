//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  CellTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Camellia_FieldContainer.hpp"

#include "CamelliaCellTools.h"
#include "Cell.h"
#include "MeshFactory.h"

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( Cell, BoundarySubcells )
  {
    // tests a new feature, necessary for certain meshes when imposing BCs on test functions, that allows determination
    // of subcells of lower dimension than sides that lie on the mesh boundary.  (For trial BC imposition, you can rely on
    // global conformity to take care of these entities; our DPG test spaces are taken to be discontinuous/nonconforming.)
    
    // set up a 2D mesh with a triangle that has no boundary sides, but has vertices that do lie on the boundary
    // unit square ABCD, where edge BC has midpoint E, edge CD has midpoint F.
    vector<double> A = {{0.0, 0.0}};
    vector<double> B = {{1.0, 0.0}};
    vector<double> C = {{1.0, 1.0}};
    vector<double> D = {{0.0, 1.0}};
    vector<double> E = {{1.0, 0.5}}; // on BC
    vector<double> F = {{0.5, 1.0}}; // on CD
    
    IndexType A_index = 0;
    IndexType B_index = 1;
    IndexType C_index = 2;
    IndexType D_index = 3;
    IndexType E_index = 4;
    IndexType F_index = 5;
    
    // elements:
    vector<IndexType> ABE_vertices = {A_index,B_index,E_index};
    vector<IndexType> ECF_vertices = {E_index,C_index,F_index};
    vector<IndexType> FDA_vertices = {F_index,D_index,A_index};
    vector<IndexType> AEF_vertices = {A_index,E_index,F_index};
    
    vector< vector<double> > vertices = {A,B,C,D,E,F};
    vector< vector< IndexType > > elementVertices = {ABE_vertices, ECF_vertices, FDA_vertices, AEF_vertices};
    
    auto triTopo = CellTopology::triangle();
    auto cellTopos = vector<CellTopoPtr>(elementVertices.size(), triTopo);
    
    MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos));
    
    vector<PeriodicBCPtr> periodicBCs;
    
    MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(geometry, periodicBCs) );
    
    // while cell AEF is the most interesting, we might as well test all four cells
    CellPtr ABE = meshTopology->getCell(0);
    CellPtr ECF = meshTopology->getCell(1);
    CellPtr FDA = meshTopology->getCell(2);
    CellPtr AEF = meshTopology->getCell(3);
    
    // all cells have all their vertices on the boundary
    set<unsigned> expectedVertexOrdinals;
    set<unsigned> expectedSideOrdinals;
    
    // define a lambda for checking boundary entries match expected side, vertex ordinals
    auto checkBoundarySubcells = [&](vector< pair<unsigned,unsigned> > actualEntries) -> void
    {
      int entryCountExpected = expectedVertexOrdinals.size() + expectedSideOrdinals.size();
      TEST_EQUALITY(actualEntries.size(), entryCountExpected);
      
      set< pair<unsigned,unsigned> > entriesSeen;
      for (auto entry : actualEntries)
      {
        unsigned scdim = entry.first;
        unsigned scord = entry.second;
        
        if (scdim == 0)
        {
          // vertex
          if (expectedVertexOrdinals.find(scord) == expectedVertexOrdinals.end())
          {
            out << "FAILURE: Unexpected vertex ordinal " << scord << " included in boundary subcells.\n";
            success = false;
          }
        }
        else if (scdim == 1)
        {
          // side
          if (expectedSideOrdinals.find(scord) == expectedSideOrdinals.end())
          {
            out << "FAILURE: Unexpected edge ordinal " << scord << " included in boundary subcells.\n";
            success = false;
          }
        }
        
        if (entriesSeen.find(entry) != entriesSeen.end())
        {
          out << "FAILURE: Duplicate entry for subcell with dimension " << scdim << ", ordinal " << scord << endl;
          success = false;
        }
        else
        {
          entriesSeen.insert(entry);
        }
      }
    };
    
    // ABE has boundary sides AB, BE (ordinals 0 and 1)
    auto ABE_boundary = ABE->boundarySubcells();
    expectedVertexOrdinals = {0,1,2};
    expectedSideOrdinals = {0,1};
    
    out << "Checking cell ABE\n";
    checkBoundarySubcells(ABE_boundary);
    
    // ECF has boundary sides EC, CF (ordinals 0 and 1)
    auto ECF_boundary = ECF->boundarySubcells();
    expectedVertexOrdinals = {0,1,2};
    expectedSideOrdinals = {0,1};
    
    out << "Checking cell ECF\n";
    checkBoundarySubcells(ECF_boundary);
    
    // FDA has boundary sides FD, DA (ordinals 0 and 1)
    auto FDA_boundary = FDA->boundarySubcells();
    expectedVertexOrdinals = {0,1,2};
    expectedSideOrdinals = {0,1};
    
    out << "Checking cell FDA\n";
    checkBoundarySubcells(FDA_boundary);
    
    // AEF has no boundary sides
    auto AEF_boundary = AEF->boundarySubcells();
    expectedVertexOrdinals = {0,1,2};
    expectedSideOrdinals = {};
    
    out << "Checking cell AEF\n";
    checkBoundarySubcells(AEF_boundary);
  }

TEUCHOS_UNIT_TEST( Cell, EntitiesOnNeighborInterfacesHangingNodes_2D )
{
  // this test confirms that we get neighbor interfaces at all levels of the hierarchy when there are hanging nodes
  
  double width=1.0, height=1.0;
  int horizontalElements=2, verticalElements=1;
  MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(width,height,horizontalElements,verticalElements);
  
  IndexType firstChildCellIndex = meshTopo->nextCellID();
  IndexType cellToRefine = 0;
  RefinementPatternPtr quadRefPattern = RefinementPattern::regularRefinementPatternQuad();
  meshTopo->refineCell(cellToRefine, quadRefPattern, firstChildCellIndex);
  
  CellPtr cell1 = meshTopo->getCell(1); // the neighbor, which is unrefined
  const unsigned SIDE_DIM = 1;
  
  const unsigned sideCount = cell1->getSideCount();
  unsigned sharedSideOrdinalInCell1 = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
  unsigned sharedSideOrdinalInCell0 = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    auto neighbor = cell1->getNeighbor(sideOrdinal, meshTopo);
    if (neighbor != Teuchos::null)
    {
      sharedSideOrdinalInCell1 = sideOrdinal;
      
      auto neighborInfo = cell1->getNeighborInfo(sideOrdinal, meshTopo);
      sharedSideOrdinalInCell0 = neighborInfo.second;
      break;
    }
  }
  
  std::set<IndexType> expectedSideEntityIndices;
  expectedSideEntityIndices.insert(cell1->entityIndex(SIDE_DIM, sharedSideOrdinalInCell1));
  
  CellPtr cell0 = meshTopo->getCell(0); // the parent cell, which is refined
  vector< pair<GlobalIndexType, unsigned> > childrenOnRightSide = cell0->childrenForSide(sharedSideOrdinalInCell0);
  
  for (const auto &childEntry : childrenOnRightSide )
  {
    const auto & cellID    = childEntry.first;
    const auto & childSideOrdinal = childEntry.second;
    
    CellPtr childCell = meshTopo->getCell(cellID);
    IndexType rightSideChildCell = childCell->entityIndex(SIDE_DIM, childSideOrdinal);
    expectedSideEntityIndices.insert(rightSideChildCell);
  }
  
  const bool peersOnly = false;
  auto sideEntitiesVector = cell1->entitiesOnNeighborInterfaces(SIDE_DIM, peersOnly, meshTopo);
  
  std::set<IndexType> actualSideEntityIndices(sideEntitiesVector.begin(), sideEntitiesVector.end());
  
  if (actualSideEntityIndices.size() != expectedSideEntityIndices.size())
  {
    out << "Number of side entity indices (" << actualSideEntityIndices.size() << ") does not match expected (" << expectedSideEntityIndices.size() << ")\n";
    success = false;
  }
  else
  {
    auto actualIt   = actualSideEntityIndices.begin();
    auto expectedIt = expectedSideEntityIndices.begin();
    while (actualIt != actualSideEntityIndices.end())
    {
      TEST_EQUALITY(*actualIt, *expectedIt);
      actualIt++;
      expectedIt++;
    }
  }
}

TEUCHOS_UNIT_TEST( Cell, EntitiesOnNeighborInterfacesHangingNodes_3D )
{
  // this test confirms that we get neighbor interfaces at all levels of the hierarchy when there are hanging nodes
  
  std::vector<double> widths {2.0,1.0,1.0};
  std::vector<int> elementCounts {2, 1, 1};
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(widths, elementCounts);
  
  CellPtr cell0 = meshTopo->getCell(0);
  const unsigned sideCount = cell0->getSideCount();
  unsigned sharedSideOrdinalInCell1 = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
  unsigned sharedSideOrdinalInCell0 = INVALID_CELLTOPOLOGY_SIDE_ORDINAL;
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    auto neighbor = cell0->getNeighbor(sideOrdinal, meshTopo);
    if (neighbor != Teuchos::null)
    {
      sharedSideOrdinalInCell0 = sideOrdinal;
      
      auto neighborInfo = cell0->getNeighborInfo(sideOrdinal, meshTopo);
      sharedSideOrdinalInCell1 = neighborInfo.second;
      break;
    }
  }
  
  IndexType firstChildCellIndex = meshTopo->nextCellID();
  IndexType cellToRefine = 0;
  RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPatternHexahedron();
  meshTopo->refineCell(cellToRefine, refPattern, firstChildCellIndex);
  
  CellPtr cell1 = meshTopo->getCell(1); // the neighbor, which is unrefined
  const unsigned SIDE_DIM = 2;
  
  std::set<IndexType> expectedSideEntityIndices;
  expectedSideEntityIndices.insert(cell1->entityIndex(SIDE_DIM, sharedSideOrdinalInCell1));
    
  vector< pair<GlobalIndexType, unsigned> > childrenOnRightSide = cell0->childrenForSide(sharedSideOrdinalInCell0);
  
  for (const auto &childEntry : childrenOnRightSide )
  {
    const auto & cellID    = childEntry.first;
    const auto & childSideOrdinal = childEntry.second;
    
    CellPtr childCell = meshTopo->getCell(cellID);
    IndexType rightSideChildCell = childCell->entityIndex(SIDE_DIM, childSideOrdinal);
    expectedSideEntityIndices.insert(rightSideChildCell);
  }
  
  const bool peersOnly = false;
  auto sideEntitiesVector = cell1->entitiesOnNeighborInterfaces(SIDE_DIM, peersOnly, meshTopo);
  
  std::set<IndexType> actualSideEntityIndices(sideEntitiesVector.begin(), sideEntitiesVector.end());
  
  if (actualSideEntityIndices.size() != expectedSideEntityIndices.size())
  {
    out << "Number of side entity indices (" << actualSideEntityIndices.size() << ") does not match expected (" << expectedSideEntityIndices.size() << ")\n";
    success = false;
  }
  else
  {
    auto actualIt   = actualSideEntityIndices.begin();
    auto expectedIt = expectedSideEntityIndices.begin();
    while (actualIt != actualSideEntityIndices.end())
    {
      TEST_EQUALITY(*actualIt, *expectedIt);
      actualIt++;
      expectedIt++;
    }
  }
}

TEUCHOS_UNIT_TEST( Cell, FindSubcellOrdinalInSide )
{
  double width=1.0, height=1.0;
  int horizontalElements=2, verticalElements=1;
  MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(width,height,horizontalElements,verticalElements);

  int sideDim = meshTopo->getDimension() - 1;
  for (auto cellIndex : meshTopo->getLocallyKnownActiveCellIndices())
  {
    CellPtr cell = meshTopo->getCell(cellIndex);
    CellTopoPtr cellTopo = cell->topology();
    for (int sideOrdinal=0; sideOrdinal<cell->getSideCount(); sideOrdinal++)
    {
      IndexType sideEntityIndex = cell->getEntityIndices(sideDim)[sideOrdinal];
      for (int subcdim=0; subcdim <= sideDim; subcdim++)
      {
        int subcCount = meshTopo->getSubEntityCount(sideDim, sideEntityIndex, subcdim);
        for (int subcOrdinalInGlobalSide=0; subcOrdinalInGlobalSide<subcCount; subcOrdinalInGlobalSide++)
          // subcOrdinalGlobal as opposed to as ordered by the cell side
        {
          IndexType subcEntityIndex = meshTopo->getSubEntityIndex(sideDim, sideEntityIndex, subcdim, subcOrdinalInGlobalSide);
          IndexType subcOrdinalInCellExpected = cell->findSubcellOrdinal(subcdim, subcEntityIndex);
          IndexType subcOrdinalInCellSide = cell->findSubcellOrdinalInSide(subcdim, subcEntityIndex, sideOrdinal);
          IndexType subcOrdinalInCellActual = CamelliaCellTools::subcellOrdinalMap(cellTopo, sideDim, sideOrdinal, subcdim, subcOrdinalInCellSide);
          TEST_EQUALITY(subcOrdinalInCellActual, subcOrdinalInCellExpected);
        }
      }
    }
  }
}

TEUCHOS_UNIT_TEST( Cell, Neighbors1D )
{
  int numCells = 8;
  double xLeft = 0, xRight = 1;
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology( xLeft, xRight, numCells );

  int numBoundarySides = 0;

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    CellPtr cell = meshTopo->getCell(cellIndex);

    TEST_ASSERT(cell->getSideCount() == 2);

    for (int sideOrdinal = 0; sideOrdinal < cell->getSideCount(); sideOrdinal++)
    {
      pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal,meshTopo);
      if (neighborInfo.first == -1)
      {
        numBoundarySides++;
      }
      else
      {
        CellPtr neighbor = meshTopo->getCell(neighborInfo.first);
        unsigned sideOrdinalInNeighbor = neighborInfo.second;
        pair<GlobalIndexType, unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(sideOrdinalInNeighbor,meshTopo);
        TEST_ASSERT(neighborNeighborInfo.first == cellIndex);
      }
    }
  }
  TEST_ASSERT(numBoundarySides == 2);
}
  
  TEUCHOS_UNIT_TEST( Cell, VertexNeighbors_Triangles )
  {
    bool useTriangles = true;
    
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,1};
    
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                                             elementCounts[0], elementCounts[1], useTriangles);
    
    GlobalIndexType cellID = 1, neighborCellID = 0;
    CellPtr cell = meshTopo->getCell(cellID);
    unsigned vertexDim = 0;
    set<GlobalIndexType> neighborIDs = cell->getActiveNeighborIndices(vertexDim, meshTopo);
    
    TEST_EQUALITY(neighborIDs.size(), 1);
    TEST_EQUALITY(*neighborIDs.begin(), neighborCellID);
    
    // something similar, but now use 2x2 quad mesh divided into triangles
    // the "0" triangle (SE of bottom-left quad) shares a vertex with all but the top left triangle
    cellID = 0;
    elementCounts = {2,2};
    
    meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                             elementCounts[0], elementCounts[1], useTriangles);
    
    cell = meshTopo->getCell(cellID);
    neighborIDs = cell->getActiveNeighborIndices(vertexDim, meshTopo);
    
    // "-2" below is to exclude cell 0 as well as the top left triangle
    int neighborsExpected = elementCounts[0] * elementCounts[1] * 2 - 2;
    
    TEST_EQUALITY(neighborIDs.size(), neighborsExpected);
  }
  
  TEUCHOS_UNIT_TEST( Cell, VertexNeighbors_HangingNodeTriangle )
  {
    bool useTriangles = true;
    vector<double> dimensions = {1.0,1.0};
    vector<int> elementCounts = {1,1};
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(dimensions[0], dimensions[1],
                                                             elementCounts[0], elementCounts[1], useTriangles);
    IndexType firstChildCellIndex = meshTopo->nextCellID();
    IndexType cellToRefine = 0;
    RefinementPatternPtr triangleRefPattern = RefinementPattern::regularRefinementPatternTriangle();
    meshTopo->refineCell(cellToRefine, triangleRefPattern, firstChildCellIndex);
    
    // find the edge the two original cells share:
    IndexType sharedEdgeEntityIndex = -1;
    CellPtr cell0 = meshTopo->getCell(0);
    CellPtr cell1 = meshTopo->getCell(1);
    int edgeCount = cell0->topology()->getEdgeCount();
    int vertexDim = 0;
    int edgeDim = 1;
    for (int edgeOrdinal=0; edgeOrdinal<edgeCount; edgeOrdinal++)
    {
      if (cell0->getNeighbor(edgeOrdinal, meshTopo) != Teuchos::null)
      {
        sharedEdgeEntityIndex = cell0->entityIndex(edgeDim, edgeOrdinal);
      }
    }
    // find the new vertex created by breaking that shared edge
    vector<IndexType> edgeChildren = meshTopo->getChildEntities(edgeDim, sharedEdgeEntityIndex);
    TEUCHOS_TEST_FOR_EXCEPT(edgeChildren.size() != 2);
    vector<IndexType> firstChildNodes = meshTopo->getEntityVertexIndices(edgeDim, edgeChildren[0]);
    vector<IndexType> secondChildNodes = meshTopo->getEntityVertexIndices(edgeDim, edgeChildren[1]);
    
    IndexType newVertexIndex;
    if ((firstChildNodes[0] == secondChildNodes[0]) || (firstChildNodes[0] == secondChildNodes[1]))
    {
      newVertexIndex = firstChildNodes[0];
    }
    else if ((firstChildNodes[1] == secondChildNodes[0]) || (firstChildNodes[1] == secondChildNodes[1]))
    {
      newVertexIndex = firstChildNodes[1];
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "new vertex not found!");
    }
    // which active cells match the new vertex?  All such cells should be included among cell1's vertex neighbors
    set<pair<IndexType,unsigned>> matchingCellEntries = meshTopo->getCellsContainingEntity(vertexDim, newVertexIndex);
    set<IndexType> matchingCellIDs;
    for (auto entry : matchingCellEntries)
    {
      matchingCellIDs.insert(entry.first);
    }
    
    set<GlobalIndexType> neighborIDs = cell1->getActiveNeighborIndices(vertexDim, meshTopo);

    for (IndexType newVertexCellID : matchingCellIDs)
    {
      if (neighborIDs.find(newVertexCellID) == neighborIDs.end())
      {
        success = false;
        out << newVertexCellID << " not found in neighborIDs.\n";
      }
    }
    print(out, "matchingCellIDs", matchingCellIDs);
    print(out, "neighborIDs", neighborIDs);
  }
} // namespace
