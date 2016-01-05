//
//  MeshTopologyTests
//  Camellia
//
//  Created by Nate Roberts on 12/8/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "CamelliaCellTools.h"
#include "MeshTopology.h"
#include "PoissonFormulation.h"

#include "MeshFactory.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{
  MeshTopologyPtr constructIrregularMeshTopology(int irregularity)
  {
    int spaceDim = 2;
    int meshWidth = 2;
    vector<double> dimensions(spaceDim,1.0);
    vector<int> elementCounts(spaceDim,meshWidth);
    
    MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts);

    // pick arbitrary cell to refine:
    GlobalIndexType activeCellWithBigNeighbor = 1;
    unsigned sharedSideOrdinal = -1;
    CellPtr activeCell = meshTopo->getCell(activeCellWithBigNeighbor);
    for (int sideOrdinal=0; sideOrdinal<activeCell->getSideCount(); sideOrdinal++)
    {
      if (activeCell->getNeighbor(sideOrdinal, meshTopo) != Teuchos::null)
      {
        sharedSideOrdinal = sideOrdinal;
      }
    }
    GlobalIndexType nextCellID = meshTopo->cellCount();
    for (int i=0; i<irregularity; i++)
    {
      CellPtr cellToRefine = meshTopo->getCell(activeCellWithBigNeighbor);
      RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellToRefine->topology());
      meshTopo->refineCell(activeCellWithBigNeighbor, refPattern, nextCellID);
      nextCellID += refPattern->numChildren();
      
      // setup for the next refinement, if any:
      auto childEntry = cellToRefine->childrenForSide(sharedSideOrdinal)[0];
      GlobalIndexType childWithNeighborCellID = childEntry.first;
      sharedSideOrdinal = childEntry.second;
      activeCellWithBigNeighbor = childWithNeighborCellID;
    }
    
    return meshTopo;
  }
  
void testConstraints( MeshTopology* mesh, unsigned entityDim, map<unsigned,pair<IndexType,unsigned> > &expectedConstraints, Teuchos::FancyOStream &out, bool &success)
{

  string meshName = "mesh";

  // check constraints for entities belonging to active cells
  set<unsigned> activeCells = mesh->getActiveCellIndices();

  for (set<unsigned>::iterator cellIt = activeCells.begin(); cellIt != activeCells.end(); cellIt++)
  {
    unsigned cellIndex = *cellIt;
    CellPtr cell = mesh->getCell(cellIndex);
    vector<unsigned> entitiesForCell = cell->getEntityIndices(entityDim);
    for (vector<unsigned>::iterator entityIt = entitiesForCell.begin(); entityIt != entitiesForCell.end(); entityIt++)
    {
      unsigned entityIndex = *entityIt;

      pair<IndexType,unsigned> constrainingEntity = mesh->getConstrainingEntity(entityDim, entityIndex);

      unsigned constrainingEntityIndex = constrainingEntity.first;
      unsigned constrainingEntityDim = constrainingEntity.second;
      if ((constrainingEntityIndex==entityIndex) && (constrainingEntityDim == entityDim))
      {
        // then we should expect not to have an entry in expectedConstraints:
        if (expectedConstraints.find(entityIndex) != expectedConstraints.end())
        {
          cout << "Expected entity constraint is not imposed in " << meshName << ".\n";
          cout << "Expected " << CamelliaCellTools::entityTypeString(entityDim) << " " << entityIndex << " to be constrained by ";
          cout << CamelliaCellTools::entityTypeString(expectedConstraints[entityIndex].second) << " " << expectedConstraints[entityIndex].first << endl;
          cout << CamelliaCellTools::entityTypeString(entityDim) << " " << entityIndex << " vertices:\n";
          mesh->printEntityVertices(entityDim, entityIndex);
          cout << CamelliaCellTools::entityTypeString(expectedConstraints[entityIndex].second) << " " << expectedConstraints[entityIndex].first << " vertices:\n";
          mesh->printEntityVertices(entityDim, expectedConstraints[entityIndex].first);
          success = false;
        }
      }
      else
      {
        if (expectedConstraints.find(entityIndex) == expectedConstraints.end())
        {
          cout << "Unexpected entity constraint is imposed in " << meshName << ".\n";

          string entityType;
          if (entityDim==0)
          {
            entityType = "Vertex ";
          }
          else if (entityDim==1)
          {
            entityType = "Edge ";
          }
          else if (entityDim==2)
          {
            entityType = "Face ";
          }
          else if (entityDim==3)
          {
            entityType = "Volume ";
          }
          string constrainingEntityType;
          if (constrainingEntityDim==0)
          {
            constrainingEntityType = "Vertex ";
          }
          else if (constrainingEntityDim==1)
          {
            constrainingEntityType = "Edge ";
          }
          else if (constrainingEntityDim==2)
          {
            constrainingEntityType = "Face ";
          }
          else if (constrainingEntityDim==3)
          {
            constrainingEntityType = "Volume ";
          }

          cout << entityType << entityIndex << " unexpectedly constrained by " << constrainingEntityType << constrainingEntityIndex << endl;
          cout << entityType << entityIndex << " vertices:\n";
          mesh->printEntityVertices(entityDim, entityIndex);
          cout << constrainingEntityType << constrainingEntityIndex << " vertices:\n";
          mesh->printEntityVertices(constrainingEntityDim, constrainingEntityIndex);
          success = false;
        }
        else
        {
          unsigned expectedConstrainingEntity = expectedConstraints[entityIndex].first;
          if (expectedConstrainingEntity != constrainingEntityIndex)
          {
            cout << "The constraining entity is not the expected one in " << meshName << ".\n";
            cout << "Expected " << CamelliaCellTools::entityTypeString(entityDim) << " " << entityIndex << " to be constrained by ";
            cout << CamelliaCellTools::entityTypeString(expectedConstraints[entityIndex].second) << " " << expectedConstrainingEntity;
            cout << "; was constrained by " << constrainingEntityIndex << endl;
            cout << CamelliaCellTools::entityTypeString(entityDim) << " " << entityIndex << " vertices:\n";
            mesh->printEntityVertices(entityDim, entityIndex);
            cout << CamelliaCellTools::entityTypeString(expectedConstraints[entityIndex].second) << " " << expectedConstrainingEntity << " vertices:\n";
            mesh->printEntityVertices(entityDim, expectedConstrainingEntity);
            cout << CamelliaCellTools::entityTypeString(constrainingEntityDim) << " " << constrainingEntityIndex << " vertices:\n";
            mesh->printEntityVertices(constrainingEntityDim, constrainingEntityIndex);
            success = false;
          }
        }
      }
    }
  }
}
  
  void testNeighbors(MeshTopologyPtr mesh, Teuchos::FancyOStream &out, bool &success)
  {
    // Worth noting: while the assertions this makes are necessary, they aren't sufficient for correctness of neighbor relationships.
    set<IndexType> activeCellIndices = mesh->getActiveCellIndices();
    for (IndexType activeCellIndex : activeCellIndices)
    {
      CellPtr cell = mesh->getCell(activeCellIndex);
      for (int sideOrdinal = 0; sideOrdinal < cell->getSideCount(); sideOrdinal++)
      {
        pair<GlobalIndexType,unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal, mesh);
        if (neighborInfo.first != -1)
        {
          CellPtr neighbor = mesh->getCell(neighborInfo.first);
          int sideOrdinalInNeighbor = neighborInfo.second;
          bool activeCellsOnly = true;
          vector< pair< GlobalIndexType, unsigned> > neighborDescendants = neighbor->getDescendantsForSide(sideOrdinalInNeighbor, mesh, activeCellsOnly);
          
          for (pair<GlobalIndexType,unsigned> descendantInfo : neighborDescendants)
          {
            // neighbor's descendant's neighbor on the side should be this cell, or this cell's ancestor:
            if (neighborDescendants.size() == 1)
            {
              pair<GlobalIndexType,unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(sideOrdinalInNeighbor, mesh);              
              IndexType cellAncestorIndex = activeCellIndex;
              while ((cellAncestorIndex != -1) && (cellAncestorIndex != neighborNeighborInfo.first))
              {
                CellPtr ancestor = mesh->getCell(cellAncestorIndex);
                if (ancestor->getParent() != Teuchos::null)
                {
                  cellAncestorIndex = ancestor->getParent()->cellIndex();
                }
                else
                {
                  cellAncestorIndex = -1;
                }
              }
              TEST_EQUALITY(cellAncestorIndex, neighborNeighborInfo.first);
            }
            else
            {
              // neighbor's descendant's neighbor on the side should be this cell
              CellPtr descendant = mesh->getCell(descendantInfo.first);
              pair<GlobalIndexType,unsigned> descendantNeighborInfo = descendant->getNeighborInfo(descendantInfo.second, mesh);
              TEST_EQUALITY(activeCellIndex, descendantNeighborInfo.first);
            }
          }
        }
        else // if no neighbor, then this should be a boundary side
        {
          int sideDim = mesh->getDimension() - 1;
          IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
          TEUCHOS_ASSERT(mesh->isBoundarySide(sideEntityIndex));
        }
      }
    }
  }

TEUCHOS_UNIT_TEST( MeshTopology, InitialMeshEntitiesActiveCellCount)
{
  // one easy way to create a quad mesh topology is to use MeshFactory

  int spaceDim = 2;
  bool conformingTraces = false;
  PoissonFormulation formulation(spaceDim, conformingTraces);
  BFPtr bf = formulation.bf();

  int H1Order = 1;

  int horizontalElements = 1;
  int verticalElements = 1;
  double width = 1.0, height = 1.0;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, width, height, horizontalElements, verticalElements); // creates a 1-cell mesh
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());

  unsigned sideDim = meshTopo->getDimension() - 1;

  // uniform refinement --> none of the parent cells sides should have any active cells
  CellPtr cell = meshTopo->getCell(0);
  int sideCount = cell->getSideCount();
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);

    vector< pair<unsigned,unsigned> > activeCellInfoForSide = meshTopo->getActiveCellIndices(sideDim, sideEntityIndex);

    TEST_EQUALITY(activeCellInfoForSide.size(), 1);
    TEST_EQUALITY(meshTopo->getActiveCellCount(sideDim, sideEntityIndex), activeCellInfoForSide.size());
  }

  int vertexDim = 0;
  int vertexCount = cell->topology()->getNodeCount();
  for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++)
  {
    IndexType vertexEntityIndex = cell->entityIndex(vertexDim, vertexOrdinal);

    vector< pair<unsigned,unsigned> > activeCellInfoForVertex = meshTopo->getActiveCellIndices(vertexDim, vertexEntityIndex);

    TEST_EQUALITY(activeCellInfoForVertex.size(), 1);
    TEST_EQUALITY(meshTopo->getActiveCellCount(vertexDim, vertexEntityIndex), activeCellInfoForVertex.size());
  }

  // repeat, but now with a 2x1 mesh:
  horizontalElements = 2;
  verticalElements = 1;
  mesh = MeshFactory::quadMesh(bf, H1Order, width, height, horizontalElements, verticalElements); // creates a 1-cell mesh
  meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());

  for (int d=0; d<=sideDim; d++)
  {
    int entityCount = meshTopo->getEntityCount(d);
    for (IndexType entityIndex=0; entityIndex < entityCount; entityIndex++)
    {
      vector< pair<unsigned,unsigned> > activeCellInfoForEntity = meshTopo->getActiveCellIndices(d, entityIndex);
      TEST_ASSERT((activeCellInfoForEntity.size() == 1) || (activeCellInfoForEntity.size() == 2));
      TEST_EQUALITY(meshTopo->getActiveCellCount(d, entityIndex), activeCellInfoForEntity.size());

      if (d==sideDim)
      {
        vector< IndexType > sides = meshTopo->getSidesContainingEntity(sideDim, entityIndex);
        TEST_EQUALITY(sides.size(), 1);
        TEST_EQUALITY(sides[0], entityIndex);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( MeshTopology, DeactivateCellOnRefinement)
{
  // one easy way to create a quad mesh topology is to use MeshFactory

  int spaceDim = 2;
  bool conformingTraces = false;
  PoissonFormulation formulation(spaceDim, conformingTraces);
  BFPtr bf = formulation.bf();

  int H1Order = 1;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order);
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());

  set<GlobalIndexType> cellsToRefine;
  cellsToRefine.insert(0);

  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());

  unsigned sideDim = meshTopo->getDimension() - 1;

  // uniform refinement --> none of the parent cells sides should have any active cells
  CellPtr cell = meshTopo->getCell(0);
  int sideCount = cell->getSideCount();
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);

    vector< pair<unsigned,unsigned> > activeCellInfoForSide = meshTopo->getActiveCellIndices(sideDim, sideEntityIndex);

    TEST_EQUALITY(activeCellInfoForSide.size(), 0);
  }
}

TEUCHOS_UNIT_TEST( MeshTopology, ConstrainingSideAncestryUniformMesh)
{
  // one easy way to create a quad mesh topology is to use MeshFactory

  int spaceDim = 2;
  bool conformingTraces = false;
  PoissonFormulation formulation(spaceDim, conformingTraces);
  BFPtr bf = formulation.bf();

  int H1Order = 1;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order);
  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());

  set<GlobalIndexType> cellsToRefine;
  cellsToRefine.insert(0);

  mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());

  // everything should be compatible at this point

  set<GlobalIndexType> activeCellIDs = mesh->cellIDsInPartition();

  unsigned sideDim = meshTopo->getDimension() - 1;

  for (set<GlobalIndexType>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++)
  {
    CellPtr cell = meshTopo->getCell(*cellIDIt);
    int sideCount = cell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
    {
      IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);

      vector< pair<unsigned,unsigned> > constrainingSideAncestry = meshTopo->getConstrainingSideAncestry(sideEntityIndex);

      TEST_EQUALITY(constrainingSideAncestry.size(), 0);
    }
  }
}

TEUCHOS_UNIT_TEST(MeshTopology, GetEntityGeneralizedParent_LineRefinement)
{
  int spaceDim = 1;

  CellTopoPtr lineTopo = CellTopology::line();

  vector< vector<double> > lineVertices;
  double xLeft = 0.0;
  double xRight = 1.0;
  lineVertices.push_back(vector<double>(1, xLeft));
  lineVertices.push_back(vector<double>(1, xRight));

  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
  
  GlobalIndexType cellID = 0;
  meshTopo->addCell(cellID, lineTopo, lineVertices); // cell from 0 to 1

  CellPtr cell = meshTopo->getCell(cellID);

  RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(lineTopo);

  GlobalIndexType nextCellID = cellID + 1;
  meshTopo->refineCell(cellID, refPattern, nextCellID);
  
  nextCellID += refPattern->numChildren();

  double xMiddle = (xLeft + xRight) / 2.0;

  unsigned vertexOrdinal;
  meshTopo->getVertexIndex(vector<double>(1,xMiddle), vertexOrdinal);

  int vertexDim = 0, lineDim = 1;
  pair<IndexType, unsigned> generalizedParent = meshTopo->getEntityGeneralizedParent(vertexDim, vertexOrdinal);

  TEST_EQUALITY(generalizedParent.first, cellID);
  TEST_EQUALITY(generalizedParent.second, lineDim);

  // try same for a child *cell*
  CellPtr childCell = meshTopo->getCell(cellID)->children()[0];
  generalizedParent = meshTopo->getEntityGeneralizedParent(spaceDim, childCell->cellIndex());

  TEST_EQUALITY(generalizedParent.first, cellID);
  TEST_EQUALITY(generalizedParent.second, lineDim);
}

TEUCHOS_UNIT_TEST(MeshTopology, GetRootMeshTopology)
{
  int k = 1;
  int H1Order = k + 1;
  int delta_k = 1;

  int spaceDim = 2;
  bool conformingTraces = false;
  PoissonFormulation formulation(spaceDim, conformingTraces);
  BFPtr bf = formulation.bf();

  BFPtr bilinearForm = bf;

  int rootMeshNumCells = 2;
  double width = 1.0;

  vector<double> dimensions;
  vector<int> elementCounts;
  vector<double> x0(spaceDim,0.0);
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(width);
    elementCounts.push_back(rootMeshNumCells);
  }
  MeshPtr originalMesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0);
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0);

  // get a sample cellTopo:
  CellTopoPtr cellTopo = mesh->getTopology()->getCell(0)->topology();
  RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);

  int numUniformRefinements = 3;
  for (int i=0; i<numUniformRefinements; i++)
  {
    set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
    mesh->RefinementObserver::hRefine(activeCellIDs, refPattern);
  }

  MeshTopology* meshTopo = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
  
  MeshTopologyPtr rootMeshTopology = meshTopo->getRootMeshTopology();
  MeshTopologyViewPtr originalMeshTopology = originalMesh->getTopology();

  TEST_EQUALITY(rootMeshTopology->cellCount(), originalMeshTopology->cellCount());

  std::set<IndexType> rootCellIndices = rootMeshTopology->getActiveCellIndices();
  std::set<IndexType> originalCellIndices = originalMeshTopology->getActiveCellIndices();

  // compare sets:
  std::set<IndexType>::iterator rootCellIt = rootCellIndices.begin(), originalCellIt = originalCellIndices.begin();
  for (int i=0; i<rootMeshTopology->cellCount(); i++)
  {
    TEST_EQUALITY(*rootCellIt, *originalCellIt);
    rootCellIt++;
    originalCellIt++;
  }

  for (std::set<IndexType>::iterator cellIDIt = rootCellIndices.begin(); cellIDIt != rootCellIndices.end(); cellIDIt++)
  {
    CellPtr originalCell = originalMeshTopology->getCell(*cellIDIt);
    CellPtr rootCell = rootMeshTopology->getCell(*cellIDIt);

    vector<unsigned> originalVertexIndices = originalCell->vertices();
    vector<unsigned> rootVertexIndices = rootCell->vertices();

    TEST_EQUALITY(originalVertexIndices.size(), rootVertexIndices.size());

    TEST_COMPARE_ARRAYS(originalVertexIndices, rootVertexIndices);

    for (int i=0; i<originalVertexIndices.size(); i++)
    {
      vector<double> originalVertex = originalMeshTopology->getVertex(originalVertexIndices[i]);
      vector<double> rootVertex = rootMeshTopology->getVertex(rootVertexIndices[i]);

      TEST_COMPARE_FLOATING_ARRAYS(originalVertex, rootVertex, 1e-15);
    }
  }
}
  
  TEUCHOS_UNIT_TEST( MeshTopology, UpdateNeighborsAfterAnisotropicRefinements)
  {
    // set up MeshGeometry sorta like what we have in Hemker meshes:

    bool testSpaceTime = true;
    
    // cell 0:
    vector<double> A = {-60,-60}, B = {-0.5,-60}, C = {-0.5,-0.5}, D = {-60,-0.5};
    vector<vector<double>> vertices = {A, B, C, D};
    vector<vector<IndexType>> elementVertices = {{0,1,2,3}};
    // cell 1:
    vector<double> E = {0.5,-60}, F = {0.5,-0.5};
    vertices.push_back(E);
    vertices.push_back(F);
    elementVertices.push_back({1,4,5,2});
    // cell 2:
    vector<double> G = {-0.5,0.5}, H = {-60,0.5};
    vertices.push_back(G);
    vertices.push_back(H);
    elementVertices.push_back({3,2,6,7});
    vector<CellTopoPtr> cellTopos(3,CellTopology::quad());
    MeshGeometryPtr geometry = Teuchos::rcp(new MeshGeometry(vertices,elementVertices,cellTopos));
    
    // create meshTopology:
    MeshTopologyPtr meshTopo = Teuchos::rcp(new MeshTopology(geometry));

    RefinementPatternPtr verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuad();
    RefinementPatternPtr horizontalCut = RefinementPattern::yAnisotropicRefinementPatternQuad();
    
    if (testSpaceTime)
    {
      double t0 = 0.0, t1 = 12.0;
      meshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1);
      verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuadTimeExtruded();
      horizontalCut = RefinementPattern::yAnisotropicRefinementPatternQuadTimeExtruded();
    }
    
    set<GlobalIndexType> cellsToCutVertically = {1};
    set<GlobalIndexType> cellsToCutHorizontally = {2};
    
    GlobalIndexType nextCellID = meshTopo->cellCount();
    // cut 5 times
    int cutCycles = 5;
    for (int i=0; i<cutCycles; i++)
    {
      set<GlobalIndexType> newCells;
      for (GlobalIndexType cellIndex : cellsToCutVertically)
      {
        meshTopo->refineCell(cellIndex, verticalCut, nextCellID);
        nextCellID += verticalCut->numChildren();
        
        CellPtr cell = meshTopo->getCell(cellIndex);
        vector<IndexType> childCellIndices = cell->getChildIndices(meshTopo);
        newCells.insert(childCellIndices.begin(),childCellIndices.end());
      }
      cellsToCutVertically = newCells;
    }
    
    for (int i=0; i<cutCycles; i++)
    {
      set<GlobalIndexType> newCells;
      for (GlobalIndexType cellIndex : cellsToCutHorizontally)
      {
        meshTopo->refineCell(cellIndex, horizontalCut, nextCellID);
        nextCellID += horizontalCut->numChildren();
        
        CellPtr cell = meshTopo->getCell(cellIndex);
        vector<IndexType> childCellIndices = cell->getChildIndices(meshTopo);
        newCells.insert(childCellIndices.begin(),childCellIndices.end());
      }
      cellsToCutHorizontally = newCells;
    }
    
    int spaceDim = 2;
    bool conformingTraces = false;
    PoissonFormulation formulation(spaceDim, conformingTraces);
    BFPtr bf = formulation.bf();
    
    int H1Order = 1, delta_k = 1;
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, bf, H1Order, delta_k) );
                                
    mesh->enforceOneIrregularity();
    
    testNeighbors(meshTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopology, UpdateNeighborsAfterTwoIrregularMeshRefined )
  {
    int irregularity = 2;
    MeshTopologyPtr meshTopo = constructIrregularMeshTopology(irregularity);
    IndexType irregularCellIndex = -1;
    set<IndexType> activeCells = meshTopo->getActiveCellIndices();
    IndexType activeCellWithLargeNeighbor;
    IndexType activeCellWithLargeNeighborSideOrdinal; // the side shared with the irregular cell
    for (IndexType activeCellIndex : activeCells)
    {
      CellPtr activeCell = meshTopo->getCell(activeCellIndex);
      for (int sideOrdinal=0; sideOrdinal<activeCell->getSideCount(); sideOrdinal++)
      {
        RefinementBranch sideRefBranch = activeCell->refinementBranchForSide(sideOrdinal, meshTopo);
        if (sideRefBranch.size() == irregularity)
        {
          irregularCellIndex = activeCell->getNeighborInfo(sideOrdinal, meshTopo).first;
          activeCellWithLargeNeighbor = activeCellIndex;
          activeCellWithLargeNeighborSideOrdinal = sideOrdinal;
          break;
        }
      }
      if (irregularCellIndex != -1) break;
    }
    TEST_ASSERT(irregularCellIndex != -1);
    
    CellPtr smallCell = meshTopo->getCell(activeCellWithLargeNeighbor);
    // before refinement, small cell should have irregular cell index as its neighbor on the side:
    TEST_EQUALITY(smallCell->getNeighborInfo(activeCellWithLargeNeighborSideOrdinal, meshTopo).first,
                  irregularCellIndex);
    
    // test neighbors before and after refinement
    testNeighbors(meshTopo, out, success);
    
    GlobalIndexType nextCellID = meshTopo->cellCount();
    RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(meshTopo->getCell(irregularCellIndex)->topology());
    meshTopo->refineCell(irregularCellIndex, refPattern, nextCellID);
    nextCellID += refPattern->numChildren();

    // after refinement, small cell should no longer have irregular cell index as its neighbor on the side:
    GlobalIndexType smallCellNeighborCellIndex = smallCell->getNeighborInfo(activeCellWithLargeNeighborSideOrdinal, meshTopo).first;
    TEST_INEQUALITY(smallCellNeighborCellIndex, irregularCellIndex);
    // instead, the neighbor should be one of the irregular cell's children
    CellPtr smallCellNeighbor = meshTopo->getCell(smallCell->getNeighborInfo(activeCellWithLargeNeighborSideOrdinal, meshTopo).first);
    TEST_EQUALITY(smallCellNeighbor->getParent()->cellIndex(), irregularCellIndex);
    
    testNeighbors(meshTopo, out, success);
  }

TEUCHOS_UNIT_TEST( MeshTopology, UnrefinedSpaceTimeMeshTopologyIsUnconstrained )
{
  // to start with, just a single-cell space-time MeshTopology
  CellTopoPtr spaceTopo = CellTopology::line();
  int spaceDim = spaceTopo->getDimension();
  MeshTopologyPtr spatialMeshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );

  FieldContainer<double> cellNodes(spaceTopo->getNodeCount(),spaceTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(cellNodes, spaceTopo);
  vector<vector<double>> cellVerticesVector;
  CamelliaCellTools::pointsVectorFromFC(cellVerticesVector, cellNodes);
  
  GlobalIndexType nextCellID = spatialMeshTopo->cellCount();
  spatialMeshTopo->addCell(nextCellID, spaceTopo, cellVerticesVector);

  double t0 = 0.0, t1 = 1.0;
  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

  for (int d=0; d<spaceTimeMeshTopo->getDimension(); d++)
  {
    map<unsigned,pair<IndexType,unsigned> > expectedConstraints;
    testConstraints(spaceTimeMeshTopo.get(), d, expectedConstraints, out, success);
  }
}
} // namespace
