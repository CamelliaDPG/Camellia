//
//  MeshTopologyViewTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 6/25/15
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "MeshFactory.h"
#include "MeshTopologyView.h"
#include "PoissonFormulation.h"

using namespace Camellia;
using namespace std;

namespace
{
  MeshPtr poissonUniformMesh(vector<int> elementWidths, int H1Order, bool useConformingTraces)
  {
    int spaceDim = elementWidths.size();
    int testSpaceEnrichment = spaceDim; //
    double span = 1.0; // in each spatial dimension
    
    vector<double> dimensions(spaceDim,span);
    
    PoissonFormulation poissonForm(spaceDim, useConformingTraces);
    MeshPtr mesh = MeshFactory::rectilinearMesh(poissonForm.bf(), dimensions, elementWidths, H1Order, testSpaceEnrichment);
    return mesh;
  }
  
  MeshPtr poissonUniformMesh(int spaceDim, int elementWidth, int H1Order, bool useConformingTraces)
  {
    vector<int> elementCounts(spaceDim,elementWidth);
    return poissonUniformMesh(elementCounts, H1Order, useConformingTraces);
  }
  
  MeshPtr poissonIrregularMesh(int spaceDim, int irregularity, int H1Order)
  {
    bool useConformingTraces = true;
    
    int elementWidth = 2;
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    
    int meshIrregularity = 0;
    vector<GlobalIndexType> cellsToRefine = {1};
    CellPtr cellToRefine = mesh->getTopology()->getCell(cellsToRefine[0]);
    unsigned sharedSideOrdinal = -1;
    for (int sideOrdinal=0; sideOrdinal<cellToRefine->getSideCount(); sideOrdinal++)
    {
      if (cellToRefine->getNeighbor(sideOrdinal, mesh->getTopology()) != Teuchos::null)
      {
        sharedSideOrdinal = sideOrdinal;
        break;
      }
    }
    
    while (meshIrregularity < irregularity)
    {
      //      print("refining cells", cellsToRefine);
      mesh->hRefine(cellsToRefine);
      meshIrregularity++;
      
      // setup for the next refinement, if any:
      auto childEntry = cellToRefine->childrenForSide(sharedSideOrdinal)[0];
      GlobalIndexType childWithNeighborCellID = childEntry.first;
      sharedSideOrdinal = childEntry.second;
      cellsToRefine = {childWithNeighborCellID};
      cellToRefine = mesh->getTopology()->getCell(cellsToRefine[0]);
    }
    return mesh;
  }
  
  void testMeshTopologyAgreesWithView(MeshTopologyPtr topo, MeshTopologyViewPtr view,
                                      Teuchos::FancyOStream &out, bool &success)
  {
    TEST_EQUALITY(topo->cellCount(), view->cellCount());
    
    set<IndexType> topoIndicesSet = topo->getActiveCellIndices();
    vector<IndexType> topoIndicesVector(topoIndicesSet.begin(),topoIndicesSet.end());
    set<IndexType> viewIndicesSet = view->getActiveCellIndices();
    vector<IndexType> viewIndicesVector(viewIndicesSet.begin(),viewIndicesSet.end());
    TEST_COMPARE_ARRAYS(topoIndicesVector, viewIndicesVector);
    
    set<IndexType> topoRootCellsSet = topo->getRootCellIndices();
    vector<IndexType> topoRootCells(topoRootCellsSet.begin(),topoRootCellsSet.end());
    set<IndexType> viewRootCellsSet = view->getRootCellIndices();
    vector<IndexType> viewRootCells(viewRootCellsSet.begin(),viewRootCellsSet.end());
    TEST_COMPARE_ARRAYS(topoRootCells, viewRootCells);
    
    TEST_EQUALITY(topo->getDimension(), view->getDimension());
    int dim = topo->getDimension();
    int sideDim = dim - 1;
    
    // entity methods:
    for (int d=0; d<dim; d++)
    {
      IndexType entityCount = topo->getEntityCount(d);
      for (IndexType entityIndex = 0; entityIndex < entityCount; entityIndex++)
      {
        vector<IndexType> topoSidesContaining = topo->getSidesContainingEntity(d, entityIndex);
        vector<IndexType> viewSidesContaining = view->getSidesContainingEntity(d, entityIndex);
        TEST_COMPARE_ARRAYS(topoSidesContaining, viewSidesContaining);
        
        vector< pair<IndexType,unsigned> > topoActiveCells = topo->getActiveCellIndices(d, entityIndex);
        vector< pair<IndexType,unsigned> > viewActiveCells = view->getActiveCellIndices(d, entityIndex);
        TEST_EQUALITY(topoActiveCells.size(), viewActiveCells.size());
        if (topoActiveCells.size() == viewActiveCells.size())
        {
          for (int i=0; i<topoActiveCells.size(); i++)
          {
            TEST_EQUALITY(topoActiveCells[i], viewActiveCells[i]);
          }
        }
        
        set<pair<IndexType,unsigned>> topoCellsForEntitySet = topo->getCellsContainingEntity(d, entityIndex);
        vector<pair<IndexType,unsigned>> topoCellsForEntity(topoCellsForEntitySet.begin(),topoCellsForEntitySet.end());
        set<pair<IndexType,unsigned>> viewCellsForEntitySet = view->getCellsContainingEntity(d, entityIndex);
        vector<pair<IndexType,unsigned>> viewCellsForEntity(viewCellsForEntitySet.begin(),viewCellsForEntitySet.end());
        TEST_EQUALITY(topoCellsForEntity.size(), viewCellsForEntity.size());
        if (topoCellsForEntity.size() == viewCellsForEntity.size())
        {
          for (int i=0; i<topoCellsForEntity.size(); i++)
          {
            TEST_EQUALITY(topoCellsForEntity[i], viewCellsForEntity[i]);
          }
        }
        
        TEST_EQUALITY(topoCellsForEntity.size(), viewCellsForEntity.size());
        if (topoCellsForEntity.size() == viewCellsForEntity.size())
        {
          for (int i=0; i<topoCellsForEntity.size(); i++)
          {
            TEST_EQUALITY(topoCellsForEntity[i], viewCellsForEntity[i]);
          }
        }
        else
        {
          out << "topoCellsForEntity.size() != viewCellsForEntity.size().\n";
        }

        
        pair<IndexType,unsigned> topoConstrainingEntity = topo->getConstrainingEntity(d, entityIndex);
        pair<IndexType,unsigned> viewConstrainingEntity = view->getConstrainingEntity(d, entityIndex);
        TEST_EQUALITY(topoConstrainingEntity, viewConstrainingEntity);
        
        if (topoConstrainingEntity.second == d)
        {
          TEST_ASSERT(topo->entityIsAncestor(d, topoConstrainingEntity.first, entityIndex));
          TEST_ASSERT(view->entityIsAncestor(d, viewConstrainingEntity.first, entityIndex));
          
          if (entityIndex != topoConstrainingEntity.first)
          {
            TEST_ASSERT(! topo->entityIsAncestor(d, entityIndex, topoConstrainingEntity.first));
            TEST_ASSERT(! view->entityIsAncestor(d, entityIndex, viewConstrainingEntity.first));
          }
        }
        
        pair<IndexType,IndexType> topoOwner = topo->owningCellIndexForConstrainingEntity(topoConstrainingEntity.second, topoConstrainingEntity.first);
        pair<IndexType,IndexType> viewOwner = view->owningCellIndexForConstrainingEntity(viewConstrainingEntity.second, viewConstrainingEntity.first);
        TEST_EQUALITY(topoOwner, viewOwner);
        
        IndexType topoConstrainingEntityLikeDimension = topo->getConstrainingEntityIndexOfLikeDimension(d, entityIndex);
        IndexType viewConstrainingEntityLikeDimension = view->getConstrainingEntityIndexOfLikeDimension(d, entityIndex);
        TEST_EQUALITY(topoConstrainingEntityLikeDimension, viewConstrainingEntityLikeDimension);
        
        vector<IndexType> topoVertexIndices = topo->getEntityVertexIndices(d, entityIndex);
        vector<IndexType> viewVertexIndices = view->getEntityVertexIndices(d, entityIndex);
        TEST_COMPARE_ARRAYS(topoVertexIndices, viewVertexIndices);
      }
    }
    
    // vertex methods:
    unsigned vertexDim = 0;
    IndexType vertexCount = topo->getEntityCount(vertexDim);
    for (IndexType vertexIndex=0; vertexIndex<vertexCount; vertexIndex++)
    {
      vector<double> topoVertex = topo->getVertex(vertexIndex);
      vector<double> viewVertex = view->getVertex(vertexIndex);
      TEST_COMPARE_ARRAYS(topoVertex, viewVertex); // exact comparison; could relax to floating comparison
      
      IndexType topoLookupVertexIndex = -1;
      bool topoLookupVertexSuccess = topo->getVertexIndex(topoVertex, topoLookupVertexIndex);
      TEST_ASSERT(topoLookupVertexSuccess);
      TEST_EQUALITY(topoLookupVertexIndex, vertexIndex);
      IndexType viewLookupVertexIndex = -1;
      bool viewLookupVertexSuccess = view->getVertexIndex(viewVertex, viewLookupVertexIndex);
      TEST_ASSERT(viewLookupVertexSuccess);
      TEST_EQUALITY(viewLookupVertexIndex, vertexIndex);
    }
    
    // side methods:
    IndexType sideCount = topo->getEntityCount(sideDim);
    for (IndexType sideEntityIndex=0; sideEntityIndex<sideCount; sideEntityIndex++)
    {
      vector<IndexType> topoCellsForSide = topo->getCellsForSide(sideEntityIndex);
      vector<IndexType> viewCellsForSide = view->getCellsForSide(sideEntityIndex);
      TEST_COMPARE_ARRAYS(topoCellsForSide, viewCellsForSide);
      
      vector<pair<IndexType,unsigned>> topoConstrainingSideAncestry = topo->getConstrainingSideAncestry(sideEntityIndex);
      vector<pair<IndexType,unsigned>> viewConstrainingSideAncestry = view->getConstrainingSideAncestry(sideEntityIndex);
      TEST_EQUALITY(topoConstrainingSideAncestry.size(), viewConstrainingSideAncestry.size());
      if (topoConstrainingSideAncestry.size() == viewConstrainingSideAncestry.size())
      {
        for (int i=0; i<topoConstrainingSideAncestry.size(); i++)
        {
          TEST_EQUALITY(topoConstrainingSideAncestry[i], viewConstrainingSideAncestry[i]);
        }
      }
    }
    
    // cell methods:
    for (IndexType cellIndex=0; cellIndex<topo->cellCount(); cellIndex++)
    {
      TEST_ASSERT(topo->isValidCellIndex(cellIndex));
      TEST_ASSERT(view->isValidCellIndex(cellIndex));
      
      TEST_EQUALITY(topo->isParent(cellIndex), view->isParent(cellIndex));
      
      Intrepid::FieldContainer<double> topoPhysicalNodes = topo->physicalCellNodesForCell(cellIndex);
      Intrepid::FieldContainer<double> viewPhysicalNodes = view->physicalCellNodesForCell(cellIndex);
      TEST_COMPARE_ARRAYS(topoPhysicalNodes, viewPhysicalNodes); // could relax to floating equality
      
      topo->verticesForCell(topoPhysicalNodes, cellIndex);
      TEST_COMPARE_ARRAYS(topoPhysicalNodes, viewPhysicalNodes); // could relax to floating equality
      view->verticesForCell(viewPhysicalNodes, cellIndex);
      TEST_COMPARE_ARRAYS(topoPhysicalNodes, viewPhysicalNodes); // could relax to floating equality
      
      vector<double> topoCentroid = topo->getCellCentroid(cellIndex);
      vector<double> viewCentroid = view->getCellCentroid(cellIndex);
      TEST_COMPARE_ARRAYS(topoCentroid, viewCentroid); // could relax to floating equality
    }
    
    // check that cells outside bounds return false for isValidCellIndex:
    for (IndexType cellIndex=topo->cellCount(); cellIndex < 2 * topo->cellCount(); cellIndex++)
    {
      TEST_ASSERT(!topo->isValidCellIndex(cellIndex));
      TEST_ASSERT(!view->isValidCellIndex(cellIndex));
    }
    
    // TODO: test agreement of the following:
    /*
     vector<IndexType> cellIDsForPoints(const Intrepid::FieldContainer<double> &physicalPoints);
     */
  }
  
  void testDeepCopyAgrees(int spaceDim, int irregularity, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 1; // we're interested in MeshTopology, so this doesn't matter.
    MeshPtr mesh = poissonIrregularMesh(spaceDim, irregularity, H1Order);
    MeshTopologyPtr meshTopoCopy = mesh->getTopology()->deepCopy(); // the object we'll check agreement with
    
    set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
    
    // refine original 2-irregular mesh uniformly:
    mesh->hRefine(activeCellIDs);
    // create a view on its MeshTopology:
    MeshTopology* meshTopoCast = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
    MeshTopologyPtr meshTopo = Teuchos::rcp(meshTopoCast, false);
    MeshTopologyViewPtr coarseMeshView = Teuchos::rcp(new MeshTopologyView(meshTopo, activeCellIDs));
    
    testMeshTopologyAgreesWithView(meshTopoCopy, coarseMeshView, out, success);
  }
  
  void testOneElementMeshIdentityAgrees(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 1; // we're interested in MeshTopology, so this doesn't matter.
    int elementWidth = 1;
    bool useConformingTraces = false; // doesn't matter for this test
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    MeshTopology* meshTopoCast = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
    MeshTopologyPtr meshTopo = Teuchos::rcp(meshTopoCast,false); // the object we'll check agreement with
    
    set<IndexType> activeCellIDs = meshTopo->getActiveCellIndices();
    MeshTopologyViewPtr topoView = Teuchos::rcp(new MeshTopologyView(meshTopo, activeCellIDs));
    
    testMeshTopologyAgreesWithView(meshTopo, topoView, out, success);
  }
  
  void testUniformMeshIdentityAgrees(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 1; // we're interested in MeshTopology, so this doesn't matter.
    int elementWidth = 2;
    bool useConformingTraces = false; // doesn't matter for this test
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    MeshTopology* meshTopoCast = dynamic_cast<MeshTopology*>(mesh->getTopology().get());
    MeshTopologyPtr meshTopo = Teuchos::rcp(meshTopoCast,false); // the object we'll check agreement with
    
    set<IndexType> activeCellIDs = meshTopo->getActiveCellIndices();
    MeshTopologyViewPtr topoView = Teuchos::rcp(new MeshTopologyView(meshTopo, activeCellIDs));
    
    testMeshTopologyAgreesWithView(meshTopo, topoView, out, success);
  }
  
  TEUCHOS_UNIT_TEST(MeshTopologyView, CellsContainingVertex_1D)
  {
    double xLeft=0.0, xRight=1.0;
    int numElements = 2;
    MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(xLeft, xRight, numElements);
    
    IndexType vertexIndex;
    bool vertexFound = meshTopo->getVertexIndex({xLeft}, vertexIndex);
    TEST_ASSERT(vertexFound);
    
    IndexType cellIndex = 0;
    GlobalIndexType nextCellID = meshTopo->cellCount();
    meshTopo->refineCell(cellIndex, RefinementPattern::regularRefinementPatternLine(), nextCellID);
    
    unsigned vertexDim = 0;
    set<pair<IndexType,unsigned>> cellEntriesMesh = meshTopo->getCellsContainingEntity(vertexDim, vertexIndex);
    TEST_ASSERT(cellEntriesMesh.size() == 1);
    
    MeshTopologyViewPtr meshTopoView = meshTopo->getView({0});
    set<pair<IndexType,unsigned>> cellEntriesView = meshTopoView->getCellsContainingEntity(vertexDim, vertexIndex);
    TEST_ASSERT(cellEntriesView.size() == 1);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, NeighborIsCorrectAnisotropicRefinement_OneLevel_2D )
  {
    int spaceDim = 2;
    int elementWidth = 2;
    int H1Order = 2;
    bool useConformingTraces = true;
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    
    GlobalIndexType cellIndex = 0;
    CellPtr cell = mesh->getTopology()->getCell(cellIndex);
    unsigned sideOrdinal = 1;
    
    GlobalIndexType neighborCellIndexToRefine = cell->getNeighbor(sideOrdinal, mesh->getTopology())->cellIndex();
    mesh->hRefine(set<GlobalIndexType>{neighborCellIndexToRefine}, RefinementPattern::xAnisotropicRefinementPatternQuad());
    
    // now, cell 0 has neighbor that is child of its original neighbor
    CellPtr newNeighbor = cell->getNeighbor(sideOrdinal, mesh->getTopology());
    TEST_INEQUALITY(newNeighbor->cellIndex(), neighborCellIndexToRefine);
    
    // but if we use the root mesh topology view, we should recover the original neighbor
    set<IndexType> rootCells = mesh->getTopology()->getRootCellIndices();
    MeshTopologyViewPtr rootMeshTopoView = mesh->getTopology()->getView(rootCells);
    CellPtr oldNeighbor = cell->getNeighbor(sideOrdinal, rootMeshTopoView);
    TEST_EQUALITY(oldNeighbor->cellIndex(), neighborCellIndexToRefine);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, NeighborIsCorrectAnisotropicRefinement_TwoLevel_2D )
  {
    int spaceDim = 2;
    int elementWidth = 2;
    int H1Order = 2;
    bool useConformingTraces = true;
    MeshPtr mesh = poissonUniformMesh(spaceDim, elementWidth, H1Order, useConformingTraces);
    
    GlobalIndexType cellIndex = 0;
    CellPtr cell = mesh->getTopology()->getCell(cellIndex);
    unsigned sideOrdinal = 1;
    
    CellPtr neighbor = cell->getNeighbor(sideOrdinal, mesh->getTopology());
    GlobalIndexType neighborCellIndexToRefine = neighbor->cellIndex();
    mesh->hRefine(set<GlobalIndexType>{neighborCellIndexToRefine}, RefinementPattern::xAnisotropicRefinementPatternQuad());
    
    // now, cell 0 has a new neighbor that is child of its original neighbor
    CellPtr newNeighbor = cell->getNeighbor(sideOrdinal, mesh->getTopology());
    TEST_INEQUALITY(newNeighbor->cellIndex(), neighborCellIndexToRefine);
    TEST_EQUALITY(newNeighbor->getParent()->cellIndex(), neighborCellIndexToRefine);
    
    mesh->hRefine(set<GlobalIndexType>{newNeighbor->cellIndex()}, RefinementPattern::xAnisotropicRefinementPatternQuad());

    // now, cell 0 has another new neighbor that is the grandchild of its original neighbor
    CellPtr newNewNeighbor = cell->getNeighbor(sideOrdinal, mesh->getTopology());
    TEST_INEQUALITY(newNeighbor->cellIndex(), newNewNeighbor->cellIndex());
    TEST_EQUALITY(newNewNeighbor->getParent()->cellIndex(), newNeighbor->cellIndex());
    
    // but if we use the root mesh topology view, we should recover the original neighbor
    set<IndexType> rootCells = mesh->getTopology()->getRootCellIndices();
    MeshTopologyViewPtr rootMeshTopoView = mesh->getTopology()->getView(rootCells);
    CellPtr oldNeighbor = cell->getNeighbor(sideOrdinal, rootMeshTopoView);
    TEST_EQUALITY(oldNeighbor->cellIndex(), neighborCellIndexToRefine);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, OneElementMeshIdentityAgrees_1D )
  {
    int spaceDim = 1;
    testOneElementMeshIdentityAgrees(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, OneElementMeshIdentityAgrees_2D )
  {
    int spaceDim = 2;
    testOneElementMeshIdentityAgrees(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, OneElementMeshIdentityAgrees_3D )
  {
    int spaceDim = 3;
    testOneElementMeshIdentityAgrees(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, UniformMeshIdentityAgrees_1D )
  {
    int spaceDim = 1;
    testUniformMeshIdentityAgrees(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, UniformMeshIdentityAgrees_2D )
  {
    int spaceDim = 2;
    testUniformMeshIdentityAgrees(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, UniformMeshIdentityAgrees_3D )
  {
    int spaceDim = 3;
    testUniformMeshIdentityAgrees(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, DeepCopyAgrees_2D )
  {
    int spaceDim = 2;
    for (int irregularity = 0; irregularity < 3; irregularity++)
    {
      testDeepCopyAgrees(spaceDim, irregularity, out, success);
    }
  }
  
  TEUCHOS_UNIT_TEST( MeshTopologyView, DeepCopyAgrees_3D_Slow )
  {
    int spaceDim = 3;
    for (int irregularity = 0; irregularity < 2; irregularity++)
    {
      testDeepCopyAgrees(spaceDim, irregularity, out, success);
    }
  }
} // namespace