//
//  MeshTopologyTests
//  Camellia
//
//  Created by Nate Roberts on 12/8/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "MeshTopology.h"
#include "PoissonFormulation.h"

#include "MeshFactory.h"

namespace {
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
    MeshTopologyPtr meshTopo = mesh->getTopology();
  
    unsigned sideDim = meshTopo->getSpaceDim() - 1;
    
    // uniform refinement --> none of the parent cells sides should have any active cells
    CellPtr cell = meshTopo->getCell(0);
    int sideCount = cell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
      
      vector< pair<unsigned,unsigned> > activeCellInfoForSide = meshTopo->getActiveCellIndices(sideDim, sideEntityIndex);
      
      TEST_EQUALITY(activeCellInfoForSide.size(), 1);
      TEST_EQUALITY(meshTopo->getActiveCellCount(sideDim, sideEntityIndex), activeCellInfoForSide.size());
    }
    
    int vertexDim = 0;
    int vertexCount = cell->topology()->getNodeCount();
    for (int vertexOrdinal=0; vertexOrdinal<vertexCount; vertexOrdinal++) {
      IndexType vertexEntityIndex = cell->entityIndex(vertexDim, vertexOrdinal);
      
      vector< pair<unsigned,unsigned> > activeCellInfoForVertex = meshTopo->getActiveCellIndices(vertexDim, vertexEntityIndex);
      
      TEST_EQUALITY(activeCellInfoForVertex.size(), 1);
      TEST_EQUALITY(meshTopo->getActiveCellCount(vertexDim, vertexEntityIndex), activeCellInfoForVertex.size());
    }
    
    // repeat, but now with a 2x1 mesh:
    horizontalElements = 2;
    verticalElements = 1;
    mesh = MeshFactory::quadMesh(bf, H1Order, width, height, horizontalElements, verticalElements); // creates a 1-cell mesh
    meshTopo = mesh->getTopology();
    
    for (int d=0; d<=sideDim; d++) {
      int entityCount = meshTopo->getEntityCount(d);
      for (IndexType entityIndex=0; entityIndex < entityCount; entityIndex++) {
        vector< pair<unsigned,unsigned> > activeCellInfoForEntity = meshTopo->getActiveCellIndices(d, entityIndex);
        TEST_ASSERT((activeCellInfoForEntity.size() == 1) || (activeCellInfoForEntity.size() == 2));
        TEST_EQUALITY(meshTopo->getActiveCellCount(d, entityIndex), activeCellInfoForEntity.size());
        
        if (d==sideDim) {
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
    MeshTopologyPtr meshTopo = mesh->getTopology();
    
    set<GlobalIndexType> cellsToRefine;
    cellsToRefine.insert(0);
    
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    unsigned sideDim = meshTopo->getSpaceDim() - 1;

    // uniform refinement --> none of the parent cells sides should have any active cells
    CellPtr cell = meshTopo->getCell(0);
    int sideCount = cell->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
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
    MeshTopologyPtr meshTopo = mesh->getTopology();
    
    set<GlobalIndexType> cellsToRefine;
    cellsToRefine.insert(0);
    
    mesh->hRefine(cellsToRefine, RefinementPattern::regularRefinementPatternQuad());
    
    // everything should be compatible at this point
    
    set<GlobalIndexType> activeCellIDs = mesh->cellIDsInPartition();
    
    unsigned sideDim = meshTopo->getSpaceDim() - 1;
    
    for (set<GlobalIndexType>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
      CellPtr cell = meshTopo->getCell(*cellIDIt);
      int sideCount = cell->getSideCount();
      for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
        IndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
        
        vector< pair<unsigned,unsigned> > constrainingSideAncestry = meshTopo->getConstrainingSideAncestry(sideEntityIndex);
        
        TEST_EQUALITY(constrainingSideAncestry.size(), 0);
      }
    }
  }
} // namespace