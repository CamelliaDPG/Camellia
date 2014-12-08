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