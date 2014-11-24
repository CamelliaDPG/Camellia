//
//  CellTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 11/18/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_UnitTestHelpers.hpp"

#include "Intrepid_FieldContainer.hpp"

#include "MeshFactory.h"
#include "Cell.h"

#include "PoissonFormulation.h"

namespace {
  TEUCHOS_UNIT_TEST( Cell, Neighbors1D )
  {
    int numCells = 8;
    int spaceDim = 1;
    // just want any bilinear form; we'll use Poisson
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    double xLeft = 0, xRight = 1;
    
    int H1Order = 1, delta_k = 1;
    MeshPtr mesh = MeshFactory::intervalMesh(form.bf(), xLeft, xRight, numCells, H1Order, delta_k);
    
    int numBoundarySides = 0;
    
    MeshTopologyPtr meshTopo = mesh->getTopology();
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      CellPtr cell = meshTopo->getCell(cellIndex);
      
      TEST_ASSERT(cell->getSideCount() == 2);
      
      for (int sideOrdinal = 0; sideOrdinal < cell->getSideCount(); sideOrdinal++) {
        pair<GlobalIndexType, unsigned> neighborInfo = cell->getNeighborInfo(sideOrdinal);
        if (neighborInfo.first == -1) {
          numBoundarySides++;
        } else {
          CellPtr neighbor = meshTopo->getCell(neighborInfo.first);
          unsigned sideOrdinalInNeighbor = neighborInfo.second;
          pair<GlobalIndexType, unsigned> neighborNeighborInfo = neighbor->getNeighborInfo(sideOrdinalInNeighbor);
          TEST_ASSERT(neighborNeighborInfo.first == cellIndex);
        }
      }
    }
    TEST_ASSERT(numBoundarySides == 2);
  }
} // namespace