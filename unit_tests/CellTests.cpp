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

#include "CamelliaCellTools.h"
#include "Cell.h"
#include "MeshFactory.h"

using namespace Camellia;
using namespace Intrepid;

namespace {
  TEUCHOS_UNIT_TEST( Cell, FindSubcellOrdinalInSide )
  {
    double width=1.0, height=1.0;
    int horizontalElements=2, verticalElements=1;
    MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(width,height,horizontalElements,verticalElements);
    
    int sideDim = meshTopo->getSpaceDim() - 1;
    for (auto cellIndex : meshTopo->getActiveCellIndices())
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
