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

#include "Solution.h"

#include "PoissonFormulation.h"

#include "GlobalDofAssignment.h"

namespace {
  TEUCHOS_UNIT_TEST( Solution, ImportOffRankCellData )
  {
    int numCells = 8;
    int spaceDim = 1;
    // just want any bilinear form; we'll use Poisson
    bool useConformingTraces = true;
    PoissonFormulation form(spaceDim, useConformingTraces);
    
    double xLeft = 0, xRight = 1;
    
    int H1Order = 1, delta_k = 1;
    MeshPtr mesh = MeshFactory::intervalMesh(form.bf(), xLeft, xRight, numCells, H1Order, delta_k);
    
    MeshTopologyPtr meshTopo = mesh->getTopology();
    
    SolutionPtr soln = Solution::solution(mesh);
    
    set<GlobalIndexType> myCells = mesh->cellIDsInPartition();
    
    int rank = Teuchos::GlobalMPISession::getRank();
    
    // set up some dummy data
    for (set<GlobalIndexType>::iterator cellIDIt = myCells.begin(); cellIDIt != myCells.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      FieldContainer<double> cellDofs(mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
      cellDofs.initialize((double)rank);
      soln->setLocalCoefficientsForCell(cellID, cellDofs);
    }
    
    int otherRank = Teuchos::GlobalMPISession::getNProc() - 1 - rank;
    set<GlobalIndexType> cellIDsToRequest;
    if (otherRank != rank) {
      cellIDsToRequest = mesh->globalDofAssignment()->cellsInPartition(otherRank);
    }
    
//    cout << "On rank " << rank << ", otherRank = " << otherRank << endl;
    
    soln->importSolutionForOffRankCells(cellIDsToRequest);
    
    for (set<GlobalIndexType>::iterator cellIDIt = cellIDsToRequest.begin(); cellIDIt != cellIDsToRequest.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      FieldContainer<double> cellDofs = soln->allCoefficientsForCellID(cellID, false); // false: don't warn about off-rank requests
      
      TEST_ASSERT(cellDofs.size() == mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
      
      for (int i=0; i<cellDofs.size(); i++) {
        TEST_ASSERT(otherRank == cellDofs[i]);
      }
    }
  }
} // namespace