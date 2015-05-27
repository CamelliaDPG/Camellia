//
//  DofOrderingTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/27/15.
//
//

#include "DofOrdering.h"
#include "CellTopology.h"
#include "doubleBasisConstruction.h"

using namespace Camellia;

#include "Teuchos_UnitTestHarness.hpp"
namespace
{
TEUCHOS_UNIT_TEST( DofOrdering, TrialOrderingConsistency )
{
  bool conforming = true; // a legacy choice
  int polyOrder = 3;
  // after adding a couple things to a trial ordering, check that the totalDofCount is consistent with the basis cardinality
  int numUniqueDofs = 0;

  CellTopoPtr cellTopo = CellTopology::quad();

  int numSides = cellTopo->getSideCount();

  Teuchos::RCP<DofOrdering> trialOrdering = Teuchos::rcp( new DofOrdering(cellTopo) );
  int trialID = 0;
  BasisPtr trialBasis = Camellia::intrepidLineHGRAD(polyOrder);
  for (int sideIndex=0; sideIndex<numSides; sideIndex++)
  {
    trialOrdering->addEntry(trialID, trialBasis, trialBasis->rangeRank(), sideIndex);
    numUniqueDofs += trialBasis->getCardinality();
  }
  if (conforming)
  {
    int firstVertexOrdinal = *(trialBasis->dofOrdinalsForVertex(0).begin());
    int lastVertexOrdinal = *(trialBasis->dofOrdinalsForVertex(1).begin());
    for (int sideIndex=0; sideIndex<numSides; sideIndex++)
    {
      int otherSideIndex = (sideIndex+1) % numSides;
      trialOrdering->addIdentification(trialID, sideIndex, lastVertexOrdinal, otherSideIndex, firstVertexOrdinal);
      numUniqueDofs--;
    }
  }
  trialOrdering->rebuildIndex();
  TEST_EQUALITY( trialOrdering->totalDofs(), numUniqueDofs );
}

TEUCHOS_UNIT_TEST( DofOrdering, SidesForVarID )
{
  int polyOrder = 3;

  vector<int> sidesExpected;
  sidesExpected.push_back(-1);
  sidesExpected.push_back(2);
  sidesExpected.push_back(4);

  CellTopoPtr cellTopo = CellTopology::quad();

  Teuchos::RCP<DofOrdering> trialOrdering = Teuchos::rcp( new DofOrdering(cellTopo) );
  int trialID = 2;
  BasisPtr trialBasis = Camellia::intrepidLineHGRAD(polyOrder);
  for (vector<int>::iterator sideIt = sidesExpected.begin(); sideIt != sidesExpected.end(); sideIt++)
  {
    trialOrdering->addEntry(trialID, trialBasis, trialBasis->rangeRank(), *sideIt);
  }

  TEST_COMPARE_ARRAYS( sidesExpected, trialOrdering->getSidesForVarID(trialID) );
}
//  TEUCHOS_UNIT_TEST( Int, Assignment )
//  {
//    int i1 = 4;
//    int i2 = i1;
//    TEST_EQUALITY( i2, i1 );
//  }
} // namespace