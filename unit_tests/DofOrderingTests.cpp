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
#include "ElementType.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"

using namespace Camellia;
using namespace Intrepid;

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
  sidesExpected.push_back(3);

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
  
  TEUCHOS_UNIT_TEST( DofOrdering, VariablesWithNonZeroEntries)
  {
    int spaceDim = 2;
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), {1.0,1.0}, {1,1}, 1);
    
    GlobalIndexType cellID = 0;
    DofOrderingPtr trialOrdering = mesh->getElementType(cellID)->trialOrderPtr;
    
    VarPtr phi_hat = form.phi_hat();
    
    FieldContainer<double> localCoefficients(trialOrdering->totalDofs());
    for (int sideOrdinal : trialOrdering->getSidesForVarID(phi_hat->ID()))
    {
      int numBasisDofs = trialOrdering->getBasisCardinality(phi_hat->ID(), sideOrdinal);
      for (int basisDofOrdinal = 0; basisDofOrdinal < numBasisDofs; basisDofOrdinal++)
      {
        int localDofIndex = trialOrdering->getDofIndex(phi_hat->ID(), basisDofOrdinal, sideOrdinal);
        localCoefficients(localDofIndex) = 1.0;
      }
    }
    
    double tol = 1e-15;
    vector<pair<int,vector<int>>> entries = trialOrdering->variablesWithNonZeroEntries(localCoefficients, tol);
    
    TEUCHOS_ASSERT_EQUALITY(entries.size(), 1); // just phi_hat
    
    pair<int,vector<int>> entry = entries[0];
    
    TEUCHOS_ASSERT_EQUALITY(entry.first, phi_hat->ID());
    
    TEUCHOS_ASSERT_EQUALITY(entry.second.size(), trialOrdering->getSidesForVarID(phi_hat->ID()).size());
  }
//  TEUCHOS_UNIT_TEST( Int, Assignment )
//  {
//    int i1 = 4;
//    int i2 = i1;
//    TEST_EQUALITY( i2, i1 );
//  }
} // namespace