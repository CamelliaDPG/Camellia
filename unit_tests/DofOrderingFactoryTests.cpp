//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  DofOrderingFactoryTests
//  Camellia
//
//  Created by Nate Roberts on 4/8/15
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "TypeDefs.h"

#include "CellTopology.h"
#include "DofOrderingFactory.h"
#include "PoissonFormulation.h"
#include "SpaceTimeHeatFormulation.h"
#include "SpaceTimeHeatDivFormulation.h"
#include "TensorBasis.h"

using namespace Camellia;
using namespace std;

namespace
{
TEUCHOS_UNIT_TEST( DofOrderingFactory, SpaceTimeTestsUseHGRADInTime )
{
  int spaceDim = 3;
  bool useConformingTraces = false;
  
  const double epsilon = 1.0;
  SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
  
  BFPtr bf = form.bf();

  DofOrderingFactory factory(bf);

  vector<int> polyOrder(2); // space, time
  polyOrder[0] = 2;
  polyOrder[1] = 3;

  CellTopoPtr hexTopo = CellTopology::hexahedron();
  int tensorialDegree = 1;
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(hexTopo, tensorialDegree);

  VarPtr v = form.v();
  VarPtr tau = form.tau();

  DofOrderingPtr testOrdering = factory.testOrdering(polyOrder, spaceTimeTopo);

  BasisPtr vBasis = testOrdering->getBasis(v->ID());
  BasisPtr tauBasis = testOrdering->getBasis(tau->ID());

  auto vBasisComponents   = vBasis->intrepid2TensorBasisComponents();
  auto tauBasisComponents = tauBasis->intrepid2TensorBasisComponents();
  
  TEST_EQUALITY(vBasisComponents.size(), 2);
  TEST_EQUALITY(tauBasisComponents.size(), 2);
  
  BasisPtr vSpatialBasis = vBasisComponents[0];
  TEST_EQUALITY(vSpatialBasis->functionSpace(), efsForSpace(v->space()));

  BasisPtr tauSpatialBasis = tauBasisComponents[0];
  TEST_EQUALITY(tauSpatialBasis->functionSpace(), efsForSpace(tau->space()));

  BasisPtr vTemporalBasis = vBasisComponents[1];
  TEST_EQUALITY(vTemporalBasis->functionSpace(), efsForSpace(HGRAD));

  BasisPtr tauTemporalBasis = tauBasisComponents[1];
  TEST_EQUALITY(tauTemporalBasis->functionSpace(), efsForSpace(HGRAD));
}

TEUCHOS_UNIT_TEST( DofOrderingFactory, SpaceTimeTracesTimeBasisCardinality )
{
  int spaceDim = 1;
  bool useConformingTraces = true;
  double epsilon = 1.0;
  SpaceTimeHeatFormulation form(spaceDim, epsilon, useConformingTraces);
  BFPtr bf = form.bf();

  DofOrderingFactory factory(bf);

  int spacePolyOrder = 2;
  vector<int> polyOrder(2); // space, time
  polyOrder[0] = spacePolyOrder;
  polyOrder[1] = spacePolyOrder;

  CellTopoPtr spaceTopo = CellTopology::line();
  int tensorialDegree = 1;
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, tensorialDegree);

  VarPtr u_hat = form.u_hat();

  DofOrderingPtr trialOrdering = factory.trialOrdering(polyOrder, spaceTimeTopo);

  for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
  {
    BasisPtr uHatBasis = trialOrdering->getBasis(u_hat->ID(), sideOrdinal);
    if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
    {
      BasisPtr uHatTemporalBasis = uHatBasis->intrepid2TensorBasisComponents()[1];
      TEST_EQUALITY(uHatTemporalBasis->getDegree(), polyOrder[1]);
      TEST_EQUALITY(uHatTemporalBasis->getCardinality(), polyOrder[1]+1);
    }
    else
    {
      // TODO: test something here??
    }
  }
}
  
  TEUCHOS_UNIT_TEST( DofOrderingFactory, SpaceTimeNonconformingTracesUseHVOLInTime )
  {
    int spaceDim = 1;
    bool useConformingTraces = false;
    double epsilon = 1.0;
    SpaceTimeHeatDivFormulation form(spaceDim, epsilon, useConformingTraces);
    BFPtr bf = form.bf();
    
    DofOrderingFactory factory(bf);
    
    int spacePolyOrder = 2;
    int H1Order = spacePolyOrder + 1;
    vector<int> polyOrder(2); // space, time
    polyOrder[0] = H1Order;
    polyOrder[1] = H1Order;
    
    CellTopoPtr spaceTopo = CellTopology::line();
    int tensorialDegree = 1;
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, tensorialDegree);
    
    VarPtr u_hat = form.uhat();
    
    DofOrderingPtr trialOrdering = factory.trialOrdering(polyOrder, spaceTimeTopo);
    
    for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
    {
      if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
      {
        BasisPtr uHatBasis = trialOrdering->getBasis(u_hat->ID(), sideOrdinal);
        
        TEST_EQUALITY(uHatBasis->getDegree(), H1Order-1);
        TEST_EQUALITY(uHatBasis->functionSpace(), efsForSpace(L2));
      }
      else
      {
        // TODO: test something here??
      }
    }
  }
} // namespace
