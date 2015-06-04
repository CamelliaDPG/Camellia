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
  PoissonFormulation form(spaceDim, useConformingTraces);
  BFPtr bf = form.bf();

  DofOrderingFactory factory(bf);

  vector<int> polyOrder(2); // space, time
  polyOrder[0] = 2;
  polyOrder[1] = 3;

  CellTopoPtr hexTopo = CellTopology::hexahedron();
  int tensorialDegree = 1;
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(hexTopo, tensorialDegree);

  VarPtr q = form.q();
  VarPtr tau = form.tau();

  DofOrderingPtr testOrdering = factory.testOrdering(polyOrder, spaceTimeTopo);

  BasisPtr qBasis = testOrdering->getBasis(q->ID());
  BasisPtr tauBasis = testOrdering->getBasis(tau->ID());

  TensorBasis<double>* qTensorBasis = dynamic_cast< TensorBasis<double>* >(qBasis.get());
  TensorBasis<double>* tauTensorBasis = dynamic_cast< TensorBasis<double>* >(tauBasis.get());

  TEUCHOS_TEST_FOR_EXCEPTION(qTensorBasis == NULL, std::invalid_argument, "qBasis is not a TensorBasis instance");
  TEUCHOS_TEST_FOR_EXCEPTION(tauTensorBasis == NULL, std::invalid_argument, "tauBasis is not a TensorBasis instance");

  BasisPtr qSpatialBasis = qTensorBasis->getSpatialBasis();
  TEST_EQUALITY(qSpatialBasis->functionSpace(), efsForSpace(q->space()));

  BasisPtr tauSpatialBasis = tauTensorBasis->getSpatialBasis();
  TEST_EQUALITY(tauSpatialBasis->functionSpace(), efsForSpace(tau->space()));

  BasisPtr qTemporalBasis = qTensorBasis->getTemporalBasis();
  TEST_EQUALITY(qTemporalBasis->functionSpace(), efsForSpace(HGRAD));

  BasisPtr tauTemporalBasis = tauTensorBasis->getTemporalBasis();
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
  int H1Order = spacePolyOrder + 1;
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
      TensorBasis<double>* uHatTensorBasis = dynamic_cast< TensorBasis<double>* >(uHatBasis.get());
      TEUCHOS_TEST_FOR_EXCEPTION(uHatTensorBasis == NULL, std::invalid_argument, "uHatBasis is not a TensorBasis instance");
      BasisPtr uHatTemporalBasis = uHatTensorBasis->getTemporalBasis();
      TEST_EQUALITY(uHatTemporalBasis->getDegree(), H1Order);
      TEST_EQUALITY(uHatTemporalBasis->getCardinality(), H1Order+1);
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
    polyOrder[0] = spacePolyOrder;
    polyOrder[1] = spacePolyOrder;
    
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
        
        TensorBasis<double>* uHatTensorBasis = dynamic_cast< TensorBasis<double>* >(uHatBasis.get());
        TEUCHOS_TEST_FOR_EXCEPTION(uHatTensorBasis == NULL, std::invalid_argument, "uHatBasis is not a TensorBasis instance");
        BasisPtr uHatTemporalBasis = uHatTensorBasis->getTemporalBasis();
        TEST_EQUALITY(uHatTemporalBasis->getDegree(), H1Order-1);
        TEST_EQUALITY(uHatTemporalBasis->getCardinality(), H1Order);
        
        TEST_EQUALITY(uHatTemporalBasis->functionSpace(), efsForSpace(L2));
      }
      else
      {
        // TODO: test something here??
      }
    }
  }
} // namespace
