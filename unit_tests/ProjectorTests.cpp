//
//  ProjectorTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/3/15.
//
//

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisSumFunction.h"
#include "CellTopology.h"
#include "Function.h"
#include "Projector.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

namespace {
  TEUCHOS_UNIT_TEST( Projector, TensorTopologyTrace1D )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    int tensorialDegree = 1;
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), tensorialDegree);
    
    int H1Order = 2;
    int cubatureDegree = H1Order * 2;
    bool createSideCache = true;
    BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
    
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    
    FunctionPtr f = Function::xn(2);
    
    Intrepid::FieldContainer<double> basisCoefficients;
    
    double tol = 1e-15;
    
    for (int sideOrdinal = 0; sideOrdinal < spaceTimeTopo->getSideCount(); sideOrdinal++) {
      CellTopoPtr sideTopo = spaceTimeTopo->getSide(sideOrdinal);
      BasisPtr sideBasis = basisFactory->getBasis(H1Order, sideTopo, Camellia::FUNCTION_SPACE_HGRAD);
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      
      int numCells = 1;
      basisCoefficients.resize(numCells,sideBasis->getCardinality());
      
      Projector::projectFunctionOntoBasis(basisCoefficients, f, sideBasis, sideBasisCache);
      
      basisCoefficients.resize(sideBasis->getCardinality());
      FunctionPtr projectedFunction = BasisSumFunction::basisSumFunction(sideBasis, basisCoefficients);
      
      double expectedIntegral = f->integrate(sideBasisCache);
      double actualIntegral = projectedFunction->integrate(sideBasisCache);
      
      TEST_FLOATING_EQUALITY(expectedIntegral,actualIntegral,tol);
      
      double integralOfDifference = (projectedFunction - f)->integrate(sideBasisCache);
      
      TEST_ASSERT(abs(integralOfDifference) < tol);
    }
  }
} // namespace