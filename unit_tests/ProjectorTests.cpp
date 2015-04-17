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
using namespace Intrepid;

namespace {
  void testProjectFunctionOnTensorTopoSides(CellTopoPtr spaceTopo, int H1Order, Camellia::EFunctionSpace fs, FunctionPtr<double> f,
                                            Teuchos::FancyOStream &out, bool &success) {
    int tensorialDegree = 1;
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo->getShardsTopology(), tensorialDegree);

    int cubatureDegree = H1Order * 2;
    bool createSideCache = true;
    BasisCachePtr basisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);

    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();

    Intrepid::FieldContainer<double> basisCoefficients;

    double tol = 1e-15;

    for (int sideOrdinal = 0; sideOrdinal < spaceTimeTopo->getSideCount(); sideOrdinal++) {
      CellTopoPtr sideTopo = spaceTimeTopo->getSide(sideOrdinal);
      BasisPtr sideBasis = basisFactory->getBasis(H1Order, sideTopo, fs);
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);

      int numCells = 1;
      basisCoefficients.resize(numCells,sideBasis->getCardinality());

      Projector::projectFunctionOntoBasis(basisCoefficients, f, sideBasis, sideBasisCache);

      basisCoefficients.resize(sideBasis->getCardinality());
      FunctionPtr<double> projectedFunction = BasisSumFunction::basisSumFunction(sideBasis, basisCoefficients);

      double expectedIntegral = f->integrate(sideBasisCache);
      double actualIntegral = projectedFunction->integrate(sideBasisCache);

//      cout << "f: " << f->displayString() << endl;
//      cout << "basisCoefficients:\n" << basisCoefficients;
//      cout << "physicalCubaturePoints:\n" << sideBasisCache->getPhysicalCubaturePoints();

//      cout << "expectedIntegral: " << expectedIntegral << endl;

      TEST_FLOATING_EQUALITY(expectedIntegral,actualIntegral,tol);

      double integralOfDifference = (projectedFunction - f)->integrate(sideBasisCache);

      TEST_ASSERT(abs(integralOfDifference) < tol);
    }
  }

  TEUCHOS_UNIT_TEST( Projector, TensorTopologyFlux1D )
  {
    // project a function that involves normal values
    CellTopoPtr spaceTopo = CellTopology::line();

    int H1Order = 3;
    FunctionPtr<double> n = Function<double>::normalSpaceTime();
    FunctionPtr<double> f = Function<double>::xn(2) * n->x() + Function<double>::yn(1) * n->y();

    testProjectFunctionOnTensorTopoSides(spaceTopo, H1Order, Camellia::FUNCTION_SPACE_HVOL, f, out, success);
  }

  TEUCHOS_UNIT_TEST( Projector, TensorTopologyTrace1D )
  {
    CellTopoPtr spaceTopo = CellTopology::line();

    int H1Order = 2;
    FunctionPtr<double> f = Function<double>::xn(2);

    testProjectFunctionOnTensorTopoSides(spaceTopo, H1Order, Camellia::FUNCTION_SPACE_HGRAD, f, out, success);
  }

} // namespace
