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
#include "CamelliaCellTools.h"
#include "CellTopology.h"
#include "Function.h"
#include "MeshFactory.h"
#include "MeshTopology.h"
#include "Projector.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;
using namespace Intrepid;

namespace {
  void testProjectFunctionOnTensorTopoSides(CellTopoPtr spaceTopo, int H1Order, Camellia::EFunctionSpace fs, FunctionPtr f,
                                            bool interpolate, Teuchos::FancyOStream &out, bool &success) {
    int spaceDim = spaceTopo->getDimension();
    MeshTopologyPtr spatialMeshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );

    FieldContainer<double> cellNodes(spaceTopo->getNodeCount(),spaceTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(cellNodes, spaceTopo);
    vector<vector<double>> cellVerticesVector;
    CamelliaCellTools::pointsVectorFromFC(cellVerticesVector, cellNodes);
    spatialMeshTopo->addCell(spaceTopo, cellVerticesVector);
    
    double t0 = 0.0, t1 = 1.0;
    MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);
    
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
    
    int cubatureDegree = H1Order * 2;
    bool createSideCache = true;

    IndexType cellID = 0;
    FieldContainer<double> physicalCellNodes = spaceTimeMeshTopo->physicalCellNodesForCell(cellID);
    physicalCellNodes.resize(1,spaceTimeTopo->getNodeCount(),spaceTimeTopo->getDimension());
    BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(spaceTimeTopo, cubatureDegree,
                                                                     physicalCellNodes, createSideCache);
    
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();

    Intrepid::FieldContainer<double> basisCoefficients;

    double tol = 1e-15;

    for (int sideOrdinal = 0; sideOrdinal < spaceTimeTopo->getSideCount(); sideOrdinal++) {
      CellTopoPtr sideTopo = spaceTimeTopo->getSide(sideOrdinal);
      BasisPtr sideBasis = basisFactory->getBasis(H1Order, sideTopo, fs);
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);

      int numCells = 1;
      basisCoefficients.resize(numCells,sideBasis->getCardinality());

      if (interpolate)
      {
        Projector<double>::projectFunctionOntoBasisInterpolating(basisCoefficients, f, sideBasis, sideBasisCache);
      }
      else
      {
        Projector<double>::projectFunctionOntoBasis(basisCoefficients, f, sideBasis, sideBasisCache);
      }

      basisCoefficients.resize(sideBasis->getCardinality());
      FunctionPtr projectedFunction = BasisSumFunction::basisSumFunction(sideBasis, basisCoefficients);

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
    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr f = Function::xn(2) * n->x() + Function::yn(1) * n->y();

    bool interpolate = false;
    testProjectFunctionOnTensorTopoSides(spaceTopo, H1Order, Camellia::FUNCTION_SPACE_HVOL, f, interpolate, out, success);
  }

  TEUCHOS_UNIT_TEST( Projector, TensorTopologyFluxInterpolating1D )
  {
    // project a function that involves normal values
    CellTopoPtr spaceTopo = CellTopology::line();
    
    int H1Order = 3;
    FunctionPtr n = Function::normalSpaceTime();
    FunctionPtr f = Function::xn(2) * n->x() + Function::yn(1) * n->y();
    
    bool interpolate = true;
    testProjectFunctionOnTensorTopoSides(spaceTopo, H1Order, Camellia::FUNCTION_SPACE_HVOL, f, interpolate, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Projector, TensorTopologyTrace1D )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    
    int H1Order = 2;
    FunctionPtr f = Function::xn(2);
    
    bool interpolate = false;
    testProjectFunctionOnTensorTopoSides(spaceTopo, H1Order, Camellia::FUNCTION_SPACE_HGRAD, f, interpolate, out, success);
  }
  
  TEUCHOS_UNIT_TEST( Projector, TensorTopologyTraceInterpolating1D )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    
//    int H1Order = 2;
//    FunctionPtr f = Function::xn(2);
    int H1Order = 1;
    FunctionPtr f = Function::constant(0.5);
    
    bool interpolate = true;
    testProjectFunctionOnTensorTopoSides(spaceTopo, H1Order, Camellia::FUNCTION_SPACE_HGRAD, f, interpolate, out, success);
  }

} // namespace
