//
//  BasisSumFunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/29/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisFactory.h"
#include "BasisSumFunction.h"
#include "Function.h"
#include "SpaceTimeHeatFormulation.h"
#include "TensorBasis.h"
#include "TypeDefs.h"

using namespace Camellia;
using namespace Intrepid;

namespace {
  MeshPtr singleCellSpaceTimeMesh(CellTopoPtr spaceTopo, vector<int> H1Order) {
    CellTopoPtr timeTopo = CellTopology::line();
    int tensorialDegree = 1;
    CellTopoPtr tensorTopo = CellTopology::cellTopology(spaceTopo, tensorialDegree);
    
    FieldContainer<double> timeNodes(timeTopo->getNodeCount(), timeTopo->getDimension());
    double t0 = 0, t1 = 1;
    // time will go from t0 to t1:
    timeNodes(0,0) = t0;
    timeNodes(1,0) = t1;
    
    FieldContainer<double> spaceNodes(spaceTopo->getNodeCount(), spaceTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(spaceNodes, spaceTopo);
    FieldContainer<double> tensorNodesFC(tensorTopo->getNodeCount(), tensorTopo->getDimension());
    tensorTopo->initializeNodes({spaceNodes, timeNodes}, tensorNodesFC);
    
    vector<vector<double> > tensorNodes;
    CamelliaCellTools::pointsVectorFromFC(tensorNodes, tensorNodesFC);
    
    MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(tensorTopo->getDimension()) );
    
    meshTopo->addCell(tensorTopo, tensorNodes);
    
    double epsilon = 1.0;
    SpaceTimeHeatFormulation form(spaceTopo->getDimension(), epsilon, true);
    
    int delta_k = 1;
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, form.bf(), H1Order, delta_k) );
    
    return mesh;
  }

  void testConstantBasisSpaceTime(CellTopoPtr spaceTopo, bool testSides, Teuchos::FancyOStream &out, bool &success)
  {
    // a TensorBasis made up of nodal components should itself be nodal.
    // therefore, using constant coefficients we should get a constant function
    const static double CONST_VALUE = 0.5;
    
    FunctionPtr f = Function::constant(CONST_VALUE);
    
    int spacePolyOrder = 3;
    int spaceH1Order = spacePolyOrder + 1;
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    BasisPtr spaceBasis = basisFactory->getBasis(spaceH1Order, spaceTopo, Camellia::FUNCTION_SPACE_HGRAD);
    
    int timePolyOrder = 2;
    int timeH1Order = timePolyOrder + 1;
    BasisPtr timeBasis = basisFactory->getBasis(timeH1Order, CellTopology::line(), Camellia::FUNCTION_SPACE_HGRAD);
    
    typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
    Teuchos::RCP<TensorBasis> tensorBasis = Teuchos::rcp( new TensorBasis(spaceBasis, timeBasis) );
    
    FieldContainer<double> basisCoefficients(tensorBasis->getCardinality());
    basisCoefficients.initialize(CONST_VALUE);
    
    FunctionPtr basisSumFunction = BasisSumFunction::basisSumFunction(tensorBasis, basisCoefficients);
    vector<int> H1Order = {spaceH1Order, timeH1Order};
    
    MeshPtr spaceTimeMesh = singleCellSpaceTimeMesh(spaceTopo, H1Order);
    
    double tol = 1e-14;
    if (!testSides)
    {
      double l2Err = (f - basisSumFunction)->l2norm(spaceTimeMesh);
      TEST_COMPARE(l2Err, <, tol);
    }
    else
    {
      FunctionPtr restrictor = Function::meshSkeletonCharacteristic();
      double l2Err = (f * restrictor - basisSumFunction * restrictor)->l2norm(spaceTimeMesh);
      TEST_COMPARE(l2Err, <, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisLine )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    bool testSides = false;
    testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisQuad )
  {
    CellTopoPtr spaceTopo = CellTopology::quad();
    bool testSides = false;
    testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisLineSides )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    bool testSides = true;
    testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisSumFunction, TensorBasisQuadSides )
  {
    CellTopoPtr spaceTopo = CellTopology::quad();
    bool testSides = true;
    testConstantBasisSpaceTime(spaceTopo, testSides, out, success);
  }
} // namespace