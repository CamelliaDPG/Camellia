#include "BasisCacheTests.h"

#include "MeshFactory.h"

#include "CamelliaCellTools.h"

FieldContainer<double> BasisCacheTests::referenceCubeNodes() {
  FieldContainer<double> cubePoints(8,3);
  cubePoints(0,0) = -1;
  cubePoints(0,1) = -1;
  cubePoints(0,2) = -1;
  
  cubePoints(1,0) = 1;
  cubePoints(1,1) = -1;
  cubePoints(1,2) = -1;
  
  cubePoints(2,0) = 1;
  cubePoints(2,1) = 1;
  cubePoints(2,2) = -1;
  
  cubePoints(3,0) = -1;
  cubePoints(3,1) = 1;
  cubePoints(3,2) = -1;
  
  cubePoints(4,0) = -1;
  cubePoints(4,1) = -1;
  cubePoints(4,2) = 1;
  
  cubePoints(5,0) = 1;
  cubePoints(5,1) = -1;
  cubePoints(5,2) = 1;
  
  cubePoints(6,0) = 1;
  cubePoints(6,1) = 1;
  cubePoints(6,2) = 1;
  
  cubePoints(7,0) = -1;
  cubePoints(7,1) = 1;
  cubePoints(7,2) = 1;
  return cubePoints;
}

FieldContainer<double> BasisCacheTests::unitCubeNodes() {
  FieldContainer<double> cubePoints = referenceCubeNodes();
  for (int i=0; i<cubePoints.size(); i++) {
    cubePoints[i] = (cubePoints[i]+1) / 2;
  }
  return cubePoints;
}

void BasisCacheTests::setup() {
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
  double eps = 1e-2;
  
  _confusionBF; // standard confusion bilinear form
  _confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  _confusionBF->addTerm(sigma1 / eps, tau->x());
  _confusionBF->addTerm(sigma2 / eps, tau->y());
  _confusionBF->addTerm(u, tau->div());
  _confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  _confusionBF->addTerm( sigma1, v->dx() );
  _confusionBF->addTerm( sigma2, v->dy() );
  _confusionBF->addTerm( beta_const * u, - v->grad() );
  _confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  _uhat_confusion = uhat; // confusion variable u_hat
  
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  
  int H1Order = 1, pToAdd = 0;
  int horizontalCells = 1, verticalCells = 1;
  
  // create a pointer to a new mesh:
  _spectralConfusionMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                      _confusionBF, H1Order, H1Order+pToAdd);
  
  // some 2D test points:
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {-1.0,-0.8,-0.6,-.4,-.2,0,0.2,0.4,0.6,0.8};
  double y[NUM_POINTS_1D] = {-0.8,-0.6,-.4,-.2,0,0.2,0.4,0.6,0.8,1.0};
  
  _testPoints = FieldContainer<double>(NUM_POINTS_1D*NUM_POINTS_1D,2);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    for (int j=0; j<NUM_POINTS_1D; j++) {
      _testPoints(i*NUM_POINTS_1D + j, 0) = x[i];
      _testPoints(i*NUM_POINTS_1D + j, 1) = y[i];
    }
  }
  
  _elemType = _spectralConfusionMesh->getElement(0)->elementType();
  vector<GlobalIndexType> cellIDs;
  GlobalIndexType cellID = 0;
  cellIDs.push_back(cellID);
  _basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) ); // *will* create side caches
  _basisCache->setRefCellPoints(_testPoints);
  
  _basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID), cellIDs, true );
  
}

void BasisCacheTests::teardown() {
  
}

void BasisCacheTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testJacobian3D()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testSetRefCellPoints()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool BasisCacheTests::testJacobian3D() {
  bool success = true;
  // for now, let's use the reference cell.  (Jacobian should be the identity.)
  FieldContainer<double> refCubePoints = referenceCubeNodes();
  
  // small upgrade: unit cube
  //  FieldContainer<double> cubePoints = unitCubeNodes();
  
  int numCells = 1;
  refCubePoints.resize(numCells,8,3); // first argument is cellIndex; we'll just have 1
  
  Teuchos::RCP<shards::CellTopology> hexTopoPtr;
  hexTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));

  int spaceDim = 3;
  int cubDegree = 2;
  
  shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8> >() );
  FieldContainer<double> physicalCellNodes = referenceCubeNodes();
  physicalCellNodes.resize(numCells,hexTopo.getVertexCount(),spaceDim);
  BasisCache hexBasisCache( physicalCellNodes, hexTopo, cubDegree);

  FieldContainer<double> referenceToReferenceJacobian = hexBasisCache.getJacobian();
  int numPoints = referenceToReferenceJacobian.dimension(1);
  
  FieldContainer<double> kronecker(numCells,numPoints,spaceDim,spaceDim);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      for (int d1=0; d1<spaceDim; d1++) {
        for (int d2=0; d2<spaceDim; d2++) {
          kronecker(cellIndex,ptIndex,d1,d2) = (d1==d2) ? 1 : 0;
        }
      }
    }
  }
  
  double maxDiff;
  double tol = 1e-14;
  
  if (! fcsAgree(kronecker, referenceToReferenceJacobian, tol, maxDiff) ) {
    cout << "identity map doesn't have identity Jacobian.\n";
    cout << "maxDiff = " << maxDiff << endl;
    success = false;
  }
  
  physicalCellNodes = unitCubeNodes();
  physicalCellNodes.resize(numCells, hexTopo.getVertexCount(), hexTopo.getDimension());
  hexBasisCache = BasisCache( physicalCellNodes, hexTopo, cubDegree );
  FieldContainer<double> halfKronecker = kronecker;
  BilinearForm::multiplyFCByWeight(halfKronecker, 0.5);
  
  FieldContainer<double> referenceToUnitCubeJacobian = hexBasisCache.getJacobian();
  if (! fcsAgree(halfKronecker, referenceToUnitCubeJacobian, tol, maxDiff) ) {
    cout << "map to unit cube doesn't have the expected half-identity Jacobian.\n";
    cout << "maxDiff = " << maxDiff << endl;
    success = false;
  }
  
  return success;
}

bool BasisCacheTests::testSetRefCellPoints() {
  setup();
  
  bool success = true;
  double tol = 1e-14;
  
  // for now, we just test setting these on the side cache, since that's broken right now.
  // TODO: test for volume cache as well.
  shards::CellTopology cellTopo = *(_elemType->cellTopoPtr);
  int numSides = CamelliaCellTools::getSideCount(cellTopo);
  
//  FieldContainer<double> refTracePoints(3, 1); // (P,D): three points, 1D
//  refTracePoints(0, 0) =  0.0;
//  refTracePoints(1, 0) =  0.5;
//  refTracePoints(2, 0) =  1.0;
  
  for (int sideIndex=0; sideIndex < numSides; sideIndex++)
  {
    BasisCachePtr sideBasisCache = _basisCache->getSideBasisCache(sideIndex);
    FieldContainer<double> refCellPoints = sideBasisCache->getRefCellPoints();
    FieldContainer<double> physicalCubaturePointsExpected = sideBasisCache->getPhysicalCubaturePoints();
//    cout << "side " << sideIndex << " ref cell points:\n" << refCellPoints;
//    cout << "side " << sideIndex << " physCubature points:\n" << physicalCubaturePointsExpected;
    sideBasisCache->setRefCellPoints(refCellPoints);
    FieldContainer<double> physicalCubaturePointsActual = sideBasisCache->getPhysicalCubaturePoints();
    double maxDiff = 0;
    if (! fcsAgree(physicalCubaturePointsActual, physicalCubaturePointsExpected, tol, maxDiff) ) {
      success = false;
      cout << "After resetting refCellPoints, physical cubature points are different in side basis cache.\n";
    }
    // TODO: test that the results are correct.
  }
    
  // TODO: test quad
    
  // test hexahedron
  int numCells = 1;
  int numPoints = 1;
  int spaceDim = 3;
  int cubDegree = 2; // doesn't affect the test
  
  shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8> >() );
  FieldContainer<double> physicalCellNodes = unitCubeNodes();
  physicalCellNodes.resize(numCells,hexTopo.getVertexCount(),spaceDim);
  BasisCache hexBasisCache( physicalCellNodes, hexTopo, cubDegree);
  
  FieldContainer<double> refCellPointsHex(numPoints,spaceDim);
  refCellPointsHex(0,0) = 0.0;
  refCellPointsHex(0,1) = 0.0;
  refCellPointsHex(0,2) = 0.0;
  
  FieldContainer<double> physicalPointsExpected(numCells,numPoints,spaceDim);
  physicalPointsExpected(0,0,0) = 0.5;
  physicalPointsExpected(0,0,1) = 0.5;
  physicalPointsExpected(0,0,2) = 0.5;
 
  hexBasisCache.setRefCellPoints(refCellPointsHex);
  
  FieldContainer<double> physicalPointsActual = hexBasisCache.getPhysicalCubaturePoints();
  
  double maxDiff;
  if (! fcsAgree(physicalPointsActual, physicalPointsExpected, 1e-14, maxDiff)) {
    cout << "physical points don't match expected for hexahedron.";
  }
  
  return success;
}

string BasisCacheTests::testSuiteName() {
  return "BasisCacheTests";
}