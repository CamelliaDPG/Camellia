#include "BasisCacheTests.h"

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
  _spectralConfusionMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
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
  vector<int> cellIDs;
  int cellID = 0;
  cellIDs.push_back(cellID);
  _basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) ); // *will* create side caches
  _basisCache->setRefCellPoints(_testPoints);
  
  _basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID), cellIDs, true );
  
}

void BasisCacheTests::teardown() {
  
}

void BasisCacheTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testSetRefCellPoints()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool BasisCacheTests::testSetRefCellPoints() {
  setup();
  
  bool success = true;
  double tol = 1e-14;
  
  // for now, we just test setting these on the side cache, since that's broken right now.
  // TODO: test for volume cache as well.
  shards::CellTopology cellTopo = *(_elemType->cellTopoPtr);
  int numSides = cellTopo.getSideCount();
  
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
    
//    // just some exploratory code here (TODO: delete this.)
//    refCellPoints.resize(3, 1);
//    refCellPoints(0, 0) =  -1.0;
//    refCellPoints(1, 0) =  0.5;
//    refCellPoints(2, 0) =  1.0;
//    sideBasisCache->setRefCellPoints(refCellPoints);
//    physicalCubaturePointsActual = sideBasisCache->getPhysicalCubaturePoints();
//    cout << "side " << sideIndex << " ref cell points:\n" << refCellPoints;
//    cout << "side " << sideIndex << " physCubature points:\n" << physicalCubaturePointsActual;
  }



  
  
  // TODO: test that the results are correct.
  
  return success;
}

string BasisCacheTests::testSuiteName() {
  return "BasisCacheTests";
}