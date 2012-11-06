#include "IncompressibleFormulationsTests.h"

void IncompressibleFormulationsTests::setup() {
  
  FunctionPtr x = Teuchos::rcp ( new Xn(1) );
  FunctionPtr x2 = Teuchos::rcp( new Xn(2) );
  FunctionPtr y2 = Teuchos::rcp( new Yn(2) );
  FunctionPtr y = Teuchos::rcp( new Yn(1) );
  
  double mu = 1.0;
  
  _vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
  
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
  _vgpStokesMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                       _vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
  
  FunctionPtr u1_exact = x2;
  FunctionPtr u2_exact = -2*x*y; // chosen to have zero divergence
  FunctionPtr p_exact = y; // odd function: zero mean on our domain
  
  SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere

  _vgpStokesExactSolution = _vgpStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
  
  BCPtr vgpBC = _vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
  
  _vgpStokesSolution = Teuchos::rcp( new Solution(_vgpStokesMesh, vgpBC,
                                                  _vgpStokesExactSolution->rhs(),
                                                  _vgpStokesFormulation->graphNorm()));
  
  _vgpNavierStokesSolution = Teuchos::rcp( new Solution(_vgpStokesMesh, vgpBC) );
  
  _vgpNavierStokesFormulation = Teuchos::rcp( new VGPNavierStokesFormulation(1.0 / mu,
                                                                               _vgpNavierStokesSolution));
  
  _vgpNavierStokesExactSolution = _vgpNavierStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);

  _vgpNavierStokesSolution->setIP(_vgpNavierStokesFormulation->graphNorm());
  _vgpNavierStokesSolution->setRHS(_vgpNavierStokesExactSolution->rhs());
  
  _vgpNavierStokesFormulation = Teuchos::rcp( new VGPNavierStokesFormulation(mu, _vgpNavierStokesSolution) );
  
  // some 2D test points:
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x_pts[NUM_POINTS_1D] = {-1.0,-0.8,-0.6,-.4,-.2,0,0.2,0.4,0.6,0.8};
  double y_pts[NUM_POINTS_1D] = {-0.8,-0.6,-.4,-.2,0,0.2,0.4,0.6,0.8,1.0};
  
  _testPoints = FieldContainer<double>(NUM_POINTS_1D*NUM_POINTS_1D,2);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    for (int j=0; j<NUM_POINTS_1D; j++) {
      _testPoints(i*NUM_POINTS_1D + j, 0) = x_pts[i];
      _testPoints(i*NUM_POINTS_1D + j, 1) = y_pts[j];
    }
  }
  
  _elemType = _vgpStokesMesh->getElement(0)->elementType();
  vector<int> cellIDs;
  int cellID = 0;
  cellIDs.push_back(cellID);
  _basisCache = Teuchos::rcp( new BasisCache( _elemType, _vgpStokesMesh ) );
  _basisCache->setRefCellPoints(_testPoints);
  
  _basisCache->setPhysicalCellNodes( _vgpStokesMesh->physicalCellNodesForCell(cellID), cellIDs, true );
}

void IncompressibleFormulationsTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testVGPStokesFormulation()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  setup();
  if (testVGPNavierStokesFormulation()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool IncompressibleFormulationsTests::testVGPStokesFormulation() {
  bool success = true;
  
  return success;
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulation() {
  bool success = true;
  
  return success;
}

std::string IncompressibleFormulationsTests::testSuiteName() {
  return "IncompressibleFormulationsTests";
}
