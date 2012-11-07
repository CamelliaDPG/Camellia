#include "IncompressibleFormulationsTests.h"

void IncompressibleFormulationsTests::setup() {
  
  x = Teuchos::rcp ( new Xn(1) );
  x2 = Teuchos::rcp( new Xn(2) );
  x3 = Teuchos::rcp( new Xn(3) );
  
  y = Teuchos::rcp( new Yn(1) );
  y2 = Teuchos::rcp( new Yn(2) );
  y3 = Teuchos::rcp( new Yn(3) );
  
  zero = Function::zero();
  
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  u1_vgp = varFactory.fieldVar(VGP_U1_S);
  u2_vgp = varFactory.fieldVar(VGP_U2_S);
  sigma11_vgp = varFactory.fieldVar(VGP_SIGMA11_S);
  sigma12_vgp = varFactory.fieldVar(VGP_SIGMA12_S);
  sigma21_vgp = varFactory.fieldVar(VGP_SIGMA21_S);
  sigma22_vgp = varFactory.fieldVar(VGP_SIGMA22_S);
  p_vgp = varFactory.fieldVar(VGP_P_S);
  
  vgpFields.push_back(u1_vgp);
  vgpFields.push_back(u2_vgp);
  vgpFields.push_back(sigma11_vgp);
  vgpFields.push_back(sigma12_vgp);
  vgpFields.push_back(sigma21_vgp);
  vgpFields.push_back(sigma22_vgp);
  vgpFields.push_back(p_vgp);
  
  // set up test containers.  Each test should try every combination here:
  // exact solutions:
  
  PolyExactFunctions exactFxns;
  
  exactFxns.push_back( make_pair(zero, 0) ); // u1
  exactFxns.push_back( make_pair(zero, 0) ); // u2 (chosen to have zero divergence)
  exactFxns.push_back( make_pair(zero, 0) ); // p: odd function: zero mean on our domain
  polyExactFunctions.push_back(exactFxns);
  exactFxns.clear();
  
  exactFxns.push_back( make_pair(zero, 0) ); // u1
  exactFxns.push_back( make_pair(zero, 0) ); // u2 (chosen to have zero divergence)
  exactFxns.push_back( make_pair(y, 1) ); // p: odd function: zero mean on our domain
  polyExactFunctions.push_back(exactFxns);
  exactFxns.clear();
  
  exactFxns.push_back( make_pair(x2, 2) );     // u1
  exactFxns.push_back( make_pair(-2*x*y, 2) ); // u2 (chosen to have zero divergence)
  exactFxns.push_back( make_pair(y, 1) );      // p: odd function: zero mean on our domain
  polyExactFunctions.push_back(exactFxns);
  exactFxns.clear();
  
  exactFxns.push_back( make_pair(x2 * y, 3) );  // u1
  exactFxns.push_back( make_pair(-x*y2, 3) );   // u2 (chosen to have zero divergence)
  exactFxns.push_back( make_pair(y3, 3) );      // p: odd function: zero mean on our domain
  polyExactFunctions.push_back(exactFxns);
  exactFxns.clear();
  
  // mesh choices (horizontal x vertical cells)
  meshDimensions.push_back( make_pair(1, 1));
  meshDimensions.push_back( make_pair(2, 2));
  meshDimensions.push_back( make_pair(1, 2));
  meshDimensions.push_back( make_pair(2, 1));
  
  pToAddValues.push_back(1);
  
  muValues.push_back(1.0);
  muValues.push_back(0.1);
  
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  quadPoints.resize(4,2);
  
  quadPoints(0,0) = -1.0; // x1
  quadPoints(0,1) = -1.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = -1.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = -1.0;
  quadPoints(3,1) = 1.0;
  
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
}

void IncompressibleFormulationsTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testVGPStokesFormulationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testVGPStokesFormulationCorrectness()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testVGPNavierStokesFormulationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testVGPNavierStokesFormulationCorrectness()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool IncompressibleFormulationsTests::testVGPStokesFormulationConsistency() {
  bool success = true;
  
  // consistency: check that solving using the BF, RHS, BCs, etc. in the Formulation
  //              gives the ExactSolution specified by the Formulation
  
  // starting out with a single hard-coded solve, but will switch soon to
  // doing several: varying meshes, pToAdd, mu, and which exact solutions we use...
    
  double tol = 1e-11;
  
  // exact solution functions: store these as vector< pair< Function, int > >
  // in the order u1, u2, p, where the paired int is the polynomial degree of the function
  
  for (vector< PolyExactFunctions >::iterator exactIt = polyExactFunctions.begin();
       exactIt != polyExactFunctions.end(); exactIt++) {
    PolyExactFunctions exactFxns = *exactIt;
  
    int maxPolyOrder = 0;
    for (int i=0; i<exactFxns.size(); i++) {
      int polyOrder = exactFxns[i].second;
      maxPolyOrder = max(maxPolyOrder,polyOrder);
    }
    
    int H1Order = maxPolyOrder + 1;
    for (vector<int>::iterator pToAddIt = pToAddValues.begin(); pToAddIt != pToAddValues.end(); pToAddIt++) {
      int pToAdd = *pToAddIt;

      for (vector< pair<int, int> >::iterator meshDimIt = meshDimensions.begin(); meshDimIt != meshDimensions.end(); meshDimIt++) {
        int horizontalCells = meshDimIt->first, verticalCells = meshDimIt->second;
    
        for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
          double mu = *muIt;
      
          _vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
        
          // create a pointer to a new mesh:
          _vgpStokesMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                               _vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
          
          FunctionPtr u1_exact = exactFxns[0].first;
          FunctionPtr u2_exact = exactFxns[1].first;
          FunctionPtr p_exact  = exactFxns[2].first;
          
          SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
          
          _vgpStokesExactSolution = _vgpStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
          
          BCPtr vgpBC = _vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
          
          _vgpStokesSolution = Teuchos::rcp( new Solution(_vgpStokesMesh, vgpBC,
                                                          _vgpStokesExactSolution->rhs(),
                                                          _vgpStokesFormulation->graphNorm()));
          
          _vgpStokesSolution->solve();
          
          int cubatureDegree = maxPolyOrder;
          for (vector< VarPtr >::iterator fieldIt = vgpFields.begin(); fieldIt != vgpFields.end(); fieldIt++ ) {
            VarPtr field = *fieldIt;
            double l2Error = _vgpStokesExactSolution->L2NormOfError(*_vgpStokesSolution, field->ID(), cubatureDegree);
            if (l2Error > tol) {
              success = false;
              cout << "testVGPStokesFormulationConsistency(): ";
              cout << "L^2 error of " << l2Error << " for variable " << field->displayString();
              cout << " exceeds tol " << tol << endl;
            }
          }
        }
      }
    }
  }
  return success;
}

bool IncompressibleFormulationsTests::testVGPStokesFormulationCorrectness() {
  bool success = true;
  
  cout << "Warning: testVGPStokesFormulationCorrectness() is trivial.\n";
  
  // consistency: check that the RHS is correct.  (Could also check other things,
  //              like individual solution components, etc., but RHS is the thing we're
  //              most liable to get incorrect.)
  double tol = 1e-11;
  
  // exact solution functions: store these as vector< pair< Function, int > >
  // in the order u1, u2, p, where the paired int is the polynomial degree of the function
  
  for (vector< PolyExactFunctions >::iterator exactIt = polyExactFunctions.begin();
       exactIt != polyExactFunctions.end(); exactIt++) {
    PolyExactFunctions exactFxns = *exactIt;
    
    for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
      double mu = *muIt;
      
      _vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
      
      RHSPtr rhs = _vgpStokesExactSolution->rhs();
      
    }
  }
  
  return success;
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulationConsistency() {
  bool success = true;
  
  cout << "Warning: testVGPNavierStokesFormulationConsistency() is trivial.\n";
  
  int H1Order = 1, pToAdd = 0;
  int horizontalCells = 1, verticalCells = 1;
  
  double mu = 1.0;
  
  _vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
  
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
  
  return success;
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulationCorrectness() {
  bool success = true;
  
  cout << "Warning: testVGPNavierStokesFormulationCorrectness() is trivial.\n";
  
  return success;
}

std::string IncompressibleFormulationsTests::testSuiteName() {
  return "IncompressibleFormulationsTests";
}
