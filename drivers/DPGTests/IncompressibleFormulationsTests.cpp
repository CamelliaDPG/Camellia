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
  
  v1_vgp = varFactory.testVar(VGP_V1_S, HGRAD);
  v2_vgp = varFactory.testVar(VGP_V2_S, HGRAD);
  tau1_vgp = varFactory.testVar(VGP_TAU1_S, HDIV);
  tau2_vgp = varFactory.testVar(VGP_TAU2_S, HDIV);
  q_vgp = varFactory.testVar(VGP_Q_S, HGRAD);
  
  vgpVarFactory = varFactory;
  
  vgpFields.push_back(u1_vgp);
  vgpFields.push_back(u2_vgp);
  vgpFields.push_back(sigma11_vgp);
  vgpFields.push_back(sigma12_vgp);
  vgpFields.push_back(sigma21_vgp);
  vgpFields.push_back(sigma22_vgp);
  vgpFields.push_back(p_vgp);
  
  vgpTests.push_back(v1_vgp);
  vgpTests.push_back(v2_vgp);
  vgpTests.push_back(tau1_vgp);
  vgpTests.push_back(tau2_vgp);
  vgpTests.push_back(q_vgp);
  
  // set up test containers.  Each test should try every combination here:
  // exact solutions:
  
  PolyExactFunctions exactFxns;
  
//  exactFxns.push_back( make_pair(zero, 0) ); // u1
//  exactFxns.push_back( make_pair(zero, 0) ); // u2 (chosen to have zero divergence)
//  exactFxns.push_back( make_pair(zero, 0) ); // p: odd function: zero mean on our domain
//  polyExactFunctions.push_back(exactFxns);
//  exactFxns.clear();
//
  exactFxns.push_back( make_pair(zero, 0) ); // u1
  exactFxns.push_back( make_pair(zero, 0) ); // u2 (chosen to have zero divergence)
  exactFxns.push_back( make_pair(y, 1) ); // p: odd function: zero mean on our domain
  polyExactFunctions.push_back(exactFxns);
  exactFxns.clear();
  
//  exactFxns.push_back( make_pair(x2, 2) );     // u1
//  exactFxns.push_back( make_pair(-2*x*y, 2) ); // u2 (chosen to have zero divergence)
//  exactFxns.push_back( make_pair(y, 1) );      // p: odd function: zero mean on our domain
//  polyExactFunctions.push_back(exactFxns);
//  exactFxns.clear();
  
  exactFxns.push_back( make_pair(x2 * y, 3) );  // u1
  exactFxns.push_back( make_pair(-x*y2, 3) );   // u2 (chosen to have zero divergence)
  exactFxns.push_back( make_pair(y3, 3) );      // p: odd function: zero mean on our domain
  polyExactFunctions.push_back(exactFxns);
  exactFxns.clear();
  
  // mesh choices (horizontal x vertical cells)
//  meshDimensions.push_back( make_pair(1, 1));
//  meshDimensions.push_back( make_pair(1, 2));
  meshDimensions.push_back( make_pair(2, 1));
  
  pToAddValues.push_back(1);
  
  muValues.push_back(1.0);
  muValues.push_back(0.1);
  
  entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
  
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
}

void IncompressibleFormulationsTests::runTests(int &numTestsRun, int &numTestsPassed) {
  cout << "Running IncompressibleFormulationsTests.  (This takes 30-60 seconds.)" << endl;
  setup();
  if (testVGPNavierStokesFormulationCorrectness()) {
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
  if (testVGPStokesFormulationCorrectness()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testVGPStokesFormulationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool IncompressibleFormulationsTests::functionsAgree(FunctionPtr f1, FunctionPtr f2, Teuchos::RCP<Mesh> mesh) {
  bool functionsAgree = true;
  if (f1->isZero() && f2->isZero()) {
    return true;
  }
  
  if (f1->isZero() != f2->isZero()) {
    cout << "f1->isZero() != f2->isZero()\n";
  }
  
  if (f2->rank() != f1->rank() ) {
    cout << "f1->rank() " << f1->rank() << " != f2->rank() " << f2->rank() << endl;
    return false;
  }
  if (f1->boundaryValueOnly() != f2->boundaryValueOnly()) {
    return false;
  }
  // TODO: rewrite this to compute in distributed fashion
  vector< ElementTypePtr > elementTypes = mesh->elementTypes();
  
  for (vector< ElementTypePtr >::iterator typeIt = elementTypes.begin(); typeIt != elementTypes.end(); typeIt++) {
    ElementTypePtr elemType = *typeIt;
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache( elemType, mesh, false) ); // all elements of same type
    typedef Teuchos::RCP< Element > ElementPtr;
    vector< ElementPtr > cells = mesh->elementsOfTypeGlobal(elemType); // TODO: replace with local variant
    
    int numCells = cells.size();
    vector<int> cellIDs;
    for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
      cellIDs.push_back( cells[cellIndex]->cellID() );
    }
    // TODO: replace with non-global variant...
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodesGlobal(elemType), cellIDs, f1->boundaryValueOnly());
    
    int numSides = 1; // interior only
    if (f1->boundaryValueOnly()) {
      numSides = elemType->cellTopoPtr->getSideCount();
    }
    for (int sideIndex = 0; sideIndex<numSides; sideIndex++) {
      BasisCachePtr basisCacheForTest;
      if (f1->boundaryValueOnly()) {
        basisCacheForTest = basisCache->getSideBasisCache(sideIndex);
      } else {
        basisCacheForTest = basisCache;
      }
      int rank = f1->rank();
      int numPoints = basisCacheForTest->getPhysicalCubaturePoints().dimension(1);
      int spaceDim = basisCacheForTest->getPhysicalCubaturePoints().dimension(2);
      Teuchos::Array<int> dim;
      dim.append(numCells);
      dim.append(numPoints);
      for (int i=0; i<rank; i++) {
        dim.append(spaceDim);
      }
      FieldContainer<double> f1Values(dim);
      FieldContainer<double> f2Values(dim);
      f1->values(f1Values,basisCacheForTest);
      f2->values(f2Values,basisCacheForTest);
      
      double tol = 1e-14;
      double maxDiff;
      functionsAgree = TestSuite::fcsAgree(f1Values,f2Values,tol,maxDiff);
      if ( ! functionsAgree ) {
        functionsAgree = false;
        cout << "f1->displayString(): " << f1->displayString() << endl;
        cout << "f2->displayString(): " << f2->displayString() << endl;
        cout << "Test failed: f1 and f2 disagree; maxDiff " << maxDiff << ".\n";
//        cout << "f1Values: \n" << f1Values;
//        cout << "f2Values: \n" << f2Values;
      } else {
        //    cout << "f1 and f2 agree!" << endl;
      }
    }
  }
  return functionsAgree;
}

bool IncompressibleFormulationsTests::ltsAgree(LinearTermPtr lt1, LinearTermPtr lt2,
                                               Teuchos::RCP<Mesh> mesh, VarFactory &varFactory) {
  bool ltsAgree = true;
  
  if (lt1->isZero() != lt2->isZero()) {
    if (lt1->isZero())
      cout << "lt1 != 0 but lt2 == 0\n";
    else
      cout << "lt1 == 0 but lt2 != 0\n";
    return false;
  }
  
  if (lt1->isZero() && lt2->isZero()) {
    return true;
  }
  
  // disagreement on the varIDs is an unexpected type of issue, so definitely want to output that to console
  set<int> varIDs = lt1->varIDs();
  set<int> varIDs2 = lt2->varIDs();
  if ( varIDs.size() != varIDs2.size() ) {
    cout << "LTs disagree in the # of varIDs\n";
    return false;
  } else {
    set<int>::iterator varIDIt2 = varIDs2.begin();
    for (set<int>::iterator varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++, varIDIt2++) {
      if (*varIDIt != *varIDIt2)  {
        cout << "LTs disagree on the varIDs\n";
        return false;
        
      }
    }
  }
  
  vector< FunctionPtr > fxns;
  fxns.push_back(x);
  fxns.push_back(y);
  fxns.push_back(x2);
  fxns.push_back(y2);
  fxns.push_back(x*y2);
  fxns.push_back(y2*x);
  
  vector< FunctionPtr > vect_fxns;
  vect_fxns.push_back( Function::vectorize(fxns[0], fxns[1]));
  vect_fxns.push_back( Function::vectorize(fxns[1], fxns[2]));
  vect_fxns.push_back( Function::vectorize(fxns[2], fxns[3]));
  vect_fxns.push_back( Function::vectorize(fxns[3], fxns[4]));
  vect_fxns.push_back( Function::vectorize(fxns[4], fxns[5]));
  vect_fxns.push_back( Function::vectorize(fxns[5], fxns[0]));
  
  // strategy is just to take a few Functions and check that as functions lt1 and lt2 agree
  for (int i=0; i< fxns.size(); i++) {
    FunctionPtr fxn = fxns[i];
    FunctionPtr vect_fxn = vect_fxns[i];
    
    for (set<int>::iterator varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++) {
      int varID = *varIDIt;
      map< int, FunctionPtr > var_substitution;
      VarPtr var = (lt1->termType()==TEST) ? varFactory.testVars().find(varID)->second : varFactory.trialVars().find(varID)->second;
      if (var->rank() == 0) {
        var_substitution[varID] = fxn;
      } else if (var->rank()==1) {
        var_substitution[varID] = vect_fxn;
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "We don't yet handle LT comparisons for tensor variables.");
      }

      FunctionPtr lt1_fxn_volume = lt1->evaluate(var_substitution, false);
      FunctionPtr lt2_fxn_volume = lt2->evaluate(var_substitution, false);
      
      if (! functionsAgree(lt1_fxn_volume, lt2_fxn_volume, mesh) ) {
        cout << "lt1 != lt2;\nlt1: " <<lt1->displayString() << "\nlt2: " <<lt2->displayString();
        return false;
      }
      
      FunctionPtr lt1_fxn_boundary = lt1->evaluate(var_substitution, true);
      FunctionPtr lt2_fxn_boundary = lt2->evaluate(var_substitution, true);
      
      if (! functionsAgree(lt1_fxn_boundary, lt2_fxn_boundary, mesh) ) {
        return false;
      }
    }
  }
  
  return ltsAgree;
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
      
          Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
        
          // create a pointer to a new mesh:
          Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                               vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
          
          FunctionPtr u1_exact = exactFxns[0].first;
          FunctionPtr u2_exact = exactFxns[1].first;
          FunctionPtr p_exact  = exactFxns[2].first;
          
          SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
          
          Teuchos::RCP<ExactSolution> vgpStokesExactSolution = vgpStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
          
          BCPtr vgpBC = vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
          
          Teuchos::RCP< Solution > vgpStokesSolution = Teuchos::rcp( new Solution(mesh, vgpBC,
                                                                                  vgpStokesExactSolution->rhs(),
                                                                                  vgpStokesFormulation->graphNorm()));
          
          vgpStokesSolution->solve();
          
          int cubatureDegree = maxPolyOrder;
          for (vector< VarPtr >::iterator fieldIt = vgpFields.begin(); fieldIt != vgpFields.end(); fieldIt++ ) {
            VarPtr field = *fieldIt;
            double l2Error = vgpStokesExactSolution->L2NormOfError(*vgpStokesSolution, field->ID(), cubatureDegree);
            if (l2Error > tol) {
              success = false;
              cout << "FAILURE: testVGPStokesFormulationConsistency(): ";
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
  
  Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation;
  Teuchos::RCP< Solution > vgpStokesSolution;
  Teuchos::RCP<ExactSolution> vgpStokesExactSolution;
  
//  cout << "Warning: testVGPStokesFormulationCorrectness() is trivial.\n";
  
  // consistency: check that the RHS is correct.  (Could also check other things,
  //              like individual solution components, etc., but RHS is the thing we're
  //              most liable to get incorrect.)
  
  // exact solution functions: store these as vector< pair< Function, int > >
  // in the order u1, u2, p, where the paired int is the polynomial degree of the function
  
  for (vector< PolyExactFunctions >::iterator exactIt = polyExactFunctions.begin();
       exactIt != polyExactFunctions.end(); exactIt++) {
    PolyExactFunctions exactFxns = *exactIt;
    FunctionPtr u1_exact = exactFxns[0].first;
    FunctionPtr u2_exact = exactFxns[1].first;
    FunctionPtr  p_exact = exactFxns[2].first;
    
    int maxPolyOrder = max(0,exactFxns[0].second);
    maxPolyOrder = max(maxPolyOrder,exactFxns[1].second);
    maxPolyOrder = max(maxPolyOrder,exactFxns[2].second);
    
    for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
      double mu = *muIt;
      
      vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
      
      // f = - grad p + mu delta u
      FunctionPtr f1 = - p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy());
      FunctionPtr f2 = - p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy());
      LinearTermPtr expectedRHS = f1 * v1_vgp + f2 * v2_vgp;
      
      vgpStokesExactSolution = vgpStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
      
      RHSPtr rhs = vgpStokesExactSolution->rhs();
      // this is a bit ugly, in that we're assuming the RHS is actually a subclass of RHSEasy
      // (but that's true!)
      LinearTermPtr rhsLT = ((RHSEasy *)rhs.get())->linearTerm();
      
      int H1Order = maxPolyOrder + 1;
      int pToAdd = 1;
      Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, 1, 1,
                                                    vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, vgpVarFactory)) {
        success = false;
      }
    }
  }
  
  return success;
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulationConsistency() {
  bool success = true;
    
  // consistency: check that solving using the BF, RHS, BCs, etc. in the Formulation
  //              gives the ExactSolution specified by the Formulation
  
  // starting out with a single hard-coded solve, but will switch soon to
  // doing several: varying meshes, pToAdd, mu, and which exact solutions we use...
  
  double tol = 1e-11;
  bool printToConsole = false;
    
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
    
    FunctionPtr u1_exact = exactFxns[0].first;
    FunctionPtr u2_exact = exactFxns[1].first;
    FunctionPtr p_exact  = exactFxns[2].first;

    if (printToConsole) {
      cout << "VGP Navier-Stokes consistency tests for exact solution:\n";
      cout << "u1_exact: " << u1_exact->displayString() << endl;
      cout << "u2_exact: " << u2_exact->displayString() << endl;
      cout << "p_exact: " << p_exact->displayString() << endl;
    }
    
    int H1Order = maxPolyOrder + 1;
    for (vector<int>::iterator pToAddIt = pToAddValues.begin(); pToAddIt != pToAddValues.end(); pToAddIt++) {
      int pToAdd = *pToAddIt;
      
      for (vector< pair<int, int> >::iterator meshDimIt = meshDimensions.begin(); meshDimIt != meshDimensions.end(); meshDimIt++) {
        int horizontalCells = meshDimIt->first, verticalCells = meshDimIt->second;
        
        for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
          double mu = *muIt;
          
          Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
          
          // create a pointer to a new mesh:
          Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                        vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
          
          
          
          SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered ); // SpatialFilterUnfiltered returns true everywhere
          
          Teuchos::RCP<ExactSolution> vgpStokesExactSolution = vgpStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
          
          BCPtr vgpBC = vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
          
          Teuchos::RCP<Mesh> stokesMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                              vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
          
          Teuchos::RCP< Solution > vgpNavierStokesSolution = Teuchos::rcp( new Solution(stokesMesh, vgpBC) );
          
          // the incremental solutions have zero BCs enforced:
          BCPtr zeroBC = vgpStokesFormulation->bc(zero, zero, entireBoundary);
          Teuchos::RCP< Solution > vgpNavierStokesSolutionIncrement = Teuchos::rcp( new Solution(stokesMesh, zeroBC) );
          
          Teuchos::RCP< VGPNavierStokesFormulation > vgpNavierStokesFormulation = Teuchos::rcp( new VGPNavierStokesFormulation(1.0 / mu,
                                                                                                                               vgpNavierStokesSolution));
          
          Teuchos::RCP<ExactSolution> vgpNavierStokesExactSolution = vgpNavierStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
          vgpNavierStokesSolution->setRHS( vgpNavierStokesExactSolution->rhs() );
          vgpNavierStokesSolution->setIP( vgpNavierStokesFormulation->graphNorm() );
          
          // solve with the true BCs
          vgpNavierStokesSolution->solve();
          
          vgpNavierStokesSolutionIncrement->setRHS( vgpNavierStokesExactSolution->rhs() );
          vgpNavierStokesSolutionIncrement->setIP( vgpNavierStokesFormulation->graphNorm() );
          
          int numIters = 20;
          for (int i=0; i<numIters; i++) {
            vgpNavierStokesSolutionIncrement->solve();
            vgpNavierStokesSolution->addSolution(vgpNavierStokesSolutionIncrement, 1.0); // optimistic?
          }
          
          int cubatureDegree = maxPolyOrder;
          for (vector< VarPtr >::iterator fieldIt = vgpFields.begin(); fieldIt != vgpFields.end(); fieldIt++ ) {
            VarPtr field = *fieldIt;
            double l2Error = vgpNavierStokesExactSolution->L2NormOfError(*vgpNavierStokesSolution, field->ID(), cubatureDegree);
            if (l2Error > tol) {
              success = false;
              cout << "testVGPNavierStokesFormulationConsistency(): ";
              cout << "L^2 error of " << l2Error << " for variable " << field->displayString();
              cout << " exceeds tol " << tol << endl;
            } else {
//              cout << "PASS: testVGPStokesFormulationConsistency(): ";
//              cout << "L^2 error of " << l2Error << " for variable " << field->displayString();
//              cout << " is within tolerance! " << tol << endl;
            }
          }
        }
      }
    }
  }
  return success;
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulationCorrectness() {
  bool success = true;
  bool printToConsole = false;
  
  Teuchos::RCP< VGPStokesFormulation > vgpStokesFormulation;
  Teuchos::RCP< Solution > vgpStokesSolution;
  Teuchos::RCP<ExactSolution> vgpStokesExactSolution;
  
  //  cout << "Warning: testVGPStokesFormulationCorrectness() is trivial.\n";
  
  // consistency: check that the RHS is correct.  (Could also check other things,
  //              like individual solution components, etc., but RHS is the thing we're
  //              most liable to get incorrect.)
  
  // exact solution functions: store these as vector< pair< Function, int > >
  // in the order u1, u2, p, where the paired int is the polynomial degree of the function
  
  for (vector< PolyExactFunctions >::iterator exactIt = polyExactFunctions.begin();
       exactIt != polyExactFunctions.end(); exactIt++) {
    PolyExactFunctions exactFxns = *exactIt;
    FunctionPtr u1_exact = exactFxns[0].first;
    FunctionPtr u2_exact = exactFxns[1].first;
    FunctionPtr  p_exact = exactFxns[2].first;
    
    if (printToConsole) {
      cout << "VGP Navier-Stokes correctness tests for exact solution:\n";
      cout << "u1_exact: " << u1_exact->displayString() << endl;
      cout << "u2_exact: " << u2_exact->displayString() << endl;
      cout << "p_exact: " << p_exact->displayString() << endl;
    }
    
    int maxPolyOrder = max(0,exactFxns[0].second);
    maxPolyOrder = max(maxPolyOrder,exactFxns[1].second);
    maxPolyOrder = max(maxPolyOrder,exactFxns[2].second);
    
    for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
      double mu = *muIt;
      
      vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
      int H1Order = maxPolyOrder + 1;
      int pToAdd = 1;
      Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, 1, 1,
                                                    vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
      
      BCPtr vgpBC = vgpStokesFormulation->bc(u1_exact, u2_exact, entireBoundary);
      Teuchos::RCP< Solution > vgpNavierStokesSolution = Teuchos::rcp( new Solution(mesh, vgpBC) );
      Teuchos::RCP< VGPNavierStokesFormulation > vgpNavierStokesFormulation = Teuchos::rcp( new VGPNavierStokesFormulation(1.0 / mu,
                                                                                                                           vgpNavierStokesSolution));
      
      Teuchos::RCP< ExactSolution > vgpNavierStokesExactSoln = vgpNavierStokesFormulation->exactSolution(u1_exact, u2_exact, p_exact, entireBoundary);
      RHSPtr rhs = vgpNavierStokesExactSoln->rhs();
      // this is a bit ugly, in that we're assuming the RHS is actually a subclass of RHSEasy
      // (but that's true!)
      LinearTermPtr rhsLT = ((RHSEasy *)rhs.get())->linearTerm();
      
      // f = - grad p + mu delta u - u * grad u
      FunctionPtr f1 = - p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy())
                       - u1_exact * u1_exact->dx() - u2_exact * u1_exact->dy();
      FunctionPtr f2 = - p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy())
                       - u1_exact * u2_exact->dx() - u2_exact * u2_exact->dy();
      
      // define the previous solution terms we need:
      FunctionPtr u1_prev = Function::solution(u1_vgp, vgpNavierStokesSolution);
      FunctionPtr u2_prev = Function::solution(u2_vgp, vgpNavierStokesSolution);
      FunctionPtr sigma11_prev = Function::solution(sigma11_vgp, vgpNavierStokesSolution);
      FunctionPtr sigma12_prev = Function::solution(sigma12_vgp, vgpNavierStokesSolution);
      FunctionPtr sigma21_prev = Function::solution(sigma21_vgp, vgpNavierStokesSolution);
      FunctionPtr sigma22_prev = Function::solution(sigma22_vgp, vgpNavierStokesSolution);
      
      LinearTermPtr expectedRHS = f1 * v1_vgp + f2 * v2_vgp;
      expectedRHS = expectedRHS + (u1_prev * sigma11_prev + u2_prev * sigma12_prev) * v1_vgp;
      expectedRHS = expectedRHS + (u1_prev * sigma21_prev + u2_prev * sigma22_prev) * v2_vgp;
      
      BFPtr stokesBF = vgpNavierStokesFormulation->stokesBF(mu);
      expectedRHS = expectedRHS - stokesBF->testFunctional(vgpNavierStokesSolution);
            
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, vgpVarFactory)) {
        cout << "Failure: Navier-Stokes correctedness: before first solve (i.e. with zero background flow), LTs disagree\n";
        success = false;
      }
      
      vgpNavierStokesSolution->setRHS(rhs);
      vgpNavierStokesSolution->setIP(vgpNavierStokesFormulation->graphNorm());
      vgpNavierStokesSolution->solve();
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, vgpVarFactory)) {
        cout << "Failure: Navier-Stokes correctedness: after first solve (i.e. with non-zero background flow), LTs disagree\n";
        success = false;
      }
    }
  }
 
  return success;
}

std::string IncompressibleFormulationsTests::testSuiteName() {
  return "IncompressibleFormulationsTests";
}
