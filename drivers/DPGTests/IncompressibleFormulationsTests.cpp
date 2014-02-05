#include "IncompressibleFormulationsTests.h"

#include "HessianFilter.h"
#include "RieszRep.h"
#include "ParameterFunction.h"
#include "LagrangeConstraints.h"

#include "SolutionExporter.h"
#include "MeshFactory.h"

IncompressibleFormulationsTests::IncompressibleFormulationsTests(bool thorough) {
  _thoroughMode = thorough;
}

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
  
  u1hat_vgp = vgpVarFactory.traceVar(VGP_U1HAT_S);
  u2hat_vgp = vgpVarFactory.traceVar(VGP_U2HAT_S);
  t1n_vgp = vgpVarFactory.fluxVar(VGP_T1HAT_S);
  t2n_vgp = vgpVarFactory.fluxVar(VGP_T2HAT_S);
  
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
  
  polyExactFunctions.clear();
  
  PolyExactFunctions exactFxns;
  
//  exactFxns.push_back( make_pair(zero, 0) ); // u1
//  exactFxns.push_back( make_pair(zero, 0) ); // u2 (chosen to have zero divergence)
//  exactFxns.push_back( make_pair(zero, 0) ); // p: odd function: zero mean on our domain
//  polyExactFunctions.push_back(exactFxns);
//  exactFxns.clear();
//
  if (_thoroughMode) {
    exactFxns.push_back( make_pair(zero, 0) ); // u1
    exactFxns.push_back( make_pair(zero, 0) ); // u2 (chosen to have zero divergence)
    exactFxns.push_back( make_pair(y, 1) ); // p: odd function: zero mean on our domain
    polyExactFunctions.push_back(exactFxns);
    exactFxns.clear();

    // nonzero u, sigma diagonal only
    exactFxns.push_back( make_pair(x, 1) );    // u1
    exactFxns.push_back( make_pair(-y, 1) );   // u2 (chosen to have zero divergence)
    exactFxns.push_back( make_pair(zero, 0) );
    polyExactFunctions.push_back(exactFxns);
    exactFxns.clear();
    
    // nonzero u, sigma off-diagonal only
    exactFxns.push_back( make_pair(y, 1) ); // u1
    exactFxns.push_back( make_pair(x, 1) ); // u2 (chosen to have zero divergence)
    exactFxns.push_back( make_pair(zero, 0) );
    polyExactFunctions.push_back(exactFxns);
    exactFxns.clear();

    // for some reason, this one and only this one fails to converge when using the Hessian term:
    // (fails in testVGPNavierStokesFormulationConsistency)
//    exactFxns.push_back( make_pair(x2, 2) );     // u1
//    exactFxns.push_back( make_pair(-2*x*y, 2) ); // u2 (chosen to have zero divergence)
//    exactFxns.push_back( make_pair(y, 1) );      // p: odd function: zero mean on our domain
//    polyExactFunctions.push_back(exactFxns);
//    exactFxns.clear();
  }
  
  exactFxns.push_back( make_pair(x2 * y, 3) );  // u1
  exactFxns.push_back( make_pair(-x*y2, 3) );   // u2 (chosen to have zero divergence)
  exactFxns.push_back( make_pair(y3, 3) );      // p: odd function: zero mean on our domain
  polyExactFunctions.push_back(exactFxns);
  exactFxns.clear();
  
  // mesh choices (horizontal x vertical cells)
  meshDimensions.clear();
  if (_thoroughMode) {
    meshDimensions.push_back( make_pair(1, 1));
    meshDimensions.push_back( make_pair(1, 2));
  }
  meshDimensions.push_back( make_pair(2, 1));
  
  pToAddValues.clear();
  pToAddValues.push_back(2);
  
  if ( _thoroughMode ) {
    double mu = 1.0;
    muValues.clear();
    for (int i=0; i<4; i++) {
      muValues.push_back(mu);
      mu /= 10;
    }
  } else {
    // just do mu = 0.1:
    muValues.clear();
    muValues.push_back(0.1);
  }
  
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
  
  quadPointsKovasznay.resize(4,2);
  quadPointsKovasznay(0,0) = -0.5; // x1
  quadPointsKovasznay(0,1) =  0.0; // y1
  quadPointsKovasznay(1,0) =  1.5;
  quadPointsKovasznay(1,1) =  0.0;
  quadPointsKovasznay(2,0) =  1.5;
  quadPointsKovasznay(2,1) =  2.0;
  quadPointsKovasznay(3,0) = -0.5;
  quadPointsKovasznay(3,1) =  2.0;
}

void IncompressibleFormulationsTests::teardown() {
  polyExactFunctions.clear();
  meshDimensions.clear();
  pToAddValues.clear();
  muValues.clear();
  vgpFields.clear();
  vgpTests.clear();
}

void IncompressibleFormulationsTests::runTests(int &numTestsRun, int &numTestsPassed) {
  if (_thoroughMode) {
    cout << "Running IncompressibleFormulationsTests in thorough mode.  (This can take a long while...)" << endl;
  } else {
    cout << "Running IncompressibleFormulationsTests in non-thorough mode.  (This can still take a while...)" << endl;
  }
  
  setup();
  if ( testVGPNavierStokesLocalConservation() ) {
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
  
  cout << "testVGPNavierStokesFormulationCorrectness completed\n";
  
  setup();
  if (testVGPStokesFormulationGraphNorm()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  cout << "testVGPStokesFormulationGraphNorm completed\n";
  
  setup();
  if (testVGPNavierStokesFormulationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testVGPNavierStokesFormulationKovasnayConvergence()) {
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
  
  setup();
  if (testVGPStokesFormulationCorrectness()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testVVPStokesFormulationGraphNorm()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool IncompressibleFormulationsTests::functionsAgree(FunctionPtr f1, FunctionPtr f2,
                                                     Teuchos::RCP<Mesh> mesh, double tol) {
  // if one of the functions is boundaryValueOnly, we compare the functions on the boundary.
  bool boundaryValueOnly = f1->boundaryValueOnly() || f2->boundaryValueOnly();
  bool functionsAgree = true;
  double pointwiseTol = tol * 10; // relax tolerance somewhat for pointwise comparisons
  
  if (f2->rank() != f1->rank() ) {
    cout << "f1->rank() " << f1->rank() << " != f2->rank() " << f2->rank() << endl;
    return false;
  }
  if (f1->isZero() && f2->isZero()) {
    // then we can exit early.  Note that not every zero function knows it is zero,
    // so we can't return false just because f1->isZero() != f2->isZero().
    return true;
  }
  
  FunctionPtr diff = f1-f2;
  double l2NormOfDifference = sqrt((diff * diff)->integrate(mesh));
  
  if (l2NormOfDifference > tol) {
    cout << "f1 != f2; L^2 norm of difference on mesh = " << l2NormOfDifference << endl;
    return false;
  }

  // TODO: rewrite this to compute in distributed fashion
  vector< ElementTypePtr > elementTypes = mesh->elementTypes();
  
  for (vector< ElementTypePtr >::iterator typeIt = elementTypes.begin(); typeIt != elementTypes.end(); typeIt++) {
    ElementTypePtr elemType = *typeIt;
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache( elemType, mesh, false) ); // all elements of same type
    vector< ElementPtr > cells = mesh->elementsOfTypeGlobal(elemType); // TODO: replace with local variant
    
    int numCells = cells.size();
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
      cellIDs.push_back( cells[cellIndex]->cellID() );
    }
    // TODO: replace with non-global variant...
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodesGlobal(elemType), cellIDs, boundaryValueOnly);
    
    int numSides = 1; // interior only
    if (f1->boundaryValueOnly()) {
      numSides = elemType->cellTopoPtr->getSideCount();
    }
    for (int sideIndex = 0; sideIndex<numSides; sideIndex++) {
      BasisCachePtr basisCacheForTest;
      if (boundaryValueOnly) {
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
    
      double maxDiff;
      functionsAgree = TestSuite::fcsAgree(f1Values,f2Values,pointwiseTol,maxDiff);
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

// tests norm of the difference
bool IncompressibleFormulationsTests::ltsAgree(LinearTermPtr lt1, LinearTermPtr lt2,
                                               Teuchos::RCP<Mesh> mesh, IPPtr ip, double tol) {
//  cout << "lt1: " << lt1->displayString() << endl;
//  cout << "lt2: " <<lt2->displayString() << endl;
  if (lt1->isZero() && lt2->isZero()) return true;
  LinearTermPtr diff = lt1-lt2;
  Teuchos::RCP<RieszRep> rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, diff) );
  rieszRep->computeRieszRep();
  double norm = rieszRep->getNorm();
  if (norm > tol) {
    cout << "norm of the difference between LTs is " << norm << endl;
    
    // to help get a little insight into where the difference is:
    LinearTermPtr boundaryDiff = lt1->getBoundaryOnlyPart() - lt2->getBoundaryOnlyPart();
    Teuchos::RCP<RieszRep> boundaryRep = Teuchos::rcp( new RieszRep(mesh, ip, boundaryDiff) );
    boundaryRep->computeRieszRep();
    cout << "norm of the difference of boundary-only parts is " << boundaryRep->getNorm() << endl;
    
    
    LinearTermPtr nonBoundaryDiff = lt1->getNonBoundaryOnlyPart() - lt2->getNonBoundaryOnlyPart();
    Teuchos::RCP<RieszRep> nonBoundaryRep = Teuchos::rcp( new RieszRep(mesh, ip, nonBoundaryDiff) );
    nonBoundaryRep->computeRieszRep();
    cout << "norm of the difference of non-boundary-only parts is " << nonBoundaryRep->getNorm() << endl;
    
    vector< VarPtr > nonZeros = nonZeroComponents(diff, vgpTests, mesh, ip);
    cout << "difference is non-zero in components: " << endl;
    for (vector< VarPtr >::iterator varIt = nonZeros.begin(); varIt != nonZeros.end(); varIt++) {
      VarPtr var = *varIt;
      cout << var->displayString() << ": " << diff->getPartMatchingVariable(var)->displayString() << endl;
    }
    return false;
  } else {
    return true;
  }
}

// this one tests "pointwise"--i.e. evaluated at functions.  The other one tests the norm of the difference
bool IncompressibleFormulationsTests::ltsAgree(LinearTermPtr lt1, LinearTermPtr lt2,
                                               Teuchos::RCP<Mesh> mesh, VarFactory &varFactory,
                                               double tol) {
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
      
      if (! functionsAgree(lt1_fxn_volume, lt2_fxn_volume, mesh, tol) ) {
        cout << "lt1 != lt2;\nlt1: " <<lt1->displayString() << "\nlt2: " <<lt2->displayString();
        return false;
      }
      
      FunctionPtr lt1_fxn_boundary = lt1->evaluate(var_substitution, true);
      FunctionPtr lt2_fxn_boundary = lt2->evaluate(var_substitution, true);
      
      if (! functionsAgree(lt1_fxn_boundary, lt2_fxn_boundary, mesh, tol) ) {
        return false;
      }
    }
  }
  
  return ltsAgree;
}

vector< VarPtr > IncompressibleFormulationsTests::nonZeroComponents( LinearTermPtr lt, vector< VarPtr > &varsToTry, MeshPtr mesh, IPPtr ip ) {
  vector< VarPtr > nonZeros;
  double tol = 1e-11;
  for (vector< VarPtr >::iterator varIt = varsToTry.begin(); varIt != varsToTry.end(); varIt++) {
    VarPtr var = *varIt;
    LinearTermPtr ltFiltered = lt->getPartMatchingVariable(var);
    Teuchos::RCP<RieszRep> rieszRep = Teuchos::rcp( new RieszRep(mesh, ip, ltFiltered) );
    rieszRep->computeRieszRep();
    double norm = rieszRep->getNorm();
    cout << "part matching var " << var->displayString() << " is: " << ltFiltered->displayString() << endl;
    cout << "norm is " << norm << endl;
    if (norm > tol) {
      nonZeros.push_back(var);
    }
  }
  return nonZeros;
}

bool IncompressibleFormulationsTests::testVGPStokesFormulationConsistency() {
  bool success = true;
  
  // consistency: check that solving using the BF, RHS, BCs, etc. in the Formulation
  //              gives the ExactSolution specified by the Formulation
  
  // starting out with a single hard-coded solve, but will switch soon to
  // doing several: varying meshes, pToAdd, mu, and which exact solutions we use...
    
  double tol = 2e-11;
  
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
          Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
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
          
          FunctionPtr u1_soln = Function::solution(u1_vgp, vgpStokesSolution);
          FunctionPtr u2_soln = Function::solution(u2_vgp, vgpStokesSolution);
          FunctionPtr p_soln  = Function::solution( p_vgp, vgpStokesSolution);
          
          if ( ! functionsAgree(u1_soln, u1_exact, mesh, tol) ) {
            cout << "FAILURE: testVGPStokesFormulationConsistency(): after solve, u1_soln != u1_exact.\n";
          }
          if ( ! functionsAgree(u2_soln, u2_exact, mesh, tol) ) {
            cout << "FAILURE: testVGPStokesFormulationConsistency(): after solve, u2_soln != u2_exact.\n";
          }
          if ( ! functionsAgree(p_soln, p_exact, mesh, tol) ) {
            cout << "FAILURE: testVGPStokesFormulationConsistency(): after solve, p_soln != p_exact.\n";
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
      Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, 1, 1,
                                                    vgpStokesFormulation->bf(), H1Order, H1Order+pToAdd);
      
      double tol = 1e-14;
      if (maxPolyOrder >= 10) {
        // a bit of a cheat: this means it's the Kovasznay solution, which we won't get exact:
        tol = 1e-10;
      }
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, vgpVarFactory, tol)) {
        success = false;
      }
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, vgpStokesFormulation->bf()->graphNorm(), tol)) {
        cout << "VGP Stokes Correctness Failure: rhsLT and expectedRHS differ, according to RieszRep norm.\n";
        success = false;
      }
    }
  }
  
  return success;
}

bool IncompressibleFormulationsTests::testVGPStokesFormulationGraphNorm() {
  bool success = true;
  
  double tol = 1e-15;
  
  for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
    double mu = *muIt;
    IPPtr ipExpected = Teuchos::rcp( new IP );
    { // setup ipExpected:
      // div tau - grad q
      ipExpected->addTerm(tau1_vgp->div() - q_vgp->dx());
      ipExpected->addTerm(tau2_vgp->div() - q_vgp->dy());
      // - grad(mu v) + tau
      ipExpected->addTerm(-mu * v1_vgp->dx() + tau1_vgp->x()); // mu v1,x + tau11
      ipExpected->addTerm(-mu * v2_vgp->dx() + tau2_vgp->x()); // mu v2,x + tau21
      ipExpected->addTerm(-mu * v1_vgp->dy() + tau1_vgp->y()); // mu v1,y + tau12
      ipExpected->addTerm(-mu * v2_vgp->dy() + tau2_vgp->y()); // mu v2,y + tau22
      // div v
      ipExpected->addTerm(v1_vgp->dx() + v2_vgp->dy());
      // v
      ipExpected->addTerm(v1_vgp);
      ipExpected->addTerm(v2_vgp);
      // tau
      ipExpected->addTerm(tau1_vgp);
      ipExpected->addTerm(tau2_vgp);
      // q
      ipExpected->addTerm(q_vgp);
    }
  
    VGPStokesFormulation vgpStokesFormulation(mu,false,false); // don't enrich velocity, don't scale sigma by mu
    IPPtr ipActual = vgpStokesFormulation.graphNorm();
    
    int horizontalCells = 2, verticalCells = 2;
    int H1Order = 1, pToAdd = 1;
    // create a pointer to a new mesh:
    Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                  vgpStokesFormulation.bf(), H1Order, H1Order+pToAdd);
    ElementTypePtr elemType = mesh->getElement(0)->elementType(); // all elements have same type here
    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(mesh, elemType);
    
    if (basisCache->cellIDs().size() == 0) continue;
    
    int testDofs = elemType->testOrderPtr->totalDofs();
    FieldContainer<double> ipMatrixExpected(basisCache->cellIDs().size(),testDofs,testDofs);
    FieldContainer<double> ipMatrixActual(basisCache->cellIDs().size(),testDofs,testDofs);
    
    ipExpected->computeInnerProductMatrix(ipMatrixExpected, elemType->testOrderPtr, basisCache);
    ipActual->computeInnerProductMatrix(ipMatrixActual, elemType->testOrderPtr, basisCache);
    
    double maxDiff = 0;
    if (!fcsAgree(ipMatrixExpected, ipMatrixActual, tol, maxDiff)) {
      success = false;
      cout << "testVGPStokesFormulationGraphNorm: IPs disagree with maxDiff " << maxDiff << endl;
      
      cout << "ipExpected:\n";
      ipExpected->printInteractions();
      cout << "ipActual:\n";
      ipActual->printInteractions();
    }
//    cout << "maxDiff = " << maxDiff << endl;
//    cout << "ipMatrixExpected:\n" << ipMatrixExpected;
  }
  
  return allSuccess(success);
}
bool IncompressibleFormulationsTests::testVVPStokesFormulationGraphNorm() {
  bool success = true;
  
  double tol = 1e-15;

  bool trueTraces = false; // shouldn't matter
  
  VarFactory vvpVarFactory = VVPStokesFormulation::vvpVarFactory(trueTraces);
  
  // look up the created VarPtrs:
  VarPtr v = vvpVarFactory.testVar(VVP_V_S, VECTOR_HGRAD);
  VarPtr q1 = vvpVarFactory.testVar(VVP_Q1_S, HGRAD);
  VarPtr q2 = vvpVarFactory.testVar(VVP_Q2_S, HGRAD);
  
//  if (!trueTraces) {
//    u1hat = varFactory.traceVar(VVP_U1HAT_S);
//    u2hat = varFactory.traceVar(VVP_U2HAT_S);
//  } else {
//    u_n = varFactory.fluxVar(VVP_U_DOT_HAT_S);
//    u_xn = varFactory.fluxVar(VVP_U_CROSS_HAT_S);
//  }
//  omega_hat = varFactory.traceVar(VVP_OMEGA_HAT_S);
//  p_hat = varFactory.traceVar(VVP_P_HAT_S);
//  
//  u1 = varFactory.fieldVar(VVP_U1_S);
//  u2 = varFactory.fieldVar(VVP_U2_S);
//  omega = varFactory.fieldVar(VVP_OMEGA_S);
//  p = varFactory.fieldVar(VVP_P_S);
  
  for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
    double mu = *muIt;
    VVPStokesFormulation vvpStokesFormulation(mu);
    
    IPPtr ipExpected = Teuchos::rcp( new IP );
    { // setup ipExpected:
      // curl (mu v) + q1
      ipExpected->addTerm(mu * v->curl() + q1);
      // div v
      ipExpected->addTerm(v->div());
      // grad q2 + curl q1
      ipExpected->addTerm(q2->grad() + q1->curl());
      // v
      ipExpected->addTerm(v);
      // q1
      ipExpected->addTerm(q1);
      // q2
      ipExpected->addTerm(q2);
    }
    
    IPPtr ipActual = vvpStokesFormulation.graphNorm();
    
    int horizontalCells = 2, verticalCells = 2;
    int H1Order = 1, pToAdd = 1;
    // create a pointer to a new mesh:
    Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                  vvpStokesFormulation.bf(), H1Order, H1Order+pToAdd);
    ElementTypePtr elemType = mesh->getElement(0)->elementType(); // all elements have same type here
    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(mesh, elemType);
    
    if (basisCache->cellIDs().size() == 0) continue;
    
    int testDofs = elemType->testOrderPtr->totalDofs();
    FieldContainer<double> ipMatrixExpected(basisCache->cellIDs().size(),testDofs,testDofs);
    FieldContainer<double> ipMatrixActual(basisCache->cellIDs().size(),testDofs,testDofs);
    
    ipExpected->computeInnerProductMatrix(ipMatrixExpected, elemType->testOrderPtr, basisCache);
    ipActual->computeInnerProductMatrix(ipMatrixActual, elemType->testOrderPtr, basisCache);
    
    double maxDiff = 0;
    if (!fcsAgree(ipMatrixExpected, ipMatrixActual, tol, maxDiff)) {
      success = false;
      cout << "testVVPStokesFormulationGraphNorm: IPs disagree with maxDiff " << maxDiff << endl;
    }
//    cout << "testVVPStokesFormulationGraphNorm: maxDiff = " << maxDiff << endl;
    //    cout << "ipMatrixExpected:\n" << ipMatrixExpected;
  }
  
  return allSuccess(success);
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulationConsistency() {
  bool success = true;
  bool printToConsole = false;
    
  // consistency: check that solving using the BF, RHS, BCs, etc. in the Formulation
  //              gives the ExactSolution specified by the Formulation
  
  // starting out with a single hard-coded solve, but will switch soon to
  // doing several: varying meshes, pToAdd, mu, and which exact solutions we use...
  
  double tol = 2e-11;
  
  bool useLineSearch = false;
  bool useCondensedSolve = true;
  bool enrichVelocity = false; // true would be for the "compliant" norm, which isn't working well yet
  
  vector<bool> useHessianList;
  useHessianList.push_back(false);
  useHessianList.push_back(true);
  
  // exact solution functions: store these as vector< pair< Function, int > >
  // in the order u1, u2, p, where the paired int is the polynomial degree of the function
  
  for (vector<bool>::iterator useHessianIt = useHessianList.begin();
       useHessianIt != useHessianList.end(); useHessianIt++) {
    bool useHessian = *useHessianIt;
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
            double Re = 1 / mu;
            
            bool dontEnhanceFluxes = false;
            VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re, quadPoints,
                                                                    horizontalCells, verticalCells,
                                                                    H1Order, pToAdd,
                                                                    u1_exact, u2_exact, p_exact, enrichVelocity, dontEnhanceFluxes);
            
            
            SolutionPtr solnIncrement = problem.solutionIncrement();
            SolutionPtr backgroundFlow = problem.backgroundFlow();
            
            // solve once:
            Teuchos::RCP<ExactSolution> exactSolution = problem.exactSolution();
            MeshPtr mesh = problem.mesh();
            
            Teuchos::RCP< RieszRep > rieszRep;
            if (useHessian) {
              LinearTermPtr rhsLT = ((RHSEasy*) exactSolution->rhs().get())->linearTerm();
              IPPtr ip = problem.bf()->graphNorm();
              rieszRep = Teuchos::rcp( new RieszRep(mesh,ip,rhsLT));
              FunctionPtr v1_rep = Teuchos::rcp( new RepFunction(v1_vgp, rieszRep) );
              FunctionPtr v2_rep = Teuchos::rcp( new RepFunction(v2_vgp, rieszRep) );
              // set up the hessian term itself:
              // we want basically u * sigma * v where "*" is a dot product
              // u * sigma = (u1 sigma11 + u2 sigma12, u1 sigma21 + u2 sigma22)
              BFPtr hessianBF = Teuchos::rcp( new BF(vgpVarFactory.getBubnovFactory(VarFactory::BUBNOV_TRIAL)) );
              hessianBF->addTerm(v1_rep * u1_vgp, sigma11_vgp);
              hessianBF->addTerm(v1_rep * u2_vgp, sigma12_vgp);
              hessianBF->addTerm(v2_rep * u1_vgp, sigma21_vgp);
              hessianBF->addTerm(v2_rep * u2_vgp, sigma22_vgp);
              // now the symmetric counterparts:
              hessianBF->addTerm(sigma11_vgp, v1_rep * u1_vgp);
              hessianBF->addTerm(sigma12_vgp, v1_rep * u2_vgp);
              hessianBF->addTerm(sigma21_vgp, v2_rep * u1_vgp);
              hessianBF->addTerm(sigma22_vgp, v2_rep * u2_vgp);
              
              Teuchos::RCP< HessianFilter > hessianFilter = Teuchos::rcp( new HessianFilter(hessianBF) );
              solnIncrement->setFilter(hessianFilter);
              rieszRep->computeRieszRep();
            }
            
            int maxIters = 100;

            FunctionPtr u1_incr = Function::solution(u1_vgp, solnIncrement);
            FunctionPtr u2_incr = Function::solution(u2_vgp, solnIncrement);
            FunctionPtr p_incr = Function::solution(p_vgp, solnIncrement);
            
            FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr;
            
            do {
              problem.iterate(useLineSearch,useCondensedSolve);
              if ( rieszRep.get() ) {
                rieszRep->computeRieszRep();
              }
            }  while ( (sqrt(l2_incr->integrate(mesh)) > tol) && (problem.iterationCount() < maxIters) );
            if (printToConsole) {
              string withHessian = useHessian ? "using hessian term" : "without hessian term";
              cout << "with Re = " << 1.0 / mu << " and " << withHessian;
              cout << ", # iters to converge: " << problem.iterationCount() << endl;
            }
            
            int cubatureDegree = maxPolyOrder;
            for (vector< VarPtr >::iterator fieldIt = vgpFields.begin(); fieldIt != vgpFields.end(); fieldIt++ ) {
              VarPtr field = *fieldIt;
              double l2Error = exactSolution->L2NormOfError(*backgroundFlow, field->ID(), cubatureDegree);
              if (l2Error > tol) {
                success = false;
                cout << "testVGPNavierStokesFormulationConsistency(): ";
                cout << "L^2 error of " << l2Error << " for variable " << field->displayString();
                cout << " exceeds tol " << tol << endl;
                string withHessian = useHessian ? "using hessian term" : "without hessian term";
                cout << "Failure for Re = " << 1.0 / mu << " and " << withHessian;
                cout << "; # iters to converge: " << problem.iterationCount() << endl;
              } else {
  //              cout << "PASS: testVGPStokesFormulationConsistency(): ";
  //              cout << "L^2 error of " << l2Error << " for variable " << field->displayString();
  //              cout << " is within tolerance! " << tol << endl;
              }
            }
            
            FunctionPtr u1_soln = Function::solution(u1_vgp, backgroundFlow);
            FunctionPtr u2_soln = Function::solution(u2_vgp, backgroundFlow);
            FunctionPtr p_soln  = Function::solution( p_vgp, backgroundFlow);
            
            if ( ! functionsAgree(u1_soln, u1_exact, mesh, tol) ) {
              cout << "FAILURE: testVGPNavierStokesFormulationConsistency(): after solve, u1_soln != u1_exact.\n";
            }
            if ( ! functionsAgree(u2_soln, u2_exact, mesh, tol) ) {
              cout << "FAILURE: testVGPNavierStokesFormulationConsistency(): after solve, u2_soln != u2_exact.\n";
            }
            if ( ! functionsAgree(p_soln, p_exact, mesh, tol) ) {
              cout << "FAILURE: testVGPNavierStokesFormulationConsistency(): after solve, p_soln != p_exact.\n";
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
  
  PolyExactFunctions kovasznaySoln; // not actually a polynomial, so we call it a 10th-degree polynomial...
  FunctionPtr u1_k, u2_k, p_k;
  double Re_k = 10;
  // temp mesh to allow integration of the pressure:
  Teuchos::RCP<Mesh> mesh_k = MeshFactory::buildQuadMesh(quadPoints, 5, 5,
                                                  VGPStokesFormulation(1.0/Re_k).bf(), 5, 6);
  NavierStokesFormulation::setKovasznay( Re_k, mesh_k, u1_k, u2_k, p_k );
  int polyOrder = 10;
  kovasznaySoln.push_back( make_pair(u1_k, polyOrder) );
  kovasznaySoln.push_back( make_pair(u2_k, polyOrder) );
  kovasznaySoln.push_back( make_pair( p_k, polyOrder) );
  
  // DEBUGGING: write solution to disk
//  u1_k->writeValuesToMATLABFile(mesh_k, "u1_k.m");
//  u2_k->writeValuesToMATLABFile(mesh_k, "u2_k.m");
//  p_k->writeValuesToMATLABFile(mesh_k, "p_k.m");
  
  double tol_k = 1e-3; // tolerance for kovasznay
  
  polyExactFunctions.push_back(kovasznaySoln);
  
  int horizontalCells = 2, verticalCells = 2;
  
  bool useLineSearch = false;
  bool useCondensedSolve = true;
  bool enrichVelocity = false; // true would be for the "compliant" norm, which isn't working well yet
  
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
    
    double tol = 1e-13; // relax while we sort out some issues
//    cout << "note: using tol of 1e-10 in VGP NS correctness test (used to be 1e-13).\n";
    if (maxPolyOrder >= 10) {
      // a bit of a cheat: this means it's the Kovasznay solution, which we won't get exact:
      tol = 1e-10;
    }
    
    for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
      double mu = *muIt;
      double Re = 1/mu;
      if (printToConsole) {
        cout << "testing with mu = " << mu << endl;
      }
    
      vgpStokesFormulation = Teuchos::rcp( new VGPStokesFormulation(mu) );
      int H1Order = maxPolyOrder + 1;
      int pToAdd = 1;
      bool dontEnhanceFluxes = false;
      VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re, quadPoints,
                                                               horizontalCells, verticalCells,
                                                               H1Order, pToAdd,
                                                               u1_exact, u2_exact, p_exact, enrichVelocity, dontEnhanceFluxes);
      SolutionPtr backgroundFlow = problem.backgroundFlow();
      SolutionPtr solnIncrement = problem.solutionIncrement();
      RHSPtr rhs = problem.exactSolution()->rhs();
      
      // this is a bit ugly, in that we're assuming the RHS is actually a subclass of RHSEasy
      // (but that's true!)
      LinearTermPtr rhsLT = ((RHSEasy *)rhs.get())->linearTerm();
      
      // f = - grad p + mu delta u - u * grad u
      FunctionPtr f1 = - p_exact->dx() + mu * (u1_exact->dx()->dx() + u1_exact->dy()->dy())
                       - u1_exact * u1_exact->dx() - u2_exact * u1_exact->dy();
      FunctionPtr f2 = - p_exact->dy() + mu * (u2_exact->dx()->dx() + u2_exact->dy()->dy())
                       - u1_exact * u2_exact->dx() - u2_exact * u2_exact->dy();
      
      // define the previous solution terms we need:
      FunctionPtr u1_prev = Function::solution(u1_vgp, backgroundFlow);
      FunctionPtr u2_prev = Function::solution(u2_vgp, backgroundFlow);
      FunctionPtr sigma11_prev = Function::solution(sigma11_vgp, backgroundFlow);
      FunctionPtr sigma12_prev = Function::solution(sigma12_vgp, backgroundFlow);
      FunctionPtr sigma21_prev = Function::solution(sigma21_vgp, backgroundFlow);
      FunctionPtr sigma22_prev = Function::solution(sigma22_vgp, backgroundFlow);


      LinearTermPtr expectedRHS = f1 * v1_vgp + f2 * v2_vgp;
      // VGP Oseen-ish form:
      // (f,v) - (u_i u_j, v_i,j) + (u_i u_j n_j, v_i)
//      expectedRHS = expectedRHS - u1_prev * u1_prev * v1_vgp->dx(); // i=1, j=1
//      expectedRHS = expectedRHS - u1_prev * u2_prev * v1_vgp->dy(); // i=1, j=2
//      expectedRHS = expectedRHS - u2_prev * u1_prev * v2_vgp->dx(); // i=2, j=1
//      expectedRHS = expectedRHS - u2_prev * u2_prev * v2_vgp->dy(); // i=2, j=2
//      
//      FunctionPtr n = Function::normal();
//      FunctionPtr un = u1_prev * n->x() + u2_prev * n->y();
//      expectedRHS = expectedRHS + u1_prev * un * v1_vgp; // i=1
//      expectedRHS = expectedRHS + u2_prev * un * v2_vgp; // i=2
      
      // gradient-based form:
      expectedRHS = expectedRHS + (u1_prev * sigma11_prev / mu + u2_prev * sigma12_prev / mu) * v1_vgp;
      expectedRHS = expectedRHS + (u1_prev * sigma21_prev / mu + u2_prev * sigma22_prev / mu) * v2_vgp;
      
      BFPtr stokesBF = problem.stokesBF();
      expectedRHS = expectedRHS - stokesBF->testFunctional(backgroundFlow);
      
      Teuchos::RCP<Mesh> mesh = problem.mesh();
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, vgpVarFactory, tol)) {
        cout << "Failure: Navier-Stokes correctedness: before first solve (i.e. with zero background flow), LTs disagree\n";
        success = false;
      }
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, stokesBF->graphNorm(), tol)) {
        cout << "VGP Navier-Stokes Correctness Failure: rhsLT and expectedRHS differ, according to RieszRep norm, before first solve.\n";
        success = false;
      }
      
      Teuchos::RCP<RieszRep> rieszRepRHS = Teuchos::rcp( new RieszRep(mesh, problem.bf()->graphNorm(), rhsLT));
      rieszRepRHS->computeRieszRep();
      if (rieszRepRHS->getNorm() < tol) {
        cout << "norm of RHS with zero background flow should not be zero! " << rieszRepRHS->getNorm() << endl;
        success = false;
      }
      
      Teuchos::RCP<RieszRep> rieszRepRHS_naiveNorm = Teuchos::rcp( new RieszRep(mesh, problem.bf()->naiveNorm(), rhsLT) );
      rieszRepRHS_naiveNorm->computeRieszRep();
//      cout << "norm of RHS with zero background flow, using naive norm: " << rieszRepRHS_naiveNorm->getNorm() << endl;
      
      problem.iterate(useLineSearch, useCondensedSolve); // calls backgroundFlow.solve()
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, vgpVarFactory, tol)) {
        cout << "Failure: Navier-Stokes correctedness: after first solve (i.e. with non-zero background flow), LTs disagree\n";
        success = false;
        cout << "rhsLT's boundary-only part:\n" << rhsLT->getBoundaryOnlyPart()->displayString() << endl;
        cout << "expectedRHS's boundary-only part:\n" << expectedRHS->getBoundaryOnlyPart()->displayString() << endl;
      }
      
      if ( !ltsAgree(rhsLT, expectedRHS, mesh, stokesBF->graphNorm(), tol)) {
        cout << "VGP Navier-Stokes Correctness Failure: rhsLT and expectedRHS differ, according to RieszRep norm, after first solve.\n";
        success = false;
      }
      
      // now, try projecting the exact solution:
      map<int, FunctionPtr > solnMap = vgpSolutionMap(u1_exact, u2_exact, p_exact, Re);
      backgroundFlow->clear(); // should not be necessary -- just checking...
      backgroundFlow->projectOntoMesh(solnMap);
      
      if (maxPolyOrder < 10) { // polynomial solution: should be able to nail this
        for (map<int, FunctionPtr >::iterator varIt = solnMap.begin(); varIt != solnMap.end(); varIt++) {
          int varID = varIt->first;
          VarPtr var = vgpVarFactory.trial(varID);
          FunctionPtr exactSoln = varIt->second;
          FunctionPtr projectedSoln = Function::solution(var, backgroundFlow);
          
          if (var->varType() == FLUX) { // then there will be an extra parity factor from Function::solution()
            projectedSoln = projectedSoln * Function::sideParity();
          }
          
          double tol = 1e-13;
          if (!functionsAgree(exactSoln - projectedSoln, zero, mesh, tol) ) {
            cout << "Projection of exact polynomial function failed for var " << var->displayString() << " != " << exactSoln->displayString() << endl;
          } else {
//            cout << "Projection succeeded for var "  << var->displayString() << " = " << exactSoln->displayString() << endl;
          }
        }
      }
      
//      VTKExporter solnExporter(backgroundFlow,mesh,vgpVarFactory);
//      solnExporter.exportSolution("correctness_background_flow");
      
//      vector< FunctionPtr > boundaryFunctions;
//      boundaryFunctions.push_back(solnMap[t1n_vgp->ID()]);
//      boundaryFunctions.push_back(solnMap[t2n_vgp->ID()]);
      
//      solnExporter.exportBoundaryValuedFunctions(boundaryFunctions, "boundaryFunctions");
      
//      solnExporter.exportFunction(Function::solution(u1_vgp, backgroundFlow), "u1_backFlow");
//      solnExporter.exportFunction(solnMap[u1_vgp->ID()], "u1_exact");
//      FunctionPtr u1diff = solnMap[u1_vgp->ID()] - Function::solution(u1_vgp, backgroundFlow);
//      solnExporter.exportFunction(u1diff,"u1_diff");
      
      if (u1_exact->isZero() && u2_exact->isZero()) {
        if (functionsAgree(p_exact,y,mesh)) {
          // for this case, check some hand-computed values:
          IPPtr ip = stokesBF->naiveNorm();
          // test functional: stokes background flow = - v2
//          LinearTermPtr testFunctionalExpected = - v2_vgp;
          // experimentally, try with the IBP version:
          // (y, div v) - < y, v * n>
          FunctionPtr n = Function::normal();
          LinearTermPtr testFunctionalExpected = y * v1_vgp->dx() + y * v2_vgp->dy() - y * n->x() * v1_vgp - y * n->y() * v2_vgp;
          LinearTermPtr testFunctionalActual = stokesBF->testFunctional(backgroundFlow);
          if (! ltsAgree(testFunctionalExpected, testFunctionalActual, mesh, ip, tol) ) {
            success = false;
            cout << "for p=y solution, bf->testFunctional doesn't match expected (-v2 expected, actual is ";
            cout << testFunctionalActual->displayString() << ")" << endl;
            cout << "boundary-only part of bf->testFunctional is: " << testFunctionalActual->getBoundaryOnlyPart()->displayString() << endl;
            
            // now, try to get clearer on where they differ
            vector< VarPtr > testVars = vgpTests;
            for ( vector< VarPtr >::iterator testVarIt = testVars.begin(); testVarIt != testVars.end(); testVarIt++) {
              VarPtr testVar = *testVarIt;
              map< int, FunctionPtr > varFunctions;
              if (testVar->rank() == 0) {
                varFunctions[testVar->ID()] = y;
              } else if (testVar->rank() == 1) {
                varFunctions[testVar->ID()] = Function::vectorize(x, y);
              }
              FunctionPtr fxnExpected = testFunctionalExpected->evaluate(varFunctions, true); // true: boundary part
              FunctionPtr fxnActual = testFunctionalActual->evaluate(varFunctions, true);
              if ( functionsAgree(fxnExpected, fxnActual, mesh) ) {
                cout << "boundary parts appear to agree for var " << testVar->displayString() << endl;
              } else {
                cout << "boundary parts disagree for var " << testVar->displayString() << endl;
                cout << "expected: " << fxnExpected->displayString() << endl;
                cout << "actual: " << fxnActual->displayString() << endl;
              }
            }
            
            // now, let's check that the background flow is what we expect it to be:
            FunctionPtr u1hat_soln = Function::solution(u1hat_vgp, backgroundFlow);
            FunctionPtr u2hat_soln = Function::solution(u2hat_vgp, backgroundFlow);
            FunctionPtr t1n_soln = Function::solution(t1n_vgp, backgroundFlow);
            FunctionPtr t2n_soln = Function::solution(t2n_vgp, backgroundFlow);
            
            FunctionPtr n = Function::normal();
            FunctionPtr sideParity = Function::sideParity();
            FunctionPtr t1n_exact = - p_exact * n->x() * sideParity;
            FunctionPtr t2n_exact = - p_exact * n->y() * sideParity;
            
            if ( !functionsAgree(u1hat_soln - u1_exact, zero, mesh) ) {
              cout << "u1_hat != 0 \n";
            }
            if ( !functionsAgree(u2hat_soln - u2_exact, zero, mesh) ) {
              cout << "u2_hat != 0 \n";
            }
            if ( !functionsAgree(t1n_soln - t1n_exact, zero, mesh) ) {
              cout << "t1n_soln != t1n_exact \n";
            }
            if ( !functionsAgree(t2n_soln - t2n_exact, zero, mesh) ) {
              cout << "t2n_soln != t2n_exact \n";
            }
            
            success = false;
          }
//          cout << "RHS for p=y solution is " << expectedRHS->displayString() << endl;
        }
      }
      
      // now, for polynomials, the RHS should be zero.  For Kovasznay, it should be small-ish.
      
      
      rieszRepRHS_naiveNorm->computeRieszRep();
//      cout << "norm of RHS with exact solution projected, using naive norm: " << rieszRepRHS_naiveNorm->getNorm() << endl;
      
      rieszRepRHS->computeRieszRep();
      double norm = rieszRepRHS->getNorm();
      if (maxPolyOrder < 10) { // polynomial solution: should have essentially zero RHS
        
        
//        cout << "norm of RHS after projection of polynomial exact solution: " << norm << endl;
        if (norm > tol) {
          cout << "Failure: Navier-Stokes does not have a zero RHS after exact solution projected; norm of RHS is " << norm << "\n";
          cout << "u1_exact = " << u1_exact->displayString() << endl;
          cout << "u2_exact = " << u2_exact->displayString() << endl;
          cout << " p_exact = " << p_exact->displayString() << endl;
          cout << "sigma11_exact = " << solnMap[sigma11_vgp->ID()]->displayString() << endl;
          cout << "sigma12_exact = " << solnMap[sigma12_vgp->ID()]->displayString() << endl;
          cout << "sigma21_exact = " << solnMap[sigma21_vgp->ID()]->displayString() << endl;
          cout << "sigma22_exact = " << solnMap[sigma22_vgp->ID()]->displayString() << endl;
          cout << "t1n_vgp_exact = " << solnMap[t1n_vgp->ID()]->displayString() << endl;
          cout << "t2n_vgp_exact = " << solnMap[t2n_vgp->ID()]->displayString() << endl;
          cout << "u1hat_exact = " << solnMap[u1hat_vgp->ID()]->displayString() << endl;
          cout << "u2hat_exact = " << solnMap[u2hat_vgp->ID()]->displayString() << endl;
          cout << "Re = " << Re << endl;
          vector< VarPtr > nonZeros = nonZeroComponents(expectedRHS, vgpTests, mesh, stokesBF->naiveNorm());
          cout << "Expected RHS is non-zero in components: " << endl;
          for (vector< VarPtr >::iterator varIt = nonZeros.begin(); varIt != nonZeros.end(); varIt++) {
            VarPtr var = *varIt;
            cout << var->displayString() << ": " << expectedRHS->getPartMatchingVariable(var)->displayString() << endl;
          }
          
          success = false;
        }
      } else {
        if (norm > tol_k) {
          success = false;
          cout << "Failure: Norm of RHS after Kovasznay solution projected onto mesh exceeds tolerance: " << norm << endl;
        }
      }
    }
  }
 
  return success;
}

// U1_0: used in lid-driven cavity flow BCs (for local conservation test below)
class U1_0 : public SimpleFunction {
  double _eps;
public:
  U1_0(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    double tol = 1e-14;
    if (abs(y-1.0) < tol) { // top boundary
      if ( (abs(x) < _eps) ) { // top left
        return x / _eps;
      } else if ( abs(1.0-x) < _eps) { // top right
        return (1.0-x) / _eps;
      } else { // top middle
        return 1;
      }
    } else { // not top boundary: 0.0
      return 0.0;
    }
  }
};

bool IncompressibleFormulationsTests::testVGPNavierStokesLocalConservation() {
  bool success = true;
  
  // really just checking against a single bug right now: getting 0 rows and columns
  // in the second iterate for Navier-Stokes when local conservation is turned on...
  
  int pToAdd = 3; // for optimal test function approximation
  double eps = 1.0/64.0; // width of ramp up to 1.0 for top BC
  bool enforceLocalConservation = true;
  int horizontalCells = 3, verticalCells = 3;
  
  int polyOrder = 1;
  double Re = 400;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  // define meshes:
  int H1Order = polyOrder + 1;
  
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
  FunctionPtr u2_0 = Function::zero();
  FunctionPtr zero = Function::zero();
  ParameterFunctionPtr Re_param = ParameterFunction::parameterFunction(Re);
  VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re_param,quadPoints,
                                                          horizontalCells,verticalCells,
                                                          H1Order, pToAdd,
                                                          u1_0, u2_0,  // BC for u
                                                          zero, zero); // zero forcing function
  SolutionPtr solution = problem.backgroundFlow();
  SolutionPtr solnIncrement = problem.solutionIncrement();
  
  // get variable definitions:
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  VarPtr u1hat = varFactory.traceVar(VGP_U1HAT_S);
  VarPtr u2hat = varFactory.traceVar(VGP_U2HAT_S);
  
  if (enforceLocalConservation) {
    solution     ->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
    solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  bool useLineSearch = false;
  bool useCondensedSolve = false; // condensed solve doesn't support lagrange constraints right now...
  problem.iterate(useLineSearch,useCondensedSolve);
  
  problem.iterate(useLineSearch,useCondensedSolve); // get zero rows and column warnings here
  
  return success;
}

map<int, FunctionPtr > IncompressibleFormulationsTests::vgpSolutionMap(FunctionPtr u1_exact, FunctionPtr u2_exact, FunctionPtr p_exact,
                                                                       double Re) {
  
  double mu = 1.0 / Re;
  
  FunctionPtr skeletonRestrictor = Function::meshSkeletonCharacteristic();
  
  map<int, FunctionPtr > solnMap;
  solnMap[ u1_vgp->ID() ] = u1_exact;
  solnMap[ u2_vgp->ID() ] = u2_exact;
  solnMap[  p_vgp->ID() ] =  p_exact;
  solnMap[  sigma11_vgp->ID() ] = mu * u1_exact->dx();
  solnMap[  sigma12_vgp->ID() ] = mu * u1_exact->dy();
  solnMap[  sigma21_vgp->ID() ] = mu * u2_exact->dx();
  solnMap[  sigma22_vgp->ID() ] = mu * u2_exact->dy();
  
  solnMap[ u1hat_vgp->ID() ] = u1_exact * skeletonRestrictor; // skeletonRestrictor makes clear that these are boundary values
  solnMap[ u2hat_vgp->ID() ] = u2_exact * skeletonRestrictor;
  
  FunctionPtr t1 = Function::vectorize( mu * u1_exact->dx() - p_exact, mu * u1_exact->dy() );
  FunctionPtr t2 = Function::vectorize( mu * u2_exact->dx(), mu * u2_exact->dy() - p_exact);
  FunctionPtr n = Function::normal();
  FunctionPtr sgn_n = Function::sideParity();
  // since \hat {t}_n is defined as t * n * sgn_n to make it uniquely valued,
  // that's what we should project...
  solnMap[ t1n_vgp->ID() ] = t1 * n * sgn_n;
  solnMap[ t2n_vgp->ID() ] = t2 * n * sgn_n;
  
  return solnMap;
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulationLocalConservation() {
  bool success = true;
  
  int pToAdd = 2; // for optimal test function approximation
  double eps = 1.0/64.0; // width of ramp up to 1.0 for top BC;  eps == 0 ==> soln not in H1
  bool enforceLocalConservation = false;
  
  int horizontalCells = 2, verticalCells = 2;
  int polyOrder = 1;

  double Re = 1;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  // define meshes:
  int H1Order = polyOrder + 1;
  
  // get variable definitions:
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  VarPtr u1hat = varFactory.traceVar(VGP_U1HAT_S);
  VarPtr u2hat = varFactory.traceVar(VGP_U2HAT_S);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
  FunctionPtr u2_0 = zero;
  ParameterFunctionPtr Re_param = ParameterFunction::parameterFunction(Re);
  VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re_param,quadPoints,
                                                          horizontalCells,verticalCells,
                                                          H1Order, pToAdd,
                                                          u1_0, u2_0,  // BC for u
                                                          zero, zero); // zero forcing function
  SolutionPtr solution = problem.backgroundFlow();
  SolutionPtr solnIncrement = problem.solutionIncrement();
  
  // see if we do better with naive norm, which is better conditioned:
  solution->setIP(problem.bf()->naiveNorm());
  solnIncrement->setIP(problem.bf()->naiveNorm());
  
  Teuchos::RCP<Mesh> mesh = problem.mesh();
  mesh->registerSolution(solution);
  mesh->registerSolution(solnIncrement);
  
  if (enforceLocalConservation) {
    FunctionPtr zero = Function::zero();
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
    solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  set<GlobalIndexType> cellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt = cellIDs.begin(); cellIt != cellIDs.end(); cellIt++) {
    int cellID = *cellIt;
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    DofOrderingPtr testSpace = mesh->getElement(cellID)->elementType()->testOrderPtr;
    double stokesConditionNumber = problem.stokesBF()->graphNorm()->computeMaxConditionNumber(testSpace, basisCache);
    cout << "Gram matrix condition number of stokes graph norm for cell " << cellID << ": " << stokesConditionNumber << endl;
//    double conditionNumber = problem.bf()->graphNorm()->computeMaxConditionNumber(testSpace, basisCache);
//    cout << "Gram matrix condition number of Navier-Stokes graph norm for cell " << cellID << ": " << conditionNumber << endl;
    double naiveStokesConditionNumber = problem.stokesBF()->naiveNorm()->computeMaxConditionNumber(testSpace, basisCache);
    cout << "Gram matrix condition number of stokes naive norm for cell " << cellID << ": " << naiveStokesConditionNumber << endl;
//    double naiveConditionNumber = problem.bf()->naiveNorm()->computeMaxConditionNumber(testSpace, basisCache);
//    cout << "Gram matrix condition number of Navier-Stokes naive norm for cell " << cellID << ": " << naiveConditionNumber << endl;
    
    double l2ConditionNumber = problem.bf()->l2Norm()->computeMaxConditionNumber(testSpace, basisCache);
    cout << "Gram matrix condition number of stokes L^2 norm for cell " << cellID << ": " << l2ConditionNumber << endl;
  }
  
  bool useLineSearch = false;
  bool useCondensedSolve = true;
  problem.iterate(useLineSearch, useCondensedSolve);
  problem.iterate(useLineSearch, useCondensedSolve);
  
  return success;
}

bool IncompressibleFormulationsTests::testVGPNavierStokesFormulationKovasnayConvergence() {
  bool success = true;
  double tol = 1e-11;
  bool printToConsole = false;
  vector<bool> useHessianList;
  useHessianList.push_back(false);
  // we skip using the hessian for now--it's not working (diverging), but we don't know whether it should...
  //  useHessianList.push_back(true);
  
  bool useNaiveNorm = false;
  if (printToConsole) {
    cout << "testVGPNavierStokesFormulationKovasnayConvergence: useNaiveNorm = " << useNaiveNorm << endl;
  }
  
  meshDimensions.clear();
  if (_thoroughMode) {
    meshDimensions.push_back( make_pair(5,5) ); // fine-ish mesh...
  } else {
    meshDimensions.push_back( make_pair(2,2) ); // less fine mesh...
  }
  
  muValues.clear();
  muValues.push_back(0.1);  // this may be the only valid one for Kovasznay; I'm not sure...
  
  // exact solution functions: store these as vector< pair< Function, int > >
  // in the order u1, u2, p, where the paired int is the polynomial degree of the function
  
  int overIntegrationForKovasznay = 5; // since the RHS isn't a polynomial...
//  double nonlinear_step_weight = 1.0;
  
  bool useLineSearch = false; // we don't converge nearly as quickly (if at all) when using line search (a problem!)
  bool useCondensedSolve = true;
  bool enrichVelocity = false; // true would be for the "compliant" norm, which isn't working well yet
  
  for (vector<bool>::iterator useHessianIt = useHessianList.begin();
       useHessianIt != useHessianList.end(); useHessianIt++) {
    bool useHessian = *useHessianIt;
    double maxPolyOrder = 3; // just picking something for Kovasznay...
    
    FunctionPtr u1_exact, u2_exact, p_exact;
        
    int H1Order = maxPolyOrder + 1;
    for (vector<int>::iterator pToAddIt = pToAddValues.begin(); pToAddIt != pToAddValues.end(); pToAddIt++) {
      int pToAdd = *pToAddIt;
      
      for (vector< pair<int, int> >::iterator meshDimIt = meshDimensions.begin(); meshDimIt != meshDimensions.end(); meshDimIt++) {
        int horizontalCells = meshDimIt->first, verticalCells = meshDimIt->second;
        
        for (vector<double>::iterator muIt = muValues.begin(); muIt != muValues.end(); muIt++) {
          double mu = *muIt;
          
          double Re = 1.0 / mu;
          bool dontEnhanceFluxes = false;
          VGPNavierStokesProblem zeroProblem = VGPNavierStokesProblem(Re, quadPointsKovasznay,
                                                                      horizontalCells, verticalCells,
                                                                      H1Order, pToAdd,
                                                                      zero, zero, zero, enrichVelocity, dontEnhanceFluxes);
          
          NavierStokesFormulation::setKovasznay( Re, zeroProblem.mesh(), u1_exact, u2_exact, p_exact );
          
          VGPNavierStokesProblem kProblem = VGPNavierStokesProblem(Re, quadPointsKovasznay,
                                                                   horizontalCells, verticalCells,
                                                                   H1Order, pToAdd,
                                                                   u1_exact, u2_exact, p_exact, enrichVelocity, dontEnhanceFluxes);
          
          if (printToConsole) {
            cout << "VGP Navier-Stokes consistency tests for Kovasznay solution with Re = " << Re << endl;
            cout << "u1_exact: " << u1_exact->displayString() << endl;
            cout << "u2_exact: " << u2_exact->displayString() << endl;
            cout << "p_exact: " << p_exact->displayString() << endl;
          }
          
          int cubatureEnrichmentDegree = kProblem.solutionIncrement()->cubatureEnrichmentDegree();
          cubatureEnrichmentDegree += overIntegrationForKovasznay;
          
          SolutionPtr solnIncrement = kProblem.solutionIncrement();
          SolutionPtr backgroundFlow = kProblem.backgroundFlow();
          solnIncrement->setCubatureEnrichmentDegree(cubatureEnrichmentDegree);
          backgroundFlow->setCubatureEnrichmentDegree(cubatureEnrichmentDegree);
          
          IPPtr ip;
          if ( useNaiveNorm ) {
            ip = kProblem.bf()->naiveNorm();
          } else {
            ip = kProblem.bf()->graphNorm();
          }
          
          RHSPtr rhs = solnIncrement->rhs();
          
          Teuchos::RCP< RieszRep > rieszRep;
          if (useHessian) {
            LinearTermPtr rhsLT = ((RHSEasy*) rhs.get())->linearTerm();

            rieszRep = Teuchos::rcp( new RieszRep(kProblem.mesh(),ip,rhsLT));
            FunctionPtr v1_rep = Teuchos::rcp( new RepFunction(v1_vgp, rieszRep) );
            FunctionPtr v2_rep = Teuchos::rcp( new RepFunction(v2_vgp, rieszRep) );
            // set up the hessian term itself:
            // we want basically u * sigma * v where "*" is a dot product
            // u * sigma = (u1 sigma11 + u2 sigma12, u1 sigma21 + u2 sigma22)
            BFPtr hessianBF = Teuchos::rcp( new BF(vgpVarFactory.getBubnovFactory(VarFactory::BUBNOV_TRIAL)) );
            hessianBF->addTerm(v1_rep * u1_vgp, sigma11_vgp);
            hessianBF->addTerm(v1_rep * u2_vgp, sigma12_vgp);
            hessianBF->addTerm(v2_rep * u1_vgp, sigma21_vgp);
            hessianBF->addTerm(v2_rep * u2_vgp, sigma22_vgp);
            // now the symmetric counterparts:
            hessianBF->addTerm(sigma11_vgp, v1_rep * u1_vgp);
            hessianBF->addTerm(sigma12_vgp, v1_rep * u2_vgp);
            hessianBF->addTerm(sigma21_vgp, v2_rep * u1_vgp);
            hessianBF->addTerm(sigma22_vgp, v2_rep * u2_vgp);
            
            Teuchos::RCP< HessianFilter > hessianFilter = Teuchos::rcp( new HessianFilter(hessianBF) );
            solnIncrement->setFilter(hessianFilter);
            rieszRep->computeRieszRep();
          }
          
//          map<int, FunctionPtr > solnMap = vgpSolutionMap(u1_exact, u2_exact, p_exact, Re);
//          vgpNavierStokesSolution->projectOntoMesh( solnMap );
          int maxIters = 100;

          FunctionPtr u1_incr = Function::solution(u1_vgp, solnIncrement);
          FunctionPtr u2_incr = Function::solution(u2_vgp, solnIncrement);
          FunctionPtr p_incr = Function::solution(p_vgp, solnIncrement);
          FunctionPtr sigma11_incr = Function::solution(sigma11_vgp, solnIncrement);
          FunctionPtr sigma12_incr = Function::solution(sigma12_vgp, solnIncrement);
          FunctionPtr sigma21_incr = Function::solution(sigma21_vgp, solnIncrement);
          FunctionPtr sigma22_incr = Function::solution(sigma22_vgp, solnIncrement);
          
          FunctionPtr u1hat_incr = Function::solution(u1hat_vgp, solnIncrement);
          FunctionPtr u2hat_incr = Function::solution(u2hat_vgp, solnIncrement);
          FunctionPtr t1n_incr = Function::solution(t1n_vgp, solnIncrement);
          FunctionPtr t2n_incr = Function::solution(t2n_vgp, solnIncrement);
          
          FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr;
          
          do {
            kProblem.iterate(useLineSearch, useCondensedSolve);
            
            if ( rieszRep.get() ) {
              rieszRep->computeRieszRep();
            }
            
            int cubatureDegree = maxPolyOrder;
            double combinedL2Error = 0.0;
            if (printToConsole) {
              cout << "After " << kProblem.iterationCount() << " nonlinear steps, Kovasznay solution:\n";
            }
            for (vector< VarPtr >::iterator fieldIt = vgpFields.begin(); fieldIt != vgpFields.end(); fieldIt++ ) {
              VarPtr field = *fieldIt;
              double l2Error = kProblem.exactSolution()->L2NormOfError(*backgroundFlow, field->ID(), cubatureDegree);
              if (printToConsole) {
                cout << "L^2 error of " << l2Error << " for variable " << field->displayString() << "." << endl;
              }
              combinedL2Error += l2Error * l2Error;
            }
            combinedL2Error = sqrt(combinedL2Error);
            if (printToConsole) {
              cout << "combined L2 error: " << combinedL2Error << endl;
            }
            if (combinedL2Error > 1e6) {
              cout << "combined L2 error > 1e6; giving up.\n";
              break;
            }
            
            
          } while ( (sqrt(l2_incr->integrate(kProblem.mesh())) > tol) && (kProblem.iterationCount()<maxIters) );
          
          if (printToConsole) {
            string withHessian = useHessian ? " using hessian term " : " without hessian term ";
            cout << "with Re = " << 1.0 / mu << " and " << withHessian;
            cout << ", # iters to converge: " << kProblem.iterationCount() << endl;
          }
          
          double l2NormOfIncr = sqrt(l2_incr->integrate(kProblem.mesh()));
          if (l2NormOfIncr > tol) {
            string withHessian = useHessian ? "using the hessian" : "not using the hessian";
            cout << "Kovasnay solution failed to converge while " << withHessian << "; l2NormOfIncr = " << l2NormOfIncr << endl;
            success = false;
          }
        }
      }
    }
  }
  return success;
}

std::string IncompressibleFormulationsTests::testSuiteName() {
  return "IncompressibleFormulationsTests";
}
