#include "LobattoBasisTests.h"

#include "Function.h"

#include "MeshFactory.h"

#include "Lobatto.hpp"

#include "BF.h"

void LobattoBasisTests::setup() {}

void LobattoBasisTests::teardown() {}

void LobattoBasisTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testLegendreValues()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testLobattoDerivativeValues()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testLobattoValues()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testH1Classifications()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool LobattoBasisTests::testLegendreValues() {  
  bool success = true;
  
  FunctionPtr x = Function::xn(1);
  vector< FunctionPtr > legendreFunctionsExpected;
  legendreFunctionsExpected.push_back( Function::constant(1.0) );
  legendreFunctionsExpected.push_back( x );
  legendreFunctionsExpected.push_back( (3 * x*x - 1) / 2);
  legendreFunctionsExpected.push_back( (5 * x*x*x - 3*x) / 2);
  legendreFunctionsExpected.push_back( (35 * x * x * x * x - 30 * x * x + 3) / 8);
  legendreFunctionsExpected.push_back( (63 * x * x * x * x * x - 70 * x * x * x + 15 * x) / 8);
  
  vector< FunctionPtr > legendreFunctions;
  
  int n_max = 4;
  for (int n=0; n<=n_max; n++) {
    legendreFunctions.push_back( Teuchos::rcp( new LegendreFunction(n) ) );
  }
  
  VarFactory varFactory;
  varFactory.testVar("v", HGRAD);
  varFactory.fieldVar("u", L2);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  MeshPtr mesh = MeshFactory::quadMesh(bf, n_max);
  
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0);
  
  double tol = 1e-8; // relax to make sure that failure isn't just roundoff
  for (int n=0; n<=n_max; n++) {
    if (! legendreFunctions[n]->equals(legendreFunctionsExpected[n], basisCache, tol) ) {
      cout << "Legendre function " << n << " does not match expected.\n";
      success = false;
    }
  }
  
  return success;
}

bool LobattoBasisTests::testLobattoDerivativeValues() {
  bool success = true;
  FunctionPtr x = Function::xn(1);
  vector< FunctionPtr > legendreFunctions; // manually specified
  vector< FunctionPtr > lobattoDerivatives;
  
  FunctionPtr one = Function::constant(1);
  legendreFunctions.push_back(one);
  legendreFunctions.push_back(x);
  legendreFunctions.push_back(0.5 * (3 * x * x - 1) );
  legendreFunctions.push_back( (5 * x * x * x - 3 * x) / 2);
  legendreFunctions.push_back( (1.0 / 8.0) * (35 * x * x * x * x - 30 * x * x + 3) );
  legendreFunctions.push_back( (1.0 / 8.0) * (63 * x * x * x * x * x - 70 * x * x * x + 15 * x) );
  
  int n_max = 4;
  for (int n=0; n<=n_max+1; n++) {
    lobattoDerivatives.push_back( Teuchos::rcp( new LobattoFunction<>(n,true) ) );
  }
  
  VarFactory varFactory;
  varFactory.testVar("v", HGRAD);
  varFactory.fieldVar("u", L2);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  MeshPtr mesh = MeshFactory::quadMesh(bf, n_max);
  
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0);
  
  double tol = 1e-8;
  for (int n=0; n<=n_max; n++) {
    if (! legendreFunctions[n]->equals(lobattoDerivatives[n+1], basisCache, tol) ) {
      cout << "Legendre function " << n << " != Lobatto function " << n + 1 << " derivative.\n";
      cout << "L_" << n << "(0.5) = " << Function::evaluate(legendreFunctions[n],0.5) << endl;
      cout << "l'_" << n+1 << "(0.5) = " << Function::evaluate(lobattoDerivatives[n+1],0.5) << endl;
      success = false;
    }
  }
  
  return success;
}

bool LobattoBasisTests::testLobattoValues() {  
  bool success = true;
  
  FunctionPtr x = Function::xn(1);
  vector< FunctionPtr > lobattoFunctionsExpected;
  // Pavel Solin's first two Lobatto functions:
//  lobattoFunctionsExpected.push_back( (1 - x) / 2 );
//  lobattoFunctionsExpected.push_back( (1 + x) / 2 );
  // Demkowicz's:
  lobattoFunctionsExpected.push_back( Function::constant(1.0) );
  lobattoFunctionsExpected.push_back( x );
  
  lobattoFunctionsExpected.push_back( (x*x - 1) / 2);
  lobattoFunctionsExpected.push_back( (x*x - 1) * x / 2);
  lobattoFunctionsExpected.push_back( (x*x - 1) * (5 * x * x - 1) / 8);
  lobattoFunctionsExpected.push_back( (x*x - 1) * (7 * x * x - 3) * x / 8);
  
  vector< FunctionPtr > lobattoFunctions;
  
  int n_max = 4;
  for (int n=0; n<=n_max; n++) {
    lobattoFunctions.push_back( Teuchos::rcp( new LobattoFunction<>(n,false) ) );
  }
  
  VarFactory varFactory;
  varFactory.testVar("v", HGRAD);
  varFactory.fieldVar("u", L2);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  MeshPtr mesh = MeshFactory::quadMesh(bf, n_max);
  
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0);
  
  double tol = 1e-12;
  for (int n=0; n<=n_max; n++) {
    if (! lobattoFunctions[n]->equals(lobattoFunctionsExpected[n], basisCache, tol) ) {
      cout << "Lobatto function " << n << " does not match expected.\n";
      success = false;
    }
  }
  
  return success;
}

bool testBasisClassifications(BasisPtr basis) {
  bool success = true;
  
  shards::CellTopology cellTopo = basis->domainTopology();
  
  int numVertices = cellTopo.getVertexCount();
  int numEdges = cellTopo.getEdgeCount();
  
  int degree = basis->getDegree();
  
  // TODO: finish this
  
  // get the points in reference space for each vertex
  
  // test that the points are correctly classified
  
  return success;
}

bool LobattoBasisTests::testH1Classifications() {
  // checks that edge functions, vertex functions, etc. are correctly listed for the H^1 Lobatto basis
  bool success = true;
  
  // TODO: implement this
  cout << "Warning: testH1Classification unimplemented.\n";
  
  return success;
}

string LobattoBasisTests::testSuiteName() {
  return "LobattoBasisTests";
}