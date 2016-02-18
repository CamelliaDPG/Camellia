#include "LobattoBasisTests.h"

#include "Function.h"

#include "MeshFactory.h"

#include "Lobatto.hpp"

#include "LobattoHGRAD_LineBasis.h"
#include "LobattoHGRAD_QuadBasis.h"

#include "BF.h"

#include "doubleBasisConstruction.h"

#include <Teuchos_GlobalMPISession.hpp>

// Shards includes
#include "Shards_CellTopology.hpp"

using namespace Camellia;

void LobattoBasisTests::setup() {}

void LobattoBasisTests::teardown() {}

void LobattoBasisTests::runTests(int &numTestsRun, int &numTestsPassed)
{
  setup();
  if (testSimpleStiffnessMatrix())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testLegendreValues())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testLobattoDerivativeValues())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testLobattoValues())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testH1Classifications())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testLobattoLineClassifications())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool LobattoBasisTests::testLegendreValues()
{
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
  for (int n=0; n<=n_max; n++)
  {
    legendreFunctions.push_back( Teuchos::rcp( new LegendreFunction(n) ) );
  }

  VarFactoryPtr varFactory = VarFactory::varFactory();
  varFactory->testVar("v", HGRAD);
  varFactory->fieldVar("u", L2);

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  MeshPtr mesh = MeshFactory::quadMesh(bf, n_max);

  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0);

  double tol = 1e-8; // relax to make sure that failure isn't just roundoff
  for (int n=0; n<=n_max; n++)
  {
    if (! legendreFunctions[n]->equals(legendreFunctionsExpected[n], basisCache, tol) )
    {
      cout << "Legendre function " << n << " does not match expected.\n";
      success = false;
    }
  }

  return success;
}

bool LobattoBasisTests::testLobattoDerivativeValues()
{
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

  bool conformingFalse = false;

  int n_max = 4;
  for (int n=0; n<=n_max+1; n++)
  {
    lobattoDerivatives.push_back( Teuchos::rcp( new LobattoFunction<>(n,conformingFalse,true) ) );
  }

  VarFactoryPtr varFactory = VarFactory::varFactory();
  varFactory->testVar("v", HGRAD);
  varFactory->fieldVar("u", L2);

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  MeshPtr mesh = MeshFactory::quadMesh(bf, n_max);

  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0);

  double tol = 1e-8;
  for (int n=0; n<=n_max; n++)
  {
    if (! legendreFunctions[n]->equals(lobattoDerivatives[n+1], basisCache, tol) )
    {
      cout << "Legendre function " << n << " != Lobatto function " << n + 1 << " derivative.\n";
      cout << "L_" << n << "(0.5) = " << Function::evaluate(legendreFunctions[n],0.5) << endl;
      cout << "l'_" << n+1 << "(0.5) = " << Function::evaluate(lobattoDerivatives[n+1],0.5) << endl;
      success = false;
    }
  }

  return success;
}

bool LobattoBasisTests::testLobattoValues()
{
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

  bool conformingFalse = false;

  int n_max = 4;
  for (int n=0; n<=n_max; n++)
  {
    lobattoFunctions.push_back( Teuchos::rcp( new LobattoFunction<>(n,conformingFalse,false) ) );
  }

  VarFactoryPtr varFactory = VarFactory::varFactory();
  varFactory->testVar("v", HGRAD);
  varFactory->fieldVar("u", L2);

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  MeshPtr mesh = MeshFactory::quadMesh(bf, n_max);

  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, 0);

  double tol = 1e-12;
  for (int n=0; n<=n_max; n++)
  {
    if (! lobattoFunctions[n]->equals(lobattoFunctionsExpected[n], basisCache, tol) )
    {
      cout << "Lobatto function " << n << " does not match expected.\n";
      success = false;
    }
  }

  return success;
}

bool checkVertexOrdinalsQuad(BasisPtr basis, vector<int> &vertexOrdinals)
{
  // check that the given indices are exactly the vertex basis functions:
  // a) these are (1,0) or (0,1) at the corresponding vertices
  // b) others are (0,0) at the vertices

  int numVertices = 4;

  FieldContainer<double> refCellPoints(numVertices,2); // vertices, in order
  refCellPoints(0,0) = -1;
  refCellPoints(0,1) = -1;
  refCellPoints(1,0) =  1;
  refCellPoints(1,1) = -1;
  refCellPoints(2,0) =  1;
  refCellPoints(2,1) =  1;
  refCellPoints(3,0) = -1;
  refCellPoints(3,1) =  1;

  // assume scalar basis for now -- we'll throw an exception if not...
  FieldContainer<double> values(basis->getCardinality(), numVertices); // F, P
  basis->getValues(values, refCellPoints, OPERATOR_VALUE);

  double tol = 1e-14;
  for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++)
  {
    int vertexOrdinal = vertexOrdinals[vertexIndex];
    for (int fieldIndex=0; fieldIndex<basis->getCardinality(); fieldIndex++)
    {
      double value = values(fieldIndex,vertexIndex);
      if (fieldIndex==vertexOrdinal)
      {
        // expect non-zero
        if (value < tol)
        {
          return false;
        }
      }
      else
      {
        // expect zero
        if (value > tol)
        {
          return false;
        }
      }
    }
  }
  return true;
}

bool testBasisClassifications(BasisPtr basis)
{
  bool success = true;

  CellTopoPtr cellTopo = basis->domainTopology();

  int numVertices = cellTopo->getVertexCount();
  int numEdges = cellTopo->getEdgeCount();

  int degree = basis->getDegree();

  // TODO: finish this
  vector<int> vertexOrdinals;
  for (int vertexIndex=0; vertexIndex < numVertices; vertexIndex++)
  {
    vector<int> dofOrdinals = basis->dofOrdinalsForVertex(vertexIndex);
    if (dofOrdinals.size() == 0) TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No dofOrdinal for vertex...");
    if (dofOrdinals.size() > 1) TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "More than one dofOrdinal per vertex...");
    vertexOrdinals.push_back(*(dofOrdinals.begin()));
  }
//
//  if (! checkVertexOrdinalsQuad(basis, vertexOrdinals) ) {
//    success = false;
//    cout << "vertex dof ordinals don't match expected\n";
//    cout << "ordinals: ";
//    for (int vertexIndex=0; vertexIndex < numVertices; vertexIndex++) {
//      cout << vertexOrdinals[vertexIndex] << " ";
//    }
//    cout << endl;
//  }

  // get the points in reference space for each vertex
  FieldContainer<double> points;
  if (numVertices == 2)   // line
  {
    points.resize(2,1);
    points(0,0) = -1;
    points(1,0) = 1;
  }
  else if (numVertices == 3)     // triangle
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "triangles not yet supported");
  }
  else if (numVertices == 4)     // quad
  {
    points.resize(4,2);
    points(0,0) = -1;
    points(0,1) = -1;
    points(1,0) =  1;
    points(1,1) = -1;
    points(2,0) =  1;
    points(2,1) =  1;
    points(3,0) = -1;
    points(3,1) =  1;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported topology");
  }

  FieldContainer<double> vertexValues;
  if (basis->rangeRank() > 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "rank > 0 bases not yet supported");
  }
  else
  {
    vertexValues.resize(basis->getCardinality(),numVertices);
  }

  basis->getValues(vertexValues, points, Intrepid::OPERATOR_VALUE);

  // test that the points are correctly classified
  for (int fieldIndex=0; fieldIndex<basis->getCardinality(); fieldIndex++)
  {
    for (int ptIndex=0; ptIndex<numVertices; ptIndex++)
    {
      int dofOrdinalForPoint = vertexOrdinals[ptIndex];
      bool expectZero = (dofOrdinalForPoint != fieldIndex);
      if (expectZero)
      {
        if (vertexValues(fieldIndex,ptIndex) != 0)
        {
          success = false;
          cout << "Expected 0 for fieldIndex " << fieldIndex << " and ptIndex " << ptIndex;
          cout << ", but got " << vertexValues(fieldIndex,ptIndex) << endl;
        }
      }
      else
      {
        if (vertexValues(fieldIndex,ptIndex) == 0)
        {
          cout << "Expected nonzero for fieldIndex " << fieldIndex << " and ptIndex " << ptIndex << endl;
          success = false;
        }
      }
    }
  }

  if (!success)
  {
    cout << "Failed testBasisClassifications; vertexValues:\n" << vertexValues;
  }

  return success;
}

bool LobattoBasisTests::testLobattoLineClassifications()
{
  bool success = true;
  bool conformingTrue = true;
  for (int polyOrder=1; polyOrder<20; polyOrder++)
  {
    BasisPtr lobattoBasis = Teuchos::rcp( new LobattoHGRAD_LineBasis<>(polyOrder,conformingTrue) );
    if (! testBasisClassifications(lobattoBasis) )
    {
      cout << "LobattoBasisTests::testLobattoLineClassifications() failed for polyOrder " << polyOrder << endl;
    }
  }
  // TODO: implement this
//  cout << "Warning: testLegendreLineClassifications unfinished.\n";

  return success;
}

bool LobattoBasisTests::testH1Classifications()
{
  // checks that edge functions, vertex functions, etc. are correctly listed for the H^1 Lobatto basis
  bool success = true;

  int rank = Teuchos::GlobalMPISession::getRank();

  bool conformingTrue = true;
  for (int polyOrder=1; polyOrder<20; polyOrder++)
  {
    BasisPtr lobattoBasis = Teuchos::rcp( new LobattoHGRAD_QuadBasis<>(polyOrder,conformingTrue) );
    if (! testBasisClassifications(lobattoBasis) )
    {
      if (rank==0)
        cout << "LobattoBasisTests::testH1Classifications() failed for polyOrder " << polyOrder << endl;
    }
  }
  // TODO: implement this
  if (rank==0)
    cout << "Warning: testH1Classification unfinished.\n";

  return success;
}

bool LobattoBasisTests::testSimpleStiffnessMatrix()
{
  bool success = true;

  int rank = Teuchos::GlobalMPISession::getRank();

  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr u = varFactory->fieldVar("u");
  VarPtr un = varFactory->fluxVar("un_hat");
  VarPtr v = varFactory->testVar("v", HGRAD);

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  vector<double> beta;
  beta.push_back(1);
  beta.push_back(1);
  bf->addTerm(beta * u, v->grad());
  bf->addTerm(un, v);

  DofOrderingPtr trialSpace = Teuchos::rcp( new DofOrdering(CellTopology::quad()) );
  DofOrderingPtr testSpace = Teuchos::rcp( new DofOrdering(CellTopology::quad()) );

  const int numSides = 4;
  const int spaceDim = 2;

  int fieldOrder = 3;
  int testOrder = fieldOrder+2;
  BasisPtr fieldBasis = Camellia::intrepidQuadHGRAD(fieldOrder);
  BasisPtr fluxBasis = Camellia::intrepidLineHGRAD(fieldOrder);
  trialSpace->addEntry(u->ID(), fieldBasis, fieldBasis->rangeRank());
  for (int i=0; i<numSides; i++)
  {
    trialSpace->addEntry(un->ID(), fluxBasis, fluxBasis->rangeRank(), i);
  }

  BasisPtr testBasis = Camellia::lobattoQuadHGRAD(testOrder+1,false); // +1 because it lives in HGRAD
  testSpace->addEntry(v->ID(), testBasis, testBasis->rangeRank());

  int numTrialDofs = trialSpace->totalDofs();
  int numTestDofs = testSpace->totalDofs();
  int numCells = 1;

  FieldContainer<double> cellNodes(numCells,numSides,spaceDim);
  cellNodes(0,0,0) = 0;
  cellNodes(0,0,1) = 0;
  cellNodes(0,1,0) = 1;
  cellNodes(0,1,1) = 0;
  cellNodes(0,2,0) = 1;
  cellNodes(0,2,1) = 1;
  cellNodes(0,3,0) = 0;
  cellNodes(0,3,1) = 1;

  FieldContainer<double> stiffness(numCells,numTestDofs,numTrialDofs);

  FieldContainer<double> cellSideParities(numCells,numSides);
  cellSideParities.initialize(1.0);

  CellTopoPtr quad_4 = Camellia::CellTopology::quad();
  Teuchos::RCP<ElementType> elemType = Teuchos::rcp( new ElementType(trialSpace, testSpace, quad_4));

  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType) );
  vector<GlobalIndexType> cellIDs;
  cellIDs.push_back(0);
  basisCache->setPhysicalCellNodes(cellNodes, cellIDs, true);
  bf->stiffnessMatrix(stiffness, elemType, cellSideParities, basisCache);

  // TODO: finish this test

//  cout << stiffness;
  if (rank==0)
    cout << "Warning: testSimpleStiffnessMatrix() unfinished.\n";

  return success;
}

string LobattoBasisTests::testSuiteName()
{
  return "LobattoBasisTests";
}
