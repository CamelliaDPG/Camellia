//
//  ScratchPadTests.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/5/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "ScratchPadTests.h"
#include "PenaltyConstraints.h"
#include "IP.h"
#include "PreviousSolutionFunction.h"
#include "RieszRep.h"
#include "TestingUtilities.h"
#include "MeshUtilities.h"
#include "MeshFactory.h"
#include "RefinementStrategy.h"


class UnitSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x+1.0) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y+1.0) < tol) || (abs(y-1.0) < tol);
    //    cout << "UnitSquareBoundary: for (" << x << ", " << y << "), (xMatch, yMatch) = (" << xMatch << ", " << yMatch << ")\n";
    return xMatch || yMatch;
  }
};

class SquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y) < tol) || (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

class InflowSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol);
    bool yMatch = (abs(y) < tol);
    return xMatch || yMatch;
  }
};

class LRInflowSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol); // left inflow
    bool yMatch = ((abs(y)<tol) || (abs(y-1.0)<tol)); // top/bottom
    return xMatch || yMatch;
  }
};
class LROutflowSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x-1.0)<tol);
    return xMatch;
  }
};

class Uinflow : public Function
{
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    double tol = 1e-11;
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0; i<cellIDs.size(); i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        double x = points(i,j,0);
        double y = points(i,j,1);
        values(i,j) = 0.0;
        if (abs(y)<tol)
        {
          values(i,j) = 1.0;
        }
        if (abs(x)<tol)
        {
          values(i,j) = -1.0;
        }
      }
    }
  }
};

// just for a discontinuity
class CellIDFunction : public Function
{
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0; i<cellIDs.size(); i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        values(i,j) = cellIDs[i];
      }
    }
  }
};

// is zero except on the edge (.5, y) on a 2x1 unit quad mesh - an edge restriction function
class EdgeFunction : public Function
{
public:
  bool boundaryValueOnly()
  {
    return true;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    double tol = 1e-11;
    vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0; i<cellIDs.size(); i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        double x = points(i,j,0);
        double y = points(i,j,1);
        if (abs(x-.5)<tol)
        {
          values(i,j) = 1.0;
        }
        else
        {
          values(i,j) = 0.0;
        }
      }
    }
  }
};

class PositiveX : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    return x > 0;
  }
};

void ScratchPadTests::setup()
{
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr tau = varFactory->testVar("\\tau", HDIV);
  VarPtr v = varFactory->testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory->traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory->fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory->fieldVar("u");
  VarPtr sigma1 = varFactory->fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory->fieldVar("\\sigma_2");

  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);

  double eps = 1e-2;

  // standard confusion bilinear form
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
  for (int i=0; i<NUM_POINTS_1D; i++)
  {
    for (int j=0; j<NUM_POINTS_1D; j++)
    {
      _testPoints(i*NUM_POINTS_1D + j, 0) = x[i];
      _testPoints(i*NUM_POINTS_1D + j, 1) = y[i];
    }
  }

  _elemType = _spectralConfusionMesh->getElement(0)->elementType();
  vector<GlobalIndexType> cellIDs;
  int cellID = 0;
  cellIDs.push_back(cellID);
  _basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) );
  _basisCache->setRefCellPoints(_testPoints);

  _basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID), cellIDs, true );

}

void ScratchPadTests::teardown()
{

}

void ScratchPadTests::runTests(int &numTestsRun, int &numTestsPassed)
{
  setup();
  if (testIntegrateDiscontinuousFunction())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testPenaltyConstraints())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testSpatiallyFilteredFunction())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testConstantFunctionProduct())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testLinearTermEvaluationConsistency())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();


  setup();
  if (testRieszIntegration())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testLTResidualSimple())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testLTResidual())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testResidualMemoryError())
  {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  /*
  setup();
  if (testGalerkinOrthogonality()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  */
}

bool ScratchPadTests::testConstantFunctionProduct()
{
  bool success = true;
  // set up basisCache (even though it won't really be used here)
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) );
  vector<GlobalIndexType> cellIDs;
  int cellID = 0;
  cellIDs.push_back(cellID);
  basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID),
                                    cellIDs, true );

  int numCells = _basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = _testPoints.dimension(0);
  FunctionPtr three = Function::constant(3.0);
  FunctionPtr two = Function::constant(2.0);

  FieldContainer<double> values(numCells,numPoints);
  two->values(values,basisCache);
  three->scalarMultiplyBasisValues( values, basisCache );

  FieldContainer<double> expectedValues(numCells,numPoints);
  expectedValues.initialize( 3.0 * 2.0 );

  double tol = 1e-15;
  double maxDiff = 0.0;
  if ( ! fcsAgree(expectedValues, values, tol, maxDiff) )
  {
    success = false;
    cout << "Expected product differs from actual; maxDiff: " << maxDiff << endl;
  }
  return success;
}

bool ScratchPadTests::testPenaltyConstraints()
{
  bool success = true;
  int numCells = 1;
  FunctionPtr one = Function::constant(1.0);

  SpatialFilterPtr entireBoundary = Teuchos::rcp( new UnitSquareBoundary );

  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  pc->addConstraint(_uhat_confusion==one,entireBoundary);

  FieldContainer<double> localRHSVector(numCells,_elemType->trialOrderPtr->totalDofs());
  FieldContainer<double> localStiffness(numCells,_elemType->trialOrderPtr->totalDofs(),
                                        _elemType->trialOrderPtr->totalDofs());

  // Our basis for uhat is 1-x, 1+x -- we should figure out what that means for
  // the values of the integrals that go into expectedStiffness.  For now, focus
  // on the sparsity pattern.

  int trialDofs = _elemType->trialOrderPtr->totalDofs();
  FieldContainer<double> expectedSparsity(numCells,_elemType->trialOrderPtr->totalDofs(),
                                          _elemType->trialOrderPtr->totalDofs());
  FieldContainer<double> expectedRHSSparsity(numCells,_elemType->trialOrderPtr->totalDofs());

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int sideIndex=0; sideIndex<4; sideIndex++)
    {
      vector<int> uhat_dofIndices = _elemType->trialOrderPtr->getDofIndices(_uhat_confusion->ID(),sideIndex);

      for (int dofOrdinal1=0; dofOrdinal1 < uhat_dofIndices.size(); dofOrdinal1++)
      {
        int dofIndex1 = uhat_dofIndices[dofOrdinal1];
        expectedRHSSparsity(cellIndex,dofIndex1) = 1.0;
        for (int dofOrdinal2=0; dofOrdinal2 < uhat_dofIndices.size(); dofOrdinal2++)
        {
          int dofIndex2 = uhat_dofIndices[dofOrdinal2];
          expectedSparsity(cellIndex,dofIndex1,dofIndex2) = 1.0;
        }
      }
    }
  }

  BCPtr bc = BC::bc();

  pc->filter(localStiffness, localRHSVector, _basisCache, _spectralConfusionMesh, bc);

  //  cout << "testPenaltyConstraints: expectedStiffnessSparsity:\n" << expectedSparsity;
  //  cout << "testPenaltyConstraints: localStiffness:\n" << localStiffness;
  //
  //  cout << "testPenaltyConstraints: expectedRHSSparsity:\n" << expectedRHSSparsity;
  //  cout << "testPenaltyConstraints: localRHSVector:\n" << localRHSVector;

  // compare sparsity
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int i=0; i<trialDofs; i++)
    {
      double rhsValue = localRHSVector(cellIndex,i);
      double rhsSparsityValue = expectedRHSSparsity(cellIndex,i);
      if ((rhsSparsityValue == 0.0) && (rhsValue != 0.0))
      {
        cout << "testPenaltyConstraints rhs: expected 0 but found " << rhsValue << " at i = " << i << ".\n";
        success = false;
      }
      if ((rhsSparsityValue != 0.0) && (rhsValue == 0.0))
      {
        cout << "testPenaltyConstraints rhs: expected nonzero but found 0 at i = " << i << ".\n";
        success = false;
      }
      for (int j=0; j<trialDofs; j++)
      {
        double stiffValue = localStiffness(cellIndex,i,j);
        double sparsityValue = expectedSparsity(cellIndex,i,j);
        if ((sparsityValue == 0.0) && (stiffValue != 0.0))
        {
          cout << "testPenaltyConstraints stiffness: expected 0 but found " << stiffValue << " at (" << i << ", " << j << ").\n";
          success = false;
        }
        if ((sparsityValue != 0.0) && (stiffValue == 0.0))
        {
          cout << "testPenaltyConstraints stiffness: expected nonzero but found 0 at (" << i << ", " << j << ").\n";
          success = false;
        }
      }
    }
  }
  return success;
}

bool ScratchPadTests::testSpatiallyFilteredFunction()
{
  bool success = true;
  FunctionPtr one = Function::constant(1.0);
  SpatialFilterPtr positiveX = Teuchos::rcp( new PositiveX );
  FunctionPtr heaviside = Teuchos::rcp( new SpatiallyFilteredFunction<double>(one, positiveX) );

  int numCells = _basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = _testPoints.dimension(0);

  FieldContainer<double> values(numCells,numPoints);
  FieldContainer<double> expectedValues(numCells,numPoints);

  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      double x = _basisCache->getPhysicalCubaturePoints()(cellIndex,ptIndex,0);
      if (x > 0)
      {
        expectedValues(cellIndex,ptIndex) = 1.0;
      }
      else
      {
        expectedValues(cellIndex,ptIndex) = 0.0;
      }
    }
  }

  heaviside->values(values,_basisCache);

  double tol = 1e-15;
  double maxDiff = 0.0;
  if ( ! fcsAgree(expectedValues, values, tol, maxDiff) )
  {
    success = false;
    cout << "testSpatiallyFilteredFunction: Expected values differ from actual; maxDiff: " << maxDiff << endl;
  }
  return success;
}

// tests whether a mixed type LT
bool ScratchPadTests::testLinearTermEvaluationConsistency()
{
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr v = varFactory->testVar("v", HGRAD);

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  // define trial variables
  VarPtr beta_n_u = varFactory->fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory->fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );

  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  // define nodes for mesh
  int order = 1;
  int H1Order = order+1;
  int pToAdd = 1;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(1, convectionBF, H1Order, H1Order+pToAdd);

  ////////////////////   get fake residual   ///////////////////////

  LinearTermPtr lt = Teuchos::rcp(new LinearTerm);
  FunctionPtr edgeFxn = Teuchos::rcp(new EdgeFunction);
  FunctionPtr Xsq = Function::xn(2);
  FunctionPtr Ysq = Function::yn(2);
  FunctionPtr XYsq = Xsq*Ysq;
  lt->addTerm(edgeFxn*v + (beta*XYsq)*v->grad());

  Teuchos::RCP<RieszRep> ltRiesz = Teuchos::rcp(new RieszRep(mesh, ip, lt));
  ltRiesz->computeRieszRep();
  FunctionPtr repFxn = RieszRep::repFunction(v,ltRiesz);
  map<int,FunctionPtr> rep_map;
  rep_map[v->ID()] = repFxn;

  FunctionPtr edgeLt = lt->evaluate(rep_map, true) ;
  FunctionPtr elemLt = lt->evaluate(rep_map, false);

  double edgeVal = edgeLt->integrate(mesh,10);
  double elemVal = elemLt->integrate(mesh,10);
  LinearTermPtr edgeOnlyLt = Teuchos::rcp(new LinearTerm);// residual
  edgeOnlyLt->addTerm(edgeFxn*v);
  FunctionPtr edgeOnly = edgeOnlyLt->evaluate(rep_map,true);
  double edgeOnlyVal = edgeOnly->integrate(mesh,10);

  double diff = edgeOnlyVal-edgeVal;
  if (abs(diff)>1e-11)
  {
    success = false;
    cout << "Failed testLinearTermEvaluationConsistency() with diff = " << diff << endl;
  }

  return success;
}

class IndicatorFunction : public Function
{
  set<int> _cellIDs;
public:
  IndicatorFunction(set<int> cellIDs) : Function(0)
  {
    _cellIDs = cellIDs;
  }
  IndicatorFunction(int cellID) : Function(0)
  {
    _cellIDs.insert(cellID);
  }
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    values.initialize(1.0);
    vector<GlobalIndexType> contextCellIDs = basisCache->cellIDs();
    int cellIndex=0; // keep track of index into values

    int entryCount = values.size();
    int numCells = values.dimension(0);
    int numEntriesPerCell = entryCount / numCells;

    for (vector<GlobalIndexType>::iterator cellIt = contextCellIDs.begin(); cellIt != contextCellIDs.end(); cellIt++)
    {
      GlobalIndexType cellID = *cellIt;
      if (_cellIDs.find(cellID) == _cellIDs.end())
      {
        // clear out the associated entries
        for (int j=0; j<numEntriesPerCell; j++)
        {
          values[cellIndex*numEntriesPerCell + j] = 0;
        }
      }
      cellIndex++;
    }
  }
};

// tests whether a mixed type LT
bool ScratchPadTests::testIntegrateDiscontinuousFunction()
{
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr v = varFactory->testVar("v", HGRAD);

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  // for projections
  IPPtr ipL2 = Teuchos::rcp(new IP);
  ipL2->addTerm(v);

  // define trial variables
  VarPtr beta_n_u = varFactory->fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory->fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );

  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  // define nodes for mesh
  int order = 1;
  int H1Order = order+1;
  int pToAdd = 1;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(2, 1, convectionBF, H1Order, H1Order+pToAdd);

  ////////////////////   integrate discontinuous function - cellIDFunction   ///////////////////////

  //  FunctionPtr cellIDFxn = Teuchos::rcp(new CellIDFunction); // should be 0 on cellID 0, 1 on cellID 1
  set<int> cellIDs;
  cellIDs.insert(1); // 0 on cell 0, 1 on cell 1
  FunctionPtr indicator = Teuchos::rcp(new IndicatorFunction(cellIDs)); // should be 0 on cellID 0, 1 on cellID 1
  double jumpWeight = 13.3; // some random number
  FunctionPtr edgeRestrictionFxn = Teuchos::rcp(new EdgeFunction);
  FunctionPtr X = Function::xn(1);
  LinearTermPtr integrandLT = Function::constant(1.0)*v + Function::constant(jumpWeight)*X*edgeRestrictionFxn*v;

  // make riesz representation function to more closely emulate the error rep
  LinearTermPtr indicatorLT = Teuchos::rcp(new LinearTerm);// residual
  indicatorLT->addTerm(indicator*v);
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ipL2, indicatorLT));
  riesz->computeRieszRep();
  map<int,FunctionPtr> vmap;
  vmap[v->ID()] = RieszRep::repFunction(v,riesz); // SHOULD BE L2 projection = same thing!!!

  FunctionPtr volumeIntegrand = integrandLT->evaluate(vmap,false);
  FunctionPtr edgeRestrictedIntegrand = integrandLT->evaluate(vmap,true);

  double edgeRestrictedValue = volumeIntegrand->integrate(mesh,10) + edgeRestrictedIntegrand->integrate(mesh,10);

  double expectedValue = .5 + .5*jumpWeight;
  double diff = abs(expectedValue-edgeRestrictedValue);
  if (abs(diff)>1e-11)
  {
    success = false;
    cout << "Failed testIntegrateDiscontinuousFunction() with expectedValue = " << expectedValue << " and actual value = " << edgeRestrictedValue << endl;
  }
  return success;
}

struct DofInfo
{
  int cellID;
  int trialID;
  int basisOrdinal;
  int basisCardinality;
  int sideIndex;
  int numSides;
  int localDofIndex; // index into trial ordering
  int totalDofs;     // number of dofs in the trial ordering
};

string dofInfoString(const DofInfo &info)
{
  ostringstream dis;
  dis << "cellID = " << info.cellID << "; trialID = " << info.trialID;
  dis << "; sideIndex = " << info.sideIndex << " (" << info.numSides << " total sides)";
  dis << "; basisOrdinal = " << info.basisOrdinal << "; cardinality = " << info.basisCardinality;
  return dis.str();
}

string dofInfoString(const vector<DofInfo> infoVector)
{
  ostringstream dis;
  for (vector<DofInfo>::const_iterator infoIt=infoVector.begin(); infoIt != infoVector.end(); infoIt++)
  {
    dis << dofInfoString(*infoIt) << endl;
  }
  return dis.str();
}

map< int, vector<DofInfo> > constructGlobalDofToLocalDofInfoMap(MeshPtr mesh)
{
  // go through the mesh as a whole, and collect info for each dof
  map< int, vector<DofInfo> > infoMap;
  DofInfo info;
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  for (set<GlobalIndexType>::iterator cellIt = activeCellIDs.begin(); cellIt != activeCellIDs.end(); cellIt++)
  {
    GlobalIndexType cellID = *cellIt;
    info.cellID = cellID;
    ElementPtr element = mesh->getElement(cellID);
    DofOrderingPtr trialOrder = element->elementType()->trialOrderPtr;
    set<int> trialIDs = trialOrder->getVarIDs();
    info.totalDofs = trialOrder->totalDofs();
    for (set<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++)
    {
      info.trialID = *trialIt;
      const vector<int>* sidesForVar = &trialOrder->getSidesForVarID(info.trialID);
      for (vector<int>::const_iterator sideIt = sidesForVar->begin(); sideIt != sidesForVar->end(); sideIt++)
      {
        int sideIndex = *sideIt;
        info.sideIndex = sideIndex;
        info.basisCardinality = trialOrder->getBasisCardinality(info.trialID, info.sideIndex);
        for (int basisOrdinal=0; basisOrdinal < info.basisCardinality; basisOrdinal++)
        {
          info.basisOrdinal = basisOrdinal;
          info.localDofIndex = trialOrder->getDofIndex(info.trialID, info.basisOrdinal, info.sideIndex);
          pair<int, int> localDofIndexKey = make_pair(info.cellID, info.localDofIndex);
          int globalDofIndex = mesh->getLocalToGlobalMap().find(localDofIndexKey)->second;
//          cout << "(" << info.cellID << "," << info.localDofIndex << ") --> " << globalDofIndex << endl;
          infoMap[globalDofIndex].push_back(info);
        }
      }
    }
  }
  return infoMap;
}

bool ScratchPadTests::testGalerkinOrthogonality()
{

  double tol = 1e-11;
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr v = varFactory->testVar("v", HGRAD);

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  // define trial variables
  VarPtr beta_n_u = varFactory->fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory->fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );

  FunctionPtr n = Function::normal();
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  // define nodes for mesh
  int order = 2;
  int H1Order = order+1;
  int pToAdd = 1;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(4, convectionBF, H1Order, H1Order+pToAdd);

  ////////////////////   SOLVE   ///////////////////////

  RHSPtr rhs = RHS::rhs();
  BCPtr bc = BC::bc();
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(inflowBoundary) );

  FunctionPtr uIn;
  uIn = Teuchos::rcp(new Uinflow); // uses a discontinuous piecewise-constant basis function on left and bottom sides of square
  bc->addDirichlet(beta_n_u, inflowBoundary, beta*n*uIn);

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  solution->solve(false);
  FunctionPtr uFxn = Function::solution(u, solution);
  FunctionPtr fnhatFxn = Function::solution(beta_n_u,solution);

  // make residual for riesz representation function
  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  FunctionPtr parity = Function::sideParity();
  residual->addTerm(-fnhatFxn*v + (beta*uFxn)*v->grad());
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  riesz->computeRieszRep();
  map<int,FunctionPtr> err_rep_map;
  err_rep_map[v->ID()] = RieszRep::repFunction(v,riesz);

  ////////////////////   GET BOUNDARY CONDITION DATA    ///////////////////////

  FieldContainer<GlobalIndexType> bcGlobalIndices;
  FieldContainer<double> bcGlobalValues;
  mesh->boundary().bcsToImpose(bcGlobalIndices,bcGlobalValues,*(solution->bc()), NULL, NULL);
  set<int> bcInds;
  for (int i=0; i<bcGlobalIndices.dimension(0); i++)
  {
    bcInds.insert(bcGlobalIndices(i));
  }

  ////////////////////   CHECK GALERKIN ORTHOGONALITY   ///////////////////////

  BCPtr nullBC;
  RHSPtr nullRHS;
  IPPtr nullIP;
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  map< int, vector<DofInfo> > infoMap = constructGlobalDofToLocalDofInfoMap(mesh);

  for (map< int, vector<DofInfo> >::iterator mapIt = infoMap.begin();
       mapIt != infoMap.end(); mapIt++)
  {
    int dofIndex = mapIt->first;
    vector< DofInfo > dofInfoVector = mapIt->second; // all the local dofs that map to dofIndex
    // create perturbation in direction du
    solnPerturbation->clear(); // clear all solns
    // set each corresponding local dof to 1.0
    for (vector< DofInfo >::iterator dofInfoIt = dofInfoVector.begin();
         dofInfoIt != dofInfoVector.end(); dofInfoIt++)
    {
      DofInfo info = *dofInfoIt;
      FieldContainer<double> solnCoeffs(info.basisCardinality);
      solnCoeffs(info.basisOrdinal) = 1.0;
      solnPerturbation->setSolnCoeffsForCellID(solnCoeffs, info.cellID, info.trialID, info.sideIndex);
    }
    //    solnPerturbation->setSolnCoeffForGlobalDofIndex(1.0,dofIndex);

    LinearTermPtr b_du =  convectionBF->testFunctional(solnPerturbation);
    FunctionPtr gradient = b_du->evaluate(err_rep_map, TestingUtilities::isFluxOrTraceDof(mesh,dofIndex)); // use boundary part only if flux
    double grad = gradient->integrate(mesh,10);
    if (!TestingUtilities::isFluxOrTraceDof(mesh,dofIndex) && abs(grad)>tol)  // if we're not single-precision zero FOR FIELDS
    {
      //      int cellID = mesh->getGlobalToLocalMap()[dofIndex].first;
      cout << "Failed testGalerkinOrthogonality() for fields with diff " << abs(grad) << " at dof " << dofIndex << "; info:" << endl;
      cout << dofInfoString(infoMap[dofIndex]);
      success = false;
    }
  }
  FieldContainer<double> errorJumps(mesh->numGlobalDofs()); //initialized to zero
  // just test fluxes ON INTERNAL SKELETON here
  vector<ElementPtr> elems = mesh->activeElements();
  for (vector<ElementPtr>::iterator elemIt=elems.begin(); elemIt!=elems.end(); elemIt++)
  {
    for (int sideIndex = 0; sideIndex < 4; sideIndex++)
    {
      ElementPtr elem = *elemIt;
      ElementTypePtr elemType = elem->elementType();
      vector<int> localDofIndices = elemType->trialOrderPtr->getDofIndices(beta_n_u->ID(), sideIndex);
      for (int i = 0; i<localDofIndices.size(); i++)
      {
        int globalDofIndex = mesh->globalDofIndex(elem->cellID(), localDofIndices[i]);
        vector< DofInfo > dofInfoVector = infoMap[globalDofIndex];

        solnPerturbation->clear();
        TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,1.0,globalDofIndex);
        // also add in BCs
        for (int i = 0; i<bcGlobalIndices.dimension(0); i++)
        {
          TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,bcGlobalValues(i),bcGlobalIndices(i));
        }

        LinearTermPtr b_du =  convectionBF->testFunctional(solnPerturbation);
        FunctionPtr gradient = b_du->evaluate(err_rep_map, TestingUtilities::isFluxOrTraceDof(mesh,globalDofIndex)); // use boundary part only if flux
        double jump = gradient->integrate(mesh,10);
        errorJumps(globalDofIndex) += jump;
      }
    }
  }
  for (int i = 0; i<mesh->numGlobalDofs(); i++)
  {
    if (abs(errorJumps(i))>tol)
    {
      cout << "Failing Galerkin orthogonality test for fluxes with diff " << errorJumps(i) << " at dof " << i << endl;
      cout << dofInfoString(infoMap[i]);
      success = false;
    }
  }

  return success;
}

// tests to make sure that the rieszNorm computed via matrices is the same as the one computed thru direct integration
bool ScratchPadTests::testRieszIntegration()
{
  double tol = 1e-11;
  bool success = true;

  int nCells = 2;
  double eps = .25;

  ////////////////////   DECLARE VARIABLES   ///////////////////////

  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr tau = varFactory->testVar("\\tau", HDIV);
  VarPtr v = varFactory->testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory->traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory->fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory->fieldVar("u");
  VarPtr sigma1 = varFactory->fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory->fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());

  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);

  // just H1 projection
  ip->addTerm(v->grad());
  ip->addTerm(v);
  ip->addTerm(tau);
  ip->addTerm(tau->div());

  ////////////////////   SPECIFY RHS AND HELPFUL FUNCTIONS   ///////////////////////

  FunctionPtr n = Function::normal();
  vector<double> e1,e2;
  e1.push_back(1.0);
  e1.push_back(0.0);
  e2.push_back(0.0);
  e2.push_back(1.0);
  FunctionPtr one = Function::constant(1.0);

  FunctionPtr zero = Function::constant(0.0);
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = one;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr squareBoundary = Teuchos::rcp( new SquareBoundary );

  bc->addDirichlet(uhat, squareBoundary, zero);

  ////////////////////   BUILD MESH   ///////////////////////

  // define nodes for mesh
  int order = 2;
  int H1Order = order+1;
  int pToAdd = 2;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);

  ////////////////////   SOLVE & REFINE   ///////////////////////

  LinearTermPtr lt = Teuchos::rcp(new LinearTerm);
  FunctionPtr fxn = Function::xn(1); // fxn = x
  lt->addTerm(fxn*v + fxn->grad()*v->grad());
  lt->addTerm(fxn*tau->x() + fxn*tau->y() + (fxn->dx() + fxn->dy())*tau->div());
  Teuchos::RCP<RieszRep> rieszLT = Teuchos::rcp(new RieszRep(mesh, ip, lt));
  rieszLT->computeRieszRep();
  double rieszNorm = rieszLT->getNorm();
  FunctionPtr e_v = RieszRep::repFunction(v,rieszLT);
  FunctionPtr e_tau = RieszRep::repFunction(tau,rieszLT);
  map<int,FunctionPtr> repFxns;
  repFxns[v->ID()] = e_v;
  repFxns[tau->ID()] = e_tau;

  double integratedNorm = sqrt((lt->evaluate(repFxns,false))->integrate(mesh,5,true));
  success = abs(rieszNorm-integratedNorm)<tol;
  if (success==false)
  {
    cout << "Failed testRieszIntegration; riesz norm is computed to be = " << rieszNorm << ", while using integration it's computed to be " << integratedNorm << endl;
    return success;
  }
  return success;
}

// tests residual computation on simple convection
bool ScratchPadTests::testLTResidualSimple()
{
  double tol = 1e-11;
  int rank = Teuchos::GlobalMPISession::getRank();

  bool success = true;

  int nCells = 2;

  ////////////////////   DECLARE VARIABLES   ///////////////////////

  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr v = varFactory->testVar("v", HGRAD);

  // define trial variables
  VarPtr beta_n_u = varFactory->fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory->fieldVar("u");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // v terms:
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);

  // choose the mesh-independent norm even though it may have BLs
  ip->addTerm(v->grad());
  ip->addTerm(v);

  ////////////////////   SPECIFY RHS AND HELPFUL FUNCTIONS   ///////////////////////

  FunctionPtr n = Function::normal();
  vector<double> e1,e2;
  e1.push_back(1.0);
  e1.push_back(0.0);
  e2.push_back(0.0);
  e2.push_back(1.0);
  FunctionPtr one = Function::constant(1.0);

  FunctionPtr zero = Function::constant(0.0);
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = one;
  rhs->addTerm( f * v );

  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr boundary = Teuchos::rcp( new InflowSquareBoundary );
  FunctionPtr u_in = Teuchos::rcp(new Uinflow);
  bc->addDirichlet(beta_n_u, boundary, beta*n*u_in);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  int H1Order = order+1;
  int pToAdd = 2;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);

  ////////////////////   SOLVE & REFINE   ///////////////////////

  int cubEnrich = 0;

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  solution->solve(false);
  double energyError = solution->energyErrorTotal();

  LinearTermPtr residual = rhs->linearTermCopy();
  residual->addTerm(-confusionBF->testFunctional(solution),true);

  Teuchos::RCP<RieszRep> rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  rieszResidual->computeRieszRep(cubEnrich);
  double energyErrorLT = rieszResidual->getNorm();

  bool testVsTest = true;
  FunctionPtr e_v = RieszRep::repFunction(v,rieszResidual);
  map<int,FunctionPtr> errFxns;
  errFxns[v->ID()] = e_v;
  FunctionPtr err = (ip->evaluate(errFxns,false))->evaluate(errFxns,false); // don't need boundary terms unless they're in IP
  double energyErrorIntegrated = sqrt(err->integrate(mesh,cubEnrich,testVsTest));
  // check that energy error computed thru Solution and through rieszRep are the same
  success = abs(energyError-energyErrorLT) < tol;
  if (success==false)
  {
    if (rank==0)
      cout << "Failed testLTResidualSimple; energy error = " << energyError << ", while linearTerm error is computed to be " << energyErrorLT << endl;
    return success;
  }
  // checks that matrix-computed and integrated errors are the same
  success = abs(energyErrorLT-energyErrorIntegrated)<tol;
  if (success==false)
  {
    if (rank==0)
      cout << "Failed testLTResidualSimple; energy error = " << energyError << ", while error computed via integration is " << energyErrorIntegrated << endl;
    return success;
  }
  return success;
}

// tests to make sure the energy error calculated thru direct integration works for vector valued test functions too
bool ScratchPadTests::testLTResidual()
{
  double tol = 1e-11;
  int rank = Teuchos::GlobalMPISession::getRank();

  bool success = true;

  int nCells = 2;
  double eps = .1;

  ////////////////////   DECLARE VARIABLES   ///////////////////////

  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr tau = varFactory->testVar("\\tau", HDIV);
  VarPtr v = varFactory->testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory->traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory->fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory->fieldVar("u");
  VarPtr sigma1 = varFactory->fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory->fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());

  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);

  // choose the mesh-independent norm even though it may have boundary layers
  ip->addTerm(v->grad());
  ip->addTerm(v);
  ip->addTerm(tau);
  ip->addTerm(tau->div());

  ////////////////////   SPECIFY RHS AND HELPFUL FUNCTIONS   ///////////////////////

  FunctionPtr n = Function::normal();
  vector<double> e1,e2;
  e1.push_back(1.0);
  e1.push_back(0.0);
  e2.push_back(0.0);
  e2.push_back(1.0);
  FunctionPtr one = Function::constant(1.0);

  FunctionPtr zero = Function::constant(0.0);
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = one; // if this is set to zero instead, we pass the test (a clue?)
  rhs->addTerm( f * v );

  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr squareBoundary = Teuchos::rcp( new SquareBoundary );

  bc->addDirichlet(uhat, squareBoundary, one);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  int H1Order = order+1;
  int pToAdd = 2;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);

  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  solution->solve(false);
  double energyError = solution->energyErrorTotal();

  LinearTermPtr residual = rhs->linearTermCopy();
  residual->addTerm(-confusionBF->testFunctional(solution),true);

//  FunctionPtr uh = Function::solution(uhat,solution);
//  FunctionPtr fn = Function::solution(beta_n_u_minus_sigma_n,solution);
//  FunctionPtr uF = Function::solution(u,solution);
//  FunctionPtr sigma = e1*Function::solution(sigma1,solution)+e2*Function::solution(sigma2,solution);
//  residual->addTerm(- (fn*v - uh*tau->dot_normal()));
//  residual->addTerm(- (uF*(tau->div() - beta*v->grad()) + sigma*((1/eps)*tau + v->grad())));
//  residual->addTerm(-(fn*v - uF*beta*v->grad() + sigma*v->grad())); // just v portion
//  residual->addTerm(uh*tau->dot_normal() - uF*tau->div() - sigma*((1/eps)*tau)); // just tau portion

  Teuchos::RCP<RieszRep> rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  rieszResidual->computeRieszRep();
  double energyErrorLT = rieszResidual->getNorm();

  int cubEnrich = 0;
  bool testVsTest = true;
  FunctionPtr e_v = RieszRep::repFunction(v,rieszResidual);
  FunctionPtr e_tau = RieszRep::repFunction(tau,rieszResidual);
  // experiment by Nate: manually specify the error (this appears to produce identical results, as it should)
//  FunctionPtr err = e_v * e_v + e_tau * e_tau + e_v->grad() * e_v->grad() + e_tau->div() * e_tau->div();
  map<int,FunctionPtr> errFxns;
  errFxns[v->ID()] = e_v;
  errFxns[tau->ID()] = e_tau;
  LinearTermPtr ipAtErrFxns = ip->evaluate(errFxns);
  FunctionPtr err = ip->evaluate(errFxns)->evaluate(errFxns);
  double energyErrorIntegrated = sqrt(err->integrate(mesh,cubEnrich,testVsTest));

  // check that energy error computed thru Solution and through rieszRep are the same
  bool success1 = abs(energyError-energyErrorLT)<tol;
  // checks that matrix-computed and integrated errors are the same
  bool success2 = abs(energyErrorLT-energyErrorIntegrated)<tol;
  success = success1==true && success2==true;
  if (!success)
  {
    if (rank==0)
      cout << "Failed testLTResidual; energy error = " << energyError << ", while linearTerm error is computed to be " << energyErrorLT << ", and when computing through integration of the Riesz rep function, error = " << energyErrorIntegrated << endl;
  }
  //  VTKExporter exporter(solution, mesh, varFactory);
  //  exporter.exportSolution("testLTRes");
  //  cout << endl;

  return success;
}

bool ScratchPadTests::testResidualMemoryError()
{

  int rank = Teuchos::GlobalMPISession::getRank();

  double tol = 1e-11;
  bool success = true;

  int nCells = 2;
  double eps = 1e-2;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactoryPtr varFactory = VarFactory::varFactory();
  VarPtr tau = varFactory->testVar("\\tau", HDIV);
  VarPtr v = varFactory->testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory->traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory->fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory->fieldVar("u");
  VarPtr sigma1 = varFactory->fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory->fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());

  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  robIP->addTerm(tau);
  robIP->addTerm(tau->div());
  robIP->addTerm(v->grad());
  robIP->addTerm(v);

  ////////////////////   SPECIFY RHS   ///////////////////////

  FunctionPtr zero = Function::constant(0.0);
  FunctionPtr one = Function::constant(1.0);
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = zero;
  //  FunctionPtr f = one;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new LRInflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new LROutflowSquareBoundary);

  FunctionPtr n = Function::normal();

  vector<double> e1,e2;
  e1.push_back(1.0);
  e1.push_back(0.0);
  e2.push_back(0.0);
  e2.push_back(1.0);

  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*one);
  bc->addDirichlet(uhat, outflowBoundary, zero);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  int H1Order = order+1;
  int pToAdd = 2;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);
  //  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  solution->solve(false);
  mesh->registerSolution(solution);
  double energyErr1 = solution->energyErrorTotal();

  LinearTermPtr residual = rhs->linearTermCopy();
  residual->addTerm(-confusionBF->testFunctional(solution));
  RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, robIP, residual));
  rieszResidual->computeRieszRep();
  FunctionPtr e_v = RieszRep::repFunction(v,rieszResidual);
  FunctionPtr e_tau = RieszRep::repFunction(tau,rieszResidual);

  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );

  refinementStrategy.refine();
  solution->solve(false);
  double energyErr2 = solution->energyErrorTotal();

  // if energy error rises
  if (energyErr1 < energyErr2)
  {
    if (rank==0)
      cout << "energy error increased from " << energyErr1 << " to " << energyErr2 << " after refinement.\n";
    success = false;
  }

  return success;
}


std::string ScratchPadTests::testSuiteName()
{
  return "ScratchPadTests";
}
