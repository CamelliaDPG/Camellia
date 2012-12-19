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

class UnitSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x+1.0) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y+1.0) < tol) || (abs(y-1.0) < tol);
    //    cout << "UnitSquareBoundary: for (" << x << ", " << y << "), (xMatch, yMatch) = (" << xMatch << ", " << yMatch << ")\n";
    return xMatch || yMatch;
  }
};

class InflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol);
    bool yMatch = (abs(y) < tol);
    return xMatch || yMatch;
  }
};

class Uinflow : public Function {
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    double tol = 1e-11;
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0;i<cellIDs.size();i++){
      for (int j = 0;j<numPoints;j++){
        double x = points(i,j,0);
        double y = points(i,j,1);
        values(i,j) = 0.0;
        if (abs(y)<tol){
          values(i,j) = 1.0;
        }
        if (abs(x)<tol){
          values(i,j) = -1.0;
        }
      }
    }
  }
};

// just for a discontinuity
class CellIDFunction : public Function {
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0;i<cellIDs.size();i++){
      for (int j = 0;j<numPoints;j++){
        values(i,j) = cellIDs[i];
      }
    }
  }
};

// is zero except on the edge (.5, y) on a 2x1 unit quad mesh - an edge restriction function
class EdgeFunction : public Function {
public:
  bool boundaryValueOnly() { 
    return true; 
  } 
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    double tol = 1e-11;
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0;i<cellIDs.size();i++){
      for (int j = 0;j<numPoints;j++){
        double x = points(i,j,0);
        double y = points(i,j,1);
        if (abs(x-.5)<tol){
          values(i,j) = 1.0;
        } else {
          values(i,j) = 0.0;
        }
      }
    }
  }
};

// is zero on inflow
class InflowCutoffFunction : public Function {
public:
  bool boundaryValueOnly() { 
    return true; 
  } 
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    double tol = 1e-11;
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0;i<cellIDs.size();i++){
      for (int j = 0;j<numPoints;j++){
        double x = points(i,j,0);
        double y = points(i,j,1);
        values(i,j) = 1.0;
        bool isOnInflow = (abs(y)<tol) || (abs(x)<tol) ;
        if (isOnInflow){
          values(i,j) = 0.0;
        }
      }
    }
  }
};


class PositiveX : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    return x > 0;
  }
};

void ScratchPadTests::setup() {
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
  _basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) );
  _basisCache->setRefCellPoints(_testPoints);
  
  _basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID), cellIDs, true );

}

void ScratchPadTests::teardown() {
  
}

void ScratchPadTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testPenaltyConstraints()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testSpatiallyFilteredFunction()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testConstantFunctionProduct()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testLinearTermEvaluationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();   

  setup();
  if (testIntegrateDiscontinuousFunction()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();   

  setup();
  if (testErrorRepConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();   
     
  setup();
  if (testGalerkinOrthogonality()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();     
  
}

bool ScratchPadTests::testConstantFunctionProduct() {
  bool success = true;
  // set up basisCache (even though it won't really be used here)
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache( _elemType, _spectralConfusionMesh ) );
  vector<int> cellIDs;
  int cellID = 0;
  cellIDs.push_back(cellID);
  basisCache->setPhysicalCellNodes( _spectralConfusionMesh->physicalCellNodesForCell(cellID), 
				    cellIDs, true );
  
  int numCells = _basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = _testPoints.dimension(0);
  FunctionPtr three = Teuchos::rcp( new ConstantScalarFunction(3.0) );
  FunctionPtr two = Teuchos::rcp( new ConstantScalarFunction(2.0) );

  FieldContainer<double> values(numCells,numPoints);
  two->values(values,basisCache);
  three->scalarMultiplyBasisValues( values, basisCache );
  
  FieldContainer<double> expectedValues(numCells,numPoints);
  expectedValues.initialize( 3.0 * 2.0 );
  
  double tol = 1e-15;
  double maxDiff = 0.0;
  if ( ! fcsAgree(expectedValues, values, tol, maxDiff) ) {
    success = false;
    cout << "Expected product differs from actual; maxDiff: " << maxDiff << endl;
  }
  return success;
}

bool ScratchPadTests::testPenaltyConstraints() {
  bool success = true;
  int numCells = 1;
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  
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
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int sideIndex=0; sideIndex<4; sideIndex++) {
      vector<int> uhat_dofIndices = _elemType->trialOrderPtr->getDofIndices(_uhat_confusion->ID(),sideIndex);
    
      for (int dofOrdinal1=0; dofOrdinal1 < uhat_dofIndices.size(); dofOrdinal1++) {
        int dofIndex1 = uhat_dofIndices[dofOrdinal1];
        expectedRHSSparsity(cellIndex,dofIndex1) = 1.0;
        for (int dofOrdinal2=0; dofOrdinal2 < uhat_dofIndices.size(); dofOrdinal2++) {
          int dofIndex2 = uhat_dofIndices[dofOrdinal2];
          expectedSparsity(cellIndex,dofIndex1,dofIndex2) = 1.0;
        }
      }
    }
  }
  
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  pc->filter(localStiffness, localRHSVector, _basisCache, _spectralConfusionMesh, bc);
  
  //  cout << "testPenaltyConstraints: expectedStiffnessSparsity:\n" << expectedSparsity;
  //  cout << "testPenaltyConstraints: localStiffness:\n" << localStiffness;
  //  
  //  cout << "testPenaltyConstraints: expectedRHSSparsity:\n" << expectedRHSSparsity;
  //  cout << "testPenaltyConstraints: localRHSVector:\n" << localRHSVector;
  
  // compare sparsity
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int i=0; i<trialDofs; i++) {
      double rhsValue = localRHSVector(cellIndex,i);
      double rhsSparsityValue = expectedRHSSparsity(cellIndex,i);
      if ((rhsSparsityValue == 0.0) && (rhsValue != 0.0)) {
        cout << "testPenaltyConstraints rhs: expected 0 but found " << rhsValue << " at i = " << i << ".\n";
        success = false;
      }
      if ((rhsSparsityValue != 0.0) && (rhsValue == 0.0)) {
        cout << "testPenaltyConstraints rhs: expected nonzero but found 0 at i = " << i << ".\n";
        success = false;
      }
      for (int j=0; j<trialDofs; j++) {
        double stiffValue = localStiffness(cellIndex,i,j);
        double sparsityValue = expectedSparsity(cellIndex,i,j);
        if ((sparsityValue == 0.0) && (stiffValue != 0.0)) {
          cout << "testPenaltyConstraints stiffness: expected 0 but found " << stiffValue << " at (" << i << ", " << j << ").\n";
          success = false;
        }
        if ((sparsityValue != 0.0) && (stiffValue == 0.0)) {
          cout << "testPenaltyConstraints stiffness: expected nonzero but found 0 at (" << i << ", " << j << ").\n";
          success = false;
        }
      }
    }
  }
  return success;
}

bool ScratchPadTests::testSpatiallyFilteredFunction() {
  bool success = true;
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  SpatialFilterPtr positiveX = Teuchos::rcp( new PositiveX );
  FunctionPtr heaviside = Teuchos::rcp( new SpatiallyFilteredFunction(one, positiveX) );
  
  int numCells = _basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = _testPoints.dimension(0);
  
  FieldContainer<double> values(numCells,numPoints);
  FieldContainer<double> expectedValues(numCells,numPoints);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints;ptIndex++) {
      double x = _basisCache->getPhysicalCubaturePoints()(cellIndex,ptIndex,0);
      if (x > 0) {
        expectedValues(cellIndex,ptIndex) = 1.0;
      } else {
        expectedValues(cellIndex,ptIndex) = 0.0;
      }
    }
  }
  
  heaviside->values(values,_basisCache);
  
  double tol = 1e-15;
  double maxDiff = 0.0;
  if ( ! fcsAgree(expectedValues, values, tol, maxDiff) ) {
    success = false;
    cout << "testSpatiallyFilteredFunction: Expected values differ from actual; maxDiff: " << maxDiff << endl;
  }
  return success;
}

// tests whether a mixed type LT
bool ScratchPadTests::testLinearTermEvaluationConsistency(){
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  // define trial variables
  VarPtr beta_n_u = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );
  
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  // define nodes for mesh
  int order = 1;
  int H1Order = order+1; int pToAdd = 1;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildUnitQuadMesh(1, convectionBF, H1Order, H1Order+pToAdd);
  
  ////////////////////   get fake residual   ///////////////////////

  LinearTermPtr lt = Teuchos::rcp(new LinearTerm);
  FunctionPtr edgeFxn = Teuchos::rcp(new EdgeFunction);
  FunctionPtr Xsq = Teuchos::rcp(new Xn(2));
  FunctionPtr Ysq = Teuchos::rcp(new Yn(2));
  FunctionPtr XYsq = Xsq*Ysq;
  lt->addTerm(edgeFxn*v + (beta*XYsq)*v->grad());

  Teuchos::RCP<RieszRep> ltRiesz = Teuchos::rcp(new RieszRep(mesh, ip, lt));
  ltRiesz->computeRieszRep();
  FunctionPtr repFxn = Teuchos::rcp(new RepFunction(v,ltRiesz));
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
  if (abs(diff)>1e-11){
    success = false;
    cout << "Failed testLinearTermEvaluationConsistency() with diff = " << diff << endl;
  }
  
  return success;
}

class IndicatorFunction : public Function {
  set<int> _cellIDs;
public:
  IndicatorFunction(set<int> cellIDs) : Function(0) {
    _cellIDs = cellIDs;
  }
  IndicatorFunction(int cellID) : Function(0) {
    _cellIDs.insert(cellID);
  }
  virtual void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    values.initialize(1.0);
    vector<int> contextCellIDs = basisCache->cellIDs();
    int cellIndex=0; // keep track of index into values
    
    int entryCount = values.size();
    int numCells = values.dimension(0);
    int numEntriesPerCell = entryCount / numCells;
    
    for (vector<int>::iterator cellIt = contextCellIDs.begin(); cellIt != contextCellIDs.end(); cellIt++) {
      int cellID = *cellIt;
      if (_cellIDs.find(cellID) == _cellIDs.end()) {
        // clear out the associated entries
        for (int j=0; j<numEntriesPerCell; j++) {
          values[cellIndex*numEntriesPerCell + j] = 0;
        }
      }
      cellIndex++;
    }
  }
};

// tests whether a mixed type LT
bool ScratchPadTests::testIntegrateDiscontinuousFunction(){
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr v = varFactory.testVar("v", HGRAD);
  
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
  ip->addTerm(v);

  // define trial variables
  VarPtr beta_n_u = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );
  
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  // define nodes for mesh
  int order = 1;
  int H1Order = order+1; int pToAdd = 1;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildUnitQuadMesh(2, 1, convectionBF, H1Order, H1Order+pToAdd);
  
  ////////////////////   integrate discontinuous function - cellIDFunction   ///////////////////////

  //  FunctionPtr cellIDFxn = Teuchos::rcp(new CellIDFunction); // should be 0 on cellID 0, 1 on cellID 1
  set<int> cellIDs;
  cellIDs.insert(1); // 0 on cell 0, 1 on cell 1
  FunctionPtr indicator = Teuchos::rcp(new IndicatorFunction(cellIDs)); // should be 0 on cellID 0, 1 on cellID 1
  double jumpWeight = 13.3; // some random number
  FunctionPtr edgeRestrictionFxn = Teuchos::rcp(new EdgeFunction);
  FunctionPtr X = Teuchos::rcp(new Xn(1));
  LinearTermPtr integrandLT = Function::constant(1.0)*v + Function::constant(jumpWeight)*X*edgeRestrictionFxn*v;
  
  // make riesz representation function to more closely emulate the error rep
  LinearTermPtr indicatorLT = Teuchos::rcp(new LinearTerm);// residual 
  indicatorLT->addTerm(indicator*v);
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ipL2, indicatorLT));
  riesz->computeRieszRep();
  map<int,FunctionPtr> vmap;
  vmap[v->ID()] = Teuchos::rcp(new RepFunction(v,riesz)); // SHOULD BE L2 projection = same thing!!!  

  FunctionPtr volumeIntegrand = integrandLT->evaluate(vmap,false); 
  FunctionPtr edgeRestrictedIntegrand = integrandLT->evaluate(vmap,true);
 
  double edgeRestrictedValue = volumeIntegrand->integrate(mesh,10) + edgeRestrictedIntegrand->integrate(mesh,10);

  double expectedValue = .5 + .5*jumpWeight;
  double diff = abs(expectedValue-edgeRestrictedValue);
  if (abs(diff)>1e-11){
    success = false;
    cout << "Failed testIntegrateDiscontinuousFunction() with expectedValue = " << expectedValue << " and actual value = " << edgeRestrictedValue << endl;
  }  
  return success;
}

struct DofInfo {
  int cellID;
  int trialID;
  int basisOrdinal;
  int basisCardinality;
  int sideIndex;
  int numSides;
  int localDofIndex; // index into trial ordering
  int totalDofs;     // number of dofs in the trial ordering
};

string dofInfoString(const DofInfo &info) {
  ostringstream dis;
  dis << "cellID = " << info.cellID << "; trialID = " << info.trialID;
  dis << "; sideIndex = " << info.sideIndex << " (" << info.numSides << " total sides)";
  dis << "; basisOrdinal = " << info.basisOrdinal << "; cardinality = " << info.basisCardinality;
  return dis.str();
}

string dofInfoString(const vector<DofInfo> infoVector) {
  ostringstream dis;
  for (vector<DofInfo>::const_iterator infoIt=infoVector.begin(); infoIt != infoVector.end(); infoIt++) {
    dis << dofInfoString(*infoIt) << endl;
  }
  return dis.str();
}

map< int, vector<DofInfo> > constructGlobalDofToLocalDofInfoMap(MeshPtr mesh) {
  // go through the mesh as a whole, and collect info for each dof
  map< int, vector<DofInfo> > infoMap;
  int numCells = mesh->numActiveElements();
  DofInfo info;
  for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
    ElementPtr cell = mesh->getActiveElement(cellIndex);
    info.cellID = cell->cellID();
    DofOrderingPtr trialOrder = cell->elementType()->trialOrderPtr;
    set<int> trialIDs = trialOrder->getVarIDs();
    info.totalDofs = trialOrder->totalDofs();
    for (set<int>::iterator trialIt=trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
      info.trialID = *trialIt;
      info.numSides = trialOrder->getNumSidesForVarID(info.trialID);
      for (int sideIndex=0; sideIndex < info.numSides; sideIndex++) {
        info.sideIndex = sideIndex;
        info.basisCardinality = trialOrder->getBasisCardinality(info.trialID, info.sideIndex);
        for (int basisOrdinal=0; basisOrdinal < info.basisCardinality; basisOrdinal++) {
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

bool ScratchPadTests::testGalerkinOrthogonality(){
  double tol = 1e-11;
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  // define trial variables
  VarPtr beta_n_u = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );

  FunctionPtr n = Function::normal();
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  // define nodes for mesh
  int order = 0;
  int H1Order = order+1; int pToAdd = 1;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildUnitQuadMesh(2,1, convectionBF, H1Order, H1Order+pToAdd);
  
  ////////////////////   SOLVE   ///////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
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
  FunctionPtr parity = Teuchos::rcp(new SideParityFunction);
  residual->addTerm(-fnhatFxn*v + (beta*uFxn)*v->grad());
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  riesz->computeRieszRep();
  map<int,FunctionPtr> err_rep_map;
  err_rep_map[v->ID()] = Teuchos::rcp(new RepFunction(v,riesz));

  ////////////////////   CHECK GALERKIN ORTHOGONALITY   ///////////////////////

  BCPtr nullBC; RHSPtr nullRHS; IPPtr nullIP;
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  map< int, vector<DofInfo> > infoMap = constructGlobalDofToLocalDofInfoMap(mesh);
  
  for (map< int, vector<DofInfo> >::iterator mapIt = infoMap.begin();
       mapIt != infoMap.end(); mapIt++) {
    int dofIndex = mapIt->first;
    vector< DofInfo > dofInfoVector = mapIt->second; // all the local dofs that map to dofIndex
    // create perturbation in direction du
    solnPerturbation->clear(); // clear all solns
    // set each corresponding local dof to 1.0
    for (vector< DofInfo >::iterator dofInfoIt = dofInfoVector.begin();
         dofInfoIt != dofInfoVector.end(); dofInfoIt++) {
      DofInfo info = *dofInfoIt;
      FieldContainer<double> solnCoeffs(info.basisCardinality);
      solnCoeffs(info.basisOrdinal) = 1.0;
      solnPerturbation->setSolnCoeffsForCellID(solnCoeffs, info.cellID, info.trialID, info.sideIndex);
    }
    //    solnPerturbation->setSolnCoeffForGlobalDofIndex(1.0,dofIndex);
      
    LinearTermPtr b_du =  convectionBF->testFunctional(solnPerturbation);
    FunctionPtr gradient = b_du->evaluate(err_rep_map, TestingUtilities::isFluxOrTraceDof(mesh,dofIndex)); // use boundary part only if flux
    double grad = gradient->integrate(mesh,10);
    if (!TestingUtilities::isFluxOrTraceDof(mesh,dofIndex) && abs(grad)>tol){ // if we're not single-precision zero FOR FIELDS
      //      int cellID = mesh->getGlobalToLocalMap()[dofIndex].first;
      cout << "Failed testGalerkinOrthogonality() for fields with diff " << abs(grad) << " at dof " << dofIndex << "; info:" << endl;
      cout << dofInfoString(infoMap[dofIndex]);
      success = false;
    }
  }
  // just test fluxes ON INTERNAL SKELETON here (by merit of the integralOfJump routine returning 0 for boundaries)
  vector<ElementPtr> elems = mesh->activeElements();
  for (vector<ElementPtr>::iterator elemIt=elems.begin();elemIt!=elems.end();elemIt++){  
    for (int sideIndex = 0;sideIndex < 4;sideIndex++){
      ElementPtr elem = *elemIt;
      ElementTypePtr elemType = elem->elementType();
      vector<int> localDofIndices = elemType->trialOrderPtr->getDofIndices(beta_n_u->ID(), sideIndex);
      for (int i = 0;i<localDofIndices.size();i++){
        int globalDofIndex = mesh->globalDofIndex(elem->cellID(), localDofIndices[i]);
        vector< DofInfo > dofInfoVector = infoMap[globalDofIndex];

	solnPerturbation->clear();
	TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,1.0,globalDofIndex);
	

        LinearTermPtr b_du =  convectionBF->testFunctional(solnPerturbation);
        FunctionPtr gradient = b_du->evaluate(err_rep_map, TestingUtilities::isFluxOrTraceDof(mesh,globalDofIndex)); // use boundary part only if flux
        double jump = gradient->integrate(mesh,10);
	//        double jump = gradient->integralOfJump(mesh,(*elemIt)->cellID(),sideIndex,10);
        //	double jump = gradient->integralOfJump(mesh,10);
        //	cout << "Jump for dof " << globalDofIndex << " is " << jump << endl;
        if (abs(jump)>tol && !mesh->boundary().boundaryElement((*elemIt)->cellID(),sideIndex)){
          cout << "Failing Galerkin orthogonality test for fluxes with diff " << jump << " at dof " << globalDofIndex << "; info:" << endl;
          cout << dofInfoString(infoMap[globalDofIndex]);
          /*
           FunctionPtr dfn = Function::solution(beta_n_u,solnPerturbation);
           FunctionPtr fluxTerm = dfn*err_rep_map[v->ID()];
           double secondJump = fluxTerm->integralOfJump(mesh,(*elemIt)->cellID(),sideIndex,10);
           cout << "second jump check = " << jump << endl;
           
           err_rep_map[v->ID()]->writeBoundaryValuesToMATLABFile(mesh,"err_rep_test.dat");
           err_rep_map[v->ID()]->writeValuesToMATLABFile(mesh,"err_rep_test.m");
           fluxTerm->writeBoundaryValuesToMATLABFile(mesh,"fn.dat");
           */
          success = false;
        }
      }
    }
  }

  return success;
}


// Testing to make sure b(du,e) = (e,v_du)_V = b(u,v_du) - l(v_du)
bool ScratchPadTests::testErrorRepConsistency(){
  double tol = 1e-11;
  bool success = true;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  // define trial variables
  VarPtr beta_n_u = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );

  FunctionPtr n = Function::normal();
  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);

  // define nodes for mesh
  int order = 2;
  int H1Order = order+1; int pToAdd = 1;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildUnitQuadMesh(2,1, convectionBF, H1Order, H1Order+pToAdd);
  
  ////////////////////   SOLVE   ///////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
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
  FunctionPtr parity = Teuchos::rcp(new SideParityFunction);
  residual->addTerm(-fnhatFxn*v + (beta*uFxn)*v->grad());
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  riesz->computeRieszRep();
  map<int,FunctionPtr> err_rep_map;
  err_rep_map[v->ID()] = Teuchos::rcp(new RepFunction(v,riesz));

  ////////////////////   CHECK CONSISTENCY   ///////////////////////

  BCPtr nullBC; RHSPtr nullRHS; IPPtr nullIP;
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  map< int, vector<DofInfo> > infoMap = constructGlobalDofToLocalDofInfoMap(mesh);
  
  for (int dofIndex=0;dofIndex<mesh->numGlobalDofs();dofIndex++){
    solnPerturbation->clear();
    TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,1.0,dofIndex);

    LinearTermPtr b_du =  convectionBF->testFunctional(solnPerturbation);

    FunctionPtr b_du_e = b_du->evaluate(err_rep_map, TestingUtilities::isFluxOrTraceDof(mesh,dofIndex)); // use boundary part only if flux
    double b_du_e_val = b_du_e->integrate(mesh,10); // b_du_e->integralOfJump(mesh,10);
    
    // create optimal test function 
    Teuchos::RCP<RieszRep> v_du_riesz = Teuchos::rcp(new RieszRep(mesh, ip, b_du)); 
    v_du_riesz->computeRieszRep();
    map<int,FunctionPtr> v_du; 
    v_du[v->ID()] = Teuchos::rcp(new RepFunction(v, v_du_riesz));

    // evaluate residual at optimal test (should be zero)
    FunctionPtr residual_edge = residual->evaluate(v_du,true); // get boundary portion
    FunctionPtr residual_vol = residual->evaluate(v_du,false); // get volume portion
    double res_at_v_du = residual_vol->integrate(mesh,10) + residual_edge->integrate(mesh,10);

    FunctionPtr e_v_du_ip = err_rep_map[v->ID()]*v_du[v->ID()] + (beta*err_rep_map[v->ID()]->grad())*(beta*v_du[v->ID()]->grad());
    double ip_val = e_v_du_ip->integrate(mesh,10);

    double diff1 = res_at_v_du - ip_val;
    double diff2 = ip_val - b_du_e_val;
    if (abs(diff1)>tol){
      cout << "Failed err rep consistency test: (e,v_du) and b(u,v_du)-l(v_du) differ with diff = " << diff1 << " for dof " << dofIndex << "; info:" << endl;
      success = false;
    }

    if (abs(diff2)>tol){
      cout << "Failed err rep consistency test: (v_du,e)_V and b(du,e) differ with diff  = " << diff2 << " for dof " << dofIndex << "; info:" << endl;      
      success = false;
    }
    /*
    if (abs(diff1)>tol || abs(diff2)>tol){
      //      cout << dofInfoString(infoMap[dofIndex]);
    }
    int cellID = infoMap[dofIndex][0].cellID;
    int sideIndex = infoMap[dofIndex][0].sideIndex;
    //    if (abs(res_at_v_du)>tol && abs(ip_val)>tol && abs(b_du_e_val)>tol && !mesh->boundary().boundaryElement(cellID,sideIndex)){
    if ( abs(b_du_e_val)>tol && !mesh->boundary().boundaryElement(cellID,sideIndex)){
      cout << "Not Galerkin-orthogonal: for dof " << dofIndex << ", ip val = " << ip_val << ", residual val = " << res_at_v_du << ", b(du,e) = " << b_du_e_val << ", " << dofInfoString(infoMap[dofIndex]) << endl;
    }
    */

  }

  return success;
}


std::string ScratchPadTests::testSuiteName() {
  return "ScratchPadTests";
}
