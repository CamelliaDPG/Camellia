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


std::string ScratchPadTests::testSuiteName() {
  return "ScratchPadTests";
}
