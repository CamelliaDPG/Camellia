
#include "PatchBasisTests.h"

#include "Intrepid_FieldContainer.hpp"

#include "BasisFactory.h"

#include "BilinearForm.h" // defines IntrepidExtendedTypes

void PatchBasisTests::setup() {
  // for tests, we'll do a simple division of a line segment into thirds
  // (for now, PatchBasis only supports 1D bases--sufficient for 2D DPG meshes)
  // setup bases:
  int polyOrder = 3;
  _parentBasis = BasisFactory::getBasis( polyOrder, shards::Line<2>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD );
  FieldContainer<double> nodesLeft(2,1), nodesMiddle(2,1), nodesRight(2,1);
  nodesLeft(0,0)   = -1.0;
  nodesLeft(1,0)   = -1.0 / 3.0;
  nodesMiddle(0,0) = -1.0 / 3.0;
  nodesMiddle(1,0) = 1.0 / 3.0;
  nodesRight(0,0)  = 1.0 / 3.0;
  nodesRight(1,0)  = 1.0;
  _patchBasisLeft   = BasisFactory::getPatchBasis(_parentBasis,nodesLeft);
  _patchBasisMiddle = BasisFactory::getPatchBasis(_parentBasis,nodesMiddle);
  _patchBasisRight  = BasisFactory::getPatchBasis(_parentBasis,nodesRight);
  
  double refCellLeft = -1.0;
  double refCellRight = 1.0;
  
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  
  _testPoints1D = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    _testPoints1D(i, 0) = x[i];
  }
  
  _testPoints1DLeftParent   = FieldContainer<double>(NUM_POINTS_1D,1);
  _testPoints1DMiddleParent = FieldContainer<double>(NUM_POINTS_1D,1);
  _testPoints1DRightParent  = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    double offset = (x[i] - refCellLeft) / 3.0;
    _testPoints1DLeftParent(i,0)   = -1.0       + offset;
    _testPoints1DMiddleParent(i,0) = -1.0 / 3.0 + offset;
    _testPoints1DRightParent(i,0)  =  1.0 / 3.0 + offset;
  }  
}

void PatchBasisTests::teardown() {
  _testPoints1D.resize(0);
  _testPoints1DLeftParent.resize(0);
  _testPoints1DMiddleParent.resize(0);
  _testPoints1DRightParent.resize(0);
}

void PatchBasisTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testPatchBasis1D()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
}

bool PatchBasisTests::testPatchBasis1D() {
  bool success = true;
  
  double tol = 1e-15;
  int numPoints = _testPoints1D.size();
  int numFields = _parentBasis->getCardinality();
  FieldContainer<double> valuesLeft(numFields,numPoints),   expectedValuesLeft(numFields,numPoints);
  FieldContainer<double> valuesMiddle(numFields,numPoints), expectedValuesMiddle(numFields,numPoints);
  FieldContainer<double> valuesRight(numFields,numPoints),  expectedValuesRight(numFields,numPoints);
  
  // get the expected values
  _parentBasis->getValues(expectedValuesLeft,   _testPoints1DLeftParent,   Intrepid::OPERATOR_VALUE);
  _parentBasis->getValues(expectedValuesMiddle, _testPoints1DMiddleParent, Intrepid::OPERATOR_VALUE);
  _parentBasis->getValues(expectedValuesRight,  _testPoints1DRightParent,  Intrepid::OPERATOR_VALUE);
  
  // get the actual values:
  _patchBasisLeft  ->getValues(valuesLeft,   _testPoints1D, Intrepid::OPERATOR_VALUE);
  _patchBasisMiddle->getValues(valuesMiddle, _testPoints1D, Intrepid::OPERATOR_VALUE);
  _patchBasisRight ->getValues(valuesRight,  _testPoints1D, Intrepid::OPERATOR_VALUE);
  
  for (int fieldIndex=0; fieldIndex < numFields; fieldIndex++) {
    for (int pointIndex=0; pointIndex < numPoints; pointIndex++) {
      double diff = abs(valuesLeft(fieldIndex,pointIndex) - expectedValuesLeft(fieldIndex,pointIndex));
      if (diff > tol) {
        success = false;
        cout << "expected value of left basis: " << expectedValuesLeft(fieldIndex,pointIndex) << "; actual: " << valuesLeft(fieldIndex,pointIndex) << endl;
      }
      
      diff = abs(valuesMiddle(fieldIndex,pointIndex) - expectedValuesMiddle(fieldIndex,pointIndex));
      if (diff > tol) {
        success = false;
        cout << "expected value of middle basis: " << expectedValuesMiddle(fieldIndex,pointIndex) << "; actual: " << valuesMiddle(fieldIndex,pointIndex) << endl;
      }
      
      diff = abs(valuesRight(fieldIndex,pointIndex) - expectedValuesRight(fieldIndex,pointIndex));
      if (diff > tol) {
        success = false;
        cout << "expected value of right basis: " << expectedValuesRight(fieldIndex,pointIndex) << "; actual: " << valuesRight(fieldIndex,pointIndex) << endl;
      }
    }
  }
  
  return success;
}