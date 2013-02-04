//
//  ParametricCurveTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/16/13.
//
//

#include "ParametricCurveTests.h"

#include "BasisSumFunction.h"

void ParametricCurveTests::setup() {
  
}

void ParametricCurveTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testLine()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testBubble()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testProjectionBasedInterpolation()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testPolygon()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testParametricCurveRefinement()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}


bool ParametricCurveTests::testBubble() {
  // tests the kind of thing that will happen to parametric functions during mesh refinement
  bool success = true;
  
  double x0=3, y0=-3, x1=5, y1=4;
  ParametricCurvePtr myLine = ParametricCurve::line(x0, y0, x1, y1);

  ParametricCurvePtr myBubble = ParametricCurve::bubble(myLine); // should be just 0.
  
  myBubble->value(0, x0,y0);
  myBubble->value(1, x1,y1);
  
  double tol = 1e-14;
  if ((abs(x0)>tol) || (abs(y0)>tol)) {
    cout << "Bubble is not 0 at t=0.\n";
    success = false;
  }
  if ((abs(x1)>tol) || (abs(y1)>tol)) {
    cout << "Bubble is not 0 at t=1.\n";
    success = false;
  }
  
  double x,y; // for points in the middle of parameter space
  double t=0.5;
  myBubble->value(t,x,y);
  
  if ((abs(x)>tol) || (abs(y)>tol)) {
    cout << "Bubble for line is not 0 at t=0.5.\n";
    success = false;
  }
  
  return success;
}

bool ParametricCurveTests::testLine() {
  bool success = true;
  
  ParametricCurvePtr unitLine = ParametricCurve::line(0, 0, 1, 0);
  ParametricCurvePtr unitLine_dt = unitLine->dt();
  FunctionPtr unitLine_x = unitLine->x(); // x=t
  FunctionPtr unitLine_y = unitLine->y(); // 0
  
  vector<double> t_values;
  t_values.push_back(0);
  t_values.push_back(0.5);
  t_values.push_back(0.75);
  t_values.push_back(1.0);
  
  double tol=1e-14;
  for (int i=0; i<t_values.size(); i++) {
    double t = t_values[i];
    double x = Function::evaluate(unitLine_x, t);
    double expected = t;
    if (abs(x-expected)>tol) {
      cout << "testLine(): unitLine_x differs from expected value.\n";
    }
  }
  for (int i=0; i<t_values.size(); i++) {
    double t = t_values[i];
    double x = Function::evaluate(unitLine_y, t);
    double expected = 0;
    if (abs(x-expected)>tol) {
      cout << "testLine(): unitLine_y differs from expected value.\n";
    }
  }

  return success;
}

bool ParametricCurveTests::testParametricCurveRefinement() {
  // tests the kind of thing that will happen to parametric functions during mesh refinement
  bool success = true;
  
  ParametricCurvePtr unitLine = ParametricCurve::line(0, 0, 1, 0);
  
  ParametricCurvePtr halfLine = ParametricCurve::subCurve(unitLine, 0, 0.5);
  
  double expected_x, expected_y, t, actual_x, actual_y;
  
  // t = 0
  t = 0;
  expected_x = 0; expected_y = 0;
  halfLine->value(t, actual_x, actual_y);
  
  if (actual_x != expected_x) {
    success = false;
    cout << "expected " << expected_x << " but actual x is " << actual_x << endl;
  }
  if (actual_y != expected_y) {
    success = false;
    cout << "expected " << expected_y << " but actual y is " << actual_y << endl;
  }
  
  // t = 1
  t = 1;
  expected_x = 0.5; expected_y = 0;
  halfLine->value(t, actual_x, actual_y);
  
  if (actual_x != expected_x) {
    success = false;
    cout << "expected " << expected_x << " but actual x is " << actual_x << endl;
  }
  if (actual_y != expected_y) {
    success = false;
    cout << "expected " << expected_y << " but actual y is " << actual_y << endl;
  }
  
  // again, but with 1/3 line
  ParametricCurvePtr thirdLine = ParametricCurve::subCurve(unitLine, 1.0/3.0, 2.0/3.0);
  // t = 0
  t = 0;
  expected_x = 1.0/3.0; expected_y = 0;
  thirdLine->value(t, actual_x, actual_y);
  
  if (actual_x != expected_x) {
    success = false;
    cout << "expected " << expected_x << " but actual x is " << actual_x << endl;
  }
  if (actual_y != expected_y) {
    success = false;
    cout << "expected " << expected_y << " but actual y is " << actual_y << endl;
  }
  
  // t = 1
  t = 1;
  expected_x = 2.0/3.0; expected_y = 0;
  thirdLine->value(t, actual_x, actual_y);
  
  if (actual_x != expected_x) {
    success = false;
    cout << "expected " << expected_x << " but actual x is " << actual_x << endl;
  }
  if (actual_y != expected_y) {
    success = false;
    cout << "expected " << expected_y << " but actual y is " << actual_y << endl;
  }
  
  return allSuccess(success);
}

bool ParametricCurveTests::testPolygon() {
  bool success = true;
  double width = 2.0; double height = 1.0;
  vector< pair<double, double> > rectVertices;
  rectVertices.push_back(make_pair(0,0));
  rectVertices.push_back(make_pair(width, 0));
  rectVertices.push_back(make_pair(width, height));
  rectVertices.push_back(make_pair(0, height));
  ParametricCurvePtr rect = ParametricCurve::polygon(rectVertices);
  
  double perimeter = 2 * (width + height);
  vector<double> t_values;
  vector<double> x_values_expected;
  vector<double> y_values_expected;
  
  // first vertex:
  t_values.push_back(0);
  x_values_expected.push_back(rectVertices[0].first);
  y_values_expected.push_back(rectVertices[0].second);
  
  // second vertex:
  t_values.push_back(width/perimeter);
  x_values_expected.push_back(rectVertices[1].first);
  y_values_expected.push_back(rectVertices[1].second);
  
  // third vertex:
  t_values.push_back((width+height)/perimeter);
  x_values_expected.push_back(rectVertices[2].first);
  y_values_expected.push_back(rectVertices[2].second);
  
  // fourth vertex:
  t_values.push_back((2*width+height)/perimeter);
  x_values_expected.push_back(rectVertices[3].first);
  y_values_expected.push_back(rectVertices[3].second);
  
  // and now a handful of in-between values:
  t_values.push_back((width/perimeter)/2);
  x_values_expected.push_back((rectVertices[1].first -rectVertices[0].first)  / 2);
  y_values_expected.push_back((rectVertices[1].second-rectVertices[0].second) / 2);
  
  double tol = 1e-14;
  for (int i=0; i<t_values.size(); i++) {
    double t = t_values[i];
    double x,y;
    rect->value(t, x, y);
    double x_err = abs(x-x_values_expected[i]);
    double y_err = abs(y-y_values_expected[i]);
    if (x_err > tol) {
      cout << "rect(" << t << "): expected x = " << x_values_expected[i] << " but was " << x << endl;
      success = false;
    }
    if (y_err > tol) {
      cout << "rect(" << t << "): expected y = " << y_values_expected[i] << " but was " << y << endl;
      success = false;
    }
  }
  return success;
}

double basisSumAtParametricPoint(FieldContainer<double> &basisCoefficients, BasisPtr basis, double tValue, BasisCachePtr parametricBasisCache) {
  int numPoints = 1;
  int spaceDim = 1;
  FieldContainer<double> parametricPoints(numPoints,spaceDim);
  parametricPoints[0] = tValue;
  FieldContainer<double> refCellPoints = parametricBasisCache->getRefCellPointsForPhysicalPoints(parametricPoints);
  parametricBasisCache->setRefCellPoints(refCellPoints);
  Teuchos::RCP< const FieldContainer<double> > basisValues = parametricBasisCache->getValues(basis, OP_VALUE);
  double sum = 0;
  int ptIndex = 0; // one point
  for (int fieldIndex=0; fieldIndex<basisCoefficients.size(); fieldIndex++) {
    sum += (*basisValues)(fieldIndex,ptIndex) * basisCoefficients(fieldIndex);
  }
  return sum;
}

bool basisSumInterpolatesCurveEndPoints(FieldContainer<double> &basisCoefficients_x, FieldContainer<double> &basisCoefficients_y,
                                        BasisPtr basis, ParametricCurvePtr curve) {
  double curve_x0, curve_y0, curve_x1, curve_y1;
  curve->value(0, curve_x0, curve_y0);
  curve->value(1, curve_x1, curve_y1);
  BasisCachePtr basisCache = BasisCache::basisCache1D(0, 1, basis->getDegree()*2);
  double sum_x0 = basisSumAtParametricPoint(basisCoefficients_x, basis, 0, basisCache);
  double sum_x1 = basisSumAtParametricPoint(basisCoefficients_x, basis, 1, basisCache);
  double sum_y0 = basisSumAtParametricPoint(basisCoefficients_y, basis, 0, basisCache);
  double sum_y1 = basisSumAtParametricPoint(basisCoefficients_y, basis, 1, basisCache);
  double tol = 1e-14;
  double x0_err = abs(sum_x0-curve_x0);
  double y0_err = abs(sum_y0-curve_y0);
  double x1_err = abs(sum_x1-curve_x1);
  double y1_err = abs(sum_y1-curve_y1);
  double sum_err = x0_err + y0_err + x1_err + y1_err;
  return sum_err < tol;
}

bool basisSumEqualsFunction(FieldContainer<double> &basisCoefficients, BasisPtr basis, FunctionPtr f) {
  // tests on [0,1]
  FunctionPtr sumFunction = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients) );
  
  int cubatureDegree = basis->getDegree() * 2;
  BasisCachePtr basisCache = BasisCache::basisCache1D(0, 1, cubatureDegree);

  return sumFunction->equals(f, basisCache);
}

bool ParametricCurveTests::testProjectionBasedInterpolation() {
  bool success = true;
  // to start with, project a line onto a linear basis
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  
  /////////////////// TEST LINEAR CURVES RECOVERED //////////////////////
  
  BasisPtr linearBasis = BasisFactory::getBasis(1, line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  double x0=3, y0=-3, x1=5, y1=4;
//  double dist = sqrt((x1-x0)*(x1-x0) + (y1-y0)*(y1-y0));
  double dist = 1; // the length of the parametric space
  BasisCachePtr basisCache = BasisCache::basisCache1D(0, dist, linearBasis->getDegree()*2);
  ParametricCurvePtr myLine = ParametricCurve::line(x0, y0, x1, y1);
  
  FieldContainer<double> basisCoefficients_x, basisCoefficients_y;
  myLine->projectionBasedInterpolant(basisCoefficients_x, linearBasis, 0);
  myLine->projectionBasedInterpolant(basisCoefficients_y, linearBasis, 1);
  
  if ( ! basisSumInterpolatesCurveEndPoints(basisCoefficients_x,basisCoefficients_y, linearBasis, myLine)) {
    cout << "testProjectionBasedInterpolation() failed: projection-based interpolant doesn't interpolate line endpoints.\n";
    cout << "basisCoefficients_x:\n" << basisCoefficients_x;
    cout << "basisCoefficients_y:\n" << basisCoefficients_y;
    success = false;
  }
  
  // in fact, we should recover the line in x and y:
  if ( !basisSumEqualsFunction(basisCoefficients_x, linearBasis, myLine->x()) ) {
    cout << "testProjectionBasedInterpolation() failed: projection-based interpolant doesn't recover the line in the x component.\n";
    success = false;
  }
  if ( !basisSumEqualsFunction(basisCoefficients_y, linearBasis, myLine->y()) ) {
    cout << "testProjectionBasedInterpolation() failed: projection-based interpolant doesn't recover the line in the y component.\n";
    success = false;
  }
  
  /////////////////// TEST CUBIC CURVES RECOVERED //////////////////////
  FunctionPtr t = Teuchos::rcp( new Xn(1) );
  // define x and y as functions of t:
  FunctionPtr x_t = t*t*t-2*t;
  FunctionPtr y_t = t*t*t+8*t*t;
  
  ParametricCurvePtr myCurve = ParametricCurve::curve(x_t,y_t);
  
  BasisPtr cubicBasis = BasisFactory::getBasis(3, line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  myCurve->projectionBasedInterpolant(basisCoefficients_x, cubicBasis, 0);
  myCurve->projectionBasedInterpolant(basisCoefficients_y, cubicBasis, 1);
  
  // we should again recover the curve exactly:
  if ( !basisSumEqualsFunction(basisCoefficients_x, cubicBasis, myCurve->x()) ) {
    cout << "testProjectionBasedInterpolation() failed: projection-based interpolant doesn't recover the cubic curve in the x component.\n";
    success = false;
  }
  if ( !basisSumEqualsFunction(basisCoefficients_y, cubicBasis, myCurve->y()) ) {
    cout << "testProjectionBasedInterpolation() failed: projection-based interpolant doesn't recover the cubic curve in the y component.\n";
    success = false;
  }
  
  return success;
}

std::string ParametricCurveTests::testSuiteName() {
  return "ParametricCurveTests";
}
