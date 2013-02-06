//
//  ParametricCurveTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/16/13.
//
//

#include "ParametricCurveTests.h"

#include "BasisSumFunction.h"

static const double PI  = 3.141592653589793238462;

void ParametricCurveTests::setup() {
  
}

void ParametricCurveTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testCircularArc()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
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

bool ParametricCurveTests::testCircularArc() {
  bool success = true;
  
  // the arc details are copied from CurvilinearMeshTests -- motivation is to diagnose test failure there with a more granular test here
  double radius = 1.0;
  double meshWidth = sqrt(2);
  
  ParametricCurvePtr circle = ParametricCurve::circle(radius, meshWidth / 2.0, meshWidth / 2.0);
  ParametricCurvePtr circularArc = ParametricCurve::subCurve(circle,  5.0/8.0, 7.0/8.0);
  
  BasisCachePtr basisCache = BasisCache::parametric1DCache(15); // overintegrate to be safe
  
  FunctionPtr cos_part = Teuchos::rcp( new Cos_ax(PI/2, 1.25*PI));
  FunctionPtr sin_part = Teuchos::rcp( new Sin_ax(PI/2, 1.25*PI));
  FunctionPtr x_t = meshWidth / 2 + cos_part;
  FunctionPtr y_t = meshWidth / 2 + sin_part;
  
  FunctionPtr dx_dt = (- PI / 2) * sin_part;
  FunctionPtr dy_dt = (PI / 2) * cos_part;
  
  // check that we have the right idea for all those functions:
  if (! x_t->equals(circularArc->x(),basisCache)) {
    cout << "expected x(0) = " << Function::evaluate(x_t,0);
    cout << "; actual = " << Function::evaluate(circularArc->x(),0) << endl;
    cout << "x part of circularArc doesn't match expected.\n";
    success = false;
  }
  if (! y_t->equals(circularArc->y(),basisCache)) {
    cout << "expected y(0) = " << Function::evaluate(y_t,0);
    cout << "; actual = " << Function::evaluate(circularArc->y(),0) << endl;
    cout << "y part of circularArc doesn't match expected.\n";
    success = false;
  }
  
  if (! dx_dt->equals(circularArc->dt()->x(),basisCache)) {
    cout << "dx/dt of circularArc doesn't match expected.\n";
    success = false;
  }
  if (! dy_dt->equals(circularArc->dt()->y(),basisCache)) {
    cout << "dy/dt of circularArc doesn't match expected.\n";
    success = false;
  }
  
  // test exact curve at t=0.5
  
  double tol=1e-14;
  double t = 0.5;
  double x_expected = meshWidth / 2;
  double y_expected = meshWidth / 2 - radius;
  
  double x,y,xErr,yErr;
  // check value
  circularArc->value(t, x, y);
  xErr = abs(x-x_expected);
  yErr = abs(y-y_expected);
  if (xErr > tol) {
    cout << "exact arc x at t=0.5 is incorrect.\n";
    success = false;
  }
  if (yErr > tol) {
    cout << "exact arc y at t=0.5 is incorrect.\n";
    success = false;
  }
  
  // check derivatives
  // figuring out what the x derivative should be is a bit of work, I think,
  // but the y value is at a minimum, so its derivative should be zero
  y_expected = 0;
  circularArc->dt()->value(t, x, y);
  yErr = abs(y-y_expected);
  if (yErr > tol) {
    cout << "exact arc dy/dt at t=0.5 is nonzero.\n";
    success = false;
  }
  
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  BasisPtr quadraticBasis = BasisFactory::getBasis(2, line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  // figure out what the weights for the quadratic "middle node" basis function should be:
  double expected_H1_weight_x, expected_H1_weight_y;
  double expected_L2_weight_x, expected_L2_weight_y;
  
  FunctionPtr middleBasis;
  {
    FunctionPtr t = Teuchos::rcp( new Xn(1) );
    middleBasis = 4 * t * (1-t);
  }
  
  double middleBasisL2_squared = (middleBasis*middleBasis)->integrate(basisCache);
  double middleBasisH1_squared = ( middleBasis->dx() * middleBasis->dx() )->integrate(basisCache) + middleBasisL2_squared;
  
  ParametricCurvePtr circularArcBubble = ParametricCurve::bubble(circularArc);
  
  FunctionPtr bubble_x = circularArcBubble->x();
  FunctionPtr bubble_y = circularArcBubble->y();
  
  double x_against_middle_L2 = (bubble_x * middleBasis)->integrate(basisCache);
  double x_against_middle_H1 = (bubble_x->dx() * middleBasis->dx())->integrate(basisCache) + x_against_middle_L2;
  
  double y_against_middle_L2 = (bubble_y * middleBasis)->integrate(basisCache);
  double y_against_middle_H1 = (bubble_y->dx() * middleBasis->dx())->integrate(basisCache) + y_against_middle_L2;
  
  expected_L2_weight_x = x_against_middle_L2 / middleBasisL2_squared;
  expected_H1_weight_x = x_against_middle_H1 / middleBasisH1_squared;
  
  expected_L2_weight_y = y_against_middle_L2 / middleBasisL2_squared;
  expected_H1_weight_y = y_against_middle_H1 / middleBasisH1_squared;
  
  int middleBasisOrdinal = quadraticBasis->getDofOrdinal(1,0,0);
  
  FieldContainer<double> basisCoefficients_x, basisCoefficients_y;
  bool useH1 = false; // just trying to diagnose whether the issue is in derivatives or values (most likely derivatives)
  circularArcBubble->projectionBasedInterpolant(basisCoefficients_x, quadraticBasis, 0, useH1);
  circularArcBubble->projectionBasedInterpolant(basisCoefficients_y, quadraticBasis, 1, useH1);
  
  double weightError_x = abs(expected_L2_weight_x-basisCoefficients_x[middleBasisOrdinal]);
  double weightError_y = abs(expected_L2_weight_y-basisCoefficients_y[middleBasisOrdinal]);
  
  if (weightError_x > tol) {
    success = false;
    cout << "testCircularArc(): L2 projection doesn't match expected basis weight in x.\n";
    cout << "expected " << expected_L2_weight_x << ", was " << basisCoefficients_x[middleBasisOrdinal] << endl;
  }
  if (weightError_y > tol) {
    success = false;
    cout << "testCircularArc(): L2 projection doesn't match expected basis weight in y.\n";
    cout << "expected " << expected_L2_weight_y << ", was " << basisCoefficients_y[middleBasisOrdinal] << endl;
  }

  useH1 = true;
  circularArcBubble->projectionBasedInterpolant(basisCoefficients_x, quadraticBasis, 0, useH1);
  circularArcBubble->projectionBasedInterpolant(basisCoefficients_y, quadraticBasis, 1, useH1);

  weightError_x = abs(expected_H1_weight_x-basisCoefficients_x[middleBasisOrdinal]);
  weightError_y = abs(expected_H1_weight_y-basisCoefficients_y[middleBasisOrdinal]);

  if (weightError_x > tol) {
    success = false;
    cout << "testCircularArc(): H1 projection doesn't match expected basis weight in x.\n";
    cout << "expected " << expected_H1_weight_x << ", was " << basisCoefficients_x[middleBasisOrdinal] << endl;
  }
  if (weightError_y > tol) {
    success = false;
    cout << "testCircularArc(): H1 projection doesn't match expected basis weight in y.\n";
    cout << "expected " << expected_H1_weight_y << ", was " << basisCoefficients_y[middleBasisOrdinal] << endl;
  }
  /*
  FunctionPtr projection_x = NewBasisSumFunction::basisSumFunction(quadraticBasis, basisCoefficients_x);
  FunctionPtr projection_y = NewBasisSumFunction::basisSumFunction(quadraticBasis, basisCoefficients_y);
  
  FieldContainer<double> parametricPoint(1,1);
  parametricPoint[0] = t;
  FieldContainer<double> refPoint = basisCache->getRefCellPointsForPhysicalPoints(parametricPoint);
  basisCache->setRefCellPoints(refPoint);
  FieldContainer<double> value(1,1);
  projection_x->values(value, basisCache);
  x = value[0];
  projection_y->values(value, basisCache);
  y = value[0];
  
  // same expectations at the beginning, except of course now we don't expect to nail it.
  // but we do expect to be closer than the linear interpolation of the vertices
  x_expected = meshWidth / 2;
  y_expected = meshWidth / 2 - radius;

  double linearErr_x = 0; // linear interpolant nails the x value
  double linearErr_y = abs(y_expected);
  
  xErr = abs(x-x_expected);
  yErr = abs(y-y_expected);
  
  if (xErr > linearErr_x + tol) {
    cout << "quadratic projection-based interpolant has greater error in x than linear interpolant.\n";
    success = false;
  }
  if (yErr > linearErr_y + tol) {
    cout << "quadratic projection-based interpolant has greater error in y than linear interpolant.\n";
    success = false;
  }*/
  
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
  double tol = 1e-13;
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
  
  /////////////////// TEST UNRECOVERABLE CURVE INTERPOLATED //////////////////////
  
  // finally, project the cubic curve onto a quadratic basis, and check that it interpolates the endpoints
  BasisPtr quadraticBasis = BasisFactory::getBasis(2, line_2.getKey(), IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD);
  
  myCurve->projectionBasedInterpolant(basisCoefficients_x, quadraticBasis, 0);
  myCurve->projectionBasedInterpolant(basisCoefficients_y, quadraticBasis, 1);
  
  if ( ! basisSumInterpolatesCurveEndPoints(basisCoefficients_x,basisCoefficients_y, quadraticBasis, myCurve)) {
    cout << "testProjectionBasedInterpolation() failed: quadratic projection-based interpolant doesn't interpolate cubic curve endpoints.\n";
    cout << "basisCoefficients_x:\n" << basisCoefficients_x;
    cout << "basisCoefficients_y:\n" << basisCoefficients_y;
    success = false;
  }
  
  return success;
}

std::string ParametricCurveTests::testSuiteName() {
  return "ParametricCurveTests";
}
