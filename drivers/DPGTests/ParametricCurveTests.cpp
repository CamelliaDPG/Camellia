//
//  ParametricCurveTests.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/16/13.
//
//

#include "ParametricCurveTests.h"

void ParametricCurveTests::setup() {
  
}

void ParametricCurveTests::runTests(int &numTestsRun, int &numTestsPassed) {
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

std::string ParametricCurveTests::testSuiteName() {
  return "ParametricCurveTests";
}
