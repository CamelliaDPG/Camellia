//
//  NavierStokesCavityFlowDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "choice.hpp"
#include "mpi_choice.hpp"

#include "HConvergenceStudy.h"

#include "InnerProductScratchPad.h"

#include "PreviousSolutionFunction.h"

#include "LagrangeConstraints.h"

#include "BasisFactory.h"

#include "ParameterFunction.h"

#include "RefinementHistory.h"
#include "MeshFactory.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "NavierStokesFormulation.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "LidDrivenFlowRefinementStrategy.h"
#include "RefinementPattern.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "MeshPolyOrderFunction.h"
#include "MeshTestUtility.h"
#include "NonlinearSolveStrategy.h"
#include "PenaltyConstraints.h"

#include "MassFluxFunction.h"

#include "GnuPlotUtil.h"

#include "MeshUtilities.h"

#include <Teuchos_GlobalMPISession.hpp>

using namespace std;

// static double REYN = 100;
static double Re = 400; // matches John Evans's dissertation, p. 183

VarFactory varFactory; 
// test variables:
VarPtr tau1, tau2, v1, v2, q;
// traces and fluxes:
VarPtr u1hat, u2hat, t1n, t2n;
// field variables:
VarPtr u1, u2, sigma11, sigma12, sigma21, sigma22, p;


class U1_0 : public SimpleFunction {
  double _eps;
public:
  U1_0(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    double tol = 1e-14;
    if (abs(y-1.0) < tol) { // top boundary
      // negated all these to agree with Botella & Peyret (main velocity was 1, is now -1)
      if ( (abs(x) < _eps) ) { // top left
        return -x / _eps;
      } else if ( abs(1.0-x) < _eps) { // top right
        return -(1.0-x) / _eps;
      } else { // top middle
        return -1;
      }
    } else { // not top boundary: 0.0
      return 0.0;
    }
  }
};

class U2_0 : public SimpleFunction {
public:
  double value(double x, double y) {
    return 0.0;
  }
};

class Un_0 : public ScalarFunctionOfNormal {
  SimpleFunctionPtr _u1, _u2;
public:
  Un_0(double eps) {
    _u1 = Teuchos::rcp(new U1_0(eps));
    _u2 = Teuchos::rcp(new U2_0);
  }
  double value(double x, double y, double n1, double n2) {
    double u1 = _u1->value(x,y);
    double u2 = _u2->value(x,y);
    return u1 * n1 + u2 * n2;
  }
};

class U0_cross_n : public ScalarFunctionOfNormal {
  SimpleFunctionPtr _u1, _u2;
public:
  U0_cross_n(double eps) {
    _u1 = Teuchos::rcp(new U1_0(eps));
    _u2 = Teuchos::rcp(new U2_0);
  }
  double value(double x, double y, double n1, double n2) {
    double u1 = _u1->value(x,y);
    double u2 = _u2->value(x,y);
    return u1 * n2 - u2 * n1;
  }
};

class SqrtFunction : public Function {
  FunctionPtr _f;
public:
  SqrtFunction(FunctionPtr f) : Function(0) {
    _f = f;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    _f->values(values,basisCache);
    
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double value = values(cellIndex,ptIndex);
        values(cellIndex,ptIndex) = sqrt(value);
      }
    }
  }
};

FieldContainer<double> pointGrid(double xMin, double xMax, double yMin, double yMax, int numPoints) {
  vector<double> points1D_x, points1D_y;
  for (int i=0; i<numPoints; i++) {
    points1D_x.push_back( xMin + (xMax - xMin) * ((double) i) / (numPoints-1) );
    points1D_y.push_back( yMin + (yMax - yMin) * ((double) i) / (numPoints-1) );
  }
  int spaceDim = 2;
  FieldContainer<double> points(numPoints*numPoints,spaceDim);
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      points(pointIndex,0) = points1D_x[i];
      points(pointIndex,1) = points1D_y[j];
    }
  }
  return points;
}

FieldContainer<double> solutionData(FieldContainer<double> &points, SolutionPtr solution, VarPtr u1) {
  int numPoints = points.dimension(0);
  FieldContainer<double> values(numPoints);
  solution->solutionValues(values, u1->ID(), points);
  
  FieldContainer<double> xyzData(numPoints, 3);
  for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
    xyzData(ptIndex,0) = points(ptIndex,0);
    xyzData(ptIndex,1) = points(ptIndex,1);
    xyzData(ptIndex,2) = values(ptIndex);
  }
  return xyzData;
}

set<double> diagonalContourLevels(FieldContainer<double> &pointData, int pointsPerLevel=1) {
  // traverse diagonal of (i*numPoints + j) data from solutionData()
  int numPoints = sqrt(pointData.dimension(0));
  set<double> levels;
  for (int i=0; i<numPoints; i++) {
    levels.insert(pointData(i*numPoints + i,2)); // format for pointData has values at (ptIndex, 2)
  }
  // traverse the counter-diagonal
  for (int i=0; i<numPoints; i++) {
    levels.insert(pointData(i*numPoints + numPoints-1-i,2)); // format for pointData has values at (ptIndex, 2)
  }
  set<double> filteredLevels;
  int i=0;
  pointsPerLevel *= 2;
  for (set<double>::iterator levelIt = levels.begin(); levelIt != levels.end(); levelIt++) {
    if (i%pointsPerLevel==0) {
      filteredLevels.insert(*levelIt);
    }
    i++;
  }
  return filteredLevels;
}

vector<double> horizontalCenterLinePoints() {
  // points where values are often reported in the literature
  vector<double> xPoints;
  xPoints.push_back(0.0000);
  xPoints.push_back(0.0312);
  xPoints.push_back(0.0391);
  xPoints.push_back(0.0469);
  xPoints.push_back(0.0547);
  xPoints.push_back(0.0937);
  xPoints.push_back(0.1406);
  xPoints.push_back(0.1953);
  xPoints.push_back(0.5000);
  xPoints.push_back(0.7656);
  xPoints.push_back(0.7734);
  xPoints.push_back(0.8437);
  xPoints.push_back(0.9062);
  xPoints.push_back(0.9219);
  xPoints.push_back(0.9297);
  xPoints.push_back(0.9375);
  xPoints.push_back(1.0000);
  return xPoints;
}

vector<double> verticalCenterLinePoints() {
  vector<double> yPoints;
  yPoints.push_back(1.0000);
  yPoints.push_back(0.9766);
  yPoints.push_back(0.9688);
  yPoints.push_back(0.9609);
  yPoints.push_back(0.9531);
  yPoints.push_back(0.8516);
  yPoints.push_back(0.7344);
  yPoints.push_back(0.6172);
  yPoints.push_back(0.5000);
  yPoints.push_back(0.4531);
  yPoints.push_back(0.2813);
  yPoints.push_back(0.1719);
  yPoints.push_back(0.1016);
  yPoints.push_back(0.0703);
  yPoints.push_back(0.0625);
  yPoints.push_back(0.0547);
  yPoints.push_back(0.0000);
  return yPoints;
}

void writePatchValues(double xMin, double xMax, double yMin, double yMax,
                      SolutionPtr solution, VarPtr u1, string filename,
                      int numPoints=100) {
  FieldContainer<double> points = pointGrid(xMin,xMax,yMin,yMax,numPoints);
  FieldContainer<double> values(numPoints*numPoints);
  solution->solutionValues(values, u1->ID(), points);
  ofstream fout(filename.c_str());
  fout << setprecision(15);
  
  fout << "X = zeros(" << numPoints << ",1);\n";
  //    fout << "Y = zeros(numPoints);\n";
  fout << "U = zeros(" << numPoints << "," << numPoints << ");\n";
  for (int i=0; i<numPoints; i++) {
    fout << "X(" << i+1 << ")=" << points(i,0) << ";\n";
  }
  for (int i=0; i<numPoints; i++) {
    fout << "Y(" << i+1 << ")=" << points(i,1) << ";\n";
  }
  
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      fout << "U("<<i+1<<","<<j+1<<")=" << values(pointIndex) << ";" << endl;
    }
  }
  fout.close();
}

enum XYVarying {
  X_VARIES,
  Y_VARIES
};

// Newton's method
bool findZeroOnLine(FunctionPtr f, FunctionPtr f_prime, XYVarying variableChoice, double fixedValue,
                    double &extremumCoordinate, double initialGuess = 0.5 ) {
  double tol = 1e-8; // if we move less than this in a single step, we figure we've converged.
  
  // set initial guess
  double z = initialGuess;  // calling the one that varies z (might be x or y)

  double x, y;
  if (variableChoice == X_VARIES) {
    y = fixedValue;
  } else {
    x = fixedValue;
  }
  
  double incr = 1;
  while (abs(incr) > tol) {
    if (variableChoice == X_VARIES) {
      x = z;
    } else {
      y = z;
    }
    double f_value = Function::evaluate(f, x, y);
    double f_prime_value = Function::evaluate(f_prime, x, y);
    incr = - f_value / f_prime_value;
    z += incr;
    
    if ((z < 0) || (z > 1)) {
      cout << "ERROR: findExtremumOnLine diverged (left the mesh).\n";
      return false;
    }
    extremumCoordinate = z;
  }
  return true;
}

// gradient ascent/descent
bool findExtremum(bool findMax, FunctionPtr fxn, FunctionPtr grad, MeshPtr mesh, double &value, double &x, double &y,
                  double x0 = 0.5, double y0 = 0.5) {
  double xyPointTol = 1e-8; // convergence criterion (euclidean distance)
  x = x0; // initial guess
  y = y0;
  double dx, dy;

  double parity = findMax ? 1.0 : -1.0; // ascent vs. descent
  
  double l2incr = 1.0;
  double gamma = 1.0; // unsure what gamma should be…
  int iterCount = 0;
  while (l2incr > xyPointTol) {
    value = Function::evaluate(fxn, x, y);
    dx = Function::evaluate(grad->x(), x, y);
    dy = Function::evaluate(grad->y(), x, y);
    
    gamma = sqrt( abs(dx * dy) ); // geometric mean of the gradient components.  No idea if this is reasonable...

    double x_incr = parity * gamma * dx;
    double y_incr = parity * gamma * dy;
    
    x = x + x_incr;
    y = y + y_incr;
    
    if ((x < 0) || (x > 1) || (y < 0) || (y > 1)) {
      cout << "ERROR: findExtremum diverged (left the mesh).\n";
      return false;
    }
    
//    cout << "(x,y) = (" << x << ", " << y << ")\n";
    
    iterCount++;
    l2incr = sqrt( x_incr * x_incr + y_incr * y_incr );
  }
  cout << "gradient descent/ascent took " << iterCount << " iterations.\n";
  return true;
}

bool findMaximum(FunctionPtr fxn, FunctionPtr grad, MeshPtr mesh, double &value, double &x, double &y,
                 double x0 = 0.5, double y0 = 0.5) {
  return findExtremum(true, fxn, grad, mesh, value, x, y, x0, y0);
}

bool findMinimum(FunctionPtr fxn, FunctionPtr grad, MeshPtr mesh, double &value, double &x, double &y,
                 double x0 = 0.5, double y0 = 0.5) {
  return findExtremum(false, fxn, grad, mesh, value, x, y, x0, y0);
}

void reportStreamfunctionMaxValue(SolutionPtr streamSolution, VarPtr phi, FunctionPtr vorticity, double Re) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  FunctionPtr phiFxn = Teuchos::rcp( new PreviousSolutionFunction( streamSolution, phi ) );
  FunctionPtr phiGrad = Function::vectorize(Teuchos::rcp( new PreviousSolutionFunction( streamSolution, phi->dx() ) ),
                                            Teuchos::rcp( new PreviousSolutionFunction( streamSolution, phi->dy() ) ));
  
  double x_primary = 0.5, y_primary = 0.5;
  double x_left_secondary = 0.05, y_left_secondary = 0.05;
  double x_right_secondary = 0.95, y_right_secondary = 0.05;
  
  if (Re == 1000) {
    x_primary = 0.4692;
    y_primary = 0.5652;
    x_left_secondary = 0.1360;
    y_left_secondary = 0.1118;
    x_right_secondary = 0.9167;
    y_right_secondary = 0.0781;
  }
  
  map< string, pair<double, double> > initialGuesses;
  initialGuesses["primary"] = make_pair(x_primary, y_primary);
  initialGuesses["left secondary"] = make_pair(x_left_secondary, y_left_secondary);
  initialGuesses["right secondary"] = make_pair(x_right_secondary, y_right_secondary);
  
  map< string, bool > vortexIsMaximum;
  vortexIsMaximum["primary"] = true;
  vortexIsMaximum["left secondary"] = false;
  vortexIsMaximum["right secondary"] = false;
  
  if (Re == 1000) {
    initialGuesses["lower left tertiary"] = make_pair(0.00768, 0.00765);
    vortexIsMaximum["lower left tertiary"] = true;
  }
  
  if (rank==0) {
    for (map< string, pair<double, double> >::iterator guessIt = initialGuesses.begin();
         guessIt != initialGuesses.end(); guessIt++) {
      string vortexID = guessIt->first;
      double x0 = guessIt->second.first;
      double y0 = guessIt->second.second;
      
      double x, y, value;
      bool success;
      if (vortexIsMaximum[vortexID]) {
        success = findMaximum(phiFxn, phiGrad, streamSolution->mesh(), value, x, y, x0, y0);
      } else {
        success = findMinimum(phiFxn, phiGrad, streamSolution->mesh(), value, x, y, x0, y0);
      }
      if (success) {
        cout << "phiMax for " << vortexID << " vortex is " << value << " at (" << x << ", " << y << ")\n";
        ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(false);
        cout << "vorticity at (" << x << ", " << y << ") = " << Function::evaluate(vorticity, x, y) << endl;
      } else {
        cout << "search for " << vortexID << " vortex diverged--left the mesh to point (" << x << ", " << y << ")\n";
      }
    }
  }
  
}

void reportCenterlineVelocityValues(FunctionPtr u1_prev, FunctionPtr u2_prev, FunctionPtr p_prev, FunctionPtr vorticity,
                                    double &u1_max_y, double &u2_max_x, double &u2_min_x) {
  int rank = Teuchos::GlobalMPISession::getRank();

  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(false); // allows Function::evaluate() call, below
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(false);
  // the next bit commented out -- Function::evaluate() must depend only on space, not on the mesh (prev soln depends on mesh)
  vector<double> u2values, pValues, vorticityValues;
  
  double x,y;
  y = 0.5;
  vector<double> xPoints = horizontalCenterLinePoints();
  for (int i=0; i<xPoints.size(); i++) {
    x = xPoints[i];
    u2values.push_back( Function::evaluate(u2_prev, x, y) );
    pValues.push_back( Function::evaluate(p_prev, x, y) );
    vorticityValues.push_back( Function::evaluate(vorticity, x, y) );
  }
  
  // botella and peyret set the center pressure to 0, so we do that too
  double centerPressure = Function::evaluate(p_prev, 0.5, 0.5);
  for (int i=0; i<xPoints.size(); i++) {
    pValues[i] -= centerPressure;
  }
  
  // search for min/max values
  double u1_max = 0, u2_max = 0, u2_min = 1;
  for (int i=0; i<xPoints.size(); i++) {
    if (u2values[i] > u2_max) {
      u2_max = u2values[i];
      u2_max_x = xPoints[i];
    }
    if (u2values[i] < u2_min) {
      u2_min = u2values[i];
      u2_min_x = xPoints[i];
    }
  }
  
  if (rank==0) {
    cout << "**** horizontal center line, values ****\n";
    int w = 20;
    cout << setw(w) << "x" << setw(w) << "u2" << setw(w) << "p" << setw(w) << "omega" << endl;
    for (int i=0; i<xPoints.size(); i++) {
      cout << setw(w) << xPoints[i] << setw(w) << u2values[i] << setw(w) << pValues[i] << setw(w) << vorticityValues[i] << endl;
    }
  }
  pValues.clear();
  vorticityValues.clear();
  vector<double> u1values;
  x = 0.5;
  vector<double> yPoints = verticalCenterLinePoints();
  
  for (int i=0; i<yPoints.size(); i++) {
    y = yPoints[i];
    u1values.push_back( Function::evaluate(u1_prev, x, y) );
    pValues.push_back( Function::evaluate(p_prev, x, y) );
    vorticityValues.push_back( Function::evaluate(vorticity, x, y) );
  }
  
  // botella and peyret set the center pressure to 0, so we do that too
  for (int i=0; i<xPoints.size(); i++) {
    pValues[i] -= centerPressure;
  }
  
  // search for min/max values
  for (int i=0; i<yPoints.size(); i++) {
    if (u1values[i] > u1_max) {
      u1_max = u1values[i];
      u1_max_y = yPoints[i];
    }
  }
  
  if (rank==0) {
    cout << "**** vertical center line, values ****\n";
    int w = 20;
    cout << setw(w) << "y" << setw(w) << "u1" << setw(w) << "p" << setw(w) << "omega" << endl;
    for (int i=0; i<yPoints.size(); i++) {
      cout << setw(w) << yPoints[i] << setw(w) << u1values[i] << setw(w) << pValues[i] << setw(w) << vorticityValues[i] << endl;
    }
  }
}

void reportVelocityExtrema(FunctionPtr u1, FunctionPtr u1dy, FunctionPtr u1dydy,
                           FunctionPtr u2, FunctionPtr u2dx, FunctionPtr u2dxdx,
                           double u1_max_guess_y, double u2_max_guess_x, double u2_min_guess_x) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  if (rank==0) {
    // find the extrema of the velocity components on the centerlines
    double u1_max, y_max;
    double u2_max, x_max;
    double u2_min, x_min;
    
    y_max = u1_max_guess_y;
    x_max = u2_max_guess_x;
    x_min = u2_min_guess_x;
    
    if (! findZeroOnLine(u1dy, u1dydy, Y_VARIES, 0.5, y_max, y_max) ) {
      cout << "could not resolve y_max\n";
    }
    if (! findZeroOnLine(u2dx, u2dxdx, X_VARIES, 0.5, x_max, x_max) ) {
      cout << "could not resolve x_max\n";
    }
    if (! findZeroOnLine(u2dx, u2dxdx, X_VARIES, 0.5, x_min, x_min) ) {
      cout << "could not resolve x_min\n";
    }
    
    u1_max = Function::evaluate(u1, 0.5, y_max);
    u2_max = Function::evaluate(u2, x_max, 0.5);
    u2_min = Function::evaluate(u2, x_min, 0.5);

    cout << "*************************** velocity extrema ***************************\n";
    int w = 20;
    cout << setw(w) << "u1max" << setw(w) << "ymax" << setw(w) << "u2max" << setw(w) << "xmax" << setw(w) << "u2min" << setw(w) << "xmin" << endl;
    cout << setw(w) << u1_max << setw(w) << y_max << setw(w) << u2_max << setw(w) << x_max << setw(w) << u2_min << setw(w) << x_min << endl;
  }
}

vector<double> verticalLinePoints(double pointIncrement = -1) {
  vector<double> yPoints;
  int rank = Teuchos::GlobalMPISession::getRank();

//  if (rank==0) cout << "verticalLinePoints: pointIncrement " << pointIncrement << endl;
  if (pointIncrement < 0) {
    // points where values are often reported in the literature (in Botella and Peyret)
    yPoints.push_back(1.0000);
    yPoints.push_back(0.9766);
    yPoints.push_back(0.9688);
    yPoints.push_back(.9609);
    yPoints.push_back(.9531);
    yPoints.push_back(.8516);
    yPoints.push_back(.7344);
    yPoints.push_back(.6172);
    yPoints.push_back(.5000);
    yPoints.push_back(.4531);
    yPoints.push_back(.2813);
    yPoints.push_back(.1719);
    yPoints.push_back(.1016);
    yPoints.push_back(.0703);
    yPoints.push_back(.0625);
    yPoints.push_back(.0547);
    yPoints.push_back(0);
  } else {
    double y;
    for (y = 1.0; y >= 0; y -= pointIncrement) {
      yPoints.push_back(y);
    }
  }
  
  return yPoints;
}

struct VerticalLineSolutionValues {
  double x;
  vector<double> yPoints;
  vector<double> u1;
  vector<double> u2;
  vector<double> omega;
  vector<double> p;
  vector<double> sigma11; // these last 4 are only populated by the "Exhaustive" guy
  vector<double> sigma12;
  vector<double> sigma21;
  vector<double> sigma22;
};

VerticalLineSolutionValues computeVerticalLineSolutionValuesExhaustive(double xValue, FunctionPtr u1_prev, FunctionPtr u2_prev,
                                                                       FunctionPtr p_prev, FunctionPtr vorticity,
                                                                       FunctionPtr sigma11_prev, FunctionPtr sigma12_prev,
                                                                       FunctionPtr sigma21_prev, FunctionPtr sigma22_prev) {
  
  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(false); // allows Function::evaluate() call, below
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) p_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) sigma11_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) sigma12_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) sigma21_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) sigma22_prev.get())->setOverrideMeshCheck(false);
  
  double pOffset = Function::evaluate(p_prev, 0.5, 0.5);
  VerticalLineSolutionValues values;
  
  double y;
  double x = xValue;
  values.x = xValue;
  values.yPoints = verticalLinePoints(); // matches Botella & Peyret's points
  for (int i=0; i<values.yPoints.size(); i++) {
    y = values.yPoints[i];
    values.u1.push_back( Function::evaluate(u1_prev, x, y) );
    values.u2.push_back( Function::evaluate(u2_prev, x, y) );
    values.p.push_back( Function::evaluate(p_prev, x, y) - pOffset );
    values.omega.push_back( Function::evaluate(vorticity, x, y) );
    values.sigma11.push_back( Function::evaluate(sigma11_prev, x, y) );
    values.sigma12.push_back( Function::evaluate(sigma12_prev, x, y) );
    values.sigma21.push_back( Function::evaluate(sigma21_prev, x, y) );
    values.sigma22.push_back( Function::evaluate(sigma22_prev, x, y) );
  }
  return values;
}

VerticalLineSolutionValues computeVerticalLineSolutionValues(double xValue, FunctionPtr u1_prev, FunctionPtr u2_prev,
                                                             FunctionPtr p_prev, FunctionPtr vorticity) {
  
  
  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(false); // allows Function::evaluate() call, below
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) p_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(false);
  // the next bit commented out -- Function::evaluate() must depend only on space, not on the mesh (prev soln depends on mesh)
  
  double pOffset = Function::evaluate(p_prev, 0.5, 0.5);
  
  VerticalLineSolutionValues values;
  
  double y;
  double x = xValue;
  values.x = xValue;
  double pointIncrement = .005; // this one for file output, ultimately visualization; 200 points seems a reasonable number
  values.yPoints = verticalLinePoints(pointIncrement);
  for (int i=0; i<values.yPoints.size(); i++) {
    y = values.yPoints[i];
    values.u1.push_back( Function::evaluate(u1_prev, x, y) );
    values.u2.push_back( Function::evaluate(u2_prev, x, y) );
    values.p.push_back( Function::evaluate(p_prev, x, y) - pOffset );
    values.omega.push_back( Function::evaluate(vorticity, x, y) );
  }
  return values;
}

int main(int argc, char *argv[]) {
  int rank = 0;
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
#ifdef HAVE_MPI
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args(argc, argv );
#endif
  
  try {
  // read args:
    int polyOrder = args.Input<int>("--polyOrder", "L^2 (field) polynomial order");
    int numRefs = args.Input<int>("--numRefinements", "Number of refinements", 6);
    double Re = args.Input<double>("--Re", "Reynolds number", 400);
    bool longDoubleGramInversion = args.Input<bool>("--longDoubleGramInversion", "use long double Cholesky factorization for Gram matrix", false);
    int horizontalCells = args.Input<int>("--horizontalCells", "horizontal cell count for initial mesh (if vertical unspecified, will match horizontal)", 2);
    int verticalCells = args.Input<int>("--verticalCells", "vertical cell count for initial mesh", horizontalCells);
    bool outputStiffnessMatrix = args.Input<bool>("--writeFinalStiffnessToDisk", "write the final stiffness matrix to disk.", false);
    bool computeMaxConditionNumber = args.Input<bool>("--computeMaxConditionNumber", "compute the maximum Gram matrix condition number for final mesh.", false);
    bool enforceLocalConservation = args.Input<bool>("--enforceLocalConservation", "enforce local conservation using Lagrange constraints", false);
    bool useCompliantGraphNorm = args.Input<bool>("--useCompliantNorm", "use the 'scale-compliant' graph norm", false);
    bool useCondensedSolve = args.Input<bool>("--useCondensedSolve", "use static condensation", true);
    bool reportConditionNumber = args.Input<bool>("--reportGlobalConditionNumber", "report the 2-norm condition number for the global system matrix", false);
    
    bool enhanceFluxOrders = args.Input<bool>("--enhanceFluxOrders", "enhance flux polynomial orders to match test space", false);

    //bool adaptForLRCornerVorticity = args.Input<bool>("--adaptForLRCornerVorticity", "do goal-oriented ", false);
    
    bool reportStreamfunctionMax = args.Input<bool>("--reportStreamfunctionMax", "report streamfunction max value", true);
    bool reportCenterlineVelocities = args.Input<bool>("--reportCenterlineVelocities", "report centerline velocities", true);
    bool reportVelocityExtremeValues = args.Input<bool>("--reportVelocityExtrema", "report velocity extrema", false);
    
    double dt = args.Input<double>("--timeStep", "time step (0 for none)", 0); // 0.5 used to be the standard value

    bool weightIncrementL2Norm = useCompliantGraphNorm; // if using the compliant graph norm, weight the measure of the L^2 increment accordingly
    
    bool induceCornerRefinements = args.Input<bool>("--induceCornerRefinements", "induce refinements in the recirculating corner", false);
    
    int maxIters = args.Input<int>("--maxIters", "maximum number of Newton-Raphson iterations to take to try to match tolerance", 50);
    double minL2Increment = args.Input<double>("--NRtol", "Newton-Raphson tolerance, L^2 norm of increment", 3e-8);
    string replayFile = args.Input<string>("--replayFile", "file with refinement history to replay", "");
    string solnFile = args.Input<string>("--solnFile", "file with solution data", "");
    string solnSaveFile = args.Input<string>("--solnSaveFile", "file to which to save solution data", "");
    string saveFile = args.Input<string>("--saveReplay", "file to which to save refinement history", "");
    
//    double finalSolveMinL2Increment = args.Input<double>("--finalNRtol", "Newton-Raphson tolerance for final solve, L^2 norm of increment", minL2Increment / 10);
    
    bool useAdHocHPRefinements = args.Input<bool>("--useAdHocHPRefinements", "use ad hoc hp refinements", false);
    
    double eps = args.Input<double>("--rampWidth", "width of 'ramp' in BCs", 1.0/64.0);
    
    args.Process();
    
    bool useLineSearch = false;
    
    int pToAdd = 2; // for optimal test function approximation
    int pToAddForStreamFunction = 2;
//    double nonlinearStepSize = 1.0;

//    double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  //  double nonlinearRelativeEnergyTolerance = 0.15; // used to determine convergence of the nonlinear solution
    // epsilon above is chosen to match our initial 16x16 mesh, to avoid quadrature errors.
  //  double eps = 0.0; // John Evans's problem: not in H^1
//    bool enforceLocalConservationInFinalSolve = false; // only works correctly for Picard (and maybe not then!)
    bool enforceOneIrregularity = true;
    bool reportPerCellErrors  = true;
    bool useMumps = true;
    bool compareWithOverkillMesh = false;
    bool startWithZeroSolutionAfterRefinement = false;
    
    bool artificialTimeStepping = (dt > 0);
    
    int overkillMeshSize = 8;
    int overkillPolyOrder = 7; // H1 order
    
    if (rank == 0) {
      cout << "numRefinements = " << numRefs << endl;
      cout << "Re = " << Re << endl;
      cout << "initial mesh: " << horizontalCells << " x " << verticalCells << endl;
      if (artificialTimeStepping) cout << "dt = " << dt << endl;
      if (!startWithZeroSolutionAfterRefinement) {
        cout << "NOTE: experimentally, NOT starting with 0 solution after refinement...\n";
      }
    }
    
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
    bool useTriangles = false;
    bool meshHasTriangles = useTriangles;
    
    // get variable definitions:
    VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
    u1 = varFactory.fieldVar(VGP_U1_S);
    u2 = varFactory.fieldVar(VGP_U2_S);
    sigma11 = varFactory.fieldVar(VGP_SIGMA11_S);
    sigma12 = varFactory.fieldVar(VGP_SIGMA12_S);
    sigma21 = varFactory.fieldVar(VGP_SIGMA21_S);
    sigma22 = varFactory.fieldVar(VGP_SIGMA22_S);
    p = varFactory.fieldVar(VGP_P_S);
    
    u1hat = varFactory.traceVar(VGP_U1HAT_S);
    u2hat = varFactory.traceVar(VGP_U2HAT_S);
    t1n = varFactory.fluxVar(VGP_T1HAT_S);
    t2n = varFactory.fluxVar(VGP_T2HAT_S);
    
    v1 = varFactory.testVar(VGP_V1_S, HGRAD);
    v2 = varFactory.testVar(VGP_V2_S, HGRAD);
    tau1 = varFactory.testVar(VGP_TAU1_S, HDIV);
    tau2 = varFactory.testVar(VGP_TAU2_S, HDIV);
    q = varFactory.testVar(VGP_Q_S, HGRAD);
    
  //  // create a pointer to a new mesh:
  //  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
  //                                                navierStokesBF, H1Order, H1Order+pToAdd, useTriangles);

  //  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  RHSPtr rhs = RHS::rhs(); // zero for now...
  //  IPPtr ip = initGraphInnerProductStokes(mu);

  //  SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) ); // accumulated solution
  //  SolutionPtr solnIncrement = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  //  solnIncrement->setReportConditionNumber(false);
    
    FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
    FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
    FunctionPtr zero = Function::zero();
    ParameterFunctionPtr Re_param = ParameterFunction::parameterFunction(Re);
    VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re_param,quadPoints,
                                                            horizontalCells,verticalCells,
                                                            H1Order, pToAdd,
                                                            u1_0, u2_0,  // BC for u
                                                            zero, zero,  // zero forcing function
                                                            useCompliantGraphNorm, // enrich velocity if using compliant graph norm
                                                            enhanceFluxOrders);
    
    SolutionPtr solution = problem.backgroundFlow();
    solution->setReportConditionNumber(reportConditionNumber);
    SolutionPtr solnIncrement = problem.solutionIncrement();
    solnIncrement->setReportConditionNumber(reportConditionNumber);
    
    problem.bf()->setUseExtendedPrecisionSolveForOptimalTestFunctions(longDoubleGramInversion);
    
    Teuchos::RCP<Mesh> mesh = problem.mesh();
    mesh->registerSolution(solution);
    mesh->registerSolution(solnIncrement);

    Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
    mesh->registerObserver(refHistory);
    
  //  if ( ! usePicardIteration ) { // we probably could afford to do pseudo-time with Picard, but choose not to
  //    // add time marching terms for momentum equations (v1 and v2):
    ParameterFunctionPtr dt_inv = ParameterFunction::parameterFunction(1.0 / dt); //Teuchos::rcp( new ConstantScalarFunction(1.0 / dt, "\\frac{1}{dt}") );
    if (artificialTimeStepping) {
  //    // LHS gets u_inc / dt:
      BFPtr bf = problem.bf();
      FunctionPtr dt_inv_fxn = Teuchos::rcp(dynamic_cast< Function* >(dt_inv.get()), false);
      bf->addTerm(-dt_inv_fxn * u1, v1);
      bf->addTerm(-dt_inv_fxn * u2, v2);
      problem.setIP( bf->graphNorm() ); // graph norm has changed...
    }
    
    if (useCompliantGraphNorm) {
      if (artificialTimeStepping) {
        problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm(dt_inv));
      } else {
        problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm());
      }
      // (otherwise, will use graph norm)
    }
    
  //  }
    
  //  if (rank==0) {
  //    cout << "********** STOKES BF **********\n";
  //    stokesBFMath->printTrialTestInteractions();
  //    cout << "\n\n********** NAVIER-STOKES BF **********\n";
  //    navierStokesBF->printTrialTestInteractions();
  //    cout << "\n\n";
  //  }
    
    // set initial guess (all zeros is probably a decent initial guess here)
  //  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0) );
  //  map< int, FunctionPtr > initialGuesses;
  //  initialGuesses[u1->ID()] = zero;
  //  initialGuesses[u2->ID()] = zero;
  //  initialGuesses[sigma11->ID()] = zero;
  //  initialGuesses[sigma12->ID()] = zero;
  //  initialGuesses[sigma21->ID()] = zero;
  //  initialGuesses[sigma22->ID()] = zero;
  //  initialGuesses[p->ID()] = zero;
  //  initialGuesses[u1hat->ID()] = zero;
  //  initialGuesses[u2hat->ID()] = zero;
  //  initialGuesses[t1n->ID()] = zero;
  //  initialGuesses[t2n->ID()] = zero;
  //  solution->projectOntoMesh(initialGuesses);
    
    ///////////////////////////////////////////////////////////////////////////
    
    // define bilinear form for stream function:
    VarFactory streamVarFactory;
    VarPtr phi_hat = streamVarFactory.traceVar("\\widehat{\\phi}");
    VarPtr psin_hat = streamVarFactory.fluxVar("\\widehat{\\psi}_n");
    VarPtr psi_1 = streamVarFactory.fieldVar("\\psi_1");
    VarPtr psi_2 = streamVarFactory.fieldVar("\\psi_2");
    VarPtr phi = streamVarFactory.fieldVar("\\phi");
    VarPtr q_s = streamVarFactory.testVar("q_s", HGRAD);
    VarPtr v_s = streamVarFactory.testVar("v_s", HDIV);
    BFPtr streamBF = Teuchos::rcp( new BF(streamVarFactory) );
    streamBF->addTerm(psi_1, q_s->dx());
    streamBF->addTerm(psi_2, q_s->dy());
    streamBF->addTerm(-psin_hat, q_s);
    
    streamBF->addTerm(psi_1, v_s->x());
    streamBF->addTerm(psi_2, v_s->y());
    streamBF->addTerm(phi, v_s->div());
    streamBF->addTerm(-phi_hat, v_s->dot_normal());
    
    Teuchos::RCP<Mesh> streamMesh, overkillMesh;
    
    streamMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                     streamBF, H1Order+pToAddForStreamFunction,
                                     H1Order+pToAdd+pToAddForStreamFunction, useTriangles);

    mesh->registerObserver(streamMesh); // will refine streamMesh in the same way as mesh.
    
    if (replayFile.length() > 0) {
      RefinementHistory refHistory;
      refHistory.loadFromFile(replayFile);
      refHistory.playback(mesh);
    }
    if (solnFile.length() > 0) {
      solution->readFromFile(solnFile);
    }
    
    Teuchos::RCP<Solution> overkillSolution;
    map<int, double> dofsToL2error; // key: numGlobalDofs, value: total L2error compared with overkill
    vector< VarPtr > fields;
    fields.push_back(u1);
    fields.push_back(u2);
    fields.push_back(sigma11);
    fields.push_back(sigma12);
    fields.push_back(sigma21);
    fields.push_back(sigma22);
    fields.push_back(p);
    
    if (rank == 0) {
      cout << "Starting mesh has " << mesh->activeElements().size() << " elements and ";
      cout << mesh->numGlobalDofs() << " total dofs.\n";
      cout << "polyOrder = " << polyOrder << endl; 
      cout << "pToAdd = " << pToAdd << endl;
      cout << "eps for top BC = " << eps << endl;
      
      if (useTriangles) {
        cout << "Using triangles.\n";
      }
      if (enforceLocalConservation) {
        cout << "Enforcing local conservation.\n";
      } else {
        cout << "NOT enforcing local conservation.\n";
      }
      if (enforceOneIrregularity) {
        cout << "Enforcing 1-irregularity.\n";
      } else {
        cout << "NOT enforcing 1-irregularity.\n";
      }
      if (saveFile.length() > 0) {
        cout << "will save refinement history to file " << saveFile << endl;
      }
      if (replayFile.length() > 0) {
        cout << "will replay refinements from file " << replayFile << endl;
      }
      if (useCondensedSolve) {
        cout << "using condensed solve.\n";
      } else {
        cout << "not using condensed solve.\n";
      }
    }
    
//    cout << "Processor #" << rank << " got to: " << __LINE__ << endl;
    
    ////////////////////   CREATE BCs   ///////////////////////
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered );

    FunctionPtr u1_prev = Teuchos::rcp(new PreviousSolutionFunction(solution, u1));
    FunctionPtr u2_prev = Teuchos::rcp(new PreviousSolutionFunction(solution, u2));
    FunctionPtr p_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,p) );
    
    FunctionPtr u1hat_prev = Function::solution(u1hat,solution);
    FunctionPtr u2hat_prev = Function::solution(u2hat,solution);
    
    FunctionPtr sigma12_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma12) );
    FunctionPtr sigma12_dy = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma12->dy()) );
    FunctionPtr sigma21_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma21) );
    FunctionPtr sigma21_dx = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma21->dx()) );
    
    FunctionPtr sigma11_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma11 ) );
    FunctionPtr sigma22_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma22 ) );
    
  //  if ( ! usePicardIteration ) {
  //    bc->addDirichlet(u1hat, entireBoundary, u1_0 - u1hat_prev);
  //    bc->addDirichlet(u2hat, entireBoundary, u2_0 - u2hat_prev);
  //  // as long as we don't subtract from the RHS, I think the following is actually right:
  ////    bc->addDirichlet(u1hat, entireBoundary, u1_0);
  ////    bc->addDirichlet(u2hat, entireBoundary, u2_0);
  //  } else {
  ////    bc->addDirichlet(u1hat, entireBoundary, u1_0);
  ////    bc->addDirichlet(u2hat, entireBoundary, u2_0);
  //    // experiment:
  //    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  //    pc->addConstraint(u1hat==u1_0,entireBoundary);
  //    pc->addConstraint(u2hat==u2_0,entireBoundary);
  //    solnIncrement->setFilter(pc);
  //  }
  //  bc->addZeroMeanConstraint(p);
  //  
    /////////////////// SOLVE OVERKILL //////////////////////
  //  if (compareWithOverkillMesh) {
  //    // TODO: fix this to make it work with Navier-Stokes
  //    cout << "WARNING: still need to switch overkill to handle nonlinear iteration...\n";
  //    overkillMesh = MeshFactory::buildQuadMesh(quadPoints, overkillMeshSize, overkillMeshSize,
  //                                       stokesBFMath, overkillPolyOrder, overkillPolyOrder+pToAdd, useTriangles);
  //    
  //    if (rank == 0) {
  //      cout << "Solving on overkill mesh (" << overkillMeshSize << " x " << overkillMeshSize << " elements, ";
  //      cout << overkillMesh->numGlobalDofs() <<  " dofs).\n";
  //    }
  //    overkillSolution = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
  //    overkillSolution->solve();
  //    if (rank == 0)
  //      cout << "...solved.\n";
  //    double overkillEnergyError = overkillSolution->energyErrorTotal();
  //    if (rank == 0)
  //      cout << "overkill energy error: " << overkillEnergyError << endl;
  //  }
    
    
//    cout << "Processor #" << rank << " got to: " << __LINE__ << endl;
    ////////////////////   SOLVE & REFINE   ///////////////////////
    
    FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, (-Re) * sigma12 + Re * sigma21 ) );
    //  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution,sigma12 - sigma21) );
    RHSPtr streamRHS = RHS::rhs();
    streamRHS->addTerm(vorticity * q_s);
    ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
    ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true);
    ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);
    
    Teuchos::RCP<BCEasy> streamBC = Teuchos::rcp( new BCEasy );
  //  streamBC->addDirichlet(psin_hat, entireBoundary, u0_cross_n);
    streamBC->addDirichlet(phi_hat, entireBoundary, zero);
  //  streamBC->addZeroMeanConstraint(phi);
    
    IPPtr streamIP = Teuchos::rcp( new IP );
    streamIP->addTerm(q_s);
    streamIP->addTerm(q_s->grad());
    streamIP->addTerm(v_s);
    streamIP->addTerm(v_s->div());
    SolutionPtr streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );
    
    if (enforceLocalConservation) {
      FunctionPtr zero = Function::zero();
      solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
      solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
    }
    
    FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
    
    double energyThreshold = 0.20; // for mesh refinements
    Teuchos::RCP<RefinementStrategy> refinementStrategy;
    if (useAdHocHPRefinements) {
      if (! compareWithOverkillMesh )
        refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solution, energyThreshold, 0, 15 )); // no minimum h, and a generous p maximum
      else
        refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solnIncrement, energyThreshold, 1.0 / overkillMeshSize, overkillPolyOrder, rank==0 ));
    } else {
//      if (rank==0) cout << "NOTE: using solution, not solnIncrement, for refinement strategy.\n";
//      refinementStrategy = Teuchos::rcp( new RefinementStrategy( solution, energyThreshold ));
      refinementStrategy = Teuchos::rcp( new RefinementStrategy( solnIncrement, energyThreshold ));
    }
    
    refinementStrategy->setEnforceOneIrregularity(enforceOneIrregularity);
    refinementStrategy->setReportPerCellErrors(reportPerCellErrors);

//    Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
//    Teuchos::RCP<NonlinearSolveStrategy> solveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
//                                                                                                 stepSize,
//                                                                                                 nonlinearRelativeEnergyTolerance));
//    
//    Teuchos::RCP<NonlinearSolveStrategy> finalSolveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
//                                                                                                 stepSize,
//                                                                                                 nonlinearRelativeEnergyTolerance / 10));

    
    
  //  solveStrategy->setUsePicardIteration(usePicardIteration);
    
    // run some refinements on the initial linear problem
  //  int numInitialRefs = 5;
  //  for (int refIndex=0; refIndex<numInitialRefs; refIndex++){    
  //    solnIncrement->solve();
  //    refinementStrategy->refine(rank==0); // print to console on rank 0
  //  }
  //  solveStrategy->solve(rank==0);
    
    vector<double> relativeEnergyErrors(numRefs + 1);
    vector<long> dofCounts(numRefs+1);
    vector<long> fluxDofCounts(numRefs+1);
    vector<long> elementCounts(numRefs+1);
    vector<int> iterationCounts(numRefs+1);
    vector<double> tolerances(numRefs+1);
    
    map<double, vector< VerticalLineSolutionValues > > verticalCutValues; // keys are x locations
    vector< VerticalLineSolutionValues > verticalCutValuesExhaustive;
    
    // just one x location of interest for cavity flow -- the center line:
    verticalCutValues[0.5] = vector< VerticalLineSolutionValues >();
    
    double initialMinL2Increment = minL2Increment;
    if (rank==0) cout << "Initial relative L^2 tolerance: " << minL2Increment << endl;
    
    LinearTermPtr backgroundSolnFunctional = problem.bf()->testFunctional(problem.backgroundFlow());
    RieszRep solnRieszRep(mesh, problem.solutionIncrement()->ip(), backgroundSolnFunctional);
    
    LinearTermPtr incrementalSolnFunctional = problem.bf()->testFunctional(problem.solutionIncrement());
    RieszRep incrementRieszRep(mesh, problem.solutionIncrement()->ip(), incrementalSolnFunctional);
    
    tolerances[0] = initialMinL2Increment;
    
    if (true) { // do regular refinement strategy...
      FieldContainer<double> bottomCornerPoints(2,2);
      bottomCornerPoints(0,0) = 1e-10;
      bottomCornerPoints(0,1) = 1e-10;
      bottomCornerPoints(1,0) = 1 - 1e-10;
      bottomCornerPoints(1,1) = 1e-10;
      
      FieldContainer<double> topCornerPoints(4,2);
      topCornerPoints(0,0) = 1e-10;
      topCornerPoints(0,1) = 1 - 1e-12;
      topCornerPoints(1,0) = 1 - 1e-10;
      topCornerPoints(1,1) = 1 - 1e-12;
      topCornerPoints(2,0) = 1e-12;
      topCornerPoints(2,1) = 1 - 1e-10;
      topCornerPoints(3,0) = 1 - 1e-12;
      topCornerPoints(3,1) = 1 - 1e-10;
      
      bool printToConsole = rank==0;
      FunctionPtr u1_incr = Function::solution(u1, solnIncrement);
      FunctionPtr u2_incr = Function::solution(u2, solnIncrement);
      FunctionPtr sigma11_incr = Function::solution(sigma11, solnIncrement);
      FunctionPtr sigma12_incr = Function::solution(sigma12, solnIncrement);
      FunctionPtr sigma21_incr = Function::solution(sigma21, solnIncrement);
      FunctionPtr sigma22_incr = Function::solution(sigma22, solnIncrement);
      FunctionPtr p_incr = Function::solution(p, solnIncrement);
      
      FunctionPtr l2_incr;
      
      if (! weightIncrementL2Norm) {
        l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
        + sigma11_incr * sigma11_incr + sigma12_incr * sigma12_incr
        + sigma21_incr * sigma21_incr + sigma22_incr * sigma22_incr;
      } else {
        double Re2 = Re * Re;
        l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
        + Re2 * sigma11_incr * sigma11_incr + Re2 * sigma12_incr * sigma12_incr
        + Re2 * sigma21_incr * sigma21_incr + Re2 * sigma22_incr * sigma22_incr;
      }
      
      for (int refIndex=0; refIndex<numRefs; refIndex++){
        if (startWithZeroSolutionAfterRefinement) {
          // start with a fresh (zero) initial guess for each adaptive mesh:
          solution->clear();
          problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
        }
        
        if (computeMaxConditionNumber) {
          IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(problem.solutionIncrement()->ip().get()), false );
          bool jacobiScalingTrue = true;
          double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, problem.solutionIncrement()->mesh(), jacobiScalingTrue);
          if (rank==0) {
            cout << "max jacobi-scaled Gram matrix condition number estimate with zero background flow: " << maxConditionNumber << endl;
          }
        }
        
        double incr_norm;
        do {
          problem.iterate(useLineSearch, useCondensedSolve);
          incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
          
          if (rank==0) {
            cout << "\x1B[2K"; // Erase the entire current line.
            cout << "\x1B[0E"; // Move to the beginning of the current line.
            cout << "Refinement # " << refIndex << ", iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
            flush(cout);
          }
        } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
        
        iterationCounts[refIndex] = (refIndex==0) ? problem.iterationCount() : problem.iterationCount() - 1;

        
        if (rank==0)
          cout << "\nFor refinement " << refIndex << ", num iterations: " << problem.iterationCount() << endl;
        
        if (computeMaxConditionNumber) {
          IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(problem.solutionIncrement()->ip().get()), false );
          bool jacobiScalingTrue = true;
          double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, problem.solutionIncrement()->mesh(), jacobiScalingTrue);
          if (rank==0) {
            cout << "max jacobi-scaled Gram matrix condition number estimate with nonzero background flow: " << maxConditionNumber << endl;
          }
        }
        
        if (reportCenterlineVelocities) {
          double u1_max_y, u2_max_x, u2_min_x; // the maxes and mins we find along the reported centerline points
          reportCenterlineVelocityValues(u1_prev,u2_prev,p_prev,vorticity, u1_max_y, u2_max_x, u2_min_x);
        }
        
        if (reportVelocityExtremeValues) {
          double u1_max_y, u2_max_x, u2_min_x; // the maxes and mins we find along the reported centerline points
          reportVelocityExtrema(u1_prev, sigma12_prev, sigma12_dy,
                                u2_prev, sigma21_prev, sigma21_dx,
                                u1_max_y, u2_max_x, u2_min_x);
        }
        
        // stream solution:
        if (reportStreamfunctionMax) {
          
          ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true); // speeds up the stream solution solve
          ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);
          ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);

          if (rank == 0) {
            cout << "solving for approximate stream function...\n";
          }
          
          if (!useCondensedSolve){
            streamSolution->solve(useMumps);
          } else {
            streamSolution->condensedSolve();
          }
          
          double energyErrorTotal = streamSolution->energyErrorTotal();
          if (rank == 0) {
            cout << "...solved.\n";
            cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
          }
          
          reportStreamfunctionMaxValue(streamSolution,phi,vorticity,Re);
        }
        
        double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
        solnRieszRep.computeRieszRep();
        double solnEnergyNormTotal = solnRieszRep.getNorm();
        //    incrementRieszRep.computeRieszRep();
        //    double incrementEnergyNormTotal = incrementRieszRep.getNorm();
        
        double relativeEnergyError = incrementalEnergyErrorTotal / solnEnergyNormTotal;
        minL2Increment = initialMinL2Increment * relativeEnergyError;
        
        relativeEnergyErrors[refIndex] = relativeEnergyError;
        dofCounts[refIndex] = mesh->numGlobalDofs();
        fluxDofCounts[refIndex] = mesh->numFluxDofs();
        elementCounts[refIndex] = mesh->numActiveElements();
        
        iterationCounts[refIndex] = (refIndex==0) ? problem.iterationCount() : problem.iterationCount() - 1;
        tolerances[refIndex+1] = minL2Increment;
        
        for (map<double, vector< VerticalLineSolutionValues > >::iterator cutVectorsIt = verticalCutValues.begin();
             cutVectorsIt != verticalCutValues.end(); cutVectorsIt++) {
          double x = cutVectorsIt->first;
          VerticalLineSolutionValues values = computeVerticalLineSolutionValues(x, u1_prev, u2_prev, p_prev, vorticity);
          cutVectorsIt->second.push_back(values);
          VerticalLineSolutionValues valuesExhaustive = computeVerticalLineSolutionValuesExhaustive(x, u1_prev, u2_prev, p_prev, vorticity, sigma11_prev, sigma12_prev, sigma21_prev, sigma22_prev);
          verticalCutValuesExhaustive.push_back(valuesExhaustive);
        }
        
        // reset iteration count to 1 (for the background flow):
        problem.setIterationCount(1);
        // reset iteration count to 0 (to start from 0 initial guess):
  //      problem.setIterationCount(0);
        
  //      solveStrategy->solve(printToConsole);
        
        refinementStrategy->refine(false); //rank==0); // print to console on rank 0
        
        if (induceCornerRefinements) {
          // induce refinements in bottom corner:
          vector< Teuchos::RCP<Element> > corners = mesh->elementsForPoints(bottomCornerPoints);
          vector<int> cornerIDs;
          cornerIDs.push_back(corners[0]->cellID());
          mesh->hRefine(cornerIDs, RefinementPattern::regularRefinementPatternQuad());
        }
        
        if (saveFile.length() > 0) {
          if (rank == 0) {
            refHistory->saveToFile(saveFile);
          }
        }
        
        // find top corner cells:
        vector< Teuchos::RCP<Element> > topCorners = mesh->elementsForPoints(topCornerPoints);
        if (rank==0) {// print out top corner cellIDs
          cout << "Refinement # " << refIndex+1 << " complete.\n";
          vector<int> cornerIDs;
          cout << "top-left corner ID: " << topCorners[0]->cellID() << endl;
          cout << "top-right corner ID: " << topCorners[1]->cellID() << endl;
          cout << mesh->activeElements().size() << " elements, " << mesh->numGlobalDofs() << " dofs.\n";
        }
        
      }
      if ((solnFile.length() == 0) || (numRefs > 0)) {

        // one more solve on the final refined mesh:
        if (rank==0) cout << "Final solve:\n";
        if (startWithZeroSolutionAfterRefinement) {
          // start with a fresh (zero) initial guess for each adaptive mesh:
          solution->clear();
          problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
        }
        double incr_norm;
        do {
          problem.iterate(useLineSearch, useCondensedSolve);
          incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
          if (rank==0) {
            cout << "\x1B[2K"; // Erase the entire current line.
            cout << "\x1B[0E"; // Move to the beginning of the current line.
            cout << "Iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
            flush(cout);
          }
        } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
        if (rank==0) cout << endl;
        
        double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
        solnRieszRep.computeRieszRep();
        double solnEnergyNormTotal = solnRieszRep.getNorm();
        //    incrementRieszRep.computeRieszRep();
        //    double incrementEnergyNormTotal = incrementRieszRep.getNorm();
        
        double relativeEnergyError = incrementalEnergyErrorTotal / solnEnergyNormTotal;
        minL2Increment = initialMinL2Increment * relativeEnergyError;
        
        int refIndex = numRefs;
        relativeEnergyErrors[refIndex] = relativeEnergyError;
        dofCounts[refIndex] = mesh->numGlobalDofs();
        fluxDofCounts[refIndex] = mesh->numFluxDofs();
        elementCounts[refIndex] = mesh->numActiveElements();
        iterationCounts[refIndex] = (refIndex==0) ? problem.iterationCount() : problem.iterationCount() - 1;
        
        for (map<double, vector< VerticalLineSolutionValues > >::iterator cutVectorsIt = verticalCutValues.begin();
             cutVectorsIt != verticalCutValues.end(); cutVectorsIt++) {
          double x = cutVectorsIt->first;
          VerticalLineSolutionValues values = computeVerticalLineSolutionValues(x, u1_prev, u2_prev, p_prev, vorticity);
          cutVectorsIt->second.push_back(values);
          VerticalLineSolutionValues valuesExhaustive = computeVerticalLineSolutionValuesExhaustive(x, u1_prev, u2_prev, p_prev, vorticity, sigma11_prev, sigma12_prev, sigma21_prev, sigma22_prev);
          verticalCutValuesExhaustive.push_back(valuesExhaustive);
        }
      }
      
      if (computeMaxConditionNumber) {
        string fileName = "nsCavity_maxConditionIPMatrix.dat";
        IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(problem.solutionIncrement()->ip().get()), false );
        bool jacobiScalingTrue = true;
        double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, problem.solutionIncrement()->mesh(), jacobiScalingTrue, fileName);
        if (rank==0) {
          cout << "max Gram matrix condition number estimate: " << maxConditionNumber << endl;
          cout << "putative worst-conditioned Gram matrix written to: " << fileName << "." << endl;
        }
      }
      
      if (outputStiffnessMatrix) {
        if (rank==0) {
          cout << "performing one extra iteration and outputting its stiffness matrix to disk.\n";
        }
        problem.solutionIncrement()->setWriteMatrixToFile(true, "nsCavity_final_stiffness.dat");
        problem.iterate(useLineSearch, useCondensedSolve);
        double incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
        if (rank==0) {
          cout << "Final iteration, L^2(incr) = " << incr_norm << endl;
        }
      }
      
  //    if (enforceLocalConservationInFinalSolve && !enforceLocalConservation) {
  //      solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  //    }
  //    
  //    finalSolveStrategy->solve(printToConsole);
    }
  //  if (compareWithOverkillMesh) {
  //    Teuchos::RCP<Solution> projectedSoln = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
  //    solution->projectFieldVariablesOntoOtherSolution(projectedSoln);
  //    
  //    projectedSoln->addSolution(overkillSolution,-1.0);
  //    double L2errorSquared = 0.0;
  //    for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
  //      VarPtr var = *fieldIt;
  //      int fieldID = var->ID();
  //      double L2error = projectedSoln->L2NormOfSolutionGlobal(fieldID);
  //      if (rank==0)
  //        cout << "L2error for " << var->name() << ": " << L2error << endl;
  //      L2errorSquared += L2error * L2error;
  //    }
  //    int numGlobalDofs = mesh->numGlobalDofs();
  //    if (rank==0)
  //      cout << "for " << numGlobalDofs << " dofs, total L2 error: " << sqrt(L2errorSquared) << endl;
  //    dofsToL2error[numGlobalDofs] = sqrt(L2errorSquared);
  //  }
    
    double energyErrorTotal = solution->energyErrorTotal();
    double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
    if (rank == 0) {
      cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
      cout << "Final energy error: " << energyErrorTotal << endl;
      cout << "  (Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".)\n";
    }
    
    if (rank==0) {
      if (solnSaveFile.length() > 0) {
        solution->writeToFile(solnSaveFile);
      }
    }
    
    if (rank==0) {
      ofstream fout;
      ostringstream fileName;
      fileName << "cavityRefinements_Re" << Re << ".txt";

      fout.open(fileName.str().c_str());
      fout << "ref. #\trel. energy error\tL^2 tol.\tNR iterations\telements\tdofs\tflux dofs\n";
      for (int refIndex=0; refIndex<=numRefs; refIndex++) {
        fout << setprecision(8) << fixed;
        fout << refIndex << "\t";
        fout << setprecision(3) << scientific;;
        fout << "\t" << relativeEnergyErrors[refIndex];
        fout << "\t" << tolerances[refIndex];
        fout << "\t" << iterationCounts[refIndex];
        fout << "\t" << elementCounts[refIndex];
        fout << "\t" << dofCounts[refIndex];
        fout << "\t" << fluxDofCounts[refIndex];
        fout << endl;
      }
      fout.close();
      
      
      // add our "exhaustive" (that is, benchmark) data to the end of the verticalCutValues lists
      for (vector<VerticalLineSolutionValues>::iterator exhaustiveIt=verticalCutValuesExhaustive.begin();
           exhaustiveIt != verticalCutValuesExhaustive.end(); exhaustiveIt++) {
        double x = exhaustiveIt->x;
        verticalCutValues[x].push_back(*exhaustiveIt);
      }
      
      for (map<double, vector< VerticalLineSolutionValues > >::iterator cutVectorsIt = verticalCutValues.begin();
           cutVectorsIt != verticalCutValues.end(); cutVectorsIt++) {
        double x = cutVectorsIt->first;
        int listSize = cutVectorsIt->second.size();
        for (int refIndex=0; refIndex<listSize; refIndex++) {
          ostringstream fileName;
          int modifiedRefIndex = (refIndex > numRefs) ? refIndex - (numRefs + 1) : refIndex;
          if (refIndex <= numRefs)
            fileName << "cavityVerticalCutData_Re" << Re << "ref" << refIndex << ".txt";
          else {
            fileName << "cavityVerticalCutDataExhaustive_Re" << Re << "ref" << modifiedRefIndex << ".txt";
          }
          fout.open(fileName.str().c_str());
          fout << setprecision(15);
          vector< VerticalLineSolutionValues > valuesList = cutVectorsIt->second;
          VerticalLineSolutionValues values = valuesList[refIndex];
          if (values.sigma11.size() == 0)
            fout << "refno\txval\tyval\tu1\tu2\t|u|\tp\tomega\n";
          else
            fout << "refno\txval\tyval\tu1\tu2\t|u|\tp\tomega\tsigma11\tsigma12\tsigma21\tsigma22\n";
          int yCount = values.yPoints.size();
          for (int yIndex=0; yIndex<yCount; yIndex++) {
            double y = values.yPoints[yIndex];
            double u1 = values.u1[yIndex];
            double u2 = values.u2[yIndex];
            double u = sqrt(u1*u1 + u2*u2);
            double p = values.p[yIndex];
            double omega = values.omega[yIndex];
            
            fout << modifiedRefIndex << "\t" << x << "\t" << y;
            fout << "\t" << u1 << "\t" << u2 << "\t" << u;
            fout << "\t" << p << "\t" << omega;
            
            if (values.sigma11.size() > 0) {
              double sigma11 = values.sigma11[yIndex];
              double sigma12 = values.sigma12[yIndex];
              double sigma21 = values.sigma21[yIndex];
              double sigma22 = values.sigma22[yIndex];
              fout << "\t" << sigma11 << "\t" << sigma12;
              fout << "\t" << sigma21 << "\t" << sigma22;
            }
            fout << endl;
          }
          fout.close();
        }
      }
    }
    
    FunctionPtr u1_sq = u1_prev * u1_prev;
    FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
    FunctionPtr u_mag = Teuchos::rcp( new SqrtFunction( u_dot_u ) );
    FunctionPtr u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dx() + u2->dy() ) );
    FunctionPtr u_n = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat->times_normal_x() + u2hat->times_normal_y()) );
    
    // check that the zero mean pressure is being correctly imposed:
    double p_avg = p_prev->integrate(mesh);
    if (rank==0)
      cout << "Integral of pressure: " << p_avg << endl;

    FunctionPtr massFlux = Teuchos::rcp( new MassFluxFunction(u_n) );
    FunctionPtr absMassFlux = Teuchos::rcp( new MassFluxFunction(u_n,true) );
    
    double totalAbsMassFlux = absMassFlux->integrate(mesh,0,false,true);
    double totalMassFlux = massFlux->integrate(mesh,0,false,true);
    
    // examine cell sizes:
    double maxCellMeasure = 0;
    double minCellMeasure = 1;
    
    vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
    for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
      ElementTypePtr elemType = *elemTypeIt;
      vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
      vector<GlobalIndexType> cellIDs;
      for (int i=0; i<elems.size(); i++) {
        cellIDs.push_back(elems[i]->cellID());
      }
      FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
      BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh,polyOrder) ); // enrich by trial space order
      basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
      FieldContainer<double> cellMeasures = basisCache->getCellMeasures();

      for (int i=0; i<elems.size(); i++) {
        maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
        minCellMeasure = min(minCellMeasure,cellMeasures(i));
      }
    }
    
    if (rank==0) {
      cout << "total mass flux: " << totalMassFlux << endl;
      cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
      cout << "largest h: " << sqrt(maxCellMeasure) << endl;
      cout << "smallest h: " << sqrt(minCellMeasure) << endl;
      cout << "ratio of largest / smallest h: " << sqrt(maxCellMeasure) / sqrt(minCellMeasure) << endl;
    }
    
    if (reportCenterlineVelocities) {
      double u1_max_y, u2_max_x, u2_min_x; // the maxes and mins we find along the reported centerline points
      reportCenterlineVelocityValues(u1_prev,u2_prev,p_prev,vorticity, u1_max_y, u2_max_x, u2_min_x);
      reportVelocityExtrema(u1_prev, sigma12_prev, sigma12_dy,
                            u2_prev, sigma21_prev, sigma21_dx,
                            u1_max_y, u2_max_x, u2_min_x);
    }
    
    if (rank == 0) {
      cout << "phi ID: " << phi->ID() << endl;
      cout << "psi1 ID: " << psi_1->ID() << endl;
      cout << "psi2 ID: " << psi_2->ID() << endl;
      
      cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
      cout << "solving for approximate stream function...\n";
    }
    //  mesh->unregisterObserver(streamMesh);
    //  streamMesh->registerObserver(mesh);
    //  RefinementStrategy streamRefinementStrategy( streamSolution, energyThreshold );
    //  for (int refIndex=0; refIndex < 3; refIndex++) {
    //    streamSolution->solve(false);
    //    streamRefinementStrategy.refine(rank==0);
    //  }
    
    if (!useCondensedSolve){
      streamSolution->solve(useMumps);
    } else {
      streamSolution->condensedSolve();
    }
    energyErrorTotal = streamSolution->energyErrorTotal();
    if (rank == 0) {  
      cout << "...solved.\n";
      cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
    }
  
    if (reportStreamfunctionMax) {
      reportStreamfunctionMaxValue(streamSolution,phi,vorticity,Re);
    }

    if (rank==0){
      solution->writeToVTK("nsCavitySoln.vtk");
      if (! meshHasTriangles ) {
//        massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
//        u_mag->writeValuesToMATLABFile(solution->mesh(), "u_mag.m");
//        u_div->writeValuesToMATLABFile(solution->mesh(), "u_div.m");
//        solution->writeFieldsToFile(u1->ID(), "u1.m");
//        solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
//        solution->writeFieldsToFile(u2->ID(), "u2.m");
//        solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
//        solution->writeFieldsToFile(p->ID(), "p.m");
//        streamSolution->writeFieldsToFile(phi->ID(), "phi.m");
//        
//        streamSolution->writeFluxesToFile(phi_hat->ID(), "phi_hat.dat");
//        streamSolution->writeFieldsToFile(psi_1->ID(), "psi1.m");
//        streamSolution->writeFieldsToFile(psi_2->ID(), "psi2.m");
//        vorticity->writeValuesToMATLABFile(streamMesh, "vorticity.m");
//        
//        FunctionPtr ten = Teuchos::rcp( new ConstantScalarFunction(10) );
//        ten->writeBoundaryValuesToMATLABFile(solution->mesh(), "skeleton.dat");
//        cout << "wrote files: u_mag.m, u_div.m, u1.m, u1_hat.dat, u2.m, u2_hat.dat, p.m, phi.m, vorticity.m.\n";
      } else {
//        solution->writeToFile(u1->ID(), "u1.dat");
//        solution->writeToFile(u2->ID(), "u2.dat");
//        solution->writeToFile(u2->ID(), "p.dat");
//        cout << "wrote files: u1.dat, u2.dat, p.dat\n";
      }
//      polyOrderFunction->writeValuesToMATLABFile(mesh, "cavityFlowPolyOrders.m");
      
      FieldContainer<double> points = pointGrid(0, 1, 0, 1, 100);
      FieldContainer<double> pointData = solutionData(points, streamSolution, phi);
      GnuPlotUtil::writeXYPoints("phi_patch_navierStokes_cavity.dat", pointData);
      set<double> patchContourLevels = diagonalContourLevels(pointData,1);
      vector<string> patchDataPath;
      patchDataPath.push_back("phi_patch_navierStokes_cavity.dat");
      GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "lidCavityNavierStokes.p");
      GnuPlotUtil::writeComputationalMeshSkeleton("nsCavityMesh", mesh);

//      writePatchValues(0, 1, 0, 1, streamSolution, phi, "phi_patch.m");
//      writePatchValues(0, .1, 0, .1, streamSolution, phi, "phi_patch_detail.m");
//      writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
//      writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
    }
    
    if (compareWithOverkillMesh) {
      cout << "******* Adaptivity Convergence Report *******\n";
      cout << "dofs\tL2 error\n";
      for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
        int dofs = entryIt->first;
        double err = entryIt->second;
        cout << dofs << "\t" << err << endl;
      }
      ofstream fout("overkillComparison.txt");
      fout << "******* Adaptivity Convergence Report *******\n";
      fout << "dofs\tL2 error\n";
      for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
        int dofs = entryIt->first;
        double err = entryIt->second;
        fout << dofs << "\t" << err << endl;
      }
      fout.close();
    }
    
  } catch ( choice::ArgException& e )
  {
    // There is no reason to do anything
  }
  
  return 0;
}
