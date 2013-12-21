//
//  StreamDriverUtil.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/26/13.
//
//

#ifndef Camellia_debug_StreamDriverUtil_h
#define Camellia_debug_StreamDriverUtil_h

#include "ParametricCurve.h"

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

set<double> diagonalContourLevels(FieldContainer<double> &pointData, int pointsPerLevel=1) {
  // traverse diagonal of (i*numPoints + j) data from solutionData()
  int numPoints = sqrt(pointData.dimension(0));
  set<double> levels;
  for (int i=0; i<numPoints; i+=pointsPerLevel) {
    levels.insert(pointData(i*numPoints + i,2)); // format for pointData has values at (ptIndex, 2)
  }
  return levels;
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

void writePatchValues(double xMin, double xMax, double yMin, double yMax,
                      SolutionPtr solution, VarPtr u1, string filename) {
  vector<double> points1D_x, points1D_y;
  int numPoints = 100;
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
  FieldContainer<double> values1(numPoints*numPoints);
  FieldContainer<double> values2(numPoints*numPoints);
  solution->solutionValues(values1, u1->ID(), points);
  ofstream fout(filename.c_str());
  fout << setprecision(15);
  
  fout << "X = zeros(" << numPoints << ",1);\n";
  //    fout << "Y = zeros(numPoints);\n";
  fout << "U = zeros(" << numPoints << "," << numPoints << ");\n";
  for (int i=0; i<numPoints; i++) {
    fout << "X(" << i+1 << ")=" << points1D_x[i] << ";\n";
  }
  for (int i=0; i<numPoints; i++) {
    fout << "Y(" << i+1 << ")=" << points1D_y[i] << ";\n";
  }
  
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      fout << "U("<<i+1<<","<<j+1<<")=" << values1(pointIndex) << ";" << endl;
    }
  }
  fout.close();
}

double getSolutionValueAtPoint(double x, double y, SolutionPtr soln, VarPtr var) {
  double spaceDim = 2;
  FieldContainer<double> point(1,spaceDim);
  FieldContainer<double> value(1); // one value
  point(0,0) = x;
  point(0,1) = y;
  soln->solutionValues(value, var->ID(), point);
  return value[0];
}

double getFunctionValueAtPoint(FunctionPtr scalarFunction, double x, double y, MeshPtr mesh) {
  static FieldContainer<double> value(1,1);
  static FieldContainer<double> physPoint(1,2);
  physPoint[0] = x;
  physPoint[1] = y;
  
  ElementPtr elem = mesh->elementsForPoints(physPoint)[0];
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, elem->cellID());
  FieldContainer<double> refPoint = basisCache->getRefCellPointsForPhysicalPoints(physPoint);
  basisCache->setRefCellPoints(refPoint);
  
  if (scalarFunction->rank() != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Function::evaluate requires a rank 1 Function.");
  }
  scalarFunction->values(value,basisCache);
  return value[0];
}

double findSignReversal(ParametricCurvePtr arc, double tGuessLeft, double tGuessRight, FunctionPtr scalarFunction, MeshPtr mesh) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  double xLeft, yLeft, xRight, yRight;
  
  arc->value(tGuessLeft, xLeft, yLeft);
  arc->value(tGuessRight, xRight, yRight);
  
  double leftValue = getFunctionValueAtPoint(scalarFunction, xLeft, yLeft, mesh); // getSolutionValueAtPoint(xLeft, yLeft, soln, u1);
  double rightValue = getFunctionValueAtPoint(scalarFunction, xRight, yRight, mesh); // getSolutionValueAtPoint(xRight, yRight, soln, u1);
  if (leftValue * rightValue > 0) {
    double tGuess = (tGuessLeft + tGuessRight) / 2;
    if (rank==0) {
      string fxnName = scalarFunction->displayString();
      cout << "Error: " << fxnName << " at t=" << tGuessLeft << " = " << leftValue << " and " << fxnName << " at t=" << tGuessRight << ") = " << rightValue;
      cout << " have the same sign.  Returning " << -tGuess << endl;
    }
    return -tGuess;
  }
  int numIterations = 30;
  double t = 0;
  for (int i=0; i<numIterations; i++) {
    double tGuess = (tGuessLeft + tGuessRight) / 2;
    double x,y;
    arc->value(tGuess,x,y);
    double middleValue = getFunctionValueAtPoint(scalarFunction, x, y, mesh); // getSolutionValueAtPoint(x, y, soln, u1);
    if (middleValue * leftValue > 0) { // same sign
      tGuessLeft = tGuess;
      leftValue = middleValue;
    }
    if (middleValue * rightValue > 0) {
      tGuessRight = tGuess;
      rightValue = middleValue;
    }
    t = tGuess;
  }
  return t;
}

#endif
