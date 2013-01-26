//
//  GnuPlotUtil.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 1/22/13.
//
//

#ifndef Camellia_debug_GnuPlotUtil_h
#define Camellia_debug_GnuPlotUtil_h

#include <fstream>
#include "Mesh.h"
#include "ParametricCurve.h"

class GnuPlotUtil {
public:
  static void writeComputationalMeshSkeleton(const string &filePath, MeshPtr mesh) {
    FunctionPtr transformationFunction = mesh->getTransformationFunction();
    if (transformationFunction.get()==NULL) {
      // then the computational and exact meshes are the same: call the other method:
      writeExactMeshSkeleton(filePath,mesh,2);
    }
    
    int spaceDim = mesh->getDimension(); // not that this will really work in 3D...
    
    ofstream fout(filePath.c_str());
    fout << setprecision(15);
    
    fout << "# Camellia GnuPlotUtil Mesh Points\n";
    fout << "# x                  y\n";
    
    int numActiveElements = mesh->numActiveElements();
    
    double minX = 1e10, minY = 1e10, maxX = -1e10, maxY = -1e10;
    
    for (int cellIndex=0; cellIndex<numActiveElements; cellIndex++) {
      ElementPtr cell = mesh->getActiveElement(cellIndex);
      bool neglectCurves = true;
      vector< ParametricCurvePtr > edgeLines = ParametricCurve::referenceCellEdges(cell->elementType()->cellTopoPtr->getKey());
      int numEdges = edgeLines.size();
      int numPointsPerEdge = cell->elementType()->testOrderPtr->maxBasisDegree() * 2; // 2 points for linear, 4 for quadratic, etc.
      // to start, compute edgePoints on the reference cell
      int numPointsTotal = numEdges*(numPointsPerEdge-1)+1; // -1 because edges share vertices, +1 because we repeat first vertex...
      FieldContainer<double> edgePoints(numPointsTotal,spaceDim);
      
      int ptIndex = 0;
      for (int edgeIndex=0; edgeIndex < edgeLines.size(); edgeIndex++) {
        ParametricCurvePtr edge = edgeLines[edgeIndex];
        double t = 0;
        double increment = 1.0 / (numPointsPerEdge - 1);
        // last edge gets one extra point (to connect to first edge):
        int thisEdgePoints = (edgeIndex < edgeLines.size()-1) ? numPointsPerEdge-1 : numPointsPerEdge;
        for (int i=0; i<thisEdgePoints; i++) {
          double x, y;
          edge->value(t,x,y);
          edgePoints(ptIndex,0) = x;
          edgePoints(ptIndex,1) = y;
          ptIndex++;
          t += increment;
        }
      }
      // make a one-cell BasisCache initialized with the edgePoints on the ref cell:
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cell->cellID());
      basisCache->setRefCellPoints(edgePoints);
      
//      cout << "--- cellID " << cell->cellID() << " ---\n";
//      cout << "edgePoints:\n" << edgePoints;
      
      FieldContainer<double> transformedPoints(1,numPointsTotal,spaceDim);
      // compute the transformed points:
      transformationFunction->values(transformedPoints,basisCache);
      
//      cout << "transformedPoints:\n" << transformedPoints;

      ptIndex = 0;
      for (int i=0; i<numEdges; i++) {
        int thisEdgePoints = (i < edgeLines.size()-1) ? numPointsPerEdge-1 : numPointsPerEdge;
        for (int j=0; j<thisEdgePoints; j++) {
          double x = transformedPoints(0,ptIndex,0);
          double y = transformedPoints(0,ptIndex,1);
          fout << x << "   " << y << endl;
          ptIndex++;
          minX = min(x,minX);
          minY = min(y,minY);
          maxX = max(x,maxX);
          maxY = max(y,maxY);
        }
      }
      
      fout << endl; // line break to separate elements
    }
    
    double xDiff = maxX - minX;
    double yDiff = maxY - minY;
    
    fout << "# Plot with:\n";
    fout << setprecision(2);
    fout << "# set size square\n";
    fout << "# set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
    fout << "# set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
    fout << "# plot \"" << filePath << "\" using 1:2 title 'mesh' with lines\n";
    fout << "# set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
    fout << "# set out '" << filePath << ".eps'\n";
    fout << "# replot\n";
    fout << "# set term pop\n";
    fout.close();
    
    ofstream scriptOut((filePath + ".p").c_str());
    scriptOut << "set size square\n";
    scriptOut << "set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
    scriptOut << "set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
    scriptOut << "plot \"" << filePath << "\" using 1:2 title 'mesh' with lines\n";
    scriptOut << "set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
    scriptOut << "set out '" << filePath << ".eps'\n";
    scriptOut << "replot\n";
    scriptOut << "set term pop\n";
    scriptOut.close();
  }
  
  static void writeExactMeshSkeleton(const string &filePath, MeshPtr mesh, int numPointsPerEdge) {
    int spaceDim = mesh->getDimension(); // not that this will really work in 3D...
    
    ofstream fout(filePath.c_str());
    fout << setprecision(15);
    
    fout << "# Camellia GnuPlotUtil Mesh Points\n";
    fout << "# x                  y\n";
    
    int numActiveElements = mesh->numActiveElements();
    
    double minX = 1e10, minY = 1e10, maxX = -1e10, maxY = -1e10;
    
    for (int cellIndex=0; cellIndex<numActiveElements; cellIndex++) {
      ElementPtr cell = mesh->getActiveElement(cellIndex);
      vector< ParametricCurvePtr > edgeCurves = mesh->parametricEdgesForCell(cell->cellID());
      for (int edgeIndex=0; edgeIndex < edgeCurves.size(); edgeIndex++) {
        ParametricCurvePtr edge = edgeCurves[edgeIndex];
        double t = 0;
        double increment = 1.0 / (numPointsPerEdge - 1);
        // last edge gets one extra point (to connect to first edge):
        int thisEdgePoints = (edgeIndex < edgeCurves.size()-1) ? numPointsPerEdge-1 : numPointsPerEdge;
        for (int i=0; i<thisEdgePoints; i++) {
          double x, y;
          edge->value(t,x,y);
          fout << x << "   " << y << endl;
          t += increment;
          minX = min(x,minX);
          minY = min(y,minY);
          maxX = max(x,maxX);
          maxY = max(y,maxY);
        }
      }
      fout << endl; // line break to separate elements
    }
    
    double xDiff = maxX - minX;
    double yDiff = maxY - minY;
    
    fout << "# Plot with:\n";
    fout << setprecision(2);
    fout << "# set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
    fout << "# set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
    fout << "# plot \"" << filePath << "\" using 1:2 title 'mesh' with lines\n";
    fout.close();
    
    ofstream scriptOut((filePath + ".p").c_str());
    scriptOut << "set size square\n";
    scriptOut << "set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
    scriptOut << "set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
    scriptOut << "plot \"" << filePath << "\" using 1:2 title 'mesh' with lines\n";
    scriptOut << "set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
    scriptOut << "set out '" << filePath << ".eps'\n";
    scriptOut << "replot\n";
    scriptOut << "set term pop\n";
    scriptOut.close();
  }
  
  static void writeXYPoints(const string &filePath, FieldContainer<double> &dataPoints) {
    int numPoints = dataPoints.dimension(0);
    int spaceDim = dataPoints.dimension(1);
    
    ofstream fout(filePath.c_str());
    fout << setprecision(15);
    
    fout << "# Camellia GnuPlotUtil output\n";
    fout << "# x                  y\n";
    
    for (int i=0; i<numPoints; i++) {
      for (int d=0; d<spaceDim; d++) {
        fout << dataPoints(i,d) << "   ";
      }
      fout << "\n";
    }
    
    fout.close();
  }
};

#endif
