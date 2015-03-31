//
//  GnuPlotUtil.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 6/6/14.
//
//

#include "GnuPlotUtil.h"

#include "BasisCache.h"

using namespace Intrepid;
using namespace Camellia;

FieldContainer<double> GnuPlotUtil::cellCentroids(MeshTopology* meshTopo) {
  // this only works on quads right now
  
  int spaceDim = meshTopo->getSpaceDim(); // not that this will really work in 3D...
  int numActiveElements = meshTopo->activeCellCount();
  
  FieldContainer<double> cellCentroids(numActiveElements,spaceDim); // used for labelling cells
  
  set<IndexType> cellIDset = meshTopo->getActiveCellIndices();
  vector<GlobalIndexType> cellIDs(cellIDset.begin(),cellIDset.end());
  
  for (int cellIndex=0; cellIndex<numActiveElements; cellIndex++) {
    vector<double> cellCentroid = meshTopo->getCellCentroid(cellIDs[cellIndex]);
    
    for (int i=0; i<spaceDim; i++) {
      cellCentroids(cellIndex,i) = cellCentroid[i];
    }
  }
  return cellCentroids;
}

FieldContainer<double> GnuPlotUtil::cellCentroids(MeshPtr mesh) {
  // this only works on quads right now
  
  int spaceDim = mesh->getDimension(); // not that this will really work in 3D...
  int numActiveElements = mesh->numActiveElements();
  
  FieldContainer<double> cellCentroids(numActiveElements,spaceDim); // used for labelling cells
  
  set<GlobalIndexType> cellIDset = mesh->getActiveCellIDs();
  vector<GlobalIndexType> cellIDs(cellIDset.begin(),cellIDset.end());
  
  for (int cellOrdinal=0; cellOrdinal<numActiveElements; cellOrdinal++) {
    vector<double> centroid = mesh->getTopology()->getCellCentroid(cellIDs[cellOrdinal]);
    for (int d=0; d<spaceDim; d++) {
      cellCentroids(cellOrdinal,d) = centroid[d];
    }
  }
  return cellCentroids;
}

void GnuPlotUtil::writeComputationalMeshSkeleton(const string &filePath, MeshPtr mesh, bool labelCells, string rgbColor, string title) {
  FunctionPtr transformationFunction = mesh->getTransformationFunction();
  if (transformationFunction.get()==NULL) {
    // then the computational and exact meshes are the same: call the other method:
    writeExactMeshSkeleton(filePath,mesh,2,labelCells,rgbColor,title);
    return;
  }
  
  int spaceDim = mesh->getDimension(); // not that this will really work in 3D...
  
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  
  fout << "# Camellia GnuPlotUtil Mesh Points\n";
  fout << "# x                  y\n";
  
  int numActiveElements = mesh->numActiveElements();
  
  double minX = 1e10, minY = 1e10, maxX = -1e10, maxY = -1e10;
  
  FieldContainer<double> cellCentroids;
  set<GlobalIndexType> cellIDset = mesh->getActiveCellIDs();
  vector<GlobalIndexType> cellIDs(cellIDset.begin(),cellIDset.end());
  
  for (int cellIndex=0; cellIndex<numActiveElements; cellIndex++) {
    ElementPtr cell = mesh->getElement(cellIDs[cellIndex]);
    vector< ParametricCurvePtr > edgeLines = ParametricCurve::referenceCellEdges(cell->elementType()->cellTopoPtr->getKey());
    int numEdges = edgeLines.size();
    int numPointsPerEdge = max(10,cell->elementType()->testOrderPtr->maxBasisDegree() * 2); // 2 points for linear, 4 for quadratic, etc.
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
    
    if (labelCells) {
      // this only works on quads right now
      cellCentroids = GnuPlotUtil::cellCentroids(mesh);
    }
  }
  
  double xDiff = maxX - minX;
  double yDiff = maxY - minY;
  
  fout << "# Plot with:\n";
  fout << setprecision(2);
  fout << "# set size ratio -1\n";
  fout << "# set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
  fout << "# set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
  fout << "# plot \"" << filePath << "\" using 1:2 title 'mesh' with lines lc rgb \"" << rgbColor << "\"\n";
  if (labelCells) {
    for (int i=0; i<cellIDs.size(); i++) {
      int cellID = cellIDs[i];
      fout << "set label \"" << cellID << "\" at " << cellCentroids(i,0) << ",";
      fout << cellCentroids(i,1) << " center " << endl;
    }
  }
  fout << "# set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
  fout << "# set out '" << filePath << ".eps'\n";
  fout << "# replot\n";
  fout << "# set term pop\n";
  fout.close();
  
  ofstream scriptOut((filePath + ".p").c_str());
  scriptOut << "set size ratio -1\n";
  scriptOut << "set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
  scriptOut << "set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
  scriptOut << "plot \"" << filePath << "\" using 1:2 title 'mesh' with lines lc rgb \"" << rgbColor << "\"\n";
  if (labelCells) {
    for (int i=0; i<cellIDs.size(); i++) {
      int cellID = cellIDs[i];
      scriptOut << "set label \"" << cellID << "\" at " << cellCentroids(i,0) << ",";
      scriptOut << cellCentroids(i,1) << " center " << endl;
    }
  }
  scriptOut << "set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
  scriptOut << "set out '" << filePath << ".eps'\n";
  //    scriptOut << "replot\n";
  //    scriptOut << "set terminal png\n";
  //    scriptOut << "set out '" << filePath << ".png'\n";
  scriptOut << "replot\n";
  scriptOut << "set term pop\n";
  scriptOut << "replot\n";
  scriptOut.close();
}

void GnuPlotUtil::writeExactMeshSkeleton(const string &filePath, MeshTopology* meshTopo, int numPointsPerEdge,
                                         bool labelCells, string rgbColor, string title) {
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  
  fout << "# Camellia GnuPlotUtil Mesh Points\n";
  fout << "# x                  y\n";
  
  int numActiveElements = meshTopo->activeCellCount();
  FieldContainer<double> cellCentroids;
  if (labelCells) {
    cellCentroids = GnuPlotUtil::cellCentroids(meshTopo);
  }
  
  double minX = 1e10, minY = 1e10, maxX = -1e10, maxY = -1e10;
  
  set<IndexType> cellIDset = meshTopo->getActiveCellIndices();
  vector<GlobalIndexType> cellIDs(cellIDset.begin(),cellIDset.end());
  
  for (int cellIndex=0; cellIndex<numActiveElements; cellIndex++) {
    CellPtr cell = meshTopo->getCell(cellIDs[cellIndex]);
    cellIDs.push_back(cell->cellIndex());
    vector< ParametricCurvePtr > edgeCurves = meshTopo->parametricEdgesForCell(cell->cellIndex(), false);
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
  fout << "# plot \"" << filePath << "\" using 1:2 title '" << title << "' with lines lc rgb \"" << rgbColor << "\"\n";
  if (labelCells) {
    for (int i=0; i<numActiveElements; i++) {
      int cellID = cellIDs[i];
      fout << "# set label \"" << cellID << "\" at " << cellCentroids(i,0) << ",";
      fout << cellCentroids(i,1) << " center " << endl;
    }
  }
  fout.close();
  
  ofstream scriptOut((filePath + ".p").c_str());
  scriptOut << "set size ratio -1\n";
  scriptOut << "set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
  scriptOut << "set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
  scriptOut << "plot \"" << filePath << "\" using 1:2 title '" << title << "' with lines lc rgb \"" << rgbColor << "\"\n";
  if (labelCells) {
    for (int i=0; i<numActiveElements; i++) {
      int cellID = cellIDs[i];
      scriptOut << "set label \"" << cellID << "\" at " << cellCentroids(i,0) << ",";
      scriptOut << cellCentroids(i,1) << " center " << endl;
    }
  }
  scriptOut << "set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
  scriptOut << "set out '" << filePath << ".eps'\n";
  //    scriptOut << "replot\n";
  //    scriptOut << "set terminal png\n";
  //    scriptOut << "set out '" << filePath << ".png'\n";
  scriptOut << "replot\n";
  scriptOut << "set term pop\n";
  scriptOut << "replot\n";
  scriptOut.close();
}

void GnuPlotUtil::writeExactMeshSkeleton(const string &filePath, MeshPtr mesh, int numPointsPerEdge, bool labelCells, string rgbColor, string title) {
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  
  fout << "# Camellia GnuPlotUtil Mesh Points\n";
  fout << "# x                  y\n";
  
  int numActiveElements = mesh->numActiveElements();
  FieldContainer<double> cellCentroids;
  if (labelCells) {
    cellCentroids = GnuPlotUtil::cellCentroids(mesh);
  }
  
  double minX = 1e10, minY = 1e10, maxX = -1e10, maxY = -1e10;
  
  set<GlobalIndexType> cellIDset = mesh->getActiveCellIDs();
  vector<GlobalIndexType> cellIDs(cellIDset.begin(),cellIDset.end());
  
  for (int cellIndex=0; cellIndex<numActiveElements; cellIndex++) {
    ElementPtr cell = mesh->getElement(cellIDs[cellIndex]);
    cellIDs.push_back(cell->cellID());
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
  fout << "# plot \"" << filePath << "\" using 1:2 title '" << title << "' with lines lc rgb \"" << rgbColor << "\"\n";
  if (labelCells) {
    for (int i=0; i<numActiveElements; i++) {
      int cellID = cellIDs[i];
      fout << "# set label \"" << cellID << "\" at " << cellCentroids(i,0) << ",";
      fout << cellCentroids(i,1) << " center " << endl;
    }
  }
  fout.close();
  
  ofstream scriptOut((filePath + ".p").c_str());
  scriptOut << "set size ratio -1\n";
  scriptOut << "set xrange [" << minX- 0.1*xDiff << ":" << maxX+0.1*xDiff << "] \n";
  scriptOut << "set yrange [" << minY- 0.1*yDiff << ":" << maxY+0.1*yDiff << "] \n";
  scriptOut << "plot \"" << filePath << "\" using 1:2 title '" << title << "' with lines lc rgb \"" << rgbColor << "\"\n";
  if (labelCells) {
    for (int i=0; i<numActiveElements; i++) {
      int cellID = cellIDs[i];
      scriptOut << "set label \"" << cellID << "\" at " << cellCentroids(i,0) << ",";
      scriptOut << cellCentroids(i,1) << " center " << endl;
    }
  }
  scriptOut << "set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
  scriptOut << "set out '" << filePath << ".eps'\n";
  //    scriptOut << "replot\n";
  //    scriptOut << "set terminal png\n";
  //    scriptOut << "set out '" << filePath << ".png'\n";
  scriptOut << "replot\n";
  scriptOut << "set term pop\n";
  scriptOut << "replot\n";
  scriptOut.close();
}

// badly named, maybe: we support arbitrary space dimension...
void GnuPlotUtil::writeXYPoints(const string &filePath, const FieldContainer<double> &dataPoints) {
  FieldContainer<double> dataPointsCopy = dataPoints;
  
  if (dataPoints.rank()==3) {
    int numCells = dataPoints.dimension(0);
    int numPoints = dataPoints.dimension(1);
    int spaceDim = dataPoints.dimension(2);
    dataPointsCopy.resize(numCells*numPoints, spaceDim);
  }
  
  if (dataPointsCopy.rank() != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"writeXYPoints only supports containers of rank 2 or 3");
  }
  
  int numPoints = dataPointsCopy.dimension(0);
  int spaceDim = dataPointsCopy.dimension(1);
  
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  
  fout << "# Camellia GnuPlotUtil output\n";
  if (spaceDim==2) {
    fout << "# x                  y\n";
  } else if (spaceDim==3) {
    fout << "# x                  y                  z\n";
  }
  
  for (int i=0; i<numPoints; i++) {
    if ((spaceDim==3) && (i>0)) {
      // for 3D point sets, separate new x values with a new line.
      if (dataPointsCopy(i,0) != dataPointsCopy(i-1,0)) {
        fout << "\n";
      }
    }
    for (int d=0; d<spaceDim; d++) {
      fout << dataPointsCopy(i,d) << "   ";
    }
    fout << "\n";
  }
  
  fout.close();
}

void GnuPlotUtil::writeContourPlotScript(set<double> contourLevels, const vector<string> &filePathsOfData,
                                         const string &outputFile, double xTics, double yTics) {
  ofstream fout((outputFile).c_str());
  ostringstream splotLine;
  splotLine << "splot ";
  for (int i=0; i<filePathsOfData.size(); i++) {
    string filePath = filePathsOfData[i];
    splotLine << "\"" << filePath << "\" using 1:2:3 notitle with lines lc rgb \"#0000ff\"";
    if (i != filePathsOfData.size()-1) {
      splotLine << ", ";
    }
  }
  splotLine << "\n";
  int levelNumber = 0;
  ostringstream levelsString;
  for (set<double>::iterator levelIt = contourLevels.begin(); levelIt != contourLevels.end(); levelIt++, levelNumber++) {
    levelsString << *levelIt;
    if (levelNumber != contourLevels.size() - 1) {
      levelsString << ", ";
    }
  }
  fout << splotLine.str() << endl;
  fout << "unset surface" << endl;
  fout << "set cntr levels discrete " << levelsString.str() << endl;
  fout << "set contour base" << endl;
  fout << "set view map" << endl;
  fout << "unset clabel" << endl;
  fout << "set cntrparam bspline" << endl;
  fout << "set size ratio -1" << endl;
  if (xTics > 0) {
    fout << "set xtics " << xTics << endl;
  }
  if (yTics > 0) {
    fout << "set ytics" << yTics << endl;
  }
  fout << "set style data lines" << endl;
  fout << "set terminal postscript eps color lw 1 \"Helvetica\" 20\n";
  fout << "set out '" << outputFile << ".eps'\n";
  //    fout << "replot" << endl;
  //    fout << "set terminal png\n";
  //    fout << "set out '" << outputFile << ".png'\n";
  fout << "replot\n";
  fout << "set term pop\n";
  fout << "replot" << endl;
  
  fout.close();
}
