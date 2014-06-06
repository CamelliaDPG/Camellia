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
  static FieldContainer<double> cellCentroids(MeshTopology* meshTopo);
  static FieldContainer<double> cellCentroids(MeshPtr mesh);
  
public:
  static void writeComputationalMeshSkeleton(const string &filePath, MeshPtr mesh, bool labelCells = false, string rgbColor = "red", string title = "mesh");

  static void writeExactMeshSkeleton(const string &filePath, MeshTopology* meshTopo, int numPointsPerEdge, bool labelCells=false, string rgbColor = "red", string title = "mesh");
  
  static void writeExactMeshSkeleton(const string &filePath, MeshPtr mesh, int numPointsPerEdge, bool labelCells=false, string rgbColor = "red", string title = "mesh");
  
  // badly named, maybe: we support arbitrary space dimension...
  static void writeXYPoints(const string &filePath, const FieldContainer<double> &dataPoints);
  
  static void writeContourPlotScript(set<double> contourLevels, const vector<string> &filePathsOfData,
                                     const string &outputFile, double xTics=-1, double yTics=-1);
};

#endif
