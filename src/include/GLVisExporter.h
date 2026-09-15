#ifndef GLVISEXPORTER_H
#define GLVISEXPORTER_H

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

/*
 *  GLVisExporter.h
 *
 *  Writes MFEM/GLVis ASCII files (.mesh + .gf) for GLVis.
 *  Version 1 supports straight-sided segment, quadrilateral, and hexahedral
 *  meshes.  Field variables are exported on the volume mesh; traces and fluxes
 *  are exported on a separate lower-dimensional side mesh.  Curvilinear geometry,
 *  simplices, and 4D/time-slab export are intentionally out of scope.
 */

#include "TypeDefs.h"

#include "Solution.h"
#include "Mesh.h"
#include "MeshTopology.h"
#include "VarFactory.h"

#include <set>
#include <string>
#include <vector>

namespace Camellia
{
class GLVisExporter
{
private:
  std::string _dirName;
  std::string _dirSuperPath;
  MeshPtr _mesh;
  bool _gatherToRankZero = false;
  int _exportStep = 0;
  int _minDegree = -1;

  void getGaussLobattoPointsFC(Camellia::FieldContainer<double> &points, int spaceDim, int degree);
  void writeExport(const std::vector<TFunctionPtr<double>> &functions, const std::vector<std::string> &functionNames,
                   int localDegree, double timeVal, const std::set<GlobalIndexType> &cellIndices);
  void writeBoundaryExport(const std::vector<TFunctionPtr<double>> &functions, const std::vector<std::string> &functionNames,
                           int localDegree, double timeVal, const std::set<GlobalIndexType> &cellIndices);
public:
  // By default, protects existing output by writing to outputDirName-2,
  // outputDirName-3, ... when outputDirName already exists.  Set
  // allowOverwrite=true to reuse the requested directory and overwrite step files.
  GLVisExporter(MeshPtr mesh, std::string outputDirName="output", std::string outputDirSuperPath = ".", bool allowOverwrite=false);
  ~GLVisExporter();
  void setMesh(MeshPtr mesh)
  {
    _mesh = mesh;
  }
  std::string outputDirName() const
  {
    return _dirName;
  }
  std::string outputDirPath() const;

  // no num1DPts arguments: display subdivision is a GLVis viewer setting, and the export
  // degree is inferred from the mesh and variables (setMinimumDegree can raise it)
  void exportFunction(TFunctionPtr<double> function, std::string functionName="function", double timeVal=0,
                      std::set<GlobalIndexType> cellIndices=std::set<GlobalIndexType>());
  void exportFunction(std::vector<TFunctionPtr<double>> functions, std::vector<std::string> functionNames, double timeVal=0,
                      std::set<GlobalIndexType> cellIndices=std::set<GlobalIndexType>());
  void exportSolution(TSolutionPtr<double> solution, double timeVal=0,
                      std::set<GlobalIndexType> cellIndices=std::set<GlobalIndexType>());
  void exportTimeSlab(TFunctionPtr<double> function, std::string functionName="function", double tInit=0, double tFinal=1,
                      unsigned int numSlices=2, unsigned int sliceH1Order=2);

  // raises the exported polynomial degree; never lowers it
  void setMinimumDegree(int degree);
  // rank 0 writes one file per step, byte-stable across rank counts;
  // default is one piece file per rank, viewed with glvis -np
  void setGatherToRankZero(bool value);

  // allows one-line export without storing an exporter object
  static void exportFunction(std::string directoryPath, std::string functionName, TFunctionPtr<double> function, MeshPtr mesh);
  // allows one-line export without storing an exporter object
  static void exportSolution(std::string directoryPath, std::string solutionName, TSolutionPtr<double> solution);
};
}

#endif
