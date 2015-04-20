#ifndef HDF5EXPORTER_H
#define HDF5EXPORTER_H

/*
 *  HDF5Exporter.h
 *
 *  Created by Truman Ellis on 6/24/2014.
 *
 */

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5

#include "TypeDefs.h"

#include "Solution.h"
#include "Mesh.h"
#include "MeshTopology.h"
#include "VarFactory.h"

#include "Teuchos_XMLObject.hpp"

#include <string>
#include <map>
#include <vector>

namespace Camellia {
  class HDF5Exporter {
  private:
    std::string _dirName;
    std::string _dirSuperPath;
    MeshPtr _mesh;
    Teuchos::XMLObject _fieldXdmf;
    Teuchos::XMLObject _traceXdmf;
    Teuchos::XMLObject _fieldDomain;
    Teuchos::XMLObject _traceDomain;
    Teuchos::XMLObject _fieldGrids;
    Teuchos::XMLObject _traceGrids;
    set<double> _fieldTimeVals;
    set<double> _traceTimeVals;

    void getPoints(Intrepid::FieldContainer<double> &points, CellTopoPtr cellTopo, int num1DPts);
  public:
    HDF5Exporter(MeshPtr mesh, std::string outputDirName="output", std::string outputDirSuperPath = ".");
    ~HDF5Exporter();
    void setMesh(MeshPtr mesh) {_mesh = mesh;}
    typedef std::map<int, int> map_int_int;
    void exportFunction(TFunctionPtr<double> function, std::string functionName="function", double timeVal=0,
                        unsigned int defaultNum1DPts=4, map_int_int cellIDToNum1DPts=map_int_int(),
                        std::set<GlobalIndexType> cellIndices=std::set<GlobalIndexType>());
    void exportFunction(std::vector<TFunctionPtr<double>> functions, std::vector<std::string> functionNames, double timeVal=0,
                        unsigned int defaultNum1DPts=4, map_int_int cellIDToNum1DPts=map_int_int(),
                        std::set<GlobalIndexType> cellIndices=std::set<GlobalIndexType>());
    void exportSolution(SolutionPtr<double> solution, double timeVal=0, unsigned int defaultNum1DPts=4,
                        map_int_int cellIDToNum1DPts=map_int_int(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
    void exportTimeSlab(TFunctionPtr<double> function, std::string functionName="function", double tInit=0, double tFinal=1, unsigned int numSlices=2,
      unsigned int sliceH1Order=2, unsigned int defaultNum1DPts=4);
    void exportTimeSlab(std::vector<TFunctionPtr<double>> functions, std::vector<std::string> functionNames, double tInit=0, double tFinal=1, unsigned int numSlices=2,
      unsigned int sliceH1Order=2, unsigned int defaultNum1DPts=4);

    // DEPRECATED METHOD:
    void exportSolution(SolutionPtr<double> solution, VarFactory varFactory, double timeVal=0,
                        unsigned int defaultNum1DPts=4, map_int_int cellIDToNum1DPts=map_int_int(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());

    // allows one-line export without storing an exporter object
    static void exportFunction(std::string directoryPath, std::string functionName, TFunctionPtr<double> function, MeshPtr mesh);
    // allows one-line export without storing an exporter object
    static void exportSolution(std::string directoryPath, std::string solutionName, SolutionPtr<double> solution);
  };

  // creates a map from cell index to number of 1D points (number of subdivisions + 1)
  // num1DPts = max(subdivisionFactor*(polyOrder-1) + 1, 2)
  map<int,int> cellIDToSubdivision(MeshPtr mesh, unsigned int subdivisionFactor=2, set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
}

#else

/* DUMMY (NO-OP) IMPLEMENTATION FOR WHEN HDF5 IS UNAVAILABLE */
// class HDF5Exporter {
// private:
// public:
//   HDF5Exporter(MeshPtr mesh, std::string saveDirectory="output") {}
//   ~HDF5Exporter() {}
//   void exportFunction(TFunctionPtr<double> function, std::string functionName="function", double timeVal=0,
//                       unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>()) {}
//   void exportFunction(std::vector<TFunctionPtr<double>> functions, std::vector<std::string> functionNames, double timeVal=0,
//                       unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>()) {}
//   void exportSolution(SolutionPtr<double> solution, VarFactory varFactory, double timeVal=0,
//                       unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>()) {}
//   static void exportSolution(std::string saveDirectory, SolutionPtr<double> solution) {}
// };
/* END OF DUMMY IMPLEMENTATION */

#endif

#endif /* end of include guard: HDF5EXPORTER_H */
