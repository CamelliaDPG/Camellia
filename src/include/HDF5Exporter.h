#ifndef HDF5EXPORTER_H
#define HDF5EXPORTER_H

/*
 *  HDF5Exporter.h
 *
 *  Created by Truman Ellis on 6/24/2014.
 *
 */

#include "CamelliaConfig.h"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5

#include "Solution.h"
#include "Mesh.h"
#include "MeshTopology.h"
#include "VarFactory.h"

#include <Teuchos_XMLObject.hpp>

class HDF5Exporter {
private:
  string _filename;
  MeshPtr _mesh;
  // ofstream _xmfFile;
  XMLObject _xdmf;
  XMLObject _domain;
  XMLObject _fieldGrids;
  XMLObject _traceGrids;
  set<double> _fieldTimeVals;
  set<double> _traceTimeVals;
public:
  HDF5Exporter(MeshPtr mesh, string saveDirectory="output", bool deleteOldFiles=false);
  ~HDF5Exporter();
  void exportFunction(FunctionPtr function, string functionName="function", double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
  void exportFunction(vector<FunctionPtr> functions, vector<string> functionNames, double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
  void exportSolution(SolutionPtr solution, VarFactory varFactory, double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
};

// creates a map from cell index to number of 1D points (number of subdivisions + 1)
// num1DPts = max(subdivisionFactor*(polyOrder-1) + 1, 2)
map<int,int> cellIDToSubdivision(MeshPtr mesh, unsigned int subdivisionFactor=2, set<GlobalIndexType> cellIndices=set<GlobalIndexType>());

#endif

#endif /* end of include guard: HDF5EXPORTER_H */
