#ifndef HDF5EXPORTER_H
#define HDF5EXPORTER_H

/*
 *  HDF5Exporter.h
 *
 *  Created by Truman Ellis on 6/24/2014.
 *
 */

#include "CamelliaConfig.h"

#ifdef USE_HDF5

#include "Solution.h"
#include "Mesh.h"
#include "MeshTopology.h"
#include "VarFactory.h"

class HDF5Exporter {
private:
  string _filename;
  MeshPtr _mesh;
  // SolutionPtr _solution;
  // VarFactory& _varFactory;
public:
  HDF5Exporter(MeshPtr mesh, string filename="output", bool deleteOldFiles=false) : _mesh(mesh), _filename(filename)
    {
      if (deleteOldFiles)
      {
        system("rm -rf *.xmf");
        system("rm -rf HDF5/*");
      }
      system("mkdir -p HDF5");

      // root.SetDOM(&dom);
      // root.SetVersion(2.0);
      // root.Build();
      // // Domain
      // root.Insert(&domain);
      // // Grid
      // gridCollection.SetName("Grid Collection");
      // gridCollection.SetGridTypeFromString("Collection");
      // gridCollection.SetCollectionTypeFromString("Spatial");
      // domain.Insert(&gridCollection);
      // fieldTemporalCollection.SetName("Field Temporal Collection");
      // fieldTemporalCollection.SetGridTypeFromString("Collection");
      // fieldTemporalCollection.SetCollectionTypeFromString("Temporal");
      // traceTemporalCollection.SetName("Trace Temporal Collection");
      // traceTemporalCollection.SetGridTypeFromString("Collection");
      // traceTemporalCollection.SetCollectionTypeFromString("Temporal");
      // gridCollection.Insert(&fieldTemporalCollection);
      // gridCollection.Insert(&traceTemporalCollection);
    }
    ~HDF5Exporter()
    {
    }
  void exportFunction(FunctionPtr function, string functionName="function", double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), MeshPtr mesh=Teuchos::rcp((Mesh*)NULL),  set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
  void exportFunction(vector<FunctionPtr> functions, vector<string> functionNames, double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), MeshPtr mesh=Teuchos::rcp((Mesh*)NULL), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
  void exportSolution(SolutionPtr solution, MeshPtr mesh, VarFactory varFactory, double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
};

// creates a map from cell index to number of 1D points (number of subdivisions + 1)
// num1DPts = max(subdivisionFactor*(polyOrder-1) + 1, 2)
map<int,int> cellIDToSubdivision(MeshPtr mesh, unsigned int subdivisionFactor=2, set<GlobalIndexType> cellIndices=set<GlobalIndexType>());

#endif

#endif /* end of include guard: HDF5EXPORTER_H */
