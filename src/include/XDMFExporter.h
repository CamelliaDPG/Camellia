#ifndef XDMFEXPORTER_H
#define XDMFEXPORTER_H

/*
 *  XDMFExporter.h
 *
 *  Created by Truman Ellis on 6/19/2014.
 *
 */

#include "CamelliaConfig.h"
 
#ifdef USE_XDMF
#include <Xdmf.h>

#include "Solution.h"
#include "Mesh.h"
#include "MeshTopology.h"
#include "VarFactory.h"

class XDMFExporter {
private:
  XdmfDOM         dom;
  XdmfRoot        root;
  XdmfDomain      domain;
  XdmfGrid        gridCollection;
  XdmfGrid        fieldTemporalCollection;
  XdmfGrid        traceTemporalCollection;
  string          filename;
protected:
  // SolutionPtr _solution;
  MeshTopologyPtr _meshTopology;
  // VarFactory& _varFactory;
public:
  XDMFExporter(MeshTopologyPtr meshTopology, string filename="output", bool deleteOldFiles=false) : _meshTopology(meshTopology), filename(filename)
    {
      if (deleteOldFiles)
      {
        system("rm -rf *.xmf");
        system("rm -rf HDF5/*");
      }
      system("mkdir -p HDF5");

      root.SetDOM(&dom);
      root.SetVersion(2.0);
      root.Build();
      // Domain
      root.Insert(&domain);
      // Grid
      gridCollection.SetName("Grid Collection");
      gridCollection.SetGridTypeFromString("Collection");
      gridCollection.SetCollectionTypeFromString("Spatial");
      domain.Insert(&gridCollection);
      fieldTemporalCollection.SetName("Field Temporal Collection");
      fieldTemporalCollection.SetGridTypeFromString("Collection");
      fieldTemporalCollection.SetCollectionTypeFromString("Temporal");
      traceTemporalCollection.SetName("Trace Temporal Collection");
      traceTemporalCollection.SetGridTypeFromString("Collection");
      traceTemporalCollection.SetCollectionTypeFromString("Temporal");
      gridCollection.Insert(&fieldTemporalCollection);
      gridCollection.Insert(&traceTemporalCollection);
    }
    ~XDMFExporter()
    {
    }
  void exportFunction(FunctionPtr function, string functionName="function", double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), MeshPtr mesh=Teuchos::rcp((Mesh*)NULL),  set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
  void exportFunction(vector<FunctionPtr> functions, vector<string> functionNames, double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), MeshPtr mesh=Teuchos::rcp((Mesh*)NULL), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
  void exportSolution(SolutionPtr solution, MeshPtr mesh, VarFactory varFactory, double timeVal=0, 
    unsigned int defaultNum1DPts=4, map<int, int> cellIDToNum1DPts=map<int,int>(), set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
};

//Convenience functions for cellIDTONum1DPts
map<int,int> cellIDToPolyOrder(MeshPtr mesh, set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
map<int,int> coarseSubdivisions(MeshPtr mesh, set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
map<int,int> mediumSubdivisions(MeshPtr mesh, set<GlobalIndexType> cellIndices=set<GlobalIndexType>());
map<int,int> fineSubdivisions(MeshPtr mesh, set<GlobalIndexType> cellIndices=set<GlobalIndexType>());

#endif

#endif /* end of include guard: XDMFEXPORTER_H */
