//
//  MeshTools.h
//  Camellia-debug
//
//  Created by Nate Roberts on 7/9/14.
//
//

#ifndef __Camellia_debug__MeshTools__
#define __Camellia_debug__MeshTools__

#include "TypeDefs.h"

#include "Mesh.h"
#include <map>
#include <string>

class MeshTools {
public:
  static MeshPtr timeSliceMesh(MeshPtr spaceTimeMesh, double t,
                               map<GlobalIndexType, GlobalIndexType> &sliceCellIDToSpaceTimeCellID, int H1OrderForSlice);
  
  static void timeSliceExport(std::string dirPath, MeshPtr mesh, FunctionPtr spaceTimeFunction, std::vector<double> tValues, std::string functionName="function");
  
  static FunctionPtr timeSliceFunction(MeshPtr spaceTimeMesh, map<GlobalIndexType, GlobalIndexType> &cellIDMap, FunctionPtr spaceTimeFunction, double t);
};

#endif /* defined(__Camellia_debug__MeshTools__) */
