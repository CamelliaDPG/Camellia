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

namespace Camellia
{
class MeshTools
{
public:
  // ! For two MeshTopologies whose cells have same geometry (vertices), generates a map from the cell indices of
  // ! one to the cell indices of the other.  (Not meant for large-scale/production use.)  Keys are the cell indices
  // ! in meshTopoFrom; values are cell indices in meshTopoTo.
  static map<IndexType,IndexType> mapActiveCellIndices(MeshTopologyViewPtr meshTopoFrom, MeshTopologyViewPtr meshTopoTo);
  
  static MeshPtr timeSliceMesh(MeshPtr spaceTimeMesh, double t,
                               map<GlobalIndexType, GlobalIndexType> &sliceCellIDToSpaceTimeCellID, int H1OrderForSlice);

  static void timeSliceExport(std::string dirPath, MeshPtr mesh, TFunctionPtr<double> spaceTimeFunction, std::vector<double> tValues, std::string functionName="function");

  static TFunctionPtr<double> timeSliceFunction(MeshPtr spaceTimeMesh, map<GlobalIndexType, GlobalIndexType> &cellIDMap, TFunctionPtr<double> spaceTimeFunction, double t);
};
}

#endif /* defined(__Camellia_debug__MeshTools__) */
