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

namespace Camellia {
	class MeshTools {
	public:
	  static MeshPtr timeSliceMesh(MeshPtr spaceTimeMesh, double t,
	                               map<GlobalIndexType, GlobalIndexType> &sliceCellIDToSpaceTimeCellID, int H1OrderForSlice);

	  static void timeSliceExport(std::string dirPath, MeshPtr mesh, TFunctionPtr<double> spaceTimeFunction, std::vector<double> tValues, std::string functionName="function");

	  static TFunctionPtr<double> timeSliceFunction(MeshPtr spaceTimeMesh, map<GlobalIndexType, GlobalIndexType> &cellIDMap, TFunctionPtr<double> spaceTimeFunction, double t);
	};
}

#endif /* defined(__Camellia_debug__MeshTools__) */
