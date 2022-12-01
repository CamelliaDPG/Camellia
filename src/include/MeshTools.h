// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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
  
  //! For two MeshTopologies that have the same geometry (and root-level topology) at a specified boundary, generate a map of cellIDs (and side ordinals) from one to the other.
  //! Intended for use in time slabs; can be called prior to reconciling by refining in the new time slab to match the prior; look for inactive cells in the map to identify where refinements should be made.
  static void mapCellIDsAtBoundary(std::map<IndexType,IndexType> &cellIDMap, std::map<int,int> &sideOrdinalMap,
                                   ConstMeshTopologyPtr meshTopoFrom, int fromBoundaryTag,
                                   ConstMeshTopologyPtr meshTopoTo);
  
  //! For two Meshes that have the same geometry (and root-level topology) at a specified boundary, refine so that the leaf-level cells in each Mesh have direct correlates
  //! in the other Mesh (i.e., they match in refinement at the interface).  Calls mapCellIDsAtBoundary() to fill the cellIDMap and sideOrdinalMap initially; on return, the cellIDMap
  //! will have only cellIDs belonging to active cells.
  static void hRefineToMatchAtBoundary(std::map<IndexType,IndexType> &cellIDMap, std::map<int,int> &sideOrdinalMap,
                                       MeshPtr meshFrom, int fromBoundaryTag,
                                       MeshPtr meshTo);
  
  //! Returns a vector with numElements+1 equispaced vertices from x0 to x0 + xWidth.
  static vector<double> verticesFor1DIntervalMesh(const double &x0, const double &xWidth, const int &numElements);
  
  static MeshPtr timeSliceMesh(MeshPtr spaceTimeMesh, double t,
                               map<GlobalIndexType, GlobalIndexType> &sliceCellIDToSpaceTimeCellID, int H1OrderForSlice);

  static void timeSliceExport(std::string dirPath, MeshPtr mesh, TFunctionPtr<double> spaceTimeFunction, std::vector<double> tValues, std::string functionName="function");

  static TFunctionPtr<double> timeSliceFunction(MeshPtr spaceTimeMesh, map<GlobalIndexType, GlobalIndexType> &cellIDMap, TFunctionPtr<double> spaceTimeFunction, double t);
};
}

#endif /* defined(__Camellia_debug__MeshTools__) */
