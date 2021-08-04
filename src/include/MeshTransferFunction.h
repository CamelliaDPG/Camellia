#ifndef MESH_TRANSFER_FUNCTION
#define MESH_TRANSFER_FUNCTION

// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// ***********************************************************************
//
//                  Camellia Mesh Transfer Function:
//
//  A function that is meant to transfer data from one mesh to another.
//  Implemented for the purpose of space-time meshes, but might also be
//  useful in the context of loosely coupled multiphysics implementations.
//

// ***********************************************************************
//@HEADER

#include "Function.h"
#include "Mesh.h"
#include "BasisCache.h"

#include "RefinementObserver.h"

//! MeshTransferFunction: Given two meshes with a shared interface...
//
/*!

 \author Nathan V. Roberts, ALCF.

 \date Last modified on 25-Nov-2014.
 */

namespace Camellia
{
class MeshTransferFunction : public TFunction<double>, public RefinementObserver
{
  MeshPtr _originalMesh, _newMesh;
  MeshTopologyPtr _gatheredOriginalMeshTopologyOnInterface; // refreshed on each call to rebuild()
  
  TFunctionPtr<double> _originalFunction;
  double _interface_t;

  typedef std::pair<GlobalIndexType,unsigned> CellSide; // cellID, side ordinal
  std::map<CellSide,CellSide> _newToOriginalMap;
  std::map<CellSide,CellSide> _originalToNewMap;
  std::map<CellSide,CellSide> _activeSideToAncestralSideInNewMesh;
  std::map<CellSide, unsigned> _permutationForNewMeshCellSide; // permutation goes from cell side in _newMesh to that in _originalMesh

  void rebuildMaps();
public:
  MeshTransferFunction(TFunctionPtr<double> originalFunction, MeshPtr originalMesh, MeshPtr newMesh, double interface_t);
  
  using TFunction<double>::values;
  virtual void values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache);

  bool boundaryValueOnly();

  bool findAncestralPairForNewMeshCellSide(const CellSide &newMeshCellSide, CellSide &newMeshCellSideAncestor,
      CellSide &originalMeshCellSideAncestor, unsigned &newCellSideAncestorPermutation);

  const std::map<CellSide,CellSide> & mapToOriginalMesh()
  {
    return _newToOriginalMap;
  }
  const std::map<CellSide,CellSide> & mapToNewMesh()
  {
    return _originalToNewMap;
  }

  // RefinementObserver method:
  void didRepartition(MeshTopologyPtr meshTopology);

  ~MeshTransferFunction();
};
}


#endif
