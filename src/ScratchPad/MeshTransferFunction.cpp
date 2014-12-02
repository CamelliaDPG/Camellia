#include "MeshTransferFunction.h"

MeshTransferFunction::MeshTransferFunction(FunctionPtr originalFunction, MeshPtr originalMesh,
                                           MeshPtr newMesh, double interface_t) : Function(originalFunction->rank()) {
  _originalFunction = originalFunction;
  _originalMesh = originalMesh;
  _newMesh = newMesh;
  
  // 1. Find (cellID, sideOrdinal) belonging to this rank that have interface_t values in originalMesh.
  
  // initial strategy: examine the vertices that belong to cells owned by this rank. Sides that are wholly
  //                   comprised of vertices on the interface are the sides of interest
  
  // 2. Find corresponding cells in newMesh.  Typically, these will be less refined than originalMesh,
  //    so that the relationship is many-to-one from originalMesh.  We look for a common ancestor to
  //    make it one-to-one.  I.e. when we find a newMesh CellSide matching one from originalMesh, we
  //    first check whether the newMesh entry already has a corresponding originalMesh entry.  If so,
  //    we seek a common ancestor of the two entries.
  // 3. Create a map in each direction.  This should be a bijection, but not all cells will be active;
  //    some will be inactive parent cells.
  // 4. Register for refinement notifications on both meshes.  When a notification is received, update
  //    the map.  For simplicity in the initial implementation, it may be useful to make the assumption
  //    that originalMesh does not change after MeshTransferFunction is constructed.  This can be enforced
  //    by throwing an exception if originalMesh is refined.
  
  // One question is how to deal with the fact that corresponding cells in original and new mesh
  // may belong to different ranks.  In such a case, values() will be called with data (that is, physical
  // points and cell IDs) corresponding to newMesh.  It seems that the most general approach will be to
  // do a round of MPI communication (via Epetra_MpiDistributor) to get appropriate data to the owner
  // of the corresponding cell on originalMesh.  Then each rank will translate the data to originalMesh
  // (possibly by appropriate permutations of the reference points), create a corresponding BasisCache,
  // and compute values.  Finally, another round of MPI communication will send the data back to the requestor.
  
  // The above is not a communications-optimized approach, in that cubature points, e.g., can be locally determined
  // without being sent.  Similarly, in the case of solution data, simply sending the solution coefficients to the
  // requesting MPI rank would give sufficient information that all necessary computation could be done locally.  However,
  // this is making an assumption about the nature of originalFunction.  It is worth noting that the communications
  // here typically only need to be done once per solve on newMesh, and this during the determination of boundary conditions.
}

bool MeshTransferFunction::boundaryValueOnly() {
  // This function is only valid on the interface (boundary) between the two meshes.
  return true;
}


void MeshTransferFunction::didHRefine(MeshTopologyPtr meshTopo, const set<GlobalIndexType> &cellIDs,
                                      Teuchos::RCP<RefinementPattern> refPattern) {
  
}

void MeshTransferFunction::didHUnrefine(MeshTopologyPtr meshTopo, const set<GlobalIndexType> &cellIDs) {
  
}

void MeshTransferFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
  // TODO: implement this
  
  // MPI-communicating method.  Must be called on all ranks.
  
  /*
   basisCache contains info on cells in newMesh.
   Basic idea: for each CellSide in basisCache, determine the corresponding CellSide in originalMesh.
   This may not be an active cell.  If it is not, we need to subdivide the physical points, assigning
   subsets to active cells.  We might do this by sending *all* the physical points to all the owners of
   the CellSide's descendants.  It would then be the owner's responsibility to determine which physical 
   points belong to them, and to communicate regarding these.  (It might be that this is done via an
   integer array indicating which point ordinals are being claimed followed by a double array indicating
   which values correspond.)
   */
  
}