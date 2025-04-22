//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include <iostream>

#include "MeshPartitionPolicy.h"

#include "GlobalDofAssignment.h"
#include "InducedMeshPartitionPolicy.h"
#include "MeshTools.h"
#include "MPIWrapper.h"
#include "ZoltanMeshPartitionPolicy.h"

using namespace Camellia;
using namespace std;

MeshPartitionPolicy::MeshPartitionPolicy(Teuchos_CommPtr Comm) : _Comm(Comm) {
  TEUCHOS_TEST_FOR_EXCEPTION(Comm == Teuchos::null, std::invalid_argument, "Comm may not be null!");
}

Teuchos_CommPtr MeshPartitionPolicy::Comm()
{
  return _Comm;
}

void MeshPartitionPolicy::partitionMesh(Mesh *mesh, PartitionIndexType numPartitions)
{
  // default simply divides the active cells into equally-sized partitions, in the order listed in activeCells…
  MeshTopologyViewPtr meshTopology = mesh->getTopology();
  int numActiveCells = meshTopology->activeCellCount(); // leaf nodes
  std::vector<std::set<GlobalIndexType> > partitionedActiveCells(numPartitions);

  int chunkSize = numActiveCells / numPartitions;
  int remainder = numActiveCells % numPartitions;
  IndexType activeCellOrdinal = 0;
  set<GlobalIndexType> activeCellIDsSet = mesh->getActiveCellIDsGlobal();
  vector<GlobalIndexType> activeCellIDs(activeCellIDsSet.begin(),activeCellIDsSet.end());
  TEUCHOS_TEST_FOR_EXCEPTION(activeCellIDs.size() != numActiveCells, std::invalid_argument, "meshTopology->getActiveCellCount() != mesh->getActiveCellIDsGlobal().size()");
  for (int i=0; i<numPartitions; i++)
  {
    int chunkSizeWithRemainder = (i < remainder) ? chunkSize + 1 : chunkSize;
    for (int j=0; j<chunkSizeWithRemainder; j++)
    {
      partitionedActiveCells[i].insert(activeCellIDs[activeCellOrdinal]);
      activeCellOrdinal++;
    }
  }
  
  mesh->globalDofAssignment()->setPartitions(partitionedActiveCells);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh)
{
  return InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(thisMesh, otherMesh);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType,GlobalIndexType> &cellIDMap)
{
  return InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(thisMesh, otherMesh, cellIDMap);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicy(MeshPtr inducingMesh) // for two meshes that have the same cell indices, uses inducingMesh to define partitioning
{
  return inducedPartitionPolicy(Teuchos::null, inducingMesh);
}

MeshPartitionPolicyPtr MeshPartitionPolicy::inducedPartitionPolicyFromRefinedMesh(MeshTopologyViewPtr inducedMeshTopo, MeshPtr inducingRefinedMesh)
{
  // generate using the inducing mesh
  vector<GlobalIndexTypeToCast> myEntries;
  
  auto myCellIDs = &inducingRefinedMesh->cellIDsInPartition();
  bool rotateChildOrdinalThatOwns = false; // the 6-27-16 modification -- I don't think this actually helps with load balance, the way things are presently implemented, and it may introduce additional communication costs
  if (rotateChildOrdinalThatOwns)
  {
    /*
     Modification 6-27-16: instead of assigning parent to owner of first child,
     assign parent to owner of child with ordinal equal to the level of the
     parent, modulo the number of children.  This should result in better load
     balancing for multigrid.
     */
    for (GlobalIndexType myCellID : *myCellIDs)
    {
      IndexType ancestralCellIndex = myCellID;
      bool hasMatchingChild = true;
      while (inducingRefinedMesh->getTopology()->isValidCellIndex(ancestralCellIndex)
             && !inducedMeshTopo->isValidCellIndex(ancestralCellIndex))
      {
        CellPtr myCell = inducingRefinedMesh->getTopology()->getCell(ancestralCellIndex);
        CellPtr parent = myCell->getParent();
        TEUCHOS_TEST_FOR_EXCEPTION(parent == Teuchos::null, std::invalid_argument, "ancestor not found in inducedMeshTopo");
        int childOrdinal = parent->findChildOrdinal(myCell->cellIndex());
        int numChildren = parent->numChildren();
        hasMatchingChild = ((parent->level() % numChildren) == childOrdinal);
        ancestralCellIndex = parent->cellIndex();
      }
      if (hasMatchingChild)
      {
        myEntries.push_back(ancestralCellIndex);
        myEntries.push_back(myCellID);
      }
    }
  }
  else
  {
    // first child is always owner
    for (GlobalIndexType myCellID : *myCellIDs)
    {
      IndexType ancestralCellIndex = myCellID;
      bool isFirstChild = true;
      while (isFirstChild && !inducedMeshTopo->isValidCellIndex(ancestralCellIndex))
      {
        CellPtr myCell = inducingRefinedMesh->getTopology()->getCell(ancestralCellIndex);
        CellPtr parent = myCell->getParent();
        TEUCHOS_TEST_FOR_EXCEPTION(parent == Teuchos::null, std::invalid_argument, "ancestor not found in inducedMeshTopo");
        int childOrdinal = parent->findChildOrdinal(myCell->cellIndex());
        isFirstChild = (childOrdinal == 0);
        ancestralCellIndex = parent->cellIndex();
      }
      if (isFirstChild)
      {
        myEntries.push_back(ancestralCellIndex);
        myEntries.push_back(myCellID);
      }
    }
  }
  // all-gather the entries
  vector<GlobalIndexTypeToCast> allEntries;
  vector<int> offsets;
  MPIWrapper::allGatherVariable(inducingRefinedMesh->Comm(), allEntries, myEntries, offsets);
  
  map<GlobalIndexType,GlobalIndexType> cellIDMap;
  for (int i=0; i<allEntries.size()/2; i++)
  {
    GlobalIndexType ancestralCellIndex = allEntries[2*i+0];
    GlobalIndexType inducingMeshCellID = allEntries[2*i+1];
    cellIDMap[ancestralCellIndex] = inducingMeshCellID;
  }
  
  return Teuchos::rcp( new InducedMeshPartitionPolicy(inducingRefinedMesh, cellIDMap) );
}

MeshPartitionPolicyPtr MeshPartitionPolicy::standardPartitionPolicy(Teuchos_CommPtr Comm)
{
  MeshPartitionPolicyPtr partitionPolicy = Teuchos::rcp( new ZoltanMeshPartitionPolicy(Comm) );
  return partitionPolicy;
}

Teuchos_CommPtr MeshPartitionPolicy::TeuchosComm()
{
  return _Comm;
}
