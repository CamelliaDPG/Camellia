//
//  InducedMeshPartitionPolicy.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/24/15.
//
//

#include "InducedMeshPartitionPolicy.h"

#include "GlobalDofAssignment.h"

using namespace Camellia;

Teuchos::RCP<InducedMeshPartitionPolicy> InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh)
{
  return Teuchos::rcp( new InducedMeshPartitionPolicy(thisMesh, otherMesh) );
}

// ! Returns an induced mesh partition policy with the specified initial cellIDMap (keys are cellIDs that belong to thisMesh; values belong to otherMesh)
Teuchos::RCP<InducedMeshPartitionPolicy> InducedMeshPartitionPolicy::inducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType, GlobalIndexType> & cellIDMap)
{
  return Teuchos::rcp( new InducedMeshPartitionPolicy(thisMesh, otherMesh, cellIDMap) );
}

InducedMeshPartitionPolicy::InducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh)
{
  _thisMesh = Teuchos::rcp(thisMesh.get(), false); // weak RCP to avoid circular references
  _otherMesh = otherMesh;
  
  set<GlobalIndexType> cellIDs = thisMesh->getActiveCellIDs();
  set<GlobalIndexType> otherCellIDs = otherMesh->getActiveCellIDs();
  
  TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != otherCellIDs.size(), std::invalid_argument, "thisMesh and otherMesh must match in their activeCellIDs");
  set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin();
  set<GlobalIndexType>::iterator otherCellIDIt = otherCellIDs.begin();
  for (int i=0; i<cellIDs.size(); i++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(*cellIDIt != *otherCellIDIt, std::invalid_argument, "thisMesh and otherMesh must match in their activeCellIDs");
    cellIDIt++;
    otherCellIDIt++;
  }
  
  for (GlobalIndexType cellID : cellIDs)
  {
    _cellIDMap[cellID] = cellID;
  }
  
  Teuchos::RCP<RefinementObserver> thisObserver = Teuchos::rcp(this,false); // weak RCP
  _thisMesh->registerObserver(thisObserver);
  _otherMesh->registerObserver(thisObserver);
}

InducedMeshPartitionPolicy::InducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType, GlobalIndexType> & cellIDMap)
{
  _thisMesh = thisMesh;
  _otherMesh = otherMesh;
  _cellIDMap = cellIDMap; // copy
}

void InducedMeshPartitionPolicy::didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, RefinementPatternPtr refPattern)
{
  // TODO: update _cellIDMap
}

void InducedMeshPartitionPolicy::didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs)
{
  // TODO: update _cellIDMap
}

void InducedMeshPartitionPolicy::partitionMesh(Mesh *mesh, PartitionIndexType numPartitions)
{
  TEUCHOS_TEST_FOR_EXCEPTION(mesh != _thisMesh.get(), std::invalid_argument, "InducedMeshPartitionPolicy may only be used to partition the mesh passed as thisMesh to the constructor");
  
  int otherPartitionCount = _otherMesh->globalDofAssignment()->getPartitionCount();
  if (numPartitions < otherPartitionCount)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Induced partition count must be greater than or equal to otherMesh's");
  }
  
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
  vector< set<GlobalIndexType> > partitions(numPartitions);
  
  for (set<GlobalIndexType>::iterator myCellIDIt = activeCellIDs.begin(); myCellIDIt != activeCellIDs.end(); myCellIDIt++)
  {
    GlobalIndexType myCellID = *myCellIDIt;
    if (_cellIDMap.find(myCellID) == _cellIDMap.end())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID not found in _cellIDMap");
    }
    GlobalIndexType otherCellID = _cellIDMap[myCellID];
    int otherPartitionNumber = _otherMesh->globalDofAssignment()->partitionForCellID(otherCellID);
    partitions[otherPartitionNumber].insert(myCellID);
  }
  
  mesh->globalDofAssignment()->setPartitions(partitions);
}