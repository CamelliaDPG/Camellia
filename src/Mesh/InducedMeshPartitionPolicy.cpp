//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
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
: MeshPartitionPolicy(otherMesh->TeuchosComm())
{
  _otherMesh = otherMesh;
  
  set<GlobalIndexType> otherCellIDs = otherMesh->getActiveCellIDsGlobal();
  
  if (_thisMesh != Teuchos::null)
  {
    _thisMesh = Teuchos::rcp(thisMesh.get(), false); // weak RCP to avoid circular references
    
    set<GlobalIndexType> cellIDs = thisMesh->getActiveCellIDsGlobal();
    set<GlobalIndexType> otherCellIDs = otherMesh->getActiveCellIDsGlobal();
    
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != otherCellIDs.size(), std::invalid_argument, "thisMesh and otherMesh must match in their activeCellIDs");
    set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin();
    set<GlobalIndexType>::iterator otherCellIDIt = otherCellIDs.begin();
    for (int i=0; i<cellIDs.size(); i++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(*cellIDIt != *otherCellIDIt, std::invalid_argument, "thisMesh and otherMesh must match in their activeCellIDs");
      cellIDIt++;
      otherCellIDIt++;
    }
  }
  
  for (GlobalIndexType cellID : otherCellIDs)
  {
    _cellIDMap[cellID] = cellID;
  }

  // for now, we don't actually have reworking of the cell map implemented, so we skip registering
//  Teuchos::RCP<RefinementObserver> thisObserver = Teuchos::rcp(this,false); // weak RCP
//  if (_thisMesh != Teuchos::null)
//  {
//    _thisMesh->registerObserver(thisObserver);
//  }
//  _otherMesh->registerObserver(thisObserver);
}

InducedMeshPartitionPolicy::InducedMeshPartitionPolicy(MeshPtr thisMesh, MeshPtr otherMesh, const map<GlobalIndexType, GlobalIndexType> & cellIDMap)
: MeshPartitionPolicy(otherMesh->TeuchosComm())
{
  _thisMesh = thisMesh;
  _otherMesh = otherMesh;
  _cellIDMap = cellIDMap; // copy
}

InducedMeshPartitionPolicy::InducedMeshPartitionPolicy(MeshPtr otherMesh, const map<GlobalIndexType, GlobalIndexType> & cellIDMap)
: MeshPartitionPolicy(otherMesh->TeuchosComm())
{
  _thisMesh = Teuchos::null;
  _otherMesh = otherMesh;
  _cellIDMap = cellIDMap; // copy
}

InducedMeshPartitionPolicy::~InducedMeshPartitionPolicy()
{
  // if/when we turn on registering, we'll want to uncomment this
//  if (_thisMesh != Teuchos::null)
//  {
//    _thisMesh->unregisterObserver(this);
//  }
//  if (_otherMesh != Teuchos::null)
//  {
//    _otherMesh->unregisterObserver(this);
//  }
}

void InducedMeshPartitionPolicy::didHRefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, RefinementPatternPtr refPattern, bool didRepartitionAndRebuild)
{
  // TODO: update _cellIDMap
}

void InducedMeshPartitionPolicy::didHUnrefine(MeshTopologyPtr meshToRefine, const set<GlobalIndexType> &cellIDs, bool didRepartitionAndRebuild)
{
  // TODO: update _cellIDMap
}

void InducedMeshPartitionPolicy::partitionMesh(Mesh *mesh, PartitionIndexType numPartitions)
{
  if (_thisMesh != Teuchos::null)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(mesh != _thisMesh.get(), std::invalid_argument, "InducedMeshPartitionPolicy may only be used to partition the mesh passed as thisMesh to the constructor");
  }
  
  int otherPartitionCount = _otherMesh->globalDofAssignment()->getPartitionCount();
  if (numPartitions < otherPartitionCount)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Induced partition count must be greater than or equal to otherMesh's");
  }
  
  set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDsGlobal();
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
