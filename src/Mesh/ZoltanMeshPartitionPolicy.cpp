//
//  ZoltanMeshPartitionPolicy.cpp
//  Camellia
//
//  Copyright 2011 Nathan Roberts. All rights reserved.
//

#include <iostream>
#include <stdlib.h> 
#include <vector>
#include "Mesh.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "CellDataMigration.h"

#include "GlobalDofAssignment.h"

#include "CamelliaDebugUtility.h"

#include "MPIWrapper.h"

#include "Solution.h"

#include <Teuchos_GlobalMPISession.hpp>

ZoltanMeshPartitionPolicy::ZoltanMeshPartitionPolicy(){
  string partitionerName = "HSFC"; // was "BLOCK"
  string debug_level = "0"; // was "10"
//  cout << "ZoltanMeshPartitionPolicy: Defaulting to HSFC partitioner" << endl;
  _ZoltanPartitioner = partitionerName;
  _debug_level = debug_level;
}
ZoltanMeshPartitionPolicy::ZoltanMeshPartitionPolicy(string partitionerName){
  string debug_level = "0";
  _ZoltanPartitioner = partitionerName;  
  _debug_level = debug_level;
}

void ZoltanMeshPartitionPolicy::partitionMesh(Mesh *mesh, PartitionIndexType numPartitions) {
  int myNode = Teuchos::GlobalMPISession::getRank();
//  cout << "Entered ZoltanMeshPartitionPolicy::partitionMesh() on rank " << myNode << endl;
//  cout << "ZoltanMeshPartitionPolicy::partitionMesh, registered solution count: " << mesh->globalDofAssignment()->getRegisteredSolutions().size() << endl;
  int numNodes = numPartitions;
  int maxPartitionSize = mesh->numActiveElements();
  GlobalIndexType numActiveElements = mesh->numActiveElements();
  
  FieldContainer<GlobalIndexType> partitionedActiveCells(numNodes,maxPartitionSize);

  partitionedActiveCells.initialize(-1); // cellID == -1 signals end of partition
  
  float version;
  //these arguments are ignored by Zoltan initialize if MPI_init is called
  int argc = 0;
  Zoltan_Initialize(argc, NULL, &version);
  
  MeshTopologyPtr meshTopology = mesh->getTopology();
  
  if (numNodes>1){
#ifdef HAVE_MPI
    Zoltan *zz = new Zoltan(MPI::COMM_WORLD);
    if (zz == NULL){
      cout << "ZoltanMeshPartititionPolicy: construction of new Zoltan object failed.\n";
      MPI::Finalize();
      exit(0);
    }
    bool useLocalIDs = false;
    
    /* Calling Zoltan Load-balancing routine */
    //cout << "Setting zoltan params" << endl;
    zz->Set_Param( "LB_METHOD", _ZoltanPartitioner.c_str());    /* Zoltan method */
    zz->Set_Param( "NUM_GID_ENTRIES", "1");  /* global ID is 1 integer */
    if (useLocalIDs){
      zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is 1 integer */
    }else{
      zz->Set_Param( "NUM_LID_ENTRIES", "0");  /* local ID is null */
    }
    zz->Set_Param( "OBJ_WEIGHT_DIM", "0");   
    zz->Set_Param( "DEBUG_LEVEL", _debug_level);
    //  zz->Set_Param( "REFTREE_INITPATH", "CONNECTED"); // no SFC on coarse meshTopology
    zz->Set_Param( "RANDOM_MOVE_FRACTION", "1.0");    /* Zoltan "random" partition param */
    
    zz->Set_Param( "IMBALANCE_TOL", "1.1"); // the default is 1.1; measured as the max. load divided by the average load -- worth noting that this is sometimes clearly unattainable (if you have e.g. 5 elements and 4 MPI ranks, you will have an average load of 1.25, and a maximum load of at least 2), and even in such cases zoltan issues a warning.  If we wanted to eliminate such warnings, it would be easy enough to compute the best case as determined by the pigeonhole principle (assuming equal weights, as we have now), and take the more tolerant of the best case versus e.g. 1.1.  But if you think of the warning as simply saying hey your work is imbalanced, that's true, even if that's totally unavoidable.
    
    Mesh* myData = mesh;
    
    // Testing query functions
    zz->Set_Num_Obj_Fn(&get_number_of_objects, myData);
    zz->Set_Obj_List_Fn(&get_object_list, myData);
    
    // HSFC query functions   
    zz->Set_Num_Geom_Fn(&get_num_geom, myData);
    zz->Set_Geom_Multi_Fn(&get_geom_list, myData);
    
    // object sizing/packing functions:
    zz->Set_Obj_Size_Fn( &get_elem_data_size, myData);
    zz->Set_Pack_Obj_Fn( &pack_elem_data, myData);
    zz->Set_Unpack_Obj_Fn( &unpack_elem_data, myData);
    
    int changes;
    int numGidEntries;
    int numLidEntries;
    int numImport;
    ZOLTAN_ID_PTR importGlobalIds;
    ZOLTAN_ID_PTR importLocalIds;
    int *importProcs;
    int *importToPart;
    int numExport;
    ZOLTAN_ID_PTR exportGlobalIds;
    ZOLTAN_ID_PTR exportLocalIds;
    int *exportProcs;
    int *exportToPart;
    
    int rc = zz->LB_Partition(changes, numGidEntries, numLidEntries,
                              numImport, importGlobalIds, importLocalIds, importProcs, importToPart,
                              numExport, exportGlobalIds, exportLocalIds, exportProcs, exportToPart);
    
    if (rc == ZOLTAN_WARN) {
      if (numActiveElements / numPartitions > 0) { // if the warning is just because not every process received an element, don't even note that the warning was issued
        if (myNode == 0) {
//          printf("Partitioning with Zoltan on process %d returned a warning.  # active elements = %d\n",myNode,numActiveElements);
          printf("Partitioning with Zoltan returned a warning.  # active elements = %d\n",numActiveElements);
        }
      } else
        rc = ZOLTAN_OK;
    }
    
    if (rc == ZOLTAN_FATAL) {
      printf("Partitioning failed on process %d with a fatal error.  Exiting...\n",myNode);
      exit(1);
    } else {
      
      /* ----------- modify output array partitionedActiveCells ------- */
      
      set<GlobalIndexType> rankLocalCells = getRankLocalCellIDs(mesh);
      // remove export IDs FOR THIS NODE
      for (int i=0;i<numExport;i++){
        rankLocalCells.erase(exportGlobalIds[i]);
      }
      // add import IDs FOR THIS NODE
      for (int i=0;i<numImport;i++){
        rankLocalCells.insert(importGlobalIds[i]);
      }
      // compute total number of IDs for this proc
      bool reportAssignment = false;
//      bool reportAssignment = (rc==ZOLTAN_WARN);
      if (reportAssignment) {
        ostringstream rankListLabel;
        rankListLabel << "For rank " << myNode << ", Zoltan assigned IDs: ";
        Camellia::print(rankListLabel.str(), rankLocalCells);
      }
      
      // need to pass around information about partitions here thru MPI - each processor must know all other processors' partitions
      FieldContainer<int> myPartition(maxPartitionSize);
      myPartition.initialize(-1);

      int index = 0;
      for (set<GlobalIndexType>::iterator myCellIt = rankLocalCells.begin(); myCellIt != rankLocalCells.end(); myCellIt++) {
        myPartition[index] = *myCellIt;
        index++;
      }
      FieldContainer<int> allPartitions(numNodes,maxPartitionSize);
      MPIWrapper::allGather(allPartitions, myPartition);
      
      // convert the ints to GlobalIndexType -- if sizeof(GlobalIndexType) ever is bigger than sizeof(int), then we'll want to do something else above to pack the cell IDs into ints, etc.
      for (int node=0;node<numNodes;node++){
        for (int i=0;i<maxPartitionSize;i++){
          partitionedActiveCells(node,i) = allPartitions(node,i);
        }	
      }
      
      // now that we have the new partition, communicate it:
      mesh->globalDofAssignment()->setPartitions(partitionedActiveCells);
      
//      cout << "about to call zz->Migrate on rank " << myNode << endl;
      int rc = zz->Migrate(numImport, importGlobalIds, importLocalIds, importProcs, importToPart,
                           numExport, exportGlobalIds, exportLocalIds, exportProcs, exportToPart);
      
      if (rc == ZOLTAN_FATAL) {
        printf("Zoltan: migration failed on process %d with a fatal error.  Exiting...\n",myNode);
        exit(1);
      }
//      cout << "zz->Migrate returned on rank " << myNode << " with result code " << rc << endl;
      
      /*
       for (int node=0;node<numNodes;node++){
       cout << "ids for node " << node << " are: ";
       for (int i = 0;i<maxPartitionSize;i++){
       if (partitionedActiveCells(node,i) !=-1){
       cout << partitionedActiveCells(node,i)<< ", ";
       }
       }
       cout << endl;
       }  
       */
      
    }//end else
    
    delete zz;
#endif
  } else { // if just one node, partition = active cellID array
    set<GlobalIndexType> activeCellIDSet = mesh->getTopology()->getActiveCellIndices();
    vector< GlobalIndexType > activeCellIDs(activeCellIDSet.begin(), activeCellIDSet.end());
    for (int i=0;i<numActiveElements;i++){
      //    for (vector<Teuchos::RCP< Element > >::iterator elemIt=activeElements.begin();elemIt!=activeElements.end();elemIt++){
      partitionedActiveCells(0,i) = activeCellIDs[i];
    }
    // now that we have the new partition, communicate it:
    mesh->globalDofAssignment()->setPartitions(partitionedActiveCells);
  }
}    

//GlobalIndexType ZoltanMeshPartitionPolicy::getIndexOfGID(int myNode,FieldContainer<GlobalIndexType> &partitionedActiveCells,GlobalIndexType globalID){
//  int maxPartitionSize = partitionedActiveCells.dimension(1);
//  for (int i=0;i<maxPartitionSize;i++){
//    if (partitionedActiveCells(myNode,i)==globalID){
//      return i;
//    }
//  }
//  //  cout << "ZoltanMeshPartitionPolicy::getIndexOfGID - GlobalID not found, returning -1" << endl;
//  return -1;    
//}

/*-------------------------ZOLTAN QUERY FUNCTIONS -------------------*/

// get number of active elements
int ZoltanMeshPartitionPolicy::get_number_of_objects(void *data, int *ierr){
//  int myNode = Teuchos::GlobalMPISession::getRank();
//  int numNodes = Teuchos::GlobalMPISession::getNProc();

  Mesh* mesh = (Mesh*) data;
  MeshTopologyPtr meshTopo = mesh->getTopology();

  *ierr = ZOLTAN_OK;

  return getRankLocalCellIDs(mesh).size();
//  int maxPartitionSize = partitionedActiveCells.dimension(1);
//  int numActiveCellsInPartition = 0;
//  for (int i = 0;i<maxPartitionSize;i++){
//    if ((partitionedActiveCells(myNode,i))!=(-1)){
//      numActiveCellsInPartition++;
//    }
//  }  
//  //  cout << "--------------------" << endl;
//  //  cout << "Num active cells in node " << myNode << " is " << numActiveCellsInPartition << endl;
//  //  cout << "--------------------" << endl;
//  return numActiveCellsInPartition;//meshTopology->numElements();
}

void ZoltanMeshPartitionPolicy::get_object_list(void *data, int sizeGID, int sizeLID,
                                                ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                                                int wgt_dim, float *obj_wgts, int *ierr) {
  Mesh* mesh = (Mesh*) data;
  
  set<GlobalIndexType> rankLocalCellIDs = getRankLocalCellIDs(mesh);
  int i=0;
  for (set<unsigned>::const_iterator cellIDIt = rankLocalCellIDs.begin(); cellIDIt != rankLocalCellIDs.end(); cellIDIt++) {
    globalID[i]= *cellIDIt;
    i++;
  }
  //  cout << endl;
  //  cout << "--------------------" << endl;
  *ierr = ZOLTAN_OK;
  return; 
}

int ZoltanMeshPartitionPolicy::get_num_geom(void *data, int *ierr){
  Mesh *mesh = (Mesh*)data;
  MeshTopologyPtr meshTopology = mesh->getTopology();
  *ierr = ZOLTAN_OK;
  /*
   cout << "--------------------" << endl;
   cout << "dimensions : " << meshTopology->vertexCoordinates(0).dimension(0) << endl;
   cout << "--------------------" << endl;
   */
  return meshTopology->getSpaceDim(); // spatial dimension
}

// get a single coordinate identifying an element
void ZoltanMeshPartitionPolicy::get_geom_list(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr){
  
  Mesh *mesh = (Mesh*) data;
  
  // loop thru all objects
  for (int i=0;i<num_obj;i++){
    GlobalIndexType cellID = global_ids[i];
    vector<double> centroid = mesh->getTopology()->getCellCentroid(cellID);
    
    // store vertex centroid
    for (int k=0;k<num_dim;k++){
      geom_vec[i*num_dim+k] = centroid[k];
    }
  }
  *ierr = ZOLTAN_OK;
  return;
}

/* ------------------------REFTREE query functions -------------------*/

// num elems in initial meshTopology
/*int ZoltanMeshPartitionPolicy::get_num_coarse_elem(void *data, int *ierr){
  int myNode = Teuchos::GlobalMPISession::getRank();
  
  pair< Mesh *, FieldContainer<GlobalIndexType> * > *myData = (pair< Mesh *, FieldContainer<GlobalIndexType> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<GlobalIndexType>* partitionedCells = myData->second;
  *ierr = ZOLTAN_OK; 
  //  cout << "in num_coarse_elem fn, num coarse elems is " << meshTopology->numInitialElements() <<endl;
  MeshTopologyPtr meshTopology = mesh->getTopology();
  set<unsigned> rootCellIDSet = meshTopology->getRootCellIndices();
  vector<unsigned> rootCellIDs(rootCellIDSet.begin(),rootCellIDSet.end());
  int numTotalInitialElems = rootCellIDs.size();
  int numInitialElemsOnProc = 0;
  int maxPartitionSize = partitionedCells->dimension(1);
  for (int i=0;i < maxPartitionSize; i++){
    if ((*partitionedCells)(myNode,i)<=numTotalInitialElems){
      numInitialElemsOnProc++;
    }
  }
  return numTotalInitialElems;//numInitialElemsOnProc; // TODO: figure out which is needed
}

void ZoltanMeshPartitionPolicy::get_coarse_elem_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, int *in_order, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){
  //  cout << "in get_coarse_elem_list" << endl;
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< Mesh *, FieldContainer<GlobalIndexType> * > *myData = (pair< Mesh *, FieldContainer<GlobalIndexType> * > *)data;
  Mesh *mesh = myData->first;
  MeshTopologyPtr meshTopology = mesh->getTopology();
  FieldContainer<GlobalIndexType> partitionedCells = *(myData->second);  
  
  *ierr = ZOLTAN_OK;
  
  set<unsigned> rootCellIDSet = meshTopology->getRootCellIndices();
  vector<unsigned> rootCellIDs(rootCellIDSet.begin(),rootCellIDSet.end());
  int numInitialElems = rootCellIDs.size();
  
  int vertexInd=0;
  
  *in_order = 0;
  for (unsigned int i=0;i<numInitialElems;i++){        
    global_ids[i] = rootCellIDs[i];
    if (num_lid_entries>0){
      local_ids[i] = i;   
    }
    //    cout << "initial element " << i << endl;
    unsigned cellID = global_ids[i];
    vector<unsigned> vertIDs = meshTopology->getCell(cellID)->vertices();
    num_vert[i] = (int)vertIDs.size();
    
    //    cout << "vertices for cell " << global_ids[i] << " are ";
    for (int j=0; j<num_vert[i]; j++){
      //      cout << vertIDs[j] << ", ";
      vertices[vertexInd] = vertIDs[j];
      vertexInd++;
    }
    //    cout << endl;
    
    // determine whether on this processor (ignored if elem refined, in which case assigned[i]=0 anyways)
    assigned[i] = 0;
    
    if (getIndexOfGID(myNode,partitionedCells,global_ids[i])!=-1){
      assigned[i]=1;
      //      cout << "Object " << global_ids[i] << " is assigned to this proc" << endl;
    }
    
  }
  return;
}

int ZoltanMeshPartitionPolicy::get_num_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr){
  //  cout << "in get_num_children" << endl;
  
  pair< Mesh *, FieldContainer<GlobalIndexType> * > *myData = (pair< Mesh *, FieldContainer<GlobalIndexType> * > *)data;
  Mesh *mesh = myData->first;
  MeshTopologyPtr meshTopology = mesh->getTopology();

  int parentID = *global_id;
  CellPtr parentCell = meshTopology->getCell(parentID);
  int numChildren = parentCell->children().size();
  
  //  cout << "parent elem " << parentID << " has " << numChildren << " kids" << endl;
  *ierr = ZOLTAN_OK;
  return numChildren;
}

void ZoltanMeshPartitionPolicy::get_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR parent_gid, ZOLTAN_ID_PTR parent_lid, ZOLTAN_ID_PTR child_gids, ZOLTAN_ID_PTR child_lids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, ZOLTAN_REF_TYPE *ref_type, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){
  //  cout << "in get_children" << endl;
  
  int myNode = Teuchos::GlobalMPISession::getRank();
  
  pair< Mesh *, FieldContainer<GlobalIndexType> * > *myData = (pair< Mesh *, FieldContainer<GlobalIndexType> * > *)data;
  Mesh *mesh = myData->first;
  MeshTopologyPtr meshTopology = mesh->getTopology();
  FieldContainer<GlobalIndexType>* partitionedActiveCells = myData->second;
  
  int parentID = *parent_gid;
  CellPtr parentCell = meshTopology->getCell(parentID);
  int numChildren = parentCell->children().size();
  //  cout << "num children of " << parentID << " is: " << numChildren << endl;
  int vertexInd = 0;
  *ref_type = ZOLTAN_IN_ORDER;
  for (int i=0; i<numChildren; i++) {
    CellPtr child = parentCell->children()[i];
    int childID = child->cellIndex();
    child_gids[i]=childID;
    //    cout << "parent elem " << parentID << " has kid " << child_gids[i] << endl;
    if (num_lid_entries>0){
      child_lids[i]=i;
    }
    
    // query if this object is assigned to this processor
    assigned[i]=0;
    if (getIndexOfGID(myNode,*partitionedActiveCells,child_gids[i])!=-1){
      assigned[i]=1;
    }
    
    vector<unsigned> vertIDs = child->vertices();
    num_vert[i] = (int)vertIDs.size();
    for (int j=0;j<num_vert[i];j++){
      //      cout << "vertex ids for child element " << child_gids[i] << " is " << vertIDs[j]<<endl;
      vertices[vertexInd] = vertIDs[j];
      vertexInd++;
    }
  }  
  *ierr = ZOLTAN_OK;
}

void ZoltanMeshPartitionPolicy::get_child_weight(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim, float *obj_wgt, int *ierr){
  //  cout << "in get_child_weight" << endl;
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< Mesh *, FieldContainer<GlobalIndexType> * > *myData = (pair< Mesh *, FieldContainer<GlobalIndexType> * > *)data;
  Mesh *mesh = myData->first;
  MeshTopologyPtr meshTopology = mesh->getTopology();
  
  if (wgt_dim>0){
    obj_wgt[0]=1;
  }  
  *ierr = ZOLTAN_OK;
} */


int ZoltanMeshPartitionPolicy::get_elem_data_size(void *data,
                                                  int num_gid_entries,
                                                  int num_lid_entries,
                                                  ZOLTAN_ID_PTR global_id,
                                                  ZOLTAN_ID_PTR local_id,
                                                  int *ierr) {
  
  // returns size in bytes
  Mesh *mesh = (Mesh*) data;
  GlobalIndexType cellID = *global_id;
  *ierr = ZOLTAN_OK; // CellDataMigration throws exceptions if it's not OK
  return CellDataMigration::dataSize(mesh, cellID);
}
void ZoltanMeshPartitionPolicy::pack_elem_data(void *data,
                                               int num_gid_entries,
                                               int num_lid_entries,
                                               ZOLTAN_ID_PTR global_id,
                                               ZOLTAN_ID_PTR local_id,
                                               int dest,
                                               int size,
                                               char *buf,
                                               int *ierr) {
  Mesh *mesh = (Mesh*) data;
  GlobalIndexType cellID = *global_id;
  CellPtr cell = mesh->getTopology()->getCell(cellID);
  bool isChild = cell->getParent().get() != NULL;
  bool hasData = false;
  if (mesh->globalDofAssignment()->getRegisteredSolutions().size() > 0) {
    Solution* soln = mesh->globalDofAssignment()->getRegisteredSolutions()[0];
    hasData = soln->cellHasCoefficientsAssigned(cellID);
  }
  CellDataMigration::packData(mesh, cellID, isChild && !hasData, buf, size);
//  CellDataMigration::packData(mesh, cellID, false, buf, size);
  *ierr = ZOLTAN_OK; // CellDataMigration throws exceptions if it's not OK
}

void ZoltanMeshPartitionPolicy::unpack_elem_data(void *data,
                                                 int num_gid_entries,
                                                 ZOLTAN_ID_PTR global_id,
                                                 int size,
                                                 char *buf,
                                                 int *ierr) {
  Mesh *mesh = (Mesh*) data;
  GlobalIndexType cellID = *global_id;
  CellDataMigration::unpackData(mesh, cellID, buf, size);
  *ierr = ZOLTAN_OK; // CellDataMigration throws exceptions if it's not OK
}

set<GlobalIndexType> ZoltanMeshPartitionPolicy::getRankLocalCellIDs(Mesh *mesh) {
  MeshTopologyPtr meshTopo = mesh->getTopology();
  
  set<GlobalIndexType> rankLocalCells = mesh->globalDofAssignment()->cellsInPartition(-1);
  
  // this will include parents of refined children, but not the children.  We should eliminate the parents, add the children
  set<GlobalIndexType> parentCellIDs;
  for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCells.begin(); cellIDIt != rankLocalCells.end(); cellIDIt++) {
    CellPtr cell = meshTopo->getCell(*cellIDIt);
    if (cell->isParent()) {
      parentCellIDs.insert(*cellIDIt);
    }
  }
  for (set<GlobalIndexType>::iterator parentIDIt = parentCellIDs.begin(); parentIDIt != parentCellIDs.end(); parentIDIt++) {
    rankLocalCells.erase(*parentIDIt);
    CellPtr parent = meshTopo->getCell(*parentIDIt);
    vector<IndexType> childIDs = parent->getChildIndices();
    for (vector<IndexType>::iterator childIDIt = childIDs.begin(); childIDIt != childIDs.end(); childIDIt++) {
      rankLocalCells.insert(*childIDIt);
    }
  }
  return rankLocalCells;
}