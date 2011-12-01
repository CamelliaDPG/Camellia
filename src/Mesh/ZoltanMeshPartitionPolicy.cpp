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

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#endif

ZoltanMeshPartitionPolicy::ZoltanMeshPartitionPolicy(){
  string partitionerName = "BLOCK";
  string debug_level = "10";
  cout << "Defaulting to block partitioner" << endl;
  _ZoltanPartitioner = partitionerName;
  _debug_level = debug_level;
}
ZoltanMeshPartitionPolicy::ZoltanMeshPartitionPolicy(string partitionerName){
  string debug_level = "0";
  _ZoltanPartitioner = partitionerName;  
  _debug_level = debug_level;
}

void ZoltanMeshPartitionPolicy::partitionMesh(Mesh *mesh, int numPartitions, FieldContainer<int> &partitionedActiveCells) {

  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif  
  
  TEST_FOR_EXCEPTION(numPartitions != partitionedActiveCells.dimension(0), std::invalid_argument,"numPartitions must match the first dimension of partitionedActiveCells");
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  int numActiveElements = mesh->activeElements().size();
  TEST_FOR_EXCEPTION(numActiveElements > maxPartitionSize, std::invalid_argument,"second dimension of partitionedActiveCells must be at least as large as the number of active cells.");
  
  partitionedActiveCells.initialize(-1); // cellID == -1 signals end of partition

  float version;
  //these arguments are ignored by Zoltan initialize if MPI_init is called
  int argc = 0;

  Zoltan_Initialize(argc, NULL, &version);
  
  Zoltan *zz = new Zoltan(MPI::COMM_WORLD);
  if (zz == NULL){
    MPI::Finalize();
    exit(0);
  }

  // store all nodes on first processor to start
  if (myNode==0){
    int activeCellIndex = 0;
    for (int i=0;i<numActiveElements;i++){
      partitionedActiveCells(0,i) = mesh->activeElements()[activeCellIndex]->cellID();
      //cout << "storing elem " << partitionedActiveCells(0,i) << " for node 0" << endl;
      activeCellIndex++;
    }
  }
    
  /* Calling Zoltan Load-balancing routine */
  cout << "Setting zoltan params" << endl;
  zz->Set_Param( "LB_METHOD", _ZoltanPartitioner.c_str());    /* Zoltan method */
  //zz->Set_Param( "LB_METHOD", "REFTREE");
  zz->Set_Param( "NUM_GID_ENTRIES", "1");  /* global ID is 1 integer */
  zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is 1 integer */
  zz->Set_Param( "OBJ_WEIGHT_DIM", "0");   
  zz->Set_Param( "DEBUG_LEVEL", "0");
  zz->Set_Param( "REFTREE_INITPATH", "CONNECTED"); // no SFC on coarse mesh
  zz->Set_Param( "RANDOM_MOVE_FRACTION", "1.0");    /* Zoltan "random" partition param */
  
  pair< Mesh *, FieldContainer<int> * > myData;
  myData.first = mesh;
  myData.second = &partitionedActiveCells;
  
  cout << "Setting zoltan query functions" << endl;
  // General query functions
  zz->Set_Num_Obj_Fn(&get_number_of_objects, &myData);
  zz->Set_Obj_List_Fn(&get_object_list, &myData);
  
  // HSFC query functions   
  zz->Set_Num_Geom_Fn(&get_num_geom, &myData);
  zz->Set_Geom_Multi_Fn(&get_geom_list, &myData);
  
  // reftree query functions
  zz->Set_Num_Coarse_Obj_Fn(&get_num_coarse_elem, &myData);
  zz->Set_Coarse_Obj_List_Fn(&get_coarse_elem_list, &myData);
  zz->Set_Num_Child_Fn(&get_num_children, &myData);
  zz->Set_Child_List_Fn(&get_children, &myData);
  zz->Set_Child_Weight_Fn(&get_child_weight, &myData);  
  
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

  if (rc != ZOLTAN_OK){
    printf("Partitioning failed on process %d\n",myNode);
    //    exit(0);
  }else{
    
    vector<int> globalIDs = getListOfActiveGlobalIDs(partitionedActiveCells);
    cout << "Global IDs for node " << myNode << " are " << endl;
    for (unsigned int i=0;i<globalIDs.size();i++){
      cout << globalIDs[i] << ", ";
    }
    cout <<endl;
    
    cout << "For node: " << myNode << ", num exported gids: " << numExport << endl;
    cout << "For node: " << myNode << ", exported globalIDs are " << endl;
    for (int i=0;i<numExport;i++){
      cout << exportGlobalIds[i] << ", "<< endl;
    } 
    
    cout << "For node: " << myNode << ", num imported gids: " << numImport << endl;
    cout << "For node: " << myNode << ", imported globalIDs should be " << endl;
    for (int i=0;i<numImport;i++){
      cout << importGlobalIds[i] << ", " << endl;
    }
  } 
  delete zz;
  
}    

// for current node, get active globalIDs
vector<int> ZoltanMeshPartitionPolicy::getListOfActiveGlobalIDs(FieldContainer<int> partitionedActiveCells){
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  vector<int> globalIDs;
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  for (int i=0;i<maxPartitionSize;i++){
    if (partitionedActiveCells(myNode,i)!=-1){
      globalIDs.push_back(partitionedActiveCells(myNode,i));
    }
  }  
  return globalIDs;
}

int ZoltanMeshPartitionPolicy::getNextActiveIndex(FieldContainer<int> &partitionedActiveCells){
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  for (int i=0;i<maxPartitionSize;i++){
    if (partitionedActiveCells(myNode,i)==-1){
      return i;
    }
  } 
}

int ZoltanMeshPartitionPolicy::getIndexOfGID(int myNode,FieldContainer<int> &partitionedActiveCells,int globalID){
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  for (int i=0;i<maxPartitionSize;i++){
    if (partitionedActiveCells(myNode,i)==globalID){
      return i;
    }
  }
  //  cout << "ZoltanMeshPartitionPolicy::getIndexOfGID - GlobalID not found, returning -1" << endl;
  return -1;    
}

/*-------------------------ZOLTAN QUERY FUNCTIONS -------------------*/

// get number of active elements
int ZoltanMeshPartitionPolicy::get_number_of_objects(void *data, int *ierr){
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);

  //  *ierr = ZOLTAN_OK;

  int numPartitions = partitionedActiveCells.dimension(0);
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  int numActiveCellsInPartition = 0;
  for (int i = 0;i<maxPartitionSize;i++){
    if ((partitionedActiveCells(myNode,i))!=(-1)){
      numActiveCellsInPartition++;
    }
  }  
  cout << "--------------------" << endl;
  cout << "Num active cells in node " << myNode << " is " << numActiveCellsInPartition << endl;
  cout << "--------------------" << endl;
  return numActiveCellsInPartition;//mesh->numElements();
}

void ZoltanMeshPartitionPolicy::get_object_list(void *data, int sizeGID, int sizeLID,ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,int wgt_dim, float *obj_wgts, int *ierr){
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);
  vector <Teuchos::RCP< Element > > elements = mesh->elements();
  int *ierr2;
  int num_obj = get_number_of_objects(data, ierr2);
  cout << "--------------------" << endl;
  cout << "objects in node " << myNode << " are ";
  for (int i=0;i<num_obj;i++){
    globalID[i]= partitionedActiveCells(myNode,i);
    cout << globalID[i] << ", ";
    localID[i] = i;
  }
  cout << endl;
  cout << "--------------------" << endl;
  //  *ierr = ZOLTAN_OK;
  return; 
}

int ZoltanMeshPartitionPolicy::get_num_geom(void *data, int *ierr){
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);
  //  *ierr = ZOLTAN_OK;
  cout << "--------------------" << endl;
  cout << "dimensions : " << mesh->vertexCoordinates(0).dimension(0) << endl;
  cout << "--------------------" << endl;
  return mesh->vertexCoordinates(0).dimension(0); // spatial dimension  
}

// get a single coordinate identifying an element
void ZoltanMeshPartitionPolicy::get_geom_list(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr){  
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);

  // loop thru all objects
  for (int i=0;i<num_obj;i++){
    FieldContainer<double> verts; // gets resized inside verticesForCell
    mesh->verticesForCell(verts, global_ids[i]);    
    //average vertex positions together to get a centroid (avoids using vertex in case local enumeration overlaps)
    int numVertices = verts.dimension(0);
    vector<double> coords(num_dim,0.0);
    cout << "Centroid for GID " << global_ids[i] << " is at ";
    for (int k=0;k<num_dim;k++){
      for (int j=0;j<numVertices;j++){
	coords[k] += verts(j,k);
      }
      coords[k] = coords[k]/((double)(numVertices));
      cout << coords[k] << ", ";
    }
    cout << endl;
    
    // store vertex centroid
    for (int k=0;k<num_dim;k++){
      geom_vec[i*num_dim+k] = coords[k];
    }
  }
  //  *ierr = ZOLTAN_OK;
  return;
}

/* ------------------------REFTREE query functions -------------------*/

// num elems in initial mesh
int ZoltanMeshPartitionPolicy::get_num_coarse_elem(void *data, int *ierr){   
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);
  //  *ierr = ZOLTAN_OK; 
  cout << "in num_coarse_elem fn, num coarse elems is " << mesh->numInitialElements() <<endl;
  return mesh->numInitialElements();
}

void ZoltanMeshPartitionPolicy::get_coarse_elem_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, int *in_order, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){
  cout << "in get_coarse_elem_list" << endl;
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);  
  //  *ierr = ZOLTAN_OK;

  int numInitialElems = mesh->numInitialElements();
  vector< Teuchos::RCP< Element > > elems = mesh->activeElements();
  
  int vertexInd=0;
  for (unsigned int i=0;i<numInitialElems;i++){    
    global_ids[i] = elems[i]->cellID(); //equivalent to global_ids[i] = i, but safer.  
    local_ids[i] = i;
    in_order[i] = 0;    

    vector<int> vertIDs = mesh->vertexIndicesForCell(global_ids[i]);
    num_vert[i] = (int)vertIDs.size();

    cout << "vertices for cell " << global_ids[i] << " are ";
    for (int j=0; j<num_vert[i]; j++){
      cout << vertIDs[j] << ", ";
      vertices[vertexInd] = vertIDs[j];
      vertexInd++;
    }
    cout << endl;
    
    // determine whether on this processor (ignored if elem refined, in which case assigned[i]=0 anyways)
    assigned[i] = 0;
    if (getIndexOfGID(myNode,partitionedActiveCells,global_ids[i])!=-1){
      assigned[i]=1;
      //      cout << "Object " << global_ids[i] << " is assigned to this proc" << endl;
    }
  }
  return;
}

int ZoltanMeshPartitionPolicy::get_num_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr){
  cout << "in get_num_children" << endl;
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);

  vector< Teuchos::RCP< Element > > elems = mesh->elements();  
  int parentID = *global_id;
  int numChildren = elems[parentID]->numChildren();

  cout << "parent elem " << parentID << " has " << numChildren << " kids" << endl;
  //  *ierr = ZOLTAN_OK;
  return numChildren;
}

void ZoltanMeshPartitionPolicy::get_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR parent_gid, ZOLTAN_ID_PTR parent_lid, ZOLTAN_ID_PTR child_gids, ZOLTAN_ID_PTR child_lids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, ZOLTAN_REF_TYPE *ref_type, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){
  cout << "in get_children" << endl;

  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);
  
  vector< Teuchos::RCP< Element > > elems = mesh->elements();
  int parentID = *parent_gid;
  int numChildren = elems[parentID]->numChildren();
  cout << "num children of " << parentID << " is: " << numChildren << endl;
  int vertexInd = 0;
  for (int i=0; i<numChildren; i++) {
    int childID = elems[parentID]->getChild(i)->cellID();
    child_gids[i]=childID;
    cout << "parent elem " << parentID << " has kid " << child_gids[i] << endl;
    child_lids[i]=i;

    // query if this object is assigned to this processor
    assigned[i]=0;
    if (getIndexOfGID(myNode,partitionedActiveCells,child_gids[i])!=-1){      
      assigned[i]=1;
    }

    vector<int> vertIDs = mesh->vertexIndicesForCell(child_gids[i]);
    num_vert[i] = (int)vertIDs.size();
    for (int j=0;j<num_vert[i];j++){
      cout << "vertex ids for element " << child_gids[i] << " is " << vertIDs[j]<<endl;
      vertices[vertexInd] = vertIDs[j];
      vertexInd++;
    }
    ref_type[i] = ZOLTAN_IN_ORDER;
  }  
  //  *ierr = ZOLTAN_OK;
}

void ZoltanMeshPartitionPolicy::get_child_weight(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim, float *obj_wgt, int *ierr){
  cout << "in get_child_weight" << endl;
  int myNode = 0;
  int numNodes = 1;
#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif 
  pair< Mesh *, FieldContainer<int> * > *myData = (pair< Mesh *, FieldContainer<int> * > *)data;
  Mesh *mesh = myData->first;
  FieldContainer<int> partitionedActiveCells = *(myData->second);

  if (wgt_dim>0){
    obj_wgt[0]=1;
  }  
  //  *ierr = ZOLTAN_OK;
}
