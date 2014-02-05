//
//  ZoltanMeshPartitionPolicy.cpp
//  Camellia
//
//  Copyright 2011 Nathan Roberts. All rights reserved.
//

#include <iostream>
#include <stdlib.h> 
#include <vector>
#include "MeshTopology.h"
#include "ZoltanMeshPartitionPolicy.h"

#include <Teuchos_GlobalMPISession.hpp>

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

void ZoltanMeshPartitionPolicy::partitionMesh(MeshTopology *meshTopology, PartitionIndexType numPartitions, FieldContainer<GlobalIndexType> &partitionedActiveCells) {
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  TEUCHOS_TEST_FOR_EXCEPTION(numPartitions != partitionedActiveCells.dimension(0), std::invalid_argument,"numPartitions must match the first dimension of partitionedActiveCells");
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  int numActiveElements = meshTopology->activeCellCount();
  TEUCHOS_TEST_FOR_EXCEPTION(numActiveElements > maxPartitionSize, std::invalid_argument,"second dimension of partitionedActiveCells must be at least as large as the number of active cells.");
  
  partitionedActiveCells.initialize(-1); // cellID == -1 signals end of partition
  
  float version;
  //these arguments are ignored by Zoltan initialize if MPI_init is called
  int argc = 0;
  Zoltan_Initialize(argc, NULL, &version);
  
  if (numNodes>1){
#ifdef HAVE_MPI
    Zoltan *zz = new Zoltan(MPI::COMM_WORLD);
    if (zz == NULL){
      cout << "ZoltanMeshPartititionPolicy: construction of new Zoltan object failed.\n";
      MPI::Finalize();
      exit(0);
    }
    
    // store all nodes on first processor to start
    FieldContainer<GlobalIndexType> partitionedInitialCells(numNodes,maxPartitionSize);
    partitionedInitialCells.initialize(-1);
    set<GlobalIndexType> activeCellIDSet = meshTopology->getActiveCellIndices();
    vector<GlobalIndexType> activeCellIDs(activeCellIDSet.begin(), activeCellIDSet.end());
    if (myNode==0){
      int activeCellIndex = 0;
      for (int i=0;i<numActiveElements;i++){
        partitionedActiveCells(0,i) = activeCellIDs[activeCellIndex];
        //cout << "storing elem " << partitionedActiveCells(0,i) << " for node 0" << endl;
        activeCellIndex++;
      }
      
      for (int i=0;i<meshTopology->getRootCellIndices().size();i++){
        partitionedInitialCells(0,i) = i; //assign ALL THE ELEMENTS
      }    
    }
    
    bool useLocalIDs = false;
    
    /* Calling Zoltan Load-balancing routine */
    //cout << "Setting zoltan params" << endl;
    zz->Set_Param( "LB_METHOD", _ZoltanPartitioner.c_str());    /* Zoltan method */
    zz->Set_Param( "NUM_GID_ENTRIES", "1");  /* global ID is 1 integer */
    if (useLocalIDs){
      zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is 1 integer */
    }else{
      zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is null */
    }
    zz->Set_Param( "OBJ_WEIGHT_DIM", "0");   
    zz->Set_Param( "DEBUG_LEVEL", "0");
    //  zz->Set_Param( "REFTREE_INITPATH", "CONNECTED"); // no SFC on coarse meshTopology
    zz->Set_Param( "RANDOM_MOVE_FRACTION", "1.0");    /* Zoltan "random" partition param */
    
    pair< MeshTopology *, FieldContainer<GlobalIndexType> * > myData;
    myData.first = meshTopology;
    myData.second = &partitionedActiveCells;
    
    pair< MeshTopology *, FieldContainer<GlobalIndexType> * > myCoarseData;
    myCoarseData.first = meshTopology;
    myCoarseData.second = &partitionedInitialCells;
    
    //cout << "Setting zoltan query functions" << endl;
    
    // Testing query functions
    zz->Set_Num_Obj_Fn(&get_number_of_objects, &myData);
    zz->Set_Obj_List_Fn(&get_object_list, &myData);
    
    // HSFC query functions   
    zz->Set_Num_Geom_Fn(&get_num_geom, &myData);
    zz->Set_Geom_Multi_Fn(&get_geom_list, &myData);
    
    // reftree query functions
    zz->Set_Num_Coarse_Obj_Fn(&get_num_coarse_elem, &myCoarseData);
    zz->Set_Coarse_Obj_List_Fn(&get_coarse_elem_list, &myCoarseData);
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
      
      /* ----------- modify output array partitionedActiveCells ------- */
      
      // remove export IDs FOR THIS NODE
      for (int i=0;i<numExport;i++){
        for (int j = 0;j<maxPartitionSize;j++){
          if (partitionedActiveCells(myNode,j)==exportGlobalIds[i]){
            partitionedActiveCells(myNode,j) = -1;
          }
        }
      }
      // add import IDs FOR THIS NODE
      for (int i=0;i<numImport;i++){
        bool haveSet = false;
        for (int j = 0;j<maxPartitionSize;j++){
          if ((partitionedActiveCells(myNode,j)==-1)&&(!haveSet)){
            partitionedActiveCells(myNode,j) = importGlobalIds[i];
            haveSet = true;
          }
        }
      }   
      // compute total number of IDs for this proc
      int numIDsForThisNode = 0;    
      for (int j = 0;j<maxPartitionSize;j++){
        if (partitionedActiveCells(myNode,j)!=-1){
          numIDsForThisNode++;
        }      
      }       
      
      if (numNodes>1){
        // need to pass around information about partitions here thru MPI - each processors must know all other processors' partitions
        
        int sendbuf[maxPartitionSize];
        int recvbuf[numNodes][maxPartitionSize];
        for (int i=0;i<maxPartitionSize;i++){
          sendbuf[i] = partitionedActiveCells(myNode,i);
        }
        /*
         cout << "for this node, partitioned cells = " ;      
         for (int i=0;i<maxPartitionSize;i++){
         if (partitionedActiveCells(myNode,i)!=-1){
         cout << partitionedActiveCells(myNode,i) << ", ";
         }
         }	
         cout << endl;
         */
        MPI::COMM_WORLD.Allgather(sendbuf,maxPartitionSize, MPI::INT,recvbuf, maxPartitionSize , MPI::INT);      
        
        for (int node=0;node<numNodes;node++){
          vector<int> activeCellVec(maxPartitionSize);
          for (int i=0;i<maxPartitionSize;i++){
            activeCellVec[i] = recvbuf[node][i];
          }	
          sort(activeCellVec.begin(), activeCellVec.end(), greater<int>());
          for (int i=0;i<maxPartitionSize;i++){
            partitionedActiveCells(node,i)=activeCellVec[i];
          }	       
        }           
      }
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
    set<unsigned> activeCellIDSet = meshTopology->getActiveCellIndices();
    vector< unsigned > activeCellIDs(activeCellIDSet.begin(), activeCellIDSet.end());
    for (int i=0;i<numActiveElements;i++){
      //    for (vector<Teuchos::RCP< Element > >::iterator elemIt=activeElements.begin();elemIt!=activeElements.end();elemIt++){
      partitionedActiveCells(0,i) = activeCellIDs[i];
    }
  }  
}    

GlobalIndexType ZoltanMeshPartitionPolicy::getIndexOfGID(int myNode,FieldContainer<GlobalIndexType> &partitionedActiveCells,GlobalIndexType globalID){
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
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  FieldContainer<GlobalIndexType> partitionedActiveCells = *(myData->second);
  
  *ierr = ZOLTAN_OK;
  
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  int numActiveCellsInPartition = 0;
  for (int i = 0;i<maxPartitionSize;i++){
    if ((partitionedActiveCells(myNode,i))!=(-1)){
      numActiveCellsInPartition++;
    }
  }  
  //  cout << "--------------------" << endl;
  //  cout << "Num active cells in node " << myNode << " is " << numActiveCellsInPartition << endl;
  //  cout << "--------------------" << endl;
  return numActiveCellsInPartition;//meshTopology->numElements();
}

void ZoltanMeshPartitionPolicy::get_object_list(void *data, int sizeGID, int sizeLID,ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,int wgt_dim, float *obj_wgts, int *ierr){
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
  FieldContainer<GlobalIndexType> partitionedActiveCells = *(myData->second);
  
  set<unsigned> activeCellIDSet = meshTopology->getActiveCellIndices();
  vector<unsigned> activeCellIDs(activeCellIDSet.begin(),activeCellIDSet.end());
  
  int num_obj = activeCellIDs.size();//get_number_of_objects(data, ierr2);
  //  cout << "--------------------" << endl;
  //  cout << "objects in node " << myNode << " are ";
  for (int i=0;i<num_obj;i++){
    globalID[i]= partitionedActiveCells(myNode,i);
    //    cout << globalID[i] << ", ";
    if (sizeLID>0){
      localID[i] = i;
    }
  }
  //  cout << endl;
  //  cout << "--------------------" << endl;
  *ierr = ZOLTAN_OK;
  return; 
}

int ZoltanMeshPartitionPolicy::get_num_geom(void *data, int *ierr){
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
  FieldContainer<GlobalIndexType> partitionedActiveCells = *(myData->second);
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
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
  FieldContainer<GlobalIndexType> partitionedActiveCells = *(myData->second);
  
  // loop thru all objects
  for (int i=0;i<num_obj;i++){
    unsigned cellID = global_ids[i];
    vector<double> centroid = meshTopology->getCellCentroid(cellID);
    
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
int ZoltanMeshPartitionPolicy::get_num_coarse_elem(void *data, int *ierr){   
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
  FieldContainer<GlobalIndexType> partitionedCells = *(myData->second);
  *ierr = ZOLTAN_OK; 
  //  cout << "in num_coarse_elem fn, num coarse elems is " << meshTopology->numInitialElements() <<endl;
  set<unsigned> rootCellIDSet = meshTopology->getRootCellIndices();
  vector<unsigned> rootCellIDs(rootCellIDSet.begin(),rootCellIDSet.end());
  int numTotalInitialElems = rootCellIDs.size();
  int numInitialElemsOnProc = 0;
  int maxPartitionSize = partitionedCells.dimension(1);
  for (int i=0;i < maxPartitionSize; i++){
    if (partitionedCells(myNode,i)<=numTotalInitialElems){
      numInitialElemsOnProc++;
    }
  }
  return numTotalInitialElems;//numInitialElemsOnProc; // TODO: figure out which is needed
}

void ZoltanMeshPartitionPolicy::get_coarse_elem_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, int *in_order, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){
  //  cout << "in get_coarse_elem_list" << endl;
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
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
  int myNode = Teuchos::GlobalMPISession::getRank();
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
  FieldContainer<GlobalIndexType> partitionedActiveCells = *(myData->second);

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
  int numNodes = Teuchos::GlobalMPISession::getNProc();
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
  FieldContainer<GlobalIndexType> partitionedActiveCells = *(myData->second);
  
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
    if (getIndexOfGID(myNode,partitionedActiveCells,child_gids[i])!=-1){      
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
  
  pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *myData = (pair< MeshTopology *, FieldContainer<GlobalIndexType> * > *)data;
  MeshTopology *meshTopology = myData->first;
  FieldContainer<GlobalIndexType> partitionedActiveCells = *(myData->second);
  
  if (wgt_dim>0){
    obj_wgt[0]=1;
  }  
  *ierr = ZOLTAN_OK;
}
