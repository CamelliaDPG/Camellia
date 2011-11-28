//
//  ZoltanMeshPartitionPolicy.cpp
//  Camellia
//
//  Copyright 2011 Nathan Roberts. All rights reserved.
//
//#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
//#endif

#include <iostream>
#include "ZoltanMeshPartitionPolicy.h"

ZoltanMeshPartitionPolicy::ZoltanMeshPartitionPolicy(string partitionerName){
  _ZoltanPartitioner = partitionerName;  
}
void ZoltanMeshPartitionPolicy::partitionMesh(Mesh *mesh, int numPartitions, FieldContainer<int> &partitionedActiveCells) {
  int myNode = 0;
  int numNodes = 1;

#ifdef HAVE_MPI
  myNode   = Teuchos::GlobalMPISession::getRank();
  numNodes = Teuchos::GlobalMPISession::getNProc();
#endif  
  
  TEST_FOR_EXCEPTION(numPartitions != partitionedActiveCells.dimension(0), std::invalid_argument,
                     "numPartitions must match the first dimension of partitionedActiveCells");
  int maxPartitionSize = partitionedActiveCells.dimension(1);
  int numActiveElements = mesh->activeElements().size();
  TEST_FOR_EXCEPTION(numActiveElements > maxPartitionSize, std::invalid_argument,
                     "second dimension of partitionedActiveCells must be at least as large as the number of active cells.");
  
  partitionedActiveCells.initialize(-1); // cellID == -1 signals end of partition

  float version;
  //these arguments are ignored by Zoltan initialize if MPI_init is called
  int argc = 0;
  char **argv;
  Zoltan_Initialize(argc, argv, &version);
  
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
      activeCellIndex++;
    }
  }

  /* Calling Zoltan Load-balancing routine */
  zz->Set_Param( "LB_METHOD", _ZoltanPartitioner.c_str());    /* Zoltan method */
  //  zz->Set_Param( "LB_METHOD", "HSFC");    /* Zoltan method */
  //  zz->Set_Param( "LB_METHOD", "REFTREE");    /* Zoltan method */
  zz->Set_Param( "RANDOM_MOVE_FRACTION", ".5");    /* Zoltan "random" partition param */
  zz->Set_Param( "NUM_GID_ENTRIES", "1");  /* global ID is 1 integer */
  zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is 1 integer */
  zz->Set_Param( "OBJ_WEIGHT_DIM", "0");   /* we omit object weights */
  zz->Set_Param( "DEBUG_LEVEL", "10");   /* no output */
  zz->Set_Param( "REFTREE_INITPATH", "CONNECTED"); // no SFC on coarse mesh

  // General query functions
  zz->Set_Num_Obj_Fn(get_number_of_objects, mesh);
  zz->Set_Obj_List_Fn(get_object_list, mesh);

  // HSFC query functions   
  zz->Set_Num_Geom_Fn(get_num_geom, mesh);
  zz->Set_Geom_Multi_Fn(get_geom_list, mesh);

  // reftree query functions
  zz->Set_Num_Coarse_Obj_Fn(get_num_coarse_elem, mesh);
  zz->Set_Coarse_Obj_List_Fn(get_coarse_elem_list, mesh);
  zz->Set_Num_Child_Fn(get_num_children, mesh);
  zz->Set_Child_List_Fn(get_children, mesh);
  zz->Set_Child_Weight_Fn(get_child_weight, mesh);

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
    //    printf("Partitioning failed on process %d\n",rank);
    delete zz;
    MPI::Finalize();
    exit(0);
  }

  delete zz;
  MPI::Finalize();

}    

static int get_number_of_objects(void *data, int *ierr){
  Mesh *mesh = (Mesh *)data;   
  return mesh->numElements();
}

static void get_object_list(void *data, int sizeGID, int sizeLID,ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,int wgt_dim, float *obj_wgts, int *ierr){
  
}

static int get_num_geom(void *data, int *ierr){
  return -1;
}

static void get_geom_list(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr){  
}


static void get_geom_fn(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, double *geom_vec, int *ierr){
}

static int get_num_coarse_elem(void *data, int *ierr){
  return -1;
}

static void get_coarse_elem_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, int *in_order, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){
  
}

static int get_num_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr){
  return -1;
}

static void get_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR parent_gid, ZOLTAN_ID_PTR parent_lid, ZOLTAN_ID_PTR child_gids, ZOLTAN_ID_PTR child_lids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, ZOLTAN_REF_TYPE *ref_type, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){

}

static void get_child_weight(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim, float *obj_wgt, int *ierr){

}
