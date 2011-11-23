#ifdef MPICPP
#undef MPICPP
#endif /* MPICPP */

#define MPICPP // Uncomment to use C++ interface for MPI.

#include <iostream>
#include <vector>
#include <mpi.h>
#include <stdio.h>
#include <zoltan_cpp.h>

using namespace std;

// Class representing collection of objects to be partitioned.
class element{
private: 
  vector<int> _vertices;
  int _parentID;
  vector<int> _childIDs; // vector of child IDs
  
public:
  element(vector<int> vertices){
    _vertices = vertices;
    _childIDs.clear();//init to size 0
  };
  void addChild(int newChildID){
    //    cout << "adding child " << newChildID << endl;
    _childIDs.push_back(newChildID);    
  }; 
  int numChildren(){
    return _childIDs.size();
  }
  vector<int> children(){
    return _childIDs;
  }
  void setParent(int parentID){
    _parentID = parentID;
  }
  vector<int> vertices(){
    return _vertices;
  }
};

// every processor owns the entire mesh, but is only responsible for the elements in _myPartitionGlobalIDs
class exampleSquareMesh{
private:
  int _numCoarseElements; 
  int _numElements; 
  int _numVertices;
  int *_globalIDs;
  vector<element> _elements;  
  vector<int> _myPartitionGlobalIDs;  //vector of global IDs for this given partition
  
public: 
  //constructor
  exampleSquareMesh(){
    _numCoarseElements = 4;
    _numElements = _numCoarseElements; 
    _numVertices = 9;
    vector<int> vertexIDs;

    // first element
    vertexIDs.push_back(0);
    vertexIDs.push_back(1);
    vertexIDs.push_back(4);
    vertexIDs.push_back(3);   
    _elements.push_back(element(vertexIDs));
    vertexIDs.clear();

    // second element
    vertexIDs.push_back(1);
    vertexIDs.push_back(2);
    vertexIDs.push_back(5);
    vertexIDs.push_back(4);       
    _elements.push_back(element(vertexIDs));
    vertexIDs.clear();

    // third element
    vertexIDs.push_back(4);
    vertexIDs.push_back(5);
    vertexIDs.push_back(8);
    vertexIDs.push_back(7);       
    _elements.push_back(element(vertexIDs));
    vertexIDs.clear();


    // fourth element
    vertexIDs.push_back(3);
    vertexIDs.push_back(4);
    vertexIDs.push_back(7);
    vertexIDs.push_back(6);       
    _elements.push_back(element(vertexIDs));
    vertexIDs.clear();        
    
  };
  
  void refineElement(int elemID){
    element elem = _elements[elemID];
    vector<int> oldVertices = elem.vertices();
    /*
    cout << "Old vertices = ";
    for (int i = 0;i<4;i++){
      cout << oldVertices[i];
    }
    cout << endl;
    cout << "Refining element: " << elemID << endl;
    cout << "Number of elements: " << _numElements << endl;
    */
    for (int i=0;i<4;i++){
      
      int newElemID = _numElements+1; // new element ID = next element number
      //      cout << "adding " << newElemID << " child to elemID = " << elemID << endl;
      _elements[elemID].addChild(newElemID);

      vector<int> newVertices; 
      // numbering of new verts is counterclockwise from bottom
      switch(i){
      case 0:
	newVertices.push_back(oldVertices[0]);
	newVertices.push_back(_numVertices+1);
	newVertices.push_back(_numVertices+5);
	newVertices.push_back(_numVertices+4);
	break;
      case 1:
	newVertices.push_back(_numVertices+1);
	newVertices.push_back(oldVertices[1]);
	newVertices.push_back(_numVertices+2);
	newVertices.push_back(_numVertices+5);
	break;
      case 2:
	newVertices.push_back(_numVertices+5);
	newVertices.push_back(_numVertices+2);
	newVertices.push_back(oldVertices[2]);
	newVertices.push_back(_numVertices+3);
	break;
      case 3:
	newVertices.push_back(_numVertices+4);
	newVertices.push_back(_numVertices+4);
	newVertices.push_back(_numVertices+3);
	newVertices.push_back(oldVertices[3]);
	break;
      };
      element newElem = element(newVertices);
      _elements.push_back(newElem);
      newVertices.clear();
      _numVertices+=5; // add 5 new vertices in an isotropic refinement
      _numElements += 1;
    }
    //    cout << "New number of elements: " << _numElements << endl;       
  }

  void printElements(){
    for (int i = 0;i < _numElements; i++){
      element elem = _elements[i];
      if (elem.numChildren()==0){
	cout << "Elem " << i << " has 0 children" << endl;
      }else{
	vector<int> children = elem.children();
	cout << "Elem " << i << " has " << children.size() << " children, who are ";
	for (unsigned int j = 0; j < children.size();j++){
	  cout << children[j] << " ";
	}
	cout << endl;
      }     
    }
  }
  //gets activeElem(index), or "index"th active elem
  int getActiveElemGlobalIndex(int index){
    int numActive = 0;
    for (int i=0;i<_numElements;i++){
      if (isActiveElem(i)){
	if (numActive == index){
	  return i;
	}else{
	  numActive++;
	}
      }       
    }   
    return -1;
  }

  int numActiveElems(){
    int numActive = 0;
    for (int i=0;i<_numElements;i++){
      if (isActiveElem(i)){
	numActive++;
      } 
    }
    return numActive;
  }

  int numElems(){
    return _numElements;
  }

  int numCoarseElems(){
    return _numCoarseElements;
  }

  bool isActiveElem(int elemID){
    bool isActive = false;
    if (_elements[elemID].numChildren()==0){
      isActive = true;
    }
    return isActive;    
  }

  // add a global ID to this partition
  void addPartitionGlobalID(int globalID){
    _myPartitionGlobalIDs.push_back(globalID);
    return;
  }

  int numPartitionGlobalIDs(){
    return _myPartitionGlobalIDs.size();
  }

  // get ith global id back
  int getPartitionGlobalID(int index){
    return _myPartitionGlobalIDs[index];
  }

  void printPartitionIDs(){
    cout << "Partition IDs are: " << endl;
    for (unsigned int i = 0;i<_myPartitionGlobalIDs.size();i++){
      cout << _myPartitionGlobalIDs[i] << endl;
    }
    return;
  }
 
  //------------------------- zoltan interface functions --------------------------------------------

  static int get_number_of_objects(void *data, int *ierr){

    exampleSquareMesh *mesh = (exampleSquareMesh *)data;
    *ierr = ZOLTAN_OK;

    return mesh->numPartitionGlobalIDs();
  }

 
  static void get_object_list(void *data, int sizeGID, int sizeLID,
            ZOLTAN_ID_PTR globalID, ZOLTAN_ID_PTR localID,
                  int wgt_dim, float *obj_wgts, int *ierr){

    exampleSquareMesh *mesh = (exampleSquareMesh *)data;
    *ierr = ZOLTAN_OK;

    // In this example, return the IDs of our objects, but no weights.
    // Zoltan will assume equally weighted objects.

    for (int i=0; i<mesh->numPartitionGlobalIDs(); i++){
      globalID[i] = mesh->getPartitionGlobalID(i);
      localID[i] = i;
    }
    return;
  }

  // ----------------BELOW FUNCTIONS: interface functions for reftree, not working --------------


  static int get_num_coarse_elem(void *data, int *ierr){
    exampleSquareMesh *mesh = (exampleSquareMesh *)data;
    *ierr = ZOLTAN_OK;
    return mesh->numCoarseElems();
  }

  static void get_coarse_elem_list(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, int *in_order, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){

    exampleSquareMesh *mesh = (exampleSquareMesh *)data;   
    *ierr = ZOLTAN_OK;
    
    // In this example, return the IDs of our objects, but no weights.
    // Zoltan will assume equally weighted objects.
    
    for (int i=0; i<mesh->numCoarseElems(); i++){
      global_ids[i] = i;
      local_ids[i] = i;
      in_order[i] = 0;
    }
    return;	   
  }    

  static int get_num_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr){
    exampleSquareMesh *mesh = (exampleSquareMesh *)data;   
    *ierr = ZOLTAN_OK;
    
    int parentID = *global_id;
    element parentElem = mesh->_elements[parentID];
    
    return parentElem.numChildren();
  }

  static void get_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR parent_gid, ZOLTAN_ID_PTR parent_lid, ZOLTAN_ID_PTR child_gids, ZOLTAN_ID_PTR child_lids, int *assigned, int *num_vert, ZOLTAN_ID_PTR vertices, ZOLTAN_REF_TYPE *ref_type, ZOLTAN_ID_PTR in_vertex, ZOLTAN_ID_PTR out_vertex, int *ierr){

    exampleSquareMesh *mesh = (exampleSquareMesh *)data;   
    *ierr = ZOLTAN_OK;
    
    int parentID = *parent_gid;
    element parentElem = mesh->_elements[parentID];
    vector<int> childIDs = parentElem.children();    
    int vertexInd = 0;
    for (int i = 0; i<parentElem.numChildren();i++){
      child_gids[i] = childIDs[i];
      num_vert[i] = 4;
      vector<int> childVertices = mesh->_elements[childIDs[i]].vertices();
      for (int j = 0; j<num_vert[i];j++){       	
	vertices[vertexInd++] = childVertices[j];	
      }
      ref_type[i] = ZOLTAN_IN_ORDER;
    }
    

  }

  
}; // end of class
  
static void MPIExit()
{
#ifdef MPICPP
  MPI::Finalize();
#else
  MPI_Finalize();
#endif
}

int main(int argc, char *argv[]){

  /////////////////////////////////
  // Initialize MPI and Zoltan
  /////////////////////////////////

  int rank, size;
  float version;

#ifdef MPICPP
  MPI::Init(argc, argv);
  rank = MPI::COMM_WORLD.Get_rank();
  size = MPI::COMM_WORLD.Get_size();
  MPI::Status status;
  int tag=0;
#else
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif

  Zoltan_Initialize(argc, argv, &version);

  /////////////////////////////////
  // Create a Zoltan object
  /////////////////////////////////

#ifdef MPICPP
  Zoltan *zz = new Zoltan(MPI::COMM_WORLD);
#else
  Zoltan *zz = new Zoltan(MPI_COMM_WORLD);
#endif

  if (zz == NULL){
    MPIExit();
    exit(0);
  }

  //////////////////////////////////////////////////////////////////
  // Read objects from input file and distribute them unevenly
  //////////////////////////////////////////////////////////////////

  int myNode = rank;
  int numNodes = size;
  int masterNode = 0; // designate first node as master node


  exampleSquareMesh mesh = exampleSquareMesh();
  mesh.refineElement(1);
  mesh.refineElement(2);

  // in master node, distribute element IDs to other nodes
  if (myNode==masterNode){
    mesh.printElements();      
    cout << "Number of active elements = " << mesh.numActiveElems()<<endl;

    // make global IDs list
    int totalObj = mesh.numActiveElems();
    int objList[totalObj];
    for (int i=0;i<totalObj;i++){
      objList[i] = mesh.getActiveElemGlobalIndex(i);
    }

    // divide up total objects to pass around
    int objPerNode[numNodes];
    for (int i = 0;i<numNodes;i++){
      objPerNode[i] = totalObj/numNodes;
    }    
    // add remainder cyclically
    int remainder = totalObj%numNodes;
    for (int i = 0;i<remainder;i++){
      objPerNode[i]++;
    }    

    // store object IDs for master node 
    for (int i = 0;i<objPerNode[0];i++){
      mesh.addPartitionGlobalID(objList[i]);
    }

    // send out the actual objects to each other node
    int listIndex = objPerNode[0]; //start at offset from objects for first node
    for (int node = 1;node<numNodes;node++){
      MPI::COMM_WORLD.Send(&objPerNode[node],1,MPI::INT,node,tag);

      // build list of objects to send out
      int objListPart[objPerNode[node]];
      for (int i = 0;i<objPerNode[node];i++){
	objListPart[i] = objList[listIndex];
	listIndex++;
      }
      MPI::COMM_WORLD.Send(&(objListPart[0]),objPerNode[node],MPI::INT,node,tag);

    }

  }else{ // if you're another node, receive input
    
    int recvObjPerNode;
    MPI::COMM_WORLD.Recv(&recvObjPerNode,1,MPI::INT,masterNode,tag,status);    
    
    int objPart[recvObjPerNode];
    MPI::COMM_WORLD.Recv(&(objPart[0]),recvObjPerNode,MPI::INT,masterNode,tag,status);    
    for (int i = 0;i<recvObjPerNode;i++){
      mesh.addPartitionGlobalID(objPart[i]);
    }
  }
  
  cout << "For node: " << myNode << " ";
  mesh.printPartitionIDs(); 

  delete zz;
  MPIExit();
  return 0;

  ///////////////////////////////////////////////////////////////////
  // Set the Zoltan parameters, and the names of the query functions
  ///////////////////////////////////////////////////////////////////
 
  // General parameters 

  zz->Set_Param( "LB_METHOD", "RANDOM");    /* Zoltan method: "BLOCK" */
  zz->Set_Param( "NUM_GID_ENTRIES", "1");  /* global ID is 1 integer */
  zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is 1 integer */
  zz->Set_Param( "OBJ_WEIGHT_DIM", "0");   /* we omit object weights */

  // Query functions 
  
  zz->Set_Num_Obj_Fn(exampleSquareMesh::get_number_of_objects, &mesh);
  zz->Set_Obj_List_Fn(exampleSquareMesh::get_object_list, &mesh);

  ////////////////////////////////////////////////////////////////
  // Zoltan can now partition the objects in this collection.
  // In this simple example, we assume the number of partitions is
  // equal to the number of processes.  Process rank 0 will own
  // partition 0, process rank 1 will own partition 1, and so on.
  ////////////////////////////////////////////////////////////////

 
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
    printf("Partitioning failed on process %d\n",rank);
    MPIExit();
    delete zz;
    exit(0);
  }
  
  delete zz;
  MPIExit();
}
