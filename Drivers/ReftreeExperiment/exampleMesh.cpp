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
  int numVertices(){
    return _vertices.size();
  }
  vector<int> vertices(){
    return _vertices;
  }
  void addVertex(int vertexID){
    _vertices.push_back(vertexID);
    return;
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
  int _thisNode;
public: 
  //constructor
  exampleSquareMesh(int nodeID){
    _thisNode = nodeID;
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
      
      int newElemID = _numElements; // new element ID = next element number
      //      cout << "adding " << newElemID << " child to elemID = " << elemID << endl;
      _elements[elemID].addChild(newElemID);

      vector<int> newVertices; 
      // numbering of new verts is counterclockwise from bottom
      switch(i){
      case 0:
	newVertices.push_back(oldVertices[0]);
	newVertices.push_back(_numVertices+0);
	newVertices.push_back(_numVertices+4);
	newVertices.push_back(_numVertices+3);
	break;
      case 1:
	newVertices.push_back(_numVertices+0);
	newVertices.push_back(oldVertices[1]);
	newVertices.push_back(_numVertices+1);
	newVertices.push_back(_numVertices+4);
	break;
      case 2:
	newVertices.push_back(_numVertices+4);
	newVertices.push_back(_numVertices+1);
	newVertices.push_back(oldVertices[2]);
	newVertices.push_back(_numVertices+2);
	break;
      case 3:
	newVertices.push_back(_numVertices+3);
	newVertices.push_back(_numVertices+4);
	newVertices.push_back(_numVertices+2);
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
  void addVertex(int elementID, int vertexID){
    _elements[elementID].addVertex(vertexID);
    return;
  }
  vector<int> getVertices(int elementID){
    return _elements[elementID].vertices();
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

  int isInPartition(int ID){
    for (int j = 0;j<numPartitionGlobalIDs();j++){
      if (getPartitionGlobalID(j)==ID){
	return 1;
      }
    }
    return 0; //no match

  }

  void printPartitionIDs(){
    cout << "Partition IDs are: " << endl;
    for (unsigned int i = 0;i<_myPartitionGlobalIDs.size();i++){
      cout << _myPartitionGlobalIDs[i] << endl;
    }
    return;
  }
  
  // returns back the node associated with this copy of the mesh
  int thisNode(){
    return _thisNode;
  }
 
  //---------------------- zoltan interface functions ----------------------------------------

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
      //      localID[i] = i;
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
      
    // assumes that the n initial mesh elements are ordered 1:n. 
    int count = 0;
    int num_assigned = 0;
    for (int i=0; i<mesh->numCoarseElems(); i++){
      global_ids[i] = i;
      local_ids[i] = i;
      in_order[i] = 0; // let zoltan figure out ordering
      num_vert[i] = mesh->_elements[global_ids[i]].numVertices();
      // warning - assumes num_gid_entries = 1!!!
      for (int j=0;j<num_vert[i];j++){
	vertices[count] = mesh->_elements[global_ids[i]].vertices()[j];
	count++;
      }
      assigned[i] = mesh->isInPartition(i);            
      if (assigned[i]==1){
	num_assigned++;
      }
    }    
    cout<< "num assigned: "<<num_assigned<<endl;
  
    return;	   
  }    

  static int get_num_children(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int *ierr){
    exampleSquareMesh *mesh = (exampleSquareMesh *)data;   
    *ierr = ZOLTAN_OK;
    
    int parentID = *global_id;
    element parentElem = mesh->_elements[parentID];
    cout << "----num children for elem " << parentID << " are: " << parentElem.numChildren() << endl;
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
      child_lids[i] = i;
      assigned[i] = mesh->isInPartition(child_gids[i]);
      num_vert[i] = mesh->_elements[child_gids[i]].numVertices();
      vector<int> childVertices = mesh->_elements[childIDs[i]].vertices();
      for (int j = 0; j<num_vert[i];j++){       	
	vertices[vertexInd] = childVertices[j];	
	vertexInd++;
      }
      ref_type[i] = ZOLTAN_IN_ORDER;
      cout << "---Children of " << parentID << " are: " << childIDs[i] << " and assigned = "<< mesh->isInPartition(child_gids[i])<< " and have " << num_vert[i] << " vertices"<<endl;
    }    
  }

  static void get_child_weight(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, int wgt_dim, float *obj_wgt, int *ierr){
    if (wgt_dim>0){
      obj_wgt[0]=1;
    }
    return;
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

  exampleSquareMesh mesh = exampleSquareMesh(myNode);

  //manually add in extra vertices on neighbors - HARD CODED
  /*
  mesh.refineElement(1);
  int newVertex;
  newVertex = mesh.getVertices(8)[0];
  mesh.addVertex(0,newVertex);

  mesh.refineElement(2);
  newVertex = mesh.getVertices(10)[0];
  mesh.addVertex(3,newVertex);
  */

  cout << "Mesh has node: " << mesh.thisNode() << endl;

  // in master node, distribute element IDs to other nodes
  if (myNode==masterNode){
    mesh.printElements();      
    cout << "Number of active elements = " << mesh.numActiveElems()<<endl;

    // make global IDs list - is it for active elems, coarse elems, or everything both active/inactive?
    int totalObj = mesh.numActiveElems();
    //        int totalObj = mesh.numElems();
    //    int totalObj = mesh.numCoarseElems();
    int objList[totalObj];
    for (int i=0;i<totalObj;i++){
      objList[i] = mesh.getActiveElemGlobalIndex(i);
      //        objList[i] = i;
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
  
  cout << " for node: " << myNode << endl;
  mesh.printPartitionIDs();
  

  ///////////////////////////////////////////////////////////////////
  // Set the Zoltan parameters, and the names of the query functions
  ///////////////////////////////////////////////////////////////////
 
  // General parameters 

  zz->Set_Param( "LB_METHOD", "REFTREE");    /* Zoltan method */
  zz->Set_Param( "NUM_GID_ENTRIES", "1");  /* global ID is 1 integer */
  zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is 1 integer */
  zz->Set_Param( "OBJ_WEIGHT_DIM", "0");   /* we omit object weights */
  zz->Set_Param( "DEBUG_LEVEL", "10");   /* no output */
  zz->Set_Param( "REFTREE_INITPATH", "CONNECTED"); // no SFC on coarse mesh
  // Query functions 
  
  //  zz->Set_Num_Obj_Fn(exampleSquareMesh::get_number_of_objects, &mesh);
  //  zz->Set_Obj_List_Fn(exampleSquareMesh::get_object_list, &mesh);
  zz->Set_Num_Coarse_Obj_Fn(exampleSquareMesh::get_num_coarse_elem, &mesh);
  zz->Set_Coarse_Obj_List_Fn(exampleSquareMesh::get_coarse_elem_list, &mesh);
  zz->Set_Num_Child_Fn(exampleSquareMesh::get_num_children, &mesh);
  zz->Set_Child_List_Fn(exampleSquareMesh::get_children, &mesh);
  zz->Set_Child_Weight_Fn(exampleSquareMesh::get_child_weight, &mesh);

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

  cout << "For node: " << myNode << ", num exported gids: " << numExport << endl;
  cout << "For node: " << myNode << ", original globalIDs are " << endl;
  for (int i=0;i<numExport;i++){
    cout << exportGlobalIds[i] << endl;
  }

  cout << "For node: " << myNode << ", new globalIDs should be " << endl;
  cout << "For node: " << myNode << ", num imported gids: " << numImport << endl;
  for (int i=0;i<numImport;i++){
    cout << importGlobalIds[i] << endl;
  }
  
  delete zz;
  MPIExit();
}
