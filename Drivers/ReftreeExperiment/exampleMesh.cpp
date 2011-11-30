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
class vertex{
private:
  int _vertexID;
  vector<double> _coords;
public:
  vertex(int vertexID, double x, double y){
    _vertexID = vertexID;
    _coords.push_back(x);
    _coords.push_back(y);
  }
  int ID(){
    return _vertexID;
  }
  vector<double> coords(){
    return _coords;
  }
};

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
  vector<vertex> _vertexList;
public: 
  //constructor
  exampleSquareMesh(int nodeID){
    _thisNode = nodeID;
    _numCoarseElements = 4;
    _numElements = _numCoarseElements; 
    _numVertices = 9;
    vector<int> vertexIDs;

    addVertexCoord(0,0.0,0.0);
    addVertexCoord(1,0.5,0.0);
    addVertexCoord(2,1.0,0.0);
    addVertexCoord(3,0.0,0.5);
    addVertexCoord(4,0.5,0.5);
    addVertexCoord(5,1.0,0.5);
    addVertexCoord(6,0.0,1.0);
    addVertexCoord(7,0.5,1.0);
    addVertexCoord(8,1.0,1.0);

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
      
      vector<vector<double> > vertCoordVec = getVertexCoords(elemID);
      
      // set new vertice positions as averages of other vertexes
      vector<double> xp(2,0.0);
      for (int j = 0;j<4;j++){
	for (int i = 0;i<2;i++){
	  xp[i] = (vertCoordVec[j][i]+vertCoordVec[j+1][i])/2.0;
	}	  
	addVertexCoord(_numVertices+j,xp[0],xp[1]);
      }
      // last vertex in middle
      xp = getElementCentroid(elemID);
      if (1==1){
	// or try just a vertex coordinate
	xp[0] = vertCoordVec[0][0];
	xp[1] = vertCoordVec[0][1];
      }
      addVertexCoord(_numVertices+4,xp[0],xp[1]);


      _numElements += 1;
    }
    _numVertices+=5; // add 5 new vertices in an isotropic refinement
    //    cout << "New number of elements: " << _numElements << endl;       
  }
  int numVertices(int elementID){
    return _elements[elementID].vertices().size();
  }
  void addVertex(int elementID, int vertexID){
    _elements[elementID].addVertex(vertexID);
    return;
  }

  void addVertexCoord(int vertexID, double x, double y){
    _vertexList.push_back(vertex(vertexID,x,y));
  }

  vector<int> getVertices(int elementID){
    return _elements[elementID].vertices();
  }
  
  vector<vertex> getElemVertices(int elementID){
    vector<int> vertices =  getVertices(elementID);
    vector<vertex> elemVertices;
    for (unsigned int i=0;i<vertices.size();i++){
      cout << "Vertex ID for element " << elementID << " is: " << vertices[i] << endl;
      elemVertices.push_back(getVertex(i));
    }
    return elemVertices;
  }

  vertex getVertex(int vertexID){
    bool returned = false;
    for (int i=0;i<_vertexList.size();i++){
      if (_vertexList[i].ID()==vertexID){
	returned = true;
	return _vertexList[i];
      }
    }
    if (~returned){
      cout << "Did not find vertex, returning empty vertex" << endl;
      return vertex(-1,0.0,0.0);
    }
  }

  vector<vector<double> > getVertexCoords(int elementID){
    vector<vector<double> > vertCoordVec;
    vector<int> vertIDs = getVertices(elementID);
    for (int i=0;i<numVertices(elementID);i++){
      cout << "Vertex coords for " << elementID << ": ";      
      vector<double> vertCoords = getVertex(vertIDs[i]).coords();
      for (int j = 0;j<2;j++){
	vertCoordVec.push_back(vertCoords);
      }
    }
    return vertCoordVec;
  }

  void printVertexCoords(int elementID){
    vector<int> vertIDs = getVertices(elementID);
    for (int i=0;i<numVertices(elementID);i++){
      cout << "Vertex coords for " << elementID << ": ";      
      vector<double> vertCoords = getVertex(vertIDs[i]).coords();
      for (int j = 0;j<2;j++){
	cout << vertCoords[j] << " ";
      }
      cout << endl;
    }
    return;
  }

  //averages vertex coordinates
  vector<double> getElementCentroid(int elementID){
    vector<int> vertIDs = getVertices(elementID);
    vector<double> vertSum(2,0.0);
    for (int i=0;i<numVertices(elementID);i++){
 
      vector<double> vertCoords = getVertex(vertIDs[i]).coords();      
      for (int j = 0;j<2;j++){
	vertSum[j]+=vertCoords[j];
      }
    }
    for (int i=0;i<2;i++){
      vertSum[i] = vertSum[i]/numVertices(elementID);
    }
    return vertSum;
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
  
  void removePartitionIDs(vector<double> exportIDs){
    for (unsigned int i = 0;i<_myPartitionGlobalIDs.size();i++){
      for (unsigned int j = 0;j<exportIDs.size();j++){
	if (_myPartitionGlobalIDs[i]==exportIDs[j]){
	  _myPartitionGlobalIDs.erase(_myPartitionGlobalIDs.begin() + i);
	}
      }
    }
  }
  
  void addPartitionIDs(vector<double> importIDs){    
    for (unsigned int j = 0;j<importIDs.size();j++){
      _myPartitionGlobalIDs.push_back(importIDs[j]);
    }
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

  static int get_num_geom(void *data, int *ierr){
    exampleSquareMesh *mesh = (exampleSquareMesh *)data;
    *ierr = ZOLTAN_OK;
    return 2; // it's always 2d here
  }

  static void get_geom_list(void *data, int num_gid_entries, int num_lid_entries, int num_obj, ZOLTAN_ID_PTR global_ids, ZOLTAN_ID_PTR local_ids, int num_dim, double *geom_vec, int *ierr){
    exampleSquareMesh *mesh = (exampleSquareMesh *)data;
    *ierr = ZOLTAN_OK;
    cout << "Num objects is " << num_obj << endl;
    cout << "Num dim is " << num_dim << endl;
    for (int i=0;i<num_obj;i++){      
      cout << "Global ids are " << global_ids[i] << endl;
      vector<double> coords = mesh->getElementCentroid(global_ids[i]);
      for (int j = 0;j<num_dim;j++){
	geom_vec[i*num_dim+j] = coords[j];
	cout << "Centroid is " << coords[j] << endl;
      }
    }       
  }

  static void get_geom_fn(void *data, int num_gid_entries, int num_lid_entries, ZOLTAN_ID_PTR global_id, ZOLTAN_ID_PTR local_id, double *geom_vec, int *ierr){
    exampleSquareMesh *mesh = (exampleSquareMesh *)data;
    *ierr = ZOLTAN_OK;
    vector<double> coords = mesh->getElementCentroid(*global_id);
    int num_dim = 2;
    for (int j = 0;j<num_dim;j++){
      geom_vec[j] = coords[j];
      cout << "Centroid is " << coords[j] << endl;
    }
    

  } 

  // ----------------BELOW FUNCTIONS: interface functions for reftree --------------


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
  vector<double> center =  mesh.getElementCentroid(0);
  /*
  cout << "Centroid: " << center[0] << "," << center[1] << endl;
  center =  mesh.getElementCentroid(1);
  cout << "Centroid: " << center[0] << "," << center[1] << endl;
  center =  mesh.getElementCentroid(2);
  cout << "Centroid: " << center[0] << "," << center[1] << endl;
  center =  mesh.getElementCentroid(3);
  cout << "Centroid: " << center[0] << "," << center[1] << endl;

  */


  //manually add in extra vertices on neighbors - HARD CODED
  bool refine = true;
  if (refine==true){
    int newVertex;
    mesh.refineElement(1);
    newVertex = mesh.getVertices(7)[0];
    mesh.addVertex(0,newVertex);
    
    mesh.refineElement(2);
    newVertex = mesh.getVertices(11)[0];
    mesh.addVertex(3,newVertex);
    
    /*
    mesh.refineElement(9);
    newVertex = mesh.getVertices(15)[0];
    mesh.addVertex(8,newVertex);

    mesh.refineElement(10);
    newVertex = mesh.getVertices(19)[0];
    mesh.addVertex(11,newVertex);
    */
    
  }

  for (int i = 0;i<mesh.numActiveElems();i++){
    int elemID = mesh.getActiveElemGlobalIndex(i);
    cout << "element " << elemID << " has vertices ";
    vector<int> verts = mesh.getVertices(elemID);
    for (int j = 0;j<verts.size();j++){
      cout << verts[j] << ", ";
    }
    cout << endl;
  }


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

    // store object IDs for master node 
    for (int i = 0;i<totalObj;i++){
      mesh.addPartitionGlobalID(objList[i]);
    }


  }else{ // if you're another node, store nothing
  }
  
  cout << " for node: " << myNode << endl;
  mesh.printPartitionIDs();
  

  ///////////////////////////////////////////////////////////////////
  // Set the Zoltan parameters, and the names of the query functions
  ///////////////////////////////////////////////////////////////////
 
  // General parameters 

  zz->Set_Param( "LB_METHOD", "REFTREE");    /* Zoltan method */
  zz->Set_Param( "RANDOM_MOVE_FRACTION", ".5");    /* Zoltan "random" partition param */
  zz->Set_Param( "NUM_GID_ENTRIES", "1");  /* global ID is 1 integer */
  zz->Set_Param( "NUM_LID_ENTRIES", "1");  /* local ID is 1 integer */
  zz->Set_Param( "OBJ_WEIGHT_DIM", "0");   /* we omit object weights */
  zz->Set_Param( "DEBUG_LEVEL", "10");   /* no output */
  zz->Set_Param( "REFTREE_INITPATH", "CONNECTED"); // no SFC on coarse mesh
  // Query functions 
  
  zz->Set_Num_Obj_Fn(exampleSquareMesh::get_number_of_objects, &mesh);
  zz->Set_Obj_List_Fn(exampleSquareMesh::get_object_list, &mesh);
  zz->Set_Num_Geom_Fn(exampleSquareMesh::get_num_geom, &mesh);
  zz->Set_Geom_Multi_Fn(exampleSquareMesh::get_geom_list, &mesh);
  //  zz->Set_Geom_Fn(exampleSquareMesh::get_geom_fn, &mesh);


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
  cout << "For node: " << myNode << ", exported globalIDs are " << endl;
  vector<double> exportGlobalIDvector;
  for (int i=0;i<numExport;i++){
    cout << exportGlobalIds[i] << endl;
    exportGlobalIDvector.push_back(exportGlobalIds[i]);
  }

  cout << "For node: " << myNode << ", num imported gids: " << numImport << endl;
  cout << "For node: " << myNode << ", imported globalIDs should be " << endl;
  vector<double> importGlobalIDvector;
  for (int i=0;i<numImport;i++){
    cout << importGlobalIds[i] << endl;
    importGlobalIDvector.push_back(importGlobalIds[i]);
  }

  mesh.printPartitionIDs();
  mesh.removePartitionIDs(exportGlobalIDvector);
  mesh.addPartitionIDs(importGlobalIDvector);  
  cout << "After import/exports: " << endl;
  mesh.printPartitionIDs();

  
  delete zz;
  MPIExit();
}
