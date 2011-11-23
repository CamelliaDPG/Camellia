#include <iostream>
#include <mpi.h>

using namespace std;
int main( int argc, char** argv ){
 
  MPI::Init();
  MPI::Status status;
  int tag=0;
  int numNodes, myNode;  
  numNodes = MPI::COMM_WORLD.Get_size();
  myNode   = MPI::COMM_WORLD.Get_rank();

  if (numNodes==1){
    cout << "Needs more than one node to work" << endl;
    return -1;
  }

  //  cout << "Node " << myNode+1 <<  " of " << numNodes << endl;
  int masterNode = 0; 

  // pass a set of numbers to different nodes
  if (myNode==masterNode){
    int totalObj = 12;    
    int objList[totalObj];
    for (int ind = 0;ind<totalObj;ind++){
      objList[ind] = ind; //pretend these are global IDs
      cout << "storing " << ind << endl;
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
   
    // send out the actual objects to each other node
    int listIndex = objPerNode[0];
    cout << "Node 0 gets " << objPerNode[0] << " objects" << endl;
    for (int node = 1;node<numNodes;node++){
      cout << "Sending " << objPerNode[node] << " objects for node " << node << endl;
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
    //    cout << "Node " << myNode << " is receiving " << recvObjPerNode << " objects for node " << myNode << endl;
    int objPart[recvObjPerNode];
    MPI::COMM_WORLD.Recv(&(objPart[0]),recvObjPerNode,MPI::INT,masterNode,tag,status);    
    for (int i = 0;i<recvObjPerNode;i++){
      cout << "Node " << myNode << " received object = " << objPart[i] << endl;
      cout << " " << endl;
    }
    
  }
  
  MPI::Finalize();
  
}
