#include <mpi.h>

#include "Epetra_MpiComm.h"
#include "Teuchos_GlobalMPISession.hpp"

#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"
#include "TimeSteppingConstants.h"

// MOAB includes:
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#include "moab/Core.hpp"

using namespace Camellia;
using namespace moab;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  MPI_Comm comm = MPI_COMM_WORLD;
  
  Epetra_MpiComm EpetraComm(comm);
  EpetraComm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.
  
  string options = "PARALLEL=BCAST";
  
  // Get MOAB instance
  Interface* mb = new (std::nothrow) Core;
  if (NULL == mb)
    return 1;

  string meshFileName = "./64bricks_512hex_256part.h5m";
  
  // Get the MOAB ParallelComm instance
  ParallelComm* pcomm = new ParallelComm(mb, comm);
  
  // Read the file with the specified options
  ErrorCode rval = mb->load_file(meshFileName.c_str(), 0, options.c_str());MB_CHK_ERR(rval);
  
  Range shared_ents;
  // Get entities shared with all other processors
  rval = pcomm->get_shared_entities(-1, shared_ents);MB_CHK_ERR(rval);
  
  // Filter shared entities with not not_owned, which means owned
  Range owned_entities;
  rval = pcomm->filter_pstatus(shared_ents, PSTATUS_NOT_OWNED, PSTATUS_NOT, -1, &owned_entities);MB_CHK_ERR(rval);
  
  
  unsigned int nums[4] = {0}; // to store the owned entities per dimension
  for (int i = 0; i < 4; i++)
    nums[i] = (int)owned_entities.num_of_dimension(i);
  vector<int> rbuf(numProcs*4, 0);
  MPI_Gather(nums, 4, MPI_INT, &rbuf[0], 4, MPI_INT, 0, comm);
  // Print the stats gathered:
  if (0 == rank) {
    for (int i = 0; i < numProcs; i++)
      cout << " Shared, owned entities on proc " << i << ": " << rbuf[4*i] << " verts, " <<
      rbuf[4*i + 1] << " edges, " << rbuf[4*i + 2] << " faces, " << rbuf[4*i + 3] << " elements" << endl;
  }
  
  // Filter shared entities with not not_owned, which means owned
  Range all_vertices, all_edges, all_faces, all_elements;
  rval = mb->get_entities_by_dimension(0, 0, all_vertices);
  rval = mb->get_entities_by_dimension(0, 1, all_edges);
  rval = mb->get_entities_by_dimension(0, 2, all_faces);
  rval = mb->get_entities_by_dimension(0, 3, all_elements);
  
  //  Range all_edges2;
  //  rval = mb->get_adjacencies(all_elements, 1, true, all_edges2);
  
  //  mb->list_entities(all_elements);
  
  nums[0] = all_vertices.size();
  nums[1] = all_edges.size();
  nums[2] = all_faces.size();
  nums[3] = all_elements.size();
  
  MPI_Gather(nums, 4, MPI_INT, &rbuf[0], 4, MPI_INT, 0, comm);
  // Print the stats gathered:
  if (0 == rank) {
    for (int i = 0; i < numProcs; i++)
      cout << " All entities on proc " << i << ": " << rbuf[4*i] << " verts, " <<
      rbuf[4*i + 1] << " edges, " << rbuf[4*i + 2] << " faces, " << rbuf[4*i + 3] << " elements" << endl;
  }
  
  EntityHandle element = all_elements.pop_front();
  int vertexDim = 0, oneElement = 1;
  vector<EntityHandle> elem_vertices;
  mb->get_adjacencies(&element, oneElement, vertexDim, true, elem_vertices);
  
  vector<double> vertexCoords(elem_vertices.size() * 3);
  mb->get_coords(&elem_vertices[0],elem_vertices.size(),&vertexCoords[0]);
  for (int i=0; i<elem_vertices.size(); i++)
  {
    double x = vertexCoords[i*3 + 0];
    double y = vertexCoords[i*3 + 1];
    double z = vertexCoords[i*3 + 2];
    cout << "element vertex " << i << ": (" << x << "," << y << "," << z << ")\n";
  }
  
  // Now exchange 1 layer of ghost elements, using vertices as bridge
  // (we could have done this as part of reading process, using the PARALLEL_GHOSTS read option)
  rval = pcomm->exchange_ghost_cells(3, // int ghost_dim
                                     0, // int bridge_dim
                                     1, // int num_layers
                                     0, // int addl_ents
                                     true);MB_CHK_ERR(rval); // bool store_remote_handles
  
  // Repeat the reports, after ghost exchange
  shared_ents.clear();
  owned_entities.clear();
  rval = pcomm->get_shared_entities(-1, shared_ents);MB_CHK_ERR(rval);
  rval = pcomm->filter_pstatus(shared_ents, PSTATUS_NOT_OWNED, PSTATUS_NOT, -1, &owned_entities);MB_CHK_ERR(rval);
  
  // Find out how many shared entities of each dimension are owned on this processor
  for (int i = 0; i < 4; i++)
    nums[i] = (int)owned_entities.num_of_dimension(i);
  
  // Gather the statistics on processor 0
  MPI_Gather(nums, 4, MPI_INT, &rbuf[0], 4, MPI_INT, 0, comm);
  if (0 == rank) {
    cout << " \n\n After exchanging one ghost layer: \n";
    for (int i = 0; i < numProcs; i++) {
      cout << " Shared, owned entities on proc " << i << ": " << rbuf[4*i] << " verts, " <<
      rbuf[4*i + 1] << " edges, " << rbuf[4*i + 2] << " faces, " << rbuf[4*i + 3] << " elements" << endl;
    }
  }
  
  delete mb;
  
  return 0;
}