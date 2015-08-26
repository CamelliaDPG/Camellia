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
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  string options = "PARALLEL=BCAST";
  
  // Get MOAB instance
  Interface* mb = new (std::nothrow) Core;
  if (NULL == mb)
    return 1;

  string meshFileName = "./64bricks_512hex_256part.h5m";
  
  // Read the file with the specified options
  ErrorCode rval = mb->load_file(meshFileName.c_str(), 0, options.c_str());MB_CHK_ERR(rval);

  // Filter shared entities with not not_owned, which means owned
  Range all_vertices, all_edges, all_faces, all_elements;
  rval = mb->get_entities_by_dimension(0, 0, all_vertices);
  rval = mb->get_entities_by_dimension(0, 1, all_edges);
  rval = mb->get_entities_by_dimension(0, 2, all_faces);
  rval = mb->get_entities_by_dimension(0, 3, all_elements);
  
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
    if (rank==0) cout << "element vertex " << i << ": (" << x << "," << y << "," << z << ")\n";
  }
  
  delete mb;
  
  return 0;
}
