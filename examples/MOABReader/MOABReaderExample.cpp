#include "CellTopology.h"
#include "Function.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "SimpleFunction.h"
#include "StokesVGPFormulation.h"
#include "TimeSteppingConstants.h"

#ifdef HAVE_MOAB

// MOAB includes:
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#include "moab/Core.hpp"

using namespace Camellia;
using namespace moab;
using namespace std;

CellTopoPtr cellTopoForEntityType(EntityType entityType)
{
  switch (entityType) {
    case moab::MBVERTEX:
      return CellTopology::point();
    case moab::MBEDGE:
      return CellTopology::line();
    case moab::MBTRI:
      return CellTopology::triangle();
    case moab::MBQUAD:
      return CellTopology::quad();
    case moab::MBTET:
      return CellTopology::tetrahedron();
    case moab::MBHEX:
      return CellTopology::hexahedron();
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported entity type");
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  string meshFileName;
  
  cmdp.setOption("meshFile", &meshFileName );
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  string options = "PARALLEL=BCAST";
  
  // Get MOAB instance
  Interface* mb = new (std::nothrow) Core;
  if (NULL == mb)
    return 1;
  
  // Read the file with the specified options
  ErrorCode rval = mb->load_file(meshFileName.c_str(), 0, options.c_str());MB_CHK_ERR(rval);

  int spaceDim;
  mb->get_dimension(spaceDim);

  Range all_elements;
  rval = mb->get_entities_by_dimension(0, spaceDim, all_elements);
  
  static const int VERTICES_PER_POINT = 3;

  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
  
  while (all_elements.size() > 0)
  {
    EntityHandle element = all_elements.pop_front();
    int vertexDim = 0, oneElement = 1;
    vector<EntityHandle> elem_vertices;
    mb->get_adjacencies(&element, oneElement, vertexDim, true, elem_vertices);
    
    EntityType moabType = mb->type_from_handle(element);
    CellTopoPtr cellTopo = cellTopoForEntityType(moabType);
    
    vector<double> vertexCoords(elem_vertices.size() * VERTICES_PER_POINT);
    mb->get_coords(&elem_vertices[0],elem_vertices.size(),&vertexCoords[0]);
    
    int vertexCount = elem_vertices.size();
    vector<vector<double>> vertices(vertexCount,vector<double>(spaceDim)); // Camellia container
    for (int i=0; i<elem_vertices.size(); i++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        vertices[i][d] = vertexCoords[i*VERTICES_PER_POINT + d];
      }
//      double x = vertexCoords[i*VERTICES_PER_POINT + 0];
//      double y = vertexCoords[i*VERTICES_PER_POINT + 1];
//      double z = vertexCoords[i*VERTICES_PER_POINT + 2];
//      if (rank==0) cout << "element vertex " << i << ": (" << x << "," << y << "," << z << ")\n";
    }
    meshTopo->addCell(cellTopo, vertices);
  }

  int cellCount = meshTopo->activeCellCount();
  
  if (rank==0) cout << spaceDim << "D mesh topology has " << cellCount << " cells.\n";
  
  delete mb;
  
  return 0;
}
#else

int main(int argc, char *argv[])
{
  cout << "Error - HAVE_MOAB preprocessor macro not defined.\n";
  
  return 0;
}

#endif
