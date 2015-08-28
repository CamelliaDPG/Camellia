//
//  MOABReader.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/27/15.
//
//

#include "MOABReader.h"

#ifdef HAVE_MOAB
// MOAB includes:
#include "moab/ParallelComm.hpp"
#include "MBParallelConventions.h"
#endif

using namespace Camellia;

#ifdef HAVE_MOAB
CellTopoPtr MOABReader::cellTopoForMOABType(moab::EntityType entityType)
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
#endif

MeshTopologyPtr MOABReader::readMOABMesh(string filePath)
{
#ifdef HAVE_MOAB
  string options = "PARALLEL=BCAST";
  
  using namespace moab;
  
  // Get MOAB instance
  Interface* mb = new (std::nothrow) Core;
  TEUCHOS_TEST_FOR_EXCEPTION(NULL == mb, std::invalid_argument, "Error during construction of MOAB Interface object");
  
  // Read the file with the specified options
  ErrorCode rval = mb->load_file(filePath.c_str(), 0, options.c_str());
  
  int spaceDim;
  mb->get_dimension(spaceDim);
  
  Range all_elements;
  rval = mb->get_entities_by_dimension(0, spaceDim, all_elements);
  
  // sometimes, return value of get_dimension can exceed the actual dimension of the mesh.
  // if there aren't any entities of a given dimension, try the next lower dimension.
  // if there aren't any entities that are at least 1D, we'll return null
  while ((all_elements.size() == 0) && (spaceDim > 1))
  {
    spaceDim--;
    rval = mb->get_entities_by_dimension(0, spaceDim, all_elements);
  }
  if (all_elements.size() == 0)
  {
    return Teuchos::null;
  }
  
  static const int VERTICES_PER_POINT = 3;
  
  MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(spaceDim) );
  
  while (all_elements.size() > 0)
  {
    EntityHandle element = all_elements.pop_front();
    int vertexDim = 0, oneElement = 1;
    vector<EntityHandle> elem_vertices;
    mb->get_adjacencies(&element, oneElement, vertexDim, true, elem_vertices);
    
    EntityType moabType = mb->type_from_handle(element);
    CellTopoPtr cellTopo = cellTopoForMOABType(moabType);
    
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
    }
    meshTopo->addCell(cellTopo, vertices);
  }
  
  delete mb;
  return meshTopo;
#else
  cout << "Error: HAVE_MOAB is false; perhaps you didn't build Camellia with MOAB?  Returning null MeshTopologyPtr.\n";
  return Teuchos::null;
#endif
}