//
//  MOABReader.h
//  Camellia
//
//  Created by Nate Roberts on 8/27/15.
//
//

#ifndef Camellia_MOABReader_h
#define Camellia_MOABReader_h

#ifdef HAVE_MOAB
#include "moab/Core.hpp"
#endif

#include "CellTopology.h"
#include "MeshTopology.h"

namespace Camellia {
  class MOABReader {
  private:
#ifdef HAVE_MOAB
    static CellTopoPtr cellTopoForMOABType(moab::EntityType entityType);
#endif
  public:
    static MeshTopologyPtr readMOABMesh(string filePath);
  };
}

#endif
