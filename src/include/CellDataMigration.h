//
//  CellDataMigration.h
//  Camellia-debug
//
//  Created by Nate Roberts on 7/15/14.
//
//

#ifndef __Camellia_debug__CellDataMigration__
#define __Camellia_debug__CellDataMigration__

#include "Mesh.h"

namespace Camellia
{
class CellDataMigration
{
public:
  static int dataSize(Mesh* mesh, GlobalIndexType cellID);
  static void packData(Mesh* mesh, GlobalIndexType cellID, bool packParentDofs, char *dataBuffer, int size);
  static void unpackData(Mesh* mesh, GlobalIndexType cellID, const char *dataBuffer, int size);
};
}

#endif /* defined(__Camellia_debug__CellDataMigration__) */
