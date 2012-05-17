//
//  MeshTestUtility.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/17/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_MeshTestUtility_h
#define Camellia_debug_MeshTestUtility_h

#include "Mesh.h"

class MeshTestUtility {
public:
  static bool checkMeshDofConnectivities(Teuchos::RCP<Mesh> mesh);
  static bool checkMeshConsistency(Teuchos::RCP<Mesh> mesh);
};

#endif
