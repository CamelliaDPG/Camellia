//
//  MeshPolyOrderFunction.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 5/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#ifndef Camellia_debug_MeshPolyOrderFunction_h
#define Camellia_debug_MeshPolyOrderFunction_h

#include "TypeDefs.h"

#include "Mesh.h"

namespace Camellia {
	class MeshPolyOrderFunction : public Function {
	  Teuchos::RCP<Mesh> _mesh;
	public:
	  MeshPolyOrderFunction(Teuchos::RCP<Mesh> mesh) : Function(0) { _mesh = mesh;} // scalar
	  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache);
	};
}

#endif
