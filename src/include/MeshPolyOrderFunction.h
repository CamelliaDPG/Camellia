// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// @HEADER

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

namespace Camellia
{
class MeshPolyOrderFunction : public TFunction<double>
{
  Teuchos::RCP<Mesh> _mesh;
public:
  MeshPolyOrderFunction(Teuchos::RCP<Mesh> mesh) : TFunction<double>(0)
  {
    _mesh = mesh; // scalar
  }
  
  using TFunction<double>::values;
  void values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache) override;
};
}

#endif
