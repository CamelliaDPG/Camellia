//
//  DofInterpreter.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/6/14.
//
//

#ifndef Camellia_debug_DofInterpreter_h
#define Camellia_debug_DofInterpreter_h

#include "Intrepid_FieldContainer.hpp"
#include "Epetra_Vector.h"

#include "IndexType.h"

class DofInterpreter {
public:
  virtual void interpretLocalDofs(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                          FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  virtual void interpretGlobalDofs(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs) = 0;
};

#endif
