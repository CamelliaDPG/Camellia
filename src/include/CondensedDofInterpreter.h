//
//  CondensedDofInterpreter.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/6/14.
//
//

#ifndef Camellia_debug_CondensedDofInterpreter_h
#define Camellia_debug_CondensedDofInterpreter_h

#include "DofInterpreter.h"
#include "Intrepid_FieldContainer.hpp"
#include "Epetra_Vector.h"
#include "Mesh.h"
#include "LagrangeConstraints.h"

using namespace Intrepid;

class CondensedDofInterpreter : public DofInterpreter {
  bool _storeLocalStiffnessMatrices;
  Mesh* _mesh; // for element type lookup, and for determination of which dofs are trace dofs
  LagrangeConstraints* _lagrangeConstraints;
  set<int> _uncondensableVarIDs;
  map<GlobalIndexType, FieldContainer<double> > _localStiffnessMatrices; // will be used by interpretGlobalData if _storeLocalStiffnessMatrices is true
public:
  CondensedDofInterpreter(Mesh* mesh, LagrangeConstraints* lagrangeConstraints, const set<int> &fieldIDsToExclude, bool storeLocalStiffnessMatrices);
  
  void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                          FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices);
  void interpretGlobalData(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs);
};


#endif
