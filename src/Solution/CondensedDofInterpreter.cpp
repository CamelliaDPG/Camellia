//
//  CondensedDofInterpreter.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/6/14.
//
//

#include "CondensedDofInterpreter.h"

CondensedDofInterpreter::CondensedDofInterpreter(Mesh* mesh, LagrangeConstraints* lagrangeConstraints, const set<int> &fieldIDsToExclude, bool storeLocalStiffnessMatrices) {
  _mesh = mesh;
  _lagrangeConstraints = lagrangeConstraints;
  _storeLocalStiffnessMatrices = storeLocalStiffnessMatrices;
  _uncondensableVarIDs.insert(fieldIDsToExclude.begin(),fieldIDsToExclude.end());
  
}

void CondensedDofInterpreter::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localDofs,
                                                 FieldContainer<double> &globalDofs, FieldContainer<GlobalIndexType> &globalDofIndices) {
  if (_storeLocalStiffnessMatrices && (localDofs.rank() == 2)) {
    _localStiffnessMatrices[cellID] = localDofs;
  }
}
void CondensedDofInterpreter::interpretGlobalData(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs) {
  
}