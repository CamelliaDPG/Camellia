//
//  DofInterpreter.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/23/14.
//
//

#include "DofInterpreter.h"
#include "Mesh.h"

void DofInterpreter::interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients) {
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  FieldContainer<double> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++) {
    int trialID = *trialIDIt;
    int sideCount = trialOrder->getNumSidesForVarID(trialID);
    for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++) {
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++) {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      FieldContainer<double> fittedGlobalCoefficients;
      FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++) {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, 0, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
        //        cout << "global coefficient " << globalDofIndex << " = " << fittedGlobalCoefficients[i] << endl;
      }
    }
  }
}

void DofInterpreter::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localStiffnessData, const FieldContainer<double> &localLoadData,
                                        FieldContainer<double> &globalStiffnessData, FieldContainer<double> &globalLoadData, FieldContainer<GlobalIndexType> &globalDofIndices) {
  this->interpretLocalData(cellID,localStiffnessData,globalStiffnessData,globalDofIndices);
  FieldContainer<GlobalIndexType> globalDofIndicesForStiffness = globalDofIndices; // copy (for debugging/inspection purposes)
  this->interpretLocalData(cellID,localLoadData,globalLoadData,globalDofIndices);
  for (int i=0; i<globalDofIndicesForStiffness.size(); i++) {
    if (globalDofIndicesForStiffness[i] != globalDofIndices[i]) {
      cout << "ERROR: the vector and matrix dof indices differ...\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: the vector and matrix dof indices differ...\n");
    }
  }
}
