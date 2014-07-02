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

#include <set>

using namespace Intrepid;
using namespace std;

class DofInterpreter {
public:
  virtual GlobalIndexType globalDofCount() = 0;
  virtual set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType rank) = 0;
  
  virtual void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localData,
                                  FieldContainer<double> &globalData, FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  
  virtual void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localStiffnessData, const FieldContainer<double> &localLoadData,
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
  
  virtual void interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_MultiVector &globalDofs) = 0;
  
};

#endif
