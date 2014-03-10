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
                                  FieldContainer<double> &globalData, FieldContainer<GlobalIndexType> &globalDofIndices, bool accumulate=true) = 0;
  
  virtual void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localStiffnessData, const FieldContainer<double> &localLoadData,
                                  FieldContainer<double> &globalStiffnessData, FieldContainer<double> &globalLoadData, FieldContainer<GlobalIndexType> &globalDofIndices) {
    bool accumulate = true;
    this->interpretLocalData(cellID,localStiffnessData,globalStiffnessData,globalDofIndices, accumulate);
    this->interpretLocalData(cellID,localLoadData,globalLoadData,globalDofIndices, accumulate);
  }
  
  virtual void interpretGlobalData(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_Vector &globalDofs, bool accumulate=true) = 0;
  
  
};

#endif
