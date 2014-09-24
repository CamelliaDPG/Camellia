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

class Mesh;
typedef Teuchos::RCP<Mesh> MeshPtr;

using namespace Intrepid;
using namespace std;

class DofInterpreter {
  MeshPtr _mesh;
public:
  DofInterpreter(MeshPtr mesh) : _mesh(mesh) {}
  virtual GlobalIndexType globalDofCount() = 0;
  virtual set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType rank) = 0;
  
  virtual void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localData,
                                  FieldContainer<double> &globalData, FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  
  virtual void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localStiffnessData, const FieldContainer<double> &localLoadData,
                                  FieldContainer<double> &globalStiffnessData, FieldContainer<double> &globalLoadData, FieldContainer<GlobalIndexType> &globalDofIndices);
  virtual void interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients);
  
  virtual void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
                                               FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices) = 0;
  
  virtual void interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_MultiVector &globalDofs) = 0;
  
  virtual set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID) = 0;
  
  virtual ~DofInterpreter() {}
};

#endif
