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
#include "Epetra_SerialDenseVector.h"

using namespace Intrepid;

/**
 Note on naming of variables:
 Essentially, the notion is that the condensed dof interpreter sits on a layer above the GDA object (owned by the Mesh),
 which sits atop individual elements.  When we use "interpreted" below we mean interpreted by the GDA.  When we say
 "global" we mean the level above the condensed dof interpreter, the dofs that actually enter the global solve.
 
 See Nate's dissertation, pp. 44-45 for a brief explanation of static condensation.
 **/

class CondensedDofInterpreter : public DofInterpreter {
  bool _storeLocalStiffnessMatrices;
  Mesh* _mesh; // for element type lookup, and for determination of which dofs are trace dofs
  LagrangeConstraints* _lagrangeConstraints;
  set<int> _uncondensibleVarIDs;
  map<GlobalIndexType, FieldContainer<double> > _localStiffnessMatrices; // will be used by interpretGlobalData if _storeLocalStiffnessMatrices is true
  map<GlobalIndexType, FieldContainer<double> > _localLoadVectors;       // will be used by interpretGlobalData if _storeLocalStiffnessMatrices is true
  map<GlobalIndexType, FieldContainer<GlobalIndexType> > _localInterpretedDofIndices;       // will be used by interpretGlobalData if _storeLocalStiffnessMatrices is true

  GlobalIndexType _myGlobalDofIndexOffset;
  IndexType _myGlobalDofIndexCount;
  
  set<GlobalIndexType> _interpretedFluxDofIndices; // the "global" dof indices prior to condensation
  
  map< GlobalIndexType, map< pair<int, int>, FieldContainer<GlobalIndexType> > > _interpretedDofIndicesForBasis; // outer map: cellID is index.  Inner: (varID, sideOrdinal)
  
  map<GlobalIndexType, GlobalIndexType> _interpretedToGlobalDofIndexMap; // maps from the interpreted dof indices to the new ("outer") global dof indices (we only store the ones that are seen by the local MPI rank)
  
  void getSubmatrices(set<int> fieldIndices, set<int> fluxIndices,
                      const FieldContainer<double> &K, Epetra_SerialDenseMatrix &K_field,
                      Epetra_SerialDenseMatrix &K_coupl, Epetra_SerialDenseMatrix &K_flux);
  
  void getSubvectors(set<int> fieldIndices, set<int> fluxIndices, const FieldContainer<double> &b, Epetra_SerialDenseVector &b_field, Epetra_SerialDenseVector &b_flux);
  
  void initializeGlobalDofIndices();
  map<GlobalIndexType, GlobalIndexType> interpretedFluxMapForPartition(PartitionIndexType partition, bool storeFluxDofIndices);
  
public:
  CondensedDofInterpreter(Mesh* mesh, LagrangeConstraints* lagrangeConstraints, const set<int> &fieldIDsToExclude, bool storeLocalStiffnessMatrices);
  
  GlobalIndexType globalDofCount();
  set<GlobalIndexType> globalDofIndicesForPartition(PartitionIndexType rank);
  
  void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localData,
                          FieldContainer<double> &globalData, FieldContainer<GlobalIndexType> &globalDofIndices) {
    if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CondensedDofInterpreter requires both stiffness and load data to be provided.");
    } else {
      FieldContainer<double> globalStiffnessData; // dummy container
      interpretLocalData(cellID, _localStiffnessMatrices[cellID], localData, globalStiffnessData, globalData, globalDofIndices);
    }
  }
  
  void interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localStiffnessData, const FieldContainer<double> &localLoadData,
                          FieldContainer<double> &globalStiffnessData, FieldContainer<double> &globalLoadData, FieldContainer<GlobalIndexType> &globalDofIndices);
  
  virtual void interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<double> &localCoefficients, Epetra_MultiVector &globalCoefficients);
  
  void interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
                                       FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices);
  
  void interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localDofs, const Epetra_MultiVector &globalDofs);

  set<GlobalIndexType> globalDofIndicesForCell(GlobalIndexType cellID);
  
  bool varDofsAreCondensible(int varID, int sideOrdinal, DofOrderingPtr dofOrdering);
};


#endif
