//
//  CondensedDofInterpreter.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/6/14.
//
//


#include "CondensedDofInterpreter.h"

#include "Epetra_SerialDenseSolver.h"
#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"
#include "Epetra_DataAccess.h"

#include <Teuchos_GlobalMPISession.hpp>
#include "MPIWrapper.h"

CondensedDofInterpreter::CondensedDofInterpreter(Mesh* mesh, LagrangeConstraints* lagrangeConstraints, const set<int> &fieldIDsToExclude, bool storeLocalStiffnessMatrices) {
  _mesh = mesh;
  _lagrangeConstraints = lagrangeConstraints;
  _storeLocalStiffnessMatrices = storeLocalStiffnessMatrices;
  _uncondensibleVarIDs.insert(fieldIDsToExclude.begin(),fieldIDsToExclude.end());
  
  int numGlobalConstraints = lagrangeConstraints->numGlobalConstraints();
  for (int i=0; i<numGlobalConstraints; i++) {
    set<int> constrainedVars = lagrangeConstraints->getGlobalConstraint(i).linearTerm()->varIDs();
    _uncondensibleVarIDs.insert(constrainedVars.begin(), constrainedVars.end());
  }
  
  int numElementConstraints = lagrangeConstraints->numElementConstraints();
  for (int i=0; i<numElementConstraints; i++) {
    set<int> constrainedVars = lagrangeConstraints->getElementConstraint(i).linearTerm()->varIDs();
    _uncondensibleVarIDs.insert(constrainedVars.begin(), constrainedVars.end());
  }
  
  initializeGlobalDofIndices();
}

void CondensedDofInterpreter::getSubmatrices(set<int> fieldIndices, set<int> fluxIndices,
                                             const FieldContainer<double> &K, Epetra_SerialDenseMatrix &K_field,
                                             Epetra_SerialDenseMatrix &K_coupl, Epetra_SerialDenseMatrix &K_flux) {
  int numFieldDofs = fieldIndices.size();
  int numFluxDofs = fluxIndices.size();
  K_field.Reshape(numFieldDofs,numFieldDofs);
  K_flux.Reshape(numFluxDofs,numFluxDofs);
  K_coupl.Reshape(numFieldDofs,numFluxDofs); // upper right hand corner matrix - symmetry gets the other
  
  set<int>::iterator dofIt1;
  set<int>::iterator dofIt2;
  
  int i,j,j_flux,j_field;
  i = 0;
  for (dofIt1 = fieldIndices.begin();dofIt1!=fieldIndices.end();dofIt1++){
    int rowInd = *dofIt1;
    j_flux = 0;
    j_field = 0;
    
    // get block field matrices
    for (dofIt2 = fieldIndices.begin();dofIt2!=fieldIndices.end();dofIt2++){
      int colInd = *dofIt2;
      //      cout << "rowInd, colInd = " << rowInd << ", " << colInd << endl;
      K_field(i,j_field) = K(rowInd,colInd);
      j_field++;
    }
    
    // get field/flux couplings
    for (dofIt2 = fluxIndices.begin();dofIt2!=fluxIndices.end();dofIt2++){
      int colInd = *dofIt2;
      K_coupl(i,j_flux) = K(rowInd,colInd);
      j_flux++;
    }
    i++;
  }
  
  // get flux coupling terms
  i = 0;
  for (dofIt1 = fluxIndices.begin();dofIt1!=fluxIndices.end();dofIt1++){
    int rowInd = *dofIt1;
    j = 0;
    for (dofIt2 = fluxIndices.begin();dofIt2!=fluxIndices.end();dofIt2++){
      int colInd = *dofIt2;
      K_flux(i,j) = K(rowInd,colInd);
      j++;
    }
    i++;
  }
}

void CondensedDofInterpreter::getSubvectors(set<int> fieldIndices, set<int> fluxIndices, const FieldContainer<double> &b, Epetra_SerialDenseVector &b_field, Epetra_SerialDenseVector &b_flux){
  
  int numFieldDofs = fieldIndices.size();
  int numFluxDofs = fluxIndices.size();
  
  b_field.Resize(numFieldDofs);
  b_flux.Resize(numFluxDofs);
  set<int>::iterator dofIt;
  int i;
  i = 0;
  for (dofIt=fieldIndices.begin();dofIt!=fieldIndices.end();dofIt++){
    int ind = *dofIt;
    b_field(i) = b(ind);
    i++;
  }
  i = 0;
  for (dofIt=fluxIndices.begin();dofIt!=fluxIndices.end();dofIt++){
    int ind = *dofIt;
    b_flux(i) = b(ind);
    i++;
  }
}

bool CondensedDofInterpreter::varDofsAreCondensible(int varID, int sideOrdinal, DofOrderingPtr dofOrdering) {
  int sideCount = dofOrdering->getNumSidesForVarID(varID);
  BasisPtr basis = dofOrdering->getBasis(varID, sideOrdinal);
  
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  
  bool isL2 =  (fs==IntrepidExtendedTypes::FUNCTION_SPACE_HVOL)
  || (fs==IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HVOL)
  || (fs==IntrepidExtendedTypes::FUNCTION_SPACE_TENSOR_HVOL);
  
  return (isL2) && (sideCount==1) && (_uncondensibleVarIDs.find(varID) == _uncondensibleVarIDs.end());
}

map<GlobalIndexType, IndexType> CondensedDofInterpreter::interpretedFluxMapForPartition(PartitionIndexType partition, bool storeFluxDofIndices) { // add the partitionDofOffset to get the globalDofIndices
  
  map<GlobalIndexType, IndexType> interpretedFluxMap;
  
  vector< GlobalIndexType > localCellIDs = _mesh->globalDofAssignment()->cellsInPartition(partition);
  
  set<GlobalIndexType> innerGlobalFluxDofs;
  
  set<GlobalIndexType> globalDofIndicesForPartition = _mesh->globalDofIndicesForPartition(partition);
  
  vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
  vector< GlobalIndexType >::iterator cellIDIt;
  
  IndexType partitionLocalDofIndex = 0;
  
  for (cellIDIt=localCellIDs.begin(); cellIDIt!=localCellIDs.end(); cellIDIt++){
    GlobalIndexType cellID = *cellIDIt;
    
    DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
    
    for (vector<int>::iterator idIt = trialIDs.begin();idIt!=trialIDs.end();idIt++){
      int trialID = *idIt;
      int numSides = trialOrder->getNumSidesForVarID(trialID);
      
      for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
        BasisPtr basis = trialOrder->getBasis(trialID, sideOrdinal);
        
        FieldContainer<double> dummyLocalBasisData(basis->getCardinality());
        FieldContainer<double> dummyGlobalData;
        FieldContainer<GlobalIndexType> interpretedDofIndicesForBasis;
        vector< int > localDofIndicesForBasis = trialOrder->getDofIndices(trialID,sideOrdinal);
        
        _mesh->interpretLocalBasisData(cellID, trialID, sideOrdinal, dummyLocalBasisData, dummyGlobalData, interpretedDofIndicesForBasis);
        
        if (storeFluxDofIndices) {
          pair< int, int > basisIdentifier = make_pair(trialID,sideOrdinal);
          _interpretedDofIndicesForBasis[cellID][basisIdentifier] = interpretedDofIndicesForBasis;
        }
        
        bool isCondensible = varDofsAreCondensible(trialID, sideOrdinal, trialOrder);
        
        for (int dofOrdinal=0; dofOrdinal < interpretedDofIndicesForBasis.size(); dofOrdinal++) {
          GlobalIndexType globalDofIndex = interpretedDofIndicesForBasis(dofOrdinal);
          
          bool isOwnedByThisPartition = (globalDofIndicesForPartition.find(globalDofIndex) == globalDofIndicesForPartition.end());
          
          if (!isCondensible) {
            if (storeFluxDofIndices) {
              _interpretedFluxDofIndices.insert(globalDofIndex);
            }
          }
          
          if (isOwnedByThisPartition && !isCondensible) {
            if (innerGlobalFluxDofs.find(globalDofIndex) != innerGlobalFluxDofs.end()) {
              interpretedFluxMap[globalDofIndex] = partitionLocalDofIndex++;
              innerGlobalFluxDofs.insert(globalDofIndex);
            }
          }
        }
      }
    }
  }
  
  return interpretedFluxMap;
}

void CondensedDofInterpreter::initializeGlobalDofIndices() {
  PartitionIndexType rank = Teuchos::GlobalMPISession::getRank();
  map<GlobalIndexType, IndexType> partitionLocalFluxMap = interpretedFluxMapForPartition(rank, true);
  
  int numRanks = Teuchos::GlobalMPISession::getNProc();
  FieldContainer<GlobalIndexType> fluxDofCountForRank(numRanks);
  
  _myGlobalDofIndexCount = partitionLocalFluxMap.size();
  fluxDofCountForRank(rank) = _myGlobalDofIndexCount;
  
  MPIWrapper::entryWiseSum(fluxDofCountForRank);
  
  _myGlobalDofIndexOffset = 0;
  for (int i=0; i<rank; i++){
    _myGlobalDofIndexOffset += fluxDofCountForRank(i);
  }

  // initialize _interpretedToGlobalDofIndexMap for the guys we own
  for (map<GlobalIndexType, IndexType>::iterator entryIt = partitionLocalFluxMap.begin(); entryIt != partitionLocalFluxMap.end(); entryIt++) {
    _interpretedToGlobalDofIndexMap[entryIt->first] = entryIt->second + _myGlobalDofIndexOffset;
  }
  
  map< PartitionIndexType, map<GlobalIndexType, GlobalIndexType> > partitionInterpretedFluxMap;

  // fill in the guys we don't own but do see
  for (set<GlobalIndexType>::iterator interpretedFluxIt=_interpretedFluxDofIndices.begin(); interpretedFluxIt != _interpretedFluxDofIndices.end(); interpretedFluxIt++) {
    GlobalIndexType interpretedFlux = *interpretedFluxIt;
    if (_interpretedToGlobalDofIndexMap.find(interpretedFlux) == _interpretedToGlobalDofIndexMap.end()) {
      // not a local guy, then
      PartitionIndexType owningPartition = _mesh->partitionForGlobalDofIndex(interpretedFlux);
      if (partitionInterpretedFluxMap.find(owningPartition) == partitionInterpretedFluxMap.end()) {
        partitionLocalFluxMap = interpretedFluxMapForPartition(owningPartition, false);
        GlobalIndexType owningPartitionDofOffset = 0;
        for (int i=0; i<owningPartition; i++){
          owningPartitionDofOffset += fluxDofCountForRank(i);
        }
        map<GlobalIndexType, GlobalIndexType> owningPartitionInterpretedToGlobalDofIndexMap;
        for (map<GlobalIndexType, IndexType>::iterator entryIt = partitionLocalFluxMap.begin(); entryIt != partitionLocalFluxMap.end(); entryIt++) {
          owningPartitionInterpretedToGlobalDofIndexMap[entryIt->first] = entryIt->second + _myGlobalDofIndexOffset;
        }
      }
      _interpretedToGlobalDofIndexMap[interpretedFlux] = partitionInterpretedFluxMap[owningPartition][interpretedFlux];
    }
  }
}

GlobalIndexType CondensedDofInterpreter::globalDofCount() {
  return MPIWrapper::sum(_myGlobalDofIndexCount);
}

set<GlobalIndexType> CondensedDofInterpreter::globalDofIndicesForPartition(PartitionIndexType rank) {
  if (rank == Teuchos::GlobalMPISession::getRank()) {
    set<GlobalIndexType> myGlobalDofIndices;
    for (GlobalIndexType dofIndex = _myGlobalDofIndexOffset; dofIndex < _myGlobalDofIndexCount; dofIndex++) {
      myGlobalDofIndices.insert(dofIndex);
    }
    return myGlobalDofIndices;
  } else {
    cout << "globalDofIndicesForPartition() requires that rank be the local MPI rank!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndicesForPartition() requires that rank be the local MPI rank!");
  }
}

void CondensedDofInterpreter::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localStiffnessData, const FieldContainer<double> &localLoadData,
                                                 FieldContainer<double> &globalStiffnessData, FieldContainer<double> &globalLoadData,
                                                 FieldContainer<GlobalIndexType> &globalDofIndices) {
  // NOTE: cellID *MUST* belong to this partition.
  
  FieldContainer<double> interpretedStiffnessData, interpretedLoadData;
  
  FieldContainer<GlobalIndexType> interpretedDofIndices;
  
  _mesh->DofInterpreter::interpretLocalData(cellID, localStiffnessData, localLoadData,
                                            interpretedStiffnessData, interpretedLoadData, interpretedDofIndices);
  
  if (_storeLocalStiffnessMatrices) {
    _localStiffnessMatrices[cellID] = localStiffnessData;
    _localLoadVectors[cellID] = interpretedLoadData;
    _localInterpretedDofIndices[cellID] = globalDofIndices;
  }
  
  set<int> fieldIndices, fluxIndices; // which are fields and which are fluxes in the interpreted data containers
  for (int dofOrdinal=0; dofOrdinal < interpretedDofIndices.size(); dofOrdinal++) {
    GlobalIndexType interpretedDofIndex = interpretedDofIndices(dofOrdinal);
    if (_interpretedFluxDofIndices.find(interpretedDofIndex) == _interpretedFluxDofIndices.end()) {
      fieldIndices.insert(dofOrdinal);
    } else {
      fluxIndices.insert(dofOrdinal);
    }
  }
  
  int fieldCount = fieldIndices.size();
  int fluxCount = fluxIndices.size();
  
  Epetra_SerialDenseMatrix D, B, K_flux;
 
  getSubmatrices(fieldIndices, fluxIndices, interpretedStiffnessData, D, B, K_flux);
  
  // reduce matrix
  Epetra_SerialDenseMatrix Bcopy = B;
  Epetra_SerialDenseSolver solver;

  Epetra_SerialDenseMatrix DinvB(fieldCount,fluxCount);
  solver.SetMatrix(D);
  solver.SetVectors(DinvB, Bcopy);
  bool equilibrated = false;
  if ( solver.ShouldEquilibrate() ) {
    solver.EquilibrateMatrix();
    solver.EquilibrateRHS();
    equilibrated = true;
  }
  solver.Solve();
  if (equilibrated)
    solver.UnequilibrateLHS();
  
  K_flux.Multiply('T','N',-1.0,B,DinvB,1.0); // assemble condensed matrix - A - B^T*inv(D)*B
  
  // reduce vector
  Epetra_SerialDenseVector Dinvf(fieldCount);
  Epetra_SerialDenseVector BtDinvf(fluxCount);
  Epetra_SerialDenseVector b_field, b_flux;
  getSubvectors(fieldIndices, fluxIndices, interpretedLoadData, b_field, b_flux);
  //    solver.SetMatrix(D);
  solver.SetVectors(Dinvf, b_field);
  equilibrated = false;
  if ( solver.ShouldEquilibrate() ) {
    solver.EquilibrateMatrix();
    solver.EquilibrateRHS();
    equilibrated = true;
  }
  solver.Solve();
  if (equilibrated)
    solver.UnequilibrateLHS();
  
  b_flux.Multiply('T','N',-1.0,B,Dinvf,1.0); // condensed RHS - f - B^T*inv(D)*g
  
  // resize output FieldContainers
  globalDofIndices.resize(fluxCount);
  globalStiffnessData.resize( fluxCount, fluxCount );
  globalLoadData.resize( fluxCount );
  
  set<int>::iterator indexIt;
  int i = 0;
  for (indexIt = fluxIndices.begin();indexIt!=fluxIndices.end();indexIt++){
    int localFluxIndex = *indexIt;
    GlobalIndexType interpretedDofIndex = interpretedDofIndices(localFluxIndex);
    int condensedIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
    globalDofIndices(i) = condensedIndex;
    i++;
  }
  
  for (int i=0; i<fluxCount; i++) {
    globalLoadData(i) = b_flux(i);
    for (int j=0; j<fluxCount; j++) {
      globalStiffnessData(i,j) = K_flux(i,j);
    }
  }
}

void CondensedDofInterpreter::interpretGlobalData(GlobalIndexType cellID, FieldContainer<double> &localData, const Epetra_Vector &globalData) {
  // get elem data and submatrix data
  FieldContainer<double> K,rhs;
  FieldContainer<GlobalIndexType> interpretedDofIndices;
  if (! _storeLocalStiffnessMatrices ){
    // getElemData(elem,K,rhs);
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CondensedDofInterpreter::interpretGlobalData() doesn't yet support _storeLocalStiffnessMatrices = false");
  } else {
    K = _localStiffnessMatrices[cellID];
    rhs = _localLoadVectors[cellID];
    interpretedDofIndices = _localInterpretedDofIndices[cellID];
  }
  
  map<GlobalIndexType, int> interpretedDofIndexToInterpretedCellDofOrdinal;
  for (int i=0; i<interpretedDofIndices.size(); i++) {
    interpretedDofIndexToInterpretedCellDofOrdinal[interpretedDofIndices(i)] = i;
  }
  
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  
  set<int> fieldIndices, fluxIndices; // element-local indices -- except that these are now row/column indices into the element's interpreted stiffness matrix and load vector
  
  vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
  for (vector<int>::iterator idIt = trialIDs.begin();idIt!=trialIDs.end();idIt++){
    int trialID = *idIt;
    int numSides = trialOrder->getNumSidesForVarID(trialID);
    
    for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
      FieldContainer<GlobalIndexType> interpretedDofIndices = _interpretedDofIndicesForBasis[cellID][make_pair(trialID,sideOrdinal)];
      bool isCondensible = varDofsAreCondensible(trialID, sideOrdinal, trialOrder);
      for (int i=0; i<interpretedDofIndices.size(); i++) {
        int ordinal = interpretedDofIndexToInterpretedCellDofOrdinal[interpretedDofIndices[i]];
        if (isCondensible) {
          fieldIndices.insert( ordinal );
        } else {
          fluxIndices.insert( ordinal );
        }
      }
    }
  }
  
  int fieldCount = fieldIndices.size();
  int fluxCount = fluxIndices.size();
  
    Epetra_SerialDenseVector flux_dofs(fluxCount); // fill this in with data from globalData
  
  for (int fluxOrdinal=0; fluxOrdinal<fluxCount; fluxOrdinal++) {
    int interpretedDofIndex = interpretedDofIndices[fluxOrdinal];
    flux_dofs[fluxOrdinal] = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
  }
  
  Epetra_SerialDenseMatrix D, B, fluxMat;
  Epetra_SerialDenseVector b_field, b_flux, field_dofs(fieldCount);
  getSubmatrices(fieldIndices, fluxIndices, K, D, B, fluxMat);
  getSubvectors(fieldIndices, fluxIndices, rhs, b_field, b_flux);
  b_field.Multiply('N','N',-1.0,B,flux_dofs,1.0);
  
  // solve for field dofs
  Epetra_SerialDenseSolver solver;
  solver.SetMatrix(D);
  solver.SetVectors(field_dofs,b_field);
  bool equilibrated = false;
  if ( solver.ShouldEquilibrate() ) {
    solver.EquilibrateMatrix();
    solver.EquilibrateRHS();
    equilibrated = true;
  }
  solver.Solve();
  if (equilibrated)
    solver.UnequilibrateLHS();
  
  // TODO: construct an Epetra_Vector for the globalData as seen by the Mesh (i.e. fill in the field and flux data in the appropriate slots
  //       then, call mesh's interpretGlobalData()...
  
  cout << "ERROR: interpretGlobalData implementation not yet completed...\n";
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: interpretGlobalData implementation not yet completed...\n");
  
}