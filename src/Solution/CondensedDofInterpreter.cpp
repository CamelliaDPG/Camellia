//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "CondensedDofInterpreter.h"

#include "Epetra_SerialDenseSolver.h"
#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"
#include "Epetra_DataAccess.h"

#include "Epetra_Distributor.h"
#include "Epetra_SerialComm.h"

#include "CamelliaDebugUtility.h"
#include "GDAMinimumRule.h"
#include "GlobalDofAssignment.h"
#include "MPIWrapper.h"
#include "RHS.h"
#include "SerialDenseWrapper.h"

using namespace Camellia;

template <typename Scalar>
CondensedDofInterpreter<Scalar>::CondensedDofInterpreter(MeshPtr mesh, TIPPtr<Scalar> ip, TRHSPtr<Scalar> rhs,
                                                         TBCPtr<Scalar> bc, LagrangeConstraints* lagrangeConstraints,
                                                         const set<int> &fieldIDsToExclude,
                                                         bool storeLocalStiffnessMatrices,
                                                         set<GlobalIndexType> offRankCellsToInclude ) : DofInterpreter(mesh)
{
  _mesh = mesh;
  _ip = ip;
  _rhs = rhs;
  _bc = bc;
  _lagrangeConstraints = lagrangeConstraints;
  _storeLocalStiffnessMatrices = storeLocalStiffnessMatrices;
  _uncondensibleVarIDs.insert(fieldIDsToExclude.begin(),fieldIDsToExclude.end());
  _offRankCellsToInclude = offRankCellsToInclude;
  _skipLocalFields = false;
  
  _meshLastKnownGlobalDofCount = _mesh->globalDofCount();

  int numGlobalConstraints = lagrangeConstraints->numGlobalConstraints();
  for (int i=0; i<numGlobalConstraints; i++)
  {
    set<int> constrainedVars = lagrangeConstraints->getGlobalConstraint(i).linearTerm()->varIDs();
    _uncondensibleVarIDs.insert(constrainedVars.begin(), constrainedVars.end());
  }

  int numElementConstraints = lagrangeConstraints->numElementConstraints();
  for (int i=0; i<numElementConstraints; i++)
  {
    set<int> constrainedVars = lagrangeConstraints->getElementConstraint(i).linearTerm()->varIDs();
    _uncondensibleVarIDs.insert(constrainedVars.begin(), constrainedVars.end());
  }
  
  initializeGlobalDofIndices();
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::reinitialize()
{
  _meshLastKnownGlobalDofCount = _mesh->globalDofCount(); // stored for a basic staleness check (used right now in numGlobalDofs()).
  
  _localLoadVectors.clear();
  _localStiffnessMatrices.clear();
  _localInterpretedDofIndices.clear();

  initializeGlobalDofIndices();
}

template <typename Scalar>
long long CondensedDofInterpreter<Scalar>::approximateStiffnessAndLoadMemoryCost()
{
  long long memoryCost = 0;
  for (auto entry : _localLoadVectors)
  {
    memoryCost += entry.second.size() * sizeof(Scalar);
  }
  
  for (auto entry : _localStiffnessMatrices)
  {
    memoryCost += entry.second.size() * sizeof(Scalar);
  }
  return memoryCost;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::clearStiffnessAndLoad()
{
  _localLoadVectors.clear();
  _localStiffnessMatrices.clear();
  _fluxToFieldMapForIterativeSolves.clear();
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::computeAndStoreLocalStiffnessAndLoad(GlobalIndexType cellID)
{
//  cout << "CondensedDofInterpreter: computing stiffness and load for cell " << cellID << endl;
  int numTrialDofs = _mesh->getElementType(cellID)->trialOrderPtr->totalDofs();
  BasisCachePtr cellBasisCache = BasisCache::basisCacheForCell(_mesh, cellID);
  BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(_mesh, cellID, true);
  _localStiffnessMatrices[cellID] = Camellia::FieldContainer<Scalar>(1,numTrialDofs,numTrialDofs);
  _localLoadVectors[cellID] = Camellia::FieldContainer<Scalar>(1,numTrialDofs);
  _mesh->bilinearForm()->localStiffnessMatrixAndRHS(_localStiffnessMatrices[cellID], _localLoadVectors[cellID], _ip, ipBasisCache, _rhs, cellBasisCache);

  _localStiffnessMatrices[cellID].resize(numTrialDofs,numTrialDofs);
  _localLoadVectors[cellID].resize(numTrialDofs);

  Camellia::FieldContainer<Scalar> interpretedStiffnessData, interpretedLoadData;

  Camellia::FieldContainer<GlobalIndexType> interpretedDofIndices;

  _mesh->DofInterpreter::interpretLocalData(cellID, _localStiffnessMatrices[cellID], _localLoadVectors[cellID],
      interpretedStiffnessData, interpretedLoadData, interpretedDofIndices);

  _localInterpretedDofIndices[cellID] = interpretedDofIndices;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::getLocalData(GlobalIndexType cellID, Camellia::FieldContainer<Scalar> &stiffness, Camellia::FieldContainer<Scalar> &load, Camellia::FieldContainer<GlobalIndexType> &interpretedDofIndices)
{
  if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end())
  {
    computeAndStoreLocalStiffnessAndLoad(cellID);
  }

  stiffness = _localStiffnessMatrices[cellID];
  load = _localLoadVectors[cellID];
  interpretedDofIndices = _localInterpretedDofIndices[cellID];
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::getLocalData(GlobalIndexType cellID, Teuchos::RCP<Epetra_SerialDenseSolver> &fieldSolver,
                                                   Epetra_SerialSymDenseMatrix &FieldField, Epetra_SerialDenseMatrix &FieldFlux, Epetra_SerialDenseVector &b_field,
                                                   Camellia::FieldContainer<GlobalIndexType> &interpretedDofIndices, set<int> &fieldIndices, set<int> &fluxIndices)
{
  if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end())
  {
    computeAndStoreLocalStiffnessAndLoad(cellID);
  }
  
  // TODO: add caching of fieldSolver, B, b_field
  // (NOTE: when we do this, need to copy b_field before returning; it's modified by caller)
  
  Camellia::FieldContainer<double> K = _localStiffnessMatrices[cellID];
  Camellia::FieldContainer<double> rhs = _localLoadVectors[cellID];
  interpretedDofIndices = _localInterpretedDofIndices[cellID];
  
//  cout << "rhs for cell " << cellID << ":\n" << rhs;
  
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  vector<int> fluxIndicesVector = fluxIndexLookupLocalCell(cellID);
  fluxIndices.insert(fluxIndicesVector.begin(),fluxIndicesVector.end());
  for (int localDofOrdinal=0; localDofOrdinal<trialOrder->totalDofs(); localDofOrdinal++)
  {
    if (fluxIndices.find(localDofOrdinal) == fluxIndices.end())
    {
      fieldIndices.insert(localDofOrdinal);
    }
  }
  
  Epetra_SerialDenseMatrix fluxMat;
  Epetra_SerialDenseVector b_flux;
  getSubmatrices(fieldIndices, fluxIndices, K, FieldField, FieldFlux, fluxMat);
  
//  cout << "rhs for cell " << cellID << ":\n" << rhs;
//  print("fieldIndices",fieldIndices);
//  print("fluxIndices",fluxIndices);
  
  getSubvectors(fieldIndices, fluxIndices, rhs, b_field, b_flux);
  
//  cout << "b_field:\n" << b_field;
//  cout << "b_flux:\n" << b_flux;
  
  Epetra_SerialSpdDenseSolver* spdSolver = new Epetra_SerialSpdDenseSolver();
  spdSolver->SetMatrix(FieldField);
  
  fieldSolver = Teuchos::rcp( spdSolver );
  
//  cout << "FieldField:\n" << FieldField;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::getSubmatrices(set<int> fieldIndices, set<int> fluxIndices,
    const Camellia::FieldContainer<Scalar> &K, Epetra_SerialDenseMatrix &K_field,
    Epetra_SerialDenseMatrix &K_coupl, Epetra_SerialDenseMatrix &K_flux)
{
  int numFieldDofs = fieldIndices.size();
  int numFluxDofs = fluxIndices.size();
  K_field.Reshape(numFieldDofs,numFieldDofs);
  K_flux.Reshape(numFluxDofs,numFluxDofs);
  K_coupl.Reshape(numFieldDofs,numFluxDofs); // upper right hand corner matrix - symmetry gets the other

  int i,j,j_flux,j_field;
  i = 0;
  for (int fieldRowIndex : fieldIndices)
  {
    j_flux = 0;
    j_field = 0;

    // get block field matrices
    for (int fieldColIndex : fieldIndices)
    {
      //      cout << "rowInd, colInd = " << rowInd << ", " << colInd << endl;
      K_field(i,j_field) = K(fieldRowIndex,fieldColIndex);
      j_field++;
    }

    // get field/flux couplings
    for (int fluxColIndex : fluxIndices)
    {
      K_coupl(i,j_flux) = K(fieldRowIndex,fluxColIndex);
      j_flux++;
    }
    i++;
  }

  // get flux coupling terms
  i = 0;
  for (int fluxRowIndex : fluxIndices)
  {
    j = 0;
    for (int fluxColIndex : fluxIndices)
    {
      K_flux(i,j) = K(fluxRowIndex,fluxColIndex);
      j++;
    }
    i++;
  }
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::getSubvectors(set<int> fieldIndices, set<int> fluxIndices, const Camellia::FieldContainer<Scalar> &b, Epetra_SerialDenseVector &b_field, Epetra_SerialDenseVector &b_flux)
{

  int numFieldDofs = fieldIndices.size();
  int numFluxDofs = fluxIndices.size();

  b_field.Resize(numFieldDofs);
  b_flux.Resize(numFluxDofs);
  set<int>::iterator dofIt;
  int i;
  i = 0;
  for (dofIt=fieldIndices.begin(); dofIt!=fieldIndices.end(); dofIt++)
  {
    int ind = *dofIt;
    b_field(i) = b(ind);
    i++;
  }
  i = 0;
  for (dofIt=fluxIndices.begin(); dofIt!=fluxIndices.end(); dofIt++)
  {
    int ind = *dofIt;
    b_flux(i) = b(ind);
    i++;
  }
}

template <typename Scalar>
bool CondensedDofInterpreter<Scalar>::canSkipLocalFieldInInterpretGlobalCoefficients() const
{
  return _skipLocalFields;
}

template <typename Scalar>
GlobalIndexType CondensedDofInterpreter<Scalar>::condensedGlobalIndex(GlobalIndexType meshGlobalIndex)
{
  if (_interpretedToGlobalDofIndexMap.find(meshGlobalIndex) != _interpretedToGlobalDofIndexMap.end())
  {
    return _interpretedToGlobalDofIndexMap[meshGlobalIndex];
  }
  else
  {
    return -1;
  }
}

template <typename Scalar>
set<int> CondensedDofInterpreter<Scalar>::condensibleVariableIDs()
{
  set<int> condensibleVariableIDs;
  vector<VarPtr> fields = _mesh->varFactory()->fieldVars();
  for (VarPtr fieldVar : fields)
  {
    if (_uncondensibleVarIDs.find(fieldVar->ID()) == _uncondensibleVarIDs.end())
    {
      condensibleVariableIDs.insert(fieldVar->ID());
    }
  }
  return condensibleVariableIDs;
}

template <typename Scalar>
vector<int> CondensedDofInterpreter<Scalar>::fieldRowIndices(GlobalIndexType cellID, int condensibleVarID)
{
  // this is not a particularly efficient way of doing this, but it's not likely to add up to much total expense, especially
  // now that we cache the result, for all but the cells with local exceptions to the condensation pattern.
  
  auto uncondensibleLocalDofsEntry = _cellLocalUncondensibleDofIndices.find(cellID);
  bool hasLocalExceptions = uncondensibleLocalDofsEntry != _cellLocalUncondensibleDofIndices.end();
  
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  
  pair<DofOrdering*,int> key = {trialOrder.get(),condensibleVarID};
  if ((_fieldRowIndices.find(key) == _fieldRowIndices.end()) || hasLocalExceptions)
  {
    // the way we order field dof indices is according to their index order in the local uncondensed stiffness matrix
    
    set<int> fieldIndices; // all field indices for the cell
    set<int> trialIDs = trialOrder->getVarIDs();
    for (int trialID : trialIDs)
    {
      const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
      for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
      {
        int sideOrdinal = *sideIt;
        vector<int> varIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
        if (varDofsAreUsuallyCondensible(trialID, sideOrdinal, trialOrder))
        {
          if (!hasLocalExceptions)
          {
            fieldIndices.insert(varIndices.begin(), varIndices.end());
          }
          else
          {
            const vector<int>* uncondensibleDofs = &uncondensibleLocalDofsEntry->second;
            for (int varIndex : varIndices)
            {
              // note: call to find does require that uncondensibleDofs be in order
              if (std::find(uncondensibleDofs->begin(), uncondensibleDofs->end(), varIndex) == uncondensibleDofs->end())
              {
                // this dof is condensible: add it
                fieldIndices.insert(varIndex);
              }
            }
          }
        }
      }
    }

    vector<int> rowIndices;
    const vector<int>* sides = &trialOrder->getSidesForVarID(condensibleVarID);
    TEUCHOS_TEST_FOR_EXCEPTION(sides->size() != 1, std::invalid_argument, "got request for condensible var ID with multiple sides");
    for (int sideOrdinal : *sides)
    {
      vector<int> varIndices = trialOrder->getDofIndices(condensibleVarID, sideOrdinal);
      for (int dofIndexForBasisOrdinal : varIndices)
      {
        int row = 0;
        for (int fieldDofIndex : fieldIndices)
        {
          if (fieldDofIndex == dofIndexForBasisOrdinal)
          {
            rowIndices.push_back(row);
            break;
          }
          row++;
        }
        if (row == fieldIndices.size()) // then we found no match
        {
          rowIndices.push_back(-1); // -1 to signal no matching field index (can happen with singletons, aka local exceptions)
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(rowIndices.size() != varIndices.size(), std::invalid_argument, "Internal error: number of rowIndices does not match the number of varIndices");
    }
    if (hasLocalExceptions)
    {
      return rowIndices;
    }
    else
    {
      _fieldRowIndices[key] = rowIndices;
    }
  }
  
  return _fieldRowIndices[key];
}


template <typename Scalar>
std::vector<int> CondensedDofInterpreter<Scalar>::fluxIndexLookupLocalCell(GlobalIndexType cellID)
{
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  
  // the way we order field dof indices is according to their index order in the local uncondensed stiffness matrix
  set<int> fluxIndices; // all flux indices for the cell
  set<int> trialIDs = trialOrder->getVarIDs();
  for (int trialID : trialIDs)
  {
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      vector<int> varIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      if (!varDofsAreUsuallyCondensible(trialID, sideOrdinal, trialOrder))
      {
        fluxIndices.insert(varIndices.begin(), varIndices.end());
      }
    }
  }
  
  /*
   If the cell has a singleton BC imposed on it (for example), then one of the field row indices is treated as a flux index.
   */
  auto uncondensibleLocalDofsEntry = _cellLocalUncondensibleDofIndices.find(cellID);
  if (uncondensibleLocalDofsEntry != _cellLocalUncondensibleDofIndices.end())
  {
    const vector<int>* uncondensibleDofs = &uncondensibleLocalDofsEntry->second;
    for (int localUncondensibleIndex : *uncondensibleDofs)
    {
      fluxIndices.insert(localUncondensibleIndex);
    }
  }
  
  vector<int> fluxIndicesVector(fluxIndices.begin(),fluxIndices.end());
  return fluxIndicesVector;
}

template <typename Scalar>
Teuchos::RCP<Epetra_SerialDenseMatrix> CondensedDofInterpreter<Scalar>::fluxToFieldMapForIterativeSolves(GlobalIndexType cellID)
{
  // if K_11 is the field-field part of the local stiffness matrix, and K_12 is the field-flux part,
  // return -K_11^(-1) * K_12
  
  if (_fluxToFieldMapForIterativeSolves.find(cellID) == _fluxToFieldMapForIterativeSolves.end())
  {
    
    set<int> fieldIndices, fluxIndices; // which are fields and which are fluxes in the local cell coefficients
    
    Epetra_SerialDenseVector b_field;
    
    Camellia::FieldContainer<Scalar> K,rhs;
    Camellia::FieldContainer<GlobalIndexType> interpretedDofIndices;
    
    Teuchos::RCP<Epetra_SerialDenseSolver> fieldSolver;
    Epetra_SerialSymDenseMatrix FieldField;
    Epetra_SerialDenseMatrix FieldFlux;
    
    getLocalData(cellID, fieldSolver, FieldField, FieldFlux, b_field, interpretedDofIndices, fieldIndices, fluxIndices);
    
    Teuchos::RCP<Epetra_SerialDenseMatrix> fluxToFieldMap = Teuchos::rcp( new Epetra_SerialDenseMatrix(fieldIndices.size(),fluxIndices.size()) );

    fieldSolver->SetVectors(*fluxToFieldMap, FieldFlux);
    
    bool didEquilibriate = false;
    if (fieldSolver->ShouldEquilibrate())
    {
      fieldSolver->EquilibrateMatrix();
      fieldSolver->EquilibrateRHS();
      didEquilibriate = true;
    }
    
    int err = fieldSolver->Solve();
    if (err != 0)
    {
      cout << "WARNING: in CondensedDofInterpreter, fieldSolver returned error code " << err << endl;
    }
    if (didEquilibriate)
    {
      fieldSolver->UnequilibrateLHS();
    }
    
    // negate
    fluxToFieldMap->Scale(-1.0);
    
    _fluxToFieldMapForIterativeSolves[cellID] = fluxToFieldMap;
  }
  
  return _fluxToFieldMapForIterativeSolves[cellID];
}

template <typename Scalar>
set<GlobalIndexType> CondensedDofInterpreter<Scalar>::globalDofIndicesForCell(GlobalIndexType cellID)
{
  set<GlobalIndexType> interpretedDofIndicesForCell = _mesh->globalDofIndicesForCell(cellID);
  set<GlobalIndexType> globalDofIndicesForCell;

  for (set<GlobalIndexType>::iterator interpretedDofIndexIt = interpretedDofIndicesForCell.begin();
       interpretedDofIndexIt != interpretedDofIndicesForCell.end(); interpretedDofIndexIt++)
  {
    GlobalIndexType interpretedDofIndex = *interpretedDofIndexIt;
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) == _interpretedToGlobalDofIndexMap.end())
    {
      // that's OK; we skip the fields...
    }
    else
    {
      GlobalIndexType globalDofIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
      globalDofIndicesForCell.insert(globalDofIndex);
    }
  }

  return globalDofIndicesForCell;
}

template <typename Scalar>
set<GlobalIndexType> CondensedDofInterpreter<Scalar>::globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal, bool upgradeToContinuousBasis, bool tensorTopologyMeansSpaceTime)
{
  set<GlobalIndexType> interpretedDofIndicesForCell = _mesh->globalDofIndicesForVarOnSubcell(varID, cellID, dim, subcellOrdinal, upgradeToContinuousBasis, tensorTopologyMeansSpaceTime);
  set<GlobalIndexType> globalDofIndices;
  
  for (GlobalIndexType interpretedDofIndex : interpretedDofIndicesForCell)
  {
    auto foundEntry = _interpretedToGlobalDofIndexMap.find(interpretedDofIndex);
    if (foundEntry == _interpretedToGlobalDofIndexMap.end())
    {
      // that's OK; we skip the fields...
    }
    else
    {
      GlobalIndexType globalDofIndex = foundEntry->second;
      globalDofIndices.insert(globalDofIndex);
    }
  }
  
  return globalDofIndices;
}

template <typename Scalar>
bool CondensedDofInterpreter<Scalar>::varDofsAreCondensible(GlobalIndexType cellID, int varID, int sideOrdinal, DofOrderingPtr dofOrdering)
{
  // eventually it would be nice to determine which sub-basis ordinals can be condensed, but right now we only
  // condense out the truly discontinuous bases defined for variables on the element interior.
  return varDofsAreUsuallyCondensible(varID, sideOrdinal, dofOrdering) && (_cellLocalUncondensibleDofIndices.find(cellID) == _cellLocalUncondensibleDofIndices.end());
}

template <typename Scalar>
bool CondensedDofInterpreter<Scalar>::varDofsAreUsuallyCondensible(int varID, int sideOrdinal, DofOrderingPtr dofOrdering) const
{
  // eventually it would be nice to determine which sub-basis ordinals can be condensed, but right now we only
  // condense out the truly discontinuous bases defined for variables on the element interior.

  const bool isVolumeVar = dofOrdering->isVolumeVar(varID);
  if (!isVolumeVar) return false;
  
  BasisPtr basis = dofOrdering->getBasis(varID);
  Camellia::EFunctionSpace fs = basis->functionSpace();
  
  bool isDiscontinuous = functionSpaceIsDiscontinuous(fs);
  
  return isDiscontinuous && isVolumeVar && (_uncondensibleVarIDs.find(varID) == _uncondensibleVarIDs.end());
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::importInterpretedMapForNeighborGlobalIndices(const map<PartitionIndexType,set<GlobalIndexType>> &partitionToMeshGlobalIndices)
{
  int rank = _mesh->Comm()->getRank();
  
  // myRequestOwners should be in nondecreasing order (it appears)
  // this is accomplished by virtue of the order in which we traverse partitionToMeshGlobalIndices
  vector<int> myRequestOwners;
  vector<GlobalIndexTypeToCast> myRequest;
  for (auto entry : partitionToMeshGlobalIndices)
  {
    for (GlobalIndexType interpretedGlobalDofIndex : entry.second)
    {
      myRequest.push_back(interpretedGlobalDofIndex);
      myRequestOwners.push_back(entry.first);
    }
  }
  int myRequestCount = myRequest.size();
  
  auto epetraComm = MPIWrapper::EpetraComm(_mesh->Comm());
  Teuchos::RCP<Epetra_Distributor> distributor = MPIWrapper::getDistributor(*epetraComm);
  
  GlobalIndexTypeToCast* myRequestPtr = NULL;
  int *myRequestOwnersPtr = NULL;
  if (myRequest.size() > 0)
  {
    myRequestPtr = &myRequest[0];
    myRequestOwnersPtr = &myRequestOwners[0];
  }
  int numEntriesToExport = 0;
  GlobalIndexTypeToCast* dofEntriesToExport = NULL;  // we are responsible for deleting the allocated arrays
  int* exportRecipients = NULL;
  
  distributor->CreateFromRecvs(myRequestCount, myRequestPtr, myRequestOwnersPtr, true, numEntriesToExport, dofEntriesToExport, exportRecipients);
  
//  const std::set<GlobalIndexType>* myCells = &_mesh->globalDofAssignment()->cellsInPartition(-1);
  
  vector<int> sizes(numEntriesToExport);
  vector<GlobalIndexTypeToCast> indicesToExport;
  for (int entryOrdinal=0; entryOrdinal<numEntriesToExport; entryOrdinal++)
  {
    GlobalIndexType interpretedDofIndex = dofEntriesToExport[entryOrdinal];
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) == _interpretedToGlobalDofIndexMap.end())
    {
      cout << "interpreted dof index " << interpretedDofIndex << " does not belong to rank " << rank << endl;
      ostringstream myRankDescriptor;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested interpretedDofIndex does not belong to this rank!");
    }
    
    GlobalIndexType condensedGlobalIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
    indicesToExport.push_back(interpretedDofIndex);
    indicesToExport.push_back(condensedGlobalIndex);
    sizes[entryOrdinal] = 2; // key, value
  }
  //  print("Export vector", indicesToExport);
  
  int objSize = sizeof(GlobalIndexTypeToCast) / sizeof(char);
  
  int importLength = 0;
  char* globalIndexData = NULL;
  int* sizePtr = NULL;
  char* indicesToExportPtr = NULL;
  if (numEntriesToExport > 0)
  {
    sizePtr = &sizes[0];
    indicesToExportPtr = (char *) &indicesToExport[0];
  }
  distributor->Do(indicesToExportPtr, objSize, sizePtr, importLength, globalIndexData);
  const char* copyFromLocation = globalIndexData;
  int numDofsImport = importLength / objSize;
  vector<GlobalIndexTypeToCast> globalIndicesMapVector(numDofsImport);
  GlobalIndexTypeToCast* copyToLocation = &globalIndicesMapVector[0];
  for (int dofOrdinal=0; dofOrdinal<numDofsImport; dofOrdinal++)
  {
    memcpy(copyToLocation, copyFromLocation, objSize);
    copyFromLocation += objSize;
    copyToLocation++; // copyToLocation has type GlobalIndexTypeToCast*, so this moves the pointer by objSize bytes
  }
  //  print("Import vector", globalIndicesMapVector);
  for (int i=0; i<globalIndicesMapVector.size()/2; i++)
  {
    GlobalIndexType meshGlobalIndex      = globalIndicesMapVector[i*2+0];
    GlobalIndexType condensedGlobalIndex = globalIndicesMapVector[i*2+1];
    _interpretedToGlobalDofIndexMap[meshGlobalIndex] = condensedGlobalIndex;
    _interpretedFluxDofIndices.insert(meshGlobalIndex);
  }
  
  if( dofEntriesToExport != 0 ) delete [] dofEntriesToExport;
  if( exportRecipients != 0 ) delete [] exportRecipients;
  if (globalIndexData != 0 ) delete [] globalIndexData;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::importInterpretedMapForOffRankCells(const std::set<GlobalIndexType> &cellIDs)
{
//  set<GlobalIndexType> dofIndicesSet;
  int rank = _mesh->Comm()->getRank();
  
  // myRequestOwners should be in nondecreasing order (it appears)
  // this is accomplished by requestMap
  map<int, vector<GlobalIndexTypeToCast>> requestMap;
  
  for (GlobalIndexType cellID : cellIDs)
  {
    int partitionForCell = _mesh->globalDofAssignment()->partitionForCellID(cellID);
    if (partitionForCell != rank)
    {
      requestMap[partitionForCell].push_back(cellID);
    }
  }

  vector<int> myRequestOwners;
  vector<GlobalIndexTypeToCast> myRequest;
  for (auto entry : requestMap)
  {
    int partition = entry.first;
    for (auto cellIDInPartition : entry.second)
    {
      myRequest.push_back(cellIDInPartition);
      myRequestOwners.push_back(partition);
    }
  }
  
  int myRequestCount = int(myRequest.size());
  
  auto epetraComm = MPIWrapper::EpetraComm(_mesh->Comm());
  Teuchos::RCP<Epetra_Distributor> distributor = MPIWrapper::getDistributor(*epetraComm);
  
  GlobalIndexTypeToCast* myRequestPtr = NULL;
  int *myRequestOwnersPtr = NULL;
  if (myRequest.size() > 0)
  {
    myRequestPtr = &myRequest[0];
    myRequestOwnersPtr = &myRequestOwners[0];
  }
  int numCellsToExport = 0;
  GlobalIndexTypeToCast* cellIDsToExport = NULL;  // we are responsible for deleting the allocated arrays
  int* exportRecipients = NULL;
  
  distributor->CreateFromRecvs(myRequestCount, myRequestPtr, myRequestOwnersPtr, true, numCellsToExport, cellIDsToExport, exportRecipients);
  
  const std::set<GlobalIndexType>* myCells = &_mesh->globalDofAssignment()->cellsInPartition(-1);
  
  vector<int> sizes(numCellsToExport);
  vector<GlobalIndexTypeToCast> indicesToExport;
  for (int cellOrdinal=0; cellOrdinal<numCellsToExport; cellOrdinal++)
  {
    GlobalIndexType cellID = cellIDsToExport[cellOrdinal];
    if (myCells->find(cellID) == myCells->end())
    {
      cout << "cellID " << cellID << " does not belong to rank " << rank << endl;
      ostringstream myRankDescriptor;
      myRankDescriptor << "rank " << rank << ", cellID ownership";
      Camellia::print(myRankDescriptor.str().c_str(), *myCells);
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested cellID does not belong to this rank!");
    }
    
    set<GlobalIndexType> meshGlobalIndices = _mesh->globalDofIndicesForCell(cellID);
    int cellIndicesSize = 0;
    for (GlobalIndexType meshGlobalIndex : meshGlobalIndices)
    {
      if (_interpretedToGlobalDofIndexMap.find(meshGlobalIndex) != _interpretedToGlobalDofIndexMap.end())
      {
        GlobalIndexType condensedGlobalIndex = _interpretedToGlobalDofIndexMap[meshGlobalIndex];
        indicesToExport.push_back(meshGlobalIndex);
        indicesToExport.push_back(condensedGlobalIndex);
        cellIndicesSize += 2;
      }
    }
    sizes[cellOrdinal] = cellIndicesSize;
  }
  
//  print("Export vector", indicesToExport);
  
  int objSize = sizeof(GlobalIndexTypeToCast) / sizeof(char);
  
  int importLength = 0;
  char* globalIndexData = NULL;
  int* sizePtr = NULL;
  char* indicesToExportPtr = NULL;
  if (numCellsToExport > 0)
  {
    sizePtr = &sizes[0];
    indicesToExportPtr = (char *) &indicesToExport[0];
  }
  distributor->Do(indicesToExportPtr, objSize, sizePtr, importLength, globalIndexData);
  const char* copyFromLocation = globalIndexData;
  int numDofsImport = importLength / objSize;
  vector<GlobalIndexTypeToCast> globalIndicesMapVector(numDofsImport);
  GlobalIndexTypeToCast* copyToLocation = &globalIndicesMapVector[0];
  for (int dofOrdinal=0; dofOrdinal<numDofsImport; dofOrdinal++)
  {
    memcpy(copyToLocation, copyFromLocation, objSize);
    copyFromLocation += objSize;
    copyToLocation++; // copyToLocation has type GlobalIndexTypeToCast*, so this moves the pointer by objSize bytes
  }
//  print("Import vector", globalIndicesMapVector);
  
  for (int i=0; i<globalIndicesMapVector.size()/2; i++)
  {
    GlobalIndexType meshGlobalIndex      = globalIndicesMapVector[i*2+0];
    GlobalIndexType condensedGlobalIndex = globalIndicesMapVector[i*2+1];
    _interpretedToGlobalDofIndexMap[meshGlobalIndex] = condensedGlobalIndex;
    _interpretedFluxDofIndices.insert(meshGlobalIndex);
  }
  
  if( cellIDsToExport != 0 ) delete [] cellIDsToExport;
  if( exportRecipients != 0 ) delete [] exportRecipients;
  if (globalIndexData != 0 ) delete [] globalIndexData;
}

template <typename Scalar>
map<GlobalIndexType, GlobalIndexType> CondensedDofInterpreter<Scalar>::interpretedFluxMapLocal(const set<GlobalIndexType> &cellsForFluxInterpretation)
{
  map<GlobalIndexType, IndexType> interpretedFluxMap; // from the interpreted dofs (the global dof indices as seen by mesh) to the partition-local condensed IDs
  const set<GlobalIndexType>* localCellIDs = &_mesh->cellIDsInPartition();
  set<GlobalIndexType> interpretedFluxDofs;
  const vector<int>* trialIDs = &_mesh->bilinearForm()->trialIDs();
  IndexType partitionLocalDofIndex = 0;
  
  /*
   4-24-16
   Today's change: we now know about BC objects, and can determine *which* global dofs the singletons
   are imposed on.  This lets us impose point constraints for e.g. pressure in Stokes while still condensing
   out all the other pressure field dofs.
   
   Going forward, we should get rid of the blunt instrument that is "varDofsAreCondensible", replacing
   it with a determination of *which* dofs are on the interior of an element.  This will allow us to do
   more standard static condensation, in which continuous fields can still be condensed out on the interior
   of an element.
   */
  
  vector<pair<GlobalIndexType,Scalar>> singletonBCsOnMeshVector;
  _mesh->boundary().singletonBCsToImpose(singletonBCsOnMeshVector, *_bc, _mesh.get());
  map<GlobalIndexType,Scalar> singletonBCsOnMesh(singletonBCsOnMeshVector.begin(),singletonBCsOnMeshVector.end());
  
  for (GlobalIndexType cellID : *localCellIDs)
  {
    bool storeFluxDofIndices = cellsForFluxInterpretation.find(cellID) != cellsForFluxInterpretation.end();
    DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
    for (int trialID : *trialIDs)
    {
      const vector<int>* sidesForTrial = &trialOrder->getSidesForVarID(trialID);
      for (int sideOrdinal : *sidesForTrial)
      {
        BasisPtr basis = trialOrder->getBasis(trialID, sideOrdinal);
        set<GlobalIndexType> interpretedDofIndices = _mesh->getGlobalDofIndices(cellID, trialID, sideOrdinal);
        bool varIsCondensible = varDofsAreUsuallyCondensible(trialID, sideOrdinal, trialOrder);
        for (GlobalIndexType interpretedDofIndex : interpretedDofIndices)
        {
          bool hasSingletonBCImposed = singletonBCsOnMesh.find(interpretedDofIndex) != singletonBCsOnMesh.end();
          if (hasSingletonBCImposed)
          {
            // find the local dof index corresponding to this
            // first, confirm that this is a field variable
            TEUCHOS_TEST_FOR_EXCEPTION(sidesForTrial->size() != 1, std::invalid_argument,"Singleton BC imposed on a trace variable; this violates assumptions in CondensedDofInterpreter");
            
            // now, get the global dof indices for the field variable:
            vector<GlobalIndexType> meshGlobalDofs = _mesh->globalDofAssignment()->globalDofIndicesForFieldVariable(cellID, trialID);
            // and the local dof indices for the same:
            vector<int> localDofs = trialOrder->getDofIndices(trialID, sideOrdinal);
            
            // for now, we are only really supporting this usage for discontinuous fields
            // (The following can throw an exception for continuous fields if the global basis is constrained.)
            TEUCHOS_TEST_FOR_EXCEPTION(localDofs.size() != meshGlobalDofs.size(), std::invalid_argument, "global dofs size does not match local dofs size");
            
            // find the local dof index corresponding to the global dof:
            int localDofIndex = -1;
            for (int i=0; i<meshGlobalDofs.size(); i++)
            {
              if (meshGlobalDofs[i] == interpretedDofIndex)
              {
                localDofIndex = localDofs[i];
                break;
              }
            }
            TEUCHOS_TEST_FOR_EXCEPTION(localDofIndex == -1, std::invalid_argument, "interpretedDofIndex not found in meshGlobalDofs");
            
//            {
//              // DEBUGGING
//              int rank = _mesh->Comm()->getRank();
//              cout << "On rank " << rank << ", CondensedDofInterpreter found singleton BC on cell " << cellID;
//              cout << ", with localDofIndex " << localDofIndex << " and interpretedDofIndex " << interpretedDofIndex << endl;
//            }
          
            _cellLocalUncondensibleDofIndices[cellID].push_back(localDofIndex);
          }
          bool dofIsCondensible = varIsCondensible && !hasSingletonBCImposed;
          bool isOwnedByThisPartition = _mesh->isLocallyOwnedGlobalDofIndex(interpretedDofIndex);
          
          if (!dofIsCondensible)
          {
            if (storeFluxDofIndices)
            {
              _interpretedFluxDofIndices.insert(interpretedDofIndex);
            }
          }
          
          if (isOwnedByThisPartition && !dofIsCondensible)
          {
            if (interpretedFluxDofs.find(interpretedDofIndex) == interpretedFluxDofs.end())
            {
              interpretedFluxMap[interpretedDofIndex] = partitionLocalDofIndex++;
              interpretedFluxDofs.insert(interpretedDofIndex);
              //              cout << interpretedDofIndex << " --> " << interpretedFluxMap[interpretedDofIndex] << endl;
            }
          }
        }
      }
    }
  }
  
  return interpretedFluxMap;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::initializeGlobalDofIndices()
{
  _interpretedFluxDofIndices.clear();
  _interpretedToGlobalDofIndexMap.clear();

  PartitionIndexType rank = _mesh->Comm()->getRank();
  set<GlobalIndexType> cellsForFluxStorage = _mesh->globalDofAssignment()->cellsInPartition(rank);
  cellsForFluxStorage.insert(_offRankCellsToInclude.begin(),_offRankCellsToInclude.end());
  map<GlobalIndexType, IndexType> partitionLocalFluxMap = interpretedFluxMapLocal(cellsForFluxStorage);

  _myGlobalDofIndexCount = partitionLocalFluxMap.size();
  int myCount = _myGlobalDofIndexCount;
  int myOffset = MPIWrapper::scanSum(_mesh->Comm(), myCount);
  _myGlobalDofIndexOffset = myOffset - myCount;
  
  int numRanks = _mesh->Comm()->getSize();
  _globalDofIndexOffsets.resize(numRanks);
  MPIWrapper::allGather(_mesh->Comm(), _globalDofIndexOffsets, _myGlobalDofIndexOffset);
  
  vector<GlobalIndexType> fluxDofCountForRank(numRanks);
//  fluxDofCountForRank(rank) = (GlobalIndexTypeToCast) _myGlobalDofIndexCount;
  MPIWrapper::allGather(_mesh->Comm(), fluxDofCountForRank, _myGlobalDofIndexCount);

  // initialize _interpretedToGlobalDofIndexMap for the guys we own
  for (map<GlobalIndexType, IndexType>::iterator entryIt = partitionLocalFluxMap.begin(); entryIt != partitionLocalFluxMap.end(); entryIt++)
  {
    _interpretedToGlobalDofIndexMap[entryIt->first] = entryIt->second + _myGlobalDofIndexOffset;
//    cout << "Rank " << rank << ": " << entryIt->first << " --> " << entryIt->second + _myGlobalDofIndexOffset << endl;
  }

  map<PartitionIndexType, set<GlobalIndexType> > partitionToMeshGlobalIndices; // key: who to ask; value: what to ask for
  // fill in the guys we don't own but do see
  for (GlobalIndexType interpretedFlux : _interpretedFluxDofIndices)
  {
    if (_interpretedToGlobalDofIndexMap.find(interpretedFlux) == _interpretedToGlobalDofIndexMap.end())
    {
      // not a local guy, then
      PartitionIndexType owningPartition = _mesh->partitionForGlobalDofIndex(interpretedFlux);
      partitionToMeshGlobalIndices[owningPartition].insert(interpretedFlux);
    }
  }
//  {
//    // DEBUGGING
//    ostringstream os;
//    os << "Rank " << rank << " _interpretedToGlobalDofIndexMap";
//    Camellia::print(os.str(), _interpretedToGlobalDofIndexMap);
//  }
  importInterpretedMapForNeighborGlobalIndices(partitionToMeshGlobalIndices);
  
  // communicate about any off-rank guys that might be of interest
  importInterpretedMapForOffRankCells(_offRankCellsToInclude);
  
//  cout << "Rank " << rank << " partitionInterpretedFluxMap.size() = " << partitionInterpretedFluxMap.size() << endl;
//  cout << "Rank " << rank << " _interpretedToGlobalDofIndexMap.size() = " << _interpretedToGlobalDofIndexMap.size() << endl;
}

template <typename Scalar>
GlobalIndexType CondensedDofInterpreter<Scalar>::globalDofCount()
{
  GlobalIndexType meshGlobalDofCount = _mesh->globalDofCount();
  if (meshGlobalDofCount != _meshLastKnownGlobalDofCount)
  {
    reinitialize();
  }
  
  return MPIWrapper::sum(_mesh->Comm(), (GlobalIndexTypeToCast)_myGlobalDofIndexCount);
}

template <typename Scalar>
set<GlobalIndexType> CondensedDofInterpreter<Scalar>::globalDofIndicesForPartition(PartitionIndexType rank)
{
  if (rank == -1)
  {
    // default to current partition, just as Mesh does.
    rank = _mesh->Comm()->getRank();
  }
  if (rank == _mesh->Comm()->getRank())
  {
    vector<GlobalIndexType> myGlobalDofIndicesVector(_myGlobalDofIndexCount);
    GlobalIndexType nextOffset = _myGlobalDofIndexOffset + _myGlobalDofIndexCount;
    int ordinal = 0;
    for (GlobalIndexType dofIndex = _myGlobalDofIndexOffset; dofIndex < nextOffset; dofIndex++)
    {
      myGlobalDofIndicesVector[ordinal] = dofIndex;
      ordinal++;
    }
    return set<GlobalIndexType>(myGlobalDofIndicesVector.begin(),myGlobalDofIndicesVector.end());
  }
  else
  {
    cout << "globalDofIndicesForPartition() requires that rank be the local MPI rank!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndicesForPartition() requires that rank be the local MPI rank!");
  }
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const Camellia::FieldContainer<Scalar> &basisCoefficients,
    Camellia::FieldContainer<Scalar> &globalCoefficients, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  int rank = _mesh->Comm()->getRank();
  // NOTE: cellID MUST belong to this partition, or have been included in "offRankCellsToInclude" constructor argument
  if ((_offRankCellsToInclude.find(cellID) == _offRankCellsToInclude.end()) && (_mesh->cellIDsInPartition().find(cellID)==_mesh->cellIDsInPartition().end()))
  {
    cout << "cellID " << cellID << " does not belong to partition " << rank;
    cout << ", and was not included in CondensedDofInterpreter constructor's offRankCellsToInclude argument.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID does not belong to partition, and isn't in offRankCellsToInclude");
  }

  Camellia::FieldContainer<Scalar> interpretedCoefficients;
  Camellia::FieldContainer<GlobalIndexType> interpretedDofIndices;
  _mesh->interpretLocalBasisCoefficients(cellID, varID, sideOrdinal, basisCoefficients,
                                         interpretedCoefficients, interpretedDofIndices);

  // all BC indices should map one to one from the mesh's "interpreted" view to our "global" view

  globalCoefficients = interpretedCoefficients;
  globalDofIndices.resize(interpretedDofIndices.size());

  for (int dofOrdinal=0; dofOrdinal<interpretedDofIndices.size(); dofOrdinal++)
  {
    GlobalIndexType interpretedDofIndex = interpretedDofIndices[dofOrdinal];
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) != _interpretedToGlobalDofIndexMap.end())
    {
      GlobalIndexType globalDofIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
      globalDofIndices[dofOrdinal] = globalDofIndex;
    }
    else if (_cellLocalUncondensibleDofIndices.find(cellID) != _cellLocalUncondensibleDofIndices.end())
    {
      // then likely there's a singleton BC imposed, meaning that we have entries in
      // _interpretedToGlobalDofIndexMap for some of the guys in interpretedDofIndices, but not all.
      // If we don't have an entry in _cellLocalUncondensibleDofIndices, then we should throw an exception.
    }
    else
    {
      cout << "On rank " << _mesh->Comm()->getRank() << ", ";
      cout << "globalDofIndex not found for specified interpretedDofIndex " << interpretedDofIndex << " (may not be a flux?)\n";
      cout << "cellID " << cellID << ", varID " << varID << ", side " << sideOrdinal << endl;
      
      cout << "Error encountered on a mesh with " << _mesh->numActiveElements() << " active elements.\n";
      
      GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(_mesh->globalDofAssignment().get());
      if (minRule != NULL)
      {
        cout << "GDAMinimumRule globalDofs for rank " << rank << ":\n";
        minRule->printGlobalDofInfo();
      }
      
//      _mesh->getTopology()->printAllEntitiesInBaseMeshTopology();
      
      
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndex not found for specified interpretedDofIndex (may not be a flux?)");
    }
  }
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localCoefficients,
                                                                 Epetra_MultiVector &globalCoefficients, int columnOrdinal)
{
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  Camellia::FieldContainer<Scalar> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (int trialID : trialOrder->getVarIDs())
  {
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    for (int sideOrdinal : *sides)
    {
      if (varDofsAreCondensible(cellID, trialID, sideOrdinal, trialOrder)) continue;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      Camellia::FieldContainer<Scalar> fittedGlobalCoefficients;
      Camellia::FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, columnOrdinal, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
        //        cout << "global coefficient " << globalDofIndex << " = " << fittedGlobalCoefficients[i] << endl;
      }
    }
  }
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localData,
    Camellia::FieldContainer<Scalar> &globalData, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end())
  {
    computeAndStoreLocalStiffnessAndLoad(cellID);
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CondensedDofInterpreter requires both stiffness and load data to be provided.");
  }
  Camellia::FieldContainer<Scalar> globalStiffnessData; // dummy container
  interpretLocalData(cellID, _localStiffnessMatrices[cellID], localData, globalStiffnessData, globalData, globalDofIndices);
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &localStiffnessData, const Camellia::FieldContainer<Scalar> &localLoadData,
    Camellia::FieldContainer<Scalar> &globalStiffnessData, Camellia::FieldContainer<Scalar> &globalLoadData,
    Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  // NOTE: cellID MUST belong to this partition, or have been included in "offRankCellsToInclude" constructor argument
  int rank = _mesh->Comm()->getRank();
  if ((_offRankCellsToInclude.find(cellID) == _offRankCellsToInclude.end()) && (_mesh->partitionForCellID(cellID) != rank))
  {
    cout << "cellID " << cellID << " does not belong to partition " << rank;
    cout << ", and was not included in CondensedDofInterpreter constructor's offRankCellsToInclude argument.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID does not belong to partition, and isn't in offRankCellsToInclude");
  }

//  if (cellID==14) {
//    cout << "cellID " << cellID << endl;
//  }

  Camellia::FieldContainer<Scalar> interpretedStiffnessData, interpretedLoadData;

  Camellia::FieldContainer<GlobalIndexType> interpretedDofIndices;

  _mesh->DofInterpreter::interpretLocalData(cellID, localStiffnessData, localLoadData,
      interpretedStiffnessData, interpretedLoadData, interpretedDofIndices);

  if (_storeLocalStiffnessMatrices)
  {
    if (_localStiffnessMatrices.find(cellID) != _localStiffnessMatrices.end())
    {
      if (&_localStiffnessMatrices[cellID] != &localStiffnessData)
      {
        _localStiffnessMatrices[cellID] = localStiffnessData;
      }
    }
    else
    {
      _localStiffnessMatrices[cellID] = localStiffnessData;
    }
    _localLoadVectors[cellID] = localLoadData;
    _localInterpretedDofIndices[cellID] = interpretedDofIndices;
  }

  set<int> fieldIndices, fluxIndices; // which are fields and which are fluxes in the interpreted data containers
//  set<GlobalIndexType> interpretedFluxIndices, interpretedFieldIndices; // debugging
  for (int dofOrdinal=0; dofOrdinal < interpretedDofIndices.size(); dofOrdinal++)
  {
    GlobalIndexType interpretedDofIndex = interpretedDofIndices(dofOrdinal);
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) == _interpretedToGlobalDofIndexMap.end())
    {
      fieldIndices.insert(dofOrdinal);
//      interpretedFieldIndices.insert(interpretedDofIndex); // debugging
    }
    else
    {
      fluxIndices.insert(dofOrdinal);
//      interpretedFluxIndices.insert(interpretedDofIndex); // debugging
    }
  }

//  { // DEBUGGING
//    cout << "CondensedDofInterpreter, field/flux division:\n";
//    ostringstream cellIDStr;
//    cellIDStr << "cell " << cellID << ", fields: ";
//    Camellia::print(cellIDStr.str(), fieldIndices);
//    Camellia::print("interpreted field indices", interpretedFieldIndices);
//
//    cellIDStr.str("");
//    cellIDStr << "cell " << cellID << ", fluxes: ";
//    Camellia::print(cellIDStr.str(), fluxIndices);
//
//    Camellia::print("interpreted flux indices", interpretedFluxIndices);
//  }

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
  if ( solver.ShouldEquilibrate() )
  {
    solver.EquilibrateMatrix();
    solver.EquilibrateRHS();
    equilibrated = true;
  }
  int err = solver.Solve();
  if (err != 0)
  {
    cout << "CondensedDofInterpreter: Epetra_SerialDenseMatrix::Solve() returned error code " << err << endl;
    cout << "matrix:\n" << D;
  }
  if (equilibrated)
    solver.UnequilibrateLHS();

  K_flux.Multiply('T','N',-1.0,B,DinvB,1.0); // assemble condensed matrix - A - B^T*inv(D)*B

  // reduce vector
  Epetra_SerialDenseVector Dinvf(fieldCount);
  Epetra_SerialDenseVector BtDinvf(fluxCount);
  Epetra_SerialDenseVector b_field, b_flux;
  getSubvectors(fieldIndices, fluxIndices, interpretedLoadData, b_field, b_flux);

  solver.SetVectors(Dinvf, b_field);
  equilibrated = false;
  //    solver.SetMatrix(D);
  if ( solver.ShouldEquilibrate() )
  {
    solver.EquilibrateMatrix();
    solver.EquilibrateRHS();
    equilibrated = true;
  }
  err = solver.Solve();
  if (err != 0)
  {
    cout << "CondensedDofInterpreter: Epetra_SerialDenseMatrix::Solve() returned error code " << err << endl;
    cout << "matrix:\n" << D;
  }

  if (equilibrated)
    solver.UnequilibrateLHS();

  b_flux.Multiply('T','N',-1.0,B,Dinvf,1.0); // condensed RHS - f - B^T*inv(D)*g

  // resize output Camellia::FieldContainers
  globalDofIndices.resize(fluxCount);
  globalStiffnessData.resize( fluxCount, fluxCount );
  globalLoadData.resize( fluxCount );

  set<int>::iterator indexIt;
  int i = 0;
  for (indexIt = fluxIndices.begin(); indexIt!=fluxIndices.end(); indexIt++)
  {
    int localFluxIndex = *indexIt;
    GlobalIndexType interpretedDofIndex = interpretedDofIndices(localFluxIndex);
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) != _interpretedToGlobalDofIndexMap.end())
    {
      int condensedIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
      globalDofIndices(i) = condensedIndex;
      i++;
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "interpretedDofIndex not found!");
    }
  }

  for (int i=0; i<fluxCount; i++)
  {
    globalLoadData(i) = b_flux(i);
    for (int j=0; j<fluxCount; j++)
    {
      globalStiffnessData(i,j) = K_flux(i,j);
    }
  }
}

// new version:
template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretGlobalCoefficients(GlobalIndexType cellID, Camellia::FieldContainer<Scalar> &localCoefficients,
                                                                  const Epetra_MultiVector &globalCoefficients)
{
  // here, globalCoefficients correspond to *flux* dofs
  
//  cout << "CondensedDofInterpreter<Scalar>::interpretGlobalCoefficients for cell " << cellID << endl;
  
  // get elem data and submatrix data
  set<int> fieldIndices, fluxIndices; // which are fields and which are fluxes in the local cell coefficients

  Epetra_SerialDenseVector b_field;
  
  Camellia::FieldContainer<Scalar> K,rhs;
  Camellia::FieldContainer<GlobalIndexType> interpretedDofIndices;
  
  Teuchos::RCP<Epetra_SerialDenseSolver> fieldSolver;
  Epetra_SerialDenseMatrix B;
  Epetra_SerialSymDenseMatrix D;
  if (! _skipLocalFields)
    getLocalData(cellID, fieldSolver, D, B, b_field, interpretedDofIndices, fieldIndices, fluxIndices);
  else
  {
    if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end())
    {
      computeAndStoreLocalStiffnessAndLoad(cellID);
    }
    interpretedDofIndices = _localInterpretedDofIndices[cellID];
  }
    
  vector<GlobalIndexTypeToCast> interpretedDofIndicesPresent(interpretedDofIndices.size());
  int numPresent = 0;
  for (int i=0; i<interpretedDofIndices.size(); i++)
  {
    GlobalIndexTypeToCast interpretedDofIndex = interpretedDofIndices[i];
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) != _interpretedToGlobalDofIndexMap.end())
    {
      GlobalIndexTypeToCast globalDofIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
      int lID_global = globalCoefficients.Map().LID(globalDofIndex);
      if (lID_global != -1)
      {
        interpretedDofIndicesPresent[numPresent++] = interpretedDofIndex;
      }
    }
  }
  // construct map for interpretedCoefficients that are represented:
  Epetra_SerialComm SerialComm; // rank-local map
  Epetra_Map    interpretedFluxIndicesMap((GlobalIndexTypeToCast)-1, numPresent, &interpretedDofIndicesPresent[0], 0, SerialComm);
  Epetra_MultiVector interpretedCoefficients(interpretedFluxIndicesMap, 1);
  
  int fieldCount = fieldIndices.size();
  int fluxCount = fluxIndices.size();
  
  for (int i=0; i<numPresent; i++)
  {
    GlobalIndexTypeToCast interpretedDofIndex = interpretedDofIndicesPresent[i];
    int lID_interpreted = interpretedFluxIndicesMap.LID(interpretedDofIndex);
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) != _interpretedToGlobalDofIndexMap.end())
    {
      GlobalIndexTypeToCast globalDofIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
      int lID_global = globalCoefficients.Map().LID(globalDofIndex);
      if (lID_global != -1)
      {
        interpretedCoefficients[0][lID_interpreted] = globalCoefficients[0][lID_global];
      }
    }
  }
  
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  
  _mesh->interpretGlobalCoefficients(cellID, localCoefficients, interpretedCoefficients); // *only* fills in fluxes in localCoefficients (fields are zeros).  We still need to back out the fields
  
  //  cout << "localCoefficients for cellID " << cellID << ":\n" << localCoefficients;
  
  if (_skipLocalFields) return; // then we are done...
  
  Epetra_SerialDenseVector field_dofs(fieldCount);
  Epetra_SerialDenseVector flux_dofs(fluxCount);
  
  int fluxOrdinal=0;
  for (set<int>::iterator fluxIt = fluxIndices.begin(); fluxIt != fluxIndices.end(); fluxIt++, fluxOrdinal++)
  {
    flux_dofs[fluxOrdinal] = localCoefficients[*fluxIt];
  }
  
  //  cout << "K:\n" << K;
  //  cout << "D:\n" << D;
//  cout << "B:\n" << B;
//  cout << "flux_dofs:\n" << flux_dofs;
//  cout << "b_field before multiplication:\n" << b_field;
  //  cout << "fluxMat:\n" << fluxMat;
  //
  
  b_field.Multiply('N','N',-1.0,B,flux_dofs,1.0);
  
  // solve for field dofs
  fieldSolver->SetVectors(field_dofs,b_field);
  bool equilibrated = false;
  if ( fieldSolver->ShouldEquilibrate() )
  {
    fieldSolver->EquilibrateMatrix();
    fieldSolver->EquilibrateRHS();
    equilibrated = true;
  }
  fieldSolver->Solve();
  if (equilibrated)
    fieldSolver->UnequilibrateLHS();
  
  int fieldOrdinal = 0; // index into field_dofs
  for (set<int>::iterator fieldIt = fieldIndices.begin(); fieldIt != fieldIndices.end(); fieldIt++, fieldOrdinal++)
  {
    localCoefficients[*fieldIt] = field_dofs[fieldOrdinal];
  }
  
//  cout << "******* b_field:\n" << b_field;
//  cout << "******* flux_dofs:\n" << flux_dofs;
//  cout << "field_dofs:\n" << field_dofs;
//  cout << "localCoefficients:\n" << localCoefficients;
}

template <typename Scalar>
bool CondensedDofInterpreter<Scalar>::isLocallyOwnedGlobalDofIndex(GlobalIndexType globalDofIndex) const
{
  return (globalDofIndex >= _myGlobalDofIndexOffset) && (globalDofIndex < _myGlobalDofIndexOffset + _myGlobalDofIndexCount);
}

template <typename Scalar>
PartitionIndexType CondensedDofInterpreter<Scalar>::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex )
{
  // find the first rank whose offset is above globalDofIndex; the one prior to that will be the owner
  auto upperBoundIt = std::upper_bound(_globalDofIndexOffsets.begin(), _globalDofIndexOffsets.end(), globalDofIndex);
  int firstRankPastGlobalDofIndex = upperBoundIt - _globalDofIndexOffsets.begin();
  return firstRankPastGlobalDofIndex - 1;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::setCanSkipLocalFieldInInterpretGlobalCoefficients(bool value)
{
  _skipLocalFields = value;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::storeLoadForCell(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &load)
{
  _localLoadVectors[cellID] = load;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::storeStiffnessForCell(GlobalIndexType cellID, const Camellia::FieldContainer<Scalar> &stiffness)
{
  _localStiffnessMatrices[cellID] = stiffness;
}

template <typename Scalar>
const Camellia::FieldContainer<Scalar> & CondensedDofInterpreter<Scalar>::storedLocalLoadForCell(GlobalIndexType cellID)
{
  if (_localLoadVectors.find(cellID) == _localLoadVectors.end())
  {
    computeAndStoreLocalStiffnessAndLoad(cellID);
  }
  
  if (_localLoadVectors.find(cellID) != _localLoadVectors.end())
  {
    return _localLoadVectors[cellID];
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no local load is stored for cell");
  }
}

template <typename Scalar>
const Camellia::FieldContainer<Scalar> & CondensedDofInterpreter<Scalar>::storedLocalStiffnessForCell(GlobalIndexType cellID)
{
  if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end())
  {
    computeAndStoreLocalStiffnessAndLoad(cellID);
  }
  
  if (_localStiffnessMatrices.find(cellID) != _localStiffnessMatrices.end())
  {
    return _localStiffnessMatrices[cellID];
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "no local stiffness matrix is stored for cell");
  }
}

namespace Camellia
{
template class CondensedDofInterpreter<double>;
}
