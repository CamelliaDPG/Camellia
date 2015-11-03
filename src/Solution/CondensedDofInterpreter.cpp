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

#include "Epetra_SerialComm.h"

#include "GlobalDofAssignment.h"

#include "SerialDenseWrapper.h"

#include "CamelliaDebugUtility.h"

#include "RHS.h"

using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
CondensedDofInterpreter<Scalar>::CondensedDofInterpreter(MeshPtr mesh, TIPPtr<Scalar> ip, TRHSPtr<Scalar> rhs,
                                                         LagrangeConstraints* lagrangeConstraints,
                                                         const set<int> &fieldIDsToExclude, bool storeLocalStiffnessMatrices,
                                                         set<GlobalIndexType> offRankCellsToInclude ) : DofInterpreter(mesh)
{
  _mesh = mesh;
  _ip = ip;
  _rhs = rhs;
  _lagrangeConstraints = lagrangeConstraints;
  _storeLocalStiffnessMatrices = storeLocalStiffnessMatrices;
  _uncondensibleVarIDs.insert(fieldIDsToExclude.begin(),fieldIDsToExclude.end());
  _offRankCellsToInclude = offRankCellsToInclude;
  _skipLocalFields = false;

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
}


template <typename Scalar>
void CondensedDofInterpreter<Scalar>::computeAndStoreLocalStiffnessAndLoad(GlobalIndexType cellID)
{
//  cout << "CondensedDofInterpreter: computing stiffness and load for cell " << cellID << endl;
  int numTrialDofs = _mesh->getElementType(cellID)->trialOrderPtr->totalDofs();
  BasisCachePtr cellBasisCache = BasisCache::basisCacheForCell(_mesh, cellID);
  BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(_mesh, cellID, true);
  _localStiffnessMatrices[cellID] = FieldContainer<Scalar>(1,numTrialDofs,numTrialDofs);
  _localLoadVectors[cellID] = FieldContainer<Scalar>(1,numTrialDofs);
  _mesh->bilinearForm()->localStiffnessMatrixAndRHS(_localStiffnessMatrices[cellID], _localLoadVectors[cellID], _ip, ipBasisCache, _rhs, cellBasisCache);

  _localStiffnessMatrices[cellID].resize(numTrialDofs,numTrialDofs);
  _localLoadVectors[cellID].resize(numTrialDofs);

  FieldContainer<Scalar> interpretedStiffnessData, interpretedLoadData;

  FieldContainer<GlobalIndexType> interpretedDofIndices;

  _mesh->DofInterpreter::interpretLocalData(cellID, _localStiffnessMatrices[cellID], _localLoadVectors[cellID],
      interpretedStiffnessData, interpretedLoadData, interpretedDofIndices);

  _localInterpretedDofIndices[cellID] = interpretedDofIndices;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::getLocalData(GlobalIndexType cellID, FieldContainer<Scalar> &stiffness, FieldContainer<Scalar> &load, FieldContainer<GlobalIndexType> &interpretedDofIndices)
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
                                                   Epetra_SerialDenseMatrix &FieldField, Epetra_SerialDenseMatrix &FieldFlux, Epetra_SerialDenseVector &b_field,
                                                   FieldContainer<GlobalIndexType> &interpretedDofIndices, set<int> &fieldIndices, set<int> &fluxIndices)
{
  if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end())
  {
    computeAndStoreLocalStiffnessAndLoad(cellID);
  }
  
  // TODO: add caching of fieldSolver, B, b_field
  // (NOTE: when we do this, need to copy b_field before returning; it's modified by caller)
  
  FieldContainer<double> K = _localStiffnessMatrices[cellID];
  FieldContainer<double> rhs = _localLoadVectors[cellID];
  interpretedDofIndices = _localInterpretedDofIndices[cellID];
  
//  cout << "rhs for cell " << cellID << ":\n" << rhs;
  
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  
  set<int> trialIDs = trialOrder->getVarIDs();
  for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      vector<int> varIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      if (varDofsAreCondensible(trialID, sideOrdinal, trialOrder))
      {
        fieldIndices.insert(varIndices.begin(), varIndices.end());
      }
      else
      {
        fluxIndices.insert(varIndices.begin(),varIndices.end());
      }
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
  
  fieldSolver = Teuchos::rcp( new Epetra_SerialDenseSolver());
  fieldSolver->SetMatrix(FieldField);
  
//  cout << "FieldField:\n" << FieldField;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::getSubmatrices(set<int> fieldIndices, set<int> fluxIndices,
    const FieldContainer<Scalar> &K, Epetra_SerialDenseMatrix &K_field,
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
void CondensedDofInterpreter<Scalar>::getSubvectors(set<int> fieldIndices, set<int> fluxIndices, const FieldContainer<Scalar> &b, Epetra_SerialDenseVector &b_field, Epetra_SerialDenseVector &b_flux)
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
  // this is not a particularly efficient way of doing this, but it's not likely to add up to much total expense
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;

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
      if (varDofsAreCondensible(trialID, sideOrdinal, trialOrder))
      {
        fieldIndices.insert(varIndices.begin(), varIndices.end());
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
    }
    TEUCHOS_TEST_FOR_EXCEPTION(rowIndices.size() != varIndices.size(), std::invalid_argument, "Internal error: number of rowIndices does not match the number of varIndices");
  }
  return rowIndices;
}


template <typename Scalar>
std::vector<int> CondensedDofInterpreter<Scalar>::fluxIndexLookupLocalCell(GlobalIndexType cellID)
{
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  
  // the way we order field dof indices is according to their index order in the local uncondensed stiffness matrix
  set<int> fluxIndices; // all field indices for the cell
  set<int> trialIDs = trialOrder->getVarIDs();
  for (int trialID : trialIDs)
  {
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      vector<int> varIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      if (!varDofsAreCondensible(trialID, sideOrdinal, trialOrder))
      {
        fluxIndices.insert(varIndices.begin(), varIndices.end());
      }
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
  set<int> fieldIndices, fluxIndices; // which are fields and which are fluxes in the local cell coefficients
  
  Epetra_SerialDenseVector b_field;
  
  FieldContainer<Scalar> K,rhs;
  FieldContainer<GlobalIndexType> interpretedDofIndices;
  
  Teuchos::RCP<Epetra_SerialDenseSolver> fieldSolver;
  Epetra_SerialDenseMatrix FieldField, FieldFlux;
  
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
  
  return fluxToFieldMap;
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
set<GlobalIndexType> CondensedDofInterpreter<Scalar>::globalDofIndicesForVarOnSubcell(int varID, GlobalIndexType cellID, unsigned dim, unsigned subcellOrdinal)
{
  set<GlobalIndexType> interpretedDofIndicesForCell = _mesh->globalDofIndicesForVarOnSubcell(varID, cellID, dim, subcellOrdinal);
  set<GlobalIndexType> globalDofIndices;
  
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
      globalDofIndices.insert(globalDofIndex);
    }
  }
  
  return globalDofIndices;
}

template <typename Scalar>
bool CondensedDofInterpreter<Scalar>::varDofsAreCondensible(int varID, int sideOrdinal, DofOrderingPtr dofOrdering)
{
  // eventually it would be nice to determine which sub-basis ordinals can be condensed, but right now we only
  // condense out the truly discontinuous bases defined for variables on the element interior.

  int sideCount = dofOrdering->getSidesForVarID(varID).size();
  if (sideCount != 1) return false;

  BasisPtr basis = dofOrdering->getBasis(varID); // sideOrdinal must be 0 since sideCount == 1
  Camellia::EFunctionSpace fs = basis->functionSpace();

  bool isDiscontinuous = functionSpaceIsDiscontinuous(fs);

  return (isDiscontinuous) && (sideCount==1) && (_uncondensibleVarIDs.find(varID) == _uncondensibleVarIDs.end());
}

// ! cellsForFluxInterpretation indicates on which cells we need to be able to interpret fluxes.
template <typename Scalar>
map<GlobalIndexType, GlobalIndexType> CondensedDofInterpreter<Scalar>::interpretedFluxMapForPartition(PartitionIndexType partition,
                                                                                                      const set<GlobalIndexType> &cellsForFluxInterpretation)
{ // add the partitionDofOffset to get the globalDofIndices

  map<GlobalIndexType, IndexType> interpretedFluxMap; // from the interpreted dofs (the global dof indices as seen by mesh) to the partition-local condensed IDs

  set< GlobalIndexType > localCellIDs = _mesh->globalDofAssignment()->cellsInPartition(partition);

  set<GlobalIndexType> interpretedFluxDofs;

  set<GlobalIndexType> interpretedDofIndicesForPartition = _mesh->globalDofIndicesForPartition(partition);

  vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
  set< GlobalIndexType >::iterator cellIDIt;

  IndexType partitionLocalDofIndex = 0;

  for (GlobalIndexType cellID : localCellIDs)
  {
    bool storeFluxDofIndices = cellsForFluxInterpretation.find(cellID) != cellsForFluxInterpretation.end();

    DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;

    for (vector<int>::iterator idIt = trialIDs.begin(); idIt!=trialIDs.end(); idIt++)
    {
      int trialID = *idIt;
      const vector<int>* sidesForTrial = &trialOrder->getSidesForVarID(trialID);

      for (int sideOrdinal : *sidesForTrial)
      {
        if ( !trialOrder->hasBasisEntry(trialID, sideOrdinal) ) continue;
        BasisPtr basis = trialOrder->getBasis(trialID, sideOrdinal);

        set<GlobalIndexType> interpretedDofIndices = _mesh->getGlobalDofIndices(cellID, trialID, sideOrdinal);

        bool isCondensible = varDofsAreCondensible(trialID, sideOrdinal, trialOrder);

        for (GlobalIndexType interpretedDofIndex : interpretedDofIndices)
        {
          bool isOwnedByThisPartition = (interpretedDofIndicesForPartition.find(interpretedDofIndex) != interpretedDofIndicesForPartition.end());

          if (!isCondensible)
          {
            if (storeFluxDofIndices)
            {
              _interpretedFluxDofIndices.insert(interpretedDofIndex);
            }
          }

          if (isOwnedByThisPartition && !isCondensible)
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
//  _interpretedDofIndicesForBasis.clear();

  PartitionIndexType rank = Teuchos::GlobalMPISession::getRank();
  set<GlobalIndexType> cellsForFluxStorage = _mesh->globalDofAssignment()->cellsInPartition(rank);
  cellsForFluxStorage.insert(_offRankCellsToInclude.begin(),_offRankCellsToInclude.end());
  map<GlobalIndexType, IndexType> partitionLocalFluxMap = interpretedFluxMapForPartition(rank, cellsForFluxStorage);

  int numRanks = Teuchos::GlobalMPISession::getNProc();
  FieldContainer<GlobalIndexType> fluxDofCountForRank(numRanks);

  _myGlobalDofIndexCount = partitionLocalFluxMap.size();
  fluxDofCountForRank(rank) = _myGlobalDofIndexCount;

  MPIWrapper::entryWiseSum(fluxDofCountForRank);

  _myGlobalDofIndexOffset = 0;
  for (int i=0; i<rank; i++)
  {
    _myGlobalDofIndexOffset += fluxDofCountForRank(i);
  }

  // initialize _interpretedToGlobalDofIndexMap for the guys we own
  for (map<GlobalIndexType, IndexType>::iterator entryIt = partitionLocalFluxMap.begin(); entryIt != partitionLocalFluxMap.end(); entryIt++)
  {
    _interpretedToGlobalDofIndexMap[entryIt->first] = entryIt->second + _myGlobalDofIndexOffset;
//    cout << "Rank " << rank << ": " << entryIt->first << " --> " << entryIt->second + _myGlobalDofIndexOffset << endl;
  }

  map< PartitionIndexType, map<GlobalIndexType, GlobalIndexType> > partitionInterpretedFluxMap;

  // now, ensure that _interpretedDofIndices includes all the off-rank cells we're interested in:
  for (GlobalIndexType offRankCell : _offRankCellsToInclude)
  {
    PartitionIndexType owningPartition = _mesh->partitionForCellID(offRankCell);
    if (partitionInterpretedFluxMap.find(owningPartition) == partitionInterpretedFluxMap.end())
    {
      partitionLocalFluxMap = interpretedFluxMapForPartition(owningPartition, cellsForFluxStorage);
      GlobalIndexType owningPartitionDofOffset = 0;
      for (int i=0; i<owningPartition; i++)
      {
        owningPartitionDofOffset += fluxDofCountForRank(i);
      }
      map<GlobalIndexType, GlobalIndexType> owningPartitionInterpretedToGlobalDofIndexMap;
      for (map<GlobalIndexType, IndexType>::iterator entryIt = partitionLocalFluxMap.begin(); entryIt != partitionLocalFluxMap.end(); entryIt++)
      {
        owningPartitionInterpretedToGlobalDofIndexMap[entryIt->first] = entryIt->second + owningPartitionDofOffset;
      }
      partitionInterpretedFluxMap[owningPartition] = owningPartitionInterpretedToGlobalDofIndexMap;
    }
  }
  
  set<GlobalIndexType> noCells;
  // fill in the guys we don't own but do see
  for (set<GlobalIndexType>::iterator interpretedFluxIt=_interpretedFluxDofIndices.begin(); interpretedFluxIt != _interpretedFluxDofIndices.end(); interpretedFluxIt++)
  {
    GlobalIndexType interpretedFlux = *interpretedFluxIt;
    if (_interpretedToGlobalDofIndexMap.find(interpretedFlux) == _interpretedToGlobalDofIndexMap.end())
    {
      // not a local guy, then
      PartitionIndexType owningPartition = _mesh->partitionForGlobalDofIndex(interpretedFlux);
      if (partitionInterpretedFluxMap.find(owningPartition) == partitionInterpretedFluxMap.end())
      {
        partitionLocalFluxMap = interpretedFluxMapForPartition(owningPartition, noCells);
        GlobalIndexType owningPartitionDofOffset = 0;
        for (int i=0; i<owningPartition; i++)
        {
          owningPartitionDofOffset += fluxDofCountForRank(i);
        }
        map<GlobalIndexType, GlobalIndexType> owningPartitionInterpretedToGlobalDofIndexMap;
        for (map<GlobalIndexType, IndexType>::iterator entryIt = partitionLocalFluxMap.begin(); entryIt != partitionLocalFluxMap.end(); entryIt++)
        {
          owningPartitionInterpretedToGlobalDofIndexMap[entryIt->first] = entryIt->second + owningPartitionDofOffset;
        }
        partitionInterpretedFluxMap[owningPartition] = owningPartitionInterpretedToGlobalDofIndexMap;
      }
      _interpretedToGlobalDofIndexMap[interpretedFlux] = partitionInterpretedFluxMap[owningPartition][interpretedFlux];
//      cout << "Rank " << rank << ": " << interpretedFlux << " --> " << partitionInterpretedFluxMap[owningPartition][interpretedFlux] << endl;
    }
  }

//  cout << "Rank " << rank << " partitionInterpretedFluxMap.size() = " << partitionInterpretedFluxMap.size() << endl;
//  cout << "Rank " << rank << " _interpretedToGlobalDofIndexMap.size() = " << _interpretedToGlobalDofIndexMap.size() << endl;
}

template <typename Scalar>
GlobalIndexType CondensedDofInterpreter<Scalar>::globalDofCount()
{
  return MPIWrapper::sum(_myGlobalDofIndexCount);
}

template <typename Scalar>
set<GlobalIndexType> CondensedDofInterpreter<Scalar>::globalDofIndicesForPartition(PartitionIndexType rank)
{
  if (rank == -1)
  {
    // default to current partition, just as Mesh does.
    rank = Teuchos::GlobalMPISession::getRank();
  }
  if (rank == Teuchos::GlobalMPISession::getRank())
  {
    set<GlobalIndexType> myGlobalDofIndices;
    GlobalIndexType nextOffset = _myGlobalDofIndexOffset + _myGlobalDofIndexCount;
    for (GlobalIndexType dofIndex = _myGlobalDofIndexOffset; dofIndex < nextOffset; dofIndex++)
    {
      myGlobalDofIndices.insert(dofIndex);
    }
    return myGlobalDofIndices;
  }
  else
  {
    cout << "globalDofIndicesForPartition() requires that rank be the local MPI rank!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndicesForPartition() requires that rank be the local MPI rank!");
  }
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<Scalar> &basisCoefficients,
    FieldContainer<Scalar> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices)
{
  // NOTE: cellID MUST belong to this partition, or have been included in "offRankCellsToInclude" constructor argument
  int rank = Teuchos::GlobalMPISession::getRank();
  if ((_offRankCellsToInclude.find(cellID) == _offRankCellsToInclude.end()) && (_mesh->partitionForCellID(cellID) != rank))
  {
    cout << "cellID " << cellID << " does not belong to partition " << rank;
    cout << ", and was not included in CondensedDofInterpreter constructor's offRankCellsToInclude argument.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID does not belong to partition, and isn't in offRankCellsToInclude");
  }

  FieldContainer<Scalar> interpretedCoefficients;
  FieldContainer<GlobalIndexType> interpretedDofIndices;
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
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalDofIndex not found for specified interpretedDofIndex (may not be a flux?)");
    }
  }
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalCoefficients(GlobalIndexType cellID, const FieldContainer<Scalar> &localCoefficients, Epetra_MultiVector &globalCoefficients)
{
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  FieldContainer<Scalar> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  for (set<int>::iterator trialIDIt = trialOrder->getVarIDs().begin(); trialIDIt != trialOrder->getVarIDs().end(); trialIDIt++)
  {
    int trialID = *trialIDIt;
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      if (varDofsAreCondensible(trialID, sideOrdinal, trialOrder)) continue;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      basisCoefficients.resize(basisCardinality);
      vector<int> localDofIndices = trialOrder->getDofIndices(trialID, sideOrdinal);
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = localDofIndices[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      FieldContainer<Scalar> fittedGlobalCoefficients;
      FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficients, fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficients.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        globalCoefficients.ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndex, 0, fittedGlobalCoefficients[i]); // for globalDofIndex not owned by this rank, doesn't do anything...
        //        cout << "global coefficient " << globalDofIndex << " = " << fittedGlobalCoefficients[i] << endl;
      }
    }
  }
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalData(GlobalIndexType cellID, const FieldContainer<Scalar> &localData,
    FieldContainer<Scalar> &globalData, FieldContainer<GlobalIndexType> &globalDofIndices)
{
  if (_localStiffnessMatrices.find(cellID) == _localStiffnessMatrices.end())
  {
    computeAndStoreLocalStiffnessAndLoad(cellID);
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "CondensedDofInterpreter requires both stiffness and load data to be provided.");
  }
  FieldContainer<Scalar> globalStiffnessData; // dummy container
  interpretLocalData(cellID, _localStiffnessMatrices[cellID], localData, globalStiffnessData, globalData, globalDofIndices);
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::interpretLocalData(GlobalIndexType cellID, const FieldContainer<Scalar> &localStiffnessData, const FieldContainer<Scalar> &localLoadData,
    FieldContainer<Scalar> &globalStiffnessData, FieldContainer<Scalar> &globalLoadData,
    FieldContainer<GlobalIndexType> &globalDofIndices)
{
  // NOTE: cellID MUST belong to this partition, or have been included in "offRankCellsToInclude" constructor argument
  int rank = Teuchos::GlobalMPISession::getRank();
  if ((_offRankCellsToInclude.find(cellID) == _offRankCellsToInclude.end()) && (_mesh->partitionForCellID(cellID) != rank))
  {
    cout << "cellID " << cellID << " does not belong to partition " << rank;
    cout << ", and was not included in CondensedDofInterpreter constructor's offRankCellsToInclude argument.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellID does not belong to partition, and isn't in offRankCellsToInclude");
  }

//  if (cellID==14) {
//    cout << "cellID " << cellID << endl;
//  }

  FieldContainer<Scalar> interpretedStiffnessData, interpretedLoadData;

  FieldContainer<GlobalIndexType> interpretedDofIndices;

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

  // resize output FieldContainers
  globalDofIndices.resize(fluxCount);
  globalStiffnessData.resize( fluxCount, fluxCount );
  globalLoadData.resize( fluxCount );

  set<int>::iterator indexIt;
  int i = 0;
  for (indexIt = fluxIndices.begin(); indexIt!=fluxIndices.end(); indexIt++)
  {
    int localFluxIndex = *indexIt;
    GlobalIndexType interpretedDofIndex = interpretedDofIndices(localFluxIndex);
    int condensedIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
    globalDofIndices(i) = condensedIndex;
    i++;
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
void CondensedDofInterpreter<Scalar>::interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<Scalar> &localCoefficients,
                                                                  const Epetra_MultiVector &globalCoefficients)
{
  // here, globalCoefficients correspond to *flux* dofs
  
//  cout << "CondensedDofInterpreter<Scalar>::interpretGlobalCoefficients for cell " << cellID << endl;
  
  // get elem data and submatrix data
  set<int> fieldIndices, fluxIndices; // which are fields and which are fluxes in the local cell coefficients

  Epetra_SerialDenseVector b_field;
  
  FieldContainer<Scalar> K,rhs;
  FieldContainer<GlobalIndexType> interpretedDofIndices;
  
  Teuchos::RCP<Epetra_SerialDenseSolver> fieldSolver;
  Epetra_SerialDenseMatrix B, D;
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
    
  
//  cout << "Got local data.\n";
  
  int fieldCount = fieldIndices.size();
  int fluxCount = fluxIndices.size();

  Epetra_SerialDenseVector field_dofs(fieldCount);
  
  FieldContainer<GlobalIndexTypeToCast> interpretedDofIndicesCast(interpretedDofIndices.size());
  for (int i=0; i<interpretedDofIndices.size(); i++)
  {
    interpretedDofIndicesCast[i] = (GlobalIndexTypeToCast) interpretedDofIndices[i];
  }
  
  // construct map for interpretedCoefficients:
  Epetra_SerialComm SerialComm; // rank-local map
  Epetra_Map    interpretedFluxIndicesMap((GlobalIndexTypeToCast)-1, (GlobalIndexTypeToCast)interpretedDofIndices.size(), &interpretedDofIndicesCast[0], 0, SerialComm);
  Epetra_MultiVector interpretedCoefficients(interpretedFluxIndicesMap, 1);
  
  for (int i=0; i<interpretedDofIndices.size(); i++)
  {
    GlobalIndexTypeToCast interpretedDofIndex = interpretedDofIndicesCast[i];
    int lID_interpreted = interpretedFluxIndicesMap.LID(interpretedDofIndex);
    if (_interpretedToGlobalDofIndexMap.find(interpretedDofIndex) != _interpretedToGlobalDofIndexMap.end())
    {
      GlobalIndexTypeToCast globalDofIndex = _interpretedToGlobalDofIndexMap[interpretedDofIndex];
      int lID_global = globalCoefficients.Map().LID(globalDofIndex);
      interpretedCoefficients[0][lID_interpreted] = globalCoefficients[0][lID_global];
      //      cout << "globalCoefficient for globalDofIndex " << globalDofIndex << ": " << globalCoefficients[0][lID_global] << endl;
    }
    else
    {
      interpretedCoefficients[0][lID_interpreted] = 0; // zeros for fields, for now
    }
  }
  
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  
  _mesh->interpretGlobalCoefficients(cellID, localCoefficients, interpretedCoefficients); // *only* fills in fluxes in localCoefficients (fields are zeros).  We still need to back out the fields
  
  //  cout << "localCoefficients for cellID " << cellID << ":\n" << localCoefficients;
  
  if (_skipLocalFields) return; // then we are done...
  
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
void CondensedDofInterpreter<Scalar>::setCanSkipLocalFieldInInterpretGlobalCoefficients(bool value)
{
  _skipLocalFields = value;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::storeLoadForCell(GlobalIndexType cellID, const FieldContainer<Scalar> &load)
{
  _localLoadVectors[cellID] = load;
}

template <typename Scalar>
void CondensedDofInterpreter<Scalar>::storeStiffnessForCell(GlobalIndexType cellID, const FieldContainer<Scalar> &stiffness)
{
  _localStiffnessMatrices[cellID] = stiffness;
}

template <typename Scalar>
const FieldContainer<Scalar> & CondensedDofInterpreter<Scalar>::storedLocalLoadForCell(GlobalIndexType cellID)
{
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
const FieldContainer<Scalar> & CondensedDofInterpreter<Scalar>::storedLocalStiffnessForCell(GlobalIndexType cellID)
{
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
