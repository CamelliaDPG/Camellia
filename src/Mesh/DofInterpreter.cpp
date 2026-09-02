//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
//
//  DofInterpreter.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/23/14.
//
//

#include "DofInterpreter.h"

#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"
#include "Mesh.h"
#include "MPIWrapper.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;
using namespace std;

set<GlobalIndexType> DofInterpreter::getGlobalDofIndices(GlobalIndexType cellID, int varID, int sideOrdinal)
{
  CellTopoPtr topo = _mesh->getElementType(cellID)->cellTopoPtr;
  int spaceDim = topo->getDimension();
  
  map<int, VarPtr> trialVars = _mesh->varFactory()->trialVars();
  
  VarPtr var = trialVars[varID];
  
  int subcellDim, subcellOrdinal;
  set<GlobalIndexType> fittableIndexSet;
  if ((var->varType() == FLUX) || (var->varType() == TRACE))
  {
    subcellDim = spaceDim - 1;
    subcellOrdinal = sideOrdinal;
  }
  else
  {
    subcellDim = spaceDim;
    subcellOrdinal = 0;
  }
  
  const bool upgradeToContinuousBasis = false;
  const bool tensorTopologyMeansSpaceTime = false; // ignored when upgradeToContinuousBasis is false
  
  return this->globalDofIndicesForVarOnSubcell(varID, cellID, subcellDim, subcellOrdinal, upgradeToContinuousBasis, tensorTopologyMeansSpaceTime);
}

void DofInterpreter::interpretLocalCoefficients(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localCoefficients,
                                                map<GlobalIndexType,double> &fittedGlobalCoefficients, const set<int> &trialIDsToExclude)
{
  DofOrderingPtr trialOrder = _mesh->getElementType(cellID)->trialOrderPtr;
  Camellia::FieldContainer<double> basisCoefficients; // declared here so that we can sometimes avoid mallocs, if we get lucky in terms of the resize()
  
  fittedGlobalCoefficients.clear();
  
  set<int> trialIDs = trialOrder->getVarIDs();
  for (int trialID : trialIDs)
  {
    const vector<int>* sides = &trialOrder->getSidesForVarID(trialID);
    if (trialIDsToExclude.find(trialID) != trialIDsToExclude.end()) continue; // skip field
    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++)
    {
      int sideOrdinal = *sideIt;
      int basisCardinality = trialOrder->getBasisCardinality(trialID, sideOrdinal);
      
      // check for non-zeros
      const vector<int>* localDofIndices = &trialOrder->getDofIndices(trialID, sideOrdinal);
      bool nonZeroFound = false;
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = (*localDofIndices)[basisOrdinal];
        if (localCoefficients[localDofIndex] != 0.0)
        {
          nonZeroFound = true;
          break;
        }
      }
      if (!nonZeroFound)
      {
        // then all zeros should be mapped; only question is which global dof indices get zeros
        std::set<GlobalIndexType> globalDofIndices = this->getGlobalDofIndices(cellID, trialID, sideOrdinal);
        for (GlobalIndexType globalDofIndex : globalDofIndices)
        {
          fittedGlobalCoefficients[globalDofIndex] = 0.0;
        }
        continue;
      }
      
      basisCoefficients.resize(basisCardinality);
      
      for (int basisOrdinal=0; basisOrdinal<basisCardinality; basisOrdinal++)
      {
        int localDofIndex = (*localDofIndices)[basisOrdinal];
        basisCoefficients[basisOrdinal] = localCoefficients[localDofIndex];
      }
      Camellia::FieldContainer<double> fittedGlobalCoefficientsFC;
      Camellia::FieldContainer<GlobalIndexType> fittedGlobalDofIndices;
      interpretLocalBasisCoefficients(cellID, trialID, sideOrdinal, basisCoefficients, fittedGlobalCoefficientsFC,
                                      fittedGlobalDofIndices);
      for (int i=0; i<fittedGlobalCoefficientsFC.size(); i++)
      {
        GlobalIndexType globalDofIndex = fittedGlobalDofIndices[i];
        fittedGlobalCoefficients[globalDofIndex] = fittedGlobalCoefficientsFC[i];
      }
    }
  }
}

void DofInterpreter::interpretLocalData(GlobalIndexType cellID, const Camellia::FieldContainer<double> &localStiffnessData, const Camellia::FieldContainer<double> &localLoadData,
                                        Camellia::FieldContainer<double> &globalStiffnessData, Camellia::FieldContainer<double> &globalLoadData, Camellia::FieldContainer<GlobalIndexType> &globalDofIndices)
{
  this->interpretLocalData(cellID,localStiffnessData,globalStiffnessData,globalDofIndices);
  Camellia::FieldContainer<GlobalIndexType> globalDofIndicesForStiffness = globalDofIndices; // copy (for debugging/inspection purposes)
  this->interpretLocalData(cellID,localLoadData,globalLoadData,globalDofIndices);
  for (int i=0; i<globalDofIndicesForStiffness.size(); i++)
  {
    if (globalDofIndicesForStiffness[i] != globalDofIndices[i])
    {
      cout << "ERROR: the vector and matrix dof indices differ...\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: the vector and matrix dof indices differ...\n");
    }
  }
}

std::set<GlobalIndexType> DofInterpreter::importGlobalIndicesForCells(const std::vector<GlobalIndexType> &cellIDs)
{
  set<GlobalIndexType> dofIndicesSet;
  int rank = _mesh->Comm()->getRank();

  // myRequestOwners should be in nondecreasing order (it appears)
  // this is accomplished by requestMap
  map<int, vector<GlobalIndexTypeToCast>> requestMap;
  
  for (GlobalIndexType cellID : cellIDs)
  {
    int partitionForCell = _mesh->globalDofAssignment()->partitionForCellID(cellID);
    if (partitionForCell == rank)
    {
      set<GlobalIndexType> dofIndicesForCell = this->globalDofIndicesForCell(cellID);
      dofIndicesSet.insert(dofIndicesForCell.begin(),dofIndicesForCell.end());
    }
    else
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

  size_t myRequestCount = myRequest.size();
  size_t globalRequestCount = MPIWrapper::sum(_mesh->Comm(), myRequestCount);
  
  if (globalRequestCount > 0)
  {
    Teuchos::RCP<Tpetra::Distributor> distributor = MPIWrapper::getDistributor(_mesh->Comm());
    
    Teuchos::ArrayView<const GlobalIndexTypeToCast> remoteCellIDs = Teuchos::arrayViewFromVector(myRequest);
    Teuchos::ArrayView<const int> remoteProcIDs = Teuchos::arrayViewFromVector(myRequestOwners);
    
    Teuchos::Array<GlobalIndexTypeToCast> cellIDsToExport;
    Teuchos::Array<int> exportRecipients;
    
    distributor->createFromRecvs(remoteCellIDs, remoteProcIDs, cellIDsToExport, exportRecipients);
    
    size_t numCellsToExport = cellIDsToExport.size();
    
    const std::set<GlobalIndexType> & myCells = _mesh->globalDofAssignment()->cellsInPartition(-1);

    vector<size_t> sizes(numCellsToExport);
    vector<GlobalIndexTypeToCast> indicesToExport;
    for (int cellOrdinal=0; cellOrdinal<numCellsToExport; cellOrdinal++)
    {
      GlobalIndexType cellID = cellIDsToExport[cellOrdinal];
      if (myCells.find(cellID) == myCells.end())
      {
        cout << "cellID " << cellID << " does not belong to rank " << rank << endl;
        ostringstream myRankDescriptor;
        myRankDescriptor << "rank " << rank << ", cellID ownership";
        Camellia::print(myRankDescriptor.str().c_str(), myCells);
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "requested cellID does not belong to this rank!");
      }

      set<GlobalIndexType> indicesForCell = this->globalDofIndicesForCell(cellID);
      indicesToExport.insert(indicesToExport.end(), indicesForCell.begin(), indicesForCell.end());
      sizes[cellOrdinal] = indicesForCell.size();
    }

    HostViewManaged<size_t> exportSizes("export dof indices sizes", sizes.size());
    UnmanagedHostView<size_t> exportSizesUnmanaged(sizes.data(), sizes.size());
    Kokkos::deep_copy(exportSizes, exportSizesUnmanaged);
    
    auto numImportCells = distributor->getTotalReceiveLength();
    HostViewManaged<size_t> importSizes("import dof indices sizes", numImportCells);
    
    distributor->doPostsAndWaits(exportSizes, 1, importSizes);
    
    size_t totalSizeImport = 0;
    for (int i=0; i<numImportCells; i++)
    {
      totalSizeImport += importSizes(i);
    }
    
    UnmanagedHostView<GlobalIndexTypeToCast> exportsUnmanaged(indicesToExport.data(), indicesToExport.size());
    HostViewManaged<GlobalIndexTypeToCast> exports("exported dof indices data", indicesToExport.size());
    Kokkos::deep_copy(exports, exportsUnmanaged);
    Teuchos::ArrayView<const size_t> exportSizesAV = Teuchos::arrayViewFromVector(sizes);
    
    HostViewManaged<GlobalIndexTypeToCast> imports("imported subcell dof indices data", totalSizeImport);
    auto importSizesAV = Teuchos::arrayView(importSizes.data(), importSizes.size());
    
    distributor->doPostsAndWaits(exports, exportSizesAV, imports, importSizesAV);
    
    Teuchos::ArrayView<GlobalIndexTypeToCast> importsAV(imports.data(), imports.size());
    
    dofIndicesSet.insert(importsAV.begin(),importsAV.end());
  }
  
  return dofIndicesSet;
}

map<GlobalIndexType,set<GlobalIndexType>> DofInterpreter::importGlobalIndicesMap(const set<GlobalIndexType> &cellIDs)
{
  /*
   First implementation of this calls importGlobalIndicesForCells repeatedly (with max. one cell requested per rank).
   
   This is not optimal, but it's relatively easy to get right, and in initial use this method will not likely
   be a bottleneck.  (The better way would be to do it all in one go, and have a data format that indicates which
   global indices belong to which cells.  But that would be more effort to implement.)
   
   */
  
  
  auto comm = _mesh->Comm();
  map<GlobalIndexType,set<GlobalIndexType>> globalIndicesMap;
  
  int myCount, maxGlobalCount; // always 0 or 1: how many cells we're asking for, whether anyone is still asking for a cell
  
  set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin();
  
  do {
    GlobalIndexType cellID;
    vector<GlobalIndexType> myRequest;
    if (cellIDIt != cellIDs.end())
    {
      myCount = 1;
      cellID = *cellIDIt++;
      myRequest = {cellID};
    }
    else
    {
      myCount = 0;
      myRequest = {};
    }
    Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &myCount, &maxGlobalCount);
    
    if (maxGlobalCount > 0)
    {
      set<GlobalIndexType> globalIndicesForMyCell = this->importGlobalIndicesForCells(myRequest);
//      { // DEBUGGING
//        int rank = Teuchos::GlobalMPISession::getRank();
//        ostringstream label;
//        label << "rank " << rank << ", cell " << cellID << " globalDofIndices received";
//        print(label.str(), globalIndicesForMyCell);
//      }
      if (myCount > 0) globalIndicesMap[cellID] = globalIndicesForMyCell;
    }
  } while (maxGlobalCount > 0);
  
  return globalIndicesMap;
}
