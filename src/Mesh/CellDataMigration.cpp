//
//  CellDataMigration.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 7/15/14.
//
//

#include "CellDataMigration.h"

#include "GlobalDofAssignment.h"

#include "Solution.h"

int CellDataMigration::dataSize(Mesh *mesh, GlobalIndexType cellID) {
//  cout << "CellDataMigration::dataSize() called for cell " << cellID << endl;
  
  int size = 0;
  
  ElementTypePtr elemType = mesh->getElementType(cellID);
  
  vector<Solution *> solutions = mesh->globalDofAssignment()->getRegisteredSolutions();
  // store # of solution objects
  int numSolutions = solutions.size();
  size += sizeof(numSolutions);
  for (int i=0; i<numSolutions; i++) {
    // # dofs per solution
    int localDofs = elemType->trialOrderPtr->totalDofs();
    size += sizeof(localDofs);
    // the dofs themselves
    size += localDofs * sizeof(double);
  }
  
  return size;
}

void CellDataMigration::packData(Mesh *mesh, GlobalIndexType cellID, char *dataBuffer, int size) {
  // ideally, we'd pack the global coefficients for this cell and simply remap them when unpacking
  // however, producing the map is an implementation challenge, particularly in the presence of refined elements
  // so what we do instead is map local data, and then use the local to global mapper that we build anyway to map
  // to global values when unpacking.
  int myRank                    = Teuchos::GlobalMPISession::getRank();

//  cout << "CellDataMigration::packData() called for cell " << cellID << " on rank " << myRank << endl;
  char* dataLocation = dataBuffer;
  if (size<dataSize(mesh, cellID)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "undersized dataBuffer");
  }
//  cout << "packed data for cell " << cellID << ": ";
//  ElementTypePtr elemType = mesh->getElementType(cellID);
  vector<Solution *> solutions = mesh->globalDofAssignment()->getRegisteredSolutions();
  int numSolutions = solutions.size();
  memcpy(dataLocation, &numSolutions, sizeof(numSolutions));
  dataLocation += sizeof(numSolutions);
//  cout << numSolutions << " ";
  for (int i=0; i<numSolutions; i++) {
    if (! solutions[i]->cellHasCoefficientsAssigned(cellID)) {
      int localDofs = 0;
      memcpy(dataLocation, &localDofs, sizeof(localDofs));
      dataLocation += sizeof(localDofs);
      continue; // no dofs to assign; proceed to next solution
    }
    // # dofs per solution
    const FieldContainer<double>* solnCoeffs = &solutions[i]->allCoefficientsForCellID(cellID, false); // false: don't warn
    int localDofs = solnCoeffs->size();
//    int localDofs = elemType->trialOrderPtr->totalDofs();
    memcpy(dataLocation, &localDofs, sizeof(localDofs));
//    cout << localDofs << " ";
    dataLocation += sizeof(localDofs);

    memcpy(dataLocation, &(*solnCoeffs)[0], localDofs * sizeof(double));
    // the dofs themselves
    dataLocation += localDofs * sizeof(double);
//    for (int j=0; j<solnCoeffs->size(); j++) {
//      cout << (*solnCoeffs)[j] << " ";
//    }
//    cout << ";";
  }
//  cout << endl;
}

void CellDataMigration::unpackData(Mesh *mesh, GlobalIndexType cellID, const char *dataBuffer, int size) {
  int myRank                    = Teuchos::GlobalMPISession::getRank();
  
//  cout << "CellDataMigration::unpackData() called for cell " << cellID << " on rank " << myRank << endl;
  const char* dataLocation = dataBuffer;
  if (size<dataSize(mesh, cellID)) {
    cout << "undersized dataBuffer\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "undersized dataBuffer");
  }

  set<GlobalIndexType> rankLocalCellIDs = mesh->cellIDsInPartition();
  if (rankLocalCellIDs.find(cellID) == rankLocalCellIDs.end()) {
    // it may be that when we do ghost cells, this shouldn't be an exception--or maybe the ghost cells will be packed in with the active cell
    cout << "unpackData called for a non-rank-local cellID\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unpackData called for a non-rank-local cellID");
  }
  ElementTypePtr elemType = mesh->getElementType(cellID);
  vector<Solution *> solutions = mesh->globalDofAssignment()->getRegisteredSolutions();
  int numSolutions = solutions.size();
  int numSolutionsPacked;
  memcpy(&numSolutionsPacked, dataLocation, sizeof(numSolutionsPacked));
  if (numSolutions != numSolutionsPacked) {
    cout << "numSolutions != numSolutionsPacked.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numSolutions != numSolutionsPacked");
  }
  dataLocation += sizeof(numSolutions);
  for (int i=0; i<numSolutions; i++) {
    // # dofs per solution
    int localDofs;
    memcpy(&localDofs, dataLocation, sizeof(localDofs));
    dataLocation += sizeof(localDofs);
    if (localDofs==0) {
      // no dofs assigned -- proceed to next solution
      continue;
    }
    
    if (localDofs != elemType->trialOrderPtr->totalDofs()) {
      cout << "localDofs != elemType->trialOrderPtr->totalDofs().\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localDofs != elemType->trialOrderPtr->totalDofs()");
    }
    
    FieldContainer<double> solnCoeffs(localDofs);
    memcpy(&solnCoeffs[0], dataLocation, localDofs * sizeof(double));
    // the dofs themselves
    dataLocation += localDofs * sizeof(double);
    solutions[i]->setSolnCoeffsForCellID(solnCoeffs, cellID);
//    cout << "setting solution coefficients for cellID " << cellID << endl << solnCoeffs;
  }
}