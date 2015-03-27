// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are 
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of 
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of 
// conditions and the following disclaimer in the documentation and/or other materials 
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from 
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY 
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR 
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF 
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING 
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

/*
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
*/
#include "RieszRep.h"
#include "Epetra_Vector.h"
#include "Epetra_Import.h"

#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"

#include "SerialDenseWrapper.h"
#include "CamelliaDebugUtility.h"
#include "GlobalDofAssignment.h"

#include "MPIWrapper.h"

#include "../../drivers/DPGTests/TestSuite.h"

LinearTermPtr RieszRep::getFunctional(){
  return _functional;
}

MeshPtr RieszRep::mesh() {
  return _mesh;
}

map<GlobalIndexType,FieldContainer<double> > RieszRep::integrateFunctional() {
  // NVR: changed this to only return integrated values for rank-local cells.

  map<GlobalIndexType,FieldContainer<double> > cellRHS;
  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt=cellIDs.begin(); cellIDIt !=cellIDs.end(); cellIDIt++){
    GlobalIndexType cellID = *cellIDIt;
    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    int cubEnrich = 0; // set to zero for release
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubEnrich);

    FieldContainer<double> rhsValues(1,numTestDofs);
    _functional->integrate(rhsValues, testOrderingPtr, basisCache);

    FieldContainer<double> rhsVals(numTestDofs);
    for (int i = 0;i<numTestDofs;i++){
      rhsVals(i) = rhsValues(0,i);
    }
    cellRHS[cellID] = rhsVals;
  }
  return cellRHS;
}

void RieszRep::computeRieszRep(int cubatureEnrichment){
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif  

  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt=cellIDs.begin(); cellIDIt !=cellIDs.end(); cellIDIt++){
    GlobalIndexType cellID = *cellIDIt;

    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubatureEnrichment);

    FieldContainer<double> rhsValues(1,numTestDofs);
    _functional->integrate(rhsValues, testOrderingPtr, basisCache);
    if (_printAll){
      cout << "RieszRep: LinearTerm values for cell " << cellID << ":\n " << rhsValues << endl;
    }
  
    FieldContainer<double> ipMatrix(1,numTestDofs,numTestDofs);      
    _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, basisCache);

    bool printOutRiesz = false;
    if (printOutRiesz){
      cout << " ============================ In RIESZ ==========================" << endl;
      cout << "matrix: \n" << ipMatrix;
    }
    
    FieldContainer<double> rieszRepDofs(numTestDofs,1);
    ipMatrix.resize(numTestDofs,numTestDofs);
    rhsValues.resize(numTestDofs,1);
    int success = SerialDenseWrapper::solveSystemUsingQR(rieszRepDofs, ipMatrix, rhsValues);
    
    if (success != 0) {
      cout << "RieszRep::computeRieszRep: Solve FAILED with error: " << success << endl;
    }

//    rieszRepDofs.Multiply(true,rhsVectorCopy, normSq); // equivalent to e^T * R_V * e
    double normSquared = SerialDenseWrapper::dot(rieszRepDofs, rhsValues);
    _rieszRepNormSquared[cellID] = normSquared;

//    cout << "normSquared for cell " << cellID << ": " << _rieszRepNormSquared[cellID] << endl;
    
    if (printOutRiesz){
      cout << "rhs: \n" << rhsValues;
      cout << "dofs: \n" << rieszRepDofs;
      cout << " ================================================================" << endl;
    }

    FieldContainer<double> dofs(numTestDofs);
    for (int i = 0;i<numTestDofs;i++){
      dofs(i) = rieszRepDofs(i,0);
    }
    _rieszRepDofs[cellID] = dofs;
  }
  distributeDofs();
  _repsNotComputed = false;
}

double RieszRep::getNorm(){

  if (_repsNotComputed){    
    cout << "Computing riesz rep dofs" << endl;
    computeRieszRep();
  }

  vector< ElementPtr > allElems = _mesh->activeElements(); 
  vector< ElementPtr >::iterator elemIt;     
  double normSum = 0.0;
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){

    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    normSum+= _rieszRepNormSquaredGlobal[cellID];
  }
  return sqrt(normSum);
}

const map<GlobalIndexType,double> & RieszRep::getNormsSquared() {
  return _rieszRepNormSquared;
}

const map<GlobalIndexType,double> & RieszRep::getNormsSquaredGlobal() { // should be renamed getNormsSquaredGlobal()
  return _rieszRepNormSquaredGlobal;
}

void RieszRep::distributeDofs(){
  int myRank = Teuchos::GlobalMPISession::getRank();
  int numRanks = Teuchos::GlobalMPISession::getNProc();
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif  

  // the code below could stand to be reworked; I'm pretty sure this is not the best way to distribute the data, and it would also be best to get rid of the iteration over the global set of active elements.  But a similar point could be made about this method as a whole: do we really need to distribute all the dofs to every rank?  It may be best to eliminate this method altogether.
  
  vector<GlobalIndexType> cellIDsByPartitionOrdering;
  for (int rank=0; rank<numRanks; rank++) {
    set<GlobalIndexType> cellIDsForRank = _mesh->globalDofAssignment()->cellsInPartition(rank);
    cellIDsByPartitionOrdering.insert(cellIDsByPartitionOrdering.end(), cellIDsForRank.begin(), cellIDsForRank.end());
  }
  // determine inverse map:
  map<GlobalIndexType,int> ordinalForCellID;
  for (int ordinal=0; ordinal<cellIDsByPartitionOrdering.size(); ordinal++) {
    GlobalIndexType cellID = cellIDsByPartitionOrdering[ordinal];
    ordinalForCellID[cellID] = ordinal;
//    cout << "ordinalForCellID[" << cellID << "] = " << ordinal << endl;
  }
  
  for (int cellOrdinal=0; cellOrdinal<cellIDsByPartitionOrdering.size(); cellOrdinal++) {
    GlobalIndexType cellID = cellIDsByPartitionOrdering[cellOrdinal];
    ElementTypePtr elemTypePtr = _mesh->getElementType(cellID);
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numDofs = testOrderingPtr->totalDofs();
    
    int cellIDPartition = _mesh->partitionForCellID(cellID);
    bool isInPartition = (cellIDPartition == myRank);

    int numMyDofs;
    FieldContainer<double> dofs(numDofs);
    if (isInPartition){  // if in partition
      numMyDofs = numDofs;
      dofs = _rieszRepDofs[cellID];
    } else{
      numMyDofs = 0;
    }
    
    Epetra_Map dofMap(numDofs,numMyDofs,0,Comm); 
    Epetra_Vector distributedRieszDofs(dofMap);
    if (isInPartition) {
      for (int i = 0;i<numMyDofs;i++) { // shouldn't activate on off-proc partitions
        distributedRieszDofs.ReplaceGlobalValues(1,&dofs(i),&i);
      }     
    }
    Epetra_Map importMap(numDofs,numDofs,0,Comm); // every proc should own their own copy of the dofs
    Epetra_Import testDofImporter(importMap, dofMap); 
    Epetra_Vector globalRieszDofs(importMap);
    globalRieszDofs.Import(distributedRieszDofs, testDofImporter, Insert);  
    if (!isInPartition){
      for (int i = 0;i<numDofs;i++){
        dofs(i) = globalRieszDofs[i];
      }      
    }
    _rieszRepDofsGlobal[cellID] = dofs;
//    { // debugging
//      ostringstream cellIDlabel;
//      cellIDlabel << "cell " << cellID << " _rieszRepDofsGlobal, after global import";
//      TestSuite::serializeOutput(cellIDlabel.str(), _rieszRepDofsGlobal[cellID]);
//    }
  }
  
  // distribute norms as well
  GlobalIndexType numElems = _mesh->numActiveElements();
  set<GlobalIndexType> rankLocalCellIDs = _mesh->cellIDsInPartition();
  IndexType numMyElems = rankLocalCellIDs.size();
  GlobalIndexType myElems[numMyElems];
  // build cell index
  GlobalIndexType myCellOrdinal = 0;

  double rankLocalRieszNorms[numMyElems];
  
  for (set<GlobalIndexType>::iterator cellIDIt = rankLocalCellIDs.begin(); cellIDIt != rankLocalCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    myElems[myCellOrdinal] = ordinalForCellID[cellID];
    rankLocalRieszNorms[myCellOrdinal] = _rieszRepNormSquared[cellID];
    myCellOrdinal++;
  }
  Epetra_Map normMap((GlobalIndexTypeToCast)numElems,(int)numMyElems,(GlobalIndexTypeToCast *)myElems,(GlobalIndexTypeToCast)0,Comm);

  Epetra_Vector distributedRieszNorms(normMap);
  int err = distributedRieszNorms.ReplaceGlobalValues(numMyElems,rankLocalRieszNorms,(GlobalIndexTypeToCast *)myElems);
  if (err != 0) {
    cout << "RieszRep::distributeDofs(): on rank" << myRank << ", ReplaceGlobalValues returned error code " << err << endl;
  }

  Epetra_Map normImportMap((GlobalIndexTypeToCast)numElems,(GlobalIndexTypeToCast)numElems,0,Comm);
  Epetra_Import normImporter(normImportMap,normMap); 
  Epetra_Vector globalNorms(normImportMap);
  globalNorms.Import(distributedRieszNorms, normImporter, Add);  // add should be OK (everything should be zeros)

  for (int cellOrdinal=0; cellOrdinal<cellIDsByPartitionOrdering.size(); cellOrdinal++) {
    GlobalIndexType cellID = cellIDsByPartitionOrdering[cellOrdinal];
    _rieszRepNormSquaredGlobal[cellID] = globalNorms[cellOrdinal];
//    if (myRank==0) cout << "_rieszRepNormSquaredGlobal[" << cellID << "] = " << globalNorms[cellOrdinal] << endl;
  }
 
}

// computes riesz representation over a single element - map is from int (testID) to FieldContainer of values (sized cellIndex, numPoints)
void RieszRep::computeRepresentationValues(FieldContainer<double> &values, int testID, Camellia::EOperator op, BasisCachePtr basisCache){

  if (_repsNotComputed){
    cout << "Computing riesz rep dofs" << endl;
    computeRieszRep();
  }

  int spaceDim = _mesh->getTopology()->getSpaceDim();
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();

  // all elems coming in should be of same type
  ElementPtr elem = _mesh->getElement(cellIDs[0]);
  ElementTypePtr elemTypePtr = elem->elementType();   
  DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
  int numTestDofsForVarID = testOrderingPtr->getBasisCardinality(testID, 0);
  BasisPtr testBasis = testOrderingPtr->getBasis(testID);
  
  bool testBasisIsVolumeBasis = (spaceDim == testBasis->domainTopology()->getDimension());  
  bool useCubPointsSideRefCell = testBasisIsVolumeBasis && basisCache->isSideCache();
  
  Teuchos::RCP< const FieldContainer<double> > transformedBasisValues = basisCache->getTransformedValues(testBasis,op,useCubPointsSideRefCell);
  
  int rank = values.rank() - 2; // if values are shaped as (C,P), scalar...
  if (rank > 1) {
    cout << "ranks greater than 1 not presently supported...\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ranks greater than 1 not presently supported...");
  }
  
//  Camellia::print("cellIDs",cellIDs);
  
  values.initialize(0.0);
  for (int cellIndex = 0;cellIndex<numCells;cellIndex++){
    int cellID = cellIDs[cellIndex];
    for (int j = 0;j<numTestDofsForVarID;j++) {
      int dofIndex = testOrderingPtr->getDofIndex(testID, j);
      for (int i = 0;i<numPoints;i++) {
        if (rank==0) {
          double basisValue = (*transformedBasisValues)(cellIndex,j,i);
          values(cellIndex,i) += basisValue*_rieszRepDofsGlobal[cellID](dofIndex);
        } else {
          for (int d = 0; d<spaceDim; d++) {
            double basisValue = (*transformedBasisValues)(cellIndex,j,i,d);
            values(cellIndex,i,d) += basisValue*_rieszRepDofsGlobal[cellID](dofIndex);
          }
        }
      }
    }
  }
//  TestSuite::serializeOutput("rep values", values);
}

map<GlobalIndexType,double> RieszRep::computeAlternativeNormSqOnCells(IPPtr ip, vector<GlobalIndexType> cellIDs){
  map<GlobalIndexType,double> altNorms;
  int numCells = cellIDs.size();
  for (int i = 0;i<numCells;i++){
    altNorms[cellIDs[i]] = computeAlternativeNormSqOnCell(ip, _mesh->getElement(cellIDs[i]));
  }
  return altNorms;
  /*
  // distribute norms as well
  int numElems = _mesh->activeElements().size();    
  int numMyElems = _mesh->elementsInPartition(rank).size();
  int myElems[numMyElems];
  // build cell index
  int cellIndex = 0;
  int myCellIndex = 0;

  vector<ElementPtr> elemsInPartition = _mesh->elementsInPartition(rank);
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    if (rank==_mesh->partitionForCellID(cellID)){ // if cell is in partition
      myElems[myCellIndex] = cellIndex;
      myCellIndex++;
    }
    cellIndex++;
  }
  Epetra_Map normMap(numElems,numMyElems,myElems,0,Comm);

  Epetra_Vector distributedRieszNorms(normMap);
  cellIndex = 0;
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    if (rank==_mesh->partitionForCellID(cellID)){ // if cell is in partition
      int ind = cellIndex;
      int err = distributedRieszNorms.ReplaceGlobalValues(1,&_rieszRepNormSquared[cellID],&ind);      
    }
    cellIndex++;
  }

  Epetra_Map normImportMap(numElems,numElems,0,Comm);
  Epetra_Import normImporter(normImportMap,normMap); 
  Epetra_Vector globalNorms(normImportMap);
  globalNorms.Import(distributedRieszNorms, normImporter, Add);  // add should be OK (everything should be zeros)

  cellIndex = 0;
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    int cellID = (*elemIt)->cellID();
    _rieszRepNormSquaredGlobal[cellID] = globalNorms[cellIndex];          
    cellIndex++;
  }
  */
}

double RieszRep::computeAlternativeNormSqOnCell(IPPtr ip, ElementPtr elem){
  GlobalIndexType cellID = elem->cellID();
  Teuchos::RCP<DofOrdering> testOrdering= elem->elementType()->testOrderPtr;
  bool testVsTest = true;
  Teuchos::RCP<BasisCache> basisCache =   BasisCache::basisCacheForCell(_mesh, cellID, testVsTest,1);

  int numDofs = testOrdering->totalDofs();
  FieldContainer<double> ipMat(1,numDofs,numDofs);
  ip->computeInnerProductMatrix(ipMat,testOrdering,basisCache);

  double sum = 0.0;
  for (int i = 0;i<numDofs;i++){
    for (int j = 0;j<numDofs;j++){
      sum += _rieszRepDofsGlobal[cellID](i)*_rieszRepDofsGlobal[cellID](j)*ipMat(0,i,j);
    }
  }
  
  return sum;
}

FunctionPtr RieszRep::repFunction( VarPtr var, RieszRepPtr rep ) {
  return Teuchos::rcp( new RepFunction(var, rep) );
}

RieszRepPtr RieszRep::rieszRep(MeshPtr mesh, IPPtr ip, LinearTermPtr rhs) {
  return Teuchos::rcp( new RieszRep(mesh,ip,rhs) );
}