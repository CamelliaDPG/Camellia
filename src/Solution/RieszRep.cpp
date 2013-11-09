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
LinearTermPtr RieszRep::getRHS(){
  return _rhs;
}

MeshPtr RieszRep::mesh() {
  return _mesh;
}

map<int,FieldContainer<double> > RieszRep::integrateRHS(){

  map<int,FieldContainer<double> > cellRHS;
  vector< ElementPtr > allElems = _mesh->activeElements(); // CHANGE TO DISTRIBUTED COMPUTATION
  vector< ElementPtr >::iterator elemIt;     
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){

    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();

    ElementTypePtr elemTypePtr = elem->elementType();   
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    int cubEnrich = 5; // set to zero for release
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubEnrich);

    FieldContainer<double> rhsValues(1,numTestDofs);
    _rhs->integrate(rhsValues, testOrderingPtr, basisCache);

    FieldContainer<double> rhsVals(numTestDofs);
    for (int i = 0;i<numTestDofs;i++){
      rhsVals(i) = rhsValues(0,i);
    }
    cellRHS[cellID] = rhsVals;
  }
  return cellRHS;
}

void RieszRep::computeRieszRep(int cubatureEnrichment){

  int numProcs=1;
  int rank=0;  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif  

  vector< ElementPtr > allElems = _mesh->elementsInPartition(rank);
  vector< ElementPtr >::iterator elemIt;     
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){

    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();

    ElementTypePtr elemTypePtr = elem->elementType();   
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();

    BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh,cellID,true,cubatureEnrichment);

    FieldContainer<double> rhsValues(1,numTestDofs);
    _rhs->integrate(rhsValues, testOrderingPtr, basisCache);
    FieldContainer<double> ipMatrix(1,numTestDofs,numTestDofs);      
    _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, basisCache);

    Epetra_SerialDenseMatrix rhsVector(numTestDofs,1);
    Epetra_SerialDenseMatrix R_V(numTestDofs,numTestDofs);
    for (int i = 0;i<numTestDofs;i++){
      rhsVector(i,0) = rhsValues(0,i);
      for (int j = 0;j<numTestDofs;j++){
        R_V(i,j) = ipMatrix(0,i,j);
      }
    }
    Epetra_SerialDenseMatrix rhsVectorCopy = rhsVector;
    rhsValues.clear();
    if (_printAll){
      cout << "rhs vector values for cell " << cellID << " are = " << rhsVector << endl;
    }
    
    Epetra_SerialDenseSolver solver;
    Epetra_SerialDenseMatrix rieszRepDofs(numTestDofs,1);
    solver.SetMatrix(R_V);
    solver.SetVectors(rieszRepDofs, rhsVector);        

    bool equilibrated = false;
    if ( solver.ShouldEquilibrate() ) {
      solver.EquilibrateMatrix();
      solver.EquilibrateRHS();
      equilibrated = true;
    }
    solver.Solve();
    if (equilibrated) 
      solver.UnequilibrateLHS();   

    Epetra_SerialDenseMatrix normSq(1,1);
    rieszRepDofs.Multiply(true,rhsVectorCopy, normSq); // equivalent to e^T * R_V * e    
    _rieszRepNormSquared[cellID] = normSq(0,0);

    bool printOutRiesz = false;
    if (printOutRiesz){
      cout << " ============================ In RIESZ ==========================" << endl;
      cout << "matrix = " << R_V << endl;
      cout << "rhs = " << rhsVectorCopy << endl;
      cout << "dofs = " << rieszRepDofs << endl;
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

const map<int,double> & RieszRep::getNormsSquared(){ // should be renamed getNormsSquaredGlobal()
  return _rieszRepNormSquaredGlobal;
}

void RieszRep::distributeDofs(){
  int numProcs=1;
  int rank=0;  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif  

  vector< ElementPtr > elems = _mesh->activeElements(); 
  vector< ElementPtr >::iterator elemIt;     
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++) {
    
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();    
    ElementTypePtr elemTypePtr = elem->elementType();   
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numDofs = testOrderingPtr->totalDofs();
    
    int cellIDPartition = _mesh->partitionForCellID(cellID);
    bool isInPartition = (cellIDPartition == rank);

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
  }
  
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
      if (err != 0) {
        cout << "RieszRep::distributeDofs(): on rank" << rank << ", ReplaceGlobalValues returned error code " << err << endl;
      }
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
 
}

// computes riesz representation over a single element - map is from int (testID) to FieldContainer of values (sized cellIndex, numPoints)
void RieszRep::computeRepresentationValues(FieldContainer<double> &values, int testID, IntrepidExtendedTypes::EOperatorExtended op, BasisCachePtr basisCache){

  if (_repsNotComputed){
    cout << "Computing riesz rep dofs" << endl;
    computeRieszRep();
  }
  
  vector< ElementPtr > allElems = _mesh->elements();

  int spaceDim = 2; // hardcoded 2D for now
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  vector<int> cellIDs = basisCache->cellIDs();

  // all elems coming in should be of same type
  ElementPtr elem = allElems[cellIDs[0]];
  ElementTypePtr elemTypePtr = elem->elementType();   
  DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
  CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
  int numTestDofsForVarID = testOrderingPtr->getBasisCardinality(testID, 0);
  BasisPtr testBasis = testOrderingPtr->getBasis(testID);
  
  bool testBasisIsVolumeBasis = true;
  if (spaceDim==2) {
    testBasisIsVolumeBasis = (testBasis->domainTopology().getBaseKey() != shards::Line<2>::key);
  } else if (spaceDim==3) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "spaceDim==3 not yet supported in testBasisIsVolumeBasis determination.");
  }
  
  bool useCubPointsSideRefCell = testBasisIsVolumeBasis && basisCache->isSideCache();
  
  Teuchos::RCP< const FieldContainer<double> > transformedBasisValues = basisCache->getTransformedValues(testBasis,op,useCubPointsSideRefCell);
  
  int rank = 0; // scalar
  if (values.rank()>2) { // if values != (C,P)
    rank = spaceDim;
  }
  values.initialize(0.0);
  for (int cellIndex = 0;cellIndex<numCells;cellIndex++){
    int cellID = cellIDs[cellIndex];
    for (int j = 0;j<numTestDofsForVarID;j++) {
      for (int i = 0;i<numPoints;i++) {
        int dofIndex = testOrderingPtr->getDofIndex(testID, j); // to index into total test dof vector	
        if (rank==0) {
          double basisValue = (*transformedBasisValues)(cellIndex,j,i);
          values(cellIndex,i) += basisValue*_rieszRepDofsGlobal[cellID](dofIndex);
        } else {
          for (int r = 0;r<rank-1;r++) {
            double basisValue = (*transformedBasisValues)(cellIndex,j,i,r);
            values(cellIndex,i,r) += basisValue*_rieszRepDofsGlobal[cellID](dofIndex);
          }
        }
      }
    }
  }
}

map<int,double> RieszRep::computeAlternativeNormSqOnCells(IPPtr ip, vector<int> cellIDs){
  map<int,double> altNorms;
  int numCells = cellIDs.size();
  for (int i = 0;i<numCells;i++){
    altNorms[cellIDs[i]] = computeAlternativeNormSqOnCell(ip, _mesh->elements()[cellIDs[i]]);
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
  int cellID = elem->cellID();
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
