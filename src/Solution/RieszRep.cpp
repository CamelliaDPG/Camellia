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
LtPtr RieszRep::getRHS(){
  return _rhs;
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
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);

    vector<int> cellIDs;
    cellIDs.push_back(cellID); // just do one cell at a time

    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, true));

    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // create side cache if ip has boundary values 

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

void RieszRep::computeRieszRep(){

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

  //  vector< ElementPtr > allElems = _mesh->activeElements(); // CHANGE TO DISTRIBUTED COMPUTATION
  vector< ElementPtr > allElems = _mesh->elementsInPartition(rank); // CHANGE TO DISTRIBUTED COMPUTATION
  vector< ElementPtr >::iterator elemIt;     
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){

    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();

    ElementTypePtr elemTypePtr = elem->elementType();   
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTestDofs = testOrderingPtr->totalDofs();
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);

    vector<int> cellIDs;
    cellIDs.push_back(cellID); // just do one cell at a time

    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, true));
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs, true); // create side cache if ip has boundary values 

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
    rhsValues.clear();
    if (_printAll){
      cout << "rhs vector values for cell " << cellID << " are = " << rhsVector << endl;
    }
    
    //    cout << "matrix = " << R_V << endl;
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
    rieszRepDofs.Multiply(true,rhsVector, normSq); // equivalent to e^T * R_V * e    
    _rieszRepNormSquared[cellID] = normSq(0,0);
  
    FieldContainer<double> dofs(numTestDofs);
    for (int i = 0;i<numTestDofs;i++){
      dofs(i) = rieszRepDofs(i,0);
      //      cout << "dofs = " << dofs(i) << endl;
    }
    _rieszRepDofs[cellID] = dofs;
  }
  distributeDofs();
}

double RieszRep::getNorm(){
  
  vector< ElementPtr > allElems = _mesh->activeElements(); 
  vector< ElementPtr >::iterator elemIt;     
  double normSum = 0.0;
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){

    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    
    normSum+= _rieszRepNormSquared[cellID];
  }
  return sqrt(normSum);
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

  // loop thru elements in partition, export data to other processors
  //  vector< ElementPtr > elems = _mesh->elementsInPartition(rank); 
  
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

  //  _rieszRepDofsGlobal = _rieszRepDofs; // to be replaced by an MPI call 
  
}

// computes riesz representation over a single element - map is from int (testID) to FieldContainer of values (sized cellIndex, numPoints)
void RieszRep::computeRepresentationValues(FieldContainer<double> &values, int testID, IntrepidExtendedTypes::EOperatorExtended op, BasisCachePtr basisCache){

  //  if (op==IntrepidExtendedTypes::OP_DX){
  //    cout << "computing rep values for op_dx" << endl;
  //  }

  vector< ElementPtr > allElems = _mesh->elements();

  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);

  values.initialize(0.0);
  vector<int> cellIDs = basisCache->cellIDs();

  for (int cellIndex = 0;cellIndex<numCells;cellIndex++){
    int cellID = cellIDs[cellIndex];
    ElementPtr elem = allElems[cellID];
    ElementTypePtr elemTypePtr = elem->elementType();   
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
    int numTestDofsForVarID = testOrderingPtr->getBasisCardinality(testID, 0);
    BasisPtr testBasis = testOrderingPtr->getBasis(testID);
    //    Teuchos::RCP< const FieldContainer<double> > basisValues = basisCache->getValues(testBasis,op);
    Teuchos::RCP< const FieldContainer<double> > transformedBasisValues = basisCache->getTransformedValues(testBasis,op);
    /*
    for (int i = 0;i<basisValues->rank();i++){
      cout << "value dimension " << i << " = " << basisValues->dimension(i) << ", ";
    }
    cout << endl;
    for (int i = 0;i<transformedBasisValues->rank();i++){
      cout << "transformed value dimension " << i << " = " << transformedBasisValues->dimension(i) << ", ";
    }
    cout << endl;
    */
    
    for (int j = 0;j<numTestDofsForVarID;j++) {
      for (int i = 0;i<numPoints;i++) {
        int dofIndex = testOrderingPtr->getDofIndex(testID, j); // to index into total test dof vector
        //	double basisValue = (*basisValues)(j,i);
        double basisValue = (*transformedBasisValues)(cellIndex,j,i);
        values(cellIndex,i) += basisValue*_rieszRepDofsGlobal[cellID](dofIndex);
      }
    }
  }
}
