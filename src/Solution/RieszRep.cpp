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

// TODO: distribute and 
double RieszRep::getNorm(){
  
  vector< ElementPtr > allElems = _mesh->activeElements(); // CHANGE TO DISTRIBUTED - THIS SHOULD GATHER AND DISTRIBUTE NORM INFO
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
  
  _rieszRepDofsGlobal = _rieszRepDofs; // to be replaced by an MPI call
  
}

// computes riesz representation over a single element - map is from int (testID) to FieldContainer of values (sized NumDofs/)
void RieszRep::computeRepresentationValues(int testID, FieldContainer<double> &values, BasisCachePtr basisCache){

  vector< ElementPtr > allElems = _mesh->activeElements(); // CHANGE TO DISTRIBUTED COMPUTATION

  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  //  cout << "num cells = " << numCells << endl;

  values.initialize(0.0);
  vector<int> cellIDs = basisCache->cellIDs();
  
  for (int cellIndex = 0;cellIndex<numCells;cellIndex++){
    int cellID = cellIDs[cellIndex];
    ElementPtr elem = allElems[cellID];
    ElementTypePtr elemTypePtr = elem->elementType();   
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
    //  int numTestDofs = testOrderingPtr->totalDofs(); 
    int numTestDofsForVarID = testOrderingPtr->getBasisCardinality(testID, 0);

    BasisPtr testBasis = testOrderingPtr->getBasis(testID);
    FieldContainer<double> basisValues = *(basisCache->getValues(testBasis,IntrepidExtendedTypes::OP_VALUE));    

    for (int i = 0;i<numPoints;i++){
      for (int j = 0;j<numTestDofsForVarID;j++){
	int dofIndex = testOrderingPtr->getDofIndex(testID, j);
	values(cellIndex,i) += basisValues(j,i)*_rieszRepDofsGlobal[cellID](dofIndex);
	//	cout << "dof values at cell " << cellID << " and pt " << i << " and basis " << j << " = " << _rieszRepDofs[cellID](dofIndex) << ", " << basisValues(j,i) << endl;
      }
    }
  }
}
