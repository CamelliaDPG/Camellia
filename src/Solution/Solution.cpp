
// @HEADER
//
// Original Version Copyright © 2011 Sandia Corporation. All Rights Reserved.
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
 *  Solution.cpp
 *
 *  Created by Nathan Roberts on 6/27/11.
 *
 */


// Intrepid includes
#include "Intrepid_DefaultCubatureFactory.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"
#include "Intrepid_Basis.hpp"

#include "Amesos_Klu.h"
#include "Amesos.h"
#include "Amesos_Utils.h"

#include "CamelliaConfig.h"

// only use MUMPS when we have MPI
#ifdef HAVE_MPI
#ifdef USE_MUMPS
#include "Amesos_Mumps.h"
#endif
#endif

// Epetra includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif
#include "Epetra_Time.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_SerialDenseSolver.h"
#include "Epetra_SerialSymDenseMatrix.h"
#include "Epetra_SerialSpdDenseSolver.h"
#include "Epetra_DataAccess.h"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "ml_epetra_utils.h"
//#include "ml_common.h"
#include "ml_epetra_preconditioner.h"

#include <stdlib.h>

#include "BilinearFormUtility.h"
#include "BasisEvaluation.h"
#include "BasisCache.h"
#include "BasisSumFunction.h"

#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"

#include "Solver.h"

#include "Function.h"

#include "Solution.h"
#include "Projector.h"

#include "CondensedDofInterpreter.h"

#include "Var.h"

#include "CamelliaCellTools.h"

#include "AztecOO_ConditionNumber.h"

#include "SerialDenseWrapper.h"

#include "MPIWrapper.h"

#ifdef HAVE_EPETRAEXT_HDF5
#include <EpetraExt_HDF5.h>
#include <Epetra_SerialComm.h>
#endif

double Solution::conditionNumberEstimate( Epetra_LinearProblem & problem ) {
  // estimates the 2-norm condition number
  AztecOOConditionNumber conditionEstimator;
  conditionEstimator.initialize(*problem.GetOperator());
  
  int maxIters = 40000;
  double tol = 1e-10;
  int status = conditionEstimator.computeConditionNumber(maxIters, tol);
  if (status!=0)
    cout << "status result from computeConditionNumber(): " << status << endl;
  double condest = conditionEstimator.getConditionNumber();
  
  return condest;
}

int Solution::cubatureEnrichmentDegree() const {
  return _cubatureEnrichmentDegree;
}

void Solution::setCubatureEnrichmentDegree(int value) {
  _cubatureEnrichmentDegree = value;
}

static const int MAX_BATCH_SIZE_IN_BYTES = 3*1024*1024; // 3 MB
static const int MIN_BATCH_SIZE_IN_CELLS = 1; // overrides the above, if it results in too-small batches

// copy constructor:
Solution::Solution(const Solution &soln) {
  _mesh = soln.mesh();
  _dofInterpreter = _mesh.get();
  _bc = soln.bc();
  _rhs = soln.rhs();
  _ip = soln.ip();
  _solutionForCellIDGlobal = soln.solutionForCellIDGlobal();
  _filter = soln.filter();
  _lagrangeConstraints = soln.lagrangeConstraints();
  _reportConditionNumber = false;
  _reportTimingResults = false;
  _writeMatrixToMatlabFile = false;
  _writeMatrixToMatrixMarketFile = false;
  _writeRHSToMatrixMarketFile = false;
  _cubatureEnrichmentDegree = soln.cubatureEnrichmentDegree();
}

Solution::Solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc, Teuchos::RCP<RHS> rhs, Teuchos::RCP<DPGInnerProduct> ip) {
  _mesh = mesh;
  _dofInterpreter = mesh.get();
  _bc = bc;
  _rhs = rhs;
  _ip = ip;
  _lagrangeConstraints = Teuchos::rcp( new LagrangeConstraints ); // empty
  
  initialize();
}

void Solution::clear() {
  // clears all solution values.  Leaves everything else intact.
  _solutionForCellIDGlobal.clear();
}

void Solution::initialize() {
  // clear the data structure in case it already stores some stuff
  _solutionForCellIDGlobal.clear();
  
  _writeMatrixToMatlabFile = false;
  _writeMatrixToMatrixMarketFile = false;
  _writeRHSToMatrixMarketFile = false;
  _residualsComputed = false;
  _energyErrorComputed = false;
  _reportConditionNumber = false;
  _reportTimingResults = false;
  _globalSystemConditionEstimate = -1;
  _cubatureEnrichmentDegree = 0;
  
  _zmcsAsRankOneUpdate = false; // I believe this works, but it's slow!
  _zmcRho = -1; // default value: stabilization parameter for zero-mean constraints
}

void Solution::addSolution(Teuchos::RCP<Solution> otherSoln, double weight, bool allowEmptyCells, bool replaceBoundaryTerms) {
  // first, add the global solution vectors together.
  if (otherSoln->getLHSVector().get() != NULL) {
    if (_lhsVector.get() != NULL) {
      _lhsVector->Update(weight, *otherSoln->getLHSVector(), 1.0);
    } else {
      _lhsVector = Teuchos::rcp( new Epetra_FEVector( *otherSoln->getLHSVector() ) );
      _lhsVector->Scale(weight);
    }
    if (replaceBoundaryTerms) {
      Epetra_Map partMap = getPartitionMap();
      set<GlobalIndexType> boundaryIndices;
      set<GlobalIndexType> fluxIndices = _mesh->globalDofAssignment()->partitionOwnedGlobalFluxIndices();
      set<GlobalIndexType> traceIndices = _mesh->globalDofAssignment()->partitionOwnedGlobalTraceIndices();
      boundaryIndices.insert(fluxIndices.begin(), fluxIndices.end());
      boundaryIndices.insert(traceIndices.begin(), traceIndices.end());
      for (set<GlobalIndexType>::iterator traceIndexIt=boundaryIndices.begin(); traceIndexIt != boundaryIndices.end(); traceIndexIt++) {
        GlobalIndexTypeToCast traceIndex = (GlobalIndexTypeToCast) *traceIndexIt;
        int localIndex = partMap.LID(traceIndex);
        double value = weight * (*otherSoln->getLHSVector())[0][localIndex];
        _lhsVector->ReplaceGlobalValue(traceIndex, 0, value);
      }
    }
    // now, interpret the global data
    importSolution();
    
    clearComputedResiduals();
  }
}

void Solution::addSolution(Teuchos::RCP<Solution> otherSoln, double weight, set<int> varsToAdd, bool allowEmptyCells) {
  set<GlobalIndexType> globalIndicesForVars = _mesh->globalDofAssignment()->partitionOwnedIndicesForVariables(varsToAdd);
  Epetra_Map partMap = getPartitionMap();

  // add the global solution vectors together.
  if (otherSoln->getLHSVector().get() != NULL) {
    if (_lhsVector.get() == NULL) {
      // then we treat this solution as 0
      _lhsVector = Teuchos::rcp(new Epetra_FEVector(partMap,1,true));
      _lhsVector->PutScalar(0); // unclear whether this is redundant with constructor or not
    }
    for (set<GlobalIndexType>::iterator gidIt = globalIndicesForVars.begin(); gidIt != globalIndicesForVars.end(); gidIt++) {
      int lid = partMap.LID((GlobalIndexTypeToCast)*gidIt);
      (*_lhsVector)[0][lid] += (*otherSoln->getLHSVector())[0][lid] * weight;
    }
    // now, interpret the global data
    importSolution();
    
    clearComputedResiduals();
  }
}

bool Solution::cellHasCoefficientsAssigned(GlobalIndexType cellID) {
  return _solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end();
}

void Solution::solve() {
#ifdef HAVE_MPI
  solve(true);
#else
  solve(false);
#endif
}

void Solution::solve(bool useMumps) {
  Teuchos::RCP<Solver> solver;
#ifdef USE_MUMPS
  if (useMumps) {
    solver = Teuchos::rcp(new MumpsSolver());
  } else {
    solver = Teuchos::rcp(new KluSolver());
  }
#else
  solver = Teuchos::rcp(new KluSolver());
#endif
  solve(solver);
}

void Solution::setSolution(Teuchos::RCP<Solution> otherSoln) {
  _solutionForCellIDGlobal = otherSoln->solutionForCellIDGlobal();
  _lhsVector = Teuchos::rcp( new Epetra_FEVector(*otherSoln->getLHSVector()) );
  clearComputedResiduals();
}

void Solution::initializeLHSVector() {
//  _lhsVector = Teuchos::rcp( (Epetra_FEVector*) NULL); // force a delete
  Epetra_Map partMap = getPartitionMap();
  _lhsVector = Teuchos::rcp(new Epetra_FEVector(partMap,1,true));
  _lhsVector->PutScalar(0); // unclear whether this is redundant with constructor or not
  
  // set initial _lhsVector (initial guess for iterative solvers)
  set<GlobalIndexType> cellIDs = _mesh->cellIDsInPartition();
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    if (_solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end()) {
      int localTrialDofCount = _mesh->getElementType(cellID)->trialOrderPtr->totalDofs();
      if (localTrialDofCount==_solutionForCellIDGlobal[cellID].size()) { // guard against cases when solutions not registered with their meshes have their meshes p-refined beneath them.  In such a case, we'll just ignore the previous solution coefficients on the cell.
        _mesh->globalDofAssignment()->interpretLocalCoefficients(cellID, _solutionForCellIDGlobal[cellID], *_lhsVector);
      }
    }
  }
}

void Solution::initializeStiffnessAndLoad() {
  Epetra_Map partMap = getPartitionMap();
  
  int maxRowSize = _mesh->rowSizeUpperBound();
  
  _globalStiffMatrix = Teuchos::rcp(new Epetra_FECrsMatrix(::Copy, partMap, maxRowSize));
  _rhsVector = Teuchos::rcp(new Epetra_FEVector(partMap));
}

void Solution::populateStiffnessAndLoad() {
  int numProcs=Teuchos::GlobalMPISession::getNProc();;
  int rank = Teuchos::GlobalMPISession::getRank();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  Epetra_Map partMap = getPartitionMap();
  
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes(rank);
  vector< ElementTypePtr >::iterator elemTypeIt;
  
  //cout << "process " << rank << " about to loop over elementTypes.\n";
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,Comm);
  Epetra_Time timer(Comm);
  Epetra_Time subTimer(Comm);
  
  double testMatrixAssemblyTime = 0, testMatrixInversionTime = 0, localStiffnessDeterminationFromTestsTime = 0;
  double localStiffnessInterpretationTime = 0, rhsIntegrationAgainstOptimalTestsTime = 0, filterApplicationTime = 0;
  
  //  cout << "Computing local matrices" << endl;
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    //cout << "Solution: elementType loop, iteration: " << elemTypeNumber++ << endl;
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr, _mesh, false, _cubatureEnrichmentDegree));
    BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh,true, _cubatureEnrichmentDegree));
    
    DofOrderingPtr trialOrderingPtr = elemTypePtr->trialOrderPtr;
    DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
    int numTrialDofs = trialOrderingPtr->totalDofs();
    int numTestDofs = testOrderingPtr->totalDofs();
    int maxCellBatch = MAX_BATCH_SIZE_IN_BYTES / 8 / (numTestDofs*numTestDofs + numTestDofs*numTrialDofs + numTrialDofs*numTrialDofs);
    maxCellBatch = max( maxCellBatch, MIN_BATCH_SIZE_IN_CELLS );
    //cout << "numTestDofs^2:" << numTestDofs*numTestDofs << endl;
    //cout << "maxCellBatch: " << maxCellBatch << endl;
    
    FieldContainer<double> myPhysicalCellNodesForType = _mesh->physicalCellNodes(elemTypePtr);
    FieldContainer<double> myCellSideParitiesForType = _mesh->cellSideParities(elemTypePtr);
    int totalCellsForType = myPhysicalCellNodesForType.dimension(0);
    int startCellIndexForBatch = 0;
    Teuchos::Array<int> nodeDimensions, parityDimensions;
    myPhysicalCellNodesForType.dimensions(nodeDimensions);
    myCellSideParitiesForType.dimensions(parityDimensions);
    while (startCellIndexForBatch < totalCellsForType) {
      int cellsLeft = totalCellsForType - startCellIndexForBatch;
      int numCells = min(maxCellBatch,cellsLeft);
      
      // determine cellIDs
      vector<GlobalIndexType> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        GlobalIndexType cellID = _mesh->cellID(elemTypePtr, cellIndex+startCellIndexForBatch, rank);
        cellIDs.push_back(cellID);
      }
      
      //cout << "testDofOrdering: " << *testOrderingPtr;
      //cout << "trialDofOrdering: " << *trialOrderingPtr;
      nodeDimensions[0] = numCells;
      parityDimensions[0] = numCells;
      FieldContainer<double> physicalCellNodes(nodeDimensions,&myPhysicalCellNodesForType(startCellIndexForBatch,0,0));
      FieldContainer<double> cellSideParities(parityDimensions,&myCellSideParitiesForType(startCellIndexForBatch,0));
      
      bool createSideCacheToo = true;
      basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
      basisCache->setCellSideParities(cellSideParities);

      // hard-coding creating side cache for IP for now, since _ip->hasBoundaryTerms() only recognizes terms explicitly passed in as boundary terms:
      ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true);//_ip->hasBoundaryTerms()); // create side cache if ip has boundary values
      ipBasisCache->setCellSideParities(cellSideParities); // I don't anticipate these being needed, though
      
      //int numCells = physicalCellNodes.dimension(0);
//      CellTopoPtrLegacy cellTopoPtr = elemTypePtr->cellTopoPtr;
//      
//      //      { // this block is not necessary for the solution.  Here just to produce debugging output
//      //        FieldContainer<double> preStiffness(numCells,numTestDofs,numTrialDofs );
//      //
//      //        BilinearFormUtility::computeStiffnessMatrix(preStiffness, _mesh->bilinearForm(),
//      //                                                    trialOrderingPtr, testOrderingPtr, *(cellTopoPtr.get()),
//      //                                                    physicalCellNodes, cellSideParities);
//      //        FieldContainer<double> preStiffnessTransposed(numCells,numTrialDofs,numTestDofs );
//      //        BilinearFormUtility::transposeFCMatrices(preStiffnessTransposed,preStiffness);
//      //
//      ////        cout << "preStiffness:\n" << preStiffness;
//      //      }
//      
//      subTimer.ResetStartTime();
//      
//      FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
//      
//      _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, ipBasisCache);
//      
//      testMatrixAssemblyTime += subTimer.ElapsedTime();
//      
//      //      cout << "ipMatrix:\n" << ipMatrix;
//      
//      subTimer.ResetStartTime();
//      FieldContainer<double> optTestCoeffs(numCells,numTrialDofs,numTestDofs);
//      
//      int optSuccess = _mesh->bilinearForm()->optimalTestWeights(optTestCoeffs, ipMatrix, elemTypePtr,
//                                                                 cellSideParities, basisCache);
//      testMatrixInversionTime += subTimer.ElapsedTime();
////      cout << "optTestCoeffs:\n" << optTestCoeffs;
//      
//      if ( optSuccess != 0 ) {
//        cout << "**** WARNING: in Solution.solve(), optimal test function computation failed with error code " << optSuccess << ". ****\n";
//      }
//      
//      //cout << "optTestCoeffs\n" << optTestCoeffs;
//      
//      subTimer.ResetStartTime();
//      FieldContainer<double> finalStiffness(numCells,numTrialDofs,numTrialDofs);
//      
//      BilinearFormUtility::computeStiffnessMatrix(finalStiffness,ipMatrix,optTestCoeffs);
//      localStiffnessDeterminationFromTestsTime += subTimer.ElapsedTime();
////      cout << "finalStiffness:\n" << finalStiffness;
//      
//      subTimer.ResetStartTime();
//      FieldContainer<double> localRHSVector(numCells, numTrialDofs);
//      _rhs->integrateAgainstOptimalTests(localRHSVector, optTestCoeffs, testOrderingPtr, basisCache);
//      rhsIntegrationAgainstOptimalTestsTime += subTimer.ElapsedTime();
      
      FieldContainer<double> localStiffness(numCells,numTrialDofs,numTrialDofs);
      FieldContainer<double> localRHSVector(numCells,numTrialDofs);
      
      _mesh->bilinearForm()->localStiffnessMatrixAndRHS(localStiffness, localRHSVector, _ip, ipBasisCache, _rhs, basisCache);
      
      // apply filter(s) (e.g. penalty method, preconditioners, etc.)
      if (_filter.get()) {
        subTimer.ResetStartTime();
        _filter->filter(localStiffness,localRHSVector,basisCache,_mesh,_bc);
        filterApplicationTime += subTimer.ElapsedTime();
        //        _filter->filter(localRHSVector,physicalCellNodes,cellIDs,_mesh,_bc);
      }
      
      //      cout << "local stiffness matrices:\n" << finalStiffness;
      //      cout << "local loads:\n" << localRHSVector;
      
      subTimer.ResetStartTime();
      
      FieldContainer<GlobalIndexType> globalDofIndices;
      
      FieldContainer<GlobalIndexTypeToCast> globalDofIndicesCast;
      
      Teuchos::Array<int> localStiffnessDim(2,numTrialDofs);
      Teuchos::Array<int> localRHSDim(1,numTrialDofs);
      
      FieldContainer<double> interpretedStiffness;
      FieldContainer<double> interpretedRHS;
      
      Teuchos::Array<int> dim;
      
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        GlobalIndexType cellID = _mesh->cellID(elemTypePtr,cellIndex+startCellIndexForBatch,rank);
        FieldContainer<double> cellStiffness(localStiffnessDim,&localStiffness(cellIndex,0,0)); // shallow copy
        FieldContainer<double> cellRHS(localRHSDim,&localRHSVector(cellIndex,0)); // shallow copy
        
        _dofInterpreter->interpretLocalData(cellID, cellStiffness, cellRHS, interpretedStiffness, interpretedRHS, globalDofIndices);
        
        // cast whatever the global index type is to a type that Epetra supports
        globalDofIndices.dimensions(dim);
        globalDofIndicesCast.resize(dim);
        
        for (int dofOrdinal = 0; dofOrdinal < globalDofIndices.size(); dofOrdinal++) {
          globalDofIndicesCast[dofOrdinal] = globalDofIndices[dofOrdinal];
        }
                
        _globalStiffMatrix->InsertGlobalValues(globalDofIndices.size(),&globalDofIndicesCast(0),
                                               globalDofIndices.size(),&globalDofIndicesCast(0),&interpretedStiffness[0]);
        _rhsVector->SumIntoGlobalValues(globalDofIndices.size(),&globalDofIndicesCast(0),&interpretedRHS[0]);
      }
      localStiffnessInterpretationTime += subTimer.ElapsedTime();
      
      startCellIndexForBatch += numCells;
    }
  }
  {
/*    cout << "testMatrixAssemblyTime: " << testMatrixAssemblyTime << " seconds.\n";
    cout << "testMatrixInversionTime: " << testMatrixInversionTime << " seconds.\n";
    cout << "localStiffnessDeterminationFromTestsTime: " << localStiffnessDeterminationFromTestsTime << " seconds.\n";
    cout << "localStiffnessInterpretationTime: " << localStiffnessInterpretationTime << " seconds.\n";
    cout << "rhsIntegrationAgainstOptimalTestsTime: " << rhsIntegrationAgainstOptimalTestsTime << " seconds.\n";
    cout << "filterApplicationTime: " << filterApplicationTime << " seconds.\n";*/
  }
  
  double timeLocalStiffness = timer.ElapsedTime();
  //  cout << "Done computing local matrices" << endl;
  Epetra_Vector timeLocalStiffnessVector(timeMap);
  timeLocalStiffnessVector[0] = timeLocalStiffness;
  
  int localRowIndex = myGlobalIndicesSet.size(); // starts where the dofs left off
  
  // order is: element-lagrange, then (on rank 0) global lagrange and ZMC
  for (int elementConstraintIndex = 0; elementConstraintIndex < _lagrangeConstraints->numElementConstraints();
       elementConstraintIndex++) {
    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
      ElementTypePtr elemTypePtr = *(elemTypeIt);
      BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh));
      
      // get cellIDs for basisCache
      vector< ElementPtr > cells = _mesh->elementsOfType(rank,elemTypePtr);
      int numCells = cells.size();
      vector<GlobalIndexType> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        int cellID = cells[cellIndex]->cellID();
        cellIDs.push_back(cellID);
      }
      // set physical cell nodes:
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
      bool createSideCacheToo = true;
      basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
      basisCache->setCellSideParities(_mesh->cellSideParities(elemTypePtr));
      
      int numTrialDofs = elemTypePtr->trialOrderPtr->totalDofs();
      FieldContainer<double> lhs(numCells,numTrialDofs);
      FieldContainer<double> rhs(numCells);
      _lagrangeConstraints->getCoefficients(lhs,rhs,elementConstraintIndex,
                                            elemTypePtr->trialOrderPtr,basisCache);
      
      FieldContainer<GlobalIndexTypeToCast> globalDofIndices(numTrialDofs+1); // max # of nonzeros
      FieldContainer<double> nonzeroValues(numTrialDofs+1);
      Teuchos::Array<int> localLHSDim(1, numTrialDofs); // changed from (numTrialDofs) by NVR, 8/27/14
      FieldContainer<double> interpretedLHS;

      FieldContainer<GlobalIndexType> interpretedGlobalDofIndices;
      
      // need to ask for local stiffness, too, for condensed dof interpreter, even though this is not used.
      FieldContainer<double> dummyLocalStiffness(numTrialDofs, numTrialDofs);
      FieldContainer<double> dummyInterpretedStiffness;
      
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        GlobalIndexTypeToCast globalRowIndex = partMap.GID(localRowIndex);
        int nnz = 0;
        FieldContainer<double> localLHS(localLHSDim,&lhs(cellIndex,0)); // shallow copy
        _dofInterpreter->interpretLocalData(cellIDs[cellIndex], dummyLocalStiffness, localLHS, dummyInterpretedStiffness,
                                            interpretedLHS, interpretedGlobalDofIndices);
        
        for (int i=0; i<interpretedLHS.size(); i++) {
          if (interpretedLHS(i) != 0.0) {
	          globalDofIndices(nnz) = interpretedGlobalDofIndices(i);
            nonzeroValues(nnz) = interpretedLHS(i);
            nnz++;
          }
        }
        // rhs:
        globalDofIndices(nnz) = globalRowIndex;
        if (nnz!=0) {
          nonzeroValues(nnz) = 0.0;
        } else { // no nonzero weights
          nonzeroValues(nnz) = 1.0; // just put a 1 in the diagonal to avoid singular matrix
        }
        // insert row:
        _globalStiffMatrix->InsertGlobalValues(1,&globalRowIndex,nnz+1,&globalDofIndices(0),
                                               &nonzeroValues(0));
        // insert column:
        _globalStiffMatrix->InsertGlobalValues(nnz+1,&globalDofIndices(0),1,&globalRowIndex,
                                               &nonzeroValues(0));
        _rhsVector->ReplaceGlobalValues(1,&globalRowIndex,&rhs(cellIndex));
        
        localRowIndex++;
      }
    }
  }
  
  //  // compute max, min h
  //  // TODO: get rid of the Global calls below (MPI-enable this code)
  //  double maxCellMeasure = 0;
  //  double minCellMeasure = 1e300;
  //  vector< ElementTypePtr > elemTypes = _mesh->elementTypes(); // global element types
  //  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
  //    ElementTypePtr elemType = *elemTypeIt;
  //    vector< ElementPtr > elems = _mesh->elementsOfTypeGlobal(elemType);
  //    vector<GlobalIndexType> cellIDs;
  //    for (int i=0; i<elems.size(); i++) {
  //      cellIDs.push_back(elems[i]->cellID());
  //    }
  //    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemType);
  //    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,_mesh) );
  //    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
  //    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
  //
  //    for (int i=0; i<elems.size(); i++) {
  //      maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
  //      minCellMeasure = min(minCellMeasure,cellMeasures(i));
  //    }
  //  }
  //  double min_h = sqrt(minCellMeasure);
  //  double max_h = sqrt(maxCellMeasure);
  
  
  // TODO: change ZMC imposition to be a distributed computation, instead of doing it all on rank 0
  //       (It's both the code below and the integrateBasisFunctions() methods that will need revision.)
  vector<int> zeroMeanConstraints = getZeroMeanConstraints();
  int numGlobalConstraints = _lagrangeConstraints->numGlobalConstraints();
  TEUCHOS_TEST_FOR_EXCEPTION(numGlobalConstraints != 0, std::invalid_argument, "global constraints not yet supported in Solution.");
  for (int lagrangeIndex = 0; lagrangeIndex < numGlobalConstraints; lagrangeIndex++) {
    int globalRowIndex = partMap.GID(localRowIndex);
    
    localRowIndex++;
  }
  
  // impose zero mean constraints:
  for (vector< int >::iterator trialIt = zeroMeanConstraints.begin(); trialIt != zeroMeanConstraints.end(); trialIt++) {
    int trialID = *trialIt;
    
    // sample an element to make sure that the basis used for trialID is nodal
    // (this is assumed in our imposition mechanism)
    GlobalIndexType firstActiveCellID = *_mesh->getActiveCellIDs().begin();
    ElementTypePtr elemTypePtr = _mesh->getElement(firstActiveCellID)->elementType();
    BasisPtr trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
    if (!trialBasis->isNodal()) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Zero-mean constraint imposition assumes a nodal basis, and this basis isn't nodal.");
    }
    
    GlobalIndexTypeToCast zmcIndex;
    if (rank==0)
      zmcIndex = partMap.GID(localRowIndex);
    else
      zmcIndex = 0;
    
    zmcIndex = MPIWrapper::sum(zmcIndex);
    
    //      cout << "Imposing zero-mean constraint for variable " << _mesh->bilinearForm()->trialName(trialID) << endl;
    FieldContainer<double> basisIntegrals;
    FieldContainer<GlobalIndexTypeToCast> globalIndices;
    integrateBasisFunctions(globalIndices,basisIntegrals, trialID);
    int numValues = globalIndices.size();
    
    if (_zmcsAsRankOneUpdate) {
      // TODO: debug this (not working)
      // first pass; can make more efficient by implementing as a symmetric SerialDenseMatrix
      FieldContainer<double> product(numValues,numValues);
      double denominator = 0.0;
      for (int i=0; i<numValues; i++) {
        denominator += basisIntegrals(i);
      }
      denominator *= denominator;
      
      for (int i=0; i<numValues; i++) {
        for (int j=0; j<numValues; j++) {
          product(i,j) = _zmcRho * basisIntegrals(i) * basisIntegrals(j) / denominator;
        }
      }
      _globalStiffMatrix->SumIntoGlobalValues(numValues, &globalIndices(0), numValues, &globalIndices(0), &product(0,0));
    } else { // otherwise, we increase the size of the system to accomodate the zmc...
      // insert row:
      _globalStiffMatrix->InsertGlobalValues(1,&zmcIndex,numValues,&globalIndices(0),&basisIntegrals(0));
      // insert column:
      _globalStiffMatrix->InsertGlobalValues(numValues,&globalIndices(0),1,&zmcIndex,&basisIntegrals(0));
      
      //      cout << "in zmc, diagonal entry: " << rho << endl;
      //rho /= numValues;
      if (rank==0) { // insert the diagonal entry on rank 0; other ranks insert basis integrals according to which cells they own
        double rho_entry = - 1.0 / _zmcRho;
        _globalStiffMatrix->InsertGlobalValues(1,&zmcIndex,1,&zmcIndex,&rho_entry);
      }
      if (rank==0) localRowIndex++;
    }
  }
  // end of ZMC imposition
  
  Comm.Barrier();  // for cleaner time measurements, let everyone else catch up before calling ResetStartTime() and GlobalAssemble()
  timer.ResetStartTime();
  
  _rhsVector->GlobalAssemble();
  
  //  EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector_before_bcs.dat",rhsVector,0,0,false);
  
  _globalStiffMatrix->GlobalAssemble(); // will call globalStiffMatrix.FillComplete();
  
  double timeGlobalAssembly = timer.ElapsedTime();
  Epetra_Vector timeGlobalAssemblyVector(timeMap);
  timeGlobalAssemblyVector[0] = timeGlobalAssembly;
  
//  cout << "debugging: outputting stiffness matrix before BC imposition to /tmp/stiffness_noBCs.dat\n";
//  EpetraExt::RowMatrixToMatlabFile("/tmp/stiffness_noBCs.dat",*_globalStiffMatrix);
  
  // determine and impose BCs
  
  timer.ResetStartTime();
  
  imposeBCs();
  
  double timeBCImposition = timer.ElapsedTime();
  Epetra_Vector timeBCImpositionVector(timeMap);
  timeBCImpositionVector[0] = timeBCImposition;
  
  _rhsVector->GlobalAssemble();
  
  Epetra_FEVector lhsVector(partMap, true);
  
  if (_writeRHSToMatrixMarketFile) {
    if (rank==0) {
      cout << "Solution: writing rhs to file: " << _rhsFilePath << endl;
    }
    EpetraExt::MultiVectorToMatrixMarketFile(_rhsFilePath.c_str(),*_rhsVector,0,0,false);
  }
  
  // Dump matrices to disk
  if (_writeMatrixToMatlabFile){
    //    EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector.dat",rhsVector,0,0,false);
    EpetraExt::RowMatrixToMatlabFile(_matrixFilePath.c_str(),*_globalStiffMatrix);
    //    EpetraExt::MultiVectorToMatrixMarketFile("lhs_vector.dat",lhsVector,0,0,false);
  }
  if (_writeMatrixToMatrixMarketFile){
    EpetraExt::RowMatrixToMatrixMarketFile(_matrixFilePath.c_str(),*_globalStiffMatrix);
  }
  
  int err = timeLocalStiffnessVector.Norm1( &_totalTimeLocalStiffness );
  err = timeGlobalAssemblyVector.Norm1( &_totalTimeGlobalAssembly );
  err = timeBCImpositionVector.Norm1( &_totalTimeBCImposition );
  
  err = timeLocalStiffnessVector.MeanValue( &_meanTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MeanValue( &_meanTimeGlobalAssembly );
  err = timeBCImpositionVector.MeanValue( &_meanTimeBCImposition );
  
  err = timeLocalStiffnessVector.MinValue( &_minTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MinValue( &_minTimeGlobalAssembly );
  err = timeBCImpositionVector.MinValue( &_minTimeBCImposition );
  
  err = timeLocalStiffnessVector.MaxValue( &_maxTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MaxValue( &_maxTimeGlobalAssembly );
  err = timeBCImpositionVector.MaxValue( &_maxTimeBCImposition );
}

void Solution::setProblem(Teuchos::RCP<Solver> solver) {
  Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&*_globalStiffMatrix, &*_lhsVector, &*_rhsVector));
  solver->setProblem(problem);
}

void Solution::solveWithPrepopulatedStiffnessAndLoad(Teuchos::RCP<Solver> solver, bool callResolveInsteadOfSolve) {
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
//  cout << "rank " << rank << " has " << myGlobalIndicesSet.size() << " locally-owned dof indices.\n";
  Epetra_Map partMap = getPartitionMap();
  
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes(rank);
  vector< ElementTypePtr >::iterator elemTypeIt;
  
  //cout << "process " << rank << " about to loop over elementTypes.\n";
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,Comm);
  Epetra_Time timer(Comm);
  
  if (_reportConditionNumber) {
    //    double oneNorm = globalStiffMatrix.NormOne();
    Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&*_globalStiffMatrix, &*_lhsVector, &*_rhsVector));
    double condest = conditionNumberEstimate(*problem);
    if (rank == 0) {
      // cout << "(one-norm) of global stiffness matrix: " << oneNorm << endl;
      cout << "condition # estimate for global stiffness matrix: " << condest << endl;
    }
    _globalSystemConditionEstimate = condest;
  }
  
  timer.ResetStartTime();
  
//  GlobalIndexType dofInterpreterGlobalDofCount = _dofInterpreter->globalDofCount();
//  GlobalIndexType meshGlobalDofCount = _mesh->globalDofCount();
//  if (rank==0) cout << "About to call global solver with " << dofInterpreterGlobalDofCount << " global dof count.\n";
//  if (rank==0) cout << "(Mesh sees " << meshGlobalDofCount << " dofs.)\n";
  
//  cout << "On rank " << rank << ", about to call global solver with " << _dofInterpreter->globalDofCount() << " global dof count.\n";
//  cout << "(On rank " << rank << ", mesh sees " << _mesh->globalDofCount() << " dofs.)\n";
  
  int solveSuccess;
  if (!callResolveInsteadOfSolve) {
    solveSuccess = solver->solve();
  } else {
    solveSuccess = solver->resolve();
  }

//  if (rank==0) cout << "Returned from global solver.\n";
  
  if (solveSuccess != 0 ) {
    cout << "**** WARNING: in Solution.solve(), solver->solve() failed with error code " << solveSuccess << ". ****\n";
  }
  
  double timeSolve = timer.ElapsedTime();
  Epetra_Vector timeSolveVector(timeMap);
  timeSolveVector[0] = timeSolve;
  
  int err = timeSolveVector.Norm1( &_totalTimeSolve );
  err = timeSolveVector.MeanValue( &_meanTimeSolve );
  err = timeSolveVector.MinValue( &_minTimeSolve );
  err = timeSolveVector.MaxValue( &_maxTimeSolve );
}

void Solution::solve(Teuchos::RCP<Solver> solver) {
//  int rank = Teuchos::GlobalMPISession::getRank();

  initializeLHSVector();
  initializeStiffnessAndLoad();
  setProblem(solver);
  populateStiffnessAndLoad();
  solveWithPrepopulatedStiffnessAndLoad(solver);
//  cout << "about to call importSolution on rank " << rank << endl;
  importSolution();
//  cout << "calling importGlobalSolution (this doesn't scale well, especially in its current form).\n";
//  importGlobalSolution();
//  cout << "about to call clearComputedResiduals on rank " << rank << endl;
  
  clearComputedResiduals(); // now that we've solved, will need to recompute residuals...
  
  if (_reportTimingResults ) {
    reportTimings();
  }
}

void Solution::reportTimings() {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  if (rank == 0) {
    cout << "****** SUM OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _totalTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _totalTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _totalTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _totalTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _totalTimeDistributeSolution << " sec." << endl << endl;
    
    cout << "****** MEAN OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _meanTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _meanTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _meanTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _meanTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _meanTimeDistributeSolution << " sec." << endl << endl;
    
    cout << "****** MAX OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _maxTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _maxTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _maxTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _maxTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _maxTimeDistributeSolution << " sec." << endl << endl;
    
    cout << "****** MIN OF TIMING REPORTS ******\n";
    cout << "localStiffness: " << _minTimeLocalStiffness << " sec." << endl;
    cout << "globalAssembly: " << _minTimeGlobalAssembly << " sec." << endl;
    cout << "impose BCs:     " << _minTimeBCImposition << " sec." << endl;
    cout << "solve:          " << _minTimeSolve << " sec." << endl;
    cout << "dist. solution: " << _minTimeDistributeSolution << " sec." << endl;
  }
}

void Solution::clearComputedResiduals() {
  _residualsComputed = false;
  _energyErrorComputed = false;
  _energyErrorForCellIDGlobal.clear();
  _residualForElementType.clear();
}

Teuchos::RCP<Mesh> Solution::mesh() const {
  return _mesh;
}

Teuchos::RCP<BC> Solution::bc() const {
  return _bc;
}
Teuchos::RCP<RHS> Solution::rhs() const {
  return _rhs;
}

void Solution::importSolution() {
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  int rank     = Teuchos::GlobalMPISession::getRank();
  Epetra_Time timer(Comm);
  
//  cout << "on rank " << rank << ", about to determine globalDofIndicesForPartition\n";
  
  set<GlobalIndexType> globalDofIndicesForMyCells;
  set<GlobalIndexType> myCellIDs = _mesh->globalDofAssignment()->cellsInPartition(-1);
  for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    set<GlobalIndexType> globalDofsForCell = _dofInterpreter->globalDofIndicesForCell(cellID);
    globalDofIndicesForMyCells.insert(globalDofsForCell.begin(),globalDofsForCell.end());
  }

//  cout << "on rank " << rank << ", about to create myDofs container of size "<< globalDofIndicesForMyCells.size() << "\n";
  GlobalIndexTypeToCast myDofs[globalDofIndicesForMyCells.size()];
  GlobalIndexTypeToCast* myDof = (globalDofIndicesForMyCells.size() > 0) ? &myDofs[0] : NULL;
  for (set<GlobalIndexType>::iterator dofIndexIt = globalDofIndicesForMyCells.begin();
       dofIndexIt != globalDofIndicesForMyCells.end(); dofIndexIt++) {
    *myDof = *dofIndexIt;
    myDof++;
  }
//  cout << "on rank " << rank << ", about to create myCellsMap\n";
  Epetra_Map     myCellsMap(-1, globalDofIndicesForMyCells.size(), myDofs, 0, Comm);
  
  // Import solution onto current processor
  Epetra_Map partMap = getPartitionMap();
  Epetra_Import  solnImporter(myCellsMap, partMap);
  Epetra_Vector  solnCoeff(myCellsMap);
//  cout << "on rank " << rank << ", about to Import\n";
  solnCoeff.Import(*_lhsVector, solnImporter, Insert);
//  cout << "on rank " << rank << ", returned from Import\n";
  
  // copy the dof coefficients into our data structure
  for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
//    cout << "on rank " << rank << ", about to interpret data for cell " << cellID << "\n";
    FieldContainer<double> cellDofs(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    _dofInterpreter->interpretGlobalCoefficients(cellID,cellDofs,solnCoeff);
    _solutionForCellIDGlobal[cellID] = cellDofs;
  }
//  cout << "on rank " << rank << ", finished interpretation\n";
  double timeDistributeSolution = timer.ElapsedTime();

  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,Comm);
  Epetra_Vector timeDistributeSolutionVector(timeMap);
  timeDistributeSolutionVector[0] = timeDistributeSolution;
  
  int err = timeDistributeSolutionVector.Norm1( &_totalTimeDistributeSolution );
  err = timeDistributeSolutionVector.MeanValue( &_meanTimeDistributeSolution );
  err = timeDistributeSolutionVector.MinValue( &_minTimeDistributeSolution );
  err = timeDistributeSolutionVector.MaxValue( &_maxTimeDistributeSolution );
}

void Solution::importGlobalSolution() {
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  Epetra_Time timer(Comm);

  GlobalIndexType globalDofCount = _mesh->globalDofAssignment()->globalDofCount();
  
  GlobalIndexTypeToCast myDofs[globalDofCount];
  GlobalIndexTypeToCast* myDof = (globalDofCount > 0) ? &myDofs[0] : NULL;
  for (GlobalIndexType dofIndex = 0; dofIndex < globalDofCount; dofIndex++) {
    *myDof = dofIndex;
    myDof++;
  }
  
  Epetra_Map     myCellsMap(-1, globalDofCount, myDofs, 0, Comm);
  
  // Import global solution onto each processor
  Epetra_Map partMap = getPartitionMap();
  Epetra_Import  solnImporter(myCellsMap, partMap);
  Epetra_Vector  solnCoeff(myCellsMap);
  solnCoeff.Import(*_lhsVector, solnImporter, Insert);
  
  set<GlobalIndexType> globalActiveCellIDs = _mesh->getActiveCellIDs();
  // copy the dof coefficients into our data structure
  for (set<GlobalIndexType>::iterator cellIDIt = globalActiveCellIDs.begin(); cellIDIt != globalActiveCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    FieldContainer<double> cellDofs(_mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
    _dofInterpreter->interpretGlobalCoefficients(cellID,cellDofs,solnCoeff);
    _solutionForCellIDGlobal[cellID] = cellDofs;
  }
  double timeDistributeSolution = timer.ElapsedTime();
  
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,Comm);
  Epetra_Vector timeDistributeSolutionVector(timeMap);
  timeDistributeSolutionVector[0] = timeDistributeSolution;
  
  int err = timeDistributeSolutionVector.Norm1( &_totalTimeDistributeSolution );
  err = timeDistributeSolutionVector.MeanValue( &_meanTimeDistributeSolution );
  err = timeDistributeSolutionVector.MinValue( &_minTimeDistributeSolution );
  err = timeDistributeSolutionVector.MaxValue( &_maxTimeDistributeSolution );
}

Teuchos::RCP<DPGInnerProduct> Solution::ip() const {
  return _ip;
}

void Solution::imposeBCs() {
  int rank     = Teuchos::GlobalMPISession::getRank();
  
  FieldContainer<GlobalIndexType> bcGlobalIndices;
  FieldContainer<double> bcGlobalValues;
  
  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  //  cout << "rank " << rank << " has " << myGlobalIndicesSet.size() << " locally-owned dof indices.\n";
  Epetra_Map partMap = getPartitionMap();

  _mesh->boundary().bcsToImpose(bcGlobalIndices,bcGlobalValues,*(_bc.get()), myGlobalIndicesSet);
  int numBCs = bcGlobalIndices.size();
  
  FieldContainer<GlobalIndexTypeToCast> bcGlobalIndicesCast;
  // cast whatever the global index type is to a type that Epetra supports
  Teuchos::Array<int> dim;
  bcGlobalIndices.dimensions(dim);
  bcGlobalIndicesCast.resize(dim);
  for (int dofOrdinal = 0; dofOrdinal < bcGlobalIndices.size(); dofOrdinal++) {
    bcGlobalIndicesCast[dofOrdinal] = bcGlobalIndices[dofOrdinal];
  }
//  cout << "bcGlobalIndices:" << endl << bcGlobalIndices;
  //  cout << "bcGlobalValues:" << endl << bcGlobalValues;
  
  Epetra_MultiVector v(partMap,1);
  v.PutScalar(0.0);
  for (int i = 0; i < numBCs; i++) {
    v.ReplaceGlobalValue(bcGlobalIndicesCast(i), 0, bcGlobalValues(i));
  }
  
  Epetra_MultiVector rhsDirichlet(partMap,1);
  _globalStiffMatrix->Apply(v,rhsDirichlet);
  
  // Update right-hand side
  _rhsVector->Update(-1.0,rhsDirichlet,1.0);
  
  if (numBCs == 0) {
    //cout << "Solution: Warning: Imposing no BCs." << endl;
  } else {
    int err = _rhsVector->ReplaceGlobalValues(numBCs,&bcGlobalIndicesCast(0),&bcGlobalValues(0));
    if (err != 0) {
      cout << "ERROR: rhsVector.ReplaceGlobalValues(): some indices non-local...\n";
    }
    err = _lhsVector->ReplaceGlobalValues(numBCs,&bcGlobalIndicesCast(0),&bcGlobalValues(0));
    if (err != 0) {
      cout << "ERROR: rhsVector.ReplaceGlobalValues(): some indices non-local...\n";
    }
  }
  // Zero out rows and columns of stiffness matrix corresponding to Dirichlet edges
  //  and add one to diagonal.
  FieldContainer<int> bcLocalIndices(bcGlobalIndices.dimension(0));
  for (int i=0; i<bcGlobalIndices.dimension(0); i++) {
    bcLocalIndices(i) = _globalStiffMatrix->LRID(bcGlobalIndicesCast(i));
  }
  if (numBCs == 0) {
    ML_Epetra::Apply_OAZToMatrix(NULL, 0, *_globalStiffMatrix);
  } else {
    ML_Epetra::Apply_OAZToMatrix(&bcLocalIndices(0), numBCs, *_globalStiffMatrix);
  }
}

Teuchos::RCP<LocalStiffnessMatrixFilter> Solution::filter() const{
  return _filter;
}

ElementTypePtr Solution::getEquivalentElementType(Teuchos::RCP<Mesh> otherMesh, ElementTypePtr elemType) {
  DofOrderingPtr otherTrial = elemType->trialOrderPtr;
  DofOrderingPtr otherTest = elemType->testOrderPtr;
  DofOrderingPtr myTrial = _mesh->getDofOrderingFactory().getTrialOrdering(*otherTrial);
  DofOrderingPtr myTest = _mesh->getDofOrderingFactory().getTestOrdering(*otherTest);
  Teuchos::RCP<shards::CellTopology> otherCellTopoPtrLegacy = elemType->cellTopoPtr;
  Teuchos::RCP<shards::CellTopology> myCellTopoPtrLegacy;
  for (int i=0; i<_mesh->activeElements().size(); i++) {
    myCellTopoPtrLegacy = _mesh->activeElements()[i]->elementType()->cellTopoPtr;
    if (myCellTopoPtrLegacy->getKey() == otherCellTopoPtrLegacy->getKey() ) {
      break; // out of for loop
    }
  }
  return _mesh->getElementTypeFactory().getElementType(myTrial,myTest,myCellTopoPtrLegacy);
}

// The following method isn't really a good idea.
//bool Solution::equals(Solution& otherSolution, double tol) {
//  double maxDiff = 0.0;
//  map< int, FieldContainer<double> > otherSolutionForCell = otherSolution.solutionForCellIDGlobal();
//  int numElements = otherSolutionForCell.size();
//  if (numElements != _solutionForCellIDGlobal.size()) return false;
//
//  for (map< int, FieldContainer<double> >::iterator entryIt=_solutionForCellIDGlobal.begin();
//       entryIt != _solutionForCellIDGlobal.end(); entryIt++) {
//    int cellID = entryIt->first;
//    FieldContainer<double> mySoln = entryIt->second;
//    vector<double> centroid = _mesh->getCellCentroid(cellID);
//
//    vector<ElementPtr>
//    int otherCellID =
//    FieldContainer<double> otherSoln = otherSolutionForCell[cellID];
//    int numSolnEntries = mySoln.size();
//    if ( numSolnEntries != otherSoln.size() ) {
//      return false;
//    }
//    for (int i=0; i<numSolnEntries; i++) {
//      double mine = mySoln[i], theirs = otherSoln[i];
//      double diff = abs(mine-theirs);
//      if (diff > tol) {
//        return false;
//      }
//      maxDiff = max(maxDiff,diff);
//    }
//  }
//  //cout << "Solution maxDiff is " << maxDiff << endl;
//  return true;
//}

Epetra_MultiVector* Solution::getGlobalCoefficients() {
  return (*_lhsVector)(0);
}

double Solution::globalCondEstLastSolve() {
  // the condition # estimate for the last system matrix used in a solve, if _reportConditionNumber is true.
  return _globalSystemConditionEstimate;
}

void Solution::integrateBasisFunctions(FieldContainer<GlobalIndexTypeToCast> &globalIndices, FieldContainer<double> &values, int trialID) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  // only supports scalar-valued field bases right now...
  int sideIndex = 0; // field variables only
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  vector<GlobalIndexType> globalIndicesVector;
  vector<double> valuesVector;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    vector<GlobalIndexType> cellIDs = _mesh->globalDofAssignment()->cellIDsOfElementType(rank,elemTypePtr);
    int numCellsOfType = cellIDs.size();
    int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
    FieldContainer<double> valuesForType(numCellsOfType, basisCardinality);
    integrateBasisFunctions(valuesForType,elemTypePtr,trialID);
    
    int numTrialDofs = elemTypePtr->trialOrderPtr->totalDofs();
    FieldContainer<double> localDiscreteValues(numTrialDofs);
    FieldContainer<double> interpretedDiscreteValues;
    FieldContainer<GlobalIndexType> globalDofIndices;
    
    // need to ask for local stiffness, too, for condensed dof interpreter, even though this is not used.
    FieldContainer<double> dummyLocalStiffness(numTrialDofs, numTrialDofs);
    FieldContainer<double> dummyInterpretedStiffness;
    
    // copy into values:
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
      GlobalIndexType cellID = cellIDs[cellIndex];
      for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++) {
        IndexType dofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID, dofOrdinal);
        localDiscreteValues(dofIndex) = valuesForType(cellIndex,dofOrdinal);
      }
      _dofInterpreter->interpretLocalData(cellID, dummyLocalStiffness, localDiscreteValues, dummyInterpretedStiffness, interpretedDiscreteValues, globalDofIndices);
      
      for (int dofIndex=0; dofIndex<globalDofIndices.size(); dofIndex++) {
        if (interpretedDiscreteValues(dofIndex) != 0) {
          globalIndicesVector.push_back(globalDofIndices(dofIndex));
          valuesVector.push_back(interpretedDiscreteValues(dofIndex));
        }
      }
    }
  }
  int numValues = globalIndicesVector.size();
  globalIndices.resize(numValues);
  values.resize(numValues);
  for (int i=0; i<numValues; i++) {
    globalIndices[i] = globalIndicesVector[i];
    values[i] = valuesVector[i];
  }
}

void Solution::integrateBasisFunctions(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID) {
  int rank = Teuchos::GlobalMPISession::getRank();
  vector<GlobalIndexType> cellIDs = _mesh->globalDofAssignment()->cellIDsOfElementType(rank,elemTypePtr);
  
  int numCellsOfType = cellIDs.size();
  if (numCellsOfType==0) {
    return;
  }
  
  int sideIndex = 0;
  int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(0) != numCellsOfType,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(1) != basisCardinality,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  BasisPtr trialBasis;
  trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
  
  int cubDegree = trialBasis->getDegree();
  
  BasisCache basisCache(_mesh->physicalCellNodes(elemTypePtr), *(elemTypePtr->cellTopoPtr), cubDegree);
  
  Teuchos::RCP < const FieldContainer<double> > trialValuesTransformedWeighted;
  
  trialValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trialBasis, OP_VALUE);
  
  if (trialValuesTransformedWeighted->rank() != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateBasisFunctions only supports scalar-valued field variables at present.");
  }
  // integrate:
  int numPoints = trialValuesTransformedWeighted->dimension(2);
  for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
    for (int basisIndex=0; basisIndex < basisCardinality; basisIndex++) {
      for (int pointIndex=0; pointIndex < numPoints; pointIndex++) {
        values(cellIndex,basisIndex) += (*trialValuesTransformedWeighted)(cellIndex,basisIndex,pointIndex);
      }
    }
  }
  //FunctionSpaceTools::integrate<double>(values,*trialValuesTransformedWeighted,ones,COMP_BLAS);
}

double Solution::meanValue(int trialID) {
  return integrateSolution(trialID) / meshMeasure();
}

double Solution::meshMeasure() {
  double value = 0.0;
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes();
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    int cubDegree = 1;
    BasisCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), *(elemTypePtr->cellTopoPtr), cubDegree);
    FieldContainer<double> cellMeasures = basisCache.getCellMeasures();
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
      value += cellMeasures(cellIndex);
    }
  }
  return value;
}

double Solution::InfNormOfSolutionGlobal(int trialID){
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int rank     = Teuchos::GlobalMPISession::getRank();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
  int indexBase = 0;
  Epetra_Map procMap(numProcs,indexBase,Comm);
  double localInfNorm = InfNormOfSolution(trialID);
  Epetra_Vector infNormVector(procMap);
  infNormVector[0] = localInfNorm;
  double globalInfNorm;
  int errCode = infNormVector.NormInf( &globalInfNorm );
  if (errCode!=0){
    cout << "Error in infNormOfSolutionGlobal, errCode = " << errCode << endl;
  }
  return globalInfNorm;
}

double Solution::InfNormOfSolution(int trialID){
  
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  double value = 0.0;
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    vector< ElementPtr > cells = _mesh->elementsOfType(rank,elemTypePtr);
    int numCells = cells.size();
    // note: basisCache below will use a greater cubature degree than strictly necessary
    //       (it'll use maxTrialDegree + maxTestDegree, when it only needs maxTrialDegree * 2)
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh));
    
    // get cellIDs for basisCache
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      GlobalIndexType cellID = cells[cellIndex]->cellID();
      cellIDs.push_back(cellID);
    }
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
    bool createSideCacheToo = false;
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
    
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    FieldContainer<double> values(numCells,numPoints);
    bool weightForCubature = false;
    solutionValues(values, trialID, basisCache, weightForCubature);
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        value = max(abs(values(cellIndex,ptIndex)),value);
      }
    }
  }
  return value;
}

double Solution::L2NormOfSolutionGlobal(int trialID){
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
  
  int indexBase = 0;
  Epetra_Map procMap(numProcs,indexBase,Comm);
  double localL2Norm = L2NormOfSolution(trialID);
  Epetra_Vector l2NormVector(procMap);
  l2NormVector[0] = localL2Norm;
  double globalL2Norm;
  int errCode = l2NormVector.Norm1( &globalL2Norm );
  if (errCode!=0){
    cout << "Error in L2NormOfSolutionGlobal, errCode = " << errCode << endl;
  }
  return globalL2Norm;
}

double Solution::L2NormOfSolutionInCell(int trialID, GlobalIndexType cellID) {
  double value = 0.0;
  ElementTypePtr elemTypePtr = _mesh->getElement(cellID)->elementType();
  int numCells = 1;
  // note: basisCache below will use a greater cubature degree than strictly necessary
  //       (it'll use maxTrialDegree + maxTestDegree, when it only needs maxTrialDegree * 2)
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh));
  
  // get cellIDs for basisCache
  vector<GlobalIndexType> cellIDs;
  cellIDs.push_back(cellID);
  
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
  
  bool createSideCacheToo = false;
  basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
  
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  FieldContainer<double> values(numCells,numPoints);
  bool weightForCubature = false;
  solutionValues(values, trialID, basisCache, weightForCubature);
  FieldContainer<double> weightedValues(numCells,numPoints);
  weightForCubature = true;
  solutionValues(weightedValues, trialID, basisCache, weightForCubature);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      value += values(cellIndex,ptIndex)*weightedValues(cellIndex,ptIndex);
    }
  }
  
  return value;
}

double Solution::L2NormOfSolution(int trialID){
  
  int numProcs=1;
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  
  double value = 0.0;
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    vector< ElementPtr > cells = _mesh->elementsOfType(rank,elemTypePtr);
    int numCells = cells.size();
    // note: basisCache below will use a greater cubature degree than strictly necessary
    //       (it'll use maxTrialDegree + maxTestDegree, when it only needs maxTrialDegree * 2)
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh));
    
    // get cellIDs for basisCache
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      GlobalIndexType cellID = cells[cellIndex]->cellID();
      cellIDs.push_back(cellID);
    }
    
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
    
    bool createSideCacheToo = false;
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
    
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    FieldContainer<double> values(numCells,numPoints);
    bool weightForCubature = false;
    solutionValues(values, trialID, basisCache, weightForCubature);
    FieldContainer<double> weightedValues(numCells,numPoints);
    weightForCubature = true;
    solutionValues(weightedValues, trialID, basisCache, weightForCubature);
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        value += values(cellIndex,ptIndex)*weightedValues(cellIndex,ptIndex);
      }
    }
  }
  
  return value;
  
}

Teuchos::RCP<LagrangeConstraints> Solution::lagrangeConstraints() const {
  return _lagrangeConstraints;
}

Teuchos::RCP<Epetra_FEVector> Solution::getLHSVector() {
  return _lhsVector;
}

double Solution::integrateSolution(int trialID) {
  double value = 0.0;
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes();
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    FieldContainer<double> valuesForType(numCellsOfType);
    integrateSolution(valuesForType,elemTypePtr,trialID);
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
      value += valuesForType(cellIndex);
    }
  }
  return value;
}

void Solution::integrateSolution(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID) {
  int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
  int sideIndex = 0;
  int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(0) != numCellsOfType,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr))");
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 1,
                             std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr))");
  BasisPtr trialBasis;
  trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
  
  int cubDegree = trialBasis->getDegree();
  
  BasisCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), *(elemTypePtr->cellTopoPtr), cubDegree);
  
  Teuchos::RCP < const FieldContainer<double> > trialValuesTransformedWeighted;
  
  trialValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trialBasis, OP_VALUE);
  
  if (trialValuesTransformedWeighted->rank() != 3) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateSolution only supports scalar-valued field variables at present.");
  }
  
  // integrate:
  FieldContainer<double> physicalCubaturePoints = basisCache.getPhysicalCubaturePoints();
  
  FieldContainer<double> solnCoeffs(basisCardinality);
  
  int numPoints = trialValuesTransformedWeighted->dimension(2);
  for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
    int cellID = _mesh->cellID(elemTypePtr,cellIndex);
    solnCoeffsForCellID(solnCoeffs, cellID, trialID, sideIndex);
    for (int basisIndex=0; basisIndex < basisCardinality; basisIndex++) {
      for (int pointIndex=0; pointIndex < numPoints; pointIndex++) {
        values(cellIndex) += solnCoeffs(basisIndex) * (*trialValuesTransformedWeighted)(cellIndex,basisIndex,pointIndex);
      }
    }
  }
}

void Solution::integrateFlux(FieldContainer<double> &values, int trialID) {
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes();
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    FieldContainer<double> valuesForType(numCellsOfType);
    integrateFlux(valuesForType,elemTypePtr,trialID);
    // copy into values:
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
      int cellID = _mesh->cellID(elemTypePtr,cellIndex);
      values(cellID) = valuesForType(cellIndex);
    }
  }
}

void Solution::integrateFlux(FieldContainer<double> &values, ElementTypePtr elemTypePtr, int trialID) {
  typedef CellTools<double>  CellTools;
  typedef FunctionSpaceTools fst;
  
  values.initialize(0.0);
  
  FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);
  
  int numCells = physicalCellNodes.dimension(0);
  unsigned spaceDim = physicalCellNodes.dimension(2);
  
  DofOrdering dofOrdering = *(elemTypePtr->trialOrderPtr.get());
  
  shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
  int numSides = CamelliaCellTools::getSideCount(cellTopo);
  
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    // Get numerical integration points and weights
    DefaultCubatureFactory<double>  cubFactory;
    BasisPtr basis = dofOrdering.getBasis(trialID,sideIndex);
    int basisRank = dofOrdering.getBasisRank(trialID);
    int cubDegree = 2*basis->getDegree();
    
    bool boundaryIntegral = _mesh()->bilinearForm()->isFluxOrTrace(trialID);
    if ( !boundaryIntegral ) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateFlux() called for field variable.");
    }
    
    shards::CellTopology side(cellTopo.getCellTopologyData(spaceDim-1,sideIndex)); // create relevant subcell (side) topology
    int sideDim = side.getDimension();
    Teuchos::RCP<Cubature<double> > sideCub = cubFactory.create(side, cubDegree);
    int numCubPoints = sideCub->getNumPoints();
    FieldContainer<double> cubPointsSide(numCubPoints, sideDim); // cubature points from the pov of the side (i.e. a 1D set)
    FieldContainer<double> cubWeightsSide(numCubPoints);
    FieldContainer<double> cubPointsSideRefCell(numCubPoints, spaceDim); // cubPointsSide from the pov of the ref cell
    FieldContainer<double> jacobianSideRefCell(numCells, numCubPoints, spaceDim, spaceDim);
    
    sideCub->getCubature(cubPointsSide, cubWeightsSide);
    
    // compute geometric cell information
    //cout << "computing geometric cell info for boundary integral." << endl;
    CellTools::mapToReferenceSubcell(cubPointsSideRefCell, cubPointsSide, sideDim, (int)sideIndex, cellTopo);
    CellTools::setJacobian(jacobianSideRefCell, cubPointsSideRefCell, physicalCellNodes, cellTopo);
    
    // map side cubature points in reference parent cell domain to physical space
    FieldContainer<double> physCubPoints(numCells, numCubPoints, spaceDim);
    CellTools::mapToPhysicalFrame(physCubPoints, cubPointsSideRefCell, physicalCellNodes, cellTopo);
    
    FieldContainer<double> weightedMeasure(numCells, numCubPoints);
    FunctionSpaceTools::computeEdgeMeasure<double>(weightedMeasure, jacobianSideRefCell,
                                                   cubWeightsSide, sideIndex, cellTopo);
    
    FieldContainer<double> computedValues(numCells,numCubPoints);
    
    solutionValues(computedValues, elemTypePtr, trialID, physCubPoints, cubPointsSide, sideIndex);
    
    // weight computedValues for integration:
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numCubPoints; ptIndex++) {
        computedValues(cellIndex,ptIndex) *= weightedMeasure(cellIndex,ptIndex);
      }
    }
    
    // compute the integral
    int numPoints = computedValues.dimension(1);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        values(cellIndex) += computedValues(cellIndex,ptIndex);
      }
    }
  }
}

void Solution::solutionValues(FieldContainer<double> &values,
                              ElementTypePtr elemTypePtr,
                              int trialID,
                              const FieldContainer<double> &physicalPoints,
                              const FieldContainer<double> &sideRefCellPoints,
                              int sideIndex) {
  // currently, we only support computing solution values on all the cells of a given type at once.
  // values(numCellsForType,numPoints[,spaceDim (for vector-valued)])
  // physicalPoints(numCellsForType,numPoints,spaceDim)
  FieldContainer<double> solnCoeffs = solutionForElementTypeGlobal(elemTypePtr); // (numcells, numLocalTrialDofs)
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
  
  int numCells = physicalCellNodes.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr.get());
  int spaceDim = cellTopo.getDimension();
  
  typedef CellTools<double>  CellTools;
  typedef FunctionSpaceTools fst;
  
  //  cout << "physicalCellNodes: " << endl << physicalCellNodes;
  //  cout << "physicalPoints: " << endl << physicalPoints;
  //  cout << "refElemPoints: " << endl << refElemPoints;
  
  // Containers for Jacobian
  FieldContainer<double> cellJacobian(numCells, numPoints, spaceDim, spaceDim);
  FieldContainer<double> cellJacobInv(numCells, numPoints, spaceDim, spaceDim);
  FieldContainer<double> cellJacobDet(numCells, numPoints);
  
  Teuchos::RCP<DofOrdering> trialOrder = elemTypePtr->trialOrderPtr;
  
  BasisPtr basis = trialOrder->getBasis(trialID,sideIndex);
  
  int basisRank = trialOrder->getBasisRank(trialID);
  int basisCardinality = basis->getCardinality();
  
  TEUCHOS_TEST_FOR_EXCEPTION( ( basisRank==0 ) && values.rank() != 2,
                             std::invalid_argument,
                             "for scalar values, values container should be dimensioned(numCells,numPoints).");
  TEUCHOS_TEST_FOR_EXCEPTION( ( basisRank==1 ) && values.rank() != 3,
                             std::invalid_argument,
                             "for scalar values, values container should be dimensioned(numCells,numPoints,spaceDim).");
  TEUCHOS_TEST_FOR_EXCEPTION( values.dimension(0) != numCells,
                             std::invalid_argument,
                             "values.dimension(0) != numCells.");
  TEUCHOS_TEST_FOR_EXCEPTION( values.dimension(1) != numPoints,
                             std::invalid_argument,
                             "values.dimension(1) != numPoints.");
  TEUCHOS_TEST_FOR_EXCEPTION( basisRank==1 && values.dimension(2) != spaceDim,
                             std::invalid_argument,
                             "vector values.dimension(1) != spaceDim.");
  TEUCHOS_TEST_FOR_EXCEPTION( physicalPoints.rank() != 3,
                             std::invalid_argument,
                             "physicalPoints.rank() != 3.");
  TEUCHOS_TEST_FOR_EXCEPTION( physicalPoints.dimension(2) != spaceDim,
                             std::invalid_argument,
                             "physicalPoints.dimension(2) != spaceDim.");
  
  FieldContainer<double> thisCellJacobian(1,numPoints, spaceDim, spaceDim);
  FieldContainer<double> thisCellJacobInv(1,numPoints, spaceDim, spaceDim);
  FieldContainer<double> thisCellJacobDet(1,numPoints);
  FieldContainer<double> thisRefElemPoints(numPoints,spaceDim);
  
  shards::CellTopology side(cellTopo.getCellTopologyData(spaceDim-1,sideIndex)); // create relevant subcell (side) topology
  int sideDim = spaceDim-1;
  FieldContainer<double> cubPointsSideRefCell(numPoints, spaceDim); // cubPointsSide from the pov of the ref cell
  
  // compute geometric cell information
  //cout << "computing geometric cell info for boundary integral." << endl;
  CellTools::mapToReferenceSubcell(cubPointsSideRefCell, sideRefCellPoints, sideDim, (int)sideIndex, cellTopo);
  CellTools::setJacobian(cellJacobian, cubPointsSideRefCell, physicalCellNodes, cellTopo);
  CellTools::setJacobianDet(cellJacobDet, cellJacobian );
  CellTools::setJacobianInv(cellJacobInv, cellJacobian );
  
  values.initialize(0.0);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    thisCellJacobian.setValues(&cellJacobian(cellIndex,0,0,0),numPoints*spaceDim*spaceDim);
    thisCellJacobInv.setValues(&cellJacobInv(cellIndex,0,0,0),numPoints*spaceDim*spaceDim);
    thisCellJacobDet.setValues(&cellJacobDet(cellIndex,0),numPoints);
    Teuchos::RCP< FieldContainer<double> > transformedValues;
    transformedValues = BasisEvaluation::getTransformedValues(basis,  OP_VALUE,
                                                              sideRefCellPoints, thisCellJacobian,
                                                              thisCellJacobInv, thisCellJacobDet);
    
    //    cout << "cellIndex " << cellIndex << " thisRefElemPoints: " << thisRefElemPoints;
    //    cout << "cellIndex " << cellIndex << " transformedValues: " << *transformedValues;
    
    // now, apply coefficient weights:
    for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
      for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
        int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
        //        cout << "localDofIndex " << localDofIndex << " solnCoeffs(cellIndex,localDofIndex): " << solnCoeffs(cellIndex,localDofIndex) << endl;
        if (basisRank == 0) {
          values(cellIndex,ptIndex) += (*transformedValues)(0,dofOrdinal,ptIndex) * solnCoeffs(cellIndex,localDofIndex);
        } else {
          for (int i=0; i<spaceDim; i++) {
            values(cellIndex,ptIndex,i) += (*transformedValues)(0,dofOrdinal,ptIndex,i) * solnCoeffs(cellIndex,localDofIndex);
          }
        }
      }
      /*if (basisRank == 0) {
       cout << "solutionValue for point (" << physicalPoints(cellIndex,ptIndex,0) << ",";
       cout << physicalPoints(cellIndex,ptIndex,1) << "): " << values(cellIndex,ptIndex) << endl;
       } else {
       cout << "solutionValue for point (" << physicalPoints(cellIndex,ptIndex,0) << ",";
       cout << physicalPoints(cellIndex,ptIndex,1) << "): " << "(" << values(cellIndex,ptIndex,0);
       cout << "," << values(cellIndex,ptIndex,1) << ")" << endl;
       }*/
    }
  }
}
/*
 double Solution::totalRHSNorm(){
 vector< Teuchos::RCP< Element > > activeElements = _mesh->activeElements();
 vector< Teuchos::RCP< Element > >::iterator activeElemIt;
 
 map<int,double> rhsNormMap;
 rhsNorm(rhsNormMap);
 double totalNorm = 0.0;
 for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
 Teuchos::RCP< Element > current_element = *(activeElemIt);
 totalNorm += rhsNormMap[current_element->cellID()];
 }
 return totalNorm;
 }
 
 
 void Solution::rhsNorm(map<int,double> &rhsNormMap){
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
 
 int numActiveElements = _mesh->activeElements().size();
 
 computeErrorRepresentation();
 
 // initialize error array to -1 (cannot have negative index...)
 int localCellIDArray[numActiveElements];
 double localNormArray[numActiveElements];
 for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
 localCellIDArray[globalCellIndex] = -1;
 localNormArray[globalCellIndex] = -1.0;
 }
 
 vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
 vector<ElementTypePtr>::iterator elemTypeIt;
 int cellIndStart = 0;
 for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
 ElementTypePtr elemTypePtr = *(elemTypeIt);
 
 vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemTypePtr);
 
 FieldContainer<double> rhs = _rhsForElementType[elemTypePtr.get()];
 FieldContainer<double> rhsReps = _rhsRepresentationForElementType[elemTypePtr.get()];
 int numTestDofs = rhs.dimension(1);
 int numCells = rhs.dimension(0);
 TEUCHOS_TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "In rhsNorm::numCells does not match number of elems in partition.");
 
 for (int cellIndex=cellIndStart;cellIndex<numCells;cellIndex++){
 double normSquared = 0.0;
 for (int i=0; i<numTestDofs; i++) {
 normSquared += rhs(cellIndex,i) * rhsReps(cellIndex,i);
 }
 localNormArray[cellIndex] = sqrt(normSquared);
 int cellID = _mesh->cellID(elemTypePtr,cellIndex,rank);
 localCellIDArray[cellIndex] = cellID;
 }
 cellIndStart += numCells; // increment to go to the next set of element types
 } // end of loop thru element types
 
 // mpi communicate all energy norms
 double normArray[numProcs][numActiveElements];
 int cellIDArray[numProcs][numActiveElements];
 #ifdef HAVE_MPI
 if (numProcs>1){
 MPI::COMM_WORLD.Allgather(localNormArray,numActiveElements, MPI::DOUBLE, normArray, numActiveElements , MPI::DOUBLE);
 MPI::COMM_WORLD.Allgather(localCellIDArray,numActiveElements, MPI::INT, cellIDArray, numActiveElements , MPI::INT);
 }else{
 #else
 #endif
 for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
 cellIDArray[0][globalCellIndex] = localCellIDArray[globalCellIndex];
 normArray[0][globalCellIndex] = localNormArray[globalCellIndex];
 }
 #ifdef HAVE_MPI
 }
 #endif
 // copy back to rhsNorm map
 for (int procIndex=0;procIndex<numProcs;procIndex++){
 for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
 if (cellIDArray[procIndex][globalCellIndex]!=-1){
 rhsNormMap[cellIDArray[procIndex][globalCellIndex]] = normArray[procIndex][globalCellIndex];
 }
 }
 }
 }
 */

double Solution::energyErrorTotal() {
  double energyErrorSquared = 0.0;
  const map<GlobalIndexType,double>* energyErrorPerCell = &(energyError());
  
  for (map<GlobalIndexType,double>::const_iterator cellEnergyIt = energyErrorPerCell->begin();
       cellEnergyIt != energyErrorPerCell->end(); cellEnergyIt++) {
    energyErrorSquared += (cellEnergyIt->second) * (cellEnergyIt->second);
  }
  return sqrt(energyErrorSquared);
}

const map<GlobalIndexType,double> & Solution::energyError() {
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
  
  if ( _energyErrorComputed ) {
    /*
     if (rank==0){
     cout << "reusing energy error\n";
     }
     */
    return _energyErrorForCellIDGlobal;
  }
  
  /*
   // ready multivector for storage of energy errors
   cout << "Initializing multivectors/maps" << endl;
   Epetra_Map cellIDPartitionMap = _mesh->getCellIDPartitionMap(rank, &Comm); // TODO FIX - should be cellIndex not cellID
   cout << "Done initing maps" << endl;
   Epetra_MultiVector energyErrMV(cellIDPartitionMap,1);
   cout << "Done initing mvs" << endl;
   */
  int numActiveElements = _mesh->activeElements().size();
  //  energyError.resize( numActiveElements );
  
  //  vector< ElementPtr > elemsInPartition = _mesh->elementsInPartition(rank);
  //  int numElemsInPartition = elemsInPartition.size();
  
  computeErrorRepresentation();
  
  // initialize error array to -1 (cannot have negative index...)
  int localCellIDArray[numActiveElements];
  double localErrArray[numActiveElements];
  for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
    localCellIDArray[globalCellIndex] = -1;
    localErrArray[globalCellIndex] = -1.0;
  }
  
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  int globalCellIndex = 0;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    
    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemTypePtr);
    
    // for error rep v_e, residual res, energyError = sqrt ( ve_^T * res)
    FieldContainer<double> residuals = _residualForElementType[elemTypePtr.get()];
    FieldContainer<double> errorReps = _errorRepresentationForElementType[elemTypePtr.get()];
    int numTestDofs = residuals.dimension(1);
    int numCells = residuals.dimension(0);
    TEUCHOS_TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "In energyError::numCells does not match number of elems in partition.");
    
    for (int cellIndex=0;cellIndex<numCells;cellIndex++){
      double errorSquared = 0.0;
      for (int i=0; i<numTestDofs; i++) {
        errorSquared += residuals(cellIndex,i) * errorReps(cellIndex,i);
      }
      localErrArray[globalCellIndex] = sqrt(errorSquared);
      int cellID = _mesh->cellID(elemTypePtr,cellIndex,rank);
      localCellIDArray[globalCellIndex] = cellID;
      //      cout << "setting energy error = " << sqrt(errorSquared) << " for cellID " << cellID << endl;
      globalCellIndex++;
    }
  } // end of loop thru element types
  
  // mpi communicate all energy errors
  double errArray[numProcs][numActiveElements];
  int cellIDArray[numProcs][numActiveElements];
#ifdef HAVE_MPI
  if (numProcs>1){
    //    cout << "sending MPI call for inds on proc " << rank << endl;
    MPI::COMM_WORLD.Allgather(localErrArray,numActiveElements, MPI::DOUBLE, errArray, numActiveElements , MPI::DOUBLE);
    MPI::COMM_WORLD.Allgather(localCellIDArray,numActiveElements, MPI::INT, cellIDArray, numActiveElements , MPI::INT);
    //    cout << "done sending MPI call" << endl;
  }else{
#else
#endif
    for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
      cellIDArray[0][globalCellIndex] = localCellIDArray[globalCellIndex];
      errArray[0][globalCellIndex] = localErrArray[globalCellIndex];
    }
#ifdef HAVE_MPI
  }
#endif
  // copy back to energyError container
  for (int procIndex=0;procIndex<numProcs;procIndex++){
    for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
      if (cellIDArray[procIndex][globalCellIndex]!=-1){
        _energyErrorForCellIDGlobal[cellIDArray[procIndex][globalCellIndex]] = errArray[procIndex][globalCellIndex];
      }
    }
  }
  _energyErrorComputed = true;
  /*
   if (rank==0){
   for (map<int,double>::iterator mapIt=_energyErrorForCellIDGlobal.begin();mapIt!=_energyErrorForCellIDGlobal.end();mapIt++){
   cout << "Energy error for cellID " << mapIt->first << " is " << mapIt->second << endl;
   }
   }
   */
  
  return _energyErrorForCellIDGlobal;
}

void Solution::computeErrorRepresentation() {
  int rank= Teuchos::GlobalMPISession::getRank();

  if (!_residualsComputed) {
    computeResiduals();
  }
  //  vector< ElementPtr > elemsInPartition = _mesh->elementsInPartition(rank);
  
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh,true,_cubatureEnrichmentDegree));
    
    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    
    vector< ElementPtr > elements = _mesh->elementsOfType(rank, elemTypePtr);
    
    int numCells = elements.size();
    int numTestDofs = testOrdering->totalDofs();

    Teuchos::Array<int> localRHSDim(2);
    localRHSDim[0] = _residualForElementType[elemTypePtr.get()].dimension(1);
    localRHSDim[1] = 1;
    
    FieldContainer<double> representationMatrix(numTestDofs, 1);
    FieldContainer<double> errorRepresentation(numCells,numTestDofs);
    
    vector<GlobalIndexType> cellIDVector(1);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
      GlobalIndexType cellID = _mesh->cellID(elemTypePtr, cellOrdinal, rank);
      cellIDVector[0] = cellID;
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
      ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDVector,_ip->hasBoundaryTerms());
      FieldContainer<double> ipMatrix(1,numTestDofs,numTestDofs);
      _ip->computeInnerProductMatrix(ipMatrix,testOrdering, ipBasisCache);
      FieldContainer<double> rhsMatrix(localRHSDim, &_residualForElementType[elemTypePtr.get()](cellOrdinal,0));
      // strip cell dimension:
      ipMatrix.resize(ipMatrix.dimension(1),ipMatrix.dimension(2));
      int result = SerialDenseWrapper::solveSystemUsingQR(representationMatrix, ipMatrix, rhsMatrix);
      if (result != 0) {
        cout << "WARNING: computeErrorRepresentation: call to solveSystemUsingQR failed with error code " << result << endl;
      }
      for (int i=0; i<numTestDofs; i++) {
        errorRepresentation(cellOrdinal,i) = representationMatrix(i,0);
      }
    }
    _errorRepresentationForElementType[elemTypePtr.get()] = errorRepresentation;
  }
}

void Solution::computeResiduals() {
  int rank=0;
  
#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  //  vector< ElementPtr > elemsInPartition = _mesh->elementsInPartition(rank);
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    
    Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;
    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemTypePtr);
    
    int numCells = elemsInPartitionOfType.size();
    int numTrialDofs = trialOrdering->totalDofs();
    int numTestDofs  = testOrdering->totalDofs();
    
    // compute l(v) and store in residuals:
    FieldContainer<double> residuals(numCells,numTestDofs);
    
    FieldContainer<double> rhs(numCells,numTestDofs);
    rhs = residuals; // copy rhs into its own separate container
    
    Teuchos::Array<int> oneCellDim(2);
    oneCellDim[0] = 1;
    oneCellDim[1] = numTestDofs;
    
    FieldContainer<double> solution = solutionForElementTypeGlobal(elemTypePtr);
    
    vector<GlobalIndexType> cellIDVector(1);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++) {
      GlobalIndexType cellID = _mesh->cellID(elemTypePtr, cellOrdinal, rank);
      cellIDVector[0] = cellID;
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
      FieldContainer<double> cellSideParities = _mesh->cellSideParitiesForCell(cellID);
      
      BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh,false,_cubatureEnrichmentDegree));
      bool createSideCacheToo = true;
      basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDVector,createSideCacheToo);
      FieldContainer<double> thisCellResidual(oneCellDim, &residuals(cellOrdinal,0));
      _rhs->integrateAgainstStandardBasis(thisCellResidual, testOrdering, basisCache);
      for (int i=0; i<numTestDofs; i++) {
        rhs(cellOrdinal,i) = thisCellResidual(0,i);
      }
      
      // compute b(u, v):
      FieldContainer<double> preStiffness(1,numTestDofs,numTrialDofs );
      _mesh->bilinearForm()->stiffnessMatrix(preStiffness, elemTypePtr, cellSideParities, basisCache);

      int globalCellIndex = elemsInPartitionOfType[cellOrdinal]->globalCellIndex();
      for (int i=0; i<numTestDofs; i++) {
        for (int j=0; j<numTrialDofs; j++) {
          residuals(cellOrdinal,i) -= solution(globalCellIndex,j) * preStiffness(0,i,j);
        }
      }
    }
    _residualForElementType[elemTypePtr.get()] = residuals;
    _rhsForElementType[elemTypePtr.get()] = rhs;
  }
  _residualsComputed = true;
}

void Solution::discardInactiveCellCoefficients() {
  set< GlobalIndexType > activeCellIDs = _mesh->getActiveCellIDs();
  vector<GlobalIndexType> cellIDsToErase;
  for (map< GlobalIndexType, FieldContainer<double> >::iterator solnIt = _solutionForCellIDGlobal.begin();
       solnIt != _solutionForCellIDGlobal.end(); solnIt++) {
    GlobalIndexType cellID = solnIt->first;
    if ( activeCellIDs.find(cellID) == activeCellIDs.end() ) {
      cellIDsToErase.push_back(cellID);
    }
  }
  for (vector<GlobalIndexType>::iterator it = cellIDsToErase.begin();it !=cellIDsToErase.end();it++){
    _solutionForCellIDGlobal.erase(*it);
  }
}

Teuchos::RCP<Epetra_FEVector> Solution::getRHSVector() {
  return _rhsVector;
}

Teuchos::RCP<Epetra_FECrsMatrix> Solution::getStiffnessMatrix() {
  return _globalStiffMatrix;
}

void Solution::solutionValues(FieldContainer<double> &values, int trialID, BasisCachePtr basisCache,
                              bool weightForCubature, EOperatorExtended op) {
  values.initialize(0.0);
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  int sideIndex = basisCache->getSideIndex();
  bool forceVolumeCoords = false; // used for evaluating fields on sides...
  if ( ( sideIndex != -1 ) && !_mesh->bilinearForm()->isFluxOrTrace(trialID)) {
    forceVolumeCoords = true;
    //    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    //                       "solutionValues doesn't support evaluation of field variables along sides (not yet anyway).");
  }
  if ( (sideIndex == -1 ) && _mesh->bilinearForm()->isFluxOrTrace(trialID) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "solutionValues doesn't support evaluation of trace or flux variables on element interiors.");
  }
  bool fluxOrTrace = _mesh->bilinearForm()->isFluxOrTrace(trialID);
  
  int numCells = cellIDs.size();
  if (numCells != values.dimension(0)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "first dimension of values should == numCells.");
  }
  int spaceDim = basisCache->getSpaceDim();
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  for (int cellIndex = 0; cellIndex < numCells; cellIndex++) {
    GlobalIndexType cellID = cellIDs[cellIndex];
    

    if ( _solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
      // cellID not known -- default values for that cell to 0
//      int rank = Teuchos::GlobalMPISession::getRank();
//      cout << "In Solution::solutionValues() on rank " << rank << ", data for cellID " << cellID << " not found; defaulting to 0.\n" ;
      continue;
    } else {
      int rank = Teuchos::GlobalMPISession::getRank();
//      cout << "In Solution::solutionValues() on rank " << rank << ", data for cellID " << cellID << " found; container size is " << _solutionForCellIDGlobal[cellID].size() << endl;
    }
    
    FieldContainer<double>* solnCoeffs = &_solutionForCellIDGlobal[cellID];
    
    DofOrderingPtr trialOrder = _mesh->getElement(cellID)->elementType()->trialOrderPtr;
    
    BasisPtr basis = fluxOrTrace ? trialOrder->getBasis(trialID, sideIndex)
    : trialOrder->getBasis(trialID);
    int basisCardinality = basis->getCardinality();
    
    Teuchos::RCP<const FieldContainer<double> > transformedValues;
    if (weightForCubature) {
      if (forceVolumeCoords) {
        transformedValues = basisCache->getVolumeBasisCache()->getTransformedWeightedValues(basis,op,sideIndex,true);
      } else {
        transformedValues = basisCache->getTransformedWeightedValues(basis, op);
      }
    } else {
      if (forceVolumeCoords) {
        transformedValues = basisCache->getVolumeBasisCache()->getTransformedValues(basis, op, sideIndex, true);
      } else {
        transformedValues = basisCache->getTransformedValues(basis, op);
      }
    }
    
//    cout << "solnCoeffs:\n" << *solnCoeffs;
    
    const vector<int> *dofIndices = fluxOrTrace ? &(trialOrder->getDofIndices(trialID,sideIndex))
    : &(trialOrder->getDofIndices(trialID));
    
    int rank = transformedValues->rank() - 3; // 3 ==> scalar valued, 4 ==> vector, etc.
    
    // now, apply coefficient weights:
    for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
      int localDofIndex = (*dofIndices)[dofOrdinal];
      //      cout << "localDofIndex " << localDofIndex << " solnCoeffs(localDofIndex): " << solnCoeffs(localDofIndex) << endl;
      for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
        if (rank == 0) {
          values(cellIndex,ptIndex) += (*transformedValues)(cellIndex,dofOrdinal,ptIndex) * (*solnCoeffs)(localDofIndex);
        } else if (rank == 1) {
          for (int i=0; i<spaceDim; i++) {
            values(cellIndex,ptIndex,i) += (*transformedValues)(cellIndex,dofOrdinal,ptIndex,i) * (*solnCoeffs)(localDofIndex);
          }
        } else if (rank == 2) {
          for (int i=0; i<spaceDim; i++) {
            for (int j=0; j<spaceDim; j++) {
              values(cellIndex,ptIndex,i,j) += (*transformedValues)(cellIndex,dofOrdinal,ptIndex,i,j) * (*solnCoeffs)(localDofIndex);
            }
          }
        } else {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "solutionValues doesn't support values with rank > 2.");
        }
      }
    }
  }
}

void Solution::solutionValues(FieldContainer<double> &values, int trialID, const FieldContainer<double> &physicalPoints) {
  // physicalPoints may have dimensions (C,P,D) or (P,D)
  // either way, this method requires searching the mesh for the points provided
  if (physicalPoints.rank()==3) { // dimensions (C,P,D)
    int numTotalCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    int spaceDim = physicalPoints.dimension(2);
    if (values.dimension(0) != numTotalCells) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values.dimension(0) != physicalPoints.dimension(0)");
    }
    if (values.dimension(1) != numPoints) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "values.dimension(1) != physicalPoints.dimension(1)");
    }
    
    values.initialize(0.0);
    for (int cellIndex=0;cellIndex<numTotalCells;cellIndex++){
      
      FieldContainer<double> cellPoint(1,spaceDim); // a single point to find elem we're in
      for (int i=0;i<spaceDim;i++){cellPoint(0,i) = physicalPoints(cellIndex,0,i);}
      vector< ElementPtr > elements = _mesh->elementsForPoints(cellPoint); // operate under assumption that all points for a given cell index are in that cell
      ElementPtr elem = elements[0];
      if (elem.get() == NULL) continue;
      ElementTypePtr elemTypePtr = elem->elementType();
      int cellID = elem->cellID();
      
      FieldContainer<double> solnCoeffs = allCoefficientsForCellID(cellID);
      if (solnCoeffs.size()==0) continue; // cell ID not known: default to zero
      int numCells = 1; // do one cell at a time
      
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
      
      // store points in local container
      FieldContainer<double> physicalPointsForCell(numCells,numPoints,spaceDim);
      for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
        for (int dim=0; dim<spaceDim; dim++) {
          physicalPointsForCell(0,ptIndex,dim) = physicalPoints(cellIndex,ptIndex,dim);
        }
      }
      
      // 1. compute refElemPoints, the evaluation points mapped to reference cell:
      FieldContainer<double> refElemPoints(numCells,numPoints, spaceDim);
      CellTools<double>::mapToReferenceFrame(refElemPoints,physicalPointsForCell,physicalCellNodes,*(elemTypePtr->cellTopoPtr.get()));
      refElemPoints.resize(numPoints,spaceDim);
      
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
      basisCache->setRefCellPoints(refElemPoints);
      Teuchos::Array<int> dim;
      values.dimensions(dim);
      dim[0] = 1; // one cell
      Teuchos::Array<int> cellOffset = dim;
      cellOffset[0] = cellIndex;
      for (int containerRank=1; containerRank<cellOffset.size(); containerRank++) {
        cellOffset[containerRank] = 0;
      }
      FieldContainer<double> cellValues(dim,&values[values.getEnumeration(cellOffset)]);
      this->solutionValues(cellValues, trialID, basisCache);
    }
    //  when the cell containing the point is off-rank, we have 0s.
    // We sum entrywise to get the missing values.
    MPIWrapper::entryWiseSum(values);
  } else { // (P,D) physicalPoints
    // the following is due to the fact that we *do not* transform basis values.
    IntrepidExtendedTypes::EFunctionSpaceExtended fs = _mesh->bilinearForm()->functionSpaceForTrial(trialID);
    TEUCHOS_TEST_FOR_EXCEPTION( (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HVOL) && (fs != IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD),
                               std::invalid_argument,
                               "This version of solutionValues only supports HVOL and HGRAD bases.");
    
    TEUCHOS_TEST_FOR_EXCEPTION( values.dimension(0) != physicalPoints.dimension(0),
                               std::invalid_argument,
                               "values.dimension(0) != physicalPoints.dimension(0).");
    
    // physicalPoints dimensions: (P,D)
    // values dimensions: (P) or (P,D)
    //int numPoints = physicalPoints.dimension(0);
    int spaceDim = physicalPoints.dimension(1);
    int valueRank = values.rank();
    Teuchos::Array<int> oneValueDimensions;
    oneValueDimensions.push_back(1);
    Teuchos::Array<int> onePointDimensions;
    onePointDimensions.push_back(1); // C (cell)
    onePointDimensions.push_back(1); // P (point)
    onePointDimensions.push_back(spaceDim); // D (space)
    if (valueRank >= 1) oneValueDimensions.push_back(spaceDim);
    FieldContainer<double> oneValue(oneValueDimensions);
    Teuchos::Array<int> oneCellDofsDimensions;
    oneCellDofsDimensions.push_back(0); // initialize according to elementType
    bool nullElementsOffRank = true;
    vector< ElementPtr > elements = _mesh->elementsForPoints(physicalPoints, nullElementsOffRank);
    vector< ElementPtr >::iterator elemIt;
    int physicalPointIndex = -1;
    values.initialize(0.0);
    for (elemIt = elements.begin(); elemIt != elements.end(); elemIt++) {
      physicalPointIndex++;
      ElementPtr elem = *elemIt;
      if (elem.get() == NULL) {
        // values for this point will already have been initialized to 0, the best we can do...
        continue;
      }
      ElementTypePtr elemTypePtr = elem->elementType();
      
      int cellID = elem->cellID();
      FieldContainer<double> solnCoeffs = allCoefficientsForCellID(cellID);
      if (solnCoeffs.size()==0) continue; // cell ID not known: default to zero
      
      int numCells = 1;
      int numPoints = 1;
      
      FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
      
      FieldContainer<double> physicalPoint(onePointDimensions);
      for (int dim=0; dim<spaceDim; dim++) {
        physicalPoint[dim] = physicalPoints(physicalPointIndex,dim);
      }
      
      // 1. Map the physicalPoints from the element specified in physicalCellNodes into reference element
      // 2. Compute each basis on those points
      // 3. Transform those basis evaluations back into the physical space
      // 4. Multiply by the solnCoeffs
      
      typedef CellTools<double>  CellTools;
      typedef FunctionSpaceTools fst;
      
      // 1. compute refElemPoints, the evaluation points mapped to reference cell:
      FieldContainer<double> refElemPoint(numCells, numPoints, spaceDim);
      shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr.get());
      CellTools::mapToReferenceFrame(refElemPoint,physicalPoint,physicalCellNodes,cellTopo);
      refElemPoint.resize(numPoints,spaceDim);

      Teuchos::RCP<DofOrdering> trialOrder = elemTypePtr->trialOrderPtr;
      int basisRank = trialOrder->getBasisRank(trialID);
      
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(_mesh, cellID);
      basisCache->setRefCellPoints(refElemPoint);
      Teuchos::Array<int> dim;
      dim.push_back(1); // one cell
      dim.push_back(1); // one point
      
      for (int rank=0; rank<basisRank; rank++) {
        dim.push_back(spaceDim);
      }
      
      Teuchos::Array<int> cellOffset = dim;
      FieldContainer<double> cellValues(dim);
      this->solutionValues(cellValues, trialID, basisCache);

      if (basisRank == 0) {
        values(physicalPointIndex) = cellValues(0,0);
      } else if (basisRank == 1) {
        for (int d=0; d<spaceDim; d++) {
          values(physicalPointIndex,d) = cellValues(0,0,d);
        }
      } else if (basisRank == 2) {
        for (int d0=0; d0<spaceDim; d0++) {
          for (int d1=0; d1<spaceDim; d1++) {
            values(physicalPointIndex,d0,d1) = cellValues(0,0,d0,d1);
          }
        }
      } else {
        cout << "unhandled basis rank.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unhandled basis rank.");
      }
    }
    // for the (P,D) version of this method, when the cell containing the point is off-rank, we have 0s.
    // We sum entrywise to get the missing values.
    MPIWrapper::entryWiseSum(values);
  } // end (P,D)
}

void determineQuadEdgeWeights(double weights[], int edgeVertexNumber, int numDivisionsPerEdge, bool xEdge) {
  if (xEdge) {
    weights[0] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
    weights[1] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
    weights[2] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
    weights[3] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
  } else {
    weights[0] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
    weights[1] = ((double)(numDivisionsPerEdge - edgeVertexNumber)) / (double)numDivisionsPerEdge;
    weights[2] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
    weights[3] = ((double)edgeVertexNumber) / (double)numDivisionsPerEdge;
  }
}

void Solution::writeStatsToFile(const string &filePath, int precision) {
  // writes out rows of the format: "cellID patchID x y solnValue"
  ofstream fout(filePath.c_str());
  fout << setprecision(precision);
  fout << "stat.\tmean\tmin\tmax\ttotal\n";
  fout << "localStiffness\t" << _meanTimeLocalStiffness << "\t" <<_minTimeLocalStiffness << "\t" <<_maxTimeLocalStiffness << "\t" << _totalTimeLocalStiffness << endl;
  fout << "globalAssembly\t" <<  _meanTimeGlobalAssembly << "\t" <<_minTimeGlobalAssembly << "\t" <<_maxTimeGlobalAssembly << "\t" << _totalTimeGlobalAssembly << endl;
  fout << "impose BCs\t" <<  _meanTimeBCImposition << "\t" <<_minTimeBCImposition << "\t" <<_maxTimeBCImposition << "\t" << _totalTimeBCImposition << endl;
  fout << "solve\t" << _meanTimeSolve << "\t" <<_minTimeSolve << "\t" <<_maxTimeSolve << "\t" << _totalTimeSolve << endl;
  fout << "dist. solution\t" <<  _meanTimeDistributeSolution << "\t" << _minTimeDistributeSolution << "\t" <<_maxTimeDistributeSolution << "\t" << _totalTimeDistributeSolution << endl;
}

void Solution::writeToFile(int trialID, const string &filePath) {
  // writes out rows of the format: "cellID patchID x y solnValue"
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...
  
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    
    CellTopoPtrLegacy cellTopo = elemTypePtr->cellTopoPtr;
    FieldContainer<double> vertexPoints(cellTopo->getVertexCount(),cellTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(vertexPoints, *cellTopo);
    
    int numVertices = vertexPoints.dimension(1);
    
    int numDivisionsPerEdge = 1; //basisDegree*basisDegree;
    int numPatchesPerCell = numDivisionsPerEdge*numDivisionsPerEdge;
    
    FieldContainer<double> refPoints(numPatchesPerCell*numVertices,spaceDim);
    
    if (numVertices == 4) { // only quads supported by the multi-patch cell stuff below...
      //if (   ( elemTypePtr->cellTopoPtr->getKey() == shards::Quadrilateral<4>::key )
      //    || (elemTypePtr->cellTopoPtr->getKey() == shards::Triangle<3>::key ) ) {
      
      FieldContainer<double> iVertex(spaceDim), jVertex(spaceDim);
      FieldContainer<double> v1(spaceDim), v2(spaceDim), v3(spaceDim);
      
      if (numVertices == 4) {
        double yWeights[numVertices], xWeights[numVertices];
        for (int i=0; i<numDivisionsPerEdge; i++) {
          for (int j=0; j<numDivisionsPerEdge; j++) {
            //cout << "weights: " << xWeights[0]*yWeights[0] << " " << xWeights[1]*yWeights[1] << " " << xWeights[2]*yWeights[2] << " " << xWeights[3]*yWeights[3] << "\n";
            int patchIndex = (i*numDivisionsPerEdge + j);
            refPoints.initialize(0.0);
            for (int patchVertexIndex=0; patchVertexIndex < numVertices; patchVertexIndex++) {
              int xOffset = ((patchVertexIndex==0) || (patchVertexIndex==3)) ? 0 : 1;
              int yOffset = ((patchVertexIndex==0) || (patchVertexIndex==1)) ? 0 : 1;
              determineQuadEdgeWeights(xWeights,i+xOffset,numDivisionsPerEdge,true);
              determineQuadEdgeWeights(yWeights,j+yOffset,numDivisionsPerEdge,false);
              
              for (int vertexIndex=0; vertexIndex < numVertices; vertexIndex++) {
                double weight = xWeights[vertexIndex] * yWeights[vertexIndex];
                for (int dim=0; dim<spaceDim; dim++) {
                  refPoints(patchIndex*numVertices + patchVertexIndex, dim) += weight*vertexPoints(vertexIndex, dim);
                }
              }
            }
          }
        }
      }
      
    } else {
      refPoints = vertexPoints;
      numPatchesPerCell = 1;
    }
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(_mesh, elemTypePtr);
    
    basisCache->setPhysicalCellNodes(_mesh->physicalCellNodesGlobal(elemTypePtr), _mesh->cellIDsOfType(elemTypePtr), false);
    basisCache->setRefCellPoints(refPoints);
    int numCells = basisCache->cellIDs().size();
    
    FieldContainer<double> values(numCells, numPatchesPerCell * numVertices);
    this->solutionValues(values, trialID, basisCache);
    
    FieldContainer<double> physPoints = basisCache->getPhysicalCubaturePoints();
    
    for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
      for (int patchIndex=0; patchIndex < numPatchesPerCell; patchIndex++) {
        for (int ptIndex=0; ptIndex< numVertices; ptIndex++) {
          double x = physPoints(cellIndex,patchIndex*numVertices + ptIndex,0);
          double y = physPoints(cellIndex,patchIndex*numVertices + ptIndex,1);
          double z = values(cellIndex,patchIndex*numVertices + ptIndex);
          fout << _mesh->cellID(elemTypePtr,cellIndex) << " " << patchIndex << " " << x << " " << y << " " << z << endl;
        }
      }
    }
  }
  fout.close();
}

void Solution::writeQuadSolutionToFile(int trialID, const string &filePath) {
  // writes out rows of the format: "cellID xIndex yIndex x y solnValue"
  // it's a goofy thing, largely because the MATLAB routine we're using
  // wants a cartesian product with every combo (x_i, y_j) somewhere in the mix...
  // The upshot is that the following will work reasonably only so long as our element
  // boundaries are in a nice Cartesian grid.  But that's more a problem for MATLAB
  // than it is for us...
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...
  
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    int basisDegree = elemTypePtr->trialOrderPtr->getBasis(trialID)->getDegree();
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    // 0. Set up Cubature
    // Get numerical integration points--these will be the points we compute the solution values for...
    DefaultCubatureFactory<double>  cubFactory;
    int cubDegree = 2*basisDegree;
    Teuchos::RCP<Cubature<double> > cellTopoCub = cubFactory.create(*(cellTopoPtr.get()), cubDegree);
    
    int cubDim       = cellTopoCub->getDimension();
    int numCubPoints = cellTopoCub->getNumPoints();
    
    FieldContainer<double> cubPoints(numCubPoints, cubDim);
    FieldContainer<double> cubWeights(numCubPoints);
    
    cellTopoCub->getCubature(cubPoints, cubWeights);
    
    // here's a hackish bit: collect all the x and y coordinates, and add the vertices
    set<double> xCoords, yCoords;
    for (int ptIndex=0; ptIndex<numCubPoints; ptIndex++) {
      xCoords.insert(cubPoints(ptIndex,0));
      yCoords.insert(cubPoints(ptIndex,1));
    }
    
    // add vertices (for ref quad)
    xCoords.insert(-1.0);
    xCoords.insert(1.0);
    yCoords.insert(-1.0);
    yCoords.insert(1.0);
    
    // now, make ourselves a new set of "cubature" points:
    numCubPoints = xCoords.size() * yCoords.size();
    cubPoints.resize(numCubPoints,cubDim);
    int ptIndex = 0;
    set<double>::iterator xIt, yIt;
    for (xIt = xCoords.begin(); xIt != xCoords.end(); xIt++) {
      for (yIt = yCoords.begin(); yIt != yCoords.end(); yIt++) {
        cubPoints(ptIndex,0) = *xIt;
        cubPoints(ptIndex,1) = *yIt;
        ptIndex++;
      }
    }
    typedef CellTools<double>  CellTools;
    
    BasisCachePtr basisCache = BasisCache::basisCacheForCellType(_mesh, elemTypePtr);
    
    basisCache->setPhysicalCellNodes(_mesh->physicalCellNodesGlobal(elemTypePtr), _mesh->cellIDsOfType(elemTypePtr), false);
    basisCache->setRefCellPoints(cubPoints);
    
    FieldContainer<double> physCubPoints = basisCache->getPhysicalCubaturePoints();
    
    FieldContainer<double> values(numCellsOfType, numCubPoints);
    solutionValues(values,trialID,basisCache);
    
    map<float,int> xIndices;
    map<float,int> yIndices; // use floats to truncate insignificant digits...
    for (int cellIndex=0; cellIndex < numCellsOfType; cellIndex++) {
      xIndices.clear();
      yIndices.clear();
      for (int ptIndex=0; ptIndex< numCubPoints; ptIndex++) {
        double x = physCubPoints(cellIndex,ptIndex,0);
        double y = physCubPoints(cellIndex,ptIndex,1);
        double z = values(cellIndex,ptIndex);
        if (xIndices.find(x) == xIndices.end()) {
          int xIndex = xIndices.size();
          xIndices[x] = xIndex;
        }
        if (yIndices.find(y) == yIndices.end()) {
          int yIndex = yIndices.size();
          yIndices[y] = yIndex;
        }
        fout << _mesh->cellID(elemTypePtr,cellIndex) << " " << xIndices[x] << " " << yIndices[y] << " " << x << " " << y << " " << z << endl;
      }
    }
  }
  fout.close();
}

FieldContainer<double> Solution::solutionForElementTypeGlobal(ElementTypePtr elemType) {
  vector< ElementPtr > elementsOfType = _mesh->elementsOfTypeGlobal(elemType);
  int numDofsForType = elemType->trialOrderPtr->totalDofs();
  int numCellsOfType = elementsOfType.size();
  FieldContainer<double> solutionCoeffs(numCellsOfType,numDofsForType);
  for (vector< ElementPtr >::iterator elemIt = elementsOfType.begin();
       elemIt != elementsOfType.end(); elemIt++) {
    int globalCellIndex = (*elemIt)->globalCellIndex();
    int cellID = (*elemIt)->cellID();
    for (int dofIndex=0; dofIndex<numDofsForType; dofIndex++) {
      if (( _solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end())
          && (_solutionForCellIDGlobal[cellID].size() == numDofsForType)) {
        solutionCoeffs(globalCellIndex,dofIndex) = _solutionForCellIDGlobal[cellID](dofIndex);
      } else { // no solution set for that cellID, return 0
        solutionCoeffs(globalCellIndex,dofIndex) = 0.0;
      }
    }
  }
  return solutionCoeffs;
}

// static method interprets a set of trial ordering coefficients in terms of a specified DofOrdering
// and returns a set of weights for the appropriate basis
void Solution::basisCoeffsForTrialOrder(FieldContainer<double> &basisCoeffs, DofOrderingPtr trialOrder,
                                        const FieldContainer<double> &allCoeffs,
                                        int trialID, int sideIndex) {
  BasisPtr basis = trialOrder->getBasis(trialID,sideIndex);
  
  int basisCardinality = basis->getCardinality();
  basisCoeffs.resize(basisCardinality);
  
  for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
    int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
    basisCoeffs(dofOrdinal) = allCoeffs(localDofIndex);
  }
}

void Solution::solnCoeffsForCellID(FieldContainer<double> &solnCoeffs, GlobalIndexType cellID, int trialID, int sideIndex) {
  Teuchos::RCP< DofOrdering > trialOrder = _mesh->getElement(cellID)->elementType()->trialOrderPtr;
  
  if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
    cout << "Warning: solution for cellID " << cellID << " not found; returning 0.\n";
    BasisPtr basis = trialOrder->getBasis(trialID,sideIndex);
    int basisCardinality = basis->getCardinality();
    solnCoeffs.resize(basisCardinality);
    solnCoeffs.initialize();
    return;
  }
  
  basisCoeffsForTrialOrder(solnCoeffs, trialOrder, _solutionForCellIDGlobal[cellID], trialID, sideIndex);
}

const FieldContainer<double>& Solution::allCoefficientsForCellID(GlobalIndexType cellID, bool warnAboutOffRankImports) {
  int myRank                    = Teuchos::GlobalMPISession::getRank();
  PartitionIndexType cellRank   = _mesh->globalDofAssignment()->partitionForCellID(cellID);

  bool cellIsRankLocal = (cellRank == myRank);
  if (cellIsRankLocal) {
    return _solutionForCellIDGlobal[cellID];
  } else {
    if ((warnAboutOffRankImports) && (cellRank != -1)) { // we don't warn about cells that don't have ranks (can happen on refinement, say)
      cout << "Warning: allCoefficientsForCellID() called on rank " << myRank << " for non-rank-local cell " << cellID;
      cout << ", which belongs to rank " << cellRank << endl;
    }
    return _solutionForCellIDGlobal[cellID];
  }
}

void Solution::setBC( Teuchos::RCP<BC> bc) {
  _bc = bc;
}

void Solution::setFilter(Teuchos::RCP<LocalStiffnessMatrixFilter> newFilter) {
  _filter = newFilter;
}

void Solution::setIP( Teuchos::RCP<DPGInnerProduct> ip) {
  _ip = ip;
  // any computed residuals will need to be recomputed with the new IP
  clearComputedResiduals();
}

void Solution::setLagrangeConstraints( Teuchos::RCP<LagrangeConstraints> lagrangeConstraints) {
  _lagrangeConstraints = lagrangeConstraints;
}

void Solution::setReportConditionNumber(bool value) {
  _reportConditionNumber = value;
}

void Solution::setReportTimingResults(bool value) {
  _reportTimingResults = value;
}

void Solution::setRHS( Teuchos::RCP<RHS> rhs) {
  _rhs = rhs;
  clearComputedResiduals();
}

void Solution::setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, GlobalIndexType cellID){
  _solutionForCellIDGlobal[cellID] = solnCoeffsToSet;
  _mesh->globalDofAssignment()->interpretLocalCoefficients(cellID,solnCoeffsToSet,*_lhsVector);
}

void Solution::setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, GlobalIndexType cellID, int trialID, int sideIndex) {
  ElementTypePtr elemTypePtr = _mesh->getElement(cellID)->elementType();
  
  Teuchos::RCP< DofOrdering > trialOrder = elemTypePtr->trialOrderPtr;
  BasisPtr basis = trialOrder->getBasis(trialID,sideIndex);
  
  int basisCardinality = basis->getCardinality();
  if ( _solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
    // allocate new storage
    _solutionForCellIDGlobal[cellID] = FieldContainer<double>(trialOrder->totalDofs());
  }
  if (_solutionForCellIDGlobal[cellID].size() != trialOrder->totalDofs()) {
    // resize
    _solutionForCellIDGlobal[cellID].resize(trialOrder->totalDofs());
  }
  TEUCHOS_TEST_FOR_EXCEPTION(solnCoeffsToSet.size() != basisCardinality, std::invalid_argument, "solnCoeffsToSet.size() != basisCardinality");
  for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
    int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
    _solutionForCellIDGlobal[cellID](localDofIndex) = solnCoeffsToSet[dofOrdinal];
  }
  FieldContainer<double> globalCoefficients;
  FieldContainer<GlobalIndexType> globalDofIndices;
  _mesh->interpretLocalBasisCoefficients(cellID, trialID, sideIndex, solnCoeffsToSet, globalCoefficients, globalDofIndices);
  
  for (int i=0; i<globalCoefficients.size(); i++) {
    _lhsVector->ReplaceGlobalValue((GlobalIndexTypeToCast)globalDofIndices[i], 0, globalCoefficients[i]);
  }
  
  // could stand to be more granular, maybe, but if we're changing the solution, the present
  // policy is to invalidate any computed residuals
  clearComputedResiduals();
}

// protected method; used for solution comparison...
const map< GlobalIndexType, FieldContainer<double> > & Solution::solutionForCellIDGlobal() const {
  return _solutionForCellIDGlobal;
}

void Solution::setWriteMatrixToFile(bool value, const string &filePath) {
  _writeMatrixToMatlabFile = value;
  _matrixFilePath = filePath;
}

void Solution::setWriteMatrixToMatrixMarketFile(bool value, const string &filePath) {
  _writeMatrixToMatrixMarketFile = value;
  _matrixFilePath = filePath;
}

void Solution::setWriteRHSToMatrixMarketFile(bool value, const string &filePath) {
  _writeRHSToMatrixMarketFile = value;
  _rhsFilePath = filePath;
}

void Solution::condensedSolve(Teuchos::RCP<Solver> globalSolver, bool reduceMemoryFootprint) {
  // when reduceMemoryFootprint is true, local stiffness matrices will be computed twice, rather than stored for reuse
  vector<int> trialIDs = _mesh->bilinearForm()->trialIDs();
  
  set< int > fieldsToExclude;
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (_bc->imposeZeroMeanConstraint(trialID) || _bc->singlePointBC(trialID) ) {
      fieldsToExclude.insert(trialID);
    }
  }
  
  // override reduceMemoryFootprint for now (since CondensedDofInterpreter doesn't yet support a true value)
  reduceMemoryFootprint = false;
  
  CondensedDofInterpreter dofInterpreter(_mesh.get(), _lagrangeConstraints.get(), fieldsToExclude, !reduceMemoryFootprint);
  
  DofInterpreter* oldDofInterpreter = _dofInterpreter;
  
  _dofInterpreter = &dofInterpreter;
  _mesh->boundary().setDofInterpreter(_dofInterpreter);
  
  solve(globalSolver);
  
  _dofInterpreter = oldDofInterpreter;
  _mesh->boundary().setDofInterpreter(_dofInterpreter);
}

// must write to .m file
void Solution::writeFieldsToFile(int trialID, const string &filePath){
  typedef CellTools<double>  CellTools;
  
  //  cout << "writeFieldsToFile for trialID: " << trialID << endl;
  
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...
  int num1DPts = 5;
  
  fout << "numCells = " << _mesh->activeElements().size() << endl;
  fout << "x=cell(numCells,1);y=cell(numCells,1);z=cell(numCells,1);" << endl;
  
  // initialize storage
  fout << "for i = 1:numCells" << endl;
  fout << "x{i} = zeros(" << num1DPts << ",1);"<<endl;
  fout << "y{i} = zeros(" << num1DPts << ",1);"<<endl;
  fout << "z{i} = zeros(" << num1DPts << ");"<<endl;
  fout << "end" << endl;
  int globalCellInd = 1; //matlab indexes from 1
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) { //thru quads/triangles/etc
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> vertexPoints, physPoints;
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);
    
    int numCells = physicalCellNodes.dimension(0);
    bool createSideCacheToo = false;
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh, createSideCacheToo));
    
    vector<GlobalIndexType> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      GlobalIndexType cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
      cellIDs.push_back(cellID);
    }
    
    int numPoints = num1DPts * num1DPts;
    FieldContainer<double> refPoints(numPoints,spaceDim);
    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
        double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
        refPoints(pointIndex,0) = x;
        refPoints(pointIndex,1) = y;
      }
    }
    
    basisCache->setRefCellPoints(refPoints);
    basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCacheToo);
    FieldContainer<double> computedValues(numCells,numPoints);
    
    this->solutionValues(computedValues, trialID, basisCache);
    const FieldContainer<double> *physicalPoints = &basisCache->getPhysicalCubaturePoints();
    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++ ) {
      for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
        int yPointIndex = 0;
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        fout << "x{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<")=" << (*physicalPoints)(cellIndex,pointIndex,0) << ";" << endl;
      }
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
        int xPointIndex = 0;
        int pointIndex = xPointIndex*num1DPts + yPointIndex;
        fout << "y{"<<globalCellInd+cellIndex<< "}("<<yPointIndex+1<<")=" << (*physicalPoints)(cellIndex,pointIndex,1) << ";" << endl;
      }
    }
    
    for (int cellIndex=0;cellIndex < numCells;cellIndex++){
      for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
        for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
          int ptIndex = xPointIndex*num1DPts + yPointIndex;
          fout << "z{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<","<<yPointIndex+1<<")=" << computedValues(cellIndex,ptIndex) << ";" << endl;
        }
      }
    }
    
    //    // NOW loop over all cells to write solution to file
    //    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
    //      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){
    //
    //        // for some odd reason, I cannot compute the ref-to-phys map for more than 1 point at a time
    //        int numPoints = 1;
    //        FieldContainer<double> refPoints(numPoints,spaceDim);
    //        double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
    //        double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
    //        refPoints(0,0) = x;
    //        refPoints(0,1) = y;
    //
    //        // map side cubature points in reference parent cell domain to physical space
    //        FieldContainer<double> physicalPoints(numCells, numPoints, spaceDim);
    //        CellTools::mapToPhysicalFrame(physicalPoints, refPoints, physicalCellNodes, cellTopo);
    //
    //        cout << "physicalPoints:\n" <<  physicalPoints;
    //
    //        FieldContainer<double> computedValues(numCells,numPoints); // first arg = 1 cell only
    //        solutionValues(computedValues, elemTypePtr, trialID, physicalPoints);
    //
    //        for (int cellIndex=0;cellIndex < numCells;cellIndex++){
    //          fout << "x{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<")=" << physicalPoints(cellIndex,0,0) << ";" << endl;
    //          fout << "y{"<<globalCellInd+cellIndex<< "}("<<yPointIndex+1<<")=" << physicalPoints(cellIndex,0,1) << ";" << endl;
    //          fout << "z{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<","<<yPointIndex+1<<")=" << computedValues(cellIndex,0) << ";" << endl;
    //        }
    //      }
    //    }
    globalCellInd+=numCells;
    
  } //end of element type loop
  fout.close();
}

void Solution::writeFluxesToFile(int trialID, const string &filePath){
  typedef CellTools<double>  CellTools;
  
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...
  
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) { //thru quads/triangles/etc
    
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    int numSides = CamelliaCellTools::getSideCount(cellTopo);
    
    FieldContainer<double> vertexPoints, physPoints;
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);
    
    int numCells = vertexPoints.dimension(0);
    // takes centroid of all cells
    int numVertices = vertexPoints.dimension(1);
    FieldContainer<double> cellIDs(numCells);
    for (int cellIndex=0;cellIndex<numCells;cellIndex++){
      FieldContainer<double> cellCentroid(spaceDim);
      cellCentroid.initialize(0.0);
      for (int vertIndex=0;vertIndex<numVertices;vertIndex++){
        for (int dimIndex=0;dimIndex<spaceDim;dimIndex++){
          cellCentroid(dimIndex) += vertexPoints(cellIndex,vertIndex,dimIndex);
        }
      }
      for (int dimIndex=0;dimIndex<spaceDim;dimIndex++){
        cellCentroid(dimIndex) /= numVertices;
      }
      cellCentroid.resize(1,spaceDim); // only one cell
      int cellID = _mesh->elementsForPoints(cellCentroid)[0]->cellID();
      cellIDs(cellIndex) = cellID;
    }
    
    for (int sideIndex=0; sideIndex < numSides; sideIndex++){
      
      DefaultCubatureFactory<double>  cubFactory;
      int cubDegree = 15;//arbitrary number of points per cell, make dep on basis degree?
      shards::CellTopology side(cellTopo.getCellTopologyData(spaceDim-1,sideIndex));
      int sideDim = side.getDimension();
      Teuchos::RCP<Cubature<double> > sideCub = cubFactory.create(side, cubDegree);
      int numCubPoints = sideCub->getNumPoints();
      FieldContainer<double> cubPointsSideRefCell(numCubPoints, spaceDim); // just need the reference cell cubature points - map to physical space in n-D space
      FieldContainer<double> cubPointsSide(numCubPoints, sideDim);
      FieldContainer<double> cubWeightsSide(numCubPoints);// dummy for now
      
      sideCub->getCubature(cubPointsSide, cubWeightsSide);
      
      // compute geometric cell information
      CellTools::mapToReferenceSubcell(cubPointsSideRefCell, cubPointsSide, sideDim, sideIndex, cellTopo);
      
      // map side cubature points in reference parent cell domain to physical space
      FieldContainer<double> physCubPoints(numCells, numCubPoints, spaceDim);
      CellTools::mapToPhysicalFrame(physCubPoints, cubPointsSideRefCell, physicalCellNodes, cellTopo);
      
      // we now have cubPointsSideRefCell
      FieldContainer<double> computedValues(numCells,numCubPoints); // first arg = 1 cell only
      solutionValues(computedValues, elemTypePtr, trialID, physCubPoints, cubPointsSide, sideIndex);
      
      // NOW loop over all cells to write solution to file
      for (int cellIndex=0;cellIndex < numCells;cellIndex++){
        FieldContainer<double> cellParities = _mesh->cellSideParitiesForCell( cellIDs(cellIndex) );
        for (int pointIndex = 0; pointIndex < numCubPoints; pointIndex++){
          for (int dimInd=0;dimInd<spaceDim;dimInd++){
            fout << physCubPoints(cellIndex,pointIndex,dimInd) << " ";
          }
          /* // if we can figure out how to undo the parity negation on fluxes, do so here
           if (_mesh->bilinearForm()->functionSpaceForTrial(trialID)==IntrepidExtendedTypes::FUNCTION_SPACE_HVOL){
           computedValues(cellIndex,pointIndex) *= cellParities(sideIndex);
           }
           */
          fout << computedValues(cellIndex,pointIndex) << endl;
        }
        // insert NaN for matlab to plot discontinuities - WILL NOT WORK IN 3D
        for (int dimInd=0;dimInd<spaceDim;dimInd++){
          fout << "NaN" << " ";
        }
        fout << "NaN" << endl;
      }
    }
  }
  fout.close();
}

double Solution::totalTimeLocalStiffness() {
  return _totalTimeLocalStiffness;
}

double Solution::totalTimeGlobalAssembly() {
  return _totalTimeGlobalAssembly;
}

double Solution::totalTimeBCImposition() {
  return _totalTimeBCImposition;
}

double Solution::totalTimeSolve() {
  return _totalTimeSolve;
}

double Solution::totalTimeDistributeSolution() {
  return _totalTimeDistributeSolution;
}

double Solution::meanTimeLocalStiffness() {
  return _meanTimeLocalStiffness;
}

double Solution::meanTimeGlobalAssembly() {
  return _meanTimeGlobalAssembly;
}

double Solution::meanTimeBCImposition() {
  return _meanTimeBCImposition;
}

double Solution::meanTimeSolve() {
  return _meanTimeSolve;
}

double Solution::meanTimeDistributeSolution() {
  return _meanTimeDistributeSolution;
}

double Solution::maxTimeLocalStiffness() {
  return _maxTimeLocalStiffness;
}

double Solution::maxTimeGlobalAssembly() {
  return _maxTimeGlobalAssembly;
}

double Solution::maxTimeBCImposition() {
  return _maxTimeBCImposition;
}

double Solution::maxTimeSolve() {
  return _maxTimeSolve;
}

double Solution::maxTimeDistributeSolution() {
  return _maxTimeDistributeSolution;
}

double Solution::minTimeLocalStiffness() {
  return _minTimeLocalStiffness;
}

double Solution::minTimeGlobalAssembly() {
  return _minTimeGlobalAssembly;
}

double Solution::minTimeBCImposition() {
  return _minTimeBCImposition;
}

double Solution::minTimeSolve() {
  return _minTimeSolve;
}

double Solution::minTimeDistributeSolution() {
  return _minTimeDistributeSolution;
}

Epetra_Map Solution::getPartitionMap() {
  int rank = Teuchos::GlobalMPISession::getRank();
  
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  vector<int> zeroMeanConstraints = getZeroMeanConstraints();
  GlobalIndexType numGlobalDofs = _dofInterpreter->globalDofCount();
  set<GlobalIndexType> myGlobalIndicesSet = _dofInterpreter->globalDofIndicesForPartition(rank);
  int numZMCDofs = _zmcsAsRankOneUpdate ? 0 : zeroMeanConstraints.size();
  
  Epetra_Map partMap = getPartitionMap(rank, myGlobalIndicesSet,numGlobalDofs,numZMCDofs,&Comm);
  return partMap;
}

Epetra_Map Solution::getPartitionMap(PartitionIndexType rank, set<GlobalIndexType> & myGlobalIndicesSet, GlobalIndexType numGlobalDofs,
                                     int zeroMeanConstraintsSize, Epetra_Comm* Comm ) {
  int numGlobalLagrange = _lagrangeConstraints->numGlobalConstraints();
  vector< ElementPtr > elements = _mesh->elementsInPartition(rank);
  IndexType numMyElements = elements.size();
  int numElementLagrange = _lagrangeConstraints->numElementConstraints() * numMyElements;
  int globalNumElementLagrange = _lagrangeConstraints->numElementConstraints() * _mesh->numActiveElements();
  
  // ordering is:
  // - regular dofs
  // - element lagrange
  // - global lagrange
  // - zero-mean constraints
  
  // determine the local dofs we have, and what their global indices are:
  int localDofsSize = myGlobalIndicesSet.size() + numElementLagrange;
  if (rank == 0) {
    // global Lagrange and zero-mean constraints belong to rank 0
    localDofsSize += zeroMeanConstraintsSize + numGlobalLagrange;
  }
  
  GlobalIndexTypeToCast *myGlobalIndices;
  if (localDofsSize!=0){
    myGlobalIndices = new GlobalIndexTypeToCast[ localDofsSize ];
  } else {
    myGlobalIndices = NULL;
  }
  
  // copy from set object into the allocated array
  GlobalIndexType offset = 0;
  for (set<GlobalIndexType>::iterator indexIt = myGlobalIndicesSet.begin(); indexIt != myGlobalIndicesSet.end(); indexIt++ ) {
    myGlobalIndices[offset++] = *indexIt;
  }
  GlobalIndexType cellOffset = _mesh->activeCellOffset() * _lagrangeConstraints->numElementConstraints();
  GlobalIndexType globalIndex = cellOffset + numGlobalDofs;
  for (int elemLagrangeIndex=0; elemLagrangeIndex<_lagrangeConstraints->numElementConstraints(); elemLagrangeIndex++) {
    for (IndexType cellIndex=0; cellIndex<numMyElements; cellIndex++) {
      myGlobalIndices[offset++] = globalIndex++;
    }
  }
  
  if ( rank == 0 ) {
    // set up the zmcs and global Lagrange constraints, which come at the end...
    for (int i=0; i<numGlobalLagrange; i++) {
      myGlobalIndices[offset++] = i + numGlobalDofs + globalNumElementLagrange;
    }
    for (int i=0; i<zeroMeanConstraintsSize; i++) {
      myGlobalIndices[offset++] = i + numGlobalDofs + globalNumElementLagrange + numGlobalLagrange;
    }
  }
  
  if (offset != localDofsSize) {
    cout << "WARNING: Apparent internal error in Solution::getPartitionMap.  # entries filled in myGlobalDofIndices does not match its size...\n";
  }
  
  int totalRows = numGlobalDofs + globalNumElementLagrange + numGlobalLagrange + zeroMeanConstraintsSize;
  
  int indexBase = 0;
  //cout << "process " << rank << " about to construct partMap.\n";
  //Epetra_Map partMap(-1, localDofsSize, myGlobalIndices, indexBase, Comm);
//  cout << "process " << rank << " about to construct partMap; totalRows = " << totalRows;
//  cout << "; localDofsSize = " << localDofsSize << ".\n";
  Epetra_Map partMap(totalRows, localDofsSize, myGlobalIndices, indexBase, *Comm);
  
  if (localDofsSize!=0){
    delete[] myGlobalIndices;
  }
  return partMap;
}

void Solution::processSideUpgrades( const map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades ) {
  set<GlobalIndexType> cellIDsToSkip; //empty
  processSideUpgrades(cellSideUpgrades,cellIDsToSkip);
}

void Solution::processSideUpgrades( const map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const set<GlobalIndexType> &cellIDsToSkip ) {
  for (map<GlobalIndexType, pair< ElementTypePtr, ElementTypePtr > >::const_iterator upgradeIt = cellSideUpgrades.begin();
       upgradeIt != cellSideUpgrades.end(); upgradeIt++) {
    GlobalIndexType cellID = upgradeIt->first;
    if (cellIDsToSkip.find(cellID) != cellIDsToSkip.end() ) continue;
    if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end())
      continue; // no previous solution for this cell
    DofOrderingPtr oldTrialOrdering = (upgradeIt->second).first->trialOrderPtr;
    DofOrderingPtr newTrialOrdering = (upgradeIt->second).second->trialOrderPtr;
    FieldContainer<double> newCoefficients(newTrialOrdering->totalDofs());
    newTrialOrdering->copyLikeCoefficients( newCoefficients, oldTrialOrdering, _solutionForCellIDGlobal[cellID] );
    //    cout << "processSideUpgrades: setting solution for cell ID " << cellID << endl;
    _solutionForCellIDGlobal[cellID] = newCoefficients;
  }
}

void Solution::projectOntoMesh(const map<int, Teuchos::RCP<Function> > &functionMap){ // map: trialID -> function
  if (_lhsVector.get()==NULL) {
    initializeLHSVector();
  }
  
  set<GlobalIndexType> cellIDs = _mesh->globalDofAssignment()->cellsInPartition(-1);
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    projectOntoCell(functionMap,cellID);
  }
}

void Solution::projectOntoCell(const map<int, FunctionPtr > &functionMap, GlobalIndexType cellID, int side) {
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
  vector<GlobalIndexType> cellIDs(1,cellID);
  
  for (map<int, FunctionPtr >::const_iterator functionIt = functionMap.begin(); functionIt !=functionMap.end(); functionIt++){
    int trialID = functionIt->first;
    
    bool fluxOrTrace = _mesh->bilinearForm()->isFluxOrTrace(trialID);
    FunctionPtr function = functionIt->second;
    ElementPtr element = _mesh->getElement(cellID);
    ElementTypePtr elemTypePtr = element->elementType();
    
    bool testVsTest = false; // in fact it's more trial vs trial, but this just means we'll over-integrate a bit
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemTypePtr,_mesh,testVsTest,_cubatureEnrichmentDegree) );
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,fluxOrTrace); // create side cache if it's a trace or flux
    
    if (fluxOrTrace) {
      int firstSide, lastSide;
      if (side == -1) { // handle all sides
        firstSide = 0;
        lastSide = CamelliaCellTools::getSideCount(*elemTypePtr->cellTopoPtr) - 1;
      } else {
        firstSide = side;
        lastSide = side;
      }
      for (int sideIndex=firstSide; sideIndex<=lastSide; sideIndex++) {
        if (! elemTypePtr->trialOrderPtr->hasBasisEntry(trialID, sideIndex)) {
          continue;
        }
        
        BasisPtr basis = elemTypePtr->trialOrderPtr->getBasis(trialID, sideIndex);
        FieldContainer<double> basisCoefficients(1,basis->getCardinality());
        Projector::projectFunctionOntoBasis(basisCoefficients, function, basis, basisCache->getSideBasisCache(sideIndex));
        basisCoefficients.resize(basis->getCardinality());
        setSolnCoeffsForCellID(basisCoefficients,cellID,trialID,sideIndex);
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(side != -1, std::invalid_argument, "sideIndex for fields must = -1");
      if (! elemTypePtr->trialOrderPtr->hasBasisEntry(trialID, 0)) { // DofOrdering uses side 0 for fields...
        continue;
      }
      
      BasisPtr basis = elemTypePtr->trialOrderPtr->getBasis(trialID);
      FieldContainer<double> basisCoefficients(1,basis->getCardinality());
      Projector::projectFunctionOntoBasis(basisCoefficients, function, basis, basisCache);
      basisCoefficients.resize(basis->getCardinality());
      setSolnCoeffsForCellID(basisCoefficients,cellID,trialID);
    }
  }
}

void Solution::projectOntoMesh(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap){
  if (_lhsVector.get()==NULL) {
    initializeLHSVector();
  }
  
  set<GlobalIndexType> cellIDs = _mesh->globalDofAssignment()->cellsInPartition(-1);
  for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    projectOntoCell(functionMap,cellID);
  }
}

void Solution::projectFieldVariablesOntoOtherSolution(SolutionPtr otherSoln) {
  vector< int > fieldIDs = _mesh->bilinearForm()->trialVolumeIDs();
  vector< VarPtr > fieldVars;
  for (vector<int>::iterator fieldIt = fieldIDs.begin(); fieldIt != fieldIDs.end(); fieldIt++) {
    int fieldID = *fieldIt;
    VarPtr var = Teuchos::rcp( new Var(fieldID, 0, "unspecified") );
    fieldVars.push_back(var);
  }
  Teuchos::RCP<Solution> thisPtr = Teuchos::rcp(this, false);
  map<int, FunctionPtr > solnMap = PreviousSolutionFunction::functionMap(fieldVars, thisPtr);
  otherSoln->projectOntoMesh(solnMap);
}

void Solution::projectOntoCell(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap, GlobalIndexType cellID){
  typedef Teuchos::RCP<AbstractFunction> AbstractFxnPtr;
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
  
  for (map<int, AbstractFxnPtr >::const_iterator functionIt = functionMap.begin(); functionIt !=functionMap.end(); functionIt++){
    int trialID = functionIt->first;
    AbstractFxnPtr function = functionIt->second;
    ElementPtr element = _mesh->getElement(cellID);
    ElementTypePtr elemTypePtr = element->elementType();
    
    BasisPtr basis = elemTypePtr->trialOrderPtr->getBasis(trialID);
    
    FieldContainer<double> basisCoefficients;
    Projector::projectFunctionOntoBasis(basisCoefficients, function, basis, physicalCellNodes);
    basisCoefficients.resize(basisCoefficients.size());
    setSolnCoeffsForCellID(basisCoefficients,cellID,trialID);
  }
}

void Solution::projectOldCellOntoNewCells(GlobalIndexType cellID,
                                          ElementTypePtr oldElemType,
                                          const vector<GlobalIndexType> &childIDs) {
  int rank = Teuchos::GlobalMPISession::getRank();

  if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end()) {
//    cout << "on rank " << rank << ", no solution for " << cellID << "; skipping projection onto children.\n";
    return; // zero solution on cell
  }
//  cout << "on rank " << rank << ", projecting " << cellID << " data onto children.\n";
  const FieldContainer<double>* oldData = &_solutionForCellIDGlobal[cellID];
//  cout << "cell " << cellID << " data: \n" << *oldData;
  projectOldCellOntoNewCells(cellID, oldElemType, *oldData, childIDs);
}

void Solution::projectOldCellOntoNewCells(GlobalIndexType cellID,
                                          ElementTypePtr oldElemType,
                                          const FieldContainer<double> &oldData,
                                          const vector<GlobalIndexType> &childIDs)
 {
   VarFactory vf = _mesh->bilinearForm()->varFactory();
   
   DofOrderingPtr oldTrialOrdering = oldElemType->trialOrderPtr;
   set<int> trialIDs = oldTrialOrdering->getVarIDs();
   
   TEUCHOS_TEST_FOR_EXCEPTION(oldTrialOrdering->totalDofs() != oldData.size(), std::invalid_argument,
                              "oldElemType trial space does not match old data coefficients size");
   map<int, FunctionPtr > fieldMap;
   
   CellPtr parentCell = _mesh->getTopology()->getCell(cellID);
   int dummyCubatureDegree = 1;
   BasisCachePtr parentRefCellCache = BasisCache::basisCacheForReferenceCell(*parentCell->topology(), dummyCubatureDegree);
   
//   cout << "projecting from cell " << cellID << " onto its children.\n";
   
   for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++) {
     int trialID = *trialIDIt;
     if (oldTrialOrdering->getNumSidesForVarID(trialID) == 1) { // field variable, the only kind we honor right now
       BasisPtr basis = oldTrialOrdering->getBasis(trialID);
       int basisCardinality = basis->getCardinality();
       FieldContainer<double> basisCoefficients(basisCardinality);
       
       for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++) {
         int dofIndex = oldElemType->trialOrderPtr->getDofIndex(trialID, dofOrdinal);
         basisCoefficients(dofOrdinal) = oldData(dofIndex);
       }
       
//       cout << "basisCoefficients for parent volume trialID " << trialID << ":\n" << basisCoefficients;
       
       FunctionPtr oldTrialFunction = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients, parentRefCellCache) );
       fieldMap[trialID] = oldTrialFunction;
     }
   }
   
   FunctionPtr sideParity = Function::sideParity();
   map<int,FunctionPtr> interiorTraceMap; // functions to use on parent interior to represent traces there
   for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++) {
     int trialID = *trialIDIt;
     if (oldTrialOrdering->getNumSidesForVarID(trialID) != 1) { // trace (flux) variable
       VarPtr var = vf.trialVars().find(trialID)->second;
       
       LinearTermPtr termTraced = var->termTraced();
       if (termTraced.get() != NULL) {
         FunctionPtr fieldTrace = termTraced->evaluate(fieldMap, true) + termTraced->evaluate(fieldMap, false);
         if (var->varType() == FLUX) { // then we do need to include side parity here
           fieldTrace = sideParity * fieldTrace;
         }
         interiorTraceMap[trialID] = fieldTrace;
       }
     }
   }

   int sideDim = _mesh->getTopology()->getSpaceDim() - 1;
  
   int sideCount = CamelliaCellTools::getSideCount(*parentCell->topology());
   vector< map<int, FunctionPtr> > traceMap(sideCount);
   for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
     shards::CellTopology sideTopo = parentCell->topology()->getCellTopologyData(sideDim, sideOrdinal);
     BasisCachePtr parentSideTopoBasisCache = BasisCache::basisCacheForReferenceCell(sideTopo, dummyCubatureDegree);
     for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++) {
       int trialID = *trialIDIt;
       if (oldTrialOrdering->getNumSidesForVarID(trialID) != 1) { // trace (flux) variable
         BasisPtr basis = oldTrialOrdering->getBasis(trialID, sideOrdinal);
         int basisCardinality = basis->getCardinality();
         FieldContainer<double> basisCoefficients(basisCardinality);
         
         for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++) {
           int dofIndex = oldElemType->trialOrderPtr->getDofIndex(trialID, dofOrdinal, sideOrdinal);
           basisCoefficients(dofOrdinal) = oldData(dofIndex);
         }
         FunctionPtr oldTrialFunction = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients, parentSideTopoBasisCache) );
         traceMap[sideOrdinal][trialID] = oldTrialFunction;
       }
     }
   }
   
   int parent_p_order = _mesh->getElementType(cellID)->trialOrderPtr->maxBasisDegree();
   
  for (int childOrdinal=0; childOrdinal < childIDs.size(); childOrdinal++) {
    GlobalIndexType childID = childIDs[childOrdinal];
    CellPtr childCell = _mesh->getTopology()->getCell(childID);
    ElementTypePtr childType = _mesh->getElementType(childID);
    int childSideCount = CamelliaCellTools::getSideCount(*childCell->topology());

    int child_p_order = _mesh->getElementType(childID)->trialOrderPtr->maxBasisDegree();
    int cubatureDegree = parent_p_order + child_p_order;
    
    BasisCachePtr volumeBasisCache;
    vector<BasisCachePtr> sideBasisCache(childSideCount);
    
    if (parentCell->children().size() > 0) {
      RefinementBranch refBranch(1,make_pair(parentCell->refinementPattern().get(), childOrdinal));
      volumeBasisCache = BasisCache::basisCacheForRefinedReferenceCell(*childCell->topology(), cubatureDegree, refBranch, true);
      for (int sideOrdinal = 0; sideOrdinal < childSideCount; sideOrdinal++) {
        shards::CellTopology sideTopo = childCell->topology()->getCellTopologyData(sideDim, sideOrdinal);
        unsigned parentSideOrdinal = (childID==cellID) ? sideOrdinal
                                   : parentCell->refinementPattern()->mapSubcellOrdinalFromChildToParent(childOrdinal, sideDim, sideOrdinal);

        RefinementBranch sideBranch;
        if (parentSideOrdinal != -1)
          sideBranch = RefinementPattern::subcellRefinementBranch(refBranch, sideDim, parentSideOrdinal);
        if (sideBranch.size()==0) {
          sideBasisCache[sideOrdinal] = BasisCache::basisCacheForReferenceCell(sideTopo, cubatureDegree);
        } else {
          sideBasisCache[sideOrdinal] = BasisCache::basisCacheForRefinedReferenceCell(sideTopo, cubatureDegree, sideBranch);
        }
      }
    } else {
      volumeBasisCache = BasisCache::basisCacheForReferenceCell(*childCell->topology(), cubatureDegree, true);
      for (int sideOrdinal = 0; sideOrdinal < childSideCount; sideOrdinal++) {
        shards::CellTopology sideTopo = childCell->topology()->getCellTopologyData(sideDim, sideOrdinal);
        sideBasisCache[sideOrdinal] = BasisCache::basisCacheForReferenceCell(sideTopo, cubatureDegree);
      }
    }
  
    // (re)initialize the FieldContainer storing the solution--element type may have changed (in case of p-refinement)
    _solutionForCellIDGlobal[childID] = FieldContainer<double>(childType->trialOrderPtr->totalDofs());
    // project fields
    FieldContainer<double> basisCoefficients;
    for (map<int,FunctionPtr>::iterator fieldFxnIt=fieldMap.begin(); fieldFxnIt != fieldMap.end(); fieldFxnIt++) {
      int varID = fieldFxnIt->first;
      FunctionPtr fieldFxn = fieldFxnIt->second;
      BasisPtr childBasis = childType->trialOrderPtr->getBasis(varID);
      basisCoefficients.resize(1,childBasis->getCardinality());
      Projector::projectFunctionOntoBasisInterpolating(basisCoefficients, fieldFxn, childBasis, volumeBasisCache);
      
//      cout << "projected basisCoefficients for child volume trialID " << varID << ":\n" << basisCoefficients;
      
      for (int basisOrdinal=0; basisOrdinal<basisCoefficients.size(); basisOrdinal++) {
        int dofIndex = childType->trialOrderPtr->getDofIndex(varID, basisOrdinal);
        _solutionForCellIDGlobal[childID][dofIndex] = basisCoefficients[basisOrdinal];
      }
    }
    
    // project traces and fluxes
    for (int sideOrdinal=0; sideOrdinal<childSideCount; sideOrdinal++) {
      unsigned parentSideOrdinal = (childID==cellID) ? sideOrdinal
                                 : parentCell->refinementPattern()->mapSubcellOrdinalFromChildToParent(childOrdinal, sideDim, sideOrdinal);
      
      map<int,FunctionPtr>* traceMapForSide = (parentSideOrdinal != -1) ? &traceMap[parentSideOrdinal] : &interiorTraceMap;
      // which BasisCache to use depends on whether we want the BasisCache's notion of "physical" space to be in the volume or on the side:
      // we want it to be on the side if parent shares the side (and we therefore have proper trace data)
      // and on the volume in parent doesn't share the side (in which case we use the interior trace map).
      BasisCachePtr basisCacheForSide = (parentSideOrdinal != -1) ? sideBasisCache[sideOrdinal] : volumeBasisCache->getSideBasisCache(sideOrdinal);
      
      basisCacheForSide->setCellSideParities(_mesh->cellSideParitiesForCell(childID));
      
      for (map<int,FunctionPtr>::iterator traceFxnIt=traceMapForSide->begin(); traceFxnIt != traceMapForSide->end(); traceFxnIt++) {
        int varID = traceFxnIt->first;
        FunctionPtr traceFxn = traceFxnIt->second;
        BasisPtr childBasis = childType->trialOrderPtr->getBasis(varID, sideOrdinal);
        basisCoefficients.resize(1,childBasis->getCardinality());
        Projector::projectFunctionOntoBasisInterpolating(basisCoefficients, traceFxn, childBasis, basisCacheForSide);
        for (int basisOrdinal=0; basisOrdinal<basisCoefficients.size(); basisOrdinal++) {
          int dofIndex = childType->trialOrderPtr->getDofIndex(varID, basisOrdinal, sideOrdinal);
          _solutionForCellIDGlobal[childID][dofIndex] = basisCoefficients[basisOrdinal];
          // worth noting that as now set up, the "field traces" may stomp on the true traces, depending on in what order the sides
          // are mapped to global dof ordinals.  For right now, I'm not too worried about this.
        }
      }
    }
  }
   
  clearComputedResiduals(); // force recomputation of energy error (could do something more incisive, just computing the energy error for the new cells)
}

void Solution::readFromFile(const string &filePath) {
  ifstream fin(filePath.c_str());
  
  while (fin.good()) {
    string refTypeStr;
    int cellID;
    
    string line;
    std::getline(fin, line, '\n');
    std::istringstream linestream(line);
    linestream >> cellID;
    
    if (_mesh->getElement(cellID).get() == NULL) {
      cout << "No cellID " << cellID << endl;
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Could not find cellID in solution file in mesh.");
    }
    ElementTypePtr elemType = _mesh->getElement(cellID)->elementType();
    int numDofsExpected = elemType->trialOrderPtr->totalDofs();
    
    if ( linestream.good() ) {
      int numDofs;
      linestream >> numDofs;
      
      // TODO: check that numDofs is right for cellID.
      if (numDofsExpected != numDofs) {
        cout << "ERROR in readFromFile: expected cellID " << cellID << " to have " << numDofsExpected;
        cout << ", but found " << numDofs << " instead.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "wrong number of dofs for cell");
      }
      
      FieldContainer<double> dofValues(numDofs);
      double dofValue;
      int dofOrdinal = 0;
      while (linestream.good()) {
        linestream >> dofValue;
        dofValues[dofOrdinal++] = dofValue;
      }
      
      _solutionForCellIDGlobal[cellID] = dofValues;
    }
  }
  fin.close();
}

SolutionPtr Solution::solution(MeshPtr mesh, BCPtr bc, RHSPtr rhs, Teuchos::RCP<DPGInnerProduct> ip ) {
  return Teuchos::rcp( new Solution(mesh,bc,rhs,ip) );
}

void Solution::writeToFile(const string &filePath) {
  ofstream fout(filePath.c_str());
  
  for (map<GlobalIndexType, FieldContainer<double> >::iterator solnEntryIt = _solutionForCellIDGlobal.begin();
       solnEntryIt != _solutionForCellIDGlobal.end(); solnEntryIt++) {
    GlobalIndexType cellID = solnEntryIt->first;
    FieldContainer<double>* solnCoeffs = &(solnEntryIt->second);
    fout << cellID << " " << solnCoeffs->size() << " ";
    for (int i=0; i<solnCoeffs->size(); i++) {
      fout << (*solnCoeffs)[i] << " ";
    }
    fout << endl;
  }
  
  fout.close();
}

#ifdef HAVE_EPETRAEXT_HDF5
void Solution::saveToHDF5(string filename)
{
  int commRank = Teuchos::GlobalMPISession::getRank();
  int nProcs = Teuchos::GlobalMPISession::getNProc();

  Epetra_SerialComm Comm;
  EpetraExt::HDF5 hdf5(Comm);
  hdf5.Create(filename+Teuchos::toString(commRank)+".h5");
  hdf5.Write("Solution", "nProcs", nProcs);
  hdf5.Write("Solution", "commRank", commRank);

  set<GlobalIndexType> myCells = mesh()->cellIDsInPartition();
  vector<GlobalIndexType> myCellsVec( myCells.begin(), myCells.end() );
  hdf5.Write("Solution", "partitionCellIDs", H5T_NATIVE_INT, myCellsVec.size(), &myCellsVec[0]);
  for (map<GlobalIndexType, FieldContainer<double> >::iterator solnEntryIt = _solutionForCellIDGlobal.begin();
       solnEntryIt != _solutionForCellIDGlobal.end(); solnEntryIt++) 
  {
    GlobalIndexType cellID = solnEntryIt->first;
    if (myCells.find(cellID) != myCells.end()) {
      FieldContainer<double>* solnCoeffs = &(solnEntryIt->second);
      cout << cellID << " " << commRank << endl;
      hdf5.Write("Solution", "cell"+Teuchos::toString(cellID), H5T_NATIVE_DOUBLE, solnCoeffs->size(), solnCoeffs);
    }
  }

  hdf5.Close();
}
#endif

vector<int> Solution::getZeroMeanConstraints() {
  // determine any zero-mean constraints:
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  vector< int > zeroMeanConstraints;
  if (_bc.get()==NULL) return zeroMeanConstraints; //empty
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (_bc->imposeZeroMeanConstraint(trialID)) {
      zeroMeanConstraints.push_back(trialID);
    }
  }
  return zeroMeanConstraints;
}

void Solution::setZeroMeanConstraintRho(double value) {
  _zmcRho = value;
}

double Solution::zeroMeanConstraintRho() {
  return _zmcRho;
}