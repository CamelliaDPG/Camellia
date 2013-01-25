
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
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "Epetra_Time.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

//#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
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

#include "Var.h"

double Solution::conditionNumberEstimate( Epetra_LinearProblem & problem ) {
  // TODO: work out how to suppress the console output here
  double condest = -1;
  AztecOO solverForConditionEstimate(problem);
  solverForConditionEstimate.SetAztecOption(AZ_solver, AZ_cg_condnum);
  solverForConditionEstimate.ConstructPreconditioner(condest);
  return condest;
}

int Solution::cubatureEnrichmentDegree() const {
  return _cubatureEnrichmentDegree;
}

void Solution::setCubatureEnrichmentDegree(int value) {
  _cubatureEnrichmentDegree = value;
}

static const int MAX_BATCH_SIZE_IN_BYTES = 3*1024*1024; // 3 MB
static const int MIN_BATCH_SIZE_IN_CELLS = 2; // overrides the above, if it results in too-small batches

// copy constructor:
Solution::Solution(const Solution &soln) {
  _mesh = soln.mesh();
  _bc = soln.bc();
  _rhs = soln.rhs();
  _ip = soln.ip();
  _solutionForCellIDGlobal = soln.solutionForCellIDGlobal();
  _filter = soln.filter();
  _lagrangeConstraints = soln.lagrangeConstraints();
  _reportConditionNumber = false;
  _reportTimingResults = false;
  _writeMatrixToMatlabFile = false;
  _cubatureEnrichmentDegree = soln.cubatureEnrichmentDegree();
}

Solution::Solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc, Teuchos::RCP<RHS> rhs, Teuchos::RCP<DPGInnerProduct> ip) {
  _mesh = mesh;
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
  _residualsComputed = false;
  _energyErrorComputed = false;
  _reportConditionNumber = false;
  _reportTimingResults = false;
  _globalSystemConditionEstimate = -1;
  _cubatureEnrichmentDegree = 0;
}

void Solution::addSolution(Teuchos::RCP<Solution> otherSoln, double weight, bool allowEmptyCells) {
  // thisSoln += weight * otherSoln
  // throws exception if the two Solutions' solutionForElementTypeMaps fail to match in any way other than in values
  const map< int, FieldContainer<double> >* otherMapPtr = &(otherSoln->solutionForCellIDGlobal());
  if ( ! allowEmptyCells ) {
    TEUCHOS_TEST_FOR_EXCEPTION(otherMapPtr->size() != _solutionForCellIDGlobal.size(),
                       std::invalid_argument, "otherSoln doesn't match Solution's solutionMap.");
  }
  map< int, FieldContainer<double> >::const_iterator mapIt;
  for (mapIt=otherMapPtr->begin(); mapIt != otherMapPtr->end(); mapIt++) {
    int cellID = mapIt->first;
    const FieldContainer<double>* otherValues = &(mapIt->second);
    map< int, FieldContainer<double> >::iterator myMapIt = _solutionForCellIDGlobal.find(cellID);
    if (myMapIt == _solutionForCellIDGlobal.end()) {
      if ( !allowEmptyCells ) {
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
                           "otherSoln doesn't match Solution's solutionMap (cellID not found).");
      } else {
        // just copy, and apply the weight
        _solutionForCellIDGlobal[cellID] = *otherValues;
        BilinearForm::multiplyFCByWeight(_solutionForCellIDGlobal[cellID],weight);
        continue;
      }
    }
    FieldContainer<double>* myValues = &(myMapIt->second);
    int numValues = myValues->size();
    TEUCHOS_TEST_FOR_EXCEPTION(numValues != otherValues->size(),
                       std::invalid_argument, "otherSoln doesn't match Solution's solutionMap (differing # of coefficients).");
    for (int dofIndex = 0; dofIndex < numValues; dofIndex++) {
      (*myValues)[dofIndex] += weight * (*otherValues)[dofIndex];
    }
  }
  // now that we've added, any computed residuals are invalid
  clearComputedResiduals();
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
  clearComputedResiduals();
}

void Solution::solve(Teuchos::RCP<Solver> solver) {
  // the following is not strictly necessary if the mesh has not changed since we were constructed:
  //initialize();
  
  bool zmcsAsRankOneUpdate = false; // seems to be working, but slow!!
  
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
  
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes(rank);
  vector< ElementTypePtr >::iterator elemTypeIt;
  // will want a CrsMatrix here in just a moment...
  
  // determine any zero-mean constraints:
  vector< int > trialIDs = _mesh->bilinearForm()->trialIDs();
  vector< int > zeroMeanConstraints;
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (_bc->imposeZeroMeanConstraint(trialID)) {
      zeroMeanConstraints.push_back(trialID);
    }
  }
  int numGlobalDofs = _mesh->numGlobalDofs();
  set<int> myGlobalIndicesSet = _mesh->globalDofIndicesForPartition(rank);
  int numZMCDofs = zmcsAsRankOneUpdate ? 0 : zeroMeanConstraints.size();
  
  Epetra_Map partMap = getPartitionMap(rank, myGlobalIndicesSet,numGlobalDofs,numZMCDofs,&Comm);
  //Epetra_Map globalMapG(numGlobalDofs+zeroMeanConstraints.size(), numGlobalDofs+zeroMeanConstraints.size(), 0, Comm);
  
  int maxRowSize = _mesh->rowSizeUpperBound();
/*  if (zeroMeanConstraints.size() > 0) {
      vector<ElementTypePtr> elemTypes = _mesh->elementTypes();
    int numEntries = 1; // 1 for the diagonal (stab. parameter)
    int trialID = zeroMeanConstraints[0];
    int sideIndex = 0; // zmc's are "inside" the cells
    for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
      ElementTypePtr elemTypePtr = *(elemTypeIt);
      int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
      int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
      numEntries += basisCardinality * numCellsOfType;
      FieldContainer<double> valuesForType(numCellsOfType, basisCardinality);
    }
    maxRowSize = max(numEntries,maxRowSize);
  }*/

  //cout << "max row size for mesh: " << maxRowSize << endl;
  //cout << "process " << rank << " about to initialize globalStiffMatrix.\n";
  Epetra_FECrsMatrix globalStiffMatrix(Copy, partMap, maxRowSize);
  //  Epetra_FECrsMatrix globalStiffMatrix(Copy, partMap, partMap, maxRowSize);
  Epetra_FEVector rhsVector(partMap);
  
  //cout << "process " << rank << " about to loop over elementTypes.\n";
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,Comm);
  Epetra_Time timer(Comm);
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
      vector<int> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        int cellID = _mesh->cellID(elemTypePtr, cellIndex+startCellIndexForBatch, rank);
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
      ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,_ip->hasBoundaryTerms()); // create side cache if ip has boundary values
      
      //int numCells = physicalCellNodes.dimension(0);
      CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
      
//      { // this block is not necessary for the solution.  Here just to produce debugging output
//        FieldContainer<double> preStiffness(numCells,numTestDofs,numTrialDofs );
//        
//        BilinearFormUtility::computeStiffnessMatrix(preStiffness, _mesh->bilinearForm(),
//                                                    trialOrderingPtr, testOrderingPtr, *(cellTopoPtr.get()), 
//                                                    physicalCellNodes, cellSideParities);
//        FieldContainer<double> preStiffnessTransposed(numCells,numTrialDofs,numTestDofs );
//        BilinearFormUtility::transposeFCMatrices(preStiffnessTransposed,preStiffness);
//        
////        cout << "preStiffness:\n" << preStiffness;
//      }
      FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
      
      _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, ipBasisCache);
      
//      cout << "ipMatrix:\n" << ipMatrix;
      
      FieldContainer<double> optTestCoeffs(numCells,numTrialDofs,numTestDofs);
      
      int optSuccess = _mesh->bilinearForm()->optimalTestWeights(optTestCoeffs, ipMatrix, elemTypePtr,
                                                                 cellSideParities, basisCache);
      
//      cout << "optTestCoeffs:\n" << optTestCoeffs;
      
      if ( optSuccess != 0 ) {
        cout << "**** WARNING: in Solution.solve(), optimal test function computation failed with error code " << optSuccess << ". ****\n";
      }
      
      //cout << "optTestCoeffs\n" << optTestCoeffs;
      
      FieldContainer<double> finalStiffness(numCells,numTrialDofs,numTrialDofs);
      
      BilinearFormUtility::computeStiffnessMatrix(finalStiffness,ipMatrix,optTestCoeffs);
      
      FieldContainer<double> localRHSVector(numCells, numTrialDofs);
      _rhs->integrateAgainstOptimalTests(localRHSVector, optTestCoeffs, testOrderingPtr, basisCache);
      
      // apply filter(s) (e.g. penalty method, preconditioners, etc.)
      if (_filter.get()) {
        _filter->filter(finalStiffness,localRHSVector,basisCache,_mesh,_bc);
	//        _filter->filter(localRHSVector,physicalCellNodes,cellIDs,_mesh,_bc);
      } 
      
      FieldContainer<int> globalDofIndices(numTrialDofs);
      
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        int cellID = _mesh->cellID(elemTypePtr,cellIndex+startCellIndexForBatch,rank);
        // we have the same local-to-global map for both rows and columns
        for (int i=0; i<numTrialDofs; i++) {
          globalDofIndices(i) = _mesh->globalDofIndex(cellID,i);
        }
        globalStiffMatrix.InsertGlobalValues(numTrialDofs,&globalDofIndices(0),numTrialDofs,&globalDofIndices(0),&finalStiffness(cellIndex,0,0));
        rhsVector.SumIntoGlobalValues(numTrialDofs,&globalDofIndices(0),&localRHSVector(cellIndex,0));
      }
      startCellIndexForBatch += numCells;
    }
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
      vector<int> cellIDs;
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
      
      FieldContainer<int> globalDofIndices(numTrialDofs+1); // max size
      FieldContainer<double> nonzeroValues(numTrialDofs+1);
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {        
        int globalRowIndex = partMap.GID(localRowIndex);
        int nnz = 0;
        for (int i=0; i<numTrialDofs; i++) {
          if (lhs(cellIndex,i) != 0.0) {
	          globalDofIndices(nnz) = _mesh->globalDofIndex(cellIDs[cellIndex],i);
            nonzeroValues(nnz) = lhs(cellIndex,i);
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
        globalStiffMatrix.InsertGlobalValues(1,&globalRowIndex,nnz+1,&globalDofIndices(0),
                                             &nonzeroValues(0));
        // insert column:
        globalStiffMatrix.InsertGlobalValues(nnz+1,&globalDofIndices(0),1,&globalRowIndex,
                                             &nonzeroValues(0));
        rhsVector.ReplaceGlobalValues(1,&globalRowIndex,&rhs(cellIndex));
        
        localRowIndex++;
      }
    }
  }
  
  // compute max, min h
  // TODO: get rid of the Global calls below (MPI-enable this code)
  double maxCellMeasure = 0;
  double minCellMeasure = 1e300;
  vector< ElementTypePtr > elemTypes = _mesh->elementTypes(); // global element types
  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemType = *elemTypeIt;
    vector< ElementPtr > elems = _mesh->elementsOfTypeGlobal(elemType);
    vector<int> cellIDs;
    for (int i=0; i<elems.size(); i++) {
      cellIDs.push_back(elems[i]->cellID());
    }
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemType);
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,_mesh) );
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();

    for (int i=0; i<elems.size(); i++) {
      maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
      minCellMeasure = min(minCellMeasure,cellMeasures(i));
    }
  }
  double min_h = sqrt(minCellMeasure); 
  double max_h = sqrt(maxCellMeasure);
 
 // define stabilizing parameter for zero-mean constraints
      double rho = 0.0; // our rho is the negative inverse of that in Bochev & Lehoucq
//      for (int i=0; i<numValues; i++) {
//        rho += basisIntegrals[i];
//      }
//  rho = -1 / (min_h * max_h);       // sorta like -1/h^2, following Bochev & Lehoucq
  rho = -1.0;
  if (rank == 0) {
    int numGlobalConstraints = _lagrangeConstraints->numGlobalConstraints();
    TEUCHOS_TEST_FOR_EXCEPTION(numGlobalConstraints != 0, std::invalid_argument, "global constraints not yet supported in Solution.");
    for (int lagrangeIndex = 0; lagrangeIndex < numGlobalConstraints; lagrangeIndex++) {
      int globalRowIndex = partMap.GID(localRowIndex);
      
      localRowIndex++;
    }
    
    // impose zero mean constraints:
    for (vector< int >::iterator trialIt = zeroMeanConstraints.begin(); trialIt != zeroMeanConstraints.end(); trialIt++) {
      int trialID = *trialIt;
      int zmcIndex = partMap.GID(localRowIndex);
      //cout << "Imposing zero-mean constraint for variable " << _mesh->bilinearForm()->trialName(trialID) << endl;
      FieldContainer<double> basisIntegrals;
      FieldContainer<int> globalIndices;
      integrateBasisFunctions(globalIndices,basisIntegrals, trialID);
      int numValues = globalIndices.size();
      
      if (zmcsAsRankOneUpdate) {
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
            product(i,j) = rho * basisIntegrals(i) * basisIntegrals(j) / denominator;
          }
        }
        globalStiffMatrix.SumIntoGlobalValues(numValues, &globalIndices(0), numValues, &globalIndices(0), &product(0,0));
      } else { // otherwise, we increase the size of the system to accomodate the zmc...
        // insert row:
        globalStiffMatrix.InsertGlobalValues(1,&zmcIndex,numValues,&globalIndices(0),&basisIntegrals(0));
        // insert column:
        globalStiffMatrix.InsertGlobalValues(numValues,&globalIndices(0),1,&zmcIndex,&basisIntegrals(0));

  //      cout << "in zmc, diagonal entry: " << rho << endl;
        //rho /= numValues;
        double rho_entry = - 1.0 / rho;
        globalStiffMatrix.InsertGlobalValues(1,&zmcIndex,1,&zmcIndex,&rho_entry);
        localRowIndex++;
      }
    }
  }
  
  timer.ResetStartTime();
  
  rhsVector.GlobalAssemble();
  
//  EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector_before_bcs.dat",rhsVector,0,0,false);
  
  globalStiffMatrix.GlobalAssemble(); // will call globalStiffMatrix.FillComplete();
  
  double timeGlobalAssembly = timer.ElapsedTime();
  Epetra_Vector timeGlobalAssemblyVector(timeMap);
  timeGlobalAssemblyVector[0] = timeGlobalAssembly;
  
//  EpetraExt::RowMatrixToMatlabFile("stiff_matrix.dat",globalStiffMatrix);
  
  // determine and impose BCs

  timer.ResetStartTime();

  FieldContainer<int> bcGlobalIndices;
  FieldContainer<double> bcGlobalValues;
  
  _mesh->boundary().bcsToImpose(bcGlobalIndices,bcGlobalValues,*(_bc.get()), myGlobalIndicesSet);
  int numBCs = bcGlobalIndices.size();
  //    cout << "bcGlobalIndices:" << endl << bcGlobalIndices;
  //    cout << "bcGlobalValues:" << endl << bcGlobalValues;

  Epetra_MultiVector v(partMap,1);
  v.PutScalar(0.0);
  for (int i = 0; i < numBCs; i++) {
    v.ReplaceGlobalValue(bcGlobalIndices(i), 0, bcGlobalValues(i));
  }
  
  Epetra_MultiVector rhsDirichlet(partMap,1);
  globalStiffMatrix.Apply(v,rhsDirichlet);
  
  // Update right-hand side
  rhsVector.Update(-1.0,rhsDirichlet,1.0);
  
  if (numBCs == 0) {
    //cout << "Solution: Warning: Imposing no BCs." << endl;
  } else {
    int err = rhsVector.ReplaceGlobalValues(numBCs,&bcGlobalIndices(0),&bcGlobalValues(0));
    if (err != 0) {
      cout << "ERROR: rhsVector.ReplaceGlobalValues(): some indices non-local...\n";
    }
  }
  // Zero out rows and columns of stiffness matrix corresponding to Dirichlet edges
  //  and add one to diagonal.
  FieldContainer<int> bcLocalIndices(bcGlobalIndices.dimension(0));
  for (int i=0; i<bcGlobalIndices.dimension(0); i++) {
    bcLocalIndices(i) = globalStiffMatrix.LRID(bcGlobalIndices(i));
  }
  if (numBCs == 0) {
    ML_Epetra::Apply_OAZToMatrix(NULL, 0, globalStiffMatrix);
  } else {
    ML_Epetra::Apply_OAZToMatrix(&bcLocalIndices(0), numBCs, globalStiffMatrix);
  }

  double timeBCImposition = timer.ElapsedTime();
  Epetra_Vector timeBCImpositionVector(timeMap);
  timeBCImpositionVector[0] = timeBCImposition;

  //cout << "MPI rank " << rank << ", numBCs: " << numBCs << endl;
  
  // solve the global matrix system..

  Epetra_FEVector lhsVector(partMap, true);
  
  // debug: check the consistency of the mesh's global -> partitionLocal index map
  for (int localIndex = partMap.MinLID(); localIndex < partMap.MaxLID(); localIndex++) {
    int globalIndex = partMap.GID(localIndex);
    if ( localIndex < myGlobalIndicesSet.size() ) { // a real dof
      int meshPartitionLocalIndex = _mesh->partitionLocalIndexForGlobalDofIndex(globalIndex);
      if (meshPartitionLocalIndex != localIndex) {
        cout << "meshPartitionLocalIndex != localIndex (" << meshPartitionLocalIndex << " != " << localIndex << ")\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
      }
      int partition = _mesh->partitionForGlobalDofIndex( globalIndex );
      if (partition != rank) {
        cout << "partition != rank (" << partition << " != " << rank << ")\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
      }
    } else {
      // a Lagrange constraint or a ZMC
      // TODO: write some check on the map here...
    }
  }
  
  Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp( new Epetra_LinearProblem(&globalStiffMatrix, &lhsVector, &rhsVector));
  
  rhsVector.GlobalAssemble();

  timer.ResetStartTime();
  solver->setProblem(problem);
  
  if (_reportConditionNumber) {
    //    double oneNorm = globalStiffMatrix.NormOne();
    double condest = conditionNumberEstimate(*problem);
    if (rank == 0) {
      // cout << "(one-norm) of global stiffness matrix: " << oneNorm << endl;
      cout << "condition # estimate for global stiffness matrix: " << condest << endl;
    }
    _globalSystemConditionEstimate = condest;
  }
  
  int solveSuccess = solver->solve();

  if (solveSuccess != 0 ) {
    cout << "**** WARNING: in Solution.solve(), solver->solve() failed with error code " << solveSuccess << ". ****\n";
  }
  
  
  double timeSolve = timer.ElapsedTime();
  Epetra_Vector timeSolveVector(timeMap);
  timeSolveVector[0] = timeSolve;
  
  timer.ResetStartTime();
  int maxLhsLength = 0;
  for (int i=0; i<numProcs; i++) {
    maxLhsLength = std::max( (int)_mesh->globalDofIndicesForPartition(i).size(), maxLhsLength );
  }
  lhsVector.GlobalAssemble();

  // Dump matrices to disk
  if (_writeMatrixToMatlabFile){
    //    EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector.dat",rhsVector,0,0,false);
    EpetraExt::RowMatrixToMatlabFile(_matrixFilePath.c_str(),globalStiffMatrix);
    //    EpetraExt::MultiVectorToMatrixMarketFile("lhs_vector.dat",lhsVector,0,0,false);
  }
  
  // Import solution onto current processor
  int numNodesGlobal = partMap.NumGlobalElements();
  Epetra_Map     solnMap(numNodesGlobal, numNodesGlobal, 0, Comm);
  Epetra_Import  solnImporter(solnMap, partMap);
  Epetra_Vector  solnCoeff(solnMap);
  solnCoeff.Import(lhsVector, solnImporter, Insert);
  
  // copy the dof coefficients into our data structure
  // get ALL element types (not just ours)-- this is a global import that we should get rid of eventually...
  elementTypes = _mesh->elementTypes();
  for ( vector< ElementTypePtr >::iterator elemTypeIt = elementTypes.begin();
       elemTypeIt != elementTypes.end(); elemTypeIt++) {
    vector< ElementPtr > elements = _mesh->elementsOfTypeGlobal(*elemTypeIt);
    vector< ElementPtr >::iterator elemIt;
    int numDofs = (*elemTypeIt)->trialOrderPtr->totalDofs();
    FieldContainer<double> elemDofs(numDofs);
    for (elemIt = elements.begin(); elemIt != elements.end(); elemIt++) {
      ElementPtr elemPtr = *(elemIt);
      int cellID = elemPtr->cellID();
      for (int dofIndex=0; dofIndex<numDofs; dofIndex++) {
        int globalIndex = _mesh->globalDofIndex(cellID, dofIndex);
        elemDofs(dofIndex) = solnCoeff[globalIndex];
      }
      _solutionForCellIDGlobal[cellID] = elemDofs;
    }
  }

  double timeDistributeSolution = timer.ElapsedTime();
  Epetra_Vector timeDistributeSolutionVector(timeMap);
  timeDistributeSolutionVector[0] = timeDistributeSolution;
  
  // DEBUGGING: print out solution coefficients
  //    for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
  //      ElementTypePtr elemTypePtr = *(elemTypeIt);
  //      cout << "solution coeffs: " << endl << _solutionForElementType[elemTypePtr.get()];
  //    }
  
  int err = timeLocalStiffnessVector.Norm1( &_totalTimeLocalStiffness );
  err = timeGlobalAssemblyVector.Norm1( &_totalTimeGlobalAssembly );
  err = timeBCImpositionVector.Norm1( &_totalTimeBCImposition );
  err = timeSolveVector.Norm1( &_totalTimeSolve );
  err = timeDistributeSolutionVector.Norm1( &_totalTimeDistributeSolution );
  
  err = timeLocalStiffnessVector.MeanValue( &_meanTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MeanValue( &_meanTimeGlobalAssembly );
  err = timeBCImpositionVector.MeanValue( &_meanTimeBCImposition );
  err = timeSolveVector.MeanValue( &_meanTimeSolve );
  err = timeDistributeSolutionVector.MeanValue( &_meanTimeDistributeSolution );
  
  err = timeLocalStiffnessVector.MinValue( &_minTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MinValue( &_minTimeGlobalAssembly );
  err = timeBCImpositionVector.MinValue( &_minTimeBCImposition );
  err = timeSolveVector.MinValue( &_minTimeSolve );
  err = timeDistributeSolutionVector.MinValue( &_minTimeDistributeSolution );
  
  err = timeLocalStiffnessVector.MaxValue( &_maxTimeLocalStiffness );
  err = timeGlobalAssemblyVector.MaxValue( &_maxTimeGlobalAssembly );
  err = timeBCImpositionVector.MaxValue( &_maxTimeBCImposition );
  err = timeSolveVector.MaxValue( &_maxTimeSolve );
  err = timeDistributeSolutionVector.MaxValue( &_maxTimeDistributeSolution );
  
  if ((rank == 0) && _reportTimingResults) {
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
  
  clearComputedResiduals(); // now that we've solved, will need to recompute residuals...
}

void Solution::clearComputedResiduals() {
  _residualsComputed = false; // now that we've solved, will need to recompute residuals...
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

Teuchos::RCP<DPGInnerProduct> Solution::ip() const { 
  return _ip;
}

Teuchos::RCP<LocalStiffnessMatrixFilter> Solution::filter() const{
  return _filter;
}

ElementTypePtr Solution::getEquivalentElementType(Teuchos::RCP<Mesh> otherMesh, ElementTypePtr elemType) {
  DofOrderingPtr otherTrial = elemType->trialOrderPtr;
  DofOrderingPtr otherTest = elemType->testOrderPtr;
  DofOrderingPtr myTrial = _mesh->getDofOrderingFactory().getTrialOrdering(*otherTrial);
  DofOrderingPtr myTest = _mesh->getDofOrderingFactory().getTestOrdering(*otherTest);
  Teuchos::RCP<shards::CellTopology> otherCellTopoPtr = elemType->cellTopoPtr;
  Teuchos::RCP<shards::CellTopology> myCellTopoPtr;
  for (int i=0; i<_mesh->activeElements().size(); i++) {
    myCellTopoPtr = _mesh->activeElements()[i]->elementType()->cellTopoPtr;
    if (myCellTopoPtr->getKey() == otherCellTopoPtr->getKey() ) {
      break; // out of for loop
    }
  }
  return _mesh->getElementTypeFactory().getElementType(myTrial,myTest,myCellTopoPtr);
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

double Solution::globalCondEstLastSolve() {
  // the condition # estimate for the last system matrix used in a solve, if _reportConditionNumber is true.
  return _globalSystemConditionEstimate;
}

void Solution::integrateBasisFunctions(FieldContainer<int> &globalIndices, FieldContainer<double> &values, int trialID) {
  // only supports scalar-valued field bases right now...
  int sideIndex = 0; // field variables only
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes();
  vector<ElementTypePtr>::iterator elemTypeIt;
  vector<int> globalIndicesVector;
  vector<double> valuesVector;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
    FieldContainer<double> valuesForType(numCellsOfType, basisCardinality);
    integrateBasisFunctions(valuesForType,elemTypePtr,trialID);
    // copy into values:
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
      int cellID = _mesh->cellID(elemTypePtr,cellIndex);
      for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++) {
        int dofIndex = elemTypePtr->trialOrderPtr->getDofIndex(trialID, dofOrdinal);
        int globalIndex = _mesh->globalDofIndex(cellID, dofIndex);
        globalIndicesVector.push_back(globalIndex);
        valuesVector.push_back(valuesForType(cellIndex,dofOrdinal));
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
  int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
  int sideIndex = 0;
  int basisCardinality = elemTypePtr->trialOrderPtr->getBasisCardinality(trialID,sideIndex);
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(0) != numCellsOfType,
                     std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(1) != basisCardinality,
                     std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis;
  trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
//  int numSides = elemTypePtr->trialOrderPtr->getNumSidesForVarID(trialID);
  
  
  int cubDegree = trialBasis->getDegree();
  
  BasisCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), *(elemTypePtr->cellTopoPtr), cubDegree);
  
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
  //FunctionSpaceTools::integrate<double>(values,*trialValuesTransformedWeighted,ones,COMP_CPP);
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
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = cells[cellIndex]->cellID();
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

double Solution::L2NormOfSolutionInCell(int trialID, int cellID) {
  double value = 0.0;
  ElementTypePtr elemTypePtr = _mesh->getElement(cellID)->elementType();
  int numCells = 1;
  // note: basisCache below will use a greater cubature degree than strictly necessary
  //       (it'll use maxTrialDegree + maxTestDegree, when it only needs maxTrialDegree * 2)
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh)); 
  
  // get cellIDs for basisCache
  vector<int> cellIDs;
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
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = cells[cellIndex]->cellID();
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
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis;
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
  int numSides = cellTopo.getSideCount();
  
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    // Get numerical integration points and weights
    DefaultCubatureFactory<double>  cubFactory;
    Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = dofOrdering.getBasis(trialID,sideIndex);
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
  
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
  
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
  const map<int,double>* energyErrorPerCell = &(energyError());
  
  for (map<int,double>::const_iterator cellEnergyIt = energyErrorPerCell->begin(); 
       cellEnergyIt != energyErrorPerCell->end(); cellEnergyIt++) {
    energyErrorSquared += (cellEnergyIt->second) * (cellEnergyIt->second);
  }
  return sqrt(energyErrorSquared);
}

const map<int,double> & Solution::energyError() { 
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
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    
    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemTypePtr);    
    
    int numCells = physicalCellNodes.dimension(0);
    int numTestDofs = testOrdering->totalDofs();
    
    TEUCHOS_TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "In computeErrorRepresentation::numCells does not match number of elems in partition.");    
    FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
    
    // determine cellIDs
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, rank);
      cellIDs.push_back(cellID);
    }
    
    ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,_ip->hasBoundaryTerms());
    
    _ip->computeInnerProductMatrix(ipMatrix,testOrdering, ipBasisCache);
    FieldContainer<double> errorRepresentation(numCells,numTestDofs);
    //    FieldContainer<double> rhsRepresentation(numCells,numTestDofs);
    
    Epetra_SerialDenseSolver solver;
    
    for (int localCellIndex=0; localCellIndex<numCells; localCellIndex++ ) {
      
      //      cout << "In compute error rep local cell ind = " << localCellIndex << ", and global cell ind = " << elemsInPartition[localCellIndex]->globalCellIndex() << endl;
      // changed to Copy from View for debugging...
      Epetra_SerialDenseMatrix ipMatrixT(Copy, &ipMatrix(localCellIndex,0,0),
                                         ipMatrix.dimension(2), // stride -- fc stores in row-major order (a.o.t. SDM)
                                         ipMatrix.dimension(2),ipMatrix.dimension(1));
      
      Epetra_SerialDenseMatrix rhs(Copy, & (_residualForElementType[elemTypePtr.get()](localCellIndex,0)),
                                   _residualForElementType[elemTypePtr.get()].dimension(1), // stride
                                   _residualForElementType[elemTypePtr.get()].dimension(1), 1);


      /*      
      int info = rhs.Reshape(numTestDofs,2); // add an extra column
      if (info!=0){
	cout << "could not reshape matrix - error code " << info << endl;
      }      
      for(int i = 0;i < numTestDofs; i++){
	rhs(i,1) = _rhsForElementType[elemTypePtr.get()](localCellIndex,i);
      }
      Epetra_SerialDenseMatrix representationMatrix(numTestDofs,2);
      */    
      
      Epetra_SerialDenseMatrix representationMatrix(numTestDofs,1);
      
      solver.SetMatrix(ipMatrixT);
      //    solver.SolveWithTranspose(true); // not that it should matter -- ipMatrix should be symmetric
      int success = solver.SetVectors(representationMatrix, rhs);
      
      if (success != 0) {
        cout << "computeErrorRepresentation: failed to SetVectors with error " << success << endl;
      }
      
      bool equilibrated = false;
      if ( solver.ShouldEquilibrate() ) {
        solver.EquilibrateMatrix();
        solver.EquilibrateRHS();
        equilibrated = true;
      }
      
      success = solver.Solve();
      
      if (success != 0) {
        cout << "computeErrorRepresentation: Solve FAILED with error: " << success << endl;
      }
      
      if (equilibrated) {
        success = solver.UnequilibrateLHS();
        if (success != 0) {
          cout << "computeErrorRepresentation: unequilibration FAILED with error: " << success << endl;
        }
      }
      
      for (int i=0; i<numTestDofs; i++) {
        errorRepresentation(localCellIndex,i) = representationMatrix(i,0);
	//        rhsRepresentation(localCellIndex,i) = representationMatrix(i,1);
      }
    }
    _errorRepresentationForElementType[elemTypePtr.get()] = errorRepresentation;
    //    _rhsRepresentationForElementType[elemTypePtr.get()] = rhsRepresentation;
  }
}

void Solution::computeResiduals() {
  
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
  //  vector< ElementPtr > elemsInPartition = _mesh->elementsInPartition(rank);    
  vector<ElementTypePtr> elemTypes = _mesh->elementTypes(rank);  
  vector<ElementTypePtr>::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    
    Teuchos::RCP<DofOrdering> trialOrdering = elemTypePtr->trialOrderPtr;
    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;

    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemTypePtr);
    
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
    FieldContainer<double> cellSideParities  = _mesh->cellSideParities(elemTypePtr);
    FieldContainer<double> solution = solutionForElementTypeGlobal(elemTypePtr);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    
    int numTrialDofs = trialOrdering->totalDofs();
    int numTestDofs  = testOrdering->totalDofs();
    int numCells = physicalCellNodes.dimension(0); // partition-local cells
    
    //    cout << "Num elems in partition " << rank << " is " << elemsInPartition.size() << endl;
    
    // determine cellIDs
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, rank);
      cellIDs.push_back(cellID);
    }
    
    TEUCHOS_TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "in computeResiduals::numCells does not match number of elems in partition.");    
    /*
      cout << "Num trial/test dofs " << rank << " is " << numTrialDofs << ", " << numTestDofs << endl;
      cout << "solution dim on " << rank << " is " << solution.dimension(0) << ", " << solution.dimension(1) << endl;
    */
      
    // set up diagonal testWeights matrices so we can reuse the existing computeRHS
//    FieldContainer<double> testWeights(numCells,numTestDofs,numTestDofs);
//    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//      for (int i=0; i<numTestDofs; i++) {
//        testWeights(cellIndex,i,i) = 1.0; 
//      }
//    }
    
    // compute l(v) and store in residuals:
    FieldContainer<double> residuals(numCells,numTestDofs);
    
    // prepare basisCache and cellIDs
    BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh,false,_cubatureEnrichmentDegree));
    bool createSideCacheToo = true;
    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
    _rhs->integrateAgainstStandardBasis(residuals, testOrdering, basisCache);
//    BilinearFormUtility::computeRHS(residuals, _mesh->bilinearForm(), *(_rhs.get()),
//                                    testWeights, testOrdering, basisCache);
//    BilinearFormUtility::computeRHS(residuals, _mesh->bilinearForm(), *(_rhs.get()), 
//                                    testWeights, testOrdering, cellTopo, physicalCellNodes);

    FieldContainer<double> rhs(numCells,numTestDofs);
    rhs = residuals; // copy rhs into its own separate container
    
    // compute b(u, v):
    FieldContainer<double> preStiffness(numCells,numTestDofs,numTrialDofs );
    _mesh->bilinearForm()->stiffnessMatrix(preStiffness, elemTypePtr, cellSideParities, basisCache);    

    // now, weight the entries in b(u,v) by the solution coefficients to compute:
    // l(v) - b(u_h,v)    
    for (int localCellIndex=0; localCellIndex<numCells; localCellIndex++) {                
      int globalCellIndex = elemsInPartitionOfType[localCellIndex]->globalCellIndex();
      //      cout << "For global cell ind = " << elemsInPartitionOfType[localCellIndex]->globalCellIndex() << " and cellID = " << elemsInPartitionOfType[localCellIndex]->cellID() << endl;          
      for (int i=0; i<numTestDofs; i++) {
        for (int j=0; j<numTrialDofs; j++) {      
          residuals(localCellIndex,i) -= solution(globalCellIndex,j) * preStiffness(localCellIndex,i,j);
        }         
      } 
    }    
    _residualForElementType[elemTypePtr.get()] = residuals;
    _rhsForElementType[elemTypePtr.get()] = rhs;
  }
  _residualsComputed = true;
}

void Solution::discardInactiveCellCoefficients() {
  vector< ElementPtr > activeElems = _mesh->activeElements();
  set< int > activeCellIDs;
  for (vector<ElementPtr >::iterator elemIt = activeElems.begin();elemIt!=activeElems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    activeCellIDs.insert(cellID);
  }
  vector<int> cellIDsToErase;
  for (map< int, FieldContainer<double> >::iterator solnIt = _solutionForCellIDGlobal.begin();
       solnIt != _solutionForCellIDGlobal.end(); solnIt++) {
    int cellID = solnIt->first;
    if ( activeCellIDs.find(cellID) == activeCellIDs.end() ) {
      cellIDsToErase.push_back(cellID);
    }
  }
  for (vector<int>::iterator it = cellIDsToErase.begin();it !=cellIDsToErase.end();it++){
    _solutionForCellIDGlobal.erase(*it);
  }
}

void Solution::solutionValuesOverCells(FieldContainer<double> &values, int trialID, const FieldContainer<double> &physicalPoints) {
  int numTotalCells = physicalPoints.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  int spaceDim = physicalPoints.dimension(2);
  for (int cellIndex=0;cellIndex<numTotalCells;cellIndex++){

    FieldContainer<double> cellPoint(1,spaceDim); // a single point to find elem we're in
    for (int i=0;i<spaceDim;i++){cellPoint(0,i) = physicalPoints(cellIndex,0,i);}
    vector< ElementPtr > elements = _mesh->elementsForPoints(cellPoint); // operate under assumption that all points for a given cell index are in that cell
    ElementPtr elem = elements[0];
    ElementTypePtr elemTypePtr = elem->elementType();
    int cellID = elem->cellID();

    if ( _solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
      // cellID not known -- default to 0
      continue;
    }

    FieldContainer<double> solnCoeffs = _solutionForCellIDGlobal[cellID];
    int numCells = 1; // do one cell at a time

    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);

    // store points in local container
    FieldContainer<double> physicalPointsForCell(numCells,numPoints,spaceDim);
    for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
      for (int dim=0; dim<spaceDim; dim++) {
        physicalPointsForCell(0,ptIndex,dim) = physicalPoints(cellIndex,ptIndex,dim);
      }
    }

    typedef CellTools<double>  CellTools;
    typedef FunctionSpaceTools fst;
  
    // 1. compute refElemPoints, the evaluation points mapped to reference cell:
    FieldContainer<double> refElemPoints(numCells,numPoints, spaceDim);
    CellTools::mapToReferenceFrame(refElemPoints,physicalPointsForCell,physicalCellNodes,*(elemTypePtr->cellTopoPtr.get()));
    refElemPoints.resize(numPoints,spaceDim);

    Teuchos::RCP<DofOrdering> trialOrder = elemTypePtr->trialOrderPtr;
    
    Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,0); // 0 assumes field var
    int basisCardinality = basis->getCardinality();

    Teuchos::RCP< FieldContainer<double> > basisValues;
    basisValues = BasisEvaluation::getValues(basis,  OP_VALUE, refElemPoints);
    
    // now, apply coefficient weights:
    for (int ptIndex=0;ptIndex<numPoints;ptIndex++){
      for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
        int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, 0); // 0 assumes field var
        values(cellIndex,ptIndex) += (*basisValues)(dofOrdinal,ptIndex) * solnCoeffs(localDofIndex);
      }
    }
  }
}

void Solution::solutionValues(FieldContainer<double> &values, int trialID, BasisCachePtr basisCache, 
                              bool weightForCubature, EOperatorExtended op) {
  values.initialize(0.0);
  vector<int> cellIDs = basisCache->cellIDs();
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
    int cellID = cellIDs[cellIndex];
    
    if ( _solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
      // cellID not known -- default values for that cell to 0
      continue;
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
    
//    cout << "solnCoeffs:\n" << solnCoeffs;
    
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
  if (physicalPoints.rank()==3) { // if we have dimensions (C,P,D), call a different method
    solutionValuesOverCells(values, trialID, physicalPoints);
    return;
  } else {

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
  vector< ElementPtr > elements = _mesh->elementsForPoints(physicalPoints);
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
    if ( _solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
      // cellID not known -- default to 0
      continue;
    }
    FieldContainer<double> solnCoeffs = _solutionForCellIDGlobal[cellID];
    
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
    
    int sideIndex = 0; // field variable assumed
    
    Teuchos::RCP<DofOrdering> trialOrder = elemTypePtr->trialOrderPtr;
    
    Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
    int basisRank = trialOrder->getBasisRank(trialID);
    int basisCardinality = basis->getCardinality();

    TEUCHOS_TEST_FOR_EXCEPTION( ( basisRank==0 ) && values.rank() != 1,
                       std::invalid_argument,
                       "for scalar values, values container should be dimensioned(numPoints).");
    TEUCHOS_TEST_FOR_EXCEPTION( ( basisRank==1 ) && values.rank() != 2,
                       std::invalid_argument,
                       "for scalar values, values container should be dimensioned(numPoints,spaceDim).");
    TEUCHOS_TEST_FOR_EXCEPTION( basisRank==1 && values.dimension(1) != spaceDim,
                       std::invalid_argument,
                       "vector values.dimension(1) != spaceDim.");
    TEUCHOS_TEST_FOR_EXCEPTION( physicalPoints.rank() != 2,
                       std::invalid_argument,
                       "physicalPoints.rank() != 2.");
    TEUCHOS_TEST_FOR_EXCEPTION( physicalPoints.dimension(1) != spaceDim,
                       std::invalid_argument,
                       "physicalPoints.dimension(1) != spaceDim.");
    TEUCHOS_TEST_FOR_EXCEPTION( _mesh->bilinearForm()->isFluxOrTrace(trialID),
                       std::invalid_argument,
                       "call the other solutionValues (with sideCellRefPoints argument) for fluxes and traces.");
    
    Teuchos::RCP< FieldContainer<double> > basisValues;
    basisValues = BasisEvaluation::getValues(basis,  OP_VALUE, refElemPoint);
    
    // now, apply coefficient weights:
    for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
      int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
      if (basisRank == 0) {
        values(physicalPointIndex) += (*basisValues)(dofOrdinal,0) * solnCoeffs(localDofIndex);
      } else {
        for (int i=0; i<spaceDim; i++) {
          values(physicalPointIndex,i) += (*basisValues)(dofOrdinal,0,i) * solnCoeffs(localDofIndex);
        }
      }
    }
  }
  }
}

void Solution::solutionValues(FieldContainer<double> &values, 
                              ElementTypePtr elemTypePtr, 
                              int trialID,
                              const FieldContainer<double> &physicalPoints) {
  int sideIndex = 0;
  // currently, we only support computing solution values on all the cells of a given type at once.
  // values(numCellsForType,numPoints[,spaceDim (for vector-valued)])
  // physicalPoints(numCellsForType,numPoints,spaceDim)
  FieldContainer<double> solnCoeffs = solutionForElementTypeGlobal(elemTypePtr); // (numcells, numLocalTrialDofs)
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
  // 1. Map the physicalPoints from the elements specified in physicalCellNodes into reference element
  // 2. Compute each basis on those points
  // 3. Transform those basis evaluations back into the physical space
  // 4. Multiply by the solnCoeffs
    
  int numCells = physicalCellNodes.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr.get());
  int spaceDim = cellTopo.getDimension();
    
  // TODO: add TEUCHOS_TEST_FOR_EXCEPTIONs to make sure all the FC dimensions/ranks agree with expectations
  // TODO: work out what to do for boundary.  (May need to separate out into separate function: how do we
  //       figure out what side we're on, e.g.?)
  typedef CellTools<double>  CellTools;
  typedef FunctionSpaceTools fst;
  
  // 1. compute refElemPoints, the evaluation points mapped to reference cell:
  FieldContainer<double> refElemPoints(numCells, numPoints, spaceDim);
  CellTools::mapToReferenceFrame(refElemPoints,physicalPoints,physicalCellNodes,cellTopo);
    
  //  cout << "physicalCellNodes: " << endl << physicalCellNodes;
  //  cout << "physicalPoints: " << endl << physicalPoints;
//  cout << "refElemPoints: " << endl << refElemPoints;
  
  // Containers for Jacobian
  FieldContainer<double> cellJacobian(numCells, numPoints, spaceDim, spaceDim);
  FieldContainer<double> cellJacobInv(numCells, numPoints, spaceDim, spaceDim);
  FieldContainer<double> cellJacobDet(numCells, numPoints);
  
  CellTools::setJacobian(cellJacobian, refElemPoints, physicalCellNodes, cellTopo);
  CellTools::setJacobianInv(cellJacobInv, cellJacobian );
  CellTools::setJacobianDet(cellJacobDet, cellJacobian );
  
  Teuchos::RCP<DofOrdering> trialOrder = elemTypePtr->trialOrderPtr;
  
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
    
  int basisRank = trialOrder->getBasisRank(trialID);
  int basisCardinality = basis->getCardinality();
  //cout << "num Cells = " << numCells << endl;
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
  TEUCHOS_TEST_FOR_EXCEPTION( _mesh->bilinearForm()->isFluxOrTrace(trialID),
		      std::invalid_argument,
		      "call the other solutionValues (with sideCellRefPoints argument) for fluxes and traces.");
  
  FieldContainer<double> thisCellJacobian(1,numPoints, spaceDim, spaceDim);
  FieldContainer<double> thisCellJacobInv(1,numPoints, spaceDim, spaceDim);
  FieldContainer<double> thisCellJacobDet(1,numPoints);
  FieldContainer<double> thisRefElemPoints(numPoints,spaceDim);
  
  values.initialize(0.0);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//    cout << "cellIndex: " << cellIndex << endl;
    thisRefElemPoints.setValues(&refElemPoints(cellIndex,0,0),numPoints*spaceDim);
//    cout << "line 5.01" << endl;
    thisCellJacobian.setValues(&cellJacobian(cellIndex,0,0,0),numPoints*spaceDim*spaceDim);
//    cout << "line 5.02" << endl;
    thisCellJacobInv.setValues(&cellJacobInv(cellIndex,0,0,0),numPoints*spaceDim*spaceDim);
//    cout << "line 5.03" << endl;
    thisCellJacobDet.setValues(&cellJacobDet(cellIndex,0),numPoints);
//    cout << "line 5.1" << endl;
    Teuchos::RCP< FieldContainer<double> > transformedValues;
    transformedValues = BasisEvaluation::getTransformedValues(basis,  OP_VALUE, 
                                                              thisRefElemPoints, thisCellJacobian, 
                                                              thisCellJacobInv, thisCellJacobDet);
    
    //    cout << "cellIndex " << cellIndex << " thisRefElemPoints: " << thisRefElemPoints;
    //    cout << "cellIndex " << cellIndex << " transformedValues: " << *transformedValues;
    
    // now, apply coefficient weights:
    for (int ptIndex=0; ptIndex < numPoints; ptIndex++) {
      // local storage to accumulate solution values to:
      double value = 0.0;
      double vectorValue[spaceDim];
      for (int d=0; d<spaceDim; d++) {
        vectorValue[spaceDim] = 0.0;
      }
      for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
        int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
        //        cout << "localDofIndex " << localDofIndex << " solnCoeffs(cellIndex,localDofIndex): " << solnCoeffs(cellIndex,localDofIndex) << endl;
        if (basisRank == 0) {
          // for watching in the debugger:
          double basisValue = (*transformedValues)(0,dofOrdinal,ptIndex);
          double weight = solnCoeffs(cellIndex,localDofIndex);
          value += weight * basisValue;
        } else {
          for (int i=0; i<spaceDim; i++) {
            vectorValue[i] += (*transformedValues)(0,dofOrdinal,ptIndex,i) * solnCoeffs(cellIndex,localDofIndex);
          }
        }
      }
      if (basisRank == 0) {
        values(cellIndex,ptIndex) = value;
      } else {
        for (int i=0; i<spaceDim; i++) {
          values(cellIndex,ptIndex,i) = vectorValue[i];
        }
      }
    }
  }
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
    
    FieldContainer<double> vertexPoints, physPoints;
    
    _mesh->verticesForElementType(vertexPoints,elemTypePtr);
    
    int numCells = vertexPoints.dimension(0);
    int numVertices = vertexPoints.dimension(1);
    
    int sideIndex = 0;
    int basisDegree = elemTypePtr->trialOrderPtr->getBasis(trialID,sideIndex)->getDegree();
    int numDivisionsPerEdge = 1; //basisDegree*basisDegree;
    
    int numPatchesPerCell = 1;
    
    if (numVertices == 4) { // only quads supported by the multi-patch cell stuff below...
      //if (   ( elemTypePtr->cellTopoPtr->getKey() == shards::Quadrilateral<4>::key )
      //    || (elemTypePtr->cellTopoPtr->getKey() == shards::Triangle<3>::key ) ) {
      
      numPatchesPerCell = numDivisionsPerEdge*numDivisionsPerEdge;
      
      physPoints.resize(numCells,numPatchesPerCell*numVertices,spaceDim);
      
      FieldContainer<double> iVertex(spaceDim), jVertex(spaceDim);
      FieldContainer<double> v1(spaceDim), v2(spaceDim), v3(spaceDim);
      
      if (numVertices == 4) {     
        double yWeights[numVertices], xWeights[numVertices];
        for (int i=0; i<numDivisionsPerEdge; i++) {
          for (int j=0; j<numDivisionsPerEdge; j++) {
            //cout << "weights: " << xWeights[0]*yWeights[0] << " " << xWeights[1]*yWeights[1] << " " << xWeights[2]*yWeights[2] << " " << xWeights[3]*yWeights[3] << "\n";
            int patchIndex = (i*numDivisionsPerEdge + j);
            
            for (int cellIndex=0; cellIndex < numCells; cellIndex++) {
              for (int patchVertexIndex=0; patchVertexIndex < numVertices; patchVertexIndex++) {
                int xOffset = ((patchVertexIndex==0) || (patchVertexIndex==3)) ? 0 : 1;
                int yOffset = ((patchVertexIndex==0) || (patchVertexIndex==1)) ? 0 : 1;
                determineQuadEdgeWeights(xWeights,i+xOffset,numDivisionsPerEdge,true);
                determineQuadEdgeWeights(yWeights,j+yOffset,numDivisionsPerEdge,false);
                
                for (int dim=0; dim<spaceDim; dim++) {
                  physPoints(cellIndex,patchIndex*numVertices + patchVertexIndex, dim) = 0.0;
                }
                for (int vertexIndex=0; vertexIndex < numVertices; vertexIndex++) {
                  double weight = xWeights[vertexIndex] * yWeights[vertexIndex];
                  //cout << "weight for vertex " << vertexIndex << ": " << weight << endl;
                  for (int dim=0; dim<spaceDim; dim++) {
                    physPoints(cellIndex,patchIndex*numVertices + patchVertexIndex, dim) += weight*vertexPoints(cellIndex, vertexIndex, dim);
                  }
                }
                for (int dim=0; dim<spaceDim; dim++) {
                  //cout << "physPoints(cellIndex, " << patchIndex*numVertices << " + " << patchVertexIndex << "," << dim << "): ";
                  //cout << physPoints(cellIndex,patchIndex*numVertices + patchVertexIndex, dim) << "\n";
                }
              }
            }
          }
        }
      }
      
    } else {
      physPoints = vertexPoints;
      numPatchesPerCell = 1;
    }
    
    FieldContainer<double> values(numCells, numPatchesPerCell * numVertices);
    solutionValues(values,elemTypePtr,trialID,physPoints);
    
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
    
    // compute physicalCubaturePoints, the transformed cubature points on each cell:
    FieldContainer<double> physCubPoints(numCellsOfType, numCubPoints, spaceDim);
    CellTools::mapToPhysicalFrame(physCubPoints,cubPoints,_mesh->physicalCellNodesGlobal(elemTypePtr),*(cellTopoPtr.get()));
    
    FieldContainer<double> values(numCellsOfType, numCubPoints);
    solutionValues(values,elemTypePtr,trialID,physCubPoints);
    
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
      if ( _solutionForCellIDGlobal.find(cellID) != _solutionForCellIDGlobal.end() ) {
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
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
  
  int basisCardinality = basis->getCardinality();
  basisCoeffs.resize(basisCardinality);
  
  for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
    int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
    basisCoeffs(dofOrdinal) = allCoeffs(localDofIndex);
  }
}

void Solution::solnCoeffsForCellID(FieldContainer<double> &solnCoeffs, int cellID, int trialID, int sideIndex) {
  Teuchos::RCP< DofOrdering > trialOrder = _mesh->getElement(cellID)->elementType()->trialOrderPtr;
  
  if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
    cout << "Warning: solution for cellID " << cellID << " not found; returning 0.\n";
    Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
    int basisCardinality = basis->getCardinality();
    solnCoeffs.resize(basisCardinality);
    solnCoeffs.initialize();
    return;
  }
  
  basisCoeffsForTrialOrder(solnCoeffs, trialOrder, _solutionForCellIDGlobal[cellID], trialID, sideIndex);
}

const FieldContainer<double>& Solution::allCoefficientsForCellID(int cellID) {
  return _solutionForCellIDGlobal[cellID];
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

void Solution::setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, int cellID){
  _solutionForCellIDGlobal[cellID] = solnCoeffsToSet;
}

void Solution::setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, int cellID, int trialID, int sideIndex) {
  ElementTypePtr elemTypePtr = _mesh->elements()[cellID]->elementType();
  
  Teuchos::RCP< DofOrdering > trialOrder = elemTypePtr->trialOrderPtr;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
  
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
  // could stand to be more granular, maybe, but if we're changing the solution, the present
  // policy is to invalidate any computed residuals
  clearComputedResiduals();
}

// protected method; used for solution comparison...
const map< int, FieldContainer<double> > & Solution::solutionForCellIDGlobal() const {
  return _solutionForCellIDGlobal;
}

// Jesse's additions below:

// =================================== CONDENSED SOLVE ======================================

void Solution::condensedSolve(){
  bool saveMemory = false;
  condensedSolve(saveMemory);
}
void Solution::condensedSolve(bool saveMemory){
 
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

  Epetra_Time totalTime(Comm);
  Epetra_Time timer(Comm);
  Epetra_Time elemTimer(Comm);
    
  totalTime.ResetStartTime();

  timer.ResetStartTime();
  map<int, FieldContainer<double> > elemMatrices; // map from cellID to local stiffness matrix
  map<int, FieldContainer<double> > elemLoads; // map from cellID to local rhs

  // get local dof indices for reduced matrices
  map<int,set<int> > localFluxInds, localFieldInds;  
  map<int,set<int> > globalFluxInds, globalFieldInds;  
  vector<int> trialIDs = mesh()->bilinearForm()->trialIDs();
  vector< ElementPtr > allElems = _mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;     
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    set<int> localFluxDofs,localFieldDofs,globalFluxDofs,globalFieldDofs;
      
    for (vector<int>::iterator idIt = trialIDs.begin();idIt!=trialIDs.end();idIt++){     
      int trialID = *idIt;
      int numSides = elem->elementType()->trialOrderPtr->getNumSidesForVarID(trialID);
      vector<int> dofInds;
      if (numSides==1) { // volume element
	dofInds = elem->elementType()->trialOrderPtr->getDofIndices(trialID, 0);
      } else {	
	for (int sideIndex=0;sideIndex<numSides;sideIndex++){
	  vector<int> inds =  elem->elementType()->trialOrderPtr->getDofIndices(trialID, sideIndex);
	  dofInds.insert(dofInds.end(),inds.begin(),inds.end());
	}
      }
      for (vector<int>::iterator dofIt = dofInds.begin();dofIt!=dofInds.end();dofIt++){
	int localDofIndex = *dofIt;
	int globalDofIndex = mesh()->globalDofIndex(cellID, localDofIndex);
	if (mesh()->bilinearForm()->isFluxOrTrace(trialID)){
	  // is flux ID
	  localFluxDofs.insert(localDofIndex);
	  globalFluxDofs.insert(globalDofIndex);
	}else{
	  // is field ID
	  localFieldDofs.insert(localDofIndex);
	  globalFieldDofs.insert(globalDofIndex);
	}	      
      }
    }
    localFluxInds[cellID]=localFluxDofs;
    localFieldInds[cellID]=localFieldDofs;
    globalFluxInds[cellID]=globalFluxDofs;
    globalFieldInds[cellID]=globalFieldDofs;
  }

  //  _mesh->getFieldFluxDofInds(localFluxInds, localFieldInds);
  //  _mesh->getGlobalFieldFluxDofInds(globalFluxInds, globalFieldInds);

  // build maps/list of all flux inds
  map<int,map<int,int> > localToCondensedMap;
  map<int,map<int,int> > globalToLocalMap;
  set<int> allFluxInds;  
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    set<int> fluxInds = localFluxInds[cellID];
    set<int>::iterator setIt;
    int i = 0;
    for (setIt=fluxInds.begin();setIt!=fluxInds.end();setIt++){
      int localInd = *setIt;
      int globalInd = _mesh->globalDofIndex(cellID,localInd);
      allFluxInds.insert(globalInd);
      localToCondensedMap[cellID][localInd] = i;
      globalToLocalMap[cellID][globalInd] = localInd;
      i++;
    }
  }

  // build global-condensed map
  map<int,int> globalToCondensedMap;
  map<int,int> condensedToGlobalMap;
  set<int>::iterator setIt;
  int i = 0;
  for (setIt=allFluxInds.begin();setIt!=allFluxInds.end();setIt++){
    int globalFluxInd = *setIt;
    globalToCondensedMap[globalFluxInd] = i; 
    condensedToGlobalMap[i] = globalFluxInd;
    i++;
  }
  if (_reportTimingResults){
    cout << "on rank " << rank << ", time to build dof maps = " << timer.ElapsedTime() << endl;
  }
  timer.ResetStartTime();

  // create partitioning - CAN BE MORE EFFICIENT if necessary
  set<int> myGlobalDofInds = _mesh->globalDofIndicesForPartition(rank);
  set<int> myGlobalFluxInds;
  for (setIt=myGlobalDofInds.begin();setIt!=myGlobalDofInds.end();setIt++){
    int ind = (*setIt);
    //    cout << "global dof ind = " << ind << endl;
    if (allFluxInds.find(ind) != allFluxInds.end()){
      myGlobalFluxInds.insert(globalToCondensedMap[ind]); // need in condensed indices
      //      cout << "global flux ind = " << globalToCondensedMap[ind] << endl;
    }
  }
  if (_reportTimingResults){
    cout << "on rank " << rank << ", time to form dof partitions = " << timer.ElapsedTime() << endl;
  }  
  timer.ResetStartTime();

  int numGlobalFluxDofs = _mesh->numFluxDofs();
  Epetra_Map fluxPartMap = getPartitionMap(rank, myGlobalFluxInds, numGlobalFluxDofs, 0, &Comm);
 
  if (_reportTimingResults){
    cout << "on rank " << rank << ", time to form partition maps = " << timer.ElapsedTime() << endl;
  }  
  timer.ResetStartTime();

  // size/create stiffness matrix
  int maxNnzPerRow = min(_mesh->condensedRowSizeUpperBound(),numGlobalFluxDofs);
  Epetra_FECrsMatrix K_cond(Copy, fluxPartMap, maxNnzPerRow); // condensed system
  Epetra_FEVector rhs_cond(fluxPartMap);
  Epetra_FEVector lhs_cond(fluxPartMap, true);
  
  if (_reportTimingResults){
    cout << "on rank " << rank << ", time to init condensed stiffness matrices = " << timer.ElapsedTime() << endl;
  }
  timer.ResetStartTime();

  vector< ElementPtr > elems = _mesh->elementsInPartition(rank);
  // loop thru elems
  double localStiffnessTime = 0.0;
  double condensationTime = 0.0;
  double assemblyTime = 0.0;
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    elemTimer.ResetStartTime();

    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    set<int> fieldInds = localFieldInds[cellID];
    set<int> fluxInds = localFluxInds[cellID];
    int numElemFluxDofs = fluxInds.size();
    int numElemFieldDofs = fieldInds.size();

    // get elem data and submatrix data
    FieldContainer<double> K,rhs;
    getElemData(elem,K,rhs);
    if (!saveMemory){
      elemMatrices[cellID] = K;
      elemLoads[cellID] = rhs;
    }
    Epetra_SerialDenseMatrix D, B, K_flux;
    getSubmatrices(fieldInds, fluxInds, K, D, B, K_flux);
    
    localStiffnessTime += elemTimer.ElapsedTime();
    elemTimer.ResetStartTime();

    // reduce matrix
    Epetra_SerialDenseMatrix Bcopy = B;
    Epetra_SerialDenseSolver solver;
    //    cout << "num elem field dofs, flux dofs = " << numElemFieldDofs << ", " << numElemFluxDofs << endl;
    Epetra_SerialDenseMatrix DinvB(numElemFieldDofs,numElemFluxDofs);
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
    Epetra_SerialDenseVector Dinvf(numElemFieldDofs);
    Epetra_SerialDenseVector BtDinvf(numElemFluxDofs);
    Epetra_SerialDenseVector b_field, b_flux;
    getSubvectors(fieldInds, fluxInds, rhs, b_field, b_flux);
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

    // create vector of dof indices (and global-local maps)
    Epetra_IntSerialDenseVector condensedFluxInds(numElemFluxDofs);
    set<int>::iterator indIt;
    int i = 0;
    for (indIt = fluxInds.begin();indIt!=fluxInds.end();indIt++){
      int localFluxInd = *indIt;
      int globalFluxInd = _mesh->globalDofIndex(cellID, localFluxInd);
      int condensedInd = globalToCondensedMap[globalFluxInd];
      condensedFluxInds(i) = condensedInd;
      i++;
    }
    
    condensationTime += elemTimer.ElapsedTime();

    /*
    // Impose BCs - WARNING: *may* not work with crack BCs b/c they're shared. or maybe it will.
    vector<int> bcGlobalIndices = bcIndices[cellID];
    vector<double> bcGlobalValues = bcValues[cellID];
    int numBCs = bcGlobalIndices.size();   
    if (numBCs > 0){
      // get BC values
      Epetra_SerialDenseVector bc_lift_dofs(numElemFluxDofs);
      for (int i = 0;i<numBCs;i++){
	int localInd = globalToLocalMap[cellID][bcGlobalIndices[i]];
	int condensedLocalInd = localToCondensedMap[cellID][localInd];
	bc_lift_dofs(condensedLocalInd) = bcGlobalValues[i];
      }
    
      // multiply bc dofs by elem matrix - rhsDirichlet stores resulting "lift" and update rhs (subtract from RHS)
      b_flux.Multiply('N','N',-1.0,K_flux,bc_lift_dofs,1.0);   

      // Zero out rows and columns of stiffness matrix corresponding to Dirichlet edges, 1 on diag
      for (int i = 0;i<numBCs;i++){      
	int globalInd = bcGlobalIndices[i];
	int ind = localToCondensedMap[cellID][globalToLocalMap[cellID][globalInd]];
	// zero out row/column
	for (int j = 0;j<numElemFluxDofs;j++){
	  K_flux(ind,j) = 0.0;
	  K_flux(j,ind) = 0.0;
	}      
	K_flux(ind,ind) = 1.0;
	b_flux(ind) = bcGlobalValues[i];
      }
    } // end of bc IF loop
    */

    // sum into FE matrices - all that's left is applying BCs
    elemTimer.ResetStartTime();

    rhs_cond.SumIntoGlobalValues(numElemFluxDofs, condensedFluxInds.Values(), b_flux.A()); 
    K_cond.InsertGlobalValues(numElemFluxDofs, condensedFluxInds.Values(), K_flux.A());    

    assemblyTime += elemTimer.ElapsedTime();

  }  
  if (_reportTimingResults){
    cout << "on rank " << rank << ", in loop, time to get local stiff = " << localStiffnessTime << endl;
    cout << "on rank " << rank << ", in loop, time to reduce matrix = " << condensationTime << endl;
    cout << "on rank " << rank << ", in loop, time to assemble into global = " << assemblyTime << endl;
  }

  // globally assemble flux matrices
  K_cond.GlobalAssemble();K_cond.FillComplete();
  rhs_cond.GlobalAssemble();

  if (_reportTimingResults)
    cout << "on rank " << rank << ", time for assembly = " << timer.ElapsedTime() << endl;
  
  // ============= MORE EFFICIENT WAY TO APPLY BCS in nate's code ===================
  
  timer.ResetStartTime();

  // getting local flux rows
  set<int> myFluxInds;
  int numGlobalElements = fluxPartMap.NumMyElements();
  int * myGlobalInds = fluxPartMap.MyGlobalElements();
  for (int i = 0;i<numGlobalElements;i++){
    myFluxInds.insert(condensedToGlobalMap[myGlobalInds[i]]); // form flux inds of global system
  }  

  // applying BCs
  FieldContainer<int> bcGlobalIndices;
  FieldContainer<double> bcGlobalValues;
  _mesh->boundary().bcsToImpose(bcGlobalIndices,bcGlobalValues,*(_bc.get()), myFluxInds);
  int numBCs = bcGlobalIndices.size();

  // get lift of the operator corresponding to BCs, modify RHS accordingly
  Epetra_MultiVector u0(fluxPartMap,1);
  u0.PutScalar(0.0);
  for (int i = 0; i < numBCs; i++) {
    u0.ReplaceGlobalValue(globalToCondensedMap[bcGlobalIndices(i)], 0, bcGlobalValues(i));
  }
  Epetra_MultiVector rhs_lift(fluxPartMap,1);
  K_cond.Apply(u0,rhs_lift);
  rhs_cond.Update(-1.0,rhs_lift,1.0);
  
  for (int i = 0; i < numBCs; i++) {
    int ind = globalToCondensedMap[bcGlobalIndices(i)];
    double value = bcGlobalValues(i);
    rhs_cond.ReplaceGlobalValues(1,&ind,&value);
  }
  //  cout << "on proc " << rank << ", applying oaz with " << numBCs << " bcs" << endl;
  // Zero out rows and columns of K corresponding to BC dofs, and add one to diagonal.
  FieldContainer<int> bcLocalIndices(bcGlobalIndices.dimension(0));
  for (int i=0; i<bcGlobalIndices.dimension(0); i++) {
    bcLocalIndices(i) = K_cond.LRID(globalToCondensedMap[bcGlobalIndices(i)]);
  }
  if (numBCs == 0) {
    ML_Epetra::Apply_OAZToMatrix(NULL, 0, K_cond);
  }else{
    ML_Epetra::Apply_OAZToMatrix(&bcLocalIndices(0), numBCs, K_cond);
  }
  if (_reportTimingResults){
    cout << "on rank " << rank << ", time for applying BCs = " << timer.ElapsedTime() << endl;
  }

  // ============= /END BCS ===================  
  
  if (_writeMatrixToMatlabFile){
    EpetraExt::RowMatrixToMatlabFile(_matrixFilePath.c_str(),K_cond);     
  }
  //  EpetraExt::MultiVectorToMatrixMarketFile("rhs_cond.dat",rhs_cond,0,0,false);

  timer.ResetStartTime();

  // solve reduced problem
  Teuchos::RCP<Epetra_LinearProblem> problem_cond = Teuchos::rcp( new Epetra_LinearProblem(&K_cond, &lhs_cond, &rhs_cond));
  rhs_cond.GlobalAssemble();

  bool useIterativeSolver = false;
  if (!useIterativeSolver){
    Teuchos::RCP<Solver> solver = Teuchos::rcp(new KluSolver()); // default to KLU for now - most stable
    solver->setProblem(problem_cond);    
    solver->solve();
  }else{
    // create the preconditioner object and compute hierarchy
    ML_Epetra::MultiLevelPreconditioner * MLPrec = 
      new ML_Epetra::MultiLevelPreconditioner(K_cond, true);

    AztecOO Solver((*problem_cond));
    Solver.SetPrecOperator(MLPrec);
    Solver.SetAztecOption(AZ_solver,AZ_cg);
    Solver.SetAztecOption(AZ_output,AZ_last);
    //    Solver.SetAztecOption(AZ_precond, AZ_Jacobi);
    int maxIter = round(numGlobalFluxDofs/4);
    Solver.Iterate(maxIter,1E-9); 
  }
  lhs_cond.GlobalAssemble();  

  if (_reportTimingResults){
    cout << "on rank " << rank << ", time for solve = " << timer.ElapsedTime() << endl;
  }

  timer.ResetStartTime();
  
  // define global dof vector to distribute
  int numGlobalDofs = _mesh->numGlobalDofs();
  set<int> myGlobalIndicesSet = _mesh->globalDofIndicesForPartition(rank);
  Epetra_Map partMap = getPartitionMap(rank, myGlobalIndicesSet,numGlobalDofs, 0,&Comm); // no zmc
  Epetra_FEVector lhs_all(partMap, true);

  // import condensed solution onto all processors
  Epetra_Map     solnMap(numGlobalFluxDofs, numGlobalFluxDofs, 0, Comm);
  Epetra_Import  solnImporter(solnMap, fluxPartMap);
  Epetra_Vector  all_flux_coeffs(solnMap);
  all_flux_coeffs.Import(lhs_cond, solnImporter, Insert);  

  if (_reportTimingResults){
    cout << "on rank " << rank << ", time for distribution of flux dofs = " << timer.ElapsedTime() << endl;
  }

  timer.ResetStartTime();

  // recover field dofs
  for (elemIt=elems.begin();elemIt!=elems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    set<int> fieldInds = localFieldInds[cellID];
    set<int> fluxInds = localFluxInds[cellID];
    int numElemFluxDofs = fluxInds.size();
    int numElemFieldDofs = fieldInds.size();

    // create vector of dof indices (and global/local maps)
    Epetra_SerialDenseVector flux_dofs(numElemFluxDofs);    
    Epetra_IntSerialDenseVector inds(numElemFluxDofs);
    set<int>::iterator indIt;
    int i = 0;
    for (indIt = fluxInds.begin();indIt!=fluxInds.end();indIt++){
      int localFluxInd = *indIt;
      int globalFluxInd = _mesh->globalDofIndex(cellID, localFluxInd);
      //      inds(i) = globalFluxInd;
      double value = all_flux_coeffs[globalToCondensedMap[globalFluxInd]];
      flux_dofs(i) = value;    
      i++;
    }
    //    lhs_all.ReplaceGlobalValues(inds, flux_dofs); // store flux dofs in multivector while we're at it

    // get elem data and submatrix data
    FieldContainer<double> K,rhs;
    if (saveMemory){
      getElemData(elem,K,rhs);
    }else{
      K = elemMatrices[cellID];
      rhs = elemLoads[cellID];
    }
    Epetra_SerialDenseMatrix D, B, fluxMat;
    Epetra_SerialDenseVector b_field, b_flux, field_dofs(numElemFieldDofs);
    getSubmatrices(fieldInds, fluxInds, K, D, B, fluxMat);
    getSubvectors(fieldInds, fluxInds, rhs, b_field, b_flux);
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

    Epetra_IntSerialDenseVector globalFieldInds(numElemFieldDofs);
    i = 0;
    for (indIt = fieldInds.begin();indIt!=fieldInds.end();indIt++){
      double value = field_dofs(i);
      int localFieldInd = *indIt;
      int globalFieldInd = _mesh->globalDofIndex(cellID,localFieldInd);
      globalFieldInds(i) = globalFieldInd;
      lhs_all.ReplaceGlobalValues(1, &globalFieldInd, &value); // store field dofs       
      i++;
    }								   
    //    lhs_all.ReplaceGlobalValues(globalFieldInds, field_dofs); // store field dofs       
  }
  
  // globally assemble LHS and import to all procs (WARNING: INEFFICIENT) 
  lhs_all.GlobalAssemble();

  if (_reportTimingResults){
    cout << "on rank " << rank << ", time for recovery of field dofs = " << timer.ElapsedTime() << endl;
  }
  timer.ResetStartTime();

  // finally store flux dofs - don't want to sum them up
  for (setIt=allFluxInds.begin();setIt!=allFluxInds.end();setIt++){
    int globalFluxInd = *setIt;
    double value = all_flux_coeffs[globalToCondensedMap[globalFluxInd]];
    lhs_all.ReplaceGlobalValues(1, &globalFluxInd, &value); 
  }

  Epetra_Map     fullSolnMap(numGlobalDofs, numGlobalDofs, 0, Comm);
  Epetra_Import  fullSolnImporter(fullSolnMap, partMap);
  Epetra_Vector  all_coeffs(fullSolnMap);
  all_coeffs.Import(lhs_all, fullSolnImporter, Insert);  

  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    ElementTypePtr elemTypePtr = elem->elementType();
    int numDofs = elemTypePtr->trialOrderPtr->totalDofs();
    FieldContainer<double> elemDofs(numDofs);
    for (int i = 0;i<numDofs;i++){
      int globalDofIndex = _mesh->globalDofIndex(cellID,i);
      elemDofs(i) = all_coeffs[globalDofIndex];
    }
    _solutionForCellIDGlobal[cellID] = elemDofs;    
  }  
  if (_reportTimingResults){
    cout << "on rank " << rank << ", time for storage of all dofs = " << timer.ElapsedTime();
    cout << ", and total time spent in solve = " << totalTime.ElapsedTime() << endl;
  }

  clearComputedResiduals();
}

void Solution::getSubmatrices(set<int> fieldInds, set<int> fluxInds, const FieldContainer<double> K, Epetra_SerialDenseMatrix &K_field, Epetra_SerialDenseMatrix &K_coupl, Epetra_SerialDenseMatrix &K_flux){
  int numFieldDofs = fieldInds.size();
  int numFluxDofs = fluxInds.size();
  K_field.Reshape(numFieldDofs,numFieldDofs);
  K_flux.Reshape(numFluxDofs,numFluxDofs);
  K_coupl.Reshape(numFieldDofs,numFluxDofs); // upper right hand corner matrix - symmetry gets the other

  set<int>::iterator dofIt1;
  set<int>::iterator dofIt2;

  int i,j,j_flux,j_field;
  i = 0;
  for (dofIt1 = fieldInds.begin();dofIt1!=fieldInds.end();dofIt1++){
    int rowInd = *dofIt1;
    j_flux = 0;
    j_field = 0;

    // get block field matrices
    for (dofIt2 = fieldInds.begin();dofIt2!=fieldInds.end();dofIt2++){      
      int colInd = *dofIt2;
      //      cout << "rowInd, colInd = " << rowInd << ", " << colInd << endl;
      K_field(i,j_field) = K(0,rowInd,colInd);
      j_field++;
    }    

    // get field/flux couplings
    for (dofIt2 = fluxInds.begin();dofIt2!=fluxInds.end();dofIt2++){
      int colInd = *dofIt2;      
      K_coupl(i,j_flux) = K(0,rowInd,colInd);
      j_flux++;
    }    
    i++;
  }

  // get flux coupling terms
  i = 0;
  for (dofIt1 = fluxInds.begin();dofIt1!=fluxInds.end();dofIt1++){
    int rowInd = *dofIt1;
    j = 0;
    for (dofIt2 = fluxInds.begin();dofIt2!=fluxInds.end();dofIt2++){
      int colInd = *dofIt2;
      K_flux(i,j) = K(0,rowInd,colInd);
      j++;
    }
    i++;
  }
}

void Solution::getSubvectors(set<int> fieldInds, set<int> fluxInds, const FieldContainer<double> b, Epetra_SerialDenseVector &b_field, Epetra_SerialDenseVector &b_flux){
  
  int numFieldDofs = fieldInds.size();
  int numFluxDofs = fluxInds.size();

  b_field.Resize(numFieldDofs);
  b_flux.Resize(numFluxDofs);
  set<int>::iterator dofIt;
  int i;
  i = 0;  
  for (dofIt=fieldInds.begin();dofIt!=fieldInds.end();dofIt++){
    int ind = *dofIt;
    b_field(i) = b(0,ind);
    i++;
  }
  i = 0;
  for (dofIt=fluxInds.begin();dofIt!=fluxInds.end();dofIt++){
    int ind = *dofIt;
    b_flux(i) = b(0,ind);
    i++;
  }
}

void Solution::getElemData(ElementPtr elem, FieldContainer<double> &finalStiffness, FieldContainer<double> &localRHSVector){

  int cellID = elem->cellID();

  ElementTypePtr elemTypePtr = elem->elementType();   
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr, _mesh, false, _cubatureEnrichmentDegree));
  BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh,true, _cubatureEnrichmentDegree));
  //  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr, _mesh));
  //  BasisCachePtr ipBasisCache = Teuchos::rcp(new BasisCache(elemTypePtr,_mesh,true));
    
  DofOrderingPtr trialOrderingPtr = elemTypePtr->trialOrderPtr;
  DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
  int numTrialDofs = trialOrderingPtr->totalDofs();
  int numTestDofs = testOrderingPtr->totalDofs();
  //  cout << "test and trial dofs = " << numTrialDofs << ", " << numTestDofs << endl;

  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
  FieldContainer<double> cellSideParities  = _mesh->cellSideParitiesForCell(cellID);

  bool createSideCacheToo = true;
  vector<int> cellIDs;
  cellIDs.push_back(cellID); // just do one cell at a time
  basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
  ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,_ip->hasBoundaryTerms()); // create side cache if ip has boundary values
  
  CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;

  FieldContainer<double> ipMatrix(1,numTestDofs,numTestDofs);
      
  _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, ipBasisCache);
  
  bool estimateElemCondition = false;
  if (estimateElemCondition){
    Epetra_SerialDenseMatrix IPK(numTestDofs,numTestDofs);
    Epetra_SerialDenseMatrix x(numTestDofs);
    Epetra_SerialDenseMatrix b(numTestDofs);
    for (int i = 0;i<numTestDofs;i++){
      for (int j = 0;j<numTestDofs;j++){
	IPK(i,j) = ipMatrix(0,i,j);
      }      
    }
    Epetra_SerialDenseSolver solver;
    solver.SetMatrix(IPK);
    solver.SetVectors(x,b);
    double invCondNumber;
    int err = solver.ReciprocalConditionEstimate(invCondNumber);    
    cout << "condition number of element " << cellID << " = " << 1.0/invCondNumber << endl;
  }

  FieldContainer<double> optTestCoeffs(1,numTrialDofs,numTestDofs);
  _mesh->bilinearForm()->optimalTestWeights(optTestCoeffs, ipMatrix, elemTypePtr,
					    cellSideParities, basisCache);

  //  FieldContainer<double> finalStiffness(1,numTrialDofs,numTrialDofs);
  finalStiffness.resize(1,numTrialDofs,numTrialDofs);

  BilinearFormUtility::computeStiffnessMatrix(finalStiffness,ipMatrix,optTestCoeffs);
      
  //  FieldContainer<double> localRHSVector(1, numTrialDofs);
  localRHSVector.resize(1, numTrialDofs);
  _rhs->integrateAgainstOptimalTests(localRHSVector, optTestCoeffs, testOrderingPtr, basisCache);

  // apply filter(s) (e.g. penalty method, preconditioners, etc.)
  if (_filter.get()) {
    _filter->filter(finalStiffness,localRHSVector,basisCache,_mesh,_bc);
  }
}

// =================================== CONDENSED SOLVE ======================================

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
    
    vector<int> cellIDs;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      int cellID = _mesh->cellID(elemTypePtr, cellIndex, -1); // -1: global cellID
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
    int numSides = cellTopo.getSideCount();
    
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

Epetra_Map Solution::getPartitionMap(int rank, set<int> & myGlobalIndicesSet, int numGlobalDofs, 
                                     int zeroMeanConstraintsSize, Epetra_Comm* Comm ) {
  int numGlobalLagrange = _lagrangeConstraints->numGlobalConstraints();
  vector< ElementPtr > elements = _mesh->elementsInPartition(rank);
  int numMyElements = elements.size();
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
  
  int *myGlobalIndices;
  if (localDofsSize!=0){
    myGlobalIndices = new int[ localDofsSize ];      
  } else {
    myGlobalIndices = NULL;
  }
    
  // copy from set object into the allocated array
  int offset = 0;
  for (set<int>::iterator indexIt = myGlobalIndicesSet.begin(); indexIt != myGlobalIndicesSet.end(); indexIt++ ) {
    myGlobalIndices[offset++] = *indexIt;
  }
  int cellOffset = _mesh->activeCellOffset() * _lagrangeConstraints->numElementConstraints();
  int globalIndex = cellOffset + numGlobalDofs;
  for (int elemLagrangeIndex=0; elemLagrangeIndex<_lagrangeConstraints->numElementConstraints(); elemLagrangeIndex++) {
    for (int cellIndex=0; cellIndex<numMyElements; cellIndex++) {
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
  
  int totalRows = numGlobalDofs + globalNumElementLagrange + numGlobalLagrange + zeroMeanConstraintsSize;
    
  int indexBase = 0;
  //cout << "process " << rank << " about to construct partMap.\n";
  //Epetra_Map partMap(-1, localDofsSize, myGlobalIndices, indexBase, Comm);
  Epetra_Map partMap(totalRows, localDofsSize, myGlobalIndices, indexBase, *Comm);

  if (localDofsSize!=0){
    delete[] myGlobalIndices;
  }
  return partMap;
}

void Solution::processSideUpgrades( const map<int, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades ) {
  set<int> cellIDsToSkip; //empty
  processSideUpgrades(cellSideUpgrades,cellIDsToSkip);
}

void Solution::processSideUpgrades( const map<int, pair< ElementTypePtr, ElementTypePtr > > &cellSideUpgrades, const set<int> &cellIDsToSkip ) {
  for (map<int, pair< ElementTypePtr, ElementTypePtr > >::const_iterator upgradeIt = cellSideUpgrades.begin();
       upgradeIt != cellSideUpgrades.end(); upgradeIt++) {
    int cellID = upgradeIt->first;
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

void Solution::projectOntoMesh(const map<int, Teuchos::RCP<Function> > &functionMap){
  // TODO: finish the commented-out MPI version of this method...
  
//#ifdef HAVE_MPI
//  int rank     = Teuchos::GlobalMPISession::getRank();
//#else
//  int rank     = 0;
//#endif

  vector< ElementPtr > activeElems = _mesh->activeElements();
//  vector< ElementPtr > activeElems = _mesh->elementsInPartition(rank);
  for (vector<ElementPtr >::iterator elemIt = activeElems.begin();elemIt!=activeElems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
    projectOntoCell(functionMap,cellID);
  }
  
  // TODO: gather the projected solutions
}

void Solution::projectOntoCell(const map<int, FunctionPtr > &functionMap, int cellID, int side) {
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
  vector<int> cellIDs(1,cellID);
  
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
        lastSide = elemTypePtr->cellTopoPtr->getSideCount() - 1;
      } else {
        firstSide = side;
        lastSide = side;
      }
      for (int sideIndex=firstSide; sideIndex<=lastSide; sideIndex++) {
        Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = elemTypePtr->trialOrderPtr->getBasis(trialID, sideIndex);
        FieldContainer<double> basisCoefficients(1,basis->getCardinality());
        Projector::projectFunctionOntoBasis(basisCoefficients, function, basis, basisCache->getSideBasisCache(sideIndex));
        setSolnCoeffsForCellID(basisCoefficients,cellID,trialID,sideIndex);
      }
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(side != -1, std::invalid_argument, "sideIndex for fields must = -1");
      Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = elemTypePtr->trialOrderPtr->getBasis(trialID);
      FieldContainer<double> basisCoefficients(1,basis->getCardinality());
      Projector::projectFunctionOntoBasis(basisCoefficients, function, basis, basisCache);
      //      cout << "setting solnCoeffs for cellID " << cellID << " and trialID " << trialID << endl;
      //      cout << basisCoefficients;
      setSolnCoeffsForCellID(basisCoefficients,cellID,trialID);
    }
  }
}

void Solution::projectOntoMesh(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap){
  vector< ElementPtr > activeElems = _mesh->activeElements();
  for (vector<ElementPtr >::iterator elemIt = activeElems.begin();elemIt!=activeElems.end();elemIt++){
    ElementPtr elem = *elemIt;
    int cellID = elem->cellID();
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

void Solution::projectOntoCell(const map<int, Teuchos::RCP<AbstractFunction> > &functionMap, int cellID){
  typedef Teuchos::RCP<AbstractFunction> AbstractFxnPtr;
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
  
  for (map<int, AbstractFxnPtr >::const_iterator functionIt = functionMap.begin(); functionIt !=functionMap.end(); functionIt++){
    int trialID = functionIt->first;
    AbstractFxnPtr function = functionIt->second;
    ElementPtr element = _mesh->getElement(cellID);
    ElementTypePtr elemTypePtr = element->elementType();
    
    Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = elemTypePtr->trialOrderPtr->getBasis(trialID);
    
    FieldContainer<double> basisCoefficients;
    Projector::projectFunctionOntoBasis(basisCoefficients, function, basis, physicalCellNodes);
    setSolnCoeffsForCellID(basisCoefficients,cellID,trialID); 
  }
}

void Solution::projectOldCellOntoNewCells(int cellID, ElementTypePtr oldElemType, const vector<int> &childIDs) {
  // NOTE: this only projects field variables for now.
  DofOrderingPtr oldTrialOrdering = oldElemType->trialOrderPtr;
  set<int> trialIDs = oldTrialOrdering->getVarIDs();
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesForCell(cellID);
  
  if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
    // they're implicit 0s, then: projection will also be implicit 0s...
    return;
  }

  FieldContainer<double>* solutionCoeffs = &(_solutionForCellIDGlobal[cellID]);
  TEUCHOS_TEST_FOR_EXCEPTION(oldTrialOrdering->totalDofs() != solutionCoeffs->size(), std::invalid_argument,
                             "oldElemType trial space does not match stored solution size");
  // TODO: rewrite this method using Functions instead of AbstractFunctions
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  
  for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++) {
    int trialID = *trialIDIt;
    if (oldTrialOrdering->getNumSidesForVarID(trialID) == 1) { // field variable, the only kind we honor right now
      BasisPtr basis = oldTrialOrdering->getBasis(trialID);
      int basisCardinality = basis->getCardinality();
      FieldContainer<double> basisCoefficients(basisCardinality);
      
      for (int dofOrdinal=0; dofOrdinal<basisCardinality; dofOrdinal++) {
        int dofIndex = oldElemType->trialOrderPtr->getDofIndex(trialID, dofOrdinal);
        basisCoefficients(dofOrdinal) = (*solutionCoeffs)(dofIndex);
      }
      Teuchos::RCP<BasisSumFunction> oldTrialFunction = Teuchos::rcp( new BasisSumFunction(basis, basisCoefficients, physicalCellNodes) );
      functionMap[trialID] = oldTrialFunction;
    }
  }
  for (vector<int>::const_iterator childIDIt=childIDs.begin(); childIDIt != childIDs.end(); childIDIt++) {
    int childID = *childIDIt;
    // (re)initialize the FieldContainer storing the solution--element type may have changed (in case of p-refinement)
    _solutionForCellIDGlobal[childID] = FieldContainer<double>(_mesh->getElement(childID)->elementType()->trialOrderPtr->totalDofs());
    projectOntoCell(functionMap,childID);
  }
  
  clearComputedResiduals(); // force recomputation of energy error (could do something more incisive, just computing the energy error for the new cells)
}

/*
void Solution::projectOldCellOntoNewCells(int cellID, ElementTypePtr oldElemType, const vector<int> &childIDs) {
  vector<int> trialVolumeIDs = _mesh->bilinearForm()->trialVolumeIDs();
  vector<int> fluxTraceIDs = _mesh->bilinearForm()->trialBoundaryIDs();
    
  if (_solutionForCellIDGlobal.find(cellID) == _solutionForCellIDGlobal.end() ) {
    // they're implicit 0s, then: projection will also be implicit 0s...
    return;
  }
  int numSides = oldElemType->cellTopoPtr->getSideCount();
  map<int, FunctionPtr > functionMap;
  map<int, map<int, FunctionPtr > > sideFunctionMap;
  
  int sideIndexForFields = 0; // someday, will probably want to make this -1, but DofOrdering doesn't yet support this
  
  for (vector<int>::iterator trialIDIt = trialVolumeIDs.begin(); trialIDIt != trialVolumeIDs.end(); trialIDIt++) {
    int trialID = *trialIDIt;
    BasisPtr basis = oldElemType->trialOrderPtr->getBasis(trialID);
    FieldContainer<double> basisCoefficients(basis->getCardinality());
    basisCoeffsForTrialOrder(basisCoefficients, oldElemType->trialOrderPtr, _solutionForCellIDGlobal[cellID], trialID, sideIndexForFields);
    functionMap[trialID] = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients));
  }
  for (vector<int>::iterator trialIDIt = fluxTraceIDs.begin(); trialIDIt != fluxTraceIDs.end(); trialIDIt++) {
    int trialID = *trialIDIt;
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      map<int, FunctionPtr> thisSideFunctions;
      BasisPtr basis = oldElemType->trialOrderPtr->getBasis(trialID,sideIndex);
      FieldContainer<double> basisCoefficients(basis->getCardinality());
      basisCoeffsForTrialOrder(basisCoefficients, oldElemType->trialOrderPtr, _solutionForCellIDGlobal[cellID], trialID, sideIndex);
      bool boundaryValued = true;
      thisSideFunctions[trialID] = Teuchos::rcp( new NewBasisSumFunction(basis, basisCoefficients, OP_VALUE, boundaryValued) );
      sideFunctionMap[sideIndex] = thisSideFunctions;
    }
  }
  
  for (vector<int>::const_iterator childIDIt=childIDs.begin(); childIDIt != childIDs.end(); childIDIt++) {
    int childID = *childIDIt;
    projectOntoCell(functionMap,childID);
    for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
      projectOntoCell(sideFunctionMap[sideIndex], childID);
    }
  }
  
  clearComputedResiduals(); // force recomputation of energy error (could do something more incisive, just computing the energy error for the new cells)
}*/
