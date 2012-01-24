
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
// only use MUMPS when we have MPI
#ifdef HAVE_MPI
//#include "Amesos_Mumps.h"
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

#include "Epetra_SerialDenseMatrix.h"
#include "Epetra_SerialDenseSolver.h"
#include "Epetra_DataAccess.h"

// Shards includes
#include "Shards_CellTopology.hpp"

#include "ml_epetra_utils.h"
//#include "ml_common.h"

#include <stdlib.h>

#include "BilinearFormUtility.h"
#include "BasisEvaluation.h"
#include "BasisValueCache.h"
#include "Solution.h"

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;
static const int MAX_BATCH_SIZE_IN_BYTES = 3*1024*1024; // 3 MB
static const int MIN_BATCH_SIZE_IN_CELLS = 2; // overrides the above, if it results in too-small batches

// copy constructor:
Solution::Solution(const Solution &soln) {
  _mesh = soln.mesh();
  _bc = soln.bc();
  _rhs = soln.rhs();
  _ip = soln.ip();
  _solutionForElementType = soln.solutionForElementTypeMap();
  initialize();
}

Solution::Solution(Teuchos::RCP<Mesh> mesh, Teuchos::RCP<BC> bc, Teuchos::RCP<RHS> rhs, Teuchos::RCP<DPGInnerProduct> ip) {
  _mesh = mesh;
  _bc = bc;
  _rhs = rhs;
  _ip = ip;
  initialize();
}

void Solution::initialize() {
  // sets up the data structure for storing the solution (will want to do this if the mesh changes!)
  
  // first, clear the data structure in case it already stores some stuff
  _solutionForElementType.clear();
  
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    int numCellsOfType = _mesh->numElementsOfType(elemTypePtr);
    int numDofsForType = elemTypePtr->trialOrderPtr->totalDofs();
    FieldContainer<double> solutionCoeffs(numCellsOfType,numDofsForType);
    _solutionForElementType[elemTypePtr.get()] = solutionCoeffs;
  }
  _residualsComputed = false;
}

void Solution::solve(bool useMumps) { // if not, KLU (TODO: make an enumerated list of choices)
  // the following is not strictly necessary if the mesh has not changed since we were constructed:
  initialize();
  
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
  
  typedef Teuchos::RCP< DofOrdering > DofOrderingPtr;
  typedef Teuchos::RCP< shards::CellTopology > CellTopoPtr;
  
  //cout << "process " << rank << " about to get elementTypes.\n";
  
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes(rank);
  vector< ElementTypePtr >::iterator elemTypeIt;
  // will want a CrsMatrix here in just a moment...
  
  // determine any zero-mean constraints:
  vector< int > trialIDs = _mesh->bilinearForm().trialIDs();
  vector< int > zeroMeanConstraints;
  for (vector< int >::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (_bc->imposeZeroMeanConstraint(trialID)) {
      zeroMeanConstraints.push_back(trialID);
    }
  }
  int numGlobalDofs = _mesh->numGlobalDofs();
  
  set<int> myGlobalIndicesSet = _mesh->globalDofIndicesForPartition(rank);
  Epetra_Map partMap = getPartitionMap(rank, myGlobalIndicesSet,numGlobalDofs,zeroMeanConstraints.size(),&Comm);
  //Epetra_Map globalMapG(numGlobalDofs+zeroMeanConstraints.size(), numGlobalDofs+zeroMeanConstraints.size(), 0, Comm);
  
  int maxRowSize = _mesh->rowSizeUpperBound();
  //cout << "max row size for mesh: " << maxRowSize << endl;
  //cout << "process " << rank << " about to initialize globalStiffMatrix.\n";
  Epetra_FECrsMatrix globalStiffMatrix(Copy, partMap, maxRowSize);
  //  Epetra_FECrsMatrix globalStiffMatrix(Copy, partMap, partMap, maxRowSize);
  Epetra_FEVector rhsVector(partMap);
  
  //cout << "process " << rank << " about to loop over elementTypes.\n";
  int indexBase = 0;
  Epetra_Map timeMap(numProcs,indexBase,Comm);
  Epetra_Time timer(Comm);
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
    //cout << "Solution: elementType loop, iteration: " << elemTypeNumber++ << endl;
    ElementTypePtr elemTypePtr = *(elemTypeIt);
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
      //cout << "testDofOrdering: " << *testOrderingPtr;
      //cout << "trialDofOrdering: " << *trialOrderingPtr;
      nodeDimensions[0] = numCells;
      parityDimensions[0] = numCells;
      FieldContainer<double> physicalCellNodes(nodeDimensions,&myPhysicalCellNodesForType(startCellIndexForBatch,0,0));
      FieldContainer<double> cellSideParities(parityDimensions,&myCellSideParitiesForType(startCellIndexForBatch,0));
      
      //int numCells = physicalCellNodes.dimension(0);
      CellTopoPtr cellTopoPtr = elemTypePtr->cellTopoPtr;
      
      { // this block is not necessary for the solution.  Here just to produce debugging output
        FieldContainer<double> preStiffness(numCells,numTestDofs,numTrialDofs );
        
        BilinearFormUtility::computeStiffnessMatrix(preStiffness, _mesh->bilinearForm(),
                                                    trialOrderingPtr, testOrderingPtr, *(cellTopoPtr.get()), 
                                                    physicalCellNodes, cellSideParities);
        FieldContainer<double> preStiffnessTransposed(numCells,numTrialDofs,numTestDofs );
        BilinearFormUtility::transposeFCMatrices(preStiffnessTransposed,preStiffness);
        //cout << "preStiffnessTransposed\n" << preStiffnessTransposed;
      }
      FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
      
      //cout << "Solution: physicalCellNodes--" << endl << physicalCellNodes;
      
      _ip->computeInnerProductMatrix(ipMatrix,testOrderingPtr, *(cellTopoPtr.get()), physicalCellNodes);
      
      //cout << "inner product matrix" << endl << ipMatrix;
      
      FieldContainer<double> optTestCoeffs(numCells,numTrialDofs,numTestDofs);
      
      int optSuccess = BilinearFormUtility::computeOptimalTest(optTestCoeffs, ipMatrix, _mesh->bilinearForm(),
                                                               trialOrderingPtr, testOrderingPtr,
                                                               *(cellTopoPtr.get()), physicalCellNodes, cellSideParities);
      
      if ( optSuccess != 0 ) {
        cout << "**** WARNING: in Solution.solve(), optimal test function computation failed with error code " << optSuccess << ". ****\n";
      }
      
      //cout << "optTestCoeffs\n" << optTestCoeffs;
      
      FieldContainer<double> finalStiffness(numCells,numTrialDofs,numTrialDofs);
      
      BilinearFormUtility::computeStiffnessMatrix(finalStiffness,ipMatrix,optTestCoeffs);
      
      // apply filter(s) (e.g. penalty method, preconditioners, etc.)
      vector<int> cellIDs;
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        int cellID = _mesh->cellID(elemTypePtr, cellIndex+startCellIndexForBatch, rank);
        cellIDs.push_back(cellID);
      }
      _filter->filter(finalStiffness,physicalCellNodes,cellIDs,_mesh,_bc);
      
      FieldContainer<double> localRHSVector(numCells, numTrialDofs);
      BilinearFormUtility::computeRHS(localRHSVector, _mesh->bilinearForm(), *(_rhs.get()),
                                      optTestCoeffs, testOrderingPtr,
                                      *(cellTopoPtr.get()), physicalCellNodes);
      
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
  
  Epetra_Vector timeLocalStiffnessVector(timeMap);
  timeLocalStiffnessVector[0] = timeLocalStiffness;
  
  // impose zero mean constraints:
  int zmcIndex = numGlobalDofs; // start zmc indices just after the regular dof indices
  for (vector< int >::iterator trialIt = zeroMeanConstraints.begin(); trialIt != zeroMeanConstraints.end(); trialIt++) {
    int trialID = *trialIt;
    //cout << "Imposing zero-mean constraint for variable " << _mesh->bilinearForm().trialName(trialID) << endl;
    FieldContainer<double> basisIntegrals;
    FieldContainer<int> globalIndices;
    integrateBasisFunctions(globalIndices,basisIntegrals, trialID);
    int numValues = globalIndices.size();
    // insert row:
    globalStiffMatrix.InsertGlobalValues(1,&zmcIndex,numValues,&globalIndices(0),&basisIntegrals(0));
    // insert column:
    globalStiffMatrix.InsertGlobalValues(numValues,&globalIndices(0),1,&zmcIndex,&basisIntegrals(0));
    // insert stabilizing parameter -- for now, just the sum of the entries in the extra row/column
    double rho = 0.0;
    for (int i=0; i<numValues; i++) {
      rho += basisIntegrals[i];
    }
    //rho /= numValues;
    globalStiffMatrix.InsertGlobalValues(1,&zmcIndex,1,&zmcIndex,&rho);
    zmcIndex++;
  }
  
  timer.ResetStartTime();
  
  rhsVector.GlobalAssemble();
  
  //EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector_before_bcs.dat",rhsVector,0,0,false);
  
  globalStiffMatrix.GlobalAssemble(); // will call globalStiffMatrix.FillComplete();
  
  double timeGlobalAssembly = timer.ElapsedTime();
  Epetra_Vector timeGlobalAssemblyVector(timeMap);
  timeGlobalAssemblyVector[0] = timeGlobalAssembly;
  
  //EpetraExt::RowMatrixToMatlabFile("stiff_matrix.dat",globalStiffMatrix);
  
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
    int meshPartitionLocalIndex = _mesh->partitionLocalIndexForGlobalDofIndex(globalIndex);
    if (meshPartitionLocalIndex != localIndex) {
      cout << "meshPartitionLocalIndex != localIndex (" << meshPartitionLocalIndex << " != " << localIndex << ")\n";
      TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
    }
    int partition = _mesh->partitionForGlobalDofIndex( globalIndex );
    if (partition != rank) {
      cout << "partition != rank (" << partition << " != " << rank << ")\n";
      TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
    }
  }
  
  Epetra_LinearProblem problem(&globalStiffMatrix, &lhsVector, &rhsVector);
  
  rhsVector.GlobalAssemble();

  timer.ResetStartTime();
  if ( !useMumps ) {
    Amesos_Klu klu(problem);
    
    //cout << "About to call klu.Solve()." << endl;
    int solveSuccess = klu.Solve();
    
    //cout << "klu.Solve() completed." << endl;
    // Amesos_Utils().ComputeTrueResidual (globalStiffMatrix, lhsVector, rhsVector, false, "TrueResidual: ");
    if (solveSuccess != 0 ) {
      cout << "**** WARNING: in Solution.solve(), klu.Solve() failed with error code " << solveSuccess << ". ****\n";
    }
  } else {
    // only use MUMPS when we have MPI
#ifdef HAVE_MPI
    if (rank == 0) {
      cout << "USING MUMPS!\n";
    }
    /*
      Amesos_Mumps mumps(problem);
      mumps.SymbolicFactorization();
      mumps.NumericFactorization();
      mumps.Solve();
    */
#else
    cout << "MUMPS disabled for non-MPI builds!\n";
#endif
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
  //EpetraExt::MultiVectorToMatrixMarketFile("rhs_vector.dat",rhsVector,0,0,false);
  //EpetraExt::RowMatrixToMatlabFile("stiff_matrix.dat",globalStiffMatrix);
  //EpetraExt::MultiVectorToMatrixMarketFile("lhs_vector.dat",lhsVector,0,0,false);
  
  // Import solution onto current processor
  int numNodesGlobal = partMap.NumGlobalElements();
  Epetra_Map     solnMap(numNodesGlobal, numNodesGlobal, 0, Comm);
  Epetra_Import  solnImporter(solnMap, partMap);
  Epetra_Vector  solnCoeff(solnMap);
  solnCoeff.Import(lhsVector, solnImporter, Insert);
  
  // copy the dof coefficients into our data structure
  vector< Teuchos::RCP< Element > > elements = _mesh->activeElements();
  vector< Teuchos::RCP< Element > >::iterator elemIt;
  for (elemIt = elements.begin(); elemIt != elements.end(); elemIt++) {
    ElementPtr elemPtr = *(elemIt);
    int cellID = elemPtr->cellID();
    int cellIndex = elemPtr->globalCellIndex();
    int numDofs = elemPtr->elementType()->trialOrderPtr->totalDofs();
    for (int dofIndex=0; dofIndex<numDofs; dofIndex++) {
      int globalIndex = _mesh->globalDofIndex(cellID, dofIndex);
      _solutionForElementType[elemPtr->elementType().get()](cellIndex,dofIndex) = solnCoeff[globalIndex];
    }
  }
  
  //  double lhsVectorGathered[numProcs][maxLhsLength];
  //  double *lhsVectorLocal = new double[ maxLhsLength ];
  //  // debugging: clear the vector first:
  //  for (int i=0; i<maxLhsLength; i++ ) {
  //    lhsVectorLocal[i] = 0.0;
  //  }
  //  
  //  lhsVector.ExtractCopy( &lhsVectorLocal ); 
  //  
  ////  //debugging output:
  ////  cout << "rank " << rank << " lhsVectorLocal:\t";
  ////  cout << setprecision(3);
  ////  // debugging: clear the vector first:
  ////  for (int i=0; i< maxLhsLength; i++ ) {
  ////    cout << lhsVectorLocal[i] << "\t";
  ////  }
  ////  cout << endl;
  //
  //  Comm.GatherAll( lhsVectorLocal, &lhsVectorGathered[0][0], maxLhsLength );
  //
  ////  //debugging output:
  ////  if (rank == 0) {
  ////    cout << "lhsVectorGathered:\n";
  ////    // debugging: clear the vector first:
  ////    for (int j=0; j< numProcs; j++) {
  ////      cout << "rank " << j << ":\t";
  ////      for (int i=0; i< maxLhsLength; i++ ) {
  ////        cout << lhsVectorGathered[j][i] << "\t";
  ////      }
  ////      cout << endl;
  ////    }
  ////  }
  ////  cout << setprecision(5);
  //  
  //  // copy the dof coefficients into our data structure
  //  vector< Teuchos::RCP< Element > > elements = _mesh->activeElements();
  //  vector< Teuchos::RCP< Element > >::iterator elemIt;
  //  
  //  for (elemIt = elements.begin(); elemIt != elements.end(); elemIt++) {
  //    ElementPtr elemPtr = *(elemIt);
  //    int cellID = elemPtr->cellID();
  //    int cellIndex = elemPtr->globalCellIndex();
  //    int numDofs = elemPtr->elementType()->trialOrderPtr->totalDofs();
  //    for (int dofIndex=0; dofIndex<numDofs; dofIndex++) {
  //      int globalIndex = _mesh->globalDofIndex(cellID, dofIndex);
  //      int partition = _mesh->partitionForGlobalDofIndex( globalIndex );
  //      int partitionLocalIndex = _mesh->partitionLocalIndexForGlobalDofIndex( globalIndex );
  //      TEST_FOR_EXCEPTION( partitionLocalIndex > maxLhsLength, std::invalid_argument, "partitionLocalIndex out of bounds");
  //      _solutionForElementType[elemPtr->elementType().get()](cellIndex,dofIndex) = lhsVectorGathered[partition][partitionLocalIndex];
  //    }
  //  }
  //
  //  delete lhsVectorLocal;

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
  
  bool printTimingReport = false;
  if ((rank == 0) && printTimingReport) {
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
  
  _residualsComputed = false; // now that we've solved, will need to recompute residuals...
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

bool Solution::equals(Solution& otherSolution, double tol) {
  vector<ElementTypePtr> myElemTypes = _mesh->elementTypes();
  vector<ElementTypePtr> otherElemTypes = otherSolution.mesh()->elementTypes();
  // check that the # of elem types are the same:
  int numElemTypes = myElemTypes.size();
  if ( numElemTypes != otherElemTypes.size() ) {
    return false;
  }
  double maxDiff = 0.0;
  map< ElementType*, FieldContainer<double> > otherSolutionForElementType = otherSolution.solutionForElementTypeMap();
  for (int elemTypeIndex=0; elemTypeIndex<numElemTypes; elemTypeIndex++) {
    ElementTypePtr otherElemType = otherElemTypes[elemTypeIndex];
    FieldContainer<double> otherSoln = otherSolutionForElementType[otherElemType.get()];
    FieldContainer<double> mySoln = _solutionForElementType[getEquivalentElementType( otherSolution.mesh(), otherElemType ).get()];
    int numSolnEntries = mySoln.size();
    if ( numSolnEntries != otherSoln.size() ) {
      return false;
    }
    for (int i=0; i<numSolnEntries; i++) {
      double mine = mySoln[i], theirs = otherSoln[i];
      double diff = abs(mine-theirs);
      if (diff > tol) {
        return false;
      }
      maxDiff = max(maxDiff,diff);
    }
  }
  cout << "Solution maxDiff is " << maxDiff << endl;
  return true;
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
  TEST_FOR_EXCEPTION(values.dimension(0) != numCellsOfType,
                     std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  TEST_FOR_EXCEPTION(values.dimension(1) != basisCardinality,
                     std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr), trialBasisCardinality)");
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis;
  trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
  
  int cubDegree = trialBasis->getDegree();
  
  BasisValueCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), *(elemTypePtr->cellTopoPtr), cubDegree);
  
  Teuchos::RCP < const FieldContainer<double> > trialValuesTransformedWeighted;
  
  trialValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trialBasis,IntrepidExtendedTypes::OPERATOR_VALUE);
  
  if (trialValuesTransformedWeighted->rank() != 3) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateBasisFunctions only supports scalar-valued field variables at present.");
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
    BasisValueCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), *(elemTypePtr->cellTopoPtr), cubDegree);
    FieldContainer<double> cellMeasures = basisCache.getCellMeasures();
    for (int cellIndex=0; cellIndex<numCellsOfType; cellIndex++) {
      value += cellMeasures(cellIndex);
    }
  }
  return value;
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
  TEST_FOR_EXCEPTION(values.dimension(0) != numCellsOfType,
                     std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr))");
  TEST_FOR_EXCEPTION(values.rank() != 1,
                     std::invalid_argument, "values must have dimensions (_mesh.numCellsOfType(elemTypePtr))");
  Teuchos::RCP < Intrepid::Basis<double,FieldContainer<double> > > trialBasis;
  trialBasis = elemTypePtr->trialOrderPtr->getBasis(trialID);
  
  int cubDegree = trialBasis->getDegree();
  
  BasisValueCache basisCache(_mesh->physicalCellNodesGlobal(elemTypePtr), *(elemTypePtr->cellTopoPtr), cubDegree);
  
  Teuchos::RCP < const FieldContainer<double> > trialValuesTransformedWeighted;
  
  trialValuesTransformedWeighted = basisCache.getTransformedWeightedValues(trialBasis,IntrepidExtendedTypes::OPERATOR_VALUE);
  
  if (trialValuesTransformedWeighted->rank() != 3) {
    TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateSolution only supports scalar-valued field variables at present.");
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
    
    bool boundaryIntegral = _mesh()->bilinearForm().isFluxOrTrace(trialID);
    if ( !boundaryIntegral ) {
      TEST_FOR_EXCEPTION(true, std::invalid_argument, "integrateFlux() called for field variable.");
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
                              FieldContainer<double> &physicalPoints,
                              FieldContainer<double> &sideRefCellPoints,
                              int sideIndex) {
  // currently, we only support computing solution values on all the cells of a given type at once.
  // values(numCellsForType,numPoints[,spaceDim (for vector-valued)])
  // physicalPoints(numCellsForType,numPoints,spaceDim)
  FieldContainer<double> solnCoeffs = _solutionForElementType[elemTypePtr.get()]; // (numcells, numLocalTrialDofs)
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
  
  TEST_FOR_EXCEPTION( ( basisRank==0 ) && values.rank() != 2,
		      std::invalid_argument,
		      "for scalar values, values container should be dimensioned(numCells,numPoints).");
  TEST_FOR_EXCEPTION( ( basisRank==1 ) && values.rank() != 3,
		      std::invalid_argument,
		      "for scalar values, values container should be dimensioned(numCells,numPoints,spaceDim).");
  TEST_FOR_EXCEPTION( values.dimension(0) != numCells,
		      std::invalid_argument,
		      "values.dimension(0) != numCells.");
  TEST_FOR_EXCEPTION( values.dimension(1) != numPoints,
		      std::invalid_argument,
		      "values.dimension(1) != numPoints.");
  TEST_FOR_EXCEPTION( basisRank==1 && values.dimension(2) != spaceDim,
		      std::invalid_argument,
		      "vector values.dimension(1) != spaceDim.");
  TEST_FOR_EXCEPTION( physicalPoints.rank() != 3,
		      std::invalid_argument,
		      "physicalPoints.rank() != 3.");
  TEST_FOR_EXCEPTION( physicalPoints.dimension(2) != spaceDim,
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
    transformedValues = BasisEvaluation::getTransformedValues(basis, IntrepidExtendedTypes::OPERATOR_VALUE, 
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

void Solution::energyError(map<int,double> &energyError){ //FieldContainer<double> &energyError) {
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
  int cellIndStart = 0;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
    ElementTypePtr elemTypePtr = *(elemTypeIt);    
    
    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemTypePtr);    
    
    // for error rep v_e, residual res, energyError = sqrt ( ve_^T * res)
    FieldContainer<double> residuals = _residualForElementType[elemTypePtr.get()];
    FieldContainer<double> errorReps = _errorRepresentationForElementType[elemTypePtr.get()];
    int numTestDofs = residuals.dimension(1);    
    int numCells = residuals.dimension(0);    
    TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "In energyError::numCells does not match number of elems in partition.");    
    
    for (int cellIndex=cellIndStart;cellIndex<numCells;cellIndex++){
      double errorSquared = 0.0;
      for (int i=0; i<numTestDofs; i++) {      
        errorSquared += residuals(cellIndex,i) * errorReps(cellIndex,i);
        //        errorSquared += errorReps(cellIndex,i) * errorReps(cellIndex,i);        
        //        errorSquared += residuals(cellIndex,i) * residuals(cellIndex,i);
      }
      localErrArray[cellIndex] = sqrt(errorSquared);
      int cellID = _mesh->cellID(elemTypePtr,cellIndex,rank);
      localCellIDArray[cellIndex] = cellID; 
      //      cout << "energy error for cellID " << cellID << " is " << sqrt(errorSquared) << endl;
    }   
    cellIndStart += numCells; // increment to go to the next set of element types
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
  // copy back to energyError field container 
  for (int procIndex=0;procIndex<numProcs;procIndex++){
    for (int globalCellIndex=0;globalCellIndex<numActiveElements;globalCellIndex++){
      if (cellIDArray[procIndex][globalCellIndex]!=-1){
        energyError[cellIDArray[procIndex][globalCellIndex]] = errArray[procIndex][globalCellIndex];
      }
    }
  }      

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
    Teuchos::RCP<DofOrdering> testOrdering = elemTypePtr->testOrderPtr;
    FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    
    vector< Teuchos::RCP< Element > > elemsInPartitionOfType = _mesh->elementsOfType(rank, elemTypePtr);    
    
    int numCells = physicalCellNodes.dimension(0);
    int numTestDofs = testOrdering->totalDofs();
    TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "In computeErrorRepresentation::numCells does not match number of elems in partition.");    
    FieldContainer<double> ipMatrix(numCells,numTestDofs,numTestDofs);
    
    _ip->computeInnerProductMatrix(ipMatrix,testOrdering, cellTopo, physicalCellNodes);
    FieldContainer<double> errorRepresentation(numCells,numTestDofs);
    
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
      
      Epetra_SerialDenseMatrix errorRepresentationMatrix(numTestDofs,1);
      
      solver.SetMatrix(ipMatrixT);
      //    solver.SolveWithTranspose(true); // not that it should matter -- ipMatrix should be symmetric
      int success = solver.SetVectors(errorRepresentationMatrix, rhs);
      
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
        errorRepresentation(localCellIndex,i) = errorRepresentationMatrix(i,0);
      }
    }
    _errorRepresentationForElementType[elemTypePtr.get()] = errorRepresentation;
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
    FieldContainer<double> solution = _solutionForElementType[elemTypePtr.get()];
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    
    int numTrialDofs = trialOrdering->totalDofs();
    int numTestDofs  = testOrdering->totalDofs();
    int numCells = physicalCellNodes.dimension(0); // partition-local cells

    //    cout << "Num elems in partition " << rank << " is " << elemsInPartition.size() << endl;
    
    TEST_FOR_EXCEPTION( numCells!=elemsInPartitionOfType.size(), std::invalid_argument, "in computeResiduals::numCells does not match number of elems in partition.");    
    /*
      cout << "Num trial/test dofs " << rank << " is " << numTrialDofs << ", " << numTestDofs << endl;
      cout << "solution dim on " << rank << " is " << solution.dimension(0) << ", " << solution.dimension(1) << endl;
    */
      
    // set up diagonal testWeights matrices so we can reuse the existing computeRHS
    FieldContainer<double> testWeights(numCells,numTestDofs,numTestDofs);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int i=0; i<numTestDofs; i++) {
        testWeights(cellIndex,i,i) = 1.0; 
      }
    }
    
    // compute l(v) and store in residuals:
    FieldContainer<double> residuals(numCells,numTestDofs);
    BilinearFormUtility::computeRHS(residuals, _mesh->bilinearForm(), *(_rhs.get()), 
                                    testWeights, testOrdering, cellTopo, physicalCellNodes);
    
    // compute b(u, v):
    FieldContainer<double> preStiffness(numCells,numTestDofs,numTrialDofs );
    BilinearFormUtility::computeStiffnessMatrix(preStiffness, _mesh->bilinearForm(),
                                                trialOrdering, testOrdering, cellTopo, 
                                                physicalCellNodes, cellSideParities);    

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
  }
  _residualsComputed = true;
}

void Solution::solutionValues(FieldContainer<double> &values, 
                              ElementTypePtr elemTypePtr, 
                              int trialID,
                              FieldContainer<double> &physicalPoints) {
  int sideIndex = 0;
  // currently, we only support computing solution values on all the cells of a given type at once.
  // values(numCellsForType,numPoints[,spaceDim (for vector-valued)])
  // physicalPoints(numCellsForType,numPoints,spaceDim)
  FieldContainer<double> solnCoeffs = _solutionForElementType[elemTypePtr.get()]; // (numcells, numLocalTrialDofs)
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodesGlobal(elemTypePtr);
  // 1. Map the physicalPoints from the elements specified in physicalCellNodes into reference element
  // 2. Compute each basis on those points
  // 3. Transform those basis evaluations back into the physical space
  // 4. Multiply by the solnCoeffs
  
  int numCells = physicalCellNodes.dimension(0);
  int numPoints = physicalPoints.dimension(1);
  shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr.get());
  int spaceDim = cellTopo.getDimension();
  
  // TODO: add TEST_FOR_EXCEPTIONs to make sure all the FC dimensions/ranks agree with expectations
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
  TEST_FOR_EXCEPTION( ( basisRank==0 ) && values.rank() != 2,
		      std::invalid_argument,
		      "for scalar values, values container should be dimensioned(numCells,numPoints).");
  TEST_FOR_EXCEPTION( ( basisRank==1 ) && values.rank() != 3,
		      std::invalid_argument,
		      "for scalar values, values container should be dimensioned(numCells,numPoints,spaceDim).");
  TEST_FOR_EXCEPTION( values.dimension(0) != numCells,
		      std::invalid_argument,
		      "values.dimension(0) != numCells.");
  TEST_FOR_EXCEPTION( values.dimension(1) != numPoints,
		      std::invalid_argument,
		      "values.dimension(1) != numPoints.");
  TEST_FOR_EXCEPTION( basisRank==1 && values.dimension(2) != spaceDim,
		      std::invalid_argument,
		      "vector values.dimension(1) != spaceDim.");
  TEST_FOR_EXCEPTION( physicalPoints.rank() != 3,
		      std::invalid_argument,
		      "physicalPoints.rank() != 3.");
  TEST_FOR_EXCEPTION( physicalPoints.dimension(2) != spaceDim,
		      std::invalid_argument,
		      "physicalPoints.dimension(2) != spaceDim.");
  TEST_FOR_EXCEPTION( _mesh->bilinearForm().isFluxOrTrace(trialID),
		      std::invalid_argument,
		      "call the other solutionValues (with sideCellRefPoints argument) for fluxes and traces.");
  
  FieldContainer<double> thisCellJacobian(1,numPoints, spaceDim, spaceDim);
  FieldContainer<double> thisCellJacobInv(1,numPoints, spaceDim, spaceDim);
  FieldContainer<double> thisCellJacobDet(1,numPoints);
  FieldContainer<double> thisRefElemPoints(numPoints,spaceDim);
  
  values.initialize(0.0);
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    thisRefElemPoints.setValues(&refElemPoints(cellIndex,0,0),numPoints*spaceDim);
    thisCellJacobian.setValues(&cellJacobian(cellIndex,0,0,0),numPoints*spaceDim*spaceDim);
    thisCellJacobInv.setValues(&cellJacobInv(cellIndex,0,0,0),numPoints*spaceDim*spaceDim);
    thisCellJacobDet.setValues(&cellJacobDet(cellIndex,0),numPoints);
    Teuchos::RCP< FieldContainer<double> > transformedValues;
    transformedValues = BasisEvaluation::getTransformedValues(basis, IntrepidExtendedTypes::OPERATOR_VALUE, 
                                                              thisRefElemPoints, thisCellJacobian, 
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

double determineQuadEdgeWeights(double weights[], int edgeVertexNumber, int numDivisionsPerEdge, bool xEdge) {
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
                    physPoints(cellIndex,patchIndex*numVertices + patchVertexIndex, dim) += 
		      weight*vertexPoints(cellIndex, vertexIndex, dim);
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

/*void Solution::integrate(FieldContainer<double> &valuePerCell, int trialID) {
  int numCells = _mesh->elements.size();
  if ( (valuePerCell.dimension(0) != numCells) || (valuePerCell.rank() != 1) ) {
  TEST_FOR_EXCEPTION( true, std::invalid_argument, "valuePerCell should have dimension numCells." );
  }
 
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
 
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) {
  //cout << "Solution: elementType loop, iteration: " << elemTypeNumber++ << endl;
  ElementTypePtr elemTypePtr = *(elemTypeIt);
  int trialDegreeMax = 0;
  int numSidesForTrial = elemTypePtr->dofOrderingPtr->getNumSidesForVarID(trialID);
  for (int sideIndex=0; sideIndex<numSidesForTrial; sideIndex++) {
  trialDegreeMax = max(trialDegreeMax, elemTypePtr->dofOrderingPtr->getBasisCardinality(trialID,sideIndex));
  }
  cubDegree = trialDegreeMax*2;
  FieldContainer<double> physicalCellNodes = _mesh->physicalCellNodes(elemTypePtr);
  BasisValueCache basisCache(physicalCellNodes, *(elemTypePtr->cellTopoPtr), cubDegree, 
  numSidesForTrial != 1); // create side caches if trialID is on side...
 
  }
  }*/

// the following was an attempt to rewrite writeToFile more sensibly, just with
// some arbitrary points for each cell.  But I don't know how to get MATLAB to
// build a surface from arbitrary points (if indeed there is any way to do so)
/*void Solution::writeToFile(int trialID, const string &filePath) {
// writes out rows of the format: "cellID x y solnValue"
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
 
typedef CellTools<double>  CellTools;
 
int cellTopoDim = cellTopoPtr->getDimension();
int numVertices = cellTopoPtr->getVertexCount(cellTopoDim,0);
FieldContainer<double> refPoints(numCubPoints+numVertices, cubDim);
FieldContainer<double> refVertices(numVertices,cellTopoDim);
CellTools::getReferenceSubcellVertices(refVertices,cellTopoDim,0,*(cellTopoPtr.get()));
for (int ptIndex=0; ptIndex<numVertices; ptIndex++) {
for (int i=0; i<cellTopoDim; i++) {
refPoints(ptIndex,i) = refVertices(ptIndex,i);
}
}
for (int ptIndex=0; ptIndex<numCubPoints; ptIndex++) {
for (int i=0; i<cubDim; i++) {
refPoints(numVertices+ptIndex,i) = cubPoints(ptIndex,i);
}
}
 
// compute physicalCubaturePoints, the transformed cubature points on each cell:
FieldContainer<double> physCubPoints(numCellsOfType, numCubPoints+numVertices, spaceDim);
CellTools::mapToPhysicalFrame(physCubPoints,refPoints,_mesh->physicalCellNodes(elemTypePtr),*(cellTopoPtr.get()));
 
FieldContainer<double> values(numCellsOfType, numCubPoints+numVertices);
solutionValues(values,elemTypePtr,trialID,physCubPoints);
 
for (int cellIndex=0; cellIndex < numCellsOfType; cellIndex++) {
for (int ptIndex=0; ptIndex< numCubPoints + numVertices; ptIndex++) {
double x = physCubPoints(cellIndex,ptIndex,0);
double y = physCubPoints(cellIndex,ptIndex,1);
double z = values(cellIndex,ptIndex);
fout << _mesh->cellID(elemTypePtr,cellIndex) << " " << x << " " << y << " " << z << endl;
}
}
}
fout.close();
}*/


void Solution::solnCoeffsForCellID(FieldContainer<double> &solnCoeffs, int cellID, int trialID, int sideIndex) {
  int cellIndex = _mesh->elements()[cellID]->globalCellIndex();
  ElementTypePtr elemTypePtr = _mesh->elements()[cellID]->elementType();
  
  Teuchos::RCP< DofOrdering > trialOrder = elemTypePtr->trialOrderPtr;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
  
  int basisRank = trialOrder->getBasisRank(trialID);
  int basisCardinality = basis->getCardinality();
  solnCoeffs.resize(basisCardinality);
  
  for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
    int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
    solnCoeffs(dofOrdinal) = _solutionForElementType[elemTypePtr.get()](cellIndex,localDofIndex);
  }
}

void Solution::setSolnCoeffsForCellID(FieldContainer<double> &solnCoeffsToSet, int cellID, int trialID, int sideIndex) {
  int cellIndex = _mesh->elements()[cellID]->globalCellIndex();
  ElementTypePtr elemTypePtr = _mesh->elements()[cellID]->elementType();
  
  Teuchos::RCP< DofOrdering > trialOrder = elemTypePtr->trialOrderPtr;
  Teuchos::RCP< Basis<double,FieldContainer<double> > > basis = trialOrder->getBasis(trialID,sideIndex);
  
  int basisRank = trialOrder->getBasisRank(trialID);
  int basisCardinality = basis->getCardinality();  
  for (int dofOrdinal=0; dofOrdinal < basisCardinality; dofOrdinal++) {
    int localDofIndex = trialOrder->getDofIndex(trialID, dofOrdinal, sideIndex);
    _solutionForElementType[elemTypePtr.get()](cellIndex,localDofIndex) = solnCoeffsToSet(dofOrdinal);
  }
}


// protected method; used for solution comparison...
map< ElementType*, FieldContainer<double> > Solution::solutionForElementTypeMap() const {
  return _solutionForElementType;
}

// Jesse's additions below:
// must write to .m file
void Solution::writeFieldsToFile(int trialID, const string &filePath){
  typedef CellTools<double>  CellTools;
  
  ofstream fout(filePath.c_str());
  fout << setprecision(15);
  vector< ElementTypePtr > elementTypes = _mesh->elementTypes();
  vector< ElementTypePtr >::iterator elemTypeIt;
  int spaceDim = 2; // TODO: generalize to 3D...
  
  fout << "numCells = " << _mesh->activeElements().size() << endl;
  fout << "x=cell(numCells,1);y=cell(numCells,1);z=cell(numCells,1);" << endl;
  int globalCellInd = 1; //matlab indexes from 1
  for (elemTypeIt = elementTypes.begin(); elemTypeIt != elementTypes.end(); elemTypeIt++) { //thru quads/triangles/etc
    ElementTypePtr elemTypePtr = *(elemTypeIt);
    shards::CellTopology cellTopo = *(elemTypePtr->cellTopoPtr);
    Teuchos::RCP<shards::CellTopology> cellTopoPtr = elemTypePtr->cellTopoPtr;
    
    FieldContainer<double> vertexPoints, physPoints;    
    _mesh->verticesForElementType(vertexPoints,elemTypePtr); //stores vertex points for this element
    FieldContainer<double> physicalCellNodes = _mesh()->physicalCellNodesGlobal(elemTypePtr);
    
    int numCells = vertexPoints.dimension(0);       
    
    // NOW loop over all cells to write solution to file
    int num1DPts = 5;
    for (int xPointIndex = 0; xPointIndex < num1DPts; xPointIndex++){
      for (int yPointIndex = 0; yPointIndex < num1DPts; yPointIndex++){

	// for some odd reason, I cannot compute the ref-to-phys map for more than 1 point 
	int numPoints = 1;
	FieldContainer<double> refPoints(numPoints,spaceDim);
	double x = -1.0 + 2.0*(double)xPointIndex/((double)num1DPts-1.0);
	double y = -1.0 + 2.0*(double)yPointIndex/((double)num1DPts-1.0);
	refPoints(0,0) = x;
	refPoints(0,1) = y;
	
	// map side cubature points in reference parent cell domain to physical space	
	FieldContainer<double> physCubPoints(numCells, numPoints, spaceDim);
	CellTools::mapToPhysicalFrame(physCubPoints, refPoints, physicalCellNodes, cellTopo);
	
	FieldContainer<double> computedValues(numCells,numPoints); // first arg = 1 cell only
	solutionValues(computedValues, elemTypePtr, trialID, physCubPoints);	
	
	for (int cellIndex=0;cellIndex < numCells;cellIndex++){	  
	  fout << "x{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<")=" << physCubPoints(cellIndex,0,0) << ";" << endl;
	  fout << "y{"<<globalCellInd+cellIndex<< "}("<<yPointIndex+1<<")=" << physCubPoints(cellIndex,0,1) << ";" << endl;
	  fout << "z{"<<globalCellInd+cellIndex<< "}("<<xPointIndex+1<<","<<yPointIndex+1<<")=" << computedValues(cellIndex,0) << ";" << endl;	  
	}
      }
    }
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
        for (int pointIndex = 0; pointIndex < numCubPoints; pointIndex++){
          for (int dimInd=0;dimInd<spaceDim;dimInd++){
            fout << physCubPoints(cellIndex,pointIndex,dimInd) << " ";
          }
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

Epetra_Map Solution::getPartitionMap(int rank, set<int> & myGlobalIndicesSet, int numGlobalDofs, int zeroMeanConstraintsSize, Epetra_Comm* Comm ) {
  // determine the local dofs we have, and what their global indices are:
  int localDofsSize;
  if (rank == 0) {
    localDofsSize = myGlobalIndicesSet.size() + zeroMeanConstraintsSize;
  } else {
    localDofsSize = myGlobalIndicesSet.size();
  }
  int *myGlobalIndices;
  if (localDofsSize!=0){
    myGlobalIndices = new int[ localDofsSize ];      
  }else{
    myGlobalIndices = NULL;
  }
    
  // copy from set object into the allocated array
  int offset = 0;
  for ( set<int>::iterator indexIt = myGlobalIndicesSet.begin();
	indexIt != myGlobalIndicesSet.end();
	indexIt++ ){
    myGlobalIndices[offset++] = *indexIt;
  }
  if ( rank == 0 ) {
    // set up the zmcs, which come at the end...
    for (int i=0; i<zeroMeanConstraintsSize; i++) {
      myGlobalIndices[offset++] = i + numGlobalDofs;
    }
  }
    
  int indexBase = 0;
  //cout << "process " << rank << " about to construct partMap.\n";
  //Epetra_Map partMap(-1, localDofsSize, myGlobalIndices, indexBase, Comm);
  Epetra_Map partMap(numGlobalDofs+zeroMeanConstraintsSize, localDofsSize, myGlobalIndices, indexBase, *Comm);

  if (localDofsSize!=0){
    delete myGlobalIndices;
  }
  return partMap;
}
