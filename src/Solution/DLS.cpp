//
//  DLS.cpp
//  Camellia
//
//  Created by Roberts, Nathan V on 3/7/17.
//
//

#include "CamelliaDebugUtility.h"
#include "DLS.h"
#include "GlobalDofAssignment.h"
#include "TimeLogger.h"

#include "Tpetra_FEMultiVector.hpp"

static const int MAX_BATCH_SIZE_IN_BYTES = 3*1024*1024; // 3 MB
static const int MIN_BATCH_SIZE_IN_CELLS = 1; // overrides the above, if it results in too-small batches

using namespace Camellia;
using namespace Teuchos;
using namespace Tpetra;

template<typename Scalar>
DLS<Scalar>::DLS(TSolutionPtr<Scalar> solution)
{
  _soln = solution;
}

template<typename Scalar>
void inverseSquareRoot(RCP<Vector<Scalar>> vector)
{
  auto diagValues_2d_device = vector->template getLocalView<Kokkos::DefaultExecutionSpace> ();
  auto diagValues_2d = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), diagValues_2d_device);
  // getLocalView returns a 2-D View by default.  We want a 1-D
  // View, so we take a subview.
  auto diagValues_1d = Kokkos::subview (diagValues_2d, Kokkos::ALL (), 0);
  
  int mySolnDofCount = vector->getMap()->getNodeNumElements();
  for (IndexType localID=0; localID<mySolnDofCount; localID++)
  {
    Scalar squaredValue = diagValues_1d(localID);
    if (squaredValue == 0.0)
    {
      auto GID = vector->getMap()->getGlobalElement(localID);
      cout << "WARNING: for globalID " << GID << ", encountered zero value in inverseSquareRoot().\n";
    }
    diagValues_1d(localID) = 1.0 / sqrt(squaredValue);
//    cout << "diagValues_1d(" << localID << ") = " << diagValues_1d(localID) << endl;
  }
}

template<typename Scalar>
int DLS<Scalar>::assemble()
{
  using namespace Tpetra;
  using namespace Teuchos;
  
  using LO = ::Tpetra::Details::DefaultTypes::local_ordinal_type;
  using GO = ::Tpetra::Details::DefaultTypes::global_ordinal_type;
  
  // will take BCs into account
  
  MeshPtr mesh = _soln->mesh();
  BCPtr bc = _soln->bc();
  auto dofInterpreter = _soln->getDofInterpreter();
  
  Epetra_CommPtr Comm = mesh->Comm();
  int rank = Comm->MyPID();
  int numProcs = Comm->NumProc();
  
  _soln->initializeLHSVector();
  RCP<Epetra_FEVector> solnLHS = _soln->getLHSVector();
  Epetra_BlockMap solnMap = solnLHS->Map();
  
  /**** Start BC determination ****/
  Camellia::FieldContainer<GlobalIndexType> bcGlobalIndicesFC;
  Camellia::FieldContainer<Scalar> bcGlobalValuesFC;
  
  mesh->boundary().bcsToImpose(bcGlobalIndicesFC,bcGlobalValuesFC,*bc,_soln->getDofInterpreter().get());
  
  map<int,vector<pair<GO,Scalar>>> bcValuesToSend; // key to outer map: recipient PID
  vector<pair<GO,Scalar>> bcsToImposeThisRank;
  for (int i=0; i<bcGlobalIndicesFC.size(); i++)
  {
    GO globalIndex = bcGlobalIndicesFC[i];
    double value = bcGlobalValuesFC[i];
    int owner = dofInterpreter->partitionForGlobalDofIndex(globalIndex);
    if (owner != rank)
    {
      bcValuesToSend[owner].push_back({globalIndex,value});
    }
    else
    {
      bcsToImposeThisRank.push_back({globalIndex,value});
    }
  }
  vector<pair<GO,Scalar>> offRankBCs;
  MPIWrapper::sendDataVectors(mesh->Comm(), bcValuesToSend, offRankBCs);
  
  bcsToImposeThisRank.insert(bcsToImposeThisRank.begin(),offRankBCs.begin(),offRankBCs.end());
  std::sort(bcsToImposeThisRank.begin(),bcsToImposeThisRank.end());
  
  int warnAboutDiscontinuousBCs = _soln->warnAboutDiscontinuousBCs();
  // we may have some duplicates (multiple ranks may have ideas about our BC's values)
  // count them:
  int numDuplicates = 0;
  int numBCsIncludingDuplicates = bcsToImposeThisRank.size();
  bool foundDiscontinuousBC = false;
  for (int i=0; i<numBCsIncludingDuplicates-1; i++)
  {
    if (bcsToImposeThisRank[i].first == bcsToImposeThisRank[i+1].first)
    {
      numDuplicates++;
      // while we're at it, let's check that the two values are within some reasonable tolerance of each other
      double tol = 1e-10;
      double firstValue = bcsToImposeThisRank[i].second;
      double secondValue = bcsToImposeThisRank[i+1].second;
      // two ways to pass:
      // (1) if their absolute difference is below tol
      // (2) if their absolute difference is above tol, but their relative difference is below tol
      double absDiff = abs(firstValue - secondValue);
      if (absDiff > tol)
      {
        double relativeDiff = absDiff / max(abs(firstValue),abs(secondValue));
        if (relativeDiff > tol)
        {
          foundDiscontinuousBC = true;
          if (warnAboutDiscontinuousBCs >= 2)
          {
            cout << "WARNING: inconsistent values for BC: " << firstValue << " and ";
            cout << secondValue << " prescribed for global dof index " << bcsToImposeThisRank[i].first;
            cout << " on rank " << rank << endl;
            print("initialH1Order for inconsistent BC mesh",mesh->globalDofAssignment()->getInitialH1Order());
          }
        }
      }
    }
  }
  
  if (warnAboutDiscontinuousBCs == 1)
  {
    // print a simple warning on rank 0
    foundDiscontinuousBC = MPIWrapper::globalOr(*mesh->Comm(), foundDiscontinuousBC);
    if (foundDiscontinuousBC && (rank == 0))
    {
      cout << "WARNING: discontinuous boundary conditions detected.  Call Solution::setWarnAboutDiscontinuousBCs() with outputLevel=0 to suppress this warning; with outputLevel=2 for full details about the differing values\n";
    }
  }
  
  int numBCs = bcsToImposeThisRank.size()-numDuplicates;
  vector<GO> bcGlobalIndices(numBCs);
  vector<double> bcGlobalValues(numBCs);
  int i_adjusted = 0; // adjusted to eliminate duplicates
  for (int i=0; i<bcsToImposeThisRank.size(); i++)
  {
    double value = bcsToImposeThisRank[i].second;
    int thisBCduplicateCount = 1;
    while ((i+1<numBCsIncludingDuplicates) && (bcsToImposeThisRank[i].first == bcsToImposeThisRank[i+1].first))
    {
      thisBCduplicateCount++;
      i++;
      value += bcsToImposeThisRank[i].second;
    }
    value /= thisBCduplicateCount; // average all values together
    bcGlobalIndices[i_adjusted] = static_cast<GO>(bcsToImposeThisRank[i].first);
    bcGlobalValues[i_adjusted] = bcsToImposeThisRank[i].second;
    i_adjusted++;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(numBCs != i_adjusted, std::invalid_argument, "internal error: numBCs != i_adjusted");
  
  Teuchos_CommPtr TeuchosComm = mesh->TeuchosComm();
  
  GO indexBase = 0;
  GO invalid = OrdinalTraits<global_size_t>::invalid();
  typedef RCP<Map<LO,GO>> MapRCP;
  MapRCP bcMap = rcp( new Map<LO,GO>(invalid, &bcGlobalIndices[0], bcGlobalIndices.size(),
                                               indexBase, TeuchosComm));
  
  int localDofCountWithBCs = solnMap.NumMyElements();
  int localDofCountNoBCs = localDofCountWithBCs - numBCs;
  vector<GO> myGlobalIndicesNoBCs(localDofCountNoBCs);
  int noBC_LID = 0;
  for (int i=0; i<localDofCountWithBCs; i++)
  {
    GO globalID = solnMap.GID(i);
    if (std::find(bcGlobalIndices.begin(),bcGlobalIndices.end(),globalID) == bcGlobalIndices.end())
    {
      myGlobalIndicesNoBCs[noBC_LID] = solnMap.GID(i);
      noBC_LID++;
    }
  }
  
  solnLHS->ReplaceGlobalValues(bcGlobalValues.size(),&bcGlobalIndices[0],&bcGlobalValues[0]);
  
  TVector<Scalar> bcValues(bcMap,1);
  size_t colZero=0;
  for (int i=0; i<bcGlobalValues.size(); i++)
  {
    bcValues.replaceLocalValue(i,colZero,bcGlobalValues[i]);
    // TODO:  figure out if it's worth revealing to the outside world which GIDs have been eliminated through BCs
    //        (if so, here would be a place to record those things)
    // GO fullSolnGID = bcGlobalIndices[i];
    
  }
  
  MapRCP solnMapNoBCs = rcp( new Map<LO,GO>(invalid, &myGlobalIndicesNoBCs[0], localDofCountNoBCs,
                                                                indexBase, TeuchosComm));
  
  /****** End BC determination ******/
  
  /*
   Unlike the usual (normal equation) case, we do need to establish a global distribution for the test
   dofs.  These are completely discontinuous, so the only communication happens due to the trial space
   overlap between ranks.
   */
//  GO testIDOffset;
//  int ownedCellCount = mesh->cellIDsInPartition().size();
  GO localTestCount = 0;
  LO maxTrialCount = 0; // on this processor
  for (GlobalIndexType cellID : mesh->cellIDsInPartition())
  {
    localTestCount += mesh->getElementType(cellID)->testOrderPtr->totalDofs();
    maxTrialCount = max(maxTrialCount,mesh->getElementType(cellID)->trialOrderPtr->totalDofs());
  }
  typedef Map<LO,GO> Map;
  typedef RCP<Map> MapRCP;
  MapRCP testMap = rcp( new Map(invalid, localTestCount, indexBase, TeuchosComm) );
  GO localTestOffset = localTestCount;
  mesh->Comm()->ScanSum(&localTestCount, &localTestOffset, 1);
  localTestOffset -= localTestCount;

  GO elementTestOffset = localTestOffset;
  map<GO,GO> elementTestIDOffsets;
  for (GlobalIndexType cellID : mesh->cellIDsInPartition())
  {
    elementTestIDOffsets[cellID] = elementTestOffset;
    elementTestOffset += mesh->getElementType(cellID)->testOrderPtr->totalDofs();
  }
  
  auto DynamicProfileType = Tpetra::ProfileType(Tpetra::StaticProfile+1);
  auto StaticProfileType  = Tpetra::ProfileType(Tpetra::StaticProfile);
  _dlsMatrix = rcp(new TMatrix<Scalar>(testMap,solnMapNoBCs,maxTrialCount,DynamicProfileType));
  CrsMatrix<Scalar,LO,GO> bcImpositionMatrix(testMap,numBCs,StaticProfileType);
  int numCols = 1;
  _rhsVector = rcp(new TVector<Scalar>(testMap,numCols));
  _lhsVector = rcp(new TVector<Scalar>(solnMapNoBCs,numCols));
  
//  cout << "Test map has " << testMap->getGlobalNumElements() << " global elements.\n";
//  cout << "Trial map has " << solnMapNoBCs->getGlobalNumElements() << " global elements.\n";

  bool diagFillerFixed = false;
  
  TEUCHOS_TEST_FOR_EXCEPTION(!diagFillerFixed, invalid_argument, "Implementation incomplete - rewrite with MultiVectorFiller replacement!");
  // commented out diagFiller here and below because Tpetra has removed MultiVectorFiller.
  // (The DLS code here is a prototype implementation that is not currently used anywhere...)
  //MultiVectorFiller<TVector<Scalar>> diagFiller(solnMapNoBCs, numCols);
 
  if (diagFillerFixed)
  {
    vector< ElementTypePtr > elementTypes = mesh->elementTypes(rank);
    
    int cubatureEnrichmentDegree = _soln->cubatureEnrichmentDegree();
    IPPtr ip = _soln->ip();
    BFPtr bf = _soln->bf();
    RHSPtr rhs = _soln->rhs();
    if (bf == null)
    {
      bf = mesh->bilinearForm();
    }
    
    Epetra_Map timeMap(static_cast<int>(numProcs),static_cast<int>(indexBase),*Comm);
    Epetra_Time timer(*Comm);
    Epetra_Time subTimer(*Comm);
    
    double localStiffnessInterpretationTime = 0;
    //  cout << "Computing local matrices" << endl;
    for (ElementTypePtr elemTypePtr : elementTypes)
    {
      //cout << "Solution: elementType loop, iteration: " << elemTypeNumber++ << endl;
      
      Camellia::FieldContainer<double> myPhysicalCellNodesForType = mesh->physicalCellNodes(elemTypePtr);
      Camellia::FieldContainer<double> myCellSideParitiesForType = mesh->cellSideParities(elemTypePtr);
      int totalCellsForType = myPhysicalCellNodesForType.dimension(0);
      int startCellIndexForBatch = 0;
      
      if (totalCellsForType == 0) continue;
      // if we get here, there is at least one, so we find a sample cellID to help us set up prototype BasisCaches:
      
      // determine cellIDs
      vector<GlobalIndexType> cellIDsOfType = mesh->globalDofAssignment()->cellIDsOfElementType(rank, elemTypePtr);
      
      GlobalIndexType sampleCellID = cellIDsOfType[0];
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh,sampleCellID,false,cubatureEnrichmentDegree);
      BasisCachePtr ipBasisCache = BasisCache::basisCacheForCell(mesh,sampleCellID,true,cubatureEnrichmentDegree);
      
      DofOrderingPtr trialOrderingPtr = elemTypePtr->trialOrderPtr;
      DofOrderingPtr testOrderingPtr = elemTypePtr->testOrderPtr;
      int numTrialDofs = trialOrderingPtr->totalDofs();
      int numTestDofs = testOrderingPtr->totalDofs();
      int maxCellBatch = MAX_BATCH_SIZE_IN_BYTES / 8 / (numTestDofs*numTestDofs + numTestDofs*numTrialDofs + numTrialDofs*numTrialDofs);
      maxCellBatch = max( maxCellBatch, MIN_BATCH_SIZE_IN_CELLS );
      
      Array<int> nodeDimensions, parityDimensions;
      myPhysicalCellNodesForType.dimensions(nodeDimensions);
      myCellSideParitiesForType.dimensions(parityDimensions);
      
      Camellia::FieldContainer<Scalar> localStiffness(maxCellBatch,numTrialDofs,numTestDofs);
      Camellia::FieldContainer<Scalar> localRHSVector(maxCellBatch,numTestDofs);
      
      while (startCellIndexForBatch < totalCellsForType)
      {
        int cellsLeft = totalCellsForType - startCellIndexForBatch;
        int numCells = min(maxCellBatch,cellsLeft);
        localStiffness.resize(numCells,numTrialDofs,numTestDofs);
        localRHSVector.resize(numCells,numTestDofs);
        
        vector<GlobalIndexType> cellIDs;
        for (int cellIndex=0; cellIndex<numCells; cellIndex++)
        {
          GlobalIndexType cellID = cellIDsOfType[cellIndex+startCellIndexForBatch];
          cellIDs.push_back(cellID);
        }
        
        nodeDimensions[0] = numCells;
        parityDimensions[0] = numCells;
        Camellia::FieldContainer<double> physicalCellNodes(nodeDimensions,&myPhysicalCellNodesForType(startCellIndexForBatch,0,0));
        Camellia::FieldContainer<double> cellSideParities(parityDimensions,&myCellSideParitiesForType(startCellIndexForBatch,0));
        
        bool createSideCacheToo = true;
        basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,createSideCacheToo);
        basisCache->setCellSideParities(cellSideParities);
        
        // hard-coding creating side cache for IP for now, since _ip->hasBoundaryTerms() only recognizes terms explicitly passed in as boundary terms:
        ipBasisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true);//_ip->hasBoundaryTerms()); // create side cache if ip has boundary values
        ipBasisCache->setCellSideParities(cellSideParities); // I don't anticipate these being needed, though
        
        bf->localStiffnessMatrixAndRHS_DLS(localStiffness, localRHSVector, ip, ipBasisCache, rhs, basisCache);
        
        subTimer.ResetStartTime();
        
        Camellia::FieldContainer<GlobalIndexType> globalDofIndicesFC;
        Array<GO> globalDofIndicesNoBCs;
        Array<GO> globalDofIndicesBCs;
        
        Array<int> localStiffnessDim = vector<int>{numTrialDofs,numTestDofs};
        Array<int> localRHSDim       = vector<int>{numTestDofs};
        
        Camellia::FieldContainer<Scalar> interpretedStiffness;
        
        Camellia::FieldContainer<Scalar> interpretedStiffnessNoBCs;
        Camellia::FieldContainer<Scalar> interpretedStiffnessBCs;
        
        Array<int> dim;
        // determine cellIDs
        vector<GlobalIndexType> cellIDsOfType = mesh->globalDofAssignment()->cellIDsOfElementType(rank, elemTypePtr);
        
        vector<GO> testGlobalIndices(numTestDofs);
        Array<Scalar> rowValues;
        
        for (int cellIndex=0; cellIndex<numCells; cellIndex++)
        {
          GlobalIndexType cellID = cellIDsOfType[cellIndex+startCellIndexForBatch];
          Camellia::FieldContainer<Scalar> cellStiffness(localStiffnessDim,&localStiffness(cellIndex,0,0)); // shallow copy
          Camellia::FieldContainer<Scalar> cellRHS(localRHSDim,&localRHSVector(cellIndex,0)); // shallow copy
          
          elementTestOffset = elementTestIDOffsets[cellID];
          for (int i=0; i<numTestDofs; i++)
          {
            testGlobalIndices[i] = static_cast<GO>(i + elementTestOffset);
            // "replace" below because we know, a priori, that this test globalID is only on this element
            // (implied by the discontinuity of test functions)
            _rhsVector->sumIntoGlobalValue(testGlobalIndices[i],colZero,cellRHS(i));
          }
          
          dofInterpreter->interpretLocalData(cellID, cellStiffness, interpretedStiffness, globalDofIndicesFC);
          
          
          int numGlobalDofIndicesNoBCs = 0;
          for (int i=0; i<globalDofIndicesFC.size(); i++)
          {
            GlobalIndexType solnGlobalID = globalDofIndicesFC(i);
            if (std::find(bcGlobalIndices.begin(),bcGlobalIndices.end(),solnGlobalID) == bcGlobalIndices.end())
            {
              // then this is not an eliminated global ID
              numGlobalDofIndicesNoBCs++;
            }
          }
          // numBCs imposed on this element:
          int numBCs = globalDofIndicesFC.size() - numGlobalDofIndicesNoBCs;
          
  //        if (numBCs == 0)
  //        {
  //          // no BCs on this element; can simply store everything in leastSquaresMatrix
  //          int numTrialValues = numGlobalDofIndicesNoBCs;
  //          rowValues.resize(numTrialValues); // trial space size
  //
  //          for (int j=0; j<numTestDofs; j++)
  //          {
  //            for (int i=0; i<numTrialValues; i++)
  //            {
  //              rowValues[i] = interpretedStiffness(i,j);
  //            }
  //            GO globalTestID = static_cast<GO>(elementTestOffset + j);
  //            _dlsMatrix->insertGlobalValues(globalTestID, numTrialValues, &rowValues[0], &globalDofIndices(0));
  //          }
  //        }
  //        else
          int bcOffset = 0, noBCOffset = 0;
          // partition things into BC and non-BC rows
          globalDofIndicesNoBCs.resize(numGlobalDofIndicesNoBCs);
          interpretedStiffnessNoBCs.resize(numGlobalDofIndicesNoBCs,numTestDofs);
          
          interpretedStiffnessBCs.resize(numBCs, numTestDofs);
          globalDofIndicesBCs.resize(numBCs);
          
          for (int i=0; i<globalDofIndicesFC.size(); i++)
          {
            GO solnGlobalID = globalDofIndicesFC(i);
            if (std::find(bcGlobalIndices.begin(),bcGlobalIndices.end(),solnGlobalID) == bcGlobalIndices.end())
            {
              // then this is not an eliminated global ID
              globalDofIndicesNoBCs[noBCOffset] = static_cast<GO>(globalDofIndicesFC(i));
              for (int j=0; j<numTestDofs; j++)
              {
                interpretedStiffnessNoBCs(noBCOffset,j) = interpretedStiffness(i,j);
              }
              
              noBCOffset++;
            }
            else
            {
              globalDofIndicesBCs[bcOffset] = static_cast<GO>(globalDofIndicesFC(i));
              for (int j=0; j<numTestDofs; j++)
              {
                interpretedStiffnessBCs(bcOffset,j) = interpretedStiffness(i,j);
              }
              
              bcOffset++;
            }
          }
          
          // insert into leastSquaresMatrix
          rowValues.resize(numGlobalDofIndicesNoBCs);
          for (int j=0; j<numTestDofs; j++)
          {
            for (int i=0; i<rowValues.size(); i++)
            {
              rowValues[i] = interpretedStiffnessNoBCs(i,j);
            }
            
            GO globalTestID = elementTestOffset + j;
            _dlsMatrix->insertGlobalValues(globalTestID, globalDofIndicesNoBCs, rowValues);
            
            for (int i=0; i<rowValues.size(); i++)
            {
              rowValues[i] = rowValues[i] * rowValues[i];
  //            cout << "adding " << rowValues[i] << " to diagonal value with global index " << globalDofIndicesNoBCs[i] << endl;
            }
  //          diagFiller.sumIntoGlobalValues(globalDofIndicesNoBCs, 0, rowValues);
          }
          
          // insert into bcImpositionMatrix
          rowValues.resize(numBCs);
          for (int j=0; j<numTestDofs; j++)
          {
            for (int i=0; i<rowValues.size(); i++)
            {
              rowValues[i] = interpretedStiffnessBCs(i,j);
            }
            
            GlobalIndexType globalTestID = elementTestOffset + j;
            bcImpositionMatrix.insertGlobalValues(globalTestID, globalDofIndicesBCs, rowValues);
          }
        }
        localStiffnessInterpretationTime += subTimer.ElapsedTime();
        
        startCellIndexForBatch += numCells;
      }
    }

    // adjust RHS to account for the BCs we've eliminated
    // compute _rhsVector -= bcImposition * bcValues
    bcImpositionMatrix.fillComplete();
    
    // It appears that apply(X,Y) requires X to have a map that matches the domain map of
    // bcImpositionMatrix, and Y to have a map that matches the range map.  It's hard to believe
    // that all of the following is actually necessary (or maybe there is something I can do better
    // in terms of setting up the various maps in the first place -- maybe it would be better to expand
    // bcMap beyond the indices that match BCs, for instance).
    TVector<Scalar> bcValues_domainMap(bcImpositionMatrix.getDomainMap(),1);
    Tpetra::Import<LO,GO> domainMapImporter(bcMap, bcImpositionMatrix.getDomainMap());
    bcValues_domainMap.doImport(bcValues, domainMapImporter, INSERT);
    TVector<Scalar> bcApplication_rangeMap(bcImpositionMatrix.getRangeMap(),1);
    bcImpositionMatrix.apply(bcValues_domainMap,bcApplication_rangeMap,NO_TRANS);
    TVector<Scalar> bcApplication_rhsMap(_rhsVector->getMap(),1);
    Tpetra::Import<LO,GO> rhsMapImporter(bcApplication_rangeMap.getMap(), bcApplication_rhsMap.getMap());
    bcApplication_rhsMap.doImport(bcApplication_rangeMap, rhsMapImporter, INSERT);
    _rhsVector->update(-1.0, bcApplication_rhsMap, 1.0);
    
    _dlsMatrix->fillComplete(solnMapNoBCs,testMap);
  //  cout << "_dlsMatrix dimensions: " << _dlsMatrix->getGlobalNumRows() << " x " << _dlsMatrix->getGlobalNumCols() << endl;
  //  cout << "_dlsMatrix column Map, global element count: " << _dlsMatrix->getColMap()->getGlobalNumElements() << endl;
  //  cout << "_dlsMatrix domain Map, global element count: " << _dlsMatrix->getDomainMap()->getGlobalNumElements() << endl;
  //  cout << "_dlsMatrix row Map, global element count: " << _dlsMatrix->getRowMap()->getGlobalNumElements() << endl;
  //  cout << "_dlsMatrix range Map, global element count: " << _dlsMatrix->getRangeMap()->getGlobalNumElements() << endl;
    
    _diag_sqrt_inverse = rcp( new Vector<Scalar>(solnMapNoBCs) );
  //  diagFiller.globalAssemble(*_diag_sqrt_inverse);
    std::cout << "WARNING: due to Tpetra's replacement of MultiVectorFiller, this code is at present incomplete.  Need to adopt Tpetra_FEMultiVector.\n";
    inverseSquareRoot(_diag_sqrt_inverse);
    
    RCP<Vector<Scalar,LO,GO>> matrixScalingVector = rcp( new Vector<Scalar,LO,GO>(_dlsMatrix->getDomainMap()) );

    Tpetra::Import<LO,GO> scalingImporter(_diag_sqrt_inverse->getMap(), matrixScalingVector->getMap());
    matrixScalingVector->doImport(*_diag_sqrt_inverse, scalingImporter, INSERT);
    
    _dlsMatrix->rightScale(*matrixScalingVector);

    // Commented out because I'm pretty sure this should be redundant with the (identical) code above
  //  // now that we've scaled the DLS matrix, let's use diagFiller to export to _diag_sqrt_inverse constructed with
  //  // the map we'll want to use to scale the solution:
  //  _diag_sqrt_inverse = rcp( new Vector<Scalar,LO,GO>(solnMapNoBCs) );
  //  diagFiller.globalAssemble(*_diag_sqrt_inverse);
  //  inverseSquareRoot(_diag_sqrt_inverse);
  } // if (diagFillerFixed)

  return 0; // success
}

template<typename Scalar>
int DLS<Scalar>::solveProblemLSQR(int maxIters, double tol)
{
  // TODO: do the actual solve here.
  
  
  // after we've solved, because of the diag scaling in assemble, we have really solved
  // something like A * D^-1/2 * y = b.  What we want is a solution to A * x = b, so here
  // x = D^-1/2 * y.
  
  _lhsVector->elementWiseMultiply(1.0, *_diag_sqrt_inverse, *_lhsVector, 0.0);

  // TODO: initialize _soln's lhsVector using our _lhsVector
  
  cout << "solveProblemLSQR: implementation incomplete!\n";
  return -1; //
}

// test by trial
template<typename Scalar>
TMatrixPtr<Scalar> DLS<Scalar>::matrix()
{
  return _dlsMatrix;
}

// the solution vector (with all Dirichlet dofs eliminated)
template<typename Scalar>
TVectorPtr<Scalar> DLS<Scalar>::lhs()
{
  return _lhsVector;
}

template<typename Scalar>
TVectorPtr<Scalar> DLS<Scalar>::normalDiagInverseSqrt()
{
  return _diag_sqrt_inverse;
}

// the "enriched" RHS vector (has length = # test dofs)
template<typename Scalar>
TVectorPtr<Scalar> DLS<Scalar>::rhs()
{
  return _rhsVector;
}

// underlying Solution object; lhs will be initialized with our solution
template<typename Scalar>
TSolutionPtr<Scalar> DLS<Scalar>::solution()
{
  return _soln;
}

// The below method commented out because it seems like a relic, before we decided to maintain
// numbering between the two -- the only difference is that solution()->lhsVector() has some extra
// dofs, which are set to whatever the BCs were...  The map is an identity map, albeit a non-bijective one...
//
// Map to allow translation between the Solution returned by DLS::solution()->lhsVector()
// and that returned by DLS::lhs()
//template<typename Scalar>
//SolutionMapPtr DLS<Scalar>::solutionMapPtr()
//{
//  return _dlsToSolnMap;
//}

// ETI for double 
template class Camellia::DLS<double>;
