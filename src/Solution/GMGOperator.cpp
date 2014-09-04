//
//  GMGOperator.cpp
//  Camellia
//
//  Created by Nate Roberts on 7/3/14.
//
//

#include "CamelliaConfig.h"

#include "GMGOperator.h"
#include "GlobalDofAssignment.h"
#include "BasisSumFunction.h"

#include "GDAMinimumRule.h"

#include "CamelliaCellTools.h"

// EpetraExt includes
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Epetra_SerialComm.h"

GMGOperator::GMGOperator(BCPtr zeroBCs, MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Epetra_Map finePartitionMap, Teuchos::RCP<Solver> coarseSolver) :  _finePartitionMap(finePartitionMap), _br(true) {
  RHSPtr zeroRHS = RHS::rhs();
  _fineMesh = fineMesh;
  _coarseMesh = coarseMesh;
  _bc = zeroBCs;
  _coarseSolution = Teuchos::rcp( new Solution(coarseMesh, zeroBCs, zeroRHS, coarseIP) );
  _coarseSolver = coarseSolver;
  
  if (( coarseMesh->meshUsesMaximumRule()) || (! fineMesh->meshUsesMinimumRule()) ) {
    cout << "GMGOperator only supports minimum rule.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "GMGOperator only supports minimum rule.");
  }
  
  _coarseSolution->initializeLHSVector();
  _coarseSolution->initializeStiffnessAndLoad();
  _coarseSolution->populateStiffnessAndLoad(); // can get away with doing this just once; after that we just manipulate the RHS vector
}

GlobalIndexType GMGOperator::getCoarseCellID(GlobalIndexType fineCellID) const {
  set<GlobalIndexType> coarseCellIDs = _coarseMesh->getActiveCellIDs();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;
  while (coarseCellIDs.find(ancestor->cellIndex()) == coarseCellIDs.end()) {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  return ancestor->cellIndex();
}

LocalDofMapperPtr GMGOperator::getLocalCoefficientMap(GlobalIndexType fineCellID) const {
  set<GlobalIndexType> coarseCellIDs = _coarseMesh->getActiveCellIDs();
  CellPtr fineCell = _fineMesh->getTopology()->getCell(fineCellID);
  CellPtr ancestor = fineCell;
  RefinementBranch refBranch;
  while (coarseCellIDs.find(ancestor->cellIndex()) == coarseCellIDs.end()) {
    CellPtr parent = ancestor->getParent();
    if (parent.get() == NULL) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ancestor for fine cell not found in coarse mesh");
    }
    unsigned childOrdinal = parent->childOrdinal(ancestor->cellIndex());
    refBranch.insert(refBranch.begin(), make_pair(parent->refinementPattern().get(), childOrdinal));
    ancestor = parent;
  }
  GlobalIndexType coarseCellID = ancestor->cellIndex();
  CellPtr coarseCell = ancestor;
  int fineOrder = _fineMesh->globalDofAssignment()->getH1Order(fineCellID);
  int coarseOrder = _coarseMesh->globalDofAssignment()->getH1Order(coarseCellID);
  
  DofOrderingPtr coarseTrialOrdering = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr;
  DofOrderingPtr fineTrialOrdering = _fineMesh->getElementType(fineCellID)->trialOrderPtr;
  
  pair< pair<int,int>, RefinementBranch > key = make_pair(make_pair(fineOrder, coarseOrder), refBranch);
  
  int fineSideCount = CamelliaCellTools::getSideCount( *fineCell->topology() );
  int sideDim = _fineMesh->getTopology()->getSpaceDim() - 1;
  vector<unsigned> ancestralSideOrdinals(fineSideCount);
  vector< RefinementBranch > sideRefBranches(fineSideCount);
  for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++) {
    ancestralSideOrdinals[sideOrdinal] = RefinementPattern::ancestralSubcellOrdinal(refBranch, sideDim, sideOrdinal);
    if (ancestralSideOrdinals[sideOrdinal] != -1) {
      sideRefBranches[sideOrdinal] = RefinementPattern::sideRefinementBranch(refBranch, sideOrdinal);
    }
  }
  
  if (_localCoefficientMap.find(key) == _localCoefficientMap.end()) {
    VarFactory vf = _fineMesh->bilinearForm()->varFactory();
    
    typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
    map< int, BasisMap > volumeMaps;
 
    vector< map< int, BasisMap > > sideMaps(fineSideCount);
    
    set<int> trialIDs = coarseTrialOrdering->getVarIDs();

    // for the moment, we skip the mapping from traces to fields based on traceTerm
    SubBasisReconciliationWeights weights;
    unsigned vertexNodePermutation = 0; // because we're "reconciling" to an ancestor, the views of the cells and sides are necessarily the same
    for (set<int>::iterator trialIDIt = trialIDs.begin(); trialIDIt != trialIDs.end(); trialIDIt++) {
      int trialID = *trialIDIt;
      if (coarseTrialOrdering->getNumSidesForVarID(trialID) == 1) { // field variable
//        cout << "Warning: for debugging purposes, skipping projection of fields in GMGOperator.\n";
        BasisPtr coarseBasis = coarseTrialOrdering->getBasis(trialID);
        BasisPtr fineBasis = fineTrialOrdering->getBasis(trialID);
        weights = _br.constrainedWeights(fineBasis, refBranch, coarseBasis, vertexNodePermutation);
        set<unsigned> fineDofOrdinals(weights.fineOrdinals.begin(),weights.fineOrdinals.end());
        
        vector<GlobalIndexType> coarseDofIndices;
        for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++) {
          unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(trialID, *coarseOrdinalIt);
          coarseDofIndices.push_back(coarseDofIndex);
        }
        BasisMap basisMap(1,SubBasisDofMapper::subBasisDofMapper(fineDofOrdinals, coarseDofIndices, weights.weights));
        volumeMaps[trialID] = basisMap;
      } else { // flux/trace
//        cout << "Warning: for debugging purposes, skipping projection of fluxes and traces in GMGOperator.\n";
        for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++) {
          unsigned coarseSideOrdinal = ancestralSideOrdinals[sideOrdinal];
          if (coarseSideOrdinal == -1) continue;
          
          BasisPtr coarseBasis = coarseTrialOrdering->getBasis(trialID, coarseSideOrdinal);
          BasisPtr fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);
          weights = _br.constrainedWeights(fineBasis, sideRefBranches[sideOrdinal], coarseBasis, vertexNodePermutation);
          set<unsigned> fineDofOrdinals(weights.fineOrdinals.begin(),weights.fineOrdinals.end());
          vector<GlobalIndexType> coarseDofIndices;
          for (set<int>::iterator coarseOrdinalIt=weights.coarseOrdinals.begin(); coarseOrdinalIt != weights.coarseOrdinals.end(); coarseOrdinalIt++) {
            unsigned coarseDofIndex = coarseTrialOrdering->getDofIndex(trialID, *coarseOrdinalIt, coarseSideOrdinal);
            coarseDofIndices.push_back(coarseDofIndex);
          }
          BasisMap basisMap(1,SubBasisDofMapper::subBasisDofMapper(fineDofOrdinals, coarseDofIndices, weights.weights));
          sideMaps[sideOrdinal][trialID] = basisMap;
        }
      }
    }
    
    // I don't think we need to do any fitting, so we leave the "fittable" containers for LocalDofMapper empty
    
    set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume;
    vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(fineSideCount);

    LocalDofMapperPtr dofMapper = Teuchos::rcp( new LocalDofMapper(fineTrialOrdering, volumeMaps, fittableGlobalDofOrdinalsInVolume,
                                                                   sideMaps, fittableGlobalDofOrdinalsOnSides) );
    _localCoefficientMap[key] = dofMapper;
  }

  LocalDofMapperPtr dofMapper = _localCoefficientMap[key];
  
  // now, correct side parities in dofMapper if the ref space situation differs from the physical space one.
  FieldContainer<double> coarseCellSideParities = _coarseMesh->globalDofAssignment()->cellSideParitiesForCell(coarseCellID);
  FieldContainer<double> fineCellSideParities = _fineMesh->globalDofAssignment()->cellSideParitiesForCell(fineCellID);
  set<unsigned> fineSidesToCorrect;
  for (unsigned fineSideOrdinal=0; fineSideOrdinal<fineSideCount; fineSideOrdinal++) {
    unsigned coarseSideOrdinal = ancestralSideOrdinals[fineSideOrdinal];
    if (coarseSideOrdinal != -1) { // ancestor shares side
      double coarseParity = coarseCellSideParities(0,coarseSideOrdinal);
      double fineParity = fineCellSideParities(0,fineSideOrdinal);
      if (coarseParity != fineParity) {
        fineSidesToCorrect.insert(fineSideOrdinal);
      }
    } else {
      // TODO: if/when we start using termTraced, should consider whether there is ever a case where the ref. space parities
      //       will be reversed relative to what happens on the fine cells.  I think the answer is that there probably is such
      //       a case; in this case, we will need to identify these and add them to fineSidesToCorrect.
    }
  }
  
  if (fineSidesToCorrect.size() > 0) {
    // copy before changing dofMapper:
    dofMapper = Teuchos::rcp( new LocalDofMapper(*dofMapper.get()) );
    set<int> fluxIDs;
    VarFactory vf = _fineMesh->bilinearForm()->varFactory();
    vector<VarPtr> fluxVars = vf.fluxVars();
    for (vector<VarPtr>::iterator fluxIt = fluxVars.begin(); fluxIt != fluxVars.end(); fluxIt++) {
      fluxIDs.insert((*fluxIt)->ID());
    }
    dofMapper->reverseParity(fluxIDs, fineSidesToCorrect);
  }
  
  return dofMapper;
}

int GMGOperator::Apply(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");
}

int GMGOperator::ApplyInverse(const Epetra_MultiVector& X_in, Epetra_MultiVector& Y) const {
//  cout << "GMGOperator::ApplyInverse.\n";
  
  Epetra_MultiVector X(X_in); // looks like Y may be stored in the same location as X_in, so that changing Y will change X, too...
  
  // the data coming in (X) is in global dofs defined on the fine mesh.  First thing we'd like to do is map it to the fine mesh's local cells
  set<GlobalIndexType> cellsInPartition = _fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local
  
  Teuchos::RCP<Epetra_FEVector> coarseRHSVector = _coarseSolution->getRHSVector();
  coarseRHSVector->PutScalar(0); // clear
  set<GlobalIndexTypeToCast> coarseDofIndicesToImport; // keep track of the coarse dof indices that this partition's fine cells talk to
  for (set<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++) {
    GlobalIndexType fineCellID = *cellIDIt;
    // filter by cell ownership to ensure that we only treat each fine global dof once:
    Teuchos::RCP<Epetra_MultiVector> X_filtered = filterMultiVectorByFineCellOwnership(X, fineCellID);
    int fineDofCount = _fineMesh->getElementType(fineCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> fineCellData(fineDofCount);
    _fineMesh->globalDofAssignment()->interpretGlobalCoefficients(fineCellID, fineCellData, *X_filtered);
//    cout << "fineCellData:\n" << fineCellData;
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
    GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
    int coarseDofCount = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> coarseCellData(coarseDofCount);
    FieldContainer<double> mappedCoarseCellData = fineMapper->mapLocalData(fineCellData, false);
//    cout << "mappedCoarseCellData:\n" << mappedCoarseCellData;
    vector<GlobalIndexType> mappedCoarseDofIndices = fineMapper->globalIndices();
    for (int mappedCoarseDofOrdinal = 0; mappedCoarseDofOrdinal < mappedCoarseDofIndices.size(); mappedCoarseDofOrdinal++) {
      GlobalIndexType coarseDofIndex = mappedCoarseDofIndices[mappedCoarseDofOrdinal];
      coarseCellData[coarseDofIndex] = mappedCoarseCellData[mappedCoarseDofOrdinal];
    }
    FieldContainer<double> interpretedCoarseData;
    FieldContainer<GlobalIndexType> interpretedGlobalDofIndices;
    _coarseMesh->interpretLocalData(coarseCellID, coarseCellData, interpretedCoarseData, interpretedGlobalDofIndices);
//    cout << "interpretedCoarseData:\n" << interpretedCoarseData;
    FieldContainer<GlobalIndexTypeToCast> interpretedGlobalDofIndicesCast(interpretedGlobalDofIndices.size());
    for (int interpretedDofOrdinal=0; interpretedDofOrdinal < interpretedGlobalDofIndices.size(); interpretedDofOrdinal++) {
      interpretedGlobalDofIndicesCast[interpretedDofOrdinal] = (GlobalIndexTypeToCast) interpretedGlobalDofIndices[interpretedDofOrdinal];
      coarseDofIndicesToImport.insert(interpretedGlobalDofIndicesCast[interpretedDofOrdinal]);
    }
    coarseRHSVector->SumIntoGlobalValues(interpretedCoarseData.size(), &interpretedGlobalDofIndicesCast[0], &interpretedCoarseData[0]);
  }
  coarseRHSVector->GlobalAssemble();
  
  // solve the coarse system:
  
  _coarseSolution->imposeBCs();
  _coarseSolution->setProblem(_coarseSolver);
  
  Teuchos::RCP<Epetra_FECrsMatrix> coarseStiffness = _coarseSolution->getStiffnessMatrix();
//  EpetraExt::RowMatrixToMatlabFile("/tmp/A_gmg.dat",*coarseStiffness);
//  EpetraExt::MultiVectorToMatlabFile("/tmp/b_gmg.dat",*coarseRHSVector);
  
  _coarseSolution->solveWithPrepopulatedStiffnessAndLoad(_coarseSolver);
  
//  { // DEBUGGING:
//    // put the RHS into FC for debugging purposes:
//    int numMyDofs = coarseRHSVector->MyLength();
//    FieldContainer<double> rhsFC(numMyDofs);
//    for (int i=0; i<numMyDofs; i++) {
//      rhsFC[i] = (*coarseRHSVector)[0][i];
//    }
//    cout << "before coarse solve, rhsFC:\n" << rhsFC;
//  }
  
  Teuchos::RCP<Epetra_FEVector> coarseLHSVector = _coarseSolution->getLHSVector();
//  EpetraExt::MultiVectorToMatlabFile("/tmp/x_gmg.dat",*coarseLHSVector);

  
  // now, map the coarse data back to the fine mesh, and add that into Y
  Y.PutScalar(0); // clear Y
  
  // import all the dofs of interest to us onto this MPI rank
  Epetra_Map coarseMap = _coarseSolution->getPartitionMap();
//  GlobalIndexTypeToCast numDofsGlobal = coarseMap.NumGlobalElements();
  GlobalIndexTypeToCast numMyDofs = coarseDofIndicesToImport.size();
  GlobalIndexTypeToCast myDofs[coarseDofIndicesToImport.size()];
  GlobalIndexTypeToCast* myDof = &myDofs[0];
  for (set<GlobalIndexTypeToCast>::iterator coarseDofIndexIt = coarseDofIndicesToImport.begin();
       coarseDofIndexIt != coarseDofIndicesToImport.end(); coarseDofIndexIt++) {
    *myDof = *coarseDofIndexIt;
    myDof++;
  }
  
  Epetra_Map     solnMap(-1, numMyDofs, myDofs, 0, Comm());
  Epetra_Import  solnImporter(solnMap, coarseMap);
  Epetra_Vector  coarseDofs(solnMap);
  coarseDofs.Import(*coarseLHSVector, solnImporter, Insert);
  
  { // DEBUGGING:
    // put the global coefficients into FC for debugging purposes:
    FieldContainer<double> globalCoefficientsFC(numMyDofs);
    for (int i=0; i<numMyDofs; i++) {
      GlobalIndexType globalIndex = myDofs[i];
      globalCoefficientsFC[i] = coarseDofs[globalIndex];
    }
//    cout << "after coarse solve, globalCoefficientsFC:\n" << globalCoefficientsFC;
  }
  
  for (set<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++) {
    GlobalIndexType fineCellID = *cellIDIt;
    int fineDofCount = _fineMesh->getElementType(fineCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> fineCellData(fineDofCount);
    
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
    GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
    
    int coarseDofCount = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> coarseCellData(coarseDofCount);
    
    _coarseMesh->globalDofAssignment()->interpretGlobalCoefficients(coarseCellID, coarseCellData, coarseDofs);
//    cout << "coarseCellData after coarse solve:\n" << coarseCellData;
    
    vector<GlobalIndexType> coarseCellMappedLocalIndices = fineMapper->globalIndices();
    
    FieldContainer<double> coarseCellMappedData(coarseCellMappedLocalIndices.size());
    for (int i=0; i<coarseCellMappedLocalIndices.size(); i++) {
      int coarseCellLocalIndex = coarseCellMappedLocalIndices[i];
      coarseCellMappedData[i] = coarseCellData[coarseCellLocalIndex];
    }
    
    FieldContainer<double> fineLocalCoefficients = fineMapper->mapGlobalCoefficients(coarseCellMappedData);
//    cout << "fineLocalCoefficients after coarse solve:\n" << fineLocalCoefficients;
    
    _fineMesh->globalDofAssignment()->interpretLocalCoefficients(fineCellID, fineLocalCoefficients, Y);
  }
  
//  static int globalIterationCount = 0; // for debugging
//  ostringstream X_file;
//  X_file << "/tmp/X_" << globalIterationCount << ".dat";
//  EpetraExt::MultiVectorToMatlabFile(X_file.str().c_str(),X);
//
//  ostringstream Y_file;
//  Y_file << "/tmp/Y_before_diag_" << globalIterationCount << ".dat";
//  
//
//  EpetraExt::MultiVectorToMatlabFile(Y_file.str().c_str(),Y);
  
  // if diag is set, add diag(A)^(-1)X to Y.
  if (_diag.get() != NULL) {
    Epetra_BlockMap partitionMap = Y.Map();
    for (int localID = 0; localID < partitionMap.NumMyElements(); localID++) {
      GlobalIndexTypeToCast globalID = partitionMap.GID(localID);
      double diagEntry = (*_diag)[0][localID];
      double xEntry = X[0][localID];
      double yEntry = Y[0][localID];
      
      
//      cout << "xEntry = " << xEntry << "; yEntry = " << yEntry << endl;
      bool applyDiagonalTermsOnlyForZeroY = false;
      if (applyDiagonalTermsOnlyForZeroY) {
        if (yEntry == 0) {
          yEntry = xEntry/diagEntry;
        }
      } else {
        yEntry += xEntry/diagEntry;
      }
      
      Y.ReplaceGlobalValue(globalID, 0, yEntry);
      
//      cout << "Adding " << xEntry / diagEntry << " to global ID " << globalID << endl;
    }
  } else {
    cout << "_diag is NULL.\n";
  }

//  Y_file.str("");
//  Y_file << "/tmp/Y_" << globalIterationCount << ".dat";
//  
//  EpetraExt::MultiVectorToMatlabFile(Y_file.str().c_str(),Y);
//  globalIterationCount++;

  return 0;
}

Teuchos::RCP<Epetra_MultiVector> GMGOperator::filterMultiVectorByFineCellOwnership(const Epetra_MultiVector &X, GlobalIndexType fineCellID) const {
  GDAMinimumRule* gda = dynamic_cast< GDAMinimumRule*>(_fineMesh->globalDofAssignment().get());

  set<GlobalIndexType> allFineGlobalIndicesForCell = gda->globalDofIndicesForCell(fineCellID);
  set<GlobalIndexType> ownedFineGlobalIndices = gda->ownedGlobalDofIndicesForCell(fineCellID); // filter by cell ownership to ensure that we only treat each fine global dof once.

  FieldContainer<GlobalIndexTypeToCast> globalIndicesFC(allFineGlobalIndicesForCell.size());
  int i=0;
  for (set<GlobalIndexType>::iterator gdofIt = allFineGlobalIndicesForCell.begin(); gdofIt != allFineGlobalIndicesForCell.end(); gdofIt++) {
    GlobalIndexTypeToCast globalDofIndex = *gdofIt;
    globalIndicesFC(i++) = globalDofIndex;
  }
  
  // construct map for interpretedCoefficients:
  Epetra_SerialComm SerialComm; // rank-local map
  Epetra_Map    cellFilteredMap((GlobalIndexTypeToCast)-1, (GlobalIndexTypeToCast)allFineGlobalIndicesForCell.size(), &globalIndicesFC[0], 0, SerialComm);
  Teuchos::RCP<Epetra_MultiVector> X_filtered = Teuchos::rcp( new Epetra_MultiVector(cellFilteredMap, 1) );
  
  for (int i=0; i<globalIndicesFC.size(); i++) {
    GlobalIndexTypeToCast globalIndex = globalIndicesFC[i];
    int lID_filtered = cellFilteredMap.LID(globalIndex);
    if (ownedFineGlobalIndices.find(globalIndex) != ownedFineGlobalIndices.end()) {
      int lID_X = X.Map().LID(globalIndex);
      if (lID_X < 0) {
        cout << "error: lID_X < 0 for globalIndex " << globalIndex << ".\n";
        Camellia::print("allFineGlobalIndicesForCell", allFineGlobalIndicesForCell);
        Camellia::print("ownedFineGlobalIndices", ownedFineGlobalIndices);
        
        set<GlobalIndexTypeToCast> myXEntries;
        for (int lID = X.Map().MinLID(); lID <= X.Map().MaxLID(); lID++) {
          myXEntries.insert(X.Map().GID(lID));
        }
        Camellia::print("myXEntries", myXEntries);
        
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
      }
      (*X_filtered)[0][lID_filtered] = X[0][lID_X];
    } else {
      (*X_filtered)[0][lID_filtered] = 0; // zero out the unowned dof coefficients that are seen by the cell
    }
  }
  return X_filtered;
}

double GMGOperator::NormInf() const {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");
}

const char * GMGOperator::Label() const {
  return "Camellia Geometric Multi-Grid Preconditioner";
}

int GMGOperator::SetUseTranspose(bool UseTranspose) {
  return -1; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

bool GMGOperator::UseTranspose() const {
  return false; // not supported for now.  (wouldn't be hard, but I don't see the point.)
}

//! Returns true if the \e this object can provide an approximate Inf-norm, false otherwise.
bool GMGOperator::HasNormInf() const {
  return false;
}

//! Returns a pointer to the Epetra_Comm communicator associated with this operator.
const Epetra_Comm & GMGOperator::Comm() const {
  return _finePartitionMap.Comm();
}

//! Returns the Epetra_Map object associated with the domain of this operator.
const Epetra_Map & GMGOperator::OperatorDomainMap() const {
  return _finePartitionMap;
}

//! Returns the Epetra_Map object associated with the range of this operator.
const Epetra_Map & GMGOperator::OperatorRangeMap() const {
  return _finePartitionMap;
}

void GMGOperator::setFineMesh(MeshPtr fineMesh, Epetra_Map finePartitionMap) {
  _fineMesh = fineMesh;
  _finePartitionMap = finePartitionMap;
}

void GMGOperator::setStiffnessDiagonal(Teuchos::RCP< Epetra_MultiVector> diagonal) {
  if (diagonal.get() != NULL) {
    EpetraExt::MultiVectorToMatlabFile("/tmp/diag.dat",*diagonal);
  }
  _diag = diagonal;
}