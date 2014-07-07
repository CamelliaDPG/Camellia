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

GMGOperator::GMGOperator(MeshPtr coarseMesh, IPPtr coarseIP, MeshPtr fineMesh, Epetra_Map finePartitionMap, Teuchos::RCP<Solver> coarseSolver) :  _finePartitionMap(finePartitionMap), _br(true) {
  RHSPtr zeroRHS = RHS::rhs();
  BCPtr noBCs = BC::bc();
  _fineMesh = fineMesh;
  _coarseMesh = coarseMesh;
  _coarseSolution = Teuchos::rcp( new Solution(coarseMesh, noBCs, zeroRHS, coarseIP) );
  _coarseSolver = coarseSolver;
  
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
  
  int fineSideCount = fineCell->topology()->getSideCount();
  int sideDim = _fineMesh->getTopology()->getSpaceDim() - 1;
  vector<unsigned> ancestralSideOrdinals(fineSideCount);
  vector< RefinementBranch > sideRefBranches(fineSideCount);
  for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++) {
    ancestralSideOrdinals[sideOrdinal] = RefinementPattern::ancestralSubcellOrdinal(refBranch, sideDim, sideOrdinal);
    sideRefBranches[sideOrdinal] = RefinementPattern::sideRefinementBranch(refBranch, sideOrdinal);
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
        BasisPtr coarseBasis = coarseTrialOrdering->getBasis(trialID);
        BasisPtr fineBasis = fineTrialOrdering->getBasis(trialID);
        weights = _br.constrainedWeights(fineBasis, refBranch, coarseBasis, vertexNodePermutation);
        set<unsigned> fineDofOrdinals(weights.fineOrdinals.begin(),weights.fineOrdinals.end());
        vector<GlobalIndexType> coarseDofOrdinals(weights.coarseOrdinals.begin(),weights.coarseOrdinals.end());
        BasisMap basisMap(1,SubBasisDofMapper::subBasisDofMapper(fineDofOrdinals, coarseDofOrdinals, weights.weights));
        volumeMaps[trialID] = basisMap;
      } else { // flux/trace
        for (int sideOrdinal=0; sideOrdinal<fineSideCount; sideOrdinal++) {
          unsigned coarseSideOrdinal = ancestralSideOrdinals[sideOrdinal];
          if (coarseSideOrdinal == -1) continue;
          
          BasisPtr coarseBasis = coarseTrialOrdering->getBasis(trialID, coarseSideOrdinal);
          BasisPtr fineBasis = fineTrialOrdering->getBasis(trialID, sideOrdinal);
          weights = _br.constrainedWeights(fineBasis, sideOrdinal, sideRefBranches[sideOrdinal], coarseBasis, coarseSideOrdinal, vertexNodePermutation);
          set<unsigned> fineDofOrdinals(weights.fineOrdinals.begin(),weights.fineOrdinals.end());
          vector<GlobalIndexType> coarseDofOrdinals(weights.coarseOrdinals.begin(),weights.coarseOrdinals.end());
          BasisMap basisMap(1,SubBasisDofMapper::subBasisDofMapper(fineDofOrdinals, coarseDofOrdinals, weights.weights));
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

int GMGOperator::ApplyInverse(const Epetra_MultiVector& X, Epetra_MultiVector& Y) const {
  // the data coming in (X) is in global dofs defined on the fine mesh.  First thing we'd like to do is map it to the fine mesh's local cells
  vector<GlobalIndexType> cellsInPartition = _fineMesh->globalDofAssignment()->cellsInPartition(-1); // rank-local
  
  Teuchos::RCP<Epetra_FEVector> coarseRHSVector = _coarseSolution->getRHSVector();
  coarseRHSVector->PutScalar(0); // clear
  set<GlobalIndexTypeToCast> coarseDofIndicesToImport; // keep track of the coarse dof indices that this partition's fine cells talk to
  for (vector<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++) {
    GlobalIndexType fineCellID = *cellIDIt;
    int fineDofCount = _fineMesh->getElementType(fineCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> fineCellData(fineDofCount);
    _fineMesh->globalDofAssignment()->interpretGlobalCoefficients(fineCellID, fineCellData, X);
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
    GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
    int coarseDofCount = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> coarseCellData(coarseDofCount);
    FieldContainer<double> mappedCoarseCellData = fineMapper->mapLocalData(fineCellData, false);
    vector<GlobalIndexType> mappedCoarseDofIndices = fineMapper->globalIndices();
    for (int mappedCoarseDofOrdinal = 0; mappedCoarseDofOrdinal < mappedCoarseDofIndices.size(); mappedCoarseDofOrdinal++) {
      GlobalIndexType coarseDofIndex = mappedCoarseDofIndices[mappedCoarseDofOrdinal];
      coarseCellData[coarseDofIndex] = mappedCoarseCellData[coarseDofIndex];
    }
    FieldContainer<double> interpretedCoarseData;
    FieldContainer<GlobalIndexType> interpretedGlobalDofIndices;
    _coarseMesh->interpretLocalData(coarseCellID, coarseCellData, interpretedCoarseData, interpretedGlobalDofIndices);
    FieldContainer<GlobalIndexTypeToCast> interpretedGlobalDofIndicesCast(interpretedGlobalDofIndices.size());
    for (int interpretedDofOrdinal=0; interpretedDofOrdinal < interpretedGlobalDofIndices.size(); interpretedDofOrdinal++) {
      interpretedGlobalDofIndicesCast[interpretedDofOrdinal] = (GlobalIndexTypeToCast) interpretedGlobalDofIndices[interpretedDofOrdinal];
      coarseDofIndicesToImport.insert(interpretedGlobalDofIndicesCast[interpretedDofOrdinal]);
    }
    coarseRHSVector->SumIntoGlobalValues(interpretedCoarseData.size(), &interpretedGlobalDofIndicesCast[0], &interpretedCoarseData[0]);
  }
  // solve the coarse system:
  _coarseSolution->setProblem(_coarseSolver);
  _coarseSolution->solve(_coarseSolver);
  
  Teuchos::RCP<Epetra_FEVector> coarseLHSVector = _coarseSolution->getLHSVector();
  
  // now, map the coarse data back to the fine mesh, and add that into Y
  Y.PutScalar(0); // clear Y
  
  // import all the dofs of interest to us onto this MPI rank
  Epetra_Map coarseMap = _coarseSolution->getPartitionMap();
  GlobalIndexTypeToCast numDofsGlobal = coarseMap.NumGlobalElements();
  GlobalIndexTypeToCast numMyDofs = coarseDofIndicesToImport.size();
  GlobalIndexTypeToCast myDofs[coarseDofIndicesToImport.size()];
  GlobalIndexTypeToCast* myDof = &myDofs[0];
  for (set<GlobalIndexTypeToCast>::iterator coarseDofIndexIt = coarseDofIndicesToImport.begin();
       coarseDofIndexIt != coarseDofIndicesToImport.end(); coarseDofIndexIt++) {
    *myDof = *coarseDofIndexIt;
    myDof++;
  }
  
  Epetra_Map     solnMap(numDofsGlobal, numMyDofs, myDofs, 0, Comm());
  Epetra_Import  solnImporter(solnMap, coarseMap);
  Epetra_Vector  coarseDofs(solnMap);
  coarseDofs.Import(*coarseLHSVector, solnImporter, Insert);
  
  for (vector<GlobalIndexType>::iterator cellIDIt=cellsInPartition.begin(); cellIDIt != cellsInPartition.end(); cellIDIt++) {
    GlobalIndexType fineCellID = *cellIDIt;
    int fineDofCount = _fineMesh->getElementType(fineCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> fineCellData(fineDofCount);
    
    LocalDofMapperPtr fineMapper = getLocalCoefficientMap(fineCellID);
    GlobalIndexType coarseCellID = getCoarseCellID(fineCellID);
    
    int coarseDofCount = _coarseMesh->getElementType(coarseCellID)->trialOrderPtr->totalDofs();
    FieldContainer<double> coarseCellData(coarseDofCount);
    
    _coarseMesh->globalDofAssignment()->interpretGlobalCoefficients(coarseCellID, coarseCellData, coarseDofs);
    
    vector<GlobalIndexType> coarseCellMappedLocalIndices = fineMapper->globalIndices();
    
    FieldContainer<double> coarseCellMappedData(coarseCellMappedLocalIndices.size());
    for (int i=0; i<coarseCellMappedLocalIndices.size(); i++) {
      int coarseCellLocalIndex = coarseCellMappedLocalIndices[i];
      coarseCellMappedData[i] = coarseCellData[coarseCellLocalIndex];
    }
    
    FieldContainer<double> fineLocalCoefficients = fineMapper->mapGlobalCoefficients(coarseCellMappedData);
    
    _fineMesh->globalDofAssignment()->interpretLocalCoefficients(fineCellID, fineLocalCoefficients, Y);
  }
  
  // if diag is set, add diag(A)^(-1)X to Y.
  if (_diag.get() != NULL) {
    Epetra_BlockMap partitionMap = Y.Map();
    for (int localID = 0; localID < partitionMap.NumMyElements(); localID++) {
      GlobalIndexTypeToCast globalID = partitionMap.GID(localID);
      double diagEntry = (*_diag)[0][globalID];
      double xEntry = X[0][globalID];
      Y.SumIntoGlobalValue(globalID, 0, xEntry/diagEntry);
    }
  }
  
  return 0;
}

double GMGOperator::NormInf() const {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported method.");
}

const char * GMGOperator::Label() const {
  return "Camellia Geometric Multi-Grid operator";
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

void GMGOperator::setStiffnessDiagonal(Teuchos::RCP< Epetra_MultiVector> diagonal) {
  _diag = diagonal;
}