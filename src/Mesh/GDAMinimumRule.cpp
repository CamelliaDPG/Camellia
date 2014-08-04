//
//  GDAMinimumRule.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 1/21/14.
//
//

#include "GDAMinimumRule.h"

#include "MPIWrapper.h"

#include "CamelliaCellTools.h"

#include "SerialDenseWrapper.h"

#include "Teuchos_GlobalMPISession.hpp"

#include "CamelliaDebugUtility.h"

#include "Solution.h"

GDAMinimumRule::GDAMinimumRule(MeshPtr mesh, VarFactory varFactory, DofOrderingFactoryPtr dofOrderingFactory, MeshPartitionPolicyPtr partitionPolicy,
                               unsigned initialH1OrderTrial, unsigned testOrderEnhancement)
: GlobalDofAssignment(mesh,varFactory,dofOrderingFactory,partitionPolicy, initialH1OrderTrial, testOrderEnhancement, false)
{

}

vector<unsigned> GDAMinimumRule::allBasisDofOrdinalsVector(int basisCardinality) {
  vector<unsigned> ordinals(basisCardinality);
  for (int i=0; i<basisCardinality; i++) {
    ordinals[i] = i;
  }
  return ordinals;
}

void GDAMinimumRule::didChangePartitionPolicy() {
  rebuildLookups();
}

void GDAMinimumRule::didHRefine(const set<GlobalIndexType> &parentCellIDs) {
  set<GlobalIndexType> neighborsOfNewElements;
  for (set<GlobalIndexType>::const_iterator cellIDIt = parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++) {
    GlobalIndexType parentCellID = *cellIDIt;
//    cout << "GDAMinimumRule: h-refining " << parentCellID << endl;
    CellPtr parentCell = _meshTopology->getCell(parentCellID);
    vector<IndexType> childIDs = parentCell->getChildIndices();
    int parentH1Order = _cellH1Orders[parentCellID];
    for (vector<IndexType>::iterator childIDIt = childIDs.begin(); childIDIt != childIDs.end(); childIDIt++) {
      GlobalIndexType childCellID = *childIDIt;
      _cellH1Orders[childCellID] = parentH1Order;
      assignInitialElementType(childCellID);
      assignParities(childCellID);
      // determine neighbors, so their parities can be updated below:
      CellPtr childCell = _meshTopology->getCell(childCellID);
      
      unsigned childSideCount = CamelliaCellTools::getSideCount(*childCell->topology());
      for (int childSideOrdinal=0; childSideOrdinal<childSideCount; childSideOrdinal++) {
        GlobalIndexType neighborCellID = childCell->getNeighbor(childSideOrdinal).first;
        if (neighborCellID != -1) {
          neighborsOfNewElements.insert(neighborCellID);
        }
      }
    }
  }
  // this set is not as lean as it might be -- we could restrict to peer neighbors, I think -- but it's a pretty cheap operation.
  for (set<GlobalIndexType>::iterator cellIDIt = neighborsOfNewElements.begin(); cellIDIt != neighborsOfNewElements.end(); cellIDIt++) {
    assignParities(*cellIDIt);
  }
  for (set<GlobalIndexType>::const_iterator cellIDIt = parentCellIDs.begin(); cellIDIt != parentCellIDs.end(); cellIDIt++) {
    GlobalIndexType parentCellID = *cellIDIt;
    ElementTypePtr elemType = elementType(parentCellID);
    for (vector< Solution* >::iterator solutionIt = _registeredSolutions.begin();
         solutionIt != _registeredSolutions.end(); solutionIt++) {
      // do projection
      vector<IndexType> childIDsLocalIndexType = _meshTopology->getCell(parentCellID)->getChildIndices();
      vector<GlobalIndexType> childIDs(childIDsLocalIndexType.begin(),childIDsLocalIndexType.end());
      (*solutionIt)->projectOldCellOntoNewCells(parentCellID,elemType,childIDs);
    }
  }
  
  this->GlobalDofAssignment::didHRefine(parentCellIDs);
}

void GDAMinimumRule::didPRefine(const set<GlobalIndexType> &cellIDs, int deltaP) {
  this->GlobalDofAssignment::didPRefine(cellIDs, deltaP);
  for (set<GlobalIndexType>::const_iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
    ElementTypePtr oldType = elementType(*cellIDIt);
    assignInitialElementType(*cellIDIt);
    for (vector< Solution* >::iterator solutionIt = _registeredSolutions.begin();
         solutionIt != _registeredSolutions.end(); solutionIt++) {
      // do projection
      vector<IndexType> childIDs(1,*cellIDIt); // "child" ID is just the cellID
      (*solutionIt)->projectOldCellOntoNewCells(*cellIDIt,oldType,childIDs);
    }
  }
//  rebuildLookups();
}

void GDAMinimumRule::didHUnrefine(const set<GlobalIndexType> &parentCellIDs) {
  this->GlobalDofAssignment::didHUnrefine(parentCellIDs);
  // TODO: implement this
  cout << "WARNING: GDAMinimumRule::didHUnrefine() unimplemented.\n";
  // will need to treat cell side parities here--probably suffices to redo those in parentCellIDs plus all their neighbors.
//  rebuildLookups();
}

ElementTypePtr GDAMinimumRule::elementType(GlobalIndexType cellID) {
  return _elementTypeForCell[cellID];
}

void GDAMinimumRule::filterSubBasisConstraintData(set<unsigned> &basisDofOrdinals, vector<GlobalIndexType> &globalDofOrdinals,
                                                  FieldContainer<double> &constraintMatrix, FieldContainer<bool> &processedDofs,
                                                  DofOrderingPtr trialOrdering, VarPtr var, int sideOrdinal) {
  double tol = 1e-12;
  vector<unsigned> localOrdinalsVector(basisDofOrdinals.begin(), basisDofOrdinals.end());
  set<unsigned> zeroRows, zeroCols;
  int numRows = constraintMatrix.dimension(0);
  int numCols = constraintMatrix.dimension(1);
  for (int i=0; i < numRows; i++) {
    bool nonZeroFound = false;
    for (int j=0; j < numCols; j++) {
      if (abs(constraintMatrix(i,j)) > tol) {
        nonZeroFound = true;
      }
    }
    if (!nonZeroFound) {
      zeroRows.insert(i);
    }
  }
  for (int j=0; j < numCols; j++) {
    bool nonZeroFound = false;
    for (int i=0; i < numRows; i++) {
      if (abs(constraintMatrix(i,j)) > tol) {
        nonZeroFound = true;
      }
    }
    if (!nonZeroFound) {
      zeroCols.insert(j);
    }
  }
  set<unsigned> rowsToSkip = zeroRows;
  set<unsigned> colsToSkip = zeroCols;
  for (int i=0; i<numRows; i++) {
    unsigned localDofOrdinal = localOrdinalsVector[i];
    unsigned localDofIndex = trialOrdering->getDofIndex(var->ID(), localDofOrdinal, sideOrdinal);
    if (processedDofs[localDofIndex]) {
      rowsToSkip.insert(i);
    }
    if (rowsToSkip.find(i) == rowsToSkip.end()) { // we will be processing this one, so mark as such
      processedDofs[localDofIndex] = true;
    }
  }
  
  FieldContainer<double> matrixCopy = constraintMatrix;
  constraintMatrix.resize(numRows - rowsToSkip.size(), numCols - colsToSkip.size());
  basisDofOrdinals.clear();
  
  int rowOffset = 0; // decrement for each skipped row
  for (int i=0; i<numRows; i++) {
    if (rowsToSkip.find(i) != rowsToSkip.end()) {
      rowOffset -= 1;
      continue;
    }
    basisDofOrdinals.insert(localOrdinalsVector[i+rowOffset]);
    int colOffset = 0;
    for (int j=0; j<numCols; j++) {
      if (colsToSkip.find(j) != colsToSkip.end()) {
        colOffset -= 1;
        continue;
      }
      constraintMatrix(i + rowOffset, j + colOffset) = matrixCopy(i,j);
    }
  }
  
  vector<GlobalIndexType> globalDofOrdinalsCopy = globalDofOrdinals;
  globalDofOrdinals.clear();
  int colOffset = 0;
  for (int j=0; j<numCols; j++) {
    if (colsToSkip.find(j) != colsToSkip.end()) {
      colOffset -= 1;
      continue;
    }
    globalDofOrdinals.push_back(globalDofOrdinalsCopy[j+colOffset]);
  }
  
  // mark dofs corresponding to the remaining rows as processed
  for (int i=0; i<numRows; i++) {
    if (rowsToSkip.find(i) == rowsToSkip.end()) { // we will be processing this one, so mark as such
      unsigned localDofOrdinal = localOrdinalsVector[i];
      unsigned localDofIndex = trialOrdering->getDofIndex(var->ID(), localDofOrdinal, sideOrdinal);
      processedDofs[localDofIndex] = true;
    }
  }
}

int GDAMinimumRule::getH1Order(GlobalIndexType cellID) {
  return _cellH1Orders[cellID];
}

GlobalIndexType GDAMinimumRule::globalDofCount() {
  // assumes the lookups have been rebuilt since the last change that would affect the count
  
  // TODO: Consider working out a way to guard against a stale value here.  E.g. could have a "dirty" flag that gets set anytime there's a change to the refinements, and cleared when lookups are rebuilt.  If we're dirty when we get here, we rebuild before returning the global dof count.
  return _globalDofCount;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForCell(GlobalIndexType cellID) {
  set<GlobalIndexType> globalDofIndices;
  
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();
  
  globalDofIndices.insert(globalIndexVector.begin(),globalIndexVector.end());
  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::globalDofIndicesForPartition(PartitionIndexType partitionNumber) {
  // Added this exception because I suspect that in our usage, partitionNumber is always -1 and therefore we can/should
  // dispense with the argument...
  int rank = Teuchos::GlobalMPISession::getRank();
  
  if (partitionNumber==-1) partitionNumber = rank;
  
  TEUCHOS_TEST_FOR_EXCEPTION(partitionNumber != rank, std::invalid_argument, "partitionNumber must be -1 or the current MPI rank");
  set<GlobalIndexType> globalDofIndices;
  // by construction, our globalDofIndices are contiguously numbered, starting with _partitionDofOffset
  for (GlobalIndexType i=0; i<_partitionDofCount; i++) {
    globalDofIndices.insert(_partitionDofOffset + i);
  }
  
  return globalDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedGlobalFieldIndices() {
  // compute complement of fluxes and traces
  set<GlobalIndexType> fieldDofIndices;
  // by construction, our globalDofIndices are contiguously numbered, starting with _partitionDofOffset
  for (GlobalIndexType i=0; i<_partitionDofCount; i++) {
    if ((_partitionTraceIndexOffsets.find(i) == _partitionTraceIndexOffsets.end()) &&
        (_partitionFluxIndexOffsets.find(i) == _partitionFluxIndexOffsets.end()) ) {
          fieldDofIndices.insert(_partitionDofOffset + i);
    }
  }
  return fieldDofIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedIndicesForVariables(set<int> varIDs) {
  set<GlobalIndexType> varIndices;
  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
    for (set<IndexType>::iterator varOffsetIt=_partitionIndexOffsetsForVarID[*varIt].begin();
         varOffsetIt != _partitionIndexOffsetsForVarID[*varIt].end(); varOffsetIt++) {
      varIndices.insert(*varOffsetIt + _partitionDofOffset);
    }
  }
  return varIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedGlobalFluxIndices() {
  set<GlobalIndexType> fluxIndices;
  for (set<IndexType>::iterator fluxOffsetIt=_partitionFluxIndexOffsets.begin(); fluxOffsetIt != _partitionFluxIndexOffsets.end(); fluxOffsetIt++) {
    fluxIndices.insert(*fluxOffsetIt + _partitionDofOffset);
  }
  return fluxIndices;
}

set<GlobalIndexType> GDAMinimumRule::partitionOwnedGlobalTraceIndices() {
  set<GlobalIndexType> traceIndices;
  for (set<IndexType>::iterator traceOffsetIt=_partitionTraceIndexOffsets.begin(); traceOffsetIt != _partitionTraceIndexOffsets.end(); traceOffsetIt++) {
    traceIndices.insert(*traceOffsetIt + _partitionDofOffset);
  }
  return traceIndices;
}

int GDAMinimumRule::H1Order(GlobalIndexType cellID, unsigned sideOrdinal) {
  // this is meant to track the cell's interior idea of what the H^1 order is along that side.  We're isotropic for now, but eventually we might want to allow anisotropy in p...
  return _cellH1Orders[cellID];
}

void GDAMinimumRule::interpretGlobalCoefficients(GlobalIndexType cellID, FieldContainer<double> &localCoefficients, const Epetra_MultiVector &globalCoefficients) {
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();
  
  // DEBUGGING
//  if (cellID==4) {
//    cout << "interpretGlobalData, mapping report for cell " << cellID << ":\n";
//    dofMapper->printMappingReport();
//  }

  FieldContainer<double> globalCoefficientsFC(globalIndexVector.size());
  Epetra_BlockMap partMap = globalCoefficients.Map();
  for (int i=0; i<globalIndexVector.size(); i++) {
    GlobalIndexTypeToCast globalIndex = globalIndexVector[i];
    int localIndex = partMap.LID(globalIndex);
    globalCoefficientsFC[i] = globalCoefficients[0][localIndex];
  }
  localCoefficients = dofMapper->mapGlobalCoefficients(globalCoefficientsFC);
//  cout << "For cellID " << cellID << ", mapping globalData:\n " << globalDataFC;
//  cout << " to localData:\n " << localDofs;
}

void GDAMinimumRule::interpretLocalData(GlobalIndexType cellID, const FieldContainer<double> &localData,
                                        FieldContainer<double> &globalData, FieldContainer<GlobalIndexType> &globalDofIndices) {
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints);
  
  // DEBUGGING
//  if (Teuchos::GlobalMPISession::getRank()==0) {
//    if (cellID==2) {
//      cout << "interpretLocalData, mapping report for cell " << cellID << ":\n";
//      dofMapper->printMappingReport();
//    }
//  }
  
  globalData = dofMapper->mapLocalData(localData, false);
  vector<GlobalIndexType> globalIndexVector = dofMapper->globalIndices();
  globalDofIndices.resize(globalIndexVector.size());
  for (int i=0; i<globalIndexVector.size(); i++) {
    globalDofIndices(i) = globalIndexVector[i];
  }
  
//  // mostly for debugging purposes, let's sort according to global dof index:
//  if (globalData.rank() == 1) {
//    map<GlobalIndexType, double> globalDataMap;
//    for (int i=0; i<globalIndexVector.size(); i++) {
//      GlobalIndexType globalIndex = globalIndexVector[i];
//      globalDataMap[globalIndex] = globalData(i);
//    }
//    int i=0;
//    for (map<GlobalIndexType, double>::iterator mapIt = globalDataMap.begin(); mapIt != globalDataMap.end(); mapIt++) {
//      globalDofIndices(i) = mapIt->first;
//      globalData(i) = mapIt->second;
//      i++;
//    }
//  } else if (globalData.rank() == 2) {
//    cout << "globalData.rank() == 2.\n";
//    FieldContainer<double> globalDataCopy = globalData;
//    map<GlobalIndexType,int> globalIndexToOrdinalMap;
//    for (int i=0; i<globalIndexVector.size(); i++) {
//      GlobalIndexType globalIndex = globalIndexVector[i];
//      globalIndexToOrdinalMap[globalIndex] = i;
//    }
//    int i=0;
//    for (map<GlobalIndexType,int>::iterator i_mapIt = globalIndexToOrdinalMap.begin();
//         i_mapIt != globalIndexToOrdinalMap.end(); i_mapIt++) {
//      int j=0;
//      globalDofIndices(i) = i_mapIt->first;
//      for (map<GlobalIndexType,int>::iterator j_mapIt = globalIndexToOrdinalMap.begin();
//           j_mapIt != globalIndexToOrdinalMap.end(); j_mapIt++) {
//        globalData(i,j) = globalDataCopy(i_mapIt->second,j_mapIt->second);
//        
//        j++;
//      }
//      i++;
//    }
//  }
  
//  cout << "localData:\n" << localData;
//  cout << "globalData:\n" << globalData;
//  cout << "globalIndices:\n" << globalDofIndices;
}

void GDAMinimumRule::interpretLocalBasisCoefficients(GlobalIndexType cellID, int varID, int sideOrdinal, const FieldContainer<double> &basisCoefficients,
                                                     FieldContainer<double> &globalCoefficients, FieldContainer<GlobalIndexType> &globalDofIndices) {
  CellConstraints constraints = getCellConstraints(cellID);
  LocalDofMapperPtr dofMapper = getDofMapper(cellID, constraints, varID, sideOrdinal);
  
//  cout << "interpretLocalBasisCoefficients, mapping report for cell " << cellID << ":\n";
//  dofMapper->printMappingReport();
  
  // DEBUGGING:
//  if ((cellID==2) && (varID==0) && (sideOrdinal==3)) {
//    cout << "(cellID==2) && (varID==0) && (sideOrdinal==3).\n";
//  }
  
  // the new, right way to do this:
  globalCoefficients = dofMapper->fitLocalCoefficients(basisCoefficients);
  vector<GlobalIndexType> globalIndexVector = dofMapper->fittableGlobalIndices();
  globalDofIndices.resize(globalIndexVector.size());
  // DEBUGGING
//  cout << "cellID " << cellID << ", side " << sideOrdinal << ", var " << varID;
//  Camellia::print(", globalIndexVector", globalIndexVector);
//  cout << "globalCoefficients:\n" << globalCoefficients;
  for (int i=0; i<globalIndexVector.size(); i++) {
    globalDofIndices(i) = globalIndexVector[i];
  }
}

IndexType GDAMinimumRule::localDofCount() {
  // TODO: implement this
  cout << "WARNING: localDofCount() unimplemented.\n";
  return 0;
}

typedef vector< SubBasisDofMapperPtr > BasisMap;
// volume variable version
BasisMap GDAMinimumRule::getBasisMap(GlobalIndexType cellID, SubCellDofIndexInfo& dofIndexInfo, VarPtr var) {
  BasisMap varVolumeMap;
  
  CellPtr cell = _meshTopology->getCell(cellID);
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  CellTopoPtr topo = cell->topology();
  unsigned spaceDim = topo->getDimension();
  unsigned sideDim = spaceDim - 1;
  
  // assumption is that the basis is defined on the whole cell
  BasisPtr basis = trialOrdering->getBasis(var->ID());
  
  // to begin, let's map the volume-interior dofs:
  vector<GlobalIndexType> globalDofOrdinals = dofIndexInfo[spaceDim][0][var->ID()];
  set<int> ordinalsInt = BasisReconciliation::interiorDofOrdinalsForBasis(basis); // basis->dofOrdinalsForInterior(); // TODO: change dofOrdinalsForInterior to return set<unsigned>...
  set<unsigned> basisDofOrdinals;
  basisDofOrdinals.insert(ordinalsInt.begin(),ordinalsInt.end());
  
  if (basisDofOrdinals.size() > 0) {
    varVolumeMap.push_back(SubBasisDofMapper::subBasisDofMapper(basisDofOrdinals, globalDofOrdinals));
  }
  
  if (globalDofOrdinals.size() != basisDofOrdinals.size()) {
    cout << "";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "volume getBasisMap() doesn't yet support non-L^2 bases.");
  }
  // TODO: reimplement the below logic in imitation of the side-oriented getBasisMap() below...
  
//  vector< vector< SubBasisReconciliationWeights> > weightsForSubcell(spaceDim);
//  vector< vector< unsigned > > minProcessedDimensionForSubcell(spaceDim); // we process dimensions from high to low -- since we deal here with the volume basis case, we initialize this to spaceDim + 1
//  vector< vector< ConstrainingSubcellInfo > > appliedSubcellConstraintInfo(spaceDim); // information about what was *last* applied to this subcell
//  
//  ConstrainingSubcellInfo emptyConstraintInfo;
//  emptyConstraintInfo.cellID = -1;
//  
//  for (int d=0; d<spaceDim; d++) {
//    int scCount = topo->getSubcellCount(d);
//    weightsForSubcell[d] = vector< SubBasisReconciliationWeights>(scCount);
//    minProcessedDimensionForSubcell[d] = vector<unsigned>(scCount, spaceDim + 1);
//    appliedSubcellConstraintInfo[d] = vector< ConstrainingSubcellInfo >(scCount, emptyConstraintInfo);
//  }
//  
//  for (int d=sideDim; d >= 0; d--) {
//    int scCount = topo->getSubcellCount(d);
//    for (int subcord=0; subcord < scCount; subcord++) { // subcell ordinals in cell
//      if (minProcessedDimensionForSubcell[d][subcord] == d) {
//        // we've already done all we need to do for this subcell, then
//        continue;
//      }
//      
//      ConstrainingSubcellInfo subcellConstraint = constraints.subcellConstraints[d][subcord];
//      CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
//      
//      DofOrderingPtr constrainingTrialOrdering = _elementTypeForCell[subcellConstraint.cellID]->trialOrderPtr;
//      BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID());
//      
//      GlobalIndexType appliedConstraintCellID;
//      appliedConstraintCellID = appliedSubcellConstraintInfo[d][subcord].cellID;
//      unsigned appliedConstraintSubcellOrdinal;
//      if ( appliedConstraintCellID == -1) { // no constraints yet applied
//        appliedConstraintCellID = cellID;
//        appliedConstraintSubcellOrdinal = subcord;
//      } else {
//        unsigned subcellOrdinalInSide = appliedSubcellConstraintInfo[d][subcord].subcellOrdinalInSide;
//        CellPtr appliedConstraintCell = _meshTopology->getCell(appliedConstraintCellID);
//        unsigned appliedConstraintSideOrdinal = appliedSubcellConstraintInfo[d][subcord].sideOrdinal;
//        shards::CellTopology appliedConstraintSideTopo = appliedConstraintCell->topology()->getCellTopologyData(sideDim, appliedConstraintSideOrdinal);
//        appliedConstraintSubcellOrdinal = CamelliaCellTools::subcellOrdinalMap(*appliedConstraintCell->topology(), sideDim, appliedConstraintSideOrdinal, d, subcellOrdinalInSide);
//      }
//      
//      DofOrderingPtr appliedConstraintTrialOrdering = _elementTypeForCell[appliedConstraintCellID]->trialOrderPtr;
//      BasisPtr appliedConstraintBasis = appliedConstraintTrialOrdering->getBasis(var->ID());
//      
//      CellPtr appliedConstraintCell = _meshTopology->getCell(appliedConstraintCellID);
//      
//      RefinementBranch refinements = appliedConstraintCell->refinementBranchForSubcell(d, appliedConstraintSubcellOrdinal);;
//      unsigned ancestralPermutation = appliedConstraintCell->ancestralPermutationForSubcell(d, appliedConstraintSubcellOrdinal);// subcell permutation as seen from the perspective of the fine cell's side's ancestor
//      
//      unsigned subcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(*topo, sideDim, subcellConstraint.sideOrdinal,
//                                                                                       d, subcellConstraint.subcellOrdinalInSide);
//      
//      unsigned constrainingPermutation = constrainingCell->subcellPermutation(d, subcellOrdinalInConstrainingCell);
//      
//      shards::CellTopology constrainingTopo = constrainingCell->topology()->getCellTopologyData(d, subcellOrdinalInConstrainingCell);
//      unsigned constrainingPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, constrainingPermutation);
//      
//      unsigned composedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, constrainingPermutationInverse, ancestralPermutation);
//      
//      SubBasisReconciliationWeights newWeightsToApply = _br.constrainedWeights(d, appliedConstraintBasis, appliedConstraintSubcellOrdinal, refinements,
//                                                                               constrainingBasis, subcellOrdinalInConstrainingCell, composedPermutation);
//      
//      // compose the new weights with any existing weights for this subcell
//      SubBasisReconciliationWeights composedWeights;
//      if (appliedSubcellConstraintInfo[d][subcord].cellID == -1) {
//        composedWeights = newWeightsToApply;
//      } else {
//        SubBasisReconciliationWeights existingWeights = weightsForSubcell[d][subcord];
//        composedWeights = BasisReconciliation::composedSubBasisReconciliationWeights(existingWeights, newWeightsToApply);
//      }
//      
//      shards::CellTopology subcellTopo = topo->getCellTopologyData(d, subcord);
//      
//      // fine sub-subcells that are interior to the coarse subcell will be entirely reconciled by virtue of the SubBasisDofMapper we create for the subcell, below
//      RefinementBranch subcellRefinements = RefinementPattern::subcellRefinementBranch(refinements, d, appliedConstraintSubcellOrdinal);
//      if (subcellRefinements.size() > 0) {
//        map<unsigned, set<unsigned> > internalSubSubcellOrdinals = RefinementPattern::getInternalSubcellOrdinals(subcellRefinements); // might like to cache this result (there is a repeated, inefficient call inside this method)
//        for (int ssd=0; ssd<internalSubSubcellOrdinals.size(); ssd++) {
//          for (set<unsigned>::iterator ssordIt=internalSubSubcellOrdinals[ssd].begin(); ssordIt != internalSubSubcellOrdinals[ssd].end(); ssordIt++) {
//            unsigned ssubcordInCell = CamelliaCellTools::subcellOrdinalMap(*topo, d, subcord, ssd, *ssordIt);
//            minProcessedDimensionForSubcell[ssd][ssubcordInCell] = ssd;
//          }
//        }
//      }
//      
//      // populate the containers for the (d-1)-dimensional constituents of this subcell
//      if (d-1 >=0) {
//        int d1 = d-1;
//        unsigned sscCount = subcellTopo.getSubcellCount(d1);
//        for (unsigned ssubcord=0; ssubcord<sscCount; ssubcord++) {
//          unsigned ssubcordInCell = CamelliaCellTools::subcellOrdinalMap(*topo, d, subcord, d1, ssubcord);
//          if (minProcessedDimensionForSubcell[d1][ssubcordInCell] > d) {
//            ConstrainingSubcellInfo ssAppliedConstraint;
//            ssAppliedConstraint.cellID = subcellConstraint.cellID;
//            ssAppliedConstraint.sideOrdinal = subcellConstraint.sideOrdinal;
//            unsigned ssubcellOrdinalInConstrainingSubcell = RefinementPattern::ancestralSubcellOrdinal(subcellRefinements, d1, ssubcord);
//
//            IndexType ssEntityIndex = cell->ancestralEntityIndexForSubcell(d1, ssubcordInCell);
//
//            CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
//            unsigned ssubcordInConstrainingCell = constrainingCell->findSubcellOrdinal(d1, ssEntityIndex);
//
//            IndexType constrainingSideEntityIndex = constrainingCell->entityIndex(sideDim, subcellConstraint.sideOrdinal);
//            
//            shards::CellTopology constrainingSideTopo = _meshTopology->getEntityTopology(sideDim, constrainingSideEntityIndex);
//            
//            ssAppliedConstraint.subcellOrdinalInSide = CamelliaCellTools::subcellReverseOrdinalMap(*constrainingCell->topology(), sideDim, subcellConstraint.sideOrdinal,
//                                                                                                   d1, ssubcordInConstrainingCell);
//            appliedSubcellConstraintInfo[d1][ssubcordInCell] = ssAppliedConstraint;
//            weightsForSubcell[d1][ssubcordInCell] = BasisReconciliation::weightsForCoarseSubcell(composedWeights, constrainingBasis, d1, ssubcellOrdinalInConstrainingSubcell, true);
//            minProcessedDimensionForSubcell[d1][ssubcordInCell] = d;
//          }
//        }
//      }
//      
//      // filter the weights whose coarse dofs are interior to this subcell, and create a SubBasisDofMapper for these (add it to varSideMap)
//      SubBasisReconciliationWeights subcellInteriorWeights = BasisReconciliation::weightsForCoarseSubcell(composedWeights, constrainingBasis, d, subcellOrdinalInConstrainingCell, false);
//      
//      if (subcellInteriorWeights.coarseOrdinals.size() > 0) {
//        vector<GlobalIndexType> globalDofOrdinals = dofIndexInfo[d][subcord][var->ID()];
//        if (subcellInteriorWeights.coarseOrdinals.size() != globalDofOrdinals.size()) {
//          cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
//          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
//        }
//        
//        set<unsigned> basisDofOrdinals;
//        basisDofOrdinals.insert(subcellInteriorWeights.fineOrdinals.begin(), subcellInteriorWeights.fineOrdinals.end()); // TODO: change fineOrdinals to be a set<unsigned>
//        
//        varVolumeMap.push_back(SubBasisDofMapper::subBasisDofMapper(basisDofOrdinals, globalDofOrdinals, subcellInteriorWeights.weights));
//      }
//      
//    }
//  }
  return varVolumeMap;
}

// trace variable version
BasisMap GDAMinimumRule::getBasisMap(GlobalIndexType cellID, SubCellDofIndexInfo& dofIndexInfo, VarPtr var, int sideOrdinal) {
  vector<SubBasisMapInfo> subBasisMaps;
  
  SubBasisMapInfo subBasisMap;
  
//  if ((cellID==2) && (sideOrdinal==1) && (var->ID() == 0)) {
//    cout << "DEBUGGING: (cellID==2) && (sideOrdinal==1) && (var->ID() == 0).\n";
//  }
  
  // TODO: move the permutation computation outside of this method -- might include in CellConstraints, e.g. -- this obviously will not change from one var to the next, but we compute it redundantly each time...
  
  CellPtr cell = _meshTopology->getCell(cellID);
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  CellTopoPtr topo = cell->topology();
  unsigned spaceDim = topo->getDimension();
  unsigned sideDim = spaceDim - 1;
  shards::CellTopology sideTopo = topo->getCellTopologyData(sideDim, sideOrdinal);
  
  // assumption is that the basis is defined on the side
  BasisPtr basis = trialOrdering->getBasis(var->ID(), sideOrdinal);
  
  typedef pair<ConstrainingSubcellInfo, SubBasisReconciliationWeights > AppliedWeightPair;
  typedef pair< unsigned, unsigned > SubcellForDofIndexInfo; // first: subcdim, second: subcordInCell
  typedef vector< AppliedWeightPair > AppliedWeightVector;
  
  vector< map< GlobalIndexType, AppliedWeightVector > > appliedWeights(sideDim+1); // map keys are the entity indices; these are used to ensure that we don't apply constraints for a given entity multiple times.
  
  ConstrainingSubcellInfo defaultConstraint;
  defaultConstraint.cellID = cellID;
  defaultConstraint.sideOrdinal = sideOrdinal;
  defaultConstraint.subcellOrdinalInSide = 0;
  defaultConstraint.dimension = sideDim;

  SubBasisReconciliationWeights unitWeights;
  unitWeights.weights.resize(basis->getCardinality(), basis->getCardinality());
  set<int> allOrdinals;
  for (int i=0; i<basis->getCardinality(); i++) {
    allOrdinals.insert(i);
    unitWeights.weights(i,i) = 1.0;
  }
  unitWeights.fineOrdinals = allOrdinals;
  unitWeights.coarseOrdinals = allOrdinals;

  GlobalIndexType sideEntityIndex = cell->entityIndex(sideDim, sideOrdinal);
  appliedWeights[sideDim][sideEntityIndex].push_back(make_pair(defaultConstraint, unitWeights));
  
  int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);
  
//  if ((cellID==21) && (sideOrdinal==2) && (var->ID() == 0)) {
//    cout << "cellID 21, side 2, varID 0 getDofMapper().\n";
//  }
  
  int appliedWeightsGreatestEntryDimension = sideDim; // the greatest dimension for which appliedWeights is non-empty
  while (appliedWeightsGreatestEntryDimension >= minimumConstraintDimension)
  {
    int d = appliedWeightsGreatestEntryDimension; // the dimension of the subcell being constrained.
    
    map< GlobalIndexType, vector< pair<ConstrainingSubcellInfo, SubBasisReconciliationWeights > > > appliedWeightsForDimension = appliedWeights[d];
    
    // clear these out from the main container:
    appliedWeights[d].clear();

    map< GlobalIndexType, AppliedWeightVector >::iterator appliedWeightsIt;
    for (appliedWeightsIt = appliedWeightsForDimension.begin(); appliedWeightsIt != appliedWeightsForDimension.end(); appliedWeightsIt++) {
      AppliedWeightVector appliedWeightVectorForSubcell = appliedWeightsIt->second;
      
      for (AppliedWeightVector::iterator appliedWeightPairIt = appliedWeightVectorForSubcell.begin();
           appliedWeightPairIt != appliedWeightVectorForSubcell.end(); appliedWeightPairIt++) {
        AppliedWeightPair appliedWeightsForSubcell = *appliedWeightPairIt;
        
        ConstrainingSubcellInfo subcellInfo = appliedWeightsForSubcell.first;
        SubBasisReconciliationWeights prevWeights = appliedWeightsForSubcell.second;
        
        if (subcellInfo.dimension != d) {
          cout << "INTERNAL ERROR: subcellInfo.dimension should be d!\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "INTERNAL ERROR: subcellInfo.dimension should be d!");
        }
        
        CellPtr appliedConstraintCell = _meshTopology->getCell(subcellInfo.cellID);
        
        unsigned subcordInAppliedConstraintCell = CamelliaCellTools::subcellOrdinalMap(*appliedConstraintCell->topology(), sideDim,
                                                                                       subcellInfo.sideOrdinal, d, subcellInfo.subcellOrdinalInSide);
        
        DofOrderingPtr appliedConstraintTrialOrdering = _elementTypeForCell[subcellInfo.cellID]->trialOrderPtr;
        BasisPtr appliedConstraintBasis = appliedConstraintTrialOrdering->getBasis(var->ID(), subcellInfo.sideOrdinal);
        
        CellConstraints cellConstraints = getCellConstraints(subcellInfo.cellID);
        ConstrainingSubcellInfo subcellConstraint = cellConstraints.subcellConstraints[d][subcordInAppliedConstraintCell];
        
        CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
        
        DofOrderingPtr constrainingTrialOrdering = _elementTypeForCell[subcellConstraint.cellID]->trialOrderPtr;
        BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID(), subcellConstraint.sideOrdinal);
        
        
        unsigned subcellOrdinalInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(*constrainingCell->topology(), sideDim, subcellConstraint.sideOrdinal,
                                                                                         subcellConstraint.dimension, subcellConstraint.subcellOrdinalInSide);
        
        // DEBUGGING
        //      if ((cellID==4) && (sideOrdinal==3) && (var->ID()==0)) {
        //        cout << "while getting basis map for cell 2, side 1: cell " << subcellInfo.cellID << ", side " << subcellInfo.sideOrdinal;
        //        cout << ", subcell " << subcellInfo.subcellOrdinalInSide << " of dimension " << subcellInfo.dimension << " is constrained by ";
        //        cout << "cell " << subcellConstraint.cellID << ", side " << subcellConstraint.sideOrdinal;
        //        cout << ", subcell " << subcellConstraint.subcellOrdinalInSide << " of dimension " << subcellConstraint.dimension << endl;
        //        IndexType appliedConstraintSubcellEntityIndex = appliedConstraintCell->entityIndex(subcellInfo.dimension, subcordInAppliedConstraintCell);
        //        IndexType constrainingSubcellEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);
        //
        //        cout << CamelliaCellTools::entityTypeString(subcellInfo.dimension) << " " << appliedConstraintSubcellEntityIndex;
        //        cout << " constrained by " << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << constrainingSubcellEntityIndex;
        //        cout << endl;
        //      }
        
        shards::CellTopology constrainingTopo = constrainingCell->topology()->getCellTopologyData(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);
        
        SubBasisReconciliationWeights composedWeights;
        
        CellPtr ancestralCell = appliedConstraintCell->ancestralCellForSubcell(d, subcordInAppliedConstraintCell);
        
        RefinementBranch volumeRefinements = appliedConstraintCell->refinementBranchForSubcell(d, subcordInAppliedConstraintCell);
        
        pair<unsigned, unsigned> ancestralSubcell = appliedConstraintCell->ancestralSubcellOrdinalAndDimension(d, subcordInAppliedConstraintCell);
        
        unsigned ancestralSubcellOrdinal = ancestralSubcell.first;
        unsigned ancestralSubcellDimension = ancestralSubcell.second;
        
        if (ancestralSubcellOrdinal == -1) {
          cout << "Internal error: ancestral subcell ordinal was not found.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal error: ancestral subcell ordinal was not found.");
        }
        
        unsigned ancestralSideOrdinal;
        if (ancestralSubcellDimension == sideDim) {
          ancestralSideOrdinal = ancestralSubcellOrdinal;
        } else {
          
          IndexType ancestralSubcellEntityIndex = ancestralCell->entityIndex(ancestralSubcellDimension, ancestralSubcellOrdinal);
          
          // for subcells constrained by subcells of unlike dimension, we can handle any side that contains the ancestral subcell,
          // but for like-dimensional constraints, we do need the ancestralSideOrdinal to be the ancestor of the side in subcellInfo...
          
          if (subcellConstraint.dimension == d) {
            IndexType descendantSideEntityIndex = appliedConstraintCell->entityIndex(sideDim, subcellInfo.sideOrdinal);
            
            ancestralSideOrdinal = -1;
            int sideCount = CamelliaCellTools::getSideCount(*ancestralCell->topology());
            for (int side=0; side<sideCount; side++) {
              IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, side);
              if (ancestralSideEntityIndex == descendantSideEntityIndex) {
                ancestralSideOrdinal = side;
                break;
              }
              
              if (_meshTopology->entityIsAncestor(sideDim, ancestralSideEntityIndex, descendantSideEntityIndex)) {
                ancestralSideOrdinal = side;
                break;
              }
            }
            
            if (ancestralSideOrdinal == -1) {
              cout << "Error: no ancestor of side found.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: no ancestor of side contains the ancestral subcell.");
            }
            
            { // a sanity check:
              set<IndexType> sidesForSubcell = _meshTopology->getSidesContainingEntity(ancestralSubcellDimension, ancestralSubcellEntityIndex);
              IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, ancestralSideOrdinal);
              if (sidesForSubcell.find(ancestralSideEntityIndex) == sidesForSubcell.end()) {
                cout << "Error: the ancestral side does not contain the ancestral subcell.\n";
                TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "the ancestral side does not contain the ancestral subcell.");
              }
            }
            
          } else {
            // find some side in the ancestral cell that contains the ancestral subcell, then (there should be at least two; which one shouldn't matter)
            set<IndexType> sidesForSubcell = _meshTopology->getSidesContainingEntity(ancestralSubcellDimension, ancestralSubcellEntityIndex);
            
            ancestralSideOrdinal = -1;
            int sideCount = CamelliaCellTools::getSideCount(*ancestralCell->topology());
            for (int side=0; side<sideCount; side++) {
              IndexType ancestralSideEntityIndex = ancestralCell->entityIndex(sideDim, side);
              if (sidesForSubcell.find(ancestralSideEntityIndex) != sidesForSubcell.end()) {
                ancestralSideOrdinal = side;
                break;
              }
            }
          }
        }
        
        //      unsigned ancestralSideOrdinal = RefinementPattern::ancestralSubcellOrdinal(volumeRefinements, sideDim, subcellInfo.sideOrdinal); // TODO: fix this line -- should be allowed to change dimensions as we ascend the hierarchy, and what we want to do is follow the subcell up the hierarchy -- i.e. we might have a vertex that's a hanging node, so the parent of this cell could have an edge that contains that vertex.  One way or another, it should be the case that the "generalized ancestor" of the subcell is contained in some side of the furthest ancestor.  I believe this line as it stands probably does work for the (subcellConstraint.dimension == d) case.
        //
        if (ancestralSideOrdinal == -1) {
          cout << "Error: ancestralSideOrdinal not found.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: ancestralSideOrdinal not found.");
        }
        
        if (subcellConstraint.dimension != d) {
          unsigned ancestralPermutation = ancestralCell->sideSubcellPermutation(ancestralSideOrdinal, sideDim, 0);// side permutation as seen from the perspective of the fine cell's side's ancestor
          unsigned constrainingPermutation = constrainingCell->sideSubcellPermutation(subcellConstraint.sideOrdinal, sideDim, 0); // side permutation as seen from the perspective of the constraining cell's side
          
          shards::CellTopology constrainingSideTopo = constrainingCell->topology()->getCellTopologyData(sideDim, subcellConstraint.sideOrdinal);
          
          unsigned constrainingPermutationInverse = CamelliaCellTools::permutationInverse(constrainingSideTopo, constrainingPermutation);
          
          unsigned composedPermutation = CamelliaCellTools::permutationComposition(constrainingSideTopo, constrainingPermutationInverse, ancestralPermutation);
          
          SubBasisReconciliationWeights newWeightsToApply = BasisReconciliation::computeConstrainedWeights(d, appliedConstraintBasis, subcellInfo.subcellOrdinalInSide,
                                                                                                           volumeRefinements, subcellInfo.sideOrdinal, subcellConstraint.dimension,
                                                                                                           constrainingBasis, subcellConstraint.subcellOrdinalInSide,
                                                                                                           ancestralSideOrdinal, composedPermutation);
          
          composedWeights = BasisReconciliation::composedSubBasisReconciliationWeights(prevWeights, newWeightsToApply);
          
          // DEBUGGING
          //        if ((cellID==4) && (sideOrdinal==3)) {
          //          cout << "prevWeights:\n";
          //          Camellia::print("prevWeights fine ordinals", prevWeights.fineOrdinals);
          //          Camellia::print("prevWeights coarse ordinals", prevWeights.coarseOrdinals);
          //          cout << "prevWeights weights:\n" << prevWeights.weights;
          //
          //          cout << "newWeightsToApply:\n";
          //          Camellia::print("newWeightsToApply fine ordinals", newWeightsToApply.fineOrdinals);
          //          Camellia::print("newWeightsToApply coarse ordinals", newWeightsToApply.coarseOrdinals);
          //          cout << "newWeightsToApply weights:\n" << newWeightsToApply.weights;
          //
          //          cout << "composedWeights:\n";
          //          Camellia::print("composedWeights fine ordinals", composedWeights.fineOrdinals);
          //          Camellia::print("composedWeights coarse ordinals", composedWeights.coarseOrdinals);
          //          cout << "composedWeights weights:\n" << composedWeights.weights;
          //        }
          
        } else {
          RefinementBranch sideRefinements = RefinementPattern::subcellRefinementBranch(volumeRefinements, sideDim, ancestralSideOrdinal);
          
          IndexType constrainingEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);
          
          unsigned ancestralSubcellOrdinalInCell = ancestralCell->findSubcellOrdinal(subcellConstraint.dimension, constrainingEntityIndex);
          
          unsigned ancestralSubcellOrdinalInSide = CamelliaCellTools::subcellReverseOrdinalMap(*ancestralCell->topology(), sideDim, ancestralSideOrdinal, subcellConstraint.dimension, ancestralSubcellOrdinalInCell);
          
          unsigned ancestralPermutation = ancestralCell->sideSubcellPermutation(ancestralSideOrdinal, d, ancestralSubcellOrdinalInSide); // subcell permutation as seen from the perspective of the fine cell's side's ancestor
          unsigned constrainingPermutation = constrainingCell->sideSubcellPermutation(subcellConstraint.sideOrdinal, subcellConstraint.dimension, subcellConstraint.subcellOrdinalInSide); // subcell permutation as seen from the perspective of the constraining cell's side
          
          unsigned constrainingPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, constrainingPermutation);
          
          unsigned composedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, constrainingPermutationInverse, ancestralPermutation);
          
          SubBasisReconciliationWeights newWeightsToApply = _br.constrainedWeights(d, appliedConstraintBasis, subcellInfo.subcellOrdinalInSide, sideRefinements,
                                                                                   constrainingBasis, subcellConstraint.subcellOrdinalInSide, composedPermutation);
          
          // compose the new weights with existing weights for this subcell
          composedWeights = BasisReconciliation::composedSubBasisReconciliationWeights(prevWeights, newWeightsToApply);
          
          // DEBUGGING
          //        if ((cellID==4) && (sideOrdinal==3)) {
          //          cout << "prevWeights:\n";
          //          Camellia::print("prevWeights fine ordinals", prevWeights.fineOrdinals);
          //          Camellia::print("prevWeights coarse ordinals", prevWeights.coarseOrdinals);
          //          cout << "prevWeights weights:\n" << prevWeights.weights;
          //
          //          cout << "newWeightsToApply:\n";
          //          Camellia::print("newWeightsToApply fine ordinals", newWeightsToApply.fineOrdinals);
          //          Camellia::print("newWeightsToApply coarse ordinals", newWeightsToApply.coarseOrdinals);
          //          cout << "newWeightsToApply weights:\n" << newWeightsToApply.weights;
          //
          //          cout << "composedWeights:\n";
          //          Camellia::print("composedWeights fine ordinals", composedWeights.fineOrdinals);
          //          Camellia::print("composedWeights coarse ordinals", composedWeights.coarseOrdinals);
          //          cout << "composedWeights weights:\n" << composedWeights.weights;
          //        }
        }
        
        // populate the containers for the (d-1)-dimensional constituents of the constraining subcell
        if (subcellConstraint.dimension >= minimumConstraintDimension + 1) {
          int d1 = subcellConstraint.dimension-1;
          unsigned sscCount = constrainingTopo.getSubcellCount(d1);
          CellTopoPtr constrainingCellTopo = constrainingCell->topology();
          for (unsigned ssubcord=0; ssubcord<sscCount; ssubcord++) {
            unsigned ssubcordInCell = CamelliaCellTools::subcellOrdinalMap(*constrainingCellTopo, subcellConstraint.dimension, subcellOrdinalInConstrainingCell, d1, ssubcord);
            IndexType ssEntityIndex = constrainingCell->entityIndex(d1, ssubcordInCell);
            
            ConstrainingSubcellInfo subsubcellConstraint;
            subsubcellConstraint.cellID = subcellConstraint.cellID;
            subsubcellConstraint.sideOrdinal = subcellConstraint.sideOrdinal;
            subsubcellConstraint.dimension = d1;
            
            CellPtr constrainingCell = _meshTopology->getCell(subcellConstraint.cellID);
            subsubcellConstraint.subcellOrdinalInSide = CamelliaCellTools::subcellReverseOrdinalMap(*constrainingCellTopo, sideDim, subsubcellConstraint.sideOrdinal,
                                                                                                    d1, ssubcordInCell);
            
            //          if ((cellID==21) && (sideOrdinal==2) && (d1==0) && (subcellConstraint.cellID==6) && ((ssubcordInCell==3) || (ssubcordInCell==0))) {
            //            cout << "About to compute composed weights for sub subcell.\n";
            //          }
            
            SubBasisReconciliationWeights composedWeightsForSubSubcell = BasisReconciliation::weightsForCoarseSubcell(composedWeights, constrainingBasis, d1,
                                                                                                                      subsubcellConstraint.subcellOrdinalInSide, true);
            // DEBUGGING
            //          if ((cellID==4) && (sideOrdinal==3)) {
            //            Camellia::print("composedWeightsForSubSubcell.coarseOrdinals", composedWeightsForSubSubcell.coarseOrdinals);
            //            Camellia::print("composedWeightsForSubSubcell.fineOrdinals", composedWeightsForSubSubcell.fineOrdinals);
            //            cout << "composed weights for subSubSubcell:\n" << composedWeightsForSubSubcell.weights;
            //          }
            
            if (composedWeightsForSubSubcell.weights.size() > 0) {
              appliedWeights[d1][ssEntityIndex].push_back(make_pair(subsubcellConstraint, composedWeightsForSubSubcell));
            }
          }
        }
      
        // add sub-basis map for dofs interior to the constraining subcell
        // filter the weights whose coarse dofs are interior to this subcell, and create a SubBasisDofMapper for these (add it to varSideMap)
        SubBasisReconciliationWeights subcellInteriorWeights = BasisReconciliation::weightsForCoarseSubcell(composedWeights, constrainingBasis, subcellConstraint.dimension, subcellConstraint.subcellOrdinalInSide, false);
        
        if ((subcellInteriorWeights.coarseOrdinals.size() > 0) && (subcellInteriorWeights.fineOrdinals.size() > 0)) {
          CellConstraints constrainingCellConstraints = getCellConstraints(subcellConstraint.cellID);
          OwnershipInfo ownershipInfo = constrainingCellConstraints.owningCellIDForSubcell[subcellConstraint.dimension][subcellOrdinalInConstrainingCell];
          CellConstraints owningCellConstraints = getCellConstraints(ownershipInfo.cellID);
          SubCellDofIndexInfo owningCellDofIndexInfo = getOwnedGlobalDofIndices(ownershipInfo.cellID, owningCellConstraints);
          unsigned owningSubcellOrdinal = _meshTopology->getCell(ownershipInfo.cellID)->findSubcellOrdinal(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
          vector<GlobalIndexType> globalDofOrdinalsForSubcell = owningCellDofIndexInfo[ownershipInfo.dimension][owningSubcellOrdinal][var->ID()];
          
          // extract the global dof ordinals corresponding to subcellInteriorWeights.coarseOrdinals
          set<int> constrainingBasisOrdinalsForSubcell = constrainingBasis->dofOrdinalsForSubcell(subcellConstraint.dimension, subcellConstraint.subcellOrdinalInSide);
          vector<int> basisOrdinalsVector(constrainingBasisOrdinalsForSubcell.begin(),constrainingBasisOrdinalsForSubcell.end());
          vector<GlobalIndexType> globalDofOrdinals;
          for (int i=0; i<basisOrdinalsVector.size(); i++) {
            if (subcellInteriorWeights.coarseOrdinals.find(basisOrdinalsVector[i]) != subcellInteriorWeights.coarseOrdinals.end()) {
              globalDofOrdinals.push_back(globalDofOrdinalsForSubcell[i]);
            }
          }
          
          if (subcellInteriorWeights.coarseOrdinals.size() != globalDofOrdinals.size()) {
            cout << "Error: coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.\n";
            Camellia::print("coarseOrdinals", subcellInteriorWeights.coarseOrdinals);
            Camellia::print("globalDofOrdinals", globalDofOrdinals);
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "coarseOrdinals container isn't the same size as the globalDofOrdinals that it's supposed to correspond to.");
          }
          
          set<unsigned> basisDofOrdinals;
          basisDofOrdinals.insert(subcellInteriorWeights.fineOrdinals.begin(), subcellInteriorWeights.fineOrdinals.end()); // TODO: change fineOrdinals to be a set<unsigned>
          
          //        // DEBUGGING:
          //        if ((cellID==4) && (sideOrdinal==3)) {
          //          set<unsigned> globalDofOrdinalsSet(globalDofOrdinals.begin(),globalDofOrdinals.end());
          //          IndexType constrainingSubcellEntityIndex = constrainingCell->entityIndex(subcellConstraint.dimension, subcellOrdinalInConstrainingCell);
          //
          //          cout << "Determined constraints imposed by ";
          //          cout << CamelliaCellTools::entityTypeString(subcellConstraint.dimension) << " " << constrainingSubcellEntityIndex;
          //          cout << " (owned by " << CamelliaCellTools::entityTypeString(ownershipInfo.dimension) << " " << ownershipInfo.owningSubcellEntityIndex << ")\n";
          //
          //          Camellia::print("basisDofOrdinals mapped on cell 2, sideOrdinal 1",basisDofOrdinals);
          //          Camellia::print("globalDofOrdinals mapped on cell 2, sideOrdinal 1", globalDofOrdinalsSet);
          //          cout << "weights:\n" << subcellInteriorWeights.weights;
          //        }
          
          subBasisMap.weights = subcellInteriorWeights.weights;
          subBasisMap.globalDofOrdinals = globalDofOrdinals;
          subBasisMap.basisDofOrdinals = basisDofOrdinals;
          
          subBasisMaps.push_back(subBasisMap);
        }
        
      }
    }
    
    appliedWeightsGreatestEntryDimension = -1;
    for (int d=sideDim; d >= minimumConstraintDimension; d--)
    {
      if (appliedWeights[d].size() > 0)
      {
        appliedWeightsGreatestEntryDimension = d;
        break;
      }
    }
  } // (appliedWeightsGreatestEntryDimension >= 0)
  
  // now, we collect the local basis coefficients corresponding to each global ordinal
  // likely there is a more efficient way to do this, but for now this is our approach
  map< GlobalIndexType, map<unsigned, double> > weightsForGlobalOrdinal;
  
  map< unsigned, set<GlobalIndexType> > globalOrdinalsForFineOrdinal;
  
  for (vector<SubBasisMapInfo>::iterator subBasisIt = subBasisMaps.begin(); subBasisIt != subBasisMaps.end(); subBasisIt++) {
    subBasisMap = *subBasisIt;
    vector<GlobalIndexType> globalDofOrdinals = subBasisMap.globalDofOrdinals;
    set<unsigned> basisDofOrdinals = subBasisMap.basisDofOrdinals;
    vector<unsigned> basisDofOrdinalsVector(basisDofOrdinals.begin(),basisDofOrdinals.end());
    // weights are fine x coarse
    for (int j=0; j<subBasisMap.weights.dimension(1); j++) {
      GlobalIndexType globalDofOrdinal = globalDofOrdinals[j];
      map<unsigned, double> fineOrdinalCoefficientsThusFar = weightsForGlobalOrdinal[globalDofOrdinal];
      for (int i=0; i<subBasisMap.weights.dimension(0); i++) {
        unsigned fineOrdinal = basisDofOrdinalsVector[i];
        double coefficient = subBasisMap.weights(i,j);
        if (coefficient != 0) {
          if (fineOrdinalCoefficientsThusFar.find(fineOrdinal) != fineOrdinalCoefficientsThusFar.end()) {
            double tol = 1e-14;
            double previousCoefficient = fineOrdinalCoefficientsThusFar[fineOrdinal];
            if (abs(previousCoefficient-coefficient) > tol) {
              cout  << "ERROR: incompatible entries for fine ordinal " << fineOrdinal << " in representation of global ordinal " << globalDofOrdinal << endl;
              cout << "previousCoefficient = " << previousCoefficient << endl;
              cout << "coefficient = " << coefficient << endl;
              cout << "diff = " << abs(previousCoefficient - coefficient) << endl;
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Internal Error: incompatible entries for fine ordinal " );
            }
          }
          fineOrdinalCoefficientsThusFar[fineOrdinal] = coefficient;
          globalOrdinalsForFineOrdinal[fineOrdinal].insert(globalDofOrdinal);
        }
      }
      weightsForGlobalOrdinal[globalDofOrdinal] = fineOrdinalCoefficientsThusFar;
    }
  }

  // partition global ordinals according to which fine ordinals they interact with -- this is definitely not super-efficient
  set<GlobalIndexType> partitionedGlobalDofOrdinals;
  vector< set<GlobalIndexType> > globalDofOrdinalPartitions;
  vector< set<unsigned> > fineOrdinalsForPartition;
  
  for (map< GlobalIndexType, map<unsigned, double> >::iterator globalWeightsIt = weightsForGlobalOrdinal.begin();
       globalWeightsIt != weightsForGlobalOrdinal.end(); globalWeightsIt++) {
    GlobalIndexType globalOrdinal = globalWeightsIt->first;
    if (partitionedGlobalDofOrdinals.find(globalOrdinal) != partitionedGlobalDofOrdinals.end()) continue;
    
    set<GlobalIndexType> partition;
    partition.insert(globalOrdinal);
    
    set<unsigned> fineOrdinals;
    
    set<GlobalIndexType> globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed;
    
    map<unsigned, double> fineCoefficients = globalWeightsIt->second;
    for (map<unsigned, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++) {
      unsigned fineOrdinal = coefficientIt->first;
      fineOrdinals.insert(fineOrdinal);
      set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
      partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
    }
    
    globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
    
    while (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.size() != partition.size()) {
      for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++) {
        GlobalIndexType globalOrdinal = *globalOrdIt;
        if (globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.find(globalOrdinal) !=
            globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.end()) continue;
        map<unsigned, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
        for (map<unsigned, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++) {
          unsigned fineOrdinal = coefficientIt->first;
          fineOrdinals.insert(fineOrdinal);
          set<GlobalIndexType> globalOrdinalsForFine = globalOrdinalsForFineOrdinal[fineOrdinal];
          partition.insert(globalOrdinalsForFine.begin(),globalOrdinalsForFine.end());
        }
        
        globalOrdinalsInPartitionWhoseFineOrdinalsHaveBeenProcessed.insert(globalOrdinal);
      }
    }
    
    for (set<GlobalIndexType>::iterator globalOrdIt = partition.begin(); globalOrdIt != partition.end(); globalOrdIt++) {
      GlobalIndexType globalOrdinal = *globalOrdIt;
      partitionedGlobalDofOrdinals.insert(globalOrdinal);
    }
    globalDofOrdinalPartitions.push_back(partition);
    fineOrdinalsForPartition.push_back(fineOrdinals);
  }
  
  BasisMap varSideMap;
  for (int i=0; i<globalDofOrdinalPartitions.size(); i++) {
    set<GlobalIndexType> partition = globalDofOrdinalPartitions[i];
    set<unsigned> fineOrdinals = fineOrdinalsForPartition[i];
    vector<unsigned> fineOrdinalsVector(fineOrdinals.begin(), fineOrdinals.end());
    map<unsigned,int> fineOrdinalRowLookup;
    for (int i=0; i<fineOrdinalsVector.size(); i++) {
      fineOrdinalRowLookup[fineOrdinalsVector[i]] = i;
    }
    FieldContainer<double> weights(fineOrdinals.size(),partition.size());
    int col=0;
    for (set<GlobalIndexType>::iterator globalDofIt=partition.begin(); globalDofIt != partition.end(); globalDofIt++) {
      GlobalIndexType globalOrdinal = *globalDofIt;
      map<unsigned, double> fineCoefficients = weightsForGlobalOrdinal[globalOrdinal];
      for (map<unsigned, double>::iterator coefficientIt = fineCoefficients.begin(); coefficientIt != fineCoefficients.end(); coefficientIt++) {
        unsigned fineOrdinal = coefficientIt->first;
        double coefficient = coefficientIt->second;
        int row = fineOrdinalRowLookup[fineOrdinal];
        weights(row,col) = coefficient;
      }
      col++;
    }
    vector<GlobalIndexType> globalOrdinals(partition.begin(),partition.end());
    SubBasisDofMapperPtr subBasisMap = SubBasisDofMapper::subBasisDofMapper(fineOrdinals, globalOrdinals, weights);
    varSideMap.push_back(subBasisMap);

    // DEBUGGING
//    if ((cellID==4) && (sideOrdinal==3) && (var->ID() ==0)) {
//      cout << "Adding subBasisDofMapper.  Details:\n";
//      Camellia::print("fineOrdinals", fineOrdinals);
//      Camellia::print("globalOrdinals", globalOrdinals);
//      cout << "weights:\n" << weights;
//    }
  }
  
  return varSideMap;
}

CellConstraints GDAMinimumRule::getCellConstraints(GlobalIndexType cellID) {
  
  if (_constraintsCache.find(cellID) == _constraintsCache.end()) {
  
//    cout << "Getting cell constraints for cellID " << cellID << endl;
    
    typedef pair< IndexType, unsigned > CellPair;
    
    CellPtr cell = _meshTopology->getCell(cellID);
    DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
    CellTopoPtr topo = cell->topology();
    unsigned spaceDim = topo->getDimension();
    unsigned sideDim = spaceDim - 1;
    
    vector< vector< bool > > processedSubcells(spaceDim+1); // we process dimensions from high to low -- since we deal here with the volume basis case, we initialize this to spaceDim + 1
    vector< vector< ConstrainingSubcellInfo > > constrainingSubcellInfo(spaceDim + 1);
    
    ConstrainingSubcellInfo emptyConstraintInfo;
    emptyConstraintInfo.cellID = -1;
    
    for (int d=0; d<=spaceDim; d++) {
      int scCount = topo->getSubcellCount(d);
      processedSubcells[d] = vector<bool>(scCount, false);
      constrainingSubcellInfo[d] = vector< ConstrainingSubcellInfo >(scCount, emptyConstraintInfo);
    }
    
    for (int d=sideDim; d >= 0; d--) {
      int scCount = topo->getSubcellCount(d);
      for (int subcord=0; subcord < scCount; subcord++) { // subcell ordinals in cell
        if (! processedSubcells[d][subcord]) { // i.e. we don't yet know the constraining entity for this subcell
          
          IndexType entityIndex = cell->entityIndex(d, subcord);

//          cout << "entity of dimension " << d << " with entity index " << entityIndex << ": " << endl;
//          _meshTopology->printEntityVertices(d, entityIndex);
          pair<IndexType,unsigned> constrainingEntity = _meshTopology->getConstrainingEntity(d, entityIndex); // the constraining entity of the same dimension as this entity
          
          unsigned constrainingEntityDimension = constrainingEntity.second;
          IndexType constrainingEntityIndex = constrainingEntity.first;
          
//          cout << "is constrained by entity of dimension " << constrainingEntity.second << "  with entity index " << constrainingEntity.first << ": " << endl;
//          _meshTopology->printEntityVertices(constrainingEntity.second, constrainingEntity.first);

          set< CellPair > cellsForSubcell = _meshTopology->getCellsContainingEntity(constrainingEntityDimension, constrainingEntityIndex);
          
          unsigned leastH1Order = (unsigned)-1;
          set< CellPair > cellsWithLeastH1Order;
          for (set< CellPair >::iterator cellForSubcellIt = cellsForSubcell.begin(); cellForSubcellIt != cellsForSubcell.end(); cellForSubcellIt++) {
            IndexType subcellCellID = cellForSubcellIt->first;
            if (_cellH1Orders[subcellCellID] == leastH1Order) {
              cellsWithLeastH1Order.insert(*cellForSubcellIt);
            } else if (_cellH1Orders[subcellCellID] < leastH1Order) {
              cellsWithLeastH1Order.clear();
              leastH1Order = _cellH1Orders[subcellCellID];
              cellsWithLeastH1Order.insert(*cellForSubcellIt);
            }
            if (cellsWithLeastH1Order.size() == 0) {
              cout << "ERROR: No cells found for constraining subside entity.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "No cells found for constraining subside entity.");
            }
          }
          CellPair constrainingCellPair = *cellsWithLeastH1Order.begin(); // first one will have the earliest cell ID, given the sorting of set/pair.
          constrainingSubcellInfo[d][subcord].cellID = constrainingCellPair.first;
          
          unsigned constrainingCellID = constrainingCellPair.first;
          unsigned constrainingSideOrdinal = constrainingCellPair.second;
          
          constrainingSubcellInfo[d][subcord].sideOrdinal = constrainingSideOrdinal;
          CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);
          
          unsigned subcellOrdinalInConstrainingSide = constrainingCell->findSubcellOrdinalInSide(constrainingEntityDimension, constrainingEntityIndex, constrainingSideOrdinal);

          constrainingSubcellInfo[d][subcord].subcellOrdinalInSide = subcellOrdinalInConstrainingSide;
          constrainingSubcellInfo[d][subcord].dimension = constrainingEntityDimension;
          
          if (constrainingEntityDimension < d) {
            cout << "Internal error: constrainingEntityDimension < entity dimension.  This should not happen!\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constrainingEntityDimension < entity dimension.  This should not happen!");
          }
          
          if ((d==0) && (constrainingEntityDimension==0)) {
            if (constrainingEntityIndex != entityIndex) {
              cout << "Internal error: one vertex constrained by another.  This should not happen!\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "one vertex constrained by another.  Shouldn't happen!");
            }
          }
          
//          cout << "constraining subcell of dimension " << d << ", ordinal " << subcord << " with:\n";
//          cout << "  cellID " << constrainingSubcellInfo[d][subcord].cellID;
//          cout << ", sideOrdinal " << constrainingSubcellInfo[d][subcord].sideOrdinal;
//          cout << ", subcellOrdinalInSide " << constrainingSubcellInfo[d][subcord].subcellOrdinalInSide;
//          cout << ", dimension " << constrainingSubcellInfo[d][subcord].dimension << endl;
        }
      }
    }
    
    // fill in constraining subcell info for the volume (namely, it constrains itself):
    constrainingSubcellInfo[spaceDim][0].cellID = cellID;
    constrainingSubcellInfo[spaceDim][0].subcellOrdinalInSide = -1;
    constrainingSubcellInfo[spaceDim][0].sideOrdinal = -1;
    constrainingSubcellInfo[spaceDim][0].dimension = spaceDim;
    
    CellConstraints cellConstraints;
    cellConstraints.subcellConstraints = constrainingSubcellInfo;

    // determine subcell ownership from the perspective of the cell
    cellConstraints.owningCellIDForSubcell = vector< vector< OwnershipInfo > >(spaceDim+1);
    for (int d=0; d<spaceDim; d++) {
      vector<IndexType> scIndices = cell->getEntityIndices(d);
      unsigned scCount = scIndices.size();
      cellConstraints.owningCellIDForSubcell[d] = vector< OwnershipInfo >(scCount);
      for (int scOrdinal = 0; scOrdinal < scCount; scOrdinal++) {
        IndexType entityIndex = scIndices[scOrdinal];
        unsigned constrainingDimension = constrainingSubcellInfo[d][scOrdinal].dimension;
        IndexType constrainingEntityIndex;
        if (d==constrainingDimension) {
          constrainingEntityIndex = _meshTopology->getConstrainingEntity(d, entityIndex).first;
        } else {
          CellPtr constrainingCell = _meshTopology->getCell(constrainingSubcellInfo[d][scOrdinal].cellID);
          unsigned constrainingSubcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(*constrainingCell->topology(), sideDim, constrainingSubcellInfo[d][scOrdinal].sideOrdinal,
                                                                                           constrainingDimension, constrainingSubcellInfo[d][scOrdinal].subcellOrdinalInSide);
          constrainingEntityIndex = constrainingCell->entityIndex(constrainingDimension, constrainingSubcellOrdinalInCell);
        }
        pair<GlobalIndexType,GlobalIndexType> owningCellInfo = _meshTopology->leastActiveCellIndexContainingEntityConstrainedByConstrainingEntity(constrainingDimension, constrainingEntityIndex);
        cellConstraints.owningCellIDForSubcell[d][scOrdinal].cellID = owningCellInfo.first;
        cellConstraints.owningCellIDForSubcell[d][scOrdinal].owningSubcellEntityIndex = owningCellInfo.second; // the constrained entity in owning cell (which is a same-dimensional descendant of the constraining entity)
        cellConstraints.owningCellIDForSubcell[d][scOrdinal].dimension = constrainingDimension;
      }
    }
    // cell owns (and constrains) its interior:
    cellConstraints.owningCellIDForSubcell[spaceDim] = vector< OwnershipInfo >(1);
    cellConstraints.owningCellIDForSubcell[spaceDim][0].cellID = cellID;
    cellConstraints.owningCellIDForSubcell[spaceDim][0].owningSubcellEntityIndex = cellID;
    cellConstraints.owningCellIDForSubcell[spaceDim][0].dimension = spaceDim;
    
    _constraintsCache[cellID] = cellConstraints;
    
//    if (cellID==4) { // DEBUGGING
//      printConstraintInfo(cellID);
//    }
  }
  
  return _constraintsCache[cellID];
}

typedef map<int, vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
typedef map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
typedef vector< SubCellOrdinalToMap > SubCellDofIndexInfo; // index to vector: subcell dimension

set<GlobalIndexType> GDAMinimumRule::getFittableGlobalDofIndices(GlobalIndexType cellID, CellConstraints &constraints, int sideOrdinal) {
  // returns the global dof indices for basis functions which have support on the given side (i.e. their support intersected with the side has positive measure).  This is determined by taking the union of the global dof indices defined on all the constraining sides for the given side (the constraining sides are by definition unconstrained).
  SubCellDofIndexInfo dofIndexInfo = getGlobalDofIndices(cellID, constraints);
  CellPtr cell = _meshTopology->getCell(cellID);
  int sideDim = _meshTopology->getSpaceDim() - 1;
  
  GlobalIndexType constrainingCellID = constraints.subcellConstraints[sideDim][sideOrdinal].cellID;
  unsigned constrainingCellSideOrdinal = constraints.subcellConstraints[sideDim][sideOrdinal].sideOrdinal;

  CellPtr constrainingCell = _meshTopology->getCell(constrainingCellID);
  
  CellConstraints constrainingCellConstraints = getCellConstraints(constrainingCellID);
  
  SubCellDofIndexInfo constrainingCellDofIndexInfo = getGlobalDofIndices(constrainingCellID, constrainingCellConstraints);
  
  set<GlobalIndexType> fittableDofIndices;
  
  /*
   Exactly *which* of the constraining side's dofs to include is a little tricky.  We want the ones interior to the side for sure (since these are 
   not further constrained), and definitely want to exclude e.g. edge dofs that are further constrained by other edges.  But then it turns out we
   also want to exclude the vertices that are included in those constrained edges.  The general rule seems to be: if a subcell is excluded, then
   all its constituent subcells should also be excluded.
   */
  
  // iterate through all the subcells of the constraining side, collecting dof indices
  CellTopoPtr constrainingCellTopo = constrainingCell->topology();
  shards::CellTopology constrainingSideTopo = constrainingCellTopo->getBaseCellTopologyData(sideDim, constrainingCellSideOrdinal);
  set<unsigned> constrainingCellConstrainedNodes; // vertices in the constraining cell that belong to subcell entities that are constrained by other cells
  for (int d=sideDim; d>=0; d--) {
    int scCount = constrainingSideTopo.getSubcellCount(d);
    for (int scOrdinal=0; scOrdinal<scCount; scOrdinal++) {
      int scordInConstrainingCell = CamelliaCellTools::subcellOrdinalMap(*constrainingCellTopo, sideDim, constrainingCellSideOrdinal, d, scOrdinal);
      set<unsigned> scNodesInConstrainingCell;
      bool nodeNotKnownToBeConstrainedFound = false;
      if (d==0) {
        scNodesInConstrainingCell.insert(scordInConstrainingCell);
        if (constrainingCellConstrainedNodes.find(scordInConstrainingCell) == constrainingCellConstrainedNodes.end()) {
          nodeNotKnownToBeConstrainedFound = true;
        }
      } else {
        int nodeCount = constrainingCellTopo->getNodeCount(d, scordInConstrainingCell);
        for (int nodeOrdinal=0; nodeOrdinal<nodeCount; nodeOrdinal++) {
          unsigned nodeInConstrainingCell = constrainingCellTopo->getNodeMap(d, scordInConstrainingCell, nodeOrdinal);
          scNodesInConstrainingCell.insert(nodeInConstrainingCell);
          if (constrainingCellConstrainedNodes.find(nodeInConstrainingCell) == constrainingCellConstrainedNodes.end()) {
            nodeNotKnownToBeConstrainedFound = true;
          }
        }
      }
      
      if (!nodeNotKnownToBeConstrainedFound) continue; // all nodes constrained, so we can skip further processing
      
      IndexType subcellIndex = constrainingCell->entityIndex(d, scordInConstrainingCell);
      pair<IndexType,unsigned> constrainingSubcell = _meshTopology->getConstrainingEntity(d, subcellIndex);
      if ((constrainingSubcell.second == d) && (constrainingSubcell.first == subcellIndex)) { // i.e. the subcell is not further constrained.
        map<int, vector<GlobalIndexType> > dofIndicesInConstrainingSide = constrainingCellDofIndexInfo[d][scordInConstrainingCell];
        for (map<int, vector<GlobalIndexType> >::iterator dofIndicesIt = dofIndicesInConstrainingSide.begin();
             dofIndicesIt != dofIndicesInConstrainingSide.end(); dofIndicesIt++) {
          fittableDofIndices.insert(dofIndicesIt->second.begin(), dofIndicesIt->second.end());
        }
      } else {
        constrainingCellConstrainedNodes.insert(scNodesInConstrainingCell.begin(),scNodesInConstrainingCell.end());
      }
    }
  }
  return fittableDofIndices;
}

SubCellDofIndexInfo GDAMinimumRule::getOwnedGlobalDofIndices(GlobalIndexType cellID, CellConstraints &constraints) {
  // there's a lot of redundancy between this method and the dof-counting bit of rebuild lookups.  May be worth factoring that out.
  
  int spaceDim = _meshTopology->getSpaceDim();
  int sideDim = spaceDim - 1;
  
  SubCellDofIndexInfo scInfo(spaceDim+1);
  
  CellTopoPtr topo = _elementTypeForCell[cellID]->cellTopoPtr;
  
  typedef vector< SubBasisDofMapperPtr > BasisMap;
  
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  map<int, VarPtr> trialVars = _varFactory.trialVars();
  
//  cout << "Owned global dof indices for cell " << cellID << endl;
  
  GlobalIndexType globalDofIndex = _globalCellDofOffsets[cellID]; // this cell's first globalDofIndex

  for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++) {
    map< pair<unsigned,IndexType>, pair<unsigned, unsigned> > entitiesClaimedForVariable; // maps from the constraining entity claimed to the (d, scord) entry that claimed it.
    VarPtr var = varIt->second;
    unsigned scordForBasis;
    bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
//    cout << " var " << var->name() << ":\n";
  
    for (int d=0; d<=spaceDim; d++) {
      int scCount = topo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++) {
        OwnershipInfo ownershipInfo = constraints.owningCellIDForSubcell[d][scord];
        if (ownershipInfo.cellID == cellID) { // owned by this cell: count all the constraining dofs as entries for this cell
          GlobalIndexType constrainingCellID = constraints.subcellConstraints[d][scord].cellID;
          unsigned constrainingDimension = constraints.subcellConstraints[d][scord].dimension;
          DofOrderingPtr trialOrdering = _elementTypeForCell[constrainingCellID]->trialOrderPtr;
          BasisPtr basis; // the constraining basis for the subcell
          if (varHasSupportOnVolume) {
            if (d==spaceDim) {
              // then there is only one subcell ordinal (and there will be -1's in sideOrdinal and subcellOrdinalInSide....
              scordForBasis = 0;
            } else {
              scordForBasis = CamelliaCellTools::subcellOrdinalMap(*_meshTopology->getCell(constrainingCellID)->topology(), sideDim,
                                                                   constraints.subcellConstraints[d][scord].sideOrdinal,
                                                                   constrainingDimension, constraints.subcellConstraints[d][scord].subcellOrdinalInSide);
            }
            basis = trialOrdering->getBasis(var->ID());
          } else {
            if (d==spaceDim) continue; // side bases don't have any support on the interior of the cell...
            scordForBasis = constraints.subcellConstraints[d][scord].subcellOrdinalInSide; // the basis sees the side, so that's the view to use for subcell ordinal
            basis = trialOrdering->getBasis(var->ID(), constraints.subcellConstraints[d][scord].sideOrdinal);
          }
          int minimumConstraintDimension = BasisReconciliation::minimumSubcellDimension(basis);
          if (minimumConstraintDimension > d) continue; // then we don't enforce (or own) anything for this subcell/basis combination
          
          pair<unsigned, IndexType> owningSubcellEntity = make_pair(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
          if (entitiesClaimedForVariable.find(owningSubcellEntity) != entitiesClaimedForVariable.end()) {
            // already processed this guy on this cell: just copy
            pair<unsigned,unsigned> previousConstrainedSubcell = entitiesClaimedForVariable[owningSubcellEntity];
            scInfo[d][scord][var->ID()] = scInfo[previousConstrainedSubcell.first][previousConstrainedSubcell.second][var->ID()];
            continue;
          } else {
            entitiesClaimedForVariable[owningSubcellEntity] = make_pair(d, scord);
          }
          
          int dofOrdinalCount = basis->dofOrdinalsForSubcell(constrainingDimension, scordForBasis).size();
          vector<GlobalIndexType> globalDofIndices;
          
//          cout << "   dim " << d << ", scord " << scord << ":";
          
          for (int i=0; i<dofOrdinalCount; i++) {
//            cout << " " << globalDofIndex;
            globalDofIndices.push_back(globalDofIndex++);
          }
//          cout << endl;
          scInfo[d][scord][var->ID()] = globalDofIndices;
        }
      }
    }
  }
  return scInfo;
}

void printDofIndexInfo(GlobalIndexType cellID, SubCellDofIndexInfo &dofIndexInfo) {
  //typedef vector< SubCellOrdinalToMap > SubCellDofIndexInfo; // index to vector: subcell dimension
  cout << "Dof Index info for cell ID " << cellID << ":\n";
  ostringstream varIDstream;
  for (int d=0; d<dofIndexInfo.size(); d++) {
    //    typedef map<int, vector<GlobalIndexType> > VarIDToDofIndices; // key: varID
    //    typedef map<unsigned, VarIDToDofIndices> SubCellOrdinalToMap; // key: subcell ordinal
    cout << "****** dimension " << d << " *******\n";
    SubCellOrdinalToMap scordMap = dofIndexInfo[d];
    for (SubCellOrdinalToMap::iterator scordMapIt = scordMap.begin(); scordMapIt != scordMap.end(); scordMapIt++) {
      cout << "  scord " << scordMapIt->first << ":\n";
      VarIDToDofIndices varMap = scordMapIt->second;
      for (VarIDToDofIndices::iterator varIt = varMap.begin(); varIt != varMap.end(); varIt++) {
        varIDstream.str("");
        varIDstream << "     var " << varIt->first << ", global dofs";
        if (varIt->second.size() > 0) Camellia::print(varIDstream.str(), varIt->second);
      }
    }
  }
}

SubCellDofIndexInfo GDAMinimumRule::getGlobalDofIndices(GlobalIndexType cellID, CellConstraints &constraints) {
  /**************** ESTABLISH OWNERSHIP ****************/
  SubCellDofIndexInfo dofIndexInfo = getOwnedGlobalDofIndices(cellID, constraints);
  
  CellPtr cell = _meshTopology->getCell(cellID);
  CellTopoPtr topo = _elementTypeForCell[cellID]->cellTopoPtr;
  
  int spaceDim = topo->getDimension();
  
  // fill in the other global dof indices (the ones not owned by this cell):
  map< GlobalIndexType, SubCellDofIndexInfo > otherDofIndexInfoCache; // local lookup, to avoid a bunch of redundant calls to getOwnedGlobalDofIndices
  for (int d=0; d<=spaceDim; d++) {
    int scCount = topo->getSubcellCount(d);
    for (int scord=0; scord<scCount; scord++) {
      if (dofIndexInfo[d].find(scord) == dofIndexInfo[d].end()) { // this one not yet filled in
        OwnershipInfo owningCellInfo = constraints.owningCellIDForSubcell[d][scord];
        GlobalIndexType owningCellID = owningCellInfo.cellID;
        CellConstraints owningConstraints = getCellConstraints(owningCellID);
        GlobalIndexType scEntityIndex = owningCellInfo.owningSubcellEntityIndex;
        CellPtr owningCell = _meshTopology->getCell(owningCellID);
        unsigned owningCellScord = owningCell->findSubcellOrdinal(owningCellInfo.dimension, scEntityIndex);
        if (otherDofIndexInfoCache.find(owningCellID) == otherDofIndexInfoCache.end()) {
          otherDofIndexInfoCache[owningCellID] = getOwnedGlobalDofIndices(owningCellID, owningConstraints);
        }
        SubCellDofIndexInfo owningDofIndexInfo = otherDofIndexInfoCache[owningCellID];
        dofIndexInfo[d][scord] = owningDofIndexInfo[owningCellInfo.dimension][owningCellScord];
      }
    }
  }

  // DEBUGGING
//  printDofIndexInfo(cellID, dofIndexInfo);
  
  return dofIndexInfo;
}

LocalDofMapperPtr GDAMinimumRule::getDofMapper(GlobalIndexType cellID, CellConstraints &constraints, int varIDToMap, int sideOrdinalToMap) {
  if ((varIDToMap == -1) && (sideOrdinalToMap == -1)) {
    // a mapper for the whole dof ordering: we cache these...
    if (_dofMapperCache.find(cellID) != _dofMapperCache.end()) {
      return _dofMapperCache[cellID];
    }
  }
  
  CellPtr cell = _meshTopology->getCell(cellID);
  CellTopoPtr topo = _elementTypeForCell[cellID]->cellTopoPtr;
  int sideCount = CamelliaCellTools::getSideCount(*topo);
  int spaceDim = topo->getDimension();
  
  DofOrderingPtr trialOrdering = _elementTypeForCell[cellID]->trialOrderPtr;
  map<int, VarPtr> trialVars = _varFactory.trialVars();
  
//  printConstraintInfo(cellID);
  
//  FieldContainer<bool> processedDofs(trialOrdering->totalDofs());
//  processedDofs.initialize(false);
  
  typedef vector< SubBasisDofMapperPtr > BasisMap;
  map< int, BasisMap > volumeMap; // keys are variable IDs
  vector< map< int, BasisMap > > sideMaps(sideCount);
  
  vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(sideCount);
  for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    if ((sideOrdinalToMap != -1) && (sideOrdinal != sideOrdinalToMap)) continue; // skip this side...
    fittableGlobalDofOrdinalsOnSides[sideOrdinal] = getFittableGlobalDofIndices(cellID, constraints, sideOrdinal);
//    cout << "sideOrdinal " << sideOrdinal;
//    Camellia::print(", fittableGlobalDofIndices", fittableGlobalDofOrdinalsOnSides[sideOrdinal]);
  }
  
  set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume; // just the interior dof ordinals
  
  /**************** ESTABLISH OWNERSHIP ****************/
  SubCellDofIndexInfo dofIndexInfo = getGlobalDofIndices(cellID, constraints);
  
  /**************** CREATE SUB-BASIS MAPPERS ****************/
  
  for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++) {
    VarPtr var = varIt->second;
    bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
    bool omitVarEntry = (varIDToMap != -1) && (var->ID() != varIDToMap);
    
    if (omitVarEntry) continue;
    
//    if ((sideOrdinalToMap != -1) && varHasSupportOnVolume) {
//      cout << "WARNING: looks like you're trying to impose BCs on a variable defined on the volume.  (For volume variables, we don't yet appropriately filter just the sides you're asking for--i.e. just the boundary--in getDofMapper).\n";
//    }
    
    if (varHasSupportOnVolume) {
      volumeMap[var->ID()] = getBasisMap(cellID, dofIndexInfo, var); // this is where we should add an argument to specify the side being requested...
      fittableGlobalDofOrdinalsInVolume.insert(dofIndexInfo[spaceDim][0][var->ID()].begin(),dofIndexInfo[spaceDim][0][var->ID()].end());
    } else {
      for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++) {
        if ((sideOrdinalToMap != -1) && (sideOrdinal != sideOrdinalToMap)) continue; // skip this side...
        sideMaps[sideOrdinal][var->ID()] = getBasisMap(cellID, dofIndexInfo, var, sideOrdinal);
      }
    }
  }
  
  LocalDofMapperPtr dofMapper = Teuchos::rcp( new LocalDofMapper(trialOrdering,volumeMap,fittableGlobalDofOrdinalsInVolume,
                                                                 sideMaps,fittableGlobalDofOrdinalsOnSides,varIDToMap,sideOrdinalToMap) );
  if ((varIDToMap == -1) && (sideOrdinalToMap == -1)) {
    // a mapper for the whole dof ordering: we cache these...
    _dofMapperCache[cellID] = dofMapper;
    return _dofMapperCache[cellID];
  } else {
    return dofMapper;
  }
}

PartitionIndexType GDAMinimumRule::partitionForGlobalDofIndex( GlobalIndexType globalDofIndex ) {
  PartitionIndexType numRanks = _partitionDofCounts.size();
  GlobalIndexType totalDofCount = 0;
  for (PartitionIndexType i=0; i<numRanks; i++) {
    totalDofCount += _partitionDofCounts(i);
    if (totalDofCount > globalDofIndex) {
      return i;
    }
  }
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid globalDofIndex");
}

void GDAMinimumRule::printConstraintInfo(GlobalIndexType cellID) {
  CellConstraints cellConstraints = getCellConstraints(cellID);
  cout << "***** Constraints for cell " << cellID << " ****** \n";
  CellPtr cell = _meshTopology->getCell(cellID);
  int spaceDim = cell->topology()->getDimension();
  for (int d=0; d<spaceDim; d++) {
    cout << "  dimension " << d << " subcells: " << endl;
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int scord=0; scord<subcellCount; scord++) {
      ConstrainingSubcellInfo constraintInfo = cellConstraints.subcellConstraints[d][scord];
      cout << "    ordinal " << scord  << " constrained by cell " << constraintInfo.cellID;
      cout << ", side " << constraintInfo.sideOrdinal << "'s dimension " << constraintInfo.dimension << " subcell ordinal " << constraintInfo.subcellOrdinalInSide << endl;
    }
  }

  cout << "Ownership info:\n";
  for (int d=0; d<spaceDim; d++) {
    cout << "  dimension " << d << " subcells: " << endl;
    int subcellCount = cell->topology()->getSubcellCount(d);
    for (int scord=0; scord<subcellCount; scord++) {
      cout << "    ordinal " << scord  << " owned by cell " << cellConstraints.owningCellIDForSubcell[d][scord].cellID << endl;
    }
  }
}

void GDAMinimumRule::rebuildLookups() {
  _constraintsCache.clear(); // to free up memory, could clear this again after the lookups are rebuilt.  Having the cache is most important during the construction below.
  _dofMapperCache.clear();
  
  _partitionFluxIndexOffsets.clear();
  _partitionTraceIndexOffsets.clear();
  _partitionIndexOffsetsForVarID.clear();
  
  int rank = Teuchos::GlobalMPISession::getRank();
//  cout << "GDAMinimumRule: Rebuilding lookups on rank " << rank << endl;
  set<GlobalIndexType> myCellIDs = _partitions[rank];
  
  map<int, VarPtr> trialVars = _varFactory.trialVars();
  
  _cellDofOffsets.clear(); // within the partition, offsets for the owned dofs in cell
  
  int spaceDim = _meshTopology->getSpaceDim();
  int sideDim = spaceDim - 1;
  
  // pieces of this remain fairly ugly--the brute force searches are limited to entities on a cell (i.e. < O(12) items to search in a hexahedron),
  // and I've done a reasonable job only doing them when we need the result, but they still are brute force searches.  By tweaking
  // the design of MeshTopology and Cell to take better advantage of regularities (or just to store better lookups), we should be able to do better.
  // But in the interest of avoiding wasting development time on premature optimization, I'm leaving it as is for now...
  
  _partitionDofCount = 0; // how many dofs we own locally
  for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++) {
    GlobalIndexType cellID = *cellIDIt;
    _cellDofOffsets[cellID] = _partitionDofCount;
    CellPtr cell = _meshTopology->getCell(cellID);
    CellTopoPtr topo = cell->topology();
    CellConstraints constraints = getCellConstraints(cellID);
    
    set< pair<unsigned,IndexType> > entitiesClaimedForCell;
    
    for (int d=0; d<=spaceDim; d++) {
      int scCount = topo->getSubcellCount(d);
      for (int scord=0; scord<scCount; scord++) {
        OwnershipInfo ownershipInfo = constraints.owningCellIDForSubcell[d][scord];
        if (ownershipInfo.cellID == cellID) { // owned by this cell: count all the constraining dofs as entries for this cell
          pair<unsigned, IndexType> owningSubcellEntity = make_pair(ownershipInfo.dimension, ownershipInfo.owningSubcellEntityIndex);
          if (entitiesClaimedForCell.find(owningSubcellEntity) != entitiesClaimedForCell.end()) {
            continue; // already processed this guy on this cell
          } else {
            entitiesClaimedForCell.insert(owningSubcellEntity);
          }
          GlobalIndexType constrainingCellID = constraints.subcellConstraints[d][scord].cellID;
          unsigned constrainingSubcellDimension = constraints.subcellConstraints[d][scord].dimension;
          DofOrderingPtr trialOrdering = _elementTypeForCell[constrainingCellID]->trialOrderPtr;
          for (map<int, VarPtr>::iterator varIt = trialVars.begin(); varIt != trialVars.end(); varIt++) {
            VarPtr var = varIt->second;
            unsigned scordForBasis;
            bool varHasSupportOnVolume = (var->varType() == FIELD) || (var->varType() == TEST);
            BasisPtr basis; // the constraining basis for the subcell
            if (varHasSupportOnVolume) {
               // volume basis => the basis sees the cell as a whole: in constraining cell, map from side scord to the volume
              if (constrainingSubcellDimension==spaceDim) {
                // then there is only one subcell ordinal (and there will be -1's in sideOrdinal and subcellOrdinalInSide....
                scordForBasis = 0;
              } else {
                scordForBasis = CamelliaCellTools::subcellOrdinalMap(*_meshTopology->getCell(constrainingCellID)->topology(), sideDim,
                                                                     constraints.subcellConstraints[d][scord].sideOrdinal,
                                                                     constrainingSubcellDimension, constraints.subcellConstraints[d][scord].subcellOrdinalInSide);
              }
              basis = trialOrdering->getBasis(var->ID());
              // field
              int ordinalCount = basis->dofOrdinalsForSubcell(constrainingSubcellDimension, scordForBasis).size();
              for (int ordinal=0; ordinal<ordinalCount; ordinal++) {
                _partitionIndexOffsetsForVarID[var->ID()].insert(ordinal+_partitionDofCount);
              }
            } else {
              if (constrainingSubcellDimension==spaceDim) continue; // side bases don't have any support on the interior of the cell...
              scordForBasis = constraints.subcellConstraints[d][scord].subcellOrdinalInSide; // the basis sees the side, so that's the view to use for subcell ordinal
              basis = trialOrdering->getBasis(var->ID(), constraints.subcellConstraints[d][scord].sideOrdinal);

              int ordinalCount = basis->dofOrdinalsForSubcell(constrainingSubcellDimension, scordForBasis).size();
              if (var->varType()==FLUX) {
                for (int ordinal=0; ordinal<ordinalCount; ordinal++) {
                  _partitionFluxIndexOffsets.insert(ordinal+_partitionDofCount);
                  _partitionIndexOffsetsForVarID[var->ID()].insert(ordinal+_partitionDofCount);
                }
              } else if (var->varType() == TRACE) {
                for (int ordinal=0; ordinal<ordinalCount; ordinal++) {
                  _partitionTraceIndexOffsets.insert(ordinal+_partitionDofCount);
                  _partitionIndexOffsetsForVarID[var->ID()].insert(ordinal+_partitionDofCount);
                }
              }
            }
            _partitionDofCount += basis->dofOrdinalsForSubcell(constrainingSubcellDimension, scordForBasis).size();
          }
        }
      }
    }
  }
  int numRanks = Teuchos::GlobalMPISession::getNProc();
  _partitionDofCounts.resize(numRanks);
  _partitionDofCounts.initialize(0.0);
  _partitionDofCounts[rank] = _partitionDofCount;
  MPIWrapper::entryWiseSum(_partitionDofCounts);
//  if (rank==0) cout << "partitionDofCounts:\n" << _partitionDofCounts;
  _partitionDofOffset = 0; // add this to a local partition dof index to get the global dof index
  for (int i=0; i<rank; i++) {
    _partitionDofOffset += _partitionDofCounts[i];
  }
  _globalDofCount = _partitionDofOffset;
  for (int i=rank; i<numRanks; i++) {
    _globalDofCount += _partitionDofCounts[i];
  }
//  if (rank==0) cout << "globalDofCount: " << _globalDofCount << endl;
  // collect and communicate global cell dof offsets:
  int activeCellCount = _meshTopology->getActiveCellIndices().size();
  FieldContainer<int> globalCellIDDofOffsets(activeCellCount);
  int partitionCellOffset = 0;
  for (int i=0; i<rank; i++) {
    partitionCellOffset += _partitions[i].size();
  }
  // fill in our _cellDofOffsets:

  int i=0;
  for (set<GlobalIndexType>::iterator cellIDIt = myCellIDs.begin(); cellIDIt != myCellIDs.end(); cellIDIt++) {
    globalCellIDDofOffsets[partitionCellOffset+i] = _cellDofOffsets[*cellIDIt] + _partitionDofOffset;
    i++;
  }
  // global copy:
  MPIWrapper::entryWiseSum(globalCellIDDofOffsets);
  // fill in the lookup table:
  _globalCellDofOffsets.clear();
  int globalCellIndex = 0;
  for (int i=0; i<numRanks; i++) {
    set<GlobalIndexType> rankCellIDs = _partitions[i];
    for (set<GlobalIndexType>::iterator cellIDIt = rankCellIDs.begin(); cellIDIt != rankCellIDs.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      _globalCellDofOffsets[cellID] = globalCellIDDofOffsets[globalCellIndex];
//      if (rank==numRanks-1) cout << "global dof offset for cell " << cellID << ": " << _globalCellDofOffsets[cellID] << endl;
      globalCellIndex++;
    }
  }
  
  _cellIDsForElementType = vector< map< ElementType*, vector<GlobalIndexType> > >(numRanks);
  for (int i=0; i<numRanks; i++) {
    set<GlobalIndexType> cellIDs = _partitions[i];
    for (set<GlobalIndexType>::iterator cellIDIt = cellIDs.begin(); cellIDIt != cellIDs.end(); cellIDIt++) {
      GlobalIndexType cellID = *cellIDIt;
      ElementTypePtr elemType = _elementTypeForCell[cellID];
      _cellIDsForElementType[i][elemType.get()].push_back(cellID);
    }
  }
}