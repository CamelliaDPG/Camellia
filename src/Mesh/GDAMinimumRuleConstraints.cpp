//
//  GDAMinimumRuleConstraints.cpp
//  Camellia
//
//  Created by Nate Roberts on 8/18/14.
//
//

#include "GDAMinimumRuleConstraints.h"

#include "CamelliaCellTools.h"

#include "GDAMinimumRule.h"

bool GDAMinimumRuleConstraints::constraintEntryExists(AnnotatedEntity &subcellInfo) {
  if (_constraintEntries.find(subcellInfo) != _constraintEntries.end()) {
    return _constraintEntries[subcellInfo].get() != NULL;
  } else {
    return false;
  }
}

void GDAMinimumRuleConstraints::computeConstraintWeights(map<AnnotatedEntity, SubBasisReconciliationWeights> & constraintWeights,
                                                         GDAMinimumRule *minRule, ConstraintEntryPtr prevEntry, ConstraintEntryPtr thisEntry, VarPtr var) {
  AnnotatedEntity prevSubcellInfo = prevEntry->subcellInfo();
  
  vector< ConstraintEntryPtr > priorEntries = prevEntry->getPriorEntries();
  
  // rule: we only fill in constraintWeights for thisEntry after we have determined all prevEntry's prior entry weights (i.e. we have entirely determined prevEntry's representation in terms of the local basis)
  for (int i=0; i<priorEntries.size(); i++) {
    ConstraintEntryPtr priorEntry = priorEntries[i];
    if (constraintWeights.find(priorEntry->subcellInfo()) == constraintWeights.end()) {
      computeConstraintWeights(constraintWeights, minRule, priorEntry, prevEntry, var);
    }
  }
  
  SubBasisReconciliationWeights prevWeights = constraintWeights[prevSubcellInfo];
  
  DofOrderingPtr prevTrialOrdering = minRule->elementType(prevSubcellInfo.cellID)->trialOrderPtr;
  BasisPtr prevBasis = prevTrialOrdering->getBasis(var->ID(), prevSubcellInfo.sideOrdinal);
  
  AnnotatedEntity constrainingSubcellInfo = thisEntry->subcellInfo();
  
  SubBasisReconciliationWeights composedWeights;
  
  if (prevEntry->isUnconstrained()) { // then the "constraint" is actually a subcell of the prev. entity
    // apply the prevWeights to the subcell in thisEntry
    composedWeights = BasisReconciliation::weightsForCoarseSubcell(prevWeights, prevBasis, thisEntry->entityDim(), thisEntry->subcellOrdinalInSide(), true);
  } else {
    DofOrderingPtr constrainingTrialOrdering = minRule->elementType(constrainingSubcellInfo.cellID)->trialOrderPtr;
    BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID(), constrainingSubcellInfo.sideOrdinal);
    
    RefinementBranch volumeRefinements = prevEntry->volumeRefinements();
    unsigned composedPermutation = prevEntry->composedPermutation();
    // TODO: work out what to do here for volume basis--in particular, what happens to the sideOrdinal arguments??  It seems like these should not apply... (they should be redundant with subcell ordinal)  It may well be that what we should do is ensure that the sideOrdinal arguments are -1, and then BR should understand that as a flag indicating that volumeRefinements really does apply to the ancestral domain, etc.
    SubBasisReconciliationWeights newWeights = BasisReconciliation::computeConstrainedWeights(prevSubcellInfo.dimension, prevBasis, prevSubcellInfo.subcellOrdinal,
                                                                                              volumeRefinements, prevSubcellInfo.sideOrdinal,
                                                                                              constrainingSubcellInfo.dimension,
                                                                                              constrainingBasis, constrainingSubcellInfo.subcellOrdinal,
                                                                                              prevEntry->ancestralSideOrdinal(), composedPermutation);
    
    composedWeights = BasisReconciliation::composedSubBasisReconciliationWeights(prevWeights, newWeights);
  }
  if (constraintWeights.find(constrainingSubcellInfo) == constraintWeights.end()) {
    constraintWeights[constrainingSubcellInfo] = composedWeights;
  } else {
    SubBasisReconciliationWeights existingWeights = constraintWeights[constrainingSubcellInfo];
    SubBasisReconciliationWeights summedWeights = BasisReconciliation::sumWeights(existingWeights,composedWeights);
    constraintWeights[constrainingSubcellInfo] = summedWeights;
  }
}

void GDAMinimumRuleConstraints::computeConstraintWeights(map<AnnotatedEntity, SubBasisReconciliationWeights> & constraintWeights,
                                                         GDAMinimumRule *minRule, ConstraintEntryPtr rootEntry, VarPtr var) {
  GDAMinimumRuleConstraints definedConstraints;
  
  AnnotatedEntity rootInfo = rootEntry->subcellInfo();
  // construct the DAG, source entry:
  definedConstraints.setConstraintEntry(rootInfo, rootEntry);
  
  BasisPtr basis = getBasis(minRule, rootEntry, var);
  
  SubBasisReconciliationWeights unitWeights;
  unitWeights.weights.resize(basis->getCardinality(), basis->getCardinality());
  set<int> allOrdinals;
  for (int i=0; i<basis->getCardinality(); i++) {
    allOrdinals.insert(i);
    unitWeights.weights(i,i) = 1.0;
  }
  unitWeights.fineOrdinals = allOrdinals;
  unitWeights.coarseOrdinals = allOrdinals;
  
  constraintWeights[rootInfo] = unitWeights;
  
  computeConstraintWeights(constraintWeights, minRule, rootEntry, rootEntry, var);
}

ConstraintEntryPtr GDAMinimumRuleConstraints::getConstraintEntry(AnnotatedEntity &subcellInfo) {
  return _constraintEntries[subcellInfo];
}

void GDAMinimumRuleConstraints::setConstraintEntry(AnnotatedEntity &subcellInfo, ConstraintEntryPtr entry) {
  _constraintEntries[subcellInfo] = entry;
}

BasisPtr GDAMinimumRuleConstraints::getBasis(GDAMinimumRule *minRule, ConstraintEntryPtr entry, VarPtr var) {
  AnnotatedEntity entryInfo = entry->subcellInfo();
  DofOrderingPtr trialOrdering = minRule->elementType(entryInfo.cellID)->trialOrderPtr;

  if (entryInfo.sideOrdinal != -1) {
    return trialOrdering->getBasis(var->ID(), entryInfo.sideOrdinal);
  } else {
    return trialOrdering->getBasis(var->ID());
  }
}

GDAMinimumRuleConstraintEntry::GDAMinimumRuleConstraintEntry(GDAMinimumRuleConstraints &definedConstraints, GDAMinimumRule* minRule, CellPtr cell, int sideOrdinal, int subcellDim, int subcellOrdinalInSide) {
  _minRule = minRule;
  _cell = cell;
  _sideOrdinal = sideOrdinal;
  _entityDim = subcellDim;
  _subcellOrdinalInSide = subcellOrdinalInSide;
  
  int sideDim = _cell->topology()->getDimension() - 1;
  
  _subcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(*_cell->topology(), sideDim, sideOrdinal,
                                                               subcellDim, subcellOrdinalInSide);
  
  _entityIndex = _cell->entityIndex(subcellDim, _subcellOrdinalInCell);
  
  CellConstraints cellConstraints = _minRule->getCellConstraints(_cell->cellIndex());
  
  AnnotatedEntity constrainingSubcellInfo = cellConstraints.subcellConstraints[_entityDim][_subcellOrdinalInCell];
  
  AnnotatedEntity thisInfo;
  thisInfo.cellID = _cell->cellIndex();
  thisInfo.sideOrdinal = _sideOrdinal;
  thisInfo.dimension = _entityDim;
  if (_sideOrdinal == -1) {
    thisInfo.subcellOrdinal = _subcellOrdinalInCell;
  } else {
    thisInfo.subcellOrdinal = _subcellOrdinalInSide;
  }
  
  ConstraintEntryPtr thisPtr = definedConstraints.getConstraintEntry(thisInfo);
  
  // define "edges" going to subcells of dimension one less
  int subsubcellDim = subcellDim - 1;
  if (subsubcellDim >= 0) {
    shards::CellTopology domainTopo;
    if (sideOrdinal==-1) {
      domainTopo = *_cell->topology();
    } else {
      domainTopo = _cell->topology()->getCellTopologyData(sideDim, sideOrdinal);
    }
    
    shards::CellTopology subcellTopo = domainTopo.getCellTopologyData(subcellDim, _subcellOrdinalInSide);
    int subsubcellCount = subcellTopo.getSubcellCount(subsubcellDim);
    for (int ssord = 0; ssord<subsubcellCount; ssord++) { // ssord in *subcell*
      int ssordInDomain = CamelliaCellTools::subcellOrdinalMap(domainTopo, _entityDim, thisInfo.subcellOrdinal,
                                                               subsubcellDim, ssord);
      AnnotatedEntity subsubcellInfo;
      subsubcellInfo.dimension = subsubcellDim;
      subsubcellInfo.sideOrdinal = sideOrdinal;
      subsubcellInfo.cellID = cell->cellIndex();
      subsubcellInfo.subcellOrdinal = ssordInDomain;
      
      ConstraintEntryPtr subsubcellEntry;
      if (definedConstraints.constraintEntryExists(subsubcellInfo)) {
        subsubcellEntry = definedConstraints.getConstraintEntry(subsubcellInfo);
      } else {
        subsubcellEntry = Teuchos::rcp( new GDAMinimumRuleConstraintEntry(definedConstraints, minRule, cell, sideOrdinal, subsubcellDim, ssordInDomain) ) ;
        definedConstraints.setConstraintEntry(subsubcellInfo, subsubcellEntry);
      }
      
      subsubcellEntry->addPriorEntry(thisPtr);
      
      _subcellEntries.push_back( subsubcellEntry );
    }
  }
  if (thisInfo != constrainingSubcellInfo) { // not self-constrained
    if (definedConstraints.constraintEntryExists(constrainingSubcellInfo)) {
      _constrainingEntity = definedConstraints.getConstraintEntry(constrainingSubcellInfo);
    } else {
      // the constraining subcell is distinct: edge goes to the constraining subcell
      CellPtr constrainingCell = cell->meshTopology()->getCell(constrainingSubcellInfo.cellID);
      _constrainingEntity = Teuchos::rcp( new GDAMinimumRuleConstraintEntry(definedConstraints, minRule, constrainingCell, constrainingSubcellInfo.sideOrdinal,
                                                                            constrainingSubcellInfo.dimension, constrainingSubcellInfo.subcellOrdinal) );
      definedConstraints.setConstraintEntry(constrainingSubcellInfo, _constrainingEntity);
      
      _ancestralCell = cell->ancestralCellForSubcell(_entityDim, _subcellOrdinalInCell);
      _volumeRefinements = cell->refinementBranchForSubcell(_entityDim, _subcellOrdinalInCell);
      pair<unsigned, unsigned> ancestralSubcell = cell->ancestralSubcellOrdinalAndDimension(_entityDim, _subcellOrdinalInCell);
      _ancestralSubcellOrdinal = ancestralSubcell.first;
      _ancestralSubcellDimension = ancestralSubcell.second;
      
      determineAncestralSideOrdinal(constrainingSubcellInfo);
      
      unsigned ancestralSubcellOrdinalInSide = CamelliaCellTools::subcellReverseOrdinalMap(*_ancestralCell->topology(), sideDim, _ancestralSideOrdinal, constrainingSubcellInfo.dimension, _ancestralSubcellOrdinal);
      
      unsigned ancestralPermutation = _ancestralCell->sideSubcellPermutation(_ancestralSideOrdinal, _ancestralSubcellDimension, ancestralSubcellOrdinalInSide); // subcell permutation as seen from the perspective of the fine cell's side's ancestor
      unsigned constrainingPermutation = constrainingCell->sideSubcellPermutation(constrainingSubcellInfo.sideOrdinal, constrainingSubcellInfo.dimension,
                                                                                  constrainingSubcellInfo.subcellOrdinal); // subcell permutation as seen from the perspective of the domain on the constraining cell
      
      shards::CellTopology constrainingTopo = constrainingCell->topology()->getCellTopologyData(constrainingSubcellInfo.dimension, _constrainingEntity->subcellOrdinalInCell());
      unsigned constrainingPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, constrainingPermutation);
      _composedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, constrainingPermutationInverse, ancestralPermutation);
    }
    _constrainingEntity->addPriorEntry(thisPtr);
  }
}

void GDAMinimumRuleConstraintEntry::addPriorEntry(ConstraintEntryPtr priorEntry) {
  _priorEntries[priorEntry->subcellInfo()] = priorEntry;
}

unsigned GDAMinimumRuleConstraintEntry::ancestralSideOrdinal() {
  return _ancestralSideOrdinal;
}

unsigned GDAMinimumRuleConstraintEntry::ancestralSubcellDimension() {
  return _ancestralSubcellDimension;
}

unsigned GDAMinimumRuleConstraintEntry::ancestralSubcellOrdinal() {
  return _ancestralSubcellOrdinal;
}

CellPtr GDAMinimumRuleConstraintEntry::cell() {
  return _cell;
}

unsigned GDAMinimumRuleConstraintEntry::composedPermutation() {
  return _composedPermutation;
}

void GDAMinimumRuleConstraintEntry::determineAncestralSideOrdinal(AnnotatedEntity &constrainingSubcellInfo) {
  int sideDim = _cell->topology()->getDimension() - 1;
  
  if (_ancestralSubcellDimension == sideDim) {
    _ancestralSideOrdinal = _ancestralSubcellOrdinal;
  } else {
    
    IndexType ancestralSubcellEntityIndex = _ancestralCell->entityIndex(_ancestralSubcellDimension, _ancestralSubcellOrdinal);
    
    // for subcells constrained by subcells of unlike dimension, we can handle any side that contains the ancestral subcell,
    // but for like-dimensional constraints, we do need the ancestralSideOrdinal to be the ancestor of the side in subcellInfo...
    
    if (constrainingSubcellInfo.dimension == _entityDim) {
      IndexType descendantSideEntityIndex = _cell->entityIndex(sideDim, _sideOrdinal);
      
      _ancestralSideOrdinal = -1;
      int sideCount = CamelliaCellTools::getSideCount(*_ancestralCell->topology());
      for (int side=0; side<sideCount; side++) {
        IndexType ancestralSideEntityIndex = _ancestralCell->entityIndex(sideDim, side);
        if (ancestralSideEntityIndex == descendantSideEntityIndex) {
          _ancestralSideOrdinal = side;
          break;
        }
        
        if (_cell->meshTopology()->entityIsAncestor(sideDim, ancestralSideEntityIndex, descendantSideEntityIndex)) {
          _ancestralSideOrdinal = side;
          break;
        }
      }
      
      if (_ancestralSideOrdinal == -1) {
        cout << "Error: no ancestor of side found.\n";
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: no ancestor of side contains the ancestral subcell.");
      }
    } else {
      // find some side in the ancestral cell that contains the ancestral subcell, then (there should be at least two; which one shouldn't matter)
      set<IndexType> sidesForSubcell = _cell->meshTopology()->getSidesContainingEntity(_ancestralSubcellDimension, ancestralSubcellEntityIndex);
      
      _ancestralSideOrdinal = -1;
      int sideCount = CamelliaCellTools::getSideCount(*_ancestralCell->topology());
      for (int side=0; side<sideCount; side++) {
        IndexType ancestralSideEntityIndex = _ancestralCell->entityIndex(sideDim, side);
        if (sidesForSubcell.find(ancestralSideEntityIndex) != sidesForSubcell.end()) {
          _ancestralSideOrdinal = side;
          break;
        }
      }
    }
  }
  
  if (_ancestralSideOrdinal == -1) {
    cout << "Error: ancestralSideOrdinal not found.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: ancestralSideOrdinal not found.");
  }
}

int GDAMinimumRuleConstraintEntry::entityDim() {
  return _entityDim;
}

GlobalIndexType GDAMinimumRuleConstraintEntry::entityIndex() {
  return _entityIndex;
}

vector< ConstraintEntryPtr > GDAMinimumRuleConstraintEntry::getPriorEntries() {
  vector< ConstraintEntryPtr > entries;
  for (map<AnnotatedEntity, ConstraintEntryPtr>::iterator mapIt = _priorEntries.begin(); mapIt != _priorEntries.end(); mapIt++) {
    entries.push_back(mapIt->second);
  }
  return entries;
}

bool GDAMinimumRuleConstraintEntry::isUnconstrained() {
  return _constrainingEntity.get() == NULL;
}

vector< ConstraintEntryPtr > GDAMinimumRuleConstraintEntry::nextEntries() { // "out edges"
  if (isUnconstrained()) {
    return _subcellEntries;
  } else {
    vector< ConstraintEntryPtr > nextEntries = _subcellEntries;
    nextEntries.push_back(_constrainingEntity);
    
    return nextEntries;
  }
}

AnnotatedEntity GDAMinimumRuleConstraintEntry::subcellInfo() {
  AnnotatedEntity scInfo;
  scInfo.dimension = _entityDim;
  scInfo.cellID = _cell->cellIndex();
  scInfo.sideOrdinal = _sideOrdinal;
  if (_sideOrdinal != -1) {
    scInfo.subcellOrdinal = _subcellOrdinalInSide;
  } else {
    scInfo.subcellOrdinal = _subcellOrdinalInCell;
  }

  return scInfo;
}

int GDAMinimumRuleConstraintEntry::subcellOrdinalInCell() {
  return _subcellOrdinalInCell;
}

int GDAMinimumRuleConstraintEntry::subcellOrdinalInSide() {
  return _subcellOrdinalInSide;
}

RefinementBranch GDAMinimumRuleConstraintEntry::volumeRefinements() {
  return _volumeRefinements;
}