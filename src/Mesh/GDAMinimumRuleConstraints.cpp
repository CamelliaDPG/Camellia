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

#include <iostream>
using namespace std;

namespace Camellia {
  bool GDAMinimumRuleConstraints::constraintEntryExists(AnnotatedEntity &subcellInfo) {
    if (_constraintEntries.find(subcellInfo) != _constraintEntries.end()) {
      return _constraintEntries[subcellInfo].get() != NULL;
    } else {
      return false;
    }
  }

  void GDAMinimumRuleConstraints::computeConstraintWeightsRecursive(map<AnnotatedEntity, SubBasisReconciliationWeights> & constraintWeights,
      GDAMinimumRule *minRule, ConstraintEntryPtr entry, VarPtr var) {
    vector< ConstraintEntryPtr > priorEntries = entry->getPriorEntries();

    AnnotatedEntity thisSubcellInfo = entry->subcellInfo();

    if (constraintWeights.find(thisSubcellInfo) == constraintWeights.end()) {
      SubBasisReconciliationWeights summedWeights;

      for (int i=0; i<priorEntries.size(); i++) {
        if (constraintWeights.find(priorEntries[i]->subcellInfo()) == constraintWeights.end()) {
          // then there is another path to this node in the graph, which we haven't yet traversed.
          // therefore, we should bail now..
          return;
        }
      }

      for (int i=0; i<priorEntries.size(); i++) {
        ConstraintEntryPtr priorEntry = priorEntries[i];
        AnnotatedEntity priorSubcellInfo = priorEntry->subcellInfo();
        SubBasisReconciliationWeights priorWeights = constraintWeights[priorSubcellInfo];

        // determine whether the edge on the directed acyclic graph is red or black
        // red:   target constrains source
        // black: target is a subcell of source
        ConstraintEntryPtr priorEntryConstrainingEntity = priorEntry->getConstrainingEntry();
        bool redEdge = (priorEntryConstrainingEntity.get() != NULL) && (priorEntryConstrainingEntity->subcellInfo() == thisSubcellInfo);

        DofOrderingPtr priorTrialOrdering = minRule->elementType(priorSubcellInfo.cellID)->trialOrderPtr;
        BasisPtr prevBasis = priorTrialOrdering->getBasis(var->ID(), priorSubcellInfo.sideOrdinal);

        SubBasisReconciliationWeights composedWeights;
        if (!redEdge) {
          // then the edge just filters the prior weights corresponding to the subcell
          // thisSubcellInfo.subcellOrdinal refers to the ordinal in the *domain*, whether that be the side or the cell:
          composedWeights = BasisReconciliation::weightsForCoarseSubcell(priorWeights, prevBasis, thisSubcellInfo.dimension, thisSubcellInfo.subcellOrdinal, true);
          cout << "computeConstraintWeightsRecursive: black (subcell) edge encountered from " <<  priorSubcellInfo << " to " << thisSubcellInfo << endl;
        } else {
          cout << "computeConstraintWeightsRecursive: red (constraining) edge encountered from " <<  priorSubcellInfo << " to " << thisSubcellInfo << endl;

          DofOrderingPtr constrainingTrialOrdering = minRule->elementType(thisSubcellInfo.cellID)->trialOrderPtr;
          BasisPtr constrainingBasis = constrainingTrialOrdering->getBasis(var->ID(), thisSubcellInfo.sideOrdinal);
          RefinementBranch volumeRefinements = priorEntry->volumeRefinements();
          unsigned composedPermutation = priorEntry->composedPermutation();
          CellTopoPtr thisCellTopology = minRule->elementType(thisSubcellInfo.cellID)->cellTopoPtr;
          
          // TODO: work out what to do here for volume basis--in particular, what happens to the sideOrdinal arguments??  It seems like these should not apply... (they should be redundant with subcell ordinal)  It may well be that what we should do is ensure that the sideOrdinal arguments are -1, and then BR should understand that as a flag indicating that volumeRefinements really does apply to the ancestral domain, etc.
          SubBasisReconciliationWeights newWeights = BasisReconciliation::computeConstrainedWeights(priorSubcellInfo.dimension, prevBasis, priorSubcellInfo.subcellOrdinal,
                                                                                                    volumeRefinements, priorSubcellInfo.sideOrdinal,
                                                                                                    thisCellTopology, thisSubcellInfo.dimension,
                                                                                                    constrainingBasis, thisSubcellInfo.subcellOrdinal,
                                                                                                    priorEntry->ancestralSideOrdinal(), composedPermutation);
          composedWeights = BasisReconciliation::composedSubBasisReconciliationWeights(priorWeights, newWeights);
        }
        summedWeights = BasisReconciliation::sumWeights(summedWeights,composedWeights);
      }

      summedWeights = BasisReconciliation::filterOutZeroRowsAndColumns(summedWeights);

      constraintWeights[thisSubcellInfo] = summedWeights;
      cout << "Set constraint weights for " << thisSubcellInfo << endl;
    }

    // follow all the out edges:
    vector< ConstraintEntryPtr > nextEntries = entry->nextEntries();
    for (vector< ConstraintEntryPtr >::iterator entryIt = nextEntries.begin(); entryIt != nextEntries.end(); entryIt++) {
      computeConstraintWeightsRecursive(constraintWeights, minRule, *entryIt, var);
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

    computeConstraintWeightsRecursive(constraintWeights, minRule, rootEntry, var);
  }

  ConstraintEntryPtr GDAMinimumRuleConstraints::getConstraintEntry(AnnotatedEntity &subcellInfo) {
    if (_constraintEntries.find(subcellInfo) == _constraintEntries.end()) {
      cout << "WARNING: constraint entry for " << subcellInfo << " not found.\n";
    }
    return _constraintEntries[subcellInfo];
  }

  void GDAMinimumRuleConstraints::setConstraintEntry(AnnotatedEntity &subcellInfo, ConstraintEntryPtr entry) {
    cout << "Set constraint entry for " << subcellInfo << endl;
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

    _subcellOrdinalInCell = CamelliaCellTools::subcellOrdinalMap(_cell->topology(), sideDim, sideOrdinal,
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

    ConstraintEntryPtr thisPtr;
    if (definedConstraints.constraintEntryExists(thisInfo)) {
      thisPtr = definedConstraints.getConstraintEntry(thisInfo);
    } else {
      thisPtr = Teuchos::rcp( this, false );
    }

    if (thisInfo == constrainingSubcellInfo) { // unconstrained / self-constrained
      // define black edges: these go to subcells of dimension one less
      int subsubcellDim = subcellDim - 1;
      if (subsubcellDim >= 0) {
        CellTopoPtr domainTopo;
        if (sideOrdinal==-1) {
          domainTopo = _cell->topology();
        } else {
          domainTopo = _cell->topology()->getSubcell(sideDim, sideOrdinal);
        }

        CellTopoPtr subcellTopo = domainTopo->getSubcell(subcellDim, _subcellOrdinalInSide);
        int subsubcellCount = subcellTopo->getSubcellCount(subsubcellDim);
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

          cout << "Adding black (subcell) edge from " << thisInfo << " to " << subsubcellInfo << endl;

          _subcellEntries.push_back( subsubcellEntry );
        }
      }
    } else { // constrained by a distinct node in the graph (draw red edges)
      CellPtr constrainingCell = cell->meshTopology()->getCell(constrainingSubcellInfo.cellID);
      if (definedConstraints.constraintEntryExists(constrainingSubcellInfo)) {
        _constrainingEntity = definedConstraints.getConstraintEntry(constrainingSubcellInfo);
      } else {
        // the constraining subcell is distinct: edge goes to the constraining subcell
        _constrainingEntity = Teuchos::rcp( new GDAMinimumRuleConstraintEntry(definedConstraints, minRule, constrainingCell, constrainingSubcellInfo.sideOrdinal,
              constrainingSubcellInfo.dimension, constrainingSubcellInfo.subcellOrdinal) );
        definedConstraints.setConstraintEntry(constrainingSubcellInfo, _constrainingEntity);
      }

      _ancestralCell = cell->ancestralCellForSubcell(_entityDim, _subcellOrdinalInCell);
      _volumeRefinements = cell->refinementBranchForSubcell(_entityDim, _subcellOrdinalInCell);
      if (_volumeRefinements.size()==0) {
        // a trick to sneak in the cell topology information that BasisReconciliation will require:
        RefinementPatternPtr noRefinement = RefinementPattern::noRefinementPattern(cell->topology());
        _volumeRefinements.push_back(make_pair(noRefinement.get(), 0));
      }

      pair<unsigned, unsigned> ancestralSubcell = cell->ancestralSubcellOrdinalAndDimension(_entityDim, _subcellOrdinalInCell);
      _ancestralSubcellOrdinal = ancestralSubcell.first;
      _ancestralSubcellDimension = ancestralSubcell.second;

      determineAncestralSideOrdinal(constrainingSubcellInfo);

      unsigned ancestralSubcellOrdinalInSide = CamelliaCellTools::subcellReverseOrdinalMap(_ancestralCell->topology(), sideDim, _ancestralSideOrdinal, constrainingSubcellInfo.dimension, _ancestralSubcellOrdinal);

      cout << "WARNING: using *side* subcell permutations to determine composed permutation for BasisReconciliation.  ";
      cout << "This is only valid for the side-centric computeConstrainedWeights; need to make a distinction between the arguments here (in GDAMinimumRuleConstraints).\n";
      
      unsigned ancestralPermutation = _ancestralCell->sideSubcellPermutation(_ancestralSideOrdinal, _ancestralSubcellDimension, ancestralSubcellOrdinalInSide); // subcell permutation as seen from the perspective of the fine cell's side's ancestor
      unsigned constrainingPermutation = constrainingCell->sideSubcellPermutation(constrainingSubcellInfo.sideOrdinal, constrainingSubcellInfo.dimension,
          constrainingSubcellInfo.subcellOrdinal); // subcell permutation as seen from the perspective of the domain on the constraining cell

      CellTopoPtr constrainingTopo = constrainingCell->topology()->getSubcell(constrainingSubcellInfo.dimension, _constrainingEntity->subcellOrdinalInCell());
      unsigned ancestralPermutationInverse = CamelliaCellTools::permutationInverse(constrainingTopo, ancestralPermutation);
      _composedPermutation = CamelliaCellTools::permutationComposition(constrainingTopo, ancestralPermutationInverse, constrainingPermutation);

      cout << "Adding red (constraining) edge from " << thisInfo << " to " << constrainingSubcellInfo << endl;

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
        int sideCount = _ancestralCell->topology()->getSideCount();
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
        vector<IndexType> sidesForSubcell = _cell->meshTopology()->getSidesContainingEntity(_ancestralSubcellDimension, ancestralSubcellEntityIndex);

        _ancestralSideOrdinal = -1;
        int sideCount = _ancestralCell->topology()->getSideCount();
        for (int side=0; side<sideCount; side++) {
          IndexType ancestralSideEntityIndex = _ancestralCell->entityIndex(sideDim, side);
          if (std::find(sidesForSubcell.begin(), sidesForSubcell.end(), ancestralSideEntityIndex) != sidesForSubcell.end()) {
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

  ConstraintEntryPtr GDAMinimumRuleConstraintEntry::getConstrainingEntry() {
    return _constrainingEntity;
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

  std::ostream& operator << (std::ostream& os, AnnotatedEntity& annotatedEntity) {
    os << "cell " << annotatedEntity.cellID;
    if (annotatedEntity.sideOrdinal != -1) {
      os << "'s side " << annotatedEntity.sideOrdinal;
    } 
    os << "'s " << CamelliaCellTools::entityTypeString(annotatedEntity.dimension);
    os << " " << annotatedEntity.subcellOrdinal;

    return os;
  }
}
