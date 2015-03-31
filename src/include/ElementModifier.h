//
//  ElementModifier.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 11/26/13.
//
//

#ifndef Camellia_debug_ElementModifier_h
#define Camellia_debug_ElementModifier_h

#include "BasisReconciliation.h"

class ElementModifier {
public:
  virtual const vector<unsigned> &modifiedGlobalDofIndices();
  virtual void getModifiedValues(Intrepid::FieldContainer<double> &modifiedValues, Intrepid::FieldContainer<int> &dofIndicesForModifiedValues, const Intrepid::FieldContainer<double> &localValues);
};

class DofIdentificationModifier : ElementModifier {
  vector<unsigned> _globalDofIndices;
public:
  DofIdentificationModifier(vector<unsigned> &globalDofIndices) {
    _globalDofIndices = globalDofIndices;
  }
  const vector<unsigned> &modifiedGlobalDofIndices() {
    return _globalDofIndices;
  }
  void getModifiedValues(Intrepid::FieldContainer<double> &modifiedValues, Intrepid::FieldContainer<int> &dofIndicesForModifiedValues, const Intrepid::FieldContainer<double> &localValues) {
    // two possibilities: shaped like stiffness (localDofs, localDofs) or shaped like load (localDofs)
    for (unsigned dofOrdinal=0; dofOrdinal < _globalDofIndices.size(); dofOrdinal++) {
      dofIndicesForModifiedValues[dofOrdinal] = _globalDofIndices[dofOrdinal];
    }
    // for this case, we don't worry about the shape...
    unsigned numEntries = localValues.size();
    for (unsigned i=0; i<numEntries; i++) {
      modifiedValues[i] = localValues[i];
    }
  }
};

class ConstrainedElementModifier : ElementModifier {
  vector<unsigned> _globalDofIndices;
  map<unsigned, unsigned> _localDofOrdinalToSubBasisWeightIndex;
  vector< SubBasisReconciliationWeights* > _weightedSubBases;
  map<unsigned,unsigned> _permutedSubBasis;
  unsigned _localDofCount;
public:
  ConstrainedElementModifier(vector< SubBasisReconciliationWeights* > &weightedSubBases, map<unsigned,unsigned> &permutedSubBasis) {
    _weightedSubBases = weightedSubBases;
    _permutedSubBasis = permutedSubBasis;
    
    unsigned weightsCount = weightedSubBases.size();
    // requirement: each local dof appears exactly once in weightedSubBases or permutedSubBasis
    for (unsigned weightsOrdinal = 0; weightsOrdinal < weightsCount; weightsOrdinal++) {
      // "fine ordinals" are the local ones
      set<int>* fineOrdinals = &(weightedSubBases[weightsOrdinal]->fineOrdinals);
      for (set<int>::iterator localOrdinalIt = fineOrdinals->begin(); localOrdinalIt != fineOrdinals->end(); localOrdinalIt++) {
        _localDofOrdinalToSubBasisWeightIndex[*localOrdinalIt] = weightsOrdinal;
      }
    }
    
    // if we knew how big the local basis is supposed to be, we could check that the total entry count matches this.
    // but I don't think we have this -- we settle for storing the total entry count, so that we can check that the inputs to getModifiedValues() match
    _localDofCount = _localDofOrdinalToSubBasisWeightIndex.size() + permutedSubBasis.size();
  }
  const vector<unsigned> &modifiedGlobalDofIndices() {
    return _globalDofIndices;
  }
  void getModifiedValues(Intrepid::FieldContainer<double> &modifiedValues, Intrepid::FieldContainer<int> &dofIndicesForModifiedValues, const Intrepid::FieldContainer<double> &localValues) {
    // two possibilities: shaped like stiffness (localDofs, localDofs) or shaped like load (localDofs)
    for (unsigned dofOrdinal=0; dofOrdinal < _globalDofIndices.size(); dofOrdinal++) {
      dofIndicesForModifiedValues[dofOrdinal] = _globalDofIndices[dofOrdinal];
    }

    // TODO: finish this method
  }
};

#endif
