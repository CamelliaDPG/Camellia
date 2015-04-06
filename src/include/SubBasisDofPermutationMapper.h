//
//  SubBasisPermutationDofMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/14/14.
//
//

#ifndef Camellia_debug_SubBasisPermutationDofMapper_h
#define Camellia_debug_SubBasisPermutationDofMapper_h

#include "SubBasisDofMapper.h"

namespace Camellia {
  class SubBasisDofPermutationMapper : public SubBasisDofMapper {
    set<unsigned> _basisDofOrdinalFilter;
    vector<GlobalIndexType> _globalDofOrdinals;
    vector<int> _inversePermutation;
    bool _negate;
  public:
    SubBasisDofPermutationMapper(const set<unsigned> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals, bool negate=false);
    const set<unsigned> &basisDofOrdinalFilter();
    Intrepid::FieldContainer<double> mapData(bool transposeConstraint, Intrepid::FieldContainer<double> &localData, bool applyOnLeftOnly = false);
    void mapDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                    bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Intrepid::FieldContainer<double> &globalData);
    
    vector<GlobalIndexType> mappedGlobalDofOrdinals();
    Intrepid::FieldContainer<double> getConstraintMatrix();
    
    SubBasisDofMapperPtr negatedDofMapper(); // this mapper, but negated.
  };
}


#endif
