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

class SubBasisDofPermutationMapper : public SubBasisDofMapper {
  set<unsigned> _basisDofOrdinalFilter;
  vector<GlobalIndexType> _globalDofOrdinals;
  vector<int> _inversePermutation;
  bool _negate;
public:
  SubBasisDofPermutationMapper(const set<unsigned> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals, bool negate=false);
  const set<unsigned> &basisDofOrdinalFilter();
  FieldContainer<double> mapData(bool transposeConstraint, FieldContainer<double> &localData);
  void mapDataIntoGlobalContainer(const FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                  bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, FieldContainer<double> &globalData);
  
  vector<GlobalIndexType> mappedGlobalDofOrdinals();
  FieldContainer<double> getConstraintMatrix();
  
  SubBasisDofMapperPtr negatedDofMapper(); // this mapper, but negated.
};


#endif
