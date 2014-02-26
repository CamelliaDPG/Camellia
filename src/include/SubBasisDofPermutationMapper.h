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
public:
  SubBasisDofPermutationMapper(const set<unsigned> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals);
  const set<unsigned> &basisDofOrdinalFilter();
  FieldContainer<double> mapData(bool transposeConstraint, FieldContainer<double> &localData);
  vector<GlobalIndexType> mappedGlobalDofOrdinals();
};


#endif
