//
//  SubBasisDofPermutationMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/14/14.
//
//

#include "SubBasisDofPermutationMapper.h"

#include <set.h>
using namespace std;

SubBasisDofPermutationMapper::SubBasisDofPermutationMapper(const set<unsigned> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals) {
  _basisDofOrdinalFilter = basisDofOrdinalFilter;
  _globalDofOrdinals = globalDofOrdinals;
}
const set<unsigned> & SubBasisDofPermutationMapper::basisDofOrdinalFilter() {
  return _basisDofOrdinalFilter;
}
FieldContainer<double> SubBasisDofPermutationMapper::mapData(const FieldContainer<double> &localData, bool transpose) {
  // for the permutation mapper, localData need not change; the permutation is implicit in the globalDofOrdinals container
  return localData;
}
vector<GlobalIndexType> SubBasisDofPermutationMapper::mappedGlobalDofOrdinals() {
  return _globalDofOrdinals;
}