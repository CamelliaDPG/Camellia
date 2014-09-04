//
//  SubBasisDofPermutationMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/14/14.
//
//

#include "SubBasisDofPermutationMapper.h"

#include <set>
using namespace std;

SubBasisDofPermutationMapper::SubBasisDofPermutationMapper(const set<unsigned> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals,
                                                           bool negate) {
  _basisDofOrdinalFilter = basisDofOrdinalFilter;
  _globalDofOrdinals = globalDofOrdinals;
  _inversePermutation = vector<int>(_basisDofOrdinalFilter.size());
  map<GlobalIndexType,int> permutation_map;
  for (int i=0; i<_basisDofOrdinalFilter.size(); i++) {
    permutation_map[globalDofOrdinals[i]] = i;
  }
  int i=0;
  for (map<GlobalIndexType,int>::iterator permIt = permutation_map.begin(); permIt != permutation_map.end(); permIt++) {
    _inversePermutation[i++] = permIt->second;
  }
  _negate = negate;
}

const set<unsigned> & SubBasisDofPermutationMapper::basisDofOrdinalFilter() {
  return _basisDofOrdinalFilter;
}
FieldContainer<double> SubBasisDofPermutationMapper::mapData(bool transposeConstraintMatrix, FieldContainer<double> &data) {
  if (transposeConstraintMatrix) {
    // data comes in ordered by basisDofOrdinal
    // caller will interpret the data by virtue of the globalDofOrdinals vector--the permutation is implicit in that
    return data;
  } else {
    // data comes in ordered by GlobalDofOrdinal -- use inversePermutation to reorder
    // data should come out such that the ordering corresponds to that of the _basisDofOrdinalFilter
    Teuchos::Array<int> dim;
    data.dimensions(dim);
    FieldContainer<double> dataPermuted(dim);
    if (dim.size() == 1) {
      for (int i=0; i<dim[0]; i++) {
        if (!_negate)
          dataPermuted(_inversePermutation[i]) = data(i);
        else
          dataPermuted(_inversePermutation[i]) = -data(i);
      }
    } else if (dim.size() == 2) {
      for (int i=0; i<dim[0]; i++) {
        for (int j=0; j<dim[1]; j++) {
          if (!_negate)
            dataPermuted(_inversePermutation[i],_inversePermutation[j]) = data(i,j);
          else
            dataPermuted(_inversePermutation[i],_inversePermutation[j]) = -data(i,j);
        }
      }
    }
    return dataPermuted;
  }
}
void SubBasisDofPermutationMapper::mapDataIntoGlobalContainer(const FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned int> &globalIndexToOrdinal,
                                                              bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, FieldContainer<double> &globalData) {
  // like calling mapData, above, with transposeConstraintMatrix = true
  
  const set<unsigned>* basisOrdinalFilter = &this->basisDofOrdinalFilter();
  vector<unsigned> dofIndices(basisOrdinalFilter->begin(),basisOrdinalFilter->end());
  
  for (int sbGlobalOrdinal_i=0; sbGlobalOrdinal_i<_globalDofOrdinals.size(); sbGlobalOrdinal_i++) {
    GlobalIndexType globalIndex_i = _globalDofOrdinals[sbGlobalOrdinal_i];
    if (fittableDofsOnly && (fittableDofIndices.find(globalIndex_i) == fittableDofIndices.end())) continue; // skip this one
    unsigned globalOrdinal_i = globalIndexToOrdinal.find(globalIndex_i)->second;
    globalData[globalOrdinal_i] += wholeBasisData[dofIndices[sbGlobalOrdinal_i]];
  }
}

FieldContainer<double> SubBasisDofPermutationMapper::getConstraintMatrix() {
  // identity (permutation comes by virtue of ordering in globalDofOrdinals)
  FieldContainer<double> matrix(_basisDofOrdinalFilter.size(),_globalDofOrdinals.size());
  for (int i=0;i<_basisDofOrdinalFilter.size(); i++) {
    if (!_negate)
      matrix(i,i) = 1;
    else
      matrix(i,i) = -1;
  }
  return matrix;
}

vector<GlobalIndexType> SubBasisDofPermutationMapper::mappedGlobalDofOrdinals() {
  return _globalDofOrdinals;
}

SubBasisDofMapperPtr SubBasisDofPermutationMapper::negatedDofMapper() {
  return Teuchos::rcp( new SubBasisDofPermutationMapper(_basisDofOrdinalFilter, _globalDofOrdinals, !_negate) );
}