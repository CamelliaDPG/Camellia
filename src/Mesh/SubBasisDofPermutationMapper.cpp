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

SubBasisDofPermutationMapper::SubBasisDofPermutationMapper(const set<unsigned> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals) {
  _basisDofOrdinalFilter = basisDofOrdinalFilter;
  _globalDofOrdinals = globalDofOrdinals;
}

const set<unsigned> & SubBasisDofPermutationMapper::basisDofOrdinalFilter() {
  return _basisDofOrdinalFilter;
}
FieldContainer<double> SubBasisDofPermutationMapper::mapData(bool transposeConstraintMatrix, FieldContainer<double> &data) {
  // not too sure about the right way to handle the transposes here -- absent them:
  //     for the permutation mapper, data need not change; the permutation is implicit in the globalDofOrdinals container
  // but with them, I'm a bit vague...  For now, I'm disabling the PermutationMapper in favor of the matrix-multiplying guy...


//  if (!transposeData) {
//    if (!transposeConstraintMatrix) {
//      return data;
//    } else {
//      // transposing the constraint matrix is equivalent to inverting the permutation given by the globalDofOrdinals container.
//      // (note that when the constraint matrix is transposed, caller will be using basisDofOrdinalFilter to interpret the result of this method.)
//      return applyInversePermutation(data);
//    }
//  } else { // DO transpose data
//    FieldContainer<double> dataCopy(data.dimension(1),data.dimension(0));
//    for (int i=0; i<data.dimension(0); i++) {
//      for (int j=0; j<data.dimension(1); j++) {
//        dataCopy(j,i) = data(i,j);
//      }
//    }
//    if (!transposeConstraintMatrix) {
//      return dataCopy;
//    } else {
//      // transposing the constraint matrix is equivalent to inverting the permutation given by the globalDofOrdinals container.
//      // (note that when the constraint matrix is transposed, caller will be using basisDofOrdinalFilter to interpret the result of this method.)
//      return applyInversePermutation(dataCopy);
//    }
//  }
  
  if (!transposeConstraintMatrix) {
    cout << "WARNING: I'm not real sure about transposeConstraintMatrix=false in SubBasisDofPermutationMapper::mapData().\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "transposeConstraintMatrix = false in SubBasisDofPermutationMapper::mapData() not yet handled.");
  }
  if ( ! transposeConstraintMatrix ) { // transpose data???
    FieldContainer<double> dataCopy(data.dimension(1),data.dimension(0));
    for (int i=0; i<data.dimension(0); i++) {
      for (int j=0; j<data.dimension(1); j++) {
        dataCopy(j,i) = data(i,j);
      }
    }
    return dataCopy;
  } else {
    return data;
  }
}
vector<GlobalIndexType> SubBasisDofPermutationMapper::mappedGlobalDofOrdinals() {
  return _globalDofOrdinals;
}
