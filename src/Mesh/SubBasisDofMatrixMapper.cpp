//
//  SubBasisDofMatrixMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#include "SubBasisDofMatrixMapper.h"

#include "SerialDenseWrapper.h"

SubBasisDofMatrixMapper::SubBasisDofMatrixMapper(const set<unsigned> &basisDofOrdinalFilter, const vector<GlobalIndexType> &mappedGlobalDofOrdinals, const FieldContainer<double> &constraintMatrix) {
  _basisDofOrdinalFilter = basisDofOrdinalFilter;
  _mappedGlobalDofOrdinals = mappedGlobalDofOrdinals;
  _constraintMatrix = constraintMatrix;
}
const set<unsigned> & SubBasisDofMatrixMapper::basisDofOrdinalFilter() {
  return _basisDofOrdinalFilter;
}
FieldContainer<double> SubBasisDofMatrixMapper::mapData(const FieldContainer<double> &localData, bool transpose) {
  int localDofOrdinalCount = _basisDofOrdinalFilter.size();
  
  // localData must be rank 2, and must have the same size as FilteredLocalDofOrdinals in its first dimension
  if ((localData.rank() != 2)) {
    cout << "localData must have rank 2.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must have rank 2");
  }
  if ((!transpose && (localData.dimension(0) != localDofOrdinalCount))
      || (transpose && (localData.dimension(1) != localDofOrdinalCount))) {
    cout << "localData dimension to be transformed must match the localDofOrdinalCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData dimension to be transformed must match the localDofOrdinalCount");
  }
  
  FieldContainer<double> result;
  
  if (!transpose) {
    result.resize(_constraintMatrix.dimension(0),localData.dimension(1));
    SerialDenseWrapper::multiply(result, _constraintMatrix, localData);
  } else {
    result.resize(_constraintMatrix.dimension(0),localData.dimension(0));
    SerialDenseWrapper::multiply(result, _constraintMatrix, localData,'N','T');
  }
  return result;
}
vector<GlobalIndexType> SubBasisDofMatrixMapper::mappedGlobalDofOrdinals() {
  return _mappedGlobalDofOrdinals;
}