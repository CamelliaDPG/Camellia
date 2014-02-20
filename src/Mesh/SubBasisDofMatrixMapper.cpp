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
FieldContainer<double> SubBasisDofMatrixMapper::mapData(bool transposeConstraint, const FieldContainer<double> &localData, bool transposeData) {
  // localData must be rank 2, and must have the same size as FilteredLocalDofOrdinals in its first dimension
  if ((localData.rank() != 2)) {
    cout << "localData must have rank 2.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must have rank 2");
  }
  int constraintRows = transposeConstraint ? _constraintMatrix.dimension(1) : _constraintMatrix.dimension(0);
  int constraintCols = transposeConstraint ? _constraintMatrix.dimension(0) : _constraintMatrix.dimension(1);
  int dataCols = transposeData ? localData.dimension(0) : localData.dimension(1);
  int dataRows = transposeData ? localData.dimension(1) : localData.dimension(0);
  
  // given the multiplication we'll do, we need constraint columns = data rows
  if (constraintCols != dataRows) {
    cout << "Missized container in SubBasisDofMatrixMapper::mapData().\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missized container in SubBasisDofMatrixMapper::mapData().");
  }
  // (could also test that the dimensions match what we expect in terms of the size of the mapped global dof ordinals or basisDofOrdinal filter)
  
  FieldContainer<double> result(constraintRows,dataCols);
  
  char constraintTransposeFlag = transposeConstraint ? 'T' : 'N';
  char dataTransposeFlag = transposeData ? 'T' : 'N';
  
  SerialDenseWrapper::multiply(result,_constraintMatrix,localData,constraintTransposeFlag,dataTransposeFlag);
  
  return result;
}
vector<GlobalIndexType> SubBasisDofMatrixMapper::mappedGlobalDofOrdinals() {
  return _mappedGlobalDofOrdinals;
}