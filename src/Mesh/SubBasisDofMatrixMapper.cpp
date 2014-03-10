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
  
  // TODO: check that input sizes are reasonable...
  
  // int constraintCols = _constraintMatrix.dimension(1);
}
const set<unsigned> & SubBasisDofMatrixMapper::basisDofOrdinalFilter() {
  return _basisDofOrdinalFilter;
}
FieldContainer<double> SubBasisDofMatrixMapper::mapData(bool transposeConstraint, FieldContainer<double> &localData) {
  // localData must be rank 2, and must have the same size as FilteredLocalDofOrdinals in its first dimension
  bool didReshape = false;
  if (localData.rank() == 1) {
    // reshape as a rank 2 container (column vector as a matrix):
    localData.resize(localData.dimension(0),1);
    didReshape = true;
  }
  if (localData.rank() != 2) {
    cout << "localData must have rank 1 or 2.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must have rank 1 or 2");
  }
  int constraintRows = transposeConstraint ? _constraintMatrix.dimension(1) : _constraintMatrix.dimension(0);
  int constraintCols = transposeConstraint ? _constraintMatrix.dimension(0) : _constraintMatrix.dimension(1);
  int dataCols = localData.dimension(1);
  int dataRows = localData.dimension(0);
  
  
  if ((dataCols==0) || (dataRows==0) || (constraintRows==0) || (constraintCols==0)) {
    cout << "degenerate matrix encountered.\n";
  }
  
  // given the multiplication we'll do, we need constraint columns = data rows
  if (constraintCols != dataRows) {
    cout << "Missized container in SubBasisDofMatrixMapper::mapData() for left-multiplication by constraint matrix.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missized container in SubBasisDofMatrixMapper::mapData().");
  }
  // (could also test that the dimensions match what we expect in terms of the size of the mapped global dof ordinals or basisDofOrdinal filter)
  
  FieldContainer<double> result1(constraintRows,dataCols);
  
  char constraintTransposeFlag = transposeConstraint ? 'T' : 'N';
  char dataTransposeFlag = 'N';
  
  SerialDenseWrapper::multiply(result1,_constraintMatrix,localData,constraintTransposeFlag,dataTransposeFlag);
  
  if (didReshape) { // change the shape of localData back, and return result
    localData.resize(localData.dimension(0));
    result1.resize(result1.size());
    return result1;
  }
  
  if (constraintCols != dataCols) {
    cout << "Missized container in SubBasisDofMatrixMapper::mapData() for right-multiplication by constraint matrix.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Missized container in SubBasisDofMatrixMapper::mapData().");
  }
  
  constraintTransposeFlag = (!transposeConstraint) ? 'T' : 'N'; // opposite of the above choice, since now we multiply on the right
  char resultTransposeFlag = 'N';
  
  FieldContainer<double> result(constraintRows,constraintRows);
  SerialDenseWrapper::multiply(result,result1,_constraintMatrix,resultTransposeFlag,constraintTransposeFlag);
  
  return result;
}
vector<GlobalIndexType> SubBasisDofMatrixMapper::mappedGlobalDofOrdinals() {
  return _mappedGlobalDofOrdinals;
}