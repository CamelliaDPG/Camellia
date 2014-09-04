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
  
  // The constraint matrix should have size (fine,coarse) -- which is to say (local, global)
  if (_constraintMatrix.dimension(0) != basisDofOrdinalFilter.size()) {
    cout << "ERROR: constraint matrix row dimension must match the local sub-basis size.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constraint matrix row dimension must match the local sub-basis size");
  }
  if (_constraintMatrix.dimension(1) != mappedGlobalDofOrdinals.size()) {
    cout << "ERROR: constraint matrix column dimension must match the number of mapped global dof ordinals.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "constraint matrix column dimension must match the number of mapped global dof ordinals");
  }
  // int constraintCols = _constraintMatrix.dimension(1);
}
const set<unsigned> & SubBasisDofMatrixMapper::basisDofOrdinalFilter() {
  return _basisDofOrdinalFilter;
}
const FieldContainer<double> &SubBasisDofMatrixMapper::constraintMatrix() {
  return _constraintMatrix;
}
FieldContainer<double> SubBasisDofMatrixMapper::getConstraintMatrix() {
  return _constraintMatrix;
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

void SubBasisDofMatrixMapper::mapDataIntoGlobalContainer(const FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                                         bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, FieldContainer<double> &globalData) {
    // like calling mapData, above, with transposeConstraint = true
    
  const set<unsigned>* basisOrdinalFilter = &this->basisDofOrdinalFilter();
  vector<unsigned> dofIndices(basisOrdinalFilter->begin(),basisOrdinalFilter->end());
  FieldContainer<double> subBasisData(basisOrdinalFilter->size());
  int dofCount = basisOrdinalFilter->size();
  if (wholeBasisData.rank()==1) {
    for (int i=0; i<dofCount; i++) {
      subBasisData[i] = wholeBasisData[dofIndices[i]];
    }
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "mapDataIntoGlobalContainer only supports rank 1 basis data");
  }
  
  // subBasisData must be rank 2, and must have the same size as FilteredLocalDofOrdinals in its first dimension
  // reshape as a rank 2 container (column vector as a matrix):
  subBasisData.resize(subBasisData.dimension(0),1);
  int constraintRows = _constraintMatrix.dimension(1);
  int constraintCols = _constraintMatrix.dimension(0);
  int dataCols = subBasisData.dimension(1);
  int dataRows = subBasisData.dimension(0);
  
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
  
  char constraintTransposeFlag = 'T';
  char dataTransposeFlag = 'N';
  
  SerialDenseWrapper::multiply(result1,_constraintMatrix,subBasisData,constraintTransposeFlag,dataTransposeFlag);
  
  for (int i=0; i<result1.size(); i++) {
    GlobalIndexType globalIndex_i = _mappedGlobalDofOrdinals[i];
    if (fittableDofsOnly && (fittableDofIndices.find(globalIndex_i) == fittableDofIndices.end())) continue; // skip this one
    unsigned globalOrdinal_i = globalIndexToOrdinal.find(globalIndex_i)->second;
    globalData[globalOrdinal_i] += result1[i];
  }
}

vector<GlobalIndexType> SubBasisDofMatrixMapper::mappedGlobalDofOrdinals() {
  return _mappedGlobalDofOrdinals;
}

SubBasisDofMapperPtr SubBasisDofMatrixMapper::negatedDofMapper() {
  FieldContainer<double> negatedConstraintMatrix = _constraintMatrix;
  SerialDenseWrapper::multiplyFCByWeight(negatedConstraintMatrix, -1);
  return Teuchos::rcp( new SubBasisDofMatrixMapper(_basisDofOrdinalFilter, _mappedGlobalDofOrdinals, negatedConstraintMatrix) );
}