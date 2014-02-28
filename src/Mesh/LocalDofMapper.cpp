//
//  LocalDofMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#include "LocalDofMapper.h"

void LocalDofMapper::filterData(const vector<int> dofIndices, const FieldContainer<double> &data, FieldContainer<double> &filteredData) {
  int dofCount = dofIndices.size();
  if (data.rank()==1) {
    filteredData.resize(dofCount);
    for (int i=0; i<dofCount; i++) {
      filteredData(i) = data(dofIndices[i]);
    }
  } else if (data.rank()==2) {
    filteredData.resize(dofCount,dofCount);
    for (int i=0; i<filteredData.dimension(0); i++) {
      for (int j=0; j<filteredData.dimension(1); j++) {
        filteredData(i,j) = data(dofIndices[i],dofIndices[j]);
      }
    }
  }
}

//void LocalDofMapper::addSubBasisMapMatrixContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData) {
//  int globalDofCount = globalData.dimension(0);
//  int localDofCount = localData.dimension(0);
//  FieldContainer<double> localDataVector(localDofCount);
//  FieldContainer<double> globalDataVector(globalDofCount);
//  
//  FieldContainer<double> globalDataIntermediateMatrix(localDofCount,globalDofCount);
//  for (int i=0; i<localDofCount; i++) {
//    for (int j=0; j<localDofCount; j++) {
//      localDataVector(j) = localData(i,j);
//    }
//    globalDataVector.initialize(0);
//    addSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, localDataVector, globalDataVector);
//    for (int j=0; j<globalDofCount; j++) {
//      globalDataIntermediateMatrix(i,j) += globalDataVector(j);
//    }
//  }
//  for (int j=0; j<globalDofCount; j++) {
//    for (int i=0; i<localDofCount; i++) {
//      localDataVector(i) = globalDataIntermediateMatrix(i,j);
//    }
//    globalDataVector.initialize(0);
//    addSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, localDataVector, globalDataVector);
//    for (int i=0; i<globalDofCount; i++) {
//      globalData(i,j) += globalDataVector(i);
//    }
//  }
//}

void LocalDofMapper::addSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData) {
  // TODO: delete the code below that treats the rank-2 case
  bool transposeConstraint = true; // that is, transpose the matrix constructed by BasisReconciliation.  (The untransposed one goes from global to local)
  
  cout << "adding sub-basis map contribution for var " << varID << " and sideOrdinal " << sideOrdinal << endl;
  
  FieldContainer<double> basisData;
  if (_varIDToMap == -1) {
    vector<int> varDofIndices = _dofOrdering->getDofIndices(varID, sideOrdinal);
    filterData(varDofIndices, localData, basisData);
  } else {
    basisData = localData;
  }
  cout << "basisData:\n" << basisData;
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    FieldContainer<double> subBasisData;
    vector<int> basisOrdinalFilter(subBasisDofMapper->basisDofOrdinalFilter().begin(), subBasisDofMapper->basisDofOrdinalFilter().end());
    filterData(basisOrdinalFilter, basisData, subBasisData);
    cout << "sub-basis, ordinals: ";
    for (vector<int>::iterator filterIt = basisOrdinalFilter.begin(); filterIt != basisOrdinalFilter.end(); filterIt++) {
      cout << *filterIt << " ";
    }
    cout << endl;
    cout << "sub-basis data:\n" << subBasisData;
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, subBasisData);
    cout << "mapped sub-basis data:\n" << mappedSubBasisData;
    vector<GlobalIndexType> globalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
    cout << "mapped global dof indices: ";
    for (vector<GlobalIndexType>::iterator dofIt = globalIndices.begin(); dofIt != globalIndices.end(); dofIt++) {
      cout << *dofIt << " ";
    }
    cout << endl;
    for (int sbGlobalOrdinal_i=0; sbGlobalOrdinal_i<globalIndices.size(); sbGlobalOrdinal_i++) {
      GlobalIndexType globalIndex_i = globalIndices[sbGlobalOrdinal_i];
      unsigned globalOrdinal_i = _globalIndexToOrdinal[globalIndex_i];
      if (globalData.rank()==1) {
        globalData(globalOrdinal_i) += mappedSubBasisData(sbGlobalOrdinal_i);
      } else {
        for (int sbGlobalOrdinal_j=0; sbGlobalOrdinal_j<globalIndices.size(); sbGlobalOrdinal_j++) {
          GlobalIndexType globalIndex_j = globalIndices[sbGlobalOrdinal_j];
          unsigned globalOrdinal_j = _globalIndexToOrdinal[globalIndex_j];
          globalData(globalOrdinal_i,globalOrdinal_j) += mappedSubBasisData(sbGlobalOrdinal_i,sbGlobalOrdinal_j);
        }
      }
    }
  }
}

//void LocalDofMapper::addReverseSubBasisMapMatrixContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData) {
//  cout << "ERROR: addReverseSubBasisMapMatrixContribution not yet implemented.\n";
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "addReverseSubBasisMapMatrixContribution not yet implemented");
//}

void LocalDofMapper::addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData) {
  // TODO: delete the code below that treats the rank-2 case
  bool transposeConstraint = false;
  
  if (_varIDToMap != -1) {
    cout << "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.");
  }
  
  vector<GlobalIndexType> globalIndices = this->globalIndices();
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    vector<int> globalOrdinalFilter;
    vector<GlobalIndexType> globalDofIndices = subBasisDofMapper->mappedGlobalDofOrdinals();
    for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices.size(); subBasisGlobalDofOrdinal++) {
      globalOrdinalFilter.push_back(_globalIndexToOrdinal[ globalDofIndices[subBasisGlobalDofOrdinal] ]);
    }
    FieldContainer<double> filteredSubBasisData;
    filterData(globalOrdinalFilter, globalData, filteredSubBasisData);
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, filteredSubBasisData);
    set<unsigned> localDofOrdinals = subBasisDofMapper->basisDofOrdinalFilter();
    int i=0;
    for (set<unsigned>::iterator localDofOrdinalIt_i = localDofOrdinals.begin(); localDofOrdinalIt_i != localDofOrdinals.end(); localDofOrdinalIt_i++, i++) {
      unsigned localDofOrdinal_i = *localDofOrdinalIt_i;
      unsigned localDofIndex_i = _dofOrdering->getDofIndex(varID, localDofOrdinal_i);
      if (localData.rank()==1) {
        localData(localDofIndex_i) += mappedSubBasisData(i);
      } else {
        int j=0;
        for (set<unsigned>::iterator localDofOrdinalIt_j = localDofOrdinals.begin(); localDofOrdinalIt_j != localDofOrdinals.end(); localDofOrdinalIt_j++, j++) {
          unsigned localDofOrdinal_j = *localDofOrdinalIt_j;
          unsigned localDofIndex_j = _dofOrdering->getDofIndex(varID, localDofOrdinal_j);
          localData(localDofIndex_i,localDofIndex_j) += mappedSubBasisData(i,j);
        }
      }
    }
  }
}

LocalDofMapper::LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps, vector< map< int, BasisMap > > sideMaps,
                               int varIDToMap, int sideOrdinalToMap) {
  _varIDToMap = varIDToMap;
  _sideOrdinalToMap = sideOrdinalToMap;
  _dofOrdering = dofOrdering;
  _volumeMaps = volumeMaps;
  _sideMaps = sideMaps;
  set<GlobalIndexType> globalIndices;
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++) {
    BasisMap basisMap = volumeMapIt->second;
    for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
      vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
      globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());
    }
  }
  for (int sideOrdinal=0; sideOrdinal<_sideMaps.size(); sideOrdinal++) {
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++) {
      BasisMap basisMap = sideMapIt->second;
      for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
        vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
        globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());
      }
    }
  }
  unsigned ordinal = 0;
//  cout << "_globalIndexToOrdinal:\n";
  for (set<GlobalIndexType>::iterator globalIndexIt = globalIndices.begin(); globalIndexIt != globalIndices.end(); globalIndexIt++) {
//    cout << *globalIndexIt << " ---> " << ordinal << endl;
    _globalIndexToOrdinal[*globalIndexIt] = ordinal++;
  }
}

vector<GlobalIndexType> LocalDofMapper::globalIndices() {
  // the implementation does not assume that the global indices will be in numerical order (which they currently are)
  vector<GlobalIndexType> indices(_globalIndexToOrdinal.size());
  
  for (map<GlobalIndexType, unsigned>::iterator globalIndexIt=_globalIndexToOrdinal.begin(); globalIndexIt != _globalIndexToOrdinal.end(); globalIndexIt++) {
    GlobalIndexType globalIndex = globalIndexIt->first;
    unsigned ordinal = globalIndexIt->second;
    indices[ordinal] = globalIndex;
  }
  return indices;
}

FieldContainer<double> LocalDofMapper::mapDataMatrix(const FieldContainer<double> &data, bool localToGlobal) {
  int dataSize = data.dimension(0);
  if (data.dimension(1) != dataSize) {
    cout << "Error: data matrix must be square.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data matrix must be square");
  }
  FieldContainer<double> dataVector(dataSize);
  FieldContainer<double> intermediateDataMatrix;
  int mappedDataSize;
  for (int i=0; i<dataSize; i++) {
    for (int j=0; j<dataSize; j++) {
      dataVector(j) = data(i,j);
    }
    FieldContainer<double> mappedDataVector = mapData(dataVector, localToGlobal);
    if (i==0) { // size intermediateDataMatrix once we know the entry for the first row
      mappedDataSize = mappedDataVector.size();
      intermediateDataMatrix.resize(dataSize,mappedDataSize);
    }
    for (int j=0; j<mappedDataSize; j++) {
      intermediateDataMatrix(i,j) = mappedDataVector(j);
    }
  }
  FieldContainer<double> globalData(mappedDataSize,mappedDataSize);
  for (int j=0; j<mappedDataSize; j++) {
    for (int i=0; i<dataSize; i++) {
      dataVector(i) = intermediateDataMatrix(i,j);
    }
    FieldContainer<double> mappedDataVector = mapData(dataVector, localToGlobal);
    for (int i=0; i<mappedDataSize; i++) {
      globalData(i,j) = mappedDataVector(i);
    }
  }
  return globalData;
}

FieldContainer<double> LocalDofMapper::mapData(const FieldContainer<double> &data, bool localToGlobal) {
  unsigned dofCount;
  if (_varIDToMap == -1) {
    dofCount = _dofOrdering->totalDofs();
  } else {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  if ((data.rank() != 1) && (data.rank() != 2)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data must be rank 1 or rank 2");
  }
  if (data.dimension(0) != dofCount) {
    cout << "data's dimension 0 must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data dimension 0 must match dofCount.");
  }
  if (data.rank()==2) {
    if (data.dimension(1) != dofCount) {
      cout << "data's dimension 1, if present, must match dof ordering's totalDofs.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data dimension 1 must match dofCount.");
    }
  }
  if (data.rank()==2) {
    return mapDataMatrix(data, localToGlobal);
  }
  // TODO: delete the code below that treats the rank-2 case
  
  int mappedDofCount = localToGlobal ? _globalIndexToOrdinal.size() : _dofOrdering->totalDofs();
  Teuchos::Array<int> dim;
  data.dimensions(dim);
  dim[0] = mappedDofCount;
  if (data.rank()==2) dim[1] = mappedDofCount;
  FieldContainer<double> mappedData(dim);
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++) {
    int varID = volumeMapIt->first;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap basisMap = volumeMapIt->second;
    int volumeSideIndex = 0;
    if (localToGlobal)
      addSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, data, mappedData);
    else
      addReverseSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, data, mappedData);
  }
  
  // map side data
  int sideCount = _sideMaps.size();
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    bool skipSide = (_sideOrdinalToMap != -1) && (sideOrdinal != _sideOrdinalToMap);
    if (skipSide) continue;
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++) {
      int varID = sideMapIt->first;
      bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
      if (skipVar) continue;
      BasisMap basisMap = sideMapIt->second;
      if (localToGlobal)
        addSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, data, mappedData);
      else
        addReverseSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, data, mappedData);
    }
  }
  return mappedData;
}