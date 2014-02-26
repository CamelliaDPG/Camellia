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

void LocalDofMapper::addSubBasisMapContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData) {
  bool transposeConstraint = true; // that is, transpose the matrix constructed by BasisReconciliation.  (The untransposed one goes from global to local)
  
  FieldContainer<double> basisData;
  if (_varIDToMap == -1) {
    vector<int> varDofIndices = _dofOrdering->getDofIndices(varID, sideOrdinal);
    filterData(varDofIndices, localData, basisData);
  } else {
    basisData = localData;
  }
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    FieldContainer<double> subBasisData;
    vector<int> basisOrdinalFilter(subBasisDofMapper->basisDofOrdinalFilter().begin(), subBasisDofMapper->basisDofOrdinalFilter().end());
    filterData(basisOrdinalFilter, basisData, subBasisData);
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, subBasisData);
    vector<GlobalIndexType> globalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
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

void LocalDofMapper::addReverseSubBasisMapContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData) {
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
  for (set<GlobalIndexType>::iterator globalIndexIt = globalIndices.begin(); globalIndexIt != globalIndices.end(); globalIndexIt++) {
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
      addSubBasisMapContribution(varID, volumeSideIndex, basisMap, data, mappedData);
    else
      addReverseSubBasisMapContribution(varID, volumeSideIndex, basisMap, data, mappedData);
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
        addSubBasisMapContribution(varID, sideOrdinal, basisMap, data, mappedData);
      else
        addReverseSubBasisMapContribution(varID, sideOrdinal, basisMap, data, mappedData);
    }
  }
  return mappedData;
}