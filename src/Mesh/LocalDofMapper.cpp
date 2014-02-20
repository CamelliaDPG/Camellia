//
//  LocalDofMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#include "LocalDofMapper.h"

void LocalDofMapper::filterData(const vector<int> dofIndices, const FieldContainer<double> &data, FieldContainer<double> &filteredData, int rankToFilter) {
  int dofCount = dofIndices.size();
  if (data.rank()==1) {
    filteredData.resize(dofCount);
    for (int i=0; i<dofCount; i++) {
      filteredData(i) = data(dofIndices[i]);
    }
  } else if (data.rank()==2) {
    if (rankToFilter==0) {
      filteredData.resize(dofCount,filteredData.dimension(1));
    } else if (rankToFilter==1) {
      filteredData.resize(filteredData.dimension(0),dofCount);
    } else {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported rank to filter.");
    }
    for (int i=0; i<filteredData.dimension(0); i++) {
      for (int j=0; j<filteredData.dimension(1); j++) {
        if (rankToFilter==0) {
          filteredData(i,j) = data(dofIndices[i],j);
        } else {
          filteredData(i,j) = data(i,dofIndices[j]);
        }
      }
    }
  }
}

void LocalDofMapper::addSubBasisMapContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData, int rankToMap) {
  bool transposeConstraint = false;
  vector<int> varDofIndices = _dofOrdering->getDofIndices(varID, sideOrdinal);
  FieldContainer<double> basisData;
  filterData(varDofIndices, localData, basisData, rankToMap);
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    bool transposeData = rankToMap==0;
    int rankNotToMap = transposeData ? 0 : 1; // this would become plural (ranksNotToMap) if ever we supported more than rank 2.  But I'm not sure why we would.
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    FieldContainer<double> subBasisData;
    vector<int> basisOrdinalFilter(subBasisDofMapper->basisDofOrdinalFilter().begin(), subBasisDofMapper->basisDofOrdinalFilter().end());
    filterData(basisOrdinalFilter, basisData, subBasisData, rankToMap);
    if (basisData.rank()==1) {
      subBasisData.resize(subBasisData.dimension(0),1); // subBasisMapper's mapData method requires rank 2.
    }
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, subBasisData, transposeData);
    vector<GlobalIndexType> globalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
    for (int sbGlobalOrdinal=0; sbGlobalOrdinal<globalIndices.size(); sbGlobalOrdinal++) {
      GlobalIndexType globalIndex = globalIndices[sbGlobalOrdinal];
      unsigned globalOrdinal = _globalIndexToOrdinal[globalIndex];
      if (!transposeData) {
        if (globalData.rank()==1) {
          globalData(globalOrdinal) += mappedSubBasisData(sbGlobalOrdinal,0);
        } else {
          for (int i=0; i<globalData.dimension(rankNotToMap); i++) {
            globalData(globalOrdinal,i) += mappedSubBasisData(sbGlobalOrdinal,i);
          }
        }
      } else { // transpose is true ==> globalData.rank() must be 2
        for (int i=0; i<globalData.dimension(rankNotToMap); i++) {
          globalData(i,globalOrdinal) += mappedSubBasisData(i,sbGlobalOrdinal);
        }
      }
    }
  }
}

void LocalDofMapper::addReverseSubBasisMapContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData, int rankToMap) {
  bool transposeConstraint = true;
  bool transposeData = rankToMap==0;
  int rankNotToMap = transposeData ? 0 : 1; // this would become plural (ranksNotToMap) if ever we supported more than rank 2.  But I'm not sure why we would.
  
  vector<GlobalIndexType> globalIndices = this->globalIndices();
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    vector<int> globalOrdinalFilter;
    vector<GlobalIndexType> globalDofIndices = subBasisDofMapper->mappedGlobalDofOrdinals();
    for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices.size(); subBasisGlobalDofOrdinal++) {
      globalOrdinalFilter.push_back(_globalIndexToOrdinal[ globalDofIndices[subBasisGlobalDofOrdinal] ]);
    }
    FieldContainer<double> filteredSubBasisData;
    filterData(globalOrdinalFilter, globalData, filteredSubBasisData, rankToMap);
    if (filteredSubBasisData.rank()==1) {
      filteredSubBasisData.resize(filteredSubBasisData.dimension(0),1); // subBasisMapper's mapData method requires rank 2.
    }
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, filteredSubBasisData, transposeData);
    set<unsigned> localDofOrdinals = subBasisDofMapper->basisDofOrdinalFilter();
    int j=0;
    for (set<unsigned>::iterator localDofOrdinalIt = localDofOrdinals.begin(); localDofOrdinalIt != localDofOrdinals.end(); localDofOrdinalIt++, j++) {
      unsigned localDofOrdinal = *localDofOrdinalIt;
      unsigned localDofIndex = _dofOrdering->getDofIndex(varID, localDofOrdinal);
      if (!transposeData) {
        if (localData.rank()==1) {
          localData(localDofIndex) += mappedSubBasisData(j,0);
        } else {
          for (int i=0; i<localData.dimension(rankNotToMap); i++) {
            localData(localDofIndex,i) += mappedSubBasisData(j,i);
          }
        }
      } else { // transpose is true ==> localData.rank() must be 2
        for (int i=0; i<localData.dimension(rankNotToMap); i++) {
          localData(i,localDofIndex) += mappedSubBasisData(i,j);
        }
      }
    }
  }
}

LocalDofMapper::LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps, vector< map< int, BasisMap > > sideMaps) {
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

FieldContainer<double> LocalDofMapper::mapData(const FieldContainer<double> &data, bool localToGlobal, int rankToMap) {
  unsigned dofCount = _dofOrdering->totalDofs();
  if ((data.rank() != 1) && (data.rank() != 2)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data must be rank 1 or rank 2");
  }
  if (rankToMap==-1) {
    FieldContainer<double> mappedData = data;
    for (int rank=0; rank<data.rank(); rank++) {
      mappedData = mapData(mappedData,localToGlobal,rank);
    }
    return mappedData;
  }
  if (data.dimension(rankToMap) != dofCount) {
    cout << "data's rankToMap dimension must match dof ordering's totalDofs.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data dimension 0 must match dof ordering's totalDofs.");
  }
  
  int mappedDofCount = localToGlobal ? _globalIndexToOrdinal.size() : _dofOrdering->totalDofs();
  Teuchos::Array<int> dim;
  data.dimensions(dim);
  dim[rankToMap] = mappedDofCount;
  FieldContainer<double> mappedData(dim);
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++) {
    int varID = volumeMapIt->first;
    BasisMap basisMap = volumeMapIt->second;
    int volumeSideIndex = 0;
    if (localToGlobal)
      addSubBasisMapContribution(varID, volumeSideIndex, basisMap, data, mappedData, rankToMap);
    else
      addReverseSubBasisMapContribution(varID, volumeSideIndex, basisMap, data, mappedData, rankToMap);
  }
  
  // map side data
  int sideCount = _sideMaps.size();
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++) {
      int varID = sideMapIt->first;
      BasisMap basisMap = sideMapIt->second;
      if (localToGlobal)
        addSubBasisMapContribution(varID, sideOrdinal, basisMap, data, mappedData, rankToMap);
      else
        addReverseSubBasisMapContribution(varID, sideOrdinal, basisMap, data, mappedData, rankToMap);
    }
  }
  return mappedData;
}