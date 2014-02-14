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
void LocalDofMapper::addSubBasisMapContribution(int varID, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData, int rankToMap) {
  vector<int> varDofIndices = _dofOrdering->getDofIndices(varID);
  FieldContainer<double> basisData;
  filterData(varDofIndices, localData, basisData, rankToMap);
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    bool transpose = rankToMap==0;
    int rankNotToMap = transpose ? 0 : 1; // this would become plural (ranksNotToMap) if ever we supported more than rank 2.  But I'm not sure why we would.
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    FieldContainer<double> subBasisData;
    vector<int> basisOrdinalFilter(subBasisDofMapper->basisDofOrdinalFilter().begin(), subBasisDofMapper->basisDofOrdinalFilter().end());
    filterData(basisOrdinalFilter, basisData, subBasisData, rankToMap);
    if (basisData.rank()==1) {
      subBasisData.resize(subBasisData.dimension(0),1); // subBasisMapper's mapData method requires rank 2.
    }
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(subBasisData, transpose);
    vector<GlobalIndexType> globalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
    for (int sbGlobalOrdinal=0; sbGlobalOrdinal<globalIndices.size(); sbGlobalOrdinal++) {
      GlobalIndexType globalIndex = globalIndices[sbGlobalOrdinal];
      unsigned globalOrdinal = _globalIndexToOrdinal[globalIndex];
      if (!transpose) {
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

FieldContainer<double> LocalDofMapper::mapData(const FieldContainer<double> &localData, int rankToMap) {
  unsigned dofCount = _dofOrdering->totalDofs();
  if ((localData.rank() != 1) && (localData.rank() != 2)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must be rank 1 or rank 2");
  }
  if (rankToMap==-1) {
    FieldContainer<double> globalData = localData;
    for (int rank=0; rank<localData.rank(); rank++) {
      mapData(globalData,rank);
    }
    return globalData;
  }
  if (localData.dimension(rankToMap) != dofCount) {
    cout << "localData's rankToMap dimension must match dof ordering's totalDofs.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData dimension 0 must match dof ordering's totalDofs.");
  }
  
  int mappedDofCount = _globalIndexToOrdinal.size();
  Teuchos::Array<int> dim;
  localData.dimensions(dim);
  dim[rankToMap] = mappedDofCount;
  FieldContainer<double> globalData(dim);
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++) {
    int varID = volumeMapIt->first;
    BasisMap basisMap = volumeMapIt->second;
    addSubBasisMapContribution(varID, basisMap, localData, globalData, rankToMap);
  }
  
  // map side data
  int sideCount = _sideMaps.size();
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++) {
      int varID = sideMapIt->first;
      BasisMap basisMap = sideMapIt->second;
      addSubBasisMapContribution(varID, basisMap, localData, globalData, rankToMap);
    }
  }
  return globalData;
}