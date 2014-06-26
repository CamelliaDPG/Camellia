//
//  LocalDofMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#include "LocalDofMapper.h"
#include <Teuchos_GlobalMPISession.hpp>

#include "CamelliaDebugUtility.h"

#include "SubBasisDofMatrixMapper.h"

#include "SerialDenseWrapper.h"

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

void LocalDofMapper::addSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap,
                                                      const FieldContainer<double> &localData,
                                                      FieldContainer<double> &globalData,
                                                      bool fittableGlobalDofsOnly) {
  bool transposeConstraint = true; // local to global
  
  // NOTE: we spend a *LOT* of time in this method.  We can probably save something by eliminating the filterData
  //       business, and having SubBasisDofMapper accumulate directly into globalData.  We might even save a lot that way.
  
  FieldContainer<double> basisData(_dofOrdering->getDofIndices(varID, sideOrdinal).size());
  if (_varIDToMap == -1) {
    vector<int> varDofIndices = _dofOrdering->getDofIndices(varID, sideOrdinal);
    filterData(varDofIndices, localData, basisData);
  } else {
    basisData = localData;
  }
  set<GlobalIndexType> *fittableDofs;
  if (_volumeMaps.find(varID) != _volumeMaps.end()) {
    fittableDofs = &_fittableGlobalDofOrdinalsInVolume;
  } else {
    fittableDofs = &_fittableGlobalDofOrdinalsOnSides[sideOrdinal];
  }
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    vector<int> basisOrdinalFilter(subBasisDofMapper->basisDofOrdinalFilter().begin(), subBasisDofMapper->basisDofOrdinalFilter().end());
    FieldContainer<double> subBasisData(basisOrdinalFilter.size());
    filterData(basisOrdinalFilter, basisData, subBasisData);
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, subBasisData);
    vector<GlobalIndexType> globalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
    for (int sbGlobalOrdinal_i=0; sbGlobalOrdinal_i<globalIndices.size(); sbGlobalOrdinal_i++) {
      GlobalIndexType globalIndex_i = globalIndices[sbGlobalOrdinal_i];
      if (fittableGlobalDofsOnly && (fittableDofs->find(globalIndex_i) == fittableDofs->end())) continue; // skip this one
      unsigned globalOrdinal_i = _globalIndexToOrdinal[globalIndex_i];
      globalData(globalOrdinal_i) += mappedSubBasisData(sbGlobalOrdinal_i);
    }
  }
}

void LocalDofMapper::addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap,
                                                             const FieldContainer<double> &globalCoefficients, FieldContainer<double> &localCoefficients) {
  bool transposeConstraint = false; // global to local
  
  if (_varIDToMap != -1) {
    cout << "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.");
  }
  
  //  cout << "*************  varID: " << varID << ", side " << sideOrdinal << "  *************" << endl;
  
  vector<GlobalIndexType> globalIndices = this->globalIndices();
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    vector<int> globalOrdinalFilter;
    vector<GlobalIndexType> globalDofIndices = subBasisDofMapper->mappedGlobalDofOrdinals();
    for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices.size(); subBasisGlobalDofOrdinal++) {
      globalOrdinalFilter.push_back(_globalIndexToOrdinal[ globalDofIndices[subBasisGlobalDofOrdinal] ]);
    }
    FieldContainer<double> filteredSubBasisData;
    filterData(globalOrdinalFilter, globalCoefficients, filteredSubBasisData);
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, filteredSubBasisData);
    set<unsigned> localDofOrdinals = subBasisDofMapper->basisDofOrdinalFilter();

    int i=0;
    for (set<unsigned>::iterator localDofOrdinalIt_i = localDofOrdinals.begin(); localDofOrdinalIt_i != localDofOrdinals.end(); localDofOrdinalIt_i++, i++) {
      unsigned localDofOrdinal_i = *localDofOrdinalIt_i;
      unsigned localDofIndex_i = _dofOrdering->getDofIndex(varID, localDofOrdinal_i, sideOrdinal);
      localCoefficients(localDofIndex_i) += mappedSubBasisData(i);
    }
  }
}

LocalDofMapper::LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps,
                               set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume,
                               vector< map< int, BasisMap > > sideMaps,
                               vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides,
                               int varIDToMap, int sideOrdinalToMap) {
  _varIDToMap = varIDToMap;
  _sideOrdinalToMap = sideOrdinalToMap;
  _dofOrdering = dofOrdering;
  _volumeMaps = volumeMaps;
  _sideMaps = sideMaps;
  _fittableGlobalDofOrdinalsInVolume = fittableGlobalDofOrdinalsInVolume;
  _fittableGlobalDofOrdinalsOnSides = fittableGlobalDofOrdinalsOnSides;
  set<GlobalIndexType> globalIndices;
//  int rank = Teuchos::GlobalMPISession::getRank();
//  int numProcs = Teuchos::GlobalMPISession::getNProc();
//  if (rank==numProcs-1) cout << "Creating local dof mapper.  Volume Map info:\n";
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

vector<GlobalIndexType> LocalDofMapper::fittableGlobalIndices() {
  // the implementation does not assume that the global indices will be in numerical order (which they currently are)
  vector<GlobalIndexType> allGlobalIndices = globalIndices();

  set<GlobalIndexType> fittableIndicesSet;
  for (int sideOrdinal=0; sideOrdinal<_fittableGlobalDofOrdinalsOnSides.size(); sideOrdinal++) {
    fittableIndicesSet.insert(_fittableGlobalDofOrdinalsOnSides[sideOrdinal].begin(), _fittableGlobalDofOrdinalsOnSides[sideOrdinal].end());
  }
  
  fittableIndicesSet.insert(_fittableGlobalDofOrdinalsInVolume.begin(),_fittableGlobalDofOrdinalsInVolume.end());
  
  vector<GlobalIndexType> fittableIndices;
  for (vector<GlobalIndexType>::iterator globalIndexIt=allGlobalIndices.begin(); globalIndexIt != allGlobalIndices.end(); globalIndexIt++) {
    GlobalIndexType globalIndex = *globalIndexIt;
    if (fittableIndicesSet.find(globalIndex) != fittableIndicesSet.end()) {
      fittableIndices.push_back(globalIndex);
    }
  }
  return fittableIndices;
}

FieldContainer<double> LocalDofMapper::mapLocalDataMatrix(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly) {
  int dataSize = localData.dimension(0);
  if (localData.dimension(1) != dataSize) {
    cout << "Error: localData matrix must be square.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData matrix must be square");
  }
  FieldContainer<double> dataVector(dataSize);
  FieldContainer<double> intermediateDataMatrix;
  int mappedDataSize;
  for (int i=0; i<dataSize; i++) {
    for (int j=0; j<dataSize; j++) {
      dataVector(j) = localData(i,j);
    }
    FieldContainer<double> mappedDataVector = mapLocalData(dataVector, fittableGlobalDofsOnly);
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
    FieldContainer<double> mappedDataVector = mapLocalData(dataVector, fittableGlobalDofsOnly);
    for (int i=0; i<mappedDataSize; i++) {
      globalData(i,j) = mappedDataVector(i);
    }
  }
  return globalData;
}

FieldContainer<double> LocalDofMapper::fitLocalCoefficients(const FieldContainer<double> &localCoefficients) {
  // solves normal equations (if the localCoefficients are in the range of the global-to-local operator, then the returned coefficients will be the preimage of localCoefficients under that operator)
  if (_varIDToMap == -1) {
    cout << "ERROR: for the present, fitLocalCoefficients is only supported when _varIDToMap has been specified.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: for the present, fitLocalCoefficients is only supported when _varIDToMap has been specified.\n");
  }
  
  unsigned localDofCount = localCoefficients.size();

  FieldContainer<double> localIdentity(localDofCount,localDofCount);
  for (int i=0; i<localDofCount; i++) {
    localIdentity(i,i) = 1.0;
  }
  
  FieldContainer<double> normalMatrix = mapLocalData(localIdentity, true);
  
  FieldContainer<double> mappedLocalCoefficients = mapLocalData(localCoefficients, true);
  
  vector<int> ordinalFilter;
  
  vector<GlobalIndexType> fittableOrdinals = fittableGlobalIndices();
  for (int i=0; i<fittableOrdinals.size(); i++) {
    ordinalFilter.push_back(_globalIndexToOrdinal[fittableOrdinals[i]]);
  }

  FieldContainer<double> filteredMappedLocalCoefficients(fittableOrdinals.size());
  filterData(ordinalFilter, mappedLocalCoefficients, filteredMappedLocalCoefficients);
  
  FieldContainer<double> filteredNormalMatrix(fittableOrdinals.size(),fittableOrdinals.size());
  filterData(ordinalFilter, normalMatrix, filteredNormalMatrix);
  
  filteredMappedLocalCoefficients.resize(filteredMappedLocalCoefficients.dimension(0),1);
  
  FieldContainer<double> fittableGlobalCoefficients(fittableOrdinals.size(),1);
  
//  cout << "localCoefficients:\n" << localCoefficients;
//  Camellia::print("fittableOrdinals", fittableOrdinals);
//  Camellia::print("ordinalFilter", ordinalFilter);
//  cout << "mappedLocalCoefficients:\n" << mappedLocalCoefficients;
//  cout << "filteredMappedLocalCoefficients:\n" << filteredMappedLocalCoefficients;
//  cout << "normalMatrix:\n" << normalMatrix;
//  cout << "filteredNormalMatrix:\n" << filteredNormalMatrix;
  
  if (fittableGlobalCoefficients.size() > 0)
    SerialDenseWrapper::solveSystemUsingQR(fittableGlobalCoefficients, filteredNormalMatrix, filteredMappedLocalCoefficients);
  
  fittableGlobalCoefficients.resize(fittableGlobalCoefficients.dimension(0));
  
  return fittableGlobalCoefficients;
}

FieldContainer<double> LocalDofMapper::mapLocalData(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly) {
  unsigned dofCount;
  if (_varIDToMap == -1) {
    dofCount = _dofOrdering->totalDofs();
  } else {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  if ((localData.rank() != 1) && (localData.rank() != 2)) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must be rank 1 or rank 2");
  }
  if (localData.dimension(0) != dofCount) {
    cout << "data's dimension 0 must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData dimension 0 must match dofCount.");
  }
  if (localData.rank()==2) {
    if (localData.dimension(1) != dofCount) {
      cout << "data's dimension 1, if present, must match dofCount.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data dimension 1 must match dofCount.");
    }
  }
  if (localData.rank()==2) {
    return mapLocalDataMatrix(localData, fittableGlobalDofsOnly);
  }
  
  int mappedDofCount =  _globalIndexToOrdinal.size();
  Teuchos::Array<int> dim;
  localData.dimensions(dim);
  dim[0] = mappedDofCount;
  FieldContainer<double> mappedData(dim);
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++) {
    int varID = volumeMapIt->first;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap basisMap = volumeMapIt->second;
    int volumeSideIndex = 0;
    addSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, localData, mappedData, fittableGlobalDofsOnly);
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
      addSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, localData, mappedData, fittableGlobalDofsOnly);
    }
  }
  return mappedData;
}

FieldContainer<double> LocalDofMapper::mapGlobalCoefficients(const FieldContainer<double> &globalCoefficients) {
  unsigned dofCount;
  if (_varIDToMap == -1) {
    dofCount = _globalIndexToOrdinal.size();
  } else {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  if (globalCoefficients.rank() != 1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalCoefficients must be rank 1");
  }
  if (globalCoefficients.dimension(0) != dofCount) {
    cout << "globalCoefficients's dimension 0 must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalCoefficients dimension 0 must match dofCount.");
  }
  
  int mappedDofCount = _dofOrdering->totalDofs();
  Teuchos::Array<int> dim;
  globalCoefficients.dimensions(dim);
  dim[0] = mappedDofCount;
  FieldContainer<double> localCoefficients(dim);
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++) {
    int varID = volumeMapIt->first;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap basisMap = volumeMapIt->second;
    int volumeSideIndex = 0;
    addReverseSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, globalCoefficients, localCoefficients);
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
      addReverseSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, globalCoefficients, localCoefficients);
    }
  }
  return localCoefficients;
}

void LocalDofMapper::printMappingReport() {
//  map< int, BasisMap > _volumeMaps; // keys are var IDs (fields)
//  vector< map< int, BasisMap > > _sideMaps; // outer index is side ordinal; map keys are var IDs
  vector< map< int, BasisMap > > allMaps = _sideMaps; // side and volume taken together...
  allMaps.insert(allMaps.begin(), _volumeMaps);
  for (int i=0; i<allMaps.size(); i++)
  {
    map<int,BasisMap> basisMaps = allMaps[i];
    if (i==0) cout << "###########  Volume Maps: ###########\n";
    else cout << "###########  Side Ordinal " << i - 1 << " Maps: ###########  "<< endl;
    for (map< int, BasisMap >::iterator mapIt = basisMaps.begin(); mapIt != basisMaps.end(); mapIt++)
    {
      int varID = mapIt->first;
      cout << "***** varID " << varID << " ***** \n";
      BasisMap basisMap = mapIt->second;
      for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
        SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
        Camellia::print("local dof ordinals", subBasisDofMapper->basisDofOrdinalFilter());
        Camellia::print("global ordinals   ", subBasisDofMapper->mappedGlobalDofOrdinals());
        
        SubBasisDofMatrixMapper * matrixMapper = (SubBasisDofMatrixMapper *) subBasisDofMapper.get();
        cout << "constraint matrix:\n" << matrixMapper->constraintMatrix();
      }
    }
  }
}