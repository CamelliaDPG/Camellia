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

void LocalDofMapper::addSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData, bool accumulate) {
  bool transposeConstraint = true; // local to global
  
//  cout << "adding sub-basis map contribution for var " << varID << " and sideOrdinal " << sideOrdinal << endl;
  
  FieldContainer<double> basisData;
  if (_varIDToMap == -1) {
    vector<int> varDofIndices = _dofOrdering->getDofIndices(varID, sideOrdinal);
    filterData(varDofIndices, localData, basisData);
  } else {
    basisData = localData;
  }
//  cout << "basisData:\n" << basisData;
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    FieldContainer<double> subBasisData;
    vector<int> basisOrdinalFilter(subBasisDofMapper->basisDofOrdinalFilter().begin(), subBasisDofMapper->basisDofOrdinalFilter().end());
    filterData(basisOrdinalFilter, basisData, subBasisData);
//    cout << "sub-basis, ordinals: ";
//    for (vector<int>::iterator filterIt = basisOrdinalFilter.begin(); filterIt != basisOrdinalFilter.end(); filterIt++) {
//      cout << *filterIt << " ";
//    }
//    cout << endl;
//    cout << "sub-basis data:\n" << subBasisData;
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, subBasisData);
//    cout << "mapped sub-basis data:\n" << mappedSubBasisData;
    vector<GlobalIndexType> globalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
//    cout << "mapped global dof indices: ";
//    for (vector<GlobalIndexType>::iterator dofIt = globalIndices.begin(); dofIt != globalIndices.end(); dofIt++) {
//      cout << *dofIt << " ";
//    }
//    cout << endl;
    for (int sbGlobalOrdinal_i=0; sbGlobalOrdinal_i<globalIndices.size(); sbGlobalOrdinal_i++) {
      GlobalIndexType globalIndex_i = globalIndices[sbGlobalOrdinal_i];
      unsigned globalOrdinal_i = _globalIndexToOrdinal[globalIndex_i];
      if (accumulate) {
        globalData(globalOrdinal_i) += mappedSubBasisData(sbGlobalOrdinal_i);
      } else {
        globalData(globalOrdinal_i)  = mappedSubBasisData(sbGlobalOrdinal_i);
      }
        
    }
  }
}

//void LocalDofMapper::addReverseSubBasisMapMatrixContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData) {
//  cout << "ERROR: addReverseSubBasisMapMatrixContribution not yet implemented.\n";
//  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "addReverseSubBasisMapMatrixContribution not yet implemented");
//}

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
    
    //    cout << "addReverseSubBasisMapVectorContribution: globalIndices ( ";
    //    for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices.size(); subBasisGlobalDofOrdinal++) {
    //      cout << globalDofIndices[subBasisGlobalDofOrdinal] << " ";
    //    }
    //    cout << ") ---> ( ";
    //    for (set<unsigned>::iterator localDofOrdinalIt_i = localDofOrdinals.begin(); localDofOrdinalIt_i != localDofOrdinals.end(); localDofOrdinalIt_i++) {
    //      unsigned localDofIndex_i = _dofOrdering->getDofIndex(varID, *localDofOrdinalIt_i, sideOrdinal);
    //      cout << localDofIndex_i << " ";
    //    }
    //    cout << ")\n";
    int i=0;
    for (set<unsigned>::iterator localDofOrdinalIt_i = localDofOrdinals.begin(); localDofOrdinalIt_i != localDofOrdinals.end(); localDofOrdinalIt_i++, i++) {
      unsigned localDofOrdinal_i = *localDofOrdinalIt_i;
      unsigned localDofIndex_i = _dofOrdering->getDofIndex(varID, localDofOrdinal_i, sideOrdinal);
      localCoefficients(localDofIndex_i) += mappedSubBasisData(i);
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
//  int rank = Teuchos::GlobalMPISession::getRank();
//  int numProcs = Teuchos::GlobalMPISession::getNProc();
//  if (rank==numProcs-1) cout << "Creating local dof mapper.  Volume Map info:\n";
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++) {
    BasisMap basisMap = volumeMapIt->second;
    for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
      vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
      globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());
      
//      // print a little report:
//      if (rank==numProcs-1) {
//        set<unsigned> localOrdinalFilter = (*subBasisMapIt)->basisDofOrdinalFilter();
//        cout << "sub-basis map--local dof ordinal filter: ";
//        for (set<unsigned>::iterator dofOrdinalIt=localOrdinalFilter.begin(); dofOrdinalIt != localOrdinalFilter.end(); dofOrdinalIt++) {
//          cout << *dofOrdinalIt << " ";
//        }
//        cout << endl;
//        cout << "sub-basis map--global dof ordinals: ";
//        for (vector<GlobalIndexType>::iterator globalDofIndexIt=subBasisGlobalIndices.begin(); globalDofIndexIt != subBasisGlobalIndices.end(); globalDofIndexIt++) {
//          cout << *globalDofIndexIt << " ";
//        }
//        cout << endl;
//        cout << endl;
//      }
    }
  }
//  if (rank==numProcs-1) cout << "Creating local dof mapper.  Side Map info:\n";
  for (int sideOrdinal=0; sideOrdinal<_sideMaps.size(); sideOrdinal++) {
//    if (rank==numProcs-1) cout << "Side ordinal " << sideOrdinal << endl;
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++) {
      BasisMap basisMap = sideMapIt->second;
      for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++) {
        vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
        globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());
        
//        // print a little report:
//        if (rank==numProcs-1) {
//          set<unsigned> localOrdinalFilter = (*subBasisMapIt)->basisDofOrdinalFilter();
//          cout << "sub-basis map--local dof ordinal filter: ";
//          for (set<unsigned>::iterator dofOrdinalIt=localOrdinalFilter.begin(); dofOrdinalIt != localOrdinalFilter.end(); dofOrdinalIt++) {
//            cout << *dofOrdinalIt << " ";
//          }
//          cout << endl;
//          cout << "sub-basis map--global dof ordinals: ";
//          for (vector<GlobalIndexType>::iterator globalDofIndexIt=subBasisGlobalIndices.begin(); globalDofIndexIt != subBasisGlobalIndices.end(); globalDofIndexIt++) {
//            cout << *globalDofIndexIt << " ";
//          }
//          cout << endl;
//          cout << endl;
//        }
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

FieldContainer<double> LocalDofMapper::mapLocalDataMatrix(const FieldContainer<double> &localData) {
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
    FieldContainer<double> mappedDataVector = mapLocalData(dataVector);
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
    FieldContainer<double> mappedDataVector = mapLocalData(dataVector);
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
  
  unsigned globalDofCount =  _globalIndexToOrdinal.size();
  unsigned localDofCount = localCoefficients.size();

  FieldContainer<double> localIdentity(localDofCount,localDofCount);
  for (int i=0; i<localDofCount; i++) {
    localIdentity(i,i) = 1.0;
  }
  
  FieldContainer<double> normalMatrix = mapLocalData(localIdentity);
  
  FieldContainer<double> mappedLocalCoefficients = mapLocalData(localCoefficients);
  
  mappedLocalCoefficients.resize(globalDofCount,1);
  
  FieldContainer<double> globalCoefficients(globalDofCount,1);
  
//  cout << "normalMatrix:\n" << normalMatrix;
  
  SerialDenseWrapper::solveSystemUsingQR(globalCoefficients, normalMatrix, mappedLocalCoefficients);
  
  globalCoefficients.resize(globalDofCount);
  
  return globalCoefficients;
}

FieldContainer<double> LocalDofMapper::mapLocalData(const FieldContainer<double> &localData) {
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
    return mapLocalDataMatrix(localData);
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
    addSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, localData, mappedData, true);
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
      addSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, localData, mappedData, true);
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


// DEPRECATED:
FieldContainer<double> LocalDofMapper::mapData(const FieldContainer<double> &data, bool localToGlobal) {
  if (localToGlobal) {
    return mapLocalData(data);
  } else {
    return mapGlobalCoefficients(data);
  }
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