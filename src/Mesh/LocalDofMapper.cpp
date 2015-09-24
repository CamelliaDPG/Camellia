//
//  LocalDofMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#include "LocalDofMapper.h"
#include <Teuchos_GlobalMPISession.hpp>

#include "BasisFactory.h"
#include "CamelliaCellTools.h"
#include "CamelliaDebugUtility.h"
#include "SerialDenseWrapper.h"
#include "SubBasisDofMatrixMapper.h"

using namespace Intrepid;
using namespace Camellia;

void LocalDofMapper::filterData(const vector<int> dofIndices, const FieldContainer<double> &data, FieldContainer<double> &filteredData)
{
  int dofCount = dofIndices.size();
  if (data.rank()==1)
  {
    filteredData.resize(dofCount);
    for (int i=0; i<dofCount; i++)
    {
      filteredData(i) = data(dofIndices[i]);
    }
  }
  else if (data.rank()==2)
  {
    filteredData.resize(dofCount,dofCount);
    for (int i=0; i<filteredData.dimension(0); i++)
    {
      for (int j=0; j<filteredData.dimension(1); j++)
      {
        filteredData(i,j) = data(dofIndices[i],dofIndices[j]);
      }
    }
  }
}

void LocalDofMapper::addSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap,
                                                      const FieldContainer<double> &localData,
                                                      FieldContainer<double> &globalData,
                                                      bool fittableGlobalDofsOnly)
{
  if (!_dofOrdering->hasBasisEntry(varID, sideOrdinal)) return; // no contribution
  FieldContainer<double> basisData(_dofOrdering->getDofIndices(varID, sideOrdinal).size());
  if (_varIDToMap == -1)
  {
    vector<int> varDofIndices = _dofOrdering->getDofIndices(varID, sideOrdinal);
    filterData(varDofIndices, localData, basisData);
  }
  else
  {
    basisData = localData;
  }
  set<GlobalIndexType> *fittableDofs;
  if (_volumeMaps.find(varID) != _volumeMaps.end())
  {
    fittableDofs = &_fittableGlobalDofOrdinalsInVolume;
  }
  else
  {
    fittableDofs = &_fittableGlobalDofOrdinalsOnSides[sideOrdinal];
  }
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++)
  {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    subBasisDofMapper->mapDataIntoGlobalContainer(basisData, _globalIndexToOrdinal, fittableGlobalDofsOnly, *fittableDofs, globalData);
  }
}

void LocalDofMapper::addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap,
                                                             const FieldContainer<double> &globalCoefficients, FieldContainer<double> &localCoefficients)
{
  bool transposeConstraint = false; // global to local
  
  bool applyOnLeftOnly = true; // mapData() otherwise will do something like L = C G C^T, where C is the constraint matrix, G the global coefficients, and L the local
  
  if (_varIDToMap != -1)
  {
    cout << "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.");
  }
  
  //  cout << "*************  varID: " << varID << ", side " << sideOrdinal << "  *************" << endl;
  
  vector<GlobalIndexType> globalIndices = this->globalIndices();
  for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++)
  {
    SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
    vector<int> globalOrdinalFilter;
    vector<GlobalIndexType> globalDofIndices = subBasisDofMapper->mappedGlobalDofOrdinals();
    for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices.size(); subBasisGlobalDofOrdinal++)
    {
      globalOrdinalFilter.push_back(_globalIndexToOrdinal[ globalDofIndices[subBasisGlobalDofOrdinal] ]);
    }
    FieldContainer<double> filteredSubBasisData;
    filterData(globalOrdinalFilter, globalCoefficients, filteredSubBasisData);
    FieldContainer<double> mappedSubBasisData = (*subBasisMapIt)->mapData(transposeConstraint, filteredSubBasisData, applyOnLeftOnly);
    set<unsigned> localDofOrdinals = subBasisDofMapper->basisDofOrdinalFilter();
    
    int i=0;
    for (set<unsigned>::iterator localDofOrdinalIt_i = localDofOrdinals.begin(); localDofOrdinalIt_i != localDofOrdinals.end(); localDofOrdinalIt_i++, i++)
    {
      unsigned localDofOrdinal_i = *localDofOrdinalIt_i;
      unsigned localDofIndex_i = _dofOrdering->getDofIndex(varID, localDofOrdinal_i, sideOrdinal);
      
      if (localCoefficients.rank()==1)
      {
        localCoefficients(localDofIndex_i) += mappedSubBasisData(i);
      }
      else if (localCoefficients.rank()==2)
      {
        int j=0;
        for (set<unsigned>::iterator localDofOrdinalIt_j = localDofOrdinals.begin(); localDofOrdinalIt_j != localDofOrdinals.end(); localDofOrdinalIt_j++, j++)
        {
          unsigned localDofOrdinal_j = *localDofOrdinalIt_j;
          unsigned localDofIndex_j = _dofOrdering->getDofIndex(varID, localDofOrdinal_j, sideOrdinal);
          localCoefficients(localDofIndex_i, localDofIndex_j) += mappedSubBasisData(i,j);
        }
      }
    }
  }
}

LocalDofMapper::LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps,
                               set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume,
                               vector< map< int, BasisMap > > sideMaps,
                               vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides,
                               set<GlobalIndexType> unmappedGlobalDofOrdinals,
                               int varIDToMap, int sideOrdinalToMap)
{
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
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++)
  {
    BasisMap basisMap = volumeMapIt->second;
    for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++)
    {
      vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
      globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());
    }
  }
  for (int sideOrdinal=0; sideOrdinal<_sideMaps.size(); sideOrdinal++)
  {
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++)
    {
      BasisMap basisMap = sideMapIt->second;
      for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++)
      {
        vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
        globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());
      }
    }
  }
  globalIndices.insert(unmappedGlobalDofOrdinals.begin(),unmappedGlobalDofOrdinals.end());
  unsigned ordinal = 0;
  //  cout << "_globalIndexToOrdinal:\n";
  for (set<GlobalIndexType>::iterator globalIndexIt = globalIndices.begin(); globalIndexIt != globalIndices.end(); globalIndexIt++)
  {
    //    cout << *globalIndexIt << " ---> " << ordinal << endl;
    _globalIndexToOrdinal[*globalIndexIt] = ordinal++;
  }
}

vector<GlobalIndexType> LocalDofMapper::globalIndices()
{
  // the implementation does not assume that the global indices will be in numerical order (which they currently are)
  vector<GlobalIndexType> indices(_globalIndexToOrdinal.size());
  
  for (map<GlobalIndexType, unsigned>::iterator globalIndexIt=_globalIndexToOrdinal.begin(); globalIndexIt != _globalIndexToOrdinal.end(); globalIndexIt++)
  {
    GlobalIndexType globalIndex = globalIndexIt->first;
    unsigned ordinal = globalIndexIt->second;
    indices[ordinal] = globalIndex;
  }
  return indices;
}

set<GlobalIndexType> LocalDofMapper::globalIndicesForSubcell(int varID, unsigned d, unsigned subcord)
{
//  typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
//  map< int, BasisMap > _volumeMaps; // keys are var IDs (fields)
//  vector< map< int, BasisMap > > _sideMaps; // outer index is side ordinal; map keys are var IDs

  CellTopoPtr volumeTopo = _dofOrdering->cellTopology();

  set<GlobalIndexType> indexSet; // guarantee uniqueness using set
  
  for (auto entry : _volumeMaps)
  {
    BasisMap volumeMap  = entry.second;
    if (entry.first != varID) continue;
    
    BasisPtr volumeBasis = BasisFactory::basisFactory()->getContinuousBasis(_dofOrdering->getBasis(varID));
    
    set<int> dofOrdinalsInt = volumeBasis->dofOrdinalsForSubcell(d, subcord, 0);
    set<unsigned> dofOrdinals(dofOrdinalsInt.begin(), dofOrdinalsInt.end());
    
    for (auto subBasisMap : volumeMap)
    {
      set<GlobalIndexType> subIndexSet = subBasisMap->mappedGlobalDofOrdinalsForBasisOrdinals(dofOrdinals);
      indexSet.insert(subIndexSet.begin(),subIndexSet.end());
    }
  }
  
  int sideDim = volumeTopo->getDimension() - 1;
  for (int sideOrdinal = 0; sideOrdinal < _sideMaps.size(); sideOrdinal++)
  {
    bool assertContainment = false;
    unsigned sideSubcellOrdinal = CamelliaCellTools::subcellReverseOrdinalMap(volumeTopo, sideDim, sideOrdinal, d, subcord,
                                                                              assertContainment);
    if (sideSubcellOrdinal == -1) continue; // subcell not found
    
    for (auto entry : _sideMaps[sideOrdinal])
    {
      BasisMap sideMap  = entry.second;
      if (entry.first != varID) continue;
      
      if (!_dofOrdering->hasBasisEntry(varID, sideOrdinal)) continue;
      
      BasisPtr sideBasis = BasisFactory::basisFactory()->getContinuousBasis(_dofOrdering->getBasis(varID,sideOrdinal));
      
      set<int> dofOrdinalsInt = sideBasis->dofOrdinalsForSubcell(d, sideSubcellOrdinal, 0);
      set<unsigned> dofOrdinals(dofOrdinalsInt.begin(), dofOrdinalsInt.end());
      
      for (auto subBasisMap : sideMap)
      {
        set<GlobalIndexType> subIndexSet = subBasisMap->mappedGlobalDofOrdinalsForBasisOrdinals(dofOrdinals);
        indexSet.insert(subIndexSet.begin(),subIndexSet.end());
      }
    }
  }
  
  return indexSet;
}



vector<GlobalIndexType> LocalDofMapper::fittableGlobalIndices()
{
  if (_fittableGlobalIndices.size() == 0)   // then we have not previously set these...
  {
    // the implementation does not assume that the global indices will be in numerical order (which they currently are)
    vector<GlobalIndexType> allGlobalIndices = globalIndices();
    
    set<GlobalIndexType> fittableIndicesSet;
    for (int sideOrdinal=0; sideOrdinal<_fittableGlobalDofOrdinalsOnSides.size(); sideOrdinal++)
    {
      fittableIndicesSet.insert(_fittableGlobalDofOrdinalsOnSides[sideOrdinal].begin(), _fittableGlobalDofOrdinalsOnSides[sideOrdinal].end());
    }
    
    fittableIndicesSet.insert(_fittableGlobalDofOrdinalsInVolume.begin(),_fittableGlobalDofOrdinalsInVolume.end());
    
    vector<GlobalIndexType> fittableIndices;
    for (vector<GlobalIndexType>::iterator globalIndexIt=allGlobalIndices.begin(); globalIndexIt != allGlobalIndices.end(); globalIndexIt++)
    {
      GlobalIndexType globalIndex = *globalIndexIt;
      if (fittableIndicesSet.find(globalIndex) != fittableIndicesSet.end())
      {
        fittableIndices.push_back(globalIndex);
      }
    }
    _fittableGlobalIndices = fittableIndices;
  }
  return _fittableGlobalIndices;
}

FieldContainer<double> LocalDofMapper::mapLocalDataMatrix(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly)
{
  int dataSize = localData.dimension(0);
  if (localData.dimension(1) != dataSize)
  {
    cout << "Error: localData matrix must be square.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData matrix must be square");
  }
  FieldContainer<double> dataVector(dataSize);
  int mappedDataSize = _globalIndexToOrdinal.size();
  FieldContainer<double> intermediateDataMatrix(dataSize,mappedDataSize);
  
  for (int i=0; i<dataSize; i++)
  {
    FieldContainer<double> mappedDataVector(mappedDataSize);
    for (int j=0; j<dataSize; j++)
    {
      dataVector(j) = localData(i,j);
    }
    mapLocalDataVector(dataVector, fittableGlobalDofsOnly, mappedDataVector);
    for (int j=0; j<mappedDataSize; j++)
    {
      intermediateDataMatrix(i,j) = mappedDataVector(j);
    }
  }
  FieldContainer<double> globalData(mappedDataSize,mappedDataSize);
  for (int j=0; j<mappedDataSize; j++)
  {
    FieldContainer<double> mappedDataVector(mappedDataSize);
    for (int i=0; i<dataSize; i++)
    {
      dataVector(i) = intermediateDataMatrix(i,j);
    }
    mapLocalDataVector(dataVector, fittableGlobalDofsOnly, mappedDataVector);
    for (int i=0; i<mappedDataSize; i++)
    {
      globalData(i,j) = mappedDataVector(i);
    }
  }
  return globalData;
}

FieldContainer<double> LocalDofMapper::fitLocalCoefficients(const FieldContainer<double> &localCoefficients)
{
  // solves normal equations (if the localCoefficients are in the range of the global-to-local operator, then the returned coefficients will be the preimage of localCoefficients under that operator)
  if (_varIDToMap == -1)
  {
    int globalIndexCount = _globalIndexToOrdinal.size();
    FieldContainer<double> allCoefficients(globalIndexCount); // includes both fittable and non-fittable
    
    set<int> varIDs = _dofOrdering->getVarIDs();
    set<pair<int,int>> varIDsAndSidesWithNonZeros;
    map<pair<int,int>, FieldContainer<double>> basisCoefficientsForVarOnSide;
    // first, pass through all variables, noting which ones have nonzero coefficients in localCoefficients
    for (int varID : varIDs)
    {
      vector<int> sides = _dofOrdering->getSidesForVarID(varID);
      for (int sideOrdinal : sides)
      {
        bool nonZeroEntryFound = false;
        vector<int> dofIndices = _dofOrdering->getDofIndices(varID,sideOrdinal);
        FieldContainer<double> basisCoefficients(dofIndices.size());
        int basisOrdinal = 0;
        for (int dofIndex : dofIndices)
        {
          if (localCoefficients(dofIndex) != 0.0)
          {
            nonZeroEntryFound = true;
            basisCoefficients(basisOrdinal) = localCoefficients(dofIndex);
          }
          basisOrdinal++;
        }
        if (nonZeroEntryFound) {
          varIDsAndSidesWithNonZeros.insert({varID,sideOrdinal});
          basisCoefficientsForVarOnSide[{varID,sideOrdinal}] = basisCoefficients;
        }
      }
    }
    
    for (pair<int,int> varIDAndSide : varIDsAndSidesWithNonZeros)
    {
      if (_localDofMapperForVarIDAndSide.find(varIDAndSide) == _localDofMapperForVarIDAndSide.end())
      {
        int varID = varIDAndSide.first;
        int sideOrdinal = varIDAndSide.second;
        
        vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides(_fittableGlobalDofOrdinalsOnSides.size());
        set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume;
        
        bool volumeVar = _dofOrdering->getSidesForVarID(varID).size() == 1;
        if (volumeVar)
        {
          BasisMap volumeMap = _volumeMaps[varID];
          for (SubBasisDofMapperPtr subBasisMap : volumeMap)
          {
            vector<GlobalIndexType> mappedDofOrdinals = subBasisMap->mappedGlobalDofOrdinals();
            for (GlobalIndexType mappedDofOrdinal : mappedDofOrdinals)
            {
              if (_fittableGlobalDofOrdinalsInVolume.find(mappedDofOrdinal) != _fittableGlobalDofOrdinalsInVolume.end())
              {
                fittableGlobalDofOrdinalsInVolume.insert(mappedDofOrdinal);
              }
            }
          }
        }
        else // side var
        {
          BasisMap sideMap = _sideMaps[sideOrdinal][varID];
          for (SubBasisDofMapperPtr subBasisMap : sideMap)
          {
            vector<GlobalIndexType> mappedDofOrdinals = subBasisMap->mappedGlobalDofOrdinals();
            for (GlobalIndexType mappedDofOrdinal : mappedDofOrdinals)
            {
              if (_fittableGlobalDofOrdinalsOnSides[sideOrdinal].find(mappedDofOrdinal) != _fittableGlobalDofOrdinalsOnSides[sideOrdinal].end())
              {
                fittableGlobalDofOrdinalsOnSides[sideOrdinal].insert(mappedDofOrdinal);
              }
            }
          }
        }
        
        _localDofMapperForVarIDAndSide[varIDAndSide] = Teuchos::rcp(new LocalDofMapper(_dofOrdering, _volumeMaps,
                                                                                       fittableGlobalDofOrdinalsInVolume,
                                                                                       _sideMaps, fittableGlobalDofOrdinalsOnSides,
                                                                                       set<GlobalIndexType>(), varID, sideOrdinal));
      }
      FieldContainer<double> fittedVarCoefficients = _localDofMapperForVarIDAndSide[varIDAndSide]->fitLocalCoefficients(basisCoefficientsForVarOnSide[varIDAndSide]);
      vector<GlobalIndexType> fittedVarGlobalIndices = _localDofMapperForVarIDAndSide[varIDAndSide]->fittableGlobalIndices();
      
      int fittedVarGlobalIndexCount = fittedVarCoefficients.size();
      double tol=1e-15; // anything below this we consider a zero entry
      for (int fittedVarEntryOrdinal=0; fittedVarEntryOrdinal<fittedVarGlobalIndexCount; fittedVarEntryOrdinal++)
      {
        if (abs(fittedVarCoefficients[fittedVarEntryOrdinal]) > tol)
        {
          GlobalIndexType fittedVarGlobalIndex = fittedVarGlobalIndices[fittedVarEntryOrdinal];
          allCoefficients[_globalIndexToOrdinal[fittedVarGlobalIndex]] = fittedVarCoefficients[fittedVarEntryOrdinal];
        }
      }
    }
    vector<GlobalIndexType> fittableIndices = fittableGlobalIndices();
    FieldContainer<double> fittedCoefficients(fittableIndices.size());
    for (int fittableOrdinal=0; fittableOrdinal<fittableIndices.size(); fittableOrdinal++)
    {
      GlobalIndexType fittableGlobalIndex = fittableIndices[fittableOrdinal];
      int allCoefficientsOrdinal = _globalIndexToOrdinal[fittableGlobalIndex];
      fittedCoefficients[fittableOrdinal] = allCoefficients[allCoefficientsOrdinal];
      //      cout << "fitted coefficient for global index " << fittableGlobalIndex << ": " << fittedCoefficients[fittableOrdinal] << endl;
    }
    
    return fittedCoefficients;
  }
  
  //  if (_varIDToMap == -1)
  //  {
  //    cout << "ERROR: for the present, fitLocalCoefficients is only supported when _varIDToMap has been specified.\n";
  //    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "ERROR: for the present, fitLocalCoefficients is only supported when _varIDToMap has been specified.\n");
  //  }
  
  unsigned localDofCount = localCoefficients.size();
  
  FieldContainer<double> mappedLocalCoefficients = mapLocalData(localCoefficients, true);
  
  vector<int> ordinalFilter;
  
  vector<GlobalIndexType> fittableOrdinals = fittableGlobalIndices();
  if (fittableOrdinals.size()==0)
  {
    return FieldContainer<double>(0);
  }
  
  for (int i=0; i<fittableOrdinals.size(); i++)
  {
    ordinalFilter.push_back(_globalIndexToOrdinal[fittableOrdinals[i]]);
  }
  
  FieldContainer<double> filteredMappedLocalCoefficients(fittableOrdinals.size());
  filterData(ordinalFilter, mappedLocalCoefficients, filteredMappedLocalCoefficients);
  
  if (_localCoefficientsFitMatrix.size()==0)
  {
    FieldContainer<double> localIdentity(localDofCount,localDofCount);
    for (int i=0; i<localDofCount; i++)
    {
      localIdentity(i,i) = 1.0;
    }
    FieldContainer<double> normalMatrix = mapLocalData(localIdentity, true);
    FieldContainer<double> filteredNormalMatrix(fittableOrdinals.size(),fittableOrdinals.size());
    filterData(ordinalFilter, normalMatrix, filteredNormalMatrix);
    
    FieldContainer<double> filteredIdentityCoefficients(fittableOrdinals.size(),fittableOrdinals.size());
    for (int i=0; i<fittableOrdinals.size(); i++)
    {
      filteredIdentityCoefficients(i,i) = 1.0;
    }
    
    _localCoefficientsFitMatrix.resize(fittableOrdinals.size(),fittableOrdinals.size());
    
    int err = SerialDenseWrapper::solveSystemUsingQR(_localCoefficientsFitMatrix, filteredNormalMatrix, filteredIdentityCoefficients);
    if (err > 0)
    {
      cout << "while trying to fit local coefficients on side " << _sideOrdinalToMap;
      cout << " for variable " << _varIDToMap << ", solveSystemUsingQR returned err = " << err << endl;
      Camellia::print("fittableGlobalOrdinals",fittableOrdinals);
      cout << "localCoefficients:\n" << localCoefficients;
    }
  }
  
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
    SerialDenseWrapper::multiply(fittableGlobalCoefficients, _localCoefficientsFitMatrix, filteredMappedLocalCoefficients);
  
  fittableGlobalCoefficients.resize(fittableGlobalCoefficients.dimension(0));
  
  //  cout << "fittableGlobalCoefficients:\n" << fittableGlobalCoefficients;
  
  return fittableGlobalCoefficients;
}

void LocalDofMapper::mapLocalDataVector(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly,
                                        FieldContainer<double> &mappedDataVector)
{
  mappedDataVector.initialize(0.0);
  unsigned dofCount;
  if (_varIDToMap == -1)
  {
    dofCount = _dofOrdering->totalDofs();
  }
  else
  {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(localData.rank() != 1, std::invalid_argument, "localData must have rank 1");
  if (localData.dimension(0) != dofCount)
  {
    cout << "data's dimension 0 must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData dimension 0 must match dofCount.");
  }
  
  int mappedDofCount =  _globalIndexToOrdinal.size();
  TEUCHOS_TEST_FOR_EXCEPTION(mappedDofCount != mappedDataVector.size(), std::invalid_argument, "Invalid size in mappedDataVector");
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++)
  {
    int varID = volumeMapIt->first;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap basisMap = volumeMapIt->second;
    int volumeSideIndex = 0;
    addSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, localData, mappedDataVector, fittableGlobalDofsOnly);
  }
  
  // map side data
  int sideCount = _sideMaps.size();
  for (unsigned sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++)
  {
    bool skipSide = (_sideOrdinalToMap != -1) && (sideOrdinal != _sideOrdinalToMap);
    if (skipSide) continue;
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++)
    {
      int varID = sideMapIt->first;
      bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
      if (skipVar) continue;
      BasisMap basisMap = sideMapIt->second;
      addSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, localData, mappedDataVector, fittableGlobalDofsOnly);
    }
  }
}

FieldContainer<double> LocalDofMapper::mapLocalData(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly)
{
  unsigned dofCount;
  if (_varIDToMap == -1)
  {
    dofCount = _dofOrdering->totalDofs();
  }
  else
  {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  if ((localData.rank() != 1) && (localData.rank() != 2))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must be rank 1 or rank 2");
  }
  if (localData.dimension(0) != dofCount)
  {
    cout << "data's dimension 0 must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData dimension 0 must match dofCount.");
  }
  if (localData.rank()==2)
  {
    if (localData.dimension(1) != dofCount)
    {
      cout << "data's dimension 1, if present, must match dofCount.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "data dimension 1 must match dofCount.");
    }
  }
  if (localData.rank()==2)
  {
    return mapLocalDataMatrix(localData, fittableGlobalDofsOnly);
  }
  
  int mappedDofCount =  _globalIndexToOrdinal.size();
  Teuchos::Array<int> dim;
  localData.dimensions(dim);
  dim[0] = mappedDofCount;
  FieldContainer<double> mappedData(dim);
  mapLocalDataVector(localData, fittableGlobalDofsOnly, mappedData);
  return mappedData;
}

void LocalDofMapper::mapLocalDataSide(const FieldContainer<double> &localData, FieldContainer<double> &mappedData, bool fittableGlobalDofsOnly, int sideOrdinal)
{
  unsigned dofCount;
  if (_varIDToMap == -1)
  {
    dofCount = _dofOrdering->totalDofs();
  }
  else
  {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  if ((localData.rank() != 1) && (localData.rank() != 2))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must be rank 1 or 2");
  }
  if (localData.dimension(localData.rank()-1) != dofCount)
  {
    cout << "localData's final dimension must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData's final dimension must match dofCount.");
  }
  
  int mappedDofCount =  _globalIndexToOrdinal.size();
  
  if (mappedData.dimension(mappedData.rank()-1) != mappedDofCount)
  {
    cout << "mappedData's final dimension 0 must match mappedDofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "mappedData's dimension 0 must match mappedDofCount.");
  }
  
  // map side data
  bool skipSide = (_sideOrdinalToMap != -1) && (sideOrdinal != _sideOrdinalToMap);
  if (skipSide) return;
  for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++)
  {
    int varID = sideMapIt->first;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap basisMap = sideMapIt->second;
    addSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, localData, mappedData, fittableGlobalDofsOnly);
  }
}

void LocalDofMapper::mapLocalDataVolume(const FieldContainer<double> &localData, FieldContainer<double> &mappedData, bool fittableGlobalDofsOnly)
{
  unsigned dofCount;
  if (_varIDToMap == -1)
  {
    dofCount = _dofOrdering->totalDofs();
  }
  else
  {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  if ((localData.rank() != 1) && (localData.rank() != 2))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData must be rank 1 or 2");
  }
  if (localData.dimension(localData.rank()-1) != dofCount)
  {
    cout << "localData's final dimension must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "localData's final dimension must match dofCount.");
  }
  
  int mappedDofCount =  _globalIndexToOrdinal.size();
  
  if (mappedData.dimension(0) != mappedDofCount)
  {
    cout << "mappedData's dimension 0 must match mappedDofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "mappedData's dimension 0 must match mappedDofCount.");
  }
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++)
  {
    int varID = volumeMapIt->first;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap basisMap = volumeMapIt->second;
    int volumeSideIndex = 0;
    addSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, localData, mappedData, fittableGlobalDofsOnly);
  }
}

FieldContainer<double> LocalDofMapper::mapGlobalCoefficients(const FieldContainer<double> &globalCoefficients)
{
  unsigned dofCount;
  if (_varIDToMap == -1)
  {
    dofCount = _globalIndexToOrdinal.size();
  }
  else
  {
    dofCount = _dofOrdering->getBasisCardinality(_varIDToMap, _sideOrdinalToMap);
  }
  if ((globalCoefficients.rank() != 1) && (globalCoefficients.rank() != 2))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalCoefficients must be rank 1 or 2");
  }
  if (globalCoefficients.dimension(globalCoefficients.rank()-1) != dofCount)
  {
    cout << "globalCoefficients's final dimension must match dofCount.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "globalCoefficients final dimension must match dofCount.");
  }
  
  int mappedDofCount = _dofOrdering->totalDofs();
  Teuchos::Array<int> dim;
  globalCoefficients.dimensions(dim);
  dim[dim.size()-1] = mappedDofCount;
  FieldContainer<double> localCoefficients(dim);
  
  // map volume data
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++)
  {
    int varID = volumeMapIt->first;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap basisMap = volumeMapIt->second;
    int volumeSideOrdinal = 0;
    addReverseSubBasisMapVectorContribution(varID, volumeSideOrdinal, basisMap, globalCoefficients, localCoefficients);
  }
  
  // map side data
  int sideCount = _sideMaps.size();
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    bool skipSide = (_sideOrdinalToMap != -1) && (sideOrdinal != _sideOrdinalToMap);
    if (skipSide) continue;
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++)
    {
      int varID = sideMapIt->first;
      bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
      if (skipVar) continue;
      BasisMap basisMap = sideMapIt->second;
      addReverseSubBasisMapVectorContribution(varID, sideOrdinal, basisMap, globalCoefficients, localCoefficients);
    }
  }
  return localCoefficients;
}

void LocalDofMapper::printMappingReport()
{
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
      for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++)
      {
        SubBasisDofMapperPtr subBasisDofMapper = *subBasisMapIt;
        Camellia::print("local dof ordinals", subBasisDofMapper->basisDofOrdinalFilter());
        Camellia::print("global ordinals   ", subBasisDofMapper->mappedGlobalDofOrdinals());
        
        SubBasisDofMatrixMapper * matrixMapper = dynamic_cast< SubBasisDofMatrixMapper* >(subBasisDofMapper.get());
        if (matrixMapper != NULL)
          cout << "constraint matrix:\n" << matrixMapper->constraintMatrix();
      }
    }
  }
}

void LocalDofMapper::reverseParity(set<int> fluxVarIDs, set<unsigned int> sideOrdinals)
{
  for (set<int>::iterator fluxIt = fluxVarIDs.begin(); fluxIt != fluxVarIDs.end(); fluxIt++)
  {
    int fluxID = *fluxIt;
    for (set<unsigned>::iterator sideOrdinalIt = sideOrdinals.begin(); sideOrdinalIt != sideOrdinals.end(); sideOrdinalIt++)
    {
      unsigned sideOrdinal = *sideOrdinalIt;
      BasisMap basisMap = _sideMaps[sideOrdinal][fluxID];
      BasisMap negatedBasisMap;
      for (BasisMap::iterator subMapIt = basisMap.begin(); subMapIt != basisMap.end(); subMapIt++)
      {
        negatedBasisMap.push_back((*subMapIt)->negatedDofMapper());
      }
      _sideMaps[sideOrdinal][fluxID] = negatedBasisMap;
    }
  }
  _localCoefficientsFitMatrix.resize(0); // this will need to be recomputed
}