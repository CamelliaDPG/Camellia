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
using namespace std;

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

void LocalDofMapper::addSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap &basisMap,
                                                      const FieldContainer<double> &localData,
                                                      FieldContainer<double> &globalData,
                                                      bool fittableGlobalDofsOnly)
{
  bool volumeRestrictedToSide = false;
  map<int,int> basisDofOrdinalReverseLookup; // for volume variables restricted to side
  
  if (!_dofOrdering->hasBasisEntry(varID, sideOrdinal))
  {
    if (_volumeMaps.find(varID) == _volumeMaps.end())
    {
      // not a volume variable restricted to the side
      return; // no contribution
    }
    else
    {
      volumeRestrictedToSide = true;
      BasisPtr volumeBasis = _dofOrdering->getBasis(varID);
      set<int> basisDofOrdinalsForSide = volumeBasis->dofOrdinalsForSide(sideOrdinal);
      int i = 0;
      for (int basisDofOrdinal : basisDofOrdinalsForSide)
      {
        basisDofOrdinalReverseLookup[basisDofOrdinal] = i++;
      }
      // we don't yet support this mode when the DofMapper has _varIDToMap = -1
      TEUCHOS_TEST_FOR_EXCEPTION(_varIDToMap == -1, std::invalid_argument, "Restriction of volume basis to side only supported on LocalDofMapper with _varIDToMap specified");
    }
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
  
  for (SubBasisDofMapperPtr subBasisDofMapper : basisMap)
  {
    // Recently added (2-12-16): now treat the case where var is a volume basis, and we have a non-interior side selected.  In this case, we skip over the
    // basis ordinals that don't match in permutation, and tweak the definition of localDofIndex_i.
    // Assumption is that the BasisMap has already been appropriately restricted.
    if (subBasisDofMapper->isPermutation())
    {
      // this does make a couple assumptions about the implementation of the permutation mapper.  If that changes, it could break the below.
      
      const vector<GlobalIndexType>* globalDofIndices = &subBasisDofMapper->mappedGlobalDofOrdinals();
      const set<int> *basisDofOrdinals = &subBasisDofMapper->basisDofOrdinalFilter();
      const vector<int>* varDofIndices;
      if (!volumeRestrictedToSide)
        varDofIndices = &_dofOrdering->getDofIndices(varID, sideOrdinal);

      bool negate = subBasisDofMapper->isNegatedPermutation();
      
      int i=-1; // loop counter; start at -1 because I want the i++ at the top of the loop, for code clarity
      for (int basisDofOrdinal_i : *basisDofOrdinals)
      {
        i++;
        GlobalIndexType globalDofIndex_i = (*globalDofIndices)[i];
        if (fittableGlobalDofsOnly && (fittableDofs->find(globalDofIndex_i) == fittableDofs->end()))
        {
          continue; // this basis dof ordinal does not correspond to a fittable global dof -- I'm not sure this will ever happen for a permutation, but logically this is correct
        }
        
        int localDofIndex_i;
        if (_varIDToMap == -1)
          localDofIndex_i = (*varDofIndices)[basisDofOrdinal_i];
        else if (!volumeRestrictedToSide)
          localDofIndex_i = basisDofOrdinal_i;
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(basisDofOrdinalReverseLookup.find(basisDofOrdinal_i) == basisDofOrdinalReverseLookup.end(),
                                     std::invalid_argument, "basis dof ordinal not found in reverse lookup");
          localDofIndex_i = basisDofOrdinalReverseLookup[basisDofOrdinal_i];
        }
        
        if (localData.rank()==1)
        {
          if (!negate)
            globalData(_globalIndexToOrdinal[globalDofIndex_i]) += localData(localDofIndex_i);
          else
            globalData(_globalIndexToOrdinal[globalDofIndex_i]) += -localData(localDofIndex_i);
        }
        else
        {
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "addSubBasisMapVectorContribution only supports rank 1 basis data");

          // the below might be OK for treating 2D data, but it's not getting invoked, so I'm commenting it out for now.
//          int j=0;
//          for (set<int>::iterator basisDofOrdinalIt_j = basisDofOrdinals->begin(); basisDofOrdinalIt_j != basisDofOrdinals->end(); basisDofOrdinalIt_j++, j++)
//          {
//            GlobalIndexType globalDofIndex_j = (*globalDofIndices)[j];
//            if (fittableGlobalDofsOnly && (fittableDofs->find(globalDofIndex_j) == fittableDofs->end()))
//            {
//              continue; // this basis dof ordinal does not correspond to a fittable global dof -- I'm not sure this will ever happen for a permutation, but logically this is correct
//            }
//            
//            int basisDofOrdinal_j = *basisDofOrdinalIt_j;
//            int localDofIndex_j;
//            if (_varIDToMap == -1)
//              localDofIndex_j = (*varDofIndices)[basisDofOrdinal_j];
//            else
//              localDofIndex_j = basisDofOrdinal_j;
//            
//            if (!negate)
//              globalData(_globalIndexToOrdinal[globalDofIndex_i], _globalIndexToOrdinal[globalDofIndex_j]) += localData(localDofIndex_i, localDofIndex_j);
//            else
//              globalData(_globalIndexToOrdinal[globalDofIndex_i], _globalIndexToOrdinal[globalDofIndex_j]) += -localData(localDofIndex_i, localDofIndex_j);
//          }
        }
      }
    }
    else
    {
      if (_varIDToMap == -1)
      {
        const vector<int>* varDofIndices = &_dofOrdering->getDofIndices(varID, sideOrdinal);
        subBasisDofMapper->mapDataIntoGlobalContainer(localData, *varDofIndices, _globalIndexToOrdinal, fittableGlobalDofsOnly, *fittableDofs, globalData);
//          basisData = new FieldContainer<double>(varDofIndices->size());
//          filterData(*varDofIndices, localData, *basisData);
      }
      else
      {
        if (!volumeRestrictedToSide)
        {
          // then localData contains coefficients for the whole basis, not just what subBasisDofMapper maps
          subBasisDofMapper->mapDataIntoGlobalContainer(localData, _globalIndexToOrdinal, fittableGlobalDofsOnly, *fittableDofs, globalData);
        }
        else
        {
          // otherwise, localData contains coefficients for the basis restricted to side
          const set<int>* subBasisDofOrdinals = &subBasisDofMapper->basisDofOrdinalFilter();
          FieldContainer<double> subBasisData(subBasisDofOrdinals->size(),1); // shaped as a matrix, because that's what mapSubBasisDataIntoGlobalContainer expects
          int i = 0;
          for (int basisDofOrdinal : *subBasisDofOrdinals)
          {
            if (basisDofOrdinalReverseLookup.find(basisDofOrdinal) != basisDofOrdinalReverseLookup.end())
            {
              subBasisData[i] = localData[basisDofOrdinalReverseLookup[basisDofOrdinal]];
            }
            i++;
          }
          subBasisDofMapper->mapSubBasisDataIntoGlobalContainer(subBasisData, _globalIndexToOrdinal, fittableGlobalDofsOnly, *fittableDofs, globalData);
        }
      }
    }
  }
}

void LocalDofMapper::addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap &basisMap,
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
  
  for (SubBasisDofMapperPtr subBasisDofMapper : basisMap)
  {
    const vector<GlobalIndexType>* globalDofIndices = &subBasisDofMapper->mappedGlobalDofOrdinals();
//    vector<int> globalOrdinalFilter(globalDofIndices->size());
//    for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices->size(); subBasisGlobalDofOrdinal++)
//    {
//      globalOrdinalFilter[subBasisGlobalDofOrdinal] = _globalIndexToOrdinal[ (*globalDofIndices)[subBasisGlobalDofOrdinal] ];
//    }
    
    // define lambda for globalOrdinal lookup:
    auto globalOrdinal = [&globalDofIndices, this] (int i) -> int
    {
      return _globalIndexToOrdinal[ (*globalDofIndices)[i] ];
    };
    
    if (subBasisDofMapper->isPermutation())
    {
      // this does make a couple assumptions about the implementation of the permutation mapper.  If that changes, it could break the below.
      
      const set<int> *localDofOrdinals = &subBasisDofMapper->basisDofOrdinalFilter();
      bool negate = subBasisDofMapper->isNegatedPermutation();
      
      const vector<int>* localDofIndices = &_dofOrdering->getDofIndices(varID, sideOrdinal);
      
      int i=0;
      for (set<int>::iterator localDofOrdinalIt_i = localDofOrdinals->begin(); localDofOrdinalIt_i != localDofOrdinals->end(); localDofOrdinalIt_i++, i++)
      {
        int localDofOrdinal_i = *localDofOrdinalIt_i;
        int localDofIndex_i = (*localDofIndices)[localDofOrdinal_i];
        
        if (localCoefficients.rank()==1)
        {
          if (!negate)
            localCoefficients(localDofIndex_i) += globalCoefficients(globalOrdinal(i));
          else
            localCoefficients(localDofIndex_i) += -globalCoefficients(globalOrdinal(i));
        }
        else
        {
          int j=0;
          for (set<int>::iterator localDofOrdinalIt_j = localDofOrdinals->begin(); localDofOrdinalIt_j != localDofOrdinals->end(); localDofOrdinalIt_j++, j++)
          {
            int localDofOrdinal_j = *localDofOrdinalIt_j;
            int localDofIndex_j = (*localDofIndices)[localDofOrdinal_j];
            if (! negate)
              localCoefficients(localDofIndex_i, localDofIndex_j) += globalCoefficients(globalOrdinal(i),globalOrdinal(j));
            else
              localCoefficients(localDofIndex_i, localDofIndex_j) += -globalCoefficients(globalOrdinal(i),globalOrdinal(j));
          }
        }
      }
    }
    else
    {
      FieldContainer<double> filteredSubBasisData;
      vector<int> globalOrdinalFilter(globalDofIndices->size());
      for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices->size(); subBasisGlobalDofOrdinal++)
      {
        globalOrdinalFilter[subBasisGlobalDofOrdinal] = globalOrdinal(subBasisGlobalDofOrdinal);
      }
      filterData(globalOrdinalFilter, globalCoefficients, filteredSubBasisData);
      FieldContainer<double> mappedSubBasisData = subBasisDofMapper->mapData(transposeConstraint, filteredSubBasisData, applyOnLeftOnly);
      const set<int>* localDofOrdinals = &subBasisDofMapper->basisDofOrdinalFilter();
      
      const vector<int>* localDofIndices = &_dofOrdering->getDofIndices(varID, sideOrdinal);
      int i=0;
      for (set<int>::const_iterator localDofOrdinalIt_i = localDofOrdinals->begin(); localDofOrdinalIt_i != localDofOrdinals->end();
           localDofOrdinalIt_i++, i++)
      {
        int localDofOrdinal_i = *localDofOrdinalIt_i;
        int localDofIndex_i = (*localDofIndices)[localDofOrdinal_i];
        
        if (localCoefficients.rank()==1)
        {
          localCoefficients(localDofIndex_i) += mappedSubBasisData(i);
        }
        else if (localCoefficients.rank()==2)
        {
          int j=0;
          for (set<int>::const_iterator localDofOrdinalIt_j = localDofOrdinals->begin(); localDofOrdinalIt_j != localDofOrdinals->end(); localDofOrdinalIt_j++, j++)
          {
            int localDofOrdinal_j = *localDofOrdinalIt_j;
            int localDofIndex_j = (*localDofIndices)[localDofOrdinal_j];
            localCoefficients(localDofIndex_i, localDofIndex_j) += mappedSubBasisData(i,j);
          }
        }
      }
    }
  }
}

void LocalDofMapper::addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap &basisMap, const std::map<GlobalIndexType,double> &globalCoefficients,
                                                             FieldContainer<double> &localCoefficients)
{
  bool transposeConstraint = false; // global to local
  
  bool applyOnLeftOnly = true; // mapData() otherwise will do something like L = C G C^T, where C is the constraint matrix, G the global coefficients, and L the local
  
  TEUCHOS_TEST_FOR_EXCEPTION(localCoefficients.rank()!=1, std::invalid_argument, "This version of addReverseSubBasisMapContribution() only supports rank 1 localCoefficients");
  
  if (_varIDToMap != -1)
  {
    cout << "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Error: LocalDofMapper::addReverseSubBasisMapContribution not supported when _varIDToMap is specified.");
  }
  
  //  cout << "*************  varID: " << varID << ", side " << sideOrdinal << "  *************" << endl;
  
  for (SubBasisDofMapperPtr subBasisDofMapper : basisMap)
  {
    const vector<GlobalIndexType>* globalDofIndices = &subBasisDofMapper->mappedGlobalDofOrdinals();
    
    if (subBasisDofMapper->isPermutation())
    {
      // this does make a couple assumptions about the implementation of the permutation mapper.  If that changes, it could break the below.
      
      const set<int> *localDofOrdinals = &subBasisDofMapper->basisDofOrdinalFilter();
      bool negate = subBasisDofMapper->isNegatedPermutation();
      
      const vector<int>* localDofIndices = &_dofOrdering->getDofIndices(varID, sideOrdinal);
      
      int i=0;
      for (set<int>::iterator localDofOrdinalIt_i = localDofOrdinals->begin(); localDofOrdinalIt_i != localDofOrdinals->end(); localDofOrdinalIt_i++, i++)
      {
        int localDofOrdinal_i = *localDofOrdinalIt_i;
        int localDofIndex_i = (*localDofIndices)[localDofOrdinal_i];
        
        if (localCoefficients.rank()==1)
        {
          auto entry = globalCoefficients.find((*globalDofIndices)[i]);
          if (entry == globalCoefficients.end()) continue;
          if (!negate)
            localCoefficients(localDofIndex_i) += entry->second;
          else
            localCoefficients(localDofIndex_i) += - entry->second;
        }
        else
        {
          int j=0;
          for (set<int>::iterator localDofOrdinalIt_j = localDofOrdinals->begin(); localDofOrdinalIt_j != localDofOrdinals->end(); localDofOrdinalIt_j++, j++)
          {
            int localDofOrdinal_j = *localDofOrdinalIt_j;
            int localDofIndex_j = (*localDofIndices)[localDofOrdinal_j];
            auto entry = globalCoefficients.find((*globalDofIndices)[j]);
            if (entry == globalCoefficients.end()) continue;
            if (! negate)
              localCoefficients(localDofIndex_i, localDofIndex_j) += entry->second;
            else
              localCoefficients(localDofIndex_i, localDofIndex_j) += - entry->second;
          }
        }
      }
    }
    else
    {
      FieldContainer<double> filteredSubBasisData(globalDofIndices->size());
      bool nonzerosFound = false;
      for (int subBasisGlobalDofOrdinal=0; subBasisGlobalDofOrdinal<globalDofIndices->size(); subBasisGlobalDofOrdinal++)
      {
        auto entry = globalCoefficients.find((*globalDofIndices)[subBasisGlobalDofOrdinal]);
        if (entry != globalCoefficients.end())
        {
          filteredSubBasisData(subBasisGlobalDofOrdinal) = entry->second;
          nonzerosFound = true;
        }
      }
      if (!nonzerosFound) continue;
      FieldContainer<double> mappedSubBasisData = subBasisDofMapper->mapData(transposeConstraint, filteredSubBasisData, applyOnLeftOnly);
      const set<int>* localDofOrdinals = &subBasisDofMapper->basisDofOrdinalFilter();
      
      const vector<int>* localDofIndices = &_dofOrdering->getDofIndices(varID, sideOrdinal);
      int i=0;
      for (set<int>::const_iterator localDofOrdinalIt_i = localDofOrdinals->begin(); localDofOrdinalIt_i != localDofOrdinals->end();
           localDofOrdinalIt_i++, i++)
      {
        int localDofOrdinal_i = *localDofOrdinalIt_i;
        int localDofIndex_i = (*localDofIndices)[localDofOrdinal_i];
        
        localCoefficients(localDofIndex_i) += mappedSubBasisData(i);
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

//  if ((_varIDToMap == -1) && (_sideOrdinalToMap == -1))
//  {
//    _mode = MAP_ALL;
//  }
//  else if ((_varIDToMap == -1) && (_sideOrdinalToMap != -1))
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "");
//  }
//  else if (volumeMaps.find(_varIDToMap) != volumeMaps.end())
//  {
//    _mode = (_sideOrdinalToMap == -1) ? MAP_VOLUME_VAR : MAP_VOLUME_VAR_SIDE;
//  }
//  else if (sideMaps.find(_varIDToMap) != sideMaps.end())
//  {
//    _mode = (_sideOrdinalToMap == -1) ? MAP_TRACE_VAR : MAP_TRACE_VAR_SIDE;
//  }
//  else
//  {
//    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "_varIDToMap != -1, but it isn't found in either volume or side maps");
//  }
  
  map<GlobalIndexType,set<int>> globalIndexToVarIDSet; // use set for easy construction; in a moment we'll copy into vectors
  set<GlobalIndexType> globalIndices;
  //  int rank = Teuchos::GlobalMPISession::getRank();
  //  int numProcs = Teuchos::GlobalMPISession::getNProc();
  //  if (rank==numProcs-1) cout << "Creating local dof mapper.  Volume Map info:\n";
  for (map< int, BasisMap >::iterator volumeMapIt = _volumeMaps.begin(); volumeMapIt != _volumeMaps.end(); volumeMapIt++)
  {
    int varID = volumeMapIt->first;
    BasisMap basisMap = volumeMapIt->second;
    for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++)
    {
      vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
      globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());
      for (GlobalIndexType subBasisGlobalIndex : subBasisGlobalIndices)
      {
        globalIndexToVarIDSet[subBasisGlobalIndex].insert(varID);
      }
    }
  }
  for (int sideOrdinal=0; sideOrdinal<_sideMaps.size(); sideOrdinal++)
  {
    for (map< int, BasisMap >::iterator sideMapIt = _sideMaps[sideOrdinal].begin(); sideMapIt != _sideMaps[sideOrdinal].end(); sideMapIt++)
    {
      int varID = sideMapIt->first;
      BasisMap basisMap = sideMapIt->second;
      for (vector<SubBasisDofMapperPtr>::iterator subBasisMapIt = basisMap.begin(); subBasisMapIt != basisMap.end(); subBasisMapIt++)
      {
        vector<GlobalIndexType> subBasisGlobalIndices = (*subBasisMapIt)->mappedGlobalDofOrdinals();
        globalIndices.insert(subBasisGlobalIndices.begin(),subBasisGlobalIndices.end());

        for (GlobalIndexType subBasisGlobalIndex : subBasisGlobalIndices)
        {
          globalIndexToVarIDSet[subBasisGlobalIndex].insert(varID);
        }
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
  for (auto entry : globalIndexToVarIDSet)
  {
    _globalIndexToVarIDs[entry.first] = vector<int>(entry.second.begin(),entry.second.end());
  }
}

map<int, GlobalIndexType> LocalDofMapper::getPermutationMap()
{
  TEUCHOS_TEST_FOR_EXCEPTION(! isPermutation(), std::invalid_argument, "getPermutionMap() requires that LocalDofMapper::isPermutation() return true");
  
  map<int, GlobalIndexType> permutationMap;
  
  for (auto volumeMapEntry : _volumeMaps)
  {
    int varID = volumeMapEntry.first;
    BasisMap volumeMap = volumeMapEntry.second;
    for (auto subBasisMap : volumeMap)
    {
      if (subBasisMap->isPermutation())
      {
        const set<int>* basisDofOrdinals = &subBasisMap->basisDofOrdinalFilter();
        const vector<GlobalIndexType>* globalDofOrdinals = &subBasisMap->mappedGlobalDofOrdinals();
        
        TEUCHOS_TEST_FOR_EXCEPTION(basisDofOrdinals->size() != globalDofOrdinals->size(), std::invalid_argument, "Internal error: sizes for permutation should match!");
        
        auto basisOrdinalIt = basisDofOrdinals->begin();
        const vector<int>* dofIndices = &_dofOrdering->getDofIndices(varID);
        for (GlobalIndexType globalDofOrdinal : *globalDofOrdinals)
        {
          int basisDofOrdinal = *basisOrdinalIt;
          int localDofIndex = (*dofIndices)[basisDofOrdinal];
          permutationMap[localDofIndex] = globalDofOrdinal;
          basisOrdinalIt++;
        }
      }
    }
  }
  for (int sideOrdinal = 0; sideOrdinal < _sideMaps.size(); sideOrdinal++)
  {
    for (auto sideMapEntry : _sideMaps[sideOrdinal])
    {
      int varID = sideMapEntry.first;
      BasisMap sideMap = sideMapEntry.second;
      for (auto subBasisMap : sideMap)
      {
        if (subBasisMap->isPermutation())
        {
          const set<int>* basisDofOrdinals = &subBasisMap->basisDofOrdinalFilter();
          const vector<GlobalIndexType>* globalDofOrdinals = &subBasisMap->mappedGlobalDofOrdinals();
          
          TEUCHOS_TEST_FOR_EXCEPTION(basisDofOrdinals->size() != globalDofOrdinals->size(), std::invalid_argument, "Internal error: sizes for permutation should match!");
          
          auto basisOrdinalIt = basisDofOrdinals->begin();
          const vector<int>* dofIndices = &_dofOrdering->getDofIndices(varID, sideOrdinal);
          for (GlobalIndexType globalDofOrdinal : *globalDofOrdinals)
          {
            int basisDofOrdinal = *basisOrdinalIt;
            int localDofIndex = (*dofIndices)[basisDofOrdinal];
            permutationMap[localDofIndex] = globalDofOrdinal;
            basisOrdinalIt++;
          }
        }
      }
    }
  }
  return permutationMap;
}

const vector<GlobalIndexType> &LocalDofMapper::globalIndices()
{
    // the implementation does not assume that the global indices will be in numerical order (which they currently are)

  // we store these lazily
  if (_globalIndices.size() != _globalIndexToOrdinal.size())
  {
    _globalIndices = vector<GlobalIndexType>(_globalIndexToOrdinal.size());
    
    for (pair<GlobalIndexType, unsigned> globalIndexEntry : _globalIndexToOrdinal)
    {
      GlobalIndexType globalIndex = globalIndexEntry.first;
      unsigned ordinal = globalIndexEntry.second;
      _globalIndices[ordinal] = globalIndex;
    }
  }
  
  return _globalIndices;
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
    set<int> dofOrdinals(dofOrdinalsInt.begin(), dofOrdinalsInt.end());
    
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
      set<int> dofOrdinals(dofOrdinalsInt.begin(), dofOrdinalsInt.end());
      
      for (auto subBasisMap : sideMap)
      {
        set<GlobalIndexType> subIndexSet = subBasisMap->mappedGlobalDofOrdinalsForBasisOrdinals(dofOrdinals);
        indexSet.insert(subIndexSet.begin(),subIndexSet.end());
      }
    }
  }
  
  return indexSet;
}

const vector<GlobalIndexType>& LocalDofMapper::fittableGlobalIndices()
{
  if (_fittableGlobalIndices.size() == 0)   // then we have not previously set these...
  {
    // the implementation does not assume that the global indices will be in numerical order (which they currently are)
    const vector<GlobalIndexType>* allGlobalIndices = &globalIndices();
    
    set<GlobalIndexType> fittableIndicesSet;
    for (int sideOrdinal=0; sideOrdinal<_fittableGlobalDofOrdinalsOnSides.size(); sideOrdinal++)
    {
      fittableIndicesSet.insert(_fittableGlobalDofOrdinalsOnSides[sideOrdinal].begin(), _fittableGlobalDofOrdinalsOnSides[sideOrdinal].end());
    }
    
    fittableIndicesSet.insert(_fittableGlobalDofOrdinalsInVolume.begin(),_fittableGlobalDofOrdinalsInVolume.end());
    
    vector<GlobalIndexType> fittableIndices;
    for (GlobalIndexType globalIndex : *allGlobalIndices)
    {
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
  
  if ( (_volumeMaps.find(_varIDToMap) != _volumeMaps.end()) && (_sideOrdinalToMap != VOLUME_INTERIOR_SIDE_ORDINAL) )
  {
    // then we're mapping a partial basis.  As it stands presently,
    
  }
  
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
      
      printMappingReport();
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

bool LocalDofMapper::isPermutation() const
{
  for (auto volumeMapEntry : _volumeMaps)
  {
    BasisMap volumeMap = volumeMapEntry.second;
    for (auto subBasisMap : volumeMap)
    {
      if (! subBasisMap->isPermutation())
      {
        return false;
      }
    }
  }
  for (auto sideEntry : _sideMaps)
  {
    for (auto sideMapEntry : sideEntry)
    {
      BasisMap sideMap = sideMapEntry.second;
      for (auto subBasisMap : sideMap)
      {
        if (! subBasisMap->isPermutation())
        {
          return false;
        }
      }
    }
  }
  return true;
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
  else if (_volumeMaps.find(_varIDToMap) != _volumeMaps.end())
  {
    if (_sideOrdinalToMap == VOLUME_INTERIOR_SIDE_ORDINAL)
      dofCount = _dofOrdering->getBasis(_varIDToMap)->getCardinality();
    else
      dofCount = _dofOrdering->getBasis(_varIDToMap)->dofOrdinalsForSide(_sideOrdinalToMap).size();
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
    // new 2-12-16: respect _sideOrdinalToMap for volume variables, too (important for imposing BCs on volume variables)
    if ((_sideOrdinalToMap == -1) || (_sideOrdinalToMap == VOLUME_INTERIOR_SIDE_ORDINAL))
    {
      // mapping all sides
      int volumeSideIndex = VOLUME_INTERIOR_SIDE_ORDINAL;
      addSubBasisMapVectorContribution(varID, volumeSideIndex, basisMap, localData, mappedDataVector, fittableGlobalDofsOnly);
    }
    else
    {
      addSubBasisMapVectorContribution(varID, _sideOrdinalToMap, basisMap, localData, mappedDataVector, fittableGlobalDofsOnly);
    }
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

void LocalDofMapper::mapLocalDataVector(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly,
                                        const vector<pair<int,vector<int>>> &varIDsAndSideOrdinalsToMap,
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

  for (auto entry : varIDsAndSideOrdinalsToMap)
  {
    int varID = entry.first;
    for (int sideOrdinal : entry.second)
    {
      bool isVolumeVariable = _volumeMaps.find(varID) != _volumeMaps.end();
      BasisMap basisMap;
      if (isVolumeVariable)
        basisMap = _volumeMaps[varID];
      else
        basisMap = _sideMaps[sideOrdinal][varID];
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
  else if (_volumeMaps.find(_varIDToMap) != _volumeMaps.end())
  {
    if (_sideOrdinalToMap == VOLUME_INTERIOR_SIDE_ORDINAL)
      dofCount = _dofOrdering->getBasis(_varIDToMap)->getCardinality();
    else
      dofCount = _dofOrdering->getBasis(_varIDToMap)->dofOrdinalsForSide(_sideOrdinalToMap).size();
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

Intrepid::FieldContainer<double> LocalDofMapper::mapLocalData(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly,
                                                              const vector<pair<int,vector<int>>> &varIDsAndSideOrdinalsToMap)
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
  mapLocalDataVector(localData, fittableGlobalDofsOnly, varIDsAndSideOrdinalsToMap, mappedData);
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
    int volumeSideIndex = VOLUME_INTERIOR_SIDE_ORDINAL;
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
    BasisMap* basisMap = &volumeMapIt->second;
    int volumeSideOrdinal = VOLUME_INTERIOR_SIDE_ORDINAL;
    addReverseSubBasisMapVectorContribution(varID, volumeSideOrdinal, *basisMap, globalCoefficients, localCoefficients);
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
      BasisMap* basisMap = &sideMapIt->second;
      addReverseSubBasisMapVectorContribution(varID, sideOrdinal, *basisMap, globalCoefficients, localCoefficients);
    }
  }
  return localCoefficients;
}

Intrepid::FieldContainer<double> LocalDofMapper::mapGlobalCoefficients(const std::map<GlobalIndexType,double> &globalCoefficients)
{
  int mappedDofCount = _dofOrdering->totalDofs();
  FieldContainer<double> localCoefficients(mappedDofCount);
  
  if (globalCoefficients.size()==0) // 0 result
    return localCoefficients;
  
  set<int> varIDsMapped;
  for (auto entry : globalCoefficients)
  {
    auto varIDsEntry = _globalIndexToVarIDs.find(entry.first);
    varIDsMapped.insert(varIDsEntry->second.begin(), varIDsEntry->second.end());
  }
  
  // map volume data
  for (int varID : varIDsMapped)
  {
    auto entry = _volumeMaps.find(varID);
    if (entry == _volumeMaps.end()) continue;
    bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
    if (skipVar) continue;
    BasisMap* basisMap = &entry->second;
    int volumeSideOrdinal = VOLUME_INTERIOR_SIDE_ORDINAL;
    addReverseSubBasisMapVectorContribution(varID, volumeSideOrdinal, *basisMap, globalCoefficients, localCoefficients);
  }
  
  // map side data
  int sideCount = _sideMaps.size();
  for (unsigned sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++)
  {
    bool skipSide = (_sideOrdinalToMap != -1) && (sideOrdinal != _sideOrdinalToMap);
    if (skipSide) continue;
    for (int varID : varIDsMapped)
    {
      auto entry = _sideMaps[sideOrdinal].find(varID);
      if (entry == _sideMaps[sideOrdinal].end()) continue;
      bool skipVar = (_varIDToMap != -1) && (varID != _varIDToMap);
      if (skipVar) continue;
      BasisMap* basisMap = &entry->second;
      addReverseSubBasisMapVectorContribution(varID, sideOrdinal, *basisMap, globalCoefficients, localCoefficients);
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