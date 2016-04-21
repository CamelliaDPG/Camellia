//
//  SubBasisDofPermutationMapper.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/14/14.
//
//

#include "SubBasisDofPermutationMapper.h"

#include <set>
using namespace std;

using namespace Intrepid;
using namespace Camellia;

SubBasisDofPermutationMapper::SubBasisDofPermutationMapper(const set<int> &basisDofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals,
    bool negate)
{
  _basisDofOrdinalFilter = basisDofOrdinalFilter;
  _globalDofOrdinals = globalDofOrdinals;
  _inversePermutation = vector<int>(_basisDofOrdinalFilter.size());
  map<GlobalIndexType,int> permutation_map;
  for (int i=0; i<_basisDofOrdinalFilter.size(); i++)
  {
    permutation_map[globalDofOrdinals[i]] = i;
  }
  int i=0;
  for (map<GlobalIndexType,int>::iterator permIt = permutation_map.begin(); permIt != permutation_map.end(); permIt++)
  {
    _inversePermutation[i++] = permIt->second;
  }
  _negate = negate;
}

const set<int> & SubBasisDofPermutationMapper::basisDofOrdinalFilter()
{
  return _basisDofOrdinalFilter;
}
FieldContainer<double> SubBasisDofPermutationMapper::mapData(bool transposeConstraintMatrix, FieldContainer<double> &data, bool applyOnLeftOnly)
{
  if (transposeConstraintMatrix)
  {
    // data comes in ordered by basisDofOrdinal
    // caller will interpret the data by virtue of the globalDofOrdinals vector--the permutation is implicit in that
    return data;
  }
  else
  {
    // data comes in ordered by GlobalDofOrdinal -- use inversePermutation to reorder
    // data should come out such that the ordering corresponds to that of the _basisDofOrdinalFilter
    Teuchos::Array<int> dim;
    data.dimensions(dim);
    FieldContainer<double> dataPermuted(dim);
    if (dim.size() == 1)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_inversePermutation.size() != dim[0], std::invalid_argument, "unexpected data length");
      for (int i=0; i<dim[0]; i++)
      {
        if (!_negate)
          dataPermuted(_inversePermutation[i]) = data(i);
        else
          dataPermuted(_inversePermutation[i]) = -data(i);
      }
    }
    else if (dim.size() == 2)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(_inversePermutation.size() != dim[0], std::invalid_argument, "unexpected dimension 0");
      TEUCHOS_TEST_FOR_EXCEPTION(_inversePermutation.size() != dim[1], std::invalid_argument, "unexpected dimension 1");
      for (int i=0; i<dim[0]; i++)
      {
        for (int j=0; j<dim[1]; j++)
        {
          if (!applyOnLeftOnly)
          {
            if (!_negate)
              dataPermuted(_inversePermutation[i],_inversePermutation[j]) = data(i,j);
            else
              dataPermuted(_inversePermutation[i],_inversePermutation[j]) = -data(i,j);
          }
          else
          {
            // applying on left only amounts to permuting the rows
            if (!_negate)
              dataPermuted(_inversePermutation[i],j) = data(i,j);
            else
              dataPermuted(_inversePermutation[i],j) = -data(i,j);
          }
        }
      }
    }
    return dataPermuted;
  }
}

void SubBasisDofPermutationMapper::mapDataIntoGlobalContainer(const FieldContainer<double> &wholeBasisData, const map<GlobalIndexType, unsigned int> &globalIndexToOrdinal,
    bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, FieldContainer<double> &globalData)
{
  // like calling mapData, above, with transposeConstraintMatrix = true

  const set<int>* basisOrdinalFilter = &this->basisDofOrdinalFilter();
  vector<int> dofIndices(basisOrdinalFilter->begin(),basisOrdinalFilter->end());

  for (int sbGlobalOrdinal_i=0; sbGlobalOrdinal_i<_globalDofOrdinals.size(); sbGlobalOrdinal_i++)
  {
    GlobalIndexType globalIndex_i = _globalDofOrdinals[sbGlobalOrdinal_i];
    if (fittableDofsOnly && (fittableDofIndices.find(globalIndex_i) == fittableDofIndices.end())) continue; // skip this one
    unsigned globalOrdinal_i = globalIndexToOrdinal.find(globalIndex_i)->second;
    globalData[globalOrdinal_i] += wholeBasisData[dofIndices[sbGlobalOrdinal_i]];
  }
}

void SubBasisDofPermutationMapper::mapDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &allLocalData, const vector<int> &basisOrdinalsInLocalData,
                                                              const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                                              bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices,
                                                              Intrepid::FieldContainer<double> &globalData)
{
  // like calling mapData, above, with transposeConstraintMatrix = true
  
  const set<int>* basisOrdinalFilter = &this->basisDofOrdinalFilter();
  vector<int> dofIndices(basisOrdinalFilter->begin(),basisOrdinalFilter->end());
  
  for (int sbGlobalOrdinal_i=0; sbGlobalOrdinal_i<_globalDofOrdinals.size(); sbGlobalOrdinal_i++)
  {
    GlobalIndexType globalIndex_i = _globalDofOrdinals[sbGlobalOrdinal_i];
    if (fittableDofsOnly && (fittableDofIndices.find(globalIndex_i) == fittableDofIndices.end())) continue; // skip this one
    unsigned globalOrdinal_i = globalIndexToOrdinal.find(globalIndex_i)->second;
    globalData[globalOrdinal_i] += allLocalData[basisOrdinalsInLocalData[dofIndices[sbGlobalOrdinal_i]]];
  }
}

FieldContainer<double> SubBasisDofPermutationMapper::getConstraintMatrix()
{
  // identity (permutation comes by virtue of ordering in globalDofOrdinals)
  FieldContainer<double> matrix(_basisDofOrdinalFilter.size(),_globalDofOrdinals.size());
  for (int i=0; i<_basisDofOrdinalFilter.size(); i++)
  {
    if (!_negate)
      matrix(i,i) = 1;
    else
      matrix(i,i) = -1;
  }
  return matrix;
}

bool SubBasisDofPermutationMapper::isNegatedPermutation()
{
  return _negate;
}

bool SubBasisDofPermutationMapper::isPermutation()
{
  return true;
}

const vector<GlobalIndexType> & SubBasisDofPermutationMapper::mappedGlobalDofOrdinals()
{
  return _globalDofOrdinals;
}

set<GlobalIndexType> SubBasisDofPermutationMapper::mappedGlobalDofOrdinalsForBasisOrdinals(set<int> &basisDofOrdinals)
{
  int i=0;
  set<GlobalIndexType> globalIndices;
  for (int myBasisDofOrdinal : _basisDofOrdinalFilter)
  {
    if (basisDofOrdinals.find(myBasisDofOrdinal) != basisDofOrdinals.end())
    {
      globalIndices.insert(_globalDofOrdinals[i]);
    }
    i++;
  }
  return globalIndices;
}

SubBasisDofMapperPtr SubBasisDofPermutationMapper::negatedDofMapper()
{
  return Teuchos::rcp( new SubBasisDofPermutationMapper(_basisDofOrdinalFilter, _globalDofOrdinals, !_negate) );
}

SubBasisDofMapperPtr SubBasisDofPermutationMapper::restrictDofOrdinalFilter(const set<int> &newDofOrdinalFilter)
{
  set<int> newBasisDofOrdinalFilter; // intersection of newDofOrdinalFilter and _basisDofOrdinalFilter
  vector<GlobalIndexType> newMappedGlobalDofOrdinals;
  
  int globalDofOrdinal_i = 0;
  for (int basisDofOrdinal : _basisDofOrdinalFilter)
  {
    if (newDofOrdinalFilter.find(basisDofOrdinal) != newDofOrdinalFilter.end())
    {
      newBasisDofOrdinalFilter.insert(basisDofOrdinal);
      newMappedGlobalDofOrdinals.push_back(_globalDofOrdinals[globalDofOrdinal_i]);
    }
    globalDofOrdinal_i++;
  }
  return Teuchos::rcp( new SubBasisDofPermutationMapper(newBasisDofOrdinalFilter, newMappedGlobalDofOrdinals, _negate));
}

SubBasisDofMapperPtr SubBasisDofPermutationMapper::restrictGlobalDofOrdinals(const set<GlobalIndexType> &newGlobalDofOrdinals)
{
  set<int> newBasisDofOrdinalFilter;
  vector<GlobalIndexType> newMappedGlobalDofOrdinals; // intersection of newGlobalDofOrdinals and _globalDofOrdinals
  
  int globalDofOrdinal_i = 0;
  for (int basisDofOrdinal : _basisDofOrdinalFilter)
  {
    if (newGlobalDofOrdinals.find(_globalDofOrdinals[globalDofOrdinal_i]) != newGlobalDofOrdinals.end())
    {
      newBasisDofOrdinalFilter.insert(basisDofOrdinal);
      newMappedGlobalDofOrdinals.push_back(_globalDofOrdinals[globalDofOrdinal_i]);
    }
    globalDofOrdinal_i++;
  }
  return Teuchos::rcp( new SubBasisDofPermutationMapper(newBasisDofOrdinalFilter, newMappedGlobalDofOrdinals, _negate));
}