//
//  SubBasisDofMatrixMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#ifndef __Camellia_debug__SubBasisDofMatrixMapper__
#define __Camellia_debug__SubBasisDofMatrixMapper__

#include "TypeDefs.h"

#include <iostream>

#include "Intrepid_FieldContainer.hpp"
#include "SubBasisDofMapper.h"

namespace Camellia
{
class SubBasisDofMatrixMapper : public SubBasisDofMapper   // subclass that multiplies by a matrix (as opposed to applying a permutation)
{
  std::set<unsigned> _basisDofOrdinalFilter;
  std::vector<GlobalIndexType> _mappedGlobalDofOrdinals;
  Intrepid::FieldContainer<double> _constraintMatrix;
  
  void mapSubBasisDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &subBasisData, const std::map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                          bool fittableDofsOnly, const std::set<GlobalIndexType> &fittableDofIndices, Intrepid::FieldContainer<double> &globalData);
public:
  SubBasisDofMatrixMapper(const std::set<unsigned> &basisDofOrdinalFilter,
                          const std::vector<GlobalIndexType> &mappedGlobalDofOrdinals,
                          const Intrepid::FieldContainer<double> &constraintMatrix);
  const set<unsigned> &basisDofOrdinalFilter();
  
  //! returns true if the sub basis map is a simple permutation, negated  -- SubBasisDofMatrixMapper always returns false
  bool isNegatedPermutation();
  
  //! returns true if the sub basis map is a simple permutation -- SubBasisDofMatrixMapper always returns false
  bool isPermutation();
  
  Intrepid::FieldContainer<double> mapData(bool transposeConstraint, Intrepid::FieldContainer<double> &localData, bool applyOnLeftOnly = false);
  
  void mapDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &allLocalData, const vector<int> &basisOrdinalsInLocalData,
                                  const map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                  bool fittableDofsOnly, const set<GlobalIndexType> &fittableDofIndices, Intrepid::FieldContainer<double> &globalData);
  void mapDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &wholeBasisData, const std::map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                  bool fittableDofsOnly, const std::set<GlobalIndexType> &fittableDofIndices, Intrepid::FieldContainer<double> &globalData);

  const std::vector<GlobalIndexType> &mappedGlobalDofOrdinals();
  std::set<GlobalIndexType> mappedGlobalDofOrdinalsForBasisOrdinals(std::set<unsigned> &basisDofOrdinals);

  SubBasisDofMapperPtr negatedDofMapper();

  const Intrepid::FieldContainer<double> &constraintMatrix();
  Intrepid::FieldContainer<double> getConstraintMatrix();
};
}

#endif /* defined(__Camellia_debug__SubBasisDofMatrixMapper__) */
