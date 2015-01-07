//
//  SubBasisDofMatrixMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#ifndef __Camellia_debug__SubBasisDofMatrixMapper__
#define __Camellia_debug__SubBasisDofMatrixMapper__

#include <iostream>

#include "Intrepid_FieldContainer.hpp"
#include "SubBasisDofMapper.h"
#include "IndexType.h"

class SubBasisDofMatrixMapper : public SubBasisDofMapper { // subclass that multiplies by a matrix (as opposed to applying a permutation)
  std::set<unsigned> _basisDofOrdinalFilter;
  std::vector<GlobalIndexType> _mappedGlobalDofOrdinals;
  Intrepid::FieldContainer<double> _constraintMatrix;
public:
  SubBasisDofMatrixMapper(const std::set<unsigned> &basisDofOrdinalFilter,
                          const std::vector<GlobalIndexType> &mappedGlobalDofOrdinals,
                          const Intrepid::FieldContainer<double> &constraintMatrix);
  const set<unsigned> &basisDofOrdinalFilter();
  Intrepid::FieldContainer<double> mapData(bool transposeConstraint, Intrepid::FieldContainer<double> &localData, bool applyOnLeftOnly = false);
  void mapDataIntoGlobalContainer(const Intrepid::FieldContainer<double> &wholeBasisData, const std::map<GlobalIndexType, unsigned> &globalIndexToOrdinal,
                                  bool fittableDofsOnly, const std::set<GlobalIndexType> &fittableDofIndices, Intrepid::FieldContainer<double> &globalData);
  
  std::vector<GlobalIndexType> mappedGlobalDofOrdinals();
  
  SubBasisDofMapperPtr negatedDofMapper();
  
  const Intrepid::FieldContainer<double> &constraintMatrix();
  Intrepid::FieldContainer<double> getConstraintMatrix();
};

#endif /* defined(__Camellia_debug__SubBasisDofMatrixMapper__) */
