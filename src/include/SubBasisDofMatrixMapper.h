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

using namespace std;
using namespace Intrepid;

class SubBasisDofMatrixMapper : public SubBasisDofMapper { // subclass that multiplies by a matrix (as opposed to applying a permutation)
  set<unsigned> _basisDofOrdinalFilter;
  vector<GlobalIndexType> _mappedGlobalDofOrdinals;
  FieldContainer<double> _constraintMatrix;
public:
  SubBasisDofMatrixMapper(const set<unsigned> &basisDofOrdinalFilter,
                          const vector<GlobalIndexType> &mappedGlobalDofOrdinals,
                          const FieldContainer<double> &constraintMatrix);
  const set<unsigned> &basisDofOrdinalFilter();
  FieldContainer<double> mapData(bool transposeConstraint, FieldContainer<double> &localData);
  vector<GlobalIndexType> mappedGlobalDofOrdinals();
  
  const FieldContainer<double> &constraintMatrix();
  FieldContainer<double> getConstraintMatrix();
};

#endif /* defined(__Camellia_debug__SubBasisDofMatrixMapper__) */
