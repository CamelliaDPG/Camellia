//
//  SubBasisDofMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#ifndef Camellia_debug_SubBasisDofMapper_h
#define Camellia_debug_SubBasisDofMapper_h



#include <set.h>
#include "Intrepid_FieldContainer.hpp"
#include "Teuchos_RCP.hpp"

#include "IndexType.h"

using namespace std;
using namespace Intrepid;

class SubBasisDofMapper;
typedef Teuchos::RCP<SubBasisDofMapper> SubBasisDofMapperPtr;

class SubBasisDofMapper {
public:
  virtual const set<unsigned> &basisDofOrdinalFilter() = 0;
  virtual FieldContainer<double> mapData(bool transposeConstraint, const FieldContainer<double> &localData, bool transposeData) = 0;
  virtual vector<GlobalIndexType> mappedGlobalDofOrdinals() = 0;
  
  virtual ~SubBasisDofMapper();
  
  static SubBasisDofMapperPtr subBasisDofMapper(const set<unsigned> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals);
  static SubBasisDofMapperPtr subBasisDofMapper(const set<unsigned> &dofOrdinalFilter, const vector<GlobalIndexType> &globalDofOrdinals, const FieldContainer<double> &constraintMatrix);
  
//  static SubBasisDofMapperPtr subBasisDofMapper(); // determines if the constraint is a permutation--if it is, then 
};

#endif
