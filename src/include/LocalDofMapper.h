//
//  LocalDofMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#ifndef __Camellia_debug__LocalDofMapper__
#define __Camellia_debug__LocalDofMapper__

#include <iostream>

#include "DofOrdering.h"
#include "SubBasisDofMapper.h"
#include "IndexType.h"

class LocalDofMapper { // maps a whole trial ordering
  DofOrderingPtr _dofOrdering;
  typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
  map< int, BasisMap > _volumeMaps; // keys are var IDs (fields)
  vector< map< int, BasisMap > > _sideMaps; // outer index is side ordinal; map keys are var IDs
  map< GlobalIndexType, unsigned > _globalIndexToOrdinal; // maps from GlobalIndex to the ordinal in our globalData container (on the present implementation, the global indices will always be in numerical order)
  
  void filterData(const vector<int> dofIndices, const FieldContainer<double> &data, FieldContainer<double> &filteredData, int rankToFilter);
  void addSubBasisMapContribution(int varID, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData, int rankToMap);
public:
  LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps, vector< map< int, BasisMap > > sideMaps);
  FieldContainer<double> mapData(const FieldContainer<double> &localData, int rankToMap = -1);
};
typedef Teuchos::RCP<LocalDofMapper> LocalDofMapperPtr;

#endif /* defined(__Camellia_debug__LocalDofMapper__) */
