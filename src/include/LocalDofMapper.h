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
  
  int _sideOrdinalToMap;
  int _varIDToMap;
  
  void filterData(const vector<int> dofIndices, const FieldContainer<double> &data, FieldContainer<double> &filteredData);
  void addSubBasisMapVectorContribution(int varID, int sideIndex, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData, bool accumulate);
//  void addSubBasisMapMatrixContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData);
  void addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData);
//  void addReverseSubBasisMapMatrixContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData);
  FieldContainer<double> mapDataMatrix(const FieldContainer<double> &localData, bool localToGlobal, bool accumulate);
public:
  LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps, vector< map< int, BasisMap > > sideMaps, int varIDToMap = -1, int sideOrdinalToMap = -1);
  FieldContainer<double> mapData(const FieldContainer<double> &localData, bool localToGlobal = true, bool accumulate = true); // can go global to local
  vector<GlobalIndexType> globalIndices();
  
  void printMappingReport();
};
typedef Teuchos::RCP<LocalDofMapper> LocalDofMapperPtr;

#endif /* defined(__Camellia_debug__LocalDofMapper__) */
