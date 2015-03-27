//
//  LocalDofMapper.h
//  Camellia-debug
//
//  Created by Nate Roberts on 2/13/14.
//
//

#ifndef __Camellia_debug__LocalDofMapper__
#define __Camellia_debug__LocalDofMapper__

#include "TypeDefs.h"

#include <iostream>

#include "DofOrdering.h"
#include "SubBasisDofMapper.h"

#include <Epetra_CrsMatrix.h>

class LocalDofMapper { // maps a whole trial ordering
  DofOrderingPtr _dofOrdering;
  typedef vector< SubBasisDofMapperPtr > BasisMap; // taken together, these maps map a whole basis
  map< int, BasisMap > _volumeMaps; // keys are var IDs (fields)
  vector< map< int, BasisMap > > _sideMaps; // outer index is side ordinal; map keys are var IDs
  map< GlobalIndexType, unsigned > _globalIndexToOrdinal; // maps from GlobalIndex to the ordinal in our globalData container (on the present implementation, the global indices will always be in numerical order)
  
  vector< set<GlobalIndexType> > _fittableGlobalDofOrdinalsOnSides;
  set<GlobalIndexType> _fittableGlobalDofOrdinalsInVolume;
  vector< GlobalIndexType > _fittableGlobalIndices;
  
  int _sideOrdinalToMap;
  int _varIDToMap;
  
  void filterData(const vector<int> dofIndices, const FieldContainer<double> &data, FieldContainer<double> &filteredData);
  void addSubBasisMapVectorContribution(int varID, int sideIndex, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData, bool fittableGlobalDofsOnly);
//  void addSubBasisMapMatrixContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &localData, FieldContainer<double> &globalData);
  void addReverseSubBasisMapVectorContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData);
//  void addReverseSubBasisMapMatrixContribution(int varID, int sideOrdinal, BasisMap basisMap, const FieldContainer<double> &globalData, FieldContainer<double> &localData);
  FieldContainer<double> mapLocalDataMatrix(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly);
  
  FieldContainer<double> _localCoefficientsFitMatrix; // used for fitLocalCoefficients
  
public:
  LocalDofMapper(DofOrderingPtr dofOrdering, map< int, BasisMap > volumeMaps,
                 set<GlobalIndexType> fittableGlobalDofOrdinalsInVolume,
                 vector< map< int, BasisMap > > sideMaps,
                 vector< set<GlobalIndexType> > fittableGlobalDofOrdinalsOnSides,
                 set<GlobalIndexType> unmappedGlobalDofOrdinals = set<GlobalIndexType>(), // extra dof ordinals which aren't mapped as such but should be included in the mapper (used in GMGOperator)
                 int varIDToMap = -1, int sideOrdinalToMap = -1);
  
  FieldContainer<double> mapLocalData(const FieldContainer<double> &localData, bool fittableGlobalDofsOnly);
  void mapLocalDataSide(const FieldContainer<double> &localData, FieldContainer<double> &mappedData, bool fittableGlobalDofsOnly, int sideOrdinal);
  void mapLocalDataVolume(const FieldContainer<double> &localData, FieldContainer<double> &mappedData, bool fittableGlobalDofsOnly);
  
  FieldContainer<double> fitLocalCoefficients(const FieldContainer<double> &localCoefficients); // solves normal equations (if the localCoefficients are in the range of the global-to-local operator, then the returned coefficients will be the preimage of localCoefficients under that operator)
  FieldContainer<double> mapGlobalCoefficients(const FieldContainer<double> &globalCoefficients);
  
  vector<GlobalIndexType> fittableGlobalIndices();
  vector<GlobalIndexType> globalIndices();
  
  void printMappingReport();
  
  void reverseParity(set<int> fluxVarIDs, set<unsigned> sideOrdinals); // multiplies corresponding sideMaps by -1
};
typedef Teuchos::RCP<LocalDofMapper> LocalDofMapperPtr;

#endif /* defined(__Camellia_debug__LocalDofMapper__) */
