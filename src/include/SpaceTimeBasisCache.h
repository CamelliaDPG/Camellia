//
//  SpaceTimeBasisCache.h
//  Camellia
//
//  Created by Nate Roberts on 3/11/15.
//
//

#ifndef __Camellia__SpaceTimeBasisCache__
#define __Camellia__SpaceTimeBasisCache__

#include "BasisCache.h"

class SpaceTimeBasisCache : public BasisCache {
  
  BasisCachePtr _spatialCache, _temporalCache;

  // side constructor:
  SpaceTimeBasisCache(int sideIndex, BasisCachePtr volumeCache, int trialDegree, int testDegree);
  
  Intrepid::EOperator spaceOp(Camellia::EOperator op);
  Intrepid::EOperator timeOp(Camellia::EOperator op);
public:
  // volume constructor:
  SpaceTimeBasisCache(MeshPtr spaceTimeMesh, ElementTypePtr spaceTimeElementType,
                      FieldContainer<double> &physicalNodesSpatial,
                      FieldContainer<double> &physicalNodesTemporal,
                      const std::vector<GlobalIndexType> &cellIDs,
                      bool testVsTest, int cubatureDegreeEnrichment);
  
  void createSideCaches();
  
  BasisCachePtr getSpatialBasisCache();
  BasisCachePtr getTemporalBasisCache();
  
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  
  // side variants:
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Intrepid::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
};


#endif /* defined(__Camellia__SpaceTimeBasisCache__) */
