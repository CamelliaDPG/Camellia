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
#include "TensorBasis.h"

class SpaceTimeBasisCache : public BasisCache {
  typedef Teuchos::RCP< Intrepid::FieldContainer<double> > FCPtr;
  typedef Teuchos::RCP< const Intrepid::FieldContainer<double> > constFCPtr;
  
  BasisCachePtr _spatialCache, _temporalCache;

  // side constructor:
  SpaceTimeBasisCache(int sideIndex, Teuchos::RCP<SpaceTimeBasisCache> volumeCache, int trialDegree, int testDegree);

  Camellia::EOperator spaceOp(Camellia::EOperator op);
  Camellia::EOperator timeOp(Camellia::EOperator op);
  
  Intrepid::EOperator spaceOpForSizing(Camellia::EOperator op);
  Intrepid::EOperator timeOpForSizing(Camellia::EOperator op);

  void getSpaceTimeCubatureDegrees(ElementTypePtr spaceTimeType, int &spaceCubature, int &timeCubature);
//  void getSpaceTimeElementTypes(ElementTypePtr spaceTimeType, ElementTypePtr &spaceType, ElementTypePtr &timeType);
  
  constFCPtr getTensorBasisValues(TensorBasis<double>* tensorBasis,
                                  int fieldIndex, int pointIndex,
                                  constFCPtr spatialValues,
                                  constFCPtr temporalValues,
                                  Intrepid::EOperator spaceOp,
                                  Intrepid::EOperator timeOp) const;
protected:
  virtual void createSideCaches();
public:
  // volume constructors:
  SpaceTimeBasisCache(MeshPtr spaceTimeMesh, ElementTypePtr spaceTimeElementType,
                      const FieldContainer<double> &physicalNodesSpatial,
                      const FieldContainer<double> &physicalNodesTemporal,
                      const FieldContainer<double> &physicalNodesSpaceTime,
                      const std::vector<GlobalIndexType> &cellIDs,
                      bool testVsTest, int cubatureDegreeEnrichment);
  SpaceTimeBasisCache(const FieldContainer<double> &physicalNodesSpatial,
                      const FieldContainer<double> &physicalNodesTemporal,
                      const FieldContainer<double> &physicalCellNodes,
                      CellTopoPtr cellTopo, int cubDegree);
  SpaceTimeBasisCache(CellTopoPtr cellTopo,
                      const FieldContainer<double> &physicalNodesSpatial,
                      const FieldContainer<double> &physicalNodesTemporal,
                      const FieldContainer<double> &physicalCellNodes,
                      const FieldContainer<double> &refCellPointsSpatial,
                      const FieldContainer<double> &refCellPointsTemporal,
                      const FieldContainer<double> &refCellPoints);
  
  
  BasisCachePtr getSpatialBasisCache();
  BasisCachePtr getTemporalBasisCache();
  
  // for now, setRefCellPoints() will throw an exception: we need to get ref cell points for
  // space and time separately.
  virtual void setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell);
  virtual void setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell,
                                const Intrepid::FieldContainer<double> &cubatureWeights);
  
  virtual constFCPtr getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual constFCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual constFCPtr getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  
  static void getTensorialComponentPoints(CellTopoPtr spaceTimeTopo, const FieldContainer<double> &tensorPoints,
                                          FieldContainer<double> &spatialPoints, FieldContainer<double> &temporalPoints);
};


#endif /* defined(__Camellia__SpaceTimeBasisCache__) */
