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

namespace Camellia
{
class SpaceTimeBasisCache : public BasisCache
{
  static bool _defaultStoreTransformedValues; // default for new SpaceTimeBasisCaches (see static setter, below)
  
  typedef Teuchos::RCP< Camellia::FieldContainer<double> > FCPtr;
  typedef Teuchos::RCP< const Camellia::FieldContainer<double> > constFCPtr;

  BasisCachePtr _spatialCache, _temporalCache;
  bool _storeTransformedValues;

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
  void getReferencePointComponents(const Camellia::FieldContainer<double> &tensorRefPoints,
                                   Camellia::FieldContainer<double> &spaceRefPoints,
                                   Camellia::FieldContainer<double> &timeRefPoints);
  double getTemporalNodeCoordinateRefSpace(); // for temporal sides
protected:
  virtual void createSideCaches();
public:
  // volume constructors:
  SpaceTimeBasisCache(MeshPtr spaceTimeMesh, ElementTypePtr spaceTimeElementType,
                      const Camellia::FieldContainer<double> &physicalNodesSpatial,
                      const Camellia::FieldContainer<double> &physicalNodesTemporal,
                      const Camellia::FieldContainer<double> &physicalNodesSpaceTime,
                      const std::vector<GlobalIndexType> &cellIDs,
                      bool testVsTest, int cubatureDegreeEnrichment);
  SpaceTimeBasisCache(const Camellia::FieldContainer<double> &physicalNodesSpatial,
                      const Camellia::FieldContainer<double> &physicalNodesTemporal,
                      const Camellia::FieldContainer<double> &physicalCellNodes,
                      CellTopoPtr cellTopo, int cubDegree);
  SpaceTimeBasisCache(CellTopoPtr cellTopo,
                      const Camellia::FieldContainer<double> &physicalNodesSpatial,
                      const Camellia::FieldContainer<double> &physicalNodesTemporal,
                      const Camellia::FieldContainer<double> &physicalCellNodes,
                      const Camellia::FieldContainer<double> &refCellPointsSpatial,
                      const Camellia::FieldContainer<double> &refCellPointsTemporal,
                      const Camellia::FieldContainer<double> &refCellPoints);


  BasisCachePtr getSpatialBasisCache();
  BasisCachePtr getTemporalBasisCache();

  virtual void setRefCellPoints(Intrepid2::TensorPoints<double,DeviceType> pointsRefCell, Intrepid2::TensorData<double,DeviceType> weightsRefCell, int cubatureDegree = -1);
  
  virtual void setRefCellPointsFC(const Camellia::FieldContainer<double> &pointsRefCell);
  virtual void setRefCellPointsFC(const Camellia::FieldContainer<double> &pointsRefCell,
                                const Camellia::FieldContainer<double> &cubatureWeights,
                                int cubatureDegree = -1);

  virtual void setPhysicalCellNodes(const Camellia::FieldContainer<double> &physicalCellNodes,
                                    const std::vector<GlobalIndexType> &cellIDs, bool createSideCacheToo);

  using BasisCache::getValues;
  virtual constFCPtr getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  
  using BasisCache::getTransformedWeightedValues;
  virtual constFCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  
  using BasisCache::getTransformedValues;
  virtual constFCPtr getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);

  static void getTensorialComponentPoints(CellTopoPtr spaceTimeTopo, const Camellia::FieldContainer<double> &tensorPoints,
                                          Camellia::FieldContainer<double> &spatialPoints, Camellia::FieldContainer<double> &temporalPoints);
  static void setDefaultStoreTransformedValues(bool storeValues);
  
  virtual ~SpaceTimeBasisCache() {
//    std::cout << "destructing SpaceTimeBasisCache.\n";
  }
};
}


#endif /* defined(__Camellia__SpaceTimeBasisCache__) */
