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
  void getReferencePointComponents(const Intrepid::FieldContainer<double> &tensorRefPoints,
                                   Intrepid::FieldContainer<double> &spaceRefPoints,
                                   Intrepid::FieldContainer<double> &timeRefPoints);
  double getTemporalNodeCoordinateRefSpace(); // for temporal sides
protected:
  virtual void createSideCaches();
public:
  // volume constructors:
  SpaceTimeBasisCache(MeshPtr spaceTimeMesh, ElementTypePtr spaceTimeElementType,
                      const Intrepid::FieldContainer<double> &physicalNodesSpatial,
                      const Intrepid::FieldContainer<double> &physicalNodesTemporal,
                      const Intrepid::FieldContainer<double> &physicalNodesSpaceTime,
                      const std::vector<GlobalIndexType> &cellIDs,
                      bool testVsTest, int cubatureDegreeEnrichment);
  SpaceTimeBasisCache(const Intrepid::FieldContainer<double> &physicalNodesSpatial,
                      const Intrepid::FieldContainer<double> &physicalNodesTemporal,
                      const Intrepid::FieldContainer<double> &physicalCellNodes,
                      CellTopoPtr cellTopo, int cubDegree);
  SpaceTimeBasisCache(CellTopoPtr cellTopo,
                      const Intrepid::FieldContainer<double> &physicalNodesSpatial,
                      const Intrepid::FieldContainer<double> &physicalNodesTemporal,
                      const Intrepid::FieldContainer<double> &physicalCellNodes,
                      const Intrepid::FieldContainer<double> &refCellPointsSpatial,
                      const Intrepid::FieldContainer<double> &refCellPointsTemporal,
                      const Intrepid::FieldContainer<double> &refCellPoints);


  BasisCachePtr getSpatialBasisCache();
  BasisCachePtr getTemporalBasisCache();

  virtual void setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell);
  virtual void setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell,
                                const Intrepid::FieldContainer<double> &cubatureWeights);

  virtual void setPhysicalCellNodes(const Intrepid::FieldContainer<double> &physicalCellNodes,
                                    const std::vector<GlobalIndexType> &cellIDs, bool createSideCacheToo);

  virtual constFCPtr getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual constFCPtr getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual constFCPtr getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);

  static void getTensorialComponentPoints(CellTopoPtr spaceTimeTopo, const Intrepid::FieldContainer<double> &tensorPoints,
                                          Intrepid::FieldContainer<double> &spatialPoints, Intrepid::FieldContainer<double> &temporalPoints);
};
}


#endif /* defined(__Camellia__SpaceTimeBasisCache__) */
