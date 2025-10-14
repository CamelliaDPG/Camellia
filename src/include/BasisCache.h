// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#ifndef CAMELLIA_BASIS_CACHE
#define CAMELLIA_BASIS_CACHE

/*
 *  BasisCache.h
 *
 */

// Shards includes
#include "Shards_CellTopology.hpp"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Basis.h"
#include "Camellia_FieldContainer.hpp"
#include "CamelliaIntrepidExtendedTypes.h"
#include "DofOrdering.h"
#include "ElementType.h"
#include "Function.h"

#include "Mesh.h"
#include "PhysicalTensorPoints.h"
#include "TypeDefs.h"

#include <array>

namespace Camellia
{
// only works properly with bases obtained from the BasisFactory.
class BasisCache
{
public:
  template<typename Scalar>
  using DeviceView = typename Camellia::FieldContainer<Scalar>::DeviceView;
protected:
  IndexType _numCells;
  int _spaceDim;
  bool _isSideCache;
  int _sideIndex;
  
private:
  MeshPtr _mesh;
  // Intrepid2 supports economical storage for tensor-product quadrature:
  Intrepid2::TensorPoints<double,DeviceType> _cubPoints;
  Intrepid2::TensorData<double, DeviceType>  _cubWeights;
  // FC containers kept around for compatibility, for now, with existing code.  Plan is to remove support for this as part of 2.0.
  Camellia::FieldContainer<double> _cubPointsFC, _cubWeightsFC;

  Camellia::FieldContainer<double> _cellJacobianFC;
  Camellia::FieldContainer<double> _cellJacobInvFC;
  Camellia::FieldContainer<double> _cellJacobDetFC;
  Camellia::FieldContainer<double> _weightedMeasure;
  Camellia::FieldContainer<double> _physCubPointsFC;
  Camellia::FieldContainer<double> _cellSideParities;
  Camellia::FieldContainer<double> _physicalCellNodes;
  
  PhysicalTensorPoints _physicalCubaturePoints;

  bool _cellJacobianIsValid, _cellJacobianInverseIsValid, _cellJacobianDeterminantIsValid;
  bool _sideNormalsIsValid, _weightedMeasureIsValid, _physCubPointsIsValid;
  
  TFunctionPtr<double> _transformationFxn;
  bool _composeTransformationFxnWithMeshTransformation;
  // bool: compose with existing ref-to-mesh-cell transformation. (false means that the function goes from ref to the physical geometry;
  //                                                                true means it goes from the straight-edge mesh to the curvilinear one)

  std::vector<GlobalIndexType> _cellIDs; // the list of cell IDs corresponding to the physicalCellNodes

  // we use *EITHER* _cubDegree or _cubDegrees
  // if _cubDegree is -1, use _cubDegrees
  // (_cubDegree == -1) <=> (_cubDegrees.size() > 0)
  int _cubDegree;
  int _cubatureEnrichmentDegree = 0;
  vector<int> _cubDegrees;

  // containers specifically for sides:
  Intrepid2::TensorPoints<double,DeviceType> _cubPointsSideRefCell; // in volume coords
  Camellia::FieldContainer<double> _cubPointsSideRefCellFC; // _cubPoints is the one in the side coordinates; this one in volume coords
  Camellia::FieldContainer<double> _sideNormals;
  Camellia::FieldContainer<double> _sideNormalsSpaceTime; // for space-time CellTopologies, a copy of _sideNormals that includes the temporal component

  CellTopoPtr _cellTopo;

  void initCubatureDegree(int maxTrialDegree, int maxTestDegree, int cubatureEnrichmentDegree);
  void initCubatureDegree(std::vector<int> &maxTrialDegrees, std::vector<int> &maxTestDegrees);

  void initVolumeCache(bool createSideCacheToo, bool interpretTensorTopologyAsSpaceTime);
  void initVolumeCache(const Camellia::FieldContainer<double> &refPoints, const Camellia::FieldContainer<double> &cubWeights);

  void determineJacobian();
  void determineJacobianInverseAndDeterminant();
  void determinePhysicalPoints();
  
  //! utility method for setting the _physicalCubaturePoints (a PhysicalTensorPoints object) from the FieldContainer
  void copyPhysicalPointsFieldContainerToPhysicalTensorPoints();
  
  int maxTestDegree();
  
  //! if testVsTest = true, maxTrialDegree refers to un-enriched degree of test space
  int maxTrialDegree();

  void findMaximumDegreeBasisForSides(DofOrdering &trialOrdering);
  
  void recomputeMeasures();
  void determineSideNormals();
protected:
  BasisCache()
  {
    _isSideCache = false;  // for the sake of some hackish subclassing
  }

  map< pair< Camellia::Basis<>*, Intrepid2::EOperator >,
       Intrepid2::BasisValues<double,DeviceType> > _knownValues;
  
  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
       Teuchos::RCP< const Camellia::FieldContainer<double> > > _knownValuesFC;

  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
       Teuchos::RCP< const Camellia::FieldContainer<double> > > _knownValuesTransformedFC;

  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
       Teuchos::RCP< const Camellia::FieldContainer<double> > > _knownValuesTransformedDottedWithNormal;

  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
       Teuchos::RCP< const Camellia::FieldContainer<double> > > _knownValuesTransformedWeighted;

  map< pair< Camellia::Basis<>*, Camellia::EOperator >,
       Teuchos::RCP< const Camellia::FieldContainer<double> > > _knownValuesTransformedWeightedDottedWithNormal;

  std::vector< BasisCachePtr > _basisCacheSides;
  BasisCachePtr _basisCacheVolume;

  virtual void createSideCaches();

  // protected side cache constructor:
  BasisCache(int sideIndex, BasisCachePtr volumeCache, int trialDegree, int testDegree, int cubatureEnrichmentDegree=0);

  // protected "fake" side cache constructor:
  BasisCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const Camellia::FieldContainer<double> &volumeRefPoints,
             const Camellia::FieldContainer<double> &sideNormals, const Camellia::FieldContainer<double> &cellSideParities,
             Camellia::FieldContainer<double> sideNormalsSpaceTime = Camellia::FieldContainer<double>());

  // protected constructor basically for the sake of the SpaceTimeBasisCache subclass, which wants to disable side cache creation during construction.
  BasisCache(ElementTypePtr elemType, MeshPtr mesh, bool testVsTest,
             int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime,
             bool createSideCacheToo, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);

  Intrepid2::EPolyType _polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX;
  std::vector< BasisPtr > _maxDegreeBasisForSide; // stored in volume cache so we can get cubature right on sides, including broken sides (if this is a multiBasis)
  int _maxTestDegree, _maxTrialDegree;

  void cubatureDegreeForElementType(ElementTypePtr elemType, bool testVsTest, int &cubatureDegree);
  void cubatureDegreeForElementType(ElementTypePtr elemType, bool testVsTest, int &cubatureDegreeSpace, int &cubatureDegreeTime);
  
  using ScratchStorageDimensions = std::vector<int>;
  
  template<typename Scalar>
  using ScratchStorageEntry = std::pair<DeviceView<Scalar>, bool>; // true: checked out right now
  
  template<typename Scalar>
  using ScratchStorageCache = std::map<ScratchStorageDimensions, std::vector< ScratchStorageEntry<Scalar> > >;
  
  ScratchStorageCache<double> _doubleScratchStorageCache; // for now, we only support double.  Could add support for int, complex, float, etc.
public:
  BasisCache(ElementTypePtr elemType, MeshPtr mesh = Teuchos::rcp( (Mesh*) NULL ), bool testVsTest=false,
             int cubatureDegreeEnrichment = 0, bool tensorProductTopologyMeansSpaceTime = false, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX); // use testVsTest=true for test space inner product

  BasisCache(CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime=false, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);
  BasisCache(shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);

  BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo = false, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);
  BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo = false, bool tensorProductTopologyMeansSpaceTime=false, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);
  
  // lighter weight volume constructor:
  BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
             const Camellia::FieldContainer<double> &refCellPoints, const Camellia::FieldContainer<double> &cubWeights, int cubatureDegree = 0);
  // lighter weight side cache constructor:
  BasisCache(int sideOrdinal, BasisCachePtr volumeCache, const Camellia::FieldContainer<double> &refPoints,
             const Camellia::FieldContainer<double> &cubWeights, int cubatureDegree = 0);

  BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
             DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo = false, bool tensorProductTopologyMeansSpaceTime=false, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);
  BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo,
             DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo = false, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);
  
  //! Volume constructor with tensor-product point/weight support.  (Side caches not generated automatically.)
  BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
             const Intrepid2::TensorPoints<double, Camellia::DeviceType> &refCellPoints,
             const Intrepid2::TensorData<double, Camellia::DeviceType> &cubWeights, int cubatureDegree);
  
  //! Volume constructor with tensor-product point support.  (Side caches not generated automatically.)
  BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
             const Intrepid2::TensorPoints<double, Camellia::DeviceType> &refCellPoints);
  
  virtual ~BasisCache();

  template<typename Scalar>
  DeviceView<Scalar> checkoutScratchStorage(DeviceView<Scalar> &dv);
  
  template<typename Scalar>
  void checkinScratchStorage(DeviceView<Scalar> &scratchView);
  
  // ! Returns true if the operator given is supported by getTransformedValues().
  // ! Right now, we can compute transformations for any first-order operators, and for d^2/{dx_i dx_j} second-order operators when i==j, but not when i!=j.
  static bool canComputeTransformedValues(Camellia::EOperator op);
  
  Camellia::FieldContainer<double> & getWeightedMeasures();
  Camellia::FieldContainer<double> getCellMeasures();

  const Intrepid2::BasisValues<double,DeviceType>& getValues(BasisPtr intrepid2Basis, Intrepid2::EOperator op, bool useCubPointsSideRefCell = false);
  
  virtual Teuchos::RCP< const Camellia::FieldContainer<double> > getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Camellia::FieldContainer<double> > getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Camellia::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell = false);

  // side variants:
  virtual Teuchos::RCP< const Camellia::FieldContainer<double> > getValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Camellia::FieldContainer<double> > getTransformedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);
  virtual Teuchos::RCP< const Camellia::FieldContainer<double> > getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell = false);

  bool isSideCache();
  BasisCachePtr getSideBasisCache(int sideOrdinal);
  BasisCachePtr getVolumeBasisCache(); // from sideCache

  const std::vector<GlobalIndexType> & cellIDs();
  void setCellIDs(const std::vector<GlobalIndexType> &cellIDs);
  
  int numCells() const;

  CellTopoPtr cellTopology();

  int cubatureDegree();
  int cubatureEnrichmentDegree();

  MeshPtr mesh();
  void setMesh(MeshPtr mesh);

  void discardPhysicalNodeInfo(); // discards physicalNodes and all transformed basis values.

  const Camellia::FieldContainer<double> & getJacobian();
  const Camellia::FieldContainer<double> & getJacobianDet();
  const Camellia::FieldContainer<double> & getJacobianInv();
  
  // ! sets an externally-computed jacobian; sets _jacobianIsValid to true
  void setJacobian(const Camellia::FieldContainer<double> &jacobian);
  // ! sets an externally-computed jacobian determinant; sets _jacobianDetIsValid to true
  void setJacobianDet(const Camellia::FieldContainer<double> &jacobianDet);
  // ! sets an externally-computed jacobian inverse; sets _jacobianInvIsValid to true
  void setJacobianInv(const Camellia::FieldContainer<double> &jacobianInv);

  double extrusionWidth(const int &d, const int &cellOrdinal);
  
  // ! Returns true if the second-order derivatives of the reference-to-physical transformation may be ignored.
  bool neglectHessian() const;
  
  Camellia::FieldContainer<double> computeParametricPoints();

  virtual const PhysicalTensorPoints & getPhysicalCubaturePoints();
  
  virtual const Camellia::FieldContainer<double> & getPhysicalCubaturePointsFC();
  const Camellia::FieldContainer<double> & getPhysicalCubaturePointsForSide(int sideOrdinal);
  const Camellia::FieldContainer<double> & getCellSideParities();
  
  virtual void setPhysicalCubaturePoints(const Camellia::FieldContainer<double> &physicalCubaturePoints);

  const Intrepid2::TensorData<double, DeviceType> getCubatureWeights() const;
  
  const Camellia::FieldContainer<double> & getCubatureWeightsFC();

  const Camellia::FieldContainer<double> & getSideUnitNormals(int sideOrdinal);

  const Camellia::FieldContainer<double> &getPhysicalCellNodes();
  virtual void setPhysicalCellNodes(const Camellia::FieldContainer<double> &physicalCellNodes,
                                    const std::vector<GlobalIndexType> &cellIDs, bool createSideCacheToo);

  /*** Methods added for BC support below ***/
  // setRefCellPointsFC overwrites _cubPoints -- for when cubature is not your interest
  // (this comes up in imposeBC)
  virtual void setRefCellPointsFC(const Camellia::FieldContainer<double> &pointsRefCell);
  virtual void setRefCellPointsFC(const Camellia::FieldContainer<double> &pointsRefCell, const Camellia::FieldContainer<double> &cubatureWeights, int cubatureDegree = -1);
  const Camellia::FieldContainer<double> &getRefCellPointsFC();
  const Camellia::FieldContainer<double> &getSideRefCellPointsInVolumeCoordinatesFC();
  
  const Intrepid2::TensorPoints<double,DeviceType> & getRefCellPoints() const;
  const Intrepid2::TensorPoints<double,DeviceType> & getSideRefCellPointsInVolumeCoordinates();
  
  void setRefCellPoints(Intrepid2::TensorPoints<double,DeviceType> pointsRefCell);
  virtual void setRefCellPoints(Intrepid2::TensorPoints<double,DeviceType> pointsRefCell, Intrepid2::TensorData<double,DeviceType> weightsRefCell, int cubatureDegree = -1);

  // physicalPoints: (P,D).  cellIndex indexes into BasisCache's physicalCellNodes
  Camellia::FieldContainer<double> getRefCellPointsForPhysicalPoints(const Camellia::FieldContainer<double> &physicalPoints, int cellIndex=0);

  /** \brief  Returns an Camellia::FieldContainer<double> populated with the side normals; dimensions are (C,P,D) or (C,P,D-1).  Tensor-product topologies are interpreted as space-time elements; in this context, the side normals provided will be the spatial part of the space-time normal.
   */
  const Camellia::FieldContainer<double> & getSideNormals();
  void setSideNormals(Camellia::FieldContainer<double> &sideNormals);
  void setCellSideParities(const Camellia::FieldContainer<double> &cellSideParities);

  /** \brief  Returns an Camellia::FieldContainer<double> populated with the full space-time side normals; dimensions are (C,P,D).  For non-tensor-product topologies, throws an exception.
   */
  const Camellia::FieldContainer<double> & getSideNormalsSpaceTime();

  int getSideIndex() const; // -1 if not sideCache

  virtual int getSpaceDim();
  
  bool cellTopologyIsSpaceTime();

  void setTransformationFunction(TFunctionPtr<double> fxn, bool composeWithMeshTransformation = true);
  
  bool testVsTest();

  // static convenience constructors:
  static BasisCachePtr parametric1DCache(int cubatureDegree);
  static BasisCachePtr parametricQuadCache(int cubatureDegree);
  static BasisCachePtr parametricQuadCache(int cubatureDegree, const Camellia::FieldContainer<double> &refCellPoints, int sideCacheIndex=-1);
  static BasisCachePtr basisCache1D(double x0, double x1, int cubatureDegree); // x0 and x1: physical space endpoints
  static BasisCachePtr basisCacheForCell(ConstMeshTopologyViewPtr meshTopology, GlobalIndexType cellID, ElementTypePtr elemType,
                                         bool testVsTest = false, int cubatureDegreeEnrichment = 0,
                                         bool tensorProductTopologyMeansSpaceTime=false);
  static BasisCachePtr basisCacheForCell(MeshPtr mesh, GlobalIndexType cellID, bool testVsTest = false,
                                         int cubatureDegreeEnrichment = 0, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);
  //! bottleneck basisCacheForCell method:
  static BasisCachePtr basisCacheForCell(MeshPtr mesh, ConstMeshTopologyViewPtr meshTopology, GlobalIndexType cellID, ElementTypePtr elemType,
                                         bool testVsTest, int cubatureDegreeEnrichment, bool tensorTopologyMeansSpaceTime, Intrepid2::EPolyType polyTypeForQuadrature = Intrepid2::POLYTYPE_MAX);
  static BasisCachePtr basisCacheForCellType(MeshPtr mesh, ElementTypePtr elemType, bool testVsTest = false,
      int cubatureDegreeEnrichment = 0); // for cells on the local MPI node
  static BasisCachePtr basisCacheForReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree, bool createSideCacheToo=false);
  static BasisCachePtr basisCacheForRefinedReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree, RefinementBranch refinementBranch, bool createSideCacheToo=false);

  static BasisCachePtr basisCacheForCellTopology(CellTopoPtr cellTopo, int cubatureDegree,
                                                 const Camellia::FieldContainer<double> &physicalCellNodes,
                                                 bool createSideCacheToo=false,
                                                 bool tensorProductTopologyMeansSpaceTime=false);

  static BasisCachePtr basisCacheForReferenceCell(CellTopoPtr cellTopo, int cubatureDegree, bool createSideCacheToo=false,
                                                  bool tensorProductTopologyMeansSpaceTime=false);
  static BasisCachePtr basisCacheForRefinedReferenceCell(CellTopoPtr cellTopo, int cubatureDegree, RefinementBranch refinementBranch,
                                                         bool createSideCacheToo=false, bool tensorProductTopologyMeansSpaceTime=false);
  
  static BasisCachePtr basisCacheForRefinedReferenceCell(int cubatureDegree, RefinementBranch refinementBranch, bool createSideCacheToo = false, bool tensorProductTopologyMeansSpaceTime=false);

  static BasisCachePtr quadBasisCache(double width, double height, int cubDegree, bool createSideCacheToo=false);

  // note that this does not inform the volumeCache about the created side cache:
  // Intended for cases where you just want to create a BasisCache for one of the sides, not all of them.
  // If you want one for all of them, you should pass createSideCacheToo = true to an appropriate volumeCache method.
  static BasisCachePtr sideBasisCache(BasisCachePtr volumeCache, int sideIndex);

  // ! As the name suggests, this method is not meant for widespread use.  Intended mainly for flux-to-field mappings
  static BasisCachePtr fakeSideCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const Camellia::FieldContainer<double> &volumeRefPoints,
                                     const Camellia::FieldContainer<double> &sideNormals, const Camellia::FieldContainer<double> &cellSideParities,
                                     Camellia::FieldContainer<double> sideNormalsSpaceTime = Camellia::FieldContainer<double>());
};
}

#endif
