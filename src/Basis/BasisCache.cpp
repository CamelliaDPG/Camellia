// @HEADER
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
// This code is derived from source governed by the license LICENSE-DPGTrilinos in the licenses directory.
//
// @HEADER

#include "TypeDefs.h"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "Intrepid2_CellTools.hpp"
#include "Intrepid2_Cubature.hpp"

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisEvaluation.h"
#include "Camellia_FieldContainerConversion.hpp"
#include "CamelliaCellTools.h"
#include "CubatureFactory.h"
#include "doubleBasisConstruction.h"
#include "Function.h"
#include "Mesh.h"
#include "SerialDenseWrapper.h"
#include "SharedScratchStorage.h"
#include "SpaceTimeBasisCache.h"

#include "Teuchos_GlobalMPISession.hpp"

#include <array>

using namespace std;
using namespace Camellia;

using fst = Intrepid::FunctionSpaceTools;

const static bool CACHE_TRANSFORMED_VALUES = false; // save some memory by not caching these

// TODO: add exceptions for side cache arguments to methods that don't make sense
// (e.g. useCubPointsSideRefCell==true when _isSideCache==false)

int boundDegreeToMaxCubatureForCellTopo(int degree, CellTopoPtr cellTopo)
{
  // TODO: update this with Intrepid2 maximums
  // limit cubature degree to max that Intrepid will support
  auto baseKey = cellTopo->getBaseTopology()->getShardsTopology().getKey();
  const bool lineOrExtrudedLine = baseKey == shards::Line<2>::key;
  const bool triOrExtrudedTri   = baseKey == shards::Triangle<3>::key;
  if (lineOrExtrudedLine)
  {
    return min(INTREPID_CUBATURE_LINE_GAUSS_MAX, degree);
  }
  else if (triOrExtrudedTri)
  {
    return min(INTREPID_CUBATURE_TRI_DEFAULT_MAX, degree);
  }
  else
  {
    // unhandled cell topo--we'll get an exception if we go beyond the max...
    return degree;
  }
}

bool BasisCache::canComputeTransformedValues(Camellia::EOperator op)
{
  if ((OP_VALUE <= op) && (op <= OP_D1)) // zero order and first order intrepid operators all supported
    return true;
  if ((OP_I1 <= op) && (op <= OP_I7)) // Camellia numbered component operators (OP_I1 synonymous with OP_X, etc.)
    return true;
  if ((OP_D2 <= op) && (op <= OP_D10)) // can't transform general second order and above
    return false;
  if ((OP_X <= op) && (op <= OP_DZDZ)) // zero and first order Camellia operators, and second-order with i==j
    return true;
  if ((OP_DI1 <= op) && (op <= OP_DI7)) // Camellia gradient component operators (OP_DI1 synonymous with OP_DX, etc.)
    return true;
  return false;
}

// ! Requires that _cellTopo be initialized
void BasisCache::initCubatureDegree(int maxTrialDegree, int maxTestDegree, int cubatureEnrichmentDegree)
{
  _cubDegree = maxTrialDegree + maxTestDegree + cubatureEnrichmentDegree;

  if (! _isSideCache)
  {
    _cubDegree = boundDegreeToMaxCubatureForCellTopo(_cubDegree, _cellTopo);
  }
  else
  {
    int sideDim = _cellTopo->getDimension() - 1;
    CellTopoPtr side = _cellTopo->getSubcell(sideDim,_sideIndex); // create relevant subcell (side) topology
    _cubDegree = boundDegreeToMaxCubatureForCellTopo(_cubDegree, side);
  }
  _cubatureEnrichmentDegree = cubatureEnrichmentDegree;
  _maxTestDegree = maxTestDegree;
  _maxTrialDegree = maxTrialDegree;
}

// ! Requires that _cellTopo be initialized
void BasisCache::initCubatureDegree(std::vector<int> &maxTrialDegrees, std::vector<int> &maxTestDegrees)
{
  TEUCHOS_TEST_FOR_EXCEPTION(maxTrialDegrees.size() != maxTestDegrees.size(), std::invalid_argument, "maxTrialDegrees must have same length as maxTestDegrees");
  _maxTestDegree = 0;
  _maxTrialDegree = 0;
  _cubDegree = -1;
  _cubDegrees.resize(maxTrialDegrees.size());
  for (int i=0; i<maxTrialDegrees.size(); i++)
  {
    _maxTrialDegree = max(_maxTrialDegree, maxTrialDegrees[i]);
    _maxTestDegree = max(_maxTestDegree, maxTestDegrees[i]);

    int cubDegree = maxTrialDegrees[i] + maxTestDegrees[i];
    _cubDegrees[i] = boundDegreeToMaxCubatureForCellTopo(cubDegree, _cellTopo);
  }
}

// ! requires that initCubatureDegree() has been called
void BasisCache::initVolumeCache(bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  _sideIndex = -1;
  _spaceDim = _cellTopo->getDimension();
  if ((tensorProductTopologyMeansSpaceTime) && (_cellTopo->getTensorialDegree() > 0))
  {
    // last dimension is time, then
    _spaceDim = _spaceDim - 1;
  }
  _isSideCache = false; // VOLUME constructor

  if (_cellTopo->getDimension() > 0)
  {
    CubatureFactory cubFactory;
    using Cubature = Intrepid2::Cubature<DeviceType,double,double>;
    Teuchos::RCP< Cubature > cellTopoCub;
    
    if (_cubDegree >= 0)
      cellTopoCub = cubFactory.create(_cellTopo, _cubDegree);
    else if (_cubDegrees.size() > 0)
      cellTopoCub = cubFactory.create(_cellTopo, _cubDegrees);
    
    if (cellTopoCub != Teuchos::null)
    {
      _cubWeights = cellTopoCub->allocateCubatureWeights();
      _cubPoints  = cellTopoCub->allocateCubaturePoints();
      
      cellTopoCub->getCubature(_cubPoints, _cubWeights);
      
      _cubPointsFC  = FieldContainerConversion<double>::fromTensorPoints(_cubPoints);
      _cubWeightsFC = FieldContainerConversion<double>::fromTensorData(_cubWeights);
    }
    else
    {
      // cubature degrees were unset; user may not be using cubature points
    }
  }
  else
  {
    _cubDegree = 1;
    int numCubPointsSide = 1;
    _cubPointsFC.resize(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
    _cubWeightsFC.resize(numCubPointsSide);

    _cubPointsFC.initialize(0.0);
    _cubWeightsFC.initialize(1.0);
  }

  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;
  
  // now, create side caches
  if ( createSideCacheToo )
  {
    this->createSideCaches();
  }
}

void BasisCache::initVolumeCache(const Camellia::FieldContainer<double> &refPoints, const Camellia::FieldContainer<double> &cubWeights)
{
  // lightweight guy for when the ref points are known in advance
  _sideIndex = -1;
  _spaceDim = _cellTopo->getDimension();
  _isSideCache = false; // VOLUME constructor
  
  _cubPointsFC = refPoints;
  _cubWeightsFC = cubWeights;
  
  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;
}

void BasisCache::copyPhysicalPointsFieldContainerToPhysicalTensorPoints()
{
  // copy the physical points into a single PhysicalTensorPoints component
  const int numCells     = _physCubPointsFC.dimension(0);
  const int numPoints    = _physCubPointsFC.dimension(1);
  const int numCoordDims = _physCubPointsFC.dimension(2);
  Kokkos::DynRankView<double, DeviceType> physPointsComponent("physical points component",numCells,numPoints,numCoordDims);
  auto physCubPointsView = _physCubPointsFC.checkoutDeviceView();
  Kokkos::deep_copy(physPointsComponent, physCubPointsView);
  vector< Kokkos::DynRankView<double, DeviceType> > physicalTensorPointsComponents {physPointsComponent};
  _physCubPointsFC.checkinDeviceView(false);
  _physicalCubaturePoints = PhysicalTensorPoints(physicalTensorPointsComponents);
}

void BasisCache::createSideCaches()
{
  _basisCacheSides.clear();
  int numSides = _cellTopo->getSideCount();

  for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
  {
    BasisPtr maxDegreeBasisOnSide = _maxDegreeBasisForSide[sideOrdinal];
    BasisPtr multiBasisIfAny;

    int maxTrialDegreeOnSide = _maxTrialDegree;
    if (maxDegreeBasisOnSide.get() != NULL)
    {
      if (BasisFactory::basisFactory()->isMultiBasis(maxDegreeBasisOnSide))
      {
        multiBasisIfAny = maxDegreeBasisOnSide;
      }
      maxTrialDegreeOnSide = maxDegreeBasisOnSide->getDegree();
    }

    BasisCachePtr thisPtr = Teuchos::rcp( this, false ); // presumption is that side cache doesn't outlive volume...
    BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideOrdinal, thisPtr, maxTrialDegreeOnSide, _maxTestDegree, multiBasisIfAny, _cubatureEnrichmentDegree));
    _basisCacheSides.push_back(sideCache);
  }
}

BasisCache::BasisCache(CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  _cellTopo = cellTopo;

  DofOrdering trialOrdering(cellTopo); // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs

  _isSideCache = false;
  initCubatureDegree(0, cubDegree, 0);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
}

BasisCache::BasisCache(ElementTypePtr elemType, MeshPtr mesh, bool testVsTest,
                       int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime)
{
  // use testVsTest=true for test space inner product (won't create side caches, and will use higher cubDegree)
  _cellTopo = elemType->cellTopoPtr;

  _maxTestDegree = elemType->testOrderPtr->maxBasisDegree();

  _mesh = mesh;
  if (_mesh.get())
  {
    _transformationFxn = _mesh->getTransformationFunction();
    if (_transformationFxn.get())
    {
      // assuming isoparametric:
      cubatureDegreeEnrichment += _maxTestDegree;
    }
    // at least for now, what the Mesh's transformation function does is transform from a straight-lined mesh to
    // one with potentially curved edges...
    _composeTransformationFxnWithMeshTransformation = true;
  }

  _maxTrialDegree = testVsTest ? _maxTestDegree : elemType->trialOrderPtr->maxBasisDegree();

  findMaximumDegreeBasisForSides( *(elemType->trialOrderPtr) );

  bool createSideCacheToo = !testVsTest && elemType->trialOrderPtr->hasSideVarIDs();

  _isSideCache = false;
  initCubatureDegree(_maxTrialDegree, _maxTestDegree, cubatureDegreeEnrichment);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
}

// protected constructor basically for the sake of the SpaceTimeBasisCache, which wants to disable side cache creation during construction.
BasisCache::BasisCache(ElementTypePtr elemType, MeshPtr mesh, bool testVsTest,
                       int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime,
                       bool createSideCacheToo)
{
  // use testVsTest=true for test space inner product (won't create side caches, and will use higher cubDegree)
  _cellTopo = elemType->cellTopoPtr;

  _maxTestDegree = elemType->testOrderPtr->maxBasisDegree();

  _mesh = mesh;
  if (_mesh.get())
  {
    _transformationFxn = _mesh->getTransformationFunction();
    if (_transformationFxn.get())
    {
      // assuming isoparametric:
      cubatureDegreeEnrichment += _maxTestDegree;
    }
    // at least for now, what the Mesh's transformation function does is transform from a straight-lined mesh to
    // one with potentially curved edges...
    _composeTransformationFxnWithMeshTransformation = true;
  }

  _maxTrialDegree = testVsTest ? _maxTestDegree : elemType->trialOrderPtr->maxBasisDegree();

  findMaximumDegreeBasisForSides( *(elemType->trialOrderPtr) );

  _isSideCache = false;
  initCubatureDegree(_maxTrialDegree, _maxTestDegree, cubatureDegreeEnrichment);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
}

BasisCache::BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
                       DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  _cellTopo = cellTopo;
  findMaximumDegreeBasisForSides(trialOrdering);

  _isSideCache = false;
  initCubatureDegree(trialOrdering.maxBasisDegree(), maxTestDegree, 0);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

BasisCache::BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes,
                       shards::CellTopology &cellTopo,
                       DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo)
{
  _cellTopo = CellTopology::cellTopology(cellTopo);
  findMaximumDegreeBasisForSides(trialOrdering);

  _isSideCache = false;
  initCubatureDegree(trialOrdering.maxBasisDegree(), maxTestDegree, 0);
  bool tensorProductTopologyMeansSpaceTime = true; // doesn't matter for shards topologies (they're not tensor products)
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

BasisCache::BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
                       const Intrepid2::TensorPoints<double, Camellia::DeviceType> &refCellPoints)
:
BasisCache(physicalCellNodes, cellTopo, refCellPoints, Intrepid2::TensorData<double, Camellia::DeviceType>(), 0)
{}

BasisCache::BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
                       const Intrepid2::TensorPoints<double, Camellia::DeviceType> &refCellPoints,
                       const Intrepid2::TensorData<double, Camellia::DeviceType> &cubWeights, int cubatureDegree)
{
  _cellTopo = cellTopo;
  _isSideCache = false;
  initCubatureDegree(cubatureDegree,0,0);
  
  _sideIndex = -1;
  _spaceDim = _cellTopo->getDimension();
  _isSideCache = false; // VOLUME constructor

  _cubPoints  = refCellPoints;
  _cubWeights = cubWeights;
  
  bool initializeFCs = true;
  if (initializeFCs)
  {
    _cubPointsFC  = FieldContainerConversion<double>::fromTensorPoints(_cubPoints);
    _cubWeightsFC = FieldContainerConversion<double>::fromTensorData(_cubWeights);
  }

  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;
  
  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),false);
}


BasisCache::BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
                       const Camellia::FieldContainer<double> &refCellPoints, const Camellia::FieldContainer<double> &cubWeights, int cubatureDegree)
{
  _cellTopo = cellTopo;
  _isSideCache = false;
  initCubatureDegree(cubatureDegree,0,0);
  
  initVolumeCache(refCellPoints, cubWeights);
  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),false);
}

BasisCache::BasisCache(shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo)
{
  // NOTE that this constructor's a bit dangerous, in that we lack information about the brokenness
  // of the sides; we may under-integrate for cells with broken sides...
  _cellTopo = CellTopology::cellTopology(cellTopo);
  DofOrdering trialOrdering(_cellTopo); // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs

  _isSideCache = false;
  initCubatureDegree(0, cubDegree, 0);
  bool tensorProductTopologyMeansSpaceTime = true; // doesn't matter for shards topologies (they're not tensor products)
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
}

BasisCache::BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo)
{
  // NOTE that this constructor's a bit dangerous, in that we lack information about the brokenness
  // of the sides; we may under-integrate for cells with broken sides...
  _cellTopo = CellTopology::cellTopology(cellTopo);
  DofOrdering trialOrdering(_cellTopo); // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs

  _isSideCache = false;
  initCubatureDegree(0, cubDegree, 0);
  bool tensorProductTopologyMeansSpaceTime = true; // doesn't matter for shards topologies (they're not tensor products)
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);

  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

BasisCache::BasisCache(const Camellia::FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  // NOTE that this constructor's a bit dangerous, in that we lack information about the brokenness
  // of the sides; we may under-integrate for cells with broken sides...
  _cellTopo = cellTopo;
  DofOrdering trialOrdering(_cellTopo); // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs

  _isSideCache = false;
  initCubatureDegree(0, cubDegree, 0);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);

  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

// "fake" side constructor
BasisCache::BasisCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const Camellia::FieldContainer<double> &volumeRefPoints,
                       const Camellia::FieldContainer<double> &sideNormals, const Camellia::FieldContainer<double> &cellSideParities,
                       Camellia::FieldContainer<double> sideNormalsSpaceTime)
{
  _cellTopo = volumeCache->cellTopology(); // VOLUME cell topo.
  _isSideCache = true;
  _sideIndex = fakeSideOrdinal;
  _basisCacheVolume = volumeCache;
  _spaceDim = volumeCache->getSpaceDim();

  _cubPointsFC.resize(0); // force an exception if true side reference points are ever accessed in fake side BasisCache

  _physicalCellNodes = volumeCache->getPhysicalCellNodes();
  
  const int numCells = volumeCache->getPhysicalCubaturePointsFC().dimension(0);
  const int numPoints = volumeRefPoints.dimension(0);
  _numCells = numCells;
  _physCubPointsFC.resize(numCells,numPoints,_spaceDim);
  _cubPointsSideRefCellFC = volumeRefPoints;
  _cubPointsSideRefCell = FieldContainerConversion<double>::toTensorPoints(_cubPointsSideRefCellFC);
  _sideNormals = sideNormals;
  _sideNormalsSpaceTime = sideNormalsSpaceTime;
  _cellSideParities = cellSideParities;
  
  // the assumption is that if you're using this constructor, the volume points provided are already in reference space
  // so that the transformations are all identities
  int cellDim = _cellTopo->getDimension(); // for space-time, cellDim = _spaceDim + 1
  _cellJacobianFC.resize(numCells, numPoints, cellDim, cellDim);
  _cellJacobInvFC.resize(numCells, numPoints, cellDim, cellDim);
  _cellJacobDetFC.resize(numCells, numPoints);
  _cellJacobianFC.initialize(0.0);
  _cellJacobInvFC.initialize(0.0);
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      for (int d=0; d<cellDim; d++)
      {
        _cellJacobianFC(cellOrdinal,pointOrdinal,d,d) = 1.0;
        _cellJacobInvFC(cellOrdinal,pointOrdinal,d,d) = 1.0;
      }
    }
  }
  
  _cellJacobDetFC.initialize(1.0);
  _cellJacobianIsValid = true;
  _cellJacobianInverseIsValid = true;
  _cellJacobianDeterminantIsValid = true;
  _sideNormalsIsValid = true;
  _physCubPointsIsValid = false;
}

// side constructor
BasisCache::BasisCache(int sideIndex, BasisCachePtr volumeCache, int trialDegree, int testDegree, BasisPtr multiBasisIfAny, int cubatureEnrichmentDegree)
{
  _cellTopo = volumeCache->cellTopology(); // VOLUME cell topo.
  _isSideCache = true;
  _sideIndex = sideIndex;
  _basisCacheVolume = volumeCache;
  _maxTestDegree = testDegree;
  _maxTrialDegree = trialDegree;
  if (volumeCache->mesh().get())
  {
    _transformationFxn = volumeCache->mesh()->getTransformationFunction();
    // at least for now, what the Mesh's transformation function does is transform from a straight-lined mesh to
    // one with potentially curved edges...
    _composeTransformationFxnWithMeshTransformation = true;
  }
  _spaceDim = volumeCache->getSpaceDim();
  int sideDim = _cellTopo->getDimension() - 1;
  CellTopoPtr side = _cellTopo->getSubcell(sideDim,sideIndex); // create relevant subcell (side) topology

  initCubatureDegree(trialDegree, testDegree, cubatureEnrichmentDegree);

  if (sideDim > 0)
  {
    CubatureFactory cubFactory;
    using Cubature = Intrepid2::Cubature<DeviceType,double,double>;
    Teuchos::RCP< Cubature > sideCub;
    if (_cubDegree >= 0)
      sideCub = cubFactory.create(side, _cubDegree);
    else if (_cubDegrees.size() > 0)
      sideCub = cubFactory.create(side, _cubDegrees);

    int numCubPointsSide;
    
    if (sideCub != Teuchos::null)
    {
      numCubPointsSide = sideCub->getNumPoints();
      _cubWeights = sideCub->allocateCubatureWeights();
      _cubPoints = sideCub->allocateCubaturePoints();
      
      sideCub->getCubature(_cubPoints, _cubWeights);
      
      _cubPointsFC  = FieldContainerConversion<double>::fromTensorPoints(_cubPoints);
      _cubWeightsFC = FieldContainerConversion<double>::fromTensorData(_cubWeights);
    }
    else
    {
      numCubPointsSide = 0;
      _cubPointsFC.resize(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
      _cubWeightsFC.resize(numCubPointsSide);
    }
    
    if ( multiBasisIfAny.get() != NULL )
    {
      MultiBasis<>* multiBasis = (MultiBasis<>*) multiBasisIfAny.get();

      int cubatureEnrichment = (multiBasis->getDegree() < _maxTrialDegree) ? _maxTrialDegree - multiBasis->getDegree() : 0;
      multiBasis->getCubature(_cubPointsFC, _cubWeightsFC, _maxTestDegree + cubatureEnrichment);

      numCubPointsSide = _cubPointsFC.dimension(0);
    }

    _cubPointsSideRefCellFC.resize(numCubPointsSide, sideDim + 1); // cubPointsSide from the pov of the ref cell
    if (numCubPointsSide > 0)
    {
      CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCellFC, _cubPointsFC, sideDim, _sideIndex, _cellTopo);
      // TODO: create the TensorPoints _cubPointsSideRefCell directly from _cubPoints
      // for now, we create it from _cubPointsSideRefCellFC (and lose any tensor product structure)
      _cubPointsSideRefCell = FieldContainerConversion<double>::toTensorPoints(_cubPointsSideRefCellFC);
    }
  }
  else
  {
    _cubDegree = 1;
    int numCubPointsSide = 1;
    _cubPointsFC.resize(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
    _cubWeightsFC.resize(numCubPointsSide);

    _cubPointsFC.initialize(0.0);
    _cubWeightsFC.initialize(1.0);

    _cubPointsSideRefCellFC.resize(numCubPointsSide, sideDim + 1); // cubPointsSide from the pov of the ref cell
    CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCellFC, _cubPointsFC, sideDim, _sideIndex, _cellTopo);
    
    // TODO: create the TensorPoints _cubPointsSideRefCell directly from _cubPoints
    // for now, we create it from _cubPointsSideRefCellFC (and lose any tensor product structure)
    _cubPointsSideRefCell = FieldContainerConversion<double>::toTensorPoints(_cubPointsSideRefCellFC);
  }
}

// side constructor with specific ref points
BasisCache::BasisCache(int sideIndex, BasisCachePtr volumeCache, const Camellia::FieldContainer<double> &refPoints,
                       const Camellia::FieldContainer<double> &cubWeights, int cubatureDegree)
{
  _cellTopo = volumeCache->cellTopology(); // VOLUME cell topo.
  _isSideCache = true;
  _sideIndex = sideIndex;
  _basisCacheVolume = volumeCache;
  if (volumeCache->mesh().get())
  {
    _transformationFxn = volumeCache->mesh()->getTransformationFunction();
    // at least for now, what the Mesh's transformation function does is transform from a straight-lined mesh to
    // one with potentially curved edges...
    _composeTransformationFxnWithMeshTransformation = true;
  }
  _spaceDim = volumeCache->getSpaceDim();
  int sideDim = _cellTopo->getDimension() - 1;
  CellTopoPtr side = _cellTopo->getSubcell(sideDim,sideIndex); // create relevant subcell (side) topology
  
  initCubatureDegree(cubatureDegree, 0, 0);

  // TODO: initialized _cubPoints (the Intrepid2 container) and _cubWeights appropriately
  _cubPointsFC = refPoints;
  _cubWeightsFC = cubWeights;
  
  int numCubPointsSide = _cubPointsFC.dimension(0);
  
  _cubPointsSideRefCellFC.resize(numCubPointsSide, sideDim + 1); // cubPointsSide from the pov of the ref cell
  CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCellFC, _cubPointsFC, sideDim, _sideIndex, _cellTopo);
  
  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;
  
  setPhysicalCellNodes(volumeCache->getPhysicalCellNodes(), volumeCache->cellIDs(), false); // false: don't create side caches...
}

BasisCache::~BasisCache()
{
//  std::cout << "BasisCache dtor.\n";
}

const vector<GlobalIndexType> & BasisCache::cellIDs()
{
  return _cellIDs;
}

CellTopoPtr BasisCache::cellTopology()
{
  return _cellTopo;
}

template<>
BasisCache::DeviceView<double> BasisCache::checkoutScratchStorage<double>(BasisCache::DeviceView<double> &dv)
{
  using Scalar = double;
  return SharedScratchStorage<Scalar>::sharedScratchStorage().checkoutScratchStorageMatchingDeviceView(dv,"BasisCache::checkoutScratchStorage()");
}

template<>
void BasisCache::checkinScratchStorage<double>(BasisCache::DeviceView<double> &scratchView)
{
  using Scalar = double;
  SharedScratchStorage<Scalar>::sharedScratchStorage().checkinScratchStorage(scratchView);
}

Camellia::FieldContainer<double> BasisCache::computeParametricPoints()
{
  if (_cubPointsFC.size()==0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "computeParametricPoints() requires reference cell points to be defined.");
  }
  if (_cellTopo->getTensorialDegree() > 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "computeParametricPoints() requires tensorial degree of cell topo to be 0");
  }
  if (_cellTopo->getShardsTopology().getKey()==shards::Quadrilateral<4>::key)
  {
    int cubatureDegree = 0;
    BasisCachePtr parametricCache = BasisCache::parametricQuadCache(cubatureDegree, getRefCellPointsFC(), this->getSideIndex());
    return parametricCache->getPhysicalCubaturePointsFC();
  }
  else if (_cellTopo->getShardsTopology().getKey()==shards::Line<2>::key)
  {
    int cubatureDegree = 0;  // we throw away the computed cubature points, so let's create as few as possible...
    BasisCachePtr parametricCache = BasisCache::parametric1DCache(cubatureDegree);
    parametricCache->setRefCellPointsFC(this->getRefCellPointsFC());
    return parametricCache->getPhysicalCubaturePointsFC();
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported cellTopo");
//    return Camellia::FieldContainer<double>(0);
  }
}

int BasisCache::cubatureDegree()
{
  return _cubDegree;
}

int BasisCache::cubatureEnrichmentDegree()
{
  return _cubatureEnrichmentDegree;
}

void BasisCache::cubatureDegreeForElementType(ElementTypePtr elemType, bool testVsTest, int &cubatureDegree)
{
  int maxTestDegree = elemType->testOrderPtr->maxBasisDegree();

  int cubatureDegreeForMesh = 0;
  if (_mesh.get())
  {
    _transformationFxn = _mesh->getTransformationFunction();
    if (_transformationFxn.get())
    {
      // assuming isoparametric:
      cubatureDegreeForMesh += maxTestDegree;
    }
    // at least for now, what the Mesh's transformation function does is transform from a straight-lined mesh to
    // one with potentially curved edges...
    _composeTransformationFxnWithMeshTransformation = true;
  }

  int maxTrialDegree = testVsTest ? maxTestDegree : elemType->trialOrderPtr->maxBasisDegree();
  cubatureDegree = maxTrialDegree + maxTestDegree + cubatureDegreeForMesh;
}

bool BasisCache::cellTopologyIsSpaceTime()
{
  return (_cellTopo->getTensorialDegree() > 0) && (_cellTopo->getDimension() > _spaceDim);
}

void BasisCache::cubatureDegreeForElementType(ElementTypePtr elemType, bool testVsTest, int &cubatureDegreeSpace, int &cubatureDegreeTime)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Method not yet implemented");
}

double BasisCache::extrusionWidth(const int &d, const int &cellOrdinal)
{
  const auto meshTopo = _mesh->getTopology();
  const auto &vertexIndices = meshTopo->getCell(_cellIDs[cellOrdinal])->vertices();
  const double tol = 1e-14;
  double maxValue = std::numeric_limits<double>::lowest();
  double minValue = std::numeric_limits<double>::max();
  for (const auto & vertexIndex : vertexIndices)
  {
    const double & coord = meshTopo->getVertex(vertexIndex)[d];
    maxValue = std::max(coord,maxValue);
    minValue = std::min(coord,minValue);
  }
  // sanity check: all vertices are either maxValue or minValue (otherwise it's not an orthogonal extrusion)
  for (const auto & vertexIndex : vertexIndices)
  {
    const double & coord = meshTopo->getVertex(vertexIndex)[d];
    const bool matchesMax = std::fabs(coord-maxValue) < tol;
    const bool matchesMin = std::fabs(coord-minValue) < tol;
    
    TEUCHOS_TEST_FOR_EXCEPTION(!matchesMax && !matchesMin, std::invalid_argument, "Not an orthogonal extrusion in the specified dimension.");
  }
  // TODO: worry about the case where the jacobian is negative (and therefore the width should be negative, too).
  return maxValue - minValue;
}

void BasisCache::findMaximumDegreeBasisForSides(DofOrdering &trialOrdering)
{
  _maxDegreeBasisForSide.clear();
  vector<int> sideTrialIDs;
  set<int> trialIDs = trialOrdering.getVarIDs();
  for (set<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++)
  {
    int trialID = *trialIt;
    if (trialOrdering.getSidesForVarID(trialID).size() > 1)
    {
      sideTrialIDs.push_back(trialID);
    }
  }
  int numSides = _cellTopo->getSideCount();

  int numSideTrialIDs = sideTrialIDs.size();
  for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
  {
    BasisPtr maxDegreeBasisOnSide;
    // loop through looking for highest-degree basis
    int maxTrialDegree = -1;
    for (int i=0; i<numSideTrialIDs; i++)
    {
      if (trialOrdering.hasBasisEntry(sideTrialIDs[i], sideOrdinal))
      {
        BasisPtr basis = trialOrdering.getBasis(sideTrialIDs[i],sideOrdinal);
        if (basis->getDegree() > maxTrialDegree)
        {
          maxDegreeBasisOnSide = basis;
          maxTrialDegree = maxDegreeBasisOnSide->getDegree();
        }
      }
    }
    _maxDegreeBasisForSide.push_back(maxDegreeBasisOnSide);
  }
}

bool BasisCache::neglectHessian() const
{
  if (_transformationFxn == Teuchos::null)
  {
    /* Then the transformation is determined by the physical vertices of the element,
       which means the second derivatives of the transformation will be zero.
     
     This does *NOT* mean that the mixed derivative terms in the Hessian (e.g. d^2/(dx dy))
     are zero, and for computing arbitrary second-order derivatives we would need to take these
     into account even for the straight-edge case.  But for computing the "diagonal" terms in
     Intrepid::OPERATOR_D2 (like d^2/dx^2) that contribute the Laplacian, all we need is those same diagonal
     terms in the Hessian.  For the straight-edge case these are zero.
     
     Right now we don't support transformations of basis values for mixed derivative terms, which
     is why it's OK to neglect the Hessian here, and also why it's OK to just check the diagonal
     derivative terms in the curvilinear case below.
     
     */
    return true;
  }
  else
  {
    // for 2nd-order operators other than the Laplacian, we'll also need dx()->dy(), etc.  But
    // for now we're focused on allowing Laplacians to be computed.
    if (_spaceDim == 1)
      return _transformationFxn->dx()->dx()->isZero();
    else if (_spaceDim == 2)
      return _transformationFxn->dx()->dx()->isZero() && _transformationFxn->dy()->dy()->isZero();
    else if (_spaceDim == 3)
      return _transformationFxn->dx()->dx()->isZero() && _transformationFxn->dy()->dy()->isZero() && _transformationFxn->dz()->dz()->isZero();
    else
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported space dimension");
  }
}

MeshPtr BasisCache::mesh()
{
  if ( ! _isSideCache )
  {
    return _mesh;
  }
  else
  {
    return _basisCacheVolume->mesh();
  }
}

void BasisCache::setMesh(MeshPtr mesh)
{
  if ( ! _isSideCache )
  {
    _mesh = mesh;
  }
  else
  {
    _basisCacheVolume->setMesh(mesh);
  }
}

void BasisCache::discardPhysicalNodeInfo()
{
  // discard physicalNodes and all transformed basis values.
  _knownValuesTransformedFC.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();
  _knownValuesTransformedWeighted.clear();

  // These resizings are a way of forcing exceptions, but otherwise
  // probably not that useful.  Plus, if we are reusing a BasisCache
  // it's not infrequently the case that the sizes next time we fill
  // them will be the same...
  // resize all the related Camellia::FieldContainers to reclaim their memory
//  _cellIDs.clear();
//  _cellJacobian.resize(0);
//  _cellJacobInv.resize(0);
//  _cellJacobDetFC.resize(0);
//  _weightedMeasure.resize(0);
//  _physCubPoints.resize(0);
  
  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _physCubPointsIsValid = false;
  _sideNormalsIsValid = false;
}

Camellia::FieldContainer<double> & BasisCache::getWeightedMeasures()
{
  if (!_weightedMeasureIsValid) recomputeMeasures();
  return _weightedMeasure;
}

const Camellia::FieldContainer<double> &BasisCache::getPhysicalCellNodes()
{
  return _physicalCellNodes;
}

const PhysicalTensorPoints & BasisCache::getPhysicalCubaturePoints()
{
  if (!_physCubPointsIsValid) determinePhysicalPoints();
  return _physicalCubaturePoints;
}

const Camellia::FieldContainer<double> & BasisCache::getPhysicalCubaturePointsFC()
{
  if (!_physCubPointsIsValid) determinePhysicalPoints();
  return _physCubPointsFC;
}

Camellia::FieldContainer<double> BasisCache::getCellMeasures()
{
  if (_cubWeightsFC.size() != 0)
  {
    if (!_weightedMeasureIsValid) recomputeMeasures();
    int numCells = _weightedMeasure.dimension(0);
    int numPoints = _weightedMeasure.dimension(1);
    Camellia::FieldContainer<double> cellMeasures(numCells);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        cellMeasures(cellIndex) += _weightedMeasure(cellIndex,ptIndex);
      }
    }
    return cellMeasures;
  }
  else if (_mesh != Teuchos::null)
  {
    int cellOrdinal = 0;
    Camellia::FieldContainer<double> cellMeasures(_cellIDs.size());
    for (GlobalIndexType cellID : _cellIDs)
    {
      cellMeasures(cellOrdinal) = _mesh->getCellMeasure(cellID);
    }
    return cellMeasures;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache::getCellMeasures() requires either a Mesh to be provided or a BasisCache with cubature points as reference points");
  }
}

const Intrepid2::TensorData<double,DeviceType> BasisCache::getCubatureWeights() const
{
  return _cubWeights;
}

const Camellia::FieldContainer<double> & BasisCache::getCubatureWeightsFC()
{
  return _cubWeightsFC;
}

const Camellia::FieldContainer<double> & BasisCache::getJacobian()
{
  if (!_cellJacobianIsValid) determineJacobian();
  return _cellJacobianFC;
}
const Camellia::FieldContainer<double> & BasisCache::getJacobianDet()
{
  if (!_cellJacobianDeterminantIsValid) determineJacobianInverseAndDeterminant();
  return _cellJacobDetFC;
}
const Camellia::FieldContainer<double> & BasisCache::getJacobianInv()
{
  if (!_cellJacobianInverseIsValid) determineJacobianInverseAndDeterminant();
  return _cellJacobInvFC;
}

Intrepid2::BasisValues<double,DeviceType>
BasisCache::getValues(Intrepid2BasisWrapper<double>* intrepid2Basis, Intrepid2::EOperator op,
                      const Intrepid2::TensorPoints<double,DeviceType> inputPoints)
{
  pair< Camellia::Basis<>*, Intrepid2::EOperator> key = make_pair(intrepid2Basis, op);
  
  auto resultFound = _knownValues.find(key);
  if (resultFound != _knownValues.end())
  {
    return resultFound->second;
  }
  else
  {
    auto result = BasisEvaluation::getValues(intrepid2Basis, op, inputPoints);
    pair< pair< Camellia::Basis<>*, Intrepid2::EOperator>, Intrepid2::BasisValues<double,DeviceType> > resultEntry {key, result};
    _knownValues.insert(resultEntry);
//    _knownValues[key] = result; // TODO: submit Intrepid2 PR with default constructor for BasisValues — that will allow use to write this instead of the more complex thing above.
    return result;
  }
}

constFCPtr BasisCache::getValues(BasisPtr basis, Camellia::EOperator op,
                                 bool useCubPointsSideRefCell)
{
  const Camellia::FieldContainer<double>* cubPoints;
  if (useCubPointsSideRefCell)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_cubPointsSideRefCellFC.size()==0,std::invalid_argument,"useCubPointsSideRefCell = true, but _cubPointsSideRefCell is empty!");
    cubPoints = &_cubPointsSideRefCellFC;
  }
  else
  {
    cubPoints = &_cubPointsFC;
  }
  // first, let's check whether the exact request is already known
  pair< Camellia::Basis<>*, Camellia::EOperator> key = make_pair(basis.get(), op);

  if (_knownValuesFC.find(key) != _knownValuesFC.end() )
  {
    return _knownValuesFC[key];
  }
  int componentOfInterest = -1;
  // otherwise, lookup to see whether a related value is already known
  Camellia::EOperator relatedOp = BasisEvaluation::relatedOperator(basis, op, componentOfInterest);

  pair<Camellia::Basis<>*, Camellia::EOperator> relatedKey = key;
  if (relatedOp != op)
  {
    relatedKey = make_pair(basis.get(), relatedOp);
    if (_knownValuesFC.find(relatedKey) == _knownValuesFC.end() )
    {
      // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDim)
      FCPtr relatedResults = BasisEvaluation::getValues(basis,(Camellia::EOperator)relatedOp,*cubPoints);
      _knownValuesFC[relatedKey] = relatedResults;
    }

    constFCPtr relatedResults = _knownValuesFC[relatedKey];
    //    constFCPtr relatedResults = _knownValues[key];
    Camellia::EFunctionSpace fs = basis->functionSpace();
    constFCPtr result = BasisEvaluation::getComponentOfInterest(relatedResults,op,fs,componentOfInterest);
    if ( result.get() == 0 )
    {
      result = relatedResults;
    }
    _knownValuesFC[key] = result;
    return result;
  }
  // if we get here, we should have a standard Intrepid operator *or* OP_DIn, in which case we should
  // be able to: size a Camellia::FieldContainer appropriately, and then call basis->getValues

  // Let's assert that that's the case:
  const bool standardIntrepidOperator = (op >= Camellia::OP_VALUE) && (op <= Camellia::OP_D10);
  const bool camelliaOpDIn            = (op >= Camellia::OP_DI1)   && (op <= Camellia::OP_DI7);
//  if (camelliaOpDIn)
//  {
//    // DEBUGGING
//    std::cout << "BasisCache::getValues(), camelliaOpDIn is TRUE.\n";
//  }
  
  const bool validOperator = standardIntrepidOperator || camelliaOpDIn;
  if ( !validOperator )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown operator.");
  }
  FCPtr result = BasisEvaluation::getValues(basis,op,*cubPoints);
  _knownValuesFC[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell)
{
  TEUCHOS_TEST_FOR_EXCEPTION(!canComputeTransformedValues(op), std::invalid_argument, "computing transformed values of this operator is not supported");
  
  Intrepid2BasisWrapper<double>* basisAsIntrepid2 = dynamic_cast<Intrepid2BasisWrapper<double>*>(basis.get());
  
  pair<Camellia::Basis<>*, Camellia::EOperator> key = make_pair(basis.get(), op);
  if (_knownValuesTransformedFC.find(key) != _knownValuesTransformedFC.end())
  {
    return _knownValuesTransformedFC[key];
  }

  int componentOfInterest;
  ::Camellia::EOperator relatedOp = BasisEvaluation::relatedOperator(basis, op, componentOfInterest);

  pair<Camellia::Basis<>*, Camellia::EOperator> relatedKey = make_pair(basis.get(),(Camellia::EOperator) relatedOp);
  if (_knownValuesTransformedFC.find(relatedKey) == _knownValuesTransformedFC.end())
  {
    constFCPtr transformedValues;
    Camellia::EFunctionSpace fs = basis->functionSpace();
    bool vectorizedBasis = functionSpaceIsVectorized(fs);
    if ( (vectorizedBasis) && (relatedOp == OP_VALUE))
    {
      VectorBasisPtr vectorBasis = Teuchos::rcp( (VectorizedBasis<double, Intrepid::FieldContainer<double> > *) basis.get(), false );
      BasisPtr componentBasis = vectorBasis->getComponentBasis();
      constFCPtr componentReferenceValuesTransformed = getTransformedValues(componentBasis, Camellia::OP_VALUE,
          useCubPointsSideRefCell);
      transformedValues = BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(vectorBasis,
                          Camellia::OP_VALUE,
                          componentReferenceValuesTransformed);
    }
    else
    {
      constFCPtr referenceValues;
      if (basisAsIntrepid2 != NULL)
      {
        const bool standardIntrepidOperator = (relatedOp >= Camellia::OP_VALUE) && (relatedOp <= Camellia::OP_D10);
        TEUCHOS_TEST_FOR_EXCEPTION(!standardIntrepidOperator, std::invalid_argument, "relatedOp must be known to Intrepid2");
        
        Intrepid2::EOperator intrepid2Op = static_cast<Intrepid2::EOperator>(relatedOp);
        const auto &points = useCubPointsSideRefCell ? _cubPointsSideRefCell : _cubPoints;
        TEUCHOS_TEST_FOR_EXCEPTION(!points.isValid(), std::logic_error, "points container is not valid!");
        auto basisValues = getValues(basisAsIntrepid2,intrepid2Op,points);
        auto convertedBasisValues = FieldContainerConversion<double>::fromBasisValues(basisValues);
        referenceValues = Teuchos::rcp(new FieldContainer<double>(convertedBasisValues));
      }
      else
      {
        referenceValues = getValues(basis,(Camellia::EOperator) relatedOp, useCubPointsSideRefCell);
      }
      // TODO: revisit the way we determine numCells??
      int numCells = getPhysicalCubaturePointsFC().dimension(0);
      if (numCells == 0)   // can happen for certain BasisCaches used in SpaceTimeBasisCache
      {
        if (_physicalCellNodes.rank() > 0)
        {
          numCells = _physicalCellNodes.dimension(0);
        }
      }
      transformedValues =
        BasisEvaluation::getTransformedValuesWithBasisValues(basis, (Camellia::EOperator) relatedOp,
            referenceValues, numCells, this);
//      cout << "transformedValues:\n" << *transformedValues;
    }
    _knownValuesTransformedFC[relatedKey] = transformedValues;
  }
  constFCPtr relatedValuesTransformed = _knownValuesTransformedFC[relatedKey];
  constFCPtr result;
  if (op == (Camellia::EOperator)relatedOp)
  {
    result = relatedValuesTransformed;
  }
  else if (   (op != Camellia::OP_CROSS_NORMAL)   && (op != Camellia::OP_DOT_NORMAL)
           && (op != Camellia::OP_TIMES_NORMAL)   && (op != Camellia::OP_VECTORIZE_VALUE)
           && (op != Camellia::OP_TIMES_NORMAL_X) && (op != Camellia::OP_TIMES_NORMAL_Y)
           && (op != Camellia::OP_TIMES_NORMAL_Z) && (op != Camellia::OP_TIMES_NORMAL_T)
     )
  {
    Camellia::EFunctionSpace fs = basis->functionSpace();
    result = BasisEvaluation::getComponentOfInterest(relatedValuesTransformed,op,fs,componentOfInterest);
    if ( result.get() == 0 )
    {
      result = relatedValuesTransformed;
    }
  }
  else
  {
    switch (op)
    {
    case OP_CROSS_NORMAL:
      result = BasisEvaluation::getValuesCrossedWithNormals(relatedValuesTransformed,getSideNormals());
      break;
    case OP_DOT_NORMAL:
      result = BasisEvaluation::getValuesDottedWithNormals(relatedValuesTransformed,getSideNormals());
      break;
    case OP_TIMES_NORMAL:
      result = BasisEvaluation::getValuesTimesNormals(relatedValuesTransformed,getSideNormals());
      break;
    case OP_VECTORIZE_VALUE:
      result = BasisEvaluation::getVectorizedValues(relatedValuesTransformed,_spaceDim);
      break;
    case OP_TIMES_NORMAL_X:
    case OP_TIMES_NORMAL_Y:
    case OP_TIMES_NORMAL_Z:
    {
      int normalComponent = op - OP_TIMES_NORMAL_X;
      result = BasisEvaluation::getValuesTimesNormals(relatedValuesTransformed,getSideNormals(),normalComponent);
    }
    break;
    case OP_TIMES_NORMAL_T:
    {
      if (_cellTopo->getTensorialDegree() == 0)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "temporal normals are not defined for pure spatial topologies");
      }
      int normalComponent = _cellTopo->getDimension() - 1; // time dimension is the last one
      result = BasisEvaluation::getValuesTimesNormals(relatedValuesTransformed,getSideNormalsSpaceTime(),normalComponent);
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled op.");
    }
  }
  if (CACHE_TRANSFORMED_VALUES)
    _knownValuesTransformedFC[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op,
    bool useCubPointsSideRefCell)
{
  pair<Camellia::Basis<>*, Camellia::EOperator> key = make_pair(basis.get(), op);
  if (_knownValuesTransformedWeighted.find(key) != _knownValuesTransformedWeighted.end())
  {
    return _knownValuesTransformedWeighted[key];
  }
  constFCPtr unWeightedValues = getTransformedValues(basis,op, useCubPointsSideRefCell);
  Teuchos::Array<int> dimensions;
  unWeightedValues->dimensions(dimensions);
  Teuchos::RCP< Camellia::FieldContainer<double> > weightedValues = Teuchos::rcp( new Camellia::FieldContainer<double>(dimensions) );
  fst::multiplyMeasure<double>(*weightedValues, getWeightedMeasures(), *unWeightedValues);
  if (CACHE_TRANSFORMED_VALUES)
    _knownValuesTransformedWeighted[key] = weightedValues;
  return weightedValues;
}

/*** SIDE VARIANTS ***/
constFCPtr BasisCache::getValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal,
                                 bool useCubPointsSideRefCell)
{
  return _basisCacheSides[sideOrdinal]->getValues(basis,op,useCubPointsSideRefCell);
}

constFCPtr BasisCache::getTransformedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal,
    bool useCubPointsSideRefCell)
{
  constFCPtr transformedValues;
  if ( ! _isSideCache )
  {
    transformedValues = _basisCacheSides[sideOrdinal]->getTransformedValues(basis,op,useCubPointsSideRefCell);
  }
  else
  {
    transformedValues = getTransformedValues(basis,op,useCubPointsSideRefCell);
  }
  return transformedValues;
}

constFCPtr BasisCache::getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op,
    int sideOrdinal, bool useCubPointsSideRefCell)
{
  return _basisCacheSides[sideOrdinal]->getTransformedWeightedValues(basis,op,useCubPointsSideRefCell);
}

const Camellia::FieldContainer<double> & BasisCache::getPhysicalCubaturePointsForSide(int sideOrdinal)
{
  return _basisCacheSides[sideOrdinal]->getPhysicalCubaturePointsFC();
}

BasisCachePtr BasisCache::getSideBasisCache(int sideOrdinal)
{
  if (sideOrdinal < _basisCacheSides.size() )
    return _basisCacheSides[sideOrdinal];
  else
    return Teuchos::rcp((BasisCache *) NULL);
}

BasisCachePtr BasisCache::getVolumeBasisCache()
{
  return _basisCacheVolume;
}

bool BasisCache::isSideCache()
{
  return _sideIndex >= 0;
}

int BasisCache::getSideIndex() const
{
  return _sideIndex;
}

const Camellia::FieldContainer<double> & BasisCache::getSideUnitNormals(int sideOrdinal)
{
  return _basisCacheSides[sideOrdinal]->getSideNormals();
}

const Intrepid2::TensorPoints<double,DeviceType> BasisCache::getRefCellPoints() const
{
  return _cubPoints;
}

const Camellia::FieldContainer<double>& BasisCache::getRefCellPointsFC()
{
  return _cubPointsFC;
}

Camellia::FieldContainer<double> BasisCache::getRefCellPointsForPhysicalPoints(const Camellia::FieldContainer<double> &physicalPoints, int cellIndex)
{
  int numPoints = physicalPoints.dimension(0);
  int spaceDim = physicalPoints.dimension(1);

  if (_cellTopo->getTensorialDegree() > 0)
  {
    cout << " BasisCache::getRefCellPointsForPhysicalPoints does not support tensorial degree > 1.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache::getRefCellPointsForPhysicalPoints does not support tensorial degree > 1.");
  }

  Camellia::FieldContainer<double> refCellPoints(numPoints,spaceDim);
  Intrepid::CellTools<double>::mapToReferenceFrame(refCellPoints,physicalPoints,_physicalCellNodes,_cellTopo->getShardsTopology(),cellIndex);
  return refCellPoints;
}

const Camellia::FieldContainer<double> &BasisCache::getSideRefCellPointsInVolumeCoordinates()
{
  if (! isSideCache())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "getSideRefCellPointsInVolumeCoordinates() only supported for side caches.");
  }
  return _cubPointsSideRefCellFC;
}

void BasisCache::setRefCellPointsFC(const Camellia::FieldContainer<double> &pointsRefCell)
{
  Camellia::FieldContainer<double> cubWeights;
  this->setRefCellPointsFC(pointsRefCell, cubWeights);
}

void BasisCache::setRefCellPointsFC(const Camellia::FieldContainer<double> &pointsRefCell, const Camellia::FieldContainer<double> &cubWeights,
                                    int cubatureDegree, bool recomputePhysicalMeasures)
{
  _cubPointsFC = pointsRefCell;
  _cubDegree = cubatureDegree;
  int numPoints = pointsRefCell.dimension(0);

  if ( isSideCache() )   // then we need to map pointsRefCell (on side) into volume coordinates, and store in _cubPointsSideRefCell
  {
    int cellDim = _cellTopo->getDimension(); // will be _spaceDim + 1 for space-time CellTopologies.
    // for side cache, cellDim is the spatial dimension of the volume cache's cellTopology
    _cubPointsSideRefCellFC.resize(numPoints, cellDim);
    // _cellTopo is the volume cell topology for side basis caches.
    int sideDim = cellDim - 1;
    CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCellFC, _cubPointsFC, sideDim, _sideIndex, _cellTopo);
  }

  _knownValues.clear();
  _knownValuesFC.clear();
  _knownValuesTransformedFC.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();

  _cubWeightsFC = cubWeights;

  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;
  
  // Keep Intrepid2 containers in sync.  (We will eventually move to using these exclusively.)
  _cubPoints  = FieldContainerConversion<double>::toTensorPoints(_cubPointsFC);
  _cubWeights = FieldContainerConversion<double>::toTensorData(  _cubWeightsFC);
  
  if ( isSideCache() )
  {
    _cubPointsSideRefCell = FieldContainerConversion<double>::toTensorPoints(_cubPointsSideRefCellFC);
  }
}

void BasisCache::setRefCellPoints(Intrepid2::TensorPoints<double,DeviceType> pointsRefCell)
{
  Intrepid2::TensorData<double,DeviceType> emptyWeights;
  setRefCellPoints(pointsRefCell, emptyWeights);
}

void BasisCache::setRefCellPoints(Intrepid2::TensorPoints<double,DeviceType> pointsRefCell, Intrepid2::TensorData<double,DeviceType> weightsRefCell)
{
  _cubPoints = pointsRefCell;
  _cubWeights = weightsRefCell;
  const int numPoints = pointsRefCell.extent_int(0);
  
  if ( isSideCache() )   // then we need to map pointsRefCell (on side) into volume coordinates, and store in _cubPointsSideRefCell
  {
    int cellDim = _cellTopo->getDimension(); // will be _spaceDim + 1 for space-time CellTopologies.
    // _cellTopo is the volume cell topology for side basis caches.
    int sideDim = cellDim - 1;
    CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
  }
  
  // the below is mostly copied from setRefCellPointsFC().
  // it basically allows us to make sure that when the FC-based interface is used, points are in sync with what is being specified
  // in this setRefCellPoints() call in terms of Intrepid2-based containers.  Ideally, we will move everything to Intrepid2-based
  // containers, and thus be able to avoid the redundancy here.
  
  auto pointsRefCellFC  = FieldContainerConversion<double>::fromTensorPoints(pointsRefCell);
  auto weightsRefCellFC = FieldContainerConversion<double>::fromTensorData(weightsRefCell);
  
  _cubPointsFC = pointsRefCellFC;
  _cubDegree = -1; // unknown

  if ( isSideCache() )   // then we need to map pointsRefCell (on side) into volume coordinates, and store in _cubPointsSideRefCellFC
  {
    int cellDim = _cellTopo->getDimension(); // will be _spaceDim + 1 for space-time CellTopologies.
    // for side cache, cellDim is the spatial dimension of the volume cache's cellTopology
    _cubPointsSideRefCellFC.resize(numPoints, cellDim);
    // _cellTopo is the volume cell topology for side basis caches.
    int sideDim = cellDim - 1;
    CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCellFC, _cubPointsFC, sideDim, _sideIndex, _cellTopo);
  }

  _knownValues.clear();
  _knownValuesFC.clear();
  _knownValuesTransformedFC.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();

  _cubWeightsFC = weightsRefCellFC;

  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;
}

const Camellia::FieldContainer<double> & BasisCache::getSideNormals()
{
  if (!_sideNormalsIsValid) determineSideNormals();
  return _sideNormals;
}

const Camellia::FieldContainer<double> & BasisCache::getSideNormalsSpaceTime()
{
  if (!_sideNormalsIsValid) determineSideNormals();
  if (_cellTopo->getTensorialDegree() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "space-time side normals are only defined for cell topologies with tensorial degree > 0.");
  }
  return _sideNormalsSpaceTime;
}

void BasisCache::setSideNormals(Camellia::FieldContainer<double> &sideNormals)
{
  _sideNormals = sideNormals;
  _sideNormalsIsValid = true;
}

const Camellia::FieldContainer<double> & BasisCache::getCellSideParities()
{
  return _cellSideParities;
}

void BasisCache::setCellIDs(const std::vector<GlobalIndexType> &cellIDs)
{
  _cellIDs = cellIDs;
}

void BasisCache::setCellSideParities(const Camellia::FieldContainer<double> &cellSideParities)
{
  TEUCHOS_TEST_FOR_EXCEPTION((cellSideParities.rank() != 2) || (cellSideParities.dimension(1) < _cellTopo->getSideCount()),
                             std::invalid_argument, "Incorrectly sized cellSideParities");
  if (&_cellSideParities != &cellSideParities) // self-assignment prohibited in Camellia::FieldContainer
  {
    _cellSideParities = cellSideParities;
  }
}

void BasisCache::setTransformationFunction(TFunctionPtr<double> fxn, bool composeWithMeshTransformation)
{
  _transformationFxn = fxn;
  _composeTransformationFxnWithMeshTransformation = composeWithMeshTransformation;
  // recompute physical points and jacobian values
  
  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;
}

void BasisCache::determinePhysicalPoints()
{
  int cellDim = _cellTopo->getDimension();
  if (cellDim==0)
  {
    // then the "cubature" points are just identical to the cell nodes
    _physCubPointsIsValid = true;
    _physCubPointsFC = _physicalCellNodes;
    
    // copy the physical points (nodes) into a single PhysicalTensorPoints component
    copyPhysicalPointsFieldContainerToPhysicalTensorPoints();
    
    return;
  }
  int numPoints = isSideCache() ? _cubPointsSideRefCellFC.dimension(0) : _cubPointsFC.dimension(0);
  if ( TFunction<double>::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation)
  {
    // _spaceDim for side cache refers to the volume cache's spatial dimension
    _physCubPointsFC.resize(_numCells, numPoints, cellDim);

    if (numPoints > 0)
    {
      const auto & volumeCubPointsFC = isSideCache() ? _cubPointsSideRefCellFC : _cubPointsFC;
      CamelliaCellTools::mapToPhysicalFrame(_physCubPointsFC,volumeCubPointsFC,_physicalCellNodes,_cellTopo);
    }
    
    const int numCells = _physicalCellNodes.dimension(0);
    using std::vector;
    using DynRankView = Kokkos::DynRankView<double, DeviceType>;
    using TensorPoints = Intrepid2::TensorPoints<double,DeviceType>;
    const auto & volumeCubPoints = isSideCache() ? _cubPointsSideRefCell : _cubPoints;
    
    if ((_cellTopo->getTensorialDegree() > 0) && (volumeCubPoints.numTensorComponents() > 1))
    {
      // TODO: factor out the following to its own CamelliaCellTools::mapToPhysicalFrame() method.
      // transform in each tensorial component separately
      // in principle, we could support cases where the tensorial components in cubPoints didn't exactly match up with those in cellTopo,
      // but that is not our immediate use case:
      
      TEUCHOS_TEST_FOR_EXCEPTION(_cellTopo->getDimension() != volumeCubPoints.extent_int(1), std::invalid_argument, "cellTopo's tensorial component count must agree in total dimension");
      
      int coordinateDimsSoFar = 0;
      int componentNodeSpan = 1; // ordinal distance between adjacent nodes in the current component; 1 for base cell topo, baseCellTopoNodeCount for first extrusion component, baseCellTopoNodeCount * 2 for next, etc.
      
      // base CellTopo may consume multiple constituent cubPoints: hexTopo takes 3 1D cubPoints, e.g.  (tensorial extrusions are all 1D, by contrast)
      // so we treat the base CellTopo separately:
      CellTopoPtr baseTopo = _cellTopo->getBaseTopology();
      
      int cubPointsComponentOrdinal = 0;
      vector<DynRankView> baseTopoRefPointComponents;
      int numDimsSoFar = 0;
      while (numDimsSoFar < baseTopo->getDimension())
      {
        TEUCHOS_TEST_FOR_EXCEPTION(cubPointsComponentOrdinal >= volumeCubPoints.numTensorComponents(), std::invalid_argument, "out of bounds in cubPoints tensor component ordinal");
        DynRankView cubPointsComponent = volumeCubPoints.getTensorComponent(cubPointsComponentOrdinal);
        baseTopoRefPointComponents.push_back(cubPointsComponent);
        
        numDimsSoFar += cubPointsComponent.extent_int(1); // (P,D)
        cubPointsComponentOrdinal++;
      }
      
      vector<DynRankView> reconciledCubPointsComponents;
      if (baseTopoRefPointComponents.size() == 1)
      {
        const DynRankView & baseTopoRefPoints = baseTopoRefPointComponents[0];
        reconciledCubPointsComponents.push_back(baseTopoRefPoints);
      }
      else
      {
        TensorPoints baseTopoTensorPoints(baseTopoRefPointComponents);
        DynRankView baseTopoRefPoints("baseTopoRefPoints", baseTopoTensorPoints.extent_int(0), baseTopoTensorPoints.extent_int(1));
        ::Camellia::deepCopy<2,DynRankView,TensorPoints>(baseTopoRefPoints, baseTopoTensorPoints);
        
        reconciledCubPointsComponents.push_back(baseTopoRefPoints);
      }
      
      for (int r=cubPointsComponentOrdinal; r<volumeCubPoints.numTensorComponents(); r++)
      {
        reconciledCubPointsComponents.push_back(volumeCubPoints.getTensorComponent(r));
      }
      
      TensorPoints reconciledCubPoints(reconciledCubPointsComponents);
      
      vector< DynRankView > physicalTensorPointsComponents;

      for (int r=0; r<reconciledCubPoints.numTensorComponents(); r++)
      {
        const auto & refPointsComponent = reconciledCubPoints.getTensorComponent(r);
        
        CellTopoPtr cellTopoComponent = (r == 0) ? _cellTopo->getBaseTopology() : CellTopology::line();
        
        TEUCHOS_TEST_FOR_EXCEPTION(refPointsComponent.extent_int(1) != cellTopoComponent->getDimension(), std::invalid_argument, "spatial dimensions for cell topo component and ref points component must match");
        const int numPointsComponent = refPointsComponent.extent_int(0);
        const int spaceDimComponent  = refPointsComponent.extent_int(1);
        
        Kokkos::DynRankView<double, DeviceType> physPointsComponent("physical points component",numCells,numPointsComponent,spaceDimComponent);
        Kokkos::DynRankView<double, DeviceType> physPointsNodes("physical component nodes",numCells,cellTopoComponent->getNodeCount(),spaceDimComponent);
        
        auto physPointsNodesHost = Kokkos::create_mirror(physPointsNodes);
        
        for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
        {
          int compositeNodeOrdinal = 0;
          for (int componentNode=0; componentNode<cellTopoComponent->getNodeCount(); componentNode++)
          {
            for (int d=0; d<spaceDimComponent; d++)
            {
              physPointsNodesHost(cellOrdinal,componentNode,d) = _physicalCellNodes(cellOrdinal,compositeNodeOrdinal,d+coordinateDimsSoFar);
            }
            compositeNodeOrdinal += componentNodeSpan;
          }
        }
        Kokkos::deep_copy(physPointsNodes,physPointsNodesHost);
        
        if (spaceDimComponent > 0)
        {
          BasisPtr lowestOrderBasis = intrepid2HGRADLowestOrder(cellTopoComponent->getShardsTopology());
          Intrepid2::CellTools<DeviceType>::mapToPhysicalFrame(physPointsComponent, refPointsComponent, physPointsNodes, lowestOrderBasis->intrepid2Basis());
        }
        else
        {
          // if spaceDimComponent == 0, the component is just a node and we can skip the ref-to-physical mapping…
        }
        
        physicalTensorPointsComponents.push_back(physPointsComponent);
        
        componentNodeSpan *= cellTopoComponent->getNodeCount();
        coordinateDimsSoFar += cellTopoComponent->getDimension();
      }
      _physicalCubaturePoints = PhysicalTensorPoints(physicalTensorPointsComponents);
    }
    else // no exploitable tensor structure in physical points
    {
      // copy all physical points into a single physicalTensorPoints component
      copyPhysicalPointsFieldContainerToPhysicalTensorPoints();
    }
  }
  else
  {
    // if we get here, then Function is meant to work on reference cell
    // unsupported for now
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Reference-cell based transformations presently unsupported");
    // (what we need to do here is either copy the _cubPoints to _numCells locations in _physCubPoints,
    //  or make the relevant Function simply work on the reference points, returning a Camellia::FieldContainer
    //  with a numCells dimension.  The latter makes more sense to me--and is more efficient.  The Function should
    //  simply call BasisCache->getRefCellPointsFC()...  On this idea, we don't even have to do anything special in
    //  BasisCache: the if clause above only serves to save us a little computational effort.)
  }
  if ( ! TFunction<double>::isNull(_transformationFxn) )
  {
    Camellia::FieldContainer<double> newPhysCubPoints(_numCells,numPoints,cellDim);
    BasisCachePtr thisPtr = Teuchos::rcp(this,false);

    // the usual story: we don't want to use the transformation Function inside the BasisCache
    // while the transformation Function is using the BasisCache to determine its values.
    // So we move _transformationFxn out of the way for a moment:
    TFunctionPtr<double> transformationFxn = _transformationFxn;
    _transformationFxn = TFunction<double>::null();
    {
      // size cell Jacobian containers—the dimensions are used even for HGRAD basis OP_VALUE, which
      // may legitimately be invoked by transformationFxn, below...
      _cellJacobianFC.resize(_numCells, numPoints, cellDim, cellDim);
      _cellJacobInvFC.resize(_numCells, numPoints, cellDim, cellDim);
      _cellJacobDetFC.resize(_numCells, numPoints);
      // (the sizes here agree with what's done in determineJacobian, so the resizing there should be
      //  basically free if we've done it here.)
    }

    transformationFxn->values(newPhysCubPoints, thisPtr);
    _transformationFxn = transformationFxn;

    _physCubPointsFC = newPhysCubPoints;
    // if there is a transformation function, we assume that there is not exploitable tensor structure in physical space:
    copyPhysicalPointsFieldContainerToPhysicalTensorPoints();
  }
  _physCubPointsIsValid = true;
}

void BasisCache::determineJacobian()
{
  int cellDim = _cellTopo->getDimension();

  if (cellDim == 0)
  {
    // Jacobians not meaningful then...
    _cellJacobianIsValid = true;
    return;
  }

  // Compute cell Jacobians, their inverses and their determinants
  int numCubPoints = isSideCache() ? _cubPointsSideRefCellFC.dimension(0) : _cubPointsFC.dimension(0);

  // Containers for Jacobian
  _cellJacobianFC.resize(_numCells, numCubPoints, cellDim, cellDim);

  if (numCubPoints == 0)
  {
    _cellJacobianIsValid = true;
    return;
  }

  if ( TFunction<double>::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation)
  {
    if (!isSideCache())
      CamelliaCellTools::setJacobian(_cellJacobianFC, _cubPointsFC, _physicalCellNodes, _cellTopo);
    else
      CamelliaCellTools::setJacobian(_cellJacobianFC, _cubPointsSideRefCellFC, _physicalCellNodes, _cellTopo);
  }
  _cellJacobianIsValid = true;

  if (_transformationFxn != Teuchos::null )
  {
    BasisCachePtr thisPtr = Teuchos::rcp(this,false);
    if (_composeTransformationFxnWithMeshTransformation)
    {
      // then we need to multiply one Jacobian by the other
      Camellia::FieldContainer<double> fxnJacobian(_numCells,numCubPoints,cellDim,cellDim);
      // a little quirky, but since _transformationFxn calls BasisCache in its values determination,
      // we disable the _transformationFxn during the call to grad()->values()
      // TODO: consider moving this logic (turning on/off the BasisCache's transformation function during the values call for gradients) into MeshTransformationFunction.  It seems like that's better from an encapsulation point of view.  Unless it is the case that sometimes we *do* want the transformationFunction gradient to be transformed according to some BasisCache's transformation function.  But that seems unlikely.
      TFunctionPtr<double> fxnCopy = _transformationFxn;
      _transformationFxn = TFunction<double>::null();
      fxnCopy->grad()->values( fxnJacobian, thisPtr );
      _transformationFxn = fxnCopy;

//      cout << "fxnJacobian:\n" << fxnJacobian;
//      cout << "_cellJacobian before multiplication:\n" << _cellJacobian;
      // TODO: check that the order of multiplication is correct!
      Camellia::FieldContainer<double> cellJacobianToMultiply(_cellJacobianFC); // tensorMultiplyDataData doesn't support multiplying in place
      fst::tensorMultiplyDataData<double>( _cellJacobianFC, fxnJacobian, cellJacobianToMultiply );
//      cout << "_cellJacobian after multiplication:\n" << _cellJacobian;
    }
    else
    {
      _transformationFxn->grad()->values( _cellJacobianFC, thisPtr );
    }
  }
}

void BasisCache::determineJacobianInverseAndDeterminant()
{
  int cellDim = _cellTopo->getDimension();
  
  if (cellDim == 0)
  {
    // Jacobians not meaningful then...
    _cellJacobianInverseIsValid = true;
    _cellJacobianDeterminantIsValid = true;
    return;
  }
  
  // Compute cell Jacobians, their inverses and their determinants
  int numCubPoints = isSideCache() ? _cubPointsSideRefCellFC.dimension(0) : _cubPointsFC.dimension(0);
  
  _cellJacobInvFC.resize(_numCells, numCubPoints, cellDim, cellDim);
  _cellJacobDetFC.resize(_numCells, numCubPoints);
  SerialDenseWrapper::determinantAndInverse(_cellJacobDetFC, _cellJacobInvFC, getJacobian());
  _cellJacobianInverseIsValid = true;
  _cellJacobianDeterminantIsValid = true;
}

void BasisCache::setJacobian(const Camellia::FieldContainer<double> &jacobian)
{
  _cellJacobianFC = jacobian;
  _cellJacobianIsValid = true;
}

void BasisCache::setJacobianDet(const Camellia::FieldContainer<double> &jacobianDet)
{
  _cellJacobDetFC = jacobianDet;
  _cellJacobianDeterminantIsValid = true;
}

void BasisCache::setJacobianInv(const Camellia::FieldContainer<double> &jacobianInv)
{
  _cellJacobInvFC = jacobianInv;
  _cellJacobianInverseIsValid = true;
}

void BasisCache::setPhysicalCellNodes(const Camellia::FieldContainer<double> &physicalCellNodes,
                                      const vector<GlobalIndexType> &cellIDs, bool createSideCacheToo)
{
  discardPhysicalNodeInfo(); // necessary to get rid of transformed values, which will no longer be valid

  _physicalCellNodes = physicalCellNodes;
  _numCells = physicalCellNodes.dimension(0);

  if (physicalCellNodes.dimension(2) != max((int)_cellTopo->getDimension(), 1))
  {
    TEUCHOS_TEST_FOR_EXCEPTION(physicalCellNodes.dimension(2) != max((int)_cellTopo->getDimension(), 1), std::invalid_argument, "physicalCellNodes.dimension(2) must equal cellTopo's dimension!");
  }

  _cellIDs = cellIDs;
  // Compute cell Jacobians, their inverses and their determinants

  _cellJacobianIsValid = false;
  _cellJacobianInverseIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _weightedMeasureIsValid = false;
  _sideNormalsIsValid = false;
  _physCubPointsIsValid = false;

  if ( ! isSideCache() && createSideCacheToo )
  {
    // we only actually create side caches anew if they don't currently exist
    if (_basisCacheSides.size() == 0)
    {
      createSideCaches();
    }
    int numSides = _cellTopo->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++)
    {
      _basisCacheSides[sideOrdinal]->setPhysicalCellNodes(physicalCellNodes, cellIDs, false);
    }
  }
  else if (! isSideCache() && ! createSideCacheToo )
  {
    // then we have side caches whose values are going to be stale: we should delete these
    _basisCacheSides.clear();
  }
}

void BasisCache::setPhysicalCubaturePoints(const Camellia::FieldContainer<double> &physicalCubaturePoints)
{
  _physCubPointsFC = physicalCubaturePoints;
  _physCubPointsIsValid = true;
  
  // invalidate jacobians
  _cellJacobianIsValid = false;
  _cellJacobianDeterminantIsValid = false;
  _cellJacobianInverseIsValid = false;
  
  // set numCells
  _numCells = physicalCubaturePoints.dimension(0);
  
  copyPhysicalPointsFieldContainerToPhysicalTensorPoints();
}

int BasisCache::maxTestDegree()
{
  return _maxTestDegree;
}

int BasisCache::maxTrialDegree()
{
  return _maxTestDegree;
}

int BasisCache::numCells() const
{
  return _numCells;
}

int BasisCache::getSpaceDim()
{
  return _spaceDim;
}

// static convenience constructors:
BasisCachePtr BasisCache::parametric1DCache(int cubatureDegree)
{
  return BasisCache::basisCache1D(0, 1, cubatureDegree);
}

BasisCachePtr BasisCache::parametricQuadCache(int cubatureDegree, const Camellia::FieldContainer<double> &refCellPoints, int sideCacheIndex)
{
  int numCells = 1;
  int numVertices = 4;
  int spaceDim = 2;
  Camellia::FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
  physicalCellNodes(0,0,0) = 0;
  physicalCellNodes(0,0,1) = 0;
  physicalCellNodes(0,1,0) = 1;
  physicalCellNodes(0,1,1) = 0;
  physicalCellNodes(0,2,0) = 1;
  physicalCellNodes(0,2,1) = 1;
  physicalCellNodes(0,3,0) = 0;
  physicalCellNodes(0,3,1) = 1;

  bool creatingSideCache = (sideCacheIndex != -1);

  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  BasisCachePtr parametricCache = Teuchos::rcp( new BasisCache(physicalCellNodes, quad_4, cubatureDegree, creatingSideCache));

  if (!creatingSideCache)
  {
    parametricCache->setRefCellPointsFC(refCellPoints);
    return parametricCache;
  }
  else
  {
    parametricCache->getSideBasisCache(sideCacheIndex)->setRefCellPointsFC(refCellPoints);
    return parametricCache->getSideBasisCache(sideCacheIndex);
  }
}

BasisCachePtr BasisCache::parametricQuadCache(int cubatureDegree)
{
  int numCells = 1;
  int numVertices = 4;
  int spaceDim = 2;
  Camellia::FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
  physicalCellNodes(0,0,0) = 0;
  physicalCellNodes(0,0,1) = 0;
  physicalCellNodes(0,1,0) = 1;
  physicalCellNodes(0,1,1) = 0;
  physicalCellNodes(0,2,0) = 1;
  physicalCellNodes(0,2,1) = 1;
  physicalCellNodes(0,3,0) = 0;
  physicalCellNodes(0,3,1) = 1;

  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  return Teuchos::rcp( new BasisCache(physicalCellNodes, quad_4, cubatureDegree));
}

bool BasisCache::testVsTest()
{
  return _maxTrialDegree == _maxTestDegree;
}

BasisCachePtr BasisCache::basisCache1D(double x0, double x1, int cubatureDegree)   // x0 and x1: physical space endpoints
{
  int numCells = 1;
  int numVertices = 2;
  int spaceDim = 1;
  Camellia::FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
  physicalCellNodes(0,0,0) = x0;
  physicalCellNodes(0,1,0) = x1;
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  return Teuchos::rcp( new BasisCache(physicalCellNodes, line_2, cubatureDegree));
}

BasisCachePtr BasisCache::basisCacheForCell(MeshPtr mesh, GlobalIndexType cellID,
                                            bool testVsTest, int cubatureDegreeEnrichment)
{
  ElementTypePtr elemType = mesh->getElementType(cellID);
  return basisCacheForCell(mesh,mesh->getTopology(),cellID,elemType,testVsTest,cubatureDegreeEnrichment,mesh->tensorTopologyMeansSpaceTime());
}

BasisCachePtr BasisCache::basisCacheForCell(ConstMeshTopologyViewPtr meshTopology, GlobalIndexType cellID, ElementTypePtr elemType,
                                            bool testVsTest, int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime)
{
  return basisCacheForCell(Teuchos::null,meshTopology,cellID,elemType,testVsTest,cubatureDegreeEnrichment,tensorProductTopologyMeansSpaceTime);
}

BasisCachePtr BasisCache::basisCacheForCell(MeshPtr mesh, ConstMeshTopologyViewPtr meshTopology, GlobalIndexType cellID, ElementTypePtr elemType,
                                            bool testVsTest, int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime)
{
  // mesh is allowed to be null
  vector<GlobalIndexType> cellIDs(1,cellID);
  bool includeCellDimension = true;
  Camellia::FieldContainer<double> physicalCellNodes = meshTopology->physicalCellNodesForCell(cellID, includeCellDimension);
  if (tensorProductTopologyMeansSpaceTime && (elemType->cellTopoPtr->getTensorialDegree() > 0))
  {
    CellTopoPtr spaceTimeTopo = elemType->cellTopoPtr;
    // check that the physical nodes are in fact in a tensor product structure:
    CellTopoPtr spaceTopo = elemType->cellTopoPtr->getTensorialComponent();
    CellTopoPtr timeTopo = CellTopology::line();
    Camellia::FieldContainer<double> physicalCellNodesSpace(1, spaceTopo->getNodeCount(), spaceTopo->getDimension());
    Camellia::FieldContainer<double> physicalCellNodesTime(1, timeTopo->getNodeCount(), timeTopo->getDimension());
    vector<unsigned> componentNodes(2);
    for (int spaceNodeOrdinal=0; spaceNodeOrdinal<spaceTopo->getNodeCount(); spaceNodeOrdinal++)
    {
      componentNodes[0] = spaceNodeOrdinal;
      componentNodes[1] = 0; // fix the 0 node ordinal in time
      int spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
      for (int d=0; d<spaceTopo->getDimension(); d++)
      {
        physicalCellNodesSpace(0,spaceNodeOrdinal,d) = physicalCellNodes(0,spaceTimeNodeOrdinal,d);
      }
      // check that the time 1 node matches the time 0 node
      componentNodes[1] = 1;
      double tol = 1e-15;
      spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
      for (int d=0; d<spaceTopo->getDimension(); d++)
      {
        double diff = abs(physicalCellNodesSpace(0,spaceNodeOrdinal,d) -physicalCellNodes(0,spaceTimeNodeOrdinal,d));
        if (diff > tol)
        {
          cout << "physical cell nodes are not in a tensor product structure; this is not supported.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-tensor product space time mesh");
        }
      }
    }
    for (int timeNodeOrdinal=0; timeNodeOrdinal<timeTopo->getNodeCount(); timeNodeOrdinal++)
    {
      componentNodes[0] = 0;
      componentNodes[1] = timeNodeOrdinal;
      int spaceDim = spaceTopo->getDimension();
      int spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
      physicalCellNodesTime(0,timeNodeOrdinal,0) = physicalCellNodes(0,spaceTimeNodeOrdinal,spaceDim);

      for (int spaceNodeOrdinal=1; spaceNodeOrdinal<spaceTopo->getNodeCount(); spaceNodeOrdinal++)
      {
        // check that the other nodes match
        componentNodes[0] = spaceNodeOrdinal;
        double tol = 1e-15;
        spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
        double diff = abs(physicalCellNodesTime(0,timeNodeOrdinal,0) - physicalCellNodes(0,spaceTimeNodeOrdinal,spaceDim));
        if (diff > tol)
        {
          cout << physicalCellNodes;
          cout << "physical cell nodes are not in a tensor product structure; this is not supported.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-tensor product space time mesh");
        }
      }
    }
    BasisCachePtr basisCache = Teuchos::rcp( new SpaceTimeBasisCache(mesh, elemType, physicalCellNodesSpace,
                               physicalCellNodesTime, physicalCellNodes, cellIDs,
                               testVsTest, cubatureDegreeEnrichment) );
    return basisCache;
  }

  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh, testVsTest, cubatureDegreeEnrichment, tensorProductTopologyMeansSpaceTime) );
  bool createSideCache = true;
  basisCache->setPhysicalCellNodes(physicalCellNodes, cellIDs, createSideCache);
  if (mesh != Teuchos::null)
  {
    basisCache->setCellSideParities(mesh->cellSideParitiesForCell(cellID));
  }

  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForCellTopology(CellTopoPtr cellTopo, int cubatureDegree,
    const Camellia::FieldContainer<double> &physicalCellNodes,
    bool createSideCacheToo,
    bool tensorProductTopologyMeansSpaceTime)
{
  if (tensorProductTopologyMeansSpaceTime && (cellTopo->getTensorialDegree() > 0))
  {
    int numCells = physicalCellNodes.dimension(0);
    CellTopoPtr spaceTimeTopo = cellTopo;

    // check that the physical nodes are in fact in a tensor product structure:
    CellTopoPtr spaceTopo = spaceTimeTopo->getTensorialComponent();
    Camellia::FieldContainer<double> physicalCellNodesSpace(numCells, spaceTopo->getNodeCount(), max((int)spaceTopo->getDimension(),1));

    CellTopoPtr timeTopo = CellTopology::line();
    Camellia::FieldContainer<double> physicalCellNodesTime(numCells, timeTopo->getNodeCount(), timeTopo->getDimension());

    if (spaceTopo->getDimension()==0)   // spatial topology is just a Node; handle this separately
    {
      physicalCellNodesSpace.initialize(0.0);
    }
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      vector<unsigned> componentNodes(2);
      if (spaceTopo->getDimension() > 0)
      {
        for (int spaceNodeOrdinal=0; spaceNodeOrdinal<spaceTopo->getNodeCount(); spaceNodeOrdinal++)
        {
          componentNodes[0] = spaceNodeOrdinal;
          componentNodes[1] = 0; // fix the 0 node ordinal in time
          int spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
          for (int d=0; d<physicalCellNodesSpace.dimension(2); d++)
          {
            physicalCellNodesSpace(cellOrdinal,spaceNodeOrdinal,d) = physicalCellNodes(cellOrdinal,spaceTimeNodeOrdinal,d);
          }
          // check that the time 1 node matches the time 0 node
          componentNodes[1] = 1;
          double tol = 1e-15;
          spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
          for (int d=0; d<physicalCellNodesSpace.dimension(2); d++)
          {
            double diff = abs(physicalCellNodesSpace(cellOrdinal,spaceNodeOrdinal,d) -physicalCellNodes(cellOrdinal,spaceTimeNodeOrdinal,d));
            if (diff > tol)
            {
              cout << "physical cell nodes are not in a tensor product structure; this is not supported.\n";
              TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-tensor product space time mesh");
            }
          }
        }
      }

      for (int timeNodeOrdinal=0; timeNodeOrdinal<timeTopo->getNodeCount(); timeNodeOrdinal++)
      {
        componentNodes[0] = 0;
        componentNodes[1] = timeNodeOrdinal;
        int spaceDim = spaceTopo->getDimension();
        int spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
        physicalCellNodesTime(cellOrdinal,timeNodeOrdinal,0) = physicalCellNodes(cellOrdinal,spaceTimeNodeOrdinal,spaceDim);

        for (int spaceNodeOrdinal=1; spaceNodeOrdinal<spaceTopo->getNodeCount(); spaceNodeOrdinal++)
        {
          // check that the other nodes match
          componentNodes[0] = spaceNodeOrdinal;
          double tol = 1e-15;
          spaceTimeNodeOrdinal = spaceTimeTopo->getNodeFromTensorialComponentNodes(componentNodes);
          double diff = abs(physicalCellNodesTime(cellOrdinal,timeNodeOrdinal,0) - physicalCellNodes(cellOrdinal,spaceTimeNodeOrdinal,spaceDim));
          if (diff > tol)
          {
            cout << physicalCellNodes;
            cout << "physical cell nodes are not in a tensor product structure; this is not supported.\n";
            TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-tensor product space time mesh");
          }
        }
      }
    }
    BasisCachePtr basisCache = Teuchos::rcp( new SpaceTimeBasisCache(physicalCellNodesSpace,
                               physicalCellNodesTime,
                               physicalCellNodes,
                               spaceTimeTopo, cubatureDegree) );
    return basisCache;
  }

  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(physicalCellNodes,cellTopo,cubatureDegree,
                                          createSideCacheToo, tensorProductTopologyMeansSpaceTime));

  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForCellType(MeshPtr mesh, ElementTypePtr elemType, bool testVsTest, int cubatureDegreeEnrichment)   // for cells on the local MPI node
{
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh, testVsTest, cubatureDegreeEnrichment) );
  bool createSideCache = true;
  vector<GlobalIndexType> cellIDs = mesh->cellIDsOfType(elemType);
  if (cellIDs.size() > 0)
  {
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, createSideCache);
  }

  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForReferenceCell(shards::CellTopology &shardsTopo, int cubatureDegree, bool createSideCacheToo)
{
  CellTopoPtr cellTopo = CellTopology::cellTopology(shardsTopo);

  return basisCacheForReferenceCell(cellTopo, cubatureDegree, createSideCacheToo);
}

BasisCachePtr BasisCache::basisCacheForReferenceCell(CellTopoPtr cellTopo, int cubatureDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  Camellia::FieldContainer<double> cellNodes(cellTopo->getNodeCount(),cellTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(cellNodes, cellTopo);
  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));

  BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubatureDegree, cellNodes, createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForRefinedReferenceCell(CellTopoPtr cellTopo, int cubatureDegree,
    RefinementBranch refinementBranch, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  if (refinementBranch.size() == 0)
  {
    // no refinements; we want the full reference cell
    return basisCacheForReferenceCell(cellTopo, cubatureDegree, createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  }
  
  Camellia::FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinementBranch);

  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  
  BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubatureDegree, cellNodes, createSideCacheToo, tensorProductTopologyMeansSpaceTime);

  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForRefinedReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree,
    RefinementBranch refinementBranch, bool createSideCacheToo)
{
  Camellia::FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinementBranch);

  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(cellNodes,cellTopo,cubatureDegree,createSideCacheToo));
  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForRefinedReferenceCell(int cubatureDegree, RefinementBranch refinementBranch,
                                                            bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  CellTopoPtr cellTopo = RefinementPattern::descendantTopology(refinementBranch);
  Camellia::FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinementBranch);
  
  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubatureDegree, cellNodes, createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  return basisCache;
}

BasisCachePtr BasisCache::quadBasisCache(double width, double height, int cubDegree, bool createSideCacheToo)
{
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  Camellia::FieldContainer<double> physicalCellNodes(1,4,2);
  physicalCellNodes(0,0,0) = 0;
  physicalCellNodes(0,0,1) = 0;
  physicalCellNodes(0,1,0) = width;
  physicalCellNodes(0,1,1) = 0;
  physicalCellNodes(0,2,0) = width;
  physicalCellNodes(0,2,1) = height;
  physicalCellNodes(0,3,0) = 0;
  physicalCellNodes(0,3,1) = height;

  return Teuchos::rcp(new BasisCache(physicalCellNodes, quad_4, cubDegree, createSideCacheToo));
}

void BasisCache::determineSideNormals()
{
  if ( isSideCache() )
  {
    int numPoints = _cubPointsSideRefCellFC.dimension(0);
    if (_cellTopo->getDimension() > 1)
    {
      Camellia::FieldContainer<double> normalLengths(_numCells, numPoints);
      
      if (_cellTopo->getTensorialDegree() == 0)
      {
        // recompute sideNormals
        _sideNormals.resize(_numCells, numPoints, _spaceDim);
        Intrepid::CellTools<double>::getPhysicalSideNormals(_sideNormals, getJacobian(), _sideIndex, _cellTopo->getShardsTopology());
        // make unit length
        Intrepid::RealSpaceTools<double>::vectorNorm(normalLengths, _sideNormals, Intrepid::NORM_TWO);
        Intrepid::FunctionSpaceTools::scalarMultiplyDataData<double>(_sideNormals, normalLengths, _sideNormals, true);
      }
      else
      {
        _sideNormalsSpaceTime.resize(_numCells, numPoints, _cellTopo->getDimension());
        CamelliaCellTools::getUnitSideNormals(_sideNormalsSpaceTime, _sideIndex, getJacobian(), _cellTopo);
        
        // next, extract the pure-spatial part of the normal (this might not be unit length)
        _sideNormals.resize(_numCells, numPoints, _spaceDim);
        for (int cellOrdinal=0; cellOrdinal<_numCells; cellOrdinal++)
        {
          for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
          {
            for (int d=0; d<_spaceDim; d++)
            {
              _sideNormals(cellOrdinal,ptOrdinal,d) = _sideNormalsSpaceTime(cellOrdinal,ptOrdinal,d);
            }
          }
        }
      }
    }
    else if (_cellTopo->getDimension()==1)
    {
      _sideNormals.resize(_numCells, numPoints, 1);
      unsigned thisSideOrdinal = _sideIndex;
      unsigned otherSideOrdinal = 1 - thisSideOrdinal;
      for (int cellOrdinal=0; cellOrdinal<_numCells; cellOrdinal++)
      {
        double x_this = _physicalCellNodes(cellOrdinal,thisSideOrdinal,0);
        double x_other = _physicalCellNodes(cellOrdinal,otherSideOrdinal,0);
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          _sideNormals(cellOrdinal,ptOrdinal,0) = (x_this > x_other) ? 1 : -1;
        }
      }
      if (_spaceDim == 0)
      {
        // space-time mesh with point topology as the spatial part.  (Not sure this is important to handle.)
        _sideNormalsSpaceTime = _sideNormals;
        _sideNormals.resize(1,numPoints,0); // empty container: no spatial normals...
      }
    }
  }
  _sideNormalsIsValid = true;
}

void BasisCache::recomputeMeasures()
{
  _weightedMeasureIsValid = true; // will be true on return from this method
  if (_cellTopo->getDimension() == 0)
  {
    // then we define the measure of the domain as 1...
    int numCubPoints = isSideCache() ? _cubPointsSideRefCellFC.dimension(0) : _cubPointsFC.dimension(0);
    _weightedMeasure.resize(_numCells, numCubPoints);
    _weightedMeasure.initialize(1.0);
    return;
  }
  if (_cubWeightsFC.size() > 0)
  {
    // a bit ugly: "_cubPoints" may not be cubature points at all, but just points of interest...  If they're not cubature points, then _cubWeights will be cleared out.  See setRefCellPointsFC, above.
    int numCubPoints = isSideCache() ? _cubPointsSideRefCellFC.dimension(0) : _cubPointsFC.dimension(0);
    // TODO: rename _cubPoints and related methods...
    _weightedMeasure.resize(_numCells, numCubPoints);
    if (! isSideCache())
    {
      fst::computeCellMeasure<double>(_weightedMeasure, getJacobianDet(), _cubWeightsFC);
    }
    else
    {
      if (_cellTopo->getDimension()==1)
      {
        // TODO: determine whether this is the right thing:
        _weightedMeasure.initialize(1.0); // not sure this is the right thing.
      }
      else
      {
        CamelliaCellTools::computeSideMeasure(_weightedMeasure, getJacobian(), _cubWeightsFC, _sideIndex, _cellTopo);
      }
    }
  }
}

BasisCachePtr BasisCache::sideBasisCache(Teuchos::RCP<BasisCache> volumeCache, int sideOrdinal)
{
  int spaceDim = volumeCache->cellTopology()->getDimension();
  int numSides = volumeCache->cellTopology()->getSideCount();

  TEUCHOS_TEST_FOR_EXCEPTION(sideOrdinal >= numSides, std::invalid_argument, "sideOrdinal out of range");

  int maxTestDegree = volumeCache->_maxTestDegree;
  int maxTrialDegreeOnSide = volumeCache->_maxTrialDegree;
  BasisPtr multiBasisIfAny;

  if (spaceDim > 1)
  {
    BasisPtr maxDegreeBasisOnSide = volumeCache->_maxDegreeBasisForSide[sideOrdinal];
    if (maxDegreeBasisOnSide.get() != NULL)
    {
      if (BasisFactory::basisFactory()->isMultiBasis(maxDegreeBasisOnSide))
      {
        multiBasisIfAny = maxDegreeBasisOnSide;
      }
      maxTrialDegreeOnSide = maxDegreeBasisOnSide->getDegree();
    }
  }
  BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideOrdinal, volumeCache, maxTrialDegreeOnSide, maxTestDegree, multiBasisIfAny));
  return sideCache;
}

// ! As the name suggests, this method is not meant for widespread use.  Intended mainly for flux-to-field mappings
BasisCachePtr BasisCache::fakeSideCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const Camellia::FieldContainer<double> &volumeRefPoints,
                                        const Camellia::FieldContainer<double> &sideNormals, const Camellia::FieldContainer<double> &cellSideParities,
                                        Camellia::FieldContainer<double> fakeSideNormalsSpaceTime)
{
  int numSides = volumeCache->cellTopology()->getSideCount();

  TEUCHOS_TEST_FOR_EXCEPTION(fakeSideOrdinal >= numSides, std::invalid_argument, "fakeSideOrdinal out of range");

  BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(fakeSideOrdinal, volumeCache, volumeRefPoints, sideNormals, cellSideParities, fakeSideNormalsSpaceTime));
  return sideCache;
}
