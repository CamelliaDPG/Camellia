
/*
 *  BasisCache.cpp
 *
 */
// @HEADER
//
// Original version copyright © 2011 Sandia Corporation. All Rights Reserved.
// Revisions copyright © 2014 Nathan Roberts. All Rights Reserved.
//
// Redistribution and use in source and binary forms, with or without modification, are
// permitted provided that the following conditions are met:
// 1. Redistributions of source code must retain the above copyright notice, this list of
// conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice, this list of
// conditions and the following disclaimer in the documentation and/or other materials
// provided with the distribution.
// 3. The name of the author may not be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
// BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Nate Roberts (nate@nateroberts.com).
//
// @HEADER

#include "TypeDefs.h"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisEvaluation.h"
#include "CamelliaCellTools.h"
#include "CubatureFactory.h"
#include "Function.h"
#include "Mesh.h"
#include "MeshTransformationFunction.h"
#include "SerialDenseWrapper.h"
#include "SpaceTimeBasisCache.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace std;
using namespace Intrepid;
using namespace Camellia;

typedef FunctionSpaceTools fst;

const static bool CACHE_TRANSFORMED_VALUES = false; // save some memory by not caching these

// TODO: add exceptions for side cache arguments to methods that don't make sense
// (e.g. useCubPointsSideRefCell==true when _isSideCache==false)

int boundDegreeToMaxCubatureForCellTopo(int degree, unsigned cellTopoKey)
{
  // limit cubature degree to max that Intrepid will support
  switch (cellTopoKey)
  {
  case shards::Line<2>::key:
  case shards::Quadrilateral<4>::key:
  case shards::Hexahedron<8>::key:
    return min(INTREPID_CUBATURE_LINE_GAUSS_MAX, degree);
    break;
  case shards::Triangle<3>::key:
    return min(INTREPID_CUBATURE_TRI_DEFAULT_MAX, degree);
    break;
  default:
    return degree; // unhandled cell topo--we'll get an exception if we go beyond the max...
  }
}

// ! Requires that _cellTopo be initialized
void BasisCache::initCubatureDegree(int maxTrialDegree, int maxTestDegree)
{
  _cubDegree = maxTrialDegree + maxTestDegree;

  if (! _isSideCache)
  {
    _cubDegree = boundDegreeToMaxCubatureForCellTopo(_cubDegree, _cellTopo->getShardsTopology().getKey());
  }
  else
  {
    int sideDim = _cellTopo->getDimension() - 1;
    CellTopoPtr side = _cellTopo->getSubcell(sideDim,_sideIndex); // create relevant subcell (side) topology
    _cubDegree = boundDegreeToMaxCubatureForCellTopo(_cubDegree, side->getShardsTopology().getKey());
  }
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
    _cubDegrees[i] = boundDegreeToMaxCubatureForCellTopo(cubDegree, _cellTopo->getShardsTopology().getKey());
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
    Teuchos::RCP<Cubature<double> > cellTopoCub;
    
    if (_cubDegree >= 0)
      cellTopoCub = cubFactory.create(_cellTopo, _cubDegree);
    else if (_cubDegrees.size() > 0)
      cellTopoCub = cubFactory.create(_cellTopo, _cubDegrees);
    
    int cubDim, numCubPoints;
    
    if (cellTopoCub != Teuchos::null)
    {
      cubDim       = cellTopoCub->getDimension();
      numCubPoints = cellTopoCub->getNumPoints();
    }
    else
    {
      cubDim = _cellTopo->getDimension();
      numCubPoints = 0;
    }

    _cubPoints = FieldContainer<double>(numCubPoints, cubDim);
    _cubWeights.resize(numCubPoints);

    if (numCubPoints > 0)
      cellTopoCub->getCubature(_cubPoints, _cubWeights);
  }
  else
  {
    _cubDegree = 1;
    int numCubPointsSide = 1;
    _cubPoints.resize(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
    _cubWeights.resize(numCubPointsSide);

    _cubPoints.initialize(0.0);
    _cubWeights.initialize(1.0);
  }

  _maxPointsPerCubaturePhase = -1;
  _cubaturePhase = 0;
  _cubaturePhaseCount = 1;
  _phasePointOrdinalOffsets.push_back(0);

  // now, create side caches
  if ( createSideCacheToo )
  {
    this->createSideCaches();
  }
}

void BasisCache::initVolumeCache(const Intrepid::FieldContainer<double> &refPoints, const Intrepid::FieldContainer<double> &cubWeights)
{
  // lightweight guy for when the ref points are known in advance
  _sideIndex = -1;
  _spaceDim = _cellTopo->getDimension();
  _isSideCache = false; // VOLUME constructor
  
  _cubPoints = refPoints;
  _cubWeights = cubWeights;
  
  _maxPointsPerCubaturePhase = -1;
  _cubaturePhase = 0;
  _cubaturePhaseCount = 1;
  _phasePointOrdinalOffsets.push_back(0);
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
    BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideOrdinal, thisPtr, maxTrialDegreeOnSide, _maxTestDegree, multiBasisIfAny));
    _basisCacheSides.push_back(sideCache);
  }
}

BasisCache::BasisCache(CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  _cellTopo = cellTopo;

  DofOrdering trialOrdering(cellTopo); // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs

  _isSideCache = false;
  initCubatureDegree(0, cubDegree);
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
  initCubatureDegree(_maxTrialDegree, _maxTestDegree + cubatureDegreeEnrichment);
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
  initCubatureDegree(_maxTrialDegree, _maxTestDegree + cubatureDegreeEnrichment);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
                       DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  _cellTopo = cellTopo;
  findMaximumDegreeBasisForSides(trialOrdering);

  _isSideCache = false;
  initCubatureDegree(trialOrdering.maxBasisDegree(), maxTestDegree);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes,
                       shards::CellTopology &cellTopo,
                       DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo)
{
  _cellTopo = CellTopology::cellTopology(cellTopo);
  findMaximumDegreeBasisForSides(trialOrdering);

  _isSideCache = false;
  initCubatureDegree(trialOrdering.maxBasisDegree(), maxTestDegree);
  bool tensorProductTopologyMeansSpaceTime = true; // doesn't matter for shards topologies (they're not tensor products)
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo,
                       const FieldContainer<double> &refCellPoints, const FieldContainer<double> &cubWeights, int cubatureDegree)
{
  _cellTopo = cellTopo;
  _isSideCache = false;
  initCubatureDegree(cubatureDegree,0);
  
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
  initCubatureDegree(0, cubDegree);
  bool tensorProductTopologyMeansSpaceTime = true; // doesn't matter for shards topologies (they're not tensor products)
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo)
{
  // NOTE that this constructor's a bit dangerous, in that we lack information about the brokenness
  // of the sides; we may under-integrate for cells with broken sides...
  _cellTopo = CellTopology::cellTopology(cellTopo);
  DofOrdering trialOrdering(_cellTopo); // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs

  _isSideCache = false;
  initCubatureDegree(0, cubDegree);
  bool tensorProductTopologyMeansSpaceTime = true; // doesn't matter for shards topologies (they're not tensor products)
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);

  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  // NOTE that this constructor's a bit dangerous, in that we lack information about the brokenness
  // of the sides; we may under-integrate for cells with broken sides...
  _cellTopo = cellTopo;
  DofOrdering trialOrdering(_cellTopo); // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs

  _isSideCache = false;
  initCubatureDegree(0, cubDegree);
  initVolumeCache(createSideCacheToo, tensorProductTopologyMeansSpaceTime);

  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

// "fake" side constructor
BasisCache::BasisCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const FieldContainer<double> &volumeRefPoints,
                       const FieldContainer<double> &sideNormals, const FieldContainer<double> &cellSideParities,
                       FieldContainer<double> sideNormalsSpaceTime)
{
  _cellTopo = volumeCache->cellTopology(); // VOLUME cell topo.
  _isSideCache = true;
  _sideIndex = fakeSideOrdinal;
  _basisCacheVolume = volumeCache;
  _spaceDim = volumeCache->getSpaceDim();

  _cubPoints.resize(0); // force an exception if true side reference points are ever accessed in fake side BasisCache

  int numCells = volumeCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = volumeRefPoints.dimension(0);
  _physCubPoints.resize(numCells,numPoints,_spaceDim);
  _cubPointsSideRefCell = volumeRefPoints;
  _sideNormals = sideNormals;
  _sideNormalsSpaceTime = sideNormalsSpaceTime;
  _cellSideParities = cellSideParities;
  _maxPointsPerCubaturePhase = -1; // default: -1 (infinite)
  _cubaturePhase = 0; // index of the cubature phase; defaults to 0
  _cubaturePhaseCount = 1; // how many phases to get through all the points
  _phasePointOrdinalOffsets.push_back(0);

  // the assumption is that if you're using this constructor, the volume points provided are already in reference space
  // so that the transformations are all identities
  int cellDim = _cellTopo->getDimension(); // for space-time, cellDim = _spaceDim + 1
  _cellJacobian.resize(numCells, numPoints, cellDim, cellDim);
  _cellJacobInv.resize(numCells, numPoints, cellDim, cellDim);
  _cellJacobDet.resize(numCells, numPoints);
  _cellJacobian.initialize(0.0);
  _cellJacobInv.initialize(0.0);
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      for (int d=0; d<cellDim; d++)
      {
        _cellJacobian(cellOrdinal,pointOrdinal,d,d) = 1.0;
        _cellJacobInv(cellOrdinal,pointOrdinal,d,d) = 1.0;
      }
    }
  }
  
  _cellJacobDet.initialize(1.0);
}

// side constructor
BasisCache::BasisCache(int sideIndex, BasisCachePtr volumeCache, int trialDegree, int testDegree, BasisPtr multiBasisIfAny)
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

  initCubatureDegree(trialDegree, testDegree);

  if (sideDim > 0)
  {
    CubatureFactory cubFactory;
    Teuchos::RCP<Cubature<double> > sideCub;
    if (_cubDegree >= 0)
      sideCub = cubFactory.create(side, _cubDegree);
    else if (_cubDegrees.size() > 0)
      sideCub = cubFactory.create(side, _cubDegrees);

    int numCubPointsSide;
    
    if (sideCub != Teuchos::null)
      numCubPointsSide = sideCub->getNumPoints();
    else
      numCubPointsSide = 0;
    
    _cubPoints.resize(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
    _cubWeights.resize(numCubPointsSide);
    if ( multiBasisIfAny.get() == NULL )
    {
      if (numCubPointsSide > 0)
        sideCub->getCubature(_cubPoints, _cubWeights);
    }
    else
    {
      MultiBasis<>* multiBasis = (MultiBasis<>*) multiBasisIfAny.get();

      int cubatureEnrichment = (multiBasis->getDegree() < _maxTrialDegree) ? _maxTrialDegree - multiBasis->getDegree() : 0;
      multiBasis->getCubature(_cubPoints, _cubWeights, _maxTestDegree + cubatureEnrichment);

      numCubPointsSide = _cubPoints.dimension(0);
    }

    _cubPointsSideRefCell.resize(numCubPointsSide, sideDim + 1); // cubPointsSide from the pov of the ref cell
    if (numCubPointsSide > 0)
      CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
  }
  else
  {
    _cubDegree = 1;
    int numCubPointsSide = 1;
    _cubPoints.resize(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
    _cubWeights.resize(numCubPointsSide);

    _cubPoints.initialize(0.0);
    _cubWeights.initialize(1.0);

    _cubPointsSideRefCell.resize(numCubPointsSide, sideDim + 1); // cubPointsSide from the pov of the ref cell
    CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
  }

  _maxPointsPerCubaturePhase = -1; // default: -1 (infinite)
  _cubaturePhase = 0; // index of the cubature phase; defaults to 0
  _cubaturePhaseCount = 1; // how many phases to get through all the points
  _phasePointOrdinalOffsets.push_back(0);
}

// side constructor with specific ref points
BasisCache::BasisCache(int sideIndex, BasisCachePtr volumeCache, const FieldContainer<double> &refPoints,
                       const FieldContainer<double> &cubWeights, int cubatureDegree)
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
  
  initCubatureDegree(cubatureDegree, 0);

  _cubPoints = refPoints;
  _cubWeights = cubWeights;
  
  int numCubPointsSide = _cubPoints.dimension(0);
  
  _cubPointsSideRefCell.resize(numCubPointsSide, sideDim + 1); // cubPointsSide from the pov of the ref cell
  CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
  
  _maxPointsPerCubaturePhase = -1; // default: -1 (infinite)
  _cubaturePhase = 0; // index of the cubature phase; defaults to 0
  _cubaturePhaseCount = 1; // how many phases to get through all the points
  _phasePointOrdinalOffsets.push_back(0);
  
  setPhysicalCellNodes(volumeCache->getPhysicalCellNodes(), volumeCache->cellIDs(), false); // false: don't create side caches...
}

const vector<GlobalIndexType> & BasisCache::cellIDs()
{
  return _cellIDs;
}

CellTopoPtr BasisCache::cellTopology()
{
  return _cellTopo;
}

FieldContainer<double> BasisCache::computeParametricPoints()
{
  if (_cubPoints.size()==0)
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
    BasisCachePtr parametricCache = BasisCache::parametricQuadCache(cubatureDegree, getRefCellPoints(), this->getSideIndex());
    return parametricCache->getPhysicalCubaturePoints();
  }
  else if (_cellTopo->getShardsTopology().getKey()==shards::Line<2>::key)
  {
    int cubatureDegree = 0;  // we throw away the computed cubature points, so let's create as few as possible...
    BasisCachePtr parametricCache = BasisCache::parametric1DCache(cubatureDegree);
    parametricCache->setRefCellPoints(this->getRefCellPoints());
    return parametricCache->getPhysicalCubaturePoints();
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported cellTopo");
    return FieldContainer<double>(0);
  }
}

int BasisCache::cubatureDegree()
{
  return _cubDegree;
}

int BasisCache::getCubaturePhaseCount()
{
  return _cubaturePhaseCount;
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

void BasisCache::setMaxPointsPerCubaturePhase(int maxPoints)
{
  if (_maxPointsPerCubaturePhase == -1)
  {
    _allCubPoints = _cubPoints;
    _allCubWeights = _cubWeights;
  }

  _maxPointsPerCubaturePhase = maxPoints;

  int totalPointCount = _allCubPoints.dimension(0);

  if (_maxPointsPerCubaturePhase != -1)
  {
    _cubaturePhaseCount = (int) ceil((double)totalPointCount / _maxPointsPerCubaturePhase);

    _phasePointOrdinalOffsets = vector<int>(_cubaturePhaseCount+1);
    for (int phaseOrdinal=0; phaseOrdinal<_cubaturePhaseCount; phaseOrdinal++)
    {
      _phasePointOrdinalOffsets[phaseOrdinal] = phaseOrdinal * (totalPointCount / _cubaturePhaseCount);
    }
    _phasePointOrdinalOffsets[_cubaturePhaseCount] = totalPointCount;
    _cubPoints.resize(0); // should trigger error if setCubaturePhase isn't called
    _cubWeights.resize(0);
  }
  else
  {
    _cubaturePhaseCount = 1;
    _phasePointOrdinalOffsets = vector<int>(2);
    _phasePointOrdinalOffsets[0] = 0;
    _phasePointOrdinalOffsets[1] = totalPointCount;

    setRefCellPoints(_allCubPoints, _allCubWeights);
  }
}

void BasisCache::setCubaturePhase(int phaseOrdinal)
{
  int offset = _phasePointOrdinalOffsets[phaseOrdinal];
  int phasePointCount = _phasePointOrdinalOffsets[phaseOrdinal+1] - offset;
  int cubSpaceDim = _allCubPoints.dimension(1);
  FieldContainer<double> cubPoints(phasePointCount, cubSpaceDim);
  FieldContainer<double> cubWeights(phasePointCount);
  for (int ptOrdinal=0; ptOrdinal<phasePointCount; ptOrdinal++)
  {
    cubWeights(ptOrdinal) = _allCubWeights(offset+ptOrdinal);
    for (int d=0; d<cubSpaceDim; d++)
    {
      cubPoints(ptOrdinal,d) = _allCubPoints(offset+ptOrdinal,d);
    }
  }
  setRefCellPoints(cubPoints, cubWeights);
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
  _knownValuesTransformed.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();
  _knownValuesTransformedWeighted.clear();

  // resize all the related fieldcontainers to reclaim their memory
  _cellIDs.clear();
  _cellJacobian.resize(0);
  _cellJacobInv.resize(0);
  _cellJacobDet.resize(0);
  _weightedMeasure.resize(0);
  _physCubPoints.resize(0);
}

FieldContainer<double> & BasisCache::getWeightedMeasures()
{
  return _weightedMeasure;
}

const Intrepid::FieldContainer<double> &BasisCache::getPhysicalCellNodes()
{
  return _physicalCellNodes;
}

const FieldContainer<double> & BasisCache::getPhysicalCubaturePoints()
{
  return _physCubPoints;
}

FieldContainer<double> BasisCache::getCellMeasures()
{
  int numCells = _weightedMeasure.dimension(0);
  int numPoints = _weightedMeasure.dimension(1);
  FieldContainer<double> cellMeasures(numCells);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      cellMeasures(cellIndex) += _weightedMeasure(cellIndex,ptIndex);
    }
  }
  return cellMeasures;
}

const Intrepid::FieldContainer<double> & BasisCache::getCubatureWeights()
{
  return _cubWeights;
}

const FieldContainer<double> & BasisCache::getJacobian()
{
  return _cellJacobian;
}
const FieldContainer<double> & BasisCache::getJacobianDet()
{
  return _cellJacobDet;
}
const FieldContainer<double> & BasisCache::getJacobianInv()
{
  return _cellJacobInv;
}

constFCPtr BasisCache::getValues(BasisPtr basis, Camellia::EOperator op,
                                 bool useCubPointsSideRefCell)
{
  const FieldContainer<double>* cubPoints;
  if (useCubPointsSideRefCell)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(_cubPointsSideRefCell.size()==0,std::invalid_argument,"useCubPointsSideRefCell = true, but _cubPointsSideRefCell is empty!");
    cubPoints = &_cubPointsSideRefCell;
  }
  else
  {
    cubPoints = &_cubPoints;
  }
  // first, let's check whether the exact request is already known
  pair< Camellia::Basis<>*, Camellia::EOperator> key = make_pair(basis.get(), op);

  if (_knownValues.find(key) != _knownValues.end() )
  {
    return _knownValues[key];
  }
  int componentOfInterest = -1;
  // otherwise, lookup to see whether a related value is already known
  Camellia::EFunctionSpace fs = basis->functionSpace();
  Intrepid::EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);

  pair<Camellia::Basis<>*, Camellia::EOperator> relatedKey = key;
  if ((Camellia::EOperator)relatedOp != op)
  {
    relatedKey = make_pair(basis.get(), (Camellia::EOperator) relatedOp);
    if (_knownValues.find(relatedKey) == _knownValues.end() )
    {
      // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDim)
      FCPtr relatedResults = BasisEvaluation::getValues(basis,(Camellia::EOperator)relatedOp,*cubPoints);
      _knownValues[relatedKey] = relatedResults;
    }

    constFCPtr relatedResults = _knownValues[relatedKey];
    //    constFCPtr relatedResults = _knownValues[key];
    constFCPtr result = BasisEvaluation::getComponentOfInterest(relatedResults,op,fs,componentOfInterest);
    if ( result.get() == 0 )
    {
      result = relatedResults;
    }
    _knownValues[key] = result;
    return result;
  }
  // if we get here, we should have a standard Intrepid operator, in which case we should
  // be able to: size a FieldContainer appropriately, and then call basis->getValues

  // But let's do just check that we have a standard Intrepid operator
  if ( (op >= Camellia::OP_X) || (op <  Camellia::OP_VALUE) )
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown operator.");
  }
  FCPtr result = BasisEvaluation::getValues(basis,op,*cubPoints);
  _knownValues[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedValues(BasisPtr basis, Camellia::EOperator op,
    bool useCubPointsSideRefCell)
{
  pair<Camellia::Basis<>*, Camellia::EOperator> key = make_pair(basis.get(), op);
  if (_knownValuesTransformed.find(key) != _knownValuesTransformed.end())
  {
    return _knownValuesTransformed[key];
  }

  int componentOfInterest;
  Camellia::EFunctionSpace fs = basis->functionSpace();
  Intrepid::EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);

  pair<Camellia::Basis<>*, Camellia::EOperator> relatedKey = make_pair(basis.get(),(Camellia::EOperator) relatedOp);
  if (_knownValuesTransformed.find(relatedKey) == _knownValuesTransformed.end())
  {
    constFCPtr transformedValues;
    bool vectorizedBasis = functionSpaceIsVectorized(fs);
    if ( (vectorizedBasis) && (relatedOp ==  Intrepid::OPERATOR_VALUE))
    {
      VectorBasisPtr vectorBasis = Teuchos::rcp( (VectorizedBasis<double, FieldContainer<double> > *) basis.get(), false );
      BasisPtr componentBasis = vectorBasis->getComponentBasis();
      constFCPtr componentReferenceValuesTransformed = getTransformedValues(componentBasis, Camellia::OP_VALUE,
          useCubPointsSideRefCell);
      transformedValues = BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(vectorBasis,
                          Camellia::OP_VALUE,
                          componentReferenceValuesTransformed);
    }
    else
    {
      constFCPtr referenceValues = getValues(basis,(Camellia::EOperator) relatedOp, useCubPointsSideRefCell);
//      cout << "_cellJacobInv:\n" << _cellJacobInv;
//      cout << "referenceValues:\n"  << *referenceValues;
      // TODO: revisit the way we determine numCells....
      int numCells = _physCubPoints.dimension(0);
      if (numCells == 0)   // can happen for certain BasisCaches used in SpaceTimeBasisCache
      {
        if (_physicalCellNodes.rank() > 0)
        {
          numCells = _physicalCellNodes.dimension(0);
        }
      }
      transformedValues =
        BasisEvaluation::getTransformedValuesWithBasisValues(basis, (Camellia::EOperator) relatedOp,
            referenceValues, numCells, _cellJacobian,
            _cellJacobInv,_cellJacobDet);
//      cout << "transformedValues:\n" << *transformedValues;
    }
    _knownValuesTransformed[relatedKey] = transformedValues;
  }
  constFCPtr relatedValuesTransformed = _knownValuesTransformed[relatedKey];
  constFCPtr result;
  if (   (op != Camellia::OP_CROSS_NORMAL)   && (op != Camellia::OP_DOT_NORMAL)
         && (op != Camellia::OP_TIMES_NORMAL)   && (op != Camellia::OP_VECTORIZE_VALUE)
         && (op != Camellia::OP_TIMES_NORMAL_X) && (op != Camellia::OP_TIMES_NORMAL_Y)
         && (op != Camellia::OP_TIMES_NORMAL_Z) && (op != Camellia::OP_TIMES_NORMAL_T)
     )
  {
    result = BasisEvaluation::BasisEvaluation::getComponentOfInterest(relatedValuesTransformed,op,fs,componentOfInterest);
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
      result = BasisEvaluation::getValuesCrossedWithNormals(relatedValuesTransformed,_sideNormals);
      break;
    case OP_DOT_NORMAL:
      result = BasisEvaluation::getValuesDottedWithNormals(relatedValuesTransformed,_sideNormals);
      break;
    case OP_TIMES_NORMAL:
      result = BasisEvaluation::getValuesTimesNormals(relatedValuesTransformed,_sideNormals);
      break;
    case OP_VECTORIZE_VALUE:
      result = BasisEvaluation::getVectorizedValues(relatedValuesTransformed,_spaceDim);
      break;
    case OP_TIMES_NORMAL_X:
    case OP_TIMES_NORMAL_Y:
    case OP_TIMES_NORMAL_Z:
    {
      int normalComponent = op - OP_TIMES_NORMAL_X;
      result = BasisEvaluation::getValuesTimesNormals(relatedValuesTransformed,_sideNormals,normalComponent);
    }
    break;
    case OP_TIMES_NORMAL_T:
    {
      if (_cellTopo->getTensorialDegree() == 0)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "temporal normals are not defined for pure spatial topologies");
      }
      int normalComponent = _cellTopo->getDimension() - 1; // time dimension is the last one
      result = BasisEvaluation::getValuesTimesNormals(relatedValuesTransformed,_sideNormalsSpaceTime,normalComponent);
    }
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled op.");
    }
  }
  if (CACHE_TRANSFORMED_VALUES)
    _knownValuesTransformed[key] = result;
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
  Teuchos::RCP< FieldContainer<double> > weightedValues = Teuchos::rcp( new FieldContainer<double>(dimensions) );
  fst::multiplyMeasure<double>(*weightedValues, _weightedMeasure, *unWeightedValues);
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

const FieldContainer<double> & BasisCache::getPhysicalCubaturePointsForSide(int sideOrdinal)
{
  return _basisCacheSides[sideOrdinal]->getPhysicalCubaturePoints();
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

const FieldContainer<double> & BasisCache::getSideUnitNormals(int sideOrdinal)
{
  return _basisCacheSides[sideOrdinal]->_sideNormals;
}

const FieldContainer<double>& BasisCache::getRefCellPoints()
{
  return _cubPoints;
}

FieldContainer<double> BasisCache::getRefCellPointsForPhysicalPoints(const FieldContainer<double> &physicalPoints, int cellIndex)
{
  int numPoints = physicalPoints.dimension(0);
  int spaceDim = physicalPoints.dimension(1);

  if (_cellTopo->getTensorialDegree() > 0)
  {
    cout << " BasisCache::getRefCellPointsForPhysicalPoints does not support tensorial degree > 1.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache::getRefCellPointsForPhysicalPoints does not support tensorial degree > 1.");
  }

  FieldContainer<double> refCellPoints(numPoints,spaceDim);
  Intrepid::CellTools<double>::mapToReferenceFrame(refCellPoints,physicalPoints,_physicalCellNodes,_cellTopo->getShardsTopology(),cellIndex);
  return refCellPoints;
}

const FieldContainer<double> &BasisCache::getSideRefCellPointsInVolumeCoordinates()
{
  if (! isSideCache())
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "getSideRefCellPointsInVolumeCoordinates() only supported for side caches.");
  }
  return _cubPointsSideRefCell;
}

void BasisCache::setRefCellPoints(const FieldContainer<double> &pointsRefCell)
{
  FieldContainer<double> cubWeights;
  this->setRefCellPoints(pointsRefCell, cubWeights);
}

void BasisCache::setRefCellPoints(const FieldContainer<double> &pointsRefCell, const FieldContainer<double> &cubWeights,
                                  int cubatureDegree, bool recomputePhysicalMeasures)
{
  _cubPoints = pointsRefCell;
  _cubDegree = cubatureDegree;
  int numPoints = pointsRefCell.dimension(0);

  if ( isSideCache() )   // then we need to map pointsRefCell (on side) into volume coordinates, and store in _cubPointsSideRefCell
  {
    int cellDim = _cellTopo->getDimension(); // will be _spaceDim + 1 for space-time CellTopologies.
    // for side cache, cellDim is the spatial dimension of the volume cache's cellTopology
    _cubPointsSideRefCell.resize(numPoints, cellDim);
    // _cellTopo is the volume cell topology for side basis caches.
    int sideDim = cellDim - 1;
    CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
  }

  _knownValues.clear();
  _knownValuesTransformed.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();
  _knownValuesTransformedWeighted.clear();

  _cubWeights = cubWeights;

  // allow reuse of physicalNode info; just map the new points...
  if ((_physCubPoints.size() > 0) && recomputePhysicalMeasures)
  {
    determinePhysicalPoints();
    determineJacobian();

    recomputeMeasures();
  }
}

const FieldContainer<double> & BasisCache::getSideNormals()
{
  return _sideNormals;
}

const FieldContainer<double> & BasisCache::getSideNormalsSpaceTime()
{
  if (_cellTopo->getTensorialDegree() == 0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "space-time side normals are only defined for cell topologies with tensorial degree > 0.");
  }
  return _sideNormalsSpaceTime;
}

void BasisCache::setSideNormals(FieldContainer<double> &sideNormals)
{
  _sideNormals = sideNormals;
}

const FieldContainer<double> & BasisCache::getCellSideParities()
{
  return _cellSideParities;
}

void BasisCache::setCellIDs(const std::vector<GlobalIndexType> &cellIDs)
{
  _cellIDs = cellIDs;
}

void BasisCache::setCellSideParities(const FieldContainer<double> &cellSideParities)
{
  TEUCHOS_TEST_FOR_EXCEPTION((cellSideParities.rank() != 2) || (cellSideParities.dimension(1) < _cellTopo->getSideCount()),
                             std::invalid_argument, "Incorrectly sized cellSideParities");
  _cellSideParities = cellSideParities;
}

void BasisCache::setTransformationFunction(TFunctionPtr<double> fxn, bool composeWithMeshTransformation)
{
  _transformationFxn = fxn;
  _composeTransformationFxnWithMeshTransformation = composeWithMeshTransformation;
  // recompute physical points and jacobian values
  determinePhysicalPoints();
  determineJacobian();
}

void BasisCache::determinePhysicalPoints()
{
  int cellDim = _cellTopo->getDimension();
  if (cellDim==0) return; // physical points not meaningful then...
  int numPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
  if ( TFunction<double>::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation)
  {
    // _spaceDim for side cache refers to the volume cache's spatial dimension
    _physCubPoints.resize(_numCells, numPoints, cellDim);

    if (numPoints > 0)
    {
      if ( ! isSideCache() )
      {
          CamelliaCellTools::mapToPhysicalFrame(_physCubPoints,_cubPoints,_physicalCellNodes,_cellTopo);
      }
      else
      {
        CamelliaCellTools::mapToPhysicalFrame(_physCubPoints,_cubPointsSideRefCell,_physicalCellNodes,_cellTopo);
      }
    }
  }
  else
  {
    // if we get here, then Function is meant to work on reference cell
    // unsupported for now
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Reference-cell based transformations presently unsupported");
    // (what we need to do here is either copy the _cubPoints to _numCells locations in _physCubPoints,
    //  or make the relevant Function simply work on the reference points, returning a FieldContainer
    //  with a numCells dimension.  The latter makes more sense to me--and is more efficient.  The Function should
    //  simply call BasisCache->getRefCellPoints()...  On this idea, we don't even have to do anything special in
    //  BasisCache: the if clause above only serves to save us a little computational effort.)
  }
  if ( ! TFunction<double>::isNull(_transformationFxn) )
  {
    FieldContainer<double> newPhysCubPoints(_numCells,numPoints,cellDim);
    BasisCachePtr thisPtr = Teuchos::rcp(this,false);

    // the usual story: we don't want to use the transformation Function inside the BasisCache
    // while the transformation Function is using the BasisCache to determine its values.
    // So we move _transformationFxn out of the way for a moment:
    TFunctionPtr<double> transformationFxn = _transformationFxn;
    _transformationFxn = TFunction<double>::null();
    {
      // size cell Jacobian containers—the dimensions are used even for HGRAD basis OP_VALUE, which
      // may legitimately be invoked by transformationFxn, below...
      _cellJacobian.resize(_numCells, numPoints, cellDim, cellDim);
      _cellJacobInv.resize(_numCells, numPoints, cellDim, cellDim);
      _cellJacobDet.resize(_numCells, numPoints);
      // (the sizes here agree with what's done in determineJacobian, so the resizing there should be
      //  basically free if we've done it here.)
    }

    transformationFxn->values(newPhysCubPoints, thisPtr);
    _transformationFxn = transformationFxn;

    _physCubPoints = newPhysCubPoints;
  }
}

void BasisCache::determineJacobian()
{
  int cellDim = _cellTopo->getDimension();

  if (cellDim == 0) return;  // Jacobians not meaningful then...

  // Compute cell Jacobians, their inverses and their determinants
  int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);

  // Containers for Jacobian
  _cellJacobian.resize(_numCells, numCubPoints, cellDim, cellDim);
  _cellJacobInv.resize(_numCells, numCubPoints, cellDim, cellDim);
  _cellJacobDet.resize(_numCells, numCubPoints);

  if (numCubPoints == 0) return;

  if ( TFunction<double>::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation)
  {
    if (!isSideCache())
      CamelliaCellTools::setJacobian(_cellJacobian, _cubPoints, _physicalCellNodes, _cellTopo);
    else
      CamelliaCellTools::setJacobian(_cellJacobian, _cubPointsSideRefCell, _physicalCellNodes, _cellTopo);
  }

  SerialDenseWrapper::determinantAndInverse(_cellJacobDet, _cellJacobInv, _cellJacobian);

//  CellTools::setJacobianDet(_cellJacobDet, _cellJacobian );
////  cout << "On rank " << Teuchos::GlobalMPISession::getRank() << ", about to compute jacobian inverse for cellJacobian of size: " << _cellJacobian.size() << endl;
//  CellTools::setJacobianInv(_cellJacobInv, _cellJacobian );

  if (_transformationFxn != Teuchos::null )
  {
    BasisCachePtr thisPtr = Teuchos::rcp(this,false);
    if (_composeTransformationFxnWithMeshTransformation)
    {
      // then we need to multiply one Jacobian by the other
      FieldContainer<double> fxnJacobian(_numCells,numCubPoints,cellDim,cellDim);
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
      FieldContainer<double> cellJacobianToMultiply(_cellJacobian); // tensorMultiplyDataData doesn't support multiplying in place
      fst::tensorMultiplyDataData<double>( _cellJacobian, fxnJacobian, cellJacobianToMultiply );
//      cout << "_cellJacobian after multiplication:\n" << _cellJacobian;
    }
    else
    {
      _transformationFxn->grad()->values( _cellJacobian, thisPtr );
    }

    SerialDenseWrapper::determinantAndInverse(_cellJacobDet, _cellJacobInv, _cellJacobian);

//    CellTools::setJacobianInv(_cellJacobInv, _cellJacobian );
//    CellTools::setJacobianDet(_cellJacobDet, _cellJacobian );
  }
}

void BasisCache::setPhysicalCellNodes(const FieldContainer<double> &physicalCellNodes,
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

  // compute physicalCubaturePoints, the transformed cubature points on each cell:
  determinePhysicalPoints(); // when using _transformationFxn, important to have physical points before Jacobian is computed
//  cout << "physicalCellNodes:\n" << physicalCellNodes;
  determineJacobian();

  // recompute weighted measure at the new physical points
  recomputeMeasures();

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

int BasisCache::maxTestDegree()
{
  return _maxTestDegree;
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

BasisCachePtr BasisCache::parametricQuadCache(int cubatureDegree, const FieldContainer<double> &refCellPoints, int sideCacheIndex)
{
  int numCells = 1;
  int numVertices = 4;
  int spaceDim = 2;
  FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
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
    parametricCache->setRefCellPoints(refCellPoints);
    return parametricCache;
  }
  else
  {
    parametricCache->getSideBasisCache(sideCacheIndex)->setRefCellPoints(refCellPoints);
    return parametricCache->getSideBasisCache(sideCacheIndex);
  }
}

BasisCachePtr BasisCache::parametricQuadCache(int cubatureDegree)
{
  int numCells = 1;
  int numVertices = 4;
  int spaceDim = 2;
  FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
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

BasisCachePtr BasisCache::basisCache1D(double x0, double x1, int cubatureDegree)   // x0 and x1: physical space endpoints
{
  int numCells = 1;
  int numVertices = 2;
  int spaceDim = 1;
  FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
  physicalCellNodes(0,0,0) = x0;
  physicalCellNodes(0,1,0) = x1;
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  return Teuchos::rcp( new BasisCache(physicalCellNodes, line_2, cubatureDegree));
}

BasisCachePtr BasisCache::basisCacheForCell(MeshPtr mesh, GlobalIndexType cellID, bool testVsTest, int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime)
{
  ElementTypePtr elemType = mesh->getElementType(cellID);
  vector<GlobalIndexType> cellIDs(1,cellID);
  if (tensorProductTopologyMeansSpaceTime && (elemType->cellTopoPtr->getTensorialDegree() > 0))
  {
    CellTopoPtr spaceTimeTopo = elemType->cellTopoPtr;
    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesForCell(cellID);
    // check that the physical nodes are in fact in a tensor product structure:
    CellTopoPtr spaceTopo = elemType->cellTopoPtr->getTensorialComponent();
    CellTopoPtr timeTopo = CellTopology::line();
    FieldContainer<double> physicalCellNodesSpace(1, spaceTopo->getNodeCount(), spaceTopo->getDimension());
    FieldContainer<double> physicalCellNodesTime(1, timeTopo->getNodeCount(), timeTopo->getDimension());
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
  basisCache->setPhysicalCellNodes(mesh->physicalCellNodesForCell(cellID), cellIDs, createSideCache);
  basisCache->setCellSideParities(mesh->cellSideParitiesForCell(cellID));

  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForCellTopology(CellTopoPtr cellTopo, int cubatureDegree,
    const FieldContainer<double> &physicalCellNodes,
    bool createSideCacheToo,
    bool tensorProductTopologyMeansSpaceTime)
{
  if (tensorProductTopologyMeansSpaceTime && (cellTopo->getTensorialDegree() > 0))
  {
    int numCells = physicalCellNodes.dimension(0);
    CellTopoPtr spaceTimeTopo = cellTopo;

    // check that the physical nodes are in fact in a tensor product structure:
    CellTopoPtr spaceTopo = spaceTimeTopo->getTensorialComponent();
    FieldContainer<double> physicalCellNodesSpace(numCells, spaceTopo->getNodeCount(), max((int)spaceTopo->getDimension(),1));

    CellTopoPtr timeTopo = CellTopology::line();
    FieldContainer<double> physicalCellNodesTime(numCells, timeTopo->getNodeCount(), timeTopo->getDimension());

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

BasisCachePtr BasisCache::basisCacheForCellType(MeshPtr mesh, ElementTypePtr elemType, bool testVsTest,
    int cubatureDegreeEnrichment, bool tensorProductTopologyMeansSpaceTime)   // for cells on the local MPI node
{
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh, testVsTest, cubatureDegreeEnrichment, tensorProductTopologyMeansSpaceTime) );
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
  FieldContainer<double> cellNodes(cellTopo->getNodeCount(),cellTopo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(cellNodes, cellTopo);
  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));

  BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubatureDegree, cellNodes, createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForRefinedReferenceCell(CellTopoPtr cellTopo, int cubatureDegree,
    RefinementBranch refinementBranch, bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinementBranch);

  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  
  BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubatureDegree, cellNodes, createSideCacheToo, tensorProductTopologyMeansSpaceTime);

  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForRefinedReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree,
    RefinementBranch refinementBranch, bool createSideCacheToo)
{
  FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinementBranch);

  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(cellNodes,cellTopo,cubatureDegree,createSideCacheToo));
  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForRefinedReferenceCell(int cubatureDegree, RefinementBranch refinementBranch,
                                                            bool createSideCacheToo, bool tensorProductTopologyMeansSpaceTime)
{
  CellTopoPtr cellTopo = RefinementPattern::descendantTopology(refinementBranch);
  FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinementBranch);
  
  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  BasisCachePtr basisCache = BasisCache::basisCacheForCellTopology(cellTopo, cubatureDegree, cellNodes, createSideCacheToo, tensorProductTopologyMeansSpaceTime);
  return basisCache;
}

BasisCachePtr BasisCache::quadBasisCache(double width, double height, int cubDegree, bool createSideCacheToo)
{
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  FieldContainer<double> physicalCellNodes(1,4,2);
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

void BasisCache::recomputeMeasures()
{
  if (_cellTopo->getDimension() == 0)
  {
    // then we define the measure of the domain as 1...
    int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
    _weightedMeasure.resize(_numCells, numCubPoints);
    _weightedMeasure.initialize(1.0);
    return;
  }
  if (_cubWeights.size() > 0)
  {
    // a bit ugly: "_cubPoints" may not be cubature points at all, but just points of interest...  If they're not cubature points, then _cubWeights will be cleared out.  See setRefCellPoints, above.
    int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
    // TODO: rename _cubPoints and related methods...
    _weightedMeasure.resize(_numCells, numCubPoints);
    if (! isSideCache())
    {
      fst::computeCellMeasure<double>(_weightedMeasure, _cellJacobDet, _cubWeights);
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
        CamelliaCellTools::computeSideMeasure(_weightedMeasure, _cellJacobian, _cubWeights, _sideIndex, _cellTopo);
      } /*else if (_cellTopo->getDimension()==2) {
        if (_cellTopo->getTensorialDegree() == 0) {
          // compute weighted edge measure
          FunctionSpaceTools::computeEdgeMeasure<double>(_weightedMeasure,
                                                         _cellJacobian,
                                                         _cubWeights,
                                                         _sideIndex,
                                                         _cellTopo->getShardsTopology());
        } else {
          cout << "ERROR: BasisCache::recomputeMeasures() does not yet support tensorial degree > 0.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "recomputeMeasures() does not yet support tensorial degree > 0.");
        }
      } else if (_cellTopo->getDimension()==3) {
        if (_cellTopo->getTensorialDegree() == 0) {
          FunctionSpaceTools::computeFaceMeasure<double>(_weightedMeasure, _cellJacobian, _cubWeights, _sideIndex, _cellTopo->getShardsTopology());
        } else {
          cout << "ERROR: BasisCache::recomputeMeasures() does not yet support tensorial degree > 0.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "recomputeMeasures() does not yet support tensorial degree > 0.");
        }
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled space dimension.");
      }*/
    }

  }

  if ( isSideCache() )
  {
    int numPoints = _cubPointsSideRefCell.dimension(0);
    if (_cellTopo->getDimension() > 1)
    {
      FieldContainer<double> normalLengths(_numCells, numPoints);

      if (_cellTopo->getTensorialDegree() == 0)
      {
        // recompute sideNormals
        _sideNormals.resize(_numCells, numPoints, _spaceDim);
        Intrepid::CellTools<double>::getPhysicalSideNormals(_sideNormals, _cellJacobian, _sideIndex, _cellTopo->getShardsTopology());
        // make unit length
        RealSpaceTools<double>::vectorNorm(normalLengths, _sideNormals, NORM_TWO);
        FunctionSpaceTools::scalarMultiplyDataData<double>(_sideNormals, normalLengths, _sideNormals, true);
      }
      else
      {
        _sideNormalsSpaceTime.resize(_numCells, numPoints, _cellTopo->getDimension());
        CamelliaCellTools::getUnitSideNormals(_sideNormalsSpaceTime, _sideIndex, _cellJacobian, _cellTopo);

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
BasisCachePtr BasisCache::fakeSideCache(int fakeSideOrdinal, BasisCachePtr volumeCache, const FieldContainer<double> &volumeRefPoints,
                                        const FieldContainer<double> &sideNormals, const FieldContainer<double> &cellSideParities,
                                        FieldContainer<double> fakeSideNormalsSpaceTime)
{
  int numSides = volumeCache->cellTopology()->getSideCount();

  TEUCHOS_TEST_FOR_EXCEPTION(fakeSideOrdinal >= numSides, std::invalid_argument, "fakeSideOrdinal out of range");

  BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(fakeSideOrdinal, volumeCache, volumeRefPoints, sideNormals, cellSideParities, fakeSideNormalsSpaceTime));
  return sideCache;
}
