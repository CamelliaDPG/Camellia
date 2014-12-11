
/*
 *  BasisCache.cpp
 *
 */
// @HEADER
//
// Original version copyright © 2011 Sandia Corporation. All Rights Reserved.
// Revisions copyright © 2013 Nathan Roberts. All Rights Reserved.
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

#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisEvaluation.h"
#include "Mesh.h"
#include "Function.h"
#include "MeshTransformationFunction.h"
#include "CamelliaCellTools.h"

#include "CubatureFactory.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace std;
using namespace Camellia;

typedef FunctionSpaceTools fst;
typedef Teuchos::RCP< FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;

// TODO: add exceptions for side cache arguments to methods that don't make sense 
// (e.g. useCubPointsSideRefCell==true when _isSideCache==false)

int boundDegreeToMaxCubatureForCellTopo(int degree, unsigned cellTopoKey) {
  // limit cubature degree to max that Intrepid will support
  switch (cellTopoKey) {
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
void BasisCache::initCubatureDegree(int maxTrialDegree, int maxTestDegree) {
  _cubDegree = maxTrialDegree + maxTestDegree;
  
  if (! _isSideCache) {
    _cubDegree = boundDegreeToMaxCubatureForCellTopo(_cubDegree, _cellTopo->getShardsTopology().getKey());
  } else {
    int sideDim = _spaceDim - 1;
    CellTopoPtr side = _cellTopo->getSubcell(sideDim,_sideIndex); // create relevant subcell (side) topology
    _cubDegree = boundDegreeToMaxCubatureForCellTopo(_cubDegree, side->getShardsTopology().getKey());
  }
  _maxTestDegree = maxTestDegree;
  _maxTrialDegree = maxTrialDegree;
}

// ! Requires that _cellTopo be initialized
void BasisCache::initCubatureDegree(std::vector<int> &maxTrialDegrees, std::vector<int> &maxTestDegrees) {
  TEUCHOS_TEST_FOR_EXCEPTION(maxTrialDegrees.size() != maxTestDegrees.size(), std::invalid_argument, "maxTrialDegrees must have same length as maxTestDegrees");
  _maxTestDegree = 0;
  _maxTrialDegree = 0;
  _cubDegree = -1;
  _cubDegrees.resize(maxTrialDegrees.size());
  for (int i=0; i<maxTrialDegrees.size(); i++) {
    _maxTrialDegree = max(_maxTrialDegree, maxTrialDegrees[i]);
    _maxTestDegree = max(_maxTestDegree, maxTestDegrees[i]);
    
    int cubDegree = maxTrialDegrees[i] + maxTestDegrees[i];
    _cubDegrees[i] = boundDegreeToMaxCubatureForCellTopo(cubDegree, _cellTopo->getShardsTopology().getKey());
  }
}

// ! requires that initCubature() has been called
void BasisCache::init(bool createSideCacheToo) {
  _sideIndex = -1;
  _spaceDim = _cellTopo->getDimension();
  _isSideCache = false; // VOLUME constructor
  
  if (_spaceDim > 0) {
    CubatureFactory cubFactory;
    Teuchos::RCP<Cubature<double> > cellTopoCub;
    if (_cubDegree >= 0)
      cellTopoCub = cubFactory.create(_cellTopo, _cubDegree);
    else
      cellTopoCub = cubFactory.create(_cellTopo, _cubDegrees);
    
    int cubDim       = cellTopoCub->getDimension();
    int numCubPoints = cellTopoCub->getNumPoints();
    
    _cubPoints = FieldContainer<double>(numCubPoints, cubDim);
    _cubWeights.resize(numCubPoints);
    
    cellTopoCub->getCubature(_cubPoints, _cubWeights);
  } else {
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
  if ( createSideCacheToo ) {
    createSideCaches();
  }
}

void BasisCache::createSideCaches() {
  _basisCacheSides.clear();
  _numSides = _cellTopo->getSideCount();

  for (int sideOrdinal=0; sideOrdinal<_numSides; sideOrdinal++) {
    BasisPtr maxDegreeBasisOnSide = _maxDegreeBasisForSide[sideOrdinal];
    BasisPtr multiBasisIfAny;
    
    int maxTrialDegreeOnSide = _maxTrialDegree;
    if (maxDegreeBasisOnSide.get() != NULL) {
      if (BasisFactory::basisFactory()->isMultiBasis(maxDegreeBasisOnSide)) {
        multiBasisIfAny = maxDegreeBasisOnSide;
      }
      maxTrialDegreeOnSide = maxDegreeBasisOnSide->getDegree();
    }
    
    BasisCachePtr thisPtr = Teuchos::rcp( this, false ); // presumption is that side cache doesn't outlive volume...
    BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideOrdinal, thisPtr, maxTrialDegreeOnSide, _maxTestDegree, multiBasisIfAny));
    _basisCacheSides.push_back(sideCache);
  }
}

BasisCache::BasisCache(CellTopoPtr cellTopo, int cubDegree, bool createSideCacheToo) {
  _spaceDim = cellTopo->getDimension();
  _numSides = cellTopo->getSideCount();
  DofOrdering trialOrdering; // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs
  
  _isSideCache = false;
  _cellTopo = cellTopo;
  initCubatureDegree(0, cubDegree);
  init(createSideCacheToo);
}

BasisCache::BasisCache(ElementTypePtr elemType, Teuchos::RCP<Mesh> mesh, bool testVsTest, int cubatureDegreeEnrichment) {
  // use testVsTest=true for test space inner product (won't create side caches, and will use higher cubDegree)
  shards::CellTopology cellTopo = *(elemType->cellTopoPtr);
  _spaceDim = cellTopo.getDimension();
  _numSides = CamelliaCellTools::getSideCount(cellTopo);
  
  _maxTestDegree = elemType->testOrderPtr->maxBasisDegree();
  
  _mesh = mesh;
  if (_mesh.get()) {
    _transformationFxn = _mesh->getTransformationFunction();
    if (_transformationFxn.get()) {
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
  _cellTopo = CellTopology::cellTopology(cellTopo);
  initCubatureDegree(_maxTrialDegree, _maxTestDegree + cubatureDegreeEnrichment);
  init(createSideCacheToo);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, 
                       shards::CellTopology &cellTopo,
                       DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo) {
  _spaceDim = cellTopo.getDimension();
  _numSides = CamelliaCellTools::getSideCount(cellTopo);
  findMaximumDegreeBasisForSides(trialOrdering);
  
  _isSideCache = false;
  _cellTopo = CellTopology::cellTopology(cellTopo);
  initCubatureDegree(trialOrdering.maxBasisDegree(), maxTestDegree);
  init(createSideCacheToo);
  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

BasisCache::BasisCache(shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo) {
  // NOTE that this constructor's a bit dangerous, in that we lack information about the brokenness
  // of the sides; we may under-integrate for cells with broken sides...
  _spaceDim = cellTopo.getDimension();
  _numSides = CamelliaCellTools::getSideCount(cellTopo);
  DofOrdering trialOrdering; // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs
  
  _isSideCache = false;
  _cellTopo = CellTopology::cellTopology(cellTopo);
  initCubatureDegree(0, cubDegree);
  init(createSideCacheToo);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree, bool createSideCacheToo) {
  // NOTE that this constructor's a bit dangerous, in that we lack information about the brokenness
  // of the sides; we may under-integrate for cells with broken sides...
  _spaceDim = cellTopo.getDimension();
  _numSides = CamelliaCellTools::getSideCount(cellTopo);
  DofOrdering trialOrdering; // dummy trialOrdering
  findMaximumDegreeBasisForSides(trialOrdering); // should fill with NULL ptrs
  
  _isSideCache = false;
  _cellTopo = CellTopology::cellTopology(cellTopo);
  initCubatureDegree(0, cubDegree);
  init(createSideCacheToo);

  setPhysicalCellNodes(physicalCellNodes,vector<GlobalIndexType>(),createSideCacheToo);
}

// side constructor
BasisCache::BasisCache(int sideIndex, BasisCachePtr volumeCache, int trialDegree, int testDegree, BasisPtr multiBasisIfAny) {
  _isSideCache = true;
  _sideIndex = sideIndex;
  _numSides = -1;
  _basisCacheVolume = volumeCache;
  _maxTestDegree = testDegree;
  _maxTrialDegree = trialDegree;
  if (volumeCache->mesh().get()) {
    _transformationFxn = volumeCache->mesh()->getTransformationFunction();
    // at least for now, what the Mesh's transformation function does is transform from a straight-lined mesh to
    // one with potentially curved edges...
    _composeTransformationFxnWithMeshTransformation = true;
  }
  _cellTopo = volumeCache->cellTopology(); // VOLUME cell topo.
  _spaceDim = _cellTopo->getDimension();
  int sideDim = _spaceDim - 1;
  CellTopoPtr side = _cellTopo->getSubcell(sideDim,sideIndex); // create relevant subcell (side) topology
  
  initCubatureDegree(trialDegree, testDegree);
  
  if (sideDim > 0) {
    CubatureFactory cubFactory;
    Teuchos::RCP<Cubature<double> > sideCub;
    if (_cubDegree >= 0)
      sideCub = cubFactory.create(side, _cubDegree);
    else
      sideCub = cubFactory.create(side, _cubDegrees);
    
    int numCubPointsSide = sideCub->getNumPoints();
    _cubPoints.resize(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
    _cubWeights.resize(numCubPointsSide);
    if ( multiBasisIfAny.get() == NULL ) {
      sideCub->getCubature(_cubPoints, _cubWeights);
    } else {
      MultiBasis<>* multiBasis = (MultiBasis<>*) multiBasisIfAny.get();
      
      int cubatureEnrichment = (multiBasis->getDegree() < _maxTrialDegree) ? _maxTrialDegree - multiBasis->getDegree() : 0;
      multiBasis->getCubature(_cubPoints, _cubWeights, _maxTestDegree + cubatureEnrichment);
      
      numCubPointsSide = _cubPoints.dimension(0);
    }
    
    _cubPointsSideRefCell.resize(numCubPointsSide, _spaceDim); // cubPointsSide from the pov of the ref cell
    if (_cellTopo->getTensorialDegree() == 0) {
      CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo->getShardsTopology());
    } else {
      cout << "Reference subcell mapping doesn't yet support tensorial degree > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Reference subcell mapping doesn't yet support tensorial degree > 0.");
    }
  } else {
    _cubDegree = 1;
    int numCubPointsSide = 1;
    _cubPoints.resize(numCubPointsSide, 1); // cubature points from the pov of the side (i.e. a (d-1)-dimensional set)
    _cubWeights.resize(numCubPointsSide);

    _cubPoints.initialize(0.0);
    _cubWeights.initialize(1.0);
    
    _cubPointsSideRefCell.resize(numCubPointsSide, _spaceDim); // cubPointsSide from the pov of the ref cell
    if (_cellTopo->getTensorialDegree() == 0) {
      CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo->getShardsTopology());
    } else {
      cout << "Reference subcell mapping doesn't yet support tensorial degree > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Reference subcell mapping doesn't yet support tensorial degree > 0.");
    }
  }
  
  _maxPointsPerCubaturePhase = -1; // default: -1 (infinite)
  _cubaturePhase = 0; // index of the cubature phase; defaults to 0
  _cubaturePhaseCount = 1; // how many phases to get through all the points
  _phasePointOrdinalOffsets.push_back(0);
}

const vector<GlobalIndexType> & BasisCache::cellIDs() {
  return _cellIDs;
}

CellTopoPtr BasisCache::cellTopology() {
  return _cellTopo;
}

FieldContainer<double> BasisCache::computeParametricPoints() {
  if (_cubPoints.size()==0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "computeParametricPoints() requires reference cell points to be defined.");
  }
  if (_cellTopo->getTensorialDegree() > 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "computeParametricPoints() requires tensorial degree of cell topo to be 0");
  }
  if (_cellTopo->getShardsTopology().getKey()==shards::Quadrilateral<4>::key) {
    int cubatureDegree = 0;
    BasisCachePtr parametricCache = BasisCache::parametricQuadCache(cubatureDegree, getRefCellPoints(), this->getSideIndex());
    return parametricCache->getPhysicalCubaturePoints();
  } else if (_cellTopo->getShardsTopology().getKey()==shards::Line<2>::key) {
    int cubatureDegree = 0;  // we throw away the computed cubature points, so let's create as few as possible...
    BasisCachePtr parametricCache = BasisCache::parametric1DCache(cubatureDegree);
    parametricCache->setRefCellPoints(this->getRefCellPoints());
    return parametricCache->getPhysicalCubaturePoints();
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported cellTopo");
    return FieldContainer<double>(0);
  }
}

int BasisCache::cubatureDegree() {
  return _cubDegree;
}

int BasisCache::getCubaturePhaseCount() {
  return _cubaturePhaseCount;
}

void BasisCache::setMaxPointsPerCubaturePhase(int maxPoints) {
  if (_maxPointsPerCubaturePhase == -1) {
    _allCubPoints = _cubPoints;
    _allCubWeights = _cubWeights;
  }
  
  _maxPointsPerCubaturePhase = maxPoints;

  int totalPointCount = _allCubPoints.dimension(0);

  if (_maxPointsPerCubaturePhase != -1) {
    _cubaturePhaseCount = (int) ceil((double)totalPointCount / _maxPointsPerCubaturePhase);
    
    _phasePointOrdinalOffsets = vector<int>(_cubaturePhaseCount+1);
    for (int phaseOrdinal=0; phaseOrdinal<_cubaturePhaseCount; phaseOrdinal++) {
      _phasePointOrdinalOffsets[phaseOrdinal] = phaseOrdinal * (totalPointCount / _cubaturePhaseCount);
    }
    _phasePointOrdinalOffsets[_cubaturePhaseCount] = totalPointCount;
    _cubPoints.resize(0); // should trigger error if setCubaturePhase isn't called
    _cubWeights.resize(0);
  } else {
    _cubaturePhaseCount = 1;
    _phasePointOrdinalOffsets = vector<int>(2);
    _phasePointOrdinalOffsets[0] = 0;
    _phasePointOrdinalOffsets[1] = totalPointCount;
    
    setRefCellPoints(_allCubPoints, _allCubWeights);
  }
}

void BasisCache::setCubaturePhase(int phaseOrdinal) {
  int offset = _phasePointOrdinalOffsets[phaseOrdinal];
  int phasePointCount = _phasePointOrdinalOffsets[phaseOrdinal+1] - offset;
  int cubSpaceDim = _allCubPoints.dimension(1);
  FieldContainer<double> cubPoints(phasePointCount, cubSpaceDim);
  FieldContainer<double> cubWeights(phasePointCount);
  for (int ptOrdinal=0; ptOrdinal<phasePointCount; ptOrdinal++) {
    cubWeights(ptOrdinal) = _allCubWeights(offset+ptOrdinal);
    for (int d=0; d<cubSpaceDim; d++) {
      cubPoints(ptOrdinal,d) = _allCubPoints(offset+ptOrdinal,d);
    }
  }
  setRefCellPoints(cubPoints, cubWeights);
}

void BasisCache::findMaximumDegreeBasisForSides(DofOrdering &trialOrdering) {
  _maxDegreeBasisForSide.clear();
  vector<int> sideTrialIDs;
  set<int> trialIDs = trialOrdering.getVarIDs();
  for (set<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
    int trialID = *trialIt;
    if (trialOrdering.getNumSidesForVarID(trialID) > 1) {
      sideTrialIDs.push_back(trialID);
    }
  }
  int numSideTrialIDs = sideTrialIDs.size();
  for (int sideOrdinal=0; sideOrdinal<_numSides; sideOrdinal++) {
    BasisPtr maxDegreeBasisOnSide;
    // loop through looking for highest-degree basis
    int maxTrialDegree = -1;
    for (int i=0; i<numSideTrialIDs; i++) {
      if (trialOrdering.getBasis(sideTrialIDs[i],sideOrdinal)->getDegree() > maxTrialDegree) {
        maxDegreeBasisOnSide = trialOrdering.getBasis(sideTrialIDs[i],sideOrdinal);
        maxTrialDegree = maxDegreeBasisOnSide->getDegree();
      }
    }
    _maxDegreeBasisForSide.push_back(maxDegreeBasisOnSide);
  }
}

Teuchos::RCP<Mesh> BasisCache::mesh() {
  if ( ! _isSideCache ) {
    return _mesh;
  } else {
    return _basisCacheVolume->mesh();
  }
}

void BasisCache::setMesh(MeshPtr mesh) {
  if ( ! _isSideCache ) {
    _mesh = mesh;
  } else {
    _basisCacheVolume->setMesh(mesh);
  }
}

void BasisCache::discardPhysicalNodeInfo() {
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

FieldContainer<double> & BasisCache::getWeightedMeasures() {
  return _weightedMeasure;
}

const Intrepid::FieldContainer<double> &BasisCache::getPhysicalCellNodes() {
  return _physicalCellNodes;
}

const FieldContainer<double> & BasisCache::getPhysicalCubaturePoints() {
  return _physCubPoints;
}

FieldContainer<double> BasisCache::getCellMeasures() {
  int numCells = _weightedMeasure.dimension(0);
  int numPoints = _weightedMeasure.dimension(1);
  FieldContainer<double> cellMeasures(numCells);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      cellMeasures(cellIndex) += _weightedMeasure(cellIndex,ptIndex);
    }
  }
  return cellMeasures;
}

const Intrepid::FieldContainer<double> & BasisCache::getCubatureWeights() {
  return _cubWeights;
}

const FieldContainer<double> & BasisCache::getJacobian() {
  return _cellJacobian;
}
const FieldContainer<double> & BasisCache::getJacobianDet() {
  return _cellJacobDet;
}
const FieldContainer<double> & BasisCache::getJacobianInv() {
  return _cellJacobInv;
}

constFCPtr BasisCache::getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                 bool useCubPointsSideRefCell) {
  FieldContainer<double> cubPoints;
  if (useCubPointsSideRefCell) {
    cubPoints = _cubPointsSideRefCell; // unnecessary copy
  } else {
    cubPoints = _cubPoints;
  }
  // first, let's check whether the exact request is already known
  pair< Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended> key = make_pair(basis.get(), op);
  
  if (_knownValues.find(key) != _knownValues.end() ) {
    return _knownValues[key];
  }
  int componentOfInterest = -1;
  // otherwise, lookup to see whether a related value is already known
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);
  
  pair<Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended> relatedKey = key;
  if ((EOperatorExtended)relatedOp != op) {
    relatedKey = make_pair(basis.get(), (IntrepidExtendedTypes::EOperatorExtended) relatedOp);
    if (_knownValues.find(relatedKey) == _knownValues.end() ) {
      // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDim)
      FCPtr relatedResults = BasisEvaluation::getValues(basis,(EOperatorExtended)relatedOp,cubPoints);
      _knownValues[relatedKey] = relatedResults;
    }
    
    constFCPtr relatedResults = _knownValues[relatedKey];
    //    constFCPtr relatedResults = _knownValues[key];
    constFCPtr result = BasisEvaluation::getComponentOfInterest(relatedResults,op,fs,componentOfInterest);
    if ( result.get() == 0 ) {
      result = relatedResults;
    }
    _knownValues[key] = result;
    return result;
  }
  // if we get here, we should have a standard Intrepid operator, in which case we should
  // be able to: size a FieldContainer appropriately, and then call basis->getValues
  
  // But let's do just check that we have a standard Intrepid operator
  if ( (op >= IntrepidExtendedTypes::OP_X) || (op <  IntrepidExtendedTypes::OP_VALUE) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown operator.");
  }
  FCPtr result = BasisEvaluation::getValues(basis,op,cubPoints);
  _knownValues[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                            bool useCubPointsSideRefCell) {
  pair<Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended> key = make_pair(basis.get(), op);
  if (_knownValuesTransformed.find(key) != _knownValuesTransformed.end()) {
    return _knownValuesTransformed[key];
  }
  
  int componentOfInterest;
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = basis->functionSpace();
  Intrepid::EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);
  
  pair<Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended> relatedKey = make_pair(basis.get(),(EOperatorExtended) relatedOp);
  if (_knownValuesTransformed.find(relatedKey) == _knownValuesTransformed.end()) {
    constFCPtr transformedValues;
    bool vectorizedBasis = functionSpaceIsVectorized(fs);
    if ( (vectorizedBasis) && (relatedOp ==  Intrepid::OPERATOR_VALUE)) {
      VectorBasisPtr vectorBasis = Teuchos::rcp( (VectorizedBasis<double, FieldContainer<double> > *) basis.get(), false );
      BasisPtr componentBasis = vectorBasis->getComponentBasis();
      constFCPtr componentReferenceValuesTransformed = getTransformedValues(componentBasis, IntrepidExtendedTypes::OP_VALUE,
                                                                            useCubPointsSideRefCell);
      transformedValues = BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(vectorBasis,
                                                                                              IntrepidExtendedTypes::OP_VALUE,
                                                                                              componentReferenceValuesTransformed);
    } else {
      constFCPtr referenceValues = getValues(basis,(EOperatorExtended) relatedOp, useCubPointsSideRefCell);
//      cout << "_cellJacobInv:\n" << _cellJacobInv;
//      cout << "referenceValues:\n"  << *referenceValues;
      transformedValues =
      BasisEvaluation::getTransformedValuesWithBasisValues(basis, (EOperatorExtended) relatedOp,
                                                           referenceValues, _cellJacobian, 
                                                           _cellJacobInv,_cellJacobDet);
//      cout << "transformedValues:\n" << *transformedValues;
    }
    _knownValuesTransformed[relatedKey] = transformedValues;
  }
  constFCPtr relatedValuesTransformed = _knownValuesTransformed[relatedKey];
  constFCPtr result;
  if (   (op != IntrepidExtendedTypes::OP_CROSS_NORMAL)   && (op != IntrepidExtendedTypes::OP_DOT_NORMAL)
      && (op != IntrepidExtendedTypes::OP_TIMES_NORMAL)   && (op != IntrepidExtendedTypes::OP_VECTORIZE_VALUE) 
      && (op != IntrepidExtendedTypes::OP_TIMES_NORMAL_X) && (op != IntrepidExtendedTypes::OP_TIMES_NORMAL_Y)
      && (op != IntrepidExtendedTypes::OP_TIMES_NORMAL_Z)
     ) {
    result = BasisEvaluation::BasisEvaluation::getComponentOfInterest(relatedValuesTransformed,op,fs,componentOfInterest);
    if ( result.get() == 0 ) {
      result = relatedValuesTransformed;
    }
  } else {
    switch (op) {
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
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled op.");
    }
  }
  _knownValuesTransformed[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedWeightedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, 
                                                    bool useCubPointsSideRefCell) {
  pair<Camellia::Basis<>*, IntrepidExtendedTypes::EOperatorExtended> key = make_pair(basis.get(), op);
  if (_knownValuesTransformedWeighted.find(key) != _knownValuesTransformedWeighted.end()) {
    return _knownValuesTransformedWeighted[key];
  }
  constFCPtr unWeightedValues = getTransformedValues(basis,op, useCubPointsSideRefCell);
  Teuchos::Array<int> dimensions;
  unWeightedValues->dimensions(dimensions);
  Teuchos::RCP< FieldContainer<double> > weightedValues = Teuchos::rcp( new FieldContainer<double>(dimensions) );
  fst::multiplyMeasure<double>(*weightedValues, _weightedMeasure, *unWeightedValues);
  _knownValuesTransformedWeighted[key] = weightedValues;
  return weightedValues;
}

/*** SIDE VARIANTS ***/
constFCPtr BasisCache::getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal,
                                 bool useCubPointsSideRefCell) {
  return _basisCacheSides[sideOrdinal]->getValues(basis,op,useCubPointsSideRefCell);
}

constFCPtr BasisCache::getTransformedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, int sideOrdinal, 
                                            bool useCubPointsSideRefCell) {
  constFCPtr transformedValues;
  if ( ! _isSideCache ) {
    transformedValues = _basisCacheSides[sideOrdinal]->getTransformedValues(basis,op,useCubPointsSideRefCell);
  } else {
    transformedValues = getTransformedValues(basis,op,useCubPointsSideRefCell);
  }
  return transformedValues;
}

constFCPtr BasisCache::getTransformedWeightedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, 
                                                    int sideOrdinal, bool useCubPointsSideRefCell) {
  return _basisCacheSides[sideOrdinal]->getTransformedWeightedValues(basis,op,useCubPointsSideRefCell);
}

const FieldContainer<double> & BasisCache::getPhysicalCubaturePointsForSide(int sideOrdinal) {
  return _basisCacheSides[sideOrdinal]->getPhysicalCubaturePoints();
}

BasisCachePtr BasisCache::getSideBasisCache(int sideOrdinal) {
  if (sideOrdinal < _basisCacheSides.size() )
    return _basisCacheSides[sideOrdinal];
  else
    return Teuchos::rcp((BasisCache *) NULL);
}

BasisCachePtr BasisCache::getVolumeBasisCache() {
  return _basisCacheVolume;
}

bool BasisCache::isSideCache() {
  return _sideIndex >= 0;
}

int BasisCache::getSideIndex() {
  return _sideIndex;
}

const FieldContainer<double> & BasisCache::getSideUnitNormals(int sideOrdinal){  
  return _basisCacheSides[sideOrdinal]->_sideNormals;
}

const FieldContainer<double>& BasisCache::getRefCellPoints() {
  return _cubPoints;
}

FieldContainer<double> BasisCache::getRefCellPointsForPhysicalPoints(const FieldContainer<double> &physicalPoints, int cellIndex) {
  int numPoints = physicalPoints.dimension(0);
  int spaceDim = physicalPoints.dimension(1);
  
  if (_cellTopo->getTensorialDegree() > 0) {
    cout << " BasisCache::getRefCellPointsForPhysicalPoints does not support tensorial degree > 1.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache::getRefCellPointsForPhysicalPoints does not support tensorial degree > 1.");
  }
  
  FieldContainer<double> refCellPoints(numPoints,spaceDim);
  CellTools<double>::mapToReferenceFrame(refCellPoints,physicalPoints,_physicalCellNodes,_cellTopo->getShardsTopology(),cellIndex);
  return refCellPoints;
}

const FieldContainer<double> &BasisCache::getSideRefCellPointsInVolumeCoordinates() {
  if (! isSideCache()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                               "getSideRefCellPointsInVolumeCoordinates() only supported for side caches.");
  }
  return _cubPointsSideRefCell;
}

void BasisCache::setRefCellPoints(const FieldContainer<double> &pointsRefCell) {
  FieldContainer<double> cubWeights;
  this->setRefCellPoints(pointsRefCell, cubWeights);
}

void BasisCache::setRefCellPoints(const FieldContainer<double> &pointsRefCell, const FieldContainer<double> &cubWeights) {
  _cubPoints = pointsRefCell;
  int numPoints = pointsRefCell.dimension(0);
  
  if ( isSideCache() ) { // then we need to map pointsRefCell (on side) into volume coordinates, and store in _cubPointsSideRefCell
    // for side cache, _spaceDim is the spatial dimension of the volume cache
    _cubPointsSideRefCell.resize(numPoints, _spaceDim); 
    // _cellTopo is the volume cell topology for side basis caches.
    int sideDim = _spaceDim - 1;
    CamelliaCellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
  }
  
  _knownValues.clear();
  _knownValuesTransformed.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();
  _knownValuesTransformedWeighted.clear();
  
  _cubWeights = cubWeights;
  
  // allow reuse of physicalNode info; just map the new points...
  if (_physCubPoints.size() > 0) {
    determinePhysicalPoints();
    determineJacobian();
    
    recomputeMeasures();
    
    if (isSideCache()) {
      if (_spaceDim > 1) {
        // recompute sideNormals
        _sideNormals.resize(_numCells, numPoints, _spaceDim);
        FieldContainer<double> normalLengths(_numCells, numPoints);
        
        if (_cellTopo->getTensorialDegree() == 0) {
          CellTools<double>::getPhysicalSideNormals(_sideNormals, _cellJacobian, _sideIndex, _cellTopo->getShardsTopology());
        } else {
          cout << "ERROR: BasisCache::setRefCellPoints does not yet support tensorial degree > 0.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache::setRefCellPoints does not yet support tensorial degree > 0.");
        }
        
        // make unit length
        RealSpaceTools<double>::vectorNorm(normalLengths, _sideNormals, NORM_TWO);
        FunctionSpaceTools::scalarMultiplyDataData<double>(_sideNormals, normalLengths, _sideNormals, true);
      } else if (_spaceDim==1) {
        _sideNormals.resize(_numCells, numPoints, _spaceDim); // here, numPoints should be 1
        unsigned thisSideOrdinal = _sideIndex;
        unsigned otherSideOrdinal = 1 - thisSideOrdinal;
        for (int cellOrdinal=0; cellOrdinal<_numCells; cellOrdinal++) {
          double x_this = _physicalCellNodes(cellOrdinal,thisSideOrdinal,0);
          double x_other = _physicalCellNodes(cellOrdinal,otherSideOrdinal,0);
          if (x_this > x_other) {
            _sideNormals(cellOrdinal,0,0) = 1;
          } else {
            _sideNormals(cellOrdinal,0,0) = -1;
          }
        }
      }
    }
  }
}

const FieldContainer<double> & BasisCache::getSideNormals() {
  return _sideNormals;
}

void BasisCache::setSideNormals(FieldContainer<double> &sideNormals) {
  _sideNormals = sideNormals;
}

const FieldContainer<double> & BasisCache::getCellSideParities() {
  return _cellSideParities;
}

void BasisCache::setCellSideParities(const FieldContainer<double> &cellSideParities) {
  _cellSideParities = cellSideParities;
}

void BasisCache::setTransformationFunction(FunctionPtr fxn, bool composeWithMeshTransformation) {
  _transformationFxn = fxn;
  _composeTransformationFxnWithMeshTransformation = composeWithMeshTransformation;
  // recompute physical points and jacobian values
  determinePhysicalPoints();
  determineJacobian();
}

void BasisCache::determinePhysicalPoints() {
  if (_spaceDim==0) return; // physical points not meaningful then...
  int numPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
  if ( Function::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation) {
    // _spaceDim for side cache refers to the volume cache's spatial dimension
    _physCubPoints.resize(_numCells, numPoints, _spaceDim);
    if (_cellTopo->getTensorialDegree() == 0) {
      if ( ! isSideCache() ) {
        CellTools<double>::mapToPhysicalFrame(_physCubPoints,_cubPoints,_physicalCellNodes,_cellTopo->getShardsTopology());
      } else {
        CellTools<double>::mapToPhysicalFrame(_physCubPoints,_cubPointsSideRefCell,_physicalCellNodes,_cellTopo->getShardsTopology());
      }
    } else {
      cout << "ERROR: BasisCache::determinePhysicalPoints() does not yet support tensorial degree > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache::determinePhysicalPoints() does not yet support tensorial degree > 0");
    }
  } else {
    // if we get here, then Function is meant to work on reference cell
    // unsupported for now
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Reference-cell based transformations presently unsupported");
    // (what we need to do here is either copy the _cubPoints to _numCells locations in _physCubPoints,
    //  or make the relevant Function simply work on the reference points, returning a FieldContainer
    //  with a numCells dimension.  The latter makes more sense to me--and is more efficient.  The Function should
    //  simply call BasisCache->getRefCellPoints()...  On this idea, we don't even have to do anything special in
    //  BasisCache: the if clause above only serves to save us a little computational effort.)
  }
  if ( ! Function::isNull(_transformationFxn) ) {
    FieldContainer<double> newPhysCubPoints(_numCells,numPoints,_spaceDim);
    BasisCachePtr thisPtr = Teuchos::rcp(this,false);
    
    // the usual story: we don't want to use the transformation Function inside the BasisCache
    // while the transformation Function is using the BasisCache to determine its values.
    // So we move _transformationFxn out of the way for a moment:
    FunctionPtr transformationFxn = _transformationFxn;
    _transformationFxn = Function::null();
    {
      // size cell Jacobian containers—the dimensions are used even for HGRAD basis OP_VALUE, which
      // may legitimately be invoked by transformationFxn, below...
      _cellJacobian.resize(_numCells, numPoints, _spaceDim, _spaceDim);
      _cellJacobInv.resize(_numCells, numPoints, _spaceDim, _spaceDim);
      _cellJacobDet.resize(_numCells, numPoints);
      // (the sizes here agree with what's done in determineJacobian, so the resizing there should be
      //  basically free if we've done it here.)
    }
    
    transformationFxn->values(newPhysCubPoints, thisPtr);
    _transformationFxn = transformationFxn;
    
    _physCubPoints = newPhysCubPoints;
  }
}

void BasisCache::determineJacobian() {
  if (_spaceDim == 0) return;  // Jacobians not meaningful then...
  
  // Compute cell Jacobians, their inverses and their determinants
  
  int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
  
  // Containers for Jacobian
  _cellJacobian.resize(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobInv.resize(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobDet.resize(_numCells, numCubPoints);
  
  typedef CellTools<double>  CellTools;
  
  if ( Function::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation) {
    if (_cellTopo->getTensorialDegree() == 0) {
      if (!isSideCache())
        CellTools::setJacobian(_cellJacobian, _cubPoints, _physicalCellNodes, _cellTopo->getShardsTopology());
      else {
        CellTools::setJacobian(_cellJacobian, _cubPointsSideRefCell, _physicalCellNodes, _cellTopo->getShardsTopology());
      }
    } else {
      cout << "ERROR: BasisCache::determineJacobian() does not yet support tensorial degree > 0.\n";
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "BasisCache::determinePhysicalPoints() does not yet support tensorial degree > 0");
    }
  }
  
  CellTools::setJacobianDet(_cellJacobDet, _cellJacobian );
//  cout << "On rank " << Teuchos::GlobalMPISession::getRank() << ", about to compute jacobian inverse for cellJacobian of size: " << _cellJacobian.size() << endl;
  CellTools::setJacobianInv(_cellJacobInv, _cellJacobian );
  
  if (! Function::isNull(_transformationFxn) ) {
    BasisCachePtr thisPtr = Teuchos::rcp(this,false);
    if (_composeTransformationFxnWithMeshTransformation) {
      // then we need to multiply one Jacobian by the other
      FieldContainer<double> fxnJacobian(_numCells,numCubPoints,_spaceDim,_spaceDim);
      // a little quirky, but since _transformationFxn calls BasisCache in its values determination,
      // we disable the _transformationFxn during the call to grad()->values()
      FunctionPtr fxnCopy = _transformationFxn;
      _transformationFxn = Function::null();
      fxnCopy->grad()->values( fxnJacobian, thisPtr );
      _transformationFxn = fxnCopy;
      
//      cout << "fxnJacobian:\n" << fxnJacobian;
//      cout << "_cellJacobian before multiplication:\n" << _cellJacobian;
      // TODO: check that the order of multiplication is correct!
      FieldContainer<double> cellJacobianToMultiply(_cellJacobian); // tensorMultiplyDataData doesn't support multiplying in place
      fst::tensorMultiplyDataData<double>( _cellJacobian, fxnJacobian, cellJacobianToMultiply );
//      cout << "_cellJacobian after multiplication:\n" << _cellJacobian;
    } else {
      _transformationFxn->grad()->values( _cellJacobian, thisPtr );
    }
    
    CellTools::setJacobianInv(_cellJacobInv, _cellJacobian );
    CellTools::setJacobianDet(_cellJacobDet, _cellJacobian );
  }
}

void BasisCache::setPhysicalCellNodes(const FieldContainer<double> &physicalCellNodes, 
                                      const vector<GlobalIndexType> &cellIDs, bool createSideCacheToo) {
  discardPhysicalNodeInfo(); // necessary to get rid of transformed values, which will no longer be valid
  
  _physicalCellNodes = physicalCellNodes;
  _numCells = physicalCellNodes.dimension(0);
  _spaceDim = physicalCellNodes.dimension(2);
  
  _cellIDs = cellIDs;
  // Compute cell Jacobians, their inverses and their determinants

  // compute physicalCubaturePoints, the transformed cubature points on each cell:
  determinePhysicalPoints(); // when using _transformationFxn, important to have physical points before Jacobian is computed
//  cout << "physicalCellNodes:\n" << physicalCellNodes;
  determineJacobian();
  
  // recompute weighted measure at the new physical points
  recomputeMeasures();
  
  if ( ! isSideCache() && createSideCacheToo ) {
    // we only actually create side caches anew if they don't currently exist
    if (_basisCacheSides.size() == 0) {
      createSideCaches();
    }
    for (int sideOrdinal=0; sideOrdinal<_numSides; sideOrdinal++) {
      _basisCacheSides[sideOrdinal]->setPhysicalCellNodes(physicalCellNodes, cellIDs, false);
    }
  } else if (! isSideCache() && ! createSideCacheToo ) {
    // then we have side caches whose values are going to be stale: we should delete these
    _basisCacheSides.clear();
  }
}

int BasisCache::maxTestDegree() {
  return _maxTestDegree;
}

int BasisCache::getSpaceDim() {
  return _spaceDim;
}

// static convenience constructors:
BasisCachePtr BasisCache::parametric1DCache(int cubatureDegree) {
  return BasisCache::basisCache1D(0, 1, cubatureDegree);
}

BasisCachePtr BasisCache::parametricQuadCache(int cubatureDegree, const FieldContainer<double> &refCellPoints, int sideCacheIndex) {
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

  if (!creatingSideCache) {
    parametricCache->setRefCellPoints(refCellPoints);
    return parametricCache;
  } else {
    parametricCache->getSideBasisCache(sideCacheIndex)->setRefCellPoints(refCellPoints);
    return parametricCache->getSideBasisCache(sideCacheIndex);
  }
}

BasisCachePtr BasisCache::parametricQuadCache(int cubatureDegree) {
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

BasisCachePtr BasisCache::basisCache1D(double x0, double x1, int cubatureDegree) { // x0 and x1: physical space endpoints
  int numCells = 1;
  int numVertices = 2;
  int spaceDim = 1;
  FieldContainer<double> physicalCellNodes(numCells,numVertices,spaceDim);
  physicalCellNodes(0,0,0) = x0;
  physicalCellNodes(0,1,0) = x1;
  shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
  return Teuchos::rcp( new BasisCache(physicalCellNodes, line_2, cubatureDegree));
}

BasisCachePtr BasisCache::basisCacheForCell(MeshPtr mesh, GlobalIndexType cellID, bool testVsTest, int cubatureDegreeEnrichment) {
  ElementTypePtr elemType = mesh->getElementType(cellID);
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh, testVsTest, cubatureDegreeEnrichment) );
  bool createSideCache = true;
  vector<GlobalIndexType> cellIDs(1,cellID);
  basisCache->setPhysicalCellNodes(mesh->physicalCellNodesForCell(cellID), cellIDs, createSideCache);
  basisCache->setCellSideParities(mesh->cellSideParitiesForCell(cellID));
  
  return basisCache;
}
BasisCachePtr BasisCache::basisCacheForCellType(MeshPtr mesh, ElementTypePtr elemType, bool testVsTest,
                                                int cubatureDegreeEnrichment) { // for cells on the local MPI node
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh, testVsTest, cubatureDegreeEnrichment) );
  bool createSideCache = true;
  vector<GlobalIndexType> cellIDs = mesh->cellIDsOfType(elemType);
  if (cellIDs.size() > 0) {
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, createSideCache);
  }
  
  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree, bool createSideCacheToo) {
  FieldContainer<double> cellNodes(cellTopo.getNodeCount(),cellTopo.getDimension());
  CamelliaCellTools::refCellNodesForTopology(cellNodes, cellTopo);
  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(cellNodes,cellTopo,cubatureDegree,createSideCacheToo));
  return basisCache;
}

BasisCachePtr BasisCache::basisCacheForRefinedReferenceCell(shards::CellTopology &cellTopo, int cubatureDegree,
                                                            RefinementBranch refinementBranch, bool createSideCacheToo) {
  FieldContainer<double> cellNodes = RefinementPattern::descendantNodesRelativeToAncestorReferenceCell(refinementBranch);
  
  cellNodes.resize(1,cellNodes.dimension(0),cellNodes.dimension(1));
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(cellNodes,cellTopo,cubatureDegree,createSideCacheToo));
  return basisCache;
}

BasisCachePtr BasisCache::quadBasisCache(double width, double height, int cubDegree, bool createSideCacheToo) {
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

void BasisCache::recomputeMeasures() {
  if (_spaceDim == 0) {
    // then we define the measure of the domain as 1...
    int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
    _weightedMeasure.resize(_numCells, numCubPoints);
    _weightedMeasure.initialize(1.0);
    return;
  }
  if (_cubWeights.size() > 0) {
    int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
    // a bit ugly: "_cubPoints" may not be cubature points at all, but just points of interest...  If they're not cubature points, then _cubWeights will be cleared out.  See setRefCellPoints, above.
    // TODO: rename _cubPoints and related methods...
    _weightedMeasure.resize(_numCells, numCubPoints);
    if (! isSideCache()) {
      fst::computeCellMeasure<double>(_weightedMeasure, _cellJacobDet, _cubWeights);
    } else {
      if (_spaceDim==1) {
        // TODO: determine whether this is the right thing:
        _weightedMeasure.initialize(1.0); // not sure this is the right thing.
      } else if (_spaceDim==2) {
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
      } else if (_spaceDim==3) {
        if (_cellTopo->getTensorialDegree() == 0) {
          FunctionSpaceTools::computeFaceMeasure<double>(_weightedMeasure, _cellJacobian, _cubWeights, _sideIndex, _cellTopo->getShardsTopology());
        } else {
          cout << "ERROR: BasisCache::recomputeMeasures() does not yet support tensorial degree > 0.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "recomputeMeasures() does not yet support tensorial degree > 0.");
        }
      } else {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled space dimension.");
      }
      //      if (_sideIndex==0) {
      //        cout << "_cellJacobian:\n" << _cellJacobian;
      //        cout << "_cubWeights:\n" << _cubWeights;
      //        cout << "_weightedMeasure:\n" << _weightedMeasure;
      //      }
      if (_spaceDim > 1) {
        // get normals
        _sideNormals.resize(_numCells, numCubPoints, _spaceDim);
        FieldContainer<double> normalLengths(_numCells, numCubPoints);
        if (_cellTopo->getTensorialDegree() == 0) {
          CellTools<double>::getPhysicalSideNormals(_sideNormals, _cellJacobian, _sideIndex, _cellTopo->getShardsTopology());
        } else {
          cout << "ERROR: BasisCache::recomputeMeasures() does not yet support tensorial degree > 0.\n";
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "recomputeMeasures() does not yet support tensorial degree > 0.");
        }

        
        // make unit length
        RealSpaceTools<double>::vectorNorm(normalLengths, _sideNormals, NORM_TWO);
        FunctionSpaceTools::scalarMultiplyDataData<double>(_sideNormals, normalLengths, _sideNormals, true);
      } else if (_spaceDim == 1) {
        _sideNormals.resize(_numCells, numCubPoints, _spaceDim); // here, numPoints should be 1
        unsigned thisSideOrdinal = _sideIndex;
        unsigned otherSideOrdinal = 1 - thisSideOrdinal;
        for (int cellOrdinal=0; cellOrdinal<_numCells; cellOrdinal++) {
          double x_this = _physicalCellNodes(cellOrdinal,thisSideOrdinal,0);
          double x_other = _physicalCellNodes(cellOrdinal,otherSideOrdinal,0);
          if (x_this > x_other) {
            _sideNormals(cellOrdinal,0,0) = 1;
          } else {
            _sideNormals(cellOrdinal,0,0) = -1;
          }
        }
      }
    }
  }
}

BasisCachePtr BasisCache::sideBasisCache(Teuchos::RCP<BasisCache> volumeCache, int sideIndex) {
  int spaceDim = volumeCache->cellTopology()->getDimension();
  int numSides = volumeCache->cellTopology()->getSideCount();
  
  TEUCHOS_TEST_FOR_EXCEPTION(sideIndex >= numSides, std::invalid_argument, "sideIndex out of range");

  int maxTestDegree = volumeCache->_maxTestDegree;
  int maxTrialDegreeOnSide = volumeCache->_maxTrialDegree;
  BasisPtr multiBasisIfAny;
  
  if (spaceDim > 1) {
    BasisPtr maxDegreeBasisOnSide = volumeCache->_maxDegreeBasisForSide[sideIndex];
    if (maxDegreeBasisOnSide.get() != NULL) {
      if (BasisFactory::basisFactory()->isMultiBasis(maxDegreeBasisOnSide)) {
        multiBasisIfAny = maxDegreeBasisOnSide;
      }
      maxTrialDegreeOnSide = maxDegreeBasisOnSide->getDegree();
    }
  }
  BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideIndex, volumeCache, maxTrialDegreeOnSide, maxTestDegree, multiBasisIfAny));
  return sideCache;
}