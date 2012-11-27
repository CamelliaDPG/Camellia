/*
 *  BasisCache.cpp
 *
 */
// @HEADER
//
// Original version Copyright © 2011 Sandia Corporation. All Rights Reserved.
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

#include "BasisCache.h"
#include "BasisFactory.h"
#include "BasisEvaluation.h"
#include "Mesh.h"
#include "Function.h"

typedef Teuchos::RCP< FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;
typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef FunctionSpaceTools fst;
typedef Teuchos::RCP<Vectorized_Basis<double, FieldContainer<double> > > VectorBasisPtr;

// TODO: add exceptions for side cache arguments to methods that don't make sense 
// (e.g. useCubPointsSideRefCell==true when _isSideCache==false)

// init is for volume caches.
void BasisCache::init(shards::CellTopology &cellTopo, DofOrdering &trialOrdering,
                      int maxTestDegree, bool createSideCacheToo) {
  _sideIndex = -1;
  _isSideCache = false; // VOLUME constructor
  
  _cellTopo = cellTopo;
  
  // _cubDegree, _maxTestDegree, and _cubFactory should become local to init() once we
  // put side cache creation back here, where it belongs.)
  // (same might be true of _trialOrdering.)
  _cubDegree = trialOrdering.maxBasisDegree() + maxTestDegree;
  _maxTestDegree = maxTestDegree;
  Teuchos::RCP<Cubature<double> > cellTopoCub = _cubFactory.create(cellTopo, _cubDegree); 
  _trialOrdering = trialOrdering; // makes a copy--would be better to have an RCP here
  
  int cubDim       = cellTopoCub->getDimension();
  int numCubPoints = cellTopoCub->getNumPoints();
  
  _cubPoints = FieldContainer<double>(numCubPoints, cubDim);
  _cubWeights.resize(numCubPoints);
  
  cellTopoCub->getCubature(_cubPoints, _cubWeights);
  
  // now, create side caches
  if ( createSideCacheToo ) {
    _numSides = _cellTopo.getSideCount();
    vector<int> sideTrialIDs;
    vector<int> trialIDs = trialOrdering.getVarIDs();
    int numTrialIDs = trialIDs.size();
    for (int i=0; i<numTrialIDs; i++) {
      if (_trialOrdering.getNumSidesForVarID(trialIDs[i]) == _numSides) {
        sideTrialIDs.push_back(trialIDs[i]);
      }
    }
    int numSideTrialIDs = sideTrialIDs.size();
    for (int sideOrdinal=0; sideOrdinal<_numSides; sideOrdinal++) {
      BasisPtr maxDegreeBasisOnSide;
      // loop through looking for highest-degree basis
      int maxTrialDegree = 0;
      for (int i=0; i<numSideTrialIDs; i++) {
        if (_trialOrdering.getBasis(sideTrialIDs[i],sideOrdinal)->getDegree() > maxTrialDegree) {
          maxDegreeBasisOnSide = _trialOrdering.getBasis(sideTrialIDs[i],sideOrdinal);
        }
      }
      BasisCachePtr thisPtr = Teuchos::rcp( this, false ); // presumption is that side cache doesn't outlive volume...
      BasisCachePtr sideCache = Teuchos::rcp( new BasisCache(sideOrdinal, thisPtr, maxDegreeBasisOnSide));
      _basisCacheSides.push_back(sideCache);
    }
  }
}

BasisCache::BasisCache(ElementTypePtr elemType, Teuchos::RCP<Mesh> mesh, bool testVsTest, int cubatureDegreeEnrichment) {
  // use testVsTest=true for test space inner product (won't create side caches, and will use higher cubDegree)
  shards::CellTopology cellTopo = *(elemType->cellTopoPtr);
  
  _mesh = mesh;

  DofOrdering trialOrdering;
  if (testVsTest)
    trialOrdering = *(elemType->testOrderPtr); // bit of a lie here -- treat the testOrdering as trialOrdering
  else
    trialOrdering = *(elemType->trialOrderPtr);
  
  bool createSideCacheToo = !testVsTest;
  
  int maxTestDegree = elemType->testOrderPtr->maxBasisDegree();

  init(cellTopo,trialOrdering,maxTestDegree + cubatureDegreeEnrichment,createSideCacheToo);
}


BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, 
                       shards::CellTopology &cellTopo,
                       DofOrdering &trialOrdering, int maxTestDegree, bool createSideCacheToo) {
  init(cellTopo, trialOrdering, maxTestDegree, createSideCacheToo);
  setPhysicalCellNodes(physicalCellNodes,vector<int>(),createSideCacheToo);
}

BasisCache::BasisCache(const FieldContainer<double> &physicalCellNodes, shards::CellTopology &cellTopo, int cubDegree) {
  DofOrdering trialOrdering; // dummy trialOrdering
  bool createSideCacheToo = false;
  init(cellTopo, trialOrdering, cubDegree, createSideCacheToo);
  setPhysicalCellNodes(physicalCellNodes,vector<int>(),createSideCacheToo);
}

// side constructor
BasisCache::BasisCache(int sideIndex, BasisCachePtr volumeCache, BasisPtr maxDegreeBasis) {
  _isSideCache = true;
  _sideIndex = sideIndex;
  _basisCacheVolume = volumeCache;
  _cubDegree = volumeCache->cubatureDegree();
  _maxTestDegree = volumeCache->maxTestDegree();

  _cellTopo = volumeCache->cellTopology(); // VOLUME cell topo.
  _spaceDim = _cellTopo.getDimension();
  
  shards::CellTopology side(_cellTopo.getCellTopologyData(_spaceDim-1,sideIndex)); // create relevant subcell (side) topology
  int sideDim = side.getDimension();
  Teuchos::RCP<Cubature<double> > sideCub = _cubFactory.create(side, _cubDegree);
  int numCubPointsSide = sideCub->getNumPoints();
  _cubPoints.resize(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a 1D set)
  _cubWeights.resize(numCubPointsSide);
  
  if ( ! BasisFactory::isMultiBasis(maxDegreeBasis) ) {
    sideCub->getCubature(_cubPoints, _cubWeights);
  } else {
    MultiBasis* multiBasis = (MultiBasis*) maxDegreeBasis.get();
    multiBasis->getCubature(_cubPoints, _cubWeights, _maxTestDegree);
    numCubPointsSide = _cubPoints.dimension(0);
  }
  
  _cubPointsSideRefCell.resize(numCubPointsSide, _spaceDim); // cubPointsSide from the pov of the ref cell
  
  // compute geometric cell information
  //cout << "computing geometric cell info for boundary integral." << endl;
  typedef CellTools<double>  CellTools;
  
  CellTools::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
}

BasisCache::BasisCache(int sideIndex, shards::CellTopology &cellTopo, int numCells, int spaceDim,
                       FieldContainer<double> &cubPointsSidePhysical,
                       FieldContainer<double> &cubPointsSide, FieldContainer<double> &cubPointsSideRefCell,
                       FieldContainer<double> &cubWeightsSide, FieldContainer<double> &sideMeasure,
                       FieldContainer<double> &sideNormals, FieldContainer<double> &jacobianSideRefCell,
                       FieldContainer<double> &jacobianInvSideRefCell, FieldContainer<double> &jacobianDetSideRefCell,
                       const vector<int> &cellIDs, FieldContainer<double> &physicalCellNodes, BasisCachePtr volumeCache) {
  _isSideCache = true; // this is the SIDE constructor: we don't have sides here!  (// TODO: think about 3D here)
  _sideIndex = sideIndex;
  _basisCacheVolume = volumeCache;
  
  _cellTopo = cellTopo;
  _numCells = numCells;
  _spaceDim = spaceDim; // VOLUME spatial dimension
  
  _cubPoints = cubPointsSide;
  _cubPointsSideRefCell = cubPointsSideRefCell;
  
  _physicalCellNodes = physicalCellNodes;
  
  _physCubPoints = cubPointsSidePhysical; // NOTE the meaning of _physCubPoints in this context: these are in volume cache's spatial dimensions...  (I.e. _physCubPoints is mapped from cubPointsSideRefCell via volumeCache->physicalCellNodes)
  _cubWeights = cubWeightsSide;
  _weightedMeasure = sideMeasure;
  
  _sideNormals = sideNormals;
  
  _cellJacobian = jacobianSideRefCell;
  _cellJacobInv = jacobianInvSideRefCell;
  _cellJacobDet = jacobianDetSideRefCell;
 
  _cellIDs = cellIDs;
}

const vector<int> & BasisCache::cellIDs() {
  return _cellIDs;
}

shards::CellTopology BasisCache::cellTopology() {
  return _cellTopo;
}

int BasisCache::cubatureDegree() {
  return _cubDegree;
}

Teuchos::RCP<Mesh> BasisCache::mesh() {
  if ( ! _isSideCache ) {
    return _mesh;
  } else {
    return _basisCacheVolume->mesh();
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

constFCPtr BasisCache::getValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op,
                                 bool useCubPointsSideRefCell) {
  FieldContainer<double> cubPoints;
  if (useCubPointsSideRefCell) {
    cubPoints = _cubPointsSideRefCell;
  } else {
    cubPoints = _cubPoints;
  }
  //cout << "cubPoints:\n" << cubPoints;
  int numPoints = cubPoints.dimension(0);
  int spaceDim = cubPoints.dimension(1);  // points dimensions are (numPoints, spaceDim)
  int basisCardinality = basis->getCardinality();
  // test to make sure that the basis is known by BasisFactory--otherwise, throw exception
  if (! BasisFactory::basisKnown(basis) ) {
    TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,
                       "Unknown basis.  BasisCache only works for bases created by BasisFactory");
  }
  // first, let's check whether the exact request is already known
  pair< Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended> key = make_pair(basis.get(), op);
  
  if (_knownValues.find(key) != _knownValues.end() ) {
    return _knownValues[key];
  }
  int componentOfInterest = -1;
  // otherwise, lookup to see whether a related value is already known
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);
  
  pair<Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended> relatedKey = key;
  if ((EOperatorExtended)relatedOp != op) {
    relatedKey = make_pair(basis.get(), (IntrepidExtendedTypes::EOperatorExtended) relatedOp);
    if (_knownValues.find(relatedKey) == _knownValues.end() ) {
      // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDim)
      FCPtr relatedResults = BasisEvaluation::getValues(basis,(EOperatorExtended)relatedOp,cubPoints);
      _knownValues[relatedKey] = relatedResults;
    }
    pair<Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended> gradKey = make_pair(basis.get(), IntrepidExtendedTypes::OP_GRAD);
  
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
  pair<Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended> key = make_pair(basis.get(), op);
  if (_knownValuesTransformed.find(key) != _knownValuesTransformed.end()) {
    return _knownValuesTransformed[key];
  }
  
  int componentOfInterest;
  IntrepidExtendedTypes::EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  Intrepid::EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);
  
  pair<Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended> relatedKey = make_pair(basis.get(),(EOperatorExtended) relatedOp);
  if (_knownValuesTransformed.find(relatedKey) == _knownValuesTransformed.end()) {
    constFCPtr transformedValues;
    if ( (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) && (relatedOp ==  Intrepid::OPERATOR_VALUE)) {
      VectorBasisPtr vectorBasis = Teuchos::rcp( (Vectorized_Basis<double, FieldContainer<double> > *) basis.get(), false );
      BasisPtr componentBasis = vectorBasis->getComponentBasis();
      constFCPtr componentReferenceValuesTransformed = getTransformedValues(componentBasis, IntrepidExtendedTypes::OP_VALUE,
                                                                            useCubPointsSideRefCell);
      transformedValues = BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(vectorBasis,
                                                                                               IntrepidExtendedTypes::OP_VALUE,
                                                                                              componentReferenceValuesTransformed);
    } else {
      constFCPtr referenceValues = getValues(basis,(EOperatorExtended) relatedOp, useCubPointsSideRefCell);
      transformedValues = 
      BasisEvaluation::getTransformedValuesWithBasisValues(basis, (EOperatorExtended) relatedOp,
                                                           referenceValues, _cellJacobian, 
                                                           _cellJacobInv,_cellJacobDet);
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
    }
  }
  _knownValuesTransformed[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedWeightedValues(BasisPtr basis, IntrepidExtendedTypes::EOperatorExtended op, 
                                                    bool useCubPointsSideRefCell) {
  pair<Basis<double,FieldContainer<double> >*, IntrepidExtendedTypes::EOperatorExtended> key = make_pair(basis.get(), op);
  if (_knownValuesTransformedWeighted.find(key) != _knownValuesTransformedWeighted.end()) {
    return _knownValuesTransformedWeighted[key];
  }
  constFCPtr unWeightedValues = getTransformedValues(basis,op, useCubPointsSideRefCell);
  typedef FunctionSpaceTools fst;
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

const FieldContainer<double> BasisCache::getRefCellPoints() {
  return _cubPoints;
}

void BasisCache::setRefCellPoints(const FieldContainer<double> &pointsRefCell) {
  _cubPoints = pointsRefCell;
  if ( isSideCache() ) { // then we need to map pointsRefCell (on side) into volume coordinates, and store in _cubPointsSideRefCell
    int numPoints = pointsRefCell.dimension(0);
    // for side cache, _spaceDim is the spatial dimension of the volume cache
    _cubPointsSideRefCell.resize(numPoints, _spaceDim); 
    // _cellTopo is the volume cell topology for side basis caches.
    int sideDim = _spaceDim - 1;
    CellTools<double>::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
  }
  
  _knownValues.clear();
  _knownValuesTransformed.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();
  _knownValuesTransformedWeighted.clear();
  
  _cubWeights.resize(0); // will force an exception if you try to compute weighted values.
//  discardPhysicalNodeInfo();
  // experimental/new: allow reuse of physicalNode info; just map the new points...
  if (_physCubPoints.size() > 0) {
    determinePhysicalPoints();
    determineJacobian();
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

void BasisCache::determinePhysicalPoints() {
  if ( ! isSideCache() ) {
    int numPoints = _cubPoints.dimension(0);
    _physCubPoints = FieldContainer<double>(_numCells, numPoints, _spaceDim);
    CellTools<double>::mapToPhysicalFrame(_physCubPoints,_cubPoints,_physicalCellNodes,_cellTopo);
  } else {
    int numPoints = _cubPointsSideRefCell.dimension(0);
    // _spaceDim for side cache refers to the volume cache's spatial dimension:
    _physCubPoints = FieldContainer<double>(_numCells, numPoints, _spaceDim);
    CellTools<double>::mapToPhysicalFrame(_physCubPoints,_cubPointsSideRefCell,_physicalCellNodes,_cellTopo);    
  }
}

void BasisCache::determineJacobian() {
  // Compute cell Jacobians, their inverses and their determinants
  
  int numCubPoints = _cubPoints.dimension(0);
  
  // Containers for Jacobian
  _cellJacobian.resize(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobInv.resize(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobDet.resize(_numCells, numCubPoints);
  
  typedef CellTools<double>  CellTools;
  
  if (!isSideCache())
    CellTools::setJacobian(_cellJacobian, _cubPoints, _physicalCellNodes, _cellTopo);
  else
    CellTools::setJacobian(_cellJacobian, _cubPointsSideRefCell, _physicalCellNodes, _cellTopo);
  
  CellTools::setJacobianInv(_cellJacobInv, _cellJacobian );
  CellTools::setJacobianDet(_cellJacobDet, _cellJacobian );
}

void BasisCache::setPhysicalCellNodes(const FieldContainer<double> &physicalCellNodes, 
                                      const vector<int> &cellIDs, bool createSideCacheToo) {
  discardPhysicalNodeInfo(); // necessary to get rid of transformed values, which will no longer be valid
  
  _physicalCellNodes = physicalCellNodes;
  _numCells = physicalCellNodes.dimension(0);
  _spaceDim = physicalCellNodes.dimension(2);
  
  _cellIDs = cellIDs;
  // 1. Determine Jacobians
  // Compute cell Jacobians, their inverses and their determinants
  
  int numCubPoints = _cubPoints.dimension(0);
  
//  cout << "physicalCellNodes:\n" << physicalCellNodes;
  
  determineJacobian();
  
  // compute weighted measure
  if (_cubWeights.size() > 0) {
    // a bit ugly: "_cubPoints" may not be cubature points at all, but just points of interest...  If they're not cubature points, then _cubWeights will be cleared out.  See setRefCellPoints, above.
    // TODO: rename _cubPoints and related methods...
    _weightedMeasure = FieldContainer<double>(_numCells, numCubPoints);
    fst::computeCellMeasure<double>(_weightedMeasure, _cellJacobDet, _cubWeights);
  }
  
  // compute physicalCubaturePoints, the transformed cubature points on each cell:
  determinePhysicalPoints();
  
  _basisCacheSides.clear();  // it would be better not to have to recreate these every time the physicalCellNodes changes, but this is a first pass.
  if ( createSideCacheToo ) {
    _numSides = _cellTopo.getSideCount();
    vector<int> sideTrialIDs;
    vector<int> trialIDs = _trialOrdering.getVarIDs();
    int numTrialIDs = trialIDs.size();
    for (int i=0; i<numTrialIDs; i++) {
      if (_trialOrdering.getNumSidesForVarID(trialIDs[i]) == _numSides) {
        sideTrialIDs.push_back(trialIDs[i]);
      }
    }
    int numSideTrialIDs = sideTrialIDs.size();
    for (int sideOrdinal=0; sideOrdinal<_numSides; sideOrdinal++) {
      shards::CellTopology side(_cellTopo.getCellTopologyData(_spaceDim-1,sideOrdinal)); // create relevant subcell (side) topology
      int sideDim = side.getDimension();                              
      Teuchos::RCP<Cubature<double> > sideCub = _cubFactory.create(side, _cubDegree);
      int numCubPointsSide = sideCub->getNumPoints();
      FieldContainer<double> cubPointsSide(numCubPointsSide, sideDim); // cubature points from the pov of the side (i.e. a 1D set)
      FieldContainer<double> cubWeightsSide(numCubPointsSide);
      bool multiBasis = false;
      if ( numSideTrialIDs > 0) {
        BasisPtr sampleBasis = _trialOrdering.getBasis(sideTrialIDs[0],sideOrdinal);
        if (BasisFactory::isMultiBasis(sampleBasis) ) {
          multiBasis = true;
        }
      }
      
      if ( ! multiBasis ) {
        sideCub->getCubature(cubPointsSide, cubWeightsSide);
      } else {
        // loop through looking for highest-degree multi-basis
        BasisPtr basis;
        int maxTrialDegree = 0;
        for (int i=0; i<numSideTrialIDs; i++) {
          if (_trialOrdering.getBasis(sideTrialIDs[i],sideOrdinal)->getDegree() > maxTrialDegree) {
            basis = _trialOrdering.getBasis(sideTrialIDs[i],sideOrdinal);
            maxTrialDegree = basis->getDegree();
          }
        }
        
        MultiBasis* multiBasis = (MultiBasis*) basis.get();
        multiBasis->getCubature(cubPointsSide, cubWeightsSide, _maxTestDegree);
        numCubPointsSide = cubPointsSide.dimension(0);
      }
      
      FieldContainer<double> cubPointsSideRefCell(numCubPointsSide, _spaceDim); // cubPointsSide from the pov of the ref cell
      FieldContainer<double> cubPointsSidePhysical(_numCells, numCubPointsSide, _spaceDim); // cubPointsSide from the pov of the physical cell
      FieldContainer<double> jacobianSideRefCell(_numCells, numCubPointsSide, _spaceDim, _spaceDim);
      FieldContainer<double> jacobianDetSideRefCell(_numCells, numCubPointsSide);
      FieldContainer<double> jacobianInvSideRefCell(_numCells, numCubPointsSide, _spaceDim, _spaceDim);
      FieldContainer<double> weightedMeasureSideRefCell(_numCells, numCubPointsSide);
      
      // compute geometric cell information
      //cout << "computing geometric cell info for boundary integral." << endl;
      typedef CellTools<double>  CellTools;

      CellTools::mapToReferenceSubcell(cubPointsSideRefCell, cubPointsSide, sideDim, (int)sideOrdinal, _cellTopo);
      CellTools::setJacobian(jacobianSideRefCell, cubPointsSideRefCell, physicalCellNodes, _cellTopo);
      
      CellTools::setJacobianDet(jacobianDetSideRefCell, jacobianSideRefCell );
      CellTools::setJacobianInv(jacobianInvSideRefCell, jacobianSideRefCell );
      
      // map side cubature points in reference parent cell domain to physical space
      CellTools::mapToPhysicalFrame(cubPointsSidePhysical, cubPointsSideRefCell, physicalCellNodes, _cellTopo);
      
      // compute weighted edge measure
      FunctionSpaceTools::computeEdgeMeasure<double>(weightedMeasureSideRefCell,
                                                     jacobianSideRefCell,
                                                     cubWeightsSide,
                                                     sideOrdinal,
                                                     _cellTopo);
      
      // get normals
      FieldContainer<double> sideNormals(_numCells, numCubPointsSide, _spaceDim);
      FieldContainer<double> normalLengths(_numCells, numCubPointsSide);
      CellTools::getPhysicalSideNormals(sideNormals, jacobianSideRefCell, sideOrdinal, _cellTopo);
      
      // make unit length
      RealSpaceTools<double>::vectorNorm(normalLengths, sideNormals, NORM_TWO);
      FunctionSpaceTools::scalarMultiplyDataData<double>(sideNormals, normalLengths, sideNormals, true);
      
      // values we want to keep around: cubPointsSide, cubPointsSideRefCell, sideNormals, jacobianSideRefCell, jacobianInvSideRefCell, jacobianDetSideRefCell
      BasisCachePtr thisPtr = Teuchos::rcp(this, false); // sideCache won't outlive us, so this is safe...
      BasisCache* sideCache = new BasisCache(sideOrdinal, _cellTopo, _numCells, _spaceDim, cubPointsSidePhysical,
                                             cubPointsSide, cubPointsSideRefCell, 
                                             cubWeightsSide, weightedMeasureSideRefCell,
                                             sideNormals, jacobianSideRefCell,
                                             jacobianInvSideRefCell, jacobianDetSideRefCell, cellIDs, 
                                             _physicalCellNodes, thisPtr);
      _basisCacheSides.push_back( Teuchos::rcp(sideCache) );
    }
  }
}

int BasisCache::maxTestDegree() {
  return _maxTestDegree;
}

void BasisCache::setTransformationFunction(FunctionPtr f, bool composeWithMeshTransformation) {
  _transformationFxn = f;
  _composeTransformationFxnWithMeshTransformation = composeWithMeshTransformation;
  // bool: compose with existing ref-to-mesh-cell transformation. (false means that the function goes from ref to the physical geometry;
  //                                                                true means it goes from the straight-edge mesh to the curvilinear one)
}