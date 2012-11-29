/*
 *  BasisCache.cpp
 *
 */
// @HEADER
//
// Original version copyright © 2011 Sandia Corporation. All Rights Reserved.
// Revisions copyright © 2012 Nathan Roberts. All Rights Reserved.
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

typedef FunctionSpaceTools fst;
typedef Teuchos::RCP< FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;
typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef Teuchos::RCP<Vectorized_Basis<double, FieldContainer<double> > > VectorBasisPtr;

// TODO: add exceptions for side cache arguments to methods that don't make sense 
// (e.g. useCubPointsSideRefCell==true when _isSideCache==false)

// init is for volume caches.
void BasisCache::init(shards::CellTopology &cellTopo, DofOrdering &trialOrdering,
                      int maxTestDegree, bool createSideCacheToo) {
  _sideIndex = -1;
  _isSideCache = false; // VOLUME constructor
  
  _cellTopo = cellTopo;
  
  // changed the following line from maxBasisDegree: we were generally overintegrating...
  _cubDegree = trialOrdering.maxBasisDegreeForVolume() + maxTestDegree;
  _maxTestDegree = maxTestDegree;
  DefaultCubatureFactory<double> cubFactory;
  Teuchos::RCP<Cubature<double> > cellTopoCub = cubFactory.create(cellTopo, _cubDegree); 
  _trialOrdering = trialOrdering; // makes a copy--would be better to have an RCP here
  
  int cubDim       = cellTopoCub->getDimension();
  int numCubPoints = cellTopoCub->getNumPoints();
  
  _cubPoints = FieldContainer<double>(numCubPoints, cubDim);
  _cubWeights.resize(numCubPoints);
  
  cellTopoCub->getCubature(_cubPoints, _cubWeights);
  
  // now, create side caches
  if ( createSideCacheToo ) {
    createSideCaches();
  }
}

void BasisCache::createSideCaches() {
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
    BasisPtr maxDegreeBasisOnSide;
    // loop through looking for highest-degree basis
    int maxTrialDegree = -1;
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

BasisCache::BasisCache(ElementTypePtr elemType, Teuchos::RCP<Mesh> mesh, bool testVsTest, int cubatureDegreeEnrichment) {
  // use testVsTest=true for test space inner product (won't create side caches, and will use higher cubDegree)
  shards::CellTopology cellTopo = *(elemType->cellTopoPtr);
  
  _mesh = mesh;

  DofOrdering trialOrdering;
  if (testVsTest)
    trialOrdering = *(elemType->testOrderPtr); // bit of a lie here -- treat the testOrdering as trialOrdering
  else
    trialOrdering = *(elemType->trialOrderPtr);
  
  bool createSideCacheToo = !testVsTest && trialOrdering.hasSideVarIDs();
  
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
  _maxTestDegree = volumeCache->maxTestDegree(); // maxTestDegree includes any cubature enrichment
  if ( maxDegreeBasis.get() != NULL ) {
    _cubDegree = maxDegreeBasis->getDegree() + _maxTestDegree;
  } else {
    // this is a "test-vs-test" type BasisCache: for IPs with boundary terms,
    // we may have a side basis cache without any bases that live on the boundaries
    // this is not quite right, in that we'll over-integrate if there's non-zero cubatureEnrichment
    _cubDegree = _maxTestDegree * 2;
  }

  _cellTopo = volumeCache->cellTopology(); // VOLUME cell topo.
  _spaceDim = _cellTopo.getDimension();
  
  shards::CellTopology side(_cellTopo.getCellTopologyData(_spaceDim-1,sideIndex)); // create relevant subcell (side) topology
  int sideDim = side.getDimension();
  DefaultCubatureFactory<double> cubFactory;
  Teuchos::RCP<Cubature<double> > sideCub = cubFactory.create(side, _cubDegree);
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
  CellTools<double>::mapToReferenceSubcell(_cubPointsSideRefCell, _cubPoints, sideDim, _sideIndex, _cellTopo);
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
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unhandled op.");
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

  // allow reuse of physicalNode info; just map the new points...
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
  int numPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
  if ( Function::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation) {
    // _spaceDim for side cache refers to the volume cache's spatial dimension
    _physCubPoints.resize(_numCells, numPoints, _spaceDim);
    if ( ! isSideCache() ) {
      CellTools<double>::mapToPhysicalFrame(_physCubPoints,_cubPoints,_physicalCellNodes,_cellTopo);
    } else {
      CellTools<double>::mapToPhysicalFrame(_physCubPoints,_cubPointsSideRefCell,_physicalCellNodes,_cellTopo);
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
    _transformationFxn->values(newPhysCubPoints, thisPtr);
    _physCubPoints = newPhysCubPoints;
  }
}

void BasisCache::determineJacobian() {
  // Compute cell Jacobians, their inverses and their determinants
  
  int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
  
  // Containers for Jacobian
  _cellJacobian.resize(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobInv.resize(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobDet.resize(_numCells, numCubPoints);
  
  typedef CellTools<double>  CellTools;
  
  if ( Function::isNull(_transformationFxn) || _composeTransformationFxnWithMeshTransformation) {
    if (!isSideCache())
      CellTools::setJacobian(_cellJacobian, _cubPoints, _physicalCellNodes, _cellTopo);
    else {
      CellTools::setJacobian(_cellJacobian, _cubPointsSideRefCell, _physicalCellNodes, _cellTopo);
    }
  }
  
  if (! Function::isNull(_transformationFxn) ) {
    BasisCachePtr thisPtr = Teuchos::rcp(this,false);
    if (_composeTransformationFxnWithMeshTransformation) {
      // then we need to multiply one Jacobian by the other
      FieldContainer<double> fxnJacobian(_numCells,numCubPoints,_spaceDim,_spaceDim);
      _transformationFxn->grad()->values( fxnJacobian, thisPtr );
      
      // TODO: check that the order of multiplication is correct!
      fst::tensorMultiplyDataData<double>( _cellJacobian, fxnJacobian, _cellJacobian );
    } else {
      _transformationFxn->grad()->values( _cellJacobian, thisPtr );
    }
  }
  
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
  
  int numCubPoints = isSideCache() ? _cubPointsSideRefCell.dimension(0) : _cubPoints.dimension(0);
  
  // compute physicalCubaturePoints, the transformed cubature points on each cell:
  determinePhysicalPoints(); // when using _transformationFxn, important to have physical points before Jacobian is computed
  determineJacobian();
  
  // compute weighted measure
  if (_cubWeights.size() > 0) {
    // a bit ugly: "_cubPoints" may not be cubature points at all, but just points of interest...  If they're not cubature points, then _cubWeights will be cleared out.  See setRefCellPoints, above.
    // TODO: rename _cubPoints and related methods...
    _weightedMeasure.resize(_numCells, numCubPoints);
    if (! isSideCache()) {
      fst::computeCellMeasure<double>(_weightedMeasure, _cellJacobDet, _cubWeights);
    } else {
      // compute weighted edge measure
      FunctionSpaceTools::computeEdgeMeasure<double>(_weightedMeasure,
                                                     _cellJacobian,
                                                     _cubWeights,
                                                     _sideIndex,
                                                     _cellTopo);
      
      // get normals
      _sideNormals.resize(_numCells, numCubPoints, _spaceDim);
      FieldContainer<double> normalLengths(_numCells, numCubPoints);
      CellTools<double>::getPhysicalSideNormals(_sideNormals, _cellJacobian, _sideIndex, _cellTopo);
      
      // make unit length
      RealSpaceTools<double>::vectorNorm(normalLengths, _sideNormals, NORM_TWO);
      FunctionSpaceTools::scalarMultiplyDataData<double>(_sideNormals, normalLengths, _sideNormals, true);
    }
  }
  
  
  if ( ! isSideCache() && createSideCacheToo ) {
    // we only actually create side caches anew if they don't currently exist
    if (_basisCacheSides.size() == 0) {
      createSideCaches();
    }
    for (int sideOrdinal=0; sideOrdinal<_numSides; sideOrdinal++) {
      _basisCacheSides[sideOrdinal]->setPhysicalCellNodes(physicalCellNodes, cellIDs, false);
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