/*
 *  BasisCache.cpp
 *
 */
// @HEADER
//
// Copyright © 2011 Sandia Corporation. All Rights Reserved.
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

typedef Teuchos::RCP< FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const FieldContainer<double> > constFCPtr;
typedef Teuchos::RCP< Basis<double,FieldContainer<double> > > BasisPtr;
typedef FunctionSpaceTools fst;
typedef Teuchos::RCP<Vectorized_Basis<double, FieldContainer<double> > > VectorBasisPtr;

// TODO: add exceptions for side cache arguments to methods that don't make sense 
// (e.g. useCubPointsSideRefCell==true when _isSideCache==false)

void BasisCache::init(shards::CellTopology &cellTopo, DofOrdering &trialOrdering,
                      int maxTestDegree, bool createSideCacheToo) {
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
}

BasisCache::BasisCache(ElementTypePtr elemType, bool testVsTest, int cubatureDegreeEnrichment) {
  // use testVsTest=true for test space inner product (won't create side caches, and will use higher cubDegree)
  shards::CellTopology cellTopo = *(elemType->cellTopoPtr);

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

BasisCache::BasisCache(shards::CellTopology &cellTopo, int numCells, int spaceDim, 
                       FieldContainer<double> &cubPointsSidePhysical,
                       FieldContainer<double> &cubPointsSide, FieldContainer<double> &cubPointsSideRefCell, 
                       FieldContainer<double> &cubWeightsSide, FieldContainer<double> &sideMeasure,
                       FieldContainer<double> &sideNormals, FieldContainer<double> &jacobianSideRefCell,
                       FieldContainer<double> &jacobianInvSideRefCell, FieldContainer<double> &jacobianDetSideRefCell) {
  _isSideCache = true; // this is the SIDE constructor: we don't have sides here!  (// TODO: think about 3D here)
  
  _cellTopo = cellTopo;
  _numCells = numCells;
  _spaceDim = spaceDim;
  
  _cubPoints = cubPointsSide;
  _cubPointsSideRefCell = cubPointsSideRefCell;
  
  _physCubPoints = cubPointsSidePhysical; // NOTE the meaning of _physCubPoints in this context: these are in _spaceDim + 1 dimensions...
  _cubWeights = cubWeightsSide;
  _weightedMeasure = sideMeasure;
  
  _sideNormals = sideNormals;
  
  _cellJacobian = jacobianSideRefCell;
  _cellJacobInv = jacobianInvSideRefCell;
  _cellJacobDet = jacobianDetSideRefCell;
  
}

const vector<int> & BasisCache::cellIDs() {
  return _cellIDs;
}

shards::CellTopology BasisCache::cellTopology() {
  return _cellTopo;
}

void BasisCache::discardPhysicalNodeInfo() {
  // discard physicalNodes and all transformed basis values.
  _knownValuesTransformed.clear();
  _knownValuesTransformedWeighted.clear();
  _knownValuesTransformedDottedWithNormal.clear();
  _knownValuesTransformedWeighted.clear();
  _cellIDs.clear();
  
  // resize all the related fieldcontainers to reclaim their memory
  _cellJacobian.resize(0);
  _cellJacobInv.resize(0);
  _cellJacobDet.resize(0);
  _weightedMeasure.resize(0);
  _physCubPoints.resize(0);
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

constFCPtr BasisCache::getValues(BasisPtr basis, EOperatorExtended op,
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
    TEST_FOR_EXCEPTION(true,std::invalid_argument,
                       "Unknown basis.  BasisCache only works for bases created by BasisFactory");
  }
  // first, let's check whether the exact request is already known
  pair< Basis<double,FieldContainer<double> >*, EOperatorExtended> key = make_pair(basis.get(), op);
  
  if (_knownValues.find(key) != _knownValues.end() ) {
    return _knownValues[key];
  }
  int componentOfInterest = -1;
  // otherwise, lookup to see whether a related value is already known
  EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);
  
  pair<Basis<double,FieldContainer<double> >*, EOperatorExtended> relatedKey = key;
  if ((EOperatorExtended)relatedOp != op) {
    relatedKey = make_pair(basis.get(), (IntrepidExtendedTypes::EOperatorExtended) relatedOp);
    if (_knownValues.find(relatedKey) == _knownValues.end() ) {
      // we can assume relatedResults has dimensions (numPoints,basisCardinality,spaceDim)
      FCPtr relatedResults = BasisEvaluation::getValues(basis,(EOperatorExtended)relatedOp,cubPoints);
      _knownValues[relatedKey] = relatedResults;
    }
    constFCPtr relatedResults = _knownValues[key];
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
  if ( (op >= IntrepidExtendedTypes::OPERATOR_X) || (op < IntrepidExtendedTypes::OPERATOR_VALUE) ) {
    TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unknown operator.");
  }
  FCPtr result = BasisEvaluation::getValues(basis,op,cubPoints);
  _knownValues[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedValues(BasisPtr basis, EOperatorExtended op,
                                            bool useCubPointsSideRefCell) {
  pair<Basis<double,FieldContainer<double> >*, EOperatorExtended> key = make_pair(basis.get(), op);
  if (_knownValuesTransformed.find(key) != _knownValuesTransformed.end()) {
    return _knownValuesTransformed[key];
  }
  
  int componentOfInterest;
  EFunctionSpaceExtended fs = BasisFactory::getBasisFunctionSpace(basis);
  Intrepid::EOperator relatedOp = BasisEvaluation::relatedOperator(op, fs, componentOfInterest);
  
  pair<Basis<double,FieldContainer<double> >*, EOperatorExtended> relatedKey = make_pair(basis.get(),(EOperatorExtended) relatedOp);
  if (_knownValuesTransformed.find(relatedKey) == _knownValuesTransformed.end()) {
    constFCPtr transformedValues;
    if ( (fs == IntrepidExtendedTypes::FUNCTION_SPACE_VECTOR_HGRAD) 
        && ((op == IntrepidExtendedTypes::OPERATOR_VALUE) || (op == IntrepidExtendedTypes::OPERATOR_CROSS_NORMAL) )) {
      VectorBasisPtr vectorBasis = Teuchos::rcp( (Vectorized_Basis<double, FieldContainer<double> > *) basis.get(), false );
      BasisPtr componentBasis = vectorBasis->getComponentBasis();
      constFCPtr componentReferenceValuesTransformed = getTransformedValues(componentBasis,IntrepidExtendedTypes::OPERATOR_VALUE,
                                                                            useCubPointsSideRefCell);
      transformedValues = BasisEvaluation::getTransformedVectorValuesWithComponentBasisValues(vectorBasis,
                                                                                              IntrepidExtendedTypes::OPERATOR_VALUE,
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
  if (   (op != IntrepidExtendedTypes::OPERATOR_CROSS_NORMAL) && (op != IntrepidExtendedTypes::OPERATOR_DOT_NORMAL)
      && (op != IntrepidExtendedTypes::OPERATOR_TIMES_NORMAL) && (op != IntrepidExtendedTypes::OPERATOR_VECTORIZE_VALUE) ) {
    result = BasisEvaluation::BasisEvaluation::getComponentOfInterest(relatedValuesTransformed,op,fs,componentOfInterest);
    if ( result.get() == 0 ) {
      result = relatedValuesTransformed;
    }
  } else {
    switch (op) {
      case OPERATOR_CROSS_NORMAL:
        result = BasisEvaluation::getValuesCrossedWithNormals(relatedValuesTransformed,_sideNormals);
        break;
      case OPERATOR_DOT_NORMAL:
        result = BasisEvaluation::getValuesDottedWithNormals(relatedValuesTransformed,_sideNormals);
        break;
      case OPERATOR_TIMES_NORMAL:
        result = BasisEvaluation::getValuesTimesNormals(relatedValuesTransformed,_sideNormals);
        break;
      case OPERATOR_VECTORIZE_VALUE:
        result = BasisEvaluation::getVectorizedValues(relatedValuesTransformed,_spaceDim);
      default:
        break;
    }
  }
  _knownValuesTransformed[key] = result;
  return result;
}

constFCPtr BasisCache::getTransformedWeightedValues(BasisPtr basis, EOperatorExtended op, 
                                                    bool useCubPointsSideRefCell) {
  pair<Basis<double,FieldContainer<double> >*, EOperatorExtended> key = make_pair(basis.get(), op);
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
constFCPtr BasisCache::getValues(BasisPtr basis, EOperatorExtended op, int sideOrdinal,
                                 bool useCubPointsSideRefCell) {
  return _basisCacheSides[sideOrdinal]->getValues(basis,op,useCubPointsSideRefCell);
}

constFCPtr BasisCache::getTransformedValues(BasisPtr basis, EOperatorExtended op, int sideOrdinal, 
                                            bool useCubPointsSideRefCell) {
  constFCPtr transformedValues;
  if ( ! _isSideCache ) {
    transformedValues = _basisCacheSides[sideOrdinal]->getTransformedValues(basis,op,useCubPointsSideRefCell);
  } else {
    transformedValues = getTransformedValues(basis,op,useCubPointsSideRefCell);
  }
  return transformedValues;
}

constFCPtr BasisCache::getTransformedWeightedValues(BasisPtr basis, EOperatorExtended op, 
                                                    int sideOrdinal, bool useCubPointsSideRefCell) {
  return _basisCacheSides[sideOrdinal]->getTransformedWeightedValues(basis,op,useCubPointsSideRefCell);
}

const FieldContainer<double> & BasisCache::getPhysicalCubaturePointsForSide(int sideOrdinal) {
  return _basisCacheSides[sideOrdinal]->getPhysicalCubaturePoints();
}

const FieldContainer<double> & BasisCache::getSideUnitNormals(int sideOrdinal){  
  return _basisCacheSides[sideOrdinal]->_sideNormals;
}

void BasisCache::setPhysicalCellNodes(const FieldContainer<double> &physicalCellNodes, const vector<int> &cellIDs, bool createSideCacheToo) {
  discardPhysicalNodeInfo(); // necessary to get rid of transformed values, which will no longer be valid
  
  _numCells = physicalCellNodes.dimension(0);
  _spaceDim = physicalCellNodes.dimension(2);
  
  _basisCacheSides.clear();  // it would be better not to have to recreate these every time the physicalCellNodes changes, but this is a first pass.
  _cellIDs = cellIDs;
  // 1. Determine Jacobians
  // Compute cell Jacobians, their inverses and their determinants
  
  int numCubPoints = _cubPoints.dimension(0);
  
  // Containers for Jacobian
  _cellJacobian = FieldContainer<double>(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobInv = FieldContainer<double>(_numCells, numCubPoints, _spaceDim, _spaceDim);
  _cellJacobDet = FieldContainer<double>(_numCells, numCubPoints);
  
  typedef CellTools<double>  CellTools;
  
  CellTools::setJacobian(_cellJacobian, _cubPoints, physicalCellNodes, _cellTopo);
  CellTools::setJacobianInv(_cellJacobInv, _cellJacobian );
  CellTools::setJacobianDet(_cellJacobDet, _cellJacobian );
  
  // compute weighted measure
  _weightedMeasure = FieldContainer<double>(_numCells, numCubPoints);
  fst::computeCellMeasure<double>(_weightedMeasure, _cellJacobDet, _cubWeights);
  
  // compute physicalCubaturePoints, the transformed cubature points on each cell:
  _physCubPoints = FieldContainer<double>(_numCells, numCubPoints, _spaceDim);
  CellTools::mapToPhysicalFrame(_physCubPoints,_cubPoints,physicalCellNodes,_cellTopo);
  
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
      BasisCache* sideCache = new BasisCache(_cellTopo, _numCells, _spaceDim, cubPointsSidePhysical,
                                             cubPointsSide, cubPointsSideRefCell, 
                                             cubWeightsSide, weightedMeasureSideRefCell,
                                             sideNormals, jacobianSideRefCell,
                                             jacobianInvSideRefCell, jacobianDetSideRefCell);
      
      _basisCacheSides.push_back( Teuchos::rcp(sideCache) );
    }
  }
}