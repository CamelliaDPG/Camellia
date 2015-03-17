//
//  SpaceTimeBasisCache.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/11/15.
//
//

#include "SpaceTimeBasisCache.h"
#include "TensorBasis.h"

typedef Teuchos::RCP< Intrepid::FieldContainer<double> > FCPtr;
typedef Teuchos::RCP< const Intrepid::FieldContainer<double> > constFCPtr;

// volume constructor
SpaceTimeBasisCache::SpaceTimeBasisCache(MeshPtr spaceTimeMesh, ElementTypePtr spaceTimeElementType,
                                         const FieldContainer<double> &physicalNodesSpatial,
                                         const FieldContainer<double> &physicalNodesTemporal,
                                         const FieldContainer<double> &physicalNodesSpaceTime,
                                         const std::vector<GlobalIndexType> &cellIDs,
                                         bool testVsTest, int cubatureDegreeEnrichment)
: BasisCache(spaceTimeElementType, spaceTimeMesh, testVsTest, cubatureDegreeEnrichment, true, false) {
  int cellCount = cellIDs.size();
  int sideCount = cellTopology()->getSideCount();
  
  // TODO: construct the spatial basis cache and the temporal basis cache
  // determine space topology
  CellTopoPtr spaceTopo;
  for (int timeSideOrdinal=0; timeSideOrdinal<sideCount; timeSideOrdinal++) {
    if (!cellTopology()->sideIsSpatial(timeSideOrdinal)) {
      spaceTopo = cellTopology()->getSide(timeSideOrdinal);
      break;
    }
  }
  CellTopoPtr timeTopo = CellTopology::line();
  ElementTypePtr spaceElemType = Teuchos::rcp( new ElementType( spaceTimeElementType->trialOrderPtr,
                                                               spaceTimeElementType->testOrderPtr, spaceTopo ) );
  ElementTypePtr timeElemType = Teuchos::rcp( new ElementType( spaceTimeElementType->trialOrderPtr,
                                                              spaceTimeElementType->testOrderPtr, timeTopo ) );
  
  bool tensorTopologyMeansSpaceTime = false; // if space topology is tensor product, don't interpret as space-time
  _spatialCache = Teuchos::rcp( new BasisCache(spaceElemType, Teuchos::null, testVsTest, cubatureDegreeEnrichment,
                                               tensorTopologyMeansSpaceTime) );
  _temporalCache = Teuchos::rcp( new BasisCache(timeElemType, Teuchos::null, testVsTest, cubatureDegreeEnrichment,
                                                tensorTopologyMeansSpaceTime) );
  
  bool createSideCache = true;
  _spatialCache->setPhysicalCellNodes(physicalNodesSpatial, cellIDs, createSideCache);
  _temporalCache->setPhysicalCellNodes(physicalNodesTemporal, cellIDs, createSideCache);
  
  // it may be that ultimately we can get away without doing any space-time construction at all;
  // everything can be done in terms of the spatial BasisCache and the temporal.  For now, we
  // construct space-time physical cell nodes, etc.
  
  setPhysicalCellNodes(physicalNodesSpaceTime, cellIDs, createSideCache);
  
  FieldContainer<double> sideParities(cellCount, sideCount);
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++) {
    GlobalIndexType cellID = cellIDs[cellOrdinal];
    FieldContainer<double> cellSideParities = spaceTimeMesh->cellSideParitiesForCell(cellID);
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      sideParities(cellOrdinal,sideOrdinal) = cellSideParities(0,sideOrdinal);
    }
  }
  setCellSideParities(sideParities);
}

//meshless volume constructor
SpaceTimeBasisCache::SpaceTimeBasisCache(const FieldContainer<double> &physicalNodesSpatial,
                                         const FieldContainer<double> &physicalNodesTemporal,
                                         const FieldContainer<double> &physicalCellNodes,
                                         CellTopoPtr cellTopo, int cubDegree)
: BasisCache(physicalCellNodes,cellTopo,cubDegree,false,true) { // false: don't create side caches during construction; true: tensor product topology (which we should have here) --> space-time
  bool createSideCache = true;
  
  int sideCount = cellTopology()->getSideCount();
  
  // determine space topology
  CellTopoPtr spaceTopo;
  for (int timeSideOrdinal=0; timeSideOrdinal<sideCount; timeSideOrdinal++) {
    if (!cellTopology()->sideIsSpatial(timeSideOrdinal)) {
      spaceTopo = cellTopology()->getSide(timeSideOrdinal);
      break;
    }
  }
  CellTopoPtr timeTopo = CellTopology::line();
  
  bool tensorTopologyMeansSpaceTime = false; // if space topology is tensor product, don't interpret as space-time
  _spatialCache = Teuchos::rcp( new BasisCache(physicalNodesSpatial, spaceTopo, cubDegree, tensorTopologyMeansSpaceTime) );
  _temporalCache = Teuchos::rcp( new BasisCache(physicalNodesTemporal, timeTopo, cubDegree, tensorTopologyMeansSpaceTime) );
  
  vector<GlobalIndexType> cellIDs; //empty
  
  _spatialCache->setPhysicalCellNodes(physicalNodesSpatial, cellIDs, createSideCache);
  _temporalCache->setPhysicalCellNodes(physicalNodesTemporal, cellIDs, createSideCache);
  
  // create side caches
  createSideCaches();
  int numSides = cellTopo->getSideCount();
  for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
    _basisCacheSides[sideOrdinal]->setPhysicalCellNodes(physicalCellNodes, cellIDs, false);
  }
}

// side constructor
SpaceTimeBasisCache::SpaceTimeBasisCache(int sideOrdinal, Teuchos::RCP<SpaceTimeBasisCache> volumeCache,
                                         int trialDegree, int testDegree)
: BasisCache(sideOrdinal, volumeCache, trialDegree, testDegree, (BasisPtr) Teuchos::null) {
  BasisCachePtr spatialCacheVolume = volumeCache->getSpatialBasisCache();
  BasisCachePtr temporalCacheVolume = volumeCache->getTemporalBasisCache();
  
  if ( cellTopology()->sideIsSpatial(sideOrdinal) ) {
    unsigned spatialSideOrdinal = cellTopology()->getSpatialComponentSideOrdinal(sideOrdinal);
    _spatialCache = spatialCacheVolume->getSideBasisCache(spatialSideOrdinal);
    _temporalCache = temporalCacheVolume;
  } else {
    _spatialCache = spatialCacheVolume;
    _temporalCache = Teuchos::null;
  }
}

void SpaceTimeBasisCache::createSideCaches() {
  _basisCacheSides.clear();
  int numSides = this->cellTopology()->getSideCount();
  
  for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
    BasisPtr maxDegreeBasisOnSide = _maxDegreeBasisForSide[sideOrdinal];
    
    int maxTrialDegreeOnSide = _maxTrialDegree;
    if (maxDegreeBasisOnSide.get() != NULL) {
      maxTrialDegreeOnSide = maxDegreeBasisOnSide->getDegree();
    }
    
    Teuchos::RCP<SpaceTimeBasisCache> thisPtr = Teuchos::rcp( this, false ); // presumption is that side cache doesn't outlive volume...
    SpaceTimeBasisCache* spaceTimeSideCache = new SpaceTimeBasisCache(sideOrdinal, thisPtr, maxTrialDegreeOnSide, _maxTestDegree);
    BasisCachePtr sideCache = Teuchos::rcp( spaceTimeSideCache );
    _basisCacheSides.push_back(sideCache);
  }
}

BasisCachePtr SpaceTimeBasisCache::getSpatialBasisCache() {
  return _spatialCache;
}
BasisCachePtr SpaceTimeBasisCache::getTemporalBasisCache() {
  return _temporalCache;
}

constFCPtr SpaceTimeBasisCache::getTensorBasisValues(TensorBasis<double>* tensorBasis,
                                                     int fieldIndex, int pointIndex,
                                                     constFCPtr spatialValues,
                                                     constFCPtr temporalValues,
                                                     Intrepid::EOperator spaceOpForSizing,
                                                     Intrepid::EOperator timeOpForSizing) const {
  Teuchos::Array<int> spatialValuesDim(spatialValues->rank()), temporalValuesDim(temporalValues->rank());
  spatialValues->dimensions(spatialValuesDim);
  temporalValues->dimensions(temporalValuesDim);
  
  int numSpaceFields = spatialValuesDim[fieldIndex];
  int numTimeFields = temporalValuesDim[fieldIndex];
  int numSpacePoints = spatialValuesDim[pointIndex];
  int numTimePoints = temporalValuesDim[pointIndex];
  
  Teuchos::Array<int> spaceTimeValuesDim = spatialValuesDim;
  spaceTimeValuesDim[fieldIndex] = numSpaceFields * numTimeFields;
  spaceTimeValuesDim[pointIndex] = numSpacePoints * numTimePoints;
  
  Intrepid::FieldContainer<double>* tensorValues = new Intrepid::FieldContainer<double>(spaceTimeValuesDim);
  
  vector< Intrepid::FieldContainer<double> > componentValues;
  // not sure there's a clean way to avoid copying the spatial/temporal values here, but it's only a temporary copy
  // (really, we probably shouldn't be storing the tensor product values, which will take significantly more memory)
  componentValues.push_back(*spatialValues);
  componentValues.push_back(*temporalValues);
  
  vector< Intrepid::EOperator > componentOps(2);
  componentOps[0] = spaceOpForSizing;
  componentOps[1] = timeOpForSizing;
  
  tensorBasis->getTensorValues(*tensorValues, componentValues, componentOps);
  
  return Teuchos::rcp( tensorValues );
}

constFCPtr SpaceTimeBasisCache::getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  if (_temporalCache == Teuchos::null) {
    // then we must be a side cache for a temporal side
    return this->BasisCache::getValues(basis, op, useCubPointsSideRefCell);
  }
  
  // For now, in this and the methods below, we do *not* store the tensor values.
  // Instead, we allow the tensor components to be stored by the component BasisCaches.
  
//  // determine key for value lookup
//  pair< Camellia::Basis<>*, Camellia::EOperator> key = make_pair(basis.get(), op);
//  
//  if (_knownValues.find(key) != _knownValues.end() ) {
//    return _knownValues[key];
//  }

  
  const int FIELD_INDEX = 0, POINT_INDEX = 1;
  
  // compute tensorial components:
  TensorBasis<double>* tensorBasis = dynamic_cast<TensorBasis<double>*>(basis.get());
  
  BasisPtr spatialBasis = tensorBasis->getSpatialBasis();
  BasisPtr temporalBasis = tensorBasis->getTemporalBasis();
  
  Intrepid::EOperator spaceOpForSizing = this->spaceOpForSizing(op), timeOpForSizing = this->timeOpForSizing(op);
  Camellia::EOperator spaceOp = this->spaceOp(op), timeOp = this->timeOp(op);
  
  constFCPtr spatialValues = _spatialCache->getValues(spatialBasis, spaceOp, useCubPointsSideRefCell);
  if (_temporalCache == Teuchos::null) {
    // then we must be a side cache for a temporal side
    return spatialValues;
  }
  
  constFCPtr temporalValues = _temporalCache->getValues(temporalBasis, timeOp, useCubPointsSideRefCell);
  
  return getTensorBasisValues(tensorBasis, FIELD_INDEX, POINT_INDEX, spatialValues, temporalValues, spaceOpForSizing, timeOpForSizing);
}

constFCPtr SpaceTimeBasisCache::getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  if (_temporalCache == Teuchos::null) {
    // then we must be a side cache for a temporal side
    return this->BasisCache::getTransformedValues(basis, op, useCubPointsSideRefCell);
  }
  
  const int FIELD_INDEX = 1, POINT_INDEX = 2;
  
  // compute tensorial components:
  TensorBasis<double>* tensorBasis = dynamic_cast<TensorBasis<double>*>(basis.get());
  
  if (tensorBasis == NULL) {
    cout << "basis must be a subclass of TensorBasis<double>!\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "basis must be a subclass of TensorBasis<double>!");
  }
  
  BasisPtr spatialBasis = tensorBasis->getSpatialBasis();
  BasisPtr temporalBasis = tensorBasis->getTemporalBasis();
  
  Intrepid::EOperator spaceOpForSizing = this->spaceOpForSizing(op), timeOpForSizing = this->timeOpForSizing(op);
  
  Camellia::EOperator spaceOp = this->spaceOp(op), timeOp = this->timeOp(op);
  
  constFCPtr spatialValues = _spatialCache->getTransformedValues(spatialBasis, spaceOp, useCubPointsSideRefCell);
  constFCPtr temporalValues = _temporalCache->getTransformedValues(temporalBasis, timeOp, useCubPointsSideRefCell);
  
  return getTensorBasisValues(tensorBasis, FIELD_INDEX, POINT_INDEX, spatialValues, temporalValues,
                              spaceOpForSizing, timeOpForSizing);
}

constFCPtr SpaceTimeBasisCache::getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  if (_temporalCache == Teuchos::null) {
    // then we must be a side cache for a temporal side
    return this->BasisCache::getTransformedWeightedValues(basis, op, useCubPointsSideRefCell);
  }
  
  const int FIELD_INDEX = 1, POINT_INDEX = 2;
  
  // compute tensorial components:
  TensorBasis<double>* tensorBasis = dynamic_cast<TensorBasis<double>*>(basis.get());
  
  BasisPtr spatialBasis = tensorBasis->getSpatialBasis();
  BasisPtr temporalBasis = tensorBasis->getTemporalBasis();
  
  Intrepid::EOperator spaceOpForSizing = this->spaceOpForSizing(op), timeOpForSizing = this->timeOpForSizing(op);
  Camellia::EOperator spaceOp = this->spaceOp(op), timeOp = this->timeOp(op);
  
  constFCPtr spatialValues = _spatialCache->getTransformedWeightedValues(spatialBasis, spaceOp, useCubPointsSideRefCell);
  constFCPtr temporalValues = _temporalCache->getTransformedWeightedValues(temporalBasis, timeOp, useCubPointsSideRefCell);
  
  return getTensorBasisValues(tensorBasis, FIELD_INDEX, POINT_INDEX, spatialValues, temporalValues, spaceOpForSizing, timeOpForSizing);
}

Camellia::EOperator SpaceTimeBasisCache::spaceOp(Camellia::EOperator op) {
  // the space op is just the op, unless it's a time-related op, in which case the space op is OP_VALUE
  switch (op) {
    case OP_T:
    case OP_DT:
      return OP_VALUE;
    default:
      return op;
      break;
  }
}

Intrepid::EOperator SpaceTimeBasisCache::spaceOpForSizing(Camellia::EOperator op) {
  switch (op) {
    case OP_GRAD:
      return OPERATOR_GRAD;
    case OP_DIV:
      return OPERATOR_DIV;
    case OP_CURL:
      return OPERATOR_CURL;
      break;
    default:
      return OPERATOR_VALUE;
      break;
  }
}

Camellia::EOperator SpaceTimeBasisCache::timeOp(Camellia::EOperator op) {
  // the time op is just OP_VALUE, unless it's a time-related op, in which case we need to take the space equivalent here
  switch (op) {
    case OP_T:
      return OP_X;
    case OP_DT:
      return OP_DX;
    default:
      return OP_VALUE;
      break;
  }
}

Intrepid::EOperator SpaceTimeBasisCache::timeOpForSizing(Camellia::EOperator op) {
  // we do not support any rank-increasing or rank-decreasing operations in time.
  return OPERATOR_VALUE;
}
