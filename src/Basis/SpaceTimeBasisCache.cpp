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
                                         FieldContainer<double> &physicalNodesSpatial,
                                         FieldContainer<double> &physicalNodesTemporal,
                                         const std::vector<GlobalIndexType> &cellIDs,
                                         bool testVsTest, int cubatureDegreeEnrichment) : BasisCache(spaceTimeElementType, spaceTimeMesh, testVsTest, cubatureDegreeEnrichment, true) {
  bool createSideCache = true;
  
  int cellCount = cellIDs.size();
  int sideCount = cellTopology()->getSideCount();
  int vertexCount = cellTopology()->getVertexCount();
  int spaceTimeDim = cellTopology()->getDimension();
  
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
  ElementTypePtr spaceElemType = Teuchos::rcp( new ElementType( spaceTimeElementType->trialOrderPtr, spaceTimeElementType->testOrderPtr, spaceTopo ) );
  ElementTypePtr timeElemType = Teuchos::rcp( new ElementType( spaceTimeElementType->trialOrderPtr, spaceTimeElementType->testOrderPtr, timeTopo ) );
  
  bool tensorTopologyMeansSpaceTime = false; // if space topology is tensor product, don't interpret as space-time
  _spatialCache = Teuchos::rcp( new BasisCache(spaceElemType, Teuchos::null, testVsTest, cubatureDegreeEnrichment, tensorTopologyMeansSpaceTime) );
  _temporalCache = Teuchos::rcp( new BasisCache(timeElemType, Teuchos::null, testVsTest, cubatureDegreeEnrichment, tensorTopologyMeansSpaceTime) );
  
  _spatialCache->setPhysicalCellNodes(physicalNodesSpatial, cellIDs, createSideCache);
  _temporalCache->setPhysicalCellNodes(physicalNodesTemporal, cellIDs, createSideCache);
  
  // it may be that ultimately we can get away without doing any space-time construction at all;
  // everything can be done in terms of the spatial BasisCache and the temporal.  For now, we
  // construct space-time physical cell nodes, etc.
  
  int spaceVertexCount = physicalNodesSpatial.dimension(1);
  int spaceDim = physicalNodesSpatial.dimension(2);
  
  int timeVertexCount = physicalNodesTemporal.dimension(1);
  int timeDim = physicalNodesTemporal.dimension(2);
  
  FieldContainer<double> spaceTimePhysicalNodes(cellCount,vertexCount,spaceTimeDim);
  
  // initialize the space-time nodes:
  Teuchos::Array<int> spaceNodeDim(2), timeNodeDim(2), spaceTimeNodeDim(2);
  spaceNodeDim[0] = spaceVertexCount;
  spaceNodeDim[1] = spaceDim;
  timeNodeDim[0] = timeVertexCount;
  timeNodeDim[1] = timeDim;
  spaceTimeNodeDim[0] = vertexCount;
  spaceTimeNodeDim[1] = spaceTimeDim;
  
  Teuchos::Array<int> spaceEnumeration(3), timeEnumeration(3), spaceTimeEnumeration(3);
  
  for (int cellOrdinal=0; cellOrdinal<cellCount; cellOrdinal++) {
    FieldContainer<double> spaceNodes(spaceNodeDim, &physicalNodesSpatial(cellOrdinal,0,0));
    FieldContainer<double> timeNodes(timeNodeDim, &physicalNodesTemporal(cellOrdinal,0,0));
    vector< FieldContainer<double> > componentNodes(2);
    componentNodes[0] = spaceNodes;
    componentNodes[1] = timeNodes;
    FieldContainer<double> spaceTimeNodes(spaceTimeNodeDim, &spaceTimePhysicalNodes(cellOrdinal,0,0));
    cellTopology()->initializeNodes(componentNodes, spaceTimeNodes);
  }
  
  setPhysicalCellNodes(spaceTimePhysicalNodes, cellIDs, createSideCache);

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

// side constructor
SpaceTimeBasisCache::SpaceTimeBasisCache(int sideIndex, BasisCachePtr volumeCache, int trialDegree, int testDegree
                                         ) : BasisCache(sideIndex, volumeCache, trialDegree, testDegree, (BasisPtr) Teuchos::null) {}

void SpaceTimeBasisCache::createSideCaches() {
  _basisCacheSides.clear();
  int numSides = this->cellTopology()->getSideCount();
  
  for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
    BasisPtr maxDegreeBasisOnSide = _maxDegreeBasisForSide[sideOrdinal];
    
    int maxTrialDegreeOnSide = _maxTrialDegree;
    if (maxDegreeBasisOnSide.get() != NULL) {
      maxTrialDegreeOnSide = maxDegreeBasisOnSide->getDegree();
    }
    
    BasisCachePtr thisPtr = Teuchos::rcp( this, false ); // presumption is that side cache doesn't outlive volume...
    BasisCachePtr sideCache = Teuchos::rcp( new SpaceTimeBasisCache(sideOrdinal, thisPtr, maxTrialDegreeOnSide, _maxTestDegree));
    _basisCacheSides.push_back(sideCache);
  }
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
  
  constFCPtr spatialValues = _spatialCache->getValues(spatialBasis, (Camellia::EOperator)spaceOp, useCubPointsSideRefCell);
  constFCPtr temporalValues = _temporalCache->getValues(temporalBasis, (Camellia::EOperator)timeOp, useCubPointsSideRefCell);
  
  return getTensorBasisValues(tensorBasis, FIELD_INDEX, POINT_INDEX, spatialValues, temporalValues, spaceOpForSizing, timeOpForSizing);
}

constFCPtr SpaceTimeBasisCache::getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  const int FIELD_INDEX = 1, POINT_INDEX = 2;
  
  // compute tensorial components:
  TensorBasis<double>* tensorBasis = dynamic_cast<TensorBasis<double>*>(basis.get());
  
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
