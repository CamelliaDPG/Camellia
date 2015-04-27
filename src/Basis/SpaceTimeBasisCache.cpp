//
//  SpaceTimeBasisCache.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/11/15.
//
//

#include "Intrepid_FunctionSpaceTools.hpp"

#include "SpaceTimeBasisCache.h"
#include "TensorBasis.h"
#include "TypeDefs.h"

using namespace Intrepid;
using namespace Camellia;

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
//  ElementTypePtr spaceElemType = Teuchos::rcp( new ElementType(spaceTimeElementType->trialOrderPtr,
//                                                               spaceTimeElementType->testOrderPtr, spaceTopo ) );
//  ElementTypePtr timeElemType = Teuchos::rcp( new ElementType( spaceTimeElementType->trialOrderPtr,
//                                                              spaceTimeElementType->testOrderPtr, timeTopo ) );
  
  // for the moment, we use same cubature degree in space as time.  By making this sharper, could reduce expense
  
  int cubDegreeSpaceTime;
  cubatureDegreeForElementType(spaceTimeElementType, testVsTest, cubDegreeSpaceTime);
  cubDegreeSpaceTime += cubatureDegreeEnrichment;
  int cubDegreeSpace = cubDegreeSpaceTime;
  int cubDegreeTime = cubDegreeSpaceTime;
  
  bool tensorTopologyMeansSpaceTime = false; // if space topology is tensor product, don't interpret as space-time
  _spatialCache = Teuchos::rcp( new BasisCache(physicalNodesSpatial, spaceTopo, cubDegreeSpace, tensorTopologyMeansSpaceTime) );
  _temporalCache = Teuchos::rcp( new BasisCache(physicalNodesTemporal, timeTopo, cubDegreeTime, tensorTopologyMeansSpaceTime) );
//  _spatialCache = Teuchos::rcp( new BasisCache(spaceElemType, Teuchos::null, testVsTest, cubatureDegreeEnrichment,
//                                               tensorTopologyMeansSpaceTime) );
//  _temporalCache = Teuchos::rcp( new BasisCache(timeElemType, Teuchos::null, testVsTest, cubatureDegreeEnrichment,
//                                                tensorTopologyMeansSpaceTime) );
  
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
                                         CellTopoPtr cellTopo, int cubDegreeSpaceTime)
: BasisCache(physicalCellNodes,cellTopo,cubDegreeSpaceTime,false,true) { // false: don't create side caches during construction; true: tensor product topology (which we should have here) --> space-time
  bool createSideCache = true;
  
  int sideCount = cellTopology()->getSideCount();
  
  if (physicalNodesSpatial.dimension(0) != physicalNodesTemporal.dimension(0)) {
    cout << "physicalNodesSpatial must have the same # of cells as physicalNodesTemporal\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "physicalNodesSpatial must have the same # of cells as physicalNodesTemporal");
  }
  
  if (physicalCellNodes.dimension(0) != physicalNodesTemporal.dimension(0)) {
    cout << "physicalCellNodes must have the same # of cells as physicalNodesTemporal\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "physicalCellNodes must have the same # of cells as physicalNodesTemporal");
  }
  
  if (physicalCellNodes.dimension(0) <= 0) {
    cout << "physicalCellNodes must have a positive # of cells\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "physicalCellNodes must have a positive # of cells");
  }
  
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
  _spatialCache = Teuchos::rcp( new BasisCache(physicalNodesSpatial, spaceTopo, cubDegreeSpaceTime, tensorTopologyMeansSpaceTime) );
  _temporalCache = Teuchos::rcp( new BasisCache(physicalNodesTemporal, timeTopo, cubDegreeSpaceTime, tensorTopologyMeansSpaceTime) );
  
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
    // copy the temporalCacheVolume so that we can set a single reference cell point:
    _temporalCache = Teuchos::rcp( new BasisCache(temporalCacheVolume->getPhysicalCellNodes(), temporalCacheVolume->cellTopology(),
                                                  temporalCacheVolume->cubatureDegree(), false) );
    FieldContainer<double> temporalNode(1,1);
    temporalNode(0,0) = getTemporalNodeCoordinateRefSpace();
    FieldContainer<double> temporalWeight(1);
    temporalWeight(0) = 1.0;
    _temporalCache->setRefCellPoints(temporalNode, temporalWeight);
    
    bool createSideCache = true;
    vector<GlobalIndexType> cellIDs; //empty
    _temporalCache->setPhysicalCellNodes(temporalCacheVolume->getPhysicalCellNodes(), cellIDs, createSideCache);
  }
}

void SpaceTimeBasisCache::createSideCaches() {
  _basisCacheSides.clear();
  int numSides = this->cellTopology()->getSideCount();
  
  int cubatureDegree = this->cubatureDegree();
  int maxTrialDegree = cubatureDegree / 2; // a bit hackish -- right now, just ensuring total cubature degree agrees with the spatial/temporal caches for volume.
  int maxTestDegree = cubatureDegree - maxTrialDegree;
  
  for (int sideOrdinal=0; sideOrdinal<numSides; sideOrdinal++) {
//    BasisPtr maxDegreeBasisOnSide = _maxDegreeBasisForSide[sideOrdinal];
//    
//    int maxTrialDegreeOnSide = _maxTrialDegree;
//    if (maxDegreeBasisOnSide.get() != NULL) {
//      maxTrialDegreeOnSide = maxDegreeBasisOnSide->getDegree();
//    }
//    
    Teuchos::RCP<SpaceTimeBasisCache> thisPtr = Teuchos::rcp( this, false ); // presumption is that side cache doesn't outlive volume...
    SpaceTimeBasisCache* spaceTimeSideCache = new SpaceTimeBasisCache(sideOrdinal, thisPtr, maxTrialDegree, maxTestDegree);
    BasisCachePtr sideCache = Teuchos::rcp( spaceTimeSideCache );
    _basisCacheSides.push_back(sideCache);
  }
}

void SpaceTimeBasisCache::getSpaceTimeCubatureDegrees(ElementTypePtr spaceTimeType, int &spaceDegree, int &timeDegree) {
  DofOrderingPtr spaceTimeTrial = spaceTimeType->trialOrderPtr;
  DofOrderingPtr spaceTimeTest = spaceTimeType->testOrderPtr;
  
  set<int> trialIDs = spaceTimeTrial->getVarIDs();
  
  int maxSpaceTrialDegree = 0, maxTimeTrialDegree = 0;
  
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "implementation incomplete");
  
  // TODO: finish this method
//  for (set<int>::iterator trialIt = trialIDs.begin(); trialIt != trialIDs.end(); trialIt++) {
//    int trialID = *trialIt;
//    const vector<int> *sides = &spaceTimeTrial->getSidesForVarID(trialID);
//    for (vector<int>::const_iterator sideIt = sides->begin(); sideIt != sides->end(); sideIt++) {
//      int side = *sideIt;
//      BasisPtr spaceTimeBasis = spaceTimeTrial->getBasis(trialID, side);
//    }
//    get the sides for each
//    when it's a volume test var, decompose the basis into space and time components
//    same thing when it's a variable that lives on a space x time side
//    when it's a variable that lives on the temporal sides, add its basis there to the spatial ordering
//  }
//  
//  then, do the same thing for trial IDs
  
//  void addEntry(int varID, BasisPtr basis, int basisRank, int sideIndex = 0);
}

BasisCachePtr SpaceTimeBasisCache::getSpatialBasisCache() {
  return _spatialCache;
}

BasisCachePtr SpaceTimeBasisCache::getTemporalBasisCache() {
  return _temporalCache;
}

double SpaceTimeBasisCache::getTemporalNodeCoordinateRefSpace() {
  TEUCHOS_TEST_FOR_EXCEPTION(!this->isSideCache(), std::invalid_argument, "getTemporalNodeCoordinateRefSpace() only supported for side caches for temporal sides");
  TEUCHOS_TEST_FOR_EXCEPTION(this->cellTopology()->sideIsSpatial(this->getSideIndex()), std::invalid_argument,
                             "getTemporalNodeCoordinateRefSpace() only supported for side caches for temporal sides");
  
  int temporalNodeOrdinal = this->cellTopology()->getTemporalComponentSideOrdinal(this->getSideIndex());
  FieldContainer<double> lineRefNodes(2,1);
  CamelliaCellTools::refCellNodesForTopology(lineRefNodes, CellTopology::line());
  return lineRefNodes(temporalNodeOrdinal,0);
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

void SpaceTimeBasisCache::getTensorialComponentPoints(CellTopoPtr spaceTimeTopo,
                                                     const FieldContainer<double> &tensorPoints,
                                                     FieldContainer<double> &spatialPoints,
                                                     FieldContainer<double> &temporalPoints) {
  
  // note that this initial implementation is hackish; it's merely meant to be a placeholder
  // until we can develop an abstraction that won't involve us *taking* the tensor product of
  // points in the first place (which costs memory and computation, and prevents us from taking
  // advantage of the tensor product structure to avoid further costs).
  
  // note also that we do assume that the tensorPoints are in a tensor product structure; we don't check
  // that all the space points that should agree do agree, e.g.
  
  // tensor points should be shaped (P,D)
  if (tensorPoints.rank() != 2) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensorPoints must have shape (P,D) or (C,P,D)");
  }
  if (tensorPoints.dimension(1) != spaceTimeTopo->getDimension()) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "tensorPoints must have shape (P,D) or (C,P,D)");
  }
  CellTopoPtr spaceTopo = spaceTimeTopo->getTensorialComponent();
  CellTopoPtr timeTopo = CellTopology::line();
  
  int numSpaceTimePoints = tensorPoints.dimension(0);
  
  int d_time = spaceTopo->getDimension();
  
  set<double> timePoints;
  for (int pointOrdinal=0; pointOrdinal<numSpaceTimePoints; pointOrdinal++) {
    // round to 6 decimal places (we're just using this to determine uniqueness)
    double timePoint = round(1.0e6 * tensorPoints(pointOrdinal,d_time)) / 1.0e6;
    timePoints.insert(timePoint);
  }
  
  int numTimePoints = timePoints.size(); // number of unique points in time
  
  int numSpacePoints = numSpaceTimePoints / numTimePoints;
  
  if (numSpacePoints * numTimePoints != numSpaceTimePoints) {
    cout << "tensorPoints:\n" << tensorPoints;
    Camellia::print("timePoints", timePoints);
    cout << "ERROR: numSpaceTimePoints is not evenly divisible by the number of distinct temporal values.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numSpaceTimePoints is not evenly divisible by the number of distinct temporal values");
  }
  
  if (spaceTopo->getDimension()==0) {
    spatialPoints.resize(0);
    temporalPoints = tensorPoints;
    return;
  }
  
  spatialPoints.resize(numSpacePoints,spaceTopo->getDimension());
  temporalPoints.resize(numTimePoints,timeTopo->getDimension());
  
  for (int spacePointOrdinal=0; spacePointOrdinal < numSpacePoints; spacePointOrdinal++) {
    int timePointOrdinal = 0;
    int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numSpacePoints);
    for (int d=0; d<spaceTopo->getDimension(); d++) {
      spatialPoints(spacePointOrdinal,d) = tensorPoints(spaceTimePointOrdinal,d);
    }
  }
  
  for (int timePointOrdinal=0; timePointOrdinal < numTimePoints; timePointOrdinal++) {
    int spacePointOrdinal = 0;
    int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numSpacePoints);
    temporalPoints(timePointOrdinal,0) = tensorPoints(spaceTimePointOrdinal,d_time);
  }
}

void SpaceTimeBasisCache::getReferencePointComponents(const FieldContainer<double> &tensorPoints, FieldContainer<double> &spacePoints,
                                                      FieldContainer<double> &timePoints) {
  // this->cellTopology() refers to volume topology; set thisTopology to be side topology if this is a side cache
  CellTopoPtr thisTopology = this->isSideCache() ? this->cellTopology()->getSide(this->getSideIndex()) : this->cellTopology();
  
  if (!this->isSideCache())
  {
    // volume cache
    getTensorialComponentPoints(thisTopology, tensorPoints, spacePoints, timePoints);
  }
  else if ( this->cellTopology()->sideIsSpatial(this->getSideIndex()))
  {
    // side cache with spatial side
    TEUCHOS_TEST_FOR_EXCEPTION(this->cellTopology()->getTensorialDegree() == 0, std::invalid_argument,
                               "Spatial sides in SpaceTimeBasisCache should have tensorial degree > 0.");
    getTensorialComponentPoints(thisTopology, tensorPoints, spacePoints, timePoints);
  }
  else
  {
    // side is temporal
    // tensorial degree = 0 : spaceRefCellPoints = pointsRefCell, and timeRefCellPoints is a single point corresponding to the temporal node
    spacePoints = tensorPoints;
    timePoints.resize(1,1);
    timePoints(0,0) = getTemporalNodeCoordinateRefSpace();
  }
}

void SpaceTimeBasisCache::setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell) {
  FieldContainer<double> spaceRefCellPoints, timeRefCellPoints;
  
  getReferencePointComponents(pointsRefCell, spaceRefCellPoints, timeRefCellPoints);
  
  if (spaceRefCellPoints.size() > 0) { // if size is zero, it's a node, and we let _spatialCache do what it likes for setting up ref cell points.
    _spatialCache->setRefCellPoints(spaceRefCellPoints);
  }
  _temporalCache->setRefCellPoints(timeRefCellPoints);
  
  this->BasisCache::setRefCellPoints(pointsRefCell);
}

void SpaceTimeBasisCache::setRefCellPoints(const Intrepid::FieldContainer<double> &pointsRefCell,
                                           const Intrepid::FieldContainer<double> &cubatureWeights) {
  FieldContainer<double> spaceRefCellPoints, timeRefCellPoints;
  
  getReferencePointComponents(pointsRefCell, spaceRefCellPoints, timeRefCellPoints);
  
  if (spaceRefCellPoints.size() == 0) {
    // then the cubature weights belong just to the temporal points:
    _temporalCache->setRefCellPoints(timeRefCellPoints, cubatureWeights);
    this->BasisCache::setRefCellPoints(pointsRefCell, cubatureWeights);
  } else {
    int numSpacePoints = spaceRefCellPoints.dimension(0);
    int numTimePoints = timeRefCellPoints.dimension(0);
    FieldContainer<double> cubatureWeightsSpace(numSpacePoints), cubatureWeightsTime(numTimePoints);

    if (cubatureWeights.size() == 0) {
      cubatureWeightsSpace.resize(0);
      cubatureWeightsTime.resize(0);
    } else {
      // tease out the weights (note we assume here that they're nonzero everywhere)
      // these are in an outer product structure
      // we additionally constrain the temporal weights to sum to 2.0
      double timeSumAtSpace0 = 0;
      for (int timePointOrdinal=0; timePointOrdinal < numTimePoints; timePointOrdinal++) {
        int spacePointOrdinal = 0;
        int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numSpacePoints);
        timeSumAtSpace0 += cubatureWeights(spaceTimePointOrdinal);
      }

      double scalingFactor = 2.0 / timeSumAtSpace0;
      for (int timePointOrdinal=0; timePointOrdinal < numTimePoints; timePointOrdinal++) {
        int spacePointOrdinal = 0;
        int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numSpacePoints);
        cubatureWeightsTime(timePointOrdinal) = cubatureWeights(spaceTimePointOrdinal) * scalingFactor;
      }

      // the inverse of the scalingFactor for temporalWeights is just what we take the "true" values of the first spatial cubature
      // weight to be.  Therefore we set the scalingFactor for the spatialWeights thus:
      scalingFactor = 1.0 / (cubatureWeights(TENSOR_POINT_ORDINAL(0, 0, numSpacePoints)) / scalingFactor);
      for (int spacePointOrdinal=0; spacePointOrdinal < numSpacePoints; spacePointOrdinal++) {
        int timePointOrdinal = 0;
        int spaceTimePointOrdinal = TENSOR_POINT_ORDINAL(spacePointOrdinal, timePointOrdinal, numSpacePoints);
        cubatureWeightsSpace(spacePointOrdinal) = cubatureWeights(spaceTimePointOrdinal) * scalingFactor;
      }
    }
    
    _spatialCache->setRefCellPoints(spaceRefCellPoints, cubatureWeightsSpace);
    _temporalCache->setRefCellPoints(timeRefCellPoints, cubatureWeightsTime);
  }
  
  this->BasisCache::setRefCellPoints(pointsRefCell, cubatureWeights);
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
  if ( isSideCache() && !cellTopology()->sideIsSpatial(getSideIndex()) ) {
    // in this case, two possibilities:
    // 1. basis is a volume basis, in which case we want to handle things in terms of tensor product, as usual
    // 2. basis is a trace basis, defined on the spatial topology only
    // In the second case, we want to use the spatial basis cache, and just pass in the basis.  In this case useCubPointsSideRefCell
    // ought to be false: this is used when we evaluate a volume basis on the side.
    if (_spatialCache->cellTopology()->getKey() == basis->domainTopology()->getKey()) {
      TEUCHOS_TEST_FOR_EXCEPTION(useCubPointsSideRefCell == true, std::invalid_argument, "useCubPointsSideRefCell should not be true for a trace basis...");
      return _spatialCache->getTransformedValues(basis, op);
    }
  }
  
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
  
  constFCPtr spatialValues, temporalValues;
  if (useCubPointsSideRefCell && !cellTopology()->sideIsSpatial(getSideIndex())) {
    // then _spatialCache is a volume cache already, so we shouldn't tell it to use volume points...
    spatialValues = _spatialCache->getTransformedValues(spatialBasis, spaceOp, false);
  } else {
    spatialValues = _spatialCache->getTransformedValues(spatialBasis, spaceOp, useCubPointsSideRefCell);
  }
  
  // _temporalCache is always a volume cache
  temporalValues = _temporalCache->getTransformedValues(temporalBasis, timeOp, false);
  return getTensorBasisValues(tensorBasis, FIELD_INDEX, POINT_INDEX, spatialValues, temporalValues,
                              spaceOpForSizing, timeOpForSizing);
}

constFCPtr SpaceTimeBasisCache::getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  constFCPtr unWeightedValues = getTransformedValues(basis,op, useCubPointsSideRefCell);
  Teuchos::Array<int> dimensions;
  unWeightedValues->dimensions(dimensions);
  Teuchos::RCP< FieldContainer<double> > weightedValues = Teuchos::rcp( new FieldContainer<double>(dimensions) );
  FunctionSpaceTools::multiplyMeasure<double>(*weightedValues, this->getWeightedMeasures(), *unWeightedValues);
  return weightedValues;
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
  // idea here is to return an Intrepid::EOperator that has the same effect on FieldContainer sizing (really, the
  // rank of the value) as the Camellia::EOperator provided.
  switch (op) {
    // rank-increasing:
    case OP_GRAD:
      return Intrepid::OPERATOR_GRAD;
    // rank-decreasing:
    case OP_DIV:
    case OP_X:
    case OP_Y:
    case OP_Z:
      return Intrepid::OPERATOR_DIV;
    // rank-switching (0 to 1, 1 to 0) in 2D, rank-preserving in 3D:
    case OP_CURL:
      return Intrepid::OPERATOR_CURL;
      break;
    // rank-preserving:
    default:
      return Intrepid::OPERATOR_VALUE;
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
  return Intrepid::OPERATOR_VALUE;
}
