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

constFCPtr SpaceTimeBasisCache::getValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  // determine key for value lookup
  pair< Camellia::Basis<>*, Camellia::EOperator> key = make_pair(basis.get(), op);
  
  if (_knownValues.find(key) != _knownValues.end() ) {
    return _knownValues[key];
  }
  
  // compute tensorial components:
  TensorBasis<double>* tensorBasis = dynamic_cast<TensorBasis<double>*>(basis.get());
  
  BasisPtr spatialBasis = tensorBasis->getSpatialBasis();
  BasisPtr temporalBasis = tensorBasis->getTemporalBasis();
  
  Intrepid::EOperator spaceOp = this->spaceOp(op), timeOp = this->timeOp(op);
  
  constFCPtr spatialValues = _spatialCache->getValues(spatialBasis, (Camellia::EOperator)spaceOp, useCubPointsSideRefCell);
  constFCPtr temporalValues = _temporalCache->getValues(temporalBasis, (Camellia::EOperator)timeOp, useCubPointsSideRefCell);
  
  Intrepid::FieldContainer<double> tensorValues;
  
  vector< Intrepid::FieldContainer<double> > componentValues;
  // not sure there's a clean way to avoid copying the spatial/temporal values here, but it's only a temporary copy
  // (really, we probably shouldn't be storing the tensor product values, which will take significantly more memory)
  componentValues.push_back(*spatialValues);
  componentValues.push_back(*temporalValues);
  
  vector< Intrepid::EOperator > componentOps(2);
  componentOps[0] = spaceOp;
  componentOps[1] = timeOp;
  
  tensorBasis->getTensorValues(tensorValues, componentValues, componentOps);
  
}

constFCPtr SpaceTimeBasisCache::getTransformedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  
}

constFCPtr SpaceTimeBasisCache::getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, bool useCubPointsSideRefCell) {
  
}

// side variants:
constFCPtr SpaceTimeBasisCache::getValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell) {
  
}

constFCPtr SpaceTimeBasisCache::getTransformedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell) {
  
}

constFCPtr SpaceTimeBasisCache::getTransformedWeightedValues(BasisPtr basis, Camellia::EOperator op, int sideOrdinal, bool useCubPointsSideRefCell) {
  
}