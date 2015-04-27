//
//  SpaceTimeBasisCacheTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 3/13/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisFactory.h"
#include "CellTopology.h"
#include "SpaceTimeBasisCache.h"
#include "PoissonFormulation.h"
#include "TensorBasis.h"

using namespace Camellia;
using namespace Intrepid;

namespace {
  typedef Intrepid::FieldContainer<double> FC;
  
  void addCellDimension(FC &fc)
  {
    Teuchos::Array<int> dim(fc.rank());
    fc.dimensions(dim);
    
    dim.insert(dim.begin(), 1);
    fc.resize(dim);
  }
  
  void stripCellDimension(FC &fc)
  {
    Teuchos::Array<int> dim(fc.rank());
    fc.dimensions(dim);
    
    dim.erase(dim.begin());
    fc.resize(dim);
  }
  
  FieldContainer<double> getScaledTranslatedRefNodes(CellTopoPtr topo, double nodeScaling, double nodeTranslation)
  {
    FieldContainer<double> nodes(topo->getNodeCount(),topo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(nodes, topo);
    for (int nodeOrdinal=0; nodeOrdinal<topo->getNodeCount(); nodeOrdinal++) {
      for (int d=0; d<topo->getDimension(); d++)
      {
        nodes(nodeOrdinal,d) *= nodeScaling;
        nodes(nodeOrdinal,d) += nodeTranslation;
      }
    }
    return nodes;
  }

  MeshPtr getSpaceTimeMesh(CellTopoPtr spaceTopo, int H1Order=2, double refCellExpansionFactor=2.0, double refCellTranslation=0.0, double t0=0, double t1=1) {
    CellTopoPtr timeTopo = CellTopology::line();
    int tensorialDegree = 1;
    CellTopoPtr tensorTopo = CellTopology::cellTopology(spaceTopo, tensorialDegree);
    
    FC spaceNodes = getScaledTranslatedRefNodes(spaceTopo, refCellExpansionFactor, refCellTranslation);
    FC timeNodes(timeTopo->getNodeCount(), timeTopo->getDimension());
    
    // time will go from t0 to t1:
    timeNodes(0,0) = t0;
    timeNodes(1,0) = t1;
    
    vector<FC> tensorComponentNodes;
    tensorComponentNodes.push_back(spaceNodes);
    tensorComponentNodes.push_back(timeNodes);
    
    FC tensorNodesFC(spaceNodes.dimension(0) * timeNodes.dimension(0), spaceNodes.dimension(1) + timeNodes.dimension(1));
    tensorTopo->initializeNodes(tensorComponentNodes, tensorNodesFC);
    
    vector<vector<double> > tensorNodes;
    CamelliaCellTools::pointsVectorFromFC(tensorNodes, tensorNodesFC);
    
    MeshTopologyPtr meshTopo = Teuchos::rcp( new MeshTopology(tensorTopo->getDimension()) );
    
    meshTopo->addCell(tensorTopo, tensorNodes);
    
    PoissonFormulation form(spaceTopo->getDimension(), false); // arbitrary; we don't actually use the BF in any meaningful way
    
    int delta_k = 1;
    MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, form.bf(), H1Order, delta_k) );
    
    return mesh;
  }
  
  void testBasisTransformation(BasisPtr spaceTimeBasis, Camellia::EOperator op,
                               CellTopoPtr spaceTopo, Teuchos::FancyOStream &out, bool &success)
  {
    /* three things to try:
     1. volume values
     2. spatial sides
     3. temporal sides
     */
    
    Camellia::EOperator spaceOp, timeOp;
    
    if (op == OP_DT)
    {
      spaceOp = OP_VALUE;
      timeOp = OP_DX;
    }
    else
    {
     // every other op that interests us will split like this:
      spaceOp = op;
      timeOp = OP_VALUE;
    }
    Intrepid::EOperator spaceOpForSizing;
    switch (spaceOp) {
        // rank-increasing:
      case OP_GRAD:
        spaceOpForSizing = Intrepid::OPERATOR_GRAD;
        // rank-decreasing:
      case OP_DIV:
      case OP_X:
      case OP_Y:
      case OP_Z:
        spaceOpForSizing = Intrepid::OPERATOR_DIV;
        // rank-switching (0 to 1, 1 to 0) in 2D, rank-preserving in 3D:
      case OP_CURL:
        spaceOpForSizing = Intrepid::OPERATOR_CURL;
        break;
        // rank-preserving:
      default:
        spaceOpForSizing = Intrepid::OPERATOR_VALUE;
        break;
    }
    Intrepid::EOperator timeOpForSizing = Intrepid::OPERATOR_VALUE;
    
    bool createSideCaches = true;
    
    typedef Camellia::TensorBasis<double, FieldContainer<double> > TensorBasis;
    TensorBasis* tensorBasis = dynamic_cast<TensorBasis*>(spaceTimeBasis.get());
    
    BasisPtr spatialBasis = tensorBasis->getSpatialBasis();
    BasisPtr temporalBasis = tensorBasis->getTemporalBasis();

    vector<FC> tensorComponentNodes;
    int cubatureDegree = tensorBasis->getDegree();
    double scaling = 0.5;
    double translation = 0.5;
    FC spaceNodes = getScaledTranslatedRefNodes(spaceTopo, scaling, translation);
    tensorComponentNodes.push_back(spaceNodes);
    addCellDimension(spaceNodes);
    BasisCachePtr spatialBasisCache = BasisCache::basisCacheForCellTopology(spaceTopo, cubatureDegree, spaceNodes, createSideCaches);
    
    double t0 = 0.0, t1 = 0.5; // 0.25 scaling in time --> different from space (better test)
    FC timeNodes(2,1);
    timeNodes(0,0) = t0;
    timeNodes(1,0) = t1;
    tensorComponentNodes.push_back(timeNodes);
    addCellDimension(timeNodes);
    CellTopoPtr timeTopo = CellTopology::line();
    BasisCachePtr temporalBasisCache = BasisCache::basisCacheForCellTopology(timeTopo, cubatureDegree, timeNodes, createSideCaches);
    
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
    FC spaceTimeNodes(spaceTimeTopo->getNodeCount(), spaceTimeTopo->getDimension());
    spaceTimeTopo->initializeNodes(tensorComponentNodes, spaceTimeNodes);
    addCellDimension(spaceTimeNodes);
    BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForCellTopology(spaceTimeTopo, cubatureDegree,
                                                                              spaceTimeNodes, createSideCaches);
    
    // Volume test:
    FC spaceValues = *spatialBasisCache->getTransformedValues(spatialBasis, spaceOp);
    FC timeValues = *temporalBasisCache->getTransformedValues(temporalBasis, timeOp);
    FC spaceTimeValues = *spaceTimeBasisCache->getTransformedValues(spaceTimeBasis, op);
    
    // strip off cell dimensions
    stripCellDimension(spaceValues);
    stripCellDimension(timeValues);
    stripCellDimension(spaceTimeValues);
    
    FC expectedValues = spaceTimeValues;
    expectedValues.initialize(0.0); // zero out
    vector<FC> componentValues = {spaceValues, timeValues};
    vector<Intrepid::EOperator> componentOps = {spaceOpForSizing, timeOpForSizing};
    tensorBasis->getTensorValues(expectedValues, componentValues, componentOps);

    FC lineRefNodes(2,1);
    CamelliaCellTools::refCellNodesForTopology(lineRefNodes, CellTopology::line());
    
    double tol = 1e-14;
    TEST_COMPARE_FLOATING_ARRAYS(expectedValues, spaceTimeValues, tol);

    for (int sideOrdinal = 0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
    {
      BasisCachePtr spaceTimeSideCache = spaceTimeBasisCache->getSideBasisCache(sideOrdinal);
      if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
      {
        // we get the time values just as usual for spatial sides:
        FC timeValues = *temporalBasisCache->getTransformedValues(temporalBasis, timeOp);
        
        // we get the side values from the spatial side cache:
        int spatialSideOrdinal = spaceTimeTopo->getSpatialComponentSideOrdinal(sideOrdinal);
        BasisCachePtr spatialSideCache = spatialBasisCache->getSideBasisCache(spatialSideOrdinal);
        bool useVolumeCoords = true; // because spatialBasis is defined on the spatial volume, and we're restricting to the side
        FC spaceValues = *spatialSideCache->getTransformedValues(spatialBasis, spaceOp, useVolumeCoords);
        
        FC spaceTimeValues = *spaceTimeSideCache->getTransformedValues(spaceTimeBasis, op, useVolumeCoords);
        
        // strip off cell dimensions
        stripCellDimension(spaceValues);
        stripCellDimension(timeValues);
        stripCellDimension(spaceTimeValues);
        
        expectedValues = spaceTimeValues;
        expectedValues.initialize(0.0); // zero out
        componentValues = {spaceValues, timeValues};
        tensorBasis->getTensorValues(expectedValues, componentValues, componentOps);
        TEST_COMPARE_FLOATING_ARRAYS(expectedValues, spaceTimeValues, tol);
      }
      else
      {
         // we get the spatial values just as usual for the temporal sides:
        FC spaceValues = *spatialBasisCache->getTransformedValues(spatialBasis, spaceOp);
        BasisCachePtr onePointTemporalBasisCache = BasisCache::basisCacheForCellTopology(timeTopo, cubatureDegree,
                                                                                         timeNodes, createSideCaches);
        unsigned temporalNodeOrdinal = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
        FC temporalRefPoint(1,1);
        temporalRefPoint(0,0) = lineRefNodes(temporalNodeOrdinal,0);
        FC temporalCubatureWeight(1);
        temporalCubatureWeight(0) = 1.0;
        onePointTemporalBasisCache->setRefCellPoints(temporalRefPoint, temporalCubatureWeight);
        FC timeValues = *onePointTemporalBasisCache->getTransformedValues(temporalBasis, timeOp);
        
//        { // DEBUGGING
//          if (timeOp==OP_DX) {
//            FC refTimeValues = *onePointTemporalBasisCache->getValues(temporalBasis, timeOp);
//            cout << "OP_DT, reference timeValues:\n" << refTimeValues;
//            cout << "OP_DT, transformed timeValues:\n" << timeValues;
//          }
//        }
        
        bool useVolumeCoords = true; // because spaceTimeBasis is defined on the volume, and we're restricting to the side
        FC spaceTimeValues = *spaceTimeSideCache->getTransformedValues(spaceTimeBasis, op, useVolumeCoords);
        
        stripCellDimension(spaceValues);
        stripCellDimension(timeValues);
        stripCellDimension(spaceTimeValues);
        
        expectedValues = spaceTimeValues;
        expectedValues.initialize(0.0); // zero out
        componentValues = {spaceValues, timeValues};
        tensorBasis->getTensorValues(expectedValues, componentValues, componentOps);
        TEST_COMPARE_FLOATING_ARRAYS(expectedValues, spaceTimeValues, tol);
      }
    }
  }
  
  void testSpaceTimeSideMeasure(CellTopoPtr spaceTopo, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2;
    int delta_k = 1;
    bool createSideCaches = true;
    double refCellExpansionFactor = 0.5;
    double refCellTranslation = 0.5;
    BasisCachePtr spatialBasisCache = BasisCache::basisCacheForReferenceCell(spaceTopo, H1Order*2+delta_k, createSideCaches);
    FC spaceNodes = getScaledTranslatedRefNodes(spaceTopo, refCellExpansionFactor, refCellTranslation);
    spaceNodes.resize(1,spaceNodes.dimension(0),spaceNodes.dimension(1));
    spatialBasisCache->setPhysicalCellNodes(spaceNodes, vector<GlobalIndexType>(), createSideCaches);
    
//    cout << "spatial cubature weights:\n" << spatialBasisCache->getCubatureWeights();
    double t0 = 0.0, t1 = 1.0;
    double temporalExtent = t1-t0; // 1.0
    double spatialMeasure = spatialBasisCache->getCellMeasures()(0); // measure of the reference-space cell
    vector<double> spatialSideMeasures;
    for (int sideOrdinal=0; sideOrdinal<spaceTopo->getSideCount(); sideOrdinal++) {
      spatialSideMeasures.push_back(spatialBasisCache->getSideBasisCache(sideOrdinal)->getCellMeasures()(0));
    }
    
    MeshPtr singleCellSpaceTimeMesh = getSpaceTimeMesh(spaceTopo,H1Order,refCellExpansionFactor,refCellTranslation,t0,t1); // reference spatial topo x (0,1) in time
    IndexType cellIndex = 0;
    CellPtr cell = singleCellSpaceTimeMesh->getTopology()->getCell(cellIndex);
    
    BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForCell(singleCellSpaceTimeMesh, cellIndex);
    
    CellTopoPtr cellTopo = cell->topology();
    for (int sideOrdinal=0; sideOrdinal<cellTopo->getSideCount(); sideOrdinal++) {
      double expectedMeasure;
      if (cellTopo->sideIsSpatial(sideOrdinal)) {
        int spatialSideOrdinal = cellTopo->getSpatialComponentSideOrdinal(sideOrdinal);
        expectedMeasure = spatialSideMeasures[spatialSideOrdinal] * temporalExtent;
//        cout << "spatial weights:\n" << spatialBasisCache->getSideBasisCache(spatialSideOrdinal)->getWeightedMeasures();
//        cout << "spatial cubature weights:\n" << spatialBasisCache->getSideBasisCache(spatialSideOrdinal)->getCubatureWeights();
      } else {
        expectedMeasure = spatialMeasure;
//        cout << "spatial cubature weights:\n" << spatialBasisCache->getCubatureWeights();
//        cout << "spatial weights:\n" << spatialBasisCache->getWeightedMeasures();
      }
//      cout << "sideOrdinal " << sideOrdinal << ", space-time cubature weights: \n" << spaceTimeBasisCache->getSideBasisCache(sideOrdinal)->getCubatureWeights();
//      cout << "sideOrdinal " << sideOrdinal << ", space-time weights: \n" << spaceTimeBasisCache->getSideBasisCache(sideOrdinal)->getWeightedMeasures();
      double actualMeasure = spaceTimeBasisCache->getSideBasisCache(sideOrdinal)->getCellMeasures()(0);
      TEST_FLOATING_EQUALITY(actualMeasure, expectedMeasure, 1e-14);
    }
  }
  
  void testPhysicalPointsForSpaceTimeBasisCache(int H1Order, CellTopoPtr spaceTopo,
                                                Teuchos::FancyOStream &out, bool &success) {
    MeshPtr mesh = getSpaceTimeMesh(spaceTopo, H1Order);
    
    TensorBasis<>* tensorBasis;
    BasisPtr spaceTimeBasis;
    {
      Camellia::EFunctionSpace spatialFS = (spaceTopo->getDimension() > 1) ? Camellia::FUNCTION_SPACE_HCURL : Camellia::FUNCTION_SPACE_HGRAD;
      Camellia::EFunctionSpace temporalFS = Camellia::FUNCTION_SPACE_HVOL;
      
      Camellia::EOperator spatialOp = (spaceTopo->getDimension() > 1) ? OP_CURL : OP_DX;
      Camellia::EOperator temporalOp = OP_VALUE;
      Camellia::EOperator spaceTimeOp = spatialOp;
      
      vector<Intrepid::EOperator> intrepidOperatorTypes(2);
      intrepidOperatorTypes[0] = OPERATOR_CURL;
      intrepidOperatorTypes[1] = OPERATOR_VALUE;
      
      CellTopoPtr timeTopo = CellTopology::line();
      CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo,1);
      
      int timePolyOrder = 1;
      BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
      BasisPtr temporalBasis = basisFactory->getBasis(timePolyOrder + 1, timeTopo, temporalFS);
      
      BasisPtr spatialBasis = basisFactory->getBasis(H1Order, spaceTopo, spatialFS);
      
      spaceTimeBasis = basisFactory->getBasis(H1Order, spaceTimeTopo, spatialFS,
                                              timePolyOrder + 1, temporalFS);
      
      GlobalIndexType cellID = 0;
      BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
      
      SpaceTimeBasisCache* spaceTimeBasisCache = dynamic_cast<SpaceTimeBasisCache*>(basisCache.get());
      
      BasisCachePtr spatialCache = spaceTimeBasisCache->getSpatialBasisCache();
      BasisCachePtr temporalCache = spaceTimeBasisCache->getTemporalBasisCache();
      
      FC transformedSpatialValues = *spatialCache->getTransformedValues(spatialBasis, spatialOp);
      FC transformedTemporalValues = *temporalCache->getTransformedValues(temporalBasis, temporalOp);
      
      FC transformedSpaceTimeValues = *spaceTimeBasisCache->getTransformedValues(spaceTimeBasis, spaceTimeOp);
      
      tensorBasis = dynamic_cast<TensorBasis<>*>(spaceTimeBasis.get());
    }
    
    GlobalIndexType cellID = 0;
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    CellTopoPtr spaceTimeTopo = mesh->getElementType(cellID)->cellTopoPtr;
    
    SpaceTimeBasisCache* spaceTimeBasisCache = dynamic_cast<SpaceTimeBasisCache*>(basisCache.get());
    
    BasisCachePtr spatialCache = spaceTimeBasisCache->getSpatialBasisCache();
    BasisCachePtr temporalCache = spaceTimeBasisCache->getTemporalBasisCache();
    
//    cout << "space-time cubature degree: " << spaceTimeBasisCache->cubatureDegree() << endl;
//    cout << "spatial cubature degree: " << spatialCache->cubatureDegree() << endl;
//    cout << "temporal cubature degree: " << temporalCache->cubatureDegree() << endl;
    
    int sideCount = spaceTimeTopo->getSideCount();
    
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
//      cout << "sideCache cubature degree: " << sideCache->cubatureDegree() << endl;
      
      if (spaceTimeTopo->sideIsSpatial(sideOrdinal)) {
        // then we expect the physical points to be a tensor product of the spatial side points and the temporal points
        
        unsigned spatialSideOrdinal = spaceTimeTopo->getSpatialComponentSideOrdinal(sideOrdinal);
        BasisCachePtr spatialBasisCacheSide = spatialCache->getSideBasisCache(spatialSideOrdinal);
//        cout << "spatialBasisCacheSide for spatial side " << spatialSideOrdinal << ", cubature degree: ";
//        cout << spatialBasisCacheSide->cubatureDegree() << endl;
        
        FC spatialPoints = spatialBasisCacheSide->getPhysicalCubaturePoints();
        int numSpatialPoints = spatialPoints.dimension(1);
        
        FC temporalPoints = temporalCache->getPhysicalCubaturePoints();
        int numTemporalPoints = temporalPoints.dimension(1);
        
        FC expectedPhysicalPoints(1,numSpatialPoints*numTemporalPoints,spaceTimeTopo->getDimension());
        tensorBasis->getTensorPoints(expectedPhysicalPoints, spatialPoints, temporalPoints);
        
        FC actualPhysicalPoints = sideCache->getPhysicalCubaturePoints();
        
        if (actualPhysicalPoints.size() != expectedPhysicalPoints.size()) {
          out << "actualPhysicalPoints:\n" << actualPhysicalPoints;
          out << "expectedPhysicalPoints:\n" << expectedPhysicalPoints;
        }
        
        TEST_COMPARE_FLOATING_ARRAYS(expectedPhysicalPoints, actualPhysicalPoints, 1e-15);
      } else {
        FC spatialPoints = spatialCache->getPhysicalCubaturePoints();
        int numSpatialPoints = spatialPoints.dimension(1);
        
        FC temporalPoints(1,1,1);
        unsigned timeSideOrdinal = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
        temporalPoints(0,0,0) = (timeSideOrdinal==0) ? 0 : 1; // time goes from 0 to 1 in mesh construction
        int numTemporalPoints = 1;
        
        FC expectedPhysicalPoints(1,numSpatialPoints*numTemporalPoints,spaceTimeTopo->getDimension());
        tensorBasis->getTensorPoints(expectedPhysicalPoints, spatialPoints, temporalPoints);
        
        FC actualPhysicalPoints = sideCache->getPhysicalCubaturePoints();
        TEST_COMPARE_FLOATING_ARRAYS(expectedPhysicalPoints, actualPhysicalPoints, 1e-15);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, BasisCacheForSpaceTimeCell )
  {
    // tests that BasisCache::basisCacheForCell() returns a SpaceTimeBasisCache when
    // a tensor-product mesh is passed in, and tensorProductTopologyMeansSpaceTime is true
    
    MeshPtr mesh = getSpaceTimeMesh(CellTopology::triangle());
    
    GlobalIndexType cellID = 0;
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    SpaceTimeBasisCache* spaceTimeBasisCache = dynamic_cast<SpaceTimeBasisCache*>(basisCache.get());
    TEST_ASSERT(spaceTimeBasisCache != NULL);
    
    // test that the side caches are created and are also space-time caches
    CellTopoPtr spaceTimeTopo = spaceTimeBasisCache->cellTopology();
    int sideCount = spaceTimeTopo->getSideCount();
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      // should be that we have a space-time cache on every spatial side:
      if (spaceTimeTopo->sideIsSpatial(sideOrdinal)) {
        BasisCachePtr sideCache = basisCache->getSideBasisCache(sideOrdinal);
        SpaceTimeBasisCache* spaceTimeSideCache = dynamic_cast<SpaceTimeBasisCache*>(sideCache.get());
        TEST_ASSERT(spaceTimeSideCache != NULL);
      } else {
        // we don't here commit to whether the side cache is SpaceTime or not (right now, it is, but it needn't be)
      }
    }
  }

  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, PhysicalPointsSpaceTimeTriangle )
  {
    CellTopoPtr triangle = CellTopology::triangle();
    int H1Order = 3;
    
    testPhysicalPointsForSpaceTimeBasisCache(H1Order, triangle, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, PhysicalPointsSpaceTimeLine )
  {
    CellTopoPtr triangle = CellTopology::line();
    int H1Order = 3;
    
    testPhysicalPointsForSpaceTimeBasisCache(H1Order, triangle, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, GetValues )
  {
    MeshPtr mesh = getSpaceTimeMesh(CellTopology::triangle());

    Camellia::EFunctionSpace spatialFS = Camellia::FUNCTION_SPACE_HCURL;
    Camellia::EFunctionSpace temporalFS = Camellia::FUNCTION_SPACE_HVOL;
    
    Camellia::EOperator spatialOp = OP_CURL;
    Camellia::EOperator temporalOp = OP_VALUE;
    Camellia::EOperator spaceTimeOp = OP_CURL;
    
    vector<Intrepid::EOperator> intrepidOperatorTypes(2);
    intrepidOperatorTypes[0] = OPERATOR_CURL;
    intrepidOperatorTypes[1] = OPERATOR_VALUE;
    
    CellTopoPtr timeTopo = CellTopology::line();
    CellTopoPtr spaceTopo = CellTopology::triangle();
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo,1);
    
    int timeH1Order = 3;
    BasisFactoryPtr basisFactory = BasisFactory::basisFactory();
    BasisPtr temporalBasis = basisFactory->getBasis(timeH1Order, timeTopo, temporalFS);
    
    int spaceH1Order = 2;
    BasisPtr spatialBasis = basisFactory->getBasis(spaceH1Order, spaceTopo, spatialFS);
    
    BasisPtr spaceTimeBasis = basisFactory->getBasis(spaceH1Order, spaceTimeTopo, spatialFS,
                                                     timeH1Order, temporalFS);
    
    GlobalIndexType cellID = 0;
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    SpaceTimeBasisCache* spaceTimeBasisCache = dynamic_cast<SpaceTimeBasisCache*>(basisCache.get());

    BasisCachePtr spatialCache = spaceTimeBasisCache->getSpatialBasisCache();
    BasisCachePtr temporalCache = spaceTimeBasisCache->getTemporalBasisCache();
    
    FC transformedSpatialValues = *spatialCache->getTransformedValues(spatialBasis, spatialOp);
    FC transformedTemporalValues = *temporalCache->getTransformedValues(temporalBasis, temporalOp);
    
    FC transformedSpaceTimeValues = *spaceTimeBasisCache->getTransformedValues(spaceTimeBasis, spaceTimeOp);
    
    TensorBasis<>* tensorBasis = dynamic_cast<TensorBasis<>*>(spaceTimeBasis.get());
    
    vector<FC> componentValues(2);
    componentValues[0] = transformedSpatialValues;
    componentValues[1] = transformedTemporalValues;
    FC transformedSpaceTimeValuesExpected = transformedSpaceTimeValues;
    transformedSpaceTimeValuesExpected.initialize(0.0);
    tensorBasis->getTensorValues(transformedSpaceTimeValuesExpected, componentValues, intrepidOperatorTypes);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, TransformedBasisValuesLine )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
    int H1Order = 2;
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(H1Order, spaceTimeTopo, Camellia::FUNCTION_SPACE_HGRAD,
                                                            H1Order, Camellia::FUNCTION_SPACE_HGRAD);
    Camellia::EOperator op = OP_VALUE;
    out << "testing with op = OP_VALUE\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
    op = OP_DT;
    out << "testing with op = OP_DT\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
    op = OP_DX; // shouldn't do OP_GRAD yet because still 1D in space
    out << "testing with op = OP_DX\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, TransformedBasisValuesQuadHGRAD )
  {
    CellTopoPtr spaceTopo = CellTopology::quad();
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
    int H1Order = 2;
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(H1Order, spaceTimeTopo, Camellia::FUNCTION_SPACE_HGRAD,
                                                            H1Order, Camellia::FUNCTION_SPACE_HGRAD);
    Camellia::EOperator op = OP_VALUE;
    out << "testing with op = OP_VALUE\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
    op = OP_DT;
    out << "testing with op = OP_DT\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
    op = OP_GRAD;
    out << "testing with op = OP_GRAD\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
  }

  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, TransformedBasisValuesQuadHDIV )
  {
    CellTopoPtr spaceTopo = CellTopology::quad();
    CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
    int H1Order = 2;
    BasisPtr basis = BasisFactory::basisFactory()->getBasis(H1Order, spaceTimeTopo, Camellia::FUNCTION_SPACE_HDIV,
                                                            H1Order, Camellia::FUNCTION_SPACE_HGRAD);
    Camellia::EOperator op = OP_VALUE;
    out << "testing with op = OP_VALUE\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
    op = OP_DT;
    out << "testing with op = OP_DT\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
    op = OP_DIV;
    out << "testing with op = OP_DIV\n";
    testBasisTransformation(basis,op,spaceTopo,out,success);
  }

  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, VolumeMeasureLine )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    int H1Order = 1;
    bool createSideCaches = false;
    BasisCachePtr spatialBasisCache = BasisCache::basisCacheForReferenceCell(spaceTopo, H1Order, createSideCaches);
    
    double temporalExtent = 1.0; // getSpaceTimeMesh() goes from 0 to 1
    double spatialMeasure = spatialBasisCache->getCellMeasures()(0); // measure of the reference-space cell
    
    double refCellExpansionFactor = 1.0;
    MeshPtr singleCellSpaceTimeMesh = getSpaceTimeMesh(spaceTopo,H1Order,refCellExpansionFactor); // reference spatial topo x (0,1) in time
    IndexType cellIndex = 0;
    CellPtr cell = singleCellSpaceTimeMesh->getTopology()->getCell(cellIndex);
    
    BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForCell(singleCellSpaceTimeMesh, cellIndex);
    double expectedMeasure = spatialMeasure * temporalExtent;
    double actualMeasure = spaceTimeBasisCache->getCellMeasures()(0);
    
    TEST_FLOATING_EQUALITY(actualMeasure, expectedMeasure, 1e-15);
  }

  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, SideMeasureLine )
  {
    CellTopoPtr spaceTopo = CellTopology::line();
    testSpaceTimeSideMeasure(spaceTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, SideMeasureTriangle )
  {
    CellTopoPtr spaceTopo = CellTopology::triangle();
    testSpaceTimeSideMeasure(spaceTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, SideMeasureQuad )
  {
    CellTopoPtr spaceTopo = CellTopology::quad();
    testSpaceTimeSideMeasure(spaceTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, SideMeasure_Hexahedron )
  {
    CellTopoPtr spaceTopo = CellTopology::hexahedron();
    testSpaceTimeSideMeasure(spaceTopo, out, success);
  }
  
  TEUCHOS_UNIT_TEST( SpaceTimeBasisCache, SideMeasure_Tetrahedron )
  {
    CellTopoPtr spaceTopo = CellTopology::tetrahedron();
    testSpaceTimeSideMeasure(spaceTopo, out, success);
  }
} // namespace
