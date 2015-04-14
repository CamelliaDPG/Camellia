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

using namespace Camellia;
using namespace Intrepid;

namespace {
  typedef Intrepid::FieldContainer<double> FC;

  MeshPtr getSpaceTimeMesh(CellTopoPtr spaceTopo, int H1Order=2, double refCellExpansionFactor=2.0, double t0=0, double t1=1) {
    CellTopoPtr timeTopo = CellTopology::line();
    int tensorialDegree = 1;
    CellTopoPtr tensorTopo = CellTopology::cellTopology(spaceTopo, tensorialDegree);
    
    FC spaceNodes(spaceTopo->getNodeCount(), spaceTopo->getDimension());
    FC timeNodes(timeTopo->getNodeCount(), timeTopo->getDimension());
    
    CamelliaCellTools::refCellNodesForTopology(spaceNodes, spaceTopo);
    // stretch the spaceNodes:
    for (int i=0; i<spaceNodes.size(); i++) {
      spaceNodes[i] *= refCellExpansionFactor;
    }
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
  
  void testSpaceTimeSideMeasure(CellTopoPtr spaceTopo, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2;
    int delta_k = 1;
    bool createSideCaches = true;
    BasisCachePtr spatialBasisCache = BasisCache::basisCacheForReferenceCell(spaceTopo, H1Order*2+delta_k, createSideCaches);
    
//    cout << "spatial cubature weights:\n" << spatialBasisCache->getCubatureWeights();
    
    double t0 = 0.0, t1 = 1.0;
    double temporalExtent = t1-t0; // 1.0
    double spatialMeasure = spatialBasisCache->getCellMeasures()(0); // measure of the reference-space cell
    vector<double> spatialSideMeasures;
    for (int sideOrdinal=0; sideOrdinal<spaceTopo->getSideCount(); sideOrdinal++) {
      spatialSideMeasures.push_back(spatialBasisCache->getSideBasisCache(sideOrdinal)->getCellMeasures()(0));
    }
    
    double refCellExpansionFactor = 1.0;
    MeshPtr singleCellSpaceTimeMesh = getSpaceTimeMesh(spaceTopo,H1Order,refCellExpansionFactor,t0,t1); // reference spatial topo x (0,1) in time
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
      TEST_FLOATING_EQUALITY(actualMeasure, expectedMeasure, 1e-15);
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
    // TODO: implement this
    success = false;
  }
} // namespace
