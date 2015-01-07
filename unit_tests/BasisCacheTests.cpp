//
//  BasisCacheTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 12/11/14.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"

#include "SerialDenseWrapper.h"

#include "CamelliaCellTools.h"
#include "CellTopology.h"

#include "Intrepid_CellTools.hpp"
#include "Intrepid_FunctionSpaceTools.hpp"

using namespace Camellia;

namespace {
  vector< CellTopoPtr > getShardsTopologies() {
    vector< CellTopoPtr > shardsTopologies;
    
    shardsTopologies.push_back(CellTopology::point());
    shardsTopologies.push_back(CellTopology::line());
    shardsTopologies.push_back(CellTopology::quad());
    shardsTopologies.push_back(CellTopology::triangle());
    shardsTopologies.push_back(CellTopology::hexahedron());
    shardsTopologies.push_back(CellTopology::tetrahedron()); // tetrahedron not yet supported by permutation
    return shardsTopologies;
  }
  
  FieldContainer<double> unitCubeNodes() {
    CellTopoPtr hex = CellTopology::hexahedron();
    // for now, let's use the reference cell.  (Jacobian should be the identity.)
    FieldContainer<double> refCubePoints(hex->getNodeCount(), hex->getDimension());
    CamelliaCellTools::refCellNodesForTopology(refCubePoints, hex);

    FieldContainer<double> cubePoints = refCubePoints;
    for (int i=0; i<cubePoints.size(); i++) {
      cubePoints[i] = (cubePoints[i]+1) / 2;
    }
    return cubePoints;
  }
  
  TEUCHOS_UNIT_TEST( BasisCache, LineCubature )
  {
    int cubDegree = 1;
    double x0 = 0, x1 = 1;
    BasisCachePtr basisCache = BasisCache::basisCache1D(x0, x1, cubDegree);
    
    const FieldContainer<double>* cubWeights = &basisCache->getCubatureWeights();
    const FieldContainer<double>* physicalCubaturePoints = &basisCache->getPhysicalCubaturePoints();
    
    // try a line m x + b; the integral on (0,1) should be m / 2 + b.
    double m = 2.0, b = -1;
    double expected_integral = m / 2.0 + b;
    
    double integral = 0;
    for (int i=0; i<cubWeights->size(); i++) {
      double x = (*physicalCubaturePoints)(0,i,0);
      double f_val = m * x + b;
      integral += (*cubWeights)(i) * f_val;
    }
    
    TEST_FLOATING_EQUALITY(expected_integral, integral, 1e-15);
  }
  
  TEUCHOS_UNIT_TEST(BasisCache, Jacobian3D)
  {
    
    CellTopoPtr hex = CellTopology::hexahedron();
    // for now, let's use the reference cell.  (Jacobian should be the identity.)
    FieldContainer<double> refCubePoints(hex->getNodeCount(), hex->getDimension());
    CamelliaCellTools::refCellNodesForTopology(refCubePoints, hex);
    
    // small upgrade: unit cube
    //  FieldContainer<double> cubePoints = unitCubeNodes();
    
    int numCells = 1;
    refCubePoints.resize(numCells,8,3); // first argument is cellIndex; we'll just have 1
    
    Teuchos::RCP<shards::CellTopology> hexTopoPtr;
    hexTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Hexahedron<8> >() ));
    
    int spaceDim = 3;
    int cubDegree = 2;
    
    shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8> >() );
    
    FieldContainer<double> physicalCellNodes = refCubePoints;
    physicalCellNodes.resize(numCells,hexTopo.getVertexCount(),spaceDim);
    BasisCache hexBasisCache( physicalCellNodes, hexTopo, cubDegree);
    
    FieldContainer<double> referenceToReferenceJacobian = hexBasisCache.getJacobian();
    int numPoints = referenceToReferenceJacobian.dimension(1);
    
    FieldContainer<double> kronecker(numCells,numPoints,spaceDim,spaceDim);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d1=0; d1<spaceDim; d1++) {
          for (int d2=0; d2<spaceDim; d2++) {
            kronecker(cellIndex,ptIndex,d1,d2) = (d1==d2) ? 1 : 0;
          }
        }
      }
    }
    
    double maxDiff = 0;
    double tol = 1e-15;
    for (int valOrdinal=0; valOrdinal<referenceToReferenceJacobian.size(); valOrdinal++) {
      double diff = abs(kronecker[valOrdinal]-referenceToReferenceJacobian[valOrdinal]);
      TEST_ASSERT( diff < tol );
      maxDiff = max(maxDiff,diff);
    }
    TEST_ASSERT(maxDiff < tol);
    
    if (maxDiff >= tol) {
      cout << "identity map doesn't have identity Jacobian.\n";
      cout << "maxDiff = " << maxDiff << endl;
      
      cout << "referenceToReferenceJacobian:\n" << referenceToReferenceJacobian ;
    }

    
//    TEST_COMPARE_FLOATING_ARRAYS(kronecker, referenceToReferenceJacobian, 1e-15);
    
//    if (! fcsAgree(kronecker, referenceToReferenceJacobian, tol, maxDiff) ) {
//      cout << "identity map doesn't have identity Jacobian.\n";
//      cout << "maxDiff = " << maxDiff << endl;
//      success = false;
//    }
    
    physicalCellNodes = unitCubeNodes();
    physicalCellNodes.resize(numCells, hexTopo.getVertexCount(), hexTopo.getDimension());
    hexBasisCache = BasisCache( physicalCellNodes, hexTopo, cubDegree );
    FieldContainer<double> halfKronecker = kronecker;
    BilinearForm::multiplyFCByWeight(halfKronecker, 0.5);
    
    maxDiff = 0;
    FieldContainer<double> referenceToUnitCubeJacobian = hexBasisCache.getJacobian();
    for (int valOrdinal=0; valOrdinal<referenceToUnitCubeJacobian.size(); valOrdinal++) {
      double diff = abs(halfKronecker[valOrdinal]-referenceToUnitCubeJacobian[valOrdinal]);
      TEST_ASSERT( diff < tol );
      maxDiff = max(maxDiff,diff);
    }
    TEST_ASSERT(maxDiff < tol);
    
    if (maxDiff >= tol) {
      cout << "map to unit cube doesn't have the expected half-identity Jacobian.\n";
      cout << "maxDiff = " << maxDiff << endl;
    }
    
//    TEST_COMPARE_FLOATING_ARRAYS(halfKronecker, referenceToUnitCubeJacobian, 1e-15);

//    if (! fcsAgree(halfKronecker, referenceToUnitCubeJacobian, tol, maxDiff) ) {
//      cout << "map to unit cube doesn't have the expected half-identity Jacobian.\n";
//      cout << "maxDiff = " << maxDiff << endl;
//      success = false;
//    }
  }
  
  TEUCHOS_UNIT_TEST( BasisCache, SideNormals_Space )
  {
    // pretty simple: just check that BasisCache gets values that agree with Intrepid's computation of side normals, on the reference cell
    // as of this writing, this is how BasisCache does it, so it should pass more or less by definition.  But we're likely to swap out
    // this implementation for a more general one that deals with tensor topologies; this will check that we still get the same values as
    // before.
    vector<CellTopoPtr> shardsTopologies = getShardsTopologies();
    for (int topoOrdinal=0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      CellTopoPtr cellTopo = shardsTopologies[topoOrdinal];
      
      if (cellTopo->getDimension() < 1) continue; // side normals only defined when dimension >= 1
      
      int cubatureDegree = 3;
      
      bool createSideCache = true;
      BasisCachePtr volumeCache = BasisCache::basisCacheForReferenceCell(cellTopo, cubatureDegree, createSideCache);
      
      for (int sideOrdinal=0; sideOrdinal<cellTopo->getSideCount(); sideOrdinal++) {
        int numCells = 1;
        BasisCachePtr sideCache = volumeCache->getSideBasisCache(sideOrdinal);
        int numCubPoints = sideCache->getRefCellPoints().dimension(0);
        int spaceDim = sideCache->getSpaceDim();
        
        FieldContainer<double> sideNormalsExpected(numCells, numCubPoints, spaceDim);
        if (spaceDim == 1) {
          for (int ptOrdinal=0; ptOrdinal<numCubPoints; ptOrdinal++) {
            if (sideOrdinal==0) { // on the -1 side of the line element
              sideNormalsExpected(0,ptOrdinal,0) = -1;
            } else {
              sideNormalsExpected(0,ptOrdinal,0) =  1;
            }
          }
        } else {
          FieldContainer<double> normalLengths(numCells, numCubPoints);

          FieldContainer<double> cellJacobian = sideCache->getJacobian();
          CellTools<double>::getPhysicalSideNormals(sideNormalsExpected, cellJacobian, sideOrdinal, cellTopo->getShardsTopology());
          // make unit length
          RealSpaceTools<double>::vectorNorm(normalLengths, sideNormalsExpected, NORM_TWO);
          FunctionSpaceTools::scalarMultiplyDataData<double>(sideNormalsExpected, normalLengths, sideNormalsExpected, true);
        }
        
        FieldContainer<double> sideNormals = sideCache->getSideNormals();
        
        TEST_COMPARE_FLOATING_ARRAYS(sideNormalsExpected, sideNormals, 1e-15);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisCache, SideNormals_SpaceTime )
  {    
    vector<CellTopoPtr> shardsTopologies = getShardsTopologies();
    for (int topoOrdinal=0; topoOrdinal < shardsTopologies.size(); topoOrdinal++) {
      CellTopoPtr shardsTopo = shardsTopologies[topoOrdinal];
      
      int tensorialDegree = 1;
      CellTopoPtr cellTopo = CellTopology::cellTopology(shardsTopo->getShardsTopology(), tensorialDegree);
      
      if (cellTopo->getDimension() < 1) continue; // side normals only defined when dimension >= 1
      
      out << "Testing side normals for space-time topo " << cellTopo->getName() << endl;
      
      int cubatureDegree = 3;
      
      bool createSideCache = true;
      BasisCachePtr volumeCacheSpaceTime = BasisCache::basisCacheForReferenceCell(cellTopo, cubatureDegree, createSideCache);
      BasisCachePtr volumeCacheSpace = BasisCache::basisCacheForReferenceCell(shardsTopo, cubatureDegree, createSideCache);

      // for reference cells, we expect the space-time normals to go as follows:
      //   - the first shardsTopo->getSideCount() sides will have normals identical to shardsTopo in its reference space, padded with 0 in time dimension
      //   - the next side will have normal equal to 0 in every spatial dimension, -1 in temporal
      //   - final side with have normal +1 in temporal dimension
      
      for (int sideOrdinal=0; sideOrdinal<cellTopo->getSideCount(); sideOrdinal++) {
        int numCells = 1;
        BasisCachePtr sideCacheSpaceTime = volumeCacheSpaceTime->getSideBasisCache(sideOrdinal);
        int numCubPoints = sideCacheSpaceTime->getRefCellPoints().dimension(0);
        int dim = sideCacheSpaceTime->getSpaceDim() + 1;
        
        FieldContainer<double> sideNormalsExpected(numCells, numCubPoints, dim);

        if (cellTopo->sideIsSpatial(sideOrdinal)) {
          unsigned spatialSideOrdinal = cellTopo->getSpatialComponentSideOrdinal(sideOrdinal);
          BasisCachePtr sideCacheSpace = volumeCacheSpace->getSideBasisCache(spatialSideOrdinal);
          // set up cubature points for shards topo
          FieldContainer<double> refPointsSpaceTime = sideCacheSpaceTime->getSideRefCellPointsInVolumeCoordinates();
          FieldContainer<double> refPointsSpace(numCubPoints, shardsTopo->getDimension());
          for (int ptOrdinal=0; ptOrdinal<numCubPoints; ptOrdinal++) {
            for (int d=0; d<shardsTopo->getDimension(); d++) {
              refPointsSpace(ptOrdinal,d) = refPointsSpaceTime(ptOrdinal,d);
            }
          }
          sideCacheSpace->setRefCellPoints(refPointsSpace);
          FieldContainer<double> sideNormalsSpace = sideCacheSpace->getSideNormals();
          for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++) {
            for (int ptOrdinal=0; ptOrdinal < numCubPoints; ptOrdinal++) {
              for (int d=0; d<shardsTopo->getDimension(); d++) {
                sideNormalsExpected(cellOrdinal,ptOrdinal,d) = sideNormalsSpace(cellOrdinal,ptOrdinal,d);
              }
            }
          }
//          {
//            out << "sideNormalsSpace:\n" << sideNormalsSpace;
//            // DEBUGGING:
//            out << "refPointsSpaceTime:\n" << refPointsSpaceTime;
//            out << "refPointsSpace:\n" << refPointsSpace;
//          }
          
        } else {
          unsigned temporalNodeOrdinal = cellTopo->getTemporalComponentSideOrdinal(sideOrdinal);
          double timeNormal = (temporalNodeOrdinal == 0) ? -1 : 1;
          FieldContainer<double> spaceTimeCellJacobian = sideCacheSpaceTime->getJacobian();
          int d_time = shardsTopo->getDimension();
          for (int cellOrdinal=0; cellOrdinal < numCells; cellOrdinal++) {
            for (int ptOrdinal=0; ptOrdinal < numCubPoints; ptOrdinal++) {
              double timeJacobian = spaceTimeCellJacobian(cellOrdinal,ptOrdinal,d_time,d_time);
              if (timeJacobian > 0) {
                sideNormalsExpected(cellOrdinal,ptOrdinal,d_time) = timeNormal;
              } else {
                sideNormalsExpected(cellOrdinal,ptOrdinal,d_time) = -timeNormal;
              }
            }
          }
        }
        
        FieldContainer<double> sideNormals = sideCacheSpaceTime->getSideNormalsSpaceTime();
        
        {
          // DEBUGGING:
          out << "testing with sideOrdinal " << sideOrdinal << endl;
          out << "sideNormals:\n" << sideNormals;
          out << "sideNormalsExpected:\n" << sideNormalsExpected;
        }
        
        SerialDenseWrapper::roundZeros(sideNormals, 1e-15);
        SerialDenseWrapper::roundZeros(sideNormalsExpected, 1e-15);
        
        TEST_COMPARE_FLOATING_ARRAYS(sideNormalsExpected, sideNormals, 1e-15);
      }
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisCache, SetRefCellPoints )
  {
    double tol = 1e-15;
      
    // test setting ref points on the side cache
    CellTopoPtr cellTopo = CellTopology::quad();
    int numSides = cellTopo->getSideCount();
    
    int cubDegree = 2;
    shards::CellTopology shardsTopo = cellTopo->getShardsTopology();
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(shardsTopo, cubDegree, true) ); // true: create side cache, too
    
    FieldContainer<double> unitQuadNodes(1,cellTopo->getNodeCount(), cellTopo->getDimension());
    unitQuadNodes(0,0,0) = 0.0;
    unitQuadNodes(0,0,1) = 0.0;
    unitQuadNodes(0,1,0) = 1.0;
    unitQuadNodes(0,1,1) = 0.0;
    unitQuadNodes(0,2,0) = 1.0;
    unitQuadNodes(0,2,1) = 1.0;
    unitQuadNodes(0,3,0) = 0.0;
    unitQuadNodes(0,3,1) = 1.0;
    
    basisCache->setPhysicalCellNodes(unitQuadNodes, vector<GlobalIndexType>(), true);
    
    for (int sideOrdinal=0; sideOrdinal < numSides; sideOrdinal++)
    {
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      FieldContainer<double> refCellPoints = sideBasisCache->getRefCellPoints();
      FieldContainer<double> physicalCubaturePointsExpected = sideBasisCache->getPhysicalCubaturePoints();
      //    cout << "side " << sideIndex << " ref cell points:\n" << refCellPoints;
      //    cout << "side " << sideIndex << " physCubature points:\n" << physicalCubaturePointsExpected;
      sideBasisCache->setRefCellPoints(refCellPoints);
      FieldContainer<double> physicalCubaturePointsActual = sideBasisCache->getPhysicalCubaturePoints();
      
      double maxDiff = 0;
      for (int valOrdinal=0; valOrdinal<physicalCubaturePointsActual.size(); valOrdinal++) {
        double diff = abs(physicalCubaturePointsExpected[valOrdinal]-physicalCubaturePointsActual[valOrdinal]);
        TEST_ASSERT( diff < tol );
        double maxDiff = max(maxDiff,diff);
      }
      TEST_ASSERT(maxDiff < tol);
      
      if (maxDiff >= tol) {
        cout << "After resetting refCellPoints, physical cubature points are different in side basis cache.\n";
        cout << "maxDiff = " << maxDiff << endl;
      }

    }
      // TODO: test quad
      
      // test hexahedron
      int numCells = 1;
      int numPoints = 1;
      int spaceDim = 3;
    
      shards::CellTopology hexTopo(shards::getCellTopologyData<shards::Hexahedron<8> >() );
      FieldContainer<double> physicalCellNodes = unitCubeNodes();
      physicalCellNodes.resize(numCells,hexTopo.getVertexCount(),spaceDim);
      BasisCache hexBasisCache( physicalCellNodes, hexTopo, cubDegree);
      
      FieldContainer<double> refCellPointsHex(numPoints,spaceDim);
      refCellPointsHex(0,0) = 0.0;
      refCellPointsHex(0,1) = 0.0;
      refCellPointsHex(0,2) = 0.0;
      
      FieldContainer<double> physicalPointsExpected(numCells,numPoints,spaceDim);
      physicalPointsExpected(0,0,0) = 0.5;
      physicalPointsExpected(0,0,1) = 0.5;
      physicalPointsExpected(0,0,2) = 0.5;
      
      hexBasisCache.setRefCellPoints(refCellPointsHex);
      
      FieldContainer<double> physicalPointsActual = hexBasisCache.getPhysicalCubaturePoints();
    
    double maxDiff = 0;
    for (int valOrdinal=0; valOrdinal<physicalPointsActual.size(); valOrdinal++) {
      double diff = abs(physicalPointsExpected[valOrdinal]-physicalPointsActual[valOrdinal]);
      TEST_ASSERT( diff < tol );
      maxDiff = max(maxDiff,diff);
    }
    TEST_ASSERT(maxDiff < tol);
    
    if (maxDiff >= tol) {
      cout << "physical points don't match expected for hexahedron.";
      cout << "maxDiff = " << maxDiff << endl;
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisCache, SetRefCellPointsSpaceTimeSide )
  {
    double tol = 1e-15;
    
    // test setting ref points on the side cache
    CellTopoPtr lineTopo = CellTopology::line();
    CellTopoPtr lineLineTopo = CellTopology::cellTopology(lineTopo->getShardsTopology(), 1);

    int cubDegree = 2;
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(lineLineTopo, cubDegree, true) ); // true: create side caches, too

    FieldContainer<double> refLineNodes(lineTopo->getNodeCount(), lineTopo->getDimension());
    CamelliaCellTools::refCellNodesForTopology(refLineNodes, lineTopo);

    int numPoints = 2;
    FieldContainer<double> sideRefCubaturePoints(numPoints,lineTopo->getDimension());
    sideRefCubaturePoints(0,0) = -0.5;
    sideRefCubaturePoints(1,0) =  0.5;
    
    int sideCount = lineLineTopo->getSideCount();
    for (int sideOrdinal=0; sideOrdinal < sideCount; sideOrdinal++)
    {
      BasisCachePtr sideBasisCache = basisCache->getSideBasisCache(sideOrdinal);
      FieldContainer<double> volumeRefPointsExpected(numPoints,lineLineTopo->getDimension());
      
      if (lineLineTopo->sideIsSpatial(sideOrdinal)) {
        // then the space coordinate is fixed at ±1, and the time coordinate where the cubature happens
        unsigned spatialNodeOrdinal = lineLineTopo->getSpatialComponentSideOrdinal(sideOrdinal);
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++) {
          volumeRefPointsExpected(ptOrdinal,0) = refLineNodes(spatialNodeOrdinal,0);
          volumeRefPointsExpected(ptOrdinal,1) = sideRefCubaturePoints(ptOrdinal,0);
        }
      } else {
        unsigned temporalNodeOrdinal = lineLineTopo->getTemporalComponentSideOrdinal(sideOrdinal);
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++) {
          volumeRefPointsExpected(ptOrdinal,0) = sideRefCubaturePoints(ptOrdinal,0);
          volumeRefPointsExpected(ptOrdinal,1) = refLineNodes(temporalNodeOrdinal,0);
        }
      }
      
      sideBasisCache->setRefCellPoints(sideRefCubaturePoints);
      FieldContainer<double> volumeRefPoints = sideBasisCache->getSideRefCellPointsInVolumeCoordinates();
      
      TEST_COMPARE_FLOATING_ARRAYS(volumeRefPoints, volumeRefPointsExpected, tol);
      
      double maxDiff = 0;
      for (int valOrdinal=0; valOrdinal<volumeRefPoints.size(); valOrdinal++) {
        double diff = abs(volumeRefPointsExpected[valOrdinal]-volumeRefPoints[valOrdinal]);
        TEST_ASSERT( diff < tol );
        maxDiff = max(maxDiff,diff);
      }
      TEST_ASSERT(maxDiff < tol);
    }
  }
} // namespace