//
// © 2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  BasisCacheFactoryTests
//  Camellia
//
//  Created by Nate Roberts on 10/11/18.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"

#include "BasisCacheFactory.hpp"

using namespace Camellia;

namespace
{
  void testUnionCacheHasCorrectJacobian(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2; // linear field order
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int meshWidth = 1;
    vector<double> domainDim(spaceDim, 1.0);
    vector<int> meshDim(spaceDim, meshWidth);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), domainDim, meshDim, H1Order);
    
    double expectedJacobianDiagonal = 0.5; // ref space element has width 2 in each dimension; physical element has width 1
    double expectedJacobianDet = 1.0;
    for (int d=0; d<spaceDim; d++)
    {
      expectedJacobianDet *= expectedJacobianDiagonal;
    }
    double expectedJacobianInverseDiagonal = 1.0 / expectedJacobianDiagonal;
    
    GlobalIndexType parentCellID = 0;
    
    bool parentCellIsLocallyKnown = mesh->getTopology()->isValidCellIndex(parentCellID);
    
    double tol=1e-15;
    
    if (parentCellIsLocallyKnown)
    {
      CellPtr parentCell = mesh->getTopology()->getCell(parentCellID);
      
      auto parentCache = BasisCache::basisCacheForCell(mesh, parentCellID);
      
      auto jacobian = parentCache->getJacobian();
      auto jacobianInv = parentCache->getJacobianInv();
      auto jacobianDet = parentCache->getJacobianDet();
      int numCells = 1;
      int numPoints = jacobian.dimension(1);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          TEST_FLOATING_EQUALITY(expectedJacobianDet, jacobianDet(cellOrdinal,ptOrdinal), tol);
          for (int d1=0; d1<spaceDim; d1++)
          {
            for (int d2=0; d2<spaceDim; d2++)
            {
              if (d1 == d2)
              {
                TEST_FLOATING_EQUALITY(expectedJacobianDiagonal, jacobian(cellOrdinal,ptOrdinal,d1,d2), tol);
                TEST_FLOATING_EQUALITY(expectedJacobianInverseDiagonal, jacobianInv(cellOrdinal,ptOrdinal,d1,d2), tol);
              }
              else
              {
                TEST_COMPARE(abs(jacobian(cellOrdinal,ptOrdinal,d1,d2)), <, tol);
                TEST_COMPARE(abs(jacobianInv(cellOrdinal,ptOrdinal,d1,d2)), <, tol);
              }
            }
          }
        }
      }
    }
      
    vector<GlobalIndexType> cellsToRefine {parentCellID};
    mesh->hRefine(cellsToRefine);
    
    parentCellIsLocallyKnown = mesh->getTopology()->isValidCellIndex(parentCellID);
    if (parentCellIsLocallyKnown)
    {
      CellPtr parentCell = mesh->getTopology()->getCell(parentCellID);
      
      std::vector<BasisCachePtr> subdomainCaches;
      auto childIDs = parentCell->getChildIndices(mesh->getTopology());
      int totalPointCount = 0;
      for (auto childCellID : childIDs)
      {
        auto childCache = BasisCache::basisCacheForCell(mesh, childCellID);
        subdomainCaches.push_back(childCache);
        totalPointCount += childCache->getRefCellPointsFC().dimension(0);
      }
      
      auto unionCache = BasisCacheFactory<double>::unionBasisCacheForParentCell(parentCell->refinementPattern(), subdomainCaches);
      
      TEST_EQUALITY(totalPointCount, unionCache->getRefCellPointsFC().dimension(0));
      TEST_EQUALITY(totalPointCount, unionCache->getPhysicalCubaturePointsFC().dimension(1));
      
      auto jacobian = unionCache->getJacobian();
      auto jacobianInv = unionCache->getJacobianInv();
      auto jacobianDet = unionCache->getJacobianDet();
    
      int numCells = 1;
      int numPoints = totalPointCount;
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
        {
          TEST_FLOATING_EQUALITY(expectedJacobianDet, jacobianDet(cellOrdinal,ptOrdinal), tol);
          for (int d1=0; d1<spaceDim; d1++)
          {
            for (int d2=0; d2<spaceDim; d2++)
            {
              if (d1 == d2)
              {
                TEST_FLOATING_EQUALITY(expectedJacobianDiagonal, jacobian(cellOrdinal,ptOrdinal,d1,d2), tol);
                TEST_FLOATING_EQUALITY(expectedJacobianInverseDiagonal, jacobianInv(cellOrdinal,ptOrdinal,d1,d2), tol);
              }
              else
              {
                TEST_COMPARE(abs(jacobian(cellOrdinal,ptOrdinal,d1,d2)), <, tol);
                TEST_COMPARE(abs(jacobianInv(cellOrdinal,ptOrdinal,d1,d2)), <, tol);
              }
            }
          }
        }
      }
    }
  }
  
  void testUnionCacheHasCorrectPointCount(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2; // linear field order
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int meshWidth = 1;
    vector<double> domainDim(spaceDim, 1.0);
    vector<int> meshDim(spaceDim, meshWidth);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), domainDim, meshDim, H1Order);
    
    GlobalIndexType parentCellID = 0;
    
    vector<GlobalIndexType> cellsToRefine = {parentCellID};
    mesh->hRefine(cellsToRefine);
    
    bool parentCellIsLocallyKnown = mesh->getTopology()->isValidCellIndex(parentCellID);
    if (parentCellIsLocallyKnown)
    {
      CellPtr parentCell = mesh->getTopology()->getCell(parentCellID);
    
      std::vector<BasisCachePtr> subdomainCaches;
      auto childIDs = parentCell->getChildIndices(mesh->getTopology());
      int totalPointCount = 0;
      for (auto childCellID : childIDs)
      {
        auto childCache = BasisCache::basisCacheForCell(mesh, childCellID);
        subdomainCaches.push_back(childCache);
        totalPointCount += childCache->getRefCellPointsFC().dimension(0);
      }
      
      auto unionCache = BasisCacheFactory<double>::unionBasisCacheForParentCell(parentCell->refinementPattern(), subdomainCaches);
      
      TEST_EQUALITY(totalPointCount, unionCache->getRefCellPointsFC().dimension(0));
      TEST_EQUALITY(totalPointCount, unionCache->getPhysicalCubaturePointsFC().dimension(1));
    }
  }
  
  void testUnionCacheIntegrationWorks(int spaceDim, Teuchos::FancyOStream &out, bool &success)
  {
    int H1Order = 2; // linear field order
    bool conformingTraces = true;
    PoissonFormulation form(spaceDim,conformingTraces);
    
    int meshWidth = 1;
    vector<double> domainDim(spaceDim, 1.0);
    vector<int> meshDim(spaceDim, meshWidth);
    MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), domainDim, meshDim, H1Order);
    
    FunctionPtr f = Function::xn(1);
    
    double expectedIntegral = 0.5; // integral of x on [0,1] = [(x^2)/2 from 0 to 1] = 0.5.  Multi-d just multiplies by 1.0
    
    // sanity check: verify that integral on the parent element matches expectations
    double tol = 1e-14;
    GlobalIndexType parentCellID = 0;
    
    bool parentCellIsLocallyKnown = mesh->getTopology()->isValidCellIndex(parentCellID);
    if (parentCellIsLocallyKnown)
    {
      CellPtr parentCell = mesh->getTopology()->getCell(parentCellID);
      BasisCachePtr parentCache = BasisCache::basisCacheForCell(mesh, parentCellID);
      double actualIntegral = f->integrate(parentCache);
      double err = abs(expectedIntegral - actualIntegral);
      TEST_COMPARE(err, <, tol);
    }
    
    vector<GlobalIndexType> cellsToRefine = {0};
    mesh->hRefine(cellsToRefine);
    
    parentCellIsLocallyKnown = mesh->getTopology()->isValidCellIndex(parentCellID);
    if (parentCellIsLocallyKnown)
    {
      CellPtr parentCell = mesh->getTopology()->getCell(parentCellID);
      std::vector<BasisCachePtr> subdomainCaches;
      auto childIDs = parentCell->getChildIndices(mesh->getTopology());
      for (auto childCellID : childIDs)
      {
        auto childCache = BasisCache::basisCacheForCell(mesh, childCellID);
        subdomainCaches.push_back(childCache);
      }
      
      auto unionCache = BasisCacheFactory<double>::unionBasisCacheForParentCell(parentCell->refinementPattern(), subdomainCaches);
    
      double actualIntegral = f->integrate(unionCache);
      TEST_FLOATING_EQUALITY(expectedIntegral, actualIntegral, tol);
    }
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheHasCorrectJacobian_1D )
  {
    int spaceDim = 1;
    testUnionCacheHasCorrectJacobian(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheHasCorrectJacobian_2D )
  {
    int spaceDim = 2;
    testUnionCacheHasCorrectJacobian(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheHasCorrectJacobian_3D )
  {
    int spaceDim = 3;
    testUnionCacheHasCorrectJacobian(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheHasCorrectPointCount_1D )
  {
    int spaceDim = 1;
    testUnionCacheHasCorrectPointCount(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheHasCorrectPointCount_2D )
  {
    int spaceDim = 2;
    testUnionCacheHasCorrectPointCount(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheHasCorrectPointCount_3D )
  {
    int spaceDim = 3;
    testUnionCacheHasCorrectPointCount(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheIntegrationWorks_1D )
  {
    int spaceDim = 1;
    testUnionCacheIntegrationWorks(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheIntegrationWorks_2D )
  {
    int spaceDim = 2;
    testUnionCacheIntegrationWorks(spaceDim, out, success);
  }
  
  TEUCHOS_UNIT_TEST( BasisCacheFactory, UnionCacheIntegrationWorks_3D )
  {
    int spaceDim = 3;
    testUnionCacheIntegrationWorks(spaceDim, out, success);
  }
  
} // namespace
