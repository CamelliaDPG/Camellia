//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/20/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "Camellia.h"

#include "CamelliaDataHelpers.h"
#include "CamelliaTestingHelpers.h"
#include "Function.h"
#include "MonomialFunctions.h"

#include <Intrepid2_TestUtils.hpp>

using namespace Camellia;

namespace
{
  TEUCHOS_UNIT_TEST( MonomialFunction, DerivativeOnExtrudedMesh1D1V )
  {
    const int baseDim = 1;
    const int extrusionDim = 1;
    const double tol = 1e-15;
    std::vector<int> H1Orders {2,2};
    
    auto mesh = makeSingleElementExtrudedTestMesh(baseDim, extrusionDim, H1Orders);
    
    FunctionPtr x  = Function::xn(1);
    FunctionPtr y  = Function::yn(1);
    FunctionPtr xy = x * y;
    
    auto xy_dx = xy->dx(); // d/dx ( xy ) = y
    double xy_dx_err = (xy_dx - y)->l2norm(mesh);
    TEST_COMPARE(xy_dx_err, <, tol);
    
    auto xy_dy = xy->dy(); // d/dy ( xy ) = x
    double xy_dy_err = (xy_dy - x)->l2norm(mesh);
    TEST_COMPARE(xy_dy_err, <, tol);
    
    // same thing, but now using MonomialFunction class directly
    x = MonomialFunction::xi_n(0,1);
    y = MonomialFunction::xi_n(1,1);
    xy = x * y;
    
    xy_dy = xy->dy(); // d/dy ( xy ) = x
    xy_dx_err = (xy_dx - y)->l2norm(mesh);
    TEST_COMPARE(xy_dy_err, <, tol);
    
    xy_dy = xy->dy(); // d/dy ( xy ) = x
    xy_dy_err = (xy_dy - x)->l2norm(mesh);
    TEST_COMPARE(xy_dy_err, <, tol);
  }

  TEUCHOS_UNIT_TEST( MonomialFunction, XnEqualsMonomialFunction1D1V )
  {
    MPIWrapper::CommWorld()->Barrier();
    
    const int baseDim = 1;
    const int extrusionDim = 1;
    const double tol = 1e-15;
    
    const int maxPolyOrder = 5;
    std::vector<int> H1Orders {maxPolyOrder,2};
    
    auto mesh = makeSingleElementExtrudedTestMesh(baseDim, extrusionDim, H1Orders);
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    
    for (int n=0; n<=maxPolyOrder; n++)
    {
      FunctionPtr x_expected = Function::xn(n);
      FunctionPtr x_actual   = MonomialFunction::xi_n(0,n);

      double err = (x_expected - x_actual)->l2norm(mesh);
      TEST_COMPARE(err, <, tol);
      
      for (const auto & cellID : myCellIDs)
      {
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
        auto data_expected = x_expected->checkoutValuesData(basisCache);
        auto data_actual   = x_actual->checkoutValuesData(basisCache);
        
        x_expected->values(data_expected, basisCache);
        x_actual->values(data_actual, basisCache);
        
        // data nominal shape is (C,P1,P2)
        Intrepid2::testFloatingEquality3(data_expected, data_actual, tol, tol, out, success);
        
        x_expected->checkinValuesData(data_expected);
        x_actual->checkinValuesData(data_actual);
      }
    }
  }

  TEUCHOS_UNIT_TEST( MonomialFunction, XnEqualsMonomialFunction2D3V )
  {
    const int baseDim = 2;
    const int extrusionDim = 3;
    const double tol = 1e-15;
    
    const int maxPolyOrder = 2;
    std::vector<int> H1Orders(1+extrusionDim,maxPolyOrder);
    
    auto mesh = makeSingleElementExtrudedTestMesh(baseDim, extrusionDim, H1Orders);
    
    using Scalar = double;
    using Data = Intrepid2::Data<Scalar,DeviceType>;
    //template<class Scalar, class DataContainer, int expandedRank, int startCollapseDim, int lastCollapseDim>
    using DataInterpreter = ViewCollapseDimensions<Scalar,Data,5,1,4>; // maps (C,P0,P1,P2,P3) --> (C,P)
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    for (int n=0; n<=maxPolyOrder; n++)
    {
      FunctionPtr x_expected = Function::xn(n);
      FunctionPtr x_actual   = MonomialFunction::xi_n(0,n);

      double err = (x_expected - x_actual)->l2norm(mesh);
      TEST_COMPARE(err, <, tol);
      
      for (const auto & cellID : myCellIDs)
      {
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
        auto data_expected = x_expected->checkoutValuesData(basisCache);
        auto data_actual   = x_actual->checkoutValuesData(basisCache);
        
        x_expected->values(data_expected, basisCache);
        x_actual->values(data_actual, basisCache);
        
        // data nominal shape is (C,P2D,P1V,P2V,P3V)
        Intrepid2::testFloatingEquality5(data_expected, data_actual, tol, tol, out, success, "data_expected", "data_actual");
        
        if (success == false)
        {
          // check whether points agree
          auto physicalTensorPoints = basisCache->getPhysicalCubaturePoints();
          auto physicalPointsFC     = basisCache->getPhysicalCubaturePointsFC();
          
          auto physicalPointsFCView = physicalPointsFC.checkoutDeviceView();
          
          // points have shape (C,P,D)
          Intrepid2::testFloatingEquality3(physicalPointsFCView, physicalTensorPoints, tol, tol, out, success, "physicalPointsFCView", "physicalTensorPoints");
        
          physicalPointsFC.checkinDeviceView(false);
        }
        
        x_expected->checkinValuesData(data_expected);
        x_actual->checkinValuesData(data_actual);
      }
    }
  }

  TEUCHOS_UNIT_TEST( MonomialFunction, XnDataPathMatchesViewPath2D3V )
  {
    const int baseDim = 2;
    const int extrusionDim = 3;
    const double tol = 1e-15;
    
    const int maxPolyOrder = 2;
    std::vector<int> H1Orders(1+extrusionDim,maxPolyOrder);
    
    auto mesh = makeSingleElementExtrudedTestMesh(baseDim, extrusionDim, H1Orders);
    
    auto myCellIDs = mesh->cellIDsInPartition(); // will be empty on all but one rank, which will have one cell w/ ID 0
    
    for (int n=0; n<=maxPolyOrder; n++)
    {
      FunctionPtr x   = MonomialFunction::xi_n(0,n);
      
      for (const auto & cellID : myCellIDs)
      {
        BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
        testFunctionDataPathMatchesViewPath3V(x, basisCache, tol, out, success);
      }
    }
  }
} // namespace
