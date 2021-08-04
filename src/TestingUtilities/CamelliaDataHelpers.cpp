//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "CamelliaDataHelpers.h"

#include "BasisCache.h"

namespace Camellia
{
  void testFunctionDataPathMatchesViewPath1V(FunctionPtr f, BasisCachePtr basisCache1V, const double &tol, Teuchos::FancyOStream &out, bool &success)
  {
    using Scalar = double;
    using ViewType = Camellia::DeviceView<Scalar>;
    //template<class Scalar, class DataContainer, int expandedRank, int startCollapseDim, int lastCollapseDim>
    using ViewInterpreter = ViewCollapseDimensions<Scalar,ViewType,3,1,2>; // maps (C,P0,P1) --> (C,P)
    
    static_assert(Intrepid2::supports_rank<ViewInterpreter, 3>::value, "ViewInterpreter should support rank 3");
    auto data_actual = f->checkoutValuesData(basisCache1V);
    f->values(data_actual, basisCache1V);
      
    const int numCells = basisCache1V->numCells();
    
    const int numPoints = basisCache1V->getPhysicalCubaturePointsFC().dimension(1);
    Camellia::FieldContainer<Scalar> valuesFC(numCells, numPoints);
    f->values(valuesFC, basisCache1V);
    auto fcView = valuesFC.checkoutDeviceView(true); // true: deep_copy from host
      
    auto tensorPoints = basisCache1V->getPhysicalCubaturePoints();
    Kokkos::Array<int,2> pointComponentBounds;
    for (int r=0; r<2; r++)
    {
      const auto & pointComponent = tensorPoints.getTensorComponent(r);
      pointComponentBounds[r] = pointComponent.extent_int(1); // (C,P)
    }
    
    if (data_actual.rank() == 3)
    {
      // data nominal shape is (C,P1D,P1V); hence we interpret to the (C,P) fcView
      ViewInterpreter interpretedView(fcView, pointComponentBounds);
      Intrepid2::testFloatingEquality3(interpretedView, data_actual, tol, tol, out, success, "view path", "data path");
    }
    else
    {
      // if data's nominal rank is not 3, then it should have shape (C,P)
      TEUCHOS_TEST_FOR_EXCEPTION(data_actual.rank() != 2, std::invalid_argument, "Unexpected rank for Data");
      Intrepid2::testFloatingEquality2(fcView, data_actual, tol, tol, out, success, "view path", "data path");
    }
    
    valuesFC.checkinDeviceView(false); // false: don't deep copy back to host
    
    f->checkinValuesData(data_actual);
  }
    
  void testFunctionDataPathMatchesViewPath3V(FunctionPtr f, BasisCachePtr basisCache3V, const double &tol, Teuchos::FancyOStream &out, bool &success)
  {
    using Scalar = double;
    using ViewType = Camellia::DeviceView<Scalar>;
    //template<class Scalar, class DataContainer, int expandedRank, int startCollapseDim, int lastCollapseDim>
    using ViewInterpreter = ViewCollapseDimensions<Scalar,ViewType,5,1,4>; // maps (C,P0,P1,P2,P3) --> (C,P)
    
    static_assert(Intrepid2::supports_rank<ViewInterpreter, 5>::value, "ViewInterpreter should support rank 5");
    auto data_actual = f->checkoutValuesData(basisCache3V);
    f->values(data_actual, basisCache3V);
      
    const int numCells = basisCache3V->numCells();
    
    const int numPoints = basisCache3V->getPhysicalCubaturePointsFC().dimension(1);
    Camellia::FieldContainer<Scalar> valuesFC(numCells, numPoints);
    f->values(valuesFC, basisCache3V);
    auto fcView = valuesFC.checkoutDeviceView(true); // true: deep_copy from host
      
    auto tensorPoints = basisCache3V->getPhysicalCubaturePoints();
    Kokkos::Array<int,4> pointComponentBounds;
    for (int r=0; r<4; r++)
    {
      const auto & pointComponent = tensorPoints.getTensorComponent(r);
      pointComponentBounds[r] = pointComponent.extent_int(1); // (C,P)
    }
    
    if (data_actual.rank() == 5)
    {
      // data nominal shape is (C,P2D,P1V,P2V,P3V); hence we interpret to the (C,P) fcView
      ViewInterpreter interpretedView(fcView, pointComponentBounds);
      Intrepid2::testFloatingEquality5(interpretedView, data_actual, tol, tol, out, success, "view path", "data path");
    }
    else
    {
      // if data's nominal rank is not 5, then it should have shape (C,P)
      TEUCHOS_TEST_FOR_EXCEPTION(data_actual.rank() != 2, std::invalid_argument, "Unexpected rank for Data");
      Intrepid2::testFloatingEquality2(fcView, data_actual, tol, tol, out, success, "view path", "data path");
    }
    
    valuesFC.checkinDeviceView(false); // false: don't deep copy back to host
    
    f->checkinValuesData(data_actual);
  }
} // namespace Camellia
