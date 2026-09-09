//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "MonomialFunctions.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace std;

string Xn::displayString()
{
  ostringstream ss;
  if ((_n != 1) && (_n != 0))
  {
    ss << "x^" << _n ;
  }
  else if (_n == 1)
  {
    ss << "x";
  }
  else
  {
    ss << "(1)";
  }
  return ss.str();
}
Xn::Xn(int n) : TFunction<double>("Xn",0,(n != 0) ? VARIES_IN_DIMENSION_0 : VARIES_IN_NO_DIMENSION)
{
  _n = n;
}

void Xn::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Xn values()");
  auto & points = basisCache->getPhysicalCubaturePointsFC(); // (C,P,D) FieldContainer
  auto devicePoints = points.checkoutDeviceView();
  const int n = _n;
  Kokkos::parallel_for("Xn values() functor", Camellia::FieldContainer<double>::getMDRangePolicy2(values),
  KOKKOS_LAMBDA (const int i0, const int i1) {
    auto & entry = values(i0, i1);
    const int xDim = 0;
    const auto x = devicePoints(i0,i1,xDim);
    const double value = pow(x,n);
    assignValue(entry, modType, value);
  });
  points.checkinDeviceView(true);
}

TFunctionPtr<double> Xn::dx()
{
  if (_n == 0)
  {
    return TFunction<double>::zero();
  }
  else if (_n == 1)
  {
    return TFunction<double>::constant(1.0);
  }
  TFunctionPtr<double> x_n_minus = Teuchos::rcp( new Xn(_n-1) );
  return (double)_n * x_n_minus;
}
TFunctionPtr<double> Xn::dy()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Xn::dz()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Xn::dt()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Xn::di(int i)
{
  if (i == 1)
  {
    return dx();
  }
  else
  {
    return TFunction<double>::zero();
  }
}

string Yn::displayString()
{
  ostringstream ss;
  const int n = this->n();
  if ((n != 1) && (n != 0))
  {
    ss << "y^" << n ;
  }
  else if (n == 1)
  {
    ss << "y";
  }
  else
  {
    ss << "(1)";
  }
  return ss.str();
}

Yn::Yn(int n) : MonomialFunction(1, n)
{}

TFunctionPtr<double> Yn::dt()
{
  return TFunction<double>::zero();
}

string Zn::displayString()
{
  ostringstream ss;
  if ((_n != 1) && (_n != 0))
  {
    ss << "z^" << _n ;
  }
  else if (_n == 1)
  {
    ss << "z";
  }
  else
  {
    ss << "(1)";
  }
  return ss.str();
}
Zn::Zn(int n) : TFunction<double>("Zn",0,(n != 0) ? VARIES_IN_DIMENSION_2 : VARIES_IN_NO_DIMENSION)
{
  _n = n;
}

void Zn::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Zn values()");
  const auto & points = basisCache->getPhysicalCubaturePointsFC(); // (C,P,D) FieldContainer
  auto devicePoints = points.checkoutDeviceView();
  const int n = _n;
  Kokkos::parallel_for("Zn values() functor", Camellia::FieldContainer<double>::getMDRangePolicy2(values),
  KOKKOS_LAMBDA (const int i0, const int i1) {
    auto & entry = values(i0, i1);
    const int zDim = 2;
    const auto z = devicePoints(i0,i1,zDim);
    const double value = pow(z,n);
    assignValue(entry, modType, value);
  });
  points.checkinDeviceView(true);
}
TFunctionPtr<double> Zn::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Zn::dy()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Zn::dz()
{
  if (_n == 0)
  {
    return TFunction<double>::zero();
  }
  else if (_n == 1)
  {
    return TFunction<double>::constant(1.0);
  }
  TFunctionPtr<double> z_n_minus = Teuchos::rcp( new Zn(_n-1) );
  return (double)_n * z_n_minus;
}
TFunctionPtr<double> Zn::dt()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Zn::di(int i)
{
  if (i == 3)
  {
    return dz();
  }
  else
  {
    return TFunction<double>::zero();
  }
}

string Tn::displayString()
{
  ostringstream ss;
  if ((_n != 1) && (_n != 0))
  {
    ss << "t^" << _n ;
  }
  else if (_n == 1)
  {
    ss << "t";
  }
  else
  {
    ss << "(1)";
  }
  return ss.str();
}
Tn::Tn(int n) : TFunction<double>("Tn")
{
  _n = n;
}

void Tn::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Tn values()");
  const auto & points = basisCache->getPhysicalCubaturePointsFC(); // (C,P,D) FieldContainer
  auto devicePoints = points.checkoutDeviceView();
  const int n = _n;
  const int spaceDim = basisCache->cellTopology()->getDimension() - 1; // we're assuming a space-time context
  Kokkos::parallel_for("Tn values() functor", Camellia::FieldContainer<double>::getMDRangePolicy2(values),
  KOKKOS_LAMBDA (const int i0, const int i1) {
    auto & entry = values(i0, i1);
    const int tDim = spaceDim;
    const auto t = devicePoints(i0,i1,tDim);
    const double value = pow(t,n);
    assignValue(entry, modType, value);
  });
  points.checkinDeviceView(true);
}

TFunctionPtr<double> Tn::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Tn::dy()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Tn::dz()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Tn::dt()
{
  if (_n == 0)
  {
    return TFunction<double>::zero();
  }
  else if (_n == 1)
  {
    return TFunction<double>::constant(1.0);
  }
  TFunctionPtr<double> t_n_minus = Teuchos::rcp( new Tn(_n-1) );
  return (double)_n * t_n_minus;
}

string MonomialFunction::displayString()
{
  ostringstream ss;
  if ((_n != 1) && (_n != 0))
  {
    ss << "x_" << _comp << "^" << _n ;
  }
  else if (_n == 1)
  {
    ss << "x_" << _comp;
  }
  else
  {
    ss << "(1)";
  }
  return ss.str();
}

MonomialFunction::MonomialFunction(int comp, int n)
: TFunction<double>("MonomialFunction"),
_comp(comp),
_n(n)
{
  if (n == 0)
  {
    // constant
    setVariationDimensions(VARIES_IN_NO_DIMENSION);
  }
  else
  {
    int variationDimensions = VARIES_IN_DIMENSION_0 << comp; // will give VARIES_IN_DIMENSION_1 for comp == 1, etc.
    setVariationDimensions(variationDimensions);
  }
}

void MonomialFunction::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("MonomialFunction values()");
  const auto & points = basisCache->getPhysicalCubaturePointsFC(); // (C,P,D) FieldContainer
  auto devicePoints = points.checkoutDeviceView();
  const int n = _n;
  auto comp = _comp;
  Kokkos::parallel_for("MonomialFunction values() functor", Camellia::FieldContainer<double>::getMDRangePolicy2(values),
  KOKKOS_LAMBDA (const int i0, const int i1) {
    auto & entry = values(i0, i1);
    const auto xi = devicePoints(i0,i1,comp);
    const double value = pow(xi,n);
    assignValue(entry, modType, value);
  });
  points.checkinDeviceView(true);
}

void MonomialFunction::values(Intrepid2::Data<double, DeviceType> &values, BasisCachePtr basisCache)
{
  if (_n==0)
  {
    // underlying view should have rank 1, and just one entry
    Kokkos::deep_copy(values.getUnderlyingView<1>(), 1.0);
    return;
  }
  
  const auto & physicalTensorPoints = basisCache->getPhysicalCubaturePoints();
  
  int numPointComponents = physicalTensorPoints.numTensorComponents();
  int tensorComponentOrdinalForMyComp = -1;
  int numDimsSoFar = 0;
  int componentDimIndex = -1; // the spatial dimension index to use in the component point container
  for (int r=0; r<numPointComponents; r++)
  {
    const int componentDims = physicalTensorPoints.getTensorComponent(r).extent_int(2);  // shape: (C,P,D)
    if (componentDims + numDimsSoFar >= _comp + 1)
    {
      tensorComponentOrdinalForMyComp = r;
      componentDimIndex = _comp - numDimsSoFar;
      break;
    }
    numDimsSoFar += componentDims;
  }
  TEUCHOS_TEST_FOR_EXCEPTION(tensorComponentOrdinalForMyComp == -1, std::invalid_argument, "points container is of too low a dimension");
  
  auto pointComponent = physicalTensorPoints.getTensorComponent(tensorComponentOrdinalForMyComp);
  auto underlyingData = values.getUnderlyingView<2>(); // (C,Pn)
  
  using ExecutionSpace = Camellia::DeviceExecutionSpace;
  Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{underlyingData.extent(0),underlyingData.extent(1)});
  
  // avoid implicit references to this
  const int n = _n;
  
  Kokkos::parallel_for("MonomialFunction values(Data) functor", policy,
  KOKKOS_LAMBDA (const int &cellOrdinal, const int &pointCompOrdinal) {
    auto & result = underlyingData(cellOrdinal, pointCompOrdinal);
    const auto & xi = pointComponent(cellOrdinal, pointCompOrdinal, componentDimIndex);
    const double value = pow(xi,n);
    result = value;
  });
}

TFunctionPtr<double> MonomialFunction::dx()
{
  return di(1);
}
TFunctionPtr<double> MonomialFunction::dy()
{
  return di(2);
}
TFunctionPtr<double> MonomialFunction::dz()
{
  return di(3);
}
TFunctionPtr<double> MonomialFunction::di(int i)
{
  if ((_n == 0) || (i != _comp+1))
  {
    return TFunction<double>::zero();
  }
  else if (_n == 1)
  {
    return TFunction<double>::constant(1.0);
  }
  TFunctionPtr<double> xi_n_minus = Teuchos::rcp( new MonomialFunction(_comp,_n-1) );
  return (double)_n * xi_n_minus;
}

int MonomialFunction::n() const
{
  return _n;
}

TFunctionPtr<double> MonomialFunction::xi_n(int i, int n)
{
  return Teuchos::rcp(new MonomialFunction(i,n));
}
