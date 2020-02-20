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
Xn::Xn(int n) : TFunction<double>("Xn")
{
  _n = n;
}

void Xn::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Xn values()");
  auto & points = basisCache->getPhysicalCubaturePoints(); // (C,P,D) FieldContainer
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
  points.checkinDeviceView();
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

string Yn::displayString()
{
  ostringstream ss;
  if ((_n != 1) && (_n != 0))
  {
    ss << "y^" << _n ;
  }
  else if (_n == 1)
  {
    ss << "y";
  }
  else
  {
    ss << "(1)";
  }
  return ss.str();
}
Yn::Yn(int n) : TFunction<double>("Yn")
{
  _n = n;
}

void Yn::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Yn values()");
  const auto & points = basisCache->getPhysicalCubaturePoints(); // (C,P,D) FieldContainer
  auto devicePoints = points.checkoutDeviceView();
  const int n = _n;
  Kokkos::parallel_for("Yn values() functor", Camellia::FieldContainer<double>::getMDRangePolicy2(values),
  KOKKOS_LAMBDA (const int i0, const int i1) {
    auto & entry = values(i0, i1);
    const int yDim = 1;
    const auto y = devicePoints(i0,i1,yDim);
    const double value = pow(y,n);
    assignValue(entry, modType, value);
  });
  points.checkinDeviceView();
}

TFunctionPtr<double> Yn::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Yn::dy()
{
  if (_n == 0)
  {
    return TFunction<double>::zero();
  }
  else if (_n == 1)
  {
    return TFunction<double>::constant(1.0);
  }
  TFunctionPtr<double> y_n_minus = Teuchos::rcp( new Yn(_n-1) );
  return (double)_n * y_n_minus;
}
TFunctionPtr<double> Yn::dz()
{
  return TFunction<double>::zero();
}
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
Zn::Zn(int n) : TFunction<double>("Zn")
{
  _n = n;
}

void Zn::values(typename Camellia::FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("Zn values()");
  const auto & points = basisCache->getPhysicalCubaturePoints(); // (C,P,D) FieldContainer
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
  points.checkinDeviceView();
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
  const auto & points = basisCache->getPhysicalCubaturePoints(); // (C,P,D) FieldContainer
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
  points.checkinDeviceView();
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
