//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ExpFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using std::log;

template<class Scalar>
Exp<Scalar>::Exp(TFunctionPtr<Scalar> power) : TFunction<Scalar>("Exp",0), _power(power)
{}

template<class Scalar>
void Exp<Scalar>::values(FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

template<class Scalar>
void Exp<Scalar>::values(typename FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto rangePolicy = FieldContainer<Scalar>::getMDRangePolicy2(values);
  if (modType == ASSIGN)
  {
    _power->values(values, ASSIGN, basisCache);
    Kokkos::fence();
    Kokkos::parallel_for("Exp_values", rangePolicy,
    KOKKOS_LAMBDA (const int i0, const int i1) {
      values(i0,i1) = std::exp(values(i0,i1));
    });
  }
  else
  {
    auto dv = basisCache->checkoutScratchStorage<Scalar>(values);
    _power->values(dv, ASSIGN, basisCache);
    Kokkos::fence();
    Kokkos::parallel_for("Exp_values", rangePolicy,
    KOKKOS_LAMBDA (const int i0, const int i1) {
      assignValue(values(i0,i1), modType, std::exp(dv(i0,i1)));
    });
    basisCache->checkinScratchStorage<Scalar>(dv);
  }
}

/*
 d/dx (e^(f)) = e^(f) * d/dx(f)
 */

template<class Scalar>
TFunctionPtr<Scalar> Exp<Scalar>::dx()
{
  TFunctionPtr<Scalar> thisPtr = Teuchos::rcp( new Exp(_power) );
  return thisPtr * _power->dx();
}

template<class Scalar>
TFunctionPtr<Scalar> Exp<Scalar>::dy()
{
  TFunctionPtr<Scalar> thisPtr = Teuchos::rcp( new Exp(_power) );
  return thisPtr * _power->dy();
}


template<class Scalar>
TFunctionPtr<Scalar> Exp<Scalar>::dz()
{
  TFunctionPtr<Scalar> thisPtr = Teuchos::rcp( new Exp(_power) );
  return thisPtr * _power->dz();
}

template<class Scalar>
std::string Exp<Scalar>::displayString()
{
  std::ostringstream str;
  str << "e^{" << _power->displayString() << "}";
  return str.str();
}

template<class Scalar>
std::vector< TFunctionPtr<Scalar> > Exp<Scalar>::memberFunctions()
{
  return {{_power}};
}

template<class Scalar>
TFunctionPtr<Scalar> Exp<Scalar>::exp(TFunctionPtr<Scalar> f)
{
  return Teuchos::rcp(new Exp<Scalar>(f));
}

Exp_x::Exp_x()
:
SimpleFunction<double>("Exp_x")
{}

string Exp_x::displayString()
{
  return "e^x";
}
double Exp_x::value(double x, double y)
{
  return exp(x);
}
TFunctionPtr<double> Exp_x::dx()
{
  return Teuchos::rcp( new Exp_x );
}
TFunctionPtr<double> Exp_x::dy()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_x::dz()
{
  return Function::zero();
}

Exp_y::Exp_y()
:
SimpleFunction<double>("Exp_y")
{}
string Exp_y::displayString()
{
  return "e^y";
}
double Exp_y::value(double x, double y)
{
  return exp(y);
}
TFunctionPtr<double> Exp_y::dx()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_y::dy()
{
  return Teuchos::rcp( new Exp_y );
}
TFunctionPtr<double> Exp_y::dz()
{
  return Function::zero();
}

Exp_z::Exp_z()
:
SimpleFunction<double>("Exp_z")
{}

string Exp_z::displayString()
{
  return "e^z";
}
double Exp_z::value(double x, double y, double z)
{
  return exp(z);
}
TFunctionPtr<double> Exp_z::dx()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_z::dy()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_z::dz()
{
  return Teuchos::rcp( new Exp_z );
}

Exp_ax::Exp_ax(double a)
: SimpleFunction<double>("Exp_ax")
{
  _a = a;
}
double Exp_ax::value(double x, double y)
{
  return exp( _a * x);
}
TFunctionPtr<double> Exp_ax::dx()
{
  return _a * (TFunctionPtr<double>) Teuchos::rcp(new Exp_ax(_a));
}
TFunctionPtr<double> Exp_ax::dy()
{
  return Function::zero();
}
std::string Exp_ax::displayString()
{
  std::ostringstream ss;
  ss << "\\exp( " << _a << " x )";
  return ss.str();
}

Exp_ay::Exp_ay(double a)
: SimpleFunction<double>("Exp_ay")
{
  _a = a;
}
double Exp_ay::value(double x, double y)
{
  return exp( _a * y);
}
TFunctionPtr<double> Exp_ay::dx()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_ay::dy()
{
  return _a * (TFunctionPtr<double>) Teuchos::rcp(new Exp_ay(_a));
}
std::string Exp_ay::displayString()
{
  std::ostringstream ss;
  ss << "\\exp( " << _a << " y )";
  return ss.str();
}

Exp_at::Exp_at(double a)
: SimpleFunction<double>("Exp_at")
{
  _a = a;
}
double Exp_at::value(double x, double t)
{
  return exp( _a * t);
}
double Exp_at::value(double x, double y, double t)
{
  return exp( _a * t);
}
double Exp_at::value(double x, double y, double z, double t)
{
  return exp( _a * t);
}
TFunctionPtr<double> Exp_at::dx()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_at::dy()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_at::dz()
{
  return Function::zero();
}
TFunctionPtr<double> Exp_at::dt()
{
  return _a * (TFunctionPtr<double>) Teuchos::rcp(new Exp_at(_a));
}
std::string Exp_at::displayString()
{
  std::ostringstream ss;
  ss << "\\exp( " << _a << " t )";
  return ss.str();
}

template<class Scalar>
Ln<Scalar>::Ln(TFunctionPtr<Scalar> argument)
:
TFunction<Scalar>("Ln"),
_argument(argument)
{}

template<class Scalar>
void Ln<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

template<class Scalar>
void Ln<Scalar>::values(typename Camellia::FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto dv = basisCache->checkoutScratchStorage<Scalar>(values);
  _argument->values(dv, ASSIGN, basisCache);
  Kokkos::fence();
  const int dim0 = values.extent_int(0);
  const int dim1 = values.extent_int(1);
  Kokkos::parallel_for("Ln_values", dim0, KOKKOS_LAMBDA (const int i0) {
    for (int i1 = 0; i1 < dim1; i1++) {
      assignValue(values(i0,i1), modType, log(dv(i0,i1)));
    }
  });
  basisCache->checkinScratchStorage<Scalar>(dv);
}

/*
 d/dx (ln(f)) = d/dx(f) / f
 */

template<class Scalar>
TFunctionPtr<Scalar> Ln<Scalar>::dx()
{
  return _argument->dx() / _argument;
}

template<class Scalar>
TFunctionPtr<Scalar> Ln<Scalar>::dy()
{
  return _argument->dy() / _argument;
}

template<class Scalar>
TFunctionPtr<Scalar> Ln<Scalar>::dz()
{
  return _argument->dy() / _argument;
}

template<class Scalar>
std::string Ln<Scalar>::displayString()
{
  std::ostringstream str;
  str << "{\\rm ln} (" << _argument->displayString() << ")";
  return str.str();
}

template<class Scalar>
std::vector< TFunctionPtr<Scalar> > Ln<Scalar>::memberFunctions()
{
  return {{_argument}};
}

// explicitly instantiate Exp<double>, Ln<double> classes:
template class Camellia::Exp<double>;
template class Camellia::Ln<double>;
