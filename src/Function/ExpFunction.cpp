//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ExpFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

template<class Scalar>
Exp<Scalar>::Exp(TFunctionPtr<Scalar> power) : _power(power)
{}

template<class Scalar>
void Exp<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  // first, fill values container with _power values:
  _power->values(values,basisCache);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      values(cellOrdinal,pointOrdinal) = std::exp(values(cellOrdinal,pointOrdinal));
    }
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
  ostringstream str;
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
string Exp_ax::displayString()
{
  ostringstream ss;
  ss << "\\exp( " << _a << " x )";
  return ss.str();
}

Exp_ay::Exp_ay(double a)
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
string Exp_ay::displayString()
{
  ostringstream ss;
  ss << "\\exp( " << _a << " y )";
  return ss.str();
}

Exp_at::Exp_at(double a)
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
string Exp_at::displayString()
{
  ostringstream ss;
  ss << "\\exp( " << _a << " t )";
  return ss.str();
}

template<class Scalar>
Ln<Scalar>::Ln(TFunctionPtr<Scalar> argument) : _argument(argument)
{}

template<class Scalar>
void Ln<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  // first, fill values container with _argument values:
  _argument->values(values,basisCache);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      values(cellOrdinal,pointOrdinal) = log(values(cellOrdinal,pointOrdinal));
    }
  }
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
  ostringstream str;
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
