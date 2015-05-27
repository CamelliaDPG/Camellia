#include "ExpFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

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