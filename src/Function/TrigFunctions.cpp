#include "TrigFunctions.h"

// TODO: move this to TrigFunctions.h/cpp

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

string Sin_y::displayString()
{
  return "\\sin y";
}

double Sin_y::value(double x, double y)
{
  return sin(y);
}
TFunctionPtr<double> Sin_y::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Sin_y::dy()
{
  return Teuchos::rcp( new Cos_y );
}
TFunctionPtr<double> Sin_y::dz()
{
  return TFunction<double>::zero();
}

string Cos_y::displayString()
{
  return "\\cos y";
}
double Cos_y::value(double x, double y)
{
  return cos(y);
}
TFunctionPtr<double> Cos_y::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Cos_y::dy()
{
  TFunctionPtr<double> sin_y = Teuchos::rcp( new Sin_y );
  return - sin_y;
}
TFunctionPtr<double> Cos_y::dz()
{
  return TFunction<double>::zero();
}

string Sin_x::displayString()
{
  return "\\sin x";
}

double Sin_x::value(double x, double y)
{
  return sin(x);
}
TFunctionPtr<double> Sin_x::dx()
{
  return Teuchos::rcp( new Cos_x );
}
TFunctionPtr<double> Sin_x::dy()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Sin_x::dz()
{
  return TFunction<double>::zero();
}

string Cos_x::displayString()
{
  return "\\cos x";
}
double Cos_x::value(double x, double y)
{
  return cos(x);
}
TFunctionPtr<double> Cos_x::dx()
{
  TFunctionPtr<double> sin_x = Teuchos::rcp( new Sin_x );
  return - sin_x;
}
TFunctionPtr<double> Cos_x::dy()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Cos_x::dz()
{
  return TFunction<double>::zero();
}

Cos_ax::Cos_ax(double a, double b)
{
  _a = a;
  _b = b;
}
double Cos_ax::value(double x)
{
  return cos( _a * x + _b);
}
TFunctionPtr<double> Cos_ax::dx()
{
  return -_a * (TFunctionPtr<double>) Teuchos::rcp(new Sin_ax(_a,_b));
}
TFunctionPtr<double> Cos_ax::dy()
{
  return TFunction<double>::zero();
}

string Cos_ax::displayString()
{
  ostringstream ss;
  ss << "\\cos( " << _a << " x )";
  return ss.str();
}

Cos_ay::Cos_ay(double a)
{
  _a = a;
}
double Cos_ay::value(double x, double y)
{
  return cos( _a * y );
}
TFunctionPtr<double> Cos_ay::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Cos_ay::dy()
{
  return -_a * (TFunctionPtr<double>) Teuchos::rcp(new Sin_ay(_a));
}

string Cos_ay::displayString()
{
  ostringstream ss;
  ss << "\\cos( " << _a << " y )";
  return ss.str();
}


Sin_ax::Sin_ax(double a, double b)
{
  _a = a;
  _b = b;
}
double Sin_ax::value(double x)
{
  return sin( _a * x + _b);
}
TFunctionPtr<double> Sin_ax::dx()
{
  return _a * (TFunctionPtr<double>) Teuchos::rcp(new Cos_ax(_a,_b));
}
TFunctionPtr<double> Sin_ax::dy()
{
  return TFunction<double>::zero();
}
string Sin_ax::displayString()
{
  ostringstream ss;
  ss << "\\sin( " << _a << " x )";
  return ss.str();
}

Sin_ay::Sin_ay(double a)
{
  _a = a;
}
double Sin_ay::value(double x, double y)
{
  return sin( _a * y);
}
TFunctionPtr<double> Sin_ay::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Sin_ay::dy()
{
  return _a * (TFunctionPtr<double>) Teuchos::rcp(new Cos_ay(_a));
}
string Sin_ay::displayString()
{
  ostringstream ss;
  ss << "\\sin( " << _a << " y )";
  return ss.str();
}
