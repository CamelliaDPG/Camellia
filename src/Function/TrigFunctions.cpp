//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "TrigFunctions.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace std;

template<class Scalar>
class Sin : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _arg;
public:
  Sin(TFunctionPtr<Scalar> arg) : TFunction<Scalar>("Sin",0) // Sin is scalar-valued
  {
    _arg = arg;
  }
  string displayString()
  {
    ostringstream name;
    name << "sin(" << _arg->displayString() << ")";
    return name.str();
  }
  
  virtual void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
  {
    this->CHECK_VALUES_RANK(values);
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    _arg->values(values, basisCache);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        values(cellOrdinal,ptOrdinal) = std::sin(values(cellOrdinal,ptOrdinal));
      }
    }
  }
  
  TFunctionPtr<Scalar> dx()
  {
    // chain rule
    return TrigFunctions<Scalar>::cos(_arg) * _arg->dx();
  }
  
  TFunctionPtr<Scalar> dy()
  {
    // chain rule
    return TrigFunctions<Scalar>::cos(_arg) * _arg->dy();
  }
  
  TFunctionPtr<Scalar> dz()
  {
    // chain rule
    return TrigFunctions<Scalar>::cos(_arg) * _arg->dz();
  }
  
  TFunctionPtr<Scalar> dt()
  {
    // chain rule
    return TrigFunctions<Scalar>::cos(_arg) * _arg->dt();
  }
};

template<class Scalar>
class Cos : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _arg;
public:
  Cos(TFunctionPtr<Scalar> arg) : TFunction<Scalar>("Cos",0) // Cos is scalar-valued
  {
    _arg = arg;
  }
  string displayString()
  {
    ostringstream name;
    name << "cos(" << _arg->displayString() << ")";
    return name.str();
  }
  virtual void values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
  {
    this->CHECK_VALUES_RANK(values);
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    _arg->values(values, basisCache);
    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        values(cellOrdinal,ptOrdinal) = std::cos(values(cellOrdinal,ptOrdinal));
      }
    }
  }
  
  TFunctionPtr<Scalar> dx()
  {
    // chain rule
    return -TrigFunctions<Scalar>::sin(_arg) * _arg->dx();
  }
  
  TFunctionPtr<Scalar> dy()
  {
    // chain rule
    return -TrigFunctions<Scalar>::sin(_arg) * _arg->dy();
  }
  
  TFunctionPtr<Scalar> dz()
  {
    // chain rule
    return -TrigFunctions<Scalar>::sin(_arg) * _arg->dz();
  }
  
  TFunctionPtr<Scalar> dt()
  {
    // chain rule
    return -TrigFunctions<Scalar>::sin(_arg) * _arg->dt();
  }
};

string Sin_y::displayString()
{
  return "\\sin y";
}

double Sin_y::value(double x, double y)
{
  return std::sin(y);
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
  return std::cos(y);
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
  return std::sin(x);
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
  return std::cos(x);
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
: SimpleFunction<double>("Cos_ax")
{
  _a = a;
  _b = b;
}
double Cos_ax::value(double x)
{
  return std::cos( _a * x + _b);
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
: SimpleFunction<double>("Cos_ay")
{
  _a = a;
}
double Cos_ay::value(double x, double y)
{
  return std::cos( _a * y );
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
: SimpleFunction<double>("Sin_ax")
{
  _a = a;
  _b = b;
}
double Sin_ax::value(double x)
{
  return std::sin( _a * x + _b);
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
: SimpleFunction<double>("Sin_ay")
{
  _a = a;
}
double Sin_ay::value(double x, double y)
{
  return std::sin( _a * y);
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

ArcTan_ax::ArcTan_ax(double a, double b)
: SimpleFunction<double>("ArcTan_ax")
{
  _a = a;
  _b = b;
}
double ArcTan_ax::value(double x)
{
  return atan( _a * x + _b);
}
TFunctionPtr<double> ArcTan_ax::dx()
{
  TFunctionPtr<double> one = TFunction<double>::constant(1);
  TFunctionPtr<double> x2  = TFunction<double>::xn(2);
  return _a * one/(x2 + one);
}
TFunctionPtr<double> ArcTan_ax::dy()
{
  return TFunction<double>::zero();
}
string ArcTan_ax::displayString()
{
  ostringstream ss;
  ss << "\\atan( " << _a << " x )";
  return ss.str();
}

ArcTan_ay::ArcTan_ay(double a, double b)
: SimpleFunction<double>("ArcTan_ay")
{
  _a = a;
  _b = b;
}
double ArcTan_ay::value(double x, double y)
{
  return atan( _a * y + _b);
}
TFunctionPtr<double> ArcTan_ay::dx()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> ArcTan_ay::dy()
{
  TFunctionPtr<double> one = TFunction<double>::constant(1);
  TFunctionPtr<double> y2  = TFunction<double>::yn(2);
  return _a * one/(y2 + one);
}
string ArcTan_ay::displayString()
{
  ostringstream ss;
  ss << "\\atan( " << _a << " y )";
  return ss.str();
}

template<class Scalar>
TFunctionPtr<Scalar> TrigFunctions<Scalar>::sin(TFunctionPtr<Scalar> argument)
{
  return Teuchos::rcp(new Sin<Scalar>(argument));
}

template<class Scalar>
TFunctionPtr<Scalar> TrigFunctions<Scalar>::cos(TFunctionPtr<Scalar> argument)
{
  return Teuchos::rcp(new Cos<Scalar>(argument));
}

// ETIs below
namespace Camellia {
  template class TrigFunctions<double>;
}
