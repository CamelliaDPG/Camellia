//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "TrigFunctions.h"

#include "BasisCache.h"

using namespace Camellia;
using std::sin;
using std::cos;

template<class Scalar>
class Sin : public TFunction<Scalar>
{
  TFunctionPtr<Scalar> _arg;
public:
  Sin(TFunctionPtr<Scalar> arg) : TFunction<Scalar>("Sin",0,arg->variationDimensions()) // Sin is scalar-valued
  {
    _arg = arg;
  }
  
  std::string displayString()
  {
    std::ostringstream name;
    name << "sin(" << _arg->displayString() << ")";
    return name.str();
  }
  
  using TFunction<Scalar>::values;
  virtual void values(FieldContainer<Scalar> &values, BasisCachePtr basisCache)
  {
    this->CHECK_VALUES_RANK(values);
    auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
    this->values(dv, ASSIGN, basisCache);
    values.checkinDeviceView(true);
  }
  
  virtual void values(typename FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
    auto dv = basisCache->checkoutScratchStorage<Scalar>(values);
    _arg->values(dv, ASSIGN, basisCache);
    auto rangePolicy = FieldContainer<Scalar>::getMDRangePolicy2(values);
    Kokkos::parallel_for("Sin_values", rangePolicy,
    KOKKOS_LAMBDA (const int i0, const int i1) {
      assignValue(values(i0,i1), modType, sin(dv(i0,i1)));
    });
    basisCache->checkinScratchStorage<Scalar>(dv);
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
  Cos(TFunctionPtr<Scalar> arg) : TFunction<Scalar>("Cos",0,arg->variationDimensions()) // Cos is scalar-valued
  {
    _arg = arg;
  }
  
  std::string displayString()
  {
    std::ostringstream name;
    name << "cos(" << _arg->displayString() << ")";
    return name.str();
  }
  
  using TFunction<Scalar>::values;
  virtual void values(FieldContainer<Scalar> &values, BasisCachePtr basisCache)
  {
    this->CHECK_VALUES_RANK(values);
    auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
    this->values(dv, ASSIGN, basisCache);
    values.checkinDeviceView(true);
  }
  
  virtual void values(typename FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
    auto dv = basisCache->checkoutScratchStorage<Scalar>(values);
    _arg->values(dv, ASSIGN, basisCache);
    auto rangePolicy = FieldContainer<Scalar>::getMDRangePolicy2(values);
    Kokkos::parallel_for("Cos_values", rangePolicy,
    KOKKOS_LAMBDA (const int i0, const int i1) {
      assignValue(values(i0,i1), modType, cos(dv(i0,i1)));
    });
    basisCache->checkinScratchStorage<Scalar>(dv);
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

std::string Sin_y::displayString()
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

std::string Cos_y::displayString()
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

std::string Sin_x::displayString()
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

std::string Cos_x::displayString()
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
: TFunction<double>("Cos_ax",0,VARIES_IN_DIMENSION_0)
{
  _a = a;
  _b = b;
}

void Cos_ax::values(Intrepid2::Data<double, Camellia::DeviceType> &values, BasisCachePtr basisCache)
{
  auto xPoints = basisCache->getPhysicalCubaturePoints().getTensorComponent(0); // xPoints might have more coordinates than just x, but it will certainly have x in the first entry
  
  auto underlyingData = values.template getUnderlyingView<2>(); // (C,P0) -- Again, it may be that data has some additional point dimensions, but the underlying view must support variation in both C and P0 (x coordinate) dimensions, and thus have rank exactly 2
  
  using ExecutionSpace = Camellia::DeviceExecutionSpace;
  Kokkos::MDRangePolicy<ExecutionSpace,Kokkos::Rank<2>> policy({0,0},{underlyingData.extent(0),underlyingData.extent(1)});
  
  // avoid implicit references to this
  const double a = _a;
  const double b = _b;
  
  const int xCoordIndex = 0; // in xPoints
  
  Kokkos::parallel_for("Cos_ax Data values", policy,
  KOKKOS_LAMBDA (const int &cellOrdinal, const int &xPointOrdinal) {
    auto & result = underlyingData(cellOrdinal, xPointOrdinal);
    
    const auto & x = xPoints(cellOrdinal,xPointOrdinal,xCoordIndex);
    result = std::cos(a * x + b);
  });
}

void Cos_ax::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  const Camellia::FieldContainer<double> &physicalPoints = basisCache->getPhysicalCubaturePointsFC();
  auto pointsView = physicalPoints.checkoutDeviceView(); // (C,P,D)
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values); // (C,P) dimension
  double a = this->_a;
  double b = this->_b;
  Kokkos::parallel_for("Cos_ax values", rangePolicy,
  KOKKOS_LAMBDA (const int cellOrdinal, const int pointOrdinal) {
    const double &x = pointsView(cellOrdinal,pointOrdinal,0);
    assignValue(values(cellOrdinal,pointOrdinal), modType, std::cos(a * x + b) );
  });
  physicalPoints.checkinDeviceView(false);
}

TFunctionPtr<double> Cos_ax::dx()
{
  return -_a * (TFunctionPtr<double>) Teuchos::rcp(new Sin_ax(_a,_b));
}
TFunctionPtr<double> Cos_ax::dy()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Cos_ax::dz()
{
  return TFunction<double>::zero();
}

std::string Cos_ax::displayString()
{
  std::ostringstream ss;
  ss << "\\cos( " << _a << " x )";
  return ss.str();
}

Cos_ay::Cos_ay(double a)
: SimpleFunction<double>("Cos_ay",VARIES_IN_DIMENSION_1)
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
TFunctionPtr<double> Cos_ay::dz()
{
  return TFunction<double>::zero();
}

std::string Cos_ay::displayString()
{
  std::ostringstream ss;
  ss << "\\cos( " << _a << " y )";
  return ss.str();
}


Sin_ax::Sin_ax(double a, double b)
: TFunction<double>("Sin_ax",0)
{
  _a = a;
  _b = b;
}

void Sin_ax::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  const Camellia::FieldContainer<double> &physicalPoints = basisCache->getPhysicalCubaturePointsFC();
  auto pointsView = physicalPoints.checkoutDeviceView(); // (C,P,D)
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values); // (C,P) dimension
  double a = this->_a;
  double b = this->_b;
  Kokkos::parallel_for("Cos_ax values", rangePolicy,
  KOKKOS_LAMBDA (const int cellOrdinal, const int pointOrdinal) {
    const double &x = pointsView(cellOrdinal,pointOrdinal,0);
    assignValue(values(cellOrdinal,pointOrdinal), modType, std::sin(a * x + b) );
  });
  physicalPoints.checkinDeviceView(false);
}

TFunctionPtr<double> Sin_ax::dx()
{
  return _a * (TFunctionPtr<double>) Teuchos::rcp(new Cos_ax(_a,_b));
}
TFunctionPtr<double> Sin_ax::dy()
{
  return TFunction<double>::zero();
}
TFunctionPtr<double> Sin_ax::dz()
{
  return TFunction<double>::zero();
}
std::string Sin_ax::displayString()
{
  std::ostringstream ss;
  ss << "\\sin( " << _a << " x )";
  return ss.str();
}

Sin_ay::Sin_ay(double a)
: SimpleFunction<double>("Sin_ay",VARIES_IN_DIMENSION_1)
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
TFunctionPtr<double> Sin_ay::dz()
{
  return TFunction<double>::zero();
}
std::string Sin_ay::displayString()
{
  std::ostringstream ss;
  ss << "\\sin( " << _a << " y )";
  return ss.str();
}

ArcTan_ax::ArcTan_ax(double a, double b)
: SimpleFunction<double>("ArcTan_ax",VARIES_IN_DIMENSION_0)
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
TFunctionPtr<double> ArcTan_ax::dz()
{
  return TFunction<double>::zero();
}
std::string ArcTan_ax::displayString()
{
  std::ostringstream ss;
  ss << "\\atan( " << _a << " x )";
  return ss.str();
}

ArcTan_ay::ArcTan_ay(double a, double b)
: SimpleFunction<double>("ArcTan_ay",VARIES_IN_DIMENSION_1)
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
TFunctionPtr<double> ArcTan_ay::dz()
{
  return TFunction<double>::zero();
}
std::string ArcTan_ay::displayString()
{
  std::ostringstream ss;
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
