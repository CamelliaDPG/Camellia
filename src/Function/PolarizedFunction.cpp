//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
#include "PolarizedFunction.h"

#include "BasisCache.h"
#include "MonomialFunctions.h"
#include "PhysicalPointCache.h"
#include "TrigFunctions.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
PolarizedFunction<Scalar>::PolarizedFunction( TFunctionPtr<Scalar> f_of_xAsR_yAsTheta ) : TFunction<Scalar>("PolarizedFunction", f_of_xAsR_yAsTheta->rank())
{
  _f = f_of_xAsR_yAsTheta;
}

template <typename Scalar>
std::vector< TFunctionPtr<Scalar> > PolarizedFunction<Scalar>::memberFunctions()
{
  return {{_f}};
}

template <typename Scalar>
Teuchos::RCP<PolarizedFunction<double>> PolarizedFunction<Scalar>::r()
{
  static Teuchos::RCP<PolarizedFunction<Scalar>> _r = Teuchos::rcp( new PolarizedFunction<Scalar>( Teuchos::rcp( new Xn(1) ) ) );
  return _r;
}

template <typename Scalar>
Teuchos::RCP<PolarizedFunction<double>> PolarizedFunction<Scalar>::sin_theta()
{
  static Teuchos::RCP<PolarizedFunction<Scalar>> _sin_theta = Teuchos::rcp( new PolarizedFunction<Scalar>( Teuchos::rcp( new Sin_y ) ) );
  return _sin_theta;
}

template <typename Scalar>
Teuchos::RCP<PolarizedFunction<double>> PolarizedFunction<Scalar>::cos_theta()
{
  static Teuchos::RCP<PolarizedFunction<Scalar>> _cos_theta = Teuchos::rcp( new PolarizedFunction<Scalar>( Teuchos::rcp( new Cos_y ) ) );
  return _cos_theta;
}

void findAndReplace(string &str, const string &findStr, const string &replaceStr)
{
  size_t found = str.find( findStr );
  while (found!=string::npos)
  {
    str.replace( found, findStr.length(), replaceStr );
    found = str.find( findStr );
  }
}

template <typename Scalar>
string PolarizedFunction<Scalar>::displayString()
{
  string displayString = _f->displayString();
  findAndReplace(displayString, "x", "r");
  findAndReplace(displayString, "y", "\\theta");
  return displayString;
  //  ostringstream ss( _f->displayString());
  //  ss << "(r,\\theta)";
  //  return ss.str();
}

template <typename Scalar>
TFunctionPtr<Scalar> PolarizedFunction<Scalar>::dx()
{
  // cast everything to FunctionPtrs:
  TFunctionPtr<double> sin_theta_fxn = sin_theta();
  TFunctionPtr<Scalar> dtheta_fxn = dtheta();
  TFunctionPtr<Scalar> dr_fxn = dr();
  TFunctionPtr<double> r_fxn = r();
  TFunctionPtr<double> cos_theta_fxn = cos_theta();
  return dr_fxn * cos_theta_fxn - dtheta_fxn * sin_theta_fxn / r_fxn;
}

template <typename Scalar>
TFunctionPtr<Scalar> PolarizedFunction<Scalar>::dy()
{
  TFunctionPtr<double> sin_theta_fxn = sin_theta();
  TFunctionPtr<Scalar> dtheta_fxn = dtheta();
  TFunctionPtr<Scalar> dr_fxn = dr();
  TFunctionPtr<double> r_fxn = r();
  TFunctionPtr<double> cos_theta_fxn = cos_theta();
  return dr_fxn * sin_theta_fxn + dtheta_fxn * cos_theta_fxn / r_fxn;
}

template <typename Scalar>
Teuchos::RCP<PolarizedFunction<Scalar>> PolarizedFunction<Scalar>::dtheta()
{
  return Teuchos::rcp( new PolarizedFunction<Scalar>( _f->dy() ) );
}

template <typename Scalar>
Teuchos::RCP<PolarizedFunction<Scalar>> PolarizedFunction<Scalar>::dr()
{
  return Teuchos::rcp( new PolarizedFunction<Scalar>( _f->dx() ) );
}

template <typename Scalar>
bool PolarizedFunction<Scalar>::isZero()
{
  return _f->isZero();
}

template <typename Scalar>
void PolarizedFunction<Scalar>::values(typename FieldContainer<Scalar>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("PolarizedFunction values()");
  const double PI = 3.141592653589793238462;
  const Camellia::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePointsFC());
  Camellia::FieldContainer<double> polarPoints = *points;
  auto dv_polarPoints = polarPoints.checkoutDeviceView();
  TEUCHOS_TEST_FOR_EXCEPTION(dv_polarPoints.rank() != 3, std::invalid_argument, "rank is not equal to 3");
  using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<2> >::array_index_type;
  Kokkos::parallel_for("PolarizedFunction_values", FieldContainer<Scalar>::getMDRangePolicy2(dv_polarPoints),
  KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1)
  {
    double x = dv_polarPoints(i0,i1,0);
    double y = dv_polarPoints(i0,i1,1);
    double r = sqrt(x * x + y * y);
    double theta = (r != 0) ? acos(x/r) : 0;
    // now x = r cos(theta), but need to guarantee that y = r sin(theta); (might differ in sign)
    // according to the acos docs, theta will be in [0, pi], so the rule is: (y < 0) ==> theta := 2 pi - theta;
    if (y < 0) theta = 2*PI-theta;
    dv_polarPoints(i0,i1,0) = r;
    dv_polarPoints(i0,i1,1) = theta;
  });
  polarPoints.checkinDeviceView(true);
  BasisCachePtr dummyBasisCache = Teuchos::rcp( new PhysicalPointCache( polarPoints ) );
  _f->values(values, modType, dummyBasisCache);
  if (_f->isZero())
  {
    cout << "Warning: in PolarizedFunction, we are being asked for values when _f is zero. This shouldn't happen.\n";
  }
}

namespace Camellia
{
template class PolarizedFunction<double>;
}

