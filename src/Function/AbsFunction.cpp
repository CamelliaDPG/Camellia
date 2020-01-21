//
//  AbsFunction.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/12/16.
//
//

#include "AbsFunction.h"

using namespace Camellia;
using std::fabs;

AbsFunction::AbsFunction(TFunctionPtr<double> f) : TFunction<double>(f->rank())
{
  TEUCHOS_TEST_FOR_EXCEPTION( f->rank() != 0, std::invalid_argument, "AbsFunction only supports Functions of rank 0.");
  _f = f;
}

bool AbsFunction::boundaryValueOnly()
{
  // if f is BVO, then so is the abs...
  return _f->boundaryValueOnly();
}

string AbsFunction::displayString()
{
  ostringstream ss;
  ss << "|" << _f->displayString() << "|";
  return ss.str();
}

std::vector< TFunctionPtr<double> > AbsFunction::memberFunctions()
{
  return {{_f}};
}

void AbsFunction::values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  _f->values(values,basisCache);
  auto dv = values.checkoutDeviceView();
  const int dim0 = dv.extent(0);
  const int dim1 = dv.extent(1);
  Kokkos::parallel_for("AbsFunction_values", dim0, KOKKOS_LAMBDA (const int i0) {
    for (int i1 = 0; i1 < dim1; i1++) {
      dv(i0,i1) = fabs(dv(i0,i1));
    }
  });
  values.checkinDeviceView();
}

TFunctionPtr<double> AbsFunction::dx()
{
  // undefined
  return null();
}

TFunctionPtr<double> AbsFunction::dy()
{
  // undefined
  return null();
}
TFunctionPtr<double> AbsFunction::dz()
{
  // undefined
  return null();
}
