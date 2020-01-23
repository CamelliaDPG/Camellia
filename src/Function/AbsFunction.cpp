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

AbsFunction::AbsFunction(TFunctionPtr<double> f) : TFunction<double>("AbsFunction",f->rank())
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

void AbsFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

void AbsFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto dv = FieldContainer<double>::getDeviceView(values);
  _f->values(dv, ASSIGN, basisCache);
  Kokkos::fence();
  const int dim0 = values.extent_int(0);
  const int dim1 = values.extent_int(1);
  Kokkos::parallel_for("AbsFunction_values", dim0, KOKKOS_LAMBDA (const int i0) {
    for (int i1 = 0; i1 < dim1; i1++) {
      assignValue(values(i0,i1), modType, fabs(dv(i0,i1)));
    }
  });
  FieldContainer<double>::recordDeviceViewDeallocation(dv);
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
