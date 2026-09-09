//
//  AbsFunction.cpp
//  Camellia
//
//  Created by Nate Roberts on 9/12/16.
//
//

#include "AbsFunction.h"
#include "BasisCache.h"

using namespace Camellia;
using std::fabs;

AbsFunction::AbsFunction(TFunctionPtr<double> f) : TFunction<double>("AbsFunction",f->rank())
{
  TEUCHOS_TEST_FOR_EXCEPTION( f->rank() != 0, std::invalid_argument, "AbsFunction only supports Functions of rank 0.");
  _f = f;
}

FunctionPtr AbsFunction::abs(TFunctionPtr<double> f)
{
  return Teuchos::rcp(new AbsFunction(f));
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

void AbsFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("AbsFunction values()");
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  auto dv = SharedScratchStorage<double>::sharedScratchStorage().checkoutScratchStorageMatchingDeviceView(values, "AbsFunction dv");
  _f->values(dv, ASSIGN, basisCache);
  auto rangePolicy = Camellia::FieldContainer<double>::getMDRangePolicy2(values);
  Kokkos::parallel_for("AbsFunction_values", rangePolicy,
  KOKKOS_LAMBDA (const int i0, const int i1) {
    assignValue(values(i0,i1), modType, fabs(dv(i0,i1)));
  });
  SharedScratchStorage<double>::sharedScratchStorage().checkinScratchStorage(dv);
}

void AbsFunction::values(Intrepid2::Data<double, Camellia::DeviceType> &data, BasisCachePtr basisCache)
{
  _f->values(data, basisCache);
  
  auto absOp = KOKKOS_LAMBDA(const double &a) -> double
  {
#ifndef KOKKOS_ENABLE_CUDA
    using std::fabs;
#endif
    return fabs(a);
  };
  applyOperator(data,absOp);
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
