//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "MinMaxFunctions.h"

#include "BasisCache.h"

using namespace Camellia;
using std::fmin;
using std::fmax;

MinFunction::MinFunction(TFunctionPtr<double> f1, TFunctionPtr<double> f2) : TFunction<double>("MinFunction", f1->rank())
{
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "both functions must be of like rank.");
  _f1 = f1;
  _f2 = f2;
}

bool MinFunction::boundaryValueOnly()
{
  // if either summand is BVO, then so is the min...
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

std::string MinFunction::displayString()
{
  std::ostringstream ss;
  ss << "\\min( " << _f1->displayString() << " , " << _f2->displayString() << " )";
  return ss.str();
}

std::vector< TFunctionPtr<double> > MinFunction::memberFunctions()
{
  return {{_f1,_f2}};
}

void MinFunction::values(DeviceView<double> &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("MinFunction values()");
  if (modType == ASSIGN)
  {
    _f1->values(values, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _f2->values(values, ValueModificationType::MINIMUM, basisCache);
  }
  else
  {
    auto minValues = Allocation<double>::checkoutDeviceView(values,"MinFunction: minValues");
    _f1->values(minValues, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _f2->values(minValues, ValueModificationType::MINIMUM, basisCache);
    Kokkos::fence();
    assignValues<double>(values, modType, minValues);
    Allocation<double>::checkinDeviceView(minValues);
  }
}

TFunctionPtr<double> MinFunction::x()
{
  if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) )
  {
    return null();
  }
  return min(_f1->x(),_f2->x());
}

TFunctionPtr<double> MinFunction::y()
{
  if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) )
  {
    return null();
  }
  return min(_f1->y(),_f2->y());
}
TFunctionPtr<double> MinFunction::z()
{
  if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) )
  {
    return null();
  }
  return min(_f1->z(),_f2->z());
}

MaxFunction::MaxFunction(TFunctionPtr<double> f1, TFunctionPtr<double> f2) : TFunction<double>("MaxFunction", f1->rank())
{
  TEUCHOS_TEST_FOR_EXCEPTION( f1->rank() != f2->rank(), std::invalid_argument, "both functions must be of like rank.");
  _f1 = f1;
  _f2 = f2;
}

bool MaxFunction::boundaryValueOnly()
{
  // if either summand is BVO, then so is the max...
  return _f1->boundaryValueOnly() || _f2->boundaryValueOnly();
}

std::string MaxFunction::displayString()
{
  std::ostringstream ss;
  ss << "\\max( " << _f1->displayString() << " , " << _f2->displayString() << " )";
  return ss.str();
}

std::vector< TFunctionPtr<double> > MaxFunction::memberFunctions()
{
  return {{_f1,_f2}};
}

void MaxFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("MaxFunction values()");
  if (modType == ASSIGN)
  {
    _f1->values(values, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _f2->values(values, ValueModificationType::MAXIMUM, basisCache);
  }
  else
  {
    auto maxValues = Allocation<double>::checkoutDeviceView(values, "MaxFunction maxValues");
    _f1->values(maxValues, ValueModificationType::ASSIGN, basisCache);
    Kokkos::fence();
    _f2->values(maxValues, ValueModificationType::MAXIMUM, basisCache);
    Kokkos::fence();
    assignValues<double>(values, modType, maxValues);
    Allocation<double>::checkinDeviceView(maxValues);
  }
}

TFunctionPtr<double> MaxFunction::x()
{
  if ( (_f1->x().get() == NULL) || (_f2->x().get() == NULL) )
  {
    return null();
  }
  return max(_f1->x(),_f2->x());
}

TFunctionPtr<double> MaxFunction::y()
{
  if ( (_f1->y().get() == NULL) || (_f2->y().get() == NULL) )
  {
    return null();
  }
  return max(_f1->y(),_f2->y());
}
TFunctionPtr<double> MaxFunction::z()
{
  if ( (_f1->z().get() == NULL) || (_f2->z().get() == NULL) )
  {
    return null();
  }
  return max(_f1->z(),_f2->z());
}
