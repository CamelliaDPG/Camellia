//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "MinMaxFunctions.h"

#include "BasisCache.h"

using namespace Camellia;
using std::fmin;
using std::fmax;

MinFunction::MinFunction(TFunctionPtr<double> f1, TFunctionPtr<double> f2) : TFunction<double>(f1->rank())
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

void MinFunction::values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  Camellia::FieldContainer<double> values2(values);
  _f1->values(values,basisCache);
  _f2->values(values2,basisCache);
  
  auto dv = values.checkoutDeviceView();
  auto dv2 = values2.checkoutDeviceView();
  
  const int dim0 = (dv.rank() > 0) ? dv.extent(0) : 1;
  const int dim1 = (dv.rank() > 1) ? dv.extent(1) : 1;
  const int dim2 = (dv.rank() > 2) ? dv.extent(2) : 1;
  const int dim3 = (dv.rank() > 3) ? dv.extent(3) : 1;
  const int dim4 = (dv.rank() > 4) ? dv.extent(4) : 1;
  const int dim5 = (dv.rank() > 5) ? dv.extent(5) : 1;
  const int dim6 = (dv.rank() > 6) ? dv.extent(6) : 1;
  
  Kokkos::parallel_for("MinFunction_values", dim0, KOKKOS_LAMBDA (const int i0) {
    for (int i1 = 0; i1 < dim1; i1++) {
      for (int i2 = 0; i2 < dim2; i2++) {
        for (int i3 = 0; i3 < dim3; i3++) {
          for (int i4 = 0; i4 < dim4; i4++) {
            for (int i5 = 0; i5 < dim5; i5++) {
              for (int i6 = 0; i6 < dim6; i6++) {
                dv.access(i0,i1,i2,i3,i4,i5,i6) = fmin(dv.access(i0,i1,i2,i3,i4,i5,i6),dv2.access(i0,i1,i2,i3,i4,i5,i6));
              }
            }
          }
        }
      }
    }
  });
  values.checkinDeviceView();
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

MaxFunction::MaxFunction(TFunctionPtr<double> f1, TFunctionPtr<double> f2) : TFunction<double>(f1->rank())
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

void MaxFunction::values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  Camellia::FieldContainer<double> values2(values);
  _f1->values(values,basisCache);
  _f2->values(values2,basisCache);
  
  auto dv = values.checkoutDeviceView();
  auto dv2 = values2.checkoutDeviceView();
  
  const int dim0 = (dv.rank() > 0) ? dv.extent(0) : 1;
  const int dim1 = (dv.rank() > 1) ? dv.extent(1) : 1;
  const int dim2 = (dv.rank() > 2) ? dv.extent(2) : 1;
  const int dim3 = (dv.rank() > 3) ? dv.extent(3) : 1;
  const int dim4 = (dv.rank() > 4) ? dv.extent(4) : 1;
  const int dim5 = (dv.rank() > 5) ? dv.extent(5) : 1;
  const int dim6 = (dv.rank() > 6) ? dv.extent(6) : 1;
  
  Kokkos::parallel_for("MaxFunction_values", dim0, KOKKOS_LAMBDA (const int i0) {
    for (int i1 = 0; i1 < dim1; i1++) {
      for (int i2 = 0; i2 < dim2; i2++) {
        for (int i3 = 0; i3 < dim3; i3++) {
          for (int i4 = 0; i4 < dim4; i4++) {
            for (int i5 = 0; i5 < dim5; i5++) {
              for (int i6 = 0; i6 < dim6; i6++) {
                dv.access(i0,i1,i2,i3,i4,i5,i6) = fmax(dv.access(i0,i1,i2,i3,i4,i5,i6),dv2.access(i0,i1,i2,i3,i4,i5,i6));
              }
            }
          }
        }
      }
    }
  });
  values.checkinDeviceView();
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
