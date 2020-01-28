//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "UnitNormalFunction.h"

#include "BasisCache.h"

using namespace Camellia;

UnitNormalFunction::UnitNormalFunction(int comp, bool spaceTime) : TFunction<double>("UnitNormalFunction", (comp==-1)? 1 : 0)
{
  _comp = comp;
  _spaceTime = spaceTime;
}

TFunctionPtr<double> UnitNormalFunction::x()
{
  return Teuchos::rcp( new UnitNormalFunction(0,_spaceTime) );
}

TFunctionPtr<double> UnitNormalFunction::y()
{
  return Teuchos::rcp( new UnitNormalFunction(1,_spaceTime) );
}

TFunctionPtr<double> UnitNormalFunction::z()
{
  return Teuchos::rcp( new UnitNormalFunction(2,_spaceTime) );
}

TFunctionPtr<double> UnitNormalFunction::t()
{
  TEUCHOS_TEST_FOR_EXCEPTION(!_spaceTime, std::invalid_argument, "t() component of normal only supported for space-time normals");
  return Teuchos::rcp( new UnitNormalFunction(-2,_spaceTime) );
}

bool UnitNormalFunction::boundaryValueOnly()
{
  return true;
}

string UnitNormalFunction::displayString()
{
  if (_comp == -1)
  {
    return " \\boldsymbol{n} ";
  }
  else
  {
    if (_comp == 0)
    {
      return " n_x ";
    }
    if (_comp == 1)
    {
      return " n_y ";
    }
    if (_comp == 2)
    {
      return " n_z ";
    }
    if (_comp == -2)
    {
      return " n_t ";
    }
    return "UnitNormalFunction with unexpected component";
  }
}

void UnitNormalFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

void UnitNormalFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  int spaceDim = basisCache->getSpaceDim();
  if (_comp == -1)
  {
    // check that the "D" dimension of values is correct:
    if (_spaceTime)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.extent_int(2) != spaceDim+1, std::invalid_argument, "For space-time normals, values.dimension(2) should be spaceDim + 1.");
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.extent_int(2) != spaceDim, std::invalid_argument, "For spatial normals, values.dimension(2) should be spaceDim.");
    }
  }
  const Camellia::FieldContainer<double> *sideNormals = _spaceTime ? &(basisCache->getSideNormalsSpaceTime()) : &(basisCache->getSideNormals());
  auto dv_sideNormals = sideNormals->checkoutDeviceView();
  const int comp = (_comp == -2) ? spaceDim : _comp;
  const bool spaceTime = _spaceTime;
  using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<2> >::array_index_type;
  Kokkos::parallel_for("UnitNormalFunction_values", FieldContainer<double>::getMDRangePolicy2(values),
  KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1)
  {
    if (comp == -1)
    {
      for (int i2 = 0; i2 < spaceDim; i2++)
      {
        assignValue(values(i0,i1,i2), modType, dv_sideNormals(i0,i1,i2));
      }
      if (spaceTime)
      {
        assignValue(values(i0,i1,spaceDim), modType, dv_sideNormals(i0,i1,spaceDim));
      }
    }
    else
    {
      assignValue(values(i0,i1), modType, dv_sideNormals(i0,i1,comp));
    }
  });
  sideNormals->checkinDeviceView();
}
