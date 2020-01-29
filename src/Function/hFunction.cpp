//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "hFunction.h"

#include "BasisCache.h"

using namespace Camellia;

string hFunction::displayString()
{
  return "h";
}

void hFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

void hFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  const int numCells  = values.extent_int(0);
  const int numPoints = values.extent_int(1);
  const int dimension = basisCache->cellTopology()->getDimension();
  Camellia::FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
  auto dv_cellMeasures = cellMeasures.checkoutDeviceView();
  Kokkos::parallel_for("hFunction_values", numCells, KOKKOS_LAMBDA (const int cellIndex) {
    double h = pow(dv_cellMeasures(cellIndex), 1.0 / dimension);
    for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
      assignValue(values(cellIndex,ptIndex), modType, h);
    }
  });
  cellMeasures.checkinDeviceView(false);
}
