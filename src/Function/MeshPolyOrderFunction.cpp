//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Function.h"
#include "MeshPolyOrderFunction.h"
#include "BasisCache.h"

using namespace Camellia;

void MeshPolyOrderFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, basisCache);
  values.checkinDeviceView();
}

void MeshPolyOrderFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  Camellia::FieldContainer<int> cellPolyOrder(cellIDs.size());
  for (int i = 0; i < cellIDs.size(); i++) { cellPolyOrder(i) = _mesh->cellPolyOrder(cellIDs[i]); }
  auto dv_cellPolyOrder = cellPolyOrder.checkoutDeviceView();
  
  const int numPoints = values.extent_int(1);
  Kokkos::parallel_for("MeshPolyOrderFunction_values", cellIDs.size(), KOKKOS_LAMBDA (const int cellIndex) {
    int polyOrder = dv_cellPolyOrder(cellIndex);
    for (int ptIndex = 0; ptIndex < numPoints; ptIndex++) {
      assignValue(values(cellIndex, ptIndex), modType, polyOrder-1.0);
    }
  });
  cellPolyOrder.checkinDeviceView(false);
}
