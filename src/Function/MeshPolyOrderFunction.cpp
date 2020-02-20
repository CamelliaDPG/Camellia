//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Function.h"
#include "MeshPolyOrderFunction.h"
#include "BasisCache.h"

using namespace Camellia;

void MeshPolyOrderFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  TEUCHOS_FUNC_TIME_MONITOR("MeshPolyOrderFunction values()");
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  Camellia::FieldContainer<int> cellPolyOrder(cellIDs.size());
  for (int i = 0; i < cellIDs.size(); i++) { cellPolyOrder(i) = _mesh->cellPolyOrder(cellIDs[i]); }
  auto dv_cellPolyOrder = cellPolyOrder.checkoutDeviceView();
  
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values);
  Kokkos::parallel_for("MeshPolyOrderFunction_values", rangePolicy,
  KOKKOS_LAMBDA (const int cellOrdinal, const int pointOrdinal) {
    int polyOrder = dv_cellPolyOrder(cellOrdinal);
    assignValue(values(cellOrdinal, pointOrdinal), modType, polyOrder-1.0);
  });
  cellPolyOrder.checkinDeviceView(false);
}
