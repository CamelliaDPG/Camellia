//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "Function.h"
#include "MeshPolyOrderFunction.h"
#include "BasisCache.h"

using namespace Camellia;

void MeshPolyOrderFunction::values(Camellia::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  vector<GlobalIndexType> cellIDs = basisCache->cellIDs();
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  TEUCHOS_TEST_FOR_EXCEPTION(dv.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  int numPoints = dv.extent(1);
  
  Camellia::FieldContainer<int> cellPolyOrder(cellIDs.size());
  for (int i = 0; i < cellIDs.size(); i++) { cellPolyOrder(i) = _mesh->cellPolyOrder(cellIDs[i]); }
  auto dv_cellPolyOrder = cellPolyOrder.checkoutDeviceView();
  
  Kokkos::parallel_for("MeshPolyOrderFunction_values", cellIDs.size(), KOKKOS_LAMBDA (const int cellIndex) {
    int polyOrder = dv_cellPolyOrder(cellIndex);
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      dv(cellIndex, ptIndex) = polyOrder-1.0;
    }
  });
  cellPolyOrder.checkinDeviceView(false);
  values.checkinDeviceView();
}
