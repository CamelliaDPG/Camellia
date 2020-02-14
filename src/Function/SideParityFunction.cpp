//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "SideParityFunction.h"

#include "BasisCache.h"

using namespace Camellia;

SideParityFunction::SideParityFunction() : TFunction<double>("SideParityFunction",0)
{
  //  cout << "SideParityFunction constructor.\n";
}

bool SideParityFunction::boundaryValueOnly()
{
  return true;
}

string SideParityFunction::displayString()
{
  return "sgn(n)";
}

void SideParityFunction::values(Camellia::FieldContainer<double> &values, BasisCachePtr sideBasisCache)
{
  this->CHECK_VALUES_RANK(values);
  auto dv = values.checkoutDeviceView(false); // false: don't deep_copy from host (will overwrite values)
  this->values(dv, ASSIGN, sideBasisCache);
  values.checkinDeviceView();
}

void SideParityFunction::values(typename FieldContainer<double>::DeviceView &values, ValueModificationType modType, BasisCachePtr sideBasisCache)
{
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2, std::invalid_argument, "rank is not equal to 2");
  const int numCells  = values.extent_int(0);
  const int sideIndex = sideBasisCache->getSideIndex();
  TEUCHOS_TEST_FOR_EXCEPTION(sideIndex == -1, std::invalid_argument, "non-sideBasisCache passed into SideParityFunction");
  
  // populate fc_parities on host
  FieldContainer<double> fc_parities(numCells);
  if (sideBasisCache->getCellSideParities().size() > 0)
  {
    // then we'll use this, and won't require that mesh and cellIDs are set
    TEUCHOS_TEST_FOR_EXCEPTION(sideBasisCache->getCellSideParities().dimension(0) != numCells, std::invalid_argument,
                               "sideBasisCache->getCellSideParities() is non-empty, but the cell dimension doesn't match that of the values Camellia::FieldContainer.");
    const auto & cellSideParities = sideBasisCache->getCellSideParities();
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      const double parity = cellSideParities(0,sideIndex);
      fc_parities(cellIndex) = parity;
    }
  }
  else
  {
    const vector<GlobalIndexType> & cellIDs = sideBasisCache->cellIDs();
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != numCells, std::invalid_argument, "cellIDs.size() != numCells");
    Teuchos::RCP<Mesh> mesh = sideBasisCache->mesh();
    TEUCHOS_TEST_FOR_EXCEPTION(!mesh.get(), std::invalid_argument, "mesh unset in BasisCache.");
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      const double parity = mesh->cellSideParitiesForCell(cellIDs[cellIndex])(0,sideIndex);
      fc_parities(cellIndex) = parity;
    }
  }
  auto dv_parities = fc_parities.checkoutDeviceView(); // copy parities to device
  
  auto rangePolicy = FieldContainer<double>::getMDRangePolicy2(values);
  Kokkos::parallel_for("SideParity_values", rangePolicy,
  KOKKOS_LAMBDA (const int cellOrdinal, const int pointOrdinal) {
    const double parity = dv_parities(cellOrdinal);
    assignValue(values(cellOrdinal,pointOrdinal), modType, parity);
  });
  
  fc_parities.checkinDeviceView(false);
}
