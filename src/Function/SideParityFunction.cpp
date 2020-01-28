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
  const int numPoints = values.extent_int(1);
  const int sideIndex = sideBasisCache->getSideIndex();
  TEUCHOS_TEST_FOR_EXCEPTION(sideIndex == -1, std::invalid_argument, "non-sideBasisCache passed into SideParityFunction");
  if (sideBasisCache->getCellSideParities().size() > 0)
  {
    // then we'll use this, and won't require that mesh and cellIDs are set
    TEUCHOS_TEST_FOR_EXCEPTION(sideBasisCache->getCellSideParities().dimension(0) != numCells, std::invalid_argument,
                              "sideBasisCache->getCellSideParities() is non-empty, but the cell dimension doesn't match that of the values Camellia::FieldContainer.");
    auto dv_parities = sideBasisCache->getCellSideParities().checkoutDeviceView();
    Kokkos::parallel_for("SideParity_values", numCells, KOKKOS_LAMBDA (const int cellOrdinal) {
      double parity = dv_parities(cellOrdinal,sideIndex);
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        assignValue(values(cellOrdinal,ptOrdinal), modType, parity);
      }
    });
    sideBasisCache->getCellSideParities().checkinDeviceView();
  }
  else
  {
    vector<GlobalIndexType> cellIDs = sideBasisCache->cellIDs();
    TEUCHOS_TEST_FOR_EXCEPTION(cellIDs.size() != numCells, std::invalid_argument, "cellIDs.size() != numCells");
    Teuchos::RCP<Mesh> mesh = sideBasisCache->mesh();
    TEUCHOS_TEST_FOR_EXCEPTION(!mesh.get(), std::invalid_argument, "mesh unset in BasisCache.");
    FieldContainer<double> fc_parities(numCells);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      double parity = mesh->cellSideParitiesForCell(cellIDs[cellIndex])(0,sideIndex);
      fc_parities(cellIndex) = parity;
    }
    auto dv_parities = fc_parities.checkoutDeviceView();
    Kokkos::parallel_for("SideParity_values", numCells, KOKKOS_LAMBDA (const int cellIndex) {
      double parity = dv_parities(cellIndex);
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        assignValue(values(cellIndex,ptIndex), modType, parity);
      
      }
    });
    fc_parities.checkinDeviceView(false);
  }
}
