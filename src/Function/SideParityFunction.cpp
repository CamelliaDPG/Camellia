#include "SideParityFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

SideParityFunction::SideParityFunction() : TFunction<double>(0)
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

void SideParityFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr sideBasisCache)
{
  this->CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int sideIndex = sideBasisCache->getSideIndex();
  if (sideIndex == -1)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "non-sideBasisCache passed into SideParityFunction");
  }
  if (sideBasisCache->getCellSideParities().size() > 0)
  {
    // then we'll use this, and won't require that mesh and cellIDs are set
    if (sideBasisCache->getCellSideParities().dimension(0) != numCells)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sideBasisCache->getCellSideParities() is non-empty, but the cell dimension doesn't match that of the values FieldContainer.");
    }

    for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
    {
      int parity = sideBasisCache->getCellSideParities()(cellOrdinal,sideIndex);
      for (int ptOrdinal=0; ptOrdinal<numPoints; ptOrdinal++)
      {
        values(cellOrdinal,ptOrdinal) = parity;
      }
    }
  }
  else
  {
    vector<GlobalIndexType> cellIDs = sideBasisCache->cellIDs();
    if (cellIDs.size() != numCells)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "cellIDs.size() != numCells");
    }
    Teuchos::RCP<Mesh> mesh = sideBasisCache->mesh();
    if (! mesh.get())
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "mesh unset in BasisCache.");
    }
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      int parity = mesh->cellSideParitiesForCell(cellIDs[cellIndex])(0,sideIndex);
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        values(cellIndex,ptIndex) = parity;
      }
    }
  }
}
