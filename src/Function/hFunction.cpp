#include "hFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

string hFunction::displayString()
{
  return "h";
}

void hFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);

  Intrepid::FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
  int dimension = basisCache->cellTopology()->getDimension();
  
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    double h = pow(cellMeasures(cellIndex), 1.0 / dimension);
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      values(cellIndex,ptIndex) = h;
    }
  }
}
