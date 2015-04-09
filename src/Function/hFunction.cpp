#include "hFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

string hFunction::displayString() {
  return "h";
}

double hFunction::value(double x, double y, double h) {
  return h;
}
void hFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  Intrepid::FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
  const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    double h = sqrt(cellMeasures(cellIndex));
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      double x = (*points)(cellIndex,ptIndex,0);
      double y = (*points)(cellIndex,ptIndex,1);
      values(cellIndex,ptIndex) = value(x,y,h);
    }
  }
}