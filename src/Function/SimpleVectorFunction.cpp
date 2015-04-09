#include "SimpleVectorFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;
using namespace std;

SimpleVectorFunction::SimpleVectorFunction() : Function(1) {}

vector<double> SimpleVectorFunction::value(double x) {
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method. Subclasses of SimpleVectorFunction must implement value() for some number of arguments < spaceDim");
  return vector<double>();
}

vector<double> SimpleVectorFunction::value(double x, double y) {
  return value(x);
}

vector<double> SimpleVectorFunction::value(double x, double y, double z) {
  return value(x,y);
}

void SimpleVectorFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  
  const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
  int spaceDim = points->dimension(2);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      if (spaceDim == 1) {
        double x = (*points)(cellIndex,ptIndex,0);
        values(cellIndex,ptIndex,0) = value(x)[0];
      } else if (spaceDim == 2) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex,0) = value(x,y)[0];
        values(cellIndex,ptIndex,1) = value(x,y)[1];
      } else if (spaceDim == 3) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double z = (*points)(cellIndex,ptIndex,2);
        values(cellIndex,ptIndex,0) = value(x,y,z)[0];
        values(cellIndex,ptIndex,1) = value(x,y,z)[1];
        values(cellIndex,ptIndex,2) = value(x,y,z)[2];
      }
    }
  }
}