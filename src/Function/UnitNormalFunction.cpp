#include "UnitNormalFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

UnitNormalFunction::UnitNormalFunction(int comp, bool spaceTime) : Function( (comp==-1)? 1 : 0) {
  _comp = comp;
  _spaceTime = spaceTime;
}

FunctionPtr UnitNormalFunction::x() {
  return Teuchos::rcp( new UnitNormalFunction(0,_spaceTime) );
}

FunctionPtr UnitNormalFunction::y() {
  return Teuchos::rcp( new UnitNormalFunction(1,_spaceTime) );
}

FunctionPtr UnitNormalFunction::z() {
  return Teuchos::rcp( new UnitNormalFunction(2,_spaceTime) );
}

FunctionPtr UnitNormalFunction::t() {
  return Teuchos::rcp( new UnitNormalFunction(-2,_spaceTime) );
}

bool UnitNormalFunction::boundaryValueOnly() {
  return true;
}

string UnitNormalFunction::displayString() {
  if (_comp == -1) {
    return " \\boldsymbol{n} ";
  } else {
    if (_comp == 0) {
      return " n_x ";
    }
    if (_comp == 1) {
      return " n_y ";
    }
    if (_comp == 2) {
      return " n_z ";
    }
    return "UnitNormalFunction with unexpected component";
  }
}

void UnitNormalFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache) {
  CHECK_VALUES_RANK(values);
  const Intrepid::FieldContainer<double> *sideNormals = _spaceTime ? &(basisCache->getSideNormalsSpaceTime()) : &(basisCache->getSideNormals());
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getSpaceDim();
  int comp = _comp;
  if (comp == -2) {
    // want to select the temporal component, t()
    comp = spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
      if (comp == -1) {
        for (int d=0; d<spaceDim; d++) {
          double nd = (*sideNormals)(cellIndex,ptIndex,d);
          values(cellIndex,ptIndex,d) = nd;
        }
      } else {
        double ni = (*sideNormals)(cellIndex,ptIndex,comp);
        values(cellIndex,ptIndex) = ni;
      }
    }
  }
}
