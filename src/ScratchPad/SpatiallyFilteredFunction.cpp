#include "SpatiallyFilteredFunction.h"


using namespace Intrepid;
using namespace Camellia;

SpatiallyFilteredFunction::SpatiallyFilteredFunction(FunctionPtr f, SpatialFilterPtr sf) : Function<double>(f->rank()) {
  _f = f;
  _sf = sf;
}

bool SpatiallyFilteredFunction::boundaryValueOnly() {
  return _f->boundaryValueOnly();
}

void SpatiallyFilteredFunction::values(FieldContainer<double> &values, BasisCachePtr basisCache) {
//  cout << "Entered SpatiallyFilteredFunction::values()\n";
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  values.initialize(0.0);

  Teuchos::Array<int> dim;
  values.dimensions(dim);
  Teuchos::Array<int> fValuesDim = dim;
  int entriesPerPoint = 1;
  for (int d=2; d<values.rank(); d++) {
    entriesPerPoint *= dim[d];
    dim[d] = 0; // clear so that these indices point to the start of storage for (cellIndex,ptIndex)
  }
  FieldContainer<bool> pointsMatch(numCells,numPoints);
  if (_sf->matchesPoints(pointsMatch,basisCache)) { // SOME point matches
//    cout << "pointsMatch:\n" << pointsMatch;
    FieldContainer<double> fValues(fValuesDim);
    _f->values(fValues,basisCache);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      dim[0] = cellIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        dim[1] = ptIndex;
        if (pointsMatch(cellIndex,ptIndex)) {
          double* value = &values[values.getEnumeration(dim)];
          double* fValue = &fValues[fValues.getEnumeration(dim)];
          for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
            *value++ = *fValue++;
          }
        }
      }
    }
  }
}

FunctionPtr SpatiallyFilteredFunction::curl() {
  return Teuchos::rcp( new SpatiallyFilteredFunction(_f->curl(), _sf));
}

FunctionPtr SpatiallyFilteredFunction::div() {
  return Teuchos::rcp( new SpatiallyFilteredFunction(_f->div(), _sf));
}

FunctionPtr SpatiallyFilteredFunction::dx() {
  return Teuchos::rcp( new SpatiallyFilteredFunction(_f->dx(), _sf));
}

FunctionPtr SpatiallyFilteredFunction::dy() {
    return Teuchos::rcp( new SpatiallyFilteredFunction(_f->dy(), _sf));
}

FunctionPtr SpatiallyFilteredFunction::dz() {
  return Teuchos::rcp( new SpatiallyFilteredFunction(_f->dz(), _sf));
}
