#include "SpatiallyFilteredFunction.h"


using namespace Intrepid;
using namespace Camellia;

template <typename Scalar>
SpatiallyFilteredFunction<Scalar>::SpatiallyFilteredFunction(FunctionPtr<Scalar> f, SpatialFilterPtr sf) : Function<Scalar>(f->rank()) {
  _f = f;
  _sf = sf;
}

template <typename Scalar>
bool SpatiallyFilteredFunction<Scalar>::boundaryValueOnly() {
  return _f->boundaryValueOnly();
}

template <typename Scalar>
void SpatiallyFilteredFunction<Scalar>::values(FieldContainer<Scalar> &values, BasisCachePtr basisCache) {
//  cout << "Entered SpatiallyFilteredFunction<Scalar>::values()\n";
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
    FieldContainer<Scalar> fValues(fValuesDim);
    _f->values(fValues,basisCache);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      dim[0] = cellIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        dim[1] = ptIndex;
        if (pointsMatch(cellIndex,ptIndex)) {
          Scalar* value = &values[values.getEnumeration(dim)];
          Scalar* fValue = &fValues[fValues.getEnumeration(dim)];
          for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++) {
            *value++ = *fValue++;
          }
        }
      }
    }
  }
}

template <typename Scalar>
FunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::curl() {
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->curl(), _sf));
}

template <typename Scalar>
FunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::div() {
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->div(), _sf));
}

template <typename Scalar>
FunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::dx() {
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->dx(), _sf));
}

template <typename Scalar>
FunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::dy() {
    return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->dy(), _sf));
}

template <typename Scalar>
FunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::dz() {
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->dz(), _sf));
}

template class SpatiallyFilteredFunction<double>;

