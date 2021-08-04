//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "SpatiallyFilteredFunction.h"


using namespace Camellia;

template <typename Scalar>
SpatiallyFilteredFunction<Scalar>::SpatiallyFilteredFunction(TFunctionPtr<Scalar> f, SpatialFilterPtr sf) : TFunction<Scalar>("SpatiallyFilteredFunction", f->rank())
{
  _f = f;
  _sf = sf;
}

template <typename Scalar>
bool SpatiallyFilteredFunction<Scalar>::boundaryValueOnly()
{
  return _f->boundaryValueOnly();
}

template <typename Scalar>
std::vector< TFunctionPtr<Scalar> > SpatiallyFilteredFunction<Scalar>::memberFunctions()
{
  return {{_f}};
}

template <typename Scalar>
void SpatiallyFilteredFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
//  cout << "Entered SpatiallyFilteredFunction<Scalar>::values()\n";
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  values.initialize(0.0);

  Teuchos::Array<int> dim;
  values.dimensions(dim);
  Teuchos::Array<int> fValuesDim = dim;
  int entriesPerPoint = 1;
  for (int d=2; d<values.rank(); d++)
  {
    entriesPerPoint *= dim[d];
    dim[d] = 0; // clear so that these indices point to the start of storage for (cellIndex,ptIndex)
  }
  Camellia::FieldContainer<bool> pointsMatch(numCells,numPoints);
  if (_sf->matchesPoints(pointsMatch,basisCache))   // SOME point matches
  {
//    cout << "pointsMatch:\n" << pointsMatch;
    Camellia::FieldContainer<Scalar> fValues(fValuesDim);
    _f->values(fValues,basisCache);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      dim[0] = cellIndex;
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        dim[1] = ptIndex;
        if (pointsMatch(cellIndex,ptIndex))
        {
          Scalar* value = &values[values.getEnumeration(dim)];
          Scalar* fValue = &fValues[fValues.getEnumeration(dim)];
          for (int entryIndex=0; entryIndex<entriesPerPoint; entryIndex++)
          {
            *value++ = *fValue++;
          }
        }
      }
    }
  }
}

template <typename Scalar>
void SpatiallyFilteredFunction<Scalar>::values(DeviceView<Scalar> &values, ValueModificationType modType, BasisCachePtr basisCache)
{
  int numCells = values.extent_int(0);
  int numPoints = values.extent_int(1);
  
  auto &points = basisCache->getPhysicalCubaturePointsFC();
  
  DeviceView<bool> filterMatches = SharedScratchStorage<bool>::sharedScratchStorage().checkoutScratchStorage({numCells, numPoints}, "SpatiallyFilteredFunction::values(): filterMatches");
  {
    auto pointsView = points.checkoutDeviceView();
    _sf->matchesPoints(filterMatches, pointsView);
    points.checkinDeviceView(true);
  }
  
  Kokkos::fence(); // ensure matchesPoints has completed prior to reducing over filterMatches
  
  // notionally somePointMatches is boolean, but to use Kokkos automatic reduction inference, we need int
  int somePointMatches = false;
  auto rangePolicy = FieldContainer<bool>::getMDRangePolicy2(filterMatches);
  Kokkos::parallel_reduce("SpatiallyFilteredFunction: some point matches", rangePolicy,
  KOKKOS_LAMBDA (const int& i, const int &j, int& localSum )
  {
    localSum |= filterMatches(i,j);
  },somePointMatches);

  DeviceView<Scalar> functionValues = SharedScratchStorage<Scalar>::sharedScratchStorage().checkoutScratchStorageMatchingDeviceView(values, "SpatiallyFilteredFunction::values(): functionValues");
  if (somePointMatches)
  {
    _f->values(functionValues,ASSIGN,basisCache);
    Kokkos::fence(); // ensure functionValues is filled before filtering
  }
  
  if ( !somePointMatches && ((modType == ASSIGN) || (modType == MULTIPLY)))
  {
    // no point matched ==> our value is everywhere 0
    // if we're assigning:   fill with 0.
    // if we're multiplying: fill with 0.
    Kokkos::deep_copy(values, 0.0);
  }
  else if ( !somePointMatches && ((modType == ADD) || (modType == SUBTRACT)))
  {
    // no point matched ==> our value is everywhere 0
    // if we're adding to or subtracting from values, we should leave values just as it is.
  }
  else
  {
    const int dim6 = functionValues.extent_int(6);
    using array_index_type = Kokkos::MDRangePolicy< Kokkos::Rank<6> >::array_index_type;
    Kokkos::parallel_for("SpatiallyFilteredFunction assignment", FieldContainer<double>::getMDRangePolicy6(values),
    KOKKOS_LAMBDA (const array_index_type i0, const array_index_type i1, const array_index_type i2,
                   const array_index_type i3, const array_index_type i4, const array_index_type i5)
    {
      for (int i6=0; i6<dim6; i6++)
      {
        const Scalar value = filterMatches(i0, i1) ? functionValues.access(i0,i1,i2,i3,i4,i5,i6) : 0.0;
        assignValue(values.access(i0,i1,i2,i3,i4,i5,i6),modType,value);
      }
    });
  }
  
  Kokkos::fence(); // ensure we're done with the containers before we return them
  SharedScratchStorage<bool>::sharedScratchStorage().checkinScratchStorage(filterMatches);
  SharedScratchStorage<Scalar>::sharedScratchStorage().checkinScratchStorage(functionValues);
}

template <typename Scalar>
TFunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::curl()
{
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->curl(), _sf));
}

template <typename Scalar>
TFunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::div()
{
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->div(), _sf));
}

template <typename Scalar>
TFunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::dx()
{
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->dx(), _sf));
}

template <typename Scalar>
TFunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::dy()
{
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->dy(), _sf));
}

template <typename Scalar>
TFunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::dz()
{
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(_f->dz(), _sf));
}

template <typename Scalar>
TFunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::spatiallyFilteredFunction(Scalar value, SpatialFilterPtr sf)
{
  return SpatiallyFilteredFunction<Scalar>::spatiallyFilteredFunction(TFunction<Scalar>::constant(value), sf);
}

template <typename Scalar>
TFunctionPtr<Scalar> SpatiallyFilteredFunction<Scalar>::spatiallyFilteredFunction(TFunctionPtr<Scalar> f, SpatialFilterPtr sf)
{
  return Teuchos::rcp( new SpatiallyFilteredFunction<Scalar>(f,sf));
}

namespace Camellia
{
template class SpatiallyFilteredFunction<double>;
}

