#include "SimpleFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

template <typename Scalar>
Scalar SimpleFunction<Scalar>::value(double x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method. Subclasses of SimpleFunction must implement value() for some number of arguments < spaceDim");
  return 0;
}

template <typename Scalar>
Scalar SimpleFunction<Scalar>::value(double x, double y)
{
  return value(x);
}

template <typename Scalar>
Scalar SimpleFunction<Scalar>::value(double x, double y, double z)
{
  return value(x,y);
}

template <typename Scalar>
Scalar SimpleFunction<Scalar>::value(double x, double y, double z, double t)
{
  return value(x,y,z);
}

template <typename Scalar>
void SimpleFunction<Scalar>::values(Intrepid::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);

  const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());

  if (points->dimension(1) != numPoints)
  {
    cout << "numPoints in values container does not match that in BasisCache's physical points.\n";
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "numPoints in values container does not match that in BasisCache's physical points.");
  }

  int spaceDim = points->dimension(2);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      if (spaceDim == 1)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        values(cellIndex,ptIndex) = value(x);
      }
      else if (spaceDim == 2)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex,ptIndex) = value(x,y);
      }
      else if (spaceDim == 3)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double z = (*points)(cellIndex,ptIndex,2);
        values(cellIndex,ptIndex) = value(x,y,z);
      }
      else if (spaceDim == 4)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double z = (*points)(cellIndex,ptIndex,2);
        double t = (*points)(cellIndex,ptIndex,3);
        values(cellIndex,ptIndex) = value(x,y,z,t);
      }
    }
  }
}

namespace Camellia
{
template class SimpleFunction<double>;
}

