//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "SimpleVectorFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace std;

template <typename Scalar>
SimpleVectorFunction<Scalar>::SimpleVectorFunction() : TFunction<Scalar>("SimpleVectorFunction", 1) {}

template <typename Scalar>
vector<Scalar> SimpleVectorFunction<Scalar>::value(double x)
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unimplemented method. Subclasses of SimpleVectorFunction must implement value() for some number of arguments < spaceDim");
//  return vector<Scalar>();
}

template <typename Scalar>
vector<Scalar> SimpleVectorFunction<Scalar>::value(double x, double y)
{
  return value(x);
}

template <typename Scalar>
vector<Scalar> SimpleVectorFunction<Scalar>::value(double x, double y, double z)
{
  return value(x,y);
}

template <typename Scalar>
void SimpleVectorFunction<Scalar>::getValue(vector<Scalar> &vectorOut, const vector<Scalar> &vectorIn)
{
  TEUCHOS_TEST_FOR_EXCEPTION(vectorOut.size() != vectorIn.size(), std::invalid_argument, "vectorOut does not match vectorIn in length");
  vector<Scalar> result;
  if (vectorOut.size() == 0)
  {
    return;
  }
  else if (vectorOut.size() == 1)
  {
    result = value(vectorIn[0]);
  }
  else if (vectorOut.size() == 2)
  {
    result = value(vectorIn[0],vectorIn[1]);
  }
  else if (vectorOut.size() == 3)
  {
    result = value(vectorIn[0],vectorIn[1],vectorIn[2]);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error, "SimpleVectorFunction subclasses must override getValue() to support spaceDim above 3.");
  }
  for (int d=0; d<vectorOut.size(); d++)
  {
    vectorOut[d] = result[d];
  }
}

template <typename Scalar>
void SimpleVectorFunction<Scalar>::values(Camellia::FieldContainer<Scalar> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);

  const Camellia::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePointsFC());
  int spaceDim = points->dimension(2);
  vector<Scalar> point(spaceDim);
  vector<Scalar> pointValue(spaceDim);
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      for (int d=0; d<spaceDim; d++)
      {
        point[d] = (*points)(cellIndex,ptIndex,d);
      }
      getValue(pointValue, point);
      for (int d=0; d<spaceDim; d++)
      {
        values(cellIndex,ptIndex,d) = pointValue[d];
      }
    }
  }
}

namespace Camellia
{
template class SimpleVectorFunction<double>;
}

