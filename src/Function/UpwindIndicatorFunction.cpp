#include "UpwindIndicatorFunction.h"

#include "BasisCache.h"

#include "Intrepid_FieldContainer.hpp"

using namespace Intrepid;
using namespace Camellia;

UpwindIndicatorFunction::UpwindIndicatorFunction(FunctionPtr beta, bool upwind)
{
  // upwind = true  means this is the DG '-' operator  (beta * n > 0)
  // upwind = false means this is the DG '+' operator  (beta * n < 0)
  
  _beta.push_back(beta->x());
  if (beta->y() != Teuchos::null)
  {
    _beta.push_back(beta->y());
    if (beta->z() != Teuchos::null)
      _beta.push_back(beta->z());
  }
  _upwind = upwind;
  _valuesBuffers.resize(_beta.size());
}

bool UpwindIndicatorFunction::isZero(BasisCachePtr basisCache)
{
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getSpaceDim();
  
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != _beta.size(), std::invalid_argument, "spaceDim and length of beta do not match");
  
  for (int d=0; d<spaceDim; d++)
  {
    _valuesBuffers[d].resize(numCells,numPoints);
    _beta[d]->values(_valuesBuffers[d],basisCache);
  }
  
  const Intrepid::FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());
  
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      double value = 0;
      for (int d=0; d<spaceDim; d++)
      {
        value += _valuesBuffers[d](cellOrdinal,pointOrdinal) * (*sideNormals)(cellOrdinal,pointOrdinal,d);
      }
      if ((_upwind && (value > 0)) || (!_upwind && (value < 0)))
      {
        return false;
      }
    }
  }
  return true;
}

void UpwindIndicatorFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getSpaceDim();
  
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != _beta.size(), std::invalid_argument, "spaceDim and length of beta do not match");
  
  for (int d=0; d<spaceDim; d++)
  {
    _valuesBuffers[d].resize(numCells,numPoints);
    _beta[d]->values(_valuesBuffers[d],basisCache);
  }
  const Intrepid::FieldContainer<double> *sideNormals = &(basisCache->getSideNormals());
  
  for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
    {
      double value = 0;
      for (int d=0; d<spaceDim; d++)
      {
        value += _valuesBuffers[d](cellOrdinal,pointOrdinal) * (*sideNormals)(cellOrdinal,pointOrdinal,d);
      }
      if ((_upwind && (value > 0)) || (!_upwind && (value < 0)))
      {
        values(cellOrdinal,pointOrdinal) = 1;
      }
      else
      {
        values(cellOrdinal,pointOrdinal) = 0;
      }
      
    }
  }
}

FunctionPtr UpwindIndicatorFunction::minus(FunctionPtr beta)
{
  return upwindIndicator(beta, true);
}

FunctionPtr UpwindIndicatorFunction::plus(FunctionPtr beta)
{
  return upwindIndicator(beta, false);
}

FunctionPtr UpwindIndicatorFunction::upwindIndicator(FunctionPtr beta, bool upwind)
{
  return Teuchos::rcp( new UpwindIndicatorFunction(beta, upwind) );
}
