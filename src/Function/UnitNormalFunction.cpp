#include "UnitNormalFunction.h"

#include "BasisCache.h"

using namespace Camellia;
using namespace Intrepid;

UnitNormalFunction::UnitNormalFunction(int comp, bool spaceTime) : TFunction<double>( (comp==-1)? 1 : 0)
{
  _comp = comp;
  _spaceTime = spaceTime;
}

TFunctionPtr<double> UnitNormalFunction::x()
{
  return Teuchos::rcp( new UnitNormalFunction(0,_spaceTime) );
}

TFunctionPtr<double> UnitNormalFunction::y()
{
  return Teuchos::rcp( new UnitNormalFunction(1,_spaceTime) );
}

TFunctionPtr<double> UnitNormalFunction::z()
{
  return Teuchos::rcp( new UnitNormalFunction(2,_spaceTime) );
}

TFunctionPtr<double> UnitNormalFunction::t()
{
  return Teuchos::rcp( new UnitNormalFunction(-2,_spaceTime) );
}

bool UnitNormalFunction::boundaryValueOnly()
{
  return true;
}

string UnitNormalFunction::displayString()
{
  if (_comp == -1)
  {
    return " \\boldsymbol{n} ";
  }
  else
  {
    if (_comp == 0)
    {
      return " n_x ";
    }
    if (_comp == 1)
    {
      return " n_y ";
    }
    if (_comp == 2)
    {
      return " n_z ";
    }
    return "UnitNormalFunction with unexpected component";
  }
}

void UnitNormalFunction::values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
{
  this->CHECK_VALUES_RANK(values);
  int numCells = values.dimension(0);
  int numPoints = values.dimension(1);
  int spaceDim = basisCache->getSpaceDim();
  if (_comp == -1)
  {
    // check the the "D" dimension of values is correct:
    if (_spaceTime)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(2) != spaceDim+1, std::invalid_argument, "For space-time normals, values.dimension(2) should be spaceDim + 1.");
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(values.dimension(2) != spaceDim, std::invalid_argument, "For spatial normals, values.dimension(2) should be spaceDim.");
    }
  }
  const Intrepid::FieldContainer<double> *sideNormals = _spaceTime ? &(basisCache->getSideNormalsSpaceTime()) : &(basisCache->getSideNormals());

  int comp = _comp;
  if (comp == -2)
  {
    // want to select the temporal component, t()
    comp = spaceDim;
  }
  for (int cellIndex=0; cellIndex<numCells; cellIndex++)
  {
    for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
    {
      if (comp == -1)
      {
        for (int d=0; d<spaceDim; d++)
        {
          double nd = (*sideNormals)(cellIndex,ptIndex,d);
          values(cellIndex,ptIndex,d) = nd;
        }
        if (_spaceTime)
        {
          double nd = (*sideNormals)(cellIndex,ptIndex,spaceDim);
          values(cellIndex,ptIndex,spaceDim) = nd;
        }
      }
      else
      {
        double ni = (*sideNormals)(cellIndex,ptIndex,comp);
        values(cellIndex,ptIndex) = ni;
      }
    }
  }
}
