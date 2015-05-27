//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include "CellTopology.h"

#include "Legendre.hpp"

namespace Camellia
{
template<class Scalar, class ArrayScalar>
void LegendreHVOL_LineBasis<Scalar,ArrayScalar>::initializeL2normValues()
{
  _legendreL2norms.resize(this->_basisDegree+1);
  for (int i=0; i<=this->_basisDegree; i++)
  {
    _legendreL2norms[i] = sqrt(2.0 / (2*i-1)); // the squared L^2 norm of the (i-1)th Legendre polynomial...
  }
//    cout << "Legendre L^2 norms squared:\n"  << _legendreL2normsSquared;
}

template<class Scalar, class ArrayScalar>
LegendreHVOL_LineBasis<Scalar,ArrayScalar>::LegendreHVOL_LineBasis(int degree)
{
  _degree = degree;
  this->_basisDegree = degree;

  this->_functionSpace = Camellia::FUNCTION_SPACE_HVOL;
  this->_rangeDimension = 1;
  this->_rangeRank = 0; // scalar
  this->_domainTopology = CellTopology::cellTopology( shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() ) );
  this->_basisCardinality = _degree + 1;
  initializeL2normValues();
}

template<class Scalar, class ArrayScalar>
void LegendreHVOL_LineBasis<Scalar,ArrayScalar>::initializeTags() const
{
  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "initializeTags() called for non-conforming Legendre basis.");
}

template<class Scalar, class ArrayScalar>
void LegendreHVOL_LineBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const
{
  this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);

  ArrayScalar  legendreValues_x( _degree + 1 );
  ArrayScalar legendreValues_dx( _degree + 1);

  int numPoints = refPoints.dimension(0);
  for (int pointIndex=0; pointIndex < numPoints; pointIndex++)
  {
    double x = refPoints(pointIndex,0);

    Legendre<Scalar,ArrayScalar>::values(legendreValues_x,legendreValues_dx, x,_degree);
    for (int i=0; i < _degree + 1; i++)
    {
      int fieldIndex = i;
      double scalingFactor = _legendreL2norms(i);
//          cout << "scaling factor squared for " << i << " " << j << ": " << scalingFactor << endl;

      switch (operatorType)
      {
      case Intrepid::OPERATOR_VALUE:
        values(fieldIndex,pointIndex) = legendreValues_x(i) / scalingFactor;
        break;
      case Intrepid::OPERATOR_GRAD:
        values(fieldIndex,pointIndex) = legendreValues_dx(i) / scalingFactor;
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported operatorType");
        break;
      }
    }
  }
}
} // namespace Camellia