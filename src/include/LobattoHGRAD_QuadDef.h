//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include "Lobatto.hpp"

namespace Camellia {

  template<class Scalar, class ArrayScalar>
  LobattoHGRAD_Quad<Scalar,ArrayScalar>::LobattoHGRAD_Quad(int degree) {
    _degree_x = degree;
    _degree_y = degree;
    _basisDegree = degree;
  
    _rangeDimension = 2; // 2 space dim
    _rangeRank = 1; // scalar
    _domainTopology = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() );    
    _basisCardinality = (_degree_x + 1) * (_degree_y + 1);
  }
  
  template<class Scalar, class ArrayScalar>
  LobattoHGRAD_Quad<Scalar,ArrayScalar>::LobattoHGRAD_Quad(int degree_x, int degree_y) {
    _basisDegree = max(degree_x, degree_y);
    _degree_x = degree_x;
    _degree_y = degree_y;
    
    _rangeDimension = 2; // 2 space dim
    _rangeRank = 1; // scalar
    _domainTopology = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() );    
    _basisCardinality = (_degree_x + 1) * (_degree_y + 1);
  }

  template<class Scalar, class ArrayScalar>
  void LobattoHGRAD_Quad<Scalar,ArrayScalar>::initializeTags() const {
    // TODO: implement this
  }
  
  template<class Scalar, class ArrayScalar>
  void LobattoHGRAD_Quad<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const {
    this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);
    
    ArrayScalar lobattoValues_x(_degree_x+1), lobattoValues_y(_degree_y+1);
    ArrayScalar lobattoValues_dx(_degree_x+1), lobattoValues_dy(_degree_y+1);
    
    int numPoints = refPoints.dimension(0);
    for (int pointIndex=0; pointIndex < numPoints; pointIndex++) {
      double x = refPoints(pointIndex,0);
      double y = refPoints(pointIndex,1);
      
      Lobatto<Scalar,ArrayScalar>::values(lobattoValues_x,lobattoValues_dx, x,_degree_x);
      Lobatto<Scalar,ArrayScalar>::values(lobattoValues_y,lobattoValues_dy, y,_degree_y);
      int fieldIndex = 0;
      for (int i=0; i<_degree_x+1; i++) {
        for (int j=0; j<_degree_y+1; j++) {
          switch (operatorType) {
            case Intrepid::OPERATOR_VALUE:
              values(fieldIndex,pointIndex) = lobattoValues_x(i) * lobattoValues_y(j);
              break;
            case Intrepid::OPERATOR_GRAD:
              values(fieldIndex,pointIndex,0) = lobattoValues_dx(i) * lobattoValues_y(j);
              values(fieldIndex,pointIndex,1) = lobattoValues_x(i) * lobattoValues_dy(j);
              break;
              
            default:
              TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported operatorType");
              break;
          }
          fieldIndex++;
        }
      }
    }
    // TODO: implement this
    
  }
} // namespace Camellia