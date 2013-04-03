//
//  BasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

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

    // TODO: implement this
    
  }
} // namespace Camellia