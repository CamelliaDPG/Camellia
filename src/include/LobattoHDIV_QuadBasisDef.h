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
  int LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::dofOrdinalMap(int i, int j, bool xComponentNonzero) const { // i the xDofOrdinal, j the yDofOrdinal
    // ordering:
    // i = 1..k, j = 0, where k := _degree_x      (ordinals 0 to k-1)
    // i = 0, j = 1..l, where l := _degree_y      (ordinals k to k+l-1)
    // i = 1..k, j = 1..l : (l_i(x) * L_j(y), 0)  (ordinals k+l to k+l + kl -1)
    // i = 1..k, j = 1..l : (0, L_i(x) * l_j(y))  (ordinals k+l + kl to k + l + 2kl)
    if ((j==0) && (i==0)) return -1; // no such basis function
    if (j==0) {
      return i-1;
    }
    if (i==0) {
      return _degree_x + j-1;
    }
    if (xComponentNonzero) {
      return _degree_x + _degree_y + (i-1) * _degree_y + j - 1;
    } else {
      return _degree_x + _degree_y + _degree_x * _degree_y + (i-1) * _degree_y + j - 1;
    }
  }
  
  template<class Scalar, class ArrayScalar>
  void LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::initializeL2normValues() {
    // NOTE: could speed things up by statically storing the lobatto L^2 norms (lazily adding higher degrees when they're required)
    _legendreL2normsSquared.resize(this->_basisDegree+1);
    _lobattoL2normsSquared.resize(this->_basisDegree+1);
    _legendreL2normsSquared[0] = 0; // not actually Legendre: the squared L^2 norm of the derivative of the first Lobatto polynomial...
    for (int i=1; i<=this->_basisDegree; i++) {
      _legendreL2normsSquared[i] = 2.0 / (2*i-1); // the squared L^2 norm of the (i-1)th Legendre polynomial...
    }
    Lobatto<Scalar,ArrayScalar>::l2norms(_lobattoL2normsSquared,this->_basisDegree,_conforming);
    // square the L^2 norms:
    for (int i=0; i<=this->_basisDegree; i++) {
      _lobattoL2normsSquared[i] = _lobattoL2normsSquared[i] * _lobattoL2normsSquared[i];
    }
  }

  template<class Scalar, class ArrayScalar>
  LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::LobattoHDIV_QuadBasis(int degree, bool conforming) {
    _degree_x = degree;
    _degree_y = degree;
    this->_basisDegree = degree;
    _conforming = conforming;
  
    this->_rangeDimension = 2; // 2 space dim
    this->_rangeRank = 1; // vector
    this->_domainTopology = CellTopology::cellTopology( shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
    this->_basisCardinality = 2 * _degree_x * _degree_y + _degree_x + _degree_y;
    this->_functionSpace = Camellia::FUNCTION_SPACE_HDIV;
    initializeL2normValues();
  }
  
  template<class Scalar, class ArrayScalar>
  LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::LobattoHDIV_QuadBasis(int degree_x, int degree_y, bool conforming) {
    this->_basisDegree = max(degree_x, degree_y);
    _degree_x = degree_x;
    _degree_y = degree_y;
    _conforming = conforming;

    this->_rangeDimension = 2; // 2 space dim
    this->_rangeRank = 1; // scalar
    this->_domainTopology = CellTopology::cellTopology( shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ) );
    this->_basisCardinality = 2 * _degree_x * _degree_y + _degree_x + _degree_y;
    this->_functionSpace = Camellia::FUNCTION_SPACE_HDIV;
    initializeL2normValues();
  }

  template<class Scalar, class ArrayScalar>
  void LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::initializeTags() const {
    if (!_conforming) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "initializeTags() called for non-conforming Lobatto basis.");
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "LobattoHDIV_QuadBasis: conforming basis not supported yet");
    cout << "LobattoHDIV_QuadBasis::initializeTags(): untested method!  It might be that the conforming basis itself is wrong.  See notes in edgeOrdinals[] initialization...\n";
  }
  
  template<class Scalar, class ArrayScalar>
  void LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const {
    this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);
    
    ArrayScalar lobattoValues_x(_degree_x+1), lobattoValues_y(_degree_y+1);
    ArrayScalar lobattoValues_dx(_degree_x+1), lobattoValues_dy(_degree_y+1);
    
    int numPoints = refPoints.dimension(0);
    for (int pointIndex=0; pointIndex < numPoints; pointIndex++) {
      double x = refPoints(pointIndex,0);
      double y = refPoints(pointIndex,1);

      Lobatto<Scalar,ArrayScalar>::values(lobattoValues_x,lobattoValues_dx, x,_degree_x,_conforming);
      Lobatto<Scalar,ArrayScalar>::values(lobattoValues_y,lobattoValues_dy, y,_degree_y,_conforming);
      for (int i=0; i<_degree_x+1; i++) {
        for (int j=0; j<_degree_y+1; j++) {
          if ((j==0) && (i==0)) continue; // no (0,0) basis function

          if ((i==0) || (j==0)) {    // first, set the divergence-free basis values
            int fieldIndex = dofOrdinalMap(i,j,false);
            double divFreeScalingFactor = (  _legendreL2normsSquared(i) * _lobattoL2normsSquared(j)
                                           + _legendreL2normsSquared(j) * _lobattoL2normsSquared(i) );
            divFreeScalingFactor = sqrt(divFreeScalingFactor);
            
            switch (operatorType) {
              case Intrepid::OPERATOR_VALUE:
              {
                // one of these is zero:
                values(fieldIndex,pointIndex,0) =  lobattoValues_x(i) * lobattoValues_dy(j) / divFreeScalingFactor;
                values(fieldIndex,pointIndex,1) = -lobattoValues_dx(i) * lobattoValues_y(j) / divFreeScalingFactor;
              }
                break;
              case Intrepid::OPERATOR_DIV:
                values(fieldIndex,pointIndex) = 0;
                break;
                
              default:
                TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported operatorType");
                break;
            }
          } else if ((i > 0) && (j > 0)) { // then there will be some non-divergence-free members
            int xFieldIndex = dofOrdinalMap(i,j,true);
            int yFieldIndex = dofOrdinalMap(i,j,false);
            double nonDivFreeScalingFactor = _legendreL2normsSquared(i) * _legendreL2normsSquared(j);
            nonDivFreeScalingFactor = sqrt(nonDivFreeScalingFactor);
            
            switch (operatorType) {
              case Intrepid::OPERATOR_VALUE:
                values(xFieldIndex,pointIndex,0) = lobattoValues_x(i) * lobattoValues_dy(j) / nonDivFreeScalingFactor;
                values(yFieldIndex,pointIndex,1) = lobattoValues_dx(i) * lobattoValues_y(j) / nonDivFreeScalingFactor;
                break;
              case Intrepid::OPERATOR_DIV:
                values(xFieldIndex,pointIndex) = lobattoValues_dx(i) * lobattoValues_dy(j) / nonDivFreeScalingFactor;
                values(yFieldIndex,pointIndex) = lobattoValues_dx(i) * lobattoValues_dy(j) / nonDivFreeScalingFactor;
                break;
              default:
                TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported operatorType");
                break;
            }
          }
        }
      }
    }
  }
} // namespace Camellia