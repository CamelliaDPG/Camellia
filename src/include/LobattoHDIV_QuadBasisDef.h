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
  int LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::dofOrdinalMap(int i, int j, bool divFree) const { // i the xDofOrdinal, j the yDofOrdinal
    // ordering:
    // i = 1..k, j = 0, where k := _degree_x  (ordinals 0 to k-1)
    // i = 0, j = 1..l, where l := _degree_y  (ordinals k to k+l-1)
    // i = 1..k, j = 1..l : divFree           (ordinals k+l to k+l + kl -1)
    // i = 1..k, j = 1..l : not divFree       (ordinals k+l + kl to k + l + 2kl)
    if ((j==0) && (i==0)) return -1; // no such basis function
    if (j==0) {
      return i-1;
    }
    if (i==0) {
      return _degree_x + j-1;
    }
    if (divFree) {
      return _degree_x + _degree_y + i * _degree_y - 1;
    } else {
      return _degree_x + _degree_y + _degree_x * _degree_y + i * _degree_y - 1;
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
  
    this->_functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
    this->_rangeDimension = 2; // 2 space dim
    this->_rangeRank = 1; // vector
    this->_domainTopology = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
    this->_basisCardinality = 2 * _degree_x * _degree_y + _degree_x + _degree_y;
    initializeL2normValues();
  }
  
  template<class Scalar, class ArrayScalar>
  LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::LobattoHDIV_QuadBasis(int degree_x, int degree_y, bool conforming) {
    this->_basisDegree = max(degree_x, degree_y);
    _degree_x = degree_x;
    _degree_y = degree_y;
    _conforming = conforming;

    this->_functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_HDIV;
    this->_rangeDimension = 2; // 2 space dim
    this->_rangeRank = 1; // scalar
    this->_domainTopology = shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
    this->_basisCardinality = 2 * _degree_x * _degree_y + _degree_x + _degree_y;
    initializeL2normValues();
  }

  template<class Scalar, class ArrayScalar>
  void LobattoHDIV_QuadBasis<Scalar,ArrayScalar>::initializeTags() const {
    if (!_conforming) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "initializeTags() called for non-conforming Lobatto basis.");
    }
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "LobattoHDIV_QuadBasis: conforming basis not supported yet");
    cout << "LobattoHDIV_QuadBasis::initializeTags(): untested method!  It might be that the conforming basis itself is wrong.  See notes in edgeOrdinals[] initialization...\n";
 
    // the following is a plausible first effort, but I really haven't thought this through carefully.
    // hence the exception thrown above, and the code commented out below.
/*    const int numEdges = 4;
    // ordering of edges: 0 south, 1 east, 2 north, 3 west
    // normal continuity: south and north care about y component, east and west about x
    vector<int> edgeOrdinals[numEdges];
    for (int i=1; i<_degree_x+1; i++) {
      edgeOrdinals[0].push_back(dofOrdinalMap(i,0,true)); // all j=0 terms are div-free
      // would have a non-zero for (0,0), but we're not including that in the basis
    }
    for (int j=1; j<_degree_y+1; j++) {
      // I'm confused here: in terms of where the non-zeros are, we should have (1,0) as well,
      // but that's already taken by edgeOrdinals[0]
      edgeOrdinals[1].push_back(dofOrdinalMap(1,j,true));
      edgeOrdinals[1].push_back(dofOrdinalMap(1,j,false));
    }
    for (int i=1; i<_degree_x+1; i++) {
      // I'm confused here: in terms of where the non-zeros are, we should have (0,1) as well,
      // but that's taken by edgeOrdinals[3], below
      edgeOrdinals[2].push_back(dofOrdinalMap(i,1,true));
      edgeOrdinals[2].push_back(dofOrdinalMap(i,1,false));
    }
    for (int j=1; j<_degree_y+1; j++) {
      edgeOrdinals[3].push_back(dofOrdinalMap(0,j,true)); // all i=0 terms are div-free
      // would have a non-zero for (0,0), but we're not including that in the basis.
    }
    
    vector<int> interiorOrdinals;
    for (int i=2; i<_degree_x+1; i++) {
      for (int j=2; j<_degree_y+1; j++) {
        interiorOrdinals.push_back(dofOrdinalMap(i,j));
      }
    }
    
    int tagSize = 4;
    int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
    int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
    int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell

    std::vector<int> tags( tagSize * this->getCardinality() ); // flat array
    
    for (int edgeIndex=0; edgeIndex<numEdges; edgeIndex++) {
      int numDofsForEdge = edgeOrdinals[edgeIndex].size();
      int edgeSpaceDim = 1;
      for (int edgeDofOrdinal=0; edgeDofOrdinal<numDofsForEdge; edgeDofOrdinal++) {
        int i = edgeOrdinals[edgeIndex][edgeDofOrdinal];
        tags[tagSize*i]   = edgeSpaceDim; // spaceDim
        tags[tagSize*i+1] = edgeIndex; // subcell ordinal
        tags[tagSize*i+2] = edgeDofOrdinal;  // ordinal of the specified DoF relative to the subcell
        tags[tagSize*i+3] = numDofsForEdge;     // total number of DoFs associated with the subcell
      }
    }
    
    int numDofsForInterior = interiorOrdinals.size();
    for (int interiorIndex=0; interiorIndex<numDofsForInterior; interiorIndex++) {
      int i=interiorOrdinals[interiorIndex];
      int interiorSpaceDim = 2;
      tags[tagSize*i]   = interiorSpaceDim; // spaceDim
      tags[tagSize*i+1] = interiorIndex; // subcell ordinal
      tags[tagSize*i+2] = interiorIndex;  // ordinal of the specified DoF relative to the subcell
      tags[tagSize*i+3] = numDofsForInterior;     // total number of DoFs associated with the subcell
    }
    
    // call basis-independent method (declared in IntrepidUtil.hpp) to set up the data structures
    Intrepid::setOrdinalTagData(this -> _tagToOrdinal,
                                this -> _ordinalToTag,
                                &(tags[0]),
                                this -> getCardinality(),
                                tagSize,
                                posScDim,
                                posScOrd,
                                posDfOrd);*/
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
          // first, set the divergence-free basis values
          int fieldIndex = dofOrdinalMap(i,j,true);

          switch (operatorType) {
            case Intrepid::OPERATOR_VALUE:
            {
              double divFreeScalingFactor = 2 * (  _legendreL2normsSquared(i) * _lobattoL2normsSquared(j)
                                                 + _legendreL2normsSquared(j) * _lobattoL2normsSquared(i) );
              divFreeScalingFactor = sqrt(divFreeScalingFactor);

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
          
          // now, set the non-divergence-free basis values:
          if ((i > 0) && (j > 0)) { // then there will be some non-divergence-free members
            fieldIndex = dofOrdinalMap(i,j,false);
            double nonDivFreeScalingFactor = 2 * (  _legendreL2normsSquared(i) * _legendreL2normsSquared(j)
                                                  + _legendreL2normsSquared(j) * _legendreL2normsSquared(i) );
            nonDivFreeScalingFactor = sqrt(nonDivFreeScalingFactor);
            switch (operatorType) {
              case Intrepid::OPERATOR_VALUE:
                values(fieldIndex,pointIndex,0) = lobattoValues_x(i) * lobattoValues_dy(j) / nonDivFreeScalingFactor;
                values(fieldIndex,pointIndex,1) = lobattoValues_dx(i) * lobattoValues_y(j) / nonDivFreeScalingFactor;
                break;
              case Intrepid::OPERATOR_DIV:
                values(fieldIndex,pointIndex) = ( lobattoValues_dx(i) * lobattoValues_dy(j) + lobattoValues_dx(i) * lobattoValues_dy(j) ) / nonDivFreeScalingFactor;
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