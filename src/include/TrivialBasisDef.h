//
//  TrivialBasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/22/13.
//
//
#include "Teuchos_TestForException.hpp"

#include "Lobatto.hpp"

namespace Camellia {
  template<class Scalar, class ArrayScalar>
  void TrivialBasis<Scalar,ArrayScalar>::initializeL2normValues() {
    _lobattoL2norms.resize(this->_basisDegree+1);
    Lobatto<Scalar,ArrayScalar>::l2norms(_lobattoL2norms,this->_basisDegree,_conforming);
  }

  template<class Scalar, class ArrayScalar>
  TrivialBasis<Scalar,ArrayScalar>::TrivialBasis(int degree, bool conforming) {
    _degree = degree;
    this->_basisDegree = degree;
    _conforming = conforming;

    this->_functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_TRIVIAL;
    this->_rangeDimension = 1;
    this->_rangeRank = 0; // scalar
    this->_domainTopology = shards::CellTopology(shards::getCellTopologyData<shards::Line<2> >() );
    this->_basisCardinality = 0;
    initializeL2normValues();
  }

  template<class Scalar, class ArrayScalar>
  void TrivialBasis<Scalar,ArrayScalar>::initializeTags() const {
    if (!_conforming) {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "initializeTags() called for non-conforming Lobatto basis.");
    }
    // ordering of vertices: vertex 0 at -1.0, vertex 1 at 1.0
    const int numVertices = 2;
    int vertexOrdinals[numVertices];
    vertexOrdinals[0] = 0;
    vertexOrdinals[1] = 1;

    vector<int> interiorOrdinals;
    for (int i=2; i<_degree + 1; i++) {
      interiorOrdinals.push_back(i);
    }

    int tagSize = 4;
    int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
    int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
    int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell

    std::vector<int> tags( tagSize * this->getCardinality() ); // flat array

    for (int vertexIndex=0; vertexIndex<numVertices; vertexIndex++) {
      int vertexSpaceDim = 0;
      int ordinalRelativeToSubcell = 0; // just one in the subcell
      int subcellDofTotal = 1;
      int i = vertexOrdinals[vertexIndex];
      tags[tagSize*i]   = vertexSpaceDim; // spaceDim
      tags[tagSize*i+1] = vertexIndex; // subcell ordinal
      tags[tagSize*i+2] = ordinalRelativeToSubcell;  // ordinal of the specified DoF relative to the subcell
      tags[tagSize*i+3] = subcellDofTotal;     // total number of DoFs associated with the subcell
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
                                posDfOrd);
  }

  template<class Scalar, class ArrayScalar>
  void TrivialBasis<Scalar,ArrayScalar>::getValues(ArrayScalar &values, const ArrayScalar &refPoints, Intrepid::EOperator operatorType) const {
    this->CHECK_VALUES_ARGUMENTS(values,refPoints,operatorType);

    ArrayScalar  lobattoValues_x( _degree + 1 );
    ArrayScalar lobattoValues_dx( _degree + 1);

    int numPoints = refPoints.dimension(0);
    for (int pointIndex=0; pointIndex < numPoints; pointIndex++) {
      double x = refPoints(pointIndex,0);

      Lobatto<Scalar,ArrayScalar>::values(lobattoValues_x,lobattoValues_dx, x,_degree,_conforming);
      for (int i=0; i < _degree + 1; i++) {
        int fieldIndex = i;
        double scalingFactor = _lobattoL2norms(i);
//          cout << "scaling factor squared for " << i << " " << j << ": " << scalingFactor << endl;

        switch (operatorType) {
          case Intrepid::OPERATOR_VALUE:
            values(fieldIndex,pointIndex) = lobattoValues_x(i) / scalingFactor;
            break;
          case Intrepid::OPERATOR_GRAD:
            values(fieldIndex,pointIndex) = lobattoValues_dx(i) / scalingFactor;
            break;
          default:
            TEUCHOS_TEST_FOR_EXCEPTION(true,std::invalid_argument,"Unsupported operatorType");
            break;
        }
      }
    }
  }

  template<class Scalar, class ArrayScalar>
  bool TrivialBasis<Scalar,ArrayScalar>::isConforming() const {
    return _conforming;
  }
} // namespace Camellia
