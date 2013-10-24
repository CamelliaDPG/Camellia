//
//  PointBasisDef.h
//  Camellia-debug
//
//  Created by Nathan Roberts on 10/24/13.
//
//

#ifndef Camellia_debug_PointBasisDef_h
#define Camellia_debug_PointBasisDef_h

#include "CamelliaIntrepidExtendedTypes.h"

template <class Scalar, class ArrayScalar>
PointBasis<Scalar,ArrayScalar>::PointBasis() {
  this -> _basisCardinality  = 1;
  this -> _basisDegree       = 0;
  this -> _domainTopology = shards::getCellTopologyData< shards::Node >();
  this -> _functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_ONE;
  this -> _rangeRank = 0;
  this -> _rangeDimension = 0;
}

template <class Scalar, class ArrayScalar>
void PointBasis<Scalar,ArrayScalar>::initializeTags() const {
  //cout << "PointBasis::initializeTags() called.\n";

  // we need to at least set this up for the first and last vertices
  int firstVertexDofOrdinal, secondVertexDofOrdinal;
  firstVertexDofOrdinal = _bases[0]->getDofOrdinal(0,0,0);
  BasisPtr lastBasis = _bases[_bases.size() - 1];
  int lastBasisOrdinalOffset = this->_basisCardinality - lastBasis->getCardinality();
  secondVertexDofOrdinal = lastBasis->getDofOrdinal(0,1,0) + lastBasisOrdinalOffset;

  // The following adapted from Basis_HGRAD_LINE_Cn_FEM
  // unlike there, we do assume that the edge's endpoints are included in the first and last bases...

  // Basis-dependent initializations
  int tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell

  // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
  int N = this->getCardinality();

  // double-check that our assumptions about the sub-bases have not been violated:
  if (firstVertexDofOrdinal != 0) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-basis has first vertex dofOrdinal in unexpected spot." );
  }
  if (secondVertexDofOrdinal != N-1) {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "sub-basis has second vertex dofOrdinal in unexpected spot." );
  }

  int *tags = new int[ tagSize * N ];

  tags[0] = 0;
  tags[1] = 0;
  tags[2] = 0;
  tags[3] = 0;

  Intrepid::setOrdinalTagData(this -> _tagToOrdinal,
                              this -> _ordinalToTag,
                              tags,
                              this -> _basisCardinality,
                              tagSize,
                              posScDim,
                              posScOrd,
                              posDfOrd);

  delete []tags;
}

template<class Scalar, class ArrayScalar>
void PointBasis<Scalar, ArrayScalar>::getValues(ArrayScalar &outputValues, const ArrayScalar &  inputPoints,
                                                const EOperator operatorType) const {
  // TODO: add parameter checking
  outputValues[0] = 1.0;
}


#endif
