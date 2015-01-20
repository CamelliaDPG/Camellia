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

#include "CellTopology.h"

template <class Scalar, class ArrayScalar>
PointBasis<Scalar,ArrayScalar>::PointBasis() {
  this -> _basisCardinality  = 1;
  this -> _basisDegree       = 0;
  this -> _domainTopology = Camellia::CellTopology::cellTopology( shards::getCellTopologyData< shards::Node >() );
  this -> _functionSpace = IntrepidExtendedTypes::FUNCTION_SPACE_REAL_SCALAR;
  this -> _rangeRank = 0;
  this -> _rangeDimension = 0;
}

template <class Scalar, class ArrayScalar>
void PointBasis<Scalar,ArrayScalar>::initializeTags() const {
//  cout << "PointBasis::initializeTags() called.\n";

  // Basis-dependent initializations
  int tagSize  = 4;        // size of DoF tag, i.e., number of fields in the tag
  int posScDim = 0;        // position in the tag, counting from 0, of the subcell dim
  int posScOrd = 1;        // position in the tag, counting from 0, of the subcell ordinal
  int posDfOrd = 2;        // position in the tag, counting from 0, of DoF ordinal relative to the subcell

  // An array with local DoF tags assigned to the basis functions, in the order of their local enumeration
  int N = this->getCardinality();
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
                                                const Intrepid::EOperator operatorType) const {
  // TODO: add parameter checking
  for (int i=0; i<outputValues.size(); i++) {
    outputValues[i] = 1.0;
  }
}


#endif
