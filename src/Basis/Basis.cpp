//
//  Basis.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 3/21/13.
//  Copyright (c) 2013 __MyCompanyName__. All rights reserved.
//

#include "Basis.h"

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

template<class Scalar, class ArrayScalar>
void Basis<Scalar,ArrayScalar>::CHECK_VALUES_ARGUMENTS(const ArrayScalar &values, const ArrayScalar &refPoints) {
  // values should have shape: (F,P[,D,D,...]) where the # of D's = rank of the basis's range
  TEUCHOS_TEST_FOR_EXCEPTION(values.rank() != 2 + rangeRank(), std::invalid_argument, "values should have shape (F,P).");
  // refPoints should have shape: (P,D)
  TEUCHOS_TEST_FOR_EXCEPTION(refPoints.rank() != 2, std::invalid_argument, "refPoints should have shape (P,D).");
  TEUCHOS_TEST_FOR_EXCEPTION(refPoints.dimension(1) != domainTopology().getDimension(), std::invalid_argument, "refPoints should have shape (P,D).");
}

template<class Scalar, class ArrayScalar>
IntrepidBasisWrapper<Scalar,ArrayScalar>::IntrepidBasisWrapper(Intrepid::Basis<Scalar,ArrayScalar> intrepidBasis, 
                                                               int rangeDimension, int rangeRank) {
  _intrepidBasis = intrepidBasis;
  _rangeDimension = rangeDimension;
  _rangeRank = rangeRank;
}
template<class Scalar, class ArrayScalar>
int IntrepidBasisWrapper<Scalar,ArrayScalar>::getCardinality() {
  return _intrepidBasis.getCardinality();
}

// domain info on which the basis is defined:

template<class Scalar, class ArrayScalar>
shards::CellTopology IntrepidBasisWrapper<Scalar,ArrayScalar>::domainTopology() {
  return _intrepidBasis.getBaseCellTopology();
}
template<class Scalar, class ArrayScalar>
std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::getSubcellDofs(int subcellDimStart, int subcellDimEnd) {
  shards::CellTopology cellTopo = _intrepidBasis.getBaseCellTopology();
  std::set<int> indices;
  for (int subcellDim = subcellDimStart; subcellDim <= subcellDimEnd; subcellDim++) {
    int numSubcells = cellTopo.getSubcellCount(subcellDim);
    for (int subcellIndex=0; subcellIndex<numSubcells; subcellIndex++) {
      // check that there is at least one dof for the subcell before asking for the first one:
      if (   (_intrepidBasis.getDofOrdinalData().size() > subcellDim)
          && (_intrepidBasis.getDofOrdinalData()[subcellDim].size() > subcellIndex)
          && (_intrepidBasis.getDofOrdinalData()[subcellDim][subcellIndex].size() > 0) ) {
        int firstDofOrdinal = _intrepidBasis.getDofOrdinal(subcellDim, subcellIndex, 0);
        int numDofs = _intrepidBasis.getDofTag(firstDofOrdinal)[3];
        for (int dof=0; dof<numDofs; dof++) {
          indices.insert(_intrepidBasis.getDofOrdinal(subcellDim, subcellIndex, dof));
        }
      }
    }
  }
  return indices;
}

// dof ordinal subsets:
template<class Scalar, class ArrayScalar>
std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForEdges(bool includeVertices) {
  int edgeDim = 1;
  int subcellDimStart = includeVertices ? 0 : edgeDim;
  return getSubcellDofs(subcellDimStart, edgeDim);
}
template<class Scalar, class ArrayScalar>
std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForFaces(bool includeVerticesAndEdges) {
  int faceDim = 2;
  int subcellDimStart = includeVerticesAndEdges ? 0 : faceDim;
  return getSubcellDofs(subcellDimStart, faceDim);
}
template<class Scalar, class ArrayScalar>
std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForInterior() {
  shards::CellTopology cellTopo = domainTopology();
  int dim = cellTopo.getDimension();
  return getSubcellDofs(dim, dim);
}
template<class Scalar, class ArrayScalar>
std::set<int> IntrepidBasisWrapper<Scalar,ArrayScalar>::dofOrdinalsForVertices() {
  int vertexDim = 0;
  return getSubcellDofs(vertexDim, vertexDim);  
}

// range info for basis values:
template<class Scalar, class ArrayScalar>
int IntrepidBasisWrapper<Scalar,ArrayScalar>::rangeDimension() {
  return _rangeDimension;
}
template<class Scalar, class ArrayScalar>
int IntrepidBasisWrapper<Scalar,ArrayScalar>::rangeRank() {
  return _rangeRank;
}
template<class Scalar, class ArrayScalar>
void IntrepidBasisWrapper<Scalar,ArrayScalar>::values(ArrayScalar &values, const ArrayScalar &refPoints) {
  CHECK_VALUES_ARGUMENTS(values,refPoints);
  return _intrepidBasis.getValues(values,refPoints);
}
