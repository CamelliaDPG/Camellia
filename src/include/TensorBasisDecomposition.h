#ifndef Camellia_TensorBasisDecomposition_h
#define Camellia_TensorBasisDecomposition_h

//
//  TensorBasisDecomposition.h
//  Camellia
//
//  A basis on a hypercube that is a tensor product of one-dimensional bases (Intrepid2::Basis_TensorBasis), or a
//  serendipity subset of one (Intrepid2::SerendipityBasis), decomposed into its one-dimensional components: what
//  sum-factorized kernels on box cells need.
//

#include "TypeDefs.h"

#include "Intrepid2_Basis.hpp"

#include <vector>

namespace Camellia
{
  struct TensorBasisDecomposition
  {
    using Intrepid2Basis = Intrepid2::Basis<Camellia::DeviceType,double,double>;

    bool valid = false;
    std::vector<Teuchos::RCP<Intrepid2Basis>> components; // [direction]: one-dimensional bases on [-1,1]
    std::vector<int> componentCardinality;              // [direction]
    std::vector<std::vector<int>> componentOrdinals;     // [basisOrdinal][direction]
    //! [basisOrdinal]: the index in the full tensor-product space, sum_k componentOrdinals[k] * stride_k, first direction fastest
    std::vector<int> tensorIndex;
    int fullSize = 0;        // prod_k componentCardinality[k]
    bool fullTensor = false; // every function of the full tensor-product basis is present

    /**
     \brief Decompose basis (on a hypercube of dimension spaceDim) into one-dimensional components.

     Returns valid == false unless basis is an Intrepid2 tensor basis of spaceDim one-dimensional components, or a
     serendipity basis whose underlying basis is one; a serendipity basis is the subset of its underlying basis with
     superlinear H^1 degree at most the basis's H^1 degree (Intrepid2::SerendipityBasis's selection).
     */
    static TensorBasisDecomposition decompose(BasisPtr basis, int spaceDim);

    /**
     \brief Values and reference derivatives of the one-dimensional component basis of a direction at points of [-1,1].

     values and derivatives are row-major (points x cardinality); derivatives may be null (and must be, for components
     without OPERATOR_GRAD).
     */
    void componentValues(int direction, const std::vector<double> &points, std::vector<double> &values,
                         std::vector<double> *derivatives = nullptr) const;
  };
}

#endif
