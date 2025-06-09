#ifndef BASIS_ND_HPP
#define BASIS_ND_HPP

#include <Teuchos_RCP.hpp>

#include "BOBA/boba.hpp"

#include "Basis1d.hpp"

// tensor product basis
template <std::size_t numDim>
class BasisNd {
public:
  BasisNd(const boba::Array<Basis1dPtr, numDim> &basis) {
    for (std::size_t n = 0; n < numDim; ++n) {
      boba_assert(basis[n] != Teuchos::null, "function component is null");
    }
    basis_ = basis;
  }

  Basis1dPtr
  getComponent(std::size_t dimId) const {
    boba_assert(dimId < numDim, "dimension index is out of range");
    return basis_[dimId];
  }

private:
  boba::Array<Basis1dPtr, numDim> basis_;
};

template <std::size_t N>
using BasisNdPtr = Teuchos::RCP<BasisNd<N>>;

template <std::size_t N>
BasisNdPtr<N>
newBasisNd(const boba::Array<Basis1dPtr, N> &basis) {
  return Teuchos::rcp(new BasisNd<N>(basis));
}

#endif
