#ifndef FUNCTION_ND_HPP
#define FUNCTION_ND_HPP

#include <vector>

#include <Teuchos_RCP.hpp>

#include "BOBA/boba.hpp"

#include "Function1d.hpp"

// multivariate function as sum of tensor products of univariate functions
template <std::size_t numDim>
class FunctionNd {
public:
  FunctionNd() : coeffs_(), terms_() {}

  FunctionNd(const boba::Array<Function1dPtr, numDim> &term) : coeffs_(1), terms_(1) {
    coeffs_[0] = 1.0;
    terms_[0] = term;
  }

  void
  addTerm(const double newCoeff, const boba::Array<Function1dPtr, numDim> &newTerm) {
    for (std::size_t n = 0; n < numDim; ++n) {
      boba_assert(newTerm[n] != Teuchos::null, "term component is null");
    }
    coeffs_.push_back(newCoeff);
    terms_.push_back(newTerm);
  }

  std::size_t
  getNumTerm() const {
    return coeffs_.size();
  }

  double
  getTermCoeff(std::size_t termId) const {
    boba_assert(termId < coeffs_.size(), "term index is out of range");
    return coeffs_[termId];
  }

  Function1dPtr
  getTermComponent(std::size_t termId, std::size_t dimId) const {
    boba_assert(termId < terms_.size(), "term index is out of range");
    boba_assert(dimId < numDim, "dimension index is out of range");
    return terms_[termId][dimId];
  }

private:
  std::vector<double> coeffs_;
  std::vector<boba::Array<Function1dPtr, numDim>> terms_;
};

template <std::size_t N>
using FunctionNdPtr = Teuchos::RCP<FunctionNd<N>>;

template <std::size_t N>
FunctionNdPtr<N>
newFunctionNd() {
  return Teuchos::rcp(new FunctionNd<N>());
}

template <std::size_t N>
FunctionNdPtr<N>
newFunctionNd(const boba::Array<Function1dPtr, N> &term) {
  return Teuchos::rcp(new FunctionNd<N>(term));
}

#endif
