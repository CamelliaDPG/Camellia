#ifndef FUNCTION_ND_HPP
#define FUNCTION_ND_HPP

#include <vector>

#include <Teuchos_RCP.hpp>

#include "BOBA/boba.hpp"
#include "CubatureFactory.h"

#include "BasisNd.hpp"
#include "Function1d.hpp"
#include "MeshNd.hpp"
#include "Projection1d.hpp"

template <std::size_t numDim>
class FunctionNd;

template <std::size_t N>
using FunctionNdPtr = Teuchos::RCP<FunctionNd<N>>;

using namespace Camellia;

// multivariate function as sum of tensor products of univariate functions
template <std::size_t numDim>
class FunctionNd {
public:
  FunctionNd() : numTerm_(0), coeffs_(), terms_() {}

  FunctionNd(const boba::Array<Function1dPtr, numDim> &term) : numTerm_(1), coeffs_(1), terms_(1) {
    coeffs_[0] = 1.0;
    terms_[0] = term;
  }

  void
  addTerm(const double newCoeff, const boba::Array<Function1dPtr, numDim> &newTerm) {
    for (std::size_t n = 0; n < numDim; ++n) {
      boba_assert(newTerm[n] != Teuchos::null, "term component is null");
    }

    numTerm_ += 1;
    coeffs_.push_back(newCoeff);
    terms_.push_back(newTerm);
  }

  void
  addTerm(const boba::Array<Function1dPtr, numDim> &newTerm) {
    addTerm(1.0, newTerm);
  }

  friend FunctionNdPtr<numDim>
  operator+(FunctionNdPtr<numDim> f, FunctionNdPtr<numDim> g) {
    auto h = Teuchos::rcp(new FunctionNd<numDim>());
    for (std::size_t i = 0; i < f->numTerm_; ++i) {
      h->addTerm(f->coeffs_[i], f->terms_[i]);
    }
    for (std::size_t j = 0; j < g->numTerm_; ++j) {
      h->addTerm(g->coeffs_[j], g->terms_[j]);
    }
    return h;
  }

  friend FunctionNdPtr<numDim>
  operator-(FunctionNdPtr<numDim> f, FunctionNdPtr<numDim> g) {
    auto h = Teuchos::rcp(new FunctionNd<numDim>());
    for (std::size_t i = 0; i < f->numTerm_; ++i) {
      h->addTerm(f->coeffs_[i], f->terms_[i]);
    }
    for (std::size_t j = 0; j < g->numTerm_; ++j) {
      h->addTerm(-g->coeffs_[j], g->terms_[j]);
    }
    return h;
  }

  friend FunctionNdPtr<numDim>
  operator*(FunctionNdPtr<numDim> f, FunctionNdPtr<numDim> g) {
    auto h = Teuchos::rcp(new FunctionNd<numDim>());
    for (std::size_t i = 0; i < f->numTerm_; ++i) {
      for (std::size_t j = 0; j < g->numTerm_; ++j) {
        auto coeff = f->coeffs_[i] * g->coeffs_[j];
        boba::Array<Function1dPtr, numDim> term;
        for (std::size_t k = 0; k < numDim; ++k) {
          term[k] = f->terms_[i][k] * g->terms_[j][k];
        }
        h->addTerm(coeff, term);
      }
    }

    return h;
  }

  boba::tensor_train<numDim, boba::execution_space::CPU, double, std::size_t>
  getBasisExpansion(BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    boba_assert(basis != Teuchos::null, "basis pointer must be non-null");
    boba_assert(mesh != Teuchos::null, "mesh pointer must be non-null");
    auto basisExpansion = computeBasisExpansion(basis, mesh);
    return getTtRepresentation(basisExpansion, basis, mesh);
  }

  boba::tensor_train<numDim, boba::execution_space::CPU, double, std::size_t>
  getDiscretization(BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    boba_assert(basis != Teuchos::null, "basis pointer must be non-null");
    boba_assert(mesh != Teuchos::null, "mesh pointer must be non-null");
    auto discretization = computeDiscretization(basis, mesh);
    
    if (numTerm_ > 0) {
      std::cout << "DEBUG getDiscretization dim0 term0 first entry = "
                << discretization[0][0].view()(0) << std::endl;
    }
    
    return getTtRepresentation(discretization, basis, mesh);
  }

private:
  void fill_element_coeffs_1d(boba::vector<boba::execution_space::CPU,double,std::size_t>& out,
                              std::size_t elementId,
                              int numBasis,
                              const std::vector<double>& coeffs)
  {
    auto outView = out.view();
    for (int i = 0; i < numBasis; ++i) {
      outView(i + elementId * (std::size_t)numBasis) = coeffs[(std::size_t)i];
    }
  }
  
  std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>>
  computeBasisExpansion(BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>> discretization(numTerm_);

    for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
      auto basis1d = basis->getComponent(dimId);
      auto mesh1d = mesh->getComponent(dimId);

      if (basis1d != Teuchos::null) {
        const int numBasis = basis1d->getCardinality();
        const std::size_t numElement = mesh1d->getNumElement();

        for (std::size_t termId = 0; termId < numTerm_; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(numElement * numBasis);
          auto & discretizationVector = discretization[termId][dimId];

          for (std::size_t elementId = 0; elementId < numElement; ++elementId) {
            const double xL = mesh1d->getElementLeftPoint(elementId);
            const double h  = mesh1d->getElementLength(elementId);

            // project onto basis on this element
            auto coeffs = Projection1d::project_onto_1d_basis_ref(basis1d,
                                                                  [&](double x){ return (*func)(x); },
                                                                  xL, h);

            fill_element_coeffs_1d(discretizationVector, elementId, numBasis, coeffs);
          }
        }
      } else {
        // -*- null basis, only want function evaluation at element boundaries -*-

        //
        // determine element boundaries
        //

        const std::size_t numElement = mesh1d->getNumElement();

        // TODO: relax the hard-coded BCs-applied-on-all-domain-boundaries thing.
        const std::size_t nTrace     = (numElement >= 1) ? (numElement - 1) : 0;
        const std::size_t nTraceCols = std::max<std::size_t>(1, nTrace);
        
        // for now, hard-coding zero Dirichlet boundary condition
        std::vector<double> points(nTrace);
        for (std::size_t elementId = 1; elementId < numElement; ++elementId) {
          points[elementId - 1] = mesh1d->getElementLeftPoint(elementId);
        }

        //
        // evaluate function at element boundaries
        //

        for (std::size_t termId = 0; termId < numTerm_; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(nTraceCols);
          auto discretizationView = discretization[termId][dimId].view();

          for (std::size_t boundaryId = 0; boundaryId < nTrace; ++boundaryId) {
            discretizationView(boundaryId) = (*func)(points[boundaryId]);
          }
          
          // pad dummy trace dofs with 0s.
          for (std::size_t boundaryId = nTrace; boundaryId < nTraceCols; ++boundaryId) {
            discretizationView(boundaryId) = 0.0;
          }

        }
      }
    }

    return discretization;
  }

  std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>>
  computeDiscretization(BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>> discretization(numTerm_);

    for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
      auto basis1d = basis->getComponent(dimId);
      auto mesh1d = mesh->getComponent(dimId);

      if (basis1d != Teuchos::null) {
        const int numBasis = basis1d->getCardinality();
        const std::size_t numElement = mesh1d->getNumElement();

        for (std::size_t termId = 0; termId < numTerm_; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(numElement * numBasis);
          auto & discretizationVector = discretization[termId][dimId];

          for (std::size_t elementId = 0; elementId < numElement; ++elementId) {
            const double xL = mesh1d->getElementLeftPoint(elementId);
            const double h  = mesh1d->getElementLength(elementId);

            
            auto moments = Projection1d::moments_onto_1d_basis_phys(
                             basis1d,
                             [&](double x){ return (*func)(x); },
                             xL, h);

            fill_element_coeffs_1d(discretizationVector, elementId, numBasis, moments);
          }
        }
      } else {
        // -*- null basis, only want function evaluation at element boundaries -*-

        //
        // determine element boundaries
        //

        // TODO: relax the hard-coded BCs-applied-on-all-domain-boundaries thing.
        const std::size_t numElement = mesh1d->getNumElement();
        const std::size_t nTrace     = (numElement >= 1) ? (numElement - 1) : 0;
        const std::size_t nTraceCols = std::max<std::size_t>(1, nTrace);
        
        // for now, hard-coding zero Dirichlet boundary condition
        std::vector<double> points(numElement - 1);
        for (std::size_t elementId = 1; elementId < numElement; ++elementId) {
          points[elementId - 1] = mesh1d->getElementLeftPoint(elementId);
        }

        //
        // evaluate function at element boundaries
        //
        
        for (std::size_t termId = 0; termId < numTerm_; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(nTraceCols);
          auto discretizationView = discretization[termId][dimId].view();

          for (std::size_t boundaryId = 0; boundaryId < nTrace; ++boundaryId) {
            discretizationView(boundaryId) = (*func)(points[boundaryId]);
          }
          // pad dummy trace dofs with 0s.
          for (std::size_t boundaryId = nTrace; boundaryId < nTraceCols; ++boundaryId) {
            discretizationView(boundaryId) = 0.0;
          }
        }
      }
    }

    return discretization;
  }

  boba::tensor_train<numDim, boba::execution_space::CPU, double, std::size_t>
  getTtRepresentation(const std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>> &cp_tensor,
                      BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    boba::tensor_train<numDim, boba::execution_space::CPU, double, std::size_t> output;

    if (numTerm_ == 0) {
      boba::Array<std::size_t, numDim> sizes;
      for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
        auto basis1d = basis->getComponent(dimId);
        auto mesh1d = mesh->getComponent(dimId);
        if (basis1d != Teuchos::null)
        {
          sizes[dimId] = basis1d->getCardinality() * mesh1d->getNumElement();
        }
        else
        {
          // TODO: relax the hardcoded assumption that BCs are always imposed at both ends.
          auto numTraces = mesh1d->getNumElement() - 1;
          sizes[dimId] = std::max<size_t>(1, numTraces); // disallow zero sizes.
        }
      }

      output.setup(sizes);
      output.fill_with_zeros();
    } else {
      output = coeffs_[0] * boba::make_tt_from_vectors(cp_tensor[0]);
      for (std::size_t termId = 1; termId < numTerm_; ++termId) {
        output += coeffs_[termId] * boba::make_tt_from_vectors(cp_tensor[termId]);
      }

      if (numTerm_ > 1) {
        output.round();
      }
    }

    return output;
  }

private:
  std::size_t numTerm_;
  std::vector<double> coeffs_;
  std::vector<boba::Array<Function1dPtr, numDim>> terms_;
};

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
