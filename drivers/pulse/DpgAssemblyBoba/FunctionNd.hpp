#ifndef FUNCTION_ND_HPP
#define FUNCTION_ND_HPP

#include <vector>

#include <Teuchos_RCP.hpp>

#include "CubatureFactory.h"
#include "BOBA/boba.hpp"

#include "BasisNd.hpp"
#include "Function1d.hpp"
#include "MeshNd.hpp"

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
  addTerm(const boba::Array<Function1dPtr, numDim> &newTerm) {
    for (std::size_t n = 0; n < numDim; ++n) {
      boba_assert(newTerm[n] != Teuchos::null, "term component is null");

      coeffs_.push_back(1.0);
      terms_.push_back(newTerm);
    }
  }

  void
  addTerm(const double newCoeff, const boba::Array<Function1dPtr, numDim> &newTerm) {
    for (std::size_t n = 0; n < numDim; ++n) {
      boba_assert(newTerm[n] != Teuchos::null, "term component is null");
    }
    coeffs_.push_back(newCoeff);
    terms_.push_back(newTerm);
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
    return getTtRepresentation(discretization, basis, mesh);
  }

private:
  std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>>
  computeBasisExpansion(BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    const std::size_t numTerm = coeffs_.size();
    std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>> discretization(numTerm);

    for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
      auto basis1d = basis->getComponent(dimId);
      auto mesh1d = mesh->getComponent(dimId);

      if (basis1d != Teuchos::null) {
        //
        // obtain basis nodes
        //

        const std::size_t numBasis = basis1d->getCardinality();
        Intrepid::FieldContainer<double, 0> basisNodes(numBasis, 1);
        basis1d->getDofCoords(basisNodes);

        //
        // determine shifts and scalings for element points
        //

        const std::size_t numElement = mesh1d->getNumElement();

        std::vector<double> shifts(numElement);
        std::vector<double> scales(numElement);
        for (std::size_t elementId = 0; elementId < numElement; ++elementId) {
          shifts[elementId] = mesh1d->getElementMidpoint(elementId);
          scales[elementId] = 0.5 * mesh1d->getElementLength(elementId);
        }

        //
        // evaluate function at basis nodes
        //

        for (std::size_t termId = 0; termId < numTerm; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(numElement * numBasis);
          auto discretizationView = discretization[termId][dimId].view();

          for (std::size_t elementId = 0; elementId < numElement; ++elementId) {
            for (std::size_t basisId = 0; basisId < numBasis; ++basisId) {
              const int j = static_cast<int>(basisId);
              discretizationView(basisId + elementId * numBasis) = (*func)(shifts[elementId] + scales[elementId] * basisNodes(j, 0));
            }
          }
        }
      } else {
        // -*- null basis, only want function evaluation at element boundaries -*-

        //
        // determine element boundaries
        //

        const std::size_t numElement = mesh1d->getNumElement();

        // for now, hard-coding zero Dirichlet boundary condition
        std::vector<double> points(numElement - 1);
        for (std::size_t elementId = 1; elementId < numElement; ++elementId) {
          points[elementId - 1] = mesh1d->getElementLeftPoint(elementId);
        }

        //
        // evaluate function at element boundaries
        //

        for (std::size_t termId = 0; termId < numTerm; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(numElement - 1);
          auto discretizationView = discretization[termId][dimId].view();

          for (std::size_t boundaryId = 0; boundaryId < numElement - 1; ++boundaryId) {
            discretizationView(boundaryId) = (*func)(points[boundaryId]);
          }
        }
      }
    }

    return discretization;
  }

  std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>>
  computeDiscretization(BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    const std::size_t numTerm = coeffs_.size();
    std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>> discretization(numTerm);

    for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
      auto basis1d = basis->getComponent(dimId);
      auto mesh1d = mesh->getComponent(dimId);

      if (basis1d != Teuchos::null) {
        //
        // construct cubature
        //

        const int cubatureDegree = 2 * basis1d->getDegree(); // making an assumption here...

        auto cubature = Camellia::CubatureFactory().create(Camellia::CellTopology::line(), cubatureDegree);

        auto cubaturePoints = cubature->allocateCubaturePoints();
        auto cubatureWeights = cubature->allocateCubatureWeights();
        cubature->getCubature(cubaturePoints, cubatureWeights);

        const std::size_t numCubature = cubaturePoints.extent(0);

        //
        // evaluate basis at cubature points
        //

        auto basisValues = basis1d->allocateBasisValues(cubaturePoints, Intrepid2::OPERATOR_VALUE);
        basis1d->getValues(basisValues, cubaturePoints, Intrepid2::OPERATOR_VALUE);

        const std::size_t numBasis = basisValues.extent(0);

        //
        // determine shifts and scalings for element points
        //

        const std::size_t numElement = mesh1d->getNumElement();

        std::vector<double> shifts(numElement);
        std::vector<double> scales(numElement);
        for (std::size_t elementId = 0; elementId < numElement; ++elementId) {
          shifts[elementId] = mesh1d->getElementMidpoint(elementId);
          scales[elementId] = 0.5 * mesh1d->getElementLength(elementId);
        }

        //
        // compute coefficients through integration/quadrature
        //

        for (std::size_t termId = 0; termId < numTerm; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(numElement * numBasis);
          auto discretizationView = discretization[termId][dimId].view();

          for (std::size_t elementId = 0; elementId < numElement; ++elementId) {
            for (std::size_t basisId = 0; basisId < numBasis; ++basisId) {
              double value = 0.0;
              for (std::size_t cubatureId = 0; cubatureId < numCubature; ++cubatureId) {
                const int j = static_cast<int>(basisId);
                const int k = static_cast<int>(cubatureId);
                value += basisValues(j, k) * (*func)(shifts[elementId] + scales[elementId] * cubaturePoints(k, 0)) * scales[elementId] * cubatureWeights(k);
              }
              discretizationView(basisId + elementId * numBasis) = value;
            }
          }
        }
      } else {
        // -*- null basis, only want function evaluation at element boundaries -*-

        //
        // determine element boundaries
        //

        const std::size_t numElement = mesh1d->getNumElement();

        // for now, hard-coding zero Dirichlet boundary condition
        std::vector<double> points(numElement - 1);
        for (std::size_t elementId = 1; elementId < numElement; ++elementId) {
          points[elementId - 1] = mesh1d->getElementLeftPoint(elementId);
        }

        //
        // evaluate function at element boundaries
        //

        for (std::size_t termId = 0; termId < numTerm; ++termId) {
          auto func = terms_[termId][dimId];

          discretization[termId][dimId].resize(numElement - 1);
          auto discretizationView = discretization[termId][dimId].view();

          for (std::size_t boundaryId = 0; boundaryId < numElement - 1; ++boundaryId) {
            discretizationView(boundaryId) = (*func)(points[boundaryId]);
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

    std::size_t numTerm = coeffs_.size();
    if (numTerm == 0) {
      boba::Array<std::size_t, numDim> sizes;
      for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
        auto basis1d = basis->getComponent(dimId);
        auto mesh1d = mesh->getComponent(dimId);
        sizes[dimId] = (basis1d != Teuchos::null) ? (basis1d->getCardinality() * mesh1d->getNumElement())
                                                  : (mesh1d->getNumElement() - 1); // for now, hard-coding zero Dirichlet boundary condition
      }

      output.setup(sizes);
      output.fill_with_zeros();
    } else {
      output = coeffs_[0] * boba::make_tt_from_vectors(cp_tensor[0]);
      for (std::size_t termId = 1; termId < numTerm; ++termId) {
        output += coeffs_[termId] * boba::make_tt_from_vectors(cp_tensor[termId]);
      }

      if (numTerm > 1) {
        output.round();
      }
    }

    return output;
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
