#ifndef DISCRETIZED_FUNCTION_HPP
#define DISCRETIZED_FUNCTION_HPP

#include <vector>

#include "BOBA/boba.hpp"
#include "CubatureFactory.h"

#include "BasisNd.hpp"
#include "FunctionNd.hpp"
#include "MeshNd.hpp"

// multivariate function discretization over tensor-product mesh and basis
template <std::size_t numDim>
class DiscretizedFunction {
public:
  DiscretizedFunction(FunctionNdPtr<numDim> func, BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
    boba_assert(func != Teuchos::null, "function pointer must be non-null");
    boba_assert(basis != Teuchos::null, "basis pointer must be non-null");
    boba_assert(mesh != Teuchos::null, "mesh pointer must be non-null");

    func_ = func;
    mesh_ = mesh;
    basis_ = basis;

    computeDiscretization();
  }

  boba::tensor_train<numDim, boba::execution_space::CPU, double, std::size_t>
  getTtRepresentation(bool round = false) const {
    boba::tensor_train<numDim, boba::execution_space::CPU, double, std::size_t> output;

    std::size_t numTerm = func_->getNumTerm();
    if (numTerm == 0) {
      boba::Array<std::size_t, numDim> sizes;
      for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
        auto basis = basis_->getComponent(dimId);
        auto mesh = mesh_->getComponent(dimId);
        sizes[dimId] = (basis != Teuchos::null) ? basis->getCardinality() * mesh->getNumElement()
                                                : (mesh->getNumElement() - 1); // for now, hard-coding zero Dirichlet boundary condition
      }

      output.setup(sizes);
      output.fill_with_zeros();
    } else {
      output = func_->getTermCoeff(0) * boba::make_tt_from_vectors(discretization_[0]);
      for (std::size_t termId = 1; termId < numTerm; ++termId) {
        output += func_->getTermCoeff(termId) * boba::make_tt_from_vectors(discretization_[termId]);
      }

      if (round && numTerm > 1) {
        output.round();
      }
    }

    return output;
  }

private:
  void
  computeDiscretization() {
    const std::size_t numTerm = func_->getNumTerm();
    discretization_.resize(numTerm);

    for (std::size_t dimId = 0; dimId < numDim; ++dimId) {
      const auto &basis = basis_->getComponent(dimId);
      const auto &mesh = mesh_->getComponent(dimId);

      if (basis != Teuchos::null) {
        //
        // construct cubature
        //

        const int cubatureDegree = 2 * basis->getDegree(); // making an assumption here...

        auto cubature = Camellia::CubatureFactory().create(Camellia::CellTopology::line(), cubatureDegree);

        auto cubaturePoints = cubature->allocateCubaturePoints();
        auto cubatureWeights = cubature->allocateCubatureWeights();
        cubature->getCubature(cubaturePoints, cubatureWeights);

        const std::size_t numCubature = cubaturePoints.extent(0);

        //
        // evaluate basis at cubature points
        //

        auto basisValues = basis->allocateBasisValues(cubaturePoints, Intrepid2::OPERATOR_VALUE);
        basis->getValues(basisValues, cubaturePoints, Intrepid2::OPERATOR_VALUE);

        const std::size_t numBasis = basisValues.extent(0);

        //
        // determine shifts and scalings for element points
        //

        const std::size_t numElement = mesh->getNumElement();

        std::vector<double> shifts(numElement);
        std::vector<double> scales(numElement);
        for (std::size_t elementId = 0; elementId < numElement; ++elementId) {
          shifts[elementId] = mesh->getElementMidpoint(elementId);
          scales[elementId] = 0.5 * mesh->getElementLength(elementId);
        }

        //
        // compute coefficients through integration/quadrature
        //

        for (std::size_t termId = 0; termId < numTerm; ++termId) {
          auto func = func_->getTermComponent(termId, dimId);

          discretization_[termId][dimId].resize(numElement * numBasis);
          auto discretizationView = discretization_[termId][dimId].view();

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

        const std::size_t numElement = mesh->getNumElement();

       // for now, hard-coding zero Dirichlet boundary condition
        std::vector<double> points(numElement - 1);
        for (std::size_t elementId = 1; elementId < numElement; ++elementId) {
          points[elementId - 1] = mesh->getElementLeftPoint(elementId);
        }

        //
        // evaluate functions
        //

        for (std::size_t termId = 0; termId < numTerm; ++termId) {
          auto func = func_->getTermComponent(termId, dimId);

          discretization_[termId][dimId].resize(numElement - 1);
          auto discretizationView = discretization_[termId][dimId].view();

          for (std::size_t boundaryId = 0; boundaryId < numElement - 1; ++boundaryId) {
            discretizationView(boundaryId) = (*func)(points[boundaryId]);
          }
        }
      }
    }
  }

private:
  FunctionNdPtr<numDim> func_;
  MeshNdPtr<numDim> mesh_;
  BasisNdPtr<numDim> basis_;
  std::vector<boba::Array<boba::vector<boba::execution_space::CPU, double, std::size_t>, numDim>> discretization_;
};

template <std::size_t numDim>
using DiscretizedFunctionPtr = Teuchos::RCP<DiscretizedFunction<numDim>>;

template <std::size_t numDim>
DiscretizedFunctionPtr<numDim>
discretizeFunction(FunctionNdPtr<numDim> func, BasisNdPtr<numDim> basis, MeshNdPtr<numDim> mesh) {
  return Teuchos::rcp(new DiscretizedFunction<numDim>(func, basis, mesh));
}

#endif
