#ifndef MESH_1D_HPP
#define MESH_1D_HPP

#include <vector>

#include <Teuchos_RCP.hpp>

#include "BOBA/boba.hpp"

// 1D mesh
class Mesh1d {
public:
  // uniform mesh
  Mesh1d(double domainMin, double domainMax, std::size_t numElement) {
    boba_assert(domainMin < domainMax, "domain's minimum extent must be smaller than maximum extent");
    boba_assert(numElement > 0, "number of elements must be positive");

    domainMin_ = domainMin;
    domainMax_ = domainMax;

    numElement_ = numElement;

    elementBoundary_.resize(numElement + 1);
    const double elementLength = (domainMax - domainMin) / numElement;
    for (std::size_t i = 0; i <= numElement; ++i) {
      elementBoundary_[i] = domainMin + i * elementLength;
    }
  }

  std::size_t
  getNumElement() const {
    return numElement_;
  }

  double
  getElementLeftPoint(std::size_t index) {
    boba_assert(index < numElement_, "element index is out of range");
    return elementBoundary_[index];
  }

  double
  getElementRightPoint(std::size_t index) {
    boba_assert(index < numElement_, "element index is out of range");
    return elementBoundary_[index + 1];
  }

  double
  getElementMidpoint(std::size_t index) {
    boba_assert(index < numElement_, "element index is out of range");
    return 0.5 * (elementBoundary_[index] + elementBoundary_[index + 1]);
  }

  double
  getElementLength(std::size_t index) {
    boba_assert(index < numElement_, "element index is out of range");
    return elementBoundary_[index + 1] - elementBoundary_[index];
  }

private:
  double domainMin_;
  double domainMax_;
  std::size_t numElement_;
  std::vector<double> elementBoundary_;
};

using Mesh1dPtr = Teuchos::RCP<Mesh1d>;

inline Mesh1dPtr
newMesh1d(double domainMin, double domainMax, std::size_t numElement) {
  return Teuchos::rcp(new Mesh1d(domainMin, domainMax, numElement));
}

#endif
