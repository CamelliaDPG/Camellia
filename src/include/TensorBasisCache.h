#ifndef Camellia_TensorBasisCache_h
#define Camellia_TensorBasisCache_h

#include <vector>

#include <Kokkos_DualView.hpp>

#include "Basis.h"
#include "TensorMesh.h"
#include "TensorVarDiscretization.h"

namespace Camellia
{
  /**
   * \brief Cached one-dimensional basis/quadrature data for one tensor component cell.
   *
   * This is the first Phase 2 setup object.  It tabulates reference quadrature points,
   * weights, basis values, reference derivatives, and affine physical scaling once so
   * later tensor-product kernels can reuse flat Kokkos views rather than repeatedly
   * allocating FieldContainers or rebuilding basis objects inside element loops.
   */
  class TensorBasisCache1D
  {
  public:
    using PointDualView = Kokkos::DualView<double*>;
    using WeightDualView = Kokkos::DualView<double*>;
    using BasisValueDualView = Kokkos::DualView<double**>;

  private:
    int _componentOrdinal = -1;
    bool _isTestSpace = false;
    int _polyOrder = -1;
    int _basisCardinality = 0;
    int _pointCount = 0;
    double _cellLeft = 0.0;
    double _cellWidth = 0.0;
    double _jacobian = 0.0;
    double _derivativeScale = 0.0;
    PointDualView _referencePoints;
    WeightDualView _referenceWeights;
    BasisValueDualView _valuesAtQuadrature;
    BasisValueDualView _derivativesAtQuadrature;

  public:
    TensorBasisCache1D() = default;
    TensorBasisCache1D(const TensorMeshPtr &tensorMesh,
                       const TensorVarDiscretizationPtr &discretization,
                       int componentOrdinal,
                       bool isTestSpace,
                       const ComponentCellID1D &cellID,
                       int quadratureExactnessDegree);

    int componentOrdinal() const;
    bool isTestSpace() const;
    int polyOrder() const;
    int basisCardinality() const;
    int pointCount() const;
    double cellLeft() const;
    double cellWidth() const;
    double jacobian() const;
    double derivativeScale() const;

    const PointDualView &referencePoints() const;
    const WeightDualView &referenceWeights() const;
    const BasisValueDualView &valuesAtQuadrature() const;
    const BasisValueDualView &derivativesAtQuadrature() const;
  };

  /**
   * \brief Tensor-product cache assembled from one-dimensional component caches.
   *
   * The ND cache stores component point strides, the total tensor-product point count,
   * physical point coordinates, and quadrature weights for an affine tensor cell.  It
   * preserves the lower-component-fastest ordering used elsewhere by tensor topology
   * and local DOF mixed-radix numbering.
   */
  class TensorBasisCacheND
  {
  public:
    using PointStrideDualView = Kokkos::DualView<IndexType*>;
    using TensorPointDualView = Kokkos::DualView<double**>;
    using TensorWeightDualView = Kokkos::DualView<double*>;

  private:
    std::vector<TensorBasisCache1D> _componentCaches;
    PointStrideDualView _pointStrides;
    IndexType _tensorPointCount = 0;
    TensorPointDualView _physicalPoints;
    TensorWeightDualView _quadratureWeights;

    void buildTensorPointData();

  public:
    TensorBasisCacheND() = default;
    TensorBasisCacheND(const TensorMeshPtr &tensorMesh,
                       const TensorVarDiscretizationPtr &discretization,
                       bool isTestSpace,
                       const TensorMeshTopology::TensorCellTuple &cellTuple,
                       const std::vector<int> &quadratureExactnessDegrees);

    int componentCount() const;
    const TensorBasisCache1D &componentCache(int componentOrdinal) const;
    const PointStrideDualView &pointStrides() const;
    IndexType pointStride(int componentOrdinal) const;
    IndexType tensorPointCount() const;
    const TensorPointDualView &physicalPoints() const;
    const TensorWeightDualView &quadratureWeights() const;
  };
}

#endif
