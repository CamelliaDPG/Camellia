#include "TensorBasisCache.h"

#include <algorithm>
#include <stdexcept>

#include "CamelliaIntrepidExtendedTypes.h"
#include "CellTopology.h"
#include "CubatureFactory.h"
#include "Teuchos_TestForException.hpp"
#include "Var.h"

using namespace Camellia;

namespace
{
  struct LineQuadratureRule
  {
    std::vector<double> points;
    std::vector<double> weights;
  };

  LineQuadratureRule lineQuadratureFromCubatureFactory(int exactnessDegree)
  {
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), std::max(1, exactnessDegree));

    const int pointCount = cubature->getNumPoints();
    auto points = cubature->allocateCubaturePoints();
    auto weights = cubature->allocateCubatureWeights();

    cubature->getCubature(points, weights);

    auto pointsView = points.getTensorComponent(0);
    auto weightsView = weights.getTensorComponent(0).getUnderlyingView();
    auto pointsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pointsView);
    auto weightsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weightsView);

    LineQuadratureRule rule;
    rule.points.resize(pointCount);
    rule.weights.resize(pointCount);
    for (int pointOrdinal=0; pointOrdinal<pointCount; pointOrdinal++)
    {
      rule.points[pointOrdinal] = pointsHost(pointOrdinal,0);
      rule.weights[pointOrdinal] = weightsHost(pointOrdinal);
    }
    return rule;
  }
}

TensorBasisCache1D::TensorBasisCache1D(const TensorMeshPtr &tensorMesh,
                                       const TensorVarDiscretizationPtr &discretization,
                                       int componentOrdinal,
                                       bool isTestSpace,
                                       const ComponentCellID1D &cellID,
                                       int quadratureExactnessDegree)
: _componentOrdinal(componentOrdinal),
  _isTestSpace(isTestSpace)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorBasisCache1D requires a non-null TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(discretization == Teuchos::null, std::invalid_argument,
                             "TensorBasisCache1D requires a non-null TensorVarDiscretization.");
  TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= tensorMesh->topology()->componentCount()), std::invalid_argument,
                             "TensorBasisCache1D component ordinal out of range.");

  auto componentTopo = tensorMesh->topology()->componentMeshTopology(componentOrdinal);
  TEUCHOS_TEST_FOR_EXCEPTION(!componentTopo->containsCell(cellID), std::invalid_argument,
                             "TensorBasisCache1D requires an active component cell.");

  _polyOrder = tensorMesh->effectivePolyOrder(discretization, cellID, componentOrdinal, isTestSpace);
  auto basisFamily = tensorMesh->effectiveBasisFamily(discretization);
  auto componentVar = discretization->tensorVar()->componentVar(componentOrdinal);
  BasisPtr basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(componentVar->space()), _polyOrder);
  _basisCardinality = basis->getCardinality();

  const auto quadrature = lineQuadratureFromCubatureFactory(quadratureExactnessDegree);
  _pointCount = static_cast<int>(quadrature.points.size());
  _cellLeft = componentTopo->cellLeft(cellID);
  _cellWidth = componentTopo->cellWidth(cellID);
  _jacobian = 0.5 * _cellWidth;
  _derivativeScale = 2.0 / _cellWidth;

  _referencePoints = PointDualView("tensor_basis_cache_1d_reference_points", _pointCount);
  _referenceWeights = WeightDualView("tensor_basis_cache_1d_reference_weights", _pointCount);
  _valuesAtQuadrature = BasisValueDualView("tensor_basis_cache_1d_values", _basisCardinality, _pointCount);
  _derivativesAtQuadrature = BasisValueDualView("tensor_basis_cache_1d_derivatives", _basisCardinality, _pointCount);

  Camellia::FieldContainer<double> refPointContainer(_pointCount, 1);
  for (int pointOrdinal=0; pointOrdinal<_pointCount; pointOrdinal++)
  {
    _referencePoints.h_view(pointOrdinal) = quadrature.points[pointOrdinal];
    _referenceWeights.h_view(pointOrdinal) = quadrature.weights[pointOrdinal];
    refPointContainer(pointOrdinal,0) = quadrature.points[pointOrdinal];
  }

  Camellia::FieldContainer<double> basisValues(_basisCardinality, _pointCount);
  basis->getValues(basisValues, refPointContainer, OP_VALUE);
  for (int basisOrdinal=0; basisOrdinal<_basisCardinality; basisOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<_pointCount; pointOrdinal++)
    {
      _valuesAtQuadrature.h_view(basisOrdinal,pointOrdinal) = basisValues(basisOrdinal,pointOrdinal);
    }
  }

  Camellia::FieldContainer<double> basisDerivatives(_basisCardinality, _pointCount, 1);
  basis->getValues(basisDerivatives, refPointContainer, OP_GRAD);
  for (int basisOrdinal=0; basisOrdinal<_basisCardinality; basisOrdinal++)
  {
    for (int pointOrdinal=0; pointOrdinal<_pointCount; pointOrdinal++)
    {
      _derivativesAtQuadrature.h_view(basisOrdinal,pointOrdinal) = basisDerivatives(basisOrdinal,pointOrdinal,0);
    }
  }

  _referencePoints.modify_host();
  _referencePoints.sync_device();
  _referenceWeights.modify_host();
  _referenceWeights.sync_device();
  _valuesAtQuadrature.modify_host();
  _valuesAtQuadrature.sync_device();
  _derivativesAtQuadrature.modify_host();
  _derivativesAtQuadrature.sync_device();
}

int TensorBasisCache1D::componentOrdinal() const
{
  return _componentOrdinal;
}

bool TensorBasisCache1D::isTestSpace() const
{
  return _isTestSpace;
}

int TensorBasisCache1D::polyOrder() const
{
  return _polyOrder;
}

int TensorBasisCache1D::basisCardinality() const
{
  return _basisCardinality;
}

int TensorBasisCache1D::pointCount() const
{
  return _pointCount;
}

double TensorBasisCache1D::cellLeft() const
{
  return _cellLeft;
}

double TensorBasisCache1D::cellWidth() const
{
  return _cellWidth;
}

double TensorBasisCache1D::jacobian() const
{
  return _jacobian;
}

double TensorBasisCache1D::derivativeScale() const
{
  return _derivativeScale;
}

const TensorBasisCache1D::PointDualView &TensorBasisCache1D::referencePoints() const
{
  return _referencePoints;
}

const TensorBasisCache1D::WeightDualView &TensorBasisCache1D::referenceWeights() const
{
  return _referenceWeights;
}

const TensorBasisCache1D::BasisValueDualView &TensorBasisCache1D::valuesAtQuadrature() const
{
  return _valuesAtQuadrature;
}

const TensorBasisCache1D::BasisValueDualView &TensorBasisCache1D::derivativesAtQuadrature() const
{
  return _derivativesAtQuadrature;
}

TensorBasisCacheND::TensorBasisCacheND(const TensorMeshPtr &tensorMesh,
                                       const TensorVarDiscretizationPtr &discretization,
                                       bool isTestSpace,
                                       const TensorMeshTopology::TensorCellTuple &cellTuple,
                                       const std::vector<int> &quadratureExactnessDegrees)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorBasisCacheND requires a non-null TensorMesh.");
  TEUCHOS_TEST_FOR_EXCEPTION(discretization == Teuchos::null, std::invalid_argument,
                             "TensorBasisCacheND requires a non-null TensorVarDiscretization.");
  const int componentCount = tensorMesh->topology()->componentCount();
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(cellTuple.size()) != componentCount, std::invalid_argument,
                             "TensorBasisCacheND requires one component cell per tensor component.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(quadratureExactnessDegrees.size()) != componentCount, std::invalid_argument,
                             "TensorBasisCacheND requires one quadrature degree per tensor component.");

  _componentCaches.reserve(componentCount);
  for (int d=0; d<componentCount; d++)
  {
    _componentCaches.push_back(TensorBasisCache1D(tensorMesh,
                                                  discretization,
                                                  d,
                                                  isTestSpace,
                                                  cellTuple[d],
                                                  quadratureExactnessDegrees[d]));
  }
  buildTensorPointData();
}

void TensorBasisCacheND::buildTensorPointData()
{
  const int componentCount = static_cast<int>(_componentCaches.size());
  _pointStrides = PointStrideDualView("tensor_basis_cache_nd_point_strides", componentCount);

  _tensorPointCount = 1;
  for (int d=0; d<componentCount; d++)
  {
    _pointStrides.h_view(d) = _tensorPointCount;
    _tensorPointCount *= _componentCaches[d].pointCount();
  }

  _physicalPoints = TensorPointDualView("tensor_basis_cache_nd_physical_points", _tensorPointCount, componentCount);
  _quadratureWeights = TensorWeightDualView("tensor_basis_cache_nd_quadrature_weights", _tensorPointCount);

  std::vector<int> pointOrdinals(componentCount, 0);
  for (IndexType tensorPointOrdinal=0; tensorPointOrdinal<_tensorPointCount; tensorPointOrdinal++)
  {
    IndexType workingOrdinal = tensorPointOrdinal;
    double tensorWeight = 1.0;
    for (int d=0; d<componentCount; d++)
    {
      const auto &cache = _componentCaches[d];
      const int pointOrdinal = static_cast<int>(workingOrdinal % cache.pointCount());
      workingOrdinal /= cache.pointCount();
      pointOrdinals[d] = pointOrdinal;

      const double xi = cache.referencePoints().h_view(pointOrdinal);
      _physicalPoints.h_view(tensorPointOrdinal,d) = cache.cellLeft() + 0.5 * (xi + 1.0) * cache.cellWidth();
      tensorWeight *= cache.referenceWeights().h_view(pointOrdinal) * cache.jacobian();
    }
    _quadratureWeights.h_view(tensorPointOrdinal) = tensorWeight;
  }

  _pointStrides.modify_host();
  _pointStrides.sync_device();
  _physicalPoints.modify_host();
  _physicalPoints.sync_device();
  _quadratureWeights.modify_host();
  _quadratureWeights.sync_device();
}

int TensorBasisCacheND::componentCount() const
{
  return static_cast<int>(_componentCaches.size());
}

const TensorBasisCache1D &TensorBasisCacheND::componentCache(int componentOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= componentCount()), std::invalid_argument,
                             "TensorBasisCacheND::componentCache component ordinal out of range.");
  return _componentCaches[componentOrdinal];
}

const TensorBasisCacheND::PointStrideDualView &TensorBasisCacheND::pointStrides() const
{
  return _pointStrides;
}

IndexType TensorBasisCacheND::pointStride(int componentOrdinal) const
{
  TEUCHOS_TEST_FOR_EXCEPTION((componentOrdinal < 0) || (componentOrdinal >= componentCount()), std::invalid_argument,
                             "TensorBasisCacheND::pointStride component ordinal out of range.");
  return _pointStrides.h_view(componentOrdinal);
}

IndexType TensorBasisCacheND::tensorPointCount() const
{
  return _tensorPointCount;
}

const TensorBasisCacheND::TensorPointDualView &TensorBasisCacheND::physicalPoints() const
{
  return _physicalPoints;
}

const TensorBasisCacheND::TensorWeightDualView &TensorBasisCacheND::quadratureWeights() const
{
  return _quadratureWeights;
}
