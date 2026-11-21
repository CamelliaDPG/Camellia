//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  RelativisticVlasovSpaceTimeFormulation.cpp
//  Camellia
//
//  Created by Nate Roberts on 10/15/25.
//
//

#include "RelativisticVlasovSpaceTimeFormulation.h"

#include "AbsFunction.h"
#include "BasisCache.h"
#include "Cell.h"
#include "Mesh.h"
#include "MeshTopologyView.h"
#include "MeshTools.h"
#include "MonomialFunctions.h"
#include "SqrtFunction.h"
#include "SeparableTensorIP.h"
#include "SolutionMomentFunction.h"

using namespace Camellia;
using std::vector;
using std::string;

namespace
{
  //! alpha_K = max(alpha, scale / h_x^2) on each element, where h_x is the element's extent in the first (spatial)
  //! coordinate.  The transport part of the graph norm scales like (|beta| / h)^2, so on a graded spatial mesh a fixed
  //! alpha that controls the large elements leaves the small ones under-weighted; this keeps alpha at least that
  //! large on every element.
  class MeshScaledL2Weight : public TFunction<double>
  {
    double _alpha, _scale;
  public:
    MeshScaledL2Weight(double alpha, double scale) : TFunction<double>(0), _alpha(alpha), _scale(scale) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const FieldContainer<double> &nodes = basisCache->getPhysicalCellNodes();
      const int numCells  = values.dimension(0);
      const int numPoints = values.dimension(1);
      const int numNodes  = nodes.dimension(1);
      TEUCHOS_TEST_FOR_EXCEPTION((nodes.rank() != 3) || (nodes.dimension(0) != numCells) || (numNodes == 0), std::invalid_argument,
                                 "MeshScaledL2Weight: the BasisCache's physical cell nodes do not match the requested values");
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        double xMin = nodes(cellOrdinal,0,0), xMax = nodes(cellOrdinal,0,0);
        for (int node=1; node<numNodes; node++)
        {
          xMin = std::min(xMin, nodes(cellOrdinal,node,0));
          xMax = std::max(xMax, nodes(cellOrdinal,node,0));
        }
        const double h = xMax - xMin;
        const double weight = std::max(_alpha, _scale / (h * h));
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          values(cellOrdinal,pointOrdinal) = weight;
        }
      }
    }
    string displayString() override { return "alpha_K"; }
  };

  //! alpha_K = alpha * prod_d (1 + s_d / (alpha h_d(K)^2)) on each element, where h_d is the element's extent in
  //! coordinate d (x, t, u_x, u_y, ...) and s_d >= 0 is the scale for that direction (zero, or omitted, to leave d out).
  //! Each factor depends only on the element's cell in one coordinate direction, so on a tensor-product mesh the
  //! weighted mass term is a Kronecker product of one-dimensional diagonally scaled masses: it adds no tensor rank.
  //! Since prod_d (1 + a_d) >= 1 + sum_d a_d, alpha_K is at least alpha + sum_d s_d / h_d^2, the sum over directions of
  //! the transport scale (|beta_d| / h_d)^2 when s_d ~ C beta_d^2.
  class DirectionProductL2Weight : public TFunction<double>
  {
    double _alpha;
    vector<double> _scales;
  public:
    DirectionProductL2Weight(double alpha, const vector<double> &scales) : TFunction<double>(0), _alpha(alpha), _scales(scales) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const FieldContainer<double> &nodes = basisCache->getPhysicalCellNodes();
      const int numCells  = values.dimension(0);
      const int numPoints = values.dimension(1);
      const int numNodes  = nodes.dimension(1);
      TEUCHOS_TEST_FOR_EXCEPTION((nodes.rank() != 3) || (nodes.dimension(0) != numCells) || (numNodes == 0), std::invalid_argument,
                                 "DirectionProductL2Weight: the BasisCache's physical cell nodes do not match the requested values");
      const int numScales = std::min(int(_scales.size()), nodes.dimension(2));
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        double weight = _alpha;
        for (int d=0; d<numScales; d++)
        {
          if (_scales[d] == 0.0) continue;
          double dMin = nodes(cellOrdinal,0,d), dMax = nodes(cellOrdinal,0,d);
          for (int node=1; node<numNodes; node++)
          {
            dMin = std::min(dMin, nodes(cellOrdinal,node,d));
            dMax = std::max(dMax, nodes(cellOrdinal,node,d));
          }
          const double h = dMax - dMin;
          weight *= 1.0 + _scales[d] / (_alpha * h * h);
        }
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          values(cellOrdinal,pointOrdinal) = weight;
        }
      }
    }
    string displayString() override { return "alpha_K (direction product)"; }
  };

  //! alpha_K = alpha + C [ sum_{d spatial} (v_d,max(K) / h_d)^2 + (1 / h_t)^2 + sum_{d velocity} (a_d,max / h_{u_d})^2 ]: the
  //! sum over directions of the element's transport scale (|beta_d| / h_d)^2, with beta = (v, 1, a).  v_d,max(K) is the
  //! largest |u_d| / gamma over the element's velocity box, gamma = sqrt(1 + |u|^2 / c^2); a_d,max is a fixed bound on
  //! the acceleration in velocity direction d (so the weight does not depend on the fields).  Each term is a product of
  //! one-dimensional factors (the x term: a velocity-cell factor times an x-cell factor), so on a tensor-product mesh the
  //! weighted mass term has tensor rank at most the number of terms.
  class TransportScaleL2Weight : public TFunction<double>
  {
    double _alpha, _scale, _c;
    int _spaceDim, _velocityDim;
    vector<double> _accelerationBounds;
  public:
    bool _velocityTermsOnly = false; // the x- and t-terms moved to outflow-face terms (OutflowFaceWeight)
    vector<FunctionPtr> _accelerationScales; // if nonempty: per-cell scales (constant on each cell) in place of the bounds
    TransportScaleL2Weight(double alpha, double scale, double c, int spaceDim, int velocityDim, const vector<double> &accelerationBounds,
                           bool velocityTermsOnly = false)
    : TFunction<double>(0), _alpha(alpha), _scale(scale), _c(c), _spaceDim(spaceDim), _velocityDim(velocityDim), _accelerationBounds(accelerationBounds),
      _velocityTermsOnly(velocityTermsOnly) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const FieldContainer<double> &nodes = basisCache->getPhysicalCellNodes();
      const int numCells  = values.dimension(0);
      const int numPoints = values.dimension(1);
      const int numNodes  = nodes.dimension(1);
      const int dim       = _spaceDim + 1 + _velocityDim; // coordinates (x_1, ..., x_s, t, u_1, ..., u_v)
      TEUCHOS_TEST_FOR_EXCEPTION((nodes.rank() != 3) || (nodes.dimension(0) != numCells) || (numNodes == 0) || (nodes.dimension(2) != dim),
                                 std::invalid_argument, "TransportScaleL2Weight: the BasisCache's physical cell nodes do not match the requested values");
      vector<double> lo(dim), hi(dim);
      vector<FieldContainer<double>> scaleValues(_accelerationScales.size());
      for (size_t j=0; j<_accelerationScales.size(); j++)
      {
        scaleValues[j].resize(numCells, numPoints);
        _accelerationScales[j]->values(scaleValues[j], basisCache);
      }
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int d=0; d<dim; d++)
        {
          lo[d] = hi[d] = nodes(cellOrdinal,0,d);
          for (int node=1; node<numNodes; node++)
          {
            lo[d] = std::min(lo[d], nodes(cellOrdinal,node,d));
            hi[d] = std::max(hi[d], nodes(cellOrdinal,node,d));
          }
        }
        // per velocity direction: largest |u_j| and smallest |u_j| over the element (0 if the cell straddles u_j = 0)
        vector<double> uMax(_velocityDim), uMin(_velocityDim);
        for (int j=0; j<_velocityDim; j++)
        {
          const double a = lo[_spaceDim+1+j], b = hi[_spaceDim+1+j];
          uMax[j] = std::max(std::abs(a), std::abs(b));
          uMin[j] = (a <= 0.0 && b >= 0.0) ? 0.0 : std::min(std::abs(a), std::abs(b));
        }
        double transport = 0.0;
        for (int d=0; d<_spaceDim; d++)
        {
          // |v_d| = |u_d| / gamma is largest at |u_d| max and the other components at their smallest
          double uSquared = uMax[d] * uMax[d];
          for (int j=0; j<_velocityDim; j++)
          {
            if (j != d) uSquared += uMin[j] * uMin[j];
          }
          const double vMax = uMax[d] / sqrt(1.0 + uSquared / (_c * _c));
          const double h = hi[d] - lo[d];
          if (!_velocityTermsOnly) transport += (vMax / h) * (vMax / h);
        }
        const double h_t = hi[_spaceDim] - lo[_spaceDim];
        if (!_velocityTermsOnly) transport += 1.0 / (h_t * h_t);
        for (int j=0; j<_velocityDim && j<int(_accelerationBounds.size()); j++)
        {
          const double h = hi[_spaceDim+1+j] - lo[_spaceDim+1+j];
          const double scale = _accelerationScales.empty() ? _accelerationBounds[j] : scaleValues[j](cellOrdinal,0);
          transport += (scale / h) * (scale / h);
        }
        const double weight = _alpha + _scale * transport;
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          values(cellOrdinal,pointOrdinal) = weight;
        }
      }
    }
    string displayString() override { return "alpha_K (transport scale)"; }
  };
}

namespace
{
  //! a_u(K) = max(safety-free max over the cell's points of |a_u|, floor), constant on each cell: the per-cell acceleration
  //! scale of velocity direction u.  The cell maxima (all directions) are cached per cell box in a shared cache, so that
  //! every evaluation in a substep (the solve's Gram matrices, GMG's coarse levels, the matrix-free check) agrees.
  class PerCellAccelerationScale : public TFunction<double>
  {
    using Cache = std::map<std::vector<long long>, std::vector<double>>;
    vector<FunctionPtr> _a; int _direction; double _floor; int _spaceDim, _neighborhood;
    std::shared_ptr<Cache> _cache;
    // the cell's box as a key, with a last entry: 0 for the cell's own maxima, the neighborhood mode for the combined ones
    static std::vector<long long> boxKey(const FieldContainer<double> &nodes, int c, std::vector<double> *lo = nullptr, std::vector<double> *hi = nullptr)
    {
      const int numNodes = nodes.dimension(1), dim = nodes.dimension(2);
      std::vector<long long> key;
      if (lo) { lo->resize(dim); hi->resize(dim); }
      for (int k=0; k<dim; k++)
      {
        double l = nodes(c,0,k), h = l;
        for (int n=1; n<numNodes; n++) { l = std::min(l, nodes(c,n,k)); h = std::max(h, nodes(c,n,k)); }
        key.push_back(std::llround(l * 1e10)); key.push_back(std::llround(h * 1e10));
        if (lo) { (*lo)[k] = l; (*hi)[k] = h; }
      }
      key.push_back(0);
      return key;
    }
    // the cell's own maxima of |a_j| over the quadrature points of basisCache's cell c (evaluated on demand)
    const std::vector<double> &ownMaxima(const std::vector<long long> &key, BasisCachePtr basisCache, int c,
                                         std::vector<FieldContainer<double>> &aValues)
    {
      auto it = _cache->find(key);
      if (it != _cache->end()) return it->second;
      const int numCells = basisCache->getPhysicalCellNodes().dimension(0), numPoints = basisCache->getPhysicalCubaturePointsFC().dimension(1);
      if (aValues.empty())
      {
        aValues.resize(_a.size());
        for (size_t j=0; j<_a.size(); j++) { aValues[j].resize(numCells, numPoints); _a[j]->values(aValues[j], basisCache); }
      }
      std::vector<double> maxima(_a.size(), 0.0);
      for (size_t j=0; j<_a.size(); j++)
        for (int p=0; p<numPoints; p++) maxima[j] = std::max(maxima[j], std::abs(aValues[j](c,p)));
      return _cache->emplace(key, maxima).first->second;
    }
  public:
    PerCellAccelerationScale(const vector<FunctionPtr> &a, int direction, double floor, int spaceDim, int neighborhood,
                             std::shared_ptr<Cache> cache)
    : TFunction<double>(0), _a(a), _direction(direction), _floor(floor), _spaceDim(spaceDim), _neighborhood(neighborhood), _cache(cache) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const FieldContainer<double> &nodes = basisCache->getPhysicalCellNodes();
      const int numCells = values.dimension(0), numPoints = values.dimension(1);
      std::vector<FieldContainer<double>> aValues; // evaluated only if some cell is not cached
      MeshPtr mesh = (_neighborhood > 0) ? basisCache->mesh() : Teuchos::null;
      TEUCHOS_TEST_FOR_EXCEPTION((_neighborhood > 0) && (mesh.is_null() || (int(basisCache->cellIDs().size()) != numCells)), std::invalid_argument,
                                 "per-cell acceleration scales with a neighborhood need a BasisCache with its mesh and cell IDs");
      for (int c=0; c<numCells; c++)
      {
        std::vector<double> lo, hi;
        std::vector<long long> key = boxKey(nodes, c, &lo, &hi);
        const std::vector<double> *scales = &ownMaxima(key, basisCache, c, aValues);
        if (_neighborhood > 0)
        {
          std::vector<long long> combinedKey = key;
          combinedKey.back() = _neighborhood;
          auto it = _cache->find(combinedKey);
          if (it == _cache->end())
          {
            // the maximum over the cell and its face neighbors: across velocity faces only (1), or all faces (2)
            std::vector<double> combined = *scales;
            MeshTopologyViewPtr topology = mesh->getTopology();
            const GlobalIndexType cellID = basisCache->cellIDs()[c];
            CellPtr cell = topology->getCell(cellID);
            for (unsigned side=0; side<cell->getSideCount(); side++)
            {
              const GlobalIndexType neighborID = cell->getNeighborInfo(side, topology).first;
              if (neighborID == GlobalIndexType(-1)) continue;
              FieldContainer<double> neighborNodes = mesh->physicalCellNodesForCell(neighborID);
              std::vector<double> nlo, nhi;
              std::vector<long long> neighborKey = boxKey(neighborNodes, 0, &nlo, &nhi);
              int normal = -1; // the direction in which the two boxes touch
              for (int k=0; k<int(lo.size()); k++)
                if ((std::abs(nhi[k] - lo[k]) < 1e-12 * (1 + std::abs(lo[k]))) || (std::abs(nlo[k] - hi[k]) < 1e-12 * (1 + std::abs(hi[k])))) normal = k;
              if ((_neighborhood == 1) && (normal < _spaceDim + 1)) continue;
              BasisCachePtr neighborCache;
              std::vector<FieldContainer<double>> neighborValues;
              const std::vector<double> *neighborMaxima;
              auto nit = _cache->find(neighborKey);
              if (nit != _cache->end()) neighborMaxima = &nit->second;
              else
              {
                neighborCache = BasisCache::basisCacheForCell(mesh, neighborID);
                neighborMaxima = &ownMaxima(neighborKey, neighborCache, 0, neighborValues);
              }
              for (size_t j=0; j<combined.size(); j++) combined[j] = std::max(combined[j], (*neighborMaxima)[j]);
            }
            it = _cache->emplace(combinedKey, combined).first;
          }
          scales = &it->second;
        }
        const double value = std::max((*scales)[_direction], _floor);
        for (int p=0; p<numPoints; p++) values(c,p) = value;
      }
    }
    string displayString() override { return "a_u(K)"; }
  };
}

namespace
{
  //! The largest |v_d| = |u_d| / gamma over the element's velocity box, for spatial direction d: the per-element scale of the
  //! x-transport term in the direction-separable graph norm (as in TransportScaleL2Weight).
  class VelocityComponentMax : public TFunction<double>
  {
    double _c; int _spaceDim, _velocityDim, _d;
  public:
    VelocityComponentMax(double c, int spaceDim, int velocityDim, int d)
    : TFunction<double>(0), _c(c), _spaceDim(spaceDim), _velocityDim(velocityDim), _d(d) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const FieldContainer<double> &nodes = basisCache->getPhysicalCellNodes();
      const int numCells = values.dimension(0), numPoints = values.dimension(1), numNodes = nodes.dimension(1);
      const int dim = _spaceDim + 1 + _velocityDim;
      vector<double> lo(dim), hi(dim);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int k=0; k<dim; k++)
        {
          lo[k] = hi[k] = nodes(cellOrdinal,0,k);
          for (int node=1; node<numNodes; node++) { lo[k] = std::min(lo[k], nodes(cellOrdinal,node,k)); hi[k] = std::max(hi[k], nodes(cellOrdinal,node,k)); }
        }
        double uSquared = 0.0, uMaxD = 0.0;
        for (int j=0; j<_velocityDim; j++)
        {
          const double a = lo[_spaceDim+1+j], b = hi[_spaceDim+1+j];
          const double uMax = std::max(std::abs(a), std::abs(b)), uMin = (a <= 0.0 && b >= 0.0) ? 0.0 : std::min(std::abs(a), std::abs(b));
          if (j == _d) { uMaxD = uMax; uSquared += uMax * uMax; } else uSquared += uMin * uMin;
        }
        const double vMax = uMaxD / sqrt(1.0 + uSquared / (_c * _c));
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) values(cellOrdinal,pointOrdinal) = vMax;
      }
    }
    string displayString() override { return "v_d,max(K)"; }
  };
}

namespace
{
  //! theta s_d(K)^2 / h_d(K) for direction d (0-based: x_1..x_s, then t): the outflow-face weight that replaces the
  //! transport-scale L^2 weight's (s_d / h_d)^2 term with a face term of the same strength (s_x = v_x,max(K), s_t = 1)
  class OutflowFaceWeight : public TFunction<double>
  {
    double _theta, _c; int _spaceDim, _velocityDim, _direction;
  public:
    OutflowFaceWeight(double theta, double c, int spaceDim, int velocityDim, int direction)
    : TFunction<double>(0), _theta(theta), _c(c), _spaceDim(spaceDim), _velocityDim(velocityDim), _direction(direction) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const FieldContainer<double> &nodes = basisCache->getPhysicalCellNodes();
      const int numCells = values.dimension(0), numPoints = values.dimension(1), numNodes = nodes.dimension(1);
      const int dim = _spaceDim + 1 + _velocityDim;
      vector<double> lo(dim), hi(dim);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int k=0; k<dim; k++)
        {
          lo[k] = hi[k] = nodes(cellOrdinal,0,k);
          for (int node=1; node<numNodes; node++) { lo[k] = std::min(lo[k], nodes(cellOrdinal,node,k)); hi[k] = std::max(hi[k], nodes(cellOrdinal,node,k)); }
        }
        double s = 1.0; // the time direction
        if (_direction < _spaceDim)
        {
          double uSquared = 0.0, uMaxD = 0.0;
          for (int j=0; j<_velocityDim; j++)
          {
            const double a = lo[_spaceDim+1+j], b = hi[_spaceDim+1+j];
            const double uMax = std::max(std::abs(a), std::abs(b)), uMin = (a <= 0.0 && b >= 0.0) ? 0.0 : std::min(std::abs(a), std::abs(b));
            if (j == _direction) { uMaxD = uMax; uSquared += uMax * uMax; } else uSquared += uMin * uMin;
          }
          s = uMaxD / sqrt(1.0 + uSquared / (_c * _c));
        }
        const double h = hi[_direction] - lo[_direction];
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++) values(cellOrdinal,pointOrdinal) = _theta * s * s / h;
      }
    }
    string displayString() override { return "theta s_d^2 / h_d"; }
  };

  //! the physical coordinate k (0-based) at the points: the outflow side of x_d is the sign of u_d
  class CoordinateValue : public TFunction<double>
  {
    int _k;
  public:
    CoordinateValue(int k) : TFunction<double>(0), _k(k) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      const FieldContainer<double> &points = basisCache->getPhysicalCubaturePointsFC();
      for (int c=0; c<values.dimension(0); c++)
        for (int p=0; p<values.dimension(1); p++) values(c,p) = points(c,p,_k);
    }
    string displayString() override { return "z_k"; }
  };
}

double RelativisticVlasovSpaceTimeFormulation::s_perCellAccelerationFloorFraction = 0.0;
int RelativisticVlasovSpaceTimeFormulation::s_perCellAccelerationNeighborhood = 0;

void RelativisticVlasovSpaceTimeFormulation::perCellAccelerationScaleRanges(std::vector<double> &minima, std::vector<double> &maxima, int &cellCount) const
{
  minima.clear(); maxima.clear(); cellCount = 0;
  if (!_perCellAccelerationCache) return;
  const int velocityDim = int(_testSpaceL2WeightAccelerationBounds.size());
  for (const auto &entry : *_perCellAccelerationCache)
  {
    if (entry.first.back() != s_perCellAccelerationNeighborhood) continue; // the scales in use (own or combined maxima)
    if (minima.empty()) { minima.assign(entry.second.size(), std::numeric_limits<double>::max()); maxima.assign(entry.second.size(), 0.0); }
    for (size_t j=0; j<entry.second.size(); j++)
    {
      const double floor = (int(j) < velocityDim) ? s_perCellAccelerationFloorFraction * _testSpaceL2WeightAccelerationBounds[j] : 0.0;
      const double value = std::max(entry.second[j], floor);
      minima[j] = std::min(minima[j], value); maxima[j] = std::max(maxima[j], value);
    }
    cellCount++;
  }
}

RelativisticVlasovSpaceTimeFormulation::RelativisticVlasovSpaceTimeFormulation(const double c, const double q, const double m,
                                                                               Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation,
                                                                               MeshTopologyPtr spatialMeshTopology,
                                                                               vector<double> temporalVertices,
                                                                               vector<double> velocityVertices_x,
                                                                               vector<double> velocityVertices_y,
                                                                               vector<double> velocityVertices_z,
                                                                               const int H1Order, const int delta_k,
                                                                               const int numUniformRefinements,
                                                                               const TestSpaceInnerProduct testSpaceIP,
                                                                               const double testSpaceL2Weight,
                                                                               const EBasisFamily &basisFamily,
                                                                               const int maxMomentNumberForDriver,
                                                                               const bool enforcePositivity,
                                                                               const double testSpaceL2WeightMeshScale,
                                                                               const vector<double> &testSpaceL2WeightDirectionScales,
                                                                               const double testSpaceL2WeightTransportScale,
                                                                               const vector<double> &testSpaceL2WeightAccelerationBounds,
                                                                               const bool testSpaceSeparableGraphNorm,
                                                                               const double testSpaceL2ConstantModeRatio,
                                                                               const bool fullTensorTestBasis,
                                                                               const double testSpaceOutflowFaceTheta)
:
_c(c),
_testSpaceInnerProduct(testSpaceIP),
_testSpaceL2Weight(testSpaceL2Weight),
_testSpaceL2WeightMeshScale(testSpaceL2WeightMeshScale),
_testSpaceL2WeightDirectionScales(testSpaceL2WeightDirectionScales),
_testSpaceL2WeightTransportScale(testSpaceL2WeightTransportScale),
_testSpaceL2WeightAccelerationBounds(testSpaceL2WeightAccelerationBounds),
_testSpaceSeparableGraphNorm(testSpaceSeparableGraphNorm),
_testSpaceL2ConstantModeRatio(testSpaceL2ConstantModeRatio),
_fullTensorTestBasis(fullTensorTestBasis),
_testSpaceOutflowFaceTheta(testSpaceOutflowFaceTheta),
VlasovSpaceTimeFormulation(q, m, configurationSpaceFormulation,
                           spatialMeshTopology, temporalVertices,
                           velocityVertices_x,
                           velocityVertices_y,
                           velocityVertices_z,
                           H1Order, delta_k,
                           numUniformRefinements,
                           basisFamily,
                           maxMomentNumberForDriver,
                           true, // true: defer initialization; we call initialize() below.
                           enforcePositivity)
{
  const int numWeightOptions = (testSpaceL2WeightMeshScale > 0.0 ? 1 : 0) + (testSpaceL2WeightDirectionScales.empty() ? 0 : 1)
                             + (testSpaceL2WeightTransportScale > 0.0 ? 1 : 0);
  TEUCHOS_TEST_FOR_EXCEPTION(numWeightOptions > 1, std::invalid_argument,
                             "Specify at most one of testSpaceL2WeightMeshScale, testSpaceL2WeightDirectionScales, and testSpaceL2WeightTransportScale");
  for (double bound : testSpaceL2WeightAccelerationBounds)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(bound < 0.0, std::invalid_argument, "testSpaceL2WeightAccelerationBounds must be nonnegative");
  }
  TEUCHOS_TEST_FOR_EXCEPTION((testSpaceOutflowFaceTheta > 0.0) && (!testSpaceSeparableGraphNorm || !(testSpaceL2WeightTransportScale > 0.0)),
                             std::invalid_argument, "testSpaceOutflowFaceTheta needs the separable graph norm and the transport-scale L^2 weight");
  TEUCHOS_TEST_FOR_EXCEPTION(testSpaceOutflowFaceTheta < 0.0, std::invalid_argument, "testSpaceOutflowFaceTheta must be nonnegative");
  TEUCHOS_TEST_FOR_EXCEPTION(testSpaceSeparableGraphNorm && (testSpaceIP != GraphTestSpaceInnerProduct), std::invalid_argument,
                             "testSpaceSeparableGraphNorm applies to the graph test-space inner product");
  TEUCHOS_TEST_FOR_EXCEPTION(!(testSpaceL2ConstantModeRatio > 0.0) || (testSpaceL2ConstantModeRatio > 1.0), std::invalid_argument,
                             "testSpaceL2ConstantModeRatio must lie in (0, 1]");
  TEUCHOS_TEST_FOR_EXCEPTION((testSpaceL2ConstantModeRatio < 1.0) && (testSpaceIP != GraphTestSpaceInnerProduct), std::invalid_argument,
                             "testSpaceL2ConstantModeRatio applies to the graph test-space inner product");
  for (double scale : testSpaceL2WeightDirectionScales)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(scale < 0.0, std::invalid_argument, "testSpaceL2WeightDirectionScales must be nonnegative");
  }
  TEUCHOS_TEST_FOR_EXCEPTION(!testSpaceL2WeightDirectionScales.empty() && (testSpaceL2Weight <= 0.0), std::invalid_argument,
                             "testSpaceL2WeightDirectionScales requires a positive testSpaceL2Weight");
  this->initialize();
}

IPPtr RelativisticVlasovSpaceTimeFormulation::constructVlasovInnerProduct()
{
  const int spaceDim    = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
  
  vector<FunctionPtr> a = this->a();
  // per-cell acceleration scales (setPerCellAccelerationScales()), in place of the box-wide bounds
  vector<FunctionPtr> accelerationScales;
  if (s_perCellAccelerationFloorFraction > 0.0)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(int(_testSpaceL2WeightAccelerationBounds.size()) < velocityDim, std::invalid_argument,
                               "per-cell acceleration scales need a box-wide bound in every velocity direction (for the floor)");
    if (!_perCellAccelerationCache) _perCellAccelerationCache = std::make_shared<std::map<std::vector<long long>, std::vector<double>>>();
    for (int vd=0; vd<velocityDim; vd++)
      accelerationScales.push_back(Teuchos::rcp(new PerCellAccelerationScale(a, vd, s_perCellAccelerationFloorFraction * _testSpaceL2WeightAccelerationBounds[vd],
                                                                            spaceDim, s_perCellAccelerationNeighborhood, _perCellAccelerationCache)));
  }
  
  // the direction-separable graph norm's Gram matrices are assembled from one-dimensional factors on box cells
  // (also when the L^2 term's constant mode is weakened, which SeparableTensorIP applies after either assembly path)
  const bool weakenedConstantMode = (_testSpaceL2ConstantModeRatio < 1.0);
  Teuchos::RCP<SeparableTensorIP> separableIP = (_testSpaceSeparableGraphNorm || weakenedConstantMode) ? SeparableTensorIP::ip() : Teuchos::null;
  IPPtr ip = (separableIP != Teuchos::null) ? IPPtr(separableIP) : IP::ip();
  
  const int numSpecies = this->numSpecies();
  
  for (int i=0; i<numSpecies; i++)
  {
    VarPtr w = this->w(i);
    
    if (_testSpaceInnerProduct == GraphTestSpaceInnerProduct)
    {
      if (_testSpaceSeparableGraphNorm)
      {
        // Direction-separable graph norm: sum_d s_d(K)^2 |dw/dz_d|^2 with s_d >= |beta_d| on K (v_x,max(K) in space, 1 in
        // time, the fixed acceleration bounds in velocity), instead of |beta . grad w|^2.  No cross terms, so each element
        // Gram matrix is a sum of single-direction Kronecker products (exactly invertible by fast diagonalization), and
        // by Cauchy-Schwarz it dominates the graph norm: |beta . grad w|^2 <= (spaceDim+1+velocityDim) sum_d s_d^2 |dw/dz_d|^2.
        ip->addTerm(w->di(indexOffsetTimeOneBased));
        for (int d=0; d<spaceDim; d++)
        {
          FunctionPtr s_d = Teuchos::rcp(new VelocityComponentMax(_c, spaceDim, velocityDim, d));
          ip->addTerm(s_d * w->di(indexOffsetSpaceOneBased + d));
        }
        for (int vd=0; vd<velocityDim; vd++)
        {
          const double bound = (vd < int(_testSpaceL2WeightAccelerationBounds.size())) ? _testSpaceL2WeightAccelerationBounds[vd] : 0.0;
          TEUCHOS_TEST_FOR_EXCEPTION(bound <= 0.0, std::invalid_argument,
                                     "testSpaceSeparableGraphNorm needs a positive acceleration bound in every velocity direction");
          if (accelerationScales.empty()) ip->addTerm(bound * w->di(indexOffsetVelocityOneBased + vd));
          else ip->addTerm(accelerationScales[vd] * w->di(indexOffsetVelocityOneBased + vd));
        }
      }
      else
      {
        LinearTermPtr f_term = - w->di(indexOffsetTimeOneBased); // -dw/dt
        for (int d=0; d<spaceDim; d++)
        {
          auto  v_d    = v_i(d);
          auto dw_dx_d = w->di(indexOffsetSpaceOneBased + d);
          f_term = f_term - v_d * dw_dx_d; // - v \cdot dw/dx
        }
        for (int vd=0; vd<velocityDim; vd++)
        {
          auto dw_dv_vd = w->di(indexOffsetVelocityOneBased + vd);
          f_term = f_term - a[vd] * dw_dv_vd;
        }
        ip->addTerm(f_term);
      }
      
      LinearTermPtr stabilization_term;
      FunctionPtr l2Weight; // alpha, the weight of the L^2 term
      if (_testSpaceL2WeightTransportScale > 0.0)
      {
        const bool outflowFaces = (_testSpaceOutflowFaceTheta > 0.0);
        FunctionPtr weight = Teuchos::rcp(new TransportScaleL2Weight(_testSpaceL2Weight, _testSpaceL2WeightTransportScale, _c,
                                                                     spaceDim, velocityDim, _testSpaceL2WeightAccelerationBounds,
                                                                     outflowFaces));
        if (!accelerationScales.empty()) dynamic_cast<TransportScaleL2Weight&>(*weight)._accelerationScales = accelerationScales;
        stabilization_term = Teuchos::rcp(new SqrtFunction<double>(weight)) * w;
        l2Weight = weight;
        if (outflowFaces)
        {
          // variant A of the marching-compatible norm: the x- and t-parts of the transport-scale weight become outflow-face
          // terms theta s_d^2 / h_d int_{F+} w^2; t: always the upper face; x_d: the face the sign of u_d points to
          TEUCHOS_TEST_FOR_EXCEPTION(separableIP == Teuchos::null || numSpecies > 1, std::invalid_argument,
                                     "testSpaceOutflowFaceTheta needs the separable graph norm and one species");
          const int timeDirection = spaceDim;
          separableIP->addOutflowFaceTerm(timeDirection, Teuchos::rcp(new OutflowFaceWeight(_testSpaceOutflowFaceTheta, _c, spaceDim, velocityDim, timeDirection)),
                                          Function::constant(1.0));
          for (int d=0; d<spaceDim; d++)
            separableIP->addOutflowFaceTerm(d, Teuchos::rcp(new OutflowFaceWeight(_testSpaceOutflowFaceTheta, _c, spaceDim, velocityDim, d)),
                                            Teuchos::rcp(new CoordinateValue(spaceDim + 1 + d)));
        }
      }
      else if (!_testSpaceL2WeightDirectionScales.empty())
      {
        FunctionPtr weight = Teuchos::rcp(new DirectionProductL2Weight(_testSpaceL2Weight, _testSpaceL2WeightDirectionScales));
        stabilization_term = Teuchos::rcp(new SqrtFunction<double>(weight)) * w;
        l2Weight = weight;
      }
      else if (_testSpaceL2WeightMeshScale > 0.0)
      {
        FunctionPtr weight = Teuchos::rcp(new MeshScaledL2Weight(_testSpaceL2Weight, _testSpaceL2WeightMeshScale));
        stabilization_term = Teuchos::rcp(new SqrtFunction<double>(weight)) * w;
        l2Weight = weight;
      }
      else
      {
        stabilization_term = sqrt(_testSpaceL2Weight) * w;
        l2Weight = Function::constant(_testSpaceL2Weight);
      }
      ip->addTerm(stabilization_term);
      if (weakenedConstantMode)
      {
        TEUCHOS_TEST_FOR_EXCEPTION(numSpecies > 1, std::invalid_argument, "testSpaceL2ConstantModeRatio supports one species");
        separableIP->setConstantModeWeight(w, l2Weight, _testSpaceL2ConstantModeRatio);
      }
    }
    else if (_testSpaceInnerProduct == L2TestSpaceInnerProduct)
    {
      ip->addTerm(w);
    }
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported test space inner product selection");
    }
    
    if (this->enforcePositivity())
    {
      VarPtr phi = this->phi(i);
      ip->addTerm(phi);
    }
  }
  
  return ip;
}

VarFactoryPtr RelativisticVlasovSpaceTimeFormulation::constructVlasovVarFactory()
{
  VarFactoryPtr vlasovVF = VarFactory::varFactory();
  
  BasisFamilyPtr basisFamily = BasisFamily::basisFamily(this->basisFamily());
  
  if (this->enforcePositivity())
  {
    this->initializeProximalGalerkinAlpha();
  }
  
  const int numSpecies = this->numSpecies();
  std::vector<VarPtr> f(numSpecies);
  std::vector<VarPtr> w(numSpecies);
  std::vector<VarPtr> t_n_hat(numSpecies);
  
  for (int i=0; i<numSpecies; i++)
  {
    f[i] = vlasovVF->fieldVar(this->f_name(i));
    w[i] = vlasovVF->testVar(this->w_name(i), HGRAD);
    t_n_hat[i] = vlasovVF->fluxVar(this->t_n_hat_name(i));
    
    if (this->enforcePositivity())
    {
      VarPtr psi_i = vlasovVF->fieldVar(this->psi_name(i), CONSTANT_SCALAR); // element-wise constant latent positivity field on the full phase-space-time cell, including time
      VarPtr phi_i = vlasovVF->testVar(this->phi_name(i), L2);               // latent positivity test function
    }
  }
  
  vlasovVF->setTrialBasisFamily(basisFamily);
  // A full tensor-product test space (with, e.g., serendipity trial spaces): the larger test space can only help stability,
  // and with the separable graph norm its Gram matrices are Kronecker sums, inverted exactly by fast diagonalization
  // (SeparableTensorIP::applyInverseInnerProductMatrix), so that the test space costs no stored or factored matrices.
  vlasovVF->setTestBasisFamily(_fullTensorTestBasis ? BasisFamily::basisFamily(BASIS_FAMILY_HIERARCHICAL) : basisFamily);
  
  return vlasovVF;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::gamma() const
{
  int ux_component_zero_based = spaceDim() + 1; // plus 1 for temporal dimension
  
  const int uDim = VlasovSpaceTimeFormulation::velocityDim();
  
  FunctionPtr uSquared = Function::zero();
  for (int d=0; d<uDim; d++)
  {
    auto ud = MonomialFunction::xi_n(ux_component_zero_based + d, 1);
    uSquared = uSquared + ud * ud;
  }
  
  FunctionPtr gamma = Function::sqrtFunction(1 + uSquared / (_c * _c));
  
  return gamma;
}

void RelativisticVlasovSpaceTimeFormulation::initializeVlasovBilinearForm(BFPtr vlasovBF)
{
  const int spaceDim = this->spaceDim();
  const int velocityDim = this->velocityDim();
  
  const int numSpecies = this->numSpecies();
  std::vector<VarPtr> f(numSpecies);
  std::vector<VarPtr> w(numSpecies);
  std::vector<VarPtr> t_n_hat(numSpecies);
  
  Teuchos::RCP<ConfigurationSpaceFormulation> configurationSpaceFormulation = this->configurationSpaceFormulation();
  
  // the di() methods are 1-based:
  const int indexOffsetSpaceOneBased    = 1;
  const int indexOffsetTimeOneBased     = 1 + spaceDim;
  const int indexOffsetVelocityOneBased = indexOffsetTimeOneBased + 1;
    
  // The below assumes that spaceDim <= velocityDim; enforce that here:
  TEUCHOS_TEST_FOR_EXCEPTION(spaceDim > velocityDim, std::invalid_argument, "velocityDim must be at least as large as spaceDim");
  
  
  std::vector<FunctionPtr> v;
  for (int vd=0; vd<velocityDim; vd++)
  {
    v.push_back(this->v_i(vd));
  }
  
  for (int i=0; i<numSpecies; i++)
  {
    // trace
    VarPtr t_n_hat = this->t_n_hat(i);
    
    // field
    VarPtr f = this->f(i);
    
    // test
    VarPtr w = this->w(i);
    
    if (this->enforcePositivity())
    {
      VarPtr psi = this->psi(i);
      VarPtr phi = this->phi(i);
      FunctionPtr alphaPG = this->proximalGalerkinAlpha();
      auto vlasovSolnPrevWeakPtr = Teuchos::rcp(this->vlasovSolutionPreviousSubstep().get(), false);
      FunctionPtr psi_solnPrev = Function::solution(psi, vlasovSolnPrevWeakPtr);
      FunctionPtr exp_psi = Exp<double>::exp(psi_solnPrev);
      
      vlasovBF->addTerm(psi / alphaPG, w);
      vlasovBF->addTerm(f - exp_psi * psi, phi);
    }
    
    for (int d=0; d<spaceDim; d++)
    {
      vlasovBF->addTerm( -v[d] * f, w->di(indexOffsetSpaceOneBased + d));  // dx_d
    }
    vlasovBF->addTerm( -f, w->di(indexOffsetTimeOneBased)); // dt
    
    const double q = this->q(i);
    const double m = this->m(i);
    auto a = this->a(-1); // acceleration term (unweighted by species q or m)
    for (int vd=0; vd<velocityDim; vd++)
    {
      // grad u of f
      vlasovBF->addTerm(-q/m * a[vd] * f, w->di(indexOffsetVelocityOneBased + vd));
    }
    
    vlasovBF->addTerm(t_n_hat, w);
  }
}

Teuchos::RCP<VlasovSpaceTimeFormulation> RelativisticVlasovSpaceTimeFormulation::nextTimeSlab(double tWidth, int numTemporalElements)
{
  auto temporalVertices = MeshTools::verticesFor1DIntervalMesh(t1(), tWidth, numTemporalElements);
  
  // create a new spatial MeshTopology based on the root of the current _spatialMeshTopology
  auto rootSpatialMeshTopology = MeshFactory::rootMeshTopologyCopy(spatialMeshTopology());
  
  auto nextConfigurationFormulation = configurationSpaceFormulation()->newInstance();
  auto nextSlabForm = Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(_c, q(), m(), nextConfigurationFormulation, rootSpatialMeshTopology,
                                                                              temporalVertices, velocitySpaceVertices_x(), velocitySpaceVertices_y(), velocitySpaceVertices_z(),
                                                                              H1Order(), delta_k(), numUniformRefinements(),
                                                                              _testSpaceInnerProduct, _testSpaceL2Weight,
                                                                              basisFamily(),
                                                                              0, this->enforcePositivity(),
                                                                              _testSpaceL2WeightMeshScale,
                                                                              _testSpaceL2WeightDirectionScales,
                                                                              _testSpaceL2WeightTransportScale,
                                                                              _testSpaceL2WeightAccelerationBounds,
                                                                              _testSpaceSeparableGraphNorm,
                                                                              _testSpaceL2ConstantModeRatio,
                                                                              _fullTensorTestBasis,
                                                                              _testSpaceOutflowFaceTheta));
  
  configureNextSlab(nextSlabForm);
  
  return nextSlabForm;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::v_i(const int zeroBasedComp) const
{
  int ux_component_zero_based = spaceDim() + 1; // plus 1 for temporal dimension
  auto ux = MonomialFunction::xi_n(ux_component_zero_based + zeroBasedComp, 1);
  
  FunctionPtr gammaFxn = gamma();
  return ux / gammaFxn;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::relativisticFirstMoment(const int zeroBasedVelocityComponent, const int speciesOrdinal)
{
  TEUCHOS_TEST_FOR_EXCEPTION((speciesOrdinal < 0) || (speciesOrdinal >= numSpecies()), std::invalid_argument,
                             "species ordinal out of range");
  std::pair<int,int> key{zeroBasedVelocityComponent, speciesOrdinal};
  auto cached = _relativisticFirstMomentFunctions.find(key);
  if (cached != _relativisticFirstMomentFunctions.end())
  {
    return cached->second;
  }

  auto drivingMoments = solutionMoments(speciesOrdinal);
  auto firstMoment = Teuchos::rcp(new RelativisticFirstMomentFunction(drivingMoments.get(),
                                                                      vlasovSolution(), f(speciesOrdinal),
                                                                      configurationSpaceMesh(), _c,
                                                                      zeroBasedVelocityComponent));
  _relativisticFirstMomentFunctions[key] = firstMoment;
  return firstMoment;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::relativisticFirstMoment(const int zeroBasedVelocityComponent,
                                                                            const int speciesOrdinal,
                                                                            SolutionMoments* drivingMoments,
                                                                            SolutionPtr phaseSpaceSolution)
{
  TEUCHOS_TEST_FOR_EXCEPTION((speciesOrdinal < 0) || (speciesOrdinal >= numSpecies()), std::invalid_argument,
                             "species ordinal out of range");
  TEUCHOS_TEST_FOR_EXCEPTION(drivingMoments == nullptr, std::invalid_argument,
                             "drivingMoments may not be null");
  TEUCHOS_TEST_FOR_EXCEPTION(phaseSpaceSolution == Teuchos::null, std::invalid_argument,
                             "phaseSpaceSolution may not be null");

  return Teuchos::rcp(new RelativisticFirstMomentFunction(drivingMoments,
                                                          phaseSpaceSolution, f(speciesOrdinal),
                                                          configurationSpaceMesh(), _c,
                                                          zeroBasedVelocityComponent));
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::currentX() const
{
  auto nonConstThis = const_cast<RelativisticVlasovSpaceTimeFormulation*>(this);
  FunctionPtr Jx = Function::zero();
  for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
  {
    Jx = Jx + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(0, speciesOrdinal);
  }
  return Jx;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::currentY() const
{
  if (velocityDim() < 2) return Function::zero();
  auto nonConstThis = const_cast<RelativisticVlasovSpaceTimeFormulation*>(this);
  FunctionPtr Jy = Function::zero();
  for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
  {
    Jy = Jy + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(1, speciesOrdinal);
  }
  return Jy;
}

FunctionPtr RelativisticVlasovSpaceTimeFormulation::currentZ() const
{
  auto nonConstThis = const_cast<RelativisticVlasovSpaceTimeFormulation*>(this);
  FunctionPtr Jz = Function::zero();
  if (velocityDim() == 2)
  {
    for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
    {
      Jz = Jz + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(1, speciesOrdinal);
    }
    return Jz;
  }
  else if (velocityDim() >= 3)
  {
    for (int speciesOrdinal=0; speciesOrdinal<numSpecies(); speciesOrdinal++)
    {
      Jz = Jz + q(speciesOrdinal) * nonConstThis->relativisticFirstMoment(2, speciesOrdinal);
    }
    return Jz;
  }
  return Function::zero();
}

bool RelativisticVlasovSpaceTimeFormulation::useRelativisticCurrentMoments() const
{
  return true;
}

double RelativisticVlasovSpaceTimeFormulation::currentMomentLightSpeed() const
{
  return _c;
}
