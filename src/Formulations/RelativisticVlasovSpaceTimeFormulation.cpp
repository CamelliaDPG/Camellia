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
#include "MeshTools.h"
#include "MonomialFunctions.h"
#include "SqrtFunction.h"
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
    TransportScaleL2Weight(double alpha, double scale, double c, int spaceDim, int velocityDim, const vector<double> &accelerationBounds)
    : TFunction<double>(0), _alpha(alpha), _scale(scale), _c(c), _spaceDim(spaceDim), _velocityDim(velocityDim), _accelerationBounds(accelerationBounds) {}
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
          transport += (vMax / h) * (vMax / h);
        }
        const double h_t = hi[_spaceDim] - lo[_spaceDim];
        transport += 1.0 / (h_t * h_t);
        for (int j=0; j<_velocityDim && j<int(_accelerationBounds.size()); j++)
        {
          const double h = hi[_spaceDim+1+j] - lo[_spaceDim+1+j];
          transport += (_accelerationBounds[j] / h) * (_accelerationBounds[j] / h);
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
                                                                               const bool testSpaceSeparableGraphNorm)
:
_c(c),
_testSpaceInnerProduct(testSpaceIP),
_testSpaceL2Weight(testSpaceL2Weight),
_testSpaceL2WeightMeshScale(testSpaceL2WeightMeshScale),
_testSpaceL2WeightDirectionScales(testSpaceL2WeightDirectionScales),
_testSpaceL2WeightTransportScale(testSpaceL2WeightTransportScale),
_testSpaceL2WeightAccelerationBounds(testSpaceL2WeightAccelerationBounds),
_testSpaceSeparableGraphNorm(testSpaceSeparableGraphNorm),
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
  TEUCHOS_TEST_FOR_EXCEPTION(testSpaceSeparableGraphNorm && (testSpaceIP != GraphTestSpaceInnerProduct), std::invalid_argument,
                             "testSpaceSeparableGraphNorm applies to the graph test-space inner product");
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
  
  IPPtr ip = IP::ip();
  
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
          ip->addTerm(bound * w->di(indexOffsetVelocityOneBased + vd));
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
      if (_testSpaceL2WeightTransportScale > 0.0)
      {
        FunctionPtr weight = Teuchos::rcp(new TransportScaleL2Weight(_testSpaceL2Weight, _testSpaceL2WeightTransportScale, _c,
                                                                     spaceDim, velocityDim, _testSpaceL2WeightAccelerationBounds));
        stabilization_term = Teuchos::rcp(new SqrtFunction<double>(weight)) * w;
      }
      else if (!_testSpaceL2WeightDirectionScales.empty())
      {
        FunctionPtr weight = Teuchos::rcp(new DirectionProductL2Weight(_testSpaceL2Weight, _testSpaceL2WeightDirectionScales));
        stabilization_term = Teuchos::rcp(new SqrtFunction<double>(weight)) * w;
      }
      else if (_testSpaceL2WeightMeshScale > 0.0)
      {
        FunctionPtr weight = Teuchos::rcp(new MeshScaledL2Weight(_testSpaceL2Weight, _testSpaceL2WeightMeshScale));
        stabilization_term = Teuchos::rcp(new SqrtFunction<double>(weight)) * w;
      }
      else
      {
        stabilization_term = sqrt(_testSpaceL2Weight) * w;
      }
      ip->addTerm(stabilization_term);
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
  vlasovVF->setTestBasisFamily(basisFamily);
  
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
                                                                              _testSpaceSeparableGraphNorm));
  
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
