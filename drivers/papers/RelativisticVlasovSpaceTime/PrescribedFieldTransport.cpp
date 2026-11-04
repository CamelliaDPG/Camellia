//
// PrescribedFieldTransport: accuracy of the relativistic 1D1T2V Vlasov DPG discretization for linear transport in
// prescribed, uniform electric and magnetic fields, against the exact solution by characteristics.
//
// This isolates the Vlasov discretization (test norm, enrichment, basis, mesh) from the Vlasov-Maxwell coupling: the
// fields are fixed (ConfigurationSpaceFormulationTrivial), so each slab is a single linear solve (direct solver).  Same
// formulation, coordinates (x, t, u_x, u_y), and scaling as PlanarMITLRelativisticSpaceTime: c = 1, q/m = -1,
// v = u / gamma, gamma = sqrt(1 + u^2), and
//
//   f_t + v_x f_x + (q/m) (E + v x B) . grad_u f = 0,
//
// with E = (E_x, 0, 0) and B = (0, B_y, 0), so that, with the 2V velocity components (u_x, u_z) stored as (u_x, u_y),
//   a_x = E_x + (v x B)_x = E_x - v_z B_y,   a_z = (v x B)_z = v_x B_y.
//
// Initial condition: a Gaussian pulse in (x, u_x, u_y), placed so that it stays away from the domain boundaries over
// the simulated time (so that inflow data is zero).  The exact solution at (x, t, u) is the initial condition at the
// foot of the characteristic, found by integrating backward to t = 0 with RK4 (step much smaller than the slab).
//
// Reported per slab: relative L^2 error of f, max |f - f_exact|, min f, and the relative error in the integral of f.
//

#include "Camellia.h"
#include "ConfigurationSpaceFormulationTrivial.h"
#include "RelativisticVlasovSpaceTimeFormulation.h"
#include "VlasovSpaceTimeFormulation.h"

#include <Kokkos_Core.hpp>
#include <Teuchos_GlobalMPISession.hpp>

#include <cmath>
#include <iomanip>
#include <iostream>

using namespace Camellia;
using namespace std;

namespace
{
  struct Fields
  {
    double Ex = 0, By = 0;
    double qOverM = -1.0;
  };

  // characteristic right-hand side in (x, ux, uz): dx/dt = v_x, du/dt = (q/m)(E + v x B)
  inline void characteristicRHS(const Fields &F, const double x, const double ux, const double uz,
                                double &dx, double &dux, double &duz)
  {
    (void) x;
    const double gamma = std::sqrt(1.0 + ux * ux + uz * uz);
    const double vx = ux / gamma, vz = uz / gamma;
    dx  = vx;
    dux = F.qOverM * (F.Ex - vz * F.By);
    duz = F.qOverM * (vx * F.By);
  }

  //! the foot at t = 0 of the characteristic through (x, t, ux, uz): RK4, backward in time
  inline void characteristicFoot(const Fields &F, double t, double &x, double &ux, double &uz, const double dtMax)
  {
    const int numSteps = std::max(1, int(std::ceil(t / dtMax)));
    const double h = -t / numSteps;
    for (int step=0; step<numSteps; step++)
    {
      double k1x, k1u, k1z, k2x, k2u, k2z, k3x, k3u, k3z, k4x, k4u, k4z;
      characteristicRHS(F, x, ux, uz, k1x, k1u, k1z);
      characteristicRHS(F, x + 0.5*h*k1x, ux + 0.5*h*k1u, uz + 0.5*h*k1z, k2x, k2u, k2z);
      characteristicRHS(F, x + 0.5*h*k2x, ux + 0.5*h*k2u, uz + 0.5*h*k2z, k3x, k3u, k3z);
      characteristicRHS(F, x + h*k3x, ux + h*k3u, uz + h*k3z, k4x, k4u, k4z);
      x  += h / 6.0 * (k1x + 2*k2x + 2*k3x + k4x);
      ux += h / 6.0 * (k1u + 2*k2u + 2*k3u + k4u);
      uz += h / 6.0 * (k1z + 2*k2z + 2*k3z + k4z);
    }
  }

  //! Gaussian pulse in (x, u_x, u_z)
  struct Pulse
  {
    double x0 = 0.3, ux0 = 0.5, uz0 = 0.0;
    double sx = 0.06, su = 0.3;
    double operator()(const double x, const double ux, const double uz) const
    {
      const double dx = (x - x0) / sx, du = (ux - ux0) / su, dz = (uz - uz0) / su;
      return std::exp(-0.5 * (dx*dx + du*du + dz*dz));
    }
  };

  //! exact solution f(x, t, u_x, u_z) = f0(foot of the characteristic); coordinates (x, t, ux, uz)
  class CharacteristicSolution : public TFunction<double>
  {
    Fields _F;
    Pulse _f0;
    double _dtMax;
  public:
    CharacteristicSolution(const Fields &F, const Pulse &f0, double dtMax) : _F(F), _f0(f0), _dtMax(dtMax) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      auto &points = basisCache->getPhysicalCubaturePointsFC();
      const int numCells = points.dimension(0), numPoints = points.dimension(1);
      for (int c=0; c<numCells; c++)
      {
        for (int p=0; p<numPoints; p++)
        {
          double x = points(c,p,0), ux = points(c,p,2), uz = points(c,p,3);
          const double t = points(c,p,1);
          characteristicFoot(_F, t, x, ux, uz, _dtMax);
          values(c,p) = _f0(x, ux, uz);
        }
      }
    }
  };

  //! initial condition (t = 0): the pulse itself
  class PulseInitialCondition : public TFunction<double>
  {
    Pulse _f0;
  public:
    explicit PulseInitialCondition(const Pulse &f0) : _f0(f0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      auto &points = basisCache->getPhysicalCubaturePointsFC();
      const int numCells = points.dimension(0), numPoints = points.dimension(1);
      for (int c=0; c<numCells; c++)
        for (int p=0; p<numPoints; p++)
          values(c,p) = _f0(points(c,p,0), points(c,p,2), points(c,p,3));
    }
  };

  std::vector<double> uniformVertices(double a, double b, int n)
  {
    std::vector<double> v(n+1);
    for (int i=0; i<=n; i++) v[i] = a + (b - a) * double(i) / n;
    return v;
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  Kokkos::initialize(argc, argv);
  const int rank = Teuchos::GlobalMPISession::getRank();
  {  // (Camellia objects must be destroyed before Kokkos is finalized)

  Teuchos::CommandLineProcessor cmdp(false, true);
  int H1Order = 3, deltaP = 2, xElements = 8, tElements = 1, uxElements = 8, uzElements = 8, numSlabs = 3;
  double tWidth = 0.1, Ex = 2.0, By = 0.0, L2WeightGraph = 1e6;
  double uMax = 3.0, sigmaX = 0.1, sigmaU = 0.5;
  bool useL2TestNorm = false, useSerendipity = true, printPoints = false;
  cmdp.setOption("p", &H1Order, "H1 order (field degree p-1, trace degree p)");
  cmdp.setOption("dp", &deltaP, "test-space enrichment");
  cmdp.setOption("xElements", &xElements);
  cmdp.setOption("tElements", &tElements);
  cmdp.setOption("uxElements", &uxElements);
  cmdp.setOption("uzElements", &uzElements);
  cmdp.setOption("numTimeSlabs", &numSlabs);
  cmdp.setOption("tWidth", &tWidth);
  cmdp.setOption("uMax", &uMax, "velocity domain [-uMax, uMax]^2");
  cmdp.setOption("sigmaX", &sigmaX, "pulse width in x");
  cmdp.setOption("sigmaU", &sigmaU, "pulse width in u_x and u_z");
  cmdp.setOption("Ex", &Ex, "uniform electric field (scaled)");
  cmdp.setOption("By", &By, "uniform magnetic field (scaled)");
  cmdp.setOption("L2WeightGraph", &L2WeightGraph, "alpha, the weight of the L^2 term in the graph norm");
  cmdp.setOption("useL2TestNorm", "useGraphTestNorm", &useL2TestNorm);
  cmdp.setOption("useSerendipityBasis", "useFullTensorBasis", &useSerendipity);
  cmdp.setOption("printPoints", "noPrintPoints", &printPoints, "print computed and exact f along a line (diagnostic)");
  if (cmdp.parse(argc, argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) return -1;

  Fields F;
  F.Ex = Ex; F.By = By;
  Pulse f0;
  f0.sx = sigmaX; f0.su = sigmaU;
  const double dtExact = 1e-3;

  // fields: E = (Ex, 0, 0), B = (0, By, 0); the trivial configuration formulation provides them as functions
  auto configForm = Teuchos::rcp(new ConfigurationSpaceFormulationTrivial(TimeDiscretizationType::SPACE_TIME));
  configForm->setE({Function::constant(Ex), Function::zero(), Function::zero()});
  configForm->setB({Function::zero(), Function::constant(By), Function::zero()});

  MeshTopologyPtr spatialMeshTopo = MeshFactory::intervalMeshTopology(0.0, 1.0, xElements, false);
  std::vector<double> temporalVertices = uniformVertices(0.0, tWidth, tElements);
  std::vector<double> ux = uniformVertices(-uMax, uMax, uxElements);
  std::vector<double> uz = uniformVertices(-uMax, uMax, uzElements);
  std::vector<double> none;

  using TestIP = RelativisticVlasovSpaceTimeFormulation::TestSpaceInnerProduct;
  const TestIP testIP = useL2TestNorm ? TestIP::L2TestSpaceInnerProduct : TestIP::GraphTestSpaceInnerProduct;
  const double c = 1.0, q = -1.0, m = 1.0;
  Teuchos::RCP<VlasovSpaceTimeFormulation> form =
    Teuchos::rcp(new RelativisticVlasovSpaceTimeFormulation(c, q, m, configForm, spatialMeshTopo, temporalVertices, ux, uz, none,
                                                            H1Order, deltaP, 0, testIP, L2WeightGraph,
                                                            useSerendipity ? BASIS_FAMILY_SERENDIPITY : BASIS_FAMILY_HIERARCHICAL));
  FunctionPtr fInitial = Teuchos::rcp(new PulseInitialCondition(f0));
  form->setInitialAndInflowConditions(fInitial, Function::zero());

  // the first slab's solve updates velocity moments of the (not yet computed) Vlasov solution before solving, so give
  // the solutions coefficients (zero) on every cell
  {
    std::map<int, FunctionPtr> zero;
    zero[form->f(0)->ID()] = Function::zero();
    form->vlasovSolution()->projectOntoMesh(zero);
    form->vlasovSolutionPreviousSubstep()->projectOntoMesh(zero);
  }

  FunctionPtr fExact = Teuchos::rcp(new CharacteristicSolution(F, f0, dtExact));
  const int quadratureEnrichment = 4;

  if (rank == 0)
  {
    std::cout << "PrescribedFieldTransport: p = " << H1Order << ", dp = " << deltaP << ", "
              << (useL2TestNorm ? std::string("L^2 test norm") : "graph test norm, alpha = " + std::to_string(L2WeightGraph))
              << ", " << (useSerendipity ? "Serendipity" : "full tensor") << ", mesh " << xElements << " x " << tElements << " x "
              << uxElements << " x " << uzElements << ", Ex = " << Ex << ", By = " << By << "\n";
    std::cout << "slab   t_end   rel_L2_err   max_err      min_f        rel_mass_err  vlasov_dofs\n";
  }
  for (int slab=0; slab<numSlabs; slab++)
  {
    // The fields are prescribed, so the slab problem is linear: solve the Vlasov system directly.  (The formulation's
    // solve() would also run the configuration-space solve and moment updates of the fixed-point iteration, which have
    // nothing to do here; the trivial configuration formulation has no unknowns.)
    form->vlasovSolution()->solve();
    auto fSoln = Function::solution(form->f(0), form->vlasovSolution());
    auto mesh = form->vlasovMesh();
    if (printPoints && (slab == 0) && (mesh->Comm()->getSize() == 1))
    {
      // diagnostic: computed and exact f at the cubature points of each cell nearest the line u = (ux0, uz0), t ~ t1,
      // printed for the cell containing the largest exact value
      double bestExact = -1; GlobalIndexType bestCell = -1;
      for (GlobalIndexType cellID : mesh->cellIDsInPartition())
      {
        BasisCachePtr cache = BasisCache::basisCacheForCell(mesh, cellID);
        const int numPoints = cache->getPhysicalCubaturePointsFC().dimension(1);
        FieldContainer<double> ex(1, numPoints);
        fExact->values(ex, cache);
        for (int p=0; p<numPoints; p++) if (ex(0,p) > bestExact) { bestExact = ex(0,p); bestCell = cellID; }
      }
      BasisCachePtr cache = BasisCache::basisCacheForCell(mesh, bestCell);
      auto &pts = cache->getPhysicalCubaturePointsFC();
      const int numPoints = pts.dimension(1);
      FieldContainer<double> ex(1, numPoints), comp(1, numPoints);
      fExact->values(ex, cache);
      fSoln->values(comp, cache);
      std::cout << "cell " << bestCell << " (max exact " << bestExact << "): x t ux uz  computed  exact\n";
      for (int p=0; p<numPoints; p += std::max(1, numPoints / 24))
      {
        std::cout << "  " << pts(0,p,0) << " " << pts(0,p,1) << " " << pts(0,p,2) << " " << pts(0,p,3)
                  << "   " << comp(0,p) << "  " << ex(0,p) << "\n";
      }
    }
    const double errL2   = (fSoln - fExact)->l2norm(mesh, quadratureEnrichment);
    const double normL2  = fExact->l2norm(mesh, quadratureEnrichment);
    const double maxErr  = std::max((fSoln - fExact)->maximumValue(mesh, quadratureEnrichment),
                                    -(fSoln - fExact)->minimumValue(mesh, quadratureEnrichment));
    const double fMin    = fSoln->minimumValue(mesh, quadratureEnrichment);
    const double mass    = fSoln->integrate(mesh, quadratureEnrichment);
    const double massEx  = fExact->integrate(mesh, quadratureEnrichment);
    const double tEnd    = form->t1();
    if (rank == 0)
    {
      std::cout << std::setw(4) << slab << std::setw(8) << std::setprecision(3) << tEnd
                << std::scientific << std::setprecision(4)
                << std::setw(13) << errL2 / normL2 << std::setw(13) << maxErr << std::setw(13) << fMin
                << std::setw(14) << (mass - massEx) / massEx << std::defaultfloat
                << std::setw(12) << mesh->numGlobalDofs() << "\n" << std::flush;
    }
    if (slab + 1 < numSlabs) form = form->nextTimeSlab(tWidth, tElements);
  }
  }
  Camellia::finalize(false); // frees Camellia's scratch storage and finalizes Kokkos
  return 0;
}
