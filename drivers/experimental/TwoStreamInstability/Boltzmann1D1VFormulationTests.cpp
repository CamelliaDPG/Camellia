//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  Boltzmann1D1VFormulationTests.cpp
//
//  Created by Nate Roberts on 3/4/2020.
//
//

#include "Boltzmann1D1VFormulation.h"

#include "BF.h"
#include "PhaseSpaceVisualization.h"
#include "RHS.h"
#include "Solution.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

namespace {
  const double TEST_X0     = 0.0;
  const double TEST_KX     = 0.2; // wavelength
  const double TEST_XWIDTH = 2 * M_PI / TEST_KX;
  const double TEST_V0     = -10.0;
  const double TEST_VWIDTH = 20.0;

  const double TEST_Q      = 1.0;
  const double TEST_M      = 1.0;
  const double TEST_EPS0   = 1e-4;

  Boltzmann1D1VFormulation getFormulation(const int xElements, const int vElements, bool implicit, bool l2Trace, const int H1Order, const bool usePeriodicBCs, ERKTableauPtr tableau = Teuchos::null)
  {
    // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
    double q      = TEST_Q;
    double eps0   = TEST_EPS0;
    double m      = TEST_M;
    double x0     = TEST_X0;
    double xWidth = TEST_XWIDTH;
    double v0     = TEST_V0;
    double vWidth = TEST_VWIDTH;
    
    const double dt = 0.1;
    const int delta_k = 2;
    
    using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
    FormulationChoice formulation;
    if (implicit)
    {
      if (l2Trace)
      {
        formulation = FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE;
      }
      else
      {
        formulation = FormulationChoice::ULTRAWEAK_IMPLICIT_H1_TRACE;
      }
    }
    else
    {
      if (l2Trace)
      {
        formulation = FormulationChoice::ULTRAWEAK_EXPLICIT_L2_TRACE;
      }
      else
      {
        formulation = FormulationChoice::ULTRAWEAK_EXPLICIT_H1_TRACE;
      }
    }
    Boltzmann1D1VFormulation form(formulation, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, tableau);
    return form;
  }

  class Maxwellian1V : public TFunction<double>
  {
    double _u;
    double _sigma;
    double _n;
  public:
    Maxwellian1V(double u, double sigma, double n)
    :
    _u(u),
    _sigma(sigma),
    _n(n)
    {}
    
    using TFunction<double>::values;
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) override
    {
      auto & phaseSpacePoints = basisCache->getPhysicalCubaturePointsFC();
      const int numCells  = phaseSpacePoints.dimension(0);
      const int numPoints = phaseSpacePoints.dimension(1);
      
      const double weight = _n / (sqrt(2. * M_PI) * _sigma);
      for (int cellOrdinal=0; cellOrdinal<numCells; cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<numPoints; pointOrdinal++)
        {
          const double vx = phaseSpacePoints(cellOrdinal,pointOrdinal,1); // vx is the second dimension for 1D1V
          const double exponent = - 0.5 * (vx - _u) * (vx - _u) / (_sigma * _sigma);
          values(cellOrdinal,pointOrdinal) = weight * std::exp(exponent);
        }
      }
    }
  };

  TEUCHOS_UNIT_TEST( Boltzmann1D1VFormulation, MapToPhaseSpaceMesh )
  {
    const int xElements = 3;
    const int vElements = 3;
    const int H1Order   = 2;
    const bool usePeriodicBCs = true;
    
    // the map to phase space mesh shouldn't be materially affected by the formulation choice; the following are arbitrary
    const bool implicit = true;
    const bool l2Trace  = false;
    
    auto form = getFormulation(xElements, vElements, implicit, l2Trace, H1Order, usePeriodicBCs);
    
    auto gaussMeshTopo     = form.gaussMesh()->getTopology();
    auto boltzmannMeshTopo = form.boltzmannMesh()->getTopology();
    
    auto cellMap = form.spatialCellIDToPhaseSpaceCellIDsMap();
    
    double tol = 1e-16; // tolerance for x coordinate comparison
    for (auto & entry : cellMap)
    {
      GlobalIndexType spatialCellID = entry.first;
      auto & phaseSpaceCells = entry.second;
      
      auto spatialVertices = gaussMeshTopo->getCell(spatialCellID)->vertices();
      double xLeft   = gaussMeshTopo->getVertex(spatialVertices[0])[0];
      double xRight  = gaussMeshTopo->getVertex(spatialVertices[1])[0];
      
      for (auto phaseSpaceCellID : phaseSpaceCells)
      {
        auto phaseVertices = boltzmannMeshTopo->getCell(phaseSpaceCellID)->vertices();
        for (auto vertexIndex : phaseVertices)
        {
          double x = boltzmannMeshTopo->getVertex(vertexIndex)[0];
          bool matchesLeft  = std::abs(x-xLeft)  < tol;
          bool matchesRight = std::abs(x-xRight) < tol;
          if (!matchesLeft && !matchesRight)
          {
            out << "FAILURE: phase space cell " << phaseSpaceCellID << " has vertex with x coordinate " << x;
            out << ", which matches neither " << xLeft << " nor " << xRight << " on spatial cell " << spatialCellID << std::endl;
            success = false;
          }
        }
      }
    }
  }

  TEUCHOS_UNIT_TEST( Boltzmann1D1VFormulation, PatchTestInitialConditions )
  {
    const double q      = TEST_Q;
    const double eps0   = TEST_EPS0;
    const double m      = TEST_M;
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr v = Function::yn(1);
    const double meshWidth = TEST_XWIDTH;
    const double vWidth = TEST_VWIDTH;
    FunctionPtr f0   = x/vWidth - (TEST_X0 + TEST_XWIDTH) / (2.0 * vWidth); // chosen to have zero integral on the domain (n0 should be zero)
    const double n0  = 0.0;
//    FunctionPtr rho0 = q * (x - (TEST_X0 + TEST_XWIDTH) / 2.0 - n0 ); // f0, integrated in v dimension
    
    const int xElements = 3;
    const int vElements = 3;
    const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
    const bool usePeriodicBCs = false;
    
    // probably worth testing more choices than these (maybe all 4 possibilities):
    const bool implicit = false;
    const bool l2Trace  = true;
    
    auto form = getFormulation(xElements, vElements, implicit, l2Trace, H1Order, usePeriodicBCs);
    
    form.setInitialConditions(f0);
    
    auto gaussSoln = form.gaussSolution();
    FunctionPtr E_soln = Function::solution(form.E(), gaussSoln);
    
    FunctionPtr E0 = (q/eps0) * (x * x / 2.0 - ((TEST_X0 + TEST_XWIDTH) / 2.0) * x - n0 * x);
    double E0_avg = E0->integrate(gaussSoln->mesh()) / meshWidth;
    E0 = E0 - E0_avg;
    
    double soln_magnitude = E_soln->l2norm(gaussSoln->mesh());
    double err_l2 = (E_soln-E0)->l2norm(gaussSoln->mesh());
    
    double tol = 1e-13;
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
    
    auto boltzmannSoln = form.boltzmannSolution();
    FunctionPtr n = Function::normal();
    FunctionPtr sgn = Function::sideParity();
    FunctionPtr tn_0 = f0 * (v * n->x() + (q/m) * E0 * n->y()) * sgn;
    
    FunctionPtr tn_soln = Function::solution(form.t_n_hat(), boltzmannSoln, false);
    
    double actualMagnitude = tn_soln->l2norm(boltzmannSoln->mesh());
    out << "actualMagnitude = " << actualMagnitude << std::endl;
    
    TEST_ASSERT(tn_0->boundaryValueOnly());
    TEST_ASSERT(tn_soln->boundaryValueOnly());

    soln_magnitude = tn_0->l2norm(boltzmannSoln->mesh());
    err_l2 = (tn_soln-tn_0)->l2norm(boltzmannSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
    
    FunctionPtr f_soln = Function::solution(form.f(), boltzmannSoln);
    soln_magnitude = f0->l2norm(boltzmannSoln->mesh());
    err_l2 = (f_soln-f0)->l2norm(boltzmannSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
    
//    PhaseSpaceVisualization phaseSpaceViz(boltzmannSoln, form.f(), "phaseSpaceOutput1D1V", ".");
//    phaseSpaceViz.exportTimeStep(form.t());
    
    out << "Testing solve with patch test.\n";
    
    // want to set up RHS, bcs so that f0 is a stationary solution
    // d/dvx of f0 is 0
    // d/dx of f0 is 1/TEST_VWIDTH
    auto w = form.w(); // test function for Boltzmann RHS
    auto vx = Function::yn(1); // 1D1V: y dimension is v_x
    boltzmannSoln->rhs()->addTerm(w * vx / TEST_VWIDTH); // this is the result of evaluating the LHS at f0
    
    // BCs:
    BCPtr boltzmannBCs = BC::bc();
    boltzmannBCs->addDirichlet(form.t_n_hat(), SpatialFilter::allSpace(), tn_0);
    form.setBoltzmannBCs(boltzmannBCs);
    
    BCPtr gaussBCs = BC::bc();
    gaussBCs->addDirichlet(form.E_hat(), SpatialFilter::allSpace(), E0);
    form.setGaussBCs(gaussBCs);
    
    form.takeTimeStep();
    
//    phaseSpaceViz.exportTimeStep(form.t());
    
    out << "Comparing tn_0 with tn_soln after taking time step.\n";
    soln_magnitude = tn_0->l2norm(boltzmannSoln->mesh());
    err_l2 = (tn_soln-tn_0)->l2norm(boltzmannSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);

    out << "Comparing f0 with f_soln after taking time step.\n";
    soln_magnitude = f0->l2norm(boltzmannSoln->mesh());
    err_l2 = (f_soln-f0)->l2norm(boltzmannSoln->mesh());
    TEST_COMPARE(err_l2, <, tol * soln_magnitude);
  }

  TEUCHOS_UNIT_TEST( Boltzmann1D1VFormulation, TimeStepStabilityUltraweakImplicit )
  {
    // confirm, for an example problem, that || u^{n+1} || ≤ || u^n ||
    // we set everything up as in two-stream instability, and use a coarse (2 x 2) high-order (H1Order = 4) mesh.
    // We run with both our usual test space norm, and the automatic graph norm for the bilinear form.
    // We use dt's that go from 1e0 to 1e-5.
    // run test suite with --details=ALL argument to see numerical details.
    
    // set things up as in two-stream
    double k_x    = 0.2; // wavelength
    double u0     = 3.0; // initial drift velocity
    double gamma  = 1e-3; // initial perturbation
    double sigma  = 1.0; // temperature
    double n0     = 1.0; // density
    
    double q      =  1.0;
    double eps0   =  1.0;
    double m      =  1.0;
    double x0     =  0.0;
    double xWidth =  2 * M_PI / k_x;
    double v0     = -10.0;
    double vWidth =  20.0;
    
    FunctionPtr cos_kx       = Teuchos::rcp(new Cos_ax(k_x));
    double n = 0.5 * n0;
    FunctionPtr fM_u0        = Teuchos::rcp(new Maxwellian1V( u0, sigma, n) );
    FunctionPtr fM_minus_u0  = Teuchos::rcp(new Maxwellian1V(-u0, sigma, n) );
    FunctionPtr f0           = (fM_u0 + fM_minus_u0) * (1. + gamma * cos_kx);
    
    const int xElements = 2;
    const int vElements = 2;
    const int H1Order   = 4; // cubic L^2 functions
    
    const int delta_k = 2;
    const double dt = 1.0;
    using FormulationChoice = Camellia::Boltzmann1D1VFormulation::Boltzmann1D1VFormulationChoice;
    FormulationChoice formulation = FormulationChoice::ULTRAWEAK_IMPLICIT_L2_TRACE;
    const bool usePeriodicBCs = true;
    Boltzmann1D1VFormulation form(formulation, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, usePeriodicBCs, Teuchos::null, 0);
    
    auto boltzmannSoln = form.boltzmannSolution();
    IPPtr defaultIP = boltzmannSoln->ip(); // our default inner product for Boltzmann; at present, this is one inspired by Führer et al.
    IPPtr graphNorm = boltzmannSoln->bf()->graphNorm(); // the usual default choice for generic DPG discretization (has unit weight on the extra L^2 terms)
    
    std::vector<IPPtr> ipsToTest {defaultIP, graphNorm};
    
    std::vector<double> dtsToTest {1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5};
    
    int ipOrdinal = 0;
    for (const auto & ipToTest : ipsToTest)
    {
      out << "************************************************************\n";
      out << "testing IP: " << ipToTest->displayString() << std::endl;
      boltzmannSoln->setIP(ipToTest);
      
      for (const auto & dt : dtsToTest)
      {
        out << "testing dt: " << dt << std::endl;
        form.setTimeStep(dt);
        
        form.setInitialConditions(f0);
        
        // we should measure the projected f0 rather than the exact.  (For the patch test initial conditions we're using now, this should not
        // matter, but we may want to harden this test by using multiple f0's, including some that are non-polynomial.)
        
        FunctionPtr f_soln = Function::solution(form.f(), boltzmannSoln);
        const double f0_l2 = f_soln->l2norm(form.boltzmannMesh());
        out << "f0_l2: " << f0_l2 << std::endl;

        form.takeTimeStep();
        
        const double f1_l2 = f_soln->l2norm(form.boltzmannMesh());
        
        out << "f1_l2: " << f1_l2 << std::endl;
        out << "f1_l2 / f0_l2: " << f1_l2 / f0_l2 << std::endl;
        out << "1 - f1_l2 / f0_l2 = " << 1 - f1_l2 / f0_l2 << std::endl;
        
        TEST_COMPARE(f1_l2, <=, f0_l2);
      }
    }
  }
} // end anonymous namespace
