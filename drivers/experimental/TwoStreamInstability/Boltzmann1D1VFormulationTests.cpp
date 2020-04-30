//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  Boltzmann1D1VFormulationTests.cpp
//
//  Created by Nate Roberts on 3/4/2020.
//
//

#include "Boltzmann1D1VFormulation.h"
#include "Solution.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

Boltzmann1D1VFormulation getFormulation(const int xElements, const int vElements, bool implicit, bool l2Trace, const int H1Order = 2, ERKTableauPtr tableau = Teuchos::null)
{
  // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
  double q      = 1.0;
  double eps0   = 1e-4;
  double m      = 1.0;
  double x0     = 0.0;
  double k_x    = 0.2; // wavelength
  double xWidth = 2 * M_PI / k_x;
  double v0     = -10.0;
  double vWidth = 20.0;
  
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
  Boltzmann1D1VFormulation form(formulation, q, eps0, m, dt, x0, xWidth, xElements, v0, vWidth, vElements, H1Order, delta_k, tableau);
  return form;
}

TEUCHOS_UNIT_TEST( Boltzmann1D1VFormulation, MapToPhaseSpaceMesh )
{
  const int xElements = 3;
  const int vElements = 3;
  
  // the map to phase space mesh shouldn't be materially affected by the formulation choice; the following are arbitrary
  const bool implicit = true;
  const bool l2Trace  = false;
  
  auto form = getFormulation(xElements, vElements, implicit, l2Trace);
  
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
  // TODO: fix this test
  //       - For the trace component of this test, we need f0 to be periodic.
  //       - For polynomial solutions, that means that f0 must be 0 at x=0, x=meshWidth
  //       - It would be better if f0=0 at v=v0, v=v0+vWidth as well; then we wouldn't have to override the BCs as we do below.
  //       - This should do it:
  //         f0_x_dir = (meshWidth - x) * x / ((meshWidth / 2) * (meshWidth / 2))
  //         f0_y_dir = (vWidth*vWidth - v * v) / (vWidth * vWidth)
  //         f0 = f0_x_dir * f0_y_dir
  //       - This also has 0 average on the domain
  //       - Need to check for the right H1Order, once we compute E0 (which I think is cubic in v, making E0 * f0 quintic)
  FunctionPtr x = Function::xn(1);
  FunctionPtr v = Function::yn(1);
  double meshWidth = 10. * M_PI;
  FunctionPtr f0 = x/20. - meshWidth / 40.0; // chosen to have zero integral on the domain (n0 should be zero)
  FunctionPtr rho0 = x - meshWidth / 2.0; // f0, integrated in v dimension
  
  const double q      = 1.0;
  const double eps0   = 1e-4;
  const double m      = 1.0;
  
  const int xElements = 3;
  const int vElements = 3;
  const int H1Order   = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
  
  // probably worth testing more choices than these (maybe all 4 possibilities):
  const bool implicit = false;
  const bool l2Trace  = true;
  
  auto form = getFormulation(xElements, vElements, implicit, l2Trace, H1Order);
  
  form.setInitialConditions(f0);
  
  auto gaussSoln = form.gaussSolution();
  FunctionPtr E_soln = Function::solution(form.E(), gaussSoln);
  
  FunctionPtr E0 = (q/eps0) * (x * x / 2.0 - (meshWidth / 2.0) * x);
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
//  std::cout << "actualMagnitude = " << actualMagnitude << std::endl;
  
  TEST_ASSERT(tn_0->boundaryValueOnly());
  TEST_ASSERT(tn_soln->boundaryValueOnly());

  // See comment above: this comparison likely won't work until f0 is periodic
//  soln_magnitude = tn_0->l2norm(boltzmannSoln->mesh());
//  err_l2 = (tn_soln-tn_0)->l2norm(boltzmannSoln->mesh());
//  TEST_COMPARE(err_l2, <, tol * soln_magnitude);
}
