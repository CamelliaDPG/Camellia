//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  Boltzmann1D3VFormulationTests.cpp
//
//  Created by Nate Roberts on 3/4/2020.
//
//

#include "Boltzmann1D3VFormulation.h"
#include "PhaseSpaceVisualization.h"
#include "RHS.h"
#include "Solution.h"

#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

namespace  {
  const double TEST_X0     = 0.0;
  const double TEST_KX     = 0.2; // wavelength
  const double TEST_XWIDTH = 2 * M_PI / TEST_KX;
  const double TEST_V0     = -10.0;
  const double TEST_VWIDTH = 20.0;
  const double TEST_VMEASURE = TEST_VWIDTH * TEST_VWIDTH * TEST_VWIDTH;

  Boltzmann1D3VFormulation getFormulation(const int xElements, const int vElements_x, const int vElements_y, const int vElements_z,
                                          bool implicit, bool l2Trace, const int H1Order, const bool usePeriodicBCs,
                                          ERKTableauPtr tableau = Teuchos::null)
  {
    // these constants correspond to what we will be testing in the two-stream instability.  They should not affect this test, except maybe by introducing a little roundoff error here or there.
    double q      = 1.0;
    double eps0   = 1e-4;
    double m      = 1.0;
    double x0     = TEST_X0;
    double xWidth = TEST_XWIDTH;
    const Kokkos::Array<double,3> v0 {TEST_V0, TEST_V0, TEST_V0};
    double vWidth = TEST_VWIDTH;
    
    const double dt = 0.1;
    const int delta_k = 2;
    
    using FormulationChoice = Camellia::Boltzmann1D3VFormulation::Boltzmann1D3VFormulationChoice;
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
    
    
    const Kokkos::Array<double,3> vDomainWidths {vWidth, vWidth, vWidth};
    const Kokkos::Array<int,3> vMeshWidths {vElements_x, vElements_y, vElements_z};
    
    Boltzmann1D3VFormulation form(formulation, q, eps0, m, dt, x0, xWidth, xElements, v0, vDomainWidths, vMeshWidths, H1Order, delta_k, usePeriodicBCs, tableau);
    return form;
  }
}

TEUCHOS_UNIT_TEST( Boltzmann1D3VFormulation, MapToPhaseSpaceMesh )
{
  const int xElements   = 3;
  const int vElements_x = 3;
  const int vElements_y = 1;
  const int vElements_z = 1;
  
  // the map to phase space mesh shouldn't be materially affected by the formulation choice; the following are arbitrary
  const bool implicit = true;
  const bool l2Trace  = false;
  const int H1Order   = 2;
  const bool usePeriodicBCs = true;
  
  auto form = getFormulation(xElements, vElements_x, vElements_y, vElements_z, implicit, l2Trace, H1Order, usePeriodicBCs);
  
  auto gaussMeshTopo     = form.gaussMesh()->getTopology();
  auto boltzmannMeshTopo = form.boltzmannMesh()->getTopology();
  
  auto phaseSpacePhysicalMeshMap = form.phaseSpacePhysicalMeshMap();
  
  auto myPhaseSpaceCells = form.boltzmannMesh()->cellIDsInPartition();
  
  double tol = 1e-16; // tolerance for x coordinate comparison
  for (auto & phaseSpaceCellID : myPhaseSpaceCells)
  {
    GlobalIndexType spatialCellID = *phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(std::set<GlobalIndexType>{phaseSpaceCellID}).begin();
    
    auto spatialVertices = gaussMeshTopo->getCell(spatialCellID)->vertices();
    double xLeft   = gaussMeshTopo->getVertex(spatialVertices[0])[0];
    double xRight  = gaussMeshTopo->getVertex(spatialVertices[1])[0];
    
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

TEUCHOS_UNIT_TEST( Boltzmann1D3VFormulation, MapToPhaseSpaceMeshWithRefinement )
{
  const int xElements   = 4;
  const int vElements_x = 4;
  const int vElements_y = 1;
  const int vElements_z = 1;
  
  // the map to phase space mesh shouldn't be materially affected by the formulation choice; the following are arbitrary
  const bool implicit = true;
  const bool l2Trace  = false;
  const int H1Order   = 2;
  const bool usePeriodicBCs = true;
  
  auto form = getFormulation(xElements, vElements_x, vElements_y, vElements_z, implicit, l2Trace, H1Order, usePeriodicBCs);
  
  auto gaussMeshTopo     = form.gaussMesh()->getTopology();
  auto boltzmannMeshTopo = form.boltzmannMesh()->getTopology();
  
  auto boltzmannMesh = form.boltzmannMesh();
  
  const bool doRefinement = true;
  set<GlobalIndexType> boltzmannCellsToRefine;
  if (doRefinement)
  {
    boltzmannCellsToRefine = {0,5,8};
    
    auto sampleCell     = boltzmannMesh->getTopology()->getCell(0);
    auto sampleCellTopo = sampleCell->topology();
    auto refPattern = RefinementPattern::regularRefinementPattern(sampleCellTopo);
    
    boltzmannMesh->hRefine(boltzmannCellsToRefine, refPattern);
  }
  
  auto phaseSpacePhysicalMeshMap = form.phaseSpacePhysicalMeshMap();
  auto myPhaseSpaceCells = form.boltzmannMesh()->cellIDsInPartition();

  double tol = 1e-13; // tolerance for x coordinate comparison
  for (auto & phaseSpaceCellID : myPhaseSpaceCells)
  {
    GlobalIndexType spatialCellID = *phaseSpacePhysicalMeshMap->physicalCellIDsForPhaseSpaceCells(std::set<GlobalIndexType>{phaseSpaceCellID}).begin();
    
    auto spatialVertices = gaussMeshTopo->getCell(spatialCellID)->vertices();
    double xLeft   = gaussMeshTopo->getVertex(spatialVertices[0])[0];
    double xRight  = gaussMeshTopo->getVertex(spatialVertices[1])[0];
    
    // the rule is that the phase space mesh be nowhere finer than the spatial mesh
    // this means that the phase space cells should contain the left and right x coordinates for the corresponding spatial cell
    
    // determine min/max x coordinate for phase space cell
    auto phaseVertices = boltzmannMeshTopo->getCell(phaseSpaceCellID)->vertices();
    
    double phaseXLeft  = std::numeric_limits<double>::max();
    double phaseXRight = std::numeric_limits<double>::min();
    for (auto vertexIndex : phaseVertices)
    {
      double x = boltzmannMeshTopo->getVertex(vertexIndex)[0];
      phaseXLeft  = std::min(x,phaseXLeft);
      phaseXRight = std::max(x,phaseXRight);
    }
    
    const bool containsLeft  = (phaseXLeft - tol < xLeft + tol);
    const bool containsRight = (phaseXRight + tol > xRight - tol);
    
    if (!containsLeft)
    {
      out << "FAILURE: phase space cell " << phaseSpaceCellID << " has vertex with x coordinate " << phaseXLeft;
      out << "; this does not contain spatial cell " << spatialCellID << "'s left vertex " << xLeft << std::endl;
      success = false;
    }
    
    if (!containsRight)
    {
      out << "FAILURE: phase space cell " << phaseSpaceCellID << " has vertex with x coordinate " << phaseXRight;
      out << "; this does not contain spatial cell " << spatialCellID << "'s right vertex " << xRight << std::endl;
      success = false;
    }
  }
}

TEUCHOS_UNIT_TEST( Boltzmann1D3VFormulation, PatchTestInitialConditions )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr v = Function::yn(1);
  const double meshWidth = TEST_XWIDTH;
  const double vMeasure = TEST_VMEASURE;
  FunctionPtr f0   = x/vMeasure - (TEST_X0 + TEST_XWIDTH) / (2.0 * vMeasure); // chosen to have zero integral on the domain (n0 should be zero)
  const double n0  = 0.0;
//    FunctionPtr rho0 = q * (x - (TEST_X0 + TEST_XWIDTH) / 2.0 - n0 ); // f0, integrated in v dimension
  
  
  const double q      = 1.0;
  const double eps0   = 1e-4;
  const double m      = 1.0;
  
  const int xElements   = 3;
  const int vElements_x = 3;
  const int vElements_y = 1;
  const int vElements_z = 1;
  const int H1Order     = 4; // cubic L^2 functions; allows exact representation of f0 * E0, required for trace solution
  const bool usePeriodicBCs = false;
  
  // probably worth testing more choices than these (maybe all 4 possibilities):
  const bool implicit = false;
  const bool l2Trace  = true;
  
  auto form = getFormulation(xElements, vElements_x, vElements_y, vElements_z, implicit, l2Trace, H1Order, usePeriodicBCs);
  
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
  
  // f0 above only varies in x.
  FunctionPtr f_soln1D3V_dvx = f_soln->di(2); // di() is 1-based: this is dvx
  FunctionPtr f_soln1D3V_dvy = f_soln->di(3); // dvy
  FunctionPtr f_soln1D3V_dvz = f_soln->di(4); // dvz
  double f_dvx_l2 = f_soln1D3V_dvx->l2norm(form.boltzmannMesh());
  double f_dvy_l2 = f_soln1D3V_dvy->l2norm(form.boltzmannMesh());
  double f_dvz_l2 = f_soln1D3V_dvz->l2norm(form.boltzmannMesh());
  
  TEST_COMPARE(f_dvx_l2, <, tol);
  TEST_COMPARE(f_dvy_l2, <, tol);
  TEST_COMPARE(f_dvz_l2, <, tol);

//  PhaseSpaceVisualization phaseSpaceViz(boltzmannSoln, form.f(), "phaseSpaceOutput1D3V", ".");
//  phaseSpaceViz.exportTimeStep(form.t());
  out << "Testing solve with patch test.\n";
  
  // want to set up RHS, bcs so that f0 is a stationary solution
  // d/dvx of f0 is 0
  // d/dx of f0 is 1/TEST_VWIDTH
  auto w = form.w(); // test function for Boltzmann RHS
  auto vx = Function::yn(1); // 1D1V: y dimension is v_x
  boltzmannSoln->rhs()->addTerm(w * vx / TEST_VMEASURE); // this is the result of evaluating the strong form of the LHS at f0 (df0/dv is 0, as is df0/dt; df0/dx = 1/TEST_VMEASURE)
  
  // BCs:
  BCPtr boltzmannBCs = BC::bc();
  boltzmannBCs->addDirichlet(form.t_n_hat(), SpatialFilter::allSpace(), tn_0);
  form.setBoltzmannBCs(boltzmannBCs);
  
  BCPtr gaussBCs = BC::bc();
  gaussBCs->addDirichlet(form.E_hat(), SpatialFilter::allSpace(), E0);
  form.setGaussBCs(gaussBCs);
  
  form.takeTimeStep();
  
  out << "Comparing tn_0 with tn_soln after taking time step.\n";
  soln_magnitude = tn_0->l2norm(boltzmannSoln->mesh());
  err_l2 = (tn_soln-tn_0)->l2norm(boltzmannSoln->mesh());
  TEST_COMPARE(err_l2, <, tol * soln_magnitude);

  out << "Comparing f0 with f_soln after taking time step.\n";
  soln_magnitude = f0->l2norm(boltzmannSoln->mesh());
  err_l2 = (f_soln-f0)->l2norm(boltzmannSoln->mesh());
  TEST_COMPARE(err_l2, <, tol * soln_magnitude);
  
//  phaseSpaceViz.exportTimeStep(form.t());
}
