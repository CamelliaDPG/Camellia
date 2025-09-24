//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  HeatFormulationTests.cpp
//  Camellia
//
//  Created by Stefan Henneking on 12/4/19.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "HeatFormulation.h"

#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;

namespace
{
void testHeatConsistencySteady(int spaceDim, HeatFormulation::HeatFormulationChoice choice, Teuchos::FancyOStream &out, bool &success)
{
  vector<double> domainDim(spaceDim,1.0); // square domain (0,1)^d
  vector<int> elementCounts(spaceDim,2); // 2^d mesh

  bool conformingTraces = true;
  HeatFormulation form(spaceDim, conformingTraces, 0.0, 1.0, choice, false);
  BFPtr bf = form.bf();
  
  FunctionPtr u_exact; // defined for manufactured solution
  FunctionPtr u_bc;
  FunctionPtr forcing;
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  
  if (spaceDim == 2)
  {
    u_exact = x*x + y*y;
    forcing = - (u_exact->dx()->dx() + u_exact->dy()->dy());
  }
  else if (spaceDim == 3)
  {
    u_exact = x*x + y*y + z;
    forcing = - (u_exact->dx()->dx() + u_exact->dy()->dy() + u_exact->dz()->dz());
  }
  
  // RHS
  RHSPtr rhs = form.rhs(forcing);
  
  // Test inner product
  IPPtr ip = IP::ip();
  
  // Polynomial order, and enrichment order
  int p = 2, delta_p = 0;
  
  // BC
  u_bc = u_exact;
  BCPtr bc = BC::bc();
  if (choice == HeatFormulation::CONTINUOUS_GALERKIN) {
    p = 2; delta_p = 0;
    ip = Teuchos::null;
    bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
  } else if (choice == HeatFormulation::PRIMAL) {
    p = 2; delta_p = 1;
    ip = bf->naiveNorm(spaceDim);
    bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
  } else if (choice == HeatFormulation::ULTRAWEAK) {
    p = 3; delta_p = 1;
    ip = bf->graphNorm();
    bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_bc);
  }
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, domainDim, elementCounts, p, delta_p);
  
  SolutionPtr soln = Solution::solution(bf, mesh, bc, rhs, ip);
  soln->solve();
  
  bool weightFluxesByParity = false;
  auto var_soln = Function::solution(form.u(), soln, weightFluxesByParity);
  
  double cubatureEnrichment = 4;
  double l2Error = (var_soln - u_exact)->l2norm(mesh, cubatureEnrichment);
  
  double tol = 1e-13;
  TEST_COMPARE(l2Error, <, tol);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Steady_2D_Galerkin )
{
  int spaceDim = 2;
  testHeatConsistencySteady(spaceDim,HeatFormulation::CONTINUOUS_GALERKIN,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Steady_2D_Primal )
{
  int spaceDim = 2;
  testHeatConsistencySteady(spaceDim,HeatFormulation::PRIMAL,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Steady_2D_Ultraweak )
{
  int spaceDim = 2;
  testHeatConsistencySteady(spaceDim,HeatFormulation::ULTRAWEAK,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Steady_3D_Galerkin )
{
  int spaceDim = 3;
  testHeatConsistencySteady(spaceDim,HeatFormulation::CONTINUOUS_GALERKIN,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Steady_3D_Primal )
{
  int spaceDim = 3;
  testHeatConsistencySteady(spaceDim,HeatFormulation::PRIMAL,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Steady_3D_Ultraweak )
{
  int spaceDim = 3;
  testHeatConsistencySteady(spaceDim,HeatFormulation::ULTRAWEAK,out,success);
}

void testHeatConsistencyTransient(int spaceDim, HeatFormulation::HeatFormulationChoice choice, Teuchos::FancyOStream &out, bool &success)
{
  vector<double> domainDim(spaceDim,1.0); // square domain (0,1)^d
  vector<int> elementCounts(spaceDim,2); // 2^d mesh

  bool conformingTraces = true;
  double dt = 0.1;
  double theta = 1.0;
  HeatFormulation form(spaceDim, conformingTraces, dt, theta, choice, false);
  BFPtr bf = form.bf();
  
  FunctionPtr u_exact; // defined for manufactured solution
  FunctionPtr u_bc;
  FunctionPtr forcing;
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  
  if (spaceDim == 2)
  {
    u_exact = x*x + y*y;
    forcing = - (u_exact->dx()->dx() + u_exact->dy()->dy());
  }
  else if (spaceDim == 3)
  {
    u_exact = x*x + y*y + z;
    forcing = - (u_exact->dx()->dx() + u_exact->dy()->dy() + u_exact->dz()->dz());
  }
  
  // RHS
  RHSPtr rhs = form.rhs(forcing);
  rhs->addTerm((u_exact/dt) * form.v());
  
  // Test inner product
  IPPtr ip = IP::ip();
  
  // Polynomial order, and enrichment order
  int p = 2, delta_p = 0;
  
  // BC
  u_bc = u_exact;
  BCPtr bc = BC::bc();
  if (choice == HeatFormulation::CONTINUOUS_GALERKIN) {
    p = 2; delta_p = 0;
    ip = Teuchos::null;
    bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
  } else if (choice == HeatFormulation::PRIMAL) {
    p = 2; delta_p = 1;
    ip = bf->naiveNorm(spaceDim);
    bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
  } else if (choice == HeatFormulation::ULTRAWEAK) {
    p = 3; delta_p = 1;
    ip = bf->graphNorm();
    bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_bc);
  }
  
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, domainDim, elementCounts, p, delta_p);
  
  SolutionPtr soln = Solution::solution(bf, mesh, bc, rhs, ip);
  soln->solve();
  
  bool weightFluxesByParity = false;
  auto var_soln = Function::solution(form.u(), soln, weightFluxesByParity);
  
  double cubatureEnrichment = 4;
  double l2Error = (var_soln - u_exact)->l2norm(mesh, cubatureEnrichment);
  
  double tol = 1e-13;
  TEST_COMPARE(l2Error, <, tol);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Transient_2D_Galerkin )
{
  int spaceDim = 2;
  testHeatConsistencyTransient(spaceDim,HeatFormulation::CONTINUOUS_GALERKIN,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Transient_2D_Primal )
{
  int spaceDim = 2;
  testHeatConsistencyTransient(spaceDim,HeatFormulation::PRIMAL,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Transient_2D_Ultraweak )
{
  int spaceDim = 2;
  testHeatConsistencyTransient(spaceDim,HeatFormulation::ULTRAWEAK,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Transient_3D_Galerkin )
{
  int spaceDim = 3;
  testHeatConsistencyTransient(spaceDim,HeatFormulation::CONTINUOUS_GALERKIN,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Transient_3D_Primal )
{
  int spaceDim = 3;
  testHeatConsistencyTransient(spaceDim,HeatFormulation::PRIMAL,out,success);
}

TEUCHOS_UNIT_TEST( HeatFormulation, Transient_3D_Ultraweak )
{
  int spaceDim = 3;
  testHeatConsistencyTransient(spaceDim,HeatFormulation::ULTRAWEAK,out,success);
}
  
} // namespace
