//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  HeatSteadyDriver.cpp
//  Camellia
//
//  Created by Stefan Henneking on 11/14/19.
//
//

#include "Camellia.h"

#include "Teuchos_CommandLineProcessor.hpp"

using namespace Camellia;
using namespace std;

const double PI = 3.141592653589793238462;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  
  int rank = MPIWrapper::CommWorld()->MyPID();
  
  int spaceDim = 3;
  bool conformingTraces = true;
  // formulation choice: CONTINUOUS_GALERKIN,PRIMAL,ULTRAWEAK
  PoissonFormulation::PoissonFormulationChoice choice = PoissonFormulation::CONTINUOUS_GALERKIN;
  PoissonFormulation form(spaceDim, conformingTraces, choice);
  BFPtr bf = form.bf();
  if (rank==0) cout << "Number of trial vars: " << bf->varFactory()->trialVars().size() << endl;
  
  FunctionPtr u_exact; // defined for manufactured solution
  FunctionPtr u_bc;
  FunctionPtr forcing;
  
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  
  // Exact solution
  // 1) sinusoidal solution
  u_exact = TrigFunctions<double>::sin(PI * x) * TrigFunctions<double>::cos(PI * y) * z;
  // 2) polynomial solution
  //u_exact = x*x + y*y + z;
  
  // RHS
  forcing = u_exact->dx()->dx() + u_exact->dy()->dy() + u_exact->dz()->dz(); // Poisson impl. is: \Delta u = f, instead of: -\Delta u = f
  RHSPtr rhs = form.rhs(forcing);
  
  // BC
  u_bc = u_exact;
  BCPtr bc = BC::bc();
  
  int delta_p = 0; // enrichment order
  TIPPtr<double> ip; // test norm
  switch (choice) {
    case PoissonFormulation::CONTINUOUS_GALERKIN:
      bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
      delta_p = 0;
      ip = Teuchos::null;
      break;
    case PoissonFormulation::PRIMAL:
      bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
      delta_p = 1;
      ip = bf->naiveNorm(3); // Mathematician's norm
      break;
    case PoissonFormulation::ULTRAWEAK:
      bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_bc);
      delta_p = 1;
      ip = bf->graphNorm(); // Scaled Adjoint graph norm
      break;
  }
  
  vector<int> elementCounts = {1,1,1}; // x,y,z directions
  vector<double> domainDim = {1.0,1.0,1.0};
  int H1Order = 2; // uniform order of approximation
  MeshPtr mesh = MeshFactory::rectilinearMesh(bf, domainDim, elementCounts, H1Order, delta_p);
  
  // -----------------
  // convergence study
  // -----------------
  
  // refine mesh uniformly; e.g., 3 times -- result will be 8x8x8 mesh
  int numRefs = 3;
  
  int mesh_dofs[numRefs];
  double mesh_err[numRefs];
  float mesh_rate[numRefs];
  
  for (int i=0; i<numRefs; i++)
  {
    // iterate
    if (rank == 0) cout << "iterate i = " << i+1 << endl;
    
    // refine
    mesh->hRefine(mesh->getActiveCellIDsGlobal());
    
    // solve
    SolutionPtr soln;
    soln = Solution::solution(bf,mesh,bc,rhs,ip);
    soln->solve();
    // create exact solution map for trial variables
    FunctionPtr sigma_exact = u_exact->grad();
    FunctionPtr n = Function::normal();
    FunctionPtr sgn = Function::sideParity();
    map<int, FunctionPtr> exactSolutionMap;
    exactSolutionMap[form.u()->ID()] = u_exact;
    if (choice == PoissonFormulation::ULTRAWEAK) {
      exactSolutionMap[form.u_hat()->ID()] = u_exact;
      exactSolutionMap[form.sigma()->ID()] = sigma_exact;
      exactSolutionMap[form.sigma_n_hat()->ID()] = sigma_exact * n * sgn;
    } else if (choice == PoissonFormulation::PRIMAL) {
      exactSolutionMap[form.sigma_n_hat()->ID()] = sigma_exact * n * sgn;
    }
      
    // compute the L2 best approximation
    if (choice == PoissonFormulation::ULTRAWEAK) {
      SolutionPtr bae = Solution::solution(bf,mesh,bc,rhs,ip);
      bae->projectOntoMesh(exactSolutionMap);
    }
    int cubatureEnrichment = 4;
    
    map<int,VarPtr> trialVars = bf->varFactory()->trialVars();
    
    //double baeL2Squared  = 0.0;
    double l2NormSquared = 0.0;
    double h1NormSquared = 0.0;
    for (auto entry : trialVars)
    {
      auto var = entry.second;
      // skip fluxes (they need minimum energy extension norm)
      if (var->varType() != FIELD) {
        continue;
      }
      auto var_exact = exactSolutionMap[var->ID()];
      
      bool weightFluxesByParity = false; // here, interested in comparing to exact solutions; should not weight
      auto var_soln = Function::solution(var, soln, weightFluxesByParity);
      //auto var_best = Function::solution(var, bae , weightFluxesByParity);
      
      //double l2bestError   = (var_best - var_exact)->l2norm(mesh, cubatureEnrichment); // BAE in L2
      double l2actualError = (var_soln - var_exact)->l2norm(mesh, cubatureEnrichment); // Error in L2
      double h1actualError = 0.0;
      if (choice != PoissonFormulation::ULTRAWEAK) {
        h1actualError = (var_soln - var_exact)->grad()->l2norm(mesh, cubatureEnrichment); // Error in H1 seminorm
      }
      
      //cout << var->name() << " best error: " << bestError << endl;
      //cout << var->name() << " actual error: " << actualError << endl;
      
      //baeL2Squared  += l2bestError * l2bestError;
      l2NormSquared += l2actualError * l2actualError;
      h1NormSquared += h1actualError * h1actualError;
    }
    //double energyError = soln->energyErrorTotal();
    mesh_dofs[i] = mesh->globalDofCount();
    mesh_err[i]  = sqrt(l2NormSquared+h1NormSquared);
    mesh_rate[i] = 0;
    if (i > 0) {
      mesh_rate[i] = (log(mesh_err[i])-log(mesh_err[i-1]))/(log(mesh_dofs[i])-log(mesh_dofs[i-1]));
    }
    if (rank == 0) {
      cout << "Global dof:       " << mesh_dofs[i] << endl;
      cout << fixed;
      cout << "Error rate:       " << mesh_rate[i] << endl;
      cout << scientific;
      //cout << "Energy error:     " << energyError << endl;
      cout << "Error:            " << mesh_err[i] << endl; // L2 or H1 error
      //cout << "L^2 BAE  :        " << sqrt(baeL2Squared ) << endl;
    }
  }
  
  /*
  // export
  HDF5Exporter exporter(mesh,"heat_steady");
  int numSubdivisions = 30; // coarse mesh -> more subdivisions
  int refNumber = 0;
  exporter.exportSolution(soln, refNumber, numSubdivisions);
  cout << "exported" << endl;
  */
  
  return 0;
}
