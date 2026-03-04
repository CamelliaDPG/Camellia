//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  HeatTransientDriver.cpp
//  Camellia
//
//  Created by Stefan Henneking on 11/14/19.
//
//  Revised by Nate Roberts 3/4/26 (minimally, to get things running again on development-2.0 branch of Camellia).
//

#include "Camellia.h"

#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "HeatFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

#include "Teuchos_Time.hpp"

using namespace Camellia;
using namespace std;

enum ProblemChoice
{
  SINUSOIDAL_MANUFACTURED  // Example 4.1 from Führer and Heuer
};

enum FormulationChoice
{
  GALERKIN,
  PRIMAL,
  ULTRAWEAK
};

enum NormChoice
{
  GRAPH_NORM,
  MANUAL_GRAPH_NORM
};

const double PI = 3.141592653589793238462;

void runSolve(vector<vector<double>> domainDim, int numElements, double dt, double finalTime, double theta, int polyOrder, int deltaP, ProblemChoice problemChoice, FormulationChoice formulationChoice, NormChoice normChoice, bool vis)
{
  if (theta < 0.0 || theta > 1.0) {
    cout << "Invalid parameter: theta = " << theta << endl; return;
  }
  bool fieldImplicit = false; bool fixedPointScheme = false;
  if (formulationChoice == GALERKIN) {fieldImplicit = false; fixedPointScheme = false;}
  
  int rank = Teuchos::GlobalMPISession::getRank();
  int spaceDim = domainDim.size();
  vector<double> x0(spaceDim);
  vector<double> domainSize(spaceDim);
  vector<int> elementCounts(spaceDim);
  for (int d=0; d<spaceDim; d++)
  {
    x0[d] = domainDim[d][0];
    domainSize[d] = domainDim[d][1] - x0[d];
    elementCounts[d] = numElements;
  }
  
  HeatFormulation::HeatFormulationChoice choice;
  switch (formulationChoice) {
    case GALERKIN  : choice = HeatFormulation::CONTINUOUS_GALERKIN; break;
    case PRIMAL    : choice = HeatFormulation::PRIMAL             ; break;
    case ULTRAWEAK : choice = HeatFormulation::ULTRAWEAK          ; break;
  }
  bool conformingTraces = true;
  HeatFormulation form(spaceDim, conformingTraces, dt, theta, choice, fieldImplicit);
  VarPtr u,u_hat,v,tau;
  u = form.u();
  v = form.v();
  if (choice == HeatFormulation::ULTRAWEAK) {
    if (!fieldImplicit) u_hat = form.u_hat();
    tau = form.tau();
  }
  BFPtr bf = form.bf();
  
  FunctionPtr u_exact; // defined for manufactured solution
  FunctionPtr u_init;
  FunctionPtr u_bc;
  FunctionPtr forcing;
  FunctionPtr u_exact_prev;
  FunctionPtr forcing_prev;
  
  // set up a parameter function; this will allow us to have an exact solution that is time-dependent
  ParameterFunctionPtr t = ParameterFunction::parameterFunction(0.0);
  ParameterFunctionPtr t_prev = ParameterFunction::parameterFunction(0.0);
  
  if (problemChoice == SINUSOIDAL_MANUFACTURED)
  {
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr exp_decay = Exp<double>::exp(-PI*PI*FunctionPtr(t));
    FunctionPtr exp_decay_prev = Exp<double>::exp(-PI*PI*FunctionPtr(t_prev));
    u_exact = exp_decay * TrigFunctions<double>::sin(PI * x) * TrigFunctions<double>::sin(PI * y);
    u_exact_prev = exp_decay_prev * TrigFunctions<double>::sin(PI * x) * TrigFunctions<double>::sin(PI * y);
    u_init = TrigFunctions<double>::sin(PI * x) * TrigFunctions<double>::sin(PI * y);
    u_bc = u_exact; // zero on boundary of unit square
    forcing = PI*PI*u_exact;
    forcing_prev = PI*PI*u_exact_prev;
  }
  else
  {
    cout << "HeatTransientDriver: Invalid problemChoice parameter.\n";
    return;
  }
  
  // Test norm
  IPPtr ip = IP::ip();
  
  // BC
  BCPtr bc = BC::bc();
  
  switch (formulationChoice) {
    case GALERKIN:
      bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
      deltaP = 0;
      ip = Teuchos::null;
      break;
    case PRIMAL:
      bc->addDirichlet(form.u(), SpatialFilter::allSpace(), u_bc);
      // Mathematician's norm
      //ip = bf->naiveNorm(spaceDim);
      //if (rank==0) cout << "PRIMAL: Using Mathematician's norm." << endl;
      
      // Choosing test norm as the energy norm that Galerkin converges in
      ip->addTerm(v/sqrt(dt));
      ip->addTerm(v->grad());
      if (rank==0) cout << "PRIMAL: Using Galerkin energy norm." << endl;
      break;
    case ULTRAWEAK:
      if (!fieldImplicit) bc->addDirichlet(form.u_hat(), SpatialFilter::allSpace(), u_bc); // TODO: this needs adjustment in Field Implicit Version
      if (normChoice == MANUAL_GRAPH_NORM) {
        if (rank==0) cout << "ULTRAWEAK: Using manual graph norm." << endl;
        // True adjoint graph norm (not scaled):
        // this norm avoids the addition of L^2 terms; we still have a localizable norm in this case
        //ip->addTerm(tau->div() + v / dt);  // corresponds to u
        //ip->addTerm(tau + v->grad());      // corresponds to sigma
          
        // ToFu's norm:
        // introduces an additional 1/dt coefficient on some of the terms to eliminate cross-terms
        ip->addTerm(v/dt);
        ip->addTerm(v->grad()/sqrt(dt));
        ip->addTerm(tau/sqrt(dt));
        ip->addTerm(tau->div());
      } else {
        ip = bf->graphNorm(); // Scaled Adjoint graph norm
        if (rank==0) cout << "ULTRAWEAK: Using scaled adjoint graph norm." << endl;
      }
      break;
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(domainSize[0], domainSize[1], elementCounts[0], elementCounts[1]);
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, bf->varFactory(), polyOrder, deltaP) );
  mesh->setBilinearForm(bf);
  
  // RHS
  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(theta * forcing * v);
  rhs->addTerm((1.0-theta) * forcing_prev * v);
  SolutionPtr soln      = Solution::solution(bf, mesh, bc, rhs, ip);
  SolutionPtr soln_prev = Solution::solution(bf, mesh, bc, rhs, ip);
  SolutionPtr soln_temp = Solution::solution(bf, mesh, bc, rhs, ip);
  FunctionPtr u_curr    = Function::solution(u, soln);
  FunctionPtr u_prev    = Function::solution(u, soln_prev);
  FunctionPtr u_temp    = Function::solution(u, soln_temp);
  TFunctionPtr<double> n   = TFunction<double>::normal();
  TFunctionPtr<double> sgn = TFunction<double>::sideParity();
  FunctionPtr Gamma   = Function::meshBoundaryCharacteristic(); // global boundary
  FunctionPtr Gamma_h = Function::meshSkeletonCharacteristic(); // entire mesh skeleton
  rhs->addTerm((u_prev/dt) * v);
  if (formulationChoice == GALERKIN && theta < 1.0) {
    rhs->addTerm(Gamma * (1.0-theta)*(u_prev->grad(2)*n) * v);
    rhs->addTerm(-(1.0-theta)*u_prev->grad(2) * v->grad());
  } else if (formulationChoice == PRIMAL && (theta < 1.0 || fieldImplicit)) {
    rhs->addTerm(-(1.0-theta)*u_prev->grad(2) * v->grad());
    
    // IF DPG and theta<1.0, we must add a flux term to the RHS
    
    // Using neighbor function
    FunctionPtr sigma_prev       = u_prev->grad(2)*n;
    FunctionPtr sigma_exact_prev = u_exact_prev->grad(2)*n;
    auto sigma_neighbor          = Function::neighborFunction(sigma_prev, mesh);
    
    //auto myCellIDs = mesh->cellIDsInPartition();
    //std::vector<GlobalIndexType> myCellIDsVector(myCellIDs.begin(),myCellIDs.end());
    //sigma_neighbor->importCellData(myCellIDsVector); // needed whenever solution changes for MPI data exchange
    
    if (theta > 0.0 && !fieldImplicit) {
      // (1) Use flux trace unknown from previous time step
      FunctionPtr sigma_n_hat_prev = Function::solution(form.sigma_n_hat(), soln_prev, true);
      rhs->addTerm((1.0-theta) * sigma_n_hat_prev * v);
    } else {
      // Now, we must compute the trace term from the field solution u (since no trace unknown is available)
      
      // (2) Calculate flux trace from previous exact solution
      //rhs->addTerm(Gamma_h * (1.0-theta)*(u_exact_prev->grad(2)*n) * v);
      
      // (3) Calculate flux average from previous approximate solution (with neighbor function)
      rhs->addTerm((Gamma_h-Gamma) * (1.0-theta)*0.5*(sigma_prev - sigma_neighbor) * v); // sign takes care of flipped normal direction
      rhs->addTerm(         Gamma  * (1.0-theta) * sigma_prev * v);
      
      if (fieldImplicit) {
        // Field implicit version 1 (with flux at previous time step on the RHS)
        if (!fixedPointScheme) {
          rhs->addTerm((Gamma_h-Gamma) * theta*0.5*(sigma_prev - sigma_neighbor) * v); // sign takes care of flipped normal direction
          rhs->addTerm(         Gamma  * theta * sigma_prev * v);
        }
        // Field implicit version 2 (with flux at current time step on the RHS)
        else {
          FunctionPtr sigma_temp = u_temp->grad(2)*n;
          auto sigma_neighbor_temp  = Function::neighborFunction(sigma_temp, mesh);
          rhs->addTerm((Gamma_h-Gamma) * theta*0.5*(sigma_temp - sigma_neighbor_temp) * v); // sign takes care of flipped normal direction
          rhs->addTerm(         Gamma  * theta* sigma_temp * v);
        }
      }
    }
  } else if (formulationChoice == ULTRAWEAK && (theta < 1.0 || fieldImplicit)) {
    VarPtr sigma = form.sigma();
    FunctionPtr sigma_prev       = Function::solution(sigma, soln_prev);
    FunctionPtr sigma_exact_prev = u_exact_prev->grad(2);
    FunctionPtr sigma_n_hat_exact_prev = sigma_exact_prev*n;
    
    rhs->addTerm(-(1.0-theta)*sigma_prev * v->grad());
    
    // IF DPG and theta<1.0, we must add a flux term to the RHS
  
    // Using neighbor function
    auto sigma_neighbor       = Function::neighborFunction(sigma_prev  , mesh);
    auto sigma_n_hat_neighbor = Function::neighborFunction(sigma_prev*n, mesh);
  
    if (theta > 0.0 && !fieldImplicit) {
      // (1) Use flux trace unknown from previous time step
      FunctionPtr sigma_n_hat_prev = Function::solution(form.sigma_n_hat(), soln_prev, true);
      rhs->addTerm((1.0-theta) * sigma_n_hat_prev * v); // (1) working
    } else {
      // Now, we must compute the trace terms from the field solution (u,sigma) (since no trace unknown is available)
      
      // (2) Calculate flux trace from previous exact solution
      //rhs->addTerm(Gamma_h * (1.0-theta) * sigma_n_hat_exact_prev * v); // (2) working
      
      // (3) Calculate flux average from previous approximate solution (with neighbor function)
      rhs->addTerm((Gamma_h-Gamma) * (1.0-theta)*0.5*(sigma_prev*n - sigma_n_hat_neighbor) * v); // sign takes care of flipped normal direction
      rhs->addTerm(         Gamma  * (1.0-theta) * sigma_prev*n * v);
      
      if (fieldImplicit) {
        // Field implicit version 1 (with flux at previous time step on the RHS)
        if (!fixedPointScheme) {
          // trace of flux var
          rhs->addTerm((Gamma_h-Gamma) * theta*0.5*(sigma_prev*n - sigma_n_hat_neighbor) * v); // sign takes care of flipped normal direction
          rhs->addTerm(         Gamma  * theta * sigma_prev*n * v);
          //rhs->addTerm(Gamma_h * theta * u_exact->grad(2)*n * v); // using exact flux
          
          // trace of field var
          auto u_neighbor = Function::neighborFunction(u_prev, mesh);
          rhs->addTerm((Gamma_h-Gamma) * 0.5*(u_prev + u_neighbor) * tau*n);
          //rhs->addTerm( (Gamma_h-Gamma) * u_exact * tau*n); // using exact field
          
          // enforce BC weakly
          rhs->addTerm( Gamma * u_bc * tau*n);
        }
        // Field implicit version 2 (with flux at current time step on the RHS)
        else {
          // trace of flux var
          FunctionPtr sigma_temp = Function::solution(sigma, soln_temp);
          auto sigma_n_hat_neighbor_temp = Function::neighborFunction(sigma_temp*n, mesh);
          rhs->addTerm((Gamma_h-Gamma) * theta*0.5*(sigma_temp*n - sigma_n_hat_neighbor_temp) * v); // sign takes care of flipped normal direction
          rhs->addTerm(         Gamma  * theta * sigma_temp*n * v);
          
          // trace of field var
          auto u_neighbor = Function::neighborFunction(u_temp, mesh);
          rhs->addTerm( (Gamma_h-Gamma) * 0.5*(u_temp + u_neighbor) * tau*n);
          
          // enforce BC weakly
          rhs->addTerm( Gamma * u_bc * tau*n);
        }
      }
    }
  }
  
  // Initial data
  map<int, FunctionPtr> initValueMap;
  initValueMap[u->ID()] = u_init;
  if (formulationChoice == PRIMAL && theta > 0.0 && theta < 1.0 && !fieldImplicit) {
    // Initial condition for the flux trace
    VarPtr sigma_n_hat = form.sigma_n_hat();
    initValueMap[sigma_n_hat->ID()] = u_init->grad(2) * n * sgn;
  } else if (formulationChoice == ULTRAWEAK && (theta < 1.0 || fieldImplicit)) {
    // Initial condition for the flux
    VarPtr sigma = form.sigma();
    initValueMap[sigma->ID()] = u_init->grad(2);
    if (theta > 0.0 && !fieldImplicit) {
      // Initial condition for the flux trace
      VarPtr sigma_n_hat = form.sigma_n_hat();
      initValueMap[sigma_n_hat->ID()] = u_init->grad(2) * n * sgn;
    }
  }
  soln_prev->projectOntoMesh(initValueMap, 0);
  soln_temp->projectOntoMesh(initValueMap, 0); // needed for fixed point iteration (first iterate)
  
  // Vis path
  ostringstream vizPath;
  vizPath << "heat_transient";
  HDF5Exporter vizExporter(mesh, vizPath.str());
  if (vis) vizExporter.exportSolution(soln_prev, t->getConstantValue());
  
  // Time stepping
  int numTimeSteps = int((finalTime+0.99*dt) / dt);
  if (rank==0) cout << "          time step  = " << dt << endl;
  if (rank==0) cout << "number of time steps = " << numTimeSteps << endl;
  
  Teuchos::RCP<Teuchos::Time> stepTimer =
      Teuchos::TimeMonitor::getNewCounter("HeatTransientDriver: solve step");
  // TODO: update neighbor function in every iteration
  for (int timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    t_prev->setValue(timeStepNumber*dt);
    t->setValue((timeStepNumber+1)*dt); // update forcing term, exact solution to current time step
    
    stepTimer->reset();
    stepTimer->start();
    
    if (fieldImplicit && fixedPointScheme) {
      double tol = 1.0e-5; int kmax = 50;
      for (int k = 0; k < kmax; k++) {
        soln->solve();
        double l2diff = (u_curr - u_temp)->l2norm(mesh, 4);
        double l2norm = u_curr->l2norm(mesh,4);
        cout << "k = " << setw(2) << k << ", l2diff/l2norm = " << l2diff/l2norm << endl;
        if (l2diff / l2norm < tol) {
          break; // fixed point iteration converged
        }
        soln_temp->setSolution(soln);
      }
    } else {
      soln->solve();
    }
    stepTimer->stop();
    double solveTime = stepTimer->totalElapsedTime();
    if (rank == 0 && (timeStepNumber%10==0 || (fieldImplicit && fixedPointScheme)))
    { cout << setw(3) << timeStepNumber << " Solved in " << solveTime << " seconds.\n"; }
    soln_prev->setSolution(soln); // update previous solution to current time step
    
    if (vis) vizExporter.exportSolution(soln, t->getConstantValue());
  }
  
  if (vis && rank==0) cout << "Exported solution to " << vizPath.str() << endl;
  
  // Compute best approximation and display error
  if (problemChoice == SINUSOIDAL_MANUFACTURED)
  {
    // Compute best approximation (L2 projection)
    SolutionPtr bestApproximation = Solution::solution(bf, mesh, bc, rhs, ip);
    int solutionOrdinal;
    map<int, FunctionPtr> exactSolutionMap;
    FunctionPtr sigma_exact = u_exact->grad();
    exactSolutionMap[form.u()->ID()] = u_exact;
    if (choice == HeatFormulation::PRIMAL && theta > 0.0 && !fieldImplicit) {
      exactSolutionMap[form.sigma_n_hat()->ID()] = sigma_exact * n * sgn;
    }
    if (choice == HeatFormulation::ULTRAWEAK) {
      exactSolutionMap[form.sigma()->ID()] = sigma_exact;
      if (!fieldImplicit) {
        exactSolutionMap[form.u_hat()->ID()] = u_exact;
        if (theta > 0.0) {
          exactSolutionMap[form.sigma_n_hat()->ID()] = sigma_exact * n * sgn;
        }
      }
    }
    bestApproximation->projectOntoMesh(exactSolutionMap);
    
    // Compute error for each variable
    int cubatureEnrichment = 4;
    map<int,VarPtr> trialVars = bf->varFactory()->trialVars();
    if (rank==0) cout << "Number of trial vars: " << bf->varFactory()->trialVars().size() << endl;
    double l2NormSquared = 0.0;
    double h1NormSquared = 0.0;
    //double bestDifferenceL2Squared = 0.0;
    for (auto entry : trialVars)
    {
      auto var = entry.second;
      
      // skip fluxes in field implicit formulations
      if (var->varType() != FIELD && fieldImplicit) { continue; }
      
      auto var_exact = exactSolutionMap[var->ID()];
      
      bool weightFluxesByParity = false; // here, interested in comparing to exact solutions; should not weight
      auto var_soln = Function::solution(var, soln,              weightFluxesByParity);
      auto var_best = Function::solution(var, bestApproximation, weightFluxesByParity);
      
      double l2bestError   = (var_best - var_exact)->l2norm(mesh, cubatureEnrichment);
      double l2actualError = (var_soln - var_exact)->l2norm(mesh, cubatureEnrichment);
      double h1bestError = 0.0, h1actualError = 0.0;
      if (choice != HeatFormulation::ULTRAWEAK && var->varType() == FIELD) {
        h1bestError   = sqrt(dt)*(var_best - var_exact)->grad(spaceDim)->l2norm(mesh, cubatureEnrichment); // Error in weighted H1 seminorm
        h1actualError = sqrt(dt)*(var_soln - var_exact)->grad(spaceDim)->l2norm(mesh, cubatureEnrichment); // Error in weighted H1 seminorm
      }
      
      double val = 1.0;
      if (choice == HeatFormulation::ULTRAWEAK) {
        if (var->ID() == form.sigma()->ID()) val = sqrt(dt);
      }
      cout << scientific;
      if (rank==0) cout << setw(18) << var->name() << " L2 BAE  : " << val*l2bestError   << endl;
      if (rank==0) cout << setw(18) << var->name() << " L2 error: " << val*l2actualError << endl;
      if (choice != HeatFormulation::ULTRAWEAK && var->ID() == form.u()->ID()) {
        if (rank==0) cout << setw(18) << var->name() << " H1 BAE  : " << h1bestError   << endl;
        if (rank==0) cout << setw(18) << var->name() << " H1 error: " << h1actualError << endl;
      }
      if (var->varType() != FIELD) { continue; }
      
      l2NormSquared += (val*l2actualError) * (val*l2actualError);
      h1NormSquared += h1actualError * h1actualError;
    }
    
    cout << scientific;
    //if (rank==0) cout << "L^2 error   :  " << sqrt(l2NormSquared) << endl;
    //if (rank==0) cout << "H^1 error   :  " << sqrt(h1NormSquared) << endl;
    if (rank==0) cout << "Error       :  " << sqrt(l2NormSquared+h1NormSquared) << endl;
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  Kokkos::initialize(argc, argv); // initialize Kokkos
  {
    int spaceDim = 2;
    int numElements = 4; // in each dimension
    vector<vector<double>> domainDim(spaceDim,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
    int polyOrder = 3;
    int deltaP = spaceDim;
    string normChoiceString = "Graph"; // "ManualGraph" is the alternative
    string formulationChoiceString = "Ultraweak"; // "Primal","Galerkin" the alternatives
    string problemChoiceString = "Manufactured"; // "Unit Forcing" the alternative
    double dt = sqrt(1.0/numElements)/20.0; // Follows Führer and Heuer: dt = sqrt(h)/20, unit square
    double finalTime = 0.1; // Follows Führer and Heuer: T = 0.1
    double theta = 1.0; // time-stepping method parameter (default: 1.0=Backward Euler)
    bool vis = false;
    bool runAll = false;
    
    cmdp.setOption("norm", &normChoiceString);
    cmdp.setOption("formulation", &formulationChoiceString);
    cmdp.setOption("problem", &problemChoiceString);
    cmdp.setOption("meshWidth", &numElements);
    cmdp.setOption("polyOrder", &polyOrder);
    cmdp.setOption("deltaP", &deltaP);
    cmdp.setOption("dt", &dt);
    cmdp.setOption("T", &finalTime);
    cmdp.setOption("theta", &theta);
    cmdp.setOption("vis", "novis", &vis);
    cmdp.setOption("runAll", "runOne", &runAll);
    cmdp.setOption("x0", &domainDim[0][0] );
    cmdp.setOption("x1", &domainDim[0][1] );
    cmdp.setOption("y0", &domainDim[1][0] );
    cmdp.setOption("y1", &domainDim[1][1] );
    
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
    {
#ifdef HAVE_MPI
      MPI_Finalize();
#endif
      return -1;
    }
    //dt = sqrt(1.0/numElements)/20.0; // ToFu
    //dt = pow(1.0/numElements,2)/2.0; // CFL
    
    NormChoice normChoice;
    if      (normChoiceString == "ManualGraph") normChoice = MANUAL_GRAPH_NORM;
    else if (normChoiceString == "Graph"      ) normChoice = GRAPH_NORM;
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized norm choice");
    }
    
    FormulationChoice formulationChoice;
    if      (formulationChoiceString == "Ultraweak" ) formulationChoice = ULTRAWEAK;
    else if (formulationChoiceString == "Primal"    ) formulationChoice = PRIMAL;
    else if (formulationChoiceString == "Galerkin"  ) formulationChoice = GALERKIN;
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized formulation choice");
    }
    
    ProblemChoice problemChoice;
    if (problemChoiceString == "Manufactured") problemChoice = SINUSOIDAL_MANUFACTURED;
    else
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized problem choice");
    }
    
    if (!runAll) {
      runSolve(domainDim, numElements, dt, finalTime, theta, polyOrder, deltaP, problemChoice, formulationChoice, normChoice, vis);
    }
    else {
      vector<int> elementCounts = {1,2,4,8,16,32}; // should update dt correspondingly
      vector<NormChoice> normChoices = {GRAPH_NORM, MANUAL_GRAPH_NORM};
      for (int numElements : elementCounts)
      {
        cout << "*************** mesh width " << numElements << " *******************\n";
        for (NormChoice normChoice : normChoices)
        {
          runSolve(domainDim, numElements, dt, finalTime, theta, polyOrder, deltaP, problemChoice, formulationChoice, normChoice, vis);
        }
      }
    }
  }
  
  Camellia::finalize(false); // false: don't print allocation report
  
  return 0;
}
