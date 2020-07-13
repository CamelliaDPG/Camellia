//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "EnergyErrorFunction.h"
#include "Function.h"
#include "GMGSolver.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "CompressibleNavierStokesFormulation.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();

  // To begin with, let's try a steady state shock example, from the "DPG Part V" ICES Report
  /*
   Select density rho_a at left, and Mach number u_a at left.  These are the BCs (Rankine-Hugoniot):
   p_a   = rho_a / gamma
   p_b   = p_a * (1 + (2 * gamma) / (gamma + 1) * (u_a^2 - 1) )
   rho_b = rho_a * [(gamma - 1) + (gamma + 1) p_b / p_a ] / [(gamma + 1) + (gamma - 1) p_b / p_a]
   u_b   = rho_a * u_a / rho_b
   E_a   = (u_a ^ 2) / 2 + p_a / ( (gamma - 1) * rho_a)
   E_b   = (u_b ^ 2) / 2 + p_b / ( (gamma - 1) * rho_b)
   */
  
  int meshWidth = 25;
  int polyOrder = 16;
  int delta_k   = 1; // 1 is likely sufficient in 1D
  bool useCondensedSolve = true;
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  
  double x_a   = 0.0;
  double x_b   = 1.0;
  MeshTopologyPtr meshTopo = MeshFactory::intervalMeshTopology(x_a, x_b, meshWidth);

  double rho_a = 1.0; // prescribed density at left
  double u_a   = 5.0; // Mach number
  double Re    = 1e2; // Reynolds number
  
  bool useConformingTraces = true;
  auto form = CompressibleNavierStokesFormulation::steadyFormulation(spaceDim, Re, useConformingTraces,
                                                                             meshTopo, polyOrder, delta_k);
  form->solutionIncrement()->setUseCondensedSolve(useCondensedSolve);
  form->solutionIncrement()->setCubatureEnrichmentDegree(cubatureEnrichment);
  
  double gamma = form->gamma();
  double c_v   = form->Cv();

  double p_a   = rho_a / gamma;
  double T_a   = p_a / (rho_a * (gamma - 1.) * c_v);
  
  double p_b, rho_b, u_b, T_b;
  
  bool computeShockSolution = true;
  if (computeShockSolution)
  {
    p_b   = p_a * (1. + (2. * gamma) / (gamma +  1.) * (u_a * u_a - 1) );
    rho_b = rho_a * ( (gamma - 1.) + (gamma + 1.) * p_b / p_a ) / ( (gamma + 1.) + (gamma - 1.) * p_b / p_a );
    u_b   = rho_a * u_a / rho_b;
    T_b   = p_b / (rho_b * (gamma - 1.) * c_v);
    
    cout << "rho_b / rho_a = " <<  rho_b / rho_a << endl;
    cout << "u_b / u_a = " << u_b / u_a << endl;
    cout << "p_b / p_a = " << p_b / p_a << endl;
    cout << "T_b / T_a = " << T_b / T_a << endl;
  }
  else
  {
    p_b   = p_a;
    rho_b = rho_a;
    u_b   = u_a;
    T_b   = T_a;
  }
//  double E_a   = (u_a * u_a) / 2 + p_a / ( (gamma-1.) * rho_a);
//  double E_b   = (u_b * u_b) / 2 + p_b / ( (gamma-1.) * rho_b);
  
  FunctionPtr x = Function::xn(1);
  // lambdas for determining a linear interpolant
  auto slope = [&](double val_a, double val_b) {
    return (val_b - val_a) / (x_b - x_a);
  };
  auto y_intercept = [&](double val_a, double val_b) {
    return val_a * x_b - val_b * x_a;
  };
  auto linearInterpolant = [&](double val_a, double val_b)
  {
    double m = slope(val_a,val_b);
    double b = y_intercept(val_a, val_b);
    return m * x + b;
  };
  
  // Let's set initial data that linearly interpolates the BCs
  
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->x() * Function::sideParity();
  
  map<int, FunctionPtr> initialGuess;
  bool interpolatedStart = false;
  if (interpolatedStart)
  {
    FunctionPtr rho = linearInterpolant(rho_a,rho_b);
    FunctionPtr T   = linearInterpolant(T_a, T_b);
    FunctionPtr u   = linearInterpolant(u_a, u_b);
    
    double R  = form->R();
    double Cv = form->Cv();
    double Cp = form->Cp();
    double Pr = form->Pr();
    double mu = form->mu();
    double q_initial = slope(T_a, T_b) * Cp * mu / Pr;
    
    initialGuess[form->rho()->ID()]   = rho;
    initialGuess[form->T()->ID()]     = T;
    initialGuess[form->u(1)->ID()]    = u;
    initialGuess[form->q(1)->ID()]    = Function::constant(q_initial);
    initialGuess[form->D(1,1)->ID()]  = Function::zero();
    
    // fluxes and traces; setting initial guesses for these should not actually matter, I don't think, but we follow Truman here...
    // (The below expressions might elucidate somewhat how the traces/fluxes relate to the fields, however...)
    initialGuess[form->T_hat()->ID()] = T;
    initialGuess[form->tc()->ID()]    = rho * u * n_x;
    initialGuess[form->te()->ID()]    = (Cv * rho * u * T + 0.5 * rho * u * u * u + R * rho * u * T) * n_x;
    
    initialGuess[form->u_hat(1)->ID()] = u;
    initialGuess[form->tm(1)->ID()]    = (rho * u * u + form->R() * rho * T) * n_x;
  }
  else
  {
    // cheating!  Instead of interpolating, let's start at what we know to be the right answer, at least for rho, T, u
    auto H_right = Function::heaviside((x_a + x_b)/2.0); // Heaviside is 0 left of center, 1 right of center
    auto H_left  = 1.0 - H_right;  // this guy is 1 left of center, 0 right of center
    auto step = [&](double val_a, double val_b)
    {
      return H_left * val_a + H_right * val_b;
    };
    
    FunctionPtr rho = step(rho_a,rho_b);
    FunctionPtr T   = step(T_a, T_b);
    FunctionPtr u   = step(u_a, u_b);
    
    initialGuess[form->rho()->ID()]   = rho;
    initialGuess[form->T()->ID()]     = T;
    initialGuess[form->u(1)->ID()]    = u;
    initialGuess[form->q(1)->ID()]    = Function::zero();
    initialGuess[form->D(1,1)->ID()]  = Function::zero();
    
    // fluxes and traces; setting initial guesses for these should not actually matter, I don't think, but we follow Truman here...
    // (The below expressions might elucidate somewhat how the traces/fluxes relate to the fields, however...)
    double R  = form->R();
    double Cv = form->Cv();
    double Cp = form->Cp();
    double Pr = form->Pr();
    double mu = form->mu();
    initialGuess[form->T_hat()->ID()] = T;
    initialGuess[form->tc()->ID()]    = rho * u * n_x;
    initialGuess[form->te()->ID()]    = (Cv * rho * u * T + 0.5 * rho * u * u * u + R * rho * u * T) * n_x;
    
    initialGuess[form->u_hat(1)->ID()] = u;
    initialGuess[form->tm(1)->ID()]    = (rho * u * u + form->R() * rho * T) * n_x;
  }
  
  form->solution()->projectOntoMesh(initialGuess);
  
  HDF5Exporter solutionExporter(form->solutionIncrement()->mesh(), "sodShockSteadySolution", ".");
  HDF5Exporter solutionIncrementExporter(form->solutionIncrement()->mesh(), "sodShockSteadySolutionIncrement", ".");
  
  double t = 0.0; // we're steady state; we're using t = 0.0 to indicate the Newton Step
  solutionIncrementExporter.exportSolution(form->solutionIncrement(), t); // set
  solutionExporter.exportSolution(form->solution(), t);
  
  // Borrowing some from Truman's dissertation code
  SpatialFilterPtr leftX  = SpatialFilter::matchingX(x_a);
  SpatialFilterPtr rightX = SpatialFilter::matchingX(x_b);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);
  
  std::cout << "T_a = " << T_a << std::endl;
  std::cout << "T_b = " << T_b << std::endl;
//  (SpatialFilterPtr region, FunctionPtr rho_exact, FunctionPtr u_exact, FunctionPtr T_exact)
  form->addMassFluxCondition(         leftX, Function::constant(rho_a), Function::constant(u_a), Function::constant(T_a));
  form->addVelocityTraceCondition(    leftX, Function::constant(u_a)         );
  form->addTemperatureTraceCondition( leftX, Function::constant(T_a)         );
  form->addMassFluxCondition(        rightX, Function::constant(rho_b), Function::constant(u_b), Function::constant(T_b));
  form->addVelocityTraceCondition(   rightX, Function::constant(u_b)         );
  form->addTemperatureTraceCondition(rightX, Function::constant(T_b)         );
  
  VarPtr u1_hat = form->u_hat(1);

  double nonlinearTolerance = 1e-2;
  double l2NormOfIncrement = 1.0;
  int stepNumber = 0;
  int maxNonlinearSteps = 20;
  while ((l2NormOfIncrement > nonlinearTolerance) && (stepNumber < maxNonlinearSteps))
  {
    double alpha = form->solveAndAccumulate();
    int solveCode = form->getSolveCode();
    if (solveCode != 0)
    {
      if (rank==0) cout << "Solve not completed correctly; aborting..." << endl;
      exit(1);
    }
    l2NormOfIncrement = form->L2NormSolutionIncrement();
    std::cout << "In Newton step " << stepNumber << ", L^2 norm of increment = " << l2NormOfIncrement;
    std::cout << " (alpha = " << alpha << ")" << std::endl;
    
    stepNumber++;
    solutionExporter.exportSolution(form->solution(), double(stepNumber));  // use stepNumber as the "time" value for export...
    solutionIncrementExporter.exportSolution(form->solutionIncrement(), double(stepNumber));
  }
  
  // create a refStrategy, just for the purpose of uniform refinement.
  // (The RHS is likely incorrect for refinement purposes, because of the treatment of fluxes.  We can fix this; see NavierStokesVGPFormulation for how...)
  double energyThreshold = 0.2;
  auto refStrategy = RefinementStrategy::energyErrorRefinementStrategy(form->solutionIncrement(), energyThreshold);
  
  int numUniformRefinements = 2;
  int stepOffset = stepNumber;
  for (int refNumber = 0; refNumber < numUniformRefinements; refNumber++)
  {
    std::cout << "**** Performing Uniform Refinement ****\n";
    refStrategy->hRefineUniformly();
    stepNumber = 0;
    l2NormOfIncrement = 1.0;
    while ((l2NormOfIncrement > nonlinearTolerance) && (stepNumber < maxNonlinearSteps))
    {
      double alpha = form->solveAndAccumulate();
      int solveCode = form->getSolveCode();
      if (solveCode != 0)
      {
        if (rank==0) cout << "Solve not completed correctly; aborting..." << endl;
        exit(1);
      }
      l2NormOfIncrement = form->L2NormSolutionIncrement();
      std::cout << "In Newton step " << stepNumber << ", L^2 norm of increment = " << l2NormOfIncrement;
      std::cout << " (alpha = " << alpha << ")" << std::endl;
      
      stepNumber++;
      solutionExporter.exportSolution(form->solution(), double(stepNumber + stepOffset));  // use stepNumber as the "time" value for export...
      solutionIncrementExporter.exportSolution(form->solutionIncrement(), double(stepNumber + stepOffset));
    }
    stepOffset += stepNumber;
  }
  
  
  return 0;
}
