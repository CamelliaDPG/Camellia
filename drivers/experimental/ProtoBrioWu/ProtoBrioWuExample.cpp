//
// For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "EnergyErrorFunction.h"
#include "ExpFunction.h" // defines Ln
#include "Function.h"
#include "Functions.hpp"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "IdealMHDFormulation.hpp"
#include "LagrangeConstraints.h"
#include "RefinementStrategy.h"
#include "RHS.h"
#include "SimpleFunction.h"
#include "SuperLUDistSolver.h"
#include "TrigFunctions.h"

#include "Teuchos_GlobalMPISession.hpp"

using namespace Camellia;

const double PI  = 3.141592653589793238462;

void addConservationConstraint(Teuchos::RCP<IdealMHDFormulation> form)
{
  int rank = MPIWrapper::CommWorld()->MyPID();
  cout << "TRYING CONSERVATION ENFORCEMENT.\n";
  auto soln = form->solution();
  auto solnIncrement = form->solutionIncrement();
  auto prevSoln = form->solutionPreviousTimeStep();
  auto bf  = solnIncrement->bf();
  auto rhs = solnIncrement->rhs();
  auto dt = form->getTimeStep();
  
  Teuchos::RCP<LagrangeConstraints> constraints = Teuchos::rcp(new LagrangeConstraints);
  // vc constraint:
  VarPtr vc = form->vc();
  map<int, FunctionPtr> vcEqualsOne = {{vc->ID(), Function::constant(1.0)}};
  LinearTermPtr vcTrialFunctional = bf->trialFunctional(vcEqualsOne) * dt;
  FunctionPtr   vcRHSFunction     = rhs->linearTerm()->evaluate(vcEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
  constraints->addConstraint(vcTrialFunctional == vcRHSFunction);
  if (rank == 0) cout << "Added element constraint " << vcTrialFunctional->displayString() << " == " << vcRHSFunction->displayString() << endl;

  const int spaceDim = 1;
  // vm constraint(s):
  for (int d=0; d<spaceDim; d++)
  {
    // test with 1
    VarPtr vm = form->vm(d+1);
    map<int, FunctionPtr> vmEqualsOne = {{vm->ID(), Function::constant(1.0)}};
    LinearTermPtr trialFunctional = bf->trialFunctional(vmEqualsOne) * dt; // multiply both by dt in effort to improve conditioning...
    FunctionPtr rhsFxn = rhs->linearTerm()->evaluate(vmEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
    constraints->addConstraint(trialFunctional == rhsFxn);

    if (rank == 0) cout << "Added element constraint " << trialFunctional->displayString() << " == " << rhsFxn->displayString() << endl;
  }
  // ve constraint:
  VarPtr ve = form->ve();
  map<int, FunctionPtr> veEqualsOne = {{ve->ID(), Function::constant(1.0)}};
  LinearTermPtr veTrialFunctional = bf->trialFunctional(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
  FunctionPtr   veRHSFunction     = rhs->linearTerm()->evaluate(veEqualsOne) * dt;  // multiply both by dt in effort to improve conditioning...
  constraints->addConstraint(veTrialFunctional == veRHSFunction);
  if (rank == 0) cout << "Added element constraint " << veTrialFunctional->displayString() << " == " << veRHSFunction->displayString() << endl;

  // although enforcement only happens in solnIncrement, the constraints change numbering of dofs, so we need to set constraints in each Solution object
  solnIncrement->setLagrangeConstraints(constraints);
  soln->setLagrangeConstraints(constraints);
  prevSoln->setLagrangeConstraints(constraints);
}

enum TestNormChoice
{
  STEADY_GRAPH_NORM,
  TRANSIENT_GRAPH_NORM,
  EXPERIMENTAL_CONSERVATIVE_NORM
};

FunctionPtr ln(FunctionPtr arg)
{
  return Teuchos::rcp(new Ln<double>(arg));
}

template<class Form>
int runSolver(Teuchos::RCP<Form> form, double dt, double finalTime, int outputFrequency, int horizontalElements,
              int polyOrder, int cubatureEnrichment, bool useCondensedSolve, double nonlinearTolerance,
              TestNormChoice normChoice, bool enforceConservationUsingLagrangeMultipliers,
              bool spaceTime, bool useLeftState, double ByMagnitude, bool enableAdaptivity)
{
  MeshPtr mesh = form->solutionIncrement()->mesh();
  
  if (enforceConservationUsingLagrangeMultipliers)
  {
    addConservationConstraint(form);
  }
  int rank = Teuchos::GlobalMPISession::getRank();
  
  SolutionPtr solnIncrement = form->solutionIncrement();
  SolutionPtr soln = form->solution();
  SolutionPtr solnPreviousTime = form->solutionPreviousTimeStep();
  
  auto ip = solnIncrement->ip(); // this will be the transient graph norm...
  if (normChoice == STEADY_GRAPH_NORM)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported norm choice");
  }
  else if (normChoice == EXPERIMENTAL_CONSERVATIVE_NORM)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported norm choice");
  }
  
  int numTimeSteps = spaceTime ? 1 : finalTime / dt;
  
  int fieldDofCount = mesh->numFieldDofs(); // MPI-communicating method
  int traceDofCount = mesh->numFluxDofs();  // MPI-communicating method
  int totalDofCount = fieldDofCount + traceDofCount;
  if (rank == 0)
  {
    using namespace std;
    cout << "Solving with:\n";
    cout << "p  = " << polyOrder << endl;
    cout << "dt = " << dt << endl;
    int totalElements = mesh->numActiveElements();
    if (spaceTime)
    {
      int spatialElements = horizontalElements;
      
      int temporalElements = totalElements / spatialElements;
      cout << spatialElements << " spatial elements; " << temporalElements << " temporal divisions.\n";
    }
    else
    {
      cout << totalElements << " elements; " << numTimeSteps << " timesteps.\n";
    }
    cout << totalDofCount << " total degrees of freedom (" << fieldDofCount << " volumetric; " << traceDofCount << " on mesh skeleton).\n";
  }
  
  form->setTimeStep(dt);
  solnIncrement->setUseCondensedSolve(useCondensedSolve);
  solnIncrement->setCubatureEnrichmentDegree(cubatureEnrichment);
  soln->setUseCondensedSolve(useCondensedSolve);
  if (!spaceTime)
  {
    solnPreviousTime->setUseCondensedSolve(useCondensedSolve);
  }
//  solnIncrement->setWriteMatrixToMatrixMarketFile(true, "/tmp/A.dat");
//  solnIncrement->setWriteRHSToMatrixMarketFile(   true, "/tmp/b.dat");
  
  double gamma = form->gamma();
  
  // set up initial conditions for 1D problem

  // For By, we want:
  //  0.0 for x ≤ 1.0, x ≥ 2.0
  // -A   for 1.0 < x < 1.5
  // +A   for 1.5 ≤ x < 2.0
  SpatialFilterPtr leftRegion  = SpatialFilter::greaterThanX(1.0)          & SpatialFilter::lessThanX(1.5);
  SpatialFilterPtr rightRegion = SpatialFilter::greaterThanOrEqualToX(1.5) & SpatialFilter::lessThanX(2.0);
  
  auto uxInitial = Function::zero();
  auto uyInitial = Function::zero();
  auto uzInitial = Function::zero();
  
  auto BxInitial = Function::constant(0.75);
  
  auto BzInitial = Function::zero();
  
  FunctionPtr ByInitial;
  double A    = ByMagnitude;
  auto plusA  = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction( A, leftRegion);
  auto minusA = SpatiallyFilteredFunction<double>::spatiallyFilteredFunction(-A, rightRegion);
  ByInitial = plusA + minusA;
  
  FunctionPtr rhoInitial, pressureInitial;
  if (useLeftState)
  {
    rhoInitial = Function::constant(1.0);
    pressureInitial = Function::constant(1.0);
  }
  else
  {
    rhoInitial = Function::constant(0.125);
    pressureInitial = Function::constant(0.1);
  }
  
  auto BInitial  = Function::vectorize(BxInitial, ByInitial, BzInitial);
  auto uInitial  = Function::vectorize(uxInitial, uyInitial, uzInitial);

  auto MxInitial = rhoInitial * uxInitial;
  auto MyInitial = rhoInitial * uyInitial;
  auto MzInitial = Function::zero();
  
  auto EInitial = pressureInitial / (gamma - 1.) + 0.5 * dot(3,uInitial,uInitial) + 0.5 * dot(3,BInitial,BInitial);
  
  double R  = form->R();
  double Cv = form->Cv();
  
  if (rank == 0)
  {
    cout << "R =     " << R << endl;
    cout << "Cv =    " << Cv << endl;
    cout << "gamma = " << gamma << endl;
    cout << "Initial State:\n";
    cout << "rho = " << rhoInitial->displayString() << endl;
    cout << "E   = " << EInitial->displayString()   << endl;
    cout << "Bx  = " << BxInitial->displayString()  << endl;
    cout << "By  = " << ByInitial->displayString()  << endl;
    cout << "Bz  = " << BzInitial->displayString()  << endl;
  }
  
  FunctionPtr n = Function::normal();
  FunctionPtr n_x = n->x() * Function::sideParity();
  
  {
    // for Ideal MHD, even in 1D, u is a vector, as is B
    form->setInitialCondition(rhoInitial, uInitial, EInitial, BInitial);
    if (spaceTime)
    {
      // have had some issues using the discontinuous initial guess for all time in Sod problem at least
      // let's try something much more modest: just unit values for rho, E, B, zero for u
      FunctionPtr one    = Function::constant(1.0);
      FunctionPtr rhoOne = one;
      FunctionPtr EOne   = one;
      FunctionPtr BGuess = Function::vectorize(BxInitial, one, one);
      auto initialGuess = form->exactSolutionFieldMap(rhoOne, uInitial, EOne, BGuess);
      form->setInitialState(initialGuess);
    }
  }
  
  auto & prevSolnMap = form->solutionPreviousTimeStepFieldMap();
  auto & solnMap     = form->solutionFieldMap();
  auto pAbstract = form->abstractPressure();
  auto TAbstract = form->abstractTemperature();
  auto uAbstract = form->abstractVelocity()->x();
  auto vAbstract = form->abstractVelocity()->y();
  auto wAbstract = form->abstractVelocity()->z();
  auto mAbstract = form->abstractMomentum();
  auto EAbstract = form->abstractEnergy();
  
  // define the pressure so we can plot in our solution export
  FunctionPtr p_prev = pAbstract->evaluateAt(prevSolnMap);
  FunctionPtr p_soln = pAbstract->evaluateAt(solnMap);
  
  // define u so we can plot in solution export
//  cout << "uAbstract = " << uAbstract->displayString() << endl;
  FunctionPtr u1_prev = uAbstract->evaluateAt(prevSolnMap);
  FunctionPtr u2_prev = vAbstract->evaluateAt(prevSolnMap);
  
  FunctionPtr u2_soln = vAbstract->evaluateAt(solnMap);
  FunctionPtr T_soln = TAbstract->evaluateAt(solnMap);
  
  auto velocityAbstract = form->abstractVelocity();
  auto velocity_soln = velocityAbstract->evaluateAt(solnMap);
  auto velocityMagnitude_soln = Function::sqrtFunction(dot(3,velocity_soln,velocity_soln));
  
  // define change in entropy so we can plot in our solution export
  // s2 - s1 = c_p ln (T2/T1) - R ln (p2/p1)
  FunctionPtr ds;
  {
    FunctionPtr p1 = p_prev;
    FunctionPtr p2 = pAbstract->evaluateAt(solnMap);
    FunctionPtr T1 = TAbstract->evaluateAt(prevSolnMap);
    FunctionPtr T2 = TAbstract->evaluateAt(solnMap);
    double c_p = form->Cp();
    ds = c_p * ln(T2 / T1) - R * ln(p2/p1);
  }
  
  {
    // check that initial solution satisfies physicality constraints
    auto T_previous = TAbstract->evaluateAt(prevSolnMap);
    int posEnrich = 5;
    double minValue = T_previous->minimumValue(mesh, posEnrich);
    cout << "min temperature at time 0: " << minValue << endl;
  }
  
  if (spaceTime)
  {
    // DEBUGGING: print out all BF integrations (trying to see why we get zero rows)
//    solnIncrement->bf()->setPrintTermWiseIntegrationOutput(true);
  }
  
  vector<FunctionPtr> functionsToPlot = {p_soln, u2_soln, T_soln};
  vector<string> functionNames = {"pressure","u2","T"};
  
  // history export gets every nonlinear increment as a separate step
  HDF5Exporter solutionHistoryExporter(mesh, "protoBrioWuSolutionHistory", ".");
  HDF5Exporter solutionIncrementHistoryExporter(mesh, "protoBrioWuSolutionIncrementHistory", ".");
  
  ostringstream solnName;
  solnName << "protoBrioWuSolution";
  if (useLeftState)
    solnName << "Left";
  else
    solnName << "Right";
  if (ByMagnitude != 1.0)
    solnName << "_ByMag" << ByMagnitude;
  solnName << "_dt" << dt << "_k" << polyOrder;
  
  HDF5Exporter solutionExporter(mesh, solnName.str(), ".");
  
  solutionIncrementHistoryExporter.exportSolution(form->solutionIncrement(), 0.0);
  solutionHistoryExporter.exportSolution(form->solution(), 0.0);
  solutionExporter.exportSolution(form->solutionPreviousTimeStep(), functionsToPlot, functionNames, 0.0);
  
//  IPPtr naiveNorm = form->solutionIncrement()->bf()->naiveNorm(spaceDim);
//  if (rank == 0)
//  {
//    cout << "*************************************************************************\n";
//    cout << "**********************    USING NAIVE NORM    ***************************\n";
//    cout << "*************************************************************************\n";
//  }
//  form->solutionIncrement()->setIP(naiveNorm);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);

  auto velocityVector = Function::vectorize(Function::zero(), Function::zero(), Function::zero());
  
  FunctionPtr rho = Function::solution(form->rho(), form->solution());
  FunctionPtr m   = mAbstract->evaluateAt(solnMap);
  FunctionPtr E   = EAbstract->evaluateAt(solnMap);
  FunctionPtr tc  = Function::solution(form->tc(),  form->solution(), true);
  FunctionPtr tm  = Function::solution(form->tm(1), form->solution(), true);
  FunctionPtr te  = Function::solution(form->te(),  form->solution(), true);
  
  auto printConservationReport = [&]() -> void
  {
    double totalMass = rho->integrate(mesh);
    double totalXMomentum = m->spatialComponent(1)->integrate(mesh);
    double totalEnergy = E->integrate(mesh);
    double dsIntegral = ds->integrate(mesh);
    
    if (rank == 0)
    {
      cout << "Total Mass:        " << totalMass << endl;
      cout << "Total x Momentum:    " << totalXMomentum << endl;
      cout << "Total Energy:      " << totalEnergy << endl;
      cout << "Change in Entropy: " << dsIntegral << endl;
    }
  };
  
  // elementwise local momentum conservation:
  FunctionPtr rho_soln = Function::solution(form->rho(), form->solution());
  FunctionPtr rho_prev = Function::solution(form->rho(), form->solutionPreviousTimeStep());
  FunctionPtr rhoTimeStep = (rho_soln - rho_prev) / dt;
  FunctionPtr rhoFlux = Function::solution(form->tc(), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr momentum_soln = mAbstract->x()->evaluateAt(solnMap);
  FunctionPtr momentum_prev = mAbstract->x()->evaluateAt(prevSolnMap);
  FunctionPtr momentumTimeStep = (momentum_soln - momentum_prev) / dt;
  FunctionPtr momentumFlux = Function::solution(form->tm(1), form->solution(), true); // true: include sideParity weights
  
  FunctionPtr energy_soln = EAbstract->evaluateAt(solnMap);
  FunctionPtr energy_prev = EAbstract->evaluateAt(prevSolnMap);
  FunctionPtr energyTimeStep = (energy_soln - energy_prev) / dt;
  FunctionPtr energyFlux = Function::solution(form->te(), form->solution(), true); // include sideParity weights

  vector<FunctionPtr> conservationFluxes = {rhoFlux,     momentumFlux,     energyFlux};
  vector<FunctionPtr> timeDifferences    = {rhoTimeStep, momentumTimeStep, energyTimeStep};
  
  int numConserved = conservationFluxes.size();
  
  auto printLocalConservationReport = [&]() -> void
  {
    auto & myCellIDs = mesh->cellIDsInPartition();
    int cellOrdinal = 0;
    vector<double> maxConservationFailures(numConserved);
    for (int conservedOrdinal=0; conservedOrdinal<numConserved; conservedOrdinal++)
    {
      double maxConservationFailure = 0.0;
      for (auto cellID : myCellIDs)
      {
//        cout << "timeDifferences function: " << timeDifferences[conservedOrdinal]->displayString() << endl;
        double timeDifferenceIntegral = timeDifferences[conservedOrdinal]->integrate(cellID, mesh);
        double fluxIntegral = conservationFluxes[conservedOrdinal]->integrate(cellID, mesh);
        double cellIntegral = timeDifferenceIntegral + fluxIntegral;
        maxConservationFailure = std::max(abs(cellIntegral), maxConservationFailure);
        cellOrdinal++;
      }
      maxConservationFailures[conservedOrdinal] = maxConservationFailure;
    }
    vector<double> globalMaxConservationFailures(numConserved);
    mesh->Comm()->MaxAll(&maxConservationFailures[0], &globalMaxConservationFailures[0], numConserved);
    if (rank == 0) {
      cout << "Max cellwise (rho,m,E) conservation failures: ";
      for (double failure : globalMaxConservationFailures)
      {
        cout << failure << "\t";
      }
      cout << endl;
    }
//    for (int cellOrdinal=0; cellOrdinal<myCellCount; cellOrdinal++)
//    {
//      cout << "cell ordinal " << cellOrdinal << ", conservation failure: " << cellIntegrals[cellOrdinal] << endl;
//    }
  };
  
  printConservationReport();
  double t = 0;
  int timeStepNumber = 0;
  
  // hard-coded h value for now -- assumes the domain is 3.0 wide
  double h = 3.0 / horizontalElements;
  // account for higher-order by dividing h by p to get the effective grid spacing
  h /= polyOrder;
  
  // for the moment, time step adjustment only reduces time step size
  auto adjustTimeStep = [&]() -> void
  {
    if (spaceTime) return; // no time step to adjust...
    // Check that dt is reasonable vis-a-vis CFL
    FunctionPtr soundSpeed = Function::sqrtFunction(gamma * p_soln / rho_soln);
    FunctionPtr fluidSpeed = velocityMagnitude_soln;
    double maxSoundSpeed = soundSpeed->linfinitynorm(mesh);
    double maxFluidSpeed = fluidSpeed->linfinitynorm(mesh);
    double dtCFL = h / (maxSoundSpeed + maxFluidSpeed);
    if (dt > dtCFL)
    {
      if (rank == 0)
      {
        cout << "Time step " << dt << " exceeds CFL-stable value of " << dtCFL << endl;
      }
      while (dt > dtCFL)
      {
        dt /= 2.0;
      }
      numTimeSteps = (finalTime - t) / dt + timeStepNumber;
      if (rank == 0)
      {
        cout << "Set time step to " << dt;
        cout << "; set numTimeSteps to " << numTimeSteps << endl;
      }
    }
    else
    {
      cout << "Time step " << dt << " is below CFL-stable value of " << dtCFL << endl;
    }
  };
  
  adjustTimeStep();
  
  for (timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    int maxTimeStepReductions = 20;
    
    int timeStepReductions = 0;
    
    double savedDt = dt;
    
    bool timeStepSucceeded = false;
    
    while (timeStepReductions < maxTimeStepReductions)
    {
      int maxNonlinearSteps = 10;
      bool printNewtonInfo = (rank == 0);
      bool newtonSuccess = form->newtonSolve(nonlinearTolerance, maxNonlinearSteps, printNewtonInfo);
      
      if (! newtonSuccess)
      {
        if (enableAdaptivity)
        {
          // then we haven't been reducing time step; we can do so here
          if (timeStepReductions != maxTimeStepReductions-1)
          {
            dt /= 2.0;
            form->setTimeStep(dt);
            if (rank == 0)
            {
              cout << "Newton step failed to converge.  Restarting time step with dt = " << dt << endl;
            }
            // reset "background flow" to the previous time step's solution.
            form->solution()->setSolution(form->solutionPreviousTimeStep());
            continue; // go to top of while loop, to try the newton solve again.
          }
          else
          {
            double inadmissibleTime = finalTime * 2;
            if (rank == 0)
            {
              cout << "Newton step failed to converge, and time step reductions have reached maximum.  Outputting unconverged solution at t = " << inadmissibleTime << endl;
            }
            // we're about to finish the while loop with failure -- let's make the exporter output the inadmissible solution, so we can look at it
            solutionExporter.exportSolution(form->solution(),functionsToPlot,functionNames,inadmissibleTime); // similarly, since the entropy compares current and previous, need this to happen before setSolution()
            return -1;
          }
        }
        else
        {
          if (rank == 0)
          {
            cout << "Nonlinear iteration failed to converge.  Exiting...\n";
          }
          return -1;
        }
      }
      
      bool printOnFailure = (rank == 0);
      double minDistanceFromZero = 0.0;
      bool solutionIsAdmissible = form->solutionIsAdmissible(minDistanceFromZero, printOnFailure);
      if (!solutionIsAdmissible)
      {
        if (enableAdaptivity)
        {
          if (rank == 0)
          {
            cout << "Newton step converged to an inadmissible state.  Restarting time step after adapting mesh.\n";
          }
          
          RefinementStrategyPtr refStrategy = RefinementStrategy::energyErrorRefinementStrategy(form->solutionIncrement(), 0.2);
          refStrategy->refine();
          
          form->solution()->setSolution(form->solutionPreviousTimeStep());
        }
        else if (timeStepReductions != maxTimeStepReductions-1)
        {
          dt /= 2.0;
          form->setTimeStep(dt);
          if (rank == 0)
          {
            cout << "Newton step converged to an inadmissible state.  Restarting time step with dt = " << dt << endl;
          }
          // reset "background flow" to the previous time step's solution.
          form->solution()->setSolution(form->solutionPreviousTimeStep());
        }
        else
        {
          double inadmissibleTime = finalTime * 2;
          if (rank == 0)
          {
            cout << "Time step failed; outputting inadmissible solution at t = " << inadmissibleTime << endl;
          }
          // we're about to finish the while loop with failure -- let's make the exporter output the inadmissible solution, so we can look at it
          solutionExporter.exportSolution(form->solution(),functionsToPlot,functionNames,inadmissibleTime); // similarly, since the entropy compares current and previous, need this to happen before setSolution()
        }
        
        timeStepReductions++;
      }
      else
      {
        timeStepSucceeded = true;
        break;
      }
    }
    
    if (!timeStepSucceeded)
    {
      if (rank == 0)
      {
        cout << "Did not find an admissible solution even after " << maxTimeStepReductions << " time step reductions; exiting.\n";
      }
      return -1;
    }
    
    t += dt;
    
    dt = savedDt;
    form->setTimeStep(dt);

    printLocalConservationReport(); // since this depends on the difference between current/previous solution, we need to call before we set prev to current.
    if (((timeStepNumber + 1) % outputFrequency == 0) || (timeStepNumber == numTimeSteps - 1))
    {
      solutionExporter.exportSolution(form->solution(),functionsToPlot,functionNames,t); // similarly, since the entropy compares current and previous, need this to happen before setSolution()
    }
    if (rank == 0) std::cout << "========== t = " << t << ", time step number " << timeStepNumber+1 << " ==========\n";
    printConservationReport();
    
    if (timeStepNumber == numTimeSteps - 2)
    {
      // one time step left: set dt to get to the appropriate final time
      double lastDt = finalTime - t;
      form->setTimeStep(lastDt);
    }
    
    if (timeStepNumber != numTimeSteps - 1)
    {
      form->solutionPreviousTimeStep()->setSolution(form->solution());
    }
  }
//
//  // output error in By:
//  double err_By_L2_final = err_By->l2norm(mesh);
//  double err_By_L1_final = err_By->l1norm(mesh);
//  double err_Bz_L2_final = err_Bz->l2norm(mesh);
//  double err_Bz_L1_final = err_Bz->l1norm(mesh);
//
//  if (rank == 0)
//  {
//    cout << "Initial L^2 error of By: " << err_By_L2_initial << endl;
//    cout << "Initial L^2 error of Bz: " << err_Bz_L2_initial << endl;
//    cout << endl;
//    cout << "Initial L^1 error of By: " << err_By_L1_initial << endl;
//    cout << "Initial L^1 error of Bz: " << err_Bz_L1_initial << endl;
//    cout << endl;
//    cout << endl;
//    cout << "Final L^2 error of By: " << err_By_L2_final << endl;
//    cout << "Final L^2 error of Bz: " << err_Bz_L2_final << endl;
//    cout << endl;
//    cout << "Final L^1 error of By: " << err_By_L1_final << endl;
//    cout << "Final L^1 error of Bz: " << err_Bz_L1_final << endl;
//  }
  
  return 0;
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv); // initialize MPI
  
  int horizontalElements = 102; // better to have it divisible by 3, to capture the initial data correctly
  int polyOrder = 2;
  int delta_k   = 2;
  bool useCondensedSolve = true;
  int spaceDim = 1;
  int cubatureEnrichment = 3 * polyOrder; // there are places in the strong, nonlinear equations where 4 variables multiplied together.  Therefore we need to add 3 variables' worth of quadrature to the simple test v. trial quadrature.
  double nonlinearTolerance    = 1e-2;
  bool useSpaceTime = false;
  int temporalPolyOrder =  1;
  int temporalMeshWidth = -1;  // if useSpaceTime gets set to true and temporalMeshWidth is left unset, we'll use meshWidth = (finalTime / dt / temporalPolyOrder)
  std::string linearization = "Newton";
  bool useConservationVariables = true;
  bool useLeftState = true; // use left state from Brio-Wu for initial conditions of thermodynamic variables (if false, will use the right state)
  double ByMagnitude = 1.0;
  bool enableAdaptivity = false;
  
  double dt    = 0.001; // time step
  double finalTime = 5.0;
  int outputFrequency = 50; // output once every 50 time steps
  
  bool enforceConservationUsingLagrangeMultipliers = false;
  
  std::map<string, TestNormChoice> normChoices = {
    {"steadyGraph", STEADY_GRAPH_NORM},
    {"transientGraph", TRANSIENT_GRAPH_NORM},
    {"experimental", EXPERIMENTAL_CONSERVATIVE_NORM}
  };
  
  std::string normChoiceString = "transientGraph";
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  cmdp.setOption("polyOrder", &polyOrder);
  cmdp.setOption("horizontalElements", &horizontalElements);
  cmdp.setOption("spaceDim",  &spaceDim);
  cmdp.setOption("dt", &dt);
  cmdp.setOption("finalTime", &finalTime);
  cmdp.setOption("outputFrequency", &outputFrequency);
  cmdp.setOption("deltaP", &delta_k);
  cmdp.setOption("linearization", &linearization);
  cmdp.setOption("nonlinearTol", &nonlinearTolerance);
  cmdp.setOption("enforceConservation", "dontEnforceConservation", &enforceConservationUsingLagrangeMultipliers);
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("spaceTime","backwardEuler", &useSpaceTime);
  cmdp.setOption("temporalPolyOrder", &temporalPolyOrder);
  cmdp.setOption("temporalMeshWidth", &temporalMeshWidth);
  cmdp.setOption("useConservationVariables", "usePrimitiveVariables", &useConservationVariables);
  cmdp.setOption("useLeftState", "useRightState", &useLeftState);
  cmdp.setOption("ByMagnitude", &ByMagnitude);
  cmdp.setOption("enableAdaptivity", "disableAdaptivity", &enableAdaptivity);
  
  cmdp.setOption("norm", &normChoiceString);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  TestNormChoice normChoice;
  if (normChoices.find(normChoiceString) != normChoices.end())
  {
    normChoice = normChoices[normChoiceString];
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported norm choice");
  }
  
  bool usePicard = (linearization == "Picard");
  
  MeshTopologyPtr meshTopo;
  if (spaceDim == 1)
  {
    double width  =  3.0;
    double x0     =  0.0;
    vector<PeriodicBCPtr> periodicBCs;
    periodicBCs.push_back(PeriodicBC::xIdentification(x0, width));
    meshTopo = MeshFactory::intervalMeshTopology(x0, x0 + width, horizontalElements, periodicBCs);
  }
  else if (spaceDim == 2)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "2D not supported");
  }
  else if (spaceDim == 3)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "3D not supported");
  }
  
  double gamma = 5.0 / 3.0;
  
  Teuchos::RCP<IdealMHDFormulation> form;
  if (useSpaceTime)
  {
    cout << "******************************************************************************************** \n";
    cout << "*** SPACE-TIME NOTE: to date, we haven't validated space-time for Proto Brio Wu problem. *** \n";
    cout << "******************************************************************************************** \n";
    double t0 = 0.0;
    double t1 = finalTime;
    if (temporalMeshWidth == -1)
    {
      temporalMeshWidth = (finalTime / dt / temporalPolyOrder);
    }
    auto spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(meshTopo, t0, t1, temporalMeshWidth);
    form = IdealMHDFormulation::spaceTimeFormulation(spaceDim, spaceTimeMeshTopo, polyOrder, temporalPolyOrder, delta_k, gamma);
  }
  else if (!usePicard && useConservationVariables)
  {
    form = IdealMHDFormulation::timeSteppingFormulation(spaceDim, meshTopo, polyOrder, delta_k, gamma);
  }
  else if (!useConservationVariables)
  {
    form = IdealMHDFormulation::timeSteppingPrimitiveVariableFormulation(spaceDim, meshTopo, polyOrder, delta_k, gamma);
  }
  else
  {
    form = IdealMHDFormulation::timeSteppingPicardFormulation(spaceDim, meshTopo, polyOrder, delta_k, gamma);
  }

  return runSolver(form, dt, finalTime, outputFrequency, horizontalElements, polyOrder, cubatureEnrichment, useCondensedSolve,
                   nonlinearTolerance, normChoice, enforceConservationUsingLagrangeMultipliers, useSpaceTime, useLeftState, ByMagnitude, enableAdaptivity);
}
