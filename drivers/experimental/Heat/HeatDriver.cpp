//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "ExpFunction.h"
#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "ParameterFunction.h"
#include "PoissonFormulation.h"
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
  NONCONFORMING_ULTRAWEAK,
  CONFORMING_ULTRAWEAK
};

enum NormChoice
{
  GRAPH_NORM,
  MANUAL_GRAPH_NORM
};

const double PI = 3.141592653589793238462;

void runSolve(vector<vector<double>> domainDim, int numElements, double dt, double finalTime, int polyOrder, int pToAdd, ProblemChoice problemChoice, FormulationChoice formulationChoice, NormChoice normChoice)
{
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
  
  bool conformingTraces;
  if (formulationChoice == CONFORMING_ULTRAWEAK)
  {
    conformingTraces = true;
    cout << "Using conforming traces.\n";
  }
  else
  {
    conformingTraces = false; // use L^2 traces even for traces of H^1
    cout << "Using non-conforming traces.\n";
  }
  
  PoissonFormulation form(spaceDim, conformingTraces, PoissonFormulation::ULTRAWEAK);
  VarPtr q = form.v();
  VarPtr u = form.u();
  VarPtr u_hat = form.u_hat();
  BFPtr bf = form.bf();
  
  // the heat equation is the same as Poisson, except that we add a term (1.0 / dt * u, v)
  bf->addTerm( u / dt, form.v() );
  
  int pToAddTest = spaceDim;
  double width = 1.0, height = 1.0;
  int horizontalElements = numElements, verticalElements = numElements;
  MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(width, height,
                                                           horizontalElements, verticalElements);
  
  map<int,int> trialOrderEnhancements;
  // enhance order of u_hat:
  trialOrderEnhancements[form.u_hat()->ID()] = 1;
  
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, bf->varFactory(), polyOrder+1, pToAddTest, trialOrderEnhancements) );
  mesh->setBilinearForm(bf);
  
  FunctionPtr u_exact; // defined for manufactured solution
  FunctionPtr u_bc;
  FunctionPtr forcing;
  
  // set up a parameter function; this will allow us to have an exact solution that is time-dependent
  ParameterFunctionPtr t = ParameterFunction::parameterFunction(0.0);
  
  if (problemChoice == SINUSOIDAL_MANUFACTURED)
  {
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    
    FunctionPtr t_squared = FunctionPtr(t) * FunctionPtr(t);
    
    FunctionPtr exp_minus_pi_t_squared = Exp<double>::exp(-PI * t_squared);
    
    u_exact = exp_minus_pi_t_squared * TrigFunctions<double>::sin(PI * x) * TrigFunctions<double>::cos(PI * y);
    
    u_bc = u_exact;
    forcing = u_exact->dx()->dx() + u_exact->dy()->dy();
  }
  else // UNIT_FORCING
  {
    u_bc = Function::zero();
    forcing = Function::constant(1.0);
  }
  
  RHSPtr rhs = RHS::rhs();
  
  rhs->addTerm(forcing * q);
  
  IPPtr ip = IP::ip();
  if (normChoice == MANUAL_GRAPH_NORM)
  {
    // this norm avoids the addition of L^2 terms, thanks to the observation that dropping them, we still have a well-behaved norm
    cout << "Using manual graph norm.\n";
    
    VarPtr v = form.v();
    VarPtr tau = form.tau();
    ip->addTerm(tau->div() + v / dt);  // corresponds to u
    ip->addTerm(tau - v->grad());      // corresponds to sigma
  }
  else
  {
    cout << "Using graph norm.\n";
    ip = form.bf()->graphNorm();
  }
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u_hat, SpatialFilter::allSpace(), u_bc);
  
  ostringstream stiffnessPath;
  stiffnessPath << "/tmp/K_Heat_";
  for (int d=0; d<spaceDim; d++)
  {
    stiffnessPath << numElements;
    if (d<spaceDim-1)
      stiffnessPath << "x";
  }
  stiffnessPath << "_k" << polyOrder << "_";
  if (normChoice == MANUAL_GRAPH_NORM)
  {
    stiffnessPath << "manualGraphNorm";
  }
  else
  {
    stiffnessPath << "graphNorm";
  }
  stiffnessPath << ".dat";
  
  bool exportL2Projection = false; //
  // export the L^2 projection matrix on this mesh:
  if (exportL2Projection)
  {
    VarFactoryPtr vf_L2 = bf->varFactory()->getBubnovFactory(VarFactory::BUBNOV_TRIAL);
    BFPtr bf_L2 = BF::bf(vf_L2);
    bf_L2->addTerm(form.u(), form.u());
    bf_L2->addTerm(form.sigma(1),form.sigma(1));
    bf_L2->addTerm(form.sigma(2),form.sigma(2));
    bf_L2->addTerm(form.u_hat(), form.u_hat());
    bf_L2->addTerm(form.sigma_n_hat(), form.sigma_n_hat());
    
    ostringstream l2ProjectionPath;
    l2ProjectionPath << "/tmp/M_Heat_";
    for (int d=0; d<spaceDim; d++)
    {
      l2ProjectionPath << numElements;
      if (d<spaceDim-1)
        l2ProjectionPath << "x";
    }
    l2ProjectionPath << "_k" << polyOrder << "_";
    l2ProjectionPath << ".dat";
    
    MeshPtr mesh_L2 = Teuchos::rcp( new Mesh(meshTopo, vf_L2, polyOrder+1, 0, trialOrderEnhancements, trialOrderEnhancements) );
    mesh_L2->setBilinearForm(bf_L2);
    
    auto trialOrdering = mesh_L2->getElementType(0)->trialOrderPtr;
//    trialOrdering->print(cout);
    
    auto testOrdering = mesh_L2->getElementType(0)->testOrderPtr;
//    testOrdering->print(cout);
    
    SolutionPtr l2_solution = Solution::solution(bf_L2, mesh_L2, bc, rhs);
    l2_solution->setWriteMatrixToMatrixMarketFile(true, l2ProjectionPath.str());
    
    // we don't have a great way to just populate and export the matrix, so we just solve, and export on the way...
    l2_solution->solve();
  }
  
  SolutionPtr solution = Solution::solution(bf, mesh, bc, rhs, ip);
  
//  solution->setWriteMatrixToMatrixMarketFile(true, stiffnessPath.str());
  
  ostringstream vizPath;
  vizPath << "Heat_Solution_" << spaceDim << "D_";
  vizPath << "Galerkin";
  vizPath << "_p" << polyOrder;
  HDF5Exporter vizExporter(mesh, vizPath.str(), "/tmp");
  
  vizExporter.exportSolution(solution, t->getConstantValue());
  
  int numTimeSteps = int(finalTime / dt);
  
  for (int timeStepNumber = 0; timeStepNumber < numTimeSteps; timeStepNumber++)
  {
    Teuchos::Time timer("Heat solve timer", false);
    timer.start();
    solution->solve();
    double solveTime = timer.stop();
    
    if (rank == 0)
      cout << "Solved in " << solveTime << " seconds.\n";
    vizExporter.exportSolution(solution, t->getConstantValue());
  }
  
  if (rank == 0)
  {
    cout << "Exported solution to " << vizPath.str() << endl;
  }
  
  if (problemChoice == SINUSOIDAL_MANUFACTURED)
  {
    SolutionPtr bestApproximation = Solution::solution(bf, mesh, bc, rhs, ip);
    map<int, FunctionPtr> exactSolutionMap;
    exactSolutionMap[form.u()->ID()] = u_exact;
    exactSolutionMap[form.u_hat()->ID()] = u_exact;
    FunctionPtr sigma_exact = u_exact->grad();
    exactSolutionMap[form.sigma(1)->ID()] = sigma_exact->x();
    exactSolutionMap[form.sigma(2)->ID()] = sigma_exact->y();
    FunctionPtr n = Function::normal();
    FunctionPtr sgn = Function::sideParity();
    exactSolutionMap[form.sigma_n_hat()->ID()] = sigma_exact * n * sgn;
    bestApproximation->projectOntoMesh(exactSolutionMap);
    
    int cubatureEnrichment = 4;
    
    map<int,VarPtr> trialVars = bf->varFactory()->trialVars();
    
    double l2NormSquared = 0.0;
    double bestDifferenceL2Squared = 0.0;
    for (auto entry : trialVars)
    {
      auto var = entry.second;
      
      auto var_exact = exactSolutionMap[var->ID()];
      
      bool weightFluxesByParity = false; // here, interested in comparing to exact solutions; should not weight
      auto var_soln = Function::solution(var, solution, weightFluxesByParity);
      auto var_best = Function::solution(var, bestApproximation, weightFluxesByParity);
      
      double bestError = (var_best - var_exact)->l2norm(mesh, cubatureEnrichment);
      double actualError = (var_soln - var_exact)->l2norm(mesh, cubatureEnrichment);
      
      cout << var->name() << " best error: " << bestError << endl;
      cout << var->name() << " actual error: " << actualError << endl;
      
      double diffOfBest = (var_best - var_soln)->l2norm(mesh, cubatureEnrichment);
      cout << "L^2 norm of difference between solution and best approximation: " << diffOfBest << endl;
      
      bestDifferenceL2Squared += diffOfBest * diffOfBest;
      l2NormSquared += actualError * actualError;
    }
    
    double energyError = solution->energyErrorTotal();
    cout << "Energy error:     " << energyError << endl;
    cout << "L^2 error:        " << sqrt(l2NormSquared) << endl;
    cout << "L^2 diff of best: " << sqrt(bestDifferenceL2Squared) << endl;
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int spaceDim = 2;
  int numElements = 30;
  vector<vector<double>> domainDim(spaceDim,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
  int polyOrder = 3;
  int pToAdd = spaceDim;
  string normChoiceString = "Trace"; // "Graph" is the alternative
  string formulationChoiceString = "Nonconforming"; // "Conforming" the alternative
  string problemChoiceString = "Manufactured"; // "Unit Forcing" the alternative
  double dt = -1.0; // if unset, will use Führer and Heuer's default of ___
  double finalTime = 0.1; // Follows Führer and Heuer
  
  bool runAll = false;
  
  cmdp.setOption("norm", &normChoiceString);
  cmdp.setOption("formulation", &formulationChoiceString);
  cmdp.setOption("problem", &problemChoiceString);
  cmdp.setOption("meshWidth", &numElements );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("deltaP", &pToAdd);
  cmdp.setOption("dt", &dt);
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
  
  NormChoice normChoice;
  if (normChoiceString == "ManualGraph")
  {
    normChoice = MANUAL_GRAPH_NORM;
  }
  else if (normChoiceString == "Graph")
  {
    normChoice = GRAPH_NORM;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized norm choice");
  }
  
  FormulationChoice formulationChoice;
  if (formulationChoiceString == "Nonconforming")
  {
    formulationChoice = NONCONFORMING_ULTRAWEAK;
  }
  else if (formulationChoiceString == "Conforming")
  {
    formulationChoice = CONFORMING_ULTRAWEAK;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized formulation choice");
  }
  
  ProblemChoice problemChoice;
  if (problemChoiceString == "Manufactured")
  {
    problemChoice = SINUSOIDAL_MANUFACTURED;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized problem choice");
  }

  if (!runAll)
  {
    runSolve(domainDim, numElements, polyOrder, pToAdd, dt, finalTime, problemChoice, formulationChoice, normChoice);
  }
  else
  {
    vector<int> elementCounts = {1,2,4,8,16,32};
    vector<NormChoice> normChoices = {GRAPH_NORM, MANUAL_GRAPH_NORM};
    for (int numElements : elementCounts)
    {
      cout << "*************** mesh width " << numElements << " *******************\n";
      for (NormChoice normChoice : normChoices)
      {
        runSolve(domainDim, numElements, polyOrder, pToAdd, dt, finalTime, problemChoice, formulationChoice, normChoice);
      }
    }
  }
  
  return 0;
}
