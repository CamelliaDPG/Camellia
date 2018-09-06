//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"
#include "TrigFunctions.h"

using namespace Camellia;
using namespace std;

enum ProblemChoice
{
  UNIT_FORCING,
  SINUSOIDAL_MANUFACTURED
};

enum FormulationChoice
{
  NONCONFORMING_ULTRAWEAK,
  CONFORMING_ULTRAWEAK
};

enum NormChoice
{
  GRAPH_NORM,
  TRACE_NORM
};

void runSolve(vector<vector<double>> domainDim, int numElements, int polyOrder, ProblemChoice problemChoice, FormulationChoice formulationChoice, NormChoice normChoice)
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
  
  int pToAddTest = spaceDim;
  double width = 1.0, height = 1.0;
  int horizontalElements = numElements, verticalElements = numElements;
  MeshTopologyPtr meshTopo = MeshFactory::quadMeshTopology(width, height,
                                                           horizontalElements, verticalElements);
  /*
   Mesh(MeshTopologyViewPtr meshTopology, VarFactoryPtr varFactory, int H1Order, int pToAddTest,
   map<int,int> trialOrderEnhancements=_emptyIntIntMap, map<int,int> testOrderEnhancements=_emptyIntIntMap,
   MeshPartitionPolicyPtr meshPartitionPolicy = Teuchos::null, Epetra_CommPtr Comm = Teuchos::null);
   */
  
  map<int,int> trialOrderEnhancements;
  // enhance order of u_hat:
  trialOrderEnhancements[form.u_hat()->ID()] = 1;
  MeshPtr mesh = Teuchos::rcp( new Mesh(meshTopo, bf->varFactory(), polyOrder+1, pToAddTest, trialOrderEnhancements) );
  mesh->setBilinearForm(bf);
  
  FunctionPtr u_exact; // defined for manufactured solution
  FunctionPtr u_bc;
  FunctionPtr forcing;
  
  if (problemChoice == SINUSOIDAL_MANUFACTURED)
  {
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    
    u_exact = TrigFunctions<double>::sin(x) * TrigFunctions<double>::cos(y);
    
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
  if (normChoice == TRACE_NORM)
  {
    cout << "Using trace norm.\n";
    // construct ip that takes traces on the mesh skeleton, but is otherwise like the standard graph norm
    VarPtr v = form.v();
    VarPtr tau = form.tau();
    ip->addTerm(tau->div());      // corresponds to u
    ip->addTerm(tau - v->grad()); // corresponds to sigma
    
    auto trace = Function::meshSkeletonCharacteristic(); // skeleton restriction
    
    bool hWeightTraceTerms = false;
    if (hWeightTraceTerms)
    {
      auto h = Function::h();
      auto traceWeight = Function::sqrtFunction(h); // volume is h^2; trace of element is 4 h
      trace = trace * traceWeight;
    }
    ip->addTerm(tau * trace);
    // trying something (this seems to give lower L^2 error)
    // ip->addTerm(traceWeight * tau->dot_normal());

    ip->addTerm(v * trace);
  }
  else
  {
    cout << "Using graph norm.\n";
    ip = form.bf()->graphNorm();
  }
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u_hat, SpatialFilter::allSpace(), u_bc);
  
  ostringstream stiffnessPath;
  stiffnessPath << "/tmp/K_Poisson_";
  for (int d=0; d<spaceDim; d++)
  {
    stiffnessPath << numElements;
    if (d<spaceDim-1)
      stiffnessPath << "x";
  }
  stiffnessPath << "_k" << polyOrder << "_";
  if (normChoice == TRACE_NORM)
  {
    stiffnessPath << "traceNorm";
  }
  else
  {
    stiffnessPath << "graphNorm";
  }
  stiffnessPath << ".dat";
  
  bool exportL2Projection = false; //
  // export the L^2 projection matrix on this mesh:
  {
    VarFactoryPtr vf_L2 = bf->varFactory()->getBubnovFactory(VarFactory::BUBNOV_TRIAL);
    BFPtr bf_L2 = BF::bf(vf_L2);
    bf_L2->addTerm(form.u(), form.u());
    bf_L2->addTerm(form.sigma(),form.sigma());
    bf_L2->addTerm(form.u_hat(), form.u_hat());
    bf_L2->addTerm(form.sigma_n_hat(), form.sigma_n_hat());
    
    ostringstream l2ProjectionPath;
    l2ProjectionPath << "/tmp/M_Poisson_";
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
  
  solution->setWriteMatrixToMatrixMarketFile(true, stiffnessPath.str());
  
  Epetra_Time timer(*mesh->Comm());
  solution->solve();
  double solveTime = timer.ElapsedTime();
  
  if (rank == 0)
    cout << "Solved in " << solveTime << " seconds.\n";
  
  ostringstream vizPath;
  vizPath << "Poisson_Solution_" << spaceDim << "D_";
  vizPath << "Galerkin";
  vizPath << "_p" << polyOrder;
  HDF5Exporter vizExporter(mesh, vizPath.str(), "/tmp");
  vizExporter.exportSolution(solution);
  
  if (rank == 0)
  {
    cout << "Exported solution to " << vizPath.str() << endl;
  }
  
  if (problemChoice == SINUSOIDAL_MANUFACTURED)
  {
    SolutionPtr bestApproximation = Solution::solution(bf, mesh, bc, rhs, ip);
    int solutionOrdinal = 0;
    map<int, FunctionPtr> exactSolutionMap;
    exactSolutionMap[form.u()->ID()] = u_exact;
    exactSolutionMap[form.u_hat()->ID()] = u_exact;
    FunctionPtr sigma_exact = u_exact->grad();
    exactSolutionMap[form.sigma()->ID()] = sigma_exact;
    FunctionPtr n = Function::normal();
    FunctionPtr sgn = Function::sideParity();
    exactSolutionMap[form.sigma_n_hat()->ID()] = sigma_exact * n * sgn;
    bestApproximation->projectOntoMesh(exactSolutionMap, solutionOrdinal);
    
    int cubatureEnrichment = 4;
    
    map<int,VarPtr> trialVars = bf->varFactory()->trialVars();
    
    double l2NormSquared = 0.0;
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
      
      l2NormSquared += actualError;
    }
    
    double energyError = solution->energyErrorTotal();
    cout << "Energy error: " << energyError << endl;
    
    cout << "L^2 error: " << sqrt(l2NormSquared) << endl;
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
  string normChoiceString = "Trace"; // "Graph" is the alternative
  string formulationChoiceString = "Nonconforming"; // "Conforming" the alternative
  string problemChoiceString = "Manufactured"; // "Unit Forcing" the alternative
  
  bool runAll = false;
  
  cmdp.setOption("norm", &normChoiceString);
  cmdp.setOption("formulation", &formulationChoiceString);
  cmdp.setOption("problem", &problemChoiceString);
  cmdp.setOption("meshWidth", &numElements );
  cmdp.setOption("polyOrder", &polyOrder );
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
  if (normChoiceString == "Trace")
  {
    normChoice = TRACE_NORM;
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
  if (problemChoiceString == "Unit Forcing")
  {
    problemChoice = UNIT_FORCING;
  }
  else if (problemChoiceString == "Manufactured")
  {
    problemChoice = SINUSOIDAL_MANUFACTURED;
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized problem choice");
  }

  if (!runAll)
  {
    runSolve(domainDim, numElements, polyOrder, problemChoice, formulationChoice, normChoice);
  }
  else
  {
    vector<int> elementCounts = {1,2,4,8,16,32};
    vector<NormChoice> normChoices = {GRAPH_NORM, TRACE_NORM};
    for (int numElements : elementCounts)
    {
      cout << "*************** mesh width " << numElements << " *******************\n";
      for (NormChoice normChoice : normChoices)
      {
        runSolve(domainDim, numElements, polyOrder, problemChoice, formulationChoice, normChoice);
      }
    }
  }
  
  return 0;
}
