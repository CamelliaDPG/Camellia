//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "CamelliaDebugUtility.h"
#include "ExpFunction.h"
#include "MeshFactory.h"
#include "RHS.h"
#include "StokesVGPFormulation.h"
#include "SuperLUDistSolver.h"
#include "TimeLogger.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"

#include <Teuchos_GlobalMPISession.hpp>

using namespace Camellia;
using namespace std;

enum ProblemChoice
{
  Stokes
};

void initializeSolutionAndCoarseMesh(SolutionPtr &solution, IPPtr &testNorm, ProblemChoice problemChoice,
                                     int spaceDim, bool conformingTraces,
                                     bool useStaticCondensation, int numCells, int k, int delta_k, string normChoice)
{
  BFPtr bf;
  RHSPtr rhs;
  
  double width = 1.0;              // width in each dimension
  vector<double> x0(spaceDim,0.0); // origin is the default for x0
  
  vector<double> dimensions;
  vector<int> elementCounts;
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(width);
    elementCounts.push_back(numCells);
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
  
  if (problemChoice == Stokes)
  {
    double mu = 1.0;
    
    StokesVGPFormulation formulation = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);
    
    bf = formulation.bf();
    testNorm = bf->graphNorm();
    
    rhs = RHS::rhs();
    
    FunctionPtr cos_y = Teuchos::rcp( new Cos_y );
    FunctionPtr sin_y = Teuchos::rcp( new Sin_y );
    FunctionPtr exp_x = Teuchos::rcp( new Exp_x );
    FunctionPtr exp_z = Teuchos::rcp( new Exp_z );
    
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    
    FunctionPtr u1_exact, u2_exact, u3_exact, p_exact;
    
    if (spaceDim == 2)
    {
      // this one was in the Cockburn Kanschat LDG Stokes paper
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y;
      p_exact = 2.0 * exp_x * sin_y;
    }
    else
    {
      // this one is inspired by the 2D one
      u1_exact = - exp_x * ( y * cos_y + sin_y );
      u2_exact = exp_x * y * sin_y + exp_z * y * cos_y;
      u3_exact = - exp_z * (cos_y - y * sin_y);
      p_exact = 2.0 * exp_x * sin_y + 2.0 * exp_z * cos_y;
    }
    
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    
    vector<FunctionPtr> uVector = (spaceDim==2) ? vector<FunctionPtr>{u1_exact,u2_exact} : vector<FunctionPtr>{u1_exact,u2_exact,u3_exact};
    FunctionPtr u_exact = Function::vectorize(uVector);
    
    FunctionPtr forcingFunction = formulation.forcingFunction(u_exact, p_exact);
    
    rhs = formulation.rhs(forcingFunction);
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unsupported problem choice");
  }
  
  if (testNorm == Teuchos::null)
  {
    if (normChoice == "Graph")
    {
      testNorm = bf->graphNorm();
    }
    else
    {
      cout << "Norm choice " << normChoice << " is not supported for this formulation.\n";
    }
  }
  
  int H1Order = k + 1;
  MeshPtr mesh = Teuchos::rcp(new Mesh(meshTopo, bf, H1Order, delta_k));
  
  if (solution == Teuchos::null)
  {
    BCPtr emptyBC = BC::bc();
    solution = Solution::solution(mesh, emptyBC, rhs, testNorm);
  }
  solution->setUseCondensedSolve(useStaticCondensation);
}

void printTimings()
{
  Epetra_CommPtr Comm = MPIWrapper::CommWorld();

  int rank = Comm->MyPID();
  
  map<string,double> timings = TimeLogger::sharedInstance()->totalTimes();
  // let's take the maximum timings -- but first let's just do a sanity check that all
  // ranks agree on the number of timing entries.  (This is true today, but depending on
  // what timings are added going forward, it might not be true in the future.)
  
  int numTimings = timings.size();
  int maxNumTimings, minNumTimings;
  Comm->MaxAll(&numTimings, &maxNumTimings, 1);
  Comm->MinAll(&numTimings, &minNumTimings, 1);
  
  if (maxNumTimings != minNumTimings)
  {
    if (rank == 0)
    {
      cout << "WARNING: Timings lists do not agree on all ranks; instead of taking maximums, just printing rank 0's timings:\n";
      Camellia::print<string,double>("selected timings on rank 0", timings);
    }
  }
  else
  {
    // otherwise, we take it the lists are the same.  Might be better later to identify which timings are actually of interest to us,
    // and identify them by name...
    vector<double> timingValues;
    for (auto entry : timings)
    {
      timingValues.push_back(entry.second);
    }
    vector<double> maxTimingValues(numTimings);
    Comm->MaxAll(&timingValues[0], &maxTimingValues[0], numTimings);
    if (rank == 0)
    {
      cout << "************************************************\n";
      cout << " Selected timings, max values across all ranks:\n";
      
      cout << setprecision(2);
      cout << std::scientific;
      
      // report items of special interest for tracking optimizations we're currently working on:
      std::set<std::string> specialInterestTimingStrings = {BF_FACTORED_CHOLESKY_SOLVE_TIMER_STRING, BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING,
        LINEAR_TERM_INTEGRATION_TIMER_STRING, SOLUTION_INTERPRET_LOCAL_DATA_TIMER_STRING, SOLUTION_LOCAL_STIFFNESS_LOAD_TIMER_STRING};
      cout << "************************************************\n";
      cout << " Timings of interest, max values across all ranks:\n";
      int i=0;
      for (auto entry : timings)
      {
        if (specialInterestTimingStrings.find(entry.first) != specialInterestTimingStrings.end())
        {
          cout.setf(std::ios::left, std::ios::adjustfield);
          cout << std::setw(40) << entry.first;
          cout.setf(std::ios::right, std::ios::adjustfield);
          cout << std::setw(8);
          cout << maxTimingValues[i] << " sec." << endl;
        }
        i++;
      }
      cout << "************************************************\n";
    }
  }
}

struct TimingResult
{
  int k_trial;
  int k_test;
  int trialDofCount;
  int testDofCount;
  double optimalTestSolveTime;              // time spent in factored cholesky calls
  double linearTermIntegrationTime;         // time spent in LinearTerm::integrate() methods
  double bfAssemblyOverheadTime;            // beyond optimalTestSolveTime + linearTermIntegrationTime
  double solutionLocalAssemblyOverheadTime; // beyond bf assembly + interpretLocalData
  double solutionInterpetLocalDataTime;     // time spent in the interpretLocalData() loops in Solution
  double totalAssemblyTime;                 // sum of all of the above
};

// result is written to on rank 0 only
void getTimingResult(TimingResult &result)
{
  Epetra_CommPtr Comm = MPIWrapper::CommWorld();
  
  int rank = Comm->MyPID();
  
  map<string,double> timings = TimeLogger::sharedInstance()->totalTimes();
  // let's take the maximum timings -- but first let's just do a sanity check that all
  // ranks agree on the number of timing entries.  (This is true today, but depending on
  // what timings are added going forward, it might not be true in the future.)
  
  int numTimings = timings.size();
  int maxNumTimings, minNumTimings;
  Comm->MaxAll(&numTimings, &maxNumTimings, 1);
  Comm->MinAll(&numTimings, &minNumTimings, 1);
  
  if (maxNumTimings != minNumTimings)
  {
    if (rank == 0)
    {
      cout << "WARNING: Timings lists do not agree on all ranks; instead of taking maximums, just printing rank 0's timings:\n";
      Camellia::print<string,double>("selected timings on rank 0", timings);
    }
  }
  else
  {
    vector<double> timingValues;
    for (auto entry : timings)
    {
      timingValues.push_back(entry.second);
    }
    vector<double> maxTimingValues(numTimings);
    Comm->MaxAll(&timingValues[0], &maxTimingValues[0], numTimings);
    if (rank == 0)
    {
      // report items of special interest for tracking optimizations we're currently working on:
      std::set<std::string> specialInterestTimingStrings = {BF_FACTORED_CHOLESKY_SOLVE_TIMER_STRING, BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING,
        LINEAR_TERM_INTEGRATION_TIMER_STRING, SOLUTION_INTERPRET_LOCAL_DATA_TIMER_STRING, SOLUTION_LOCAL_STIFFNESS_LOAD_TIMER_STRING};
      int i=0;
      std::map<std::string, double> specialInterestTimings;
      for (auto entry : timings)
      {
        if (specialInterestTimingStrings.find(entry.first) != specialInterestTimingStrings.end())
        {
          specialInterestTimings[entry.first] = maxTimingValues[i];
        }
        i++;
      }
      double bfAssemblyTime                    = specialInterestTimings[BF_LOCAL_STIFFNESS_AND_RHS_TIMER_STRING];
      double bfFactoredCholeskyTime            = specialInterestTimings[BF_FACTORED_CHOLESKY_SOLVE_TIMER_STRING];
      double linearTermIntegrationTime         = specialInterestTimings[LINEAR_TERM_INTEGRATION_TIMER_STRING];
      double solnInterpretLocalDataTime        = specialInterestTimings[SOLUTION_INTERPRET_LOCAL_DATA_TIMER_STRING];
      double solnLocalStiffnessAndLoadTime     = specialInterestTimings[SOLUTION_LOCAL_STIFFNESS_LOAD_TIMER_STRING];
      result.bfAssemblyOverheadTime            = bfAssemblyTime - bfFactoredCholeskyTime - linearTermIntegrationTime;
      result.solutionInterpetLocalDataTime     = solnInterpretLocalDataTime;
      result.solutionLocalAssemblyOverheadTime = solnLocalStiffnessAndLoadTime - solnInterpretLocalDataTime - bfAssemblyTime;
      result.optimalTestSolveTime              = bfFactoredCholeskyTime;
      result.linearTermIntegrationTime         = linearTermIntegrationTime;
      
      result.totalAssemblyTime                 = solnLocalStiffnessAndLoadTime;
    }
  }
}

void printTimingResult(const TimingResult &timingResult)
{
  Epetra_CommPtr Comm = MPIWrapper::CommWorld();
  int rank = Comm->MyPID();
  
  if (rank == 0)
  {
    std::vector< std::pair<std::string, int> >    intResults;
    std::vector< std::pair<std::string, double> > timingResults;
    
    intResults.push_back({"Trial poly order", timingResult.k_trial});
    intResults.push_back({"Test poly order", timingResult.k_test});
    intResults.push_back({"Trial dof count", timingResult.trialDofCount});
    intResults.push_back({"Test dof count", timingResult.testDofCount});
    
    timingResults.push_back({"Total Local Assembly Time", timingResult.totalAssemblyTime});
    timingResults.push_back({" - BF Optimal Test Solve",  timingResult.optimalTestSolveTime});
    timingResults.push_back({" - LinearTerm integration", timingResult.linearTermIntegrationTime});
    timingResults.push_back({" - BF Assembly Overhead", timingResult.bfAssemblyOverheadTime});
    timingResults.push_back({" - Solution interpretLocalData()", timingResult.solutionInterpetLocalDataTime});
    timingResults.push_back({" - Solution Local Assembly Overhead", timingResult.solutionLocalAssemblyOverheadTime});
    
    cout << setprecision(2);
    cout << std::scientific;
    
    for (auto intResult : intResults)
    {
      cout.setf(std::ios::left, std::ios::adjustfield);
      cout << std::setw(40) << intResult.first;
      cout.setf(std::ios::right, std::ios::adjustfield);
      cout << std::setw(8);
      cout << intResult.second << endl;
    }
    
    for (auto timeResult : timingResults)
    {
      cout.setf(std::ios::left, std::ios::adjustfield);
      cout << std::setw(40) << timeResult.first;
      cout.setf(std::ios::right, std::ios::adjustfield);
      cout << std::setw(8);
      cout << timeResult.second << " sec." << endl;
    }
  }
}

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  
  Epetra_CommPtr Comm = MPIWrapper::CommWorld();
  int rank = Comm->MyPID();
  int numProcs = Comm->NumProc();

  Comm->Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  int k = 1; // poly order for field variables
  int delta_k = -1;   // test space enrichment

  bool conformingTraces = false;

  int numCells = -1;
  int spaceDim = 3;
  bool useCondensedSolve = true;

  bool reportTimings = false;
  
  bool useFactoredCholeskyForOptimalTests = true;
  
  bool pauseOnRankZero = false;

  string problemChoiceString = "Stokes";
  string testNormChoice = "Graph";

  cmdp.setOption("problem",&problemChoiceString,"problem choice: Stokes");
  cmdp.setOption("testNorm",&testNormChoice,"test norm: Graph, or for ConvectionDiffusion: Robust or CoupledRobust");

  cmdp.setOption("numCells",&numCells,"mesh width");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment (use -1 for spaceDim, -2 for polyOrder)");
  
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("useFactoredCholesky", "useStandardCholesky", &useFactoredCholeskyForOptimalTests, "Use factored cholesky for optimal test solve");
  
  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (1, 2, or 3)");

  cmdp.setOption("pause","dontPause",&pauseOnRankZero, "pause (to allow attachment by tracer, e.g.), waiting for user to press a key");
  cmdp.setOption("reportTimings", "dontReportTimings", &reportTimings, "Report timings in Solution");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  if (pauseOnRankZero)
  {
    if (rank==0)
    {
      cout << "Press Enter to continue.\n";
      cin.get();
    }
    Comm->Barrier();
  }

  ProblemChoice problemChoice;

  if (problemChoiceString == "Stokes")
  {
    problemChoice = Stokes;
  }
  else
  {
    if (rank==0) cout << "Problem choice not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  if (rank==0)
  {
    cout << "Timing Local Assembly for " << spaceDim << "D " << problemChoiceString << " problem with k = " << k << " on " << numProcs << " MPI ranks.  Initializing meshes...\n";
  }
  
  auto getDeltaK = [&delta_k, &spaceDim] (int polyOrder) -> int
  {
    if (delta_k == -1)
    {
      return spaceDim;
    }
    else if (delta_k == -2)
    {
      return polyOrder;
    }
    else
    {
      return delta_k;
    }
  };
  
  TimingResult timingResult;
  timingResult.k_trial = k;
  timingResult.k_test  = k + getDeltaK(k) + 1;
  
  if (numCells == -1)
  {
    numCells = (int)ceil(pow(numProcs,1.0/(double)spaceDim));
  }
  
  Epetra_Time timer(*Comm);
  
  SolutionPtr solution;
  MeshPtr coarseMesh;
  IPPtr ip;
  
  initializeSolutionAndCoarseMesh(solution, ip, problemChoice, spaceDim, conformingTraces,
                                  useCondensedSolve, numCells, k, getDeltaK(k), testNormChoice);
  
  auto mesh = solution->mesh();
  
  int numDofs      = mesh->numGlobalDofs();
  int numTraceDofs = mesh->numFluxDofs();
  int numElements  = mesh->numActiveElements();
  
  BFPtr bf = solution->mesh()->bilinearForm();

  GlobalIndexType sampleCellID = 0;
  int totalTrialDofs, totalTestDofs;
  int rankWithSampleCell = solution->mesh()->Comm()->NumProc();
  if (solution->mesh()->getTopology()->isValidCellIndex(sampleCellID))
  {
    rankWithSampleCell = solution->mesh()->Comm()->MyPID();
    ElementTypePtr sampleElementType = solution->mesh()->getElementType(sampleCellID);
    totalTrialDofs = sampleElementType->trialOrderPtr->totalDofs();
    totalTestDofs = sampleElementType->testOrderPtr->totalDofs();
  }
  
  
  Comm = solution->mesh()->Comm();
  
  int minRankWithSampleCell;
  Comm->MinAll(&rankWithSampleCell, &minRankWithSampleCell, 1);
  Comm->Broadcast(&totalTestDofs, 1, minRankWithSampleCell);
  Comm->Broadcast(&totalTrialDofs, 1, minRankWithSampleCell);
  
  timingResult.trialDofCount = totalTrialDofs;
  timingResult.testDofCount  = totalTestDofs;
  
  double cellHaloTime = solution->mesh()->getTopology()->totalTimeComputingCellHalos();
  double maxCellHaloTime;
  solution->mesh()->Comm()->MaxAll(&cellHaloTime, &maxCellHaloTime, 1);
  
  if (rank==0)
  {
    cout << setprecision(2);
    cout << "Mesh has " << numDofs;
    cout << " global degrees of freedom (" << numTraceDofs << " trace dofs) on " << numElements << " elements.\n";
    cout << totalTrialDofs << " trial dofs per element; " << totalTestDofs << " test dofs.\n";
    
    cout << "Maximum time spent determining cell halos: " << maxCellHaloTime << " seconds.\n";
  }
  
    if (useFactoredCholeskyForOptimalTests)
    {
      if (rank==0) cout << "Setting optimal test solve to *factored* Cholesky\n";
      solution->mesh()->bilinearForm()->setOptimalTestSolver(TBF<>::FACTORED_CHOLESKY);
    }
    else
    {
      if (rank==0) cout << "Setting optimal test solve to standard Cholesky\n";
      solution->mesh()->bilinearForm()->setOptimalTestSolver(TBF<>::CHOLESKY);
    }

  solution->initializeStiffnessAndLoad();
  double populateStiffnessAndLoadTime = 0.0;
  timer.ResetStartTime();
  solution->populateStiffnessAndLoad();
  populateStiffnessAndLoadTime = timer.ElapsedTime();
  
  getTimingResult(timingResult);
  printTimingResult(timingResult);
  
  return 0;
}
