//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "CamelliaDebugUtility.h"
#include "CondensedDofInterpreter.h"
#include "ConvectionDiffusionFormulation.h"
#include "ExpFunction.h"
#include "E_to_T.hpp"
#include "GMGOperator.h"
#include "GMGSolver.h"
#include "GnuPlotUtil.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "NavierStokesVGPFormulation.h"
#include "PoissonFormulation.h"
#include "RefinementStrategy.h"
#include "Solver.h"
#include "StokesVGPFormulation.h"
#include "SuperLUDistSolver.h"
#include "TimeLogger.h"
#include "TrigFunctions.h"
#include "TypeDefs.h"

#include "Amesos_config.h"

#include "Epetra_Operator_to_Epetra_Matrix.h"
#include "EpetraExt_MatrixMatrix.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"

#include "Ifpack_AdditiveSchwarz.h"
#include "Ifpack_Amesos.h"
#include "Ifpack_IC.h"
#include "Ifpack_ILU.h"

#include <Teuchos_GlobalMPISession.hpp>
#include <MatrixMarket_Tpetra.hpp>

using namespace Camellia;
using namespace std;

enum ProblemChoice
{
  Poisson,
  ConvectionDiffusion,
  Stokes,
  NavierStokes
};

void initializeSolutionAndCoarseMesh(SolutionPtr &solution, vector<MeshPtr> &meshesCoarseToFine, IPPtr &testNorm, ProblemChoice problemChoice,
                                     int spaceDim, bool conformingTraces, bool enhanceFieldsForH1TracesWhenConforming,
                                     bool useStaticCondensation, int numCells, int k, int delta_k,
                                     int k_coarse, int rootMeshNumCells, bool useZeroMeanConstraints, bool jumpToCoarsePolyOrder,
                                     bool setupMeshTopologyAndQuit, Teuchos::RCP<NavierStokesVGPFormulation> &navierStokesFormulation,
                                     string normChoice, double epsilonForConvectionDiffusion)
{
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif
  
  int rank = Comm.MyPID();
  
  BFPtr bf;
  BCPtr bc;
  RHSPtr rhs;
  MeshPtr mesh;
  
  double width = 1.0; // in each dimension
  vector<double> x0(spaceDim,0); // origin is the default
  
  if (problemChoice == NavierStokes)
  {
    // for Navier-Stokes we use different width and x0, to match the standard Kovasznay flow parameters
    TEUCHOS_TEST_FOR_EXCEPTION(spaceDim != 2, std::invalid_argument, "spaceDim != 2 not supported for Navier-Stokes (don't have a manufactured solution defined for this case)");
    width = 2.0;
    x0 = {-0.5,0.0};
  }
  
  vector<double> dimensions;
  vector<int> elementCounts;
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(width);
    elementCounts.push_back(rootMeshNumCells);
  }
  
  MeshTopologyPtr meshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
  VarPtr p; // pressure
  
  map<int,int> trialOrderEnhancements;
  
  if (problemChoice == Poisson)
  {
    PoissonFormulation formulation(spaceDim, conformingTraces);
    
    bf = formulation.bf();
    
    rhs = RHS::rhs();
    FunctionPtr f = Function::constant(1.0);
    
    VarPtr v = formulation.v();
    rhs->addTerm( f * v );
    
    bc = BC::bc();
    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    VarPtr u_hat = formulation.u_hat();
    bc->addDirichlet(u_hat, boundary, Function::zero());
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      trialOrderEnhancements[formulation.u()->ID()] = 1;
    }
  }
  else if (problemChoice == ConvectionDiffusion)
  {
    x0 = vector<double>(spaceDim,-1.0);

    FunctionPtr beta;
    FunctionPtr beta_x = Function::constant(1);
    FunctionPtr beta_y = Function::constant(2);
    FunctionPtr beta_z = Function::constant(3);
    if (spaceDim == 1)
      beta = beta_x;
    else if (spaceDim == 2)
      beta = Function::vectorize(beta_x, beta_y);
    else if (spaceDim == 3)
      beta = Function::vectorize(beta_x, beta_y, beta_z);
    ConvectionDiffusionFormulation formulation(spaceDim, conformingTraces, beta, epsilonForConvectionDiffusion);
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      trialOrderEnhancements[formulation.u()->ID()] = 1;
    }
    
    bf = formulation.bf();
    
    rhs = RHS::rhs();
    FunctionPtr f = Function::constant(1.0);
    
    VarPtr v = formulation.v();
    rhs->addTerm( f * v );
    
    bc = BC::bc();
    VarPtr uhat = formulation.uhat();
    FunctionPtr x = Function::xn(1);
    FunctionPtr y = Function::yn(1);
    FunctionPtr z = Function::zn(1);
    
    // NVR 3-14-17
    /*
     I don't understand the BCs involving tc.  It seems to me if
     we're trying to get an inflow u of (1-x)*(1-y) (which I think we are,
     but I guess I'm not sure), then there should be an epsilon somewhere
     in there, because of the definition of tc and sigma.  But there is not.
     
     What I do understand is imposing Dirichlet conditions everywhere on u.
     This is pretty easy to do...
     */
    FunctionPtr u_exact = 1.0 - x;
    if (spaceDim >= 2)
      u_exact = u_exact * (1.0 - y);
    if (spaceDim >= 3)
      u_exact = u_exact * (1.0 - z);
    bc->addDirichlet(uhat, SpatialFilter::allSpace(), u_exact);
    
    // old implementation below
//    VarPtr tc = formulation.tc();
//    SpatialFilterPtr inflowX = SpatialFilter::matchingX(-1);
//    SpatialFilterPtr inflowY = SpatialFilter::matchingY(-1);
//    SpatialFilterPtr inflowZ = SpatialFilter::matchingZ(-1);
//    SpatialFilterPtr outflowX = SpatialFilter::matchingX(1);
//    SpatialFilterPtr outflowY = SpatialFilter::matchingY(1);
//    SpatialFilterPtr outflowZ = SpatialFilter::matchingZ(1);
//    FunctionPtr zero = Function::zero();
//    FunctionPtr one = Function::constant(1);
    
//    if (spaceDim == 1)
//    {
//      bc->addDirichlet(tc, inflowX, -one);
//      bc->addDirichlet(uhat, outflowX, zero);
//    }
//    if (spaceDim == 2)
//    {
//      bc->addDirichlet(tc, inflowX, -1*.5*(one-y));
//      bc->addDirichlet(uhat, outflowX, zero);
//      bc->addDirichlet(tc, inflowY, -2*.5*(one-x));
//      bc->addDirichlet(uhat, outflowY, zero);
//    }
//    if (spaceDim == 3)
//    {
//      bc->addDirichlet(tc, inflowX, -1*.25*(one-y)*(one-z));
//      bc->addDirichlet(uhat, outflowX, zero);
//      bc->addDirichlet(tc, inflowY, -2*.25*(one-x)*(one-z));
//      bc->addDirichlet(uhat, outflowY, zero);
//      bc->addDirichlet(tc, inflowZ, -3*.25*(one-x)*(one-y));
//      bc->addDirichlet(uhat, outflowZ, zero);
//    }

    testNorm = formulation.ip(normChoice);
  }
  else if (problemChoice == Stokes)
  {
    double mu = 1.0;
    
    StokesVGPFormulation formulation = StokesVGPFormulation::steadyFormulation(spaceDim, mu, conformingTraces);
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      for (int d=0; d<spaceDim; d++)
        trialOrderEnhancements[formulation.u(d+1)->ID()] = 1;
    }
    
    p = formulation.p();
    
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
    
    // to ensure zero mean for p, need the domain carefully defined:
    x0 = vector<double>(spaceDim,-1.0);
    
    width = 2.0;
    
    bc = BC::bc();

    SpatialFilterPtr boundary = SpatialFilter::allSpace();
    bc->addDirichlet(formulation.u_hat(1), boundary, u1_exact);
    bc->addDirichlet(formulation.u_hat(2), boundary, u2_exact);
    if (spaceDim==3) bc->addDirichlet(formulation.u_hat(3), boundary, u3_exact);
    
    vector<FunctionPtr> uVector = (spaceDim==2) ? vector<FunctionPtr>{u1_exact,u2_exact} : vector<FunctionPtr>{u1_exact,u2_exact,u3_exact};
    FunctionPtr u_exact = Function::vectorize(uVector);
    
    FunctionPtr forcingFunction = formulation.forcingFunction(u_exact, p_exact);
    
    rhs = formulation.rhs(forcingFunction);
  }
  else if (problemChoice == NavierStokes)
  {
    // set up classical Kovasznay flow solution:
    double Re = 40.0;
    navierStokesFormulation = Teuchos::rcp(new NavierStokesVGPFormulation(NavierStokesVGPFormulation::steadyFormulation(spaceDim, Re, conformingTraces, meshTopo, k, delta_k)));
    
    FunctionPtr u1, u2, p;
    NavierStokesVGPFormulation::getKovasznaySolution(Re, u1, u2, p);
    
    mesh = navierStokesFormulation->solutionIncrement()->mesh();
    int kovasznayCubatureEnrichment = 20; // 20 is better than 10 for accurately measuring error on the coarser meshes.
    navierStokesFormulation->solutionIncrement()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    navierStokesFormulation->solution()->setCubatureEnrichmentDegree(kovasznayCubatureEnrichment);
    
    if (useZeroMeanConstraints)
    {
      navierStokesFormulation->addZeroMeanPressureCondition();
      double p_mean = p->integrate(mesh);
      p = p - p_mean;
    }
    else
    {
      navierStokesFormulation->addPointPressureCondition({0.5,1.0});
      double p_center = p->evaluate(0.5, 1.0);
      p = p - p_center;
    }
    
    FunctionPtr u = Function::vectorize({u1, u2});
    FunctionPtr forcingFunction = NavierStokesVGPFormulation::forcingFunctionSteady(spaceDim, Re, u, p);
    
    navierStokesFormulation->addInflowCondition(SpatialFilter::allSpace(), u);
    navierStokesFormulation->setForcingFunction(forcingFunction);
    
    if (conformingTraces && enhanceFieldsForH1TracesWhenConforming)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Unsupported option for Navier-Stokes"); // if it seems important, we can add support for this later...
    }

    bf = navierStokesFormulation->bf();
    testNorm = bf->graphNorm();
    
    solution = navierStokesFormulation->solutionIncrement();
    
    bc = solution->bc();
    rhs = solution->rhs();
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
  
  // now that we have mesh, add pressure constraint for Stokes (imposing zero at origin--want to aim for center of mesh)
  if (problemChoice == Stokes)
  {
    if (!useZeroMeanConstraints)
    {
      vector<double> origin(spaceDim,0);
      IndexType vertexIndex;
      
      bool foundVertex = meshTopo->getVertexIndex(origin, vertexIndex);
      foundVertex = MPIWrapper::globalOr(Comm, foundVertex);
      TEUCHOS_TEST_FOR_EXCEPTION(!foundVertex, std::invalid_argument, "origin vertex not found on any rank");
      bc->addSpatialPointBC(p->ID(), 0, origin);
    }
    else
    {
      bc->addZeroMeanConstraint(p);
    }
  }
  
  bool useLightWeightViews = true; // flag added to allow checking whether any pure MeshTopologyViews are at issue in issues that might arise...
  
  meshesCoarseToFine.clear();
  
  bool useGMGSolverForMeshes = true; // use static method from GMGSolver to generate meshesCoarseToFine
  if (useGMGSolverForMeshes)
  {
    if (mesh == Teuchos::null)
    {
      mesh = Teuchos::rcp(new Mesh(meshTopo, bf, H1Order, delta_k, trialOrderEnhancements));
    }

    Epetra_Time hRefinementTimer(Comm);
    int meshWidthCells = rootMeshNumCells;
    while (meshWidthCells < numCells)
    {
      vector<IndexType> activeCellIDs = meshTopo->getActiveCellIndicesGlobal();
      
      mesh->hRefine(activeCellIDs);
//      if (rank==0)
//      {
//        print("h-refining cells", activeCellIDs);
////      }
//      IndexType nextCellID = meshTopo->cellCount();
//      for (IndexType activeCellID : activeCellIDs)
//      {
//        CellTopoPtr cellTopo = meshTopo->getCell(activeCellID)->topology();
//        RefinementPatternPtr refPattern = RefinementPattern::regularRefinementPattern(cellTopo);
//        meshTopo->refineCell(activeCellID, refPattern, nextCellID);
//        nextCellID += refPattern->numChildren();
//      }
      meshWidthCells *= 2;
    }
    if (meshWidthCells != numCells)
    {
      if (rank == 0)
      {
        cout << "Warning: may have over-refined mesh; mesh has width " << meshWidthCells << ", not " << numCells << endl;
      }
    }
    if (rank==0)
    {
      int refinementTime = hRefinementTimer.ElapsedTime();
      cout << "h refinements (Mesh/MeshTopology construction) completed in " << refinementTime << " seconds.\n";
      hRefinementTimer.ResetStartTime();
    }
//    if (setupMeshTopologyAndQuit) return;
//    mesh = Teuchos::rcp(new Mesh(meshTopo, bf, H1Order, delta_k, trialOrderEnhancements));
//    if (rank==0)
//    {
//      int meshConstructionTime = hRefinementTimer.ElapsedTime();
//      cout << "Mesh construction completed in " << meshConstructionTime << " seconds.\n";
//    }
  }
  else
  {
    mesh = MeshFactory::rectilinearMesh(bf, dimensions, elementCounts, H1Order, delta_k, x0);
    
    MeshTopologyViewPtr meshTopoView;
    if (useLightWeightViews)
      meshTopoView = mesh->getTopology()->getView(mesh->cellIDsInPartition());
    else
      meshTopoView = mesh->getTopology()->deepCopy();
    
    int H1Order_coarse = k_coarse + 1;
    MeshPtr k0Mesh = Teuchos::rcp(new Mesh(meshTopoView, bf, H1Order_coarse, delta_k));
    meshesCoarseToFine.push_back(k0Mesh);
    
    int meshWidthCells = rootMeshNumCells;
    while (meshWidthCells < numCells)
    {
      set<IndexType> activeCellIDs = mesh->getActiveCellIDsGlobal(); // should match between coarseMesh and mesh
      mesh->hRefine(activeCellIDs);
      if (rank==0)
      {
        Camellia::print("h-refining cells", activeCellIDs);
      }
      
      MeshTopologyViewPtr meshTopoView;
      if (useLightWeightViews)
        meshTopoView = mesh->getTopology()->getView(mesh->cellIDsInPartition());
      else
        meshTopoView = mesh->getTopology()->deepCopy();
          
      k0Mesh = Teuchos::rcp(new Mesh(meshTopoView, bf, H1Order_coarse, delta_k));
      
      meshesCoarseToFine.push_back(k0Mesh);
      meshWidthCells *= 2;
    }
    
    // a new experiment: duplicate the finest h-mesh, so that we get a smoother application
    // that involves just the fine k0 elements.
    if (k_coarse != k)
      meshesCoarseToFine.push_back(k0Mesh);
    
    if ((k_coarse == 0) && (k > 1))
    {
      MeshTopologyViewPtr meshTopoView;
      if (useLightWeightViews)
        meshTopoView = k0Mesh->getTopology()->getView(mesh->cellIDsInPartition());
      else
        meshTopoView = k0Mesh->getTopology()->deepCopy();
      
      MeshPtr k1Mesh = Teuchos::rcp(new Mesh(meshTopoView, bf, H1Order_coarse + 1, delta_k));

      meshesCoarseToFine.push_back(k1Mesh);
    }
    
    meshesCoarseToFine.push_back(mesh);
    
    if (meshWidthCells != numCells)
    {
      if (rank == 0)
      {
        cout << "Warning: may have over-refined mesh; mesh has width " << meshWidthCells << ", not " << numCells << endl;
      }
    }
  }
  
  Epetra_Time timer(Comm);
  if (solution == Teuchos::null)
  {
    solution = Solution::solution(mesh, bc, rhs, testNorm);
  }
  solution->setUseCondensedSolve(useStaticCondensation);
  solution->setZMCsAsGlobalLagrange(false); // fine grid solution shouldn't impose ZMCs (should be handled in coarse grid solve)
  
  int solutionConstructionTime = timer.ElapsedTime();
  if (rank==0)
    cout << "Solution constructed in " << solutionConstructionTime << " seconds.\n";
  
  if (useGMGSolverForMeshes)
  {
    Teuchos::ParameterList pl;
    pl.set("kCoarse", k_coarse);
    pl.set("delta_k", delta_k);
    pl.set("jumpToCoarsePolyOrder",jumpToCoarsePolyOrder);
    timer.ResetStartTime();
    meshesCoarseToFine = GMGSolver::meshesForMultigrid(solution->mesh(), pl);
    int meshesForMultigridExecutionTime = timer.ElapsedTime();
    if (rank==0)
      cout << "meshesForMultigrid() executed in " << meshesForMultigridExecutionTime << " seconds.\n";
  }
}

long long approximateMemoryCostsForMeshTopologies(vector<MeshPtr> meshes)
{
  map<MeshTopologyView*, long long> meshTopologyCosts; // pointer as key ensures we only count each MeshTopology once, even if they are shared
  for (MeshPtr mesh : meshes)
  {
    MeshTopologyViewPtr meshTopo = mesh->getTopology();
    long long memoryCost = meshTopo->approximateMemoryFootprint();
    meshTopologyCosts[meshTopo.get()] = memoryCost;
  }
  long long memoryCostTotal = 0;
  for (auto entry : meshTopologyCosts)
  {
    memoryCostTotal += entry.second;
  }
  return memoryCostTotal;
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
      vector<double> timingValues;
      for (auto entry : timings)
      {
        timingValues.push_back(entry.second);
      }
      if (rank == 0)
      {
        cout << "************************************************\n";
        cout << " Selected timings, values on rank 0:\n";
        
        cout << setprecision(2);
        cout << std::scientific;
        
        int i=0;
        for (auto entry : timings)
        {
          cout.setf(std::ios::left, std::ios::adjustfield);
          cout << std::setw(35) << entry.first;
          cout.setf(std::ios::right, std::ios::adjustfield);
          cout << std::setw(8);
          cout << timingValues[i++] << " sec." << endl;
        }
        cout << "************************************************\n";
      }
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
      
      int i=0;
      for (auto entry : timings)
      {
        cout.setf(std::ios::left, std::ios::adjustfield);
        cout << std::setw(35) << entry.first;
        cout.setf(std::ios::right, std::ios::adjustfield);
        cout << std::setw(8);
        cout << maxTimingValues[i++] << " sec." << endl;
      }
      cout << "************************************************\n";
    }
  }
}

struct Result
{
  int k_fine;
  int k_coarse;
  int k_levels;
  int fineMeshWidth;
  int coarseMeshWidth;
  int h_levels;
  int iterations;
};

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
  int k_coarse = 0; // coarse poly order
  bool jumpToCoarsePolyOrder = false;

  bool conformingTraces = true;
  bool enhanceFieldsForH1TracesWhenConforming = false;

  int numGrids = -1;
  
  int numCells = -1;
  int numCellsRootMesh = -1;
  int spaceDim = 1;
  bool useCondensedSolve = true;

  double cgTol = 1e-10;
  double epsilonForConvectionDiffusion = 1e-2;
  int cgMaxIterations = 2000;

  bool clearFinestCondensedDofInterpreterAfterProlongation = false;
  
  bool reportTimings = false;
  bool useZeroMeanConstraints = false;
  bool useConjugateGradient = true;
  bool useDiagonalSchwarzWeighting = false;
  bool logFineOperator = false;
  
  bool solveDirectly = false;
  
  bool writeOpToFile = false;
  
  bool constructProlongationOperatorAndQuit = false;
  bool setUpMeshesAndQuit = false;
  bool setUpMeshTopologyAndQuit = false;
  
  bool useFactoredCholeskyForOptimalTests = true;
  
  bool runMany = false;
  
  string multigridStrategyString = "V-cycle";
  
  bool pauseOnRankZero = false;

  int azOutput = 10;
  
  string problemChoiceString = "Poisson";
  string coarseSolverChoiceString = "KLU";
  string testNormChoice = "Graph";

  cmdp.setOption("problem",&problemChoiceString,"problem choice: Poisson, ConvectionDiffusion, Stokes, Navier-Stokes");
  cmdp.setOption("epsilonForConvectionDiffusion",&epsilonForConvectionDiffusion,"diffusion parameter for convection-dominated diffusion");
  
  cmdp.setOption("testNorm",&testNormChoice,"test norm: Graph, or for ConvectionDiffusion: Robust or CoupledRobust");

  cmdp.setOption("numCells",&numCells,"mesh width");
  cmdp.setOption("numCellsRootMesh",&numCellsRootMesh,"mesh width of root mesh");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment (use -1 for spaceDim, -2 for polyOrder)");
  cmdp.setOption("coarsePolyOrder", &k_coarse, "polynomial order for field variables on coarse grid");

  cmdp.setOption("jumpToCoarsePolyOrder","dontJumpToCoarsePolyOrder",&jumpToCoarsePolyOrder);
  
  cmdp.setOption("coarseSolver", &coarseSolverChoiceString, "coarse solver choice: KLU, MUMPS, SuperLUDist, SimpleML");

  cmdp.setOption("CG", "GMRES", &useConjugateGradient);
  
  cmdp.setOption("azOutput", &azOutput);
  cmdp.setOption("logFineOperator", "dontLogFineOperator", &logFineOperator);
  
  cmdp.setOption("clearFinestCondensedDofInterpreterAfterProlongation", "retainFinestCondensedDofInterpreterAfterProlongation",
                 &clearFinestCondensedDofInterpreterAfterProlongation);
  
  cmdp.setOption("multigridStrategy", &multigridStrategyString, "Multigrid strategy: V-cycle, W-cycle, Full-V, Full-W, or Two-level");
  cmdp.setOption("numGrids", &numGrids, "Number of grid levels to use (-1 means all).");
  
  cmdp.setOption("useCondensedSolve", "useStandardSolve", &useCondensedSolve);
  cmdp.setOption("useConformingTraces", "useNonConformingTraces", &conformingTraces);
  cmdp.setOption("useFactoredCholesky", "useStandardCholesky", &useFactoredCholeskyForOptimalTests, "Use factored cholesky for optimal test solve");
  cmdp.setOption("enhanceFieldsForH1TracesWhenConforming", "equalOrderFieldsForH1TracesWhenConforming", &enhanceFieldsForH1TracesWhenConforming);

  cmdp.setOption("spaceDim", &spaceDim, "space dimensions (1, 2, or 3)");

  cmdp.setOption("maxIterations", &cgMaxIterations, "maximum number of CG iterations");
  cmdp.setOption("cgTol", &cgTol, "CG convergence tolerance");

  cmdp.setOption("pause","dontPause",&pauseOnRankZero, "pause (to allow attachment by tracer, e.g.), waiting for user to press a key");
  cmdp.setOption("reportTimings", "dontReportTimings", &reportTimings, "Report timings in Solution");
  cmdp.setOption("runMany", "runOne", &runMany, "Run for intermediate mesh sizes in the range specified, and report iteration counts for each.");

  cmdp.setOption("constructProlongationOperatorAndQuit", "constructProlongationOperatorAndContinue", &constructProlongationOperatorAndQuit);
  cmdp.setOption("setUpMeshTopologyAndQuit", "setUpMeshTopologyAndContinue", &setUpMeshTopologyAndQuit);
  cmdp.setOption("setUpMeshesAndQuit", "setUpMeshesAndRunNormally", &setUpMeshesAndQuit);
  cmdp.setOption("solveDirectly", "solveIteratively", &solveDirectly);

  cmdp.setOption("useDiagonalSchwarzWeighting","dontUseDiagonalSchwarzWeighting",&useDiagonalSchwarzWeighting);
  cmdp.setOption("useZeroMeanConstraint", "usePointConstraint", &useZeroMeanConstraints, "Use a zero-mean constraint for the pressure (otherwise, use a vertex constraint at the origin)");
  
  cmdp.setOption("writeOpToFile", "dontWriteOpToFile", &writeOpToFile);

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

  if (problemChoiceString == "Poisson")
  {
    problemChoice = Poisson;
  }
  else if (problemChoiceString == "ConvectionDiffusion")
  {
    problemChoice = ConvectionDiffusion;
  }
  else if (problemChoiceString == "Stokes")
  {
    problemChoice = Stokes;
  }
  else if (problemChoiceString == "Navier-Stokes")
  {
    problemChoice = NavierStokes;
  }
  else
  {
    if (rank==0) cout << "Problem choice not recognized.\n";
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  GMGOperator::MultigridStrategy multigridStrategy;
  if (multigridStrategyString == "Two-level")
  {
    multigridStrategy = GMGOperator::TWO_LEVEL;
  }
  else if (multigridStrategyString == "W-cycle")
  {
    multigridStrategy = GMGOperator::W_CYCLE;
  }
  else if (multigridStrategyString == "V-cycle")
  {
    multigridStrategy = GMGOperator::V_CYCLE;
  }
  else if (multigridStrategyString == "Full-W")
  {
    multigridStrategy = GMGOperator::FULL_MULTIGRID_W;
    useConjugateGradient = false; // not symmetric
  }
  else if (multigridStrategyString == "Full-V")
  {
    multigridStrategy = GMGOperator::FULL_MULTIGRID_V;
    useConjugateGradient = false; // not symmetric
  }
  else
  {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "unrecognized multigrid strategy");
  }
  
  if (numCellsRootMesh == -1)
  {
    if (!useZeroMeanConstraints && (problemChoice == Stokes))
    {
      numCellsRootMesh = 2;
    }
    else
    {
      numCellsRootMesh = 1;
    }
  }

  if (rank==0)
  {
    cout << "Solving " << spaceDim << "D " << problemChoiceString << " problem with k = " << k << " on " << numProcs << " MPI ranks.  Initializing meshes...\n";
  }
  
  vector<Result> runsToDo;
  if (!runMany)
  {
    Result run;
    run.k_fine = k;
    run.k_coarse = k_coarse;
    run.k_levels = 0; // can work this out by hand?
    run.coarseMeshWidth = numCellsRootMesh;
    run.fineMeshWidth = numCells;
    run.h_levels = 0;
    runsToDo.push_back(run);
  }
  else
  {
    Result run;
    run.k_coarse = k_coarse;
    run.k_fine = k_coarse;
    run.k_levels = 0;
    run.coarseMeshWidth = numCellsRootMesh;
    run.fineMeshWidth = numCellsRootMesh;
    run.h_levels = 0;
    while (run.k_fine < k)
    {
      run.k_fine = max(run.k_fine*2,1);
      if (!jumpToCoarsePolyOrder)
        run.k_levels++;
      else
        run.k_levels = 1;
      runsToDo.push_back(run);
    }
    while (run.fineMeshWidth < numCells)
    {
      run.fineMeshWidth = 2 * run.fineMeshWidth;
      run.h_levels++;
      runsToDo.push_back(run);
    }
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
  
  
  for (int i=0; i<runsToDo.size(); i++)
  {
    Result *run = &runsToDo[i];
    k = run->k_fine;
    k_coarse = run->k_coarse;
    numCellsRootMesh = run->coarseMeshWidth;
    numCells = run->fineMeshWidth;
    
    Epetra_Time timer(*Comm);

    int meshInitTimerHandle = TimeLogger::sharedInstance()->startTimer("Mesh Init.");
    
    SolutionPtr solution;
    MeshPtr coarseMesh;
    IPPtr ip;

    if (numCells == -1)
    {
      numCells = (int)ceil(pow(numProcs,1.0/(double)spaceDim));
    }
    
    Teuchos::RCP<NavierStokesVGPFormulation> nsFormulation;
    vector<MeshPtr> meshesCoarseToFine;
    initializeSolutionAndCoarseMesh(solution, meshesCoarseToFine, ip, problemChoice, spaceDim, conformingTraces, enhanceFieldsForH1TracesWhenConforming,
                                    useCondensedSolve, numCells, k, getDeltaK(k), k_coarse, numCellsRootMesh, useZeroMeanConstraints, jumpToCoarsePolyOrder,
                                    setUpMeshTopologyAndQuit, nsFormulation, testNormChoice, epsilonForConvectionDiffusion);
    
    if (setUpMeshTopologyAndQuit)
    {
      return 0;
    }
    
    if (numGrids != -1)
    {
      vector<MeshPtr> newMeshesCoarseToFine;
      for (int gridNumber=0; gridNumber < numGrids; gridNumber++)
      {
        MeshPtr lastMesh = meshesCoarseToFine[meshesCoarseToFine.size()-1];
        meshesCoarseToFine.pop_back();
        newMeshesCoarseToFine.insert(newMeshesCoarseToFine.begin(), lastMesh);
      }
      meshesCoarseToFine = newMeshesCoarseToFine;
    }
    
    TimeLogger::sharedInstance()->stopTimer(meshInitTimerHandle);
    
    double meshInitializationTime = timer.ElapsedTime();

    int numDofs = solution->mesh()->numGlobalDofs();
    int numTraceDofs = solution->mesh()->numFluxDofs();
    int numElements = solution->mesh()->numActiveElements();
    
    long long approximateMemoryCostInBytes = approximateMemoryCostsForMeshTopologies(meshesCoarseToFine);
    double bytesPerMB = (1024.0 * 1024.0);
    double memoryCostInMB = approximateMemoryCostInBytes / bytesPerMB;

    BFPtr bf = solution->mesh()->bilinearForm();

    GlobalIndexType sampleCellID = 0;
    int totalTrialDofs, totalTestDofs;
    double B_denseMatrixSize, G_denseMatrixSize, K_denseMatrixSize, B_sparseMatrixSize, G_sparseMatrixSize;
    int rankWithSampleCell = solution->mesh()->Comm()->getSize();
    if (solution->mesh()->getTopology()->isValidCellIndex(sampleCellID))
    {
      rankWithSampleCell = solution->mesh()->Comm()->getRank();
      ElementTypePtr sampleElementType = solution->mesh()->getElementType(sampleCellID);
      totalTrialDofs = sampleElementType->trialOrderPtr->totalDofs();
      totalTestDofs = sampleElementType->testOrderPtr->totalDofs();
      
      int doubleSizeInBytes = sizeof(double);
      B_denseMatrixSize = (totalTrialDofs * totalTestDofs * doubleSizeInBytes) / bytesPerMB;
      G_denseMatrixSize = (totalTestDofs * totalTestDofs * doubleSizeInBytes) / bytesPerMB;
      K_denseMatrixSize = (totalTrialDofs * totalTrialDofs * doubleSizeInBytes) / bytesPerMB;
      B_sparseMatrixSize = ( bf->nonZeroEntryCount(sampleElementType->trialOrderPtr, sampleElementType->testOrderPtr) * doubleSizeInBytes) / bytesPerMB;
      G_sparseMatrixSize = ( ip->nonZeroEntryCount(sampleElementType->testOrderPtr) * doubleSizeInBytes) / bytesPerMB;
    }

    Comm = MPIWrapper::EpetraComm(solution->mesh()->Comm());
    
    int minRankWithSampleCell;
    Comm->MinAll(&rankWithSampleCell, &minRankWithSampleCell, 1);
    Comm->Broadcast(&totalTestDofs, 1, minRankWithSampleCell);
    Comm->Broadcast(&totalTrialDofs, 1, minRankWithSampleCell);
    
    Comm->Broadcast(&B_denseMatrixSize, 1, minRankWithSampleCell);
    Comm->Broadcast(&G_denseMatrixSize, 1, minRankWithSampleCell);
    Comm->Broadcast(&K_denseMatrixSize, 1, minRankWithSampleCell);
    Comm->Broadcast(&B_sparseMatrixSize, 1, minRankWithSampleCell);
    Comm->Broadcast(&G_sparseMatrixSize, 1, minRankWithSampleCell);

    int coarseMeshGlobalDofs = meshesCoarseToFine[0]->numGlobalDofs();
    int coarseMeshNumElements = meshesCoarseToFine[0]->numElements();
    int coarseMeshTraceDofs = meshesCoarseToFine[0]->numFluxDofs();
    
    double cellHaloTime = solution->mesh()->getTopology()->totalTimeComputingCellHalos();
    double maxCellHaloTime = MPIWrapper::max(solution->mesh()->Comm(), cellHaloTime);
    
    if (rank==0)
    {
      int numLevels = static_cast<int>(meshesCoarseToFine.size());
      cout << setprecision(2);
      cout << "Mesh initialization completed in " << meshInitializationTime << " seconds.  Fine mesh has " << numDofs;
      cout << " global degrees of freedom (" << numTraceDofs << " trace dofs) on " << numElements << " elements.\n";
      cout << "Coarsest mesh has " << coarseMeshGlobalDofs << " global degrees of freedom (" << coarseMeshTraceDofs << " trace dofs) on " << coarseMeshNumElements << " elements.\n";
      cout << "Approximate (correct within a factor of 2 or so) memory cost for all mesh topologies: " << memoryCostInMB << " MB.\n";
      cout << "Number of mesh levels: " << numLevels << ".\n";
      cout << "Approximate memory cost per element (assuming dense storage): G = " << G_denseMatrixSize << " MB, B = " << B_denseMatrixSize << " MB, K = ";
      cout << K_denseMatrixSize << " MB.\n";
      cout << totalTrialDofs << " trial dofs per element; " << totalTestDofs << " test dofs.\n";
      
      cout << "Approximate memory cost per element (assuming sparse storage): G = " << G_sparseMatrixSize << " MB, B = " << B_sparseMatrixSize << " MB.\n";
      
      cout << "Maximum time spent determining cell halos: " << maxCellHaloTime << " seconds.\n";
      
      if (setUpMeshesAndQuit)
        cout << "***** setUpMeshesAndQuit option selected; now exiting *****\n";
      
      if (solveDirectly)
        cout << "******* Using direct solve in place of multigrid-preconditioned iterative solve ****** \n";
      else
        cout << "Multigrid strategy: " << multigridStrategyString << endl;
      
      if (useDiagonalSchwarzWeighting)
      {
        cout << "***********************************************************************************************************\n";
        cout << "** NOTE: USING DIAGONAL SCHWARZ WEIGHTING.  THIS IS EXPERIMENTAL, AS PREVIOUS TESTS HAVE NOT WORKED WELL. *\n";
        cout << "***********************************************************************************************************\n";
      }
    }
    
    printTimings();
    
    if (setUpMeshesAndQuit)
    {
      return 0;
    }
    
  //  if (rank==0) cout << "Setting optimal test solve to QR\n";
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
    
    double gmgSolverInitializationTime = 0, solveTime;
    
    timer.ResetStartTime();
    int gmgSolverInitTimerHandle = TimeLogger::sharedInstance()->startTimer("GMGSolver Init.");
    if (!solveDirectly)
    {
      auto setUpGMGSolver = [&solution, &meshesCoarseToFine, &cgMaxIterations, &cgTol,
                             &multigridStrategy, &useCondensedSolve, &useDiagonalSchwarzWeighting,
                             &useConjugateGradient, &azOutput, &logFineOperator,
                             &clearFinestCondensedDofInterpreterAfterProlongation] () -> Teuchos::RCP<GMGSolver>
      {
        bool reuseFactorization = true;
        SolverPtr coarseSolver = Solver::getDirectSolver(reuseFactorization);
        
  #if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
        SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(coarseSolver.get());
        if (superLUSolver)
        {
          superLUSolver->setRunSilent(true);
        }
  #endif
        
        Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp(new GMGSolver(solution, meshesCoarseToFine, cgMaxIterations, cgTol,
                                                                       multigridStrategy, coarseSolver, useCondensedSolve,
                                                                       useDiagonalSchwarzWeighting));
        gmgSolver->setUseConjugateGradient(useConjugateGradient);
        gmgSolver->setAztecOutput(azOutput);
        gmgSolver->gmgOperator()->setNarrateOnRankZero(logFineOperator,"finest GMGOperator");
        
        gmgSolver->gmgOperator()->setClearFinestCondensedDofInterpreterAfterProlongation(clearFinestCondensedDofInterpreterAfterProlongation);
        
        return gmgSolver;
      };
      Teuchos::RCP<GMGSolver> gmgSolver = setUpGMGSolver();
      TimeLogger::sharedInstance()->stopTimer(gmgSolverInitTimerHandle);
      
      gmgSolverInitializationTime = timer.ElapsedTime();
      if (rank==0)
      {
        cout << "GMGSolver initialized in " << gmgSolverInitializationTime << " seconds.\n";
        Teuchos::RCP<GMGOperator> op = gmgSolver->gmgOperator();
        while (op != Teuchos::null)
        {
          cout << "Level " << op->getOperatorLevel() << " operator has smoother overlap of " << op->getSmootherOverlap() << endl;
          op = op->getCoarseOperator();
        }
      }

      if (constructProlongationOperatorAndQuit)
      {
        timer.ResetStartTime();
        gmgSolver->gmgOperator()->constructProlongationOperator();
        double gmgFineProlongationOperatorConstructionTime = timer.ElapsedTime();
        if (rank==0)
        {
          cout << "fine GMG prolongation operator constructed in " << gmgFineProlongationOperatorConstructionTime << " seconds.\n";
          cout << "--constructProlongationOperatorAndQuit passed in; now exiting.\n";
        }
  #ifdef HAVE_MPI
        MPI_Finalize();
  #endif
        exit(0);
      }
      
      if (problemChoice == NavierStokes)
      {
        if (rank == 0) cout << "Navier-Stokes: taking an initial Newton step before recording results.\n";
        gmgSolver->setAztecOutput(0); // solve silently
        // then take a Newton step, and recreate the gmgSolver
        nsFormulation->setSolver(gmgSolver);
        nsFormulation->solveAndAccumulate();
        if (rank == 0) cout << "Navier-Stokes: recreating GMGSolver after taking initial Newton step.\n";
        gmgSolver = setUpGMGSolver();
      }
      
      timer.ResetStartTime();
      
      int totalSolveTimerHandle = TimeLogger::sharedInstance()->startTimer("Total solve");
      solution->solve(gmgSolver);
      TimeLogger::sharedInstance()->stopTimer(totalSolveTimerHandle);
      solveTime = timer.ElapsedTime();
      
      run->iterations = gmgSolver->iterationCount();

      if (useCondensedSolve)
      {
        long long memoryCost = 0;
        
        Teuchos::RCP<GMGOperator> gmgOperator = gmgSolver->gmgOperator();
        while (gmgOperator != Teuchos::null)
        {
          CondensedDofInterpreter<double>* dofInterpreter = dynamic_cast<CondensedDofInterpreter<double>*>(gmgOperator->getFineDofInterpreter().get());
          memoryCost += dofInterpreter->approximateStiffnessAndLoadMemoryCost();
          
          gmgOperator = gmgOperator->getCoarseOperator();
        }
        
        double memoryCostInMB = memoryCost / (1024.0 * 1024.0);
        if (rank==0) cout << "On rank 0, CondensedDofInterpreter used " << memoryCostInMB << " MB for stiffness and load storage.\n";
      }
      
      if (rank==0)
      {
        Teuchos::RCP<GMGOperator> op = gmgSolver->gmgOperator();
        while (op != Teuchos::null)
        {
          cout << "Level " << op->getOperatorLevel() << " operator has smoother overlap of " << op->getSmootherOverlap();
          cout << " and sigma value of " << op->getSmootherWeight() << endl;
          op = op->getCoarseOperator();
        }
      }
      
      if (rank==0) cout << "Finest GMGOperator, timing report:\n";
      gmgSolver->gmgOperator()->reportTimingsSumOfOperators(StatisticChoice::MAX);
      
      if (writeOpToFile)
      {
        Teuchos::RCP<GMGOperator> op = gmgSolver->gmgOperator();
        if (rank==0) cout << "writing op to op.dat.\n";
        Tpetra::MatrixMarket::Writer<TMatrix<double> >::writeSparseFile("op.dat", E_to_T::matrix(op->getMatrixRepresentation()));
        
        if (rank==0) cout << "writing fine stiffness to A.dat.\n";
        Tpetra::MatrixMarket::Writer<TMatrix<double> >::writeSparseFile("A.dat", E_to_T::matrix(solution->getStiffnessMatrix()));
      }
    }
    else
    {
      timer.ResetStartTime();
      SolverPtr solver = Solver::getDirectSolver();
  #if defined(HAVE_AMESOS_SUPERLUDIST) || defined(HAVE_AMESOS2_SUPERLUDIST)
      // if we are solving directly using SuperLU_Dist, there's a good chance we're memory-bound.  Let's use as many processors as are available
      SuperLUDistSolver* superLUSolver = dynamic_cast<SuperLUDistSolver*>(solver.get());
      if (superLUSolver)
      {
        superLUSolver->setMaxProcsToUse(-3);
        if (rank==0)
          cout << "****** Set SuperLUDistSolver to use all available processors (for direct solve). ***** \n";
      }
  #endif
      solution->solve(solver);
      solveTime = timer.ElapsedTime();
      
      solution->reportTimings();
    }

    double maxTimeLocalStiffness = solution->maxTimeLocalStiffness();
    
    if (rank==0)
    {
      double totalTime = solveTime + gmgSolverInitializationTime + meshInitializationTime;
      cout << "Max time spent determining local stiffness contributions (included in Solve, below): " << maxTimeLocalStiffness << " seconds.\n";
      
      cout << "Total time: " << totalTime << " seconds.\n";
      int tabWidth = 15;
      cout << setw(tabWidth) << "Mesh Init." << setw(tabWidth) << "GMG Init." << setw(tabWidth) << "Solve" << endl;
      if (gmgSolverInitializationTime ==  0)
        cout << setw(tabWidth) << meshInitializationTime << setw(tabWidth) << "-" << setw(tabWidth) << solveTime << endl;
      else
        cout << setw(tabWidth) << meshInitializationTime << setw(tabWidth) << gmgSolverInitializationTime << setw(tabWidth) << solveTime << endl;
  //    cout << "Solve completed in " << solveTime << " seconds.\n";
  //    cout << "Total time, including GMGSolver initialization (but not mesh construction): " << solveTime + gmgSolverInitializationTime << " seconds.\n";
    }
    
    printTimings();
  }
  
  if (rank == 0)
  {
    vector<int> colWidths = {20,20,20,20,20,20,20};
    cout << "Summary:\n";
    cout << setw(colWidths[0]) << "$k_{\\rm fine}$"; //
    cout << setw(colWidths[1]) << "& $k_{\\rm coarse}$";
    cout << setw(colWidths[2]) << "& $k$ levels";
    cout << setw(colWidths[3]) << "& Fine Mesh Width";
    cout << setw(colWidths[4]) << "& Coarse Mesh Width";
    cout << setw(colWidths[5]) << "& $h$ levels";
    cout << setw(colWidths[6]) << "& Iterations\\\\\n";
    
    for (Result run : runsToDo)
    {
      cout << setw(colWidths[0]) << run.k_fine;
      cout << setw(colWidths[1]) << "&" << run.k_coarse;
      cout << setw(colWidths[2]) << "&" << run.k_levels;
      cout << setw(colWidths[3]) << "&" << run.fineMeshWidth;
      cout << setw(colWidths[4]) << "&" << run.coarseMeshWidth;
      cout << setw(colWidths[5]) << "&" << run.h_levels;
      cout << setw(colWidths[6]) << "&" << run.iterations;
      cout << "\\\\\n";
    }
  }
  
  return 0;
}
