#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_TimeMonitor.hpp"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "BF.h"
#include "Function.h"
#include "RefinementStrategy.h"
#include "GMGSolver.h"
#include "SpaceTimeConvectionDiffusionFormulation.h"
#include "SpatiallyFilteredFunction.h"
#include "ExpFunction.h"

using namespace Camellia;

class ExactU1D : public Function
{
public:
  double _l;
  double _lambda1;
  double _lambda2;
  ExactU1D(double epsilon, double l=3) : Function(0), _l(l)
  {
    _l = 3;
    _lambda1 = (-1.+sqrt(1.-4*epsilon*_l))/(-2.*epsilon);
    _lambda2 = (-1.-sqrt(1.-4*epsilon*_l))/(-2.*epsilon);
  }
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double t = (*points)(cellIndex,ptIndex,1);
        values(cellIndex, ptIndex) = exp(-_l*t)*(exp(_lambda1*(x-1))-exp(_lambda2*(x-1)));
      }
    }
  }
};

class ExactU2D : public Function
{
public:
  double _l;
  double _lambda1;
  double _lambda2;
  ExactU2D(double epsilon, double l=1) : Function(0), _l(l)
  {
    double k = l;
    _lambda1 = (-1.+sqrt(1.-4*epsilon*k))/(-2.*epsilon);
    _lambda2 = (-1.-sqrt(1.-4*epsilon*k))/(-2.*epsilon);
  }
  void values(Intrepid::FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const Intrepid::FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        double t = (*points)(cellIndex,ptIndex,2);
        values(cellIndex, ptIndex) = exp(-_l*t)*(exp(_lambda1*(x-1))-exp(_lambda2*(x-1)))*(2*(y-.5));
      }
    }
  }
};

int main(int argc, char *argv[])
{

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  int commRank = Teuchos::GlobalMPISession::getRank();

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options

  // problem parameters:
  int spaceDim = 2;
  double epsilon = 1;
  int numRefs = 1;
  int k = 2, delta_k = 2;
  int numXElems = 1;
  bool useConformingTraces = false;
  string solverChoice = "KLU";
  double solverTolerance = 1e-6;
  string norm = "Graph";
  cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("numXElems",&numXElems,"number of elements in x direction");
  cmdp.setOption("epsilon", &epsilon, "epsilon");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "KLU, SuperLU, MUMPS, GMG-Direct, GMG-ILU, GMG-IC");
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  // Exact solution
  // FunctionPtr uExact1D = Teuchos::rcp(new ExactU1D(epsilon));
  // FunctionPtr uExact2D = Teuchos::rcp(new ExactU2D(epsilon));
  double l = 3;
  double lambda1 = (-1+sqrt(1-4*epsilon*l))/(-2*epsilon);
  double lambda2 = (-1-sqrt(1-4*epsilon*l))/(-2*epsilon);
  FunctionPtr explt = Teuchos::rcp(new Exp_at(-l));
  FunctionPtr explambda1x = Teuchos::rcp(new Exp_ax(lambda1));
  FunctionPtr explambda2x = Teuchos::rcp(new Exp_ax(lambda2));
  FunctionPtr u_exact = explt*(explambda1x-explambda2x);
  FunctionPtr sigma_exact = epsilon*u_exact->dx()->dx();
  FunctionPtr tc_exact = u_exact-sigma_exact;
  FunctionPtr uhat_exact = u_exact;

  FunctionPtr beta;
  FunctionPtr beta_x = Function::constant(1);
  FunctionPtr beta_y = Function::constant(0);
  FunctionPtr beta_z = Function::constant(0);
  // if (spaceDim == 1)
  //   beta = beta_x;
  // else if (spaceDim == 2)
  //   beta = Function::vectorize(beta_x, beta_y);
  // else if (spaceDim == 3)
    beta = Function::vectorize(beta_x, beta_y, beta_z);

  SpaceTimeConvectionDiffusionFormulation form(spaceDim, epsilon, beta, useConformingTraces);

  map<int, FunctionPtr> exactMap;
  exactMap[form.u()->ID()] = u_exact;
  exactMap[form.sigma(1)->ID()] = sigma_exact;
  exactMap[form.tc()->ID()] = form.tc()->termTraced()->evaluate(exactMap);
  exactMap[form.uhat()->ID()] = form.uhat()->termTraced()->evaluate(exactMap);

  // Build mesh
  vector<double> x0 = vector<double>(spaceDim,-1);
  double width = 1.0;
  vector<double> dimensions;
  vector<int> elementCounts;
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(width);
    elementCounts.push_back(numXElems);
  }
  MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);
  double t0 = 0.0, t1 = 1.0;
  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);
  // MeshPtr mesh = Teuchos::rcp( new Mesh(spaceTimeMeshTopo, form.bf(), k+1, delta_k) ) ;

  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);
  // FunctionPtr x2 = Function::xn(2);
  // FunctionPtr y2 = Function::yn(2);
  // SpatialFilterPtr lessHalfX = SpatialFilter::lessThanX(0.25);
  // SpatialFilterPtr greaterNegHalfX = SpatialFilter::greaterThanX(-0.25);
  // SpatialFilterPtr pulseX = lessHalfX & greaterNegHalfX;
  // FunctionPtr pulseFunctionX = Teuchos::rcp(new SpatiallyFilteredFunction<double>(one, pulseX));
  // SpatialFilterPtr lessHalfY = SpatialFilter::lessThanY(0.25);
  // SpatialFilterPtr greaterNegHalfY = SpatialFilter::greaterThanY(-0.25);
  // SpatialFilterPtr pulseY = lessHalfY & greaterNegHalfY;
  // FunctionPtr pulseFunctionY = Teuchos::rcp(new SpatiallyFilteredFunction<double>(one, pulseY));

  LinearTermPtr forcingTerm = Teuchos::null;
  form.initializeSolution(spaceTimeMeshTopo, k+1, delta_k, norm, forcingTerm);

  MeshPtr mesh = form.solution()->mesh();
  MeshPtr k0Mesh = Teuchos::rcp( new Mesh (spaceTimeMeshTopo->deepCopy(), form.bf(), 1, delta_k) );
  mesh->registerObserver(k0Mesh);

  // Set up boundary conditions
  BCPtr bc = form.solution()->bc();
  VarPtr uhat = form.uhat();
  VarPtr tc = form.tc();
  SpatialFilterPtr initTime = SpatialFilter::matchingT(0);
  if (spaceDim == 1)
  {
    SpatialFilterPtr leftX = SpatialFilter::matchingX(-1);
    SpatialFilterPtr rightX = SpatialFilter::matchingX(0);
    bc->addDirichlet(tc, leftX, exactMap[form.tc()->ID()]);
    bc->addDirichlet(uhat, rightX, exactMap[form.uhat()->ID()]);
    bc->addDirichlet(tc, initTime, exactMap[form.tc()->ID()]);
  }
  else if (spaceDim == 2)
  {
    // SpatialFilterPtr leftX = SpatialFilter::matchingX(-1);
    // SpatialFilterPtr rightX = SpatialFilter::matchingX(0);
    // SpatialFilterPtr leftY = SpatialFilter::matchingY(-1);
    // SpatialFilterPtr rightY = SpatialFilter::matchingY(0);
    // bc->addDirichlet(uhat, leftX, uExact2D);
    // bc->addDirichlet(uhat, rightX, uExact2D);
    // bc->addDirichlet(uhat, leftY, uExact2D);
    // bc->addDirichlet(uhat, rightY, uExact2D);
    // bc->addDirichlet(tc, initTime, -uExact2D);
  }

  // Set up solution
  // SolutionPtr soln = Solution::solution(form.bf(), mesh, bc, rhs, form.ip(norm));
  // SolutionPtr soln = Solution::solution(form.bf(), mesh, bc, rhs, form.bf()->graphNorm());
  SolutionPtr soln = form.solution();

  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);

  string outputDir = "/tmp";
  ostringstream solnName;
  solnName << "spacetimeConfusion" << spaceDim << "D_" << norm << "_" << epsilon << "_k" << k << "_" << solverChoice;
  HDF5Exporter exporter(mesh,solnName.str(), outputDir);

  Teuchos::RCP<Time> solverTime = Teuchos::TimeMonitor::getNewCounter("Solve Time");

  if (commRank == 0)
    Solver::printAvailableSolversReport();
  map<string, SolverPtr> solvers;
  solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
  // SolverPtr superluSolver = Solver::getSolver(Solver::SuperLUDist, true);
  int maxIters = 2000;
  bool useStaticCondensation = false;
  int azOutput = 20; // print residual every 20 CG iterations

  // ofstream dataFile(solnName.str()+".txt");
  // dataFile << "ref\t " << "elements\t " << "dofs\t " << "error\t " << "solvetime\t" << "iterations\t " << endl;
  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {
    solverTime->start(true);
    Teuchos::RCP<GMGSolver> gmgSolver;
    if (solverChoice[0] == 'G')
    {
      gmgSolver = Teuchos::rcp( new GMGSolver(soln, k0Mesh, maxIters, solverTolerance, solvers["KLU"], useStaticCondensation));
      gmgSolver->setAztecOutput(azOutput);
      if (solverChoice == "GMG-Direct")
        gmgSolver->gmgOperator().setSchwarzFactorizationType(GMGOperator::Direct);
      if (solverChoice == "GMG-ILU")
        gmgSolver->gmgOperator().setSchwarzFactorizationType(GMGOperator::ILU);
      if (solverChoice == "GMG-IC")
        gmgSolver->gmgOperator().setSchwarzFactorizationType(GMGOperator::IC);
      soln->solve(gmgSolver);
    }
    else
      soln->condensedSolve(solvers[solverChoice]);
    double solveTime = solverTime->stop();

    double energyError = soln->energyErrorTotal();
    if (commRank == 0)
    {
      // if (refIndex > 0)
      // refStrategy.printRefinementStatistics(refIndex-1);
      if (solverChoice[0] == 'G')
      {
        cout << "Refinement: " << refIndex
             << " \tElements: " << mesh->numActiveElements()
             << " \tDOFs: " << mesh->numGlobalDofs()
             << " \tEnergy Error: " << energyError
             << " \tSolve Time: " << solveTime
             << " \tIteration Count: " << gmgSolver->iterationCount()
             << endl;
        // dataFile << refIndex
        //          << " " << mesh->numActiveElements()
        //          << " " << mesh->numGlobalDofs()
        //          << " " << energyError
        //          << " " << solveTime
        //          << " " << gmgSolver->iterationCount()
        //          << endl;
      }
      else
      {
        cout << "Refinement: " << refIndex
             << " \tElements: " << mesh->numActiveElements()
             << " \tDOFs: " << mesh->numGlobalDofs()
             << " \tEnergy Error: " << energyError
             << " \tSolve Time: " << solveTime
             << endl;
        // dataFile << refIndex
        //          << " " << mesh->numActiveElements()
        //          << " " << mesh->numGlobalDofs()
        //          << " " << energyError
        //          << " " << solveTime
        //          << endl;
      }
    }

    exporter.exportSolution(soln, refIndex);

    if (refIndex != numRefs)
      refStrategy.refine();
  }
  // dataFile.close();

  return 0;
}
