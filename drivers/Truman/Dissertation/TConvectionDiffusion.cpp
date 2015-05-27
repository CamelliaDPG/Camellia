#include "Solution.h"
#include "RHS.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"

#include "EpetraExt_ConfigDefs.h"
#ifdef HAVE_EPETRAEXT_HDF5
#include "HDF5Exporter.h"
#endif

#include "BF.h"
#include "Function.h"
#include "RefinementStrategy.h"
#include "GMGSolver.h"
#include "ConvectionDiffusionFormulation.h"

using namespace Camellia;

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
  double epsilon = 1e-2;
  int numRefs = 0;
  int k = 2, delta_k = 2;
  int numXElems = 1;
  bool useConformingTraces = true;
  int solverChoice = 0;
  double solverTolerance = 1e-6;
  string norm = "CoupledRobust";
  cmdp.setOption("spaceDim", &spaceDim, "spatial dimension");
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("numXElems",&numXElems,"number of elements in x direction");
  cmdp.setOption("epsilon", &epsilon, "epsilon");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("conformingTraces", "nonconformingTraces", &useConformingTraces, "use conforming traces");
  cmdp.setOption("solver", &solverChoice, "0=iterative, 1=KLI, 2=SuperLu");
  cmdp.setOption("solverTolerance", &solverTolerance, "iterative solver tolerance");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

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

  ConvectionDiffusionFormulation form(spaceDim, useConformingTraces, beta, epsilon);

  // Define right hand side
  RHSPtr rhs = RHS::rhs();

  // Set up boundary conditions
  BCPtr bc = BC::bc();
  VarPtr uhat = form.uhat();
  VarPtr tc = form.tc();
  SpatialFilterPtr inflowX = SpatialFilter::matchingX(-1);
  SpatialFilterPtr inflowY = SpatialFilter::matchingY(-1);
  SpatialFilterPtr inflowZ = SpatialFilter::matchingZ(-1);
  SpatialFilterPtr outflowX = SpatialFilter::matchingX(1);
  SpatialFilterPtr outflowY = SpatialFilter::matchingY(1);
  SpatialFilterPtr outflowZ = SpatialFilter::matchingZ(1);
  FunctionPtr zero = Function::zero();
  FunctionPtr one = Function::constant(1);
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  if (spaceDim == 1)
  {
    bc->addDirichlet(tc, inflowX, -one);
    bc->addDirichlet(uhat, outflowX, zero);
  }
  if (spaceDim == 2)
  {
    bc->addDirichlet(tc, inflowX, -1*.5*(one-y));
    bc->addDirichlet(uhat, outflowX, zero);
    bc->addDirichlet(tc, inflowY, -2*.5*(one-x));
    bc->addDirichlet(uhat, outflowY, zero);
  }
  if (spaceDim == 3)
  {
    bc->addDirichlet(tc, inflowX, -1*.25*(one-y)*(one-z));
    bc->addDirichlet(uhat, outflowX, zero);
    bc->addDirichlet(tc, inflowY, -2*.25*(one-x)*(one-z));
    bc->addDirichlet(uhat, outflowY, zero);
    bc->addDirichlet(tc, inflowZ, -3*.25*(one-x)*(one-y));
    bc->addDirichlet(uhat, outflowZ, zero);
  }

  // Build mesh
  vector<double> x0 = vector<double>(spaceDim,-1.0);
  double width = 2.0;
  vector<double> dimensions;
  vector<int> elementCounts;
  for (int d=0; d<spaceDim; d++)
  {
    dimensions.push_back(width);
    elementCounts.push_back(numXElems);
  }
  MeshPtr mesh = MeshFactory::rectilinearMesh(form.bf(), dimensions, elementCounts, k+1, delta_k, x0);
  MeshPtr k0Mesh = Teuchos::rcp( new Mesh (mesh->getTopology()->deepCopy(), form.bf(), 1, delta_k) );
  mesh->registerObserver(k0Mesh);

  // Set up solution
  SolutionPtr soln = Solution::solution(form.bf(), mesh, bc, rhs, form.ip(norm));

  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);

  ostringstream refName;
  refName << "confusion" << spaceDim << "D_" << norm << "_" << epsilon << "_k" << k;
  // HDF5Exporter exporter(mesh,refName.str());

  SolverPtr kluSolver = Solver::getSolver(Solver::KLU, true);
  SolverPtr superluSolver = Solver::getSolver(Solver::SuperLUDist, true);
  int maxIters = 10000;
  bool useStaticCondensation = false;
  int azOutput = 20; // print residual every 20 CG iterations

  ofstream dataFile(refName.str()+".txt");
  dataFile << "ref\t " << "iterations\t " << "elements\t " << "dofs\t " << "error\t " << endl;
  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {
    Teuchos::RCP<GMGSolver> gmgSolver;
    if (solverChoice == 0)
    {
      gmgSolver = Teuchos::rcp( new GMGSolver(soln, k0Mesh, maxIters, solverTolerance, kluSolver, useStaticCondensation));
      gmgSolver->setAztecOutput(azOutput);
    }
    switch(solverChoice)
    {
    case 0:
      soln->solve(gmgSolver);
      break;
    case 1:
      soln->condensedSolve(kluSolver);
      break;
    case 2:
      soln->condensedSolve(superluSolver);
      break;
    }

    double energyError = soln->energyErrorTotal();
    if (commRank == 0)
    {
      // if (refIndex > 0)
      // refStrategy.printRefinementStatistics(refIndex-1);
      if (solverChoice == 0)
      {
        cout << "Refinement:\t " << refIndex
             << " \tIteration Count:\t " << gmgSolver->iterationCount()
             << " \tElements:\t " << mesh->numActiveElements()
             << " \tDOFs:\t " << mesh->numGlobalDofs()
             << " \tEnergy Error:\t " << energyError << endl;
        dataFile << refIndex
                 << " " << gmgSolver->iterationCount()
                 << " " << mesh->numActiveElements()
                 << " " << mesh->numGlobalDofs()
                 << " " << energyError << endl;
      }
      else
      {
        cout << "Refinement:\t " << refIndex
             << " \tElements:\t " << mesh->numActiveElements()
             << " \tDOFs:\t " << mesh->numGlobalDofs()
             << " \tEnergy Error:\t " << energyError << endl;
      }
    }

    // exporter.exportSolution(soln, refIndex);

    if (refIndex != numRefs)
      refStrategy.refine();
  }
  dataFile.close();

  return 0;
}
