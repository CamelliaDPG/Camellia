#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

#include "EpetraExt_RowMatrixOut.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int numElements = 3;
  vector<vector<double>> domainDim(3,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
  int polyOrder = 2, delta_k = 1;
  int spaceDim = 2;
  
  cmdp.setOption("numElements", &numElements );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("delta_k", &delta_k );
  cmdp.setOption("x0", &domainDim[0][0] );
  cmdp.setOption("x1", &domainDim[0][1] );
  cmdp.setOption("y0", &domainDim[1][0] );
  cmdp.setOption("y1", &domainDim[1][1] );
  cmdp.setOption("z0", &domainDim[2][0] );
  cmdp.setOption("z1", &domainDim[2][1] );
  cmdp.setOption("spaceDim", &spaceDim);
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  vector<double> x0(spaceDim);
  vector<double> domainSize(spaceDim);
  vector<int> elementCounts(spaceDim);
  for (int d=0; d<spaceDim; d++)
  {
    x0[d] = domainDim[d][0];
    domainSize[d] = domainDim[d][1] - x0[d];
    elementCounts[d] = numElements;
  }
  
  bool conformingTraces = true; // no difference for primal/continuous formulations
  PoissonFormulation formCG(spaceDim, conformingTraces, PoissonFormulation::CONTINUOUS_GALERKIN);
  VarPtr q = formCG.q();
  VarPtr phi = formCG.phi();
  BFPtr bf = formCG.bf();
  
  MeshPtr bubnovMesh = MeshFactory::rectilinearMesh(bf, domainSize, elementCounts, polyOrder, 0, x0);

  // Right now, hanging nodes don't work with continuous field variables
  // there is a GDAMinimumRule test demonstrating the failure, SolvePoisson2DContinuousGalerkinHangingNode.
  // make a mesh with hanging nodes (when spaceDim > 1)
//  {
//    set<GlobalIndexType> cellsToRefine = {0};
//    bubnovMesh->hRefine(cellsToRefine);
//  }

  RHSPtr rhs = RHS::rhs();
  rhs->addTerm(1.0 * q); // unit forcing
  
  IPPtr ip = Teuchos::null; // will give Bubnov-Galerkin
  BCPtr bc = BC::bc();
  bc->addDirichlet(phi, SpatialFilter::allSpace(), Function::zero());
  
  SolutionPtr solution = Solution::solution(bf, bubnovMesh, bc, rhs, ip);
  solution->solve();
  
  HDF5Exporter exporter(bubnovMesh, "PoissonContinuousGalerkin");
  exporter.exportSolution(solution);

  /**** Sort-of-primal experiment ****/
  // an experiment: try doing "primal" DPG with IBP to the boundary
//  ip = IP::ip();
//  ip->addTerm(q->grad());
//  ip->addTerm(q);
//
//  solution = Solution::solution(bf, bubnovMesh, bc, rhs, ip);
//  solution->solve();
//  
//  HDF5Exporter primalNoFluxExporter(bubnovMesh, "PoissonPrimalNoFlux");
//  primalNoFluxExporter.exportSolution(solution);
  
  //*** Primal Formulation ***//
  PoissonFormulation form(spaceDim, conformingTraces, PoissonFormulation::PRIMAL);
  q = form.q();
  phi = form.phi();
  bf = form.bf();
  
  bc = BC::bc();
  bc->addDirichlet(phi, SpatialFilter::allSpace(), Function::zero());
  
  rhs = RHS::rhs();
  rhs->addTerm(1.0 * q); // unit forcing
  
  MeshPtr primalMesh = MeshFactory::rectilinearMesh(bf, domainSize, elementCounts, polyOrder, delta_k, x0);
  
  ip = IP::ip();
  ip->addTerm(q->grad());
  ip->addTerm(q);

  // Right now, hanging nodes don't work with continuous field variables
  // there is a GDAMinimumRule test demonstrating the failure, SolvePoisson2DContinuousGalerkinHangingNode.
  // make a mesh with hanging nodes (when spaceDim > 1)
//  {
//    set<GlobalIndexType> cellsToRefine = {0};
//    primalMesh->hRefine(cellsToRefine);
//  }
  
  solution = Solution::solution(bf, primalMesh, bc, rhs, ip);
  solution->solve();
 
  HDF5Exporter primalExporter(primalMesh, "PoissonPrimal");
  primalExporter.exportSolution(solution);
  
  return 0;
}