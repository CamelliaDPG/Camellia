//
// © 2016-2018 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//

#include "GDAMinimumRule.h"
#include "HDF5Exporter.h"
#include "MeshFactory.h"
#include "PoissonFormulation.h"
#include "RHS.h"
#include "Solution.h"

using namespace Camellia;
using namespace std;

int main(int argc, char *argv[])
{
  Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL); // initialize MPI
  int rank = Teuchos::GlobalMPISession::getRank();
  
  Teuchos::CommandLineProcessor cmdp(false,true); // false: don't throw exceptions; true: do return errors for unrecognized options
  
  int spaceDim = 2;
  int numElements = 30;
  vector<vector<double>> domainDim(spaceDim,vector<double>{0.0,1.0}); // first index: spaceDim; second: 0/1 for x0, x1, etc.
  int polyOrder = 3;
  
  cmdp.setOption("meshWidth", &numElements );
  cmdp.setOption("polyOrder", &polyOrder );
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
  
  vector<double> x0(spaceDim);
  vector<double> domainSize(spaceDim);
  vector<int> elementCounts(spaceDim);
  for (int d=0; d<spaceDim; d++)
  {
    x0[d] = domainDim[d][0];
    domainSize[d] = domainDim[d][1] - x0[d];
    elementCounts[d] = numElements;
  }
  
  bool conformingTraces = false; // use L^2 traces even for traces of H^1
  PoissonFormulation form(spaceDim, conformingTraces, PoissonFormulation::ULTRAWEAK);
  VarPtr q = form.v();
  VarPtr u = form.u();
  VarPtr u_hat = form.u_hat();
  BFPtr bf = form.bf();
  
  int pToAddTest = spaceDim;
  double width = 1.0, height = 1.0;
  int horizontalElements = numElements, verticalElements = numElements;
  bool divideIntoTriangles = false;
  MeshPtr mesh = MeshFactory::quadMeshMinRule(bf, polyOrder, pToAddTest,
                                              width, height,
                                              horizontalElements, verticalElements,
                                              divideIntoTriangles);
  
  RHSPtr rhs = RHS::rhs();
  FunctionPtr f = Function::constant(1.0); // unit forcing
  
  rhs->addTerm(f * q); // unit forcing
  
  IPPtr ip = IP::ip();
//  ip = form.bf()->graphNorm();
  {
    // construct ip that takes traces on the mesh skeleton, but is otherwise like the standard graph norm
    VarPtr v = form.v();
    VarPtr tau = form.tau();
    ip->addTerm(tau->div());
    ip->addTerm(tau - v->grad());

    auto trace = Function::meshSkeletonCharacteristic(); // skeleton restriction
    ip->addTerm(tau * trace);
    ip->addTerm(v * trace);
  }
  BCPtr bc = BC::bc();
  
  bc->addDirichlet(u_hat, SpatialFilter::allSpace(), Function::zero());
  
  SolutionPtr solution = Solution::solution(bf, mesh, bc, rhs, ip);
  
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
  
  return 0;
}
