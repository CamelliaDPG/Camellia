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
  double xLeft = 0.0, xRight = 1.0;
  int polyOrder = 1, delta_k = 1;
  
  cmdp.setOption("numElements", &numElements );
  cmdp.setOption("polyOrder", &polyOrder );
  cmdp.setOption("delta_k", &delta_k );
  cmdp.setOption("xLeft", &xLeft );
  cmdp.setOption("xRight", &xRight );
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  int spaceDim = 1;
  bool conformingTraces = true; // conformingTraces argument has no effect in 1D
  PoissonFormulation poissonForm(spaceDim, conformingTraces);
  
  MeshPtr mesh = MeshFactory::intervalMesh(poissonForm.bf(), xLeft, xRight, numElements, polyOrder + 1, delta_k); // 1D equispaced
  
  RHSPtr rhs = RHS::rhs(); // zero RHS
  IPPtr ip = poissonForm.bf()->graphNorm();
  BCPtr bc = BC::bc();
  bc->addDirichlet(poissonForm.phi_hat(), SpatialFilter::allSpace(), Function::zero());
  
  SolutionPtr solution = Solution::solution(poissonForm.bf(), mesh, bc, rhs, ip);
  solution->solve();
  
  GDAMinimumRule* minRule = dynamic_cast<GDAMinimumRule*>(mesh->globalDofAssignment().get());
//  minRule->printGlobalDofInfo();
  
  Teuchos::RCP<Epetra_CrsMatrix> A = solution->getStiffnessMatrix();
  EpetraExt::RowMatrixToMatrixMarketFile("A.dat",*A, NULL, NULL, false);
  
  HDF5Exporter exporter(mesh);
  
  return 0;
}