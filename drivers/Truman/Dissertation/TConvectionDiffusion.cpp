#include "Solution.h"

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

#include "RefinementStrategy.h"

int main(int argc, char *argv[]) {
  
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
  double epsilon = 1e-8;
  int numRefs = 10;
  int k = 2, delta_k = 2;
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("epsilon", &epsilon, "epsilon");
  
  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  
  VarFactory vf;
  //fields:
  VarPtr sigma = vf.fieldVar("sigma", VECTOR_L2);
  VarPtr u = vf.fieldVar("u", L2);
  
  // traces:
  VarPtr uhat = vf.traceVar("uhat");
  VarPtr tc = vf.fluxVar("tc");
  
  // test:
  VarPtr v = vf.testVar("v", HGRAD);
  VarPtr tau = vf.testVar("tau", HDIV);
  
  FunctionPtr beta_x = Function::constant(1);
  FunctionPtr beta_y = Function::constant(2);
  FunctionPtr beta = Function::vectorize(beta_x, beta_y);
  
  BFPtr bf = Teuchos::rcp( new BF(vf) );
  
  bf->addTerm((1/epsilon) * sigma, tau);
  bf->addTerm(u, tau->div());
  bf->addTerm(-uhat, tau->dot_normal());
  
  bf->addTerm(sigma - beta * u, v->grad());
  bf->addTerm(tc, v);

  RHSPtr rhs = RHS::rhs();

  BCPtr bc = BC::bc();
  
  SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
  SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(1.0);
  SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
  
  FunctionPtr zero = Function::zero();
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  bc->addDirichlet(tc, y_equals_zero, -2 * (1-x));
  bc->addDirichlet(tc, x_equals_zero, -1 * (1-y));
  bc->addDirichlet(uhat, y_equals_one, zero);
  bc->addDirichlet(uhat, x_equals_one, zero);
  
  MeshPtr mesh = MeshFactory::quadMesh(bf, k+1, delta_k);
  
  IPPtr ip = bf->graphNorm();
  
  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);
  
  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);
  
  ostringstream refName;
  refName << "confusion";
  HDF5Exporter exporter(mesh,refName.str());
  
  for (int refIndex=0; refIndex < numRefs; refIndex++) {
    soln->solve();
    
    double energyError = soln->energyErrorTotal();
    if (commRank == 0)
      cout << "After " << refIndex << " refinements, energy error is " << energyError << endl;
    
    exporter.exportSolution(soln, refIndex);
    
    if (refIndex != numRefs)
      refStrategy.refine();
  }
  
  return 0;
}
