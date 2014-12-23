#include <iostream>

#include "Solution.h"

#include "MeshUtilities.h"
#include "MeshFactory.h"

#include "GnuPlotUtil.h"
#include <Teuchos_GlobalMPISession.hpp>

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
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif

  int commRank = Teuchos::GlobalMPISession::getRank();

  Comm.Barrier(); // set breakpoint here to allow debugger attachment to other MPI processes than the one you automatically attached to.

  VarFactory vf;
  //fields:
  VarPtr sigma = vf.fieldVar("sigma", VECTOR_L2);
  VarPtr u = vf.fieldVar("u", L2);

  // traces:
  VarPtr u_hat = vf.traceVar("u_hat");
  VarPtr t_n = vf.fluxVar("t_n");

  // test:
  VarPtr v = vf.testVar("v", HGRAD);
  VarPtr tau = vf.testVar("tau", HDIV);

  double eps = 1e-4;
  FunctionPtr beta_x = Function::constant(0);
  FunctionPtr beta_y = Function::constant(0);
  FunctionPtr beta = Function::vectorize(beta_x, beta_y);

  BFPtr bf = Teuchos::rcp( new BF(vf) );

  bf->addTerm((1/eps) * sigma, tau);
  bf->addTerm(u, tau->div());
  bf->addTerm(-u_hat, tau->dot_normal());

  bf->addTerm(sigma - beta * u, v->grad());
  bf->addTerm(t_n, v);

  RHSPtr rhs = RHS::rhs();

  BCPtr bc = BC::bc();

  SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0.0);
  SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(1.0);
  SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
  SpatialFilterPtr x_less_half = SpatialFilter::lessThanX(0.5);
  SpatialFilterPtr x_greater_half = SpatialFilter::greaterThanX(0.5);
  SpatialFilterPtr high_density_bottom = SpatialFilter::intersectionFilter(y_equals_zero, x_greater_half);
  SpatialFilterPtr low_density_bottom = SpatialFilter::intersectionFilter(y_equals_zero, x_less_half);
  SpatialFilterPtr high_density_top = SpatialFilter::intersectionFilter(y_equals_one, x_greater_half);
  SpatialFilterPtr low_density_top = SpatialFilter::intersectionFilter(y_equals_one, x_less_half);

  FunctionPtr zero = Function::zero();
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr one = Function::constant(1);
  // bc->addDirichlet(u_hat, x_greater_half, 10*one);
  bc->addDirichlet(u_hat, high_density_bottom, 10*one);
  bc->addDirichlet(u_hat, low_density_bottom, one);
  // bc->addDirichlet(u_hat, y_equals_zero, one+9*x);
  bc->addDirichlet(u_hat, low_density_top, one);
  bc->addDirichlet(u_hat, high_density_top, 10*one);
  // bc->addDirichlet(u_hat, y_equals_one, one+9*x);
  bc->addDirichlet(u_hat, x_equals_zero, one);
  bc->addDirichlet(u_hat, x_equals_one, 10*one);
  // bc->addDirichlet(t_n, y_equals_zero, -one);
  // bc->addDirichlet(t_n, high_density_region, -10*one);
  // bc->addDirichlet(t_n, low_density_region, -1*one);
  // bc->addDirichlet(t_n, x_equals_zero, -1 * (1-y));
  // bc->addDirichlet(u_hat, y_equals_one, zero);
  // bc->addDirichlet(u_hat, x_equals_zero, one*(1-y));
  // bc->addDirichlet(u_hat, x_equals_one, one*(1-y));
  // bc->addDirichlet(t_n, x_equals_zero, zero);
  // bc->addDirichlet(t_n, x_equals_one, zero);

  int k = 2;
  int delta_k = 2;
  MeshPtr mesh = MeshFactory::quadMesh(bf, k+1, delta_k);

  IPPtr ip = bf->graphNorm();

  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);

  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);

  int numRefs = 4;

  ostringstream refName;
  refName << "confusion";
  HDF5Exporter exporter(mesh,refName.str());

  for (int refIndex=0; refIndex <= numRefs; refIndex++) {
    soln->solve();

    double energyError = soln->energyErrorTotal();
    if (commRank == 0)
      cout << "After " << refIndex << " refinements, energy error is " << energyError << endl;

    exporter.exportSolution(soln, vf, refIndex);

    if (refIndex != numRefs)
      refStrategy.refine();
  }

  return 0;
}
