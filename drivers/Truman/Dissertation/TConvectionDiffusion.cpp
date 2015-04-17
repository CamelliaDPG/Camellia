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

using namespace Camellia;

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
  double epsilon = 1e-2;
  int numRefs = 0;
  int k = 2, delta_k = 2;
  string norm = "CoupledRobust";
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("epsilon", &epsilon, "epsilon");
  cmdp.setOption("norm", &norm, "norm");

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

  FunctionPtr beta_x = Function<double>::constant(1);
  FunctionPtr beta_y = Function<double>::constant(2);
  FunctionPtr beta = Function<double>::vectorize(beta_x, beta_y);

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

  FunctionPtr zero = Function<double>::zero();
  FunctionPtr one = Function<double>::constant(1);
  FunctionPtr x = Function<double>::xn(1);
  FunctionPtr y = Function<double>::yn(1);
  bc->addDirichlet(tc, y_equals_zero, -2. * (one-x));
  bc->addDirichlet(tc, x_equals_zero, -1. * (one-y));
  bc->addDirichlet(uhat, y_equals_one, zero);
  bc->addDirichlet(uhat, x_equals_one, zero);

  MeshPtr mesh = MeshFactory::quadMesh(bf, k+1, delta_k);

  map<string, IPPtr> confusionIPs;
  confusionIPs["Graph"] = bf->graphNorm();

  confusionIPs["Robust"] = Teuchos::rcp(new IP);
  confusionIPs["Robust"]->addTerm(tau->div());
  confusionIPs["Robust"]->addTerm(beta*v->grad());
  confusionIPs["Robust"]->addTerm(Function<double>::min(one/Function<double>::h(),Function<double>::constant(1./sqrt(epsilon)))*tau);
  confusionIPs["Robust"]->addTerm(sqrt(epsilon)*v->grad());
  confusionIPs["Robust"]->addTerm(beta*v->grad());
  confusionIPs["Robust"]->addTerm(Function<double>::min(sqrt(epsilon)*one/Function<double>::h(),one)*v);

  confusionIPs["CoupledRobust"] = Teuchos::rcp(new IP);
  confusionIPs["CoupledRobust"]->addTerm(tau->div()-beta*v->grad());
  confusionIPs["CoupledRobust"]->addTerm(Function<double>::min(one/Function<double>::h(),Function<double>::constant(1./sqrt(epsilon)))*tau);
  confusionIPs["CoupledRobust"]->addTerm(sqrt(epsilon)*v->grad());
  confusionIPs["CoupledRobust"]->addTerm(beta*v->grad());
  confusionIPs["CoupledRobust"]->addTerm(Function<double>::min(sqrt(epsilon)*one/Function<double>::h(),one)*v);

  IPPtr ip = confusionIPs[norm];

  SolutionPtr soln = Solution<double>::solution(mesh, bc, rhs, ip);

  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);

  ostringstream refName;
  refName << "confusion";
  HDF5Exporter exporter(mesh,refName.str());

  for (int refIndex=0; refIndex <= numRefs; refIndex++) {
    soln->solve(false);

    double energyError = soln->energyErrorTotal();
    if (commRank == 0)
    {
      // if (refIndex > 0)
        // refStrategy.printRefinementStatistics(refIndex-1);
      cout << "Refinement:\t " << refIndex << " \tElements:\t " << mesh->numActiveElements()
        << " \tDOFs:\t " << mesh->numGlobalDofs() << " \tEnergy Error:\t " << energyError << endl;
    }

    exporter.exportSolution(soln, refIndex);

    if (refIndex != numRefs)
      refStrategy.refine();
  }

  return 0;
}
