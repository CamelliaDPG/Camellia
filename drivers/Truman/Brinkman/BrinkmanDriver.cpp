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
#include "TrigFunctions.h"

#include "RefinementStrategy.h"

/*
Brinkman Equations
-mu * Delta u1 + dp/dx + mu * k * u1 = f1
-mu * Delta u2 + dp/dy + mu * k * u2 = f2
Div u = 0

First order system
sigma1/mu - Grad u1 = 0
sigma2/mu - Grad u2 = 0
-Div(sigma1) + dp/dx + mu * k * u1 = f1
-Div(sigma2) + dp/dy + mu * k * u2 = f2
Div u = 0

Variational Formulation
1/mu * (sigma1,tau1) - (Grad u1, tau1)
1/mu * (sigma2,tau2) - (Grad u2, tau2)
-(Div(sigma1 - (p,0)),v1) + mu * k * (u1,v1) = (f1,v1)
-(Div(sigma2 - (0,p)),v2) + mu * k * (u2,v2) = (f2,v2)
(Div(u),q)

Ultra-weak formulation
1/mu * (sigma1, tau1) + (u1,Div(tau1)) - <u1,tau1_n>
1/mu * (sigma2, tau2) + (u2,Div(tau2)) - <u2,tau2_n>
(sigma1 - (p,0), Grad v1) - <(sigma1 - (p,0))*n, v1> + mu * k * (u1,v1) = (f1,v1)
(sigma2 - (0,p), Grad v2) - <(sigma2 - (0,p))*n, v2> + mu * k * (u2,v2) = (f2,v2)
-(u,Grad(q)) + <u, q_n>
*/

using namespace Camellia;

double pi = 2.0*acos(0.0);

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
  double mu = 0.1;
  double permCoef = 1e4;
  int numRefs = 0;
  int k = 2, delta_k = 2;
  string norm = "Graph";
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("mu", &mu, "mu");
  cmdp.setOption("permCoef", &permCoef, "Permeability coefficient");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  FunctionPtr zero = TFunction<double>::zero();
  FunctionPtr one = TFunction<double>::constant(1);
  FunctionPtr sin2pix = Teuchos::rcp( new Sin_ax(2*pi) );
  FunctionPtr cos2pix = Teuchos::rcp( new Cos_ax(2*pi) );
  FunctionPtr sin2piy = Teuchos::rcp( new Sin_ay(2*pi) );
  FunctionPtr cos2piy = Teuchos::rcp( new Cos_ay(2*pi) );
  FunctionPtr u1_exact = sin2pix*cos2piy;
  FunctionPtr u2_exact = -cos2pix*sin2piy;
  FunctionPtr x2 = TFunction<double>::xn(2);
  FunctionPtr y2 = TFunction<double>::yn(2);
  FunctionPtr p_exact = x2*y2 - 1./9;
  FunctionPtr permInv = permCoef*(sin2pix + 1.1);

  VarFactoryPtr vf = VarFactory::varFactory();
  //fields:
  VarPtr sigma1 = vf->fieldVar("sigma1", VECTOR_L2);
  VarPtr sigma2 = vf->fieldVar("sigma2", VECTOR_L2);
  VarPtr u1 = vf->fieldVar("u1", L2);
  VarPtr u2 = vf->fieldVar("u2", L2);
  VarPtr p = vf->fieldVar("p", L2);

  // traces:
  VarPtr u1hat = vf->traceVar("u1hat");
  VarPtr u2hat = vf->traceVar("u2hat");
  VarPtr t1c = vf->fluxVar("t1c");
  VarPtr t2c = vf->fluxVar("t2c");

  // test:
  VarPtr v1 = vf->testVar("v1", HGRAD);
  VarPtr v2 = vf->testVar("v2", HGRAD);
  VarPtr tau1 = vf->testVar("tau1", HDIV);
  VarPtr tau2 = vf->testVar("tau2", HDIV);
  VarPtr q = vf->testVar("q", HGRAD);

  BFPtr bf = Teuchos::rcp( new BF(vf) );

  bf->addTerm(1./mu*sigma1, tau1);
  bf->addTerm(1./mu*sigma2, tau2);
  bf->addTerm(u1, tau1->div());
  bf->addTerm(u2, tau2->div());
  bf->addTerm(-u1hat, tau1->dot_normal());
  bf->addTerm(-u2hat, tau2->dot_normal());

  bf->addTerm(sigma1, v1->grad());
  bf->addTerm(sigma2, v2->grad());
  bf->addTerm(-p, v1->dx());
  bf->addTerm(-p, v2->dy());
  bf->addTerm(t1c, v1);
  bf->addTerm(t2c, v2);
  bf->addTerm(mu*permInv*u1, v1);
  bf->addTerm(mu*permInv*u2, v2);

  bf->addTerm(-u1, q->dx());
  bf->addTerm(-u2, q->dy());
  bf->addTerm(u1hat, q->times_normal_x());
  bf->addTerm(u2hat, q->times_normal_y());

  RHSPtr rhs = RHS::rhs();

  BCPtr bc = BC::bc();

  SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
  SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(1.0);
  SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
  bc->addDirichlet(u1hat, y_equals_zero, u1_exact);
  bc->addDirichlet(u2hat, y_equals_zero, u2_exact);
  bc->addDirichlet(u1hat, x_equals_zero, u1_exact);
  bc->addDirichlet(u2hat, x_equals_zero, u2_exact);
  bc->addDirichlet(u1hat, y_equals_one, u1_exact);
  bc->addDirichlet(u2hat, y_equals_one, u2_exact);
  bc->addDirichlet(u1hat, x_equals_one, u1_exact);
  bc->addDirichlet(u2hat, x_equals_one, u2_exact);
  bc->addZeroMeanConstraint(p);

  MeshPtr mesh = MeshFactory::quadMesh(bf, k+1, delta_k, 1, 1, 4, 4);

  map<string, IPPtr> brinkmanIPs;
  brinkmanIPs["Graph"] = bf->graphNorm();

  brinkmanIPs["Decoupled"] = Teuchos::rcp(new IP);
  brinkmanIPs["Decoupled"]->addTerm(tau1);
  brinkmanIPs["Decoupled"]->addTerm(tau2);
  brinkmanIPs["Decoupled"]->addTerm(tau1->div());
  brinkmanIPs["Decoupled"]->addTerm(tau2->div());
  brinkmanIPs["Decoupled"]->addTerm(permInv*v1);
  brinkmanIPs["Decoupled"]->addTerm(permInv*v2);
  brinkmanIPs["Decoupled"]->addTerm(v1->grad());
  brinkmanIPs["Decoupled"]->addTerm(v2->grad());
  brinkmanIPs["Decoupled"]->addTerm(q);
  brinkmanIPs["Decoupled"]->addTerm(q->grad());

  // brinkmanIPs["CoupledRobust"] = Teuchos::rcp(new IP);
  // brinkmanIPs["CoupledRobust"]->addTerm(tau->div()-beta*v->grad());
  // brinkmanIPs["CoupledRobust"]->addTerm(Function<double>::min(one/Function<double>::h(),Function<double>::constant(1./sqrt(epsilon)))*tau);
  // brinkmanIPs["CoupledRobust"]->addTerm(sqrt(epsilon)*v->grad());
  // brinkmanIPs["CoupledRobust"]->addTerm(beta*v->grad());
  // brinkmanIPs["CoupledRobust"]->addTerm(Function<double>::min(sqrt(epsilon)*one/Function<double>::h(),one)*v);

  IPPtr ip = brinkmanIPs[norm];

  SolutionPtr soln = TSolution<double>::solution(mesh, bc, rhs, ip);

  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);

  ostringstream refName;
  refName << "brinkman";
  HDF5Exporter exporter(mesh,refName.str());

  for (int refIndex=0; refIndex <= numRefs; refIndex++)
  {
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
