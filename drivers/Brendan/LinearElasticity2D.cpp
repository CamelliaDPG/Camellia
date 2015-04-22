//  LinearElasticity2D.cpp
//  Driver for Linear elasticity with weakly imposed (H(div))^3 symmetry
//  Camellia
//
//  Created by Brendan Keith, April 2015.

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

#include "RefinementStrategy.h"

int kronDelta(int i, int j) {
  return (i == j) ? 1 : 0;
}

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
  // double lambda = 123;
  // double mu = 79.3;
  double lambda = 1;
  double mu = 1;
  int numRefs = 10;
  int k = 2, delta_k = 2;
  string norm = "Graph";
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("lambda", &lambda, "lambda");
  cmdp.setOption("mu", &mu, "mu");
  cmdp.setOption("norm", &norm, "norm");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  //////////////////////   DECLARE VARIABLES   ////////////////////////
  VarFactory vf;
  // trials:
  VarPtr w       = vf.fieldVar("w", L2);
  VarPtr u       = vf.fieldVar("u", VECTOR_L2);
  VarPtr sigma11 = vf.fieldVar("\\sigma_{11}", L2);
  VarPtr sigma12 = vf.fieldVar("\\sigma_{12}", L2);
  VarPtr sigma22 = vf.fieldVar("\\sigma_{22}", L2);

  // traces:
  VarPtr u1hat = vf.traceVar("\\hat{u_1}");
  VarPtr u2hat = vf.traceVar("\\hat{u_2}");
  VarPtr t1hat = vf.fluxVar("\\hat{t_1}");
  VarPtr t2hat = vf.fluxVar("\\hat{t_2}");

  // tests:
  VarPtr tau1 = vf.testVar("\\tau_1", HDIV);
  VarPtr tau2 = vf.testVar("\\tau_2", HDIV);
  VarPtr v1   = vf.testVar("v_1", HGRAD);
  VarPtr v2   = vf.testVar("v_2", HGRAD);

  ////////////////    MISCELLANEOUS LOCAL VARIABLES    ////////////////

  // Compliance Tensor
  int N = 2;
  double C[2][2][2][2];
  for (int i = 0; i < 2; ++i){
    for (int j = 0; j < 2; ++j){
      for (int k = 0; k < 2; ++k){
        for (int l = 0; l < 2; ++l){
          C[i][j][k][l] = 1/(2*mu)*(0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                        - lambda/(2*mu+N*lambda)*kronDelta(i,j)*kronDelta(k,l));
          // cout << "C(" << i << "," << j << "," << k << "," << l << ") = " << C[i][j][k][l] << endl;
        }
      }
    }
  }

  // Stiffness Tensor
  double E[2][2][2][2];
  for (int i = 0; i < 2; ++i){
    for (int j = 0; j < 2; ++j){
      for (int k = 0; k < 2; ++k){
        for (int l = 0; l < 2; ++l){
          E[i][j][k][l] = (2*mu)*0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                        + lambda*kronDelta(i,j)*kronDelta(k,l);
          // cout << "C(" << i << "," << j << "," << k << "," << l << ") = " << C[i][j][k][l] << endl;
        }
      }
    }
  }

  LinearTermPtr sigma[2][2];
  sigma[0][0] = 1*sigma11;
  sigma[0][1] = 1*sigma12;
  sigma[1][0] = 1*sigma12;
  sigma[1][1] = 1*sigma22;

  LinearTermPtr tau[2][2];
  tau[0][0] = 1*tau1->x();
  tau[0][1] = 1*tau1->y();
  tau[1][0] = 1*tau2->x();
  tau[1][1] = 1*tau2->y();

  FunctionPtr one  = Function::constant(1);
  FunctionPtr zero = Function::zero();
  FunctionPtr x    = Function::xn(1);
  FunctionPtr y    = Function::yn(1);
  FunctionPtr n    = Function::normal();

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(vf) );

  bf->addTerm(u->x(),-tau1->div());
  bf->addTerm(u->y(),-tau2->div());
  bf->addTerm(w,-tau1->y());
  bf->addTerm(w, tau2->x());
  bf->addTerm(u1hat, tau1->dot_normal());
  bf->addTerm(u2hat, tau2->dot_normal());
  for (int i = 0; i < 2; ++i){
    for (int j = 0; j < 2; ++j){
      for (int k = 0; k < 2; ++k){
        for (int l = 0; l < 2; ++l){
          if (abs(C[i][j][k][l])>1e-14){
            bf->addTerm(sigma[k][l],-Function::constant(C[i][j][k][l])*tau[i][j]);
          }
        }
      }
    }
  }

  bf->addTerm(sigma11, v1->dx());
  bf->addTerm(sigma12, v1->dy());
  bf->addTerm(sigma12, v2->dx());
  bf->addTerm(sigma22, v2->dy());
  // omega term missing
  bf->addTerm(t1hat,-v1);
  bf->addTerm(t2hat,-v2);

  // PRINT BILINEAR FORM
  // cout << bf->displayString() << endl;

  ////////////////////   SPECIFY RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs();
  // rhs->addTerm(one*v1);
  // rhs->addTerm(one*v2);

  // A simple rhs from a manufactured solution

  // LinearTermPtr u_exact[2];
  // u_exact[0] = 1*x*y;
  // u_exact[1] = 1*x*y;

  // LinearTermPtr Du_exact[2][2];
  // Du_exact[0][0] = 1*y;
  // Du_exact[0][1] = 1*x;
  // Du_exact[1][0] = 1*y;
  // Du_exact[1][1] = 1*x;

  // LinearTermPtr D2u_exact[2][2][2];
  // D2u_exact[0][0][0] = zero;
  // D2u_exact[0][0][1] = one;
  // D2u_exact[0][1][0] = one;
  // D2u_exact[0][1][1] = zero;
  // D2u_exact[1][0][0] = zero;
  // D2u_exact[1][0][1] = one;
  // D2u_exact[1][1][0] = one;
  // D2u_exact[1][1][1] = zero;

  // LinearTermPtr v[2];
  // v[0] = 1*v1;
  // v[1] = 1*v2;

  // for (int i = 0; i < 2; ++i){
  //   for (int j = 0; j < 2; ++j){
  //     for (int k = 0; k < 2; ++k){
  //       for (int l = 0; l < 2; ++l){
  //         if (abs(E[i][j][k][l])>1e-14){
  //           rhs->addTerm(-Function::constant(E[i][j][k][l])*D2u_exact[k][l][j]*v[i]);
  //         }
  //       }
  //     }
  //   }
  // }

  BCPtr bc = BC::bc();

  SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
  SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(1.0);
  SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);

  bc->addDirichlet(t1hat, y_equals_zero, x);
  bc->addDirichlet(t2hat, y_equals_zero, zero);
  bc->addDirichlet(u1hat, x_equals_zero, zero);
  bc->addDirichlet(u2hat, x_equals_zero, zero);
  bc->addDirichlet(t1hat, y_equals_one,  x);
  bc->addDirichlet(t2hat, y_equals_one,  zero);
  bc->addDirichlet(t1hat, x_equals_one,  one);
  bc->addDirichlet(t2hat, x_equals_one,  zero);
  // bc->addDirichlet(u1hat, y_equals_zero, zero);
  // bc->addDirichlet(u1hat, x_equals_zero, zero);
  // bc->addDirichlet(u1hat, y_equals_one,  x);
  // bc->addDirichlet(u1hat, x_equals_one,  y);
  // bc->addDirichlet(u2hat, y_equals_zero, zero);
  // bc->addDirichlet(u2hat, x_equals_zero, zero);
  // bc->addDirichlet(u2hat, y_equals_one,  x);
  // bc->addDirichlet(u2hat, x_equals_one,  y);

  MeshPtr mesh = MeshFactory::quadMesh(bf, k+1, delta_k);

  map<string, IPPtr> elasticityIPs;
  elasticityIPs["Graph"] = bf->graphNorm();

  // elasticityIPs["Robust"] = Teuchos::rcp(new IP);
  // elasticityIPs["Robust"]->addTerm(tau->div());
  // elasticityIPs["Robust"]->addTerm(beta*v->grad());
  // elasticityIPs["Robust"]->addTerm(min(1./Function::h(),1./sqrt(lambda))*tau);
  // elasticityIPs["Robust"]->addTerm(sqrt(lambda)*v->grad());
  // elasticityIPs["Robust"]->addTerm(beta*v->grad());
  // elasticityIPs["Robust"]->addTerm(min(sqrt(lambda)/Function::h(),Function::constant(1.0))*v);

  IPPtr ip = elasticityIPs[norm];

  SolutionPtr soln = Solution::solution(mesh, bc, rhs, ip);

  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);

  ostringstream refName;
  refName << "elasticity";
  HDF5Exporter exporter(mesh,refName.str());

  for (int refIndex=0; refIndex < numRefs; refIndex++) {
    soln->solve();

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
