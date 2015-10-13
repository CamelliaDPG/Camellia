//  LinearElasticity2D.cpp
//  Driver for Linear elasticity with strongly imposed (H(div))^3 symmetry
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

using namespace Camellia;

int kronDelta(int i, int j)
{
  return (i == j) ? 1 : 0;
}

vector<double> makeVertex(double v0, double v1, double v2)
{
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  v.push_back(v2);
  return v;
}

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


  // Check if formulation is correct
  // Check that linearized equations converge immediately. Comment out nonlinears.
  // Check if det(G) > 0 (see SpaceTimeCompressibleFormulation.cpp)










  // PROBLEM PARAMETERS //
  double nonlinearTolerance = 1e-5;
  int maxNonlinearIterations = 20;
  double lambda = 123;
  double mu = 79.3;
  // double lambda = 1;
  // double mu = 1;
  int numRefs = 10;
  int useConformingTraces = 1;
  int k = 2, delta_k = 2;
  string norm = "Graph";
  string solverChoice = "KLU";
  bool saveToFile = false;
  int loadRefinementNumber = -1; // -1 means don't load from file
  string savePrefix = "elasticity_ref";
  string loadPrefix = "elasticity_ref";
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("useConformingTraces", &useConformingTraces, "H1 conforming traces or L2 conforming traces");
  cmdp.setOption("lambda", &lambda, "lambda");
  cmdp.setOption("mu", &mu, "mu");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("loadRefinement", &loadRefinementNumber, "Refinement number to load from previous run");
  cmdp.setOption("loadPrefix", &loadPrefix, "Filename prefix for loading solution/mesh from previous run");
  cmdp.setOption("saveToFile", "skipSave", &saveToFile, "Save solution after each refinement/solve");
  cmdp.setOption("savePrefix", &savePrefix, "Filename prefix for saved solutions if saveToFile option is selected");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL)
  {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }
  Teuchos::RCP<Time> totalTimer = Teuchos::TimeMonitor::getNewCounter("Total Time");
  totalTimer->start(true);

  //////////////////////   DECLARE VARIABLES   ////////////////////////
  VarFactoryPtr vf = VarFactory::varFactory();
  // trials:
  VarPtr u        = vf->fieldVar("u", VECTOR_L2);
  VarPtr dG11     = vf->fieldVar("\\delta\\G_{11}", L2);
  VarPtr dG12     = vf->fieldVar("\\delta\\G_{12}", L2);
  VarPtr dG13     = vf->fieldVar("\\delta\\G_{13}", L2);
  VarPtr dG21     = vf->fieldVar("\\delta\\G_{21}", L2);
  VarPtr dG22     = vf->fieldVar("\\delta\\G_{22}", L2);
  VarPtr dG23     = vf->fieldVar("\\delta\\G_{23}", L2);
  VarPtr dG31     = vf->fieldVar("\\delta\\G_{31}", L2);
  VarPtr dG32     = vf->fieldVar("\\delta\\G_{32}", L2);
  VarPtr dG33     = vf->fieldVar("\\delta\\G_{33}", L2);
  VarPtr dSigma11 = vf->fieldVar("\\delta\\sigma_{11}", L2);
  VarPtr dSigma12 = vf->fieldVar("\\delta\\sigma_{12}", L2);
  VarPtr dSigma13 = vf->fieldVar("\\delta\\sigma_{13}", L2);
  VarPtr dSigma22 = vf->fieldVar("\\delta\\sigma_{22}", L2);
  VarPtr dSigma23 = vf->fieldVar("\\delta\\sigma_{23}", L2);
  VarPtr dSigma33 = vf->fieldVar("\\delta\\sigma_{33}", L2);
  // traces:
  Space traceSpace = useConformingTraces ? HGRAD : L2;
  // cout << "useConformingTraces: " << useConformingTraces << endl;
  VarPtr u1hat = vf->traceVar("\\hat{u_1}", traceSpace);
  VarPtr u2hat = vf->traceVar("\\hat{u_2}", traceSpace);
  VarPtr u3hat = vf->traceVar("\\hat{u_3}", traceSpace);
  VarPtr t1hat = vf->fluxVar("\\hat{t_1}");
  VarPtr t2hat = vf->fluxVar("\\hat{t_2}");
  VarPtr t3hat = vf->fluxVar("\\hat{t_3}");

  // tests:
  VarPtr tau11  = vf->testVar("\\tau_{11}", HGRAD);
  VarPtr tau12  = vf->testVar("\\tau_{12}", HGRAD);
  VarPtr tau13  = vf->testVar("\\tau_{13}", HGRAD);
  VarPtr tau22  = vf->testVar("\\tau_{22}", HGRAD);
  VarPtr tau23  = vf->testVar("\\tau_{23}", HGRAD);
  VarPtr tau33  = vf->testVar("\\tau_{33}", HGRAD);
  VarPtr v1     = vf->testVar("v_1", HGRAD);
  VarPtr v2     = vf->testVar("v_2", HGRAD);
  VarPtr v3     = vf->testVar("v_3", HGRAD);
  VarPtr delta1 = vf->testVar("\\delta_1", HDIV);
  VarPtr delta2 = vf->testVar("\\delta_2", HDIV);
  VarPtr delta3 = vf->testVar("\\delta_3", HDIV);

  BFPtr bf = Teuchos::rcp( new BF(vf) );

  // Mesh
  BCPtr bc = BC::bc();

  CellTopoPtr hex = CellTopology::hexahedron();
  vector<double> V0 = makeVertex(0,0,0);
  vector<double> V1 = makeVertex(1,0,0);
  vector<double> V2 = makeVertex(1,1,0);
  vector<double> V3 = makeVertex(0,1,0);
  vector<double> V4 = makeVertex(0,0,1);
  vector<double> V5 = makeVertex(1,0,1);
  vector<double> V6 = makeVertex(1,1,1);
  vector<double> V7 = makeVertex(0,1,1);

  vector< vector<double> > vertices;
  vertices.push_back(V0);
  vertices.push_back(V1);
  vertices.push_back(V2);
  vertices.push_back(V3);
  vertices.push_back(V4);
  vertices.push_back(V5);
  vertices.push_back(V6);
  vertices.push_back(V7);

  vector<unsigned> hexVertexList;
  hexVertexList.push_back(0);
  hexVertexList.push_back(1);
  hexVertexList.push_back(2);
  hexVertexList.push_back(3);
  hexVertexList.push_back(4);
  hexVertexList.push_back(5);
  hexVertexList.push_back(6);
  hexVertexList.push_back(7);

  vector< vector<unsigned> > elementVertices;
  elementVertices.push_back(hexVertexList);

  vector< CellTopoPtr > cellTopos;
  cellTopos.push_back(hex);

  MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );

  MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );

  MeshPtr mesh = Teuchos::rcp( new Mesh (meshTopology, bf, k+1, delta_k) );

  // SOLUTION OBJECTS //
  SolutionPtr solutionUpdate = Solution::solution(bf, mesh, bc);
  SolutionPtr solutionBackground = Solution::solution(bf, mesh, bc);

  // SET INITIAL GUESS //

  map<int, FunctionPtr> initialGuess;

  if (commRank==0)
  {
    cout << "Initial guess set" << endl;
  }
  // initialGuess[u(1)->ID()] = problem->u1_exact();
  // initialGuess[u(2)->ID()] = problem->u2exact();
  // initialGuess[sigma(1,1)->ID()] = problem->sigma1_exact()->x();
  // initialGuess[sigma(1,2)->ID()] = problem->sigma1_exact()->y();
  // initialGuess[sigma(2,1)->ID()] = problem->sigma2_exact()->x();
  // initialGuess[sigma(2,2)->ID()] = problem->sigma2_exact()->y();
  // initialGuess[p()->ID()] = problem->p_exact();
  // initialGuess[uhat(1)->ID()] = problem->u1_exact();
  // initialGuess[uhat(2)->ID()] = problem->u2_exact();

  // g**,sigma** = 0 (do nothing)
  solutionBackground->projectOntoMesh(initialGuess);

  //////////////////////   PREVIOUS SOLUTIONS   ////////////////////////
  FunctionPtr g11 = Function::solution(dG11, solutionBackground);
  FunctionPtr g12 = Function::solution(dG12, solutionBackground);
  FunctionPtr g13 = Function::solution(dG13, solutionBackground);
  FunctionPtr g21 = Function::solution(dG21, solutionBackground);
  FunctionPtr g22 = Function::solution(dG22, solutionBackground);
  FunctionPtr g23 = Function::solution(dG23, solutionBackground);
  FunctionPtr g31 = Function::solution(dG31, solutionBackground);
  FunctionPtr g32 = Function::solution(dG32, solutionBackground);
  FunctionPtr g33 = Function::solution(dG33, solutionBackground);
  FunctionPtr sigma11 = Function::solution(dSigma11, solutionBackground);
  FunctionPtr sigma12 = Function::solution(dSigma12, solutionBackground);
  FunctionPtr sigma13 = Function::solution(dSigma13, solutionBackground);
  FunctionPtr sigma22 = Function::solution(dSigma22, solutionBackground);
  FunctionPtr sigma23 = Function::solution(dSigma23, solutionBackground);
  FunctionPtr sigma33 = Function::solution(dSigma33, solutionBackground);

  ////////////////    MISCELLANEOUS LOCAL VARIABLES    ////////////////

  // Compliance Tensor
  static const int N = 3;
  double A[N][N][N][N];
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      for (int k = 0; k < N; ++k)
      {
        for (int l = 0; l < N; ++l)
        {
          A[i][j][k][l] = 1/(2*mu)*(0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                                    - lambda/(2*mu+N*lambda)*kronDelta(i,j)*kronDelta(k,l));
          // cout << "A(" << i << "," << j << "," << k << "," << l << ") = " << A[i][j][k][l] << endl;
        }
      }
    }
  }

  // Stiffness Tensor
  double C[N][N][N][N];
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      for (int k = 0; k < N; ++k)
      {
        for (int l = 0; l < N; ++l)
        {
          C[i][j][k][l] = (2*mu)*0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                          + lambda*kronDelta(i,j)*kronDelta(k,l);
          // cout << "C(" << i << "," << j << "," << k << "," << l << ") = " << C[i][j][k][l] << endl;
        }
      }
    }
  }

  LinearTermPtr Sigma[N][N];
  Sigma[0][0] = 1*dSigma11;
  Sigma[0][1] = 1*dSigma12;
  Sigma[0][2] = 1*dSigma13;
  Sigma[1][0] = 1*dSigma12;
  Sigma[1][1] = 1*dSigma22;
  Sigma[1][2] = 1*dSigma23;
  Sigma[2][0] = 1*dSigma13;
  Sigma[2][1] = 1*dSigma23;
  Sigma[2][2] = 1*dSigma33;

  LinearTermPtr tau[N][N];
  tau[0][0] = 1*tau11;
  tau[0][1] = 1*tau12;
  tau[0][2] = 1*tau13;
  tau[1][0] = 1*tau12;
  tau[1][1] = 1*tau22;
  tau[1][2] = 1*tau23;
  tau[2][0] = 1*tau13;
  tau[2][1] = 1*tau23;
  tau[2][2] = 1*tau33;

  FunctionPtr one  = Function::constant(1);
  FunctionPtr zero = Function::zero();
  FunctionPtr x    = Function::xn(1);
  FunctionPtr y    = Function::yn(1);
  FunctionPtr z    = Function::zn(1);
  FunctionPtr n    = Function::normal();

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  // Constitutive equation //
  bf->addTerm(u->x(),-(tau11->dx()+tau12->dy()+tau13->dz()));
  bf->addTerm(u->y(),-(tau12->dx()+tau22->dy()+tau23->dz()));
  bf->addTerm(u->z(),-(tau13->dx()+tau23->dy()+tau33->dz()));
  bf->addTerm(u1hat, tau11*n->x()+tau12*n->y()+tau13*n->z());
  bf->addTerm(u2hat, tau12*n->x()+tau22*n->y()+tau23*n->z());
  bf->addTerm(u3hat, tau13*n->x()+tau23*n->y()+tau33*n->z());
  for (int i = 0; i < N; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      for (int k = 0; k < N; ++k)
      {
        for (int l = 0; l < N; ++l)
        {
          if (abs(A[i][j][k][l])>1e-14)
          {
            bf->addTerm(Sigma[k][l],-Function::constant(A[i][j][k][l])*tau[i][j]);
          }
        }
      }
    }
  }
  // bf->addTerm(dG11, g11*tau11 + g12*tau12 + g13*tau13);
  // bf->addTerm(dG12, g11*tau12 + g12*tau22 + g13*tau23);
  // bf->addTerm(dG13, g11*tau13 + g12*tau23 + g13*tau33);
  // bf->addTerm(dG21, g21*tau11 + g22*tau12 + g23*tau13);
  // bf->addTerm(dG22, g21*tau12 + g22*tau22 + g23*tau23);
  // bf->addTerm(dG23, g21*tau13 + g22*tau23 + g23*tau33);
  // bf->addTerm(dG31, g31*tau11 + g32*tau12 + g33*tau13);
  // bf->addTerm(dG32, g31*tau12 + g32*tau22 + g33*tau23);
  // bf->addTerm(dG33, g31*tau13 + g32*tau23 + g33*tau33);

  // Principle of virtual work (momentum balance) //
  bf->addTerm(dSigma11, v1->dx());
  bf->addTerm(dSigma12, v1->dy());
  bf->addTerm(dSigma13, v1->dz());
  bf->addTerm(dSigma12, v2->dx());
  bf->addTerm(dSigma22, v2->dy());
  bf->addTerm(dSigma23, v2->dz());
  bf->addTerm(dSigma13, v3->dx());
  bf->addTerm(dSigma23, v3->dy());
  bf->addTerm(dSigma33, v3->dz());
  bf->addTerm(t1hat,-v1);
  bf->addTerm(t2hat,-v2);
  bf->addTerm(t3hat,-v3);
  // // dsigma : (G^T)*grad(v)
  // bf->addTerm(dSigma11, g11*v1->dx() + g21*v2->dx() + g31*v3->dx());
  // bf->addTerm(dSigma12, g11*v1->dy() + g21*v2->dy() + g31*v3->dy());
  // bf->addTerm(dSigma13, g11*v1->dz() + g21*v2->dz() + g31*v3->dz());
  // bf->addTerm(dSigma12, g12*v1->dx() + g22*v2->dx() + g32*v3->dx());
  // bf->addTerm(dSigma22, g12*v1->dy() + g22*v2->dy() + g32*v3->dy());
  // bf->addTerm(dSigma23, g12*v1->dz() + g22*v2->dz() + g32*v3->dz());
  // bf->addTerm(dSigma13, g13*v1->dx() + g23*v2->dx() + g33*v3->dx());
  // bf->addTerm(dSigma23, g13*v1->dy() + g23*v2->dy() + g33*v3->dy());
  // bf->addTerm(dSigma33, g13*v1->dz() + g23*v2->dz() + g33*v3->dz());
  // // dG : grad(v)*sigma
  // bf->addTerm(dG11, sigma11*v1->dx() + sigma12*v1->dy() + sigma13*v1->dz());
  // bf->addTerm(dG12, sigma12*v1->dx() + sigma22*v1->dy() + sigma23*v1->dz());
  // bf->addTerm(dG13, sigma13*v1->dx() + sigma23*v1->dy() + sigma33*v1->dz());
  // bf->addTerm(dG21, sigma11*v2->dx() + sigma12*v2->dy() + sigma13*v2->dz());
  // bf->addTerm(dG22, sigma12*v2->dx() + sigma22*v2->dy() + sigma23*v2->dz());
  // bf->addTerm(dG23, sigma13*v2->dx() + sigma23*v2->dy() + sigma33*v2->dz());
  // bf->addTerm(dG31, sigma11*v3->dx() + sigma12*v3->dy() + sigma13*v3->dz());
  // bf->addTerm(dG32, sigma12*v3->dx() + sigma22*v3->dy() + sigma23*v3->dz());
  // bf->addTerm(dG33, sigma13*v3->dx() + sigma23*v3->dy() + sigma33*v3->dz());

  // Constraint equation //
  bf->addTerm(u->x(),-delta1->div());
  bf->addTerm(u->y(),-delta2->div());
  bf->addTerm(u->z(),-delta3->div());
  bf->addTerm(u1hat, delta1->dot_normal());
  bf->addTerm(u2hat, delta2->dot_normal());
  bf->addTerm(u3hat, delta3->dot_normal());
  bf->addTerm(dG11,-delta1->x());
  bf->addTerm(dG12,-delta1->y());
  bf->addTerm(dG13,-delta1->z());
  bf->addTerm(dG21,-delta2->x());
  bf->addTerm(dG22,-delta2->y());
  bf->addTerm(dG23,-delta2->z());
  bf->addTerm(dG31,-delta3->x());
  bf->addTerm(dG32,-delta3->y());
  bf->addTerm(dG33,-delta3->z());

  // PRINT BILINEAR FORM
  // cout << bf->displayString() << endl;


  ////////////////////   SPECIFY RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs();

  // Constitutive equation //
  // -1/2*(G^T)*G:tau
  // rhs->addTerm(-0.5*(g11*g11 + g21*g21 + g31*g31) * tau11 );
  // rhs->addTerm(-0.5*(g12*g12 + g22*g22 + g32*g32) * tau22 );
  // rhs->addTerm(-0.5*(g13*g13 + g23*g23 + g33*g33) * tau33 );
  // rhs->addTerm(-(g11*g12 + g21*g22 + g31*g32) * tau12 );
  // rhs->addTerm(-(g11*g13 + g21*g23 + g31*g33) * tau13 );
  // rhs->addTerm(-(g12*g13 + g22*g23 + g32*g33) * tau23 );

  // // Principle of virtual work (momentum balance) //
  // // no forcing term
  // // -(I+G)*sigma:grad(v)
  // rhs->addTerm(-(sigma11 + g11*sigma11 + g12*sigma12 + g13*sigma13) * v1->dx() );
  // rhs->addTerm(-(sigma12 + g11*sigma12 + g12*sigma22 + g13*sigma23) * v1->dy() );
  // rhs->addTerm(-(sigma13 + g11*sigma13 + g12*sigma23 + g13*sigma33) * v1->dz() );
  // rhs->addTerm(-(sigma12 + g21*sigma11 + g22*sigma12 + g23*sigma13) * v2->dx() );
  // rhs->addTerm(-(sigma22 + g21*sigma12 + g22*sigma22 + g23*sigma23) * v2->dy() );
  // rhs->addTerm(-(sigma23 + g21*sigma13 + g22*sigma23 + g23*sigma33) * v2->dz() );
  // rhs->addTerm(-(sigma13 + g31*sigma11 + g32*sigma12 + g33*sigma13) * v3->dx() );
  // rhs->addTerm(-(sigma23 + g31*sigma12 + g32*sigma22 + g33*sigma23) * v3->dy() );
  // rhs->addTerm(-(sigma33 + g31*sigma13 + g32*sigma23 + g33*sigma33) * v3->dz() );

  rhs->addTerm(-sigma11 * v1->dx() );
  rhs->addTerm(-sigma12 * v1->dy() );
  rhs->addTerm(-sigma13 * v1->dz() );
  rhs->addTerm(-sigma12 * v2->dx() );
  rhs->addTerm(-sigma22 * v2->dy() );
  rhs->addTerm(-sigma23 * v2->dz() );
  rhs->addTerm(-sigma13 * v3->dx() );
  rhs->addTerm(-sigma23 * v3->dy() );
  rhs->addTerm(-sigma33 * v3->dz() );

  // // Constraint equation //
  // // G:delta
  rhs->addTerm( g11 * delta1->x() );
  rhs->addTerm( g12 * delta1->y() );
  rhs->addTerm( g13 * delta1->z() );
  rhs->addTerm( g21 * delta2->x() );
  rhs->addTerm( g22 * delta2->y() );
  rhs->addTerm( g23 * delta2->z() );
  rhs->addTerm( g31 * delta3->x() );
  rhs->addTerm( g32 * delta3->y() );
  rhs->addTerm( g33 * delta3->z() );


  ////////////////////   BOUNDARY CONDITIONS   ///////////////////////
  SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(1.0);
  SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
  SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(1.0);
  SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
  SpatialFilterPtr z_equals_one = SpatialFilter::matchingZ(1.0);
  SpatialFilterPtr z_equals_zero = SpatialFilter::matchingZ(0.0);

  bc->addDirichlet(u1hat, x_equals_zero, zero);
  bc->addDirichlet(u2hat, x_equals_zero, zero);
  bc->addDirichlet(u3hat, x_equals_zero, zero);
  bc->addDirichlet(t1hat, x_equals_one,  one);
  bc->addDirichlet(t2hat, x_equals_one,  zero);
  bc->addDirichlet(t3hat, x_equals_one,  zero);
  bc->addDirichlet(t1hat, y_equals_zero, x);
  bc->addDirichlet(t2hat, y_equals_zero, zero);
  bc->addDirichlet(t3hat, y_equals_zero, zero);
  bc->addDirichlet(t1hat, y_equals_one,  x);
  bc->addDirichlet(t2hat, y_equals_one,  zero);
  bc->addDirichlet(t3hat, y_equals_one,  zero);
  bc->addDirichlet(t1hat, z_equals_zero, zero);
  bc->addDirichlet(t2hat, z_equals_zero, zero);
  bc->addDirichlet(t3hat, z_equals_zero, zero);
  bc->addDirichlet(t1hat, z_equals_one,  zero);
  bc->addDirichlet(t2hat, z_equals_one,  zero);
  bc->addDirichlet(t3hat, z_equals_one,  zero);
  // bc->addDirichlet(u1hat, y_equals_zero, zero);
  // bc->addDirichlet(u1hat, x_equals_zero, zero);
  // bc->addDirichlet(u1hat, y_equals_one,  x);
  // bc->addDirichlet(u1hat, x_equals_one,  y);
  // bc->addDirichlet(u2hat, y_equals_zero, zero);
  // bc->addDirichlet(u2hat, x_equals_zero, zero);
  // bc->addDirichlet(u2hat, y_equals_one,  x);
  // bc->addDirichlet(u2hat, x_equals_one,  y);

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

  solutionUpdate->setRHS(rhs);
  solutionUpdate->setIP(ip);

  mesh->registerSolution(solutionBackground);
  mesh->registerSolution(solutionUpdate);

  // if (loadRefinementNumber == -1) {
  //   solutionUpdate = Solution::solution(mesh, bc, rhs, ip);
  // }
  // else
  // {
    // ostringstream filePrefix;
    // filePrefix << loadPrefix << loadRefinementNumber;
    // if (commRank==0) cout << "loading " << filePrefix.str() << endl;
    // solutionUpdate = Solution::load(bf, filePrefix.str());
    // mesh = solutionUpdate->mesh();
    // solutionUpdate->setBC(bc);
    // solutionUpdate->setRHS(rhs);
    // solutionUpdate->setIP(ip);
  // }

  ostringstream refName;
  refName << "StVenantKirchhoffelasticity";
  HDF5Exporter exporter(mesh,refName.str());

  double threshold = 0.20;
  RefinementStrategy refStrategy(solutionUpdate, threshold);

  int startIndex = loadRefinementNumber + 1; // the first refinement we haven't computed (is 0 when we aren't loading from file)
  if (startIndex > 0)
  {
    // then refine first
    refStrategy.refine();
  }

  // for (int refIndex=startIndex; refIndex < numRefs; refIndex++) {
  //   soln->solve();

  //   double energyError = soln->energyErrorTotal();
  //   if (commRank == 0)
  //   {
  //     // if (refIndex > 0)
  //     // refStrategy.printRefinementStatistics(refIndex-1);
  //     cout << "Refinement:\t " << refIndex << " \tElements:\t " << mesh->numActiveElements()
  //          << " \tDOFs:\t " << mesh->numGlobalDofs() << " \tEnergy Error:\t " << energyError << endl;
  //   }

  //   // save solution to file
  //   if (saveToFile)
  //   {
  //     ostringstream filePrefix;
  //     filePrefix << savePrefix << refIndex;
  //     soln->save(filePrefix.str());
  //   }
  //   exporter.exportSolution(soln, refIndex);

  //   if (refIndex != numRefs)
  //     refStrategy.refine();
  // }

  map<string, SolverPtr> solvers;
  solvers["KLU"] = Solver::getSolver(Solver::KLU, true);
  for (int refIndex=startIndex; refIndex <= numRefs; refIndex++)
  {
    double l2Update = 1e10;
    int iterCount = 0;
    // Teuchos::RCP<GMGSolver> gmgSolver;
    // if (solverChoice[0] == 'G')
    // {
    //   bool reuseFactorization = true;
    //   SolverPtr coarseSolver = Solver::getDirectSolver(reuseFactorization);
    //   gmgSolver = Teuchos::rcp(new GMGSolver(solutionUpdate, meshesCoarseToFine, cgMaxIterations, cgTol, multigridStrategy, coarseSolver, useCondensedSolve));
    //   gmgSolver->setUseConjugateGradient(useConjugateGradient);
    //   int azOutput = 20; // print residual every 20 CG iterations
    //   gmgSolver->setAztecOutput(azOutput);
    //   gmgSolver->gmgOperator()->setNarrateOnRankZero(logFineOperator,"finest GMGOperator");

    // }
    while (l2Update > nonlinearTolerance && iterCount < maxNonlinearIterations)
    {
      // if (solverChoice[0] == 'G')
      //   solutionUpdate->solve(gmgSolver);
      // else
        solutionUpdate->condensedSolve(solvers[solverChoice]);

      // Compute L2 norm of update
      FunctionPtr g11_incr = Function::solution(dG11, solutionUpdate);
      FunctionPtr g12_incr = Function::solution(dG12, solutionUpdate);
      FunctionPtr g13_incr = Function::solution(dG13, solutionUpdate);
      FunctionPtr g21_incr = Function::solution(dG21, solutionUpdate);
      FunctionPtr g22_incr = Function::solution(dG22, solutionUpdate);
      FunctionPtr g23_incr = Function::solution(dG23, solutionUpdate);
      FunctionPtr g31_incr = Function::solution(dG31, solutionUpdate);
      FunctionPtr g32_incr = Function::solution(dG32, solutionUpdate);
      FunctionPtr g33_incr = Function::solution(dG33, solutionUpdate);
      FunctionPtr sigma11_incr = Function::solution(dSigma11, solutionUpdate);
      FunctionPtr sigma12_incr = Function::solution(dSigma12, solutionUpdate);
      FunctionPtr sigma13_incr = Function::solution(dSigma13, solutionUpdate);
      FunctionPtr sigma22_incr = Function::solution(dSigma22, solutionUpdate);
      FunctionPtr sigma23_incr = Function::solution(dSigma23, solutionUpdate);
      FunctionPtr sigma33_incr = Function::solution(dSigma33, solutionUpdate);


      // double u1L2Update = solutionUpdate->L2NormOfSolutionGlobal(u(1)->ID());
      // double u2L2Update = solutionUpdate->L2NormOfSolutionGlobal(u(2)->ID());
      FunctionPtr incrSquared;
      incrSquared = g11_incr*g11_incr + g12_incr*g12_incr + g13_incr*g13_incr
                  + g21_incr*g21_incr + g22_incr*g22_incr + g23_incr*g23_incr
                  + g31_incr*g31_incr + g32_incr*g32_incr + g33_incr*g33_incr
                  + sigma11_incr*sigma11_incr + 2.0*sigma12_incr*sigma12_incr
                  + sigma22_incr*sigma22_incr + 2.0*sigma13_incr*sigma13_incr
                  + sigma33_incr*sigma33_incr + 2.0*sigma23_incr*sigma23_incr;

      double incrSquaredInt = incrSquared->integrate(solutionUpdate->mesh());
      l2Update = sqrt(incrSquaredInt);

      if (commRank == 0)
        cout << "Nonlinear Update:\t " << l2Update << endl;

      // Update solution
      double alpha = 1;

      set<int> nlVars;
      nlVars.insert(dSigma11->ID());
      nlVars.insert(dSigma12->ID());
      nlVars.insert(dSigma13->ID());
      nlVars.insert(dSigma22->ID());
      nlVars.insert(dSigma23->ID());
      nlVars.insert(dSigma33->ID());
      nlVars.insert(dG11->ID());
      nlVars.insert(dG12->ID());
      nlVars.insert(dG13->ID());
      nlVars.insert(dG21->ID());
      nlVars.insert(dG22->ID());
      nlVars.insert(dG23->ID());
      nlVars.insert(dG31->ID());
      nlVars.insert(dG32->ID());
      nlVars.insert(dG33->ID());

      set<int> lVars;
      lVars.insert(u->ID());
      lVars.insert(u1hat->ID());
      lVars.insert(u2hat->ID());
      lVars.insert(u3hat->ID());
      lVars.insert(t1hat->ID());
      lVars.insert(t2hat->ID());
      lVars.insert(t3hat->ID());

      solutionBackground->addReplaceSolution(solutionUpdate, alpha, nlVars, lVars);

      iterCount++;
    }

    // double solveTime = solverTime->stop();
    double energyError = solutionUpdate->energyErrorTotal();

    if (commRank == 0)
    {
      cout << "Refinement: " << refIndex
        << " \tElements: " << mesh->numActiveElements()
        << " \tDOFs: " << mesh->numGlobalDofs()
        << " \tEnergy Error: " << energyError
        // << " \tSolve Time: " << solveTime
        << " \tTotal Time: " << totalTimer->totalElapsedTime(true)
        // << " \tIteration Count: " << iterationCount
        << endl;
      // dataFile << refIndex
      //   << " " << mesh->numActiveElements()
      //   << " " << mesh->numGlobalDofs()
      //   << " " << energyError
      //   // << " " << solveTime
      //   << " " << totalTimer->totalElapsedTime(true)
      //   // << " " << iterationCount
      //   << endl;
    }

    // save solution to file
    if (saveToFile)
    {
      ostringstream filePrefix;
      filePrefix << savePrefix << refIndex;
      solutionBackground->save(filePrefix.str());
    }

    exporter.exportSolution(solutionBackground, refIndex);

    if (refIndex != numRefs)
    {
      refStrategy.refine();
      // meshesCoarseToFine.push_back(mesh);
    }
  }

  double totalTime = totalTimer->stop();
  if (commRank == 0)
    cout << "Total time = " << totalTime << endl;

  return 0;
}
