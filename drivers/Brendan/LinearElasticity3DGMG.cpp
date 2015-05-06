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
#include "GMGSolver.h"
#include "MeshTestUtility.h"
#include "RefinementStrategy.h"

using namespace Camellia;

int kronDelta(int i, int j) {
  return (i == j) ? 1 : 0;
}

vector<double> makeVertex(double v0, double v1, double v2) {
  vector<double> v;
  v.push_back(v0);
  v.push_back(v1);
  v.push_back(v2);
  return v;
}

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
  bool saveToFile = false;
  int loadRefinementNumber = -1; // -1 means don't load from file
  string savePrefix = "elasticity_ref";
  string loadPrefix = "elasticity_ref";
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("lambda", &lambda, "lambda");
  cmdp.setOption("mu", &mu, "mu");
  cmdp.setOption("norm", &norm, "norm");
  cmdp.setOption("loadRefinement", &loadRefinementNumber, "Refinement number to load from previous run");
  cmdp.setOption("loadPrefix", &loadPrefix, "Filename prefix for loading solution/mesh from previous run");
  cmdp.setOption("saveToFile", "skipSave", &saveToFile, "Save solution after each refinement/solve");
  cmdp.setOption("savePrefix", &savePrefix, "Filename prefix for saved solutions if saveToFile option is selected");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
#ifdef HAVE_MPI
    MPI_Finalize();
#endif
    return -1;
  }

  //////////////////////   DECLARE VARIABLES   ////////////////////////
  VarFactoryPtr vf = VarFactory::varFactory();
  // trials:
  VarPtr u       = vf->fieldVar("u", VECTOR_L2);
  VarPtr sigma11 = vf->fieldVar("\\sigma_{11}", L2);
  VarPtr sigma12 = vf->fieldVar("\\sigma_{12}", L2);
  VarPtr sigma13 = vf->fieldVar("\\sigma_{13}", L2);
  VarPtr sigma22 = vf->fieldVar("\\sigma_{22}", L2);
  VarPtr sigma23 = vf->fieldVar("\\sigma_{23}", L2);
  VarPtr sigma33 = vf->fieldVar("\\sigma_{33}", L2);

  Space traceSpace = L2;
  
  // traces:
  VarPtr u1hat = vf->traceVar("\\hat{u_1}",traceSpace);
  VarPtr u2hat = vf->traceVar("\\hat{u_2}",traceSpace);
  VarPtr u3hat = vf->traceVar("\\hat{u_3}",traceSpace);
  VarPtr t1hat = vf->fluxVar("\\hat{t_1}");
  VarPtr t2hat = vf->fluxVar("\\hat{t_2}");
  VarPtr t3hat = vf->fluxVar("\\hat{t_3}");

  // tests:
  VarPtr tau11 = vf->testVar("\\tau_{11}", HGRAD);
  VarPtr tau12 = vf->testVar("\\tau_{12}", HGRAD);
  VarPtr tau13 = vf->testVar("\\tau_{13}", HGRAD);
  VarPtr tau22 = vf->testVar("\\tau_{22}", HGRAD);
  VarPtr tau23 = vf->testVar("\\tau_{23}", HGRAD);
  VarPtr tau33 = vf->testVar("\\tau_{33}", HGRAD);
  VarPtr v1    = vf->testVar("v_1", HGRAD);
  VarPtr v2    = vf->testVar("v_2", HGRAD);
  VarPtr v3    = vf->testVar("v_3", HGRAD);

  ////////////////    MISCELLANEOUS LOCAL VARIABLES    ////////////////

  // Compliance Tensor
  static const int N = 3;
  double C[N][N][N][N];
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      for (int k = 0; k < N; ++k){
        for (int l = 0; l < N; ++l){
          C[i][j][k][l] = 1/(2*mu)*(0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                        - lambda/(2*mu+N*lambda)*kronDelta(i,j)*kronDelta(k,l));
          // cout << "C(" << i << "," << j << "," << k << "," << l << ") = " << C[i][j][k][l] << endl;
        }
      }
    }
  }

  // Stiffness Tensor
  double E[N][N][N][N];
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      for (int k = 0; k < N; ++k){
        for (int l = 0; l < N; ++l){
          E[i][j][k][l] = (2*mu)*0.5*(kronDelta(i,k)*kronDelta(j,l)+kronDelta(i,l)*kronDelta(j,k))
                        + lambda*kronDelta(i,j)*kronDelta(k,l);
          // cout << "C(" << i << "," << j << "," << k << "," << l << ") = " << C[i][j][k][l] << endl;
        }
      }
    }
  }

  LinearTermPtr sigma[N][N];
  sigma[0][0] = 1*sigma11;
  sigma[0][1] = 1*sigma12;
  sigma[0][2] = 1*sigma13;
  sigma[1][0] = 1*sigma12;
  sigma[1][1] = 1*sigma22;
  sigma[1][2] = 1*sigma23;
  sigma[2][0] = 1*sigma13;
  sigma[2][1] = 1*sigma23;
  sigma[2][2] = 1*sigma33;

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
  BFPtr bf = Teuchos::rcp( new BF(vf) );

  bf->addTerm(u->x(),-(tau11->dx()+tau12->dy()+tau13->dz()));
  bf->addTerm(u->y(),-(tau12->dx()+tau22->dy()+tau23->dz()));
  bf->addTerm(u->z(),-(tau13->dx()+tau23->dy()+tau33->dz()));
  bf->addTerm(u1hat, tau11*n->x()+tau12*n->y()+tau13*n->z());
  bf->addTerm(u2hat, tau12*n->x()+tau22*n->y()+tau23*n->z());
  bf->addTerm(u3hat, tau13*n->x()+tau23*n->y()+tau33*n->z());
  for (int i = 0; i < N; ++i){
    for (int j = 0; j < N; ++j){
      for (int k = 0; k < N; ++k){
        for (int l = 0; l < N; ++l){
          if (abs(C[i][j][k][l])>1e-14){
            bf->addTerm(sigma[k][l],-Function::constant(C[i][j][k][l])*tau[i][j]);
          }
        }
      }
    }
  }

  bf->addTerm(sigma11, v1->dx());
  bf->addTerm(sigma12, v1->dy());
  bf->addTerm(sigma13, v1->dz());
  bf->addTerm(sigma12, v2->dx());
  bf->addTerm(sigma22, v2->dy());
  bf->addTerm(sigma23, v2->dz());
  bf->addTerm(sigma13, v3->dx());
  bf->addTerm(sigma23, v3->dy());
  bf->addTerm(sigma33, v3->dz());
  // omega term missing
  bf->addTerm(t1hat,-v1);
  bf->addTerm(t2hat,-v2);
  bf->addTerm(t3hat,-v3);

  // PRINT BILINEAR FORM
  // cout << bf->displayString() << endl;

  ////////////////////   SPECIFY RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs();
  // rhs->addTerm(one*v1);
  // rhs->addTerm(one*v2);
  // rhs->addTerm(one*v3);

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

  SolutionPtr soln;
  
  MeshPtr mesh, k0Mesh;
  
  if (loadRefinementNumber == -1) {
    // Mesh
    CellTopoPtr hex = CellTopology::hexahedron();
    vector<double> V0 = {0,0,0};
    vector<double> V1 = {1,0,0};
    vector<double> V2 = {1,1,0};
    vector<double> V3 = {0,1,0};
    vector<double> V4 = {0,0,1};
    vector<double> V5 = {1,0,1};
    vector<double> V6 = {1,1,1};
    vector<double> V7 = {0,1,1};
    
    vector< vector<double> > vertices = {V0,V1,V2,V3,V4,V5,V6,V7};
    vector<unsigned> hexVertexList = {0,1,2,3,4,5,6,7};
    
    vector< vector<unsigned> > elementVertices;
    elementVertices.push_back(hexVertexList);
    
    vector< CellTopoPtr > cellTopos;
    cellTopos.push_back(hex);
    
    MeshGeometryPtr meshGeometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices, cellTopos) );
    
    MeshTopologyPtr meshTopology = Teuchos::rcp( new MeshTopology(meshGeometry) );
    
    mesh = Teuchos::rcp( new Mesh (meshTopology, bf, k+1, delta_k) );
    k0Mesh = Teuchos::rcp( new Mesh (meshTopology->deepCopy(), bf, 1, delta_k) );
    
    mesh->registerObserver(k0Mesh);
    
    soln = Solution::solution(mesh, bc, rhs, ip);
  } else {
    ostringstream filePrefix;
    filePrefix << loadPrefix << loadRefinementNumber;
    if (commRank==0) cout << "loading " << filePrefix.str() << endl;
    soln = Solution::load(bf, filePrefix.str());
    mesh = soln->mesh();
    
    MeshTopologyPtr meshTopo = mesh->getTopology();
    k0Mesh = Teuchos::rcp( new Mesh (meshTopo->deepCopy(), bf, 1, delta_k) );
    mesh->registerObserver(k0Mesh);
    
    soln->setBC(bc);
    soln->setRHS(rhs);
    soln->setIP(ip);
  }

  ostringstream refName;
  refName << "elasticity";
  HDF5Exporter exporter(mesh,refName.str());

  double threshold = 0.20;
  RefinementStrategy refStrategy(soln, threshold);
  
//  bool meshLocalGlobalConsistencyCheck = MeshTestUtility::checkLocalGlobalConsistency(mesh,1e-6); // 1e-6: we're just looking for egregious errors
//  if (!meshLocalGlobalConsistencyCheck)
//  {
//    cout << "ERROR: Mesh does not pass local/global consistency check on rank " << commRank << endl;
//  }
//  else
//  {
//    cout << "Mesh passed local/global consistency check on rank " << commRank << endl;
//  }
  
  int startIndex = loadRefinementNumber + 1; // the first refinement we haven't computed (is 0 when we aren't loading from file)
  if (startIndex > 0) {
    // then refine first
    refStrategy.refine();
  }
  
  SolverPtr kluSolver = Solver::getSolver(Solver::KLU, true);
  double tol = 1e-6;
  int maxIters = 10000;
  bool useStaticCondensation = false;
  int azOutput = 20; // print residual every 20 CG iterations
  
  for (int refIndex=startIndex; refIndex < numRefs; refIndex++) {
    Teuchos::RCP<GMGSolver> gmgSolver = Teuchos::rcp( new GMGSolver(soln, k0Mesh, maxIters, tol, kluSolver, useStaticCondensation));
    gmgSolver->setAztecOutput(azOutput);
    soln->solve(gmgSolver);

    double energyError = soln->energyErrorTotal();
    if (commRank == 0)
    {
      // if (refIndex > 0)
        // refStrategy.printRefinementStatistics(refIndex-1);
      cout << "Refinement:\t " << refIndex << " \tElements:\t " << mesh->numActiveElements()
        << " \tDOFs:\t " << mesh->numGlobalDofs() << " \tEnergy Error:\t " << energyError << endl;
    }

    // save solution to file
    if (saveToFile) {
      ostringstream filePrefix;
      filePrefix << savePrefix << refIndex;
      soln->save(filePrefix.str());
    }
    exporter.exportSolution(soln, refIndex);

    if (refIndex != numRefs)
      refStrategy.refine();
  }

  return 0;
}
