//  Oldroyd-B.cpp
//  Driver for 2D OldroyB fluid model
//  Camellia
//
//  Created by Brendan Keith, October 2015.

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

class TopBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y-1.0) < tol);
  }
};

class RampBoundaryFunction_U1 : public SimpleFunction
{
  double _eps; // ramp width
public:
  RampBoundaryFunction_U1(double eps)
  {
    _eps = eps;
  }
  double value(double x, double y)
  {
    if ( (abs(x) < _eps) )   // top left
    {
      return x / _eps;
    }
    else if ( abs(1.0-x) < _eps)     // top right
    {
      return (1.0-x) / _eps;
    }
    else     // top middle
    {
      return 1;
    }
  }
};

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

  //////////////////////////////////////////////////////////////////////
  ///////////////////////  SET PROBLEM PARAMETERS  /////////////////////
  //////////////////////////////////////////////////////////////////////
  double rho = 1;
  double lambda = 1;
  double mu0 = 1;
  double mu1 = 1;
  int numRefs = 1;
  int k = 2, delta_k = 1;
  int loadRefinementNumber = -1; // -1 means don't load from file
  double nonlinearTolerance = 1e-5;
  int maxNonlinearIterations = 20;
  string norm = "Graph";
  string solverChoice = "KLU";
  string savePrefix = "OldroydB_ref";
  string loadPrefix = "OldroydB_ref";
  cmdp.setOption("polyOrder",&k,"polynomial order for field variable u");
  cmdp.setOption("delta_k", &delta_k, "test space polynomial order enrichment");
  cmdp.setOption("numRefs",&numRefs,"number of refinements");
  cmdp.setOption("rho", &rho, "rho");
  cmdp.setOption("lambda", &lambda, "lambda");
  cmdp.setOption("mu0", &mu0, "mu0");
  cmdp.setOption("mu1", &mu1, "mu1");
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
  //////////////////////////////////////////////////////////////////////
  ///////////////////  MISCELLANEOUS LOCAL VARIABLES  //////////////////
  //////////////////////////////////////////////////////////////////////
  FunctionPtr one  = Function::constant(1);
  FunctionPtr zero = Function::zero();
  FunctionPtr x    = Function::xn(1);
  FunctionPtr y    = Function::yn(1);
  FunctionPtr n    = Function::normal();

  //////////////////////////////////////////////////////////////////////
  ////////////////////////////  INITIALIZE  ////////////////////////////
  //////////////////////////////////////////////////////////////////////
  VarFactory vf;

  //////////////////////////   DECLARE MESH   //////////////////////////
  int H1Order = k+1;
  double width = 1.0, height = 1.0;
  int horizontalCells = 2, verticalCells = 2;
  MeshPtr mesh = MeshFactory::quadMesh(bf, H1Order, delta_k, width, height,
                                       horizontalCells, verticalCells);

  // Pointers for linear solve
  BFPtr bf   = Teuchos::rcp( new BF(vf) );
  RHSPtr rhs = RHS::rhs();
  BCPtr bc   = BC::bc();
  SolutionPtr solnUpdate = Solution::solution(bf, mesh, bc); // RHS+IP have to be reset after each iteration

  // Pointers for background approximations
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  BCPtr nullBC   = Teuchos::rcp((BC*)NULL);
  IPPtr nullIP   = Teuchos::rcp((IP*)NULL);
  SolutionPtr solnBackground = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );


  ///////////////////////   DECLARE VARIABLES   ////////////////////////
  // TRIAL VARIABLES:
  // fields:
  VarPtr du   = vf.fieldVar("u",    VECTOR_L2);
  VarPtr p    = vf.fieldVar("p",    L2);
  VarPtr dL11 = vf.fieldVar("dL11", L2);
  VarPtr dL12 = vf.fieldVar("dL12", L2);
  VarPtr dL21 = vf.fieldVar("dL21", L2);
  VarPtr dL22 = vf.fieldVar("dL22", L2);
  VarPtr dT11 = vf.fieldVar("dT11", L2);
  VarPtr dT12 = vf.fieldVar("dT12", L2);
  VarPtr dT22 = vf.fieldVar("dT22", L2);
  // traces:
  VarPtr u1hat        = vf.traceVar("\\hat{u_1}");
  VarPtr u2hat        = vf.traceVar("\\hat{u_2}");
  // VarPtr u_nhat        = vf.fluxVar("\\hat{u_n}");
  VarPtr sigma_n1hat   = vf.fluxVar("\\hat{\\sigma_{n_1}}");
  VarPtr sigma_n2hat   = vf.fluxVar("\\hat{\\sigma_{n_2}}");
  VarPtr TtensU_n11hat = vf.fluxVar("\\hat{(T\\otimes u)_{n_{11}}}");
  VarPtr TtensU_n12hat = vf.fluxVar("\\hat{(T\\otimes u)_{n_{12}}}");
  // VarPtr TtensU_n21hat = vf.fluxVar("\\hat{(T\\otimes u)_{n_{21}}}");
  VarPtr TtensU_n22hat = vf.fluxVar("\\hat{(T\\otimes u)_{n_{22}}}");

  // TEST VARIABLES:
  VarPtr v1  = vf.testVar("v_1",    HGRAD);
  VarPtr v2  = vf.testVar("v_2",    HGRAD);
  VarPtr q   = vf.testVar("q",      HGRAD);
  VarPtr S11 = vf.testVar("S_{11}", HGRAD);
  VarPtr S12 = vf.testVar("S_{12}", HGRAD);
  VarPtr S22 = vf.testVar("S_{22}", HGRAD);
  VarPtr M1  = vf.testVar("\\vec{M}_{1}", HDIV);
  VarPtr M2  = vf.testVar("\\vec{M}_{2}", HDIV);

  // BACKGROUND TRIAL VARIABLES
  FunctionPtr bu1  = Function::solution(u->x(), solnBackground);
  FunctionPtr bu2  = Function::solution(u->y(), solnBackground);
  FunctionPtr bL11 = Function::solution(dL11,   solnBackground);
  FunctionPtr bL12 = Function::solution(dL12,   solnBackground);
  FunctionPtr bL21 = Function::solution(dL21,   solnBackground);
  FunctionPtr bL22 = Function::solution(dL22,   solnBackground);
  FunctionPtr bT11 = Function::solution(dT11,   solnBackground);
  FunctionPtr bT12 = Function::solution(dT12,   solnBackground);
  FunctionPtr bT22 = Function::solution(dT22,   solnBackground);


  //////////////////////  DECLARE INITIAL GUESSES  /////////////////////
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[bu1->ID()] = Function::constant(0.0);
  functionMap[bu2->ID()] = Function::constant(0.0);
  // everything else = 0
  solnBackground->projectOntoMesh(functionMap);

  if (rank==0)
  {
    cout << "Initial guess set" << endl;
  }


  //////////////////////////   DECLARE BC'S   //////////////////////////
  // SpatialFilterPtr y_equals_one = SpatialFilter::matchingY(1.0);
  // SpatialFilterPtr y_equals_zero = SpatialFilter::matchingY(0);
  // SpatialFilterPtr x_equals_one = SpatialFilter::matchingX(1.0);
  // SpatialFilterPtr x_equals_zero = SpatialFilter::matchingX(0.0);
  SpatialFilterPtr topBoundary = Teuchos::rcp( new TopBoundary );
  SpatialFilterPtr otherBoundary = SpatialFilter::negatedFilter(topBoundary);

  // inflow

  // outflow

  // no-slips
  //   top boundary:
  FunctionPtr u1_bc_fxn = Teuchos::rcp( new RampBoundaryFunction_U1(1.0/64) );
  bc->addDirichlet(u1hat, topBoundary, u1_bc_fxn);
  bc->addDirichlet(u2hat, topBoundary, zero);

  //   everywhere else:
  bc->addDirichlet(u1hat, otherBoundary, zero);
  bc->addDirichlet(u2hat, otherBoundary, zero);

  // zero-mean constaint
  bc->addZeroMeanConstraint(p);


  //////////////////////////////////////////////////////////////////////
  //////////////////////////   DEFINE FORMS   //////////////////////////
  //////////////////////////////////////////////////////////////////////

  //  0. Definition of L
  //
  //    L = \nabla u
  //
  //  (L, M) + (u, \nabla \cdot M) - <\hat{u}, M \hat{n}> = 0
  //
  //    u = du + bu,    L = dL + bL
  //
  //  (dL,M) + (du, \nabla \cdot M) - <\hat{u}, M \hat{n}>
  //    = -(bL, M) - (bu, \nabla \cdot M)
  //
  // BF:
  bf->addTerm(dL11, M1->x());
  bf->addTerm(dL12, M1->y());
  bf->addTerm(dL21, M2->x());
  bf->addTerm(dL22, M2->y());
  //
  bf->addTerm(du->x(), M1->div());
  bf->addTerm(du->y(), M2->div());
  //
  bf->addTerm(u1hat,-M1->dot_normal());
  bf->addTerm(u2hat,-M2->dot_normal());
  //
  // RHS:
  rhs->addTerm(-bL11*M1->x());
  rhs->addTerm(-bL12*M1->y());
  rhs->addTerm(-bL21*M2->x());
  rhs->addTerm(-bL22*M2->y());
  //
  rhs->addTerm(-bu1*M1->div());
  rhs->addTerm(-bu2*M2->div());


  //  1. Conservation of mass
  //
  //    \nabla \cdot u  =  0
  //
  //  (u, \nabla q) - <\hat{u}_n, q> = 0
  //
  //    u = du + bu
  //
  //  (du, \nabla q) - <\hat{u}_n, q>
  //    = -(bu, \nabla q)
  //
  // BF:
  bf->addTerm(du->x(), q->dx());
  bf->addTerm(du->y(), q->dy());
  //
  // bf->addTerm(u_nhat, q);
  bf->addTerm(u1hat*n->x()+u2hat*n->y(),q);
  //
  // RHS:
  rhs->addTerm(-bu1*q->dx());
  rhs->addTerm(-bu2*q->dy());


  //  2. Conservation of momentum
  //
  //    \rho ( L u - f )  +  \nabla p  -  \mu_0 \nabla \cdot ( L + L^T )  -  \nabla \cdot T  =  0
  //
  //  (L u, \rho v) - (pI, \nabla v) + (L, \mu_0 ( \nabla v + (\nabla v)^T )) + (T, \nabla v) - <\hat{\sigma}_n, v> = (\rho f, v)
  //
  //
  //    u = du + bu,    L = dL + bL,    T = dT + bT
  //
  //  (du, \rho (bL)^T v) + (dL, \rho v \otimes bu) - (pI, \nabla v) + (dL, \mu_0 ( \nabla v + (\nabla v)^T )) + (dT, \nabla v) - <\hat{\sigma}_n, v>
  //    = (\rho f, v) - (\rho bL bu, v) - (\mu_0( bL + bL^T ), \nabla v) - (bT, \nabla v) + H.O.T.
  //
  // BF:
  bf->addTerm(du->x(), rho*(bL11*v1 + bL21*v2));
  bf->addTerm(du->y(), rho*(bL12*v1 + bL22*v2));
  //
  bf->addTerm(dL11, rho*(v1*bu1));
  bf->addTerm(dL12, rho*(v1*bu2));
  bf->addTerm(dL21, rho*(v2*bu1));
  bf->addTerm(dL22, rho*(v2*bu2));
  //
  bf->addTerm(p,-v1->dx());
  bf->addTerm(p,-v2->dy());
  //
  bf->addTerm(dL11, mu0*2*v1->dx());
  bf->addTerm(dL12, mu0*(v1->dy()+v2->dx()));
  bf->addTerm(dL21, mu0*(v1->dy()+v2->dx()));
  bf->addTerm(dL22, mu0*2*v2->dy());
  //
  bf->addTerm(dT11, v1->dx());
  bf->addTerm(dT12, v1->dy());
  bf->addTerm(dT21, v2->dx());
  bf->addTerm(dT22, v2->dy());
  //
  bf->addTerm(sigma_n1hat,-v1);
  bf->addTerm(sigma_n2hat,-v2);
  //
  // RHS:
  // rhs->addTerm(f*v1)
  // rhs->addTerm(f*v2)
  //
  rhs->addTerm(-rho*(bL11*bu1+bL12*bu2)*v1);
  rhs->addTerm(-rho*(bL21*bu1+bL22*bu2)*v2);
  //
  rhs->addTerm(-mu0*2*bL11*v1->x());
  rhs->addTerm(-mu0*(bL12+bL21)*v1->y());
  rhs->addTerm(-mu0*(bL12+bL21)*v2->x());
  rhs->addTerm(-mu0*2*bL22*v2->y());
  //
  rhs->addTerm(-bT11*v1->x());
  rhs->addTerm(-bT12*v1->y());
  rhs->addTerm(-bT12*v2->x());
  rhs->addTerm(-bT22*v2->y());


  //  3. Constitutive Law
  //
  //    T  +  \lambda ( \nabla \cdot ( T \otimes u )  -  ( (\nabla u) T + T (\nabla u)^T ) ) ) = \mu_1 ( L + L^T )
  //
  //  (T, S) - (T \otimes u, \lambda \nabla S) + <\hat{T \otimes u }_n, \lambda S> - (L T, 2 S) - (L, 2 \mu_1 S) = 0
  //
  //    u = du + bu,    L = dL + bL,    T = dT + bT
  //
  //  (dT, S) - (dT \otimes bu, \lambda \nabla S) - (bT \otimes du, \lambda \nabla S) + <\hat{T \otimes u }_n, \lambda S> - (dL bT, 2 S) - (bL dT, 2 S) - (dL, 2 \mu_1 S)
  //    = (bT, S) + (\lambda bT \otimes bu, \nabla S) + (2 bL bT, S) + (2 \mu_1 bL, S) + H.O.T.
  //
  // BF:
  bf->addTerm(dT11, S11);
  bf->addTerm(dT12, 2*S12);
  bf->addTerm(dT22, S22);
  //
  bf->addTerm(dT11,-lambda*(bu1*S11->dx()+bu2*S11->dy()));
  bf->addTerm(dT12,-2*lambda*(bu1*S12->dx()+bu2*S12->dy()));
  bf->addTerm(dT22,-lambda*(bu1*S22->dx()+bu2*S22->dy()));
  //
  bf->addTerm(du->x(),-lambda*(bT11*S11->x()+2*bT12*S12->x()+bT22*S22->x()));
  bf->addTerm(du->y(),-lambda*(bT11*S11->y()+2*bT12*S12->y()+bT22*S22->y()));
  //
  bf->addTerm(TtensU_n11hat, lambda*S11);
  bf->addTerm(TtensU_n12hat, 2*lambda*S12);
  bf->addTerm(TtensU_n22hat, lambda*S22);
  //
  bf->addTerm(dL11,-2*lambda*(bT11*S11+bT12*S12));
  bf->addTerm(dL12,-2*lambda*(bT11*S12+bT12*S22));
  bf->addTerm(dL21,-2*lambda*(bT12*S11+bT22*S12));
  bf->addTerm(dL22,-2*lambda*(bT12*S12+bT22*S22));
  //
  bf->addTerm(dT11,-2*lambda*(bL11*S11+bL21*S12));
  bf->addTerm(dT12,-2*lambda*(bL11*S12+bL21*S22));
  bf->addTerm(dT12,-2*lambda*(bL12*S11+bL22*S12));
  bf->addTerm(dT22,-2*lambda*(bL12*S12+bL22*S22));
  //
  bf->addTerm(dL11,-2*mu1*S11);
  bf->addTerm(dL12,-2*mu1*S12);
  bf->addTerm(dL21,-2*mu1*S12);
  bf->addTerm(dL22,-2*mu1*S22);
  //
  // RHS:
  rhs->addTerm(bT11*S11);
  rhs->addTerm(2*bT12*S12);
  rhs->addTerm(bT22*S22);
  //
  rhs->addTerm(lambda*(bT11*bu1)*S11->dx());
  rhs->addTerm(lambda*(bT11*bu2)*S11->dy());
  rhs->addTerm(2*lambda*(bT12*bu1)*S12->dx());
  rhs->addTerm(2*lambda*(bT12*bu2)*S12->dy());
  rhs->addTerm(lambda*(bT22*bu1)*S22->dx());
  rhs->addTerm(lambda*(bT22*bu2)*S22->dy());
  //
  rhs->addTerm(2*(bL11*bT11+bL12*bT12)*S11);
  rhs->addTerm(2*(bL11*bT12+bL12*bT22)*S12);
  rhs->addTerm(2*(bL21*bT11+bL22*bT12)*S12);
  rhs->addTerm(2*(bL21*bT12+bL22*bT22)*S22);
  //
  rhs->addTerm(2*mu1*bL11*S11);
  rhs->addTerm(2*mu1*bL12*S12);
  rhs->addTerm(2*mu1*bL21*S12);
  rhs->addTerm(2*mu1*bL22*S22);

  // PRINT BILINEAR FORM
  // cout << bf->displayString() << endl;


  //////////////////////////////////////////////////////////////////////
  //////////////////////   DEFINE INNER PRODUCTS   /////////////////////
  //////////////////////////////////////////////////////////////////////
  map<string, IPPtr> OldroydBGraph;
  OldroydBGraph["Graph"] = bf->graphNorm();
  IPPtr ip = OldroydBGraph[norm];

  //////////////////////////////////////////////////////////////////////
  //////////////////////////////  SOLVE  ///////////////////////////////
  //////////////////////////////////////////////////////////////////////
  solnUpdate->setRHS(rhs);
  solnUpdate->setIP(ip);

  mesh->registerSolution(solnUpdate);
  mesh->registerSolution(solnBackground);

  ostringstream refName;
  refName << "OldroyB";
  HDF5Exporter exporter(mesh,refName.str());

  double threshold = 0.25;
  RefinementStrategy refStrategy(solnUpdate, threshold);

  int startIndex = loadRefinementNumber + 1; // the first refinement we haven't computed (is 0 when we aren't loading from file)
  if (startIndex > 0)
  {
    // then refine first
    refStrategy.refine();
  }
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
    //   gmgSolver = Teuchos::rcp(new GMGSolver(solnUpdate, meshesCoarseToFine, cgMaxIterations, cgTol, multigridStrategy, coarseSolver, useCondensedSolve));
    //   gmgSolver->setUseConjugateGradient(useConjugateGradient);
    //   int azOutput = 20; // print residual every 20 CG iterations
    //   gmgSolver->setAztecOutput(azOutput);
    //   gmgSolver->gmgOperator()->setNarrateOnRankZero(logFineOperator,"finest GMGOperator");

    // }
    while (l2Update > nonlinearTolerance && iterCount < maxNonlinearIterations)
    {
      // if (solverChoice[0] == 'G')
      //   solnUpdate->solve(gmgSolver);
      // else
      solnUpdate->condensedSolve(solvers[solverChoice]);

      // Compute L2 norm of update
      FunctionPtr u1_incr  = Function::solution(du->x(), solnUpdate);
      FunctionPtr u2_incr  = Function::solution(du->y(), solnUpdate);
      FunctionPtr L11_incr = Function::solution(dL11, solnUpdate);
      FunctionPtr L12_incr = Function::solution(dL12, solnUpdate);
      FunctionPtr L21_incr = Function::solution(dL21, solnUpdate);
      FunctionPtr L22_incr = Function::solution(dL22, solnUpdate);
      FunctionPtr T11_incr = Function::solution(dT11, solnUpdate);
      FunctionPtr T12_incr = Function::solution(dT12, solnUpdate);
      FunctionPtr T22_incr = Function::solution(dT22, solnUpdate);

      FunctionPtr incrSquared;
      incrSquared = u1_incr*u1_incr + u2_incr*u2_incr + L11_incr*L11_incr
                  + L12_incr*L12_incr + L21_incr*L21_incr + L22_incr*L22_incr
                  + T11_incr*T11_incr + 2.0*T12_incr*T12_incr + T22_incr*T22_incr;

      double incrSquaredInt = incrSquared->integrate(solnUpdate->mesh());
      l2Update = sqrt(incrSquaredInt);

      if (commRank == 0)
        cout << "Nonlinear Update:\t " << l2Update << endl;

      // Update solution
      double alpha = 1;

      set<int> nlVars;
      nlVars.insert(du->ID());
      nlVars.insert(dL11->ID());
      nlVars.insert(dL12->ID());
      nlVars.insert(dL21->ID());
      nlVars.insert(dL22->ID());
      nlVars.insert(dT11->ID());
      nlVars.insert(dT12->ID());
      nlVars.insert(dT22->ID());

      set<int> lVars;
      lVars.insert(p->ID());
      lVars.insert(u1hat->ID());
      lVars.insert(u2hat->ID());
      lVars.insert(sigma_n1hat->ID());
      lVars.insert(sigma_n2hat->ID());
      lVars.insert(TtensU_n11hat->ID());
      lVars.insert(TtensU_n12hat->ID());
      lVars.insert(TtensU_n22hat->ID());

      solutionBackground->addReplaceSolution(solnUpdate, alpha, nlVars, lVars);

      iterCount++;
    }

    // double solveTime = solverTime->stop();
    double energyError = solnUpdate->energyErrorTotal();

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
