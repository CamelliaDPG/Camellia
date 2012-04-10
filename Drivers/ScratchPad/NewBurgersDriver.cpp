#include "../Burgers/BurgersBilinearForm.h"

#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "RefinementStrategy.h"
#include "NonlinearStepSize.h"
#include "NonlinearSolveStrategy.h"

// Trilinos includes
#include "Epetra_Time.h"
#include "Intrepid_FieldContainer.hpp"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "TestSuite.h"
#include "RefinementPattern.h"
#include "PenaltyConstraints.h"

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;

using namespace std;

class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(sqrt(_epsilon)/ h, 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class U0 : public SimpleFunction {
public:
  double value(double x, double y) {
    return 1 - 2 * x;
  }
};

class TopBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = (abs(y-1.0) < tol);
    return yMatch;
  }
};


int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif
  int polyOrder = 3;
  int pToAdd = 2; // for tests
  
  // define our manufactured solution or problem bilinear form:
  double epsilon = 1e-2;
  bool useTriangles = false;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int H1Order = polyOrder + 1;
  int horizontalCells = 2, verticalCells = 2;
  
  double energyThreshold = 0.2; // for mesh refinements
  double nonlinearStepSize = 0.5;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE VARIABLES 
  ////////////////////////////////////////////////////////////////////
  
  // new-style bilinear form definition
  VarFactory varFactory;
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_hat = varFactory.fluxVar("\\widehat{\\beta_n u - \\sigma_n}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  VarPtr tau = varFactory.testVar("\\tau",HDIV);
  VarPtr v = varFactory.testVar("v",HGRAD);
  BFPtr bf = Teuchos::rcp( new BF(varFactory) ); // initialize bilinear form
  
  ////////////////////////////////////////////////////////////////////
  // CREATE MESH 
  ////////////////////////////////////////////////////////////////////
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, 
                                                verticalCells, bf, H1Order, 
                                                H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  
  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, 
                                                         nullRHS, nullIP) );
  
  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  
  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u) );
  FunctionPtr beta = e1 * u_prev + Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE BILINEAR FORM
  ////////////////////////////////////////////////////////////////////
  
  // tau parts:
  // 1/eps (sigma, tau)_K + (u, div tau)_K - (u_hat, tau_n)_dK
  bf->addTerm(sigma1 / epsilon, tau->x()); 
  bf->addTerm(sigma2 / epsilon, tau->y()); 
  bf->addTerm(u, tau->div());
  bf->addTerm( - uhat, tau->dot_normal() );

  // v:
  // (sigma, grad v)_K - (sigma_hat_n, v)_dK - (u, beta dot grad v) + (u_hat * n dot beta, v)_dK
  bf->addTerm( sigma1, v->dx() );
  bf->addTerm( sigma2, v->dy() );
  bf->addTerm( -u, beta * v->grad());
  bf->addTerm( beta_n_u_minus_sigma_hat, v);

  // ==================== SET INITIAL GUESS ==========================
  mesh->registerSolution(backgroundFlow);
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[BurgersBilinearForm::U] = u0;
  functionMap[BurgersBilinearForm::SIGMA_1] = zero;
  functionMap[BurgersBilinearForm::SIGMA_2] = zero;
 
  backgroundFlow->projectOntoMesh(functionMap);
  // ==================== END SET INITIAL GUESS ==========================

  ////////////////////////////////////////////////////////////////////
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////
  // function to scale the squared guy by epsilon/h
  FunctionPtr epsilonOverHScaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(tau);
  ip->addTerm(tau->div());
  ip->addTerm( epsilonOverHScaling * v );
  ip->addTerm( sqrt(sqrt(epsilon)) * v->grad() );
  ip->addTerm( beta * v->grad() );

  ////////////////////////////////////////////////////////////////////
  // DEFINE RHS
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  rhs->addTerm( (e1 * u_prev_squared_div2 + e2 * u_prev) * v->grad() - u_prev * tau->div());

  ////////////////////////////////////////////////////////////////////
  // DEFINE PENALTY BC
  ////////////////////////////////////////////////////////////////////
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new TopBoundary );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(outflowBoundary) );
  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  LinearTermPtr sigma_hat = beta * uhat->times_normal() - beta_n_u_minus_sigma_hat;
  pc->addConstraint(sigma_hat==zero,outflowBoundary);
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE DIRICHLET BC
  ////////////////////////////////////////////////////////////////////
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  Teuchos::RCP<BCEasy> inflowBC = Teuchos::rcp( new BCEasy );
  FunctionPtr u0_squared_div_2 = 0.5 * u0 * u0;
  inflowBC->addDirichlet(beta_n_u_minus_sigma_hat,inflowBoundary, 
                         ( e1 * u0_squared_div_2 + e2 * u0) * n );
  
  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, inflowBC, rhs, ip));
  mesh->registerSolution(solution);
  solution->setFilter(pc);
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE REFINEMENT STRATEGY
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));
  
  int numRefs = 5;
  
  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy;
  solveStrategy = Teuchos::rcp( new NonlinearSolveStrategy(backgroundFlow, solution, stepSize,
                                                           nonlinearRelativeEnergyTolerance));

  ////////////////////////////////////////////////////////////////////
  // SOLVE 
  ////////////////////////////////////////////////////////////////////
  
  for (int refIndex=0;refIndex<numRefs;refIndex++){    
    solveStrategy->solve(rank==0);       // print to console on rank 0
    refinementStrategy->refine(rank==0); // print to console on rank 0
  }
  
  // one more nonlinear solve on refined mesh
  int numNRSteps = 5;
  for (int i=0;i<numNRSteps;i++){
    solution->solve(false); // false: don't use MUMPS
    backgroundFlow->addSolution(solution,1.0);
  }
  
  if (rank==0){
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::U, "u_ref.m");
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::SIGMA_1, "sigmax.m");
    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::SIGMA_2, "sigmay.m");
    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "du_hat_ref.dat");
  }
  
  return 0;
}