#include "../Burgers/BurgersBilinearForm.h"

#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "PenaltyMethodFilter.h"
#include "../Burgers/BurgersProblem.h"
#include "Projector.h"
#include "../Burgers/BurgersInnerProduct.h"
#include "ZeroFunction.h"
#include "../Burgers/InitialGuess.h"

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
  int horizontalCells = 1, verticalCells = 1;
  
  double energyThreshold = 0.2; // for mesh refinements
  double nonlinearStepSize = 0.5;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  
  ////////////////////////////////////////////////////////////////////
  // SET UP PROBLEM 
  ////////////////////////////////////////////////////////////////////
  
  Teuchos::RCP<BurgersBilinearForm> oldBurgersBF = Teuchos::rcp(new BurgersBilinearForm(epsilon));
  
  // new-style bilinear form definition
  VarFactory varFactory;
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_hat = varFactory.fluxVar("\\widehat{\\beta_n u - \\sigma_n}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  VarPtr tau = varFactory.testVar("\\tau",HDIV);
  VarPtr v = varFactory.testVar("v",HGRAD);
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  
  Teuchos::RCP<Solution> backgroundFlow = Teuchos::rcp(new Solution(mesh, Teuchos::rcp((BC*)NULL) , Teuchos::rcp((RHS*)NULL), Teuchos::rcp((DPGInnerProduct*)NULL))); // create null solution 
  oldBurgersBF->setBackgroundFlow(backgroundFlow);
  
  // tau parts:
  // 1/eps (sigma, tau)_K + (u, div tau)_K - (u_hat, tau_n)_dK
  bf->addTerm(sigma1 / epsilon, tau->x()); 
  bf->addTerm(sigma2 / epsilon, tau->y()); 
  bf->addTerm(u, tau->div());
  bf->addTerm( - uhat, tau->dot_normal() );

  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  
  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u) );
  FunctionPtr beta = e1 * u_prev + Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  
  // v:
  // (sigma, grad v)_K - (sigma_hat_n, v)_dK - (u, beta dot grad v) + (u_hat * n dot beta, v)_dK
  bf->addTerm( sigma1, v->dx() );
  bf->addTerm( sigma2, v->dy() );
  bf->addTerm( -u, beta * v->grad());
  bf->addTerm( beta_n_u_minus_sigma_hat, v);

  // ==================== SET INITIAL GUESS ==========================
  mesh->registerSolution(backgroundFlow);
  
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  functionMap[BurgersBilinearForm::U] = Teuchos::rcp(new InitialGuess());
  functionMap[BurgersBilinearForm::SIGMA_1] = Teuchos::rcp(new ZeroFunction());
  functionMap[BurgersBilinearForm::SIGMA_2] = Teuchos::rcp(new ZeroFunction());
  
  backgroundFlow->projectOntoMesh(functionMap);
  
  // ==================== END SET INITIAL GUESS ==========================
  // compare stiffness matrices for first linear step:
  int trialOrder = 1;
  pToAdd = 0;
  int testOrder = trialOrder + pToAdd;
  CellTopoPtr quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  DofOrderingFactory dofOrderingFactory(bf);
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(testOrder, *quadTopoPtr);
  DofOrderingPtr trialOrdering = dofOrderingFactory.trialOrdering(trialOrder, *quadTopoPtr);
  
  int numCells = 1;
  // just use testOrdering for both trial and test spaces (we only use to define BasisCache)
  ElementTypePtr elemType  = Teuchos::rcp( new ElementType(trialOrdering, testOrdering, quadTopoPtr) );
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType) );
  quadPoints.resize(1,quadPoints.dimension(0),quadPoints.dimension(1));
  basisCache->setPhysicalCellNodes(quadPoints,vector<int>(1),true); // true: do create side cache
  FieldContainer<double> cellSideParities(numCells,quadTopoPtr->getSideCount());
  cellSideParities.initialize(1.0); // not worried here about neighbors actually having opposite parity -- just want the two BF implementations to agree...
  FieldContainer<double> expectedValues(numCells, testOrdering->totalDofs(), trialOrdering->totalDofs() );
  FieldContainer<double> actualValues(numCells, testOrdering->totalDofs(), trialOrdering->totalDofs() );
  oldBurgersBF->stiffnessMatrix(expectedValues, elemType, cellSideParities, basisCache);
  bf->stiffnessMatrix(actualValues, elemType, cellSideParities, basisCache);
  
  // compare beta's as well
  FieldContainer<double> expectedBeta = oldBurgersBF->getBeta(basisCache);
  Teuchos::Array<int> dim;
  expectedBeta.dimensions(dim);
  FieldContainer<double> actualBeta(dim);
  beta->values(actualBeta,basisCache);
  
  double tol = 1e-14;
  double maxDiff;
  if (rank == 0) {
    if ( ! TestSuite::fcsAgree(expectedBeta,actualBeta,tol,maxDiff) ) {
      cout << "Test failed: old Burgers beta differs from new; maxDiff " << maxDiff << ".\n";
      cout << "Old beta: \n" << expectedBeta;
      cout << "New beta: \n" << actualBeta;
    } else {
      cout << "Old and new Burgers beta agree!!\n";
    }

    if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
      cout << "Test failed: old Burgers stiffness differs from new; maxDiff " << maxDiff << ".\n";
      cout << "Old: \n" << expectedValues;
      cout << "New: \n" << actualValues;
      cout << "TrialDofOrdering: \n" << *trialOrdering;
      cout << "TestDofOrdering:\n" << *testOrdering;
    } else {
      cout << "Old and new Burgers stiffness agree!!\n";
    }
  }
  
  
  // define our inner product:
  // Teuchos::RCP<BurgersInnerProduct> ip = Teuchos::rcp( new BurgersInnerProduct( bf, mesh ) );
  
  // function to scale the squared guy by epsilon/h
  FunctionPtr epsilonOverHScaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(tau);
  ip->addTerm(tau->div());
  ip->addTerm( epsilonOverHScaling * v );
  ip->addTerm( sqrt(sqrt(epsilon)) * v->grad() );
  ip->addTerm( beta * v->grad() );

  // use old IP instead, for now...
  Teuchos::RCP<BurgersInnerProduct> oldIP = Teuchos::rcp( new BurgersInnerProduct( oldBurgersBF, mesh ) );
  
  expectedValues.resize(numCells, testOrdering->totalDofs(), testOrdering->totalDofs() );
  actualValues.resize  (numCells, testOrdering->totalDofs(), testOrdering->totalDofs() );
  
  BasisCachePtr ipBasisCache = Teuchos::rcp( new BasisCache(elemType, true) ); // true: test vs. test
  ipBasisCache->setPhysicalCellNodes(quadPoints,vector<int>(1),false); // false: don't create side cache

  oldIP->computeInnerProductMatrix(expectedValues,testOrdering,ipBasisCache);
  ip->computeInnerProductMatrix(actualValues,testOrdering,ipBasisCache);
  
  tol = 1e-14;
  maxDiff = 0.0;
  if (rank==0) {
    if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
      cout << "Test failed: old inner product differs from new IP; maxDiff " << maxDiff << ".\n";
      cout << "Old: \n" << expectedValues;
      cout << "New IP: \n" << actualValues;
      cout << "testOrdering: \n" << *testOrdering;
    } else {
      cout << "Old inner product and new IP agree!!\n";
    }
  }
  
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  // the RHS as implemented by BurgersProblem divides the first component of beta by 2.0
  // so we follow that.  I've not done the math; just imitating the code...
//  vector<double> e1_div2 = e1;
//  e1_div2[0] /= 2.0;
//  FunctionPtr rhsBeta = (e1_div2 * beta * e1 + Teuchos::rcp( new ConstantVectorFunction( e2 ) )) * u_prev;
//  rhs->addTerm( rhsBeta * v->grad() - u_prev * tau->div() );

  // the below should be equivalent to the above commented-out code
  // it's probably a little more efficient this way, but the real
  // reason for trying this is because the other one isn't working just yet
  // TODO: write a test asserting that these two ways of specifying the RHS are equivalent...
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  rhs->addTerm( (e1 * u_prev_squared_div2 + e2 * u_prev) * v->grad() - u_prev * tau->div());

  // create a solution object
  Teuchos::RCP<BurgersProblem> problem = Teuchos::rcp( new BurgersProblem(oldBurgersBF) );
  
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, problem, rhs, ip));
  mesh->registerSolution(solution);
  Teuchos::RCP<LocalStiffnessMatrixFilter> penaltyBC = Teuchos::rcp(new PenaltyMethodFilter(problem));
  solution->setFilter(penaltyBC);
  
  // define refinement strategy:
  Teuchos::RCP<RefinementStrategy> refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));
  
  // =================== END INITIALIZATION CODE ==========================
  
  //  solution->solve(false);
  //  backgroundFlow->addSolution(solution,.5);
  //  solution->solve(false);
  //  backgroundFlow->addSolution(solution,.5);
  //  return 0;
  
  // refine the spectral mesh, for comparability with the original Burgers' driver
  mesh->hRefine(vector<int>(1),RefinementPattern::regularRefinementPatternQuad());
  
  int numRefs = 5;
  
  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy = Teuchos::rcp( 
       new NonlinearSolveStrategy(backgroundFlow, solution, stepSize, nonlinearRelativeEnergyTolerance)
                                                                    );
  
  for (int refIndex=0;refIndex<numRefs;refIndex++){    
    solveStrategy->solve(rank==0);
    refinementStrategy->refine(rank==0); // print to console on rank 0
  }
  
  // one more nonlinear solve on refined mesh
  int numNRSteps = 5;
  for (int i=0;i<numNRSteps;i++){
    solution->solve(false);    
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