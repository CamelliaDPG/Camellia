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

bool functionsAgree(FunctionPtr f1, FunctionPtr f2, BasisCachePtr basisCache) {
  if (f2->rank() != f1->rank() ) {
    cout << "f1->rank() " << f1->rank() << " != f2->rank() " << f2->rank() << endl;
    return false;
  }
  int rank = f1->rank();
  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  int spaceDim = basisCache->getPhysicalCubaturePoints().dimension(2);
  Teuchos::Array<int> dim;
  dim.append(numCells);
  dim.append(numPoints);
  for (int i=0; i<rank; i++) {
    dim.append(spaceDim);
  }
  FieldContainer<double> f1Values(dim);
  FieldContainer<double> f2Values(dim);
  f1->values(f1Values,basisCache);
  f2->values(f2Values,basisCache);
  
  double tol = 1e-14;
  double maxDiff;
  bool functionsAgree = TestSuite::fcsAgree(f1Values,f2Values,tol,maxDiff);
  if ( ! functionsAgree ) {
    cout << "Test failed: f1 and f2 disagree; maxDiff " << maxDiff << ".\n";
    cout << "f1Values: \n" << f1Values;
    cout << "f2Values: \n" << f2Values;
  } else {
    cout << "f1 and f2 agree!" << endl;
  }
  return functionsAgree;
}

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
  Teuchos::RCP<RHSEasy> otherRHS = Teuchos::rcp( new RHSEasy );
  vector<double> e1_div2 = e1;
  e1_div2[0] /= 2.0;
  FunctionPtr rhsBeta = (e1_div2 * beta * e1 + Teuchos::rcp( new ConstantVectorFunction( e2 ) )) * u_prev;
  otherRHS->addTerm( rhsBeta * v->grad() - u_prev * tau->div() );

  Teuchos::RCP<BurgersProblem> problem = Teuchos::rcp( new BurgersProblem(oldBurgersBF) );
  
  expectedValues.resize(numCells, testOrdering->totalDofs() );
  actualValues.resize  (numCells, testOrdering->totalDofs() );

  problem->integrateAgainstStandardBasis(expectedValues,testOrdering,basisCache);
  otherRHS->integrateAgainstStandardBasis(actualValues,testOrdering,basisCache);
  
  tol = 1e-14;
  maxDiff = 0.0;
  if (rank==0) {
    if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
      cout << "Test failed: old RHS differs from new (\"otherRHS\"); maxDiff " << maxDiff << ".\n";
      cout << "Old: \n" << expectedValues;
      cout << "New: \n" << actualValues;
      cout << "testOrdering: \n" << *testOrdering;
    } else {
      cout << "Old and new RHS (\"otherRHS\") agree!!\n";
    }
  }
  
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  rhs->addTerm( (e1 * u_prev_squared_div2 + e2 * u_prev) * v->grad() - u_prev * tau->div());
  
  if (! functionsAgree(e2 * u_prev, 
                       Teuchos::rcp( new ConstantVectorFunction( e2 ) ) * u_prev,
                       basisCache) ) {
    cout << "two like functions differ...\n";
  }
  
  FunctionPtr e1_f = Teuchos::rcp( new ConstantVectorFunction( e1 ) );
  FunctionPtr e2_f = Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  FunctionPtr one  = Teuchos::rcp( new ConstantScalarFunction( 1.0 ) );
  if (! functionsAgree( Teuchos::rcp( new ProductFunction(e1_f, (e1_f + e2_f)) ), // e1_f * (e1_f + e2_f)
                        one,
                        basisCache) ) {
    cout << "two like functions differ...\n";
  }
  
  if (! functionsAgree(u_prev_squared_div2, 
                       (e1_div2 * beta) * u_prev,
                       basisCache) ) {
    cout << "two like functions differ...\n";
  }
  
  if (! functionsAgree(e1 * u_prev_squared_div2, 
                       (e1_div2 * beta * e1) * u_prev,
                       basisCache) ) {
    cout << "two like functions differ...\n";
  }
  
  if (! functionsAgree(e1 * u_prev_squared_div2 + e2 * u_prev, 
                       (e1_div2 * beta * e1 + Teuchos::rcp( new ConstantVectorFunction( e2 ) )) * u_prev,
                       basisCache) ) {
    cout << "two like functions differ...\n";
  }
  
  problem->integrateAgainstStandardBasis(expectedValues,testOrdering,basisCache);
  rhs->integrateAgainstStandardBasis(actualValues,testOrdering,basisCache);
  
  tol = 1e-14;
  maxDiff = 0.0;
  if (rank==0) {
    if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
      cout << "Test failed: old RHS differs from new (\"rhs\"); maxDiff " << maxDiff << ".\n";
      cout << "Old: \n" << expectedValues;
      cout << "New: \n" << actualValues;
      cout << "testOrdering: \n" << *testOrdering;
    } else {
      cout << "Old and new RHS (\"rhs\") agree!!\n";
    }
  }
  
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new TopBoundary );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(outflowBoundary) );
  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  LinearTermPtr sigma_hat = beta * uhat->times_normal() - beta_n_u_minus_sigma_hat;
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  pc->addConstraint(sigma_hat==zero,outflowBoundary);
  
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  Teuchos::RCP<BCEasy> inflowBC = Teuchos::rcp( new BCEasy );
  FunctionPtr u0_squared_div_2 = 0.5 * u0 * u0;
  inflowBC->addDirichlet(beta_n_u_minus_sigma_hat,inflowBoundary, ( e1 * u0_squared_div_2 + e2 * u0) * n );
  
  // create a solution object  
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, inflowBC, rhs, ip));
  mesh->registerSolution(solution);
  
  solution->setFilter(pc);
  
  // old penalty filter:
  Teuchos::RCP<LocalStiffnessMatrixFilter> penaltyBC = Teuchos::rcp(new PenaltyMethodFilter(problem));
//  solution->setFilter(penaltyBC);
  
  // compare old and new filters
  elemType = mesh->getElement(0)->elementType();
  trialOrdering = elemType->trialOrderPtr;
  testOrdering = elemType->testOrderPtr;
  // stiffness
  expectedValues.resize(numCells, trialOrdering->totalDofs(), trialOrdering->totalDofs() );
  actualValues.resize  (numCells, trialOrdering->totalDofs(), trialOrdering->totalDofs() );
  expectedValues.initialize(0.0);
  actualValues.initialize(0.0);
  // load
  FieldContainer<double> expectedLoad(numCells, trialOrdering->totalDofs() );
  FieldContainer<double> actualLoad(numCells, trialOrdering->totalDofs() );
    
  penaltyBC->filter(expectedValues,expectedLoad,basisCache,mesh,problem);
  pc->filter(actualValues,actualLoad,basisCache,mesh,problem);
  
  maxDiff = 0.0;
  if (rank==0) {
    if ( ! TestSuite::fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
      cout << "Test failed: old penalty stiffness differs from new; maxDiff " << maxDiff << ".\n";
      cout << "Old: \n" << expectedValues;
      cout << "New: \n" << actualValues;
      cout << "trialOrdering: \n" << *trialOrdering;
    } else {
      cout << "Old and new penalty stiffness agree!!\n";
    }
  }
  if (rank==0) {
    if ( ! TestSuite::fcsAgree(expectedLoad,actualLoad,tol,maxDiff) ) {
      cout << "Test failed: old penalty load differs from new; maxDiff " << maxDiff << ".\n";
      cout << "Old: \n" << expectedValues;
      cout << "New: \n" << actualValues;
      cout << "trialOrdering: \n" << *trialOrdering;
    } else {
      cout << "Old and new penalty load agree!!\n";
    }
  }
  
  // define refinement strategy:
  Teuchos::RCP<RefinementStrategy> refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));
  
  // =================== END INITIALIZATION CODE ==========================
  
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