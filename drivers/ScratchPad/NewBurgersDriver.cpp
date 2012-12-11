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
#include "../DPGTests/TestSuite.h"
#include "RefinementPattern.h"
#include "PenaltyConstraints.h"
#include "RieszRep.h"
#include "HessianFilter.h"
#include <sstream>

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;

using namespace std;

class PositivePart : public Function {
  FunctionPtr _f;
public:
  PositivePart(FunctionPtr f) {
    _f = f;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints);
    _f->values(values,basisCache);
    
    for (int i = 0;i<numCells;i++){
      for (int j = 0;j<numPoints;j++){
	if (values(i,j)<0){
	  values(i,j) = 0.0;
	}
      }
    }
  }
};


class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
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
  int polyOrder = 2;
  
  // define our manufactured solution or problem bilinear form:
  double epsilon = 1e-3;
  bool useTriangles = false;
  
  int pToAdd = 2;
  int nCells = 2;
  if ( argc > 1) {
    nCells = atoi(argv[1]);
    if (rank==0){
      cout << "numCells = " << nCells << endl;
    }
  }
 int numSteps = 20;
  if ( argc > 2) {
    numSteps = atoi(argv[2]);
    if (rank==0){
      cout << "num NR steps = " << numSteps << endl;
    }
  }
 int useHessian = 0; // defaults to "not use"
  if ( argc > 3) {
    useHessian = atoi(argv[3]);
    if (rank==0){
      cout << "useHessian = " << useHessian << endl;
    }
  } 

  int thresh = numSteps; // threshhold for when to apply linesearch/hessian
  if ( argc > 4) {
    thresh = atoi(argv[4]);
    if (rank==0){
      cout << "thresh = " << thresh << endl;
    }
  }

  int H1Order = polyOrder + 1;
  
  double energyThreshold = 0.2; // for mesh refinements
  double nonlinearStepSize = 0.5;
  double nonlinearRelativeEnergyTolerance = 1e-8; // used to determine convergence of the nonlinear solution
  
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
  Teuchos::RCP<Mesh> mesh = Mesh::buildUnitQuadMesh(nCells, bf, H1Order, H1Order+pToAdd);
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
  functionMap[u->ID()] = u0;
  functionMap[sigma1->ID()] = zero;
  functionMap[sigma2->ID()] = zero;
 
  backgroundFlow->projectOntoMesh(functionMap);
  // ==================== END SET INITIAL GUESS ==========================

  ////////////////////////////////////////////////////////////////////
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////
  // function to scale the squared guy by epsilon/h
  FunctionPtr epsilonOverHScaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm( epsilonOverHScaling * (1.0/sqrt(epsilon))* tau);
  ip->addTerm( tau->div());
  //  ip->addTerm( epsilonOverHScaling * v );
  ip->addTerm( v );
  ip->addTerm( sqrt(epsilon) * v->grad() );
  ip->addTerm(v->grad());
  //  ip->addTerm( beta * v->grad() );

  ////////////////////////////////////////////////////////////////////
  // DEFINE RHS
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  
  rhs->addTerm((e1 * u_prev_squared_div2 + e2 * u_prev) * v->grad() - u_prev * tau->div());

  ////////////////////////////////////////////////////////////////////
  // DEFINE DIRICHLET BC
  ////////////////////////////////////////////////////////////////////
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new TopBoundary);
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(outflowBoundary) );
  Teuchos::RCP<BCEasy> inflowBC = Teuchos::rcp( new BCEasy );
  FunctionPtr u0_squared_div_2 = 0.5 * u0 * u0;
  inflowBC->addDirichlet(beta_n_u_minus_sigma_hat,inflowBoundary, 
                         ( e1 * u0_squared_div_2 + e2 * u0) * n );
  
  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, inflowBC, rhs, ip));
  mesh->registerSolution(solution);

  ////////////////////////////////////////////////////////////////////
  // WARNING: UNFINISHED HESSIAN BIT
  ////////////////////////////////////////////////////////////////////
  VarFactory hessianVars = varFactory.getBubnovFactory(VarFactory::BUBNOV_TRIAL);
  VarPtr du = hessianVars.test(u->ID());
  BFPtr hessianBF = Teuchos::rcp( new BF(hessianVars) ); // initialize bilinear form
  //  FunctionPtr e_v = Function::constant(1.0); // dummy error rep function for now - should do nothing

  FunctionPtr u_current  = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );

  FunctionPtr sig1_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
  FunctionPtr sig2_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );
  FunctionPtr sig_prev = (e1*sig1_prev + e2*sig2_prev);
  FunctionPtr fnhat = Teuchos::rcp(new PreviousSolutionFunction(solution,beta_n_u_minus_sigma_hat));
  FunctionPtr uhat_prev = Teuchos::rcp(new PreviousSolutionFunction(solution,uhat));
  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  residual->addTerm(fnhat*v - (e1 * (u_prev_squared_div2 - sig1_prev) + e2 * (u_prev - sig2_prev)) * v->grad());
  residual->addTerm((1/epsilon)*sig_prev * tau + u_prev * tau->div() - uhat_prev*tau->dot_normal());

  LinearTermPtr Bdu = Teuchos::rcp(new LinearTerm);// residual
  Bdu->addTerm( u_current*tau->div() - u_current*(beta*v->grad()));

  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  Teuchos::RCP<RieszRep> duRiesz = Teuchos::rcp(new RieszRep(mesh, ip, Bdu));
  riesz->computeRieszRep();
  FunctionPtr e_v = Teuchos::rcp(new RepFunction(v,riesz));
  e_v->writeValuesToMATLABFile(mesh, "e_v.m");
  FunctionPtr posErrPart = Teuchos::rcp(new PositivePart(e_v->dx()));
  hessianBF->addTerm(e_v->dx()*u,du); 
  //  hessianBF->addTerm(posErrPart*u,du); 
  Teuchos::RCP<HessianFilter> hessianFilter = Teuchos::rcp(new HessianFilter(hessianBF));

  if (useHessian){
    solution->setWriteMatrixToFile(true,"hessianStiffness.dat");
  }else{
    solution->setWriteMatrixToFile(true,"stiffness.dat");    
  }

  Teuchos::RCP< LineSearchStep > LS_Step = Teuchos::rcp(new LineSearchStep(riesz));
  ofstream out;
  out.open("Burgers.txt"); 
  double NL_residual = 9e99;
  for (int i = 0;i<numSteps;i++){
    solution->solve(false); // do one solve to initialize things...   
    double stepLength = 1.0;
    stepLength = LS_Step->stepSize(backgroundFlow,solution, NL_residual);
    if (useHessian){
      solution->setFilter(hessianFilter);        
    }
    backgroundFlow->addSolution(solution,stepLength);
    NL_residual = LS_Step->getNLResidual();
    if (rank==0){
      cout << "NL residual after adding = " << NL_residual << " with step size " << stepLength << endl;    
      out << NL_residual << endl; // saves initial NL error     
    }
  }
  out.close();
 
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE REFINEMENT STRATEGY
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));
  
  int numRefs = 0;
  
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
  //  solveStrategy->solve(rank==0);

  if (rank==0){ 
    backgroundFlow->writeToVTK("Burgers.vtu",min(H1Order+1,4));
    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "burgers.dat");
    cout << "wrote solution files" << endl;
  }

  return 0;
}

