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
#include "RieszRep.h"
#include "HessianFilter.h"
#include <sstream>
#include "SolutionTests.h"

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
  int polyOrder = 1;
  
  // define our manufactured solution or problem bilinear form:
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

  int H1Order = polyOrder + 1;
  
  double energyThreshold = 0.2; // for mesh refinements
  double nonlinearStepSize = 0.5;
  double nonlinearRelativeEnergyTolerance = 1e-8; // used to determine convergence of the nonlinear solution
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE VARIABLES 
  ////////////////////////////////////////////////////////////////////
  
  // new-style bilinear form definition
  VarFactory varFactory;
  VarPtr fn = varFactory.fluxVar("\\widehat{\\beta_n_u}");
  VarPtr u = varFactory.fieldVar("u");
  
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
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC, 
							   nullRHS, nullIP) );
  backgroundFlow->setCubatureEnrichmentDegree(10);
  solnPerturbation->setCubatureEnrichmentDegree(10);
  
  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  
  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u) );
  FunctionPtr beta = e1 * u_prev + Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE BILINEAR FORM
  ////////////////////////////////////////////////////////////////////
  
  // v:
  bf->addTerm( -u, beta * v->grad());
  bf->addTerm( fn, v);

  ////////////////////////////////////////////////////////////////////
  // DEFINE RHS
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;  
  rhs->addTerm((e1 * u_prev_squared_div2 + e2 * u_prev) * v->grad());

  // ==================== SET INITIAL GUESS ==========================
  mesh->registerSolution(backgroundFlow);
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  FunctionPtr parity = Teuchos::rcp(new SideParityFunction);

  FunctionPtr u0_squared_div_2 = 0.5 * u0 * u0;

  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()] = u0;
  //  functionMap[fn->ID()] = -(e1 * u0_squared_div_2 + e2 * u0) * n * parity;
  backgroundFlow->projectOntoMesh(functionMap);

  // ==================== END SET INITIAL GUESS ==========================

  ////////////////////////////////////////////////////////////////////
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////

  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm( v );
  ip->addTerm(v->grad());
  //  ip->addTerm( beta * v->grad() );

  ////////////////////////////////////////////////////////////////////
  // DEFINE DIRICHLET BC
  ////////////////////////////////////////////////////////////////////
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new TopBoundary);
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(outflowBoundary) );
  Teuchos::RCP<BCEasy> inflowBC = Teuchos::rcp( new BCEasy );
  inflowBC->addDirichlet(fn,inflowBoundary, 
                         ( e1 * u0_squared_div_2 + e2 * u0) * n );
  //  inflowBC->addDirichlet(fn,inflowBoundary,zero); 
  
  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, inflowBC, rhs, ip));
  mesh->registerSolution(solution);
  solution->setCubatureEnrichmentDegree(10);

  ////////////////////////////////////////////////////////////////////
  // HESSIAN BIT + CHECKS ON GRADIENT + HESSIAN
  ////////////////////////////////////////////////////////////////////

  VarFactory hessianVars = varFactory.getBubnovFactory(VarFactory::BUBNOV_TRIAL);
  VarPtr du = hessianVars.test(u->ID());
  BFPtr hessianBF = Teuchos::rcp( new BF(hessianVars) ); // initialize bilinear form

  FunctionPtr du_current  = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );

  FunctionPtr fnhat = Teuchos::rcp(new PreviousSolutionFunction(solution,fn));
  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  residual->addTerm(fnhat*v - (e1 * (u_prev_squared_div2) + e2 * (u_prev)) * v->grad());

  LinearTermPtr Bdu = Teuchos::rcp(new LinearTerm);// residual
  Bdu->addTerm( - du_current*(beta*v->grad()));

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
  //  ofstream out;
  //  out.open("Burgers.txt"); 
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
      //      out << NL_residual << endl; // saves initial NL error     
    }


    int numGlobalDofs = mesh->numGlobalDofs();
    double fd_gradient;
    for (int dofIndex = 0;dofIndex<numGlobalDofs;dofIndex++){
      // CHECK HESSIAN
      riesz->computeRieszRep();
      double fx = riesz->getNorm();
      
      // create perturbation in direction du
      solnPerturbation->clearSolution(); // clear all solns
      solnPerturbation->setSolnCoeffForGlobalDofIndex(1.0,dofIndex);
      double h = 1e-7;
      backgroundFlow->addSolution(solnPerturbation,h);
      
      riesz->computeRieszRep();
      double fxh = riesz->getNorm();
      double f = fx*fx*.5;
      double fh = fxh*fxh*.5;
      fd_gradient = (fh-f)/h;      
      
      // remove contribution
      backgroundFlow->addSolution(solnPerturbation,-h);
      
      // CHECK GRADIENT
      LinearTermPtr b_u =  bf->testFunctional(solnPerturbation);
      map<int,FunctionPtr> NL_err_rep_map;
      NL_err_rep_map[v->ID()] = Teuchos::rcp(new RepFunction(v,riesz));
      FunctionPtr gradient = b_u->evaluate(NL_err_rep_map, solnPerturbation->isFluxOrTraceDof(dofIndex)); // use boundary part only if flux or trace
      double grad = gradient->integrate(mesh,10);
      double fdgrad = fd_gradient;
      double diff = grad-fdgrad;
      cout << "gradient = " << grad << " and FD gradient = " << fdgrad << endl;
      if (abs(diff)>1e-6){
	int cellID = mesh->getGlobalToLocalMap()[dofIndex].first;
	int localDofIndex = mesh->getGlobalToLocalMap()[dofIndex].second;
	vector<double> centroid = mesh->getCellCentroid(cellID);
	cout << "Found difference of " << diff << " in dof " << dofIndex << ", isTraceDof = " << solnPerturbation->isFluxOrTraceDof(dofIndex) << " with support on cell " << cellID  << " at " << centroid[0] << "," << centroid[1] << ", with local dof index " << localDofIndex << endl;
	if (solnPerturbation->isFluxOrTraceDof(dofIndex)){
	  for (int sideIndex = 0;sideIndex<4;sideIndex++){
	    vector<int> fluxDofInds = mesh->elements()[cellID]->elementType()->trialOrderPtr->getDofIndices(fn->ID(), sideIndex);
	    for (int i = 0;i<fluxDofInds.size();i++){
	      cout << "Side " << sideIndex << " has dof index " << fluxDofInds[i] << endl;
	    }
	  }
	  map<int,FunctionPtr> testMap;
	  testMap[v->ID()] = Function::constant(0.0);
	  double testval = b_u->evaluate(testMap,true)->integrate(mesh,10); //evaluate for boundary flux
	  cout << endl;
	  cout << "integral of this dof = " << testval << endl;
	  cout << endl;
	  
	  NL_err_rep_map[v->ID()]->writeValuesToMATLABFile(mesh,"burgers_nl_err_rep.m");
	  
	}
      }

      cout << "consistency check for perturbation " << SolutionTests::solutionCoefficientsAreConsistent(solnPerturbation) << endl;
      cout << "consistency check for background flow " << SolutionTests::solutionCoefficientsAreConsistent(backgroundFlow) << endl;
      cout << "consistency check for solution " << SolutionTests::solutionCoefficientsAreConsistent(solution) << endl;


    }
    
    
  }

  //  out.close();
   
  if (rank==0){ 
    backgroundFlow->writeToVTK("BurgersTest.vtu",min(H1Order+1,4));
    cout << "wrote solution files" << endl;
  }

  return 0;
}

