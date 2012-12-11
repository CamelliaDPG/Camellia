#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CondensationSolver.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "RieszRep.h"
#include "BasisFactory.h" // for test
#include "HessianFilter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

typedef Teuchos::RCP<IP> IPPtr;
typedef Teuchos::RCP<BF> BFPtr;

double pi = 2.0*acos(0.0);

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

class invSqrtHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return sqrt(1.0/h);
  }
};

class InflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol); // left inflow
    bool yMatch = ((abs(y)<tol) || (abs(y-1.0)<tol)); // top/bottom
    return xMatch || yMatch;
  }
};

class OutflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x-1.0)<tol);
    return xMatch;
  }
};

// inflow values for u
class Uex : public Function {
  double _eps;
  int _returnID;
public:
  Uex(double eps) : Function(0) {
    _eps = eps;
    _returnID = 0;
  }
  Uex(double eps,int trialID) : Function(0) {
    _eps = eps;
    _returnID = trialID;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {

    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);

	double C0 = 0.0;// average of u0
	double u = C0;
	double u_x = 0.0;
	double u_y = 0.0;  	
	for (int n = 1;n<20;n++){

	  double lambda = n*n*pi*pi*_eps;
	  double d = sqrt(1.0+4.0*_eps*lambda);
	  double r1 = (1.0+d)/(2.0*_eps);
	  double r2 = (1.0-d)/(2.0*_eps);
    
	  double Cn = 0.0;            
	  //	  if (n==1){
	  //	    Cn = 1.0; // first term only
	  //	  }    
	  
	  // discontinuous hat 
	  Cn = -1 + cos(n*pi/2)+.5*n*pi*sin(n*pi/2) + sin(n*pi/4)*(n*pi*cos(n*pi/4)-2*sin(3*n*pi/4));
	  Cn /= (n*pi);
	  Cn /= (n*pi);    
	  
	  // normal stress outflow
	  double Xbottom;
	  double Xtop;
	  double dXtop;
	  // wall, zero outflow
	  Xtop = (exp(r2*(x-1))-exp(r1*(x-1)));
	  Xbottom = (exp(-r2)-exp(-r1));
	  dXtop = (exp(r2*(x-1))*r2-exp(r1*(x-1))*r1);    

	  double X = Xtop/Xbottom;
	  double dX = dXtop/Xbottom;
	  double Y = Cn*cos(n*pi*y);
	  double dY = -Cn*n*pi*sin(n*pi*y);
    
	  u += X*Y;
	  u_x += _eps * dX*Y;
	  u_y += _eps * X*dY;
	}
	if (_returnID==0){
	  values(cellIndex,ptIndex) = u;
	}else if (_returnID==1){
	  values(cellIndex,ptIndex) = u_x;
	}else if (_returnID==2){
	  values(cellIndex,ptIndex) = u_y;
	}
      }
    }
  }
};


// inflow values for u
class l2NormOfVector : public Function {
  FunctionPtr _beta;
public:
  l2NormOfVector(FunctionPtr beta) : Function(0){
    _beta = beta;
  }

  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints,2);
    _beta->values(beta_pts,basisCache);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	double b1 =  beta_pts(cellIndex,ptIndex,0);
	double b2 =  beta_pts(cellIndex,ptIndex,1);
	double beta_norm =b1*b1 + b2*b2;
	values(cellIndex,ptIndex) = sqrt(beta_norm);
      }
    }
  }
};

class DivTestFunction : public Function{
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	values(cellIndex,ptIndex) = 1.0-2.0*x;// div F = 1.0-2.0*x;
      }
    }
  }
};

class TestFunction : public Function{  
  TestFunction():Function(1){}

  /*
  FunctionPtr dx(){
    FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0));
    FunctionPtr two_x = 2.0*(FunctionPtr)Teuchos::rcp(new Xn(1));
    return Teuchos::rcp(new VectorizedFunction(one-two_x,Function::zero()));
  }
  FunctionPtr dy(){
    return Teuchos::rcp(new VectorizedFunction(Function::zero(),Function::zero());
  }
  */
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	values(cellIndex,ptIndex,0) = x*(1.0-x);// div F = 1.0-2.0*x;
	values(cellIndex,ptIndex,1) = 0.0;// div F = 1.0-2.0*x;
      }
    }
  }
};


class TestFunction1 : public Function{
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	//	values(cellIndex,ptIndex) = .5*x;
	values(cellIndex,ptIndex) = x*(1.0-x);// div F = 1.0-2.0*x;
      }
    }
  }
};

class TestFunction2 : public Function{
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	//	values(cellIndex,ptIndex) = .5*y;
	values(cellIndex,ptIndex) = 0.0;
      }
    }
  }
};

class TestFunctionBoundary : public Function{
  bool boundaryValueOnly(){ 
    return true; 
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
   
    const FieldContainer<double> normals = basisCache->getSideNormals();
    const FieldContainer<double> points =  basisCache->getPhysicalCubaturePoints();
    double tol = 1e-12;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = points(cellIndex,ptIndex,0);
        double y = points(cellIndex,ptIndex,1);
	double n1 = normals(cellIndex,ptIndex,0);
	double n2 = normals(cellIndex,ptIndex,1);
	
	//	if (abs(x)<tol || abs(x-1.0)<tol || abs(y)<tol || abs(y-1.0)<tol){
	values(cellIndex,ptIndex) =  (x*(1.0-x))*n1;
	//	}else{
	//	  values(cellIndex,ptIndex) =  0.0;
	//	}
      }
    }
  }
};

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();  
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  int rank = 0;
  int numProcs = 1;  
  Epetra_SerialComm Comm;
#endif
  
  int nCells = 2;
  if ( argc > 1) {
    nCells = atoi(argv[1]);
    if (rank==0){
      cout << "numCells = " << nCells << endl;
    }
  }
  
  double eps = .01;
  if (argc > 2){
    eps = atof(argv[2]);
    if (rank==0)
      cout << "eps = " << eps << endl;
  }

  double squareSize = 1.0;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

   // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  //  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(eps) ); 
  FunctionPtr ip_scaling = Teuchos::rcp( new ConstantScalarFunction(1.0));
  FunctionPtr tau_ip_scaling = Teuchos::rcp( new EpsilonScaling(eps) ); 

  robIP->addTerm( ip_scaling * v);
  robIP->addTerm( sqrt(eps) * v->grad() );
  robIP->addTerm( beta * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( tau_ip_scaling/sqrt(eps) * tau );
  
  ////////////////////   SPECIFY RHS   ///////////////////////

  FunctionPtr zero = Function::constant(0.0);
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary(beta) );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary);

  FunctionPtr u_exact = Teuchos::rcp( new Uex(eps,0) );
  FunctionPtr sig1_exact = Teuchos::rcp( new Uex(eps,1) );
  FunctionPtr sig2_exact = Teuchos::rcp( new Uex(eps,2) );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  bc->addDirichlet(uhat, outflowBoundary, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*one);  

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  int H1Order = order+1; int pToAdd = 2;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = squareSize;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = squareSize;
  quadPoints(2,1) = squareSize;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = squareSize;
 
  int horizontalCells = nCells, verticalCells = nCells;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
    
  ElementTypePtr elemType = mesh->getElement(0)->elementType();
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType, mesh));
  
  vector<int> cellIDs;
  vector< ElementPtr > allElems = mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){
    cellIDs.push_back((*elemIt)->cellID());
  }

  bool createSideCacheToo = true;
  
  basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);

  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  
  // PREREFINE TWICE
  double energyThreshold = .2; // for mesh refinements - just to make mesh irregular
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  int numRefs = 0;
  if (rank==0){
    cout << "solving/refining..." << endl;
  }
  for (int i = 0;i<numRefs;i++){
    solution->solve(false);
    refinementStrategy.refine(rank==0); 
  }
  solution->solve(false);  
  FunctionPtr uCopy = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  FunctionPtr sigma1Copy = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
  FunctionPtr sigma2Copy = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );
  FunctionPtr fnhatCopy = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_minus_sigma_n));
  FunctionPtr uhatCopy = Teuchos::rcp( new PreviousSolutionFunction(solution, uhat));
  
  ////////////////////   get residual   ///////////////////////

  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;
  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  FunctionPtr Sigma = e1*sigma1Copy + e2*sigma2Copy;
  
  residual->addTerm(fnhatCopy*v - (beta*uCopy - Sigma)*v->grad());
  residual->addTerm((1.0/eps)*Sigma*tau + uCopy*tau->div() - uhatCopy*tau->dot_normal());
  cout << "making riesz representations" << endl;
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, robIP, residual));
  cout << "computing riesz representations" << endl;
  riesz->computeRieszRep();
  cout << "making rep fxns" << endl;
  cout << "riesz error = " << riesz->getNorm() << endl;
  cout << "energy error = " << solution->energyErrorTotal() << endl;
  FunctionPtr rieszRepFxn = Teuchos::rcp(new RepFunction(v,riesz));

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC, 
							   nullRHS, nullIP) );

  int numGlobalDofs = mesh->numGlobalDofs();
  for (int dofIndex = 0;dofIndex<numGlobalDofs;dofIndex++){
    solnPerturbation->clearSolution(); // clear all solns
    if (solnPerturbation->isFluxOrTraceDof(dofIndex)){
      solnPerturbation->setSolnCoeffForGlobalDofIndex(1.0,dofIndex);     
      LinearTermPtr b_u =  confusionBF->testFunctional(solnPerturbation);
      map<int,FunctionPtr> NL_err_rep_map;
      NL_err_rep_map[v->ID()] = Teuchos::rcp(new RepFunction(v,riesz));
      NL_err_rep_map[tau->ID()] = Teuchos::rcp(new RepFunction(tau,riesz));
      FunctionPtr b_e = b_u->evaluate(NL_err_rep_map, true); // use boundary part only if flux or trace
      double be_int = b_e->integrate(mesh,10);      
      cout << "bilinear form evaluated at flux dof/error test = " << be_int << endl;
    }
  }


  if (rank==0){    
    rieszRepFxn->writeValuesToMATLABFile(mesh,"err_rep.m");
    solution->writeFluxesToFile(uhat->ID(), "uhatCond.dat");
    solution->writeFluxesToFile(beta_n_u_minus_sigma_n->ID(), "fhatCond.dat");
    solution->writeToVTK("U.vtu",min(H1Order+1,4));
    
    cout << "wrote files: rates.vtu, uhat.dat\n";
  }
  // =========================================
  /* 
  VarFactory hessianVars = varFactory.getBubnovFactory(VarFactory::BUBNOV_TRIAL);
  VarPtr du = hessianVars.test(u->ID());
  VarPtr dsigma1 = hessianVars.test(sigma1->ID());
  VarPtr dsigma2 = hessianVars.test(sigma2->ID());
  BFPtr hessianBF = Teuchos::rcp( new BF(hessianVars) ); // initialize bilinear form 
  FunctionPtr e_v = Teuchos::rcp(new RepFunction(v,riesz));
  FunctionPtr e_tau = Teuchos::rcp(new RepFunction(tau,riesz));
  
  hessianBF->addTerm(scalingFxn*u,du); // to scale u?
  hessianBF->addTerm(10.0*u,du); // to increase sigma (stress)

  Teuchos::RCP<HessianFilter> hessianFilter = Teuchos::rcp(new HessianFilter(hessianBF));
  solution->setFilter(hessianFilter);
  solution->solve(false);  

  if (rank==0){    
    (e_v->dx()+e_tau->x()/eps)->writeValuesToMATLABFile(mesh,"artdiff_x.m");    
    (e_v->dy()+e_tau->y()/eps)->writeValuesToMATLABFile(mesh,"artdiff_y.m");    
    (e_tau->x()/eps)->writeValuesToMATLABFile(mesh,"artdiff_tau_x.m");    
    (e_tau->y()/eps)->writeValuesToMATLABFile(mesh,"artdiff_tau_y.m");    
    solution->writeToVTK("UPost.vtu",min(H1Order+1,4));
    
    cout << "wrote 2nd set of files: rates.vtu, uhat.dat\n";
  }
  */

  return 0;
}


