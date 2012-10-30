#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CondensationSolver.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "RieszRep.h"
#include "BasisFactory.h" // for test

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
  if (argc > 3){
    squareSize = atof(argv[3]);
    if (rank==0)
      cout << "eps = " << eps << endl;
  }

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
  beta.push_back(0.0);
  
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
  int H1Order = 1; int pToAdd = 1;
  
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
  cellIDs.push_back(0); 
  cellIDs.push_back(1);
  cellIDs.push_back(2);
  cellIDs.push_back(3);
  bool createSideCacheToo = true;
  
  basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);


  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  
  // PREREFINE TWICE
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  int numRefs = 0;
  if (rank==0){
    cout << "refining..." << endl;
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
  FunctionPtr minusOne = Teuchos::rcp(new ConstantScalarFunction(-1.0));
  
  ////////////////////   get residual   ///////////////////////

  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;
  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  LinearTermPtr residualIBP = Teuchos::rcp(new LinearTerm);// residual

  FunctionPtr X = Teuchos::rcp(new Xn(1));
  FunctionPtr Y = Teuchos::rcp(new Yn(1));
  FunctionPtr testFxn1 = X;
  FunctionPtr testFxn2 = Y;
  FunctionPtr divTestFxn = testFxn1->dx() + testFxn2->dy();
  FunctionPtr vectorTest = testFxn1*e1 + testFxn2*e2;
  residual->addTerm(divTestFxn*v);
  residualIBP->addTerm(vectorTest*n*v - vectorTest*v->grad() ); // boundary term
  
  /*
  residual->addTerm(minusOne*beta*uCopy*v->grad());
  residual->addTerm((sigma1Copy*e1 + sigma2Copy*e2)*v->grad());
  residual->addTerm(fnhatCopy*v);

  residual->addTerm(minusOne*uhatCopy*tau->dot_normal());
  residual->addTerm((sigma1Copy*e1 + sigma2Copy*e2)*tau);
  residual->addTerm(uCopy*tau->div());
  */

  IPPtr sobolevIP = Teuchos::rcp(new IP);
  sobolevIP->addTerm(v);
  //  sobolevIP->addTerm(v->grad());
  sobolevIP->addTerm(tau);
  sobolevIP->addTerm(tau->div());
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, sobolevIP, residual));
  riesz->computeRieszRep();
  Teuchos::RCP<RieszRep> rieszIBP = Teuchos::rcp(new RieszRep(mesh, sobolevIP, residualIBP));
  rieszIBP->computeRieszRep();

  FunctionPtr rieszRepFxn = Teuchos::rcp(new RepFunction(v->ID(),riesz));
  FunctionPtr rieszRepIBPFxn = Teuchos::rcp(new RepFunction(v->ID(),rieszIBP));

  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPts = basisCache->getPhysicalCubaturePoints().dimension(1);
  FieldContainer<double> valOriginal( numCells, numPts);
  FieldContainer<double> valIBP( numCells, numPts);
  rieszRepFxn->values(valOriginal,basisCache);
  rieszRepIBPFxn->values(valIBP,basisCache);

  double maxDiff = 0.0;
  double tol = 1e-15;
  for (int i = 0;i<numCells;i++){
    for (int j = 0;j<numPts;j++){
      maxDiff = max(abs(valOriginal(i,j)-valIBP(i,j)),maxDiff);
    }
  }
  cout << "max diff = " << maxDiff << endl;

  if (rank==0){
    rieszRepFxn->writeValuesToMATLABFile(mesh, "rieszRep.m");
    rieszRepIBPFxn->writeValuesToMATLABFile(mesh, "rieszRepIBP.m");
    solution->writeFluxesToFile(uhat->ID(), "uhatCond.dat");
    solution->writeFluxesToFile(beta_n_u_minus_sigma_n->ID(), "fhatCond.dat");
    solution->writeToVTK("solnCond.vtu",min(H1Order+1,4));
    
    cout << "wrote files: rates.vtu, uhat.dat\n";
  }
  // =========================================

  return 0;
}


