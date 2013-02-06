#include "Solver.h"
#include "Amesos.h"
#include "Amesos_Utils.h"

#include "SolutionExporter.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "RieszRep.h"
#include "BasisFactory.h" // for test
#include "HessianFilter.h"

#include "MeshUtilities.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

#include "Epetra_LinearProblem.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "StandardAssembler.h" // for system assembly
#include "SerialDenseWrapper.h" // for system assembly
#include "TestingUtilities.h" 

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

class WallInflow : public SpatialFilter{
public: 
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = abs(x)<tol;
    return xMatch;
  }
};

class WallSquareBoundary : public SpatialFilter{
public: 
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (x>.5);
    bool yMatch = abs(y)<tol;
    return xMatch && yMatch;
  }
};

class NonWallSquareBoundary : public SpatialFilter{
public: 
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (x<.5);
    bool yMatch = abs(y)<tol || abs(y-1.0)<tol;
    return xMatch && yMatch;
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
	  if (n==1){
	    Cn = 1.0; // first term only
	  }    
	  /*
	  // discontinuous hat 
	  Cn = -1 + cos(n*pi/2)+.5*n*pi*sin(n*pi/2) + sin(n*pi/4)*(n*pi*cos(n*pi/4)-2*sin(3*n*pi/4));
	  Cn /= (n*pi);
	  Cn /= (n*pi);    
	  */
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
  
  int numRefs = 0;
  if ( argc > 3) {
    numRefs = atoi(argv[3]);
    if (rank==0){
      cout << "numRefs = " << numRefs << endl;
    }
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
  FunctionPtr C_h = Teuchos::rcp( new EpsilonScaling(eps) ); 
  /*  
  robIP->addTerm(v->grad());
  robIP->addTerm(v);
  robIP->addTerm( beta * v->grad() );
  robIP->addTerm(tau);
  robIP->addTerm(tau->div());
  */
  //  robIP->addTerm( C_h * v);
  robIP->addTerm( v);
  robIP->addTerm( sqrt(eps) * v->grad() );
  robIP->addTerm( beta * v->grad() );
  robIP->addTerm( tau->div() );
  //  robIP->addTerm( C_h/sqrt(eps) * tau );
  robIP->addTerm( 1.0/sqrt(eps) * tau );
  ////////////////////   SPECIFY RHS   ///////////////////////

  FunctionPtr zero = Function::constant(0.0);
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  //  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary);
  SpatialFilterPtr wallInflow = Teuchos::rcp( new WallInflow);
  SpatialFilterPtr wallBoundary = Teuchos::rcp( new WallSquareBoundary);
  SpatialFilterPtr nonWall = Teuchos::rcp( new NonWallSquareBoundary);

  FunctionPtr u_exact = Teuchos::rcp( new Uex(eps,0) );
  FunctionPtr sig1_exact = Teuchos::rcp( new Uex(eps,1) );
  FunctionPtr sig2_exact = Teuchos::rcp( new Uex(eps,2) );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  vector<double> e1,e2;
  e1.push_back(1.0);e1.push_back(0.0);
  e2.push_back(0.0);e2.push_back(1.0);
  //  bc->addDirichlet(uhat, outflowBoundary, zero);
  //  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u_exact - (sig1_exact*e1+sig2_exact*e2)*n);  
  bc->addDirichlet(uhat, wallBoundary, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, wallInflow, beta*n*one);
  bc->addDirichlet(beta_n_u_minus_sigma_n, nonWall, zero);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  int H1Order = order+1; int pToAdd = 2;
  
  FieldContainer<double> quadPoints(4,2);
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);
  //  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
   
  BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, mesh->activeElements()[0]->cellID());// should be same basisCache for all
  
  vector<int> cellIDs;
  vector< ElementPtr > allElems = mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){
    cellIDs.push_back((*elemIt)->cellID());
  }
  
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  solution->solve(false);
  double energyErr = solution->energyErrorTotal();
  if (rank==0){
    cout << "solved..." << endl;
  }

  /*
  Teuchos::RCP<StandardAssembler> assembler = Teuchos::rcp(new StandardAssembler(solution));
  Epetra_FECrsMatrix K = assembler->initializeMatrix();
  Epetra_FEVector b = assembler->initializeVector();
  Epetra_FEVector x = assembler->initializeVector();
  assembler->assembleProblem(K,b);  
  Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp(new Epetra_LinearProblem(&K,&x,&b));
  Teuchos::RCP<Solver> solver = Teuchos::rcp(new KluSolver());
  solver->setProblem(problem);
  solver->solve();
  assembler->distributeDofs(x);  
  */

  
  //  LinearTermPtr residual = rhs->linearTerm(); LinearTermPtr bfLT = -confusionBF->testFunctional(solution); residual->addTerm(bfLT);  
  LinearTermPtr residual = -confusionBF->testFunctional(solution);  LinearTermPtr rhsLT = rhs->linearTerm();  residual->addTerm(rhsLT);  
  Teuchos::RCP<RieszRep> rieszResidual = Teuchos::rcp(new RieszRep(mesh, robIP, residual));
  rieszResidual->computeRieszRep();
  FunctionPtr e_v = Teuchos::rcp(new RepFunction(v,rieszResidual));
  FunctionPtr e_tau = Teuchos::rcp(new RepFunction(tau,rieszResidual));
  FunctionPtr xErr = eps*e_v->dx()*e_v->dx() + (C_h*C_h/eps)*(e1*e_tau)*(e1*e_tau);
  FunctionPtr yErr = eps*e_v->dy()*e_v->dy() + (C_h*C_h/eps)*(e2*e_tau)*(e2*e_tau);    
  //  FunctionPtr restOfError = (C_h*C_h)*(e_v)*(e_v) + (beta*e_v->grad())*(beta*e_v->grad()) + (e_tau->div())*(e_tau->div()) + (C_h*C_h/eps)*(e_tau)*(e_tau) + eps*(e_v->grad())*(e_v->grad());  
  /*
  map<int,FunctionPtr > errFxns;
  errFxns[v->ID()] = e_v;
  errFxns[tau->ID()] = e_tau;
  FunctionPtr restOfError =  robIP->evaluate(errFxns,false)->evaluate(errFxns,false); 
  */
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  for (int refIndex=0;refIndex<numRefs;refIndex++){
    if (rank==0){
      cout << "on ref index " << refIndex << endl;
    }
    vector<int> cellsToRefine,xCells,yCells,regCells;
    refinementStrategy.getCellsAboveErrorThreshhold(cellsToRefine);
    rieszResidual->computeRieszRep(); // in preparation  
    for (vector<int>::iterator cellIt = cellsToRefine.begin();cellIt!=cellsToRefine.end();cellIt++){
      int cellID = *cellIt;
      vector<double> c = mesh->getCellCentroid(cellID);
      FieldContainer<double> verts(4,2);
      mesh->verticesForCell(verts, cellID);
      bool notSingularity=true;
      for (int i = 0;i<4;i++){
	if ((abs(verts(i,0)-.5)<1e-8)&&(abs(verts(i,1))<1e-8)){
	  notSingularity = false;
	}
      }
      bool onWall = c[0]>.5;
      double yError = yErr->integrate(cellID,mesh,10,true);
      double xError = xErr->integrate(cellID,mesh,10,true);
      double thresh = 25.0;
      bool anisotropicFlag = yError/xError > thresh;
      bool doAnisotropy = notSingularity && onWall && anisotropicFlag;
      //      doAnisotropy = false;
      if (doAnisotropy){ // if ratio is small = y err bigger than xErr
	yCells.push_back(cellID);
      }else if (c[0]>.25){
	regCells.push_back(cellID);
      }
    }
    mesh->hRefine(yCells, RefinementPattern::yAnisotropicRefinementPatternQuad());    
    mesh->hRefine(regCells, RefinementPattern::regularRefinementPatternQuad());        

    solution->solve(false);
  }
  double energyErrFinal = solution->energyErrorTotal();
  if (rank==0){
    cout << "num elements = " << mesh->numActiveElements() << endl;
    cout << "num dofs = " << mesh->numGlobalDofs() << endl;
    cout << "energy err " << energyErrFinal << endl;
  }
 
  ////////////////////   get residual   ///////////////////////

  VTKExporter exporter(solution, mesh, varFactory);
  if (rank==0){
    exporter.exportSolution("robustIP");
    cout << endl;
  }
 
  return 0;
}


