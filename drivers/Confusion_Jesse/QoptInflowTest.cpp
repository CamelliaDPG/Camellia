#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

bool enforceLocalConservation = false;

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

class EnergyErrorFunction : public Function {
  map<int, double> _energyErrorForCell;
public:
  EnergyErrorFunction(map<int, double> energyErrorForCell) : Function(0) {
    _energyErrorForCell = energyErrorForCell;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    for (int i = 0;i<cellIDs.size();i++){
      double energyError = _energyErrorForCell[cellIDs[i]];
      for (int j = 0;j<numPoints;j++){
	values(i,j) = energyError;
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

// inflow values for u
class WeightFunction : public SimpleFunction {
private:
  double _cut;
public:
  WeightFunction(double cut){
    _cut = cut;
  }
  double value(double x, double y){
    double val = 0.0;
    if ( x > _cut ){
      val = x-_cut;
    }
    return val;
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

  int numRefs = 0;
  if ( argc > 1) {
    numRefs = atoi(argv[1]);
    if (rank==0){
      cout << "numRefs = " << numRefs << endl;
    }
  }
  
  double eps = 1e-3;
  if ( argc > 2) {
    eps = atof(argv[2]);    
    if (rank==0){
      cout << "eps = " << eps << endl;
    }
  }
  int numPreRefs = 0;
  if (argc > 3){
    numPreRefs = atoi(argv[3]);
    if (rank==0){
      cout << "numPreRefs = " << numPreRefs << endl;
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

  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);  
  qoptIP->addTerm( v );
  FunctionPtr X =  Teuchos::rcp(new Xn(1));
  //  FunctionPtr weight = Function::constant(eps) + X;
  double cut = pow(.5,numPreRefs);
  FunctionPtr weight = Function::constant(eps) + Teuchos::rcp(new WeightFunction(cut));
  qoptIP->addTerm( weight*(tau / eps + v->grad()) );
  qoptIP->addTerm( beta * v->grad() - tau->div() );

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(eps) ); 

  robIP->addTerm( ip_scaling * v);
  robIP->addTerm( sqrt(eps) * v->grad() );
  robIP->addTerm( beta * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( ip_scaling/sqrt(eps) * tau );
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  //  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary(beta) );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary);

  FunctionPtr u_exact = Teuchos::rcp( new Uex(eps,0) );
  FunctionPtr sig1_exact = Teuchos::rcp( new Uex(eps,1) );
  FunctionPtr sig2_exact = Teuchos::rcp( new Uex(eps,2) );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  
  bc->addDirichlet(uhat, outflowBoundary, zero); // wall BC - constant throughout

  bool useRobustBC = false;
  if (useRobustBC){
    bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u_exact);  
  }else{
    bc->addDirichlet(uhat, inflowBoundary, u_exact);  
  }


  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 2, pToAdd = 2;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  

  int nCells = 2;
  if (argc > 4){
    nCells = atoi(argv[4]);
    if (rank==0){
      cout << "numCells = " << nCells << endl;
    }
  }
  int horizontalCells = nCells, verticalCells = nCells;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);
    
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  //  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
  //  solution->setReportTimingResults(true);
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
   
  if (rank==0){
    cout << "Number of pre-refinements = " << numPreRefs << endl;
  }
  for (int i =0;i<=numPreRefs;i++){   
    vector<ElementPtr> elems = mesh->activeElements();
    vector<ElementPtr>::iterator elemIt;
    vector<int> wallCells;    
    for (elemIt=elems.begin();elemIt != elems.end();elemIt++){
      int cellID = (*elemIt)->cellID();
      int numSides = mesh->getElement(cellID)->numSides();
      FieldContainer<double> vertices(numSides,2); //for quads

      mesh->verticesForCell(vertices, cellID);
      bool cellIDset = false;	
      for (int j = 0;j<numSides;j++){ 	
	if (abs(vertices(j,0))<1e-7 && !cellIDset){ // if at the inflow x =0
	  wallCells.push_back(cellID);
	  cellIDset = true;
	}
      }
    }
    if (i<numPreRefs){
      refinementStrategy.refineCells(wallCells);
    }
  }

  ofstream convOut;
  stringstream convOutFile;
  convOutFile << "erickson_conv_" << eps <<".txt";
  convOut.open(convOutFile.str().c_str());
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    solution->condensedSolve(false);

    FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
    FunctionPtr sigma1_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
    FunctionPtr sigma2_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );

    FunctionPtr uNorm = u_soln*u_soln;
    FunctionPtr sigmaNorm = sigma1_soln*sigma1_soln + sigma2_soln*sigma2_soln;
    double u_norm_sq = uNorm->integrate(mesh);
    double sigma_norm_sq = sigmaNorm->integrate(mesh);

    FunctionPtr u_diff = (u_soln - u_exact)*(u_soln - u_exact);
    FunctionPtr sig1_diff = (sigma1_soln - sig1_exact)*(sigma1_soln - sig1_exact);
    FunctionPtr sig2_diff = (sigma2_soln - sig2_exact)*(sigma2_soln - sig2_exact);
    double u_L2_error = u_diff->integrate(mesh);
    double sigma_L2_error = sig1_diff->integrate(mesh) + sig2_diff->integrate(mesh);
    double L2_error = sqrt(u_L2_error + sigma_L2_error);
    double energy_error = solution->energyErrorTotal();
    u_soln->writeValuesToMATLABFile(mesh, "u_soln.m");
    u_diff->writeValuesToMATLABFile(mesh, "u_diff.m");
    u_exact->writeValuesToMATLABFile(mesh, "u_exact.m");
    FunctionPtr total_err = u_diff + sig1_diff + sig2_diff;
    total_err->writeValuesToMATLABFile(mesh, "totalError.m");

    convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << endl;
    if (rank==0){
      cout << "L2 error = " << L2_error << ", energy error = " << energy_error << ", ratio of L2/energy = " << L2_error/energy_error << endl;
      cout << "relative u L2 error = " << sqrt(u_L2_error)/sqrt(u_norm_sq) << ", relative sigma squared l2 error = " << sqrt(sigma_L2_error)/sqrt(sigma_norm_sq) << ", num dofs = " << mesh->numGlobalDofs() << endl;
    }
    if (refIndex<numRefs){
      refinementStrategy.refine(); // print to console on rank 0
    }
  }
  //  // one more solve on the final refined mesh:

  convOut.close();  

  if (rank==0){
    solution->writeFluxesToFile(uhat->ID(), "uhatQopt.dat");
    solution->writeFluxesToFile(beta_n_u_minus_sigma_n->ID(), "fhatQopt.dat");
    solution->writeToVTK("qopt.vtu",min(H1Order+1,4));
    
    cout << "wrote files: rates.vtu, uhat.dat\n";
  }

  return 0;

}
