#include "SolutionExporter.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"
#include "MeshUtilities.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double pi = 2.0*acos(0.0);

class InvSqrtHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return sqrt(1.0/h);
  }
};
class InvHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return 1.0/h;
  }
};
class HScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return h;
  }
};
class SqrtHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return sqrt(h);
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

class ZeroMeanScaling : public hFunction {
  public:
  double value(double x, double y, double h) {
    return 1.0/(h*h);
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

class TopBottomSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = (abs(y)<tol) || (abs(y-1.0)<tol);
    return yMatch;
  }
};

class Udisc : public Function {
public:
  Udisc() : Function(0) {
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {

    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
	if (y>.5){
	  values(cellIndex,ptIndex) = (y-1.0)*(y-1.0);
	}else{
	  values(cellIndex,ptIndex) = -y*y;
	}
      }
    }
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
	bool useDiscontinuous = false; // use discontinuous soln
	int numTerms = 20;
	if (!useDiscontinuous)
	  numTerms = 1;
	for (int n = 1;n<numTerms+1;n++){

	  double lambda = n*n*pi*pi*_eps;
	  double d = sqrt(1.0+4.0*_eps*lambda);
	  double r1 = (1.0+d)/(2.0*_eps);
	  double r2 = (1.0-d)/(2.0*_eps);
    
	  double Cn = 0.0;            
	  if (!useDiscontinuous){
	    if (n==1){
	      Cn = 1.0; // first term only
	    } 	  
	  }else{
	    // discontinuous hat 
	    Cn = -1 + cos(n*pi/2)+.5*n*pi*sin(n*pi/2) + sin(n*pi/4)*(n*pi*cos(n*pi/4)-2*sin(3*n*pi/4));
	    Cn /= (n*pi);
	    Cn /= (n*pi);    
	  }
	  

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

class HeavisideFunction : public SimpleFunction{
  double value(double x, double y){
    double value = 0.0;
    if (x>.5){
      value = 1.0;
    }
    return value;
  }
};

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif

  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  
  int nCells = args.Input<int>("--nCells", "num cells",2);  
  int numRefs = args.Input<int>("--numRefs","num adaptive refinements",0);
  double eps = args.Input<double>("--epsilon","diffusion parameter",1e-2);
  double energyThreshold = args.Input<double>("--energyThreshold","adaptivity energy thresh",.2); 
  bool useGraphNorm = args.Input<bool>("--useGraphNorm","graph norm flag",false);
  bool scaleH = args.Input<bool>("--scaleH","hScale flag",false); 

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
  FunctionPtr invH = Teuchos::rcp(new InvHScaling);
  FunctionPtr h = Teuchos::rcp(new HScaling);
  if (scaleH){
    qoptIP->addTerm( invH*v );
  }else{
    qoptIP->addTerm( v );
  }
  qoptIP->addTerm( tau / eps + v->grad() );
  qoptIP->addTerm( beta * v->grad() - tau->div() );

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(eps) ); 
  if (scaleH){
    robIP->addTerm( invH*v);
    robIP->addTerm( h*beta * v->grad() );
  
  }else{
    robIP->addTerm( v);
    robIP->addTerm( beta * v->grad() );
  }
  robIP->addTerm( sqrt(eps) * v->grad() );  
  robIP->addTerm( ip_scaling/sqrt(eps) * tau );
  robIP->addTerm( tau->div() );  
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr f = Teuchos::rcp( new HeavisideFunction() );
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  rhs->addTerm(f*v);

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary(beta) );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary);
  SpatialFilterPtr topAndBottom = Teuchos::rcp(new TopBottomSquareBoundary);

  FunctionPtr u_exact = Teuchos::rcp( new Uex(eps,0) );
  FunctionPtr sig1_exact = Teuchos::rcp( new Uex(eps,1) );
  FunctionPtr sig2_exact = Teuchos::rcp( new Uex(eps,2) );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;
  FunctionPtr sigma = sig1_exact*e1 + sig2_exact*e2;
  //  bc->addDirichlet(uhat, outflowBoundary, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, topAndBottom, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, Function::constant(1.0));
  //  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u_exact-sigma*n);  
  //  FunctionPtr u_disc = Teuchos::rcp( new Udisc );
  //  bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u_disc);  

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  int H1Order = order+1, pToAdd = 2;
  
  int horizontalCells = nCells, verticalCells = nCells;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);
    
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution; 
  if (useGraphNorm){
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
  }else{
    solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  }
  
  RefinementStrategy refinementStrategy( solution, energyThreshold );
   
  ofstream convOut;
  stringstream convOutFile;
  convOutFile << "erickson_conv_" << round(-log(eps)/log(10.0)) <<".txt";
  convOut.open(convOutFile.str().c_str());
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->condensedSolve(false);
    //    solution->solve(false);

    double quadTol = 1e-7;
    FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
    FunctionPtr sigma1_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
    FunctionPtr sigma2_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );
    FunctionPtr u_diff = (u_soln - u_exact)*(u_soln - u_exact);
    FunctionPtr sig1_diff = (sigma1_soln - sig1_exact)*(sigma1_soln - sig1_exact);
    FunctionPtr sig2_diff = (sigma2_soln - sig2_exact)*(sigma2_soln - sig2_exact);
    //    double u_L2_error = u_diff->integrate(mesh,quadTol);
    //    double sigma_L2_error = sig1_diff->integrate(mesh,quadTol) + sig2_diff->integrate(mesh,quadTol);
    double u_L2_error = u_diff->integrate(mesh,10);
    double sigma_L2_error = sig1_diff->integrate(mesh,10) + sig2_diff->integrate(mesh,10);
    double L2_error = sqrt(u_L2_error + sigma_L2_error);
    double energy_error = solution->energyErrorTotal();
    u_soln->writeValuesToMATLABFile(mesh, "u_soln.m");
    u_diff->writeValuesToMATLABFile(mesh, "u_diff.m");
    u_exact->writeValuesToMATLABFile(mesh, "u_exact.m");
    sig1_exact->writeValuesToMATLABFile(mesh, "s1_exact.m");
    sig2_exact->writeValuesToMATLABFile(mesh, "s2_exact.m");

    convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << endl;
    if (rank==0){
      cout << "L2 error = " << L2_error << ", energy error = " << energy_error << ", ratio = " << L2_error/energy_error << endl;
      //      cout << "u squared L2 error = " << u_L2_error << ", sigma squared l2 error = " << sigma_L2_error << ", num dofs = " << mesh->numGlobalDofs() << endl;
    }

    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  convOut.close();  

  // one more solve on the final refined mesh:
  solution->condensedSolve(false);

  VTKExporter exporter(solution, mesh, varFactory);
  if (rank==0){
    exporter.exportSolution("robustIP");
    cout << endl;
  }

  return 0; 
}
