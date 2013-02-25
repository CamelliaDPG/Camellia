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

bool enforceLocalConservation = false;

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


// inflow values for u
class InflowIndicator : public SimpleFunction {
public:
  double value(double x, double y){
    double tol = 1e-12;
    double val = 0.0;    
    if ( abs(x)<tol){
      val = 1.0;
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

  double eps = 1e-3;
  if ( argc > 1) {
    eps = atof(argv[1]);    
    if (rank==0){
      cout << "eps = " << eps << endl;
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
  qoptIP->addTerm( tau / eps + v->grad() );
  qoptIP->addTerm( beta * v->grad() - tau->div() );
  FunctionPtr inflowIndicator = Teuchos::rcp(new InflowIndicator);
  qoptIP->addBoundaryTerm(inflowIndicator*tau->dot_normal());
  //  qoptIP->addBoundaryTerm(beta*tau);
  
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
  vector<double> e1,e2;
  e1.push_back(1.0);  e1.push_back(0.0);
  e2.push_back(0.0);  e2.push_back(1.0);
  FunctionPtr sig_exact = e1*sig1_exact + e2*sig2_exact;
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
 
  bc->addDirichlet(uhat, outflowBoundary, zero); // wall BC - constant throughout

  bool useRobustBC = true;
  if (useRobustBC){
    bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u_exact - sig_exact*n);  
  }else{
    bc->addDirichlet(uhat, inflowBoundary, u_exact);  
  }


  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  if (argc > 2){
    order = atoi(argv[2]);
    if (rank==0){
      cout << "order = " << order << endl;
    }
  }
  int H1Order = order+1;
  int pToAdd = 3;
  if (argc > 3){
    pToAdd = atoi(argv[3]);
    if (rank==0){
      cout << "pToAdd = " << pToAdd << endl;
    }
  }

  Teuchos::RCP<Mesh> mesh;
  int numUniformRefs = 0;
  int nCells = 4;
  if (argc > 4){
    nCells = atoi(argv[4]);
    if (rank==0){
      cout << "numCells = " << nCells << endl;
    }       
    mesh = MeshUtilities::buildUnitQuadMesh(nCells, confusionBF, H1Order, H1Order+pToAdd);

  }else{ // do Schwab/Suri 2-element mesh

    double c = 1.0;
    double width = c*eps*(H1Order+pToAdd);

    // schwab/suri mesh
    bool expectCrossBoundaryLayers = true;
    vector<FieldContainer<double> > vertices;
    vector< vector<int> > elementVertices;

    if (expectCrossBoundaryLayers){
      FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2), I(2), J(2), K(2), L(2);
      A(0) = 0.0; A(1) = 0.0;
      B(0) = width; B(1) = 0.0;
      C(0) = 1.0; C(1) = 0.0;

      D(0) = 0.0; D(1) = width;
      E(0) = width; E(1) = width;
      F(0) = 1.0; F(1) = width;

      G(0) = 0.0; G(1) = 1.0-width;
      H(0) = width; H(1) = 1.0-width;
      I(0) = 1.0; I(1) = 1.0-width;

      J(0) = 0.0; J(1) = 1.0;
      K(0) = width; K(1) = 1.0;
      L(0) = 1.0; L(1) = 1.0;

      vertices.push_back(A); int A_index = 0;
      vertices.push_back(B); int B_index = 1;
      vertices.push_back(C); int C_index = 2;
      vertices.push_back(D); int D_index = 3;
      vertices.push_back(E); int E_index = 4;
      vertices.push_back(F); int F_index = 5;
      vertices.push_back(G); int G_index = 6;
      vertices.push_back(H); int H_index = 7;
      vertices.push_back(I); int I_index = 8;
      vertices.push_back(J); int J_index = 9;
      vertices.push_back(K); int K_index = 10;
      vertices.push_back(L); int L_index = 11;
      vector<int> el1, el2, el3, el4, el5, el6;
      el1.push_back(A_index); el1.push_back(B_index); el1.push_back(E_index); el1.push_back(D_index);
      el2.push_back(B_index); el2.push_back(C_index); el2.push_back(F_index); el2.push_back(E_index);
      el3.push_back(D_index); el3.push_back(E_index); el3.push_back(H_index); el3.push_back(G_index);
      el4.push_back(E_index); el4.push_back(F_index); el4.push_back(I_index); el4.push_back(H_index);
      el5.push_back(G_index); el5.push_back(H_index); el5.push_back(K_index); el5.push_back(J_index);
      el6.push_back(H_index); el6.push_back(I_index); el6.push_back(L_index); el6.push_back(K_index);

      elementVertices.push_back(el1);
      elementVertices.push_back(el2);
      elementVertices.push_back(el3);
      elementVertices.push_back(el4);
      elementVertices.push_back(el5);
      elementVertices.push_back(el6);   
    }else{    
      FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2);
      A(0) = 0.0; A(1) = 0.0;
      B(0) = width; B(1) = 0.0;
      C(0) = 1.0; C(1) = 0.0;

      D(0) = 1.0; D(1) = 1.0;
      E(0) = width; E(1) = 1.0;
      F(0) = 0.0; F(1) = 1.0;

      vertices.push_back(A); int A_index = 0;
      vertices.push_back(B); int B_index = 1;
      vertices.push_back(C); int C_index = 2;
      vertices.push_back(D); int D_index = 3;
      vertices.push_back(E); int E_index = 4;
      vertices.push_back(F); int F_index = 5;
      vector<int> el1, el2, el3, el4, el5;
      // left thin element:
      el1.push_back(A_index); el1.push_back(B_index); el1.push_back(E_index); el1.push_back(F_index);
      // right element
      el2.push_back(B_index); el2.push_back(C_index); el2.push_back(D_index); el2.push_back(E_index);

      elementVertices.push_back(el1);
      elementVertices.push_back(el2);
    }
    
    mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, confusionBF, H1Order, pToAdd) );          
  }
  
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
  //  solution->setReportTimingResults(true);
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  }
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );

  for (int i = 0;i<numUniformRefs;i++){
    refinementStrategy.hRefineUniformly(mesh); 
  }  
   
  FunctionPtr u_diff;
  double L2_error;

  ofstream convOut;
  stringstream convOutFile;
  convOutFile << "erickson_qopt_" << -floor(log(eps)/log(10)) << "_" << order << "_" << pToAdd <<".txt";
  convOut.open(convOutFile.str().c_str());
  solution->condensedSolve(false);
  //  solution->solve(false);

  FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  FunctionPtr sigma1_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
  FunctionPtr sigma2_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );

  FunctionPtr uNorm = u_soln*u_soln;
  FunctionPtr sigmaNorm = sigma1_soln*sigma1_soln + sigma2_soln*sigma2_soln;
  double u_norm_sq = uNorm->integrate(mesh);
  double sigma_norm_sq = sigmaNorm->integrate(mesh);

  u_diff = (u_soln - u_exact)*(u_soln - u_exact);
  FunctionPtr sig1_diff = (sigma1_soln - sig1_exact)*(sigma1_soln - sig1_exact);
  FunctionPtr sig2_diff = (sigma2_soln - sig2_exact)*(sigma2_soln - sig2_exact);
  double u_L2_error = u_diff->integrate(mesh);
  double sigma_L2_error = sig1_diff->integrate(mesh) + sig2_diff->integrate(mesh);
  L2_error = sqrt(u_L2_error + sigma_L2_error);
  double energy_error = solution->energyErrorTotal();
  u_soln->writeValuesToMATLABFile(mesh, "u_soln.m");
  u_diff->writeValuesToMATLABFile(mesh, "u_diff.m");
  u_exact->writeValuesToMATLABFile(mesh, "u_exact.m");
  FunctionPtr total_err = u_diff + sig1_diff + sig2_diff;
  total_err->writeValuesToMATLABFile(mesh, "totalError.m");

  convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << " ";
  if (rank==0){
    //      cout << "L2 error = " << L2_error << ", energy error = " << energy_error << endl;
    cout << "L2 error = " << L2_error << ", energy error = " << energy_error << ", ratio of L2/energy = " << L2_error/energy_error << endl;      

    //    cout << "relative u L2 error = " << sqrt(u_L2_error)/sqrt(u_norm_sq) << ", relative sigma squared l2 error = " << sqrt(sigma_L2_error)/sqrt(sigma_norm_sq) << ", num dofs = " << mesh->numGlobalDofs() << endl;
  }

  /*
    ofstream convOut;
    stringstream convOutFile;
  convOutFile << "erickson_conv_" << eps <<".txt";
  convOut.open(convOutFile.str().c_str());
  FunctionPtr u_diff;
  double L2_error;
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    solution->condensedSolve(false);

    FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
    FunctionPtr sigma1_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
    FunctionPtr sigma2_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );

    FunctionPtr uNorm = u_soln*u_soln;
    FunctionPtr sigmaNorm = sigma1_soln*sigma1_soln + sigma2_soln*sigma2_soln;
    double u_norm_sq = uNorm->integrate(mesh);
    double sigma_norm_sq = sigmaNorm->integrate(mesh);

    u_diff = (u_soln - u_exact)*(u_soln - u_exact);
    FunctionPtr sig1_diff = (sigma1_soln - sig1_exact)*(sigma1_soln - sig1_exact);
    FunctionPtr sig2_diff = (sigma2_soln - sig2_exact)*(sigma2_soln - sig2_exact);
    double u_L2_error = u_diff->integrate(mesh);
    double sigma_L2_error = sig1_diff->integrate(mesh) + sig2_diff->integrate(mesh);
    L2_error = sqrt(u_L2_error + sigma_L2_error);
    double energy_error = solution->energyErrorTotal();
    u_soln->writeValuesToMATLABFile(mesh, "u_soln.m");
    u_diff->writeValuesToMATLABFile(mesh, "u_diff.m");
    u_exact->writeValuesToMATLABFile(mesh, "u_exact.m");
    FunctionPtr total_err = u_diff + sig1_diff + sig2_diff;
    total_err->writeValuesToMATLABFile(mesh, "totalError.m");

    convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << endl;
    if (rank==0){
      //      cout << "L2 error = " << L2_error << ", energy error = " << energy_error << endl;
      cout << "Ratio of L2/energy = " << L2_error/energy_error << ", ratio of u/energy error = " << sqrt(u_L2_error)/energy_error << ", and ratio of sigma/energy error = " << sqrt(sigma_L2_error)/energy_error << endl;      
      cout << "relative u L2 error = " << sqrt(u_L2_error)/sqrt(u_norm_sq) << ", relative sigma squared l2 error = " << sqrt(sigma_L2_error)/sqrt(sigma_norm_sq) << ", num dofs = " << mesh->numGlobalDofs() << endl;
    }
    if (refIndex<numRefs){
      refinementStrategy.refine(); // print to console on rank 0
    }
  }
  //  // one more solve on the final refined mesh:

  convOut.close();  
  */

  ////////////////////   CREATE SOLUTION PROJECTION   ///////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr projectedSolution = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()] = u_exact;
  functionMap[sigma1->ID()] = sig1_exact;
  functionMap[sigma2->ID()] = sig2_exact;

  // everything else = 0; previous stresses sigma_ij = 0 as well
  projectedSolution->projectOntoMesh(functionMap);
  FunctionPtr u_project = Teuchos::rcp( new PreviousSolutionFunction(projectedSolution, u) );
  FunctionPtr sig1_project = Teuchos::rcp( new PreviousSolutionFunction(projectedSolution, sigma1) );
  FunctionPtr sig2_project = Teuchos::rcp( new PreviousSolutionFunction(projectedSolution, sigma2) );
  
  FunctionPtr u_proj_err = (u_project-u_exact)*(u_project-u_exact);
  FunctionPtr s1_proj_err = (sig1_project-sig1_exact)*(sig1_project-sig1_exact);
  FunctionPtr s2_proj_err = (sig2_project-sig2_exact)*(sig2_project-sig2_exact);
  double u_proj_error = u_proj_err->integrate(mesh); 
  double s1_proj_error = s1_proj_err->integrate(mesh); 
  double s2_proj_error = s2_proj_err->integrate(mesh); 
  double proj_error = sqrt(u_proj_error + s1_proj_error + s2_proj_error);
  if (rank==0){
    cout << "DPG L2 error = " << L2_error << ", projection error = " << proj_error << ", ratio = " << L2_error/proj_error << endl;
    convOut << proj_error << endl;
  }
  convOut.close();
  /////////////////////////////////////////////////////////////////////////////

  if (rank==0){
    u_diff->writeValuesToMATLABFile(mesh,"u_error.m");   
    u_proj_err->writeValuesToMATLABFile(mesh,"u_proj_error.m");   

    solution->writeFluxesToFile(uhat->ID(), "uhatQopt.dat");
    solution->writeFluxesToFile(beta_n_u_minus_sigma_n->ID(), "fhatQopt.dat");
    solution->writeToVTK("qopt.vtu",min(H1Order+1,4));
    
    cout << "wrote files: rates.vtu, uhat.dat\n";
  }


  return 0;

}
