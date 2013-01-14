#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "PreviousSolutionFunction.h"
#include "MeshUtilities.h"
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

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

// restriction to inflow boundary
class InflowRestriction : public SimpleFunction {
public:
  bool boundaryValueOnly(){
    return true;
  }
  double value(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol); // left inflow
    bool yMatch = ((abs(y)<tol) || (abs(y-1.0)<tol)); // top/bottom
    double val = 0.0;
    if (xMatch || yMatch){
      val = 1.0;
    }
    return val;
  }
};

// restriction to inflow boundary
class ElemInflowRestriction : public Function {
  vector<double> _beta;
public:
  // assumes const beta for now
  ElemInflowRestriction(vector<double> beta):Function(0){
    _beta = beta;
  }
  bool boundaryValueOnly(){
    return true;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    const FieldContainer<double> *normals = &(basisCache->getSideNormals());
    double tol=1e-14;
    values.initialize(0.0);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	double x = (*points)(cellIndex,ptIndex,0);
	double y = (*points)(cellIndex,ptIndex,1);
	double nx = (*normals)(cellIndex,ptIndex,0);
	double ny = (*normals)(cellIndex,ptIndex,1);
	double beta_n = _beta[0]*nx + _beta[1]*ny;
	if (beta_n < 0){
	  values(cellIndex,ptIndex) = 1.0;
	}	
      }
    }
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
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  int numRefs = args.Input<int>("--numRefs", "num refinements",0);
  
  double eps = args.Input<double>("--epsilon","diffusion parameter",1e-2);
 
  int numPreRefs = args.Input<int>("--numPreRefs", "num initial refinements of mesh",0);

  int nCells = args.Input<int>("--nCells","num cells in initial mesh",2);

  bool useRobustBC = args.Input<bool>("--useRobustBC", "bool flag for BC", false);
  
  int cubEnrich = args.Input<int>("--cubEnrich", "cubature enrichment degree", 0);

  int order = args.Input<int>("--order", "L^2 polynomial order of basis",3);
  int H1Order = order+1;

  int pToAdd = 3;

  args.Process();
  if (rank==0){
    cout << "Epsilon = " << eps << endl;
    cout << "Number of refinements = " << numRefs << endl;
    cout << "Number of pre-refinements = " << numPreRefs << endl;
    cout << "Number of cells in initial mesh = " << nCells << endl;
    cout << "Use of robust BC = " << useRobustBC << endl;
    cout << "Cubature enrichment degree = " << cubEnrich << endl;
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
  //  qoptIP->addTerm( pow(eps,3/2)*tau ); // from Antti's paper
  qoptIP->addTerm( tau / eps + v->grad() );
  qoptIP->addTerm( beta * v->grad() - tau->div() );
  
  FunctionPtr inflowRestriction = Function::constant(0.0);
  //  inflowRestriction = Teuchos::rcp(new ElemInflowRestriction(beta));
  inflowRestriction = Teuchos::rcp(new InflowRestriction);
  //  qoptIP->addBoundaryTerm( inflowRestriction*(1.0/eps)*tau->dot_normal());
    
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

  if (useRobustBC){
    bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*u_exact);  
  }else{
    bc->addDirichlet(uhat, inflowBoundary, u_exact);  
  }

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells,confusionBF, H1Order, H1Order+pToAdd);
  FunctionPtr usq = u_exact*u_exact;
  double u_int = usq->integrate(mesh,cubEnrich); 
  double u_int_adapt = usq->integrate(mesh,1e-7);
  cout << "integral of u = " << u_int << ", and adaptively " << u_int_adapt << endl;
  return 0;

  
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
 
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
   
  FunctionPtr u_diff;
  double L2_error;
  for (int i =0;i<=numPreRefs;i++){   
    vector<ElementPtr> elems = mesh->activeElements();
    vector<int> wallCells;    
    for (vector<ElementPtr>::iterator elemIt=elems.begin();elemIt != elems.end();elemIt++){
      int cellID = (*elemIt)->cellID();
      int numSides = mesh->getElement(cellID)->numSides();
      FieldContainer<double> vertices(numSides,2); 

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
      refinementStrategy.setEnforceOneIrregularity(false);
      refinementStrategy.refineCells(wallCells);
    }
  }
  solution->condensedSolve(false);
  
  ofstream convOut;
  stringstream convOutFile;
  convOutFile << "erickson_conv_" << eps <<".txt";
  convOut.open(convOutFile.str().c_str());

  FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  FunctionPtr sigma1_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
  FunctionPtr sigma2_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );

  FunctionPtr uNorm = u_soln*u_soln;
  FunctionPtr sigmaNorm = sigma1_soln*sigma1_soln + sigma2_soln*sigma2_soln;
  //  double u_norm_sq = uNorm->integrate(mesh,cubEnrich);
  //  double sigma_norm_sq = sigmaNorm->integrate(mesh,cubEnrich);

  u_diff = (u_soln - u_exact)*(u_soln - u_exact);
  FunctionPtr sig1_diff = (sigma1_soln - sig1_exact)*(sigma1_soln - sig1_exact);
  FunctionPtr sig2_diff = (sigma2_soln - sig2_exact)*(sigma2_soln - sig2_exact);
  double u_L2_error = u_diff->integrate(mesh,cubEnrich);
  double sigma_L2_error = sig1_diff->integrate(mesh,cubEnrich) + sig2_diff->integrate(mesh,cubEnrich);
  L2_error = sqrt(u_L2_error + sigma_L2_error);
  double energy_error = solution->energyErrorTotal();
  FunctionPtr total_err = u_diff + sig1_diff + sig2_diff;

  convOut << mesh->numGlobalDofs() << " " << L2_error << " " << energy_error << endl;
  if (rank==0){
    cout << "L2 error = " << L2_error << ", energy error = " << energy_error << endl;
    cout << "Ratio of L2/energy = " << L2_error/energy_error << endl;
    //    cout << ", ratio of u/energy error = " << sqrt(u_L2_error)/energy_error << ", and ratio of sigma/energy error = " << sqrt(sigma_L2_error)/energy_error << endl;      
    //      cout << "u L2 error = " << sqrt(u_L2_error) << ", sigma l2 error = " << sqrt(sigma_L2_error) << ", num dofs = " << mesh->numGlobalDofs() << endl;
  }
  convOut.close();

  /*
////////////////////   CREATE SOLUTION PROJECTION   ///////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL); RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL); IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr projectedSolution = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  projectedSolution->setCubatureEnrichmentDegree(cubEnrich); //for better boundary layer integration
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()] = u_exact; functionMap[sigma1->ID()] = sig1_exact; functionMap[sigma2->ID()] = sig2_exact;

  // everything else = 0; previous stresses sigma_ij = 0 as well
  projectedSolution->projectOntoMesh(functionMap);
  FunctionPtr u_project = Function::solution(u,projectedSolution);
  FunctionPtr sig1_project = Function::solution(sigma1,projectedSolution);
  FunctionPtr sig2_project = Function::solution(sigma2,projectedSolution);
  
  FunctionPtr u_proj_err = (u_project-u_exact)*(u_project-u_exact);
  FunctionPtr s1_proj_err = (sig1_project-sig1_exact)*(sig1_project-sig1_exact);
  FunctionPtr s2_proj_err = (sig2_project-sig2_exact)*(sig2_project-sig2_exact);
  double u_proj_error = u_proj_err->integrate(mesh,cubEnrich); 
  double s1_proj_error = s1_proj_err->integrate(mesh,cubEnrich); 
  double s2_proj_error = s2_proj_err->integrate(mesh,cubEnrich); 
  double proj_error = sqrt(u_proj_error + s1_proj_error + s2_proj_error);
  if (rank==0){
    cout << "u proj error = " << sqrt(u_proj_error) << ", sigma proj error = " << sqrt(s1_proj_error+s2_proj_error) << endl;
    cout << "DPG L2 error = " << L2_error << ", projection error = " << proj_error << ", ratio = " << L2_error/proj_error << endl;
  }
  */

  /////////////////////////////////////////////////////////////////////////////
  
  for (int i =0;i<numRefs;i++){       
    solution->condensedSolve(false);
    
    FunctionPtr u_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
    FunctionPtr sigma1_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma1) );
    FunctionPtr sigma2_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, sigma2) );

    FunctionPtr uNorm = u_soln*u_soln;
    FunctionPtr sigmaNorm = sigma1_soln*sigma1_soln + sigma2_soln*sigma2_soln;

    u_diff = (u_soln - u_exact)*(u_soln - u_exact);
    FunctionPtr sig1_diff = (sigma1_soln - sig1_exact)*(sigma1_soln - sig1_exact);
    FunctionPtr sig2_diff = (sigma2_soln - sig2_exact)*(sigma2_soln - sig2_exact);
    double u_L2_error = u_diff->integrate(mesh,cubEnrich);
    double sigma_L2_error = sig1_diff->integrate(mesh,cubEnrich) + sig2_diff->integrate(mesh,cubEnrich);
    L2_error = sqrt(u_L2_error + sigma_L2_error);
    double energy_error = solution->energyErrorTotal();
    FunctionPtr total_err = u_diff + sig1_diff + sig2_diff;
    //    total_err->writeValuesToMATLABFile(mesh, "totalError.m");
    if (rank==0){
      //      cout << "L2 error = " << L2_error << ", energy error = " << energy_error << endl;
      cout << "Ratio of L2/energy = " << L2_error/energy_error << endl;
      //      cout << "u L2 error = " << sqrt(u_L2_error) << ", sigma l2 error = " << sqrt(sigma_L2_error) << ", num dofs = " << mesh->numGlobalDofs() << endl;
    }

    if (rank==0){
      cout << endl << "DOING REFINEMENTS " << endl;
    }   
    refinementStrategy.refine(rank==0); // print to console on rank 0

    /*
    ////////////////////   CREATE SOLUTION PROJECTION   ///////////////////////

    BCPtr nullBC = Teuchos::rcp((BC*)NULL); RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL); IPPtr nullIP = Teuchos::rcp((IP*)NULL);
    SolutionPtr projectedSolution = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
    projectedSolution->setCubatureEnrichmentDegree(cubEnrich); //for better boundary layer integration
    map<int, Teuchos::RCP<Function> > functionMap;
    functionMap[u->ID()] = u_exact; functionMap[sigma1->ID()] = sig1_exact; functionMap[sigma2->ID()] = sig2_exact;

    // everything else = 0; previous stresses sigma_ij = 0 as well
    projectedSolution->projectOntoMesh(functionMap);
    FunctionPtr u_project = Function::solution(u,projectedSolution);
    FunctionPtr sig1_project = Function::solution(sigma1,projectedSolution);
    FunctionPtr sig2_project = Function::solution(sigma2,projectedSolution);
  
    FunctionPtr u_proj_err = (u_project-u_exact)*(u_project-u_exact);
    FunctionPtr s1_proj_err = (sig1_project-sig1_exact)*(sig1_project-sig1_exact);
    FunctionPtr s2_proj_err = (sig2_project-sig2_exact)*(sig2_project-sig2_exact);
    double u_proj_error = u_proj_err->integrate(mesh,cubEnrich); 
    double s1_proj_error = s1_proj_err->integrate(mesh,cubEnrich); 
    double s2_proj_error = s2_proj_err->integrate(mesh,cubEnrich); 
    double proj_error = sqrt(u_proj_error + s1_proj_error + s2_proj_error);
    if (rank==0){
      cout << "u proj error = " << sqrt(u_proj_error) << ", sigma proj error = " << sqrt(s1_proj_error+s2_proj_error) << endl;
      cout << "DPG L2 error = " << L2_error << ", projection error = " << proj_error << ", ratio = " << L2_error/proj_error << endl;
    }
    */
    /////////////////////////////////////////////////////////////////////////////

  }

  //  solution->condensedSolve(false);  

  VTKExporter exporter(solution, mesh, varFactory);
  if (rank==0){
    //    exporter.exportSolution("qopt");
  }

  return 0;

}
