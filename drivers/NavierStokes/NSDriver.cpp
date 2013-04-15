#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "LagrangeConstraints.h"

#include "RefinementHistory.h"
#include "RefinementPattern.h"
#include "RefinementStrategy.h"

#include "NonlinearStepSize.h"
#include "NonlinearSolveStrategy.h"

// Trilinos includes
#include "Epetra_Time.h"
#include "Intrepid_FieldContainer.hpp"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

#include "InnerProductScratchPad.h"
#include "TestSuite.h"
#include "PenaltyConstraints.h"

#include "ElementType.h"
#include "Element.h"

#include "MeshPolyOrderFunction.h"
#include "SolutionExporter.h"

#include "StandardAssembler.h"
#include "SerialDenseWrapper.h"


typedef map< int, FunctionPtr > sparseFxnVector;    // dim = {trialID}
typedef map< int, sparseFxnVector > sparseFxnMatrix; // dim = {testID, trialID}
typedef map< int, sparseFxnMatrix > sparseFxnTensor; // dim = {spatial dim, testID, trialID}

static const double GAMMA = 1.4;
static const double PRANDTL = 0.72;
static const double YTOP = 1.0;
static const double X_BOUNDARY = 2.0;

using namespace std;

class InvH : public hFunction {
public:
  double value(double x, double y, double h) {
    return 1.0/h;
  }
};

class InvSqrtH : public hFunction {
public:
  double value(double x, double y, double h) {
    return 1.0/sqrt(h);
  }
};
// ===================== Mesh functions ====================

class MeshInfo {
  MeshPtr _mesh;
public:
  MeshInfo(MeshPtr mesh){
    _mesh = mesh;
  }
  double getMinCellMeasure(){
    double minMeasure = 1e7;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
      minMeasure = min(minMeasure, _mesh->getCellMeasure((*elemIt)->cellID()));
    }
    return minMeasure;
  }
  vector<int> getMinCellSizeCellIDs(){
    double minMeasure = getMinCellMeasure();
    vector<int> minMeasureCellIDs;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
      if (minMeasure <= _mesh->getCellMeasure((*elemIt)->cellID())){
	minMeasureCellIDs.push_back((*elemIt)->cellID());
      }
    }
    return minMeasureCellIDs;
  }
  double getMinCellSideLength(){
    double minMeasure = 1e7;
    vector<ElementPtr> elems = _mesh->activeElements();
    for (vector<ElementPtr>::iterator elemIt = elems.begin();elemIt!=elems.end();elemIt++){
      minMeasure = min(minMeasure, _mesh->getCellXSize((*elemIt)->cellID()));
      minMeasure = min(minMeasure, _mesh->getCellYSize((*elemIt)->cellID()));
    }
    return minMeasure;
  }
};

// ===================== Helper functions ====================

class ScalarParamFunction : public Function {
  double _a;
public:
  ScalarParamFunction(double a) : Function(0){
    _a = a;
  }
  void set_param(double a){
    _a = a;
  }
  double get_param(){
    return _a;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    values.initialize(_a);
  }
};

class PowerFunction : public Function {
  FunctionPtr _f;
  double _power;
  double _minVal;
public:
  PowerFunction(FunctionPtr f,double power) : Function(0) {
    _f = f;
    _power = power;
    _minVal = 1e-7;
  }
  PowerFunction(FunctionPtr f,double power,double minVal) : Function(0) {
    _f = f;
    _power = power;
    _minVal = minVal;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    _f->values(values,basisCache);    
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double value = values(cellIndex,ptIndex);
        values(cellIndex,ptIndex) = pow(max(value,_minVal),_power); // WARNING: MAX OPERATOR HACK FOR NEG TEMP
      }
    }
  }
};

class PositiveFunction : public Function {
  FunctionPtr _f;
  double _minVal;
public:
  PositiveFunction(FunctionPtr f) : Function(0) {
    _f = f;
    _minVal = 1e-7;
  }
  PositiveFunction(FunctionPtr f,double minVal) : Function(0) {
    _f = f;
    _minVal = minVal;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    _f->values(values,basisCache);    
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double value = values(cellIndex,ptIndex);
	if (value < _minVal){
	  values(cellIndex,ptIndex) = _minVal;
	}
      }
    }
  }
};

class LogFunction : public Function {
  FunctionPtr _f;
public:
  LogFunction(FunctionPtr f) : Function(0) {
    _f = f;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    _f->values(values,basisCache);    
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double value = values(cellIndex,ptIndex);
        values(cellIndex,ptIndex) = log(value);
      }
    }
  }
};

class NormSqOverElement : public Function {
  FunctionPtr _f;
  Teuchos::RCP<Mesh> _mesh;
public:
  NormSqOverElement(FunctionPtr f, Teuchos::RCP<Mesh> mesh) : Function(0) {
    _f = f;
    _mesh = mesh;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    FunctionPtr fsq = _f*_f;
    int numCells = basisCache->cellIDs().size();
    int numPoints = values.dimension(1);   
    FieldContainer<double> cellIntegs(numCells);
    fsq->integrate(cellIntegs,basisCache);    
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	values(cellIndex,ptIndex) = cellIntegs(cellIndex);
      }
    }
  }
};

class PartitionFunction : public Function {
  Teuchos::RCP<Mesh> _mesh;
public:
  PartitionFunction(  Teuchos::RCP<Mesh> mesh) : Function(0) {
    _mesh = mesh;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    for (int i = 0;i<cellIDs.size();i++){
      int partitionNumber = _mesh->partitionForCellID(cellIDs[i]);
      for (int j = 0;j<numPoints;j++){
	values(i,j) = partitionNumber;
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
    // h = sqrt(|K|), or measure of one side of a quad elem
    double scaling = min(_epsilon/(h), 1.0);
    // sqrt because it's inserted into an IP form in a symmetric fashion
    return sqrt(scaling);
  }
};

class hPowerFunction : public Function {
  double _power;
  int _xyDim;
public: 
  hPowerFunction(double power){
    _power = power;
    _xyDim = 0; // isotropic assumption
  }
  hPowerFunction(double power, int anisotropicDimension){
    _power = power;
    _xyDim = anisotropicDimension;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    MeshPtr mesh = basisCache->mesh();
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    for (int i = 0;i<cellIDs.size();i++){
      double h = mesh->getCellXSize(cellIDs[i]); // default to x-direction
      if (_xyDim == 1){
	h = mesh->getCellYSize(cellIDs[i]);
      }
      double hPower = pow(h,_power);      
      for (int j = 0;j<numPoints;j++){
	values(i,j) = hPower;
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


// ===================== Spatial filter boundary functions ====================

/* not really a spatial filter, but used like one in the penalty method */
class SubsonicIndicator : public Function {
  FunctionPtr _u1;
  FunctionPtr _T;
  double _gamma;
  double _cv;    
  double _tol;
public:
  SubsonicIndicator(FunctionPtr u1hat, FunctionPtr That, double gamma, double cv) : Function(0) {
    _u1 = u1hat;
    _T = That;
    _gamma = gamma;
    _cv = cv;  
    _tol=1e-12;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {  
    FieldContainer<double> points  = basisCache->getPhysicalCubaturePoints();
    FieldContainer<double> normals = basisCache->getSideNormals();
    int numCells = points.dimension(0);
    int numPoints = points.dimension(1);

    FieldContainer<double> Tv(numCells,numPoints);
    FieldContainer<double> u1v(numCells,numPoints);;
    _u1->values(u1v,basisCache);
    _T->values(Tv,basisCache);
    
    bool isSubsonic = false;
    double min_y = YTOP;
    double max_y = 0.0;
    values.initialize(0.0);
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	double x = points(cellIndex,ptIndex,0);
	double y = points(cellIndex,ptIndex,1);
	
	double T = Tv(cellIndex,ptIndex);
	double un = u1v(cellIndex,ptIndex); // WARNING: ASSUMES NORMAL AT OUTFLOW = (1,0)
	double c = sqrt(_gamma * (_gamma-1.0) * _cv * T);

	bool outflowMatch = ((abs(x-2.0) < _tol) && (y > 0.0) && (y < YTOP));
	bool subsonicMatch = (un < c) && (un > 0.0);
	if (subsonicMatch && outflowMatch){
	  values(cellIndex,ptIndex) = 1.0;
	  isSubsonic = true;
	  min_y = min(y,min_y);
	  max_y = max(y,max_y);
	  //	  cout << "y = " << y << endl;
	}
      }
    }
    if (isSubsonic){
      //      cout << "subsonic in interval y =(" << min_y << "," << max_y << ")" << endl;
    }
  }
};

class SubsonicOutflow : public SpatialFilter { 
  FunctionPtr _u1;
  FunctionPtr _T;
  double _gamma;
  double _cv;
public:
  SubsonicOutflow(FunctionPtr u1hat, FunctionPtr That, double gamma, double cv) {
    _u1 = u1hat;
    _T = That;
    _gamma = gamma;
    _cv = cv;
  }

  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) {  
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());    
    int numCells = (*points).dimension(0);
    int numPoints = (*points).dimension(1);

    FieldContainer<double> T(numCells,numPoints);
    FieldContainer<double> u1(numCells,numPoints);;
    _u1->values(u1,basisCache);
    _T->values(T,basisCache);
    
    double tol=1e-14;
    bool somePointMatches = false;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	double x = (*points)(cellIndex,ptIndex,0);
	double y = (*points)(cellIndex,ptIndex,1);

	double T_val = T(cellIndex,ptIndex);
	double c = sqrt(_gamma * (_gamma-1.0) * _cv * T_val);
	double un = u1(cellIndex,ptIndex); // WARNING: ASSUMES NORMAL AT OUTFLOW = (1,0)

	cout << "un = " << un << ", T = " << T_val << endl;

	double tol = 1e-14;
	bool outflowMatch = ((abs(x-2.0) < tol) && (y > 0.0) && (y < YTOP));
	bool subsonicMatch = (un < c) && (un > 0.0);

	pointsMatch(cellIndex,ptIndex) = false;
	if (outflowMatch && subsonicMatch){	  
	  pointsMatch(cellIndex,ptIndex) = true;
	  somePointMatches = true;
	}
      }
    }
    return somePointMatches;   
  }
};


class OutflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = ((abs(x-2.0) < tol) && (y > 0.0) && (y < YTOP));
    return yMatch;
  }
};

class InflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = ((abs(x) < tol) && (y > 0) && (y < YTOP));
    return yMatch;
  }
};

class FreeStreamBoundaryTop : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = (abs(y-YTOP) < tol && (x < 2.0) && (x > 0.0));
    //    bool yMatch = (abs(y-YTOP) < tol); 
    return yMatch;
  }
};

class FreeStreamBoundaryBottom : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = ((abs(y) < tol) && (x < 1.0) && (x > 0.0));
    return yMatch;
  }
};

class WallBoundary : public SpatialFilter {
public:  
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = ((abs(y) < tol) && (x > 1.0) && (x < 2.0));
    return yMatch;
  }  
};

// ===================== IP helper functions ====================

class RadialWeightFunction : public SimpleFunction {
  double _eps;
public:
  RadialWeightFunction(double epsilon){
    _eps = epsilon;
  }
  double value(double x, double y) {
    double xdiff = x-1.0;
    double ydiff = y-0.0;
    return sqrt(_eps + sqrt(xdiff*xdiff + ydiff*ydiff)); // sqrt so it can go into an IP
  }
};

class TwoDGaussian : public SimpleFunction {
  double _width,_amplitude;
public:
  TwoDGaussian(double width,double amplitude){
    _width = width;
    _amplitude = amplitude;
  }
  double value(double x, double y){
    return _amplitude*exp(-((x-1.0)*(x-1.0)+y*y)/(_width*_width));
  }
};

void initLinearTermVector(sparseFxnMatrix A, map<int, LinearTermPtr> &Mvec){
  
  sparseFxnMatrix::iterator testIt;
  for (testIt = A.begin();testIt!=A.end();testIt++){
    int testID = testIt->first;      
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;   
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;	
      Mvec[trialID] = Teuchos::rcp(new LinearTerm);
    }
  }
}


// ===================== main file ====================

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  // define mathematical "constants"
  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  FunctionPtr negOne = Function::constant(-1.0);  
  FunctionPtr one = Function::constant(1.0);

  // problem params
  double Re = args.Input<double>("--Re","Reynolds number",1e3);
  double dt = args.Input<double>("--dt","Timestep",.25);

  // solver
  int nCells = args.Input<int>("--nCells", "num cells",2);  
  int polyOrder = args.Input<int>("--p","order of approximation",2);
  int pToAdd = args.Input<int>("--pToAdd", "test space enrichment",2); 
  double time_tol_orig = args.Input<double>("--timeTol", "time step tolerance",1e-8);
  bool useLineSearch = args.Input<bool>("--useLineSearch", "flag for line search",false); // default to zero
  int maxNRIter = args.Input<int>("--maxNRIter","maximum number of NR iterations",1); // default to one per timestep

  // adaptivity
  int numRefs = args.Input<int>("--numRefs","num adaptive refinements",0);
  double energyThreshold = args.Input<double>("--energyThreshold", "energy thresh for adaptivity",0.25); // for mesh refinements 
  bool useHpStrategy = args.Input<bool>("--useHpStrategy","option to use a 'cheap' hp strategy", false);
  double anisotropicThresh = args.Input<int>("--anisotropicThresh","anisotropy threshhold",10.0);
  bool useAnisotropy = args.Input<bool>("--useAnisotropy","anisotropy flag",false);
  bool usePointViscosity = args.Input<bool>("--usePointViscosity","use extra viscosity at plate point",false);

  // conditioning for DPG
  int hScaleOption = args.Input<int>("--hScaleOption","option to scale terms to offset conditioning for small h", 0);
  int hScaleTauOption = args.Input<int>("--hScaleTauOption","option to scale tau terms to offset conditioning for small h", 0);

  // etc - experimental
  bool useHigherOrderForU = args.Input<bool>("--useHigherOrderForU","option to increase order for field vars",false); // HGRAD is one higher order 
  bool useConditioningCFL = args.Input<bool>("--useConditioningCFL","option to use a CFL limit for conditioning",false); 
  int numPreRefs = args.Input<int>("--numPreRefs","pre-refinements on singularity",0);

  // IO stuff
  string replayFile = args.Input<string>("--loadFile", "file with refinement history to replay", "");
  string saveFile = args.Input<string>("--saveFile", "file to which to save refinement history", "");
  bool reportTimingResults = args.Input<bool>("--reportTimings", "flag to report timings of solve", false);
  if (rank==0)
    cout << "saveFile is " << saveFile << endl;

  if (rank==0)
    cout << "loadFile is " << replayFile << endl;

  ///////////////////////////////////////////////////////////////////////////////////////////////
  //                            END OF INPUT ARGUMENTS
  ///////////////////////////////////////////////////////////////////////////////////////////////

  if (useHigherOrderForU){
    polyOrder = 1;
  }
  
  // define our manufactured solution or problem bilinear form:
  double Ma = 3.0;
  double cv = 1.0 / ( GAMMA * (GAMMA - 1) * (Ma * Ma) );

  if (rank==0){
    cout << "Running with polynomial order " << polyOrder << ", delta p = " << pToAdd << endl;
    cout << "Running with parameters Re = " << Re << ", Mach = " << Ma << ", and dt = " << dt << " with time tol = " << time_tol_orig << endl;
    cout << "AnisotropyFlag = " << useAnisotropy << ", and aniso thresh = " << anisotropicThresh << endl;
    cout << "Conditioning CFL = " << useConditioningCFL << endl;
  }
  
  bool useTriangles = false;  
  FieldContainer<double> domainPoints(4,2);  
  domainPoints(0,0) = 0.0; // x1
  domainPoints(0,1) = 0.0; // y1
  domainPoints(1,0) = X_BOUNDARY;
  domainPoints(1,1) = 0.0;
  domainPoints(2,0) = X_BOUNDARY;
  domainPoints(2,1) = YTOP;
  domainPoints(3,0) = 0.0;
  domainPoints(3,1) = YTOP;  
  
  int H1Order = polyOrder + 1;
  int horizontalCells = 2*nCells, verticalCells = nCells;
 
  ////////////////////////////////////////////////////////////////////
  // DEFINE VARIABLES 
  ////////////////////////////////////////////////////////////////////
  
  // new-style bilinear form definition
  // traces
  VarFactory varFactory;
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr That = varFactory.traceVar("\\widehat{T}");
  
  // fluxes
  VarPtr F1nhat = varFactory.fluxVar("\\widehat{F}_1n");
  VarPtr F2nhat = varFactory.fluxVar("\\widehat{F}_2n");
  VarPtr F3nhat = varFactory.fluxVar("\\widehat{F}_3n");
  VarPtr F4nhat = varFactory.fluxVar("\\widehat{F}_4n");
  
  // stress fields
  VarPtr sigma11 = varFactory.fieldVar("\\sigma_{11}");
  VarPtr sigma12 = varFactory.fieldVar("\\sigma_{12}");
  VarPtr sigma22 = varFactory.fieldVar("\\sigma_{22}");
  VarPtr q1 = varFactory.fieldVar("q_1");
  VarPtr q2 = varFactory.fieldVar("q_2");
  VarPtr omega = varFactory.fieldVar("\\omega");
  
  VarPtr u_1,u_2,u_3,u_4;
  // H1-ish fields
  if (useHigherOrderForU){ // HGRAD is one higher order 
    u_1= varFactory.fieldVar("u_1",HGRAD); 
    u_2 = varFactory.fieldVar("u_2",HGRAD);
    u_3 = varFactory.fieldVar("u_3",HGRAD);
    u_4 = varFactory.fieldVar("u_4",HGRAD);        
  }else{
    u_1= varFactory.fieldVar("u_1"); 
    u_2 = varFactory.fieldVar("u_2");
    u_3 = varFactory.fieldVar("u_3");
    u_4 = varFactory.fieldVar("u_4");        
  }

  // test fxns
  VarPtr tau1 = varFactory.testVar("\\tau_1",HDIV);
  VarPtr tau2 = varFactory.testVar("\\tau_2",HDIV);
  VarPtr tau3 = varFactory.testVar("\\tau_3",HDIV);
  VarPtr v1 = varFactory.testVar("v_1",HGRAD);
  VarPtr v2 = varFactory.testVar("v_2",HGRAD);
  VarPtr v3 = varFactory.testVar("v_3",HGRAD);
  VarPtr v4 = varFactory.testVar("v_4",HGRAD);
   
  ////////////////////////////////////////////////////////////////////
  // CREATE BILINEAR FORM PTR AND MESH 
  ////////////////////////////////////////////////////////////////////

  BFPtr bf = Teuchos::rcp( new BF(varFactory) ); // initialize bilinear form

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(domainPoints, horizontalCells, 
                                                verticalCells, bf, H1Order, 
                                                H1Order+pToAdd, useTriangles);
  //  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("REFTREE")));
  MeshInfo meshInfo(mesh); // gets info like cell measure, etc

  // for writing ref history to file
  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory ); 
  mesh->registerObserver(refHistory);
  
  // for loading refinement history
  if (replayFile.length() > 0) {
    RefinementHistory refHistory;
    refHistory.loadFromFile(replayFile);
    refHistory.playback(mesh);
  }
  

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  // set variables
  VarPtr u1,u2,rho,T;
  u1 = u_1;
  u2 = u_2;
  rho = u_3;
  T = u_4;

  BCPtr nullBC = Teuchos::rcp((BC*)NULL); RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL); IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  SolutionPtr prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  

  FunctionPtr u1_prev = Function::solution(u1,backgroundFlow);
  FunctionPtr u2_prev = Function::solution(u2,backgroundFlow);
  //  FunctionPtr rho_prev = Teuchos::rcp(new PositiveFunction(Function::solution(rho,backgroundFlow))); // cutoff neg parts
  FunctionPtr rho_prev = Function::solution(rho,backgroundFlow);
  FunctionPtr T_prev = Function::solution(T,backgroundFlow);

  // linearized stresses (q_i is linear, so doesn't need linearizing)
  FunctionPtr sigma11_prev = Function::solution(sigma11,backgroundFlow);
  FunctionPtr sigma12_prev = Function::solution(sigma12,backgroundFlow);
  FunctionPtr sigma22_prev = Function::solution(sigma22,backgroundFlow);

  // previous timestep quantities
  FunctionPtr u1_prev_time = Function::solution(u1,prevTimeFlow); 
  FunctionPtr u2_prev_time = Function::solution(u2,prevTimeFlow); 
  FunctionPtr rho_prev_time = Function::solution(rho,prevTimeFlow); 
  FunctionPtr T_prev_time = Function::solution(T,prevTimeFlow);
  
  FunctionPtr zero = Function::constant(0.0);    

  // ==================== SET INITIAL GUESS ==========================

  double rho_free = 1.0;
  double u1_free = 1.0;
  double u2_free = 0.0;
  double T_free = 1/(GAMMA*(GAMMA-1.0)*Ma*Ma); // TODO - check this value - from Capon paper

  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[rho->ID()] = Function::constant(rho_free);
  functionMap[u1->ID()] = Function::constant(u1_free);
  functionMap[u2->ID()] = Function::constant(u2_free);
  functionMap[T->ID()] = Function::constant(T_free);

  // everything else = 0; previous stresses sigma_ij = 0 as well
  backgroundFlow->projectOntoMesh(functionMap);
  prevTimeFlow->projectOntoMesh(functionMap);

  if (rank==0){
    cout << "Initial guess set" << endl;
  }

  // ==================== END SET INITIAL GUESS ==========================
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE PHYSICAL QUANTITIES
  ////////////////////////////////////////////////////////////////////

  double gam1 = (GAMMA-1.0);
  FunctionPtr u1sq = u1_prev*u1_prev;
  FunctionPtr u2sq = u2_prev*u2_prev;
  FunctionPtr unorm = (u1sq + u2sq);

  FunctionPtr iota = cv*T_prev; // internal energy per unit mass
  FunctionPtr p = (gam1 * cv) * rho_prev * T_prev;
  FunctionPtr e = .5*unorm + iota; // kinetic + internal energy (per unit mass)
  
  // derivatives of p and e
  FunctionPtr dpdrho = (gam1*cv)*T_prev;
  FunctionPtr dpdT = (gam1*cv)*rho_prev;
  FunctionPtr dedu1 = u1_prev;
  FunctionPtr dedu2 = u2_prev;
  double dedT = cv; 

  double beta = 2.0/3.0;
  //  double beta = 0.0;
  FunctionPtr T_visc;
  if (abs(beta)<1e-14){
    T_visc = Function::constant(1.0);
  }else{
    T_visc = Teuchos::rcp( new PowerFunction(T_prev/T_free, beta, T_free/2.0) );  // set min viscosity
  }
 
  FunctionPtr mu = T_visc / Re;

  // try a point artificial diffusion at the plate edge...  
  if (usePointViscosity){
    mu = T_visc/Re + Teuchos::rcp(new TwoDGaussian(1/1000.0,1.0));
  }

  FunctionPtr lambda = -.66 * T_visc / Re;
  FunctionPtr kappa = GAMMA * cv * mu / PRANDTL; // double check sign

  ////////////////////////////////////////////////////////////////////
  // DEFINE BILINEAR FORM
  ////////////////////////////////////////////////////////////////////

  // conservation law fluxes
  bf->addTerm(F1nhat, v1);
  bf->addTerm(F2nhat, v2);
  bf->addTerm(F3nhat, v3);
  bf->addTerm(F4nhat, v4);

  // conservation law test functions 
  map<int, VarPtr> V;
  V[v1->ID()] = v1;
  V[v2->ID()] = v2;
  V[v3->ID()] = v3;
  V[v4->ID()] = v4;

  // stress law traces
  bf->addTerm(u1hat, -tau1->dot_normal() );    
  bf->addTerm(u2hat, -tau2->dot_normal() );
  bf->addTerm(That, -tau3->dot_normal() );

  // stress law test functions
  map<int, VarPtr> TAU;
  TAU[tau1->ID()] = tau1;
  TAU[tau2->ID()] = tau2;
  TAU[tau3->ID()] = tau3;

  ///////////////////////////////////////////////////////////////////////
  // 
  ///////////////////////////////////////////////////////////////////////

  // field variables
  map<int, VarPtr> U;
  U[u1->ID()] = u1;
  U[u2->ID()] = u2;
  U[rho->ID()] = rho;
  U[T->ID()] = T;
  U[sigma11->ID()] = sigma11;
  U[sigma12->ID()] = sigma12;
  U[sigma22->ID()] = sigma22;
  U[q1->ID()] = q1;
  U[q2->ID()] = q2;
  U[omega->ID()] = omega;

  // sparse Jacobians and viscous matrices
  sparseFxnMatrix A_time; // time terms
  sparseFxnMatrix A_euler; // conservation law matrix multiplying eulerian variables
  sparseFxnMatrix A_visc; // conservation law matrix multiplying stresses
  sparseFxnMatrix eps_visc; // multiplies viscous terms (like 1/epsilon * sigma)
  sparseFxnMatrix eps_euler; // multiplies eulerian terms (like grad(u)) 

  ////////////////////////////////////////////////////////////////////
  // CONSTRUCT JACOBIANS
  ////////////////////////////////////////////////////////////////////

  // ========================================= TIMESTEPPING TERMS ====================================

  if (rank==0){
    cout << "Timestep dt = " << dt << endl;
  }
  FunctionPtr invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    
  FunctionPtr sqrtInvDt = Teuchos::rcp(new ScalarParamFunction(sqrt(1.0/dt)));    

  // mass d(rho)/dt
  A_time[v1->ID()][rho->ID()] = invDt*one; 

  // x-momentum d(rho*u1)/dt
  A_time[v2->ID()][rho->ID()] = invDt*u1_prev; 
  A_time[v2->ID()][u1->ID()] = invDt*rho_prev;

  // x-momentum d(rho*u2)/dt
  A_time[v3->ID()][rho->ID()] = invDt*u2_prev;
  A_time[v3->ID()][u2->ID()] = invDt*rho_prev;

  // x-momentum d(rho*u2)/dt
  A_time[v4->ID()][rho->ID()] = invDt*e;
  A_time[v4->ID()][u1->ID()] = invDt*dedu1*rho_prev;
  A_time[v4->ID()][u2->ID()] = invDt*dedu2*rho_prev;
  A_time[v4->ID()][T->ID()] = invDt*dedT*rho_prev;

  // ========================================= CONSERVATION LAWS ====================================

  // mass conservation
  A_euler[v1->ID()][rho->ID()] = u1_prev*e1 + u2_prev*e2;
  A_euler[v1->ID()][u1->ID()] = rho_prev*e1;
  A_euler[v1->ID()][u2->ID()] = rho_prev*e2;

  // x-momentum conservation
  A_euler[v2->ID()][rho->ID()] = (u1sq + dpdrho)*e1 + (u1_prev * u2_prev)*e2;
  A_euler[v2->ID()][u1->ID()] = (2*u1_prev*rho_prev)*e1 + (u2_prev * rho_prev)*e2;
  A_euler[v2->ID()][u2->ID()] = (u1_prev * rho_prev)*e2;
  A_euler[v2->ID()][T->ID()] = dpdT*e1;

  // x-momentum viscous terms
  A_visc[v2->ID()][sigma11->ID()] = negOne*e1;
  A_visc[v2->ID()][sigma12->ID()] = negOne*e2;

  // y-momentum conservation
  A_euler[v3->ID()][rho->ID()] = (u1_prev * u2_prev)*e1 + (u2sq + dpdrho)*e2;
  A_euler[v3->ID()][u1->ID()] = (u2_prev * rho_prev)*e1;
  A_euler[v3->ID()][u2->ID()] = (u1_prev * rho_prev)*e1 + (2 * u2_prev * rho_prev)*e2;
  A_euler[v3->ID()][T->ID()] = dpdT*e2;

  // y-momentum viscous terms
  A_visc[v3->ID()][sigma12->ID()] = negOne*e1;
  A_visc[v3->ID()][sigma22->ID()] = negOne*e2;

  // energy conservation
  FunctionPtr rho_wx = u1_prev * (e + dpdrho);
  FunctionPtr u1_wx = (rho_prev * e + p) + u1_prev*rho_prev*dedu1;
  FunctionPtr u2_wx = u1_prev*rho_prev*dedu2;
  FunctionPtr T_wx = u1_prev*(dpdT + rho_prev*dedT);

  FunctionPtr rho_wy = u2_prev * (e + dpdrho);
  FunctionPtr u1_wy = u2_prev * rho_prev * dedu1;
  FunctionPtr u2_wy = (rho_prev * e + p) + u2_prev * rho_prev * dedu2;
  FunctionPtr T_wy = u2_prev * (dpdT + rho_prev * dedT);

  A_euler[v4->ID()][rho->ID()] = rho_wx*e1 + rho_wy*e2;
  A_euler[v4->ID()][u1->ID()]  = (u1_wx-sigma11_prev)*e1 + (u1_wy-sigma12_prev)*e2;
  A_euler[v4->ID()][u2->ID()]  = (u2_wx-sigma12_prev)*e1 + (u2_wy-sigma22_prev)*e2;
  A_euler[v4->ID()][T->ID()]   = (T_wx)*e1 + (T_wy)*e2;

  // stress portions
  A_visc[v4->ID()][sigma11->ID()]  = -u1_prev*e1;;
  A_visc[v4->ID()][sigma12->ID()]  = -u2_prev*e1 -u1_prev*e2;
  A_visc[v4->ID()][sigma22->ID()]  = -u2_prev*e2;
  A_visc[v4->ID()][q1->ID()]  = negOne*e1;
  A_visc[v4->ID()][q2->ID()]  = negOne*e2;
 
  // ========================================= STRESS LAWS  =========================================

  FunctionPtr lambda_factor_fxn = lambda / (4.0 * mu * (mu + lambda) );
  FunctionPtr two_mu = 2*mu; 

  // 1st stress eqn
  eps_visc[tau1->ID()][sigma11->ID()] = (one/two_mu - lambda_factor_fxn)*e1;
  eps_visc[tau1->ID()][sigma12->ID()] = one/two_mu*e2;
  eps_visc[tau1->ID()][sigma22->ID()] = -lambda_factor_fxn*e1;
  eps_visc[tau1->ID()][omega->ID()] = -one*Re*e2;
  
  eps_euler[tau1->ID()][u1->ID()] = one;
  
  // 2nd stress eqn
  eps_visc[tau2->ID()][sigma11->ID()] = -lambda_factor_fxn*e2;
  eps_visc[tau2->ID()][sigma12->ID()] = one/two_mu*e1;
  eps_visc[tau2->ID()][sigma22->ID()] = (one/two_mu - lambda_factor_fxn)*e2;
  eps_visc[tau2->ID()][omega->ID()] = one*Re*e1;

  eps_euler[tau2->ID()][u2->ID()] = one;

  // Heat stress equation
  eps_visc[tau3->ID()][q1->ID()] = one/kappa*e1; 
  eps_visc[tau3->ID()][q2->ID()] = one/kappa*e2; 
  eps_euler[tau3->ID()][T->ID()] = one;
 
  ///////////////////////////////////////////////////////////
  // APPLICATION OF JACOBIAN DATA
  ///////////////////////////////////////////////////////////

  sparseFxnMatrix::iterator testIt;
  // timestepping terms in conservation laws
  for (testIt = A_time.begin();testIt!=A_time.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      bf->addTerm(trialWeight*U[trialID],V[testID]);
    }
  }

  // conservation (Hgrad) equations
  for (testIt = A_euler.begin();testIt!=A_euler.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      bf->addTerm(-trialWeight*U[trialID],V[testID]->grad());
    }
  }
  
  // stresses in conservation laws
  sparseFxnTensor::iterator xyIt;
  for (testIt = A_visc.begin();testIt!=A_visc.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      bf->addTerm(-trialWeight*U[trialID],V[testID]->grad());
    }
  }

  // Stress (Hdiv) equations 
  for (testIt = eps_visc.begin();testIt!=eps_visc.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      bf->addTerm(trialWeight*U[trialID],TAU[testID]);
    }
  }
 
  // Eulerian component of stress (Hdiv) equations (positive b/c of IBP)
  //  sparseFxnMatrix::iterator testIt;
  for (testIt = eps_euler.begin();testIt!=eps_euler.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      bf->addTerm(trialWeight*U[trialID],TAU[testID]->div());
    }
  } 

  ////////////////////////////////////////////////////////////////////////////
  // TIMESTEPPING RHS TERMS - RESIDUALS (independent of choice of variables)
  ////////////////////////////////////////////////////////////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  // needs prev time residual (u_t(i-1) - u_t(i))/dt
  FunctionPtr u1sq_pt = u1_prev_time*u1_prev_time;
  FunctionPtr u2sq_pt = u2_prev_time*u2_prev_time;
  FunctionPtr iota_pt = cv*T_prev_time; // internal energy
  FunctionPtr unorm_pt = (u1sq_pt + u2sq_pt);
  FunctionPtr e_prev_time = .5*unorm_pt + iota_pt; // kinetic + internal energy

  //rhs 
  LinearTermPtr time_res_LT = Teuchos::rcp(new LinearTerm);
  // mass 
  FunctionPtr time_res_1 = rho_prev_time - rho_prev;  
  time_res_LT->addTerm( (time_res_1 * invDt) * v1);
    
  // x momentum
  FunctionPtr time_res_2 = rho_prev_time * u1_prev_time - rho_prev * u1_prev;
  time_res_LT->addTerm((time_res_2*invDt) * v2);

  // y momentum  
  FunctionPtr time_res_3 = rho_prev_time * u2_prev_time - rho_prev * u2_prev;
  time_res_LT->addTerm((time_res_3 *  invDt ) *v3);

  // energy  
  FunctionPtr time_res_4 = (rho_prev_time * e_prev_time - rho_prev * e);
  time_res_LT->addTerm((time_res_4 * invDt) * v4);    
  
  rhs->addTerm(time_res_LT);

  ////////////////////////////////////////////////////////////////////
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////

  IPPtr ip = Teuchos::rcp( new IP );
  LinearTermPtr tauVecLTx = Teuchos::rcp(new LinearTerm); // for evaluating anisotropic error
  LinearTermPtr tauVecLTy = Teuchos::rcp(new LinearTerm); // for evaluating anisotropic error
  LinearTermPtr vVecLTx = Teuchos::rcp(new LinearTerm); // for evaluating anisotropic error
  LinearTermPtr vVecLTy = Teuchos::rcp(new LinearTerm); // for evaluating anisotropic error

  ////////////////////////////////////////////////////////////////////
  // H-scaling terms for conditioning/approximation V
  ////////////////////////////////////////////////////////////////////

  FunctionPtr invH = Teuchos::rcp(new InvH); // 1/h
  FunctionPtr sqrtH = Teuchos::rcp(new hPowerFunction(.5)); // sqrt(h) - squared in IP
  FunctionPtr invSqrtH = Teuchos::rcp(new InvSqrtH); // 1/sqrt(h) 

  // only really need to scale one or two of these to achieve better conditioning
  FunctionPtr streamlineHScale = Function::constant(1.0);    
  FunctionPtr l2HScale = Function::constant(1.0);
  switch (hScaleOption){
  case 1:
    l2HScale = invSqrtH;
    break;
  case 2:
    l2HScale = invH; // ||v||^2 -> ||v||^2/h^2 
    break;
  case 3:
    streamlineHScale = sqrtH; // scale beta\dot \grad v by h
    break;
  case 4:
    streamlineHScale = sqrtH; // scale beta\dot \grad v by h
    l2HScale = invSqrtH; // scale beta\dot \grad v by h
    break;
  default: // do nothing
    break;
  }

  // default to (1/eps)*||tau||^2 + ||div(tau)||^2
  FunctionPtr TauDivScaling = Function::constant(1.0);
  FunctionPtr TauReScaling = Teuchos::rcp( new EpsilonScaling(1.0/Re) );  // default to Heuer paper norm
  switch (hScaleTauOption){
  case 1:
    TauReScaling = invH;
    break;
  case 2:
    TauDivScaling = sqrtH; // ||div(tau)*sqrt(h)||^2
    break;
  default: // do nothing
    break;
  }

  FunctionPtr timeScaling = Function::constant(1.0);
  timeScaling = sqrtInvDt; // legal with first order term

  ////////////////////////////////////////////////////////////////////
  // Timestep L2 portion of V
  ////////////////////////////////////////////////////////////////////
 
  map<int,LinearTermPtr> vTime;
  initLinearTermVector(A_time,vTime); // initialize to LinearTermPtrs of dimensions of A_time
  for (testIt = A_time.begin();testIt!=A_time.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      vTime[trialID] = vTime[trialID] + trialWeight*V[testID];
    }
  } 

  // adds dual test portion to IP
  for (map<int, LinearTermPtr>::iterator vTimeIt = vTime.begin();vTimeIt != vTime.end();vTimeIt++){
    LinearTermPtr ipSum = vTimeIt->second;
    ip->addTerm(l2HScale*ipSum);
  }

  ////////////////////////////////////////////////////////////////////
  // Rescaled L2 portion of TAU - has Re built into it
  ////////////////////////////////////////////////////////////////////

  map<int, LinearTermPtr> tauVec, tauX, tauY;
  initLinearTermVector(eps_visc,tauVec); // initialize to LinearTermPtrs of dimensions of eps_visc
  initLinearTermVector(eps_visc,tauX); // initialize to LinearTermPtrs of dimensions of eps_visc
  initLinearTermVector(eps_visc,tauY); // initialize to LinearTermPtrs of dimensions of eps_visc

  for (testIt = eps_visc.begin();testIt!=eps_visc.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      tauVec[trialID] = tauVec[trialID] + trialWeight*TAU[testID];
      tauX[trialID] = tauX[trialID] + (e1*trialWeight)*(TAU[testID]->x());
      tauY[trialID] = tauY[trialID] + (e2*trialWeight)*(TAU[testID]->y());
    }
  } 
  // adds dual test portion to IP
  map<int, LinearTermPtr>::iterator tauIt;
  for (tauIt = tauVec.begin();tauIt != tauVec.end();tauIt++){
    LinearTermPtr ipSum = tauIt->second;
    ip->addTerm(TauReScaling*ipSum);
  }  
  // for anisotropic bits - x and y contributions to erro
  for (tauIt = tauX.begin();tauIt!=tauX.end();tauIt++){
    LinearTermPtr lt = tauIt->second;
    tauVecLTx->addTerm(TauReScaling*lt);
  }
  for (tauIt = tauY.begin();tauIt!=tauY.end();tauIt++){
    LinearTermPtr lt = tauIt->second;
    tauVecLTy->addTerm(TauReScaling*lt);
  }
 
  ////////////////////////////////////////////////////////////////////
  // epsilon portion of grad V
  ////////////////////////////////////////////////////////////////////

  FunctionPtr SqrtReInv = Function::constant(1.0/sqrt(Re));

  map<int, LinearTermPtr> vEpsVec, vEpsX, vEpsY;
  initLinearTermVector(A_visc,vEpsVec); // initialize to LinearTermPtrs of dimensions of A_visc
  initLinearTermVector(A_visc,vEpsX); // initialize to LinearTermPtrs of dimensions of A_euler
  initLinearTermVector(A_visc,vEpsY); // initialize to LinearTermPtrs of dimensions of A_euler

  for (testIt = A_visc.begin();testIt!=A_visc.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      vEpsVec[trialID] = vEpsVec[trialID] + trialWeight*V[testID]->grad();
      vEpsX[trialID] = vEpsX[trialID] + (e1*trialWeight)*V[testID]->dx();
      vEpsY[trialID] = vEpsY[trialID] + (e2*trialWeight)*V[testID]->dy();
    }
  } 
  // adds dual test portion to IP
  map<int, LinearTermPtr>::iterator vEpsIt;
  for (vEpsIt = vEpsVec.begin();vEpsIt != vEpsVec.end();vEpsIt++){
    LinearTermPtr ipSum = vEpsIt->second;
    ip->addTerm(timeScaling*SqrtReInv*ipSum);
  }
  // for anisotropic bits - x and y contributions to erro
  for (vEpsIt = vEpsX.begin();vEpsIt != vEpsX.end();vEpsIt++){
    LinearTermPtr lt = vEpsIt->second;
    vVecLTx->addTerm(timeScaling*SqrtReInv*lt);
  }
  for (vEpsIt = vEpsY.begin();vEpsIt != vEpsY.end();vEpsIt++){
    LinearTermPtr lt = vEpsIt->second;
    vVecLTy->addTerm(timeScaling*SqrtReInv*lt);
  }

  ////////////////////////////////////////////////////////////////////
  // "streamline" portion of grad V
  ////////////////////////////////////////////////////////////////////

  map<int, LinearTermPtr> vStreamVec;
  initLinearTermVector(A_euler,vStreamVec); // initialize to LinearTermPtrs of dimensions of A_euler
  for (testIt = A_euler.begin();testIt!=A_euler.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      vStreamVec[trialID] = vStreamVec[trialID] + trialWeight*V[testID]->grad();
    }
  } 
  // adds dual test portion to IP
  map<int, LinearTermPtr>::iterator vStreamIt;
  for (vStreamIt = vStreamVec.begin();vStreamIt != vStreamVec.end();vStreamIt++){
    LinearTermPtr ipSum = vStreamIt->second;
    ip->addTerm(streamlineHScale*ipSum); // streamlineHScale option for conditioning!
    //    ip->addTerm(ipSum);
  }

  ////////////////////////////////////////////////////////////////////
  // rest of the test terms (easier)
  ////////////////////////////////////////////////////////////////////

  ip->addTerm( l2HScale*v1 ); // doesn't get smaller with Re -> 0
  ip->addTerm( l2HScale*v2 );
  ip->addTerm( l2HScale*v3 );
  ip->addTerm( l2HScale*v4 );    
  
  // div remains the same (identity operator in classical variables)
  ip->addTerm(TauDivScaling*tau1->div());
  ip->addTerm(TauDivScaling*tau2->div());
  ip->addTerm(TauDivScaling*tau3->div());

  //  ip = bf->graphNorm();
 
  //  ////////////////////////////////////////////////////////////////////
  //  // DEFINE RHS
  //  ////////////////////////////////////////////////////////////////////

  // mass contributions
  FunctionPtr mass_1 = rho_prev*u1_prev;
  FunctionPtr mass_2 = rho_prev*u2_prev;
  FunctionPtr mass_rhs = (e1 * mass_1 + e2 *mass_2);
  rhs->addTerm( mass_rhs * v1->grad());

  // inviscid momentum contributions
  FunctionPtr momentum_x_1 = rho_prev * u1sq + p ;
  FunctionPtr momentum_x_2 = rho_prev * u1_prev * u2_prev ;
  FunctionPtr momentum_y_1 = rho_prev * u1_prev * u2_prev ;
  FunctionPtr momentum_y_2 = rho_prev * u2sq + p ;
  FunctionPtr mom1_rhs = (e1 * momentum_x_1 + e2 *momentum_x_2 - e1 * sigma11_prev - e2 * sigma12_prev);
  FunctionPtr mom2_rhs = (e1 * momentum_y_1 + e2 *momentum_y_2 - e1 * sigma12_prev - e2 * sigma22_prev);
  rhs->addTerm( mom1_rhs * v2->grad());
  rhs->addTerm( mom2_rhs * v3->grad());

  // inviscid energy contributions
  FunctionPtr rho_e_p = rho_prev * e + p;
  FunctionPtr energy_1 = rho_e_p * u1_prev;
  FunctionPtr energy_2 = rho_e_p * u2_prev;

  // viscous contributions
  FunctionPtr viscousEnergy1 = sigma11_prev * u1_prev + sigma12_prev * u2_prev;
  FunctionPtr viscousEnergy2 = sigma12_prev * u1_prev + sigma22_prev * u2_prev;
  FunctionPtr energy_rhs = (e1 * energy_1 + e2 *energy_2 - e1 * viscousEnergy1 - e2 * viscousEnergy2);
  rhs->addTerm( energy_rhs * v4->grad());

  // stress rhs - no heat flux or omega (asym tensor) accumulated, eqns are linear in those
  FunctionPtr sigmaTrace = -lambda_factor_fxn*(sigma11_prev + sigma22_prev);
  FunctionPtr viscous1 = e1 * sigma11_prev/(2*mu) + e2 * sigma12_prev/(2*mu) + e1 * sigmaTrace;
  FunctionPtr viscous2 = e1 * sigma12_prev/(2*mu) + e2 * sigma22_prev/(2*mu) + e2 * sigmaTrace;

  rhs->addTerm(u1_prev * -tau1->div() - viscous1 * tau1);
  rhs->addTerm(u2_prev * -tau2->div() - viscous2 * tau2);
  rhs->addTerm(T_prev * -tau3->div());

  ////////////////////////////////////////////////////////////////////
  // DEFINE DIRICHLET BC
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowBoundary());
  SpatialFilterPtr wallBoundary = Teuchos::rcp( new WallBoundary());

  // free stream quantities for inflow
  double p_free = (gam1 * cv) * rho_free * T_free;
  double e_free = .5*(u1_free*u1_free+u2_free*u2_free) + cv*T_free; // kinetic + internal energy

  FunctionPtr m1_free = Function::constant(rho_free*u1_free); 
  FunctionPtr m2_free = Function::constant(rho_free*u2_free); 

  // inviscid momentum contributions
  FunctionPtr mom_x1_free = Function::constant(rho_free * u1_free*u1_free + p_free); 
  FunctionPtr mom_x2_free = Function::constant(rho_free * u1_free * u2_free);
  FunctionPtr mom_y1_free = Function::constant(rho_free * u1_free * u2_free);
  FunctionPtr mom_y2_free = Function::constant(rho_free * u2_free*u2_free + p_free );

  double rho_e_p_free =  (rho_free * e_free + p_free);
  FunctionPtr energy_1_free = Function::constant(rho_e_p_free * u1_free);
  FunctionPtr energy_2_free = Function::constant(rho_e_p_free * u2_free);

  // inflow BCs   
  bc->addDirichlet(F1nhat, inflowBoundary, ( e1 * m1_free + e2 * m2_free) * n );
  bc->addDirichlet(F2nhat, inflowBoundary, ( e1 * mom_x1_free + e2 * mom_x2_free) * n );
  bc->addDirichlet(F3nhat, inflowBoundary, ( e1 * mom_y1_free + e2 * mom_y2_free) * n );
  bc->addDirichlet(F4nhat, inflowBoundary, ( e1 * energy_1_free + e2 * energy_2_free) * n ); 

  // =============================================================================================
  
  // wall BCs
  double Tscale = 1.0 + gam1*Ma*Ma/2.0; // from pj capon paper "adaptive finite element method compressible...".  Is equal to 2.8 for Mach 3 and Gamma = 1.4;

  //  bc->addDirichlet(F1nhat, wallBoundary, zero); // for consistency?
  bc->addDirichlet(u2hat, wallBoundary, zero);
  bc->addDirichlet(u1hat, wallBoundary, zero);
  bc->addDirichlet(That, wallBoundary, Function::constant(T_free*Tscale));
  
  //  bc->addDirichlet(F4nhat, wallBoundary, zero); // sets zero heat-flux in free stream bottom boundary

  // =============================================================================================

  // symmetry BCs
  SpatialFilterPtr freeTop = Teuchos::rcp( new FreeStreamBoundaryTop );
  //  bc->addDirichlet(F1nhat, freeTop, zero); // for consistency, but this one doesn't make much diff  
  bc->addDirichlet(u2hat,  freeTop, zero); // top sym bc
  bc->addDirichlet(F2nhat, freeTop, zero);
  bc->addDirichlet(F4nhat, freeTop, zero); // sets zero y-heat flux in free stream top boundary

  // =============================================================================================

  SpatialFilterPtr freeBottom = Teuchos::rcp( new FreeStreamBoundaryBottom );
  //  bc->addDirichlet(F1nhat, freeBottom, zero); // for consistency
  bc->addDirichlet(u2hat,  freeBottom, zero); // sym bcs
  bc->addDirichlet(F2nhat, freeBottom, zero); // sets zero y-stress in free stream bottom boundary
  bc->addDirichlet(F4nhat, freeBottom, zero); // sets zero heat-flux in free stream bottom boundary

  // =============================================================================================

  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, bc, rhs, ip));
  int enrichDegree = H1Order-1; // just for kicks. 
  if (rank==0)
    cout << "enriching cubature by " << enrichDegree << endl;
  solution->setCubatureEnrichmentDegree(enrichDegree); // double cubature enrichment 
  if (reportTimingResults){
    solution->setReportTimingResults(true);
  }
  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow); // u_t(i)
  mesh->registerSolution(prevTimeFlow); // u_t(i-1)
  
  if (rank==0)
    cout << "Refinement threshhold = " << energyThreshold << endl;

  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));
  //  refinementStrategy->setMinH(.5/Re); // 1/2 length of diffusion scale

  int numTimeSteps = 150; // max time steps

  ////////////////////////////////////////////////////////////////////
  // PREREFINE THE MESH
  ////////////////////////////////////////////////////////////////////

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
	//	if (vertices(j,0)>=1.0 && vertices(j,1)==0 && !cellIDset){ // if at the wall
	if ((abs(vertices(j,0)-1.0)<1e-7) && (abs(vertices(j,1))<1e-7) && !cellIDset){ // if at singularity, i.e. if a vertex is (1,0)
	  wallCells.push_back(cellID);
	  cellIDset = true;
	}
      }
    }
    if (i<numPreRefs){
      refinementStrategy->refineCells(wallCells);
    }else{
      //      mesh->pRefine(wallCells);
    }
  }
  if (numPreRefs>0){
    double minSideLength = meshInfo.getMinCellSideLength() ;
    double minCellMeasure = meshInfo.getMinCellMeasure() ;
    if (rank==0){
      cout << "after prerefs, sqrt min cell measure = " << sqrt(minCellMeasure) << ", min side length = " << minSideLength << endl;
    }
  }

  
  ////////////////////////////////////////////////////////////////////
  // PSEUDO-TIME SOLVE STRATEGY 
  ////////////////////////////////////////////////////////////////////

  VTKExporter exporter(solution, mesh, varFactory);
  VTKExporter backgroundFlowExporter(backgroundFlow, mesh, varFactory);

  LinearTermPtr residual = rhs->linearTermCopy();
  residual->addTerm(-bf->testFunctional(solution));  
  RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  FunctionPtr ev1 = Teuchos::rcp(new RepFunction(v1,rieszResidual));
  FunctionPtr ev2 = Teuchos::rcp(new RepFunction(v2,rieszResidual));
  FunctionPtr ev3 = Teuchos::rcp(new RepFunction(v3,rieszResidual));
  FunctionPtr ev4 = Teuchos::rcp(new RepFunction(v4,rieszResidual));
  FunctionPtr etau1 = Teuchos::rcp(new RepFunction(tau1,rieszResidual));
  FunctionPtr etau2 = Teuchos::rcp(new RepFunction(tau2,rieszResidual)); 
  FunctionPtr etau3 = Teuchos::rcp(new RepFunction(tau3,rieszResidual));
  map<int,FunctionPtr> errRepMap;
  errRepMap[v1->ID()] = ev1;
  errRepMap[v2->ID()] = ev2;
  errRepMap[v3->ID()] = ev3;
  errRepMap[v4->ID()] = ev4;
  errRepMap[tau1->ID()] = etau1;
  errRepMap[tau2->ID()] = etau2;
  errRepMap[tau3->ID()] = etau3;

  // for timestepping
  RieszRepPtr rieszTimeResidual = Teuchos::rcp(new RieszRep(mesh, ip, time_res_LT));

  // get entropy
  FunctionPtr rhoToTheGamma = Teuchos::rcp(new PowerFunction(rho_prev,GAMMA));
  FunctionPtr p_prev = (GAMMA-1.0)*rho_prev*cv*T_prev;
  FunctionPtr s = Teuchos::rcp(new LogFunction(p_prev/rhoToTheGamma));
  FunctionPtr H = rho_prev*s; // entropy functional
  FunctionPtr Hsq = H*H; // entropy functional sq   

  if (rank==0){
    cout << "doing timesteps";
    cout << endl;  
  }

  // start first step with very small time tolerance, then change it
  double time_tol = time_tol_orig;

  // time steps
  for (int k = 0;k < numRefs+1;k++){    

    // prevent conditioning issues (and keep robustness under control by increasing 1/dt in problem)
    if (useConditioningCFL){	
      double minSideLength = meshInfo.getMinCellSideLength();	
      double CFL = 50.0; // conservative estimate based off of low Re runs, 75 also seems to work, 100 does not.
      double newDt = min(minSideLength*CFL,dt); // take orig dt if smaller (so dt doesn't get too large)
      if (newDt<dt){
	((ScalarParamFunction*)invDt.get())->set_param(1.0/newDt);
	((ScalarParamFunction*)sqrtInvDt.get())->set_param(sqrt(1.0/newDt));
	if (rank==0)
	  cout << "setting timestep to " << 1.0/((ScalarParamFunction*)invDt.get())->get_param() << endl;
      }
    }      


    ofstream residualFile;      
    if (rank==0){
      std::ostringstream refNum;
      refNum << k;
      string filename1 = "time_res" + refNum.str()+ ".txt";
      residualFile.open(filename1.c_str());
    }
    double L2_time_residual = 1e7;
    int i = 0;
    while(L2_time_residual > time_tol && (i<numTimeSteps)){

      double alpha = 0.0; // to initialize
      int nriter = 0;
      int posEnrich = 10;
      //      while (alpha<1.0 && nriter < maxNRIter){
      double newtonNorm = 1e7; // init to big value
      while (newtonNorm > 1e-6 && nriter < maxNRIter){
	solution->condensedSolve(false);  // don't save memory (maybe turn on if ndofs > maxDofs?)      
	if (k==numRefs){
	//	  solution->setReportConditionNumber(true);
	}
	alpha = 1.0; 
	if (useLineSearch){ // to enforce positivity of density rho
	  double lineSearchFactor = .75; double eps = 1e-7;
	  FunctionPtr rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution) - Function::constant(eps); 
	  bool rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich); 
	  int iter = 0; int maxIter = 20;
	  while (!rhoIsPositive && iter < maxIter){
	    alpha = alpha*lineSearchFactor;
	    rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution); 
	    rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich); 
	    bool rhoIsPositive = Function::solution(rho,backgroundFlow)->isPositive(mesh,posEnrich); 
	    iter++;
	  }
	  if (rank==0 && alpha < 1.0){
	    cout << "line search factor alpha = " << alpha << endl;
	  }      
	}	
	backgroundFlow->addSolution(solution,alpha); // update with dU
	nriter++;

	double rhoNorm = solution->L2NormOfSolutionGlobal(rho->ID());
	double u1Norm = solution->L2NormOfSolutionGlobal(u1->ID());
	double u2Norm = solution->L2NormOfSolutionGlobal(u2->ID());
	double TNorm = solution->L2NormOfSolutionGlobal(T->ID());
	newtonNorm = sqrt(rhoNorm*rhoNorm + u1Norm*u1Norm + u2Norm*u2Norm + TNorm*TNorm);
	if (rank==0)
	  cout << "in Newton step, soln norm = " << newtonNorm << endl;

	/*
	if (useLineSearch){
	  bool rhoIsPositive = Function::solution(rho,backgroundFlow)->isPositive(mesh,posEnrich); 
	  if (rank==0 && !rhoIsPositive){
	    cout << "non positive density detected!" << endl;
	    std::ostringstream oss;
	    oss << i << "_" << nriter;
	    backgroundFlowExporter.exportSolution(string("U_NR") + oss.str());
	  }
	}
	*/
      }
     
      rieszTimeResidual->computeRieszRep();
      double timeRes = rieszTimeResidual->getNorm();
      L2_time_residual = timeRes;

      bool writeTimestepFiles = (k>7);
      std::ostringstream oss;
      oss << k << "_" << i ;
      if (writeTimestepFiles){
	solution->setWriteMatrixToFile(true,string("K")+oss.str());
      }
      if (rank==0){
	residualFile << L2_time_residual << endl;

	cout << "at timestep i = " << i << " with dt = " << 1.0/((ScalarParamFunction*)invDt.get())->get_param() << ", and time residual = " << L2_time_residual << endl;

	if (writeTimestepFiles){
	  string Ustr("U");      
	  string dUstr("dU");      
	  exporter.exportSolution(string("dU") + oss.str());
	  backgroundFlowExporter.exportSolution(string("U") + oss.str());
	}
      } 
 
      prevTimeFlow->setSolution(backgroundFlow); // reset previous time solution to current time sol           
      i++;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                          CHECK CONDITIONING 
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool checkConditioning = true;
    if (checkConditioning){
      double minSideLength = meshInfo.getMinCellSideLength() ;
      StandardAssembler assembler(solution);
      vector<int> cellIDs = meshInfo.getMinCellSizeCellIDs();
      double maxCond = 0.0;
      int maxCellID = 0;
      for (int i = 0;i<cellIDs.size();i++){
	int cellID = cellIDs[i];
	FieldContainer<double> ipMat = assembler.getIPMatrix(mesh->getElement(cellID));
	double cond = SerialDenseWrapper::getMatrixConditionNumber(ipMat);
	if (cond>maxCond){
	  maxCond = cond;
	  maxCellID = cellID;
	}
      }
      if (rank==0){
	cout << "cell ID  " << maxCellID << " has minCellLength " << minSideLength << " and condition estimate " << maxCond << endl;
      }
      std::ostringstream oss;
      oss << k;      	  
      string ipMatName = string("ipMat_")+oss.str()+string(".mat");
      ElementPtr maxCondElem = mesh->getElement(maxCellID);
      FieldContainer<double> ipMat = assembler.getIPMatrix(maxCondElem);
      SerialDenseWrapper::writeMatrixToMatlabFile(ipMatName,ipMat);
      map<int,vector<int> > dofIndices;
      dofIndices[v1->ID()] = maxCondElem->elementType()->testOrderPtr->getDofIndices(v1->ID());
      dofIndices[v2->ID()] = maxCondElem->elementType()->testOrderPtr->getDofIndices(v2->ID());
      dofIndices[v3->ID()] = maxCondElem->elementType()->testOrderPtr->getDofIndices(v3->ID());
      dofIndices[v4->ID()] = maxCondElem->elementType()->testOrderPtr->getDofIndices(v4->ID());
      dofIndices[tau1->ID()] = maxCondElem->elementType()->testOrderPtr->getDofIndices(tau1->ID());
      dofIndices[tau2->ID()] = maxCondElem->elementType()->testOrderPtr->getDofIndices(tau2->ID());
      dofIndices[tau3->ID()] = maxCondElem->elementType()->testOrderPtr->getDofIndices(tau3->ID());
      if (rank==0){
	cout << "v1 test id = " << v1->ID() << endl;
	cout << "v2 test id = " << v2->ID() << endl;
	cout << "v3 test id = " << v3->ID() << endl;
	cout << "v4 test id = " << v4->ID() << endl;
	cout << "t1 test id = " << tau1->ID() << endl;
	cout << "t2 test id = " << tau2->ID() << endl;
	cout << "t3 test id = " << tau3->ID() << endl;
	for (map<int,vector<int> >::iterator mapIt = dofIndices.begin();mapIt!=dofIndices.end();mapIt++){
	  int testID = mapIt->first;
	  std::ostringstream testIDstream;
	  testIDstream << testID;
	  string dofIndicesName = string("dofInds_")+oss.str()+string("_testID_")+testIDstream.str()+string(".txt");
	  /*
	  ofstream dofIndsFile;    
	  dofIndsFile.open(dofIndicesName.c_str());	
	  vector<int> dofInds = mapIt->second;
	  for (int i = 0;i<dofInds.size();i++){
	    dofIndsFile << dofInds[i] << endl;
	  }
	  dofIndsFile.close();
	  */
	}      
      }
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //                                          END OF CHECKING CONDITIONING 
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////

    map<int,double> errMap = solution->energyError();
    FunctionPtr energyErrorFxn = Teuchos::rcp( new EnergyErrorFunction(errMap) );    
    if (rank==0){
      std::ostringstream oss;
      oss << k ;      
      residualFile.close();
      exporter.exportSolution(string("dU")+oss.str());
      backgroundFlowExporter.exportSolution(string("U")+oss.str());
      exporter.exportFunction(energyErrorFxn, string("energyErrFxn")+oss.str());
      //      exporter.exportFunction(H, string("H")+oss.str()); exporter.exportFunction(Hsq, string("Hsq")+oss.str());
    }

    if (k<numRefs){

      // adaptive time tolerance
      double energyError = solution->energyErrorTotal();
      time_tol = max(energyError*1e-2,time_tol_orig);

      if (rank==0){
	cout << "Performing refinement number " << k << ", with energy error " << energyError << " and time tolerance = " << time_tol << endl;
      }     

      if (!useAnisotropy){
	refinementStrategy->refine(rank==0);          // isotropic option
      }else{	       
	if (rank==0)
	  cout << "doing anisotropic refs" << endl;

	rieszResidual->computeRieszRep();
	bool boundaryPart = false;
	FunctionPtr errTauX = tauVecLTx->evaluate(errRepMap,boundaryPart);
	FunctionPtr errTauY = tauVecLTy->evaluate(errRepMap,boundaryPart);
	FunctionPtr errVX = vVecLTx->evaluate(errRepMap,boundaryPart);
	FunctionPtr errVY = vVecLTy->evaluate(errRepMap,boundaryPart);
	FunctionPtr xErr = (errVX)*(errVX) + (errTauX)*(errTauX);
	FunctionPtr yErr = (errVY)*(errVY) + (errTauY)*(errTauY);

	double maxThresh = 1e7;
	vector<int> cellIDs;
	refinementStrategy->getCellsAboveErrorThreshhold(cellIDs);
	int cubEnrich = 5; bool testVsTest = true;
	map<int,double> xErrMap = xErr->cellIntegrals(cellIDs,mesh,cubEnrich,testVsTest);
	map<int,double> yErrMap = yErr->cellIntegrals(cellIDs,mesh,cubEnrich,testVsTest);
	vector<int> xCells,yCells,regCells;
	map<int,double> threshMap;
	map<int,bool> useHRefFlagMap;
	for (int i = 0;i<cellIDs.size();i++){
	  int cellID = cellIDs[i];
	  vector<double> c = mesh->getCellCentroid(cellID);
	  bool atWall = c[0]>1.0; // only do anisotropy on wall, to avoid the weird H1 error wrt singularity issue
	  FieldContainer<double> vv(4,2); mesh->verticesForCell(vv, cellID);
	  bool vertexOnWall = false; bool vertexAtSingularity = false;
	  for (int j = 0;j<4;j++){
	    if (abs(vv(j,1))<(1.0/Re)*(c[0])) // if any vertex is close to wall - if vertex y coord < (1/Re)*(1+x)
	      vertexOnWall = true;	    
	    if ((abs(vv(j,0)-1.0) + abs(vv(j,1)))<1e-10)
	      vertexAtSingularity = true;
	  }
	  bool onWall = !vertexAtSingularity; //&& vertexOnWall; // only do anisotropy on wall, to avoid the weird H1 error wrt singularity issue
	  double ratio = xErrMap[cellID]/yErrMap[cellID];
	  threshMap[cellID] = anisotropicThresh;
	  if (vertexOnWall && atWall){
	    threshMap[cellID] = 2.5; // make it easier to do anisotropic refinements at the wall (scale it with entropy functional in the future?)
	    // WARNING: A HACK TO TRIGGER ANISOTROPIC REFINEMENTS
	    yErrMap[cellID] = yErrMap[cellID]*5.0;
	  }
	  if (vertexAtSingularity || !atWall){
	    threshMap[cellID] = maxThresh; // want ISOTROPIC refinements only at or before singularity
	  }
	  // p-refinement of diffusion-scale terms (for boundary layers and singularities)	  
	  if (useHpStrategy && min(mesh->getCellXSize(cellID),mesh->getCellYSize(cellID))<(1.0/Re)){
	    useHRefFlagMap[cellID] = false;
	    cout << "setting false ref flag" << endl;
	  }else{
	    useHRefFlagMap[cellID] = true;
	  }
	}
	refinementStrategy->refine(rank==0,xErrMap,yErrMap,threshMap,useHRefFlagMap); //anisotropic hp-scheme
	if (rank==0){
	  cout << "Num elements = " << mesh->numActiveElements() << ", and num dofs = " << mesh->numGlobalDofs() << endl;
	}
      }
      if (rank==0){
	cout << "Done with  refinement number " << k << endl;
      }  
      
      // RESET solution every refinement - make sure discretization error doesn't creep in
      //      backgroundFlow->projectOntoMesh(functionMap);
      //      prevTimeFlow->projectOntoMesh(functionMap);

      if (k==numRefs-1){
	// save mesh to file
	if (saveFile.length() > 0) {
	  if (rank == 0) {
	    cout << "saving mesh file to " << saveFile << " on refinement " << k << endl;
	    refHistory->saveToFile(saveFile);
	  }
	}
      }

    } else {

   
      time_tol = time_tol_orig; // return to original time tolerance for final solve
      if (rank==0){
	cout << "Finishing it off with the final solve" << endl;
      }

    }
    
  }
  if (rank==0){
    exporter.exportSolution("dU");
    backgroundFlowExporter.exportSolution("U");
  }
  
  return 0;
}
