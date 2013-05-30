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
#include "MeshUtilities.h"

#include "IPSwitcher.h"

typedef map< int, FunctionPtr > sparseFxnVector;    // dim = {trialID}
typedef map< int, sparseFxnVector > sparseFxnMatrix; // dim = {testID, trialID}
typedef map< int, sparseFxnMatrix > sparseFxnTensor; // dim = {spatial dim, testID, trialID}

static const double GAMMA = 1.4;
static const double PRANDTL = 0.72;
static const double YTOP = 1.0;
static const double X_BOUNDARY = 2.0;
static const double rampHeight = 0.0;

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
    if (abs(rampHeight)<tol){
      return yMatch; // if it's just a flat plate
    }else{
      return ((abs(y) < tol) && (x < .50) && (x > 0.0)); // if it's a ramp
    }

  }
};

// .5 to 1.0 - in front of ramp
class NonRampPlate : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = ((abs(y) < tol) && (x < 1.0) && (x > 0.5));
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

class TwoDGaussian : public SimpleFunction {
  double _width,_amplitude,_center;
public:
  TwoDGaussian(double width,double amplitude){
    _width = width;
    _amplitude = amplitude;
    _center = 1.0;
  }
  TwoDGaussian(double width,double center, double amplitude){
    _width = width;
    _center = center;
    _amplitude = amplitude;
  }
  double value(double x, double y){
    return _amplitude*exp(-((x-_center)*(x-_center)+y*y)/(2.0*_width*_width));
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
  double Ma = args.Input<double>("--Ma","Mach number",3.0);

  // solver
  int nCells = args.Input<int>("--nCells", "num cells",2);  
  int polyOrder = args.Input<int>("--p","order of approximation",2);
  int pToAdd = args.Input<int>("--pToAdd", "test space enrichment",2); 
  double time_tol_orig = args.Input<double>("--timeTol", "time step tolerance",1e-8);
  bool useLineSearch = args.Input<bool>("--useLineSearch", "flag for line search",true); // default to zero
  int maxNRIter = args.Input<int>("--maxNRIter","maximum number of NR iterations",2); // default to one per timestep
  int numTimeSteps = args.Input<int>("--maxTimeSteps","max number of time steps",150); // max time steps
  double minDt = args.Input<double>("--minDt","min timestep for adaptive timestepping",.01); // max time steps
  double maxDt = args.Input<double>("--maxDt","max timestep for adaptive timestepping",.1); // max time steps

  // adaptivity
  int numUniformRefs = args.Input<int>("--numUniformRefs","num uniform refinements (pre-adaptivity)",0);
  int numRefs = args.Input<int>("--numRefs","num adaptive refinements",0);
  double energyThreshold = args.Input<double>("--energyThreshold", "energy thresh for adaptivity",0.25); // for mesh refinements 
  bool useHpStrategy = args.Input<bool>("--useHpStrategy","option to use a 'cheap' hp strategy", false);
  double anisotropicThresh = args.Input<int>("--anisotropicThresh","anisotropy threshhold",10.0);
  bool useAnisotropy = args.Input<bool>("--useAnisotropy","anisotropy flag",false);
  bool usePointViscosity = args.Input<bool>("--usePointViscosity","use extra viscosity at plate point",false);
  bool useAdaptiveTimestepping = args.Input<bool>("--useAdaptiveTimestepping","use adaptive timestepping a la Ben Kirk", false);

  // conditioning for DPG
  int hScaleOption = args.Input<int>("--hScaleOption","option to scale terms to offset conditioning for small h", 0);
  int hScaleTauOption = args.Input<int>("--hScaleTauOption","option to scale tau terms to offset conditioning for small h", 0);

  // etc - experimental
  bool useHigherOrderForU = args.Input<bool>("--useHigherOrderForU","option to increase order for field vars",false); // HGRAD is one higher order 
  bool useConditioningCFL = args.Input<bool>("--useConditioningCFL","option to use a CFL limit for conditioning",false); 
  int numPreRefs = args.Input<int>("--numPreRefs","pre-refinements on singularity",0);
  bool scalePlate = args.Input<bool>("--scalePlate","flag to weight plate so it matters less",false);
  double ipSwitch = args.Input<double>("--ipSwitch","smallest elem thresh to switch to graph norm",0.0); // default to not changing

  // IO stuff
  string saveFile = args.Input<string>("--meshSaveFile", "file to which to save refinement history", "");
  string replayFile = args.Input<string>("--meshLoadFile", "file with refinement history to replay", "");

  string solnSaveFile = args.Input<string>("--solnSaveFile", "file to which to save soln", "");
  string solnLoadFile = args.Input<string>("--solnLoadFile", "file from which to load soln", "");

  string dir = args.Input<string>("--dir", "dir to which we save/from which we load files","");

  bool reportTimingResults = args.Input<bool>("--reportTimings", "flag to report timings of solve", false);
  bool writeTimestepFiles = args.Input<bool>("--writeTimestepFiles","flag to turn on and off time step writing",false);

  if (rank==0){
    cout << "saveFile is " << dir + saveFile << endl;
    cout << "saveSolnFile is " << dir + solnSaveFile << endl;
  }

  if (rank==0){
    cout << "loadFile is " << dir + replayFile << endl;
    cout << "loadSolnFile is " << dir + solnLoadFile << endl;
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  //                            END OF INPUT ARGUMENTS
  ///////////////////////////////////////////////////////////////////////////////////////////////

  if (useHigherOrderForU){
    polyOrder = 1;
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
  //  int H1Order = 2; // start with linears, and keep singularity linear at singularity
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
    u_1 = varFactory.fieldVar("u_1",HGRAD); 
    u_2 = varFactory.fieldVar("u_2",HGRAD);
    u_3 = varFactory.fieldVar("u_3",HGRAD);
    u_4 = varFactory.fieldVar("u_4",HGRAD);        
  }else{
    u_1 = varFactory.fieldVar("u_1"); 
    u_2 = varFactory.fieldVar("u_2");
    u_3 = varFactory.fieldVar("rho");
    u_4 = varFactory.fieldVar("T");        
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
  //  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(domainPoints, horizontalCells, verticalCells, bf, H1Order, H1Order+pToAdd, useTriangles);
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildRampMesh(rampHeight,bf, H1Order, H1Order+pToAdd);
  //  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("REFTREE")));
  MeshInfo meshInfo(mesh); // gets info like cell measure, etc

  // for writing ref history to file
  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory ); 
  mesh->registerObserver(refHistory);
  

  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL); RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL); IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  SolutionPtr prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP));
  int enrichDegree = 2; // just for kicks. 
  if (rank==0)
    cout << "enriching cubature by " << enrichDegree << endl;
  solution->setCubatureEnrichmentDegree(enrichDegree); // double cubature enrichment 
  if (reportTimingResults){
    solution->setReportTimingResults(true);
  }
  
  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow); // u_t(i)
  mesh->registerSolution(prevTimeFlow); // u_t(i-1)
    
  // for loading refinement history
  if (replayFile.length() > 0) {
    RefinementHistory refHistory;
    replayFile = dir + replayFile;
    refHistory.loadFromFile(replayFile);
    refHistory.playback(mesh);
    int numElems = mesh->numActiveElements();
    if (rank==0){
      double minSideLength = meshInfo.getMinCellSideLength() ;    
      cout << "after replay, num elems = " << numElems << " and min side length = " << minSideLength << endl;
    }
  }  
  if (solnLoadFile.length() > 0) {
    std::ostringstream ss;
    //    ss << dir <<  "solution_" << solnLoadFile;
    //    solution->readFromFile(ss.str());
    ss.str("");
    ss << dir << "backgroundFlow_" << solnLoadFile;
    backgroundFlow->readFromFile(ss.str());
    ss.str("");
    ss << dir << "prevTimeFlow_" << solnLoadFile;
    prevTimeFlow->readFromFile(ss.str());
  }

  ////////////////////////////////////////////////////////////////////
  // PSEUDO-TIME SOLVE STRATEGY 
  ////////////////////////////////////////////////////////////////////

  VTKExporter exporter(solution, mesh, varFactory);
  VTKExporter backgroundFlowExporter(backgroundFlow, mesh, varFactory);

 
  if (rank==0){
    exporter.exportSolution("dU");
    backgroundFlowExporter.exportSolution("U");
  }
  
  return 0;
}
