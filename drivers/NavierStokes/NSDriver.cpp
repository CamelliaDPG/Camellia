#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "LagrangeConstraints.h"

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

#include "ElementType.h"
#include "Element.h"

#include "MeshPolyOrderFunction.h"
#include "CondensationSolver.h"

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;
typedef map< int, FunctionPtr > sparseFxnVector;    // dim = {trialID}
typedef map< int, sparseFxnVector > sparseFxnMatrix; // dim = {testID, trialID}
typedef map< int, sparseFxnMatrix > sparseFxnTensor; // dim = {spatial dim, testID, trialID}

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;
typedef Teuchos::RCP< Mesh > MeshPtr;

static const double GAMMA = 1.4;
static const double PRANDTL = 0.72;
static const double YTOP = 1.0;
static const double X_BOUNDARY = 2.0;

using namespace std;

// ===================== Mesh functions ====================

class MeshInfo {
  MeshPtr _mesh;
public:
  MeshInfo(MeshPtr mesh){
    _mesh = mesh;
  }
  map<int,double> getCellMeasures(){	
    map<int,double> cellMeasures;
    vector< ElementTypePtr > elemTypes = _mesh->elementTypes(); 
    vector< ElementTypePtr >::iterator typeIt;
    for (typeIt=elemTypes.begin();typeIt!=elemTypes.end();typeIt++){
      ElementTypePtr elemTypePtr = (*typeIt);
      BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr, _mesh));  
      vector< ElementPtr > elemsOfType =_mesh->elementsOfTypeGlobal(elemTypePtr);
      vector<int> cellIDs;
      for (int i = 0;i<elemsOfType.size();i++){
	cellIDs.push_back(elemsOfType[i]->cellID());
      }
      basisCache->setPhysicalCellNodes(_mesh->physicalCellNodesGlobal(elemTypePtr ), cellIDs, false); // no side cache
  
      FieldContainer<double> cell_h = basisCache->getCellMeasures();
      int numElems = _mesh->numElementsOfType( elemTypePtr );
      for (int i = 0;i < numElems;i++){
	cellMeasures[cellIDs[i]] = cell_h(i);	
      }      
    }
    return cellMeasures;
  }

  double getMinCellMeasure(){
    map<int,double> cellMeasures = getCellMeasures();
    map<int,double>::iterator hIt;
    double minMeasure = 1e7;
    for (hIt = cellMeasures.begin();hIt != cellMeasures.end();hIt++){
      minMeasure = min(minMeasure, hIt->second);
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

class SqrtHFunction : public hFunction {
public:
  SqrtHFunction() {
  }
  double value(double x, double y, double h) {
    return sqrt(h);
  }
};

class InvSqrtHFunction : public hFunction {
public:
  InvSqrtHFunction() {
  }
  double value(double x, double y, double h) {
    return 1.0/sqrt(h);
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

void initLinearTermVector(sparseFxnMatrix A, map<int, LinearTermPtr> &Mvec){

  FunctionPtr zero = Teuchos::rcp(new ConstantScalarFunction(0.0));
  
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
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif
  int polyOrder = 2;
  int pToAdd = 2; // for tests
  
  // define our manufactured solution or problem bilinear form:
  double Re = 1e3;
  double Ma = 3.0;
  double cv = 1.0 / ( GAMMA * (GAMMA - 1) * (Ma * Ma) );

  if (rank==0){
    cout << "Running with polynomial order " << polyOrder << ", delta p = " << pToAdd << endl;
    cout << "Running with parameters Re = " << Re << ", Mach = " << Ma << endl;
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
  int nCells = 1;
  if ( argc > 1) {
    nCells = atoi(argv[1]);
    if (rank==0){
      cout << "nCells = " << nCells << endl;
    }
  }
  int numRefs = 0;
  if ( argc > 2) {
    numRefs = atoi(argv[2]);
    if (rank==0){
      cout << "numRefs = " << numRefs << endl;
    }
  }
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
  
  // fields
  VarPtr u1 = varFactory.fieldVar("u_1");
  VarPtr u2 = varFactory.fieldVar("u_2");
  VarPtr rho = varFactory.fieldVar("\\rho");
  VarPtr T = varFactory.fieldVar("T");
  VarPtr sigma11 = varFactory.fieldVar("\\sigma_{11}");
  VarPtr sigma12 = varFactory.fieldVar("\\sigma_{12}");
  VarPtr sigma22 = varFactory.fieldVar("\\sigma_{22}");
  VarPtr q1 = varFactory.fieldVar("q_1");
  VarPtr q2 = varFactory.fieldVar("q_2");
  VarPtr omega = varFactory.fieldVar("\\omega");
  
  // test fxns
  VarPtr tau1 = varFactory.testVar("\\tau_1",HDIV);
  VarPtr tau2 = varFactory.testVar("\\tau_2",HDIV);
  VarPtr tau3 = varFactory.testVar("\\tau_3",HDIV);
  VarPtr v1 = varFactory.testVar("v_1",HGRAD);
  VarPtr v2 = varFactory.testVar("v_2",HGRAD);
  VarPtr v3 = varFactory.testVar("v_3",HGRAD);
  VarPtr v4 = varFactory.testVar("v_4",HGRAD);
  
  BFPtr bf = Teuchos::rcp( new BF(varFactory) ); // initialize bilinear form
  
  ////////////////////////////////////////////////////////////////////
  // CREATE MESH 
  ////////////////////////////////////////////////////////////////////
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(domainPoints, horizontalCells, 
                                                verticalCells, bf, H1Order, 
                                                H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  MeshInfo meshInfo(mesh); // gets info like cell measure, etc

  //  FunctionPtr partitions = Teuchos::rcp( new PartitionFunction(mesh) );
  
  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  SolutionPtr prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  

  vector<double> e1(2); // (1,0)
  vector<double> e2(2); // (0,1)
  e1[0] = 1;
  e2[1] = 1;
  
  FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u1) );
  FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u2) );
  FunctionPtr rho_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, rho) );  
  FunctionPtr T_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, T) );

  // linearized stresses (q_i is linear, so doesn't need linearizing)
  FunctionPtr sigma11_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, sigma11) );
  FunctionPtr sigma12_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, sigma12) );
  FunctionPtr sigma22_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, sigma22) );

  // previous timestep quantities
  FunctionPtr u1_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u1) );
  FunctionPtr u2_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u2) );
  FunctionPtr rho_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, rho) );
  FunctionPtr T_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, T) );

  // for subsonic outflow 
  FunctionPtr u1hat_prev  = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u1hat ) );
  FunctionPtr That_prev   = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, That  ) );
  FunctionPtr F2nhat_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, F2nhat) );
  FunctionPtr F3nhat_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, F3nhat) );
  FunctionPtr F4nhat_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, F4nhat) );

  FunctionPtr u1hat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u1hat) );
  FunctionPtr That_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, That) );
  FunctionPtr F2nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F2nhat) );
  FunctionPtr F3nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F3nhat) );
  FunctionPtr F4nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F4nhat) );

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );

  // ==================== SET INITIAL GUESS ==========================

  double rho_free = 1.0;
  double u1_free = 1.0;
  double u2_free = 0.0;
  double T_free = 1/(GAMMA*(GAMMA-1.0)*Ma*Ma); // TODO - check this value - from Capon paper

  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[rho->ID()] = Teuchos::rcp( new ConstantScalarFunction(rho_free) );
  functionMap[u1->ID()] = Teuchos::rcp( new ConstantScalarFunction(u1_free) );
  functionMap[u2->ID()] = Teuchos::rcp( new ConstantScalarFunction(u2_free) );
  functionMap[T->ID()] = Teuchos::rcp( new ConstantScalarFunction(T_free) );

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

  //  double beta = 2.0/3.0;
  double beta = 0.0;
  FunctionPtr T_visc;
  if (abs(beta)<1e-14){
    T_visc = Teuchos::rcp( new ConstantScalarFunction(1.0) );  
  }else{
    T_visc = Teuchos::rcp( new PowerFunction(T_prev/T_free, beta, 1.0/1000.0) );  // set 1/Re = min viscosity
  }
  FunctionPtr mu = T_visc / Re;
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

  map<int, VarPtr> V;
  V[v1->ID()] = v1;
  V[v2->ID()] = v2;
  V[v3->ID()] = v3;
  V[v4->ID()] = v4;

  map<int, VarPtr> TAU;
  TAU[tau1->ID()] = tau1;
  TAU[tau2->ID()] = tau2;
  TAU[tau3->ID()] = tau3;

  // sparse Jacobians and viscous matrices
  sparseFxnMatrix A_euler; // 
  sparseFxnMatrix A_visc; // 
  //  sparseFxnTensor eps_visc; // multiplies viscous terms (like 1/epsilon * sigma)
  sparseFxnMatrix eps_visc; // multiplies viscous terms (like 1/epsilon * sigma)
  sparseFxnMatrix eps_euler; // multiplies eulerian terms (like grad(u)) 

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
  FunctionPtr negOne =  Teuchos::rcp( new ConstantScalarFunction(-1.0));
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
  FunctionPtr u1_wx = rho_prev * e + p + u1_prev*rho_prev*dedu1;
  FunctionPtr u2_wx = u1_prev*rho_prev*dedu2;
  FunctionPtr T_wx = u1_prev*(dpdT + rho_prev*dedT);

  FunctionPtr rho_wy = u2_prev * (e + dpdrho);
  FunctionPtr u1_wy = u2_prev * rho_prev * dedu1;
  FunctionPtr u2_wy = rho_prev * e + p + u2_prev * rho_prev * dedu2;
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

  // conservation (Hgrad) equations
  sparseFxnMatrix::iterator testIt;
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

  // ========================================= STRESS LAWS  =========================================

  bf->addTerm(u1hat, -tau1->dot_normal() );    
  bf->addTerm(u2hat, -tau2->dot_normal() );
  bf->addTerm(That, -tau3->dot_normal() );

  FunctionPtr lambda_factor_fxn = lambda / (4.0 * mu * (mu + lambda) );
  FunctionPtr two_mu = 2*mu; 
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0));

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
  eps_visc[tau3->ID()][q1->ID()] = one/kappa*e1; //Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
  eps_visc[tau3->ID()][q2->ID()] = one/kappa*e2; //Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
  eps_euler[tau3->ID()][T->ID()] = one;
  
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
 
  ////////////////////////////////////////////////////////////////////
  // TIMESTEPPING TERMS
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  double CFL = .9; // not exactly CFL, just how we want our timestep to change w/min h
  double hmin = sqrt(meshInfo.getMinCellMeasure());
  bool useCFL = false; // rescale dt with min mesh size
  double dtMin = .25;

  //  double dt = max(dtMin,hmin*CFL);
  double dt = .1;

  if (rank==0){
    cout << "CFL = " << CFL << endl;
    cout << "hmin = " << hmin << endl;
    cout << "Timestep dt = " << dt << endl;
  }
  FunctionPtr invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    

  // needs prev time residual (u_t(i-1) - u_t(i))/dt
  FunctionPtr u1sq_pt = u1_prev_time*u1_prev_time;
  FunctionPtr u2sq_pt = u2_prev_time*u2_prev_time;
  FunctionPtr iota_pt = cv*T_prev_time; // internal energy
  FunctionPtr unorm_pt = (u1sq_pt + u2sq_pt);
  FunctionPtr e_prev_time = .5*unorm_pt + iota_pt; // kinetic + internal energy

  // mass 
  bf->addTerm(rho,invDt*v1);    
  FunctionPtr time_res_1 = rho_prev_time - rho_prev;  
  rhs->addTerm( (time_res_1 * invDt) * v1);
    
  // x momentum
  bf->addTerm(u1_prev * rho + rho_prev * u1, invDt * v2);
  FunctionPtr time_res_2 = rho_prev_time * u1_prev_time - rho_prev * u1_prev;
  rhs->addTerm((time_res_2*invDt) * v2);

  // y momentum
  bf->addTerm(u2_prev * rho + rho_prev * u2, invDt * v3);
  FunctionPtr time_res_3 = rho_prev_time * u2_prev_time - rho_prev * u2_prev;
  rhs->addTerm((time_res_3 *  invDt ) *v3);

  // energy  
  bf->addTerm((e) * rho + (dedu1*rho_prev) * u1 + (dedu2*rho_prev) * u2 + (dedT*rho_prev) * T, invDt * v4 );
  FunctionPtr time_res_4 = (rho_prev_time * e_prev_time - rho_prev * e);
  rhs->addTerm((time_res_4 * invDt) * v4);    

  ////////////////////////////////////////////////////////////////////
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////
  // function to scale the squared guy by epsilon/|K| 
  FunctionPtr ReScaling = Teuchos::rcp( new EpsilonScaling(1.0/Re) ); 

  IPPtr ip = Teuchos::rcp( new IP );

  ////////////////////////////////////////////////////////////////////
  // Timestep L2 portion of V
  ////////////////////////////////////////////////////////////////////

  FunctionPtr invSqrtH = Teuchos::rcp(new InvSqrtHFunction);

  // rho dt term
  ip->addTerm(invDt*(v1 + u1_prev*v2 + u2_prev*v3 + e*v4));
  // u1 dt term
  ip->addTerm(invDt*(rho_prev*v2 + (dedu1*rho_prev)*v4));
  // u2 dt term
  ip->addTerm(invDt*(rho_prev*v3 + (dedu2*rho_prev)*v4));
  // T dt term
  ip->addTerm(invDt*(dedT*rho_prev*v4) );
 
  ////////////////////////////////////////////////////////////////////
  // Rescaled L2 portion of TAU - has Re built into it
  ////////////////////////////////////////////////////////////////////

  //  FunctionPtr radialWeight = Teuchos::rcp(new RadialWeightFunction(1.0/Re));
  map<int, LinearTermPtr> tauVec;
  initLinearTermVector(eps_visc,tauVec); // initialize to LinearTermPtrs of dimensions of eps_visc

  for (testIt = eps_visc.begin();testIt!=eps_visc.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      tauVec[trialID] = tauVec[trialID] + trialWeight*TAU[testID];
    }
  } 
  // adds dual test portion to IP
  map<int, LinearTermPtr>::iterator tauIt;
  for (tauIt = tauVec.begin();tauIt != tauVec.end();tauIt++){
    LinearTermPtr ipSum = tauIt->second;
    ip->addTerm(ReScaling*ipSum);
  }
 
  ////////////////////////////////////////////////////////////////////
  // epsilon portion of grad V
  ////////////////////////////////////////////////////////////////////
  FunctionPtr SqrtReInv = Teuchos::rcp(new ConstantScalarFunction(1.0/sqrt(Re)));

  map<int, LinearTermPtr> vEpsVec;
  initLinearTermVector(A_visc,vEpsVec); // initialize to LinearTermPtrs of dimensions of A_visc

  for (testIt = A_visc.begin();testIt!=A_visc.end();testIt++){
    int testID = testIt->first;
    sparseFxnVector a = testIt->second;
    sparseFxnVector::iterator trialIt;
    for (trialIt = a.begin();trialIt!=a.end();trialIt++){
      int trialID = trialIt->first;
      FunctionPtr trialWeight = trialIt->second;
      vEpsVec[trialID] = vEpsVec[trialID] + trialWeight*V[testID]->grad();
    }
  } 
  // adds dual test portion to IP
  map<int, LinearTermPtr>::iterator vEpsIt;
  for (vEpsIt = vEpsVec.begin();vEpsIt != vEpsVec.end();vEpsIt++){
    LinearTermPtr ipSum = vEpsIt->second;
    ip->addTerm(SqrtReInv*ipSum);
  }
 
  ////////////////////////////////////////////////////////////////////
  // "streamline" portion of grad V
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Function> sqrtH = Teuchos::rcp(new SqrtHFunction);
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
    //    ip->addTerm(sqrtH*ipSum); // for conditioning!
    ip->addTerm(ipSum);
  }
 
  ////////////////////////////////////////////////////////////////////
  // rest of the test terms (easier)
  ////////////////////////////////////////////////////////////////////

  ip->addTerm( ReScaling*v1 );
  ip->addTerm( ReScaling*v2 );
  ip->addTerm( ReScaling*v3 );
  ip->addTerm( ReScaling*v4 );    
  //  ip->addTerm( v1 ); // doesn't get smaller with Re -> 0
  //  ip->addTerm( v2 );
  //  ip->addTerm( v3 );
  //  ip->addTerm( v4 );    

  // div remains the same (identity operator in classical variables)
  ip->addTerm(tau1->div());
  ip->addTerm(tau2->div());
  ip->addTerm(tau3->div());
  
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

  FunctionPtr m1_free = Teuchos::rcp( new ConstantScalarFunction(rho_free*u1_free) );
  FunctionPtr m2_free = Teuchos::rcp( new ConstantScalarFunction(rho_free*u2_free) );

  // inviscid momentum contributions
  FunctionPtr mom_x1_free = Teuchos::rcp( new ConstantScalarFunction( rho_free * u1_free*u1_free + p_free)) ;
  FunctionPtr mom_x2_free = Teuchos::rcp( new ConstantScalarFunction(rho_free * u1_free * u2_free)) ;
  FunctionPtr mom_y1_free = Teuchos::rcp( new ConstantScalarFunction( rho_free * u1_free * u2_free ));
  FunctionPtr mom_y2_free = Teuchos::rcp( new ConstantScalarFunction(rho_free * u2_free*u2_free + p_free ));

  double rho_e_p_free =  (rho_free * e_free + p_free);
  FunctionPtr energy_1_free = Teuchos::rcp( new ConstantScalarFunction(rho_e_p_free * u1_free) );
  FunctionPtr energy_2_free = Teuchos::rcp( new ConstantScalarFunction( rho_e_p_free * u2_free) );

  // inflow BCs   
  bc->addDirichlet(F1nhat, inflowBoundary, ( e1 * m1_free + e2 * m2_free) * n );
  bc->addDirichlet(F2nhat, inflowBoundary, ( e1 * mom_x1_free + e2 * mom_x2_free) * n );
  bc->addDirichlet(F3nhat, inflowBoundary, ( e1 * mom_y1_free + e2 * mom_y2_free) * n );
  bc->addDirichlet(F4nhat, inflowBoundary, ( e1 * energy_1_free + e2 * energy_2_free) * n ); 

  // =============================================================================================
  
  // wall BCs
  double Tscale = 1.0 + gam1*Ma*Ma/2.0; // from pj capon paper "adaptive finite element method compressible...".  Is equal to 2.8 for Mach 3 and Gamma = 1.4;

  bc->addDirichlet(F1nhat, wallBoundary, zero); // for consistency?
  bc->addDirichlet(u2hat, wallBoundary, zero);
  bc->addDirichlet(u1hat, wallBoundary, zero);
  bc->addDirichlet(That, wallBoundary, Teuchos::rcp(new ConstantScalarFunction(T_free*Tscale))); 
  //  bc->addDirichlet(F4nhat, wallBoundary, zero); // sets zero heat-flux in free stream bottom boundary

  // =============================================================================================

  // symmetry BCs
  SpatialFilterPtr freeTop = Teuchos::rcp( new FreeStreamBoundaryTop );
  //  bc->addDirichlet(u2hat,  freeTop, zero); // top sym bc
  bc->addDirichlet(F1nhat, freeTop, zero); // for consistency
  bc->addDirichlet(F2nhat, freeTop, zero);
  bc->addDirichlet(F4nhat, freeTop, zero); // sets zero y-heat flux in free stream top boundary

  // =============================================================================================

  SpatialFilterPtr freeBottom = Teuchos::rcp( new FreeStreamBoundaryBottom );
  bc->addDirichlet(F1nhat, freeBottom, zero); // for consistency
  bc->addDirichlet(u2hat,  freeBottom, zero); // sym bcs
  bc->addDirichlet(F2nhat, freeBottom, zero); // sets zero y-stress in free stream bottom boundary
  bc->addDirichlet(F4nhat, freeBottom, zero); // sets zero heat-flux in free stream bottom boundary

  // =============================================================================================

  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, bc, rhs, ip));
  int enrichDegree = H1Order; // just for kicks. 
  cout << "enriching cubature by " << enrichDegree << endl;
  solution->setCubatureEnrichmentDegree(enrichDegree); // double cubature enrichment 

  //  solution->setReportTimingResults(true); // print out timing 

  bool setOutflowBC = false;
  if (setOutflowBC){
    bool usePenalty = true;
    if (usePenalty){
      Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
      SpatialFilterPtr outflow = Teuchos::rcp( new OutflowBoundary);
      FunctionPtr subsonicIndicator = Teuchos::rcp( new SubsonicIndicator(u1hat_prev, That_prev, GAMMA, cv) );
      // conditions on u_n = u_1, sigma_ns = sigma_12, q_1 flux
      pc->addConstraint(subsonicIndicator*u1hat == subsonicIndicator*u1hat_prev,outflow);
      pc->addConstraint(subsonicIndicator*F3nhat == subsonicIndicator*F3nhat_prev,outflow);
      pc->addConstraint(subsonicIndicator*F4nhat == subsonicIndicator*F4nhat_prev,outflow);

      solution->setFilter(pc);

    } else {
      SpatialFilterPtr subsonicOutflow = Teuchos::rcp( new SubsonicOutflow(u1hat_prev, That_prev, GAMMA, cv));
      /*
      bc->addDirichlet(u1hat, subsonicOutflow, u1hat_prev); // u_n
      bc->addDirichlet(F3nhat, subsonicOutflow, F3nhat_prev); // sigma_12
      bc->addDirichlet(F4nhat, subsonicOutflow, F4nhat_prev); // q_1
      */
      Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
      pc->addConstraint(u1hat == u1hat_prev_time,subsonicOutflow);
      pc->addConstraint(F3nhat == F3nhat_prev_time,subsonicOutflow);
      pc->addConstraint(F4nhat == F4nhat_prev_time,subsonicOutflow);

      solution->setFilter(pc);

    }
  }

  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow); // u_t(i)
  mesh->registerSolution(prevTimeFlow); // u_t(i-1)
  
  double energyThreshold = 0.2; // for mesh refinements
  if (rank==0)
    cout << "Refinement threshhold = " << energyThreshold << endl;

  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));

  int numTimeSteps = 150; // max time steps
  int numNRSteps = 1;
  
  ////////////////////////////////////////////////////////////////////
  // PREREFINE THE MESH
  ////////////////////////////////////////////////////////////////////

  int numPreRefs = 0;
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
	if (vertices(j,0)>=1.0 && vertices(j,1)==0 && !cellIDset){ // if at the wall
	  //	if ((abs(vertices(j,0)-1.0)<1e-7) && (abs(vertices(j,1))<1e-7) && !cellIDset){ // if at singularity, i.e. if a vertex is (1,0)
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

  if (rank==0){
    FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
    polyOrderFunction->writeValuesToMATLABFile(mesh,"polyOrder.m");
  }

  ////////////////////////////////////////////////////////////////////
  // PSEUDO-TIME SOLVE STRATEGY 
  ////////////////////////////////////////////////////////////////////

  bool useAdaptTS = false;
  if (rank==0){
    cout << "doing timesteps";
    if ((rank==0) && useAdaptTS){
      cout << " using adaptive timestepping";
    }
    cout << endl;  
  }

  // time steps
  double time_tol = 1e-8;
  for (int k = 0;k<=numRefs;k++){    

    ofstream residualFile;      
    ofstream dtFile;      
    if (rank==0){
      std::ostringstream refNum;
      refNum << k;
      string filename1 = "time_res" + refNum.str()+ ".txt";
      residualFile.open(filename1.c_str());
      //      string filename2 = "dt" + refNum.str()+ ".txt";
      //      dtFile.open(filename2.c_str());      
    }

    double L2_time_residual = 1e7;
    int i = 0;
    while(L2_time_residual > time_tol && (i<numTimeSteps)){
      for (int j = 0;j<numNRSteps;j++){
	solution->solve(false); 
	//	solution->condensedSolve(false);  
	if (mesh->numActiveElements() > 2000){
	  solution->condensedSolve(true);  // turn on save memory flag	  
	}
	// clear fluxes that we use for subsonic outflow, which accumulate
	backgroundFlow->clearSolution(That->ID());
	backgroundFlow->clearSolution(u1hat->ID());
	backgroundFlow->clearSolution(F3nhat->ID());
	backgroundFlow->clearSolution(F4nhat->ID());

	backgroundFlow->addSolution(solution,1.0); // update with dU
      }         
     
      // subtract solutions to get residual
      prevTimeFlow->addSolution(backgroundFlow,-1.0);       
      double L2rho = prevTimeFlow->L2NormOfSolutionGlobal(rho->ID());
      double L2u1 = prevTimeFlow->L2NormOfSolutionGlobal(u1->ID());
      double L2u2 = prevTimeFlow->L2NormOfSolutionGlobal(u2->ID());
      double L2T = prevTimeFlow->L2NormOfSolutionGlobal(T->ID());
      double L2_time_residual_sq = L2rho*L2rho + L2u1*L2u1 + L2u2*L2u2 + L2T*L2T;
      L2_time_residual= sqrt(L2_time_residual_sq)/dt;

      if (rank==0){
       residualFile << L2_time_residual << endl;
       //       dtFile << 1.0/((ScalarParamFunction*)invDt.get())->get_param() << endl;

       cout << "at timestep i = " << i << " with dt = " << 1.0/((ScalarParamFunction*)invDt.get())->get_param() << ", and time residual = " << L2_time_residual << endl;    	

       bool writeTimestepFiles = false;
       if (writeTimestepFiles){
	 std::ostringstream oss;
	 oss << k << "_" << i ;
	 std::ostringstream dat;
	 dat<<".dat";
	 std::ostringstream vtu;
	 vtu<<".vtu";
	 string Ustr("U_NS");      
	 solution->writeFluxesToFile(u1hat->ID(),"u1hat" +oss.str()+dat.str());
	 solution->writeFluxesToFile(u2hat->ID(),"u2hat" +oss.str()+dat.str());
	 solution->writeFluxesToFile(That->ID(), "That" +oss.str()+dat.str());
	 solution->writeFluxesToFile(F1nhat->ID(),"F1nhat"+oss.str()+dat.str() );
	 solution->writeFluxesToFile(F2nhat->ID(),"F2nhat"+oss.str()+dat.str() );
	 solution->writeFluxesToFile(F3nhat->ID(),"F3nhat"+oss.str()+dat.str() );
	 solution->writeFluxesToFile(F4nhat->ID(),"F4nhat"+oss.str()+dat.str() );
	 backgroundFlow->writeFluxesToFile(u1hat->ID(),"u1hat_prev" +oss.str()+dat.str());
	 backgroundFlow->writeFluxesToFile(u2hat->ID(),"u2hat_prev" +oss.str()+dat.str());
	 backgroundFlow->writeFluxesToFile(That->ID(), "That_prev" +oss.str()+dat.str());
	 backgroundFlow->writeFluxesToFile(F1nhat->ID(),"F1nhat_prev"+oss.str()+dat.str() );
	 backgroundFlow->writeFluxesToFile(F2nhat->ID(),"F2nhat_prev"+oss.str()+dat.str() );
	 backgroundFlow->writeFluxesToFile(F3nhat->ID(),"F3nhat_prev"+oss.str()+dat.str() );
	 backgroundFlow->writeFluxesToFile(F4nhat->ID(),"F4nhat_prev"+oss.str()+dat.str() );
	 backgroundFlow->writeToVTK(Ustr+oss.str()+vtu.str(),min(polyOrder+1,4));       
       }
      }     
      prevTimeFlow->setSolution(backgroundFlow); // reset previous time solution to current time sol

      i++;
    }
    
    // Print results from processor with rank 0
    if (rank==0){
      residualFile.close();
      //      dtFile.close();	

      std::ostringstream oss;
      oss << k ;
      std::ostringstream dat;
      dat<<".dat";
      std::ostringstream vtu;
      vtu<<".vtu";
      string Ustr("U_NS");      	
      solution->writeFluxesToFile(F1nhat->ID(),"F1nhat"+oss.str()+dat.str() );      
      solution->writeFluxesToFile(u1hat->ID(),"u1hat" +oss.str()+dat.str());
      solution->writeFluxesToFile(u2hat->ID(),"u2hat" +oss.str()+dat.str());
      solution->writeFluxesToFile(That->ID(), "That" +oss.str()+dat.str());   
      solution->writeFluxesToFile(F4nhat->ID(),"F4nhat"+oss.str()+dat.str());
      backgroundFlow->writeToVTK(Ustr+oss.str()+vtu.str(),min(polyOrder+1,4));
    }

    // get entropy
    FunctionPtr rhoToTheGamma = Teuchos::rcp(new PowerFunction(rho_prev,GAMMA));
    FunctionPtr p_prev = (GAMMA-1.0)*rho_prev*cv*T_prev;
    FunctionPtr s = Teuchos::rcp(new LogFunction(p_prev/rhoToTheGamma));
    FunctionPtr H = rho_prev*s; // entropy functional
    FunctionPtr Hsq = H*H; // entropy functional sq
    //    FunctionPtr Hnorm = Teuchos::rcp(new NormSqOverElement(H,mesh));

    // compute energy error and plot
    map<int, double> energyErrorMap = solution->energyError();
    if (rank==0){
      std::ostringstream refNum;
      refNum << k;
      std::ostringstream mfile;
      mfile<<".m";
      FunctionPtr energyErrorFunction = Teuchos::rcp( new EnergyErrorFunction(energyErrorMap) );
      energyErrorFunction->writeValuesToMATLABFile(mesh,"energyError"+refNum.str()+mfile.str());
      H->writeValuesToMATLABFile(mesh,"entropy"+refNum.str()+mfile.str());
      Hsq->writeValuesToMATLABFile(mesh,"entropySq"+refNum.str()+mfile.str());
    }

    if (k<numRefs){
      if (rank==0){
	cout << "Performing refinement number " << k << endl;
      }     
      //      energyError = solution->energyErrorTotal();
      refinementStrategy->refine(rank==0);    
      if (rank==0){
	cout << "Done with  refinement number " << k << endl;
      }   

      if (useCFL){
	double hmin = sqrt(meshInfo.getMinCellMeasure());
	((ScalarParamFunction*)invDt.get())->set_param(1.0/max(dtMin,CFL*hmin));
	if (rank==0){
	  cout << "minCellSize = " << hmin << ", dt = " << 1.0/((ScalarParamFunction*)invDt.get())->get_param() << endl;
	}
      }
      // RESET solution every refinement - make sure discretization error doesn't creep in
      backgroundFlow->projectOntoMesh(functionMap);
      prevTimeFlow->projectOntoMesh(functionMap);
      
    } else {
      if (rank==0){
	cout << "Finishing it off with the final solve" << endl;
      }
    }

  }


  return 0;
}
