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

using namespace std;

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
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(sqrt(_epsilon)/ h, 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class OutflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = ((abs(x-2.0)<tol) && (y > 0) && (y < YTOP)) ;
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


/*
  class LineSearch {
  private:
  FunctionPtr rho_prev, rho_update;
  MeshPtr meshPtr;
  public:
  LineSearch(FunctionPtr Rho_p, FunctionPtr Rho_up, MeshPtr Mesh){
  rho_prev = Rho_p;
  rho_update = Rho_up;
  meshPtr = Mesh;
  }

  double getWeight(){
  double weight = 1.0;
  vector< ElementTypePtr > elemTypes = meshPtr->elementTypes(); // returns *all* elementTypes
  vector< ElementTypePtr >::iterator elemTypeIt;
  for (elemTypeIt = elemTypes.begin(); elemTypeIt!=elemTypes.end(); elemTypeIt++){
  ElementTypePtr elemTypePtr = (*elemTypeIt);
  vector< ElementPtr > elems = meshPtr->elementsOfTypeGlobal(elemTypePtr);          
  vector<int> cellIDs;
  for (int i = 0;i<elems.size();i++){
  cellIDs.push_back(elems[i]->cellID());
  }
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemTypePtr, meshPtr));
  // set cellIDs, physCellNodes
  basisCache->setPhysicalCellNodes(meshPtr->physicalCellNodes(elemTypePtr), cellIDs, false); // false = don't create side cache
  double elemTypeWeight = getWeightForElemType(basisCache);
  weight = min(weight,elemTypeWeight);
  }
  return weight;
  }

  double getWeightForElemType(BasisCachePtr basisCache){

  int numCells = basisCache->getPhysicalCubaturePoints().dimension(0);
  int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
  FieldContainer<double> rho(numCells,numPoints);
  FieldContainer<double> drho(numCells,numPoints);
  rho_prev->values(rho,basisCache);
  rho_update->values(drho,basisCache);
  double weight = 1.0;
  double min_rho = 1e-2;
  for (int i = 0;i<numCells;i++){
  for (int j = 0;j<numPoints;j++){	
  double w = ( min_rho - rho(i,j) ) / drho(i,j);
  if (w>0){
  // if it's positive, we need weight<1. take smallest weight over all points
  weight = min(w,weight); 
  }
  }
  }
  return weight;
  }
  };
*/

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
  
  bool useTriangles = false;
  
  FieldContainer<double> domainPoints(4,2);
  
  domainPoints(0,0) = 0.0; // x1
  domainPoints(0,1) = 0.0; // y1
  domainPoints(1,0) = 2.0;
  domainPoints(1,1) = 0.0;
  domainPoints(2,0) = 2.0;
  domainPoints(2,1) = YTOP;
  domainPoints(3,0) = 0.0;
  domainPoints(3,1) = YTOP;  
  
  int H1Order = polyOrder + 1;
  int nCells = 2;
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
  int horizontalCells = (2.0/YTOP)*nCells, verticalCells = nCells;
  
  double energyThreshold = 0.2; // for mesh refinements
  double nonlinearStepSize = 0.5;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  
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

  // to analyze polynomial order
  FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
  FunctionPtr partitions = Teuchos::rcp( new PartitionFunction(mesh) );
  
  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  SolutionPtr prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  

  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
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
  FunctionPtr u2hat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u2hat) );
  FunctionPtr F2nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F2nhat) );
  FunctionPtr F3nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F3nhat) );
  FunctionPtr F4nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F4nhat) );

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );

  // ==================== SET INITIAL GUESS ==========================

  double rho_free = 1.0;
  double u1_free = 1.0;
  double u2_free = 0.0;
  //  double T_free = (1/((GAMMA-1.0)*Ma*Ma)) * ( 1 + .5 * (GAMMA-1.0) * Ma*Ma); // TODO - check this value
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

  double beta = 2.0/3.0;
  FunctionPtr T_visc = Teuchos::rcp( new PowerFunction(T_prev/T_free, beta, 1.0/Re) );  // set 1/Re = min viscosity
  FunctionPtr mu = T_visc / Re;
  FunctionPtr lambda = -.66 * T_visc / Re;
  FunctionPtr kappa =  GAMMA * cv * mu / PRANDTL; // double check sign

  ////////////////////////////////////////////////////////////////////
  // DEFINE BILINEAR FORM
  ////////////////////////////////////////////////////////////////////

  // conservation law fluxes
  bf->addTerm(F1nhat, v1);
  bf->addTerm(F2nhat, v2);
  bf->addTerm(F3nhat, v3);
  bf->addTerm(F4nhat, v4);

  // sparse Jacobians and viscous matrices
  sparseFxnTensor A_euler; // 
  sparseFxnTensor A_visc; // 
  sparseFxnTensor eps_visc; // multiplies viscous terms (like 1/epsilon * sigma)
  sparseFxnMatrix eps_euler; // multiplies eulerian terms (like grad(u)) 

  int x_comp = 0; int y_comp = 1;
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

  // ========================================= CONSERVATION LAWS ====================================

  // mass conservation
  A_euler[x_comp][v1->ID()][rho->ID()] = u1_prev;
  A_euler[x_comp][v1->ID()][u1->ID()]  = rho_prev;
  A_euler[y_comp][v1->ID()][rho->ID()] = u2_prev;
  A_euler[y_comp][v1->ID()][u2->ID()]  = rho_prev;

  // x-momentum conservation
  A_euler[x_comp][v2->ID()][rho->ID()] = (u1sq + dpdrho);
  A_euler[x_comp][v2->ID()][u1->ID()] = (2*u1_prev*rho_prev);
  A_euler[x_comp][v2->ID()][T->ID()] = dpdT;
  A_euler[y_comp][v2->ID()][rho->ID()] =  (u1_prev * u2_prev);
  A_euler[y_comp][v2->ID()][u1->ID()] = (u2_prev * rho_prev);
  A_euler[y_comp][v2->ID()][u2->ID()] =(u1_prev * rho_prev);  
  // x-momentum viscous terms
  A_visc[x_comp][v2->ID()][sigma11->ID()] = Teuchos::rcp( new ConstantScalarFunction(-1.0));
  A_visc[y_comp][v2->ID()][sigma12->ID()] = Teuchos::rcp( new ConstantScalarFunction(-1.0));

  // y-momentum conservation
  A_euler[x_comp][v3->ID()][rho->ID()] = (u1_prev * u2_prev);
  A_euler[x_comp][v3->ID()][u1->ID()] = (u2_prev * rho_prev);
  A_euler[x_comp][v3->ID()][u2->ID()] = (u1_prev * rho_prev);
  A_euler[y_comp][v3->ID()][rho->ID()] = (u2sq + dpdrho);
  A_euler[y_comp][v3->ID()][u1->ID()] = (2 * u2_prev * rho_prev);
  A_euler[y_comp][v3->ID()][T->ID()] = dpdT;
  // y-momentum viscous terms
  A_visc[x_comp][v3->ID()][sigma12->ID()] = Teuchos::rcp( new ConstantScalarFunction(-1.0));
  A_visc[y_comp][v3->ID()][sigma22->ID()] = Teuchos::rcp( new ConstantScalarFunction(-1.0));

  // energy conservation
  FunctionPtr rho_wx = u1_prev * (e + dpdrho);
  FunctionPtr u1_wx = rho_prev * e + p + u1_prev*rho_prev*dedu1;
  FunctionPtr u2_wx = u1_prev*rho_prev*dedu2;
  FunctionPtr T_wx = u1_prev*(dpdT + rho_prev*dedT);

  FunctionPtr rho_wy = u2_prev * (e + dpdrho);
  FunctionPtr u1_wy = u2_prev * rho_prev * dedu1;
  FunctionPtr u2_wy = rho_prev * e + p + u2_prev * rho_prev * dedu2;
  FunctionPtr T_wy = u2_prev * (dpdT + rho_prev * dedT);

  A_euler[x_comp][v4->ID()][rho->ID()] = rho_wx;
  A_euler[x_comp][v4->ID()][u1->ID()] = u1_wx-sigma11_prev; 
  A_euler[x_comp][v4->ID()][u2->ID()] = u2_wx-sigma12_prev;;
  A_euler[x_comp][v4->ID()][T->ID()]  = T_wx;

  A_euler[y_comp][v4->ID()][rho->ID()] = rho_wy;
  A_euler[y_comp][v4->ID()][u1->ID()] = u1_wy-sigma12_prev;
  A_euler[y_comp][v4->ID()][u2->ID()] = u2_wy-sigma22_prev;;
  A_euler[y_comp][v4->ID()][T->ID()]  = T_wy;

  // stress portions
  A_visc[x_comp][v4->ID()][sigma11->ID()]  = -u1_prev;
  A_visc[x_comp][v4->ID()][sigma12->ID()]  = -u2_prev;
  A_visc[x_comp][v4->ID()][q1->ID()]  = Teuchos::rcp( new ConstantScalarFunction(-1.0));

  A_visc[y_comp][v4->ID()][sigma12->ID()]  = -u1_prev;
  A_visc[y_comp][v4->ID()][sigma22->ID()]  = -u2_prev;
  A_visc[y_comp][v4->ID()][q2->ID()]  = Teuchos::rcp( new ConstantScalarFunction(-1.0));

  // conservation (Hgrad) equations
  sparseFxnTensor::iterator xyIt;
  for (xyIt = A_euler.begin();xyIt!=A_euler.end();xyIt++){
    int component = xyIt->first;
    sparseFxnMatrix A = xyIt->second;
    sparseFxnMatrix::iterator testIt;
    for (testIt = A.begin();testIt!=A.end();testIt++){
      int testID = testIt->first;
      sparseFxnVector a = testIt->second;
      sparseFxnVector::iterator trialIt;
      for (trialIt = a.begin();trialIt!=a.end();trialIt++){
	int trialID = trialIt->first;
	FunctionPtr trialWeight = trialIt->second;
	if (component==x_comp){
	  bf->addTerm(trialWeight*U[trialID],-V[testID]->dx());
	}else if (component==y_comp){
	  bf->addTerm(trialWeight*U[trialID],-V[testID]->dy());
	}
      }
    }
  }
  // conservation (Hgrad) equations - viscous terms
  for (xyIt = A_visc.begin();xyIt!=A_visc.end();xyIt++){
    int component = xyIt->first;
    sparseFxnMatrix A = xyIt->second;
    sparseFxnMatrix::iterator testIt;
    for (testIt = A.begin();testIt!=A.end();testIt++){
      int testID = testIt->first;
      sparseFxnVector a = testIt->second;
      sparseFxnVector::iterator trialIt;
      for (trialIt = a.begin();trialIt!=a.end();trialIt++){
	int trialID = trialIt->first;
	FunctionPtr trialWeight = trialIt->second;
	if (component==x_comp){
	  bf->addTerm(trialWeight*U[trialID],-V[testID]->dx());
	}else if (component==y_comp){
	  bf->addTerm(trialWeight*U[trialID],-V[testID]->dy());
	}
      }
    }
  }
  
  // ========================================= STRESS LAWS  =========================================

  bf->addTerm(u1hat, -tau1->dot_normal() );    
  bf->addTerm(u2hat, -tau2->dot_normal() );
  bf->addTerm(That, -tau3->dot_normal() );

  FunctionPtr lambda_factor_fxn = lambda / (4.0 * mu * (mu + lambda) );
  FunctionPtr two_mu = 2*mu; 
  //  FunctionPtr lambda_factor_fxn = Teuchos::rcp(new ConstantScalarFunction(lambda_factor));
  //  FunctionPtr two_mu = Teuchos::rcp(new ConstantScalarFunction(2*mu));
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0));

  // 1st stress eqn
  eps_visc[x_comp][tau1->ID()][sigma11->ID()] = one/two_mu - lambda_factor_fxn;
  eps_visc[x_comp][tau1->ID()][sigma22->ID()] = -lambda_factor_fxn;

  eps_visc[y_comp][tau1->ID()][sigma12->ID()] = one/two_mu;
  eps_visc[y_comp][tau1->ID()][omega->ID()] = -one;
  
  eps_euler[tau1->ID()][u1->ID()] = one;
  
  // 2nd stress eqn
  eps_visc[x_comp][tau2->ID()][sigma12->ID()] = one/two_mu;
  eps_visc[x_comp][tau2->ID()][omega->ID()] = one;

  eps_visc[y_comp][tau2->ID()][sigma11->ID()] = -lambda_factor_fxn;
  eps_visc[y_comp][tau2->ID()][sigma22->ID()] = one/two_mu - lambda_factor_fxn;

  eps_euler[tau2->ID()][u2->ID()] = one;

  // Heat stress equation
  eps_visc[x_comp][tau3->ID()][q1->ID()] = one/kappa; //Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
  eps_visc[y_comp][tau3->ID()][q2->ID()] = one/kappa; //Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
  eps_euler[tau3->ID()][T->ID()] = one;
  
  // Stress (Hdiv) equations 
  for (xyIt = eps_visc.begin();xyIt!=eps_visc.end();xyIt++){
    int component = xyIt->first;
    sparseFxnMatrix A = xyIt->second;
    sparseFxnMatrix::iterator testIt;
    for (testIt = A.begin();testIt!=A.end();testIt++){
      int testID = testIt->first;
      sparseFxnVector a = testIt->second;
      sparseFxnVector::iterator trialIt;
      for (trialIt = a.begin();trialIt!=a.end();trialIt++){
	int trialID = trialIt->first;
	FunctionPtr trialWeight = trialIt->second;
	if (component==x_comp){
	  bf->addTerm(trialWeight*U[trialID],TAU[testID]->x());
	} else if (component==y_comp){
	  bf->addTerm(trialWeight*U[trialID],TAU[testID]->y());
	}
      }
    }
  }

  // Eulerian component of stress (Hdiv) equations (positive b/c of IBP)
  sparseFxnMatrix::iterator testIt;
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
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////
  // function to scale the squared guy by epsilon/h
  //  FunctionPtr epsilonOverHScaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
  
  FunctionPtr ReScaling = Teuchos::rcp( new EpsilonScaling(Re) ); 

  sparseFxnTensor visc; // rescaled viscous 
  visc[x_comp][tau1->ID()][sigma11->ID()] = eps_visc[x_comp][tau1->ID()][sigma11->ID()]*ReScaling;
  visc[x_comp][tau1->ID()][sigma22->ID()] = eps_visc[x_comp][tau1->ID()][sigma22->ID()]*ReScaling;
  visc[y_comp][tau1->ID()][sigma12->ID()] = eps_visc[y_comp][tau1->ID()][sigma12->ID()]*ReScaling;
  
  // 2nd stress eqn
  visc[x_comp][tau2->ID()][sigma12->ID()] = eps_visc[x_comp][tau2->ID()][sigma12->ID()]*ReScaling;
  visc[y_comp][tau2->ID()][sigma11->ID()] = eps_visc[y_comp][tau2->ID()][sigma11->ID()]*ReScaling;
  visc[y_comp][tau2->ID()][sigma22->ID()] = eps_visc[y_comp][tau2->ID()][sigma22->ID()]*ReScaling;

  // Heat stress equation
  visc[x_comp][tau3->ID()][q1->ID()] = eps_visc[x_comp][tau3->ID()][q1->ID()]*ReScaling; // O(Re)
  visc[y_comp][tau3->ID()][q2->ID()] = eps_visc[y_comp][tau3->ID()][q2->ID()]*ReScaling; // O(Re)

  IPPtr ip = Teuchos::rcp( new IP );
    
  // Rescaled L2 portion of TAU - has Re built into it
  for (xyIt = visc.begin();xyIt!=visc.end();xyIt++){
    int component = xyIt->first;
    sparseFxnMatrix A = xyIt->second;
    sparseFxnMatrix::iterator testIt;
    for (testIt = A.begin();testIt!=A.end();testIt++){
      int testID = testIt->first;
      sparseFxnVector a = testIt->second;
      sparseFxnVector::iterator trialIt;
      for (trialIt = a.begin();trialIt!=a.end();trialIt++){
	int trialID = trialIt->first;
	FunctionPtr trialWeight = trialIt->second;
	if (component==x_comp){
	  ip->addTerm(trialWeight*TAU[testID]->x());
	} else if (component==y_comp){
	  ip->addTerm(trialWeight*TAU[testID]->y());
	}
      }
    }
  }

  // epsilon portion of grad V
  for (xyIt = A_visc.begin();xyIt!=A_visc.end();xyIt++){
    int component = xyIt->first;
    sparseFxnMatrix A = xyIt->second;
    sparseFxnMatrix::iterator testIt;
    for (testIt = A.begin();testIt!=A.end();testIt++){
      int testID = testIt->first;
      sparseFxnVector a = testIt->second;
      sparseFxnVector::iterator trialIt;
      for (trialIt = a.begin();trialIt!=a.end();trialIt++){
	int trialID = trialIt->first;
	FunctionPtr trialWeight = trialIt->second;
	if (component==x_comp){
	  ip->addTerm(trialWeight/sqrt(Re)*V[testID]->dx());
	} else if (component==y_comp){
	  ip->addTerm(trialWeight/sqrt(Re)*V[testID]->dy());
	}
      }
    }
  }

  // "streamline" portion of grad V
  for (xyIt = A_euler.begin();xyIt!=A_euler.end();xyIt++){
    int component = xyIt->first;
    sparseFxnMatrix A = xyIt->second;
    sparseFxnMatrix::iterator testIt;
    for (testIt = A.begin();testIt!=A.end();testIt++){
      int testID = testIt->first;
      sparseFxnVector a = testIt->second;
      sparseFxnVector::iterator trialIt;
      for (trialIt = a.begin();trialIt!=a.end();trialIt++){
	int trialID = trialIt->first;
	FunctionPtr trialWeight = trialIt->second;
	if (component==x_comp){
	  ip->addTerm(trialWeight*V[testID]->dx());
	} else if (component==y_comp){
	  ip->addTerm(trialWeight*V[testID]->dy());
	}
      }
    }
  }

  ip->addTerm( ReScaling*v1 );
  ip->addTerm( ReScaling*v2 );
  ip->addTerm( ReScaling*v3 );
  ip->addTerm( ReScaling*v4 );    
 
  // div remains the same (identity operator in classical variables)
  ip->addTerm(tau1->div());
  ip->addTerm(tau2->div());
  ip->addTerm(tau3->div());

  /*
  ip->addTerm(tau1);
  ip->addTerm(tau2);
  ip->addTerm(tau3);

  ip->addTerm(v1);
  ip->addTerm(v2);
  ip->addTerm(v3);
  ip->addTerm(v4);
  ip->addTerm(v1->grad());
  ip->addTerm(v2->grad());
  ip->addTerm(v3->grad());
  ip->addTerm(v4->grad());
  */

  //  ////////////////////////////////////////////////////////////////////
  //  // DEFINE RHS
  //  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  // mass contributions
  FunctionPtr mass_1 = rho_prev*u1_prev;
  FunctionPtr mass_2 = rho_prev*u2_prev;

  rhs->addTerm( (e1 * mass_1 + e2 *mass_2) * v1->grad());

  // inviscid momentum contributions
  FunctionPtr momentum_x_1 = rho_prev * u1sq + p ;
  FunctionPtr momentum_x_2 = rho_prev * u1_prev * u2_prev ;
  FunctionPtr momentum_y_1 = rho_prev * u1_prev * u2_prev ;
  FunctionPtr momentum_y_2 = rho_prev * u2sq + p ;

  rhs->addTerm( (e1 * momentum_x_1 + e2 *momentum_x_2 - e1 * sigma11_prev - e2 * sigma12_prev) * v2->grad());
  rhs->addTerm( (e1 * momentum_y_1 + e2 *momentum_y_2 - e1 * sigma12_prev - e2 * sigma22_prev) * v3->grad());

  // inviscid energy contributions
  FunctionPtr rho_e_p = rho_prev * e + p;
  FunctionPtr energy_1 = rho_e_p * u1_prev;
  FunctionPtr energy_2 = rho_e_p * u2_prev;

  // viscous contributions
  FunctionPtr viscousEnergy1 = sigma11_prev * u1_prev + sigma12_prev * u2_prev;
  FunctionPtr viscousEnergy2 = sigma12_prev * u1_prev + sigma22_prev * u2_prev;
    
  rhs->addTerm( (e1 * energy_1 + e2 *energy_2 - e1 * viscousEnergy1 - e2 * viscousEnergy2) * v4->grad());

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
  bc->addDirichlet(u2hat, wallBoundary, zero);
  bc->addDirichlet(u1hat, wallBoundary, zero);
  double Tscale = 1.0 + gam1*Ma*Ma/2.0; // from pj capon paper "adaptive finite element method compressible...".  Is equal to 2.8 for Mach 3 and Gamma = 1.4;
  //  bc->addDirichlet(That, wallBoundary, Teuchos::rcp(new ConstantScalarFunction(T_free*Tscale))); 
  bc->addDirichlet(F4nhat, wallBoundary, zero); // sets heat flux = 0

  // =============================================================================================

  // symmetry BCs
  SpatialFilterPtr freeTop = Teuchos::rcp( new FreeStreamBoundaryTop );
  bc->addDirichlet(u2hat, freeTop, Teuchos::rcp( new ConstantScalarFunction(0.0))); // top sym bc
  bc->addDirichlet(F2nhat, freeTop, zero);
  bc->addDirichlet(F4nhat, freeTop, zero); // sets zero y-heat flux in free stream top boundary

  // =============================================================================================

  SpatialFilterPtr freeBottom = Teuchos::rcp( new FreeStreamBoundaryBottom );
  bc->addDirichlet(u2hat, freeBottom, Teuchos::rcp( new ConstantScalarFunction(0.0))); // sym bcs
  bc->addDirichlet(F2nhat, freeBottom, zero); // sets zero y-stress in free stream bottom boundary
  bc->addDirichlet(F4nhat, freeBottom, zero); // sets zero heat-flux in free stream bottom boundary

  // =============================================================================================

  /*
    SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowBoundary );
    FunctionPtr F2n_for_sigma = ( e1 * momentum_x_1 + e2 * momentum_x_2) * n; 
    FunctionPtr F3n_for_sigma = ( e1 * momentum_y_1 + e2 * momentum_y_2) * n; 
    FunctionPtr F4n_for_qn = (e1 * energy_1 + e2 *energy_2 + e1 * viscousEnergy1 + e2 * viscousEnergy2)*n; // makes q_n implicitly 0 b/c u2 = 0, sigma12 = 0, and n = (1,0)

    bc->addDirichlet(F2nhat, outflowBoundary, F2n_for_sigma);
    bc->addDirichlet(F3nhat, outflowBoundary, F3n_for_sigma);
    bc->addDirichlet(F4nhat, outflowBoundary, F4n_for_qn);
  */
  
  //  bc->addDirichlet(F2nhat, outflowBoundary, Teuchos::rcp(new ConstantScalarFunction(1.0)));

  // =============================================================================================

  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, bc, rhs, ip));
  //  solution->setReportTimingResults(true); // print out timing 

  /*
    // for use in line search enforcing rho > 0
    FunctionPtr rho_update = Teuchos::rcp( new PreviousSolutionFunction(solution, rho ));
    Teuchos::RCP<LineSearch> lineSearch;
    lineSearch = Teuchos::rcp(new LineSearch(rho_prev, rho_update, mesh));
  */

  // enforce local conservation of fluxes  
  bool enforceLocalConservation = false;
  if (enforceLocalConservation){
    solution->lagrangeConstraints()->addConstraint(F1nhat == zero);  
    solution->lagrangeConstraints()->addConstraint(F2nhat == zero);
    solution->lagrangeConstraints()->addConstraint(F3nhat == zero);
    solution->lagrangeConstraints()->addConstraint(F4nhat == zero);
  }

  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow); // u_t(i)
  mesh->registerSolution(prevTimeFlow); // u_t(i-1)
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE REFINEMENT STRATEGY
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));

  int numTimeSteps = 150; // max time steps
  int numNRSteps = 1;
  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy;
  solveStrategy = Teuchos::rcp( new NonlinearSolveStrategy(backgroundFlow, solution, stepSize,
                                                           nonlinearRelativeEnergyTolerance));
  
  ////////////////////////////////////////////////////////////////////
  // SOLVE 
  ////////////////////////////////////////////////////////////////////
  bool useAdaptiveTimesteps = false;
  double dt = .1;
  //  FunctionPtr Dt = Teuchos::rcp(new ScalarParamFunction(dt));    
  FunctionPtr invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    
  if (numTimeSteps>0){
    if (rank==0){
      cout << "Timestep dt = " << dt << endl;
    }

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
  }

  // time step 1
  // prerefine the mesh
  /*
  for (int refIndex=0;refIndex<1;refIndex++){    
    solution->solve(); // false: don't use MUMPS
    refinementStrategy->refine(rank==0); // print to console on rank 0	
  }
  solution->solve(); // false: don't use MUMPS
  */

  if (rank==0){

    // UN_minus_C->writeValuesToMATLABFile(solution->mesh(), "un_c.m");
    solution->writeFluxesToFile(u1hat->ID(), "u1hat.dat");
    solution->writeFluxesToFile(u2hat->ID(), "u2hat.dat");
    solution->writeFluxesToFile(That->ID(), "That.dat");

    solution->writeFluxesToFile(F1nhat->ID(), "F1nhat.dat");
    solution->writeFluxesToFile(F2nhat->ID(), "F2nhat.dat");
    solution->writeFluxesToFile(F3nhat->ID(), "F3nhat.dat");
    solution->writeFluxesToFile(F4nhat->ID(), "F4nhat.dat");

    solution->writeFieldsToFile(u1->ID(), "u1.m");
    solution->writeFieldsToFile(u2->ID(), "u2.m");
    solution->writeFieldsToFile(rho->ID(), "rho.m");
    solution->writeFieldsToFile(T->ID(), "T.m");

    solution->writeFieldsToFile(sigma11->ID(), "sigma11.m");
    solution->writeFieldsToFile(sigma12->ID(), "sigma12.m");
    solution->writeFieldsToFile(sigma22->ID(), "sigma22.m");
    solution->writeFieldsToFile(q1->ID(), "q1.m");
    solution->writeFieldsToFile(q2->ID(), "q2.m");
    solution->writeFieldsToFile(omega->ID(), "w.m");    
  } 

  if (rank==0){
    cout << "doing timesteps" << endl;
  }
  // time steps
  double time_tol = 5e-7;
  for (int k = 0;k<numRefs;k++){
    double L2_time_residual = 1e7;
    int i = 0;
    while(L2_time_residual > time_tol && (i<numTimeSteps)){
      //  for (int i = 0;i<numTimeSteps;i++){
      for (int j = 0;j<numNRSteps;j++){
	solution->solve(false); 
	backgroundFlow->addSolution(solution,1.0);
      }         

      prevTimeFlow->addSolution(backgroundFlow,-1.0); 

      if (useAdaptiveTimesteps){
	double inf_rho = prevTimeFlow->InfNormOfSolutionGlobal(rho->ID());
	double inf_u1 = prevTimeFlow->InfNormOfSolutionGlobal(u1->ID());
	double inf_u2 = prevTimeFlow->InfNormOfSolutionGlobal(u2->ID());
	double inf_T = prevTimeFlow->InfNormOfSolutionGlobal(T->ID());
	double inf_time_residual = max(max(inf_rho,inf_u1),max(inf_u2,inf_T));
	double first_step_residual, residual_ratio;
	int init_ts = 10; // give a few timesteps to stabilize
	if (i>=init_ts){
	  if (i==init_ts){
	    first_step_residual = inf_time_residual;
	  }else{
	    double maxDt = 2.5;
	    double minDt = 1e-2;
	    residual_ratio = inf_time_residual/first_step_residual;
	    // cout << "residual ratio = " << residual_ratio << endl;
	    dt /= residual_ratio;
	    dt = min(dt,maxDt);
	    dt = max(dt,minDt);
	  }     
	}
	((ScalarParamFunction*)invDt.get())->set_param(1.0/dt);     
      }

      double L2rho = prevTimeFlow->L2NormOfSolutionGlobal(rho->ID());
      double L2u1 = prevTimeFlow->L2NormOfSolutionGlobal(u1->ID());
      double L2u2 = prevTimeFlow->L2NormOfSolutionGlobal(u2->ID());
      double L2T = prevTimeFlow->L2NormOfSolutionGlobal(T->ID());
      double L2_time_residual_sq = L2rho*L2rho + L2u1*L2u1 + L2u2*L2u2 + L2T*L2T;
      L2_time_residual= sqrt(L2_time_residual_sq);
   
      if (rank==0){
	cout << "at timestep i = " << i << " with dt = " << dt << ", and time residual = " << L2_time_residual << endl;
      }
      prevTimeFlow->setSolution(backgroundFlow); 
      i++;
    }
    refinementStrategy->refine(rank==0);
  }

  if (numTimeSteps==0){
    for (int j = 0;j<numNRSteps;j++){
      solution->solve(false); 
      if (j<5){
	backgroundFlow->addSolution(solution,.2);
      }else{
	backgroundFlow->addSolution(solution,.5);
      }
      
      cout << "nr iter " << j << endl;
    }
  }
  if (rank==0){
    cout << "finishing it off with final solve" << endl;
  }
  solution->solve(false); 
  backgroundFlow->addSolution(solution,1.0);
  if (rank==0){
    cout << "writing solutions to file" << endl;
  }
  if (rank==0){    
    unorm->writeValuesToMATLABFile(solution->mesh(),"unorm.m");
    T_visc->writeValuesToMATLABFile(solution->mesh(),"T_visc.m");
    solution->writeFluxesToFile(u1hat->ID(), "u1hat2.dat");
    solution->writeFluxesToFile(u2hat->ID(), "u2hat2.dat");
    solution->writeFluxesToFile(That->ID(), "That2.dat");
    solution->writeFluxesToFile(F1nhat->ID(), "F1nhat2.dat");
    solution->writeFluxesToFile(F2nhat->ID(), "F2nhat2.dat");
    solution->writeFluxesToFile(F3nhat->ID(), "F3nhat2.dat");
    solution->writeFluxesToFile(F4nhat->ID(), "F4nhat2.dat");

    solution->writeFieldsToFile(rho->ID(),"drho2.m");
    backgroundFlow->writeFieldsToFile(u1->ID(), "u12.m");
    backgroundFlow->writeFieldsToFile(u2->ID(), "u22.m");
    backgroundFlow->writeFieldsToFile(rho->ID(), "rho2.m");
    backgroundFlow->writeFieldsToFile(T->ID(), "T2.m");

    backgroundFlow->writeFieldsToFile(sigma11->ID(), "sigma112.m");
    backgroundFlow->writeFieldsToFile(sigma12->ID(), "sigma122.m");
    backgroundFlow->writeFieldsToFile(sigma22->ID(), "sigma222.m");
    solution->writeFieldsToFile(q1->ID(), "q12.m");
    solution->writeFieldsToFile(q2->ID(), "q22.m");
    solution->writeFieldsToFile(omega->ID(), "w2.m");    

    //    polyOrderFunction->writeValuesToMATLABFile(mesh, "polyOrders.m");
    //    partitions->writeValuesToMATLABFile(mesh,"partitions.m");
  } 

  return 0;
}
