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

double GAMMA = 1.4;
int numRefs = 12;
int numTimeSteps = 40; // max time steps
int numNRSteps = 1;
int numPreRefs = 0;

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;
typedef map< int, FunctionPtr > sparseFxnVector;    // dim = {trialID}
typedef map< int, sparseFxnVector > sparseFxnMatrix; // dim = {testID, trialID}
typedef map< int, sparseFxnMatrix > sparseFxnTensor; // dim = {spatial dim, testID, trialID}

typedef Teuchos::RCP< ElementType > ElementTypePtr;
typedef Teuchos::RCP< Element > ElementPtr;
typedef Teuchos::RCP< Mesh > MeshPtr;

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

class EpsilonScaling : public hFunction {
  double _epsilon;
  public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // h = sqrt(|K|), or measure of one side of a quad elem
    double scaling = min(_epsilon/(h*h), 1.0);
    // sqrt because it's inserted into an IP form in a symmetric fashion
    return sqrt(scaling);
  }
};

// ===================== Spatial filter boundary functions ====================

class BottomOutflow : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = ((abs(y) < tol) && (x < 0.5));
      return match;
    }
};

class RightOutflow : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = (abs(x-1.0) < tol);
      return match;
    }
};

class RampBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = ((abs(y) < tol) && (x >= 0.5));
      return match;
    }
};

class InflowBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      bool match = ((abs(x) < tol) || (abs(y-1.0) < tol));
      return match;
    }
};

// ===================== IP helper functions ====================

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
  int pToAdd = 3; // for tests
  int H1Order = polyOrder + 1;

  // define our manufactured solution or problem bilinear form:
  double Ma = 3.0;
  // double cv = 1.0 / ( GAMMA * (GAMMA - 1) * (Ma * Ma) );
  double energyThreshold = 0.2; // for mesh refinements

  if (rank==0){
    cout << "Running with polynomial order " << polyOrder << ", delta p = " << pToAdd << endl;
  }

  ////////////////////////////////////////////////////////////////////
  // DEFINE VARIABLES 
  ////////////////////////////////////////////////////////////////////

  // new-style bilinear form definition
  // traces
  VarFactory varFactory;
  // VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  // VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  // VarPtr ehat = varFactory.traceVar("\\widehat{e}");

  // fluxes
  VarPtr F1nhat = varFactory.fluxVar("\\widehat{F}_1n");
  VarPtr F2nhat = varFactory.fluxVar("\\widehat{F}_2n");
  VarPtr F3nhat = varFactory.fluxVar("\\widehat{F}_3n");
  VarPtr F4nhat = varFactory.fluxVar("\\widehat{F}_4n");

  // fields
  VarPtr u1 = varFactory.fieldVar("u_1");
  VarPtr u2 = varFactory.fieldVar("u_2");
  VarPtr rho = varFactory.fieldVar("\\rho");
  VarPtr e = varFactory.fieldVar("e");
  // VarPtr sigma11 = varFactory.fieldVar("\\sigma_{11}");
  // VarPtr sigma12 = varFactory.fieldVar("\\sigma_{12}");
  // VarPtr sigma22 = varFactory.fieldVar("\\sigma_{22}");
  // VarPtr omega = varFactory.fieldVar("\\omega");

  // test fxns
  // VarPtr tau1 = varFactory.testVar("\\tau_1",HDIV);
  // VarPtr tau2 = varFactory.testVar("\\tau_2",HDIV);
  // VarPtr tau3 = varFactory.testVar("\\tau_3",HDIV);
  VarPtr v1 = varFactory.testVar("v_1",HGRAD);
  VarPtr v2 = varFactory.testVar("v_2",HGRAD);
  VarPtr v3 = varFactory.testVar("v_3",HGRAD);
  VarPtr v4 = varFactory.testVar("v_4",HGRAD);

  BFPtr bf = Teuchos::rcp( new BF(varFactory) ); // initialize bilinear form

  ////////////////////////////////////////////////////////////////////
  // CREATE MESH 
  ////////////////////////////////////////////////////////////////////
  FieldContainer<double> meshPoints(4,2);

  meshPoints(0,0) = 0.0; // x1
  meshPoints(0,1) = 0.0; // y1
  meshPoints(1,0) = 1.0;
  meshPoints(1,1) = 0.0;
  meshPoints(2,0) = 1.0;
  meshPoints(2,1) = 1.0;
  meshPoints(3,0) = 0.0;
  meshPoints(3,1) = 1.0;  

  int horizontalCells = 2;
  int verticalCells = 2;

  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshPoints, horizontalCells, 
      verticalCells, bf, H1Order, 
      H1Order+pToAdd, false);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

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
  FunctionPtr e_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, e) );

  // linearized stresses
  // FunctionPtr sigma11_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, sigma11) );
  // FunctionPtr sigma12_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, sigma12) );
  // FunctionPtr sigma22_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, sigma22) );

  // previous timestep quantities
  FunctionPtr u1_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u1) );
  FunctionPtr u2_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u2) );
  FunctionPtr rho_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, rho) );
  FunctionPtr e_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, e) );

  // for subsonic outflow 
  // FunctionPtr u1hat_prev  = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u1hat ) );
  // FunctionPtr ehat_prev   = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, ehat  ) );
  FunctionPtr F2nhat_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, F2nhat) );
  FunctionPtr F3nhat_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, F3nhat) );
  FunctionPtr F4nhat_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, F4nhat) );

  // FunctionPtr u1hat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u1hat) );
  // FunctionPtr ehat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, ehat) );
  FunctionPtr F2nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F2nhat) );
  FunctionPtr F3nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F3nhat) );
  FunctionPtr F4nhat_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, F4nhat) );

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );

  // ==================== SET INITIAL GUESS ==========================

  double rho_free = 1.0;
  double u1_free = 1.0;
  double u2_free = -0.01;
  double e_free = (1.0+0.5*(u1_free*u1_free + u2_free*u2_free)); // TODO - check this value - from Capon paper

  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[rho->ID()] = Teuchos::rcp( new ConstantScalarFunction(rho_free) );
  functionMap[u1->ID()] = Teuchos::rcp( new ConstantScalarFunction(u1_free) );
  functionMap[u2->ID()] = Teuchos::rcp( new ConstantScalarFunction(u2_free) );
  functionMap[e->ID()] = Teuchos::rcp( new ConstantScalarFunction(e_free) );

  // everything else = 0; previous stresses sigma_ij = 0 as well
  backgroundFlow->projectOntoMesh(functionMap);
  prevTimeFlow->projectOntoMesh(functionMap);

  // ==================== END SET INITIAL GUESS ==========================

  ////////////////////////////////////////////////////////////////////
  // DEFINE PHYSICAL QUANTITIES
  ////////////////////////////////////////////////////////////////////

  double gam1 = (GAMMA-1.0);
  FunctionPtr u1sq = u1_prev*u1_prev;
  FunctionPtr u2sq = u2_prev*u2_prev;
  FunctionPtr unorm = (u1sq + u2sq);

  FunctionPtr p = rho_prev*gam1*(e_prev-0.5*unorm);

  // derivatives of p and e
  FunctionPtr dpdrho = gam1*(e_prev-0.5*unorm);
  FunctionPtr dpde = gam1*rho_prev;
  FunctionPtr dedu1 = u1_prev;
  FunctionPtr dedu2 = u2_prev;

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
  U[e->ID()] = e;
  // U[sigma11->ID()] = sigma11;
  // U[sigma12->ID()] = sigma12;
  // U[sigma22->ID()] = sigma22;

  map<int, VarPtr> V;
  V[v1->ID()] = v1;
  V[v2->ID()] = v2;
  V[v3->ID()] = v3;
  V[v4->ID()] = v4;

  map<int, VarPtr> TAU;
  // TAU[tau1->ID()] = tau1;
  // TAU[tau2->ID()] = tau2;
  // TAU[tau3->ID()] = tau3;

  // sparse Jacobians and viscous matrices
  sparseFxnMatrix A_euler; // 
  // sparseFxnMatrix A_visc; // 
  //  sparseFxnTensor eps_visc; // multiplies viscous terms (like 1/epsilon * sigma)
  // sparseFxnMatrix eps_visc; // multiplies viscous terms (like 1/epsilon * sigma)
  // sparseFxnMatrix eps_euler; // multiplies eulerian terms (like grad(u)) 

  // ========================================= CONSERVATION LAWS ====================================

  // mass conservation
  A_euler[v1->ID()][rho->ID()] = u1_prev*e1 + u2_prev*e2;
  A_euler[v1->ID()][u1->ID()] = rho_prev*e1;
  A_euler[v1->ID()][u2->ID()] = rho_prev*e2;

  // x-momentum conservation
  A_euler[v2->ID()][rho->ID()] = (u1sq + dpdrho)*e1 + (u1_prev * u2_prev)*e2;
  A_euler[v2->ID()][u1->ID()] = (2*u1_prev*rho_prev)*e1 + (u2_prev * rho_prev)*e2;
  A_euler[v2->ID()][u2->ID()] = (u1_prev * rho_prev)*e2;
  A_euler[v2->ID()][e->ID()] = dpde*e1;

  // x-momentum viscous terms
  // FunctionPtr negOne =  Teuchos::rcp( new ConstantScalarFunction(-1.0));
  // A_visc[v2->ID()][sigma11->ID()] = negOne*e1;
  // A_visc[v2->ID()][sigma12->ID()] = negOne*e2;

  // y-momentum conservation
  A_euler[v3->ID()][rho->ID()] = (u1_prev * u2_prev)*e1 + (u2sq + dpdrho)*e2;
  A_euler[v3->ID()][u1->ID()] = (u2_prev * rho_prev)*e1;
  A_euler[v3->ID()][u2->ID()] = (u1_prev * rho_prev)*e1 + (2 * u2_prev * rho_prev)*e2;
  A_euler[v3->ID()][e->ID()] = dpde*e2;

  // y-momentum viscous terms
  // A_visc[v3->ID()][sigma12->ID()] = negOne*e1;
  // A_visc[v3->ID()][sigma22->ID()] = negOne*e2;

  // energy conservation
  A_euler[v4->ID()][rho->ID()] = u1_prev*(e_prev + dpdrho)*e1 + u2_prev*(e_prev + dpdrho)*e2;
  A_euler[v4->ID()][u1->ID()]  = (e_prev*rho_prev + p)*e1;
  A_euler[v4->ID()][u2->ID()]  = (e_prev*rho_prev + p)*e2;
  A_euler[v4->ID()][e->ID()]   = u1_prev*(rho_prev+dpde)*e1 + u2_prev*(rho_prev+dpde)*e2;

  // FunctionPtr rho_wx = u1_prev * (e + dpdrho);
  // FunctionPtr u1_wx = rho_prev * e + p + u1_prev*rho_prev*dedu1;
  // FunctionPtr u2_wx = u1_prev*rho_prev*dedu2;
  // FunctionPtr T_wx = u1_prev*(dpdT + rho_prev*dedT);

  // FunctionPtr rho_wy = u2_prev * (e + dpdrho);
  // FunctionPtr u1_wy = u2_prev * rho_prev * dedu1;
  // FunctionPtr u2_wy = rho_prev * e + p + u2_prev * rho_prev * dedu2;
  // FunctionPtr T_wy = u2_prev * (dpdT + rho_prev * dedT);

  // A_euler[v4->ID()][rho->ID()] = rho_wx*e1 + rho_wy*e2;
  // A_euler[v4->ID()][u1->ID()]  = (u1_wx-sigma11_prev)*e1 + (u1_wy-sigma12_prev)*e2;
  // A_euler[v4->ID()][u2->ID()]  = (u2_wx-sigma12_prev)*e1 + (u2_wy-sigma22_prev)*e2;
  // A_euler[v4->ID()][T->ID()]   = (T_wx)*e1 + (T_wy)*e2;

  // stress portions
  // A_visc[v4->ID()][sigma11->ID()]  = -u1_prev*e1;;
  // A_visc[v4->ID()][sigma12->ID()]  = -u2_prev*e1 -u1_prev*e2;
  // A_visc[v4->ID()][sigma22->ID()]  = -u2_prev*e2;
  // A_visc[v4->ID()][q1->ID()]  = negOne*e1;
  // A_visc[v4->ID()][q2->ID()]  = negOne*e2;

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

  // sparseFxnTensor::iterator xyIt;
  // for (testIt = A_visc.begin();testIt!=A_visc.end();testIt++){
  //   int testID = testIt->first;
  //   sparseFxnVector a = testIt->second;
  //   sparseFxnVector::iterator trialIt;
  //   for (trialIt = a.begin();trialIt!=a.end();trialIt++){
  //     int trialID = trialIt->first;
  //     FunctionPtr trialWeight = trialIt->second;
  //     bf->addTerm(-trialWeight*U[trialID],V[testID]->grad());
  //   }
  // }

  // ========================================= STRESS LAWS  =========================================

  // bf->addTerm(u1hat, -tau1->dot_normal() );    
  // bf->addTerm(u2hat, -tau2->dot_normal() );
  // bf->addTerm(That, -tau3->dot_normal() );

  // FunctionPtr lambda_factor_fxn = lambda / (4.0 * mu * (mu + lambda) );
  // FunctionPtr two_mu = 2*mu; 
  // FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0));

  // // 1st stress eqn
  // eps_visc[tau1->ID()][sigma11->ID()] = (one/two_mu - lambda_factor_fxn)*e1;
  // eps_visc[tau1->ID()][sigma12->ID()] = one/two_mu*e2;
  // eps_visc[tau1->ID()][sigma22->ID()] = -lambda_factor_fxn*e1;
  // eps_visc[tau1->ID()][omega->ID()] = -one*Re*e2;

  // eps_euler[tau1->ID()][u1->ID()] = one;

  // // 2nd stress eqn
  // eps_visc[tau2->ID()][sigma11->ID()] = -lambda_factor_fxn*e2;
  // eps_visc[tau2->ID()][sigma12->ID()] = one/two_mu*e1;
  // eps_visc[tau2->ID()][sigma22->ID()] = (one/two_mu - lambda_factor_fxn)*e2;
  // eps_visc[tau2->ID()][omega->ID()] = one*Re*e1;

  // eps_euler[tau2->ID()][u2->ID()] = one;

  // // Heat stress equation
  // eps_visc[tau3->ID()][q1->ID()] = one/kappa*e1; //Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
  // eps_visc[tau3->ID()][q2->ID()] = one/kappa*e2; //Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
  // eps_euler[tau3->ID()][T->ID()] = one;

  // // Stress (Hdiv) equations 
  // for (testIt = eps_visc.begin();testIt!=eps_visc.end();testIt++){
  //   int testID = testIt->first;
  //   sparseFxnVector a = testIt->second;
  //   sparseFxnVector::iterator trialIt;
  //   for (trialIt = a.begin();trialIt!=a.end();trialIt++){
  //     int trialID = trialIt->first;
  //     FunctionPtr trialWeight = trialIt->second;
  //     bf->addTerm(trialWeight*U[trialID],TAU[testID]);
  //   }
  // }

  // // Eulerian component of stress (Hdiv) equations (positive b/c of IBP)
  // //  sparseFxnMatrix::iterator testIt;
  // for (testIt = eps_euler.begin();testIt!=eps_euler.end();testIt++){
  //   int testID = testIt->first;
  //   sparseFxnVector a = testIt->second;
  //   sparseFxnVector::iterator trialIt;
  //   for (trialIt = a.begin();trialIt!=a.end();trialIt++){
  //     int trialID = trialIt->first;
  //     FunctionPtr trialWeight = trialIt->second;
  //     bf->addTerm(trialWeight*U[trialID],TAU[testID]->div());
  //   }
  // } 

  ////////////////////////////////////////////////////////////////////
  // TIMESTEPPING TERMS
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  double dt = .5;
  FunctionPtr invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    
  if (rank==0){
    cout << "Timestep dt = " << dt << endl;
  }

  // needs prev time residual (u_t(i-1) - u_t(i))/dt
  FunctionPtr u1sq_pt = u1_prev_time*u1_prev_time;
  FunctionPtr u2sq_pt = u2_prev_time*u2_prev_time;
  FunctionPtr unorm_pt = (u1sq_pt + u2sq_pt);

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
  bf->addTerm(e_prev * rho + rho_prev * e, invDt * v4 );
  // bf->addTerm((e) * rho + (dedu1*rho_prev) * u1 + (dedu2*rho_prev) * u2 + (dedT*rho_prev) * T, invDt * v4 );
  FunctionPtr time_res_4 = (rho_prev_time * e_prev_time - rho_prev * e_prev);
  rhs->addTerm((time_res_4 * invDt) * v4);

  ////////////////////////////////////////////////////////////////////
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////
  // function to scale the squared guy by epsilon/|K| 
  // FunctionPtr ReScaling = Teuchos::rcp( new EpsilonScaling(1.0/Re) ); 
  //  FunctionPtr ReScaling = Teuchos::rcp( new ConstantScalarFunction(1.0));
  //  FunctionPtr ReDtScaling = Teuchos::rcp( new EpsilonScaling(dt/Re) ); 
  // FunctionPtr ReDtScaling = Teuchos::rcp( new ConstantScalarFunction(1.0));

  IPPtr ip = Teuchos::rcp( new IP );

  ////////////////////////////////////////////////////////////////////
  // Timestep L2 portion of V
  ////////////////////////////////////////////////////////////////////

  // rho dt term
  ip->addTerm(invDt*(v1 + u1_prev*v2 + u2_prev*v3 + e_prev*v4));
  // u1 dt term
  ip->addTerm(invDt*(rho_prev*v2));
  // u2 dt term
  ip->addTerm(invDt*(rho_prev*v3));
  // T dt term
  ip->addTerm(invDt*(rho_prev*v4));

  // bool coupleTauTestTerms = true;
  // bool coupleEpsVTestTerms = true;
  // bool coupleStreamTestTerms = true;

  ////////////////////////////////////////////////////////////////////
  // Rescaled L2 portion of TAU - has Re built into it
  ////////////////////////////////////////////////////////////////////

  // if (coupleTauTestTerms){
  //   map<int, LinearTermPtr> tauVec;
  //   initLinearTermVector(eps_visc,tauVec); // initialize to LinearTermPtrs of dimensions of eps_visc

  //   for (testIt = eps_visc.begin();testIt!=eps_visc.end();testIt++){
  //     int testID = testIt->first;
  //     sparseFxnVector a = testIt->second;
  //     sparseFxnVector::iterator trialIt;
  //     for (trialIt = a.begin();trialIt!=a.end();trialIt++){
  //       int trialID = trialIt->first;
  //       FunctionPtr trialWeight = trialIt->second;
  //       tauVec[trialID] = tauVec[trialID] + trialWeight*TAU[testID];
  //     }
  //   } 
  //   // adds dual test portion to IP
  //   map<int, LinearTermPtr>::iterator tauIt;
  //   for (tauIt = tauVec.begin();tauIt != tauVec.end();tauIt++){
  //     LinearTermPtr ipSum = tauIt->second;
  //     ip->addTerm(ReScaling*ipSum);
  //   }
  // }else{
  //   for (testIt = eps_visc.begin();testIt!=eps_visc.end();testIt++){
  //     int testID = testIt->first;
  //     sparseFxnVector a = testIt->second;
  //     sparseFxnVector::iterator trialIt;
  //     for (trialIt = a.begin();trialIt!=a.end();trialIt++){
  //       int trialID = trialIt->first;
  //       FunctionPtr trialWeight = trialIt->second;
  //       ip->addTerm(ReScaling*trialWeight*TAU[testID]);
  //     }
  //   } 
  // }
  ////////////////////////////////////////////////////////////////////
  // epsilon portion of grad V
  ////////////////////////////////////////////////////////////////////
  // FunctionPtr SqrtReInv = Teuchos::rcp(new ConstantScalarFunction(1.0/sqrt(Re)));

  // if (coupleEpsVTestTerms){
  //   map<int, LinearTermPtr> vEpsVec;
  //   initLinearTermVector(A_visc,vEpsVec); // initialize to LinearTermPtrs of dimensions of A_visc

  //   for (testIt = A_visc.begin();testIt!=A_visc.end();testIt++){
  //     int testID = testIt->first;
  //     sparseFxnVector a = testIt->second;
  //     sparseFxnVector::iterator trialIt;
  //     for (trialIt = a.begin();trialIt!=a.end();trialIt++){
  //       int trialID = trialIt->first;
  //       FunctionPtr trialWeight = trialIt->second;
  //       vEpsVec[trialID] = vEpsVec[trialID] + trialWeight*V[testID]->grad();
  //     }
  //   } 
  //   // adds dual test portion to IP
  //   map<int, LinearTermPtr>::iterator vEpsIt;
  //   for (vEpsIt = vEpsVec.begin();vEpsIt != vEpsVec.end();vEpsIt++){
  //     LinearTermPtr ipSum = vEpsIt->second;
  //     ip->addTerm(SqrtReInv*ipSum);
  //   }
  // } else {
  //   for (testIt = A_visc.begin();testIt!=A_visc.end();testIt++){
  //     int testID = testIt->first;
  //     sparseFxnVector a = testIt->second;
  //     sparseFxnVector::iterator trialIt;
  //     for (trialIt = a.begin();trialIt!=a.end();trialIt++){
  //       int trialID = trialIt->first;
  //       FunctionPtr trialWeight = trialIt->second;
  //       ip->addTerm(SqrtReInv*trialWeight*V[testID]->grad());
  //     }
  //   } 
  // }
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
    ip->addTerm(ipSum);
  }
  ////////////////////////////////////////////////////////////////////
  // rest of the test terms (easier)
  ////////////////////////////////////////////////////////////////////

  // ip->addTerm( ReScaling*v1 );
  // ip->addTerm( ReScaling*v2 );
  // ip->addTerm( ReScaling*v3 );
  // ip->addTerm( ReScaling*v4 );    

  // div remains the same (identity operator in classical variables)
  // ip->addTerm(tau1->div());
  // ip->addTerm(tau2->div());
  // ip->addTerm(tau3->div());

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
  FunctionPtr mom1_rhs = (e1 * momentum_x_1 + e2 *momentum_x_2);
  FunctionPtr mom2_rhs = (e1 * momentum_y_1 + e2 *momentum_y_2);
  // FunctionPtr mom1_rhs = (e1 * momentum_x_1 + e2 *momentum_x_2 - e1 * sigma11_prev - e2 * sigma12_prev);
  // FunctionPtr mom2_rhs = (e1 * momentum_y_1 + e2 *momentum_y_2 - e1 * sigma12_prev - e2 * sigma22_prev);
  rhs->addTerm( mom1_rhs * v2->grad());
  rhs->addTerm( mom2_rhs * v3->grad());

  // inviscid energy contributions
  FunctionPtr rho_e_p = rho_prev * e_prev + p;
  FunctionPtr energy_1 = rho_e_p * u1_prev;
  FunctionPtr energy_2 = rho_e_p * u2_prev;

  // viscous contributions
  // FunctionPtr viscousEnergy1 = sigma11_prev * u1_prev + sigma12_prev * u2_prev;
  // FunctionPtr viscousEnergy2 = sigma12_prev * u1_prev + sigma22_prev * u2_prev;
  // FunctionPtr energy_rhs = (e1 * energy_1 + e2 *energy_2 - e1 * viscousEnergy1 - e2 * viscousEnergy2);
  FunctionPtr energy_rhs = (e1 * energy_1 + e2 *energy_2);
  rhs->addTerm( energy_rhs * v4->grad());

  // stress rhs - no heat flux or omega (asym tensor) accumulated, eqns are linear in those
  // FunctionPtr sigmaTrace = -lambda_factor_fxn*(sigma11_prev + sigma22_prev);
  // FunctionPtr viscous1 = e1 * sigma11_prev/(2*mu) + e2 * sigma12_prev/(2*mu) + e1 * sigmaTrace;
  // FunctionPtr viscous2 = e1 * sigma12_prev/(2*mu) + e2 * sigma22_prev/(2*mu) + e2 * sigmaTrace;

  // rhs->addTerm(u1_prev * -tau1->div() - viscous1 * tau1);
  // rhs->addTerm(u2_prev * -tau2->div() - viscous2 * tau2);
  // rhs->addTerm(T_prev * -tau3->div());

  ////////////////////////////////////////////////////////////////////
  // DEFINE DIRICHLET BC
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowBoundary());
  SpatialFilterPtr rampBoundary = Teuchos::rcp( new RampBoundary());

  // free stream quantities for inflow
  double p_free = rho_free * gam1 * (e_free - 0.5*(u1_free*u1_free + u2_free*u2_free));

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
  // double Tscale = 1.0 + gam1*Ma*Ma/2.0; // from pj capon paper "adaptive finite element method compressible...".  Is equal to 2.8 for Mach 3 and Gamma = 1.4;

  bc->addDirichlet(F1nhat, rampBoundary, zero);
  // bc->addDirichlet(u2hat, wallBoundary, zero);
  // bc->addDirichlet(u1hat, wallBoundary, zero);
  // bc->addDirichlet(That, wallBoundary, Teuchos::rcp(new ConstantScalarFunction(T_free*Tscale))); 
  //  bc->addDirichlet(F4nhat, wallBoundary, zero); // sets heat flux = 0

  // =============================================================================================

  // symmetry BCs
  // SpatialFilterPtr freeTop = Teuchos::rcp( new FreeStreamBoundaryTop );
  // bc->addDirichlet(F1nhat, freeTop, zero);
  // bc->addDirichlet(F3nhat, freeTop, zero);
  // bc->addDirichlet(F4nhat, freeTop, zero); // sets zero y-heat flux in free stream top boundary

  // SpatialFilterPtr freeBottom = Teuchos::rcp( new FreeStreamBoundaryBottom );
  // bc->addDirichlet(F1nhat, freeBottom, zero);
  // bc->addDirichlet(F3nhat, freeBottom, zero); // sets zero y-stress in free stream bottom boundary
  // bc->addDirichlet(F4nhat, freeBottom, zero); // sets zero heat-flux in free stream bottom boundary

  // =============================================================================================

  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, bc, rhs, ip));
  //  solution->setReportTimingResults(true); // print out timing 

  bool setOutflowBC = false;
  // if (setOutflowBC){
  //   bool usePenalty = true;
  //   if (usePenalty){
  //     Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  //     SpatialFilterPtr outflow = Teuchos::rcp( new OutflowBoundary);
  //     FunctionPtr subsonicIndicator = Teuchos::rcp( new SubsonicIndicator(u1hat_prev, That_prev, GAMMA, cv) );
  //     // conditions on u_n = u_1, sigma_ns = sigma_12, q_1 flux
  //     pc->addConstraint(subsonicIndicator*u1hat == subsonicIndicator*u1hat_prev,outflow);
  //     pc->addConstraint(subsonicIndicator*F3nhat == subsonicIndicator*F3nhat_prev,outflow);
  //     pc->addConstraint(subsonicIndicator*F4nhat == subsonicIndicator*F4nhat_prev,outflow);

  //     solution->setFilter(pc);

  //   } else {
  //     SpatialFilterPtr subsonicOutflow = Teuchos::rcp( new SubsonicOutflow(u1hat_prev, That_prev, GAMMA, cv));
  //     /*
  //        bc->addDirichlet(u1hat, subsonicOutflow, u1hat_prev); // u_n
  //        bc->addDirichlet(F3nhat, subsonicOutflow, F3nhat_prev); // sigma_12
  //        bc->addDirichlet(F4nhat, subsonicOutflow, F4nhat_prev); // q_1
  //        */
  //     Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  //     pc->addConstraint(u1hat == u1hat_prev_time,subsonicOutflow);
  //     pc->addConstraint(F3nhat == F3nhat_prev_time,subsonicOutflow);
  //     pc->addConstraint(F4nhat == F4nhat_prev_time,subsonicOutflow);

  //     solution->setFilter(pc);

  //   }
  // }

  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow); // u_t(i)
  mesh->registerSolution(prevTimeFlow); // u_t(i-1)

  MeshInfo meshInfo(mesh); // gets info like cell measure, etc

  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));

  ////////////////////////////////////////////////////////////////////
  // PREREFINE THE MESH
  ////////////////////////////////////////////////////////////////////

  // double ReferenceRe = 100; // galerkin can represent Re = 10 easily on a standard 8x8 mesh, so no prerefs there
  // int numPreRefs = round(max(ceil(log10(Re/ReferenceRe)),0.0));
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
        // if (abs(vertices(j,0))<=1.0 && !cellIDset){
        if (!cellIDset){
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
    // if (rank==0){
    //   polyOrderFunction->writeValuesToMATLABFile(mesh,"polyOrder.m");
    // }

    ////////////////////////////////////////////////////////////////////
    // PSEUDO-TIME SOLVE STRATEGY 
    ////////////////////////////////////////////////////////////////////

    // bool useAdaptTS = false;
    // if (rank==0){
    //   cout << "doing timesteps";
    //   if ((rank==0) && useAdaptTS){
    //     cout << " using adaptive timestepping";
    //   }
    //   cout << endl;  
    // }

    // time steps
    double time_tol = 1e-4;
    for (int refIndex = 0; refIndex <= numRefs; refIndex++)
    {    
      double L2_time_residual = 1e7;
      int timestepCount = 0;
      int thresh = 2; // timestep threshhold to turn on adaptive timestepping
      while(L2_time_residual > time_tol && (timestepCount < numTimeSteps)){
        solution->solve(false); 
        backgroundFlow->addSolution(solution,1.0);

        // subtract solutions to get residual
        prevTimeFlow->addSolution(backgroundFlow,-1.0);       
        double L2rho = prevTimeFlow->L2NormOfSolutionGlobal(rho->ID());
        double L2u1 = prevTimeFlow->L2NormOfSolutionGlobal(u1->ID());
        double L2u2 = prevTimeFlow->L2NormOfSolutionGlobal(u2->ID());
        double L2e = prevTimeFlow->L2NormOfSolutionGlobal(e->ID());
        double L2_time_residual_sq = L2rho*L2rho + L2u1*L2u1 + L2u2*L2u2 + L2e*L2e;
        L2_time_residual= sqrt(L2_time_residual_sq)/dt;

        double prev_time_residual, prev_prev_time_residual;

        if (rank==0){
          stringstream outfile;
          outfile << "ramp" << refIndex << "_" << timestepCount;
          backgroundFlow->writeToVTK(outfile.str(), 5);
          cout << "Timestep: " << timestepCount << ", dt = " << dt << ", Time residual = " << L2_time_residual << endl;    	
        }     
        prevTimeFlow->setSolution(backgroundFlow); // reset previous time solution to current time sol

        timestepCount++;
      }

      //////////////////////////////////////////////////////////////////////////
      // Check conservation by testing against one
      //////////////////////////////////////////////////////////////////////////

      VarPtr testOne = varFactory.testVar("1", CONSTANT_SCALAR);
      // Create a fake bilinear form for the testing
      BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
      // Define our mass flux
      FunctionPtr massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, F1nhat) );
      LinearTermPtr massFluxTerm = massFlux * testOne;

      Teuchos::RCP<shards::CellTopology> quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
      DofOrderingFactory dofOrderingFactory(fakeBF);
      int fakeTestOrder = H1Order;
      DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, *quadTopoPtr);

      int testOneIndex = testOrdering->getDofIndex(testOne->ID(),0);
      vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
      map<int, double> massFluxIntegral; // cellID -> integral
      double maxMassFluxIntegral = 0.0;
      double totalMassFlux = 0.0;
      double totalAbsMassFlux = 0.0;
      for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
        ElementTypePtr elemType = *elemTypeIt;
        vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
        vector<int> cellIDs;
        for (int i=0; i<elems.size(); i++) {
          cellIDs.push_back(elems[i]->cellID());
        }
        FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
        BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh) );
        basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
        FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
        FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
        massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
        for (int i=0; i<elems.size(); i++) {
          int cellID = cellIDs[i];
          // pick out the ones for testOne:
          massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
        }
        // find the largest:
        for (int i=0; i<elems.size(); i++) {
          int cellID = cellIDs[i];
          maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
        }
        for (int i=0; i<elems.size(); i++) {
          int cellID = cellIDs[i];
          maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
          totalMassFlux += massFluxIntegral[cellID];
          totalAbsMassFlux += abs( massFluxIntegral[cellID] );
        }
      }

      // Print results from processor with rank 0
      if (rank==0){
        cout << endl;
        cout << "largest mass flux: " << maxMassFluxIntegral << endl;
        cout << "total mass flux: " << totalMassFlux << endl;
        cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
        cout << endl;
      }

      if (refIndex < numRefs){
        if (rank==0){
          cout << "Performing refinement number " << refIndex << endl;
        }     
        refinementStrategy->refine(rank==0);    
      }
    }

    return 0;
  }
