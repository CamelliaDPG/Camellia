#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"
#include "ZoltanMeshPartitionPolicy.h"

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

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;
typedef map< int, FunctionPtr > sparseFxnVector;    // dim = {trialID}
typedef map< int, sparseFxnVector > sparseFxnMatrix; // dim = {testID, trialID}
typedef map< int, sparseFxnMatrix > sparseFxnTensor; // dim = {spatial dim, testID, trialID}

static const double GAMMA = 1.4;
static const double PRANDTL = 0.72;
static const double YTOP = 2.0;

using namespace std;

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

class InflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = (abs(x) < tol);
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
  /*
  bool matchesPoints(FieldContainer<bool> pointsMatch, BasisCachePtr basisCache){
    double tol = 1e-14;
    FieldContainer<double> pts = basisCache->getPhysicalCubaturePoints();    
  }
  */
};

class FreeStreamBoundaryBottom : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = ((abs(y) < tol) && (x <= 1.0) && (x > 0.0));
    return yMatch;
  }
};

class FreeStreamBoundaryTop : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = (abs(y-YTOP) < tol && (x < 2.0) && (x > 0.0));
    return yMatch;
  }
};

/*
class SubsonicOutflowBoundary : public SpatialFilter {
private: 
  FunctionPtr un, c;
public:
  SubsonicOutflowBoundary(UN,C){
    un = UN;
    c = C;
  }  
  bool matchesPoints(FieldContainer<bool> pointsMatch, BasisCachePtr basisCache){
    double tol = 1e-14;
    FieldContainer<double> pts = basisCache->getPhysicalCubaturePoints();
    numPts = pts.dimension(0);
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
  double Re = 1e2;
  double Ma = 3.0;
  double cv = 1.0 / ( GAMMA * (GAMMA - 1) * (Ma * Ma) );
  double mu = 1.0 / Re;
  double lambda = -.66 / Re;
  double kappa =  GAMMA * cv * mu / PRANDTL; // double check sign
  
  bool useTriangles = false;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 2.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 2.0;
  quadPoints(2,1) = YTOP;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = YTOP;  
  
  int H1Order = polyOrder + 1;
  int nCells = 1;
  int horizontalCells = (2.0/YTOP)*nCells, verticalCells = nCells;
  
  double energyThreshold = 0.25; // for mesh refinements
  double nonlinearStepSize = 0.5;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE VARIABLES 
  ////////////////////////////////////////////////////////////////////
  
  // new-style bilinear form definition
  VarFactory varFactory;
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr That = varFactory.traceVar("\\widehat{T}");
  
  VarPtr F1nhat = varFactory.fluxVar("\\widehat{F}_1n");
  VarPtr F2nhat = varFactory.fluxVar("\\widehat{F}_2n");
  VarPtr F3nhat = varFactory.fluxVar("\\widehat{F}_3n");
  VarPtr F4nhat = varFactory.fluxVar("\\widehat{F}_4n");
  
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
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, 
                                                verticalCells, bf, H1Order, 
                                                H1Order+pToAdd, useTriangles);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));
  
  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, 
                                                         nullRHS, nullIP) );  
  SolutionPtr prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, 
						       nullRHS, nullIP) );  

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

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );

  // ==================== SET INITIAL GUESS ==========================

  mesh->registerSolution(backgroundFlow); // u_t(i)
  mesh->registerSolution(prevTimeFlow); // u_t(i-1)

  double rho_free = 1.0;
  double u1_free = 1.0;
  double u2_free = 0.0;
  double T_free = (1/((GAMMA-1.0)*Ma*Ma)) * ( 1 + .5 * (GAMMA-1.0) * Ma*Ma);

  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[rho->ID()] = Teuchos::rcp( new ConstantScalarFunction(rho_free) );
  functionMap[u1->ID()] = Teuchos::rcp( new ConstantScalarFunction(u1_free) );
  functionMap[u2->ID()] = Teuchos::rcp( new ConstantScalarFunction(u2_free) );
  functionMap[T->ID()] = Teuchos::rcp( new ConstantScalarFunction(T_free) );
  //  functionMap[T->ID()] = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  functionMap[sigma11->ID()] = zero;
  functionMap[sigma12->ID()] = zero;
  functionMap[sigma22->ID()] = zero;
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
  FunctionPtr p = (gam1 * cv) * rho_prev * T_prev;
  FunctionPtr u1sq = u1_prev*u1_prev;
  FunctionPtr u2sq = u2_prev*u2_prev;
  FunctionPtr iota = cv*T_prev; // internal energy
  FunctionPtr unorm = (u1sq + u2sq);
  FunctionPtr e = .5*unorm + iota; // kinetic + internal energy
  
  // derivatives of p and e
  FunctionPtr dpdrho = cv*T_prev*gam1;
  FunctionPtr dpdT = cv*rho_prev*gam1;
  FunctionPtr dedu1 = u1_prev;
  FunctionPtr dedu2 = u2_prev;
  double dedT = cv * GAMMA;

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

  double lambda_factor = lambda / (4.0 * mu * (mu + lambda) );

  FunctionPtr lambda_factor_fxn = Teuchos::rcp(new ConstantScalarFunction(lambda_factor));
  FunctionPtr two_mu = Teuchos::rcp(new ConstantScalarFunction(2*mu));
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
  eps_visc[x_comp][tau3->ID()][q1->ID()] = Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
  eps_visc[y_comp][tau3->ID()][q2->ID()] = Teuchos::rcp(new ConstantScalarFunction(1.0/kappa));
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

  // Eulerian component of stress (Hdiv) equations 
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

  bf->addTerm(u1hat, -tau1->dot_normal() );    
  bf->addTerm(u2hat, -tau2->dot_normal() );
  bf->addTerm(That, -tau3->dot_normal() );
  
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

  // Rescaled L2 portion of TAU
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
	}  else if (component==y_comp){
	  ip->addTerm(trialWeight*V[testID]->dy());
	}
      }
    }
  }

  // div remains the same (identity operator in classical variables)
  ip->addTerm(tau1->div());
  ip->addTerm(tau2->div());
  ip->addTerm(tau3->div());

  ip->addTerm( ReScaling*v1 );
  ip->addTerm( ReScaling*v2 );
  ip->addTerm( ReScaling*v3 );
  ip->addTerm( ReScaling*v4 );  


//  ////////////////////////////////////////////////////////////////////
//  // DEFINE RHS
//  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  // mass contributions
  FunctionPtr mass_1 = rho_prev*u1_prev;
  FunctionPtr mass_2 = rho_prev*u2_prev;

  // inviscid momentum contributions
  FunctionPtr momentum_x_1 = rho_prev * u1sq + p ;
  FunctionPtr momentum_x_2 = rho_prev * u1_prev * u2_prev ;
  FunctionPtr momentum_y_1 = rho_prev * u1_prev * u2_prev ;
  FunctionPtr momentum_y_2 = rho_prev * u2sq + p ;

  // inviscid energy contributions
  FunctionPtr rho_e_p = rho_prev * e + p;
  FunctionPtr energy_1 = rho_e_p * u1_prev;
  FunctionPtr energy_2 = rho_e_p * u2_prev;

  // viscous contributions
  FunctionPtr viscousEnergy1 = sigma11_prev * u1_prev + sigma12_prev * u2_prev;
  FunctionPtr viscousEnergy2 = sigma12_prev * u1_prev + sigma22_prev * u2_prev;
    
  FunctionPtr sigmaTrace = sigma11_prev + sigma22_prev;
  FunctionPtr sig11lambda = - lambda_factor * sigma11_prev;
  FunctionPtr sig22lambda = - lambda_factor * sigma22_prev;
  FunctionPtr viscous1 = e1 * sigma11_prev/(2*mu) + e2 * sigma12_prev/(2*mu) + e1 * sigmaTrace;
  FunctionPtr viscous2 = e1 * sigma12_prev/(2*mu) + e2 * sigma22_prev/(2*mu) + e2 * sigmaTrace;

  rhs->addTerm( (e1 * mass_1 + e2 *mass_2) * v1->grad());
  rhs->addTerm( (e1 * momentum_x_1 + e2 *momentum_x_2 + e1 * sigma11_prev + e2 * sigma12_prev) * v2->grad());
  rhs->addTerm( (e1 * momentum_y_1 + e2 *momentum_y_2 + e1 * sigma12_prev + e2 * sigma22_prev) * v3->grad());
  rhs->addTerm( (e1 * energy_1 + e2 *energy_2 + e1 * viscousEnergy1 + e2 * viscousEnergy2) * v4->grad());
  rhs->addTerm(u1_prev * -tau1->div() + viscous1 * tau1);
  rhs->addTerm(u2_prev * -tau2->div() + viscous2 * tau2);
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

  /* // other inflow bcs
  bc->addDirichlet(F1nhat,inflowBoundary,(e1*mass_1 + e2 *mass_2)*n);
  bc->addDirichlet(F2nhat,inflowBoundary,(e1*momentum_x_1 + e2*momentum_x_2 + e1*sigma11_prev + e2*sigma12_prev)*n);
  bc->addDirichlet(F3nhat,inflowBoundary,(e1*momentum_y_1 + e2*momentum_y_2 + e1*sigma12_prev + e2*sigma22_prev)*n);
  bc->addDirichlet(F4nhat,inflowBoundary,(e1*energy_1 + e2*energy_2 + e1 * viscousEnergy1 + e2 * viscousEnergy2)*n);
  */

  // =============================================================================================
  
  // wall BCs
  bc->addDirichlet(u1hat, wallBoundary, zero);
  bc->addDirichlet(u2hat, wallBoundary, zero);
  bc->addDirichlet(That, wallBoundary, Teuchos::rcp(new ConstantScalarFunction(T_free*2.8))); // inferred from LD paper

  // =============================================================================================

  FunctionPtr F2n_for_sigma12 = ( e1 * momentum_x_1 + e2 * momentum_x_2) * n; // makes sigma_12 implicitly 0
  FunctionPtr F4n_for_qn = (e1 * energy_1 + e2 *energy_2 + e1 * viscousEnergy1 + e2 * viscousEnergy2)*n; // makes q_n implicitly 0  

  // symmetry BCs
  SpatialFilterPtr freeTop = Teuchos::rcp( new FreeStreamBoundaryTop );
  bc->addDirichlet(u2hat, freeTop, Teuchos::rcp( new ConstantScalarFunction(u2_free))); // top sym bc
  //  bc->addDirichlet(F2nhat, freeTop, F2n_for_sigma12);
  //  bc->addDirichlet(F4nhat, freeTop, F4n_for_qn); // sets zero y-stress in free stream top boundary 
  bc->addDirichlet(F2nhat, freeTop, zero);
  bc->addDirichlet(F4nhat, freeTop, zero); // sets zero y-heat flux in free stream top boundary

  // =============================================================================================

  SpatialFilterPtr freeBottom = Teuchos::rcp( new FreeStreamBoundaryBottom );
  bc->addDirichlet(u2hat, freeBottom, Teuchos::rcp( new ConstantScalarFunction(u2_free))); // symmetry bc
  //  bc->addDirichlet(u2hat, freeBottom, u2_prev); 
  //  bc->addDirichlet(F2nhat, freeBottom, F2n_for_sigma12);
  bc->addDirichlet(F2nhat, freeBottom, zero);
  //  bc->addDirichlet(F4nhat, freeBottom, F4n_for_qn); // sets zero y-stress in free stream bottom boundary
  bc->addDirichlet(F4nhat, freeBottom, zero); // sets zero y-stress in free stream bottom boundary
  //  bc->addDirichlet(F4nhat, wallBoundary, F4n_for_qn); // sets zero heat flux at wall
 
  ////////////////////////////////////////////////////////////////////
  // DEFINE PENALTY BC
  ////////////////////////////////////////////////////////////////////

  //  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  //    LinearTermPtr sigma_hat = beta * uhat->times_normal() - beta_n_u_minus_sigma_hat;
  //    pc->addConstraint(sigma_hat==zero,outflowBoundary);
  //    pc->addConstraint(q_n == 0, wallBoundary);
  
  
  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, bc, rhs, ip));
  mesh->registerSolution(solution);
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE REFINEMENT STRATEGY
  ////////////////////////////////////////////////////////////////////
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));

  int numTimeSteps = 0;
  int numNRSteps = 1;
  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy;
  solveStrategy = Teuchos::rcp( new NonlinearSolveStrategy(backgroundFlow, solution, stepSize,
                                                           nonlinearRelativeEnergyTolerance));
  
  ////////////////////////////////////////////////////////////////////
  // SOLVE 
  ////////////////////////////////////////////////////////////////////

  if (numTimeSteps>0){
    double dt = 1e-2;
    FunctionPtr dt_fxn;
    cout << "Timestep dt = " << dt << endl;

    // needs prev time residual (u_t(i-1) - u_t(i))/dt
    FunctionPtr u1sq_pt = u1_prev_time*u1_prev_time;
    FunctionPtr u2sq_pt = u2_prev_time*u2_prev_time;
    FunctionPtr iota_pt = cv*T_prev_time; // internal energy
    FunctionPtr unorm_pt = (u1sq_pt + u2sq_pt);
    FunctionPtr e_prev_time = .5*unorm_pt + iota_pt; // kinetic + internal energy

    // mass 
    bf->addTerm(rho,v1/dt);    
    FunctionPtr time_res_1 = rho_prev_time - rho_prev;  
    rhs->addTerm( time_res_1 * (v1/dt));

    // x momentum
    bf->addTerm(u1_prev * rho + rho_prev * u1, v2/dt);
    FunctionPtr time_res_2 = rho_prev_time * u1_prev_time - rho_prev * u1_prev;
    rhs->addTerm(time_res_2 * (v2/dt));

    // y momentum
    bf->addTerm(u2_prev * rho + rho_prev * u2, v3/dt);
    FunctionPtr time_res_3 = rho_prev_time * u2_prev_time - rho_prev * u2_prev;
    rhs->addTerm(time_res_3 * (v3/dt));

    // energy  
    bf->addTerm((e) * rho + (dedu1*rho_prev) * u1 + (dedu2*rho_prev) * u2 + (dedT*rho_prev) * T, v4/dt);
    FunctionPtr time_res_4 = (rho_prev_time * e_prev_time - rho_prev * e);
    rhs->addTerm(time_res_4 * (v4/dt));
  }
  // time step 1

  cout << "getting to the solve..." << endl;

  // prerefine the mesh
  int numRefs = 0;
  for (int refIndex=0;refIndex<numRefs;refIndex++){    
    solution->solve(); // false: don't use MUMPS
    refinementStrategy->refine(rank==0); // print to console on rank 0	
  }
  solution->solve(); // false: don't use MUMPS
  cout << "solved." << endl;
  if (rank==0){
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
  //  energy_2->writeValuesToMATLABFile(mesh, "energy_2.m");

  if (rank==0){
    cout << "doing 2-n timesteps" << endl;
  }
  // time steps
  for (int i = 0;i<numTimeSteps;i++){
    for (int j = 0;j<numNRSteps;j++){
      solution->solve(); 
      backgroundFlow->addSolution(solution,1.0);
    }
    prevTimeFlow->setSolution(backgroundFlow);  
    cout << "timestep i = " << i << endl;
  }

  if (rank==0){
    solution->writeFluxesToFile(u1hat->ID(), "u1hat2.dat");
    solution->writeFluxesToFile(u2hat->ID(), "u2hat2.dat");
    solution->writeFluxesToFile(That->ID(), "That2.dat");
    solution->writeFluxesToFile(F1nhat->ID(), "F1nhat2.dat");
    solution->writeFluxesToFile(F2nhat->ID(), "F2nhat2.dat");
    solution->writeFluxesToFile(F3nhat->ID(), "F3nhat2.dat");
    solution->writeFluxesToFile(F4nhat->ID(), "F4nhat2.dat");

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
  } 
  /*
    for (int i = 0;i<numTimeSteps;i++){
    // single NR solve
    solution->solve(); // false: don't use MUMPS
    backgroundFlow->addSolution(solution,1.0);

    // next timestep
    prevTimeFlow->setSolution(backgroundFlow);

    std::string s = "hat.dat"; 
    std::string m = ".m"; 
    std::stringstream rho_out; std::stringstream u1_out;
    std::stringstream u2_out; std::stringstream T_out;
    rho_out << "rho" << i ;
    u1_out << "u1"<< i;
    u2_out << "u2" << i ;
    T_out << "T" << i ;
    
    if (rank==0){
      backgroundFlow->writeFieldsToFile(u1->ID(), (u1_out.str()+m).c_str());
      backgroundFlow->writeFieldsToFile(u2->ID(), (u2_out.str()+m).c_str());
      backgroundFlow->writeFieldsToFile(rho->ID(), (rho_out.str()+m).c_str());
      backgroundFlow->writeFieldsToFile(T->ID(), (T_out.str()+m).c_str());
      
      solution->writeFieldsToFile(u1->ID(), "du1.m");
      solution->writeFieldsToFile(u2->ID(), "du2.m");
      solution->writeFieldsToFile(rho->ID(), "drho.m");
      solution->writeFieldsToFile(T->ID(), "dT.m");

      solution->writeFluxesToFile(u1hat->ID(), (u1_out.str()+s).c_str());
      solution->writeFluxesToFile(u2hat->ID(),(u2_out.str()+s).c_str());
      solution->writeFluxesToFile(That->ID(), (T_out.str()+s).c_str());
      solution->writeFluxesToFile(F1nhat->ID(), "F1nhat.dat");
      solution->writeFluxesToFile(F2nhat->ID(), "F2nhat.dat");
      solution->writeFluxesToFile(F3nhat->ID(), "F3nhat.dat");
      solution->writeFluxesToFile(F4nhat->ID(), "F4nhat.dat");
      cout << "done with timestep " << i << endl;
    }
  }
  
  for (int i = 0;i<numTimeSteps;i++){
    if (rank==0){
      backgroundFlow->writeFieldsToFile(u1->ID(), "u1_prev.m");
      backgroundFlow->writeFieldsToFile(u2->ID(), "u2_prev.m");
      backgroundFlow->writeFieldsToFile(rho->ID(), "rho_prev.m");
      backgroundFlow->writeFieldsToFile(T->ID(), "T_prev.m");
    }
    for (int j = 0;j<numNRSteps;j++){
      solution->solve(); // false: don't use MUMPS
      backgroundFlow->addSolution(solution,1.0);
    }
    prevTimeFlow->setSolution(backgroundFlow);
    if (rank==0){
      cout << "on timestep " << i << endl;
    }

    std::string s; std::stringstream rho_out; std::stringstream u1_out;
    std::stringstream u2_out; std::stringstream T_out;
    rho_out << "rho" << i << ".m";
    u1_out << "u1"<< i << ".m";
    u2_out << "u2" << i << ".m";
    T_out << "T" << i << ".m";
    if (rank==0){
      backgroundFlow->writeFieldsToFile(u1->ID(), u1_out.str().c_str());
      backgroundFlow->writeFieldsToFile(u2->ID(), u2_out.str().c_str());
      backgroundFlow->writeFieldsToFile(rho->ID(), rho_out.str().c_str());
      backgroundFlow->writeFieldsToFile(T->ID(), T_out.str().c_str());

      solution->writeFieldsToFile(u1->ID(), "du1.m");
      solution->writeFieldsToFile(u2->ID(), "du2.m");
      solution->writeFieldsToFile(rho->ID(), "drho.m");
      solution->writeFieldsToFile(T->ID(), "dT.m");

      solution->writeFluxesToFile(u1hat->ID(), "u1hat.dat");
      solution->writeFluxesToFile(u2hat->ID(), "u2hat.dat");
      solution->writeFluxesToFile(That->ID(), "That.dat");
    }
  }
  //  backgroundFlow->addSolution(solution,1.0);
  */
  /*
  if (numRefs==0){ //just do one solve
    solution->solve(false); // false: don't use MUMPS
  }else{
    for (int refIndex=0;refIndex<numRefs;refIndex++){    
      solution->solve(false); // false: don't use MUMPS
      //    solveStrategy->solve(rank==0);       // print to console on rank 0
      refinementStrategy->refine(rank==0); // print to console on rank 0	
    }
    solution->solve(false); // false: don't use MUMPS
  }

  int numPreRefs = 0;
  for (int refIndex=0;refIndex<numPreRefs;refIndex++){    
    solution->solve(false); // false: don't use MUMPS
    //solveStrategy->solve(rank==0);       // print to console on rank 0
    refinementStrategy->refine(rank==0); // print to console on rank 0	
  }

  solution->solve(false); // false: don't use MUMPS 
  
  // one more nonlinear solve on refined mesh
  double factor = 1/Re;
  factor = .5;
  for (int i=0;i<numNRSteps;i++){
    solution->solve(); // false: don't use MUMPS
    backgroundFlow->addSolution(solution,factor);
    factor *= 1.5;
    factor = min(factor,.5);
    double err = solution->energyErrorTotal();
    if (rank==0){
      cout << "NR step " << i << " with energy error " << err << " and factor = " << factor << endl;
    }
  }
  */
   
  return 0;
}
