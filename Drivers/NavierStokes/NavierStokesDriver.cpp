//
//  NavierStokesDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/12/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//


#include "../Burgers/BurgersBilinearForm.h"

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

static const double GAMMA = 1.4;
static const double PRANDTL = 0.72;

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

class U0 : public SimpleFunction {
public:
  double value(double x, double y) {
    return 1 - 2 * x;
  }
};

class TopBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = (abs(y-1.0) < tol);
    return yMatch;
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
  int polyOrder = 3;
  int pToAdd = 2; // for tests
  
  // define our manufactured solution or problem bilinear form:
  double Re = 1e2;
  double Ma = 3.0;
  double cv = 1.0 / ( GAMMA * (GAMMA - 1) * (Ma * Ma) );
  double mu = 1.0 / Re;
  double lambda = 0.5 / Re;
  double kappa = GAMMA * cv * mu / PRANDTL;
  
  bool useTriangles = true;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  int H1Order = polyOrder + 1;
  int horizontalCells = 2, verticalCells = 2;
  
  double energyThreshold = 0.2; // for mesh refinements
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
  
  VarPtr F1hat = varFactory.fluxVar("\\widehat{F}_1");
  VarPtr F2hat = varFactory.fluxVar("\\widehat{F}_2");
  VarPtr F3hat = varFactory.fluxVar("\\widehat{F}_3");
  VarPtr F4hat = varFactory.fluxVar("\\widehat{F}_4");
  
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
  
  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;
  
  FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u1) );
  FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u2) );
  FunctionPtr rho_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, rho) );
  FunctionPtr T_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, T) );
//  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u) );
//  FunctionPtr beta = e1 * u_prev + Teuchos::rcp( new ConstantVectorFunction( e2 ) );
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE BILINEAR FORM
  ////////////////////////////////////////////////////////////////////
  
  // Eulerian terms:
  bf->addTerm(u1_prev * rho + rho_prev * u1, v1->dx()); 
  bf->addTerm(u2_prev * rho + rho_prev * u2, v1->dy());
  
  double cv_gamma_minus_one = cv * (GAMMA - 1);
  bf->addTerm( (u1_prev * u1_prev + cv_gamma_minus_one * T_prev) * rho + 2.0 * u1_prev * rho_prev * u1 + cv_gamma_minus_one * rho_prev * T, v2->dx() ) ;
  bf->addTerm( u1_prev * u2_prev * rho + u2_prev * rho_prev * u1 + u1_prev * rho_prev * u2, v2->dy() ) ;
  
  bf->addTerm( u1_prev * u2_prev * rho + u2_prev * rho_prev * u1 + u1_prev * rho_prev * u2, v3->dx() ) ;
  bf->addTerm( (u2_prev * u2_prev + cv_gamma_minus_one * T_prev) * rho + 2.0 * u2_prev * rho_prev * u2 + cv_gamma_minus_one * rho_prev * T, v3->dy() ) ;
  
  FunctionPtr u_T_fxn = Teuchos::rcp (new SumFunction(u1_prev * u1_prev, u2_prev * u2_prev) ) + (2.0 * cv * (2.0 * GAMMA - 1) ) * T_prev;
  
  FunctionPtr rho_v4_dx_weight = 0.5 * u1_prev * u_T_fxn;
  FunctionPtr sum = u_T_fxn + 2.0 * u1_prev * u1_prev; // temporary to avoid compiler complaints about ambiguous operators
  FunctionPtr u1_v4_dx_weight = 0.5 * sum * rho_prev;
  
  FunctionPtr u2_v4_dx_weight = u1_prev * u2_prev * rho_prev;
  FunctionPtr T_v4_dx_weight = cv * (2.0 * GAMMA - 1) * u1_prev * rho_prev;
  
  FunctionPtr rho_v4_dy_weight = 0.5 * u2_prev * u_T_fxn;
  sum = u_T_fxn + 2.0 * u2_prev * u2_prev;
  FunctionPtr u2_v4_dy_weight = 0.5 * sum * rho_prev;
  FunctionPtr u1_v4_dy_weight = u1_prev * u2_prev * rho_prev;
  FunctionPtr T_v4_dy_weight = cv * (2.0 * GAMMA - 1) * u2_prev * rho_prev;
  
  bf->addTerm( rho_v4_dx_weight * rho + u1_v4_dx_weight * u1 + u2_v4_dx_weight * u2 + T_v4_dx_weight * T, v4->dx() );
  bf->addTerm( rho_v4_dy_weight * rho + u1_v4_dy_weight * u1 + u2_v4_dy_weight * u2 + T_v4_dy_weight * T, v4->dy() );
  
  // viscous terms:
  double lambda_factor = lambda / (4.0 * mu * (mu + lambda) );
  double two_mu = 2 * mu;
  bf->addTerm(-u1hat, tau1->dot_normal() );
  bf->addTerm(sigma11 / two_mu - lambda_factor * sigma11 - lambda_factor * sigma22,tau1->x());
  bf->addTerm(sigma12 / two_mu - omega,tau1->y());
  bf->addTerm( u1, tau1->div() );
  
  
  bf->addTerm(-u2hat, tau2->dot_normal() );
  bf->addTerm(sigma12 / two_mu + omega,tau2->x());
  bf->addTerm(sigma22 / two_mu - lambda_factor * sigma11 - lambda_factor * sigma22,tau2->y());
  bf->addTerm( u2, tau2->div() );
  
  bf->addTerm( - That, tau3->dot_normal() );
  bf->addTerm(q1 / kappa, tau3->x());
  bf->addTerm(q2 / kappa, tau3->y());
  bf->addTerm(T, tau3->div());
  
  // ==================== SET INITIAL GUESS ==========================
  mesh->registerSolution(backgroundFlow);
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[BurgersBilinearForm::U] = u0;
  functionMap[BurgersBilinearForm::SIGMA_1] = zero;
  functionMap[BurgersBilinearForm::SIGMA_2] = zero;
  
  backgroundFlow->projectOntoMesh(functionMap);
  // ==================== END SET INITIAL GUESS ==========================
//  
//  ////////////////////////////////////////////////////////////////////
//  // DEFINE INNER PRODUCT
//  ////////////////////////////////////////////////////////////////////
//  // function to scale the squared guy by epsilon/h
//  FunctionPtr epsilonOverHScaling = Teuchos::rcp( new EpsilonScaling(epsilon) ); 
//  IPPtr ip = Teuchos::rcp( new IP );
//  ip->addTerm(tau);
//  ip->addTerm(tau->div());
//  ip->addTerm( epsilonOverHScaling * v );
//  ip->addTerm( sqrt(sqrt(epsilon)) * v->grad() );
//  ip->addTerm( beta * v->grad() );
//  
//  ////////////////////////////////////////////////////////////////////
//  // DEFINE RHS
//  ////////////////////////////////////////////////////////////////////
//  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
//  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
//  rhs->addTerm( (e1 * u_prev_squared_div2 + e2 * u_prev) * v->grad() - u_prev * tau->div());
  
  ////////////////////////////////////////////////////////////////////
  // DEFINE PENALTY BC
  ////////////////////////////////////////////////////////////////////
//  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new TopBoundary );
//  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(outflowBoundary) );
//  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
////  LinearTermPtr sigma_hat = beta * uhat->times_normal() - beta_n_u_minus_sigma_hat;
//  pc->addConstraint(sigma_hat==zero,outflowBoundary);
//  
//  ////////////////////////////////////////////////////////////////////
//  // DEFINE DIRICHLET BC
//  ////////////////////////////////////////////////////////////////////
//  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
//  Teuchos::RCP<BCEasy> inflowBC = Teuchos::rcp( new BCEasy );
//  FunctionPtr u0_squared_div_2 = 0.5 * u0 * u0;
//  inflowBC->addDirichlet(beta_n_u_minus_sigma_hat,inflowBoundary, 
//                         ( e1 * u0_squared_div_2 + e2 * u0) * n );
//  
//  ////////////////////////////////////////////////////////////////////
//  // CREATE SOLUTION OBJECT
//  ////////////////////////////////////////////////////////////////////
////  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, inflowBC, rhs, ip));
//  mesh->registerSolution(solution);
//  solution->setFilter(pc);
//  
//  ////////////////////////////////////////////////////////////////////
//  // DEFINE REFINEMENT STRATEGY
//  ////////////////////////////////////////////////////////////////////
//  Teuchos::RCP<RefinementStrategy> refinementStrategy;
//  refinementStrategy = Teuchos::rcp(new RefinementStrategy(solution,energyThreshold));
//  
//  int numRefs = 5;
//  
//  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
//  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy;
//  solveStrategy = Teuchos::rcp( new NonlinearSolveStrategy(backgroundFlow, solution, stepSize,
//                                                           nonlinearRelativeEnergyTolerance));
//  
//  ////////////////////////////////////////////////////////////////////
//  // SOLVE 
//  ////////////////////////////////////////////////////////////////////
//  
//  for (int refIndex=0;refIndex<numRefs;refIndex++){    
//    solveStrategy->solve(rank==0);       // print to console on rank 0
//    refinementStrategy->refine(rank==0); // print to console on rank 0
//  }
//  
//  // one more nonlinear solve on refined mesh
//  int numNRSteps = 5;
//  for (int i=0;i<numNRSteps;i++){
//    solution->solve(false); // false: don't use MUMPS
//    backgroundFlow->addSolution(solution,1.0);
//  }
//  
//  if (rank==0){
//    backgroundFlow->writeToFile(BurgersBilinearForm::U, "u_ref_old_plotSolution.dat");
//    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::U, "u_ref.m");
//    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::SIGMA_1, "sigmax.m");
//    backgroundFlow->writeFieldsToFile(BurgersBilinearForm::SIGMA_2, "sigmay.m");
//    solution->writeFluxesToFile(BurgersBilinearForm::U_HAT, "du_hat_ref.dat");
//  }
//  
  return 0;
}