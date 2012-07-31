//
//  NavierStokesCavityFlowDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "LidDrivenFlowRefinementStrategy.h"
#include "RefinementPattern.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "MeshPolyOrderFunction.h"
#include "MeshTestUtility.h"
#include "NonlinearSolveStrategy.h"
#include "PenaltyConstraints.h"

typedef Teuchos::RCP<Element> ElementPtr;
typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

static double REYN = 100;
//const static double REYN = 400; // matches John Evans's dissertation, p. 183

const static string S_TAU1 = "\\tau_1";
const static string S_TAU2 = "\\tau_2";
const static string S_V1 = "v_1";
const static string S_V2 = "v_2";
const static string S_Q = "q";
const static string S_U1_HAT = "\\widehat{u}_1";
const static string S_U2_HAT = "\\widehat{u}_2";
const static string S_TRACTION_1N = "\\widehat{P - \\mu \\sigma_{1n}}";
const static string S_TRACTION_2N = "\\widehat{P - \\mu \\sigma_{2n}}";
const static string S_U1 = "u_1";
const static string S_U2 = "u_2";
const static string S_SIGMA11 = "\\sigma_{11}";
const static string S_SIGMA12 = "\\sigma_{12}";
const static string S_SIGMA21 = "\\sigma_{21}";
const static string S_SIGMA22 = "\\sigma_{22}";
const static string S_P = "p";

VarFactory varFactory; 
// test variables:
VarPtr tau1, tau2, v1, v2, q;
// traces and fluxes:
VarPtr u1hat, u2hat, t1n, t2n;
// field variables:
VarPtr u1, u2, sigma11, sigma12, sigma21, sigma22, p;

void initVariables() {
  tau1 = varFactory.testVar(S_TAU1, HDIV);  // tau_1
  tau2 = varFactory.testVar(S_TAU2, HDIV);  // tau_2
  v1 = varFactory.testVar(S_V1, HGRAD); // v_1
  v2 = varFactory.testVar(S_V2, HGRAD); // v_2
  q = varFactory.testVar(S_Q, HGRAD); // q
  
  u1hat = varFactory.traceVar(S_U1_HAT);
  u2hat = varFactory.traceVar(S_U2_HAT);
  t1n = varFactory.fluxVar(S_TRACTION_1N);
  t2n = varFactory.fluxVar(S_TRACTION_2N);
  u1 = varFactory.fieldVar(S_U1);
  u2 = varFactory.fieldVar(S_U2);
  sigma11 = varFactory.fieldVar(S_SIGMA11);
  sigma12 = varFactory.fieldVar(S_SIGMA12);
  sigma21 = varFactory.fieldVar(S_SIGMA21);
  sigma22 = varFactory.fieldVar(S_SIGMA22);
  p = varFactory.fieldVar(S_P);
}

class U1_0 : public SimpleFunction {
  double _eps;
public:
  U1_0(double eps) {
    _eps = eps;
  }
  double value(double x, double y) {
    double tol = 1e-14;
    if (abs(y-1.0) < tol) { // top boundary
      if ( (abs(x) < _eps) ) { // top left
        return x / _eps;
      } else if ( abs(1.0-x) < _eps) { // top right
        return (1.0-x) / _eps;
      } else { // top middle
        return 1;
      }
    } else { // not top boundary: 0.0
      return 0.0;
    }
  }
};

class U2_0 : public SimpleFunction {
public:
  double value(double x, double y) {
    return 0.0;
  }
};

class Un_0 : public ScalarFunctionOfNormal {
  SimpleFunctionPtr _u1, _u2;
public:
  Un_0(double eps) {
    _u1 = Teuchos::rcp(new U1_0(eps));
    _u2 = Teuchos::rcp(new U2_0);
  }
  double value(double x, double y, double n1, double n2) {
    double u1 = _u1->value(x,y);
    double u2 = _u2->value(x,y);
    return u1 * n1 + u2 * n2;
  }
};

class U0_cross_n : public ScalarFunctionOfNormal {
  SimpleFunctionPtr _u1, _u2;
public:
  U0_cross_n(double eps) {
    _u1 = Teuchos::rcp(new U1_0(eps));
    _u2 = Teuchos::rcp(new U2_0);
  }
  double value(double x, double y, double n1, double n2) {
    double u1 = _u1->value(x,y);
    double u2 = _u2->value(x,y);
    return u1 * n2 - u2 * n1;
  }
};

class SqrtFunction : public Function {
  FunctionPtr _f;
public:
  SqrtFunction(FunctionPtr f) : Function(0) {
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
        values(cellIndex,ptIndex) = sqrt(value);
      }
    }
  }
};

class UnitSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y) < tol) || (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

void writePatchValues(double xMin, double xMax, double yMin, double yMax,
                      SolutionPtr solution, VarPtr u1, string filename) {
  vector<double> points1D_x, points1D_y;
  int numPoints = 100;
  for (int i=0; i<numPoints; i++) {
    points1D_x.push_back( xMin + (xMax - xMin) * ((double) i) / (numPoints-1) );
    points1D_y.push_back( yMin + (yMax - yMin) * ((double) i) / (numPoints-1) );
  }
  int spaceDim = 2;
  FieldContainer<double> points(numPoints*numPoints,spaceDim);
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      points(pointIndex,0) = points1D_x[i];
      points(pointIndex,1) = points1D_y[j];
    }
  }
  FieldContainer<double> values1(numPoints*numPoints);
  FieldContainer<double> values2(numPoints*numPoints);
  solution->solutionValues(values1, u1->ID(), points);
  ofstream fout(filename.c_str());
  fout << setprecision(15);
  
  fout << "X = zeros(" << numPoints << ",1);\n";
  //    fout << "Y = zeros(numPoints);\n";
  fout << "U = zeros(" << numPoints << "," << numPoints << ");\n";
  for (int i=0; i<numPoints; i++) {
    fout << "X(" << i+1 << ")=" << points1D_x[i] << ";\n";
  }
  for (int i=0; i<numPoints; i++) {
    fout << "Y(" << i+1 << ")=" << points1D_y[i] << ";\n";
  }
  
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      fout << "U("<<i+1<<","<<j+1<<")=" << values1(pointIndex) << ";" << endl;
    }
  }
  fout.close();
}

void initStokesBilinearForm(BFPtr emptyBF, FunctionPtr mu) {  
  // the velocity-gradient-pressure (VGP) stokes form
  BFPtr stokesBFMath = emptyBF;
  // the tau equations define sigma = grad u
  // tau1 terms:
  stokesBFMath->addTerm(u1,tau1->div());
  stokesBFMath->addTerm(sigma11,tau1->x()); // (sigma1, tau1)
  stokesBFMath->addTerm(sigma12,tau1->y());
  stokesBFMath->addTerm(-u1hat, tau1->dot_normal());
  
  // tau2 terms:
  stokesBFMath->addTerm(u2, tau2->div());
  stokesBFMath->addTerm(sigma21,tau2->x()); // (sigma2, tau2)
  stokesBFMath->addTerm(sigma22,tau2->y());
  stokesBFMath->addTerm(-u2hat, tau2->dot_normal());
  
  // the v equations are the momentum equations
  // v1:
  stokesBFMath->addTerm(mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
  stokesBFMath->addTerm(mu * sigma12,v1->dy());
  stokesBFMath->addTerm( - p, v1->dx() );
  stokesBFMath->addTerm( t1n, v1);
  
  // v2:
  stokesBFMath->addTerm(mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
  stokesBFMath->addTerm(mu * sigma22,v2->dy());
  stokesBFMath->addTerm( -p, v2->dy());
  stokesBFMath->addTerm( t2n, v2);
  
  // the q equation is the conservation of mass
  // q:
  stokesBFMath->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBFMath->addTerm(-u2,q->dy());
  stokesBFMath->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);
  
//  return stokesBFMath;
}

void initNavierStokesBilinearForm(BFPtr emptyBF, FunctionPtr mu, SolutionPtr prevSolution) {
  initStokesBilinearForm(emptyBF, mu);
  
  FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, u1) );
  FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, u2) );
  FunctionPtr sigma11_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma11) );
  FunctionPtr sigma12_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma12) );
  FunctionPtr sigma21_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma21) );
  FunctionPtr sigma22_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma22) );
  
  emptyBF->addTerm(sigma11_prev * u1 + sigma12_prev * u2 + u1_prev * sigma11 + u2_prev * sigma12, v1);
  emptyBF->addTerm(sigma21_prev * u1 + sigma22_prev * u2 + u1_prev * sigma21 + u2_prev * sigma22, v2);
}

IPPtr initGraphInnerProductStokes(FunctionPtr mu) {
  // TODO: implement one of these for Navier-Stokes as well
  IPPtr qoptIP = Teuchos::rcp(new IP());
  
  double beta = 1.0;
  
  qoptIP->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
  qoptIP->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
  qoptIP->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
  qoptIP->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
  qoptIP->addTerm( v1->dx() + v2->dy() );       // pressure
  qoptIP->addTerm( tau1->div() - q->dx() );     // u1
  qoptIP->addTerm( tau2->div() - q->dy() );     // u2
  
  qoptIP->addTerm( sqrt(beta) * v1 );
  qoptIP->addTerm( sqrt(beta) * v2 );
  qoptIP->addTerm( sqrt(beta) * q );
  qoptIP->addTerm( sqrt(beta) * tau1 );
  qoptIP->addTerm( sqrt(beta) * tau2 );

  return qoptIP;
}

void initRHSNavierStokes( Teuchos::RCP<RHSEasy> rhs, BFPtr stokesBF, FunctionPtr mu, SolutionPtr prevSolution) {
  bool useBFTestFunctional = true; // just because this is new and untested, don't entirely trust it...

  FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, u1) );
  FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, u2) );
  FunctionPtr sigma11_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma11) );
  FunctionPtr sigma12_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma12) );
  FunctionPtr sigma21_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma21) );
  FunctionPtr sigma22_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, sigma22) );
  FunctionPtr p_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, p) );
  
  FunctionPtr u1hat_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, u1hat) );
  FunctionPtr u2hat_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, u2hat) );
  FunctionPtr t1n_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, t1n) );
  FunctionPtr t2n_prev = Teuchos::rcp( new PreviousSolutionFunction(prevSolution, t2n) );
  
  if (useBFTestFunctional) {
    // subtract stokesBF evaluated at previous solution from the RHS
    rhs->addTerm(-stokesBF->testFunctional(prevSolution));
  } else {
    // for now, omit fluxes and traces:
    
//    // tau1 terms:
//    stokesBFMath->addTerm(u1,tau1->div());
//    stokesBFMath->addTerm(sigma11,tau1->x()); // (sigma1, tau1)
//    stokesBFMath->addTerm(sigma12,tau1->y());
//    stokesBFMath->addTerm(-u1hat, tau1->dot_normal());
    
    rhs->addTerm(- u1_prev * tau1->div() - sigma11_prev * tau1->x() - sigma12_prev * tau1->y());
    rhs->addTerm(u1hat_prev * tau1->dot_normal() );
    
//    // tau2 terms:
//    stokesBFMath->addTerm(u2, tau2->div());
//    stokesBFMath->addTerm(sigma21,tau2->x()); // (sigma2, tau2)
//    stokesBFMath->addTerm(sigma22,tau2->y());
//    stokesBFMath->addTerm(-u2hat, tau2->dot_normal());
    
    rhs->addTerm( -u2_prev * tau2->div() - sigma21_prev * tau2->x() - sigma22_prev * tau2->y());
    rhs->addTerm( u2hat_prev * tau2->dot_normal() );
//    
//    // v1:
//    stokesBFMath->addTerm(mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
//    stokesBFMath->addTerm(mu * sigma12,v1->dy());
//    stokesBFMath->addTerm( - p, v1->dx() );
//    stokesBFMath->addTerm( t1n, v1);
    
    rhs->addTerm( - mu * sigma11_prev * v1->dx() - mu * sigma12_prev * v1->dy() + p_prev * v1->dx() );
    rhs->addTerm( - t1n_prev * v1 );
//    
//    // v2:
//    stokesBFMath->addTerm(mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
//    stokesBFMath->addTerm(mu * sigma22,v2->dy());
//    stokesBFMath->addTerm( -p, v2->dy());
//    stokesBFMath->addTerm( t2n, v2);
    
    rhs->addTerm( -mu * sigma21_prev * v2->dx() - mu * sigma22_prev * v2->dy() + p_prev * v2->dy() );
    rhs->addTerm( - t2n_prev * v2 );
//    
//    // q:
//    stokesBFMath->addTerm(-u1,q->dx()); // (-u, grad q)
//    stokesBFMath->addTerm(-u2,q->dy());
//    stokesBFMath->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);
    
    rhs->addTerm( u1_prev * q->dx()  + u2_prev * q->dy() );
    rhs->addTerm( - u1hat_prev * q->times_normal_x() - u2hat_prev * q->times_normal_y() );
  }
  // now add in the nonlinear (convective) term:
//  double rho = 1.0; // density fixed at 1.0
  // rho * ( - u * sigma_i, v_i)

  rhs->addTerm(- u1_prev * sigma11_prev * v1 - u2_prev * sigma21_prev * v1);
  rhs->addTerm(- u1_prev * sigma12_prev * v2 - u2_prev * sigma22_prev * v2);
}

int main(int argc, char *argv[]) {
  int rank = 0;
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
#else
#endif
  int pToAdd = 2; // for optimal test function approximation
  int pToAddForStreamFunction = 2;
  double nonlinearStepSize = 1.0;
  double dt = 0.01;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
//  double nonlinearRelativeEnergyTolerance = 0.15; // used to determine convergence of the nonlinear solution
  double eps = 1.0/64.0; // width of ramp up to 1.0 for top BC;  eps == 0 ==> soln not in H1
  // epsilon above is chosen to match our initial 16x16 mesh, to avoid quadrature errors.
  //  double eps = 0.0; // John Evans's problem: not in H^1
  bool enforceLocalConservation = false;
  bool enforceLocalConservationInFinalSolve = false; // only works correctly for Picard (and maybe not then!)
  bool enforceOneIrregularity = true;
  bool reportPerCellErrors  = true;
  bool useMumps = false;
  bool compareWithOverkillMesh = false;
  bool useAdHocHPRefinements = false;
  bool usePicardIteration = true; // instead of newton-raphson
  int overkillMeshSize = 8;
  int overkillPolyOrder = 7; // H1 order
  
  // usage: polyOrder [numRefinements]
  // parse args:
  if ((argc != 4) && (argc != 3) && (argc != 2)) {
    cout << "Usage: NavierStokesCavityFlowDriver fieldPolyOrder [numRefinements=10]\n";
    return -1;
  }
  int polyOrder = atoi(argv[1]);
  int numRefs = 10;
  if ( argc == 3) {
    numRefs = atoi(argv[2]);
  }
  if ( argc == 4) {
    numRefs = atoi(argv[2]);
    REYN = atof(argv[3]);
  }
  if (rank == 0) {
    cout << "numRefinements = " << numRefs << endl;
    cout << "REYN = " << REYN << endl;
    cout << "dt = " << dt << endl;
  }
  
  FunctionPtr mu = Teuchos::rcp( new ConstantScalarFunction(1.0 / REYN, "\\mu") );
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;

  initVariables();
  BFPtr navierStokesBF = Teuchos::rcp(new BF(varFactory));
  
  // define meshes:
  int H1Order = polyOrder + 1;
  int horizontalCells = 4, verticalCells = 4;
  bool useTriangles = false;
  bool meshHasTriangles = useTriangles;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                navierStokesBF, H1Order, H1Order+pToAdd, useTriangles);

  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy ); // zero for now...
  IPPtr ip = initGraphInnerProductStokes(mu);

  SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) ); // accumulated solution
  SolutionPtr solnIncrement = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  solnIncrement->setReportConditionNumber(false);
  mesh->registerSolution(solution);
  mesh->registerSolution(solnIncrement);
  
  initNavierStokesBilinearForm(navierStokesBF, mu, solution);
  
  BFPtr stokesBFMath = Teuchos::rcp(new BF(varFactory));
  initStokesBilinearForm( stokesBFMath, mu );

  if ( ! usePicardIteration ) { // Picard solves not for increments, but successive full solutions
    initRHSNavierStokes( rhs, stokesBFMath, mu, solution );
  }
  
  if ( ! usePicardIteration ) { // we probably could afford to do pseudo-time with Picard, but choose not to
    // add time marching terms for momentum equations (v1 and v2):
    FunctionPtr dt_inv = Teuchos::rcp( new ConstantScalarFunction(1.0 / dt, "\\frac{1}{dt}") );
    // LHS gets u_inc / dt:
    navierStokesBF->addTerm(-dt_inv * u1, v1);
    navierStokesBF->addTerm(-dt_inv * u2, v2);
  }
  
  if (rank==0) {
    cout << "********** STOKES BF **********\n";
    stokesBFMath->printTrialTestInteractions();
    cout << "\n\n********** NAVIER-STOKES BF **********\n";
    navierStokesBF->printTrialTestInteractions();
    cout << "\n\n";
  }
  
  // set initial guess (all zeros is probably a decent initial guess here)
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0) );
  map< int, FunctionPtr > initialGuesses;
  initialGuesses[u1->ID()] = zero;
  initialGuesses[u2->ID()] = zero;
  initialGuesses[sigma11->ID()] = zero;
  initialGuesses[sigma12->ID()] = zero;
  initialGuesses[sigma21->ID()] = zero;
  initialGuesses[sigma22->ID()] = zero;
  initialGuesses[p->ID()] = zero;
  initialGuesses[u1hat->ID()] = zero;
  initialGuesses[u2hat->ID()] = zero;
  initialGuesses[t1n->ID()] = zero;
  initialGuesses[t2n->ID()] = zero;
  solution->projectOntoMesh(initialGuesses);
  
  ///////////////////////////////////////////////////////////////////////////
  
  // define bilinear form for stream function:
  VarFactory streamVarFactory;
  VarPtr phi_hat = streamVarFactory.traceVar("\\widehat{\\phi}");
  VarPtr psin_hat = streamVarFactory.fluxVar("\\widehat{\\psi}_n");
  VarPtr psi_1 = streamVarFactory.fieldVar("\\psi_1");
  VarPtr psi_2 = streamVarFactory.fieldVar("\\psi_2");
  VarPtr phi = streamVarFactory.fieldVar("\\phi");
  VarPtr q_s = streamVarFactory.testVar("q_s", HGRAD);
  VarPtr v_s = streamVarFactory.testVar("v_s", HDIV);
  BFPtr streamBF = Teuchos::rcp( new BF(streamVarFactory) );
  streamBF->addTerm(psi_1, q_s->dx());
  streamBF->addTerm(psi_2, q_s->dy());
  streamBF->addTerm(-psin_hat, q_s);
  
  streamBF->addTerm(psi_1, v_s->x());
  streamBF->addTerm(psi_2, v_s->y());
  streamBF->addTerm(phi, v_s->div());
  streamBF->addTerm(-phi_hat, v_s->dot_normal());
  
  Teuchos::RCP<Mesh> streamMesh, overkillMesh;
  
  streamMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                   streamBF, H1Order+pToAddForStreamFunction, H1Order+pToAdd+pToAddForStreamFunction, useTriangles);

  mesh->registerMesh(streamMesh); // will refine streamMesh in the same way as mesh.
  
  Teuchos::RCP<Solution> overkillSolution;
  map<int, double> dofsToL2error; // key: numGlobalDofs, value: total L2error compared with overkill
  vector< VarPtr > fields;
  fields.push_back(u1);
  fields.push_back(u2);
  fields.push_back(sigma11);
  fields.push_back(sigma12);
  fields.push_back(sigma21);
  fields.push_back(sigma22);
  fields.push_back(p);
  
  if (rank == 0) {
    cout << "Starting mesh has " << horizontalCells << " x " << verticalCells << " elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    cout << "eps for top BC = " << eps << endl;
    
    if (useTriangles) {
      cout << "Using triangles.\n";
    }
    if (enforceLocalConservation) {
      cout << "Enforcing local conservation.\n";
    } else {
      cout << "NOT enforcing local conservation.\n";
    }
    if (enforceOneIrregularity) {
      cout << "Enforcing 1-irregularity.\n";
    } else {
      cout << "NOT enforcing 1-irregularity.\n";
    }
  }
  
  
  if (rank==0) 
    ip->printInteractions();
  
  ////////////////////   CREATE BCs   ///////////////////////
  SpatialFilterPtr entireBoundary = Teuchos::rcp( new UnitSquareBoundary );
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  FunctionPtr un_0 = Teuchos::rcp( new Un_0(eps) );
  FunctionPtr u0_cross_n = Teuchos::rcp( new U0_cross_n(eps) );
  
  FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u1) );
  FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u2) );
  
  FunctionPtr u1hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat) );
  FunctionPtr u2hat_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u2hat) );
    
  if ( ! usePicardIteration ) {
    bc->addDirichlet(u1hat, entireBoundary, u1_0 - u1hat_prev);
    bc->addDirichlet(u2hat, entireBoundary, u2_0 - u2hat_prev);
  // as long as we don't subtract from the RHS, I think the following is actually right:
//    bc->addDirichlet(u1hat, entireBoundary, u1_0);
//    bc->addDirichlet(u2hat, entireBoundary, u2_0);
  } else {
//    bc->addDirichlet(u1hat, entireBoundary, u1_0);
//    bc->addDirichlet(u2hat, entireBoundary, u2_0);
    // experiment:
    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
    pc->addConstraint(u1hat==u1_0,entireBoundary);
    pc->addConstraint(u2hat==u2_0,entireBoundary);
    solnIncrement->setFilter(pc);
  }
  bc->addZeroMeanConstraint(p);
  
  /////////////////// SOLVE OVERKILL //////////////////////
  if (compareWithOverkillMesh) {
    // TODO: fix this to make it work with Navier-Stokes
    cout << "WARNING: still need to switch overkill to handle nonlinear iteration...\n";
    overkillMesh = Mesh::buildQuadMesh(quadPoints, overkillMeshSize, overkillMeshSize,
                                       stokesBFMath, overkillPolyOrder, overkillPolyOrder+pToAdd, useTriangles);
    
    if (rank == 0) {
      cout << "Solving on overkill mesh (" << overkillMeshSize << " x " << overkillMeshSize << " elements, ";
      cout << overkillMesh->numGlobalDofs() <<  " dofs).\n";
    }
    overkillSolution = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
    overkillSolution->solve();
    if (rank == 0)
      cout << "...solved.\n";
    double overkillEnergyError = overkillSolution->energyErrorTotal();
    if (rank == 0)
      cout << "overkill energy error: " << overkillEnergyError << endl;
  }
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  
  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );
  //  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution,sigma12 - sigma21) );
  Teuchos::RCP<RHSEasy> streamRHS = Teuchos::rcp( new RHSEasy );
  streamRHS->addTerm(vorticity * q_s);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true);
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);
  
  Teuchos::RCP<BCEasy> streamBC = Teuchos::rcp( new BCEasy );
//  streamBC->addDirichlet(psin_hat, entireBoundary, u0_cross_n);
  streamBC->addDirichlet(phi_hat, entireBoundary, zero);
//  streamBC->addZeroMeanConstraint(phi);
  
  IPPtr streamIP = Teuchos::rcp( new IP );
  streamIP->addTerm(q_s);
  streamIP->addTerm(q_s->grad());
  streamIP->addTerm(v_s);
  streamIP->addTerm(v_s->div());
  SolutionPtr streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );
  
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
  
  double energyThreshold = 0.20; // for mesh refinements
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  if (useAdHocHPRefinements) 
//    refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solution, energyThreshold, 1.0 / horizontalCells )); // no h-refinements allowed
    refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solnIncrement, energyThreshold, 1.0 / overkillMeshSize, overkillPolyOrder, rank==0 ));
  else
    refinementStrategy = Teuchos::rcp( new RefinementStrategy( solnIncrement, energyThreshold ));
  
  refinementStrategy->setEnforceOneIrregurity(enforceOneIrregularity);
  refinementStrategy->setReportPerCellErrors(reportPerCellErrors);

  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
                                                                                               stepSize,
                                                                                               nonlinearRelativeEnergyTolerance));
  
  Teuchos::RCP<NonlinearSolveStrategy> finalSolveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
                                                                                               stepSize,
                                                                                               nonlinearRelativeEnergyTolerance / 10));

  
  
  solveStrategy->setUsePicardIteration(usePicardIteration);
  
  // run some refinements on the initial linear problem
//  int numInitialRefs = 5;
//  for (int refIndex=0; refIndex<numInitialRefs; refIndex++){    
//    solnIncrement->solve();
//    refinementStrategy->refine(rank==0); // print to console on rank 0
//  }
//  solveStrategy->solve(rank==0);
  
  if (true) { // do regular refinement strategy...
    FieldContainer<double> bottomCornerPoints(2,2);
    bottomCornerPoints(0,0) = 1e-10;
    bottomCornerPoints(0,1) = 1e-10;
    bottomCornerPoints(1,0) = 1 - 1e-10;
    bottomCornerPoints(1,1) = 1e-10;
    
    FieldContainer<double> topCornerPoints(4,2);
    topCornerPoints(0,0) = 1e-10;
    topCornerPoints(0,1) = 1 - 1e-12;
    topCornerPoints(1,0) = 1 - 1e-10;
    topCornerPoints(1,1) = 1 - 1e-12;
    topCornerPoints(2,0) = 1e-12;
    topCornerPoints(2,1) = 1 - 1e-10;
    topCornerPoints(3,0) = 1 - 1e-12;
    topCornerPoints(3,1) = 1 - 1e-10;
    
    bool printToConsole = rank==0;
    for (int refIndex=0; refIndex<numRefs; refIndex++){    
      solveStrategy->solve(printToConsole);
//      if (compareWithOverkillMesh) {
//        Teuchos::RCP<Solution> projectedSoln = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
//        solution->projectFieldVariablesOntoOtherSolution(projectedSoln);
//        if (refIndex==numRefs-1) { // last refinement
//          solution->writeFieldsToFile(p->ID(),"pressure.m");
//          overkillSolution->writeFieldsToFile(p->ID(), "pressure_overkill.m");
//        }
//
//        projectedSoln->addSolution(overkillSolution,-1.0);
//
//        if (refIndex==numRefs-1) { // last refinement
//          projectedSoln->writeFieldsToFile(p->ID(), "pressure_error_vs_overkill.m");
//        }
//        double L2errorSquared = 0.0;
//        for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
//          VarPtr var = *fieldIt;
//          int fieldID = var->ID();
//          double L2error = projectedSoln->L2NormOfSolutionGlobal(fieldID);
//          if (rank==0)
//            cout << "L2error for " << var->name() << ": " << L2error << endl;
//          L2errorSquared += L2error * L2error;
//        }
//        
//        int numGlobalDofs = mesh->numGlobalDofs();
//        if (rank==0)
//          cout << "for " << numGlobalDofs << " dofs, total L2 error: " << sqrt(L2errorSquared) << endl;
//        dofsToL2error[numGlobalDofs] = sqrt(L2errorSquared);
//      }
      refinementStrategy->refine(rank==0); // print to console on rank 0
//      if (! MeshTestUtility::checkMeshConsistency(mesh)) {
//        if (rank==0) cout << "checkMeshConsistency returned false after refinement.\n";
//      }
      // find top corner cells:
      vector< Teuchos::RCP<Element> > topCorners = mesh->elementsForPoints(topCornerPoints);
      if (rank==0) {// print out top corner cellIDs
        cout << "Refinement # " << refIndex+1 << " complete.\n";
        vector<int> cornerIDs;
        cout << "top-left corner ID: " << topCorners[0]->cellID() << endl;
        cout << "top-right corner ID: " << topCorners[1]->cellID() << endl;
      }
    }
    // one more solve on the final refined mesh:
    if (enforceLocalConservationInFinalSolve && !enforceLocalConservation) {
      solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
    }
    
    finalSolveStrategy->solve(printToConsole);
  }
//  if (compareWithOverkillMesh) {
//    Teuchos::RCP<Solution> projectedSoln = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
//    solution->projectFieldVariablesOntoOtherSolution(projectedSoln);
//    
//    projectedSoln->addSolution(overkillSolution,-1.0);
//    double L2errorSquared = 0.0;
//    for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
//      VarPtr var = *fieldIt;
//      int fieldID = var->ID();
//      double L2error = projectedSoln->L2NormOfSolutionGlobal(fieldID);
//      if (rank==0)
//        cout << "L2error for " << var->name() << ": " << L2error << endl;
//      L2errorSquared += L2error * L2error;
//    }
//    int numGlobalDofs = mesh->numGlobalDofs();
//    if (rank==0)
//      cout << "for " << numGlobalDofs << " dofs, total L2 error: " << sqrt(L2errorSquared) << endl;
//    dofsToL2error[numGlobalDofs] = sqrt(L2errorSquared);
//  }
  
  double energyErrorTotal = solution->energyErrorTotal();
  double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
  if (rank == 0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "  (Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".)\n";
  }
  
  FunctionPtr u1_sq = u1_prev * u1_prev;
  FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
  FunctionPtr u_mag = Teuchos::rcp( new SqrtFunction( u_dot_u ) );
  FunctionPtr u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dx() + u2->dy() ) );
  FunctionPtr massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat->times_normal_x() + u2hat->times_normal_y()) );
  
  // check that the zero mean pressure is being correctly imposed:
  FunctionPtr p_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,p) );
  double p_avg = p_prev->integrate(mesh);
  if (rank==0)
    cout << "Integral of pressure: " << p_avg << endl;
  
  // integrate massFlux over each element (a test):
  // fake a new bilinear form so we can integrate against 1 
  VarPtr testOne = varFactory.testVar("1",CONSTANT_SCALAR);
  BFPtr fakeBF = Teuchos::rcp( new BF(varFactory) );
  LinearTermPtr massFluxTerm = massFlux * testOne;
  
  CellTopoPtr quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
  DofOrderingFactory dofOrderingFactory(fakeBF);
  int fakeTestOrder = H1Order;
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, *quadTopoPtr);
  
  int testOneIndex = testOrdering->getDofIndex(testOne->ID(),0);
  vector< ElementTypePtr > elemTypes = mesh->elementTypes(); // global element types
  map<int, double> massFluxIntegral; // cellID -> integral
  double maxMassFluxIntegral = 0.0;
  double totalMassFlux = 0.0;
  double totalAbsMassFlux = 0.0;
  double maxCellMeasure = 0;
  double minCellMeasure = 1;
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
    //      cout << "fakeRHSIntegrals:\n" << fakeRHSIntegrals;
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      // pick out the ones for testOne:
      massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
    }
    //      int numSides = 4;
    //      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    //        for (int i=0; i<elems.size(); i++) {
    //          int cellID = cellIDs[i];
    //          // pick out the ones for testOne:
    //          massFluxIntegral[cellID] += fakeRHSIntegrals(i,testOneIndex);
    //        }
    //      }
    // find the largest:
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
    }
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
      minCellMeasure = min(minCellMeasure,cellMeasures(i));
      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
      totalMassFlux += massFluxIntegral[cellID];
      totalAbsMassFlux += abs( massFluxIntegral[cellID] );
    }
  }
  if (rank==0) {
    cout << "largest mass flux: " << maxMassFluxIntegral << endl;
    cout << "total mass flux: " << totalMassFlux << endl;
    cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
    cout << "largest h: " << sqrt(maxCellMeasure) << endl;
    cout << "smallest h: " << sqrt(minCellMeasure) << endl;
    cout << "ratio of largest / smallest h: " << sqrt(maxCellMeasure) / sqrt(minCellMeasure) << endl;
  }
  if (rank == 0) {
    cout << "phi ID: " << phi->ID() << endl;
    cout << "psi1 ID: " << psi_1->ID() << endl;
    cout << "psi2 ID: " << psi_2->ID() << endl;
    
    cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
    cout << "solving for approximate stream function...\n";
  }
  //  mesh->unregisterMesh(streamMesh);
  //  streamMesh->registerMesh(mesh);
  //  RefinementStrategy streamRefinementStrategy( streamSolution, energyThreshold );
  //  for (int refIndex=0; refIndex < 3; refIndex++) {
  //    streamSolution->solve(false);
  //    streamRefinementStrategy.refine(rank==0);
  //  }
  
  streamSolution->solve(useMumps);
  energyErrorTotal = streamSolution->energyErrorTotal();
  if (rank == 0) {  
    cout << "...solved.\n";
    cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
  }
  
  if (rank==0){
    solution->writeToVTK("nsCavitySoln.vtk");
    if (! meshHasTriangles ) {
      massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
      u_mag->writeValuesToMATLABFile(solution->mesh(), "u_mag.m");
      u_div->writeValuesToMATLABFile(solution->mesh(), "u_div.m");
      solution->writeFieldsToFile(u1->ID(), "u1.m");
      solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
      solution->writeFieldsToFile(u2->ID(), "u2.m");
      solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
      solution->writeFieldsToFile(p->ID(), "p.m");
      streamSolution->writeFieldsToFile(phi->ID(), "phi.m");
      
      streamSolution->writeFluxesToFile(phi_hat->ID(), "phi_hat.dat");
      streamSolution->writeFieldsToFile(psi_1->ID(), "psi1.m");
      streamSolution->writeFieldsToFile(psi_2->ID(), "psi2.m");
      vorticity->writeValuesToMATLABFile(streamMesh, "vorticity.m");
      
      FunctionPtr ten = Teuchos::rcp( new ConstantScalarFunction(10) );
      ten->writeBoundaryValuesToMATLABFile(solution->mesh(), "skeleton.dat");
      cout << "wrote files: u_mag.m, u_div.m, u1.m, u1_hat.dat, u2.m, u2_hat.dat, p.m, phi.m, vorticity.m.\n";
    } else {
      solution->writeToFile(u1->ID(), "u1.dat");
      solution->writeToFile(u2->ID(), "u2.dat");
      solution->writeToFile(u2->ID(), "p.dat");
      cout << "wrote files: u1.dat, u2.dat, p.dat\n";
    }
    polyOrderFunction->writeValuesToMATLABFile(mesh, "cavityFlowPolyOrders.m");
    
    writePatchValues(0, 1, 0, 1, streamSolution, phi, "phi_patch.m");
    writePatchValues(0, .1, 0, .1, streamSolution, phi, "phi_patch_detail.m");
    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
    writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
}
  
  if (compareWithOverkillMesh) {
    cout << "******* Adaptivity Convergence Report *******\n";
    cout << "dofs\tL2 error\n";
    for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
      int dofs = entryIt->first;
      double err = entryIt->second;
      cout << dofs << "\t" << err << endl;
    }
    ofstream fout("overkillComparison.txt");
    fout << "******* Adaptivity Convergence Report *******\n";
    fout << "dofs\tL2 error\n";
    for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
      int dofs = entryIt->first;
      double err = entryIt->second;
      fout << dofs << "\t" << err << endl;
    }
    fout.close();
  }
  
  return 0;
}
