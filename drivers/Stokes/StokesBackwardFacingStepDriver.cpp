//
//  StokesBackwardFacingStep.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/27/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "RefinementStrategy.h"
#include "RefinementPattern.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "GnuPlotUtil.h"
#include "SolutionExporter.h"

#include "StokesFormulation.h"
#include "MeshUtilities.h"

#include "RefinementPattern.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "RefinementHistory.h"

#include "StreamDriverUtil.h"

using namespace std;

static double tol=1e-14;

static const double RIGHT_OUTFLOW = 8.0;

bool topWall(double x, double y) {
  return abs(y-2.0) < tol;
}

bool bottomWallRight(double x, double y) {
  return (y < tol) && (x-4.0 >= tol);
}

bool bottomWallLeft(double x, double y) {
  return (abs(y-1.0) < tol) && (x-4.0 < tol);
}

bool step(double x, double y) {
  return (abs(x-4.0) < tol) && (y-1.0 <=tol);
}

bool inflow(double x, double y) {
  return abs(x) < tol;
}

bool outflow(double x, double y) {
  return abs(x-RIGHT_OUTFLOW) < tol;
}

class U1_0 : public SimpleFunction {
public:
  double value(double x, double y) {
    bool isTopWall = topWall(x,y);
    bool isBottomWallLeft = bottomWallLeft(x,y);
    bool isBottomWallRight = bottomWallRight(x,y);
    bool isStep = step(x,y);
    bool isInflow = inflow(x,y);
    //    bool isOutflow = outflow(x,y);
    if (isTopWall || isBottomWallLeft || isBottomWallRight || isStep ) { // walls: no slip
      return 0.0;
    } else if (isInflow) {
      return - 6.0 * (y-2.0) * (y-1.0);
    }
    return 0;
  }
};

class U2_0 : public SimpleFunction {
public:
  double value(double x, double y) {
    // everywhere u2 is prescribed, it's 0.
    return 0.0;
  }
};

class PHI_0 : public SimpleFunction {
  // for inflow, use the condition that volumetric flow rate is equal to the difference of the stream function
  // so this should be the integral of the inflow BC on u1
public:
  double value(double x, double y) {
    bool isTopWall = topWall(x,y);
    bool isBottomWallLeft = bottomWallLeft(x,y);
    bool isBottomWallRight = bottomWallRight(x,y);
    bool isStep = step(x,y);
    bool isInflow = inflow(x,y);
    bool isOutflow = outflow(x,y);
    if (isBottomWallLeft || isBottomWallRight || isStep ) { // walls: no slip
      return 0.0;
    } else if (isTopWall || isInflow) {
      return 9 * y * y - 2 * y * y * y - 12 * y + 5;
    } 
    //    else if (isOutflow) {
    //      // cheating: assume that we simply stretch the inflow uniformly
    //      // (the right way would be numerical integration of the actual solution u1 across outflow)
    //      // map from (0,2) to (1,2)
    //      y = y/2 + 1;
    //      return 9 * y * y - 2 * y * y * y - 12 * y + 5;
    //    }
    return 0;
  }
};

class Un_0 : public ScalarFunctionOfNormal {
  SimpleFunctionPtr _u1, _u2;
public:
  Un_0(double eps) {
    _u1 = Teuchos::rcp(new U1_0);
    _u2 = Teuchos::rcp(new U2_0);
  }
  double value(double x, double y, double n1, double n2) {
    double u1 = _u1->value(x,y);
    double u2 = _u2->value(x,y);
    return u1 * n1 + u2 * n2;
  }
};

class OutflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    return outflow(x,y);
  }
};

class NonOutflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    return ! outflow(x,y);
  }
};

class WallBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    bool isTopWall = topWall(x,y);
    bool isBottomWallLeft = bottomWallLeft(x,y);
    bool isBottomWallRight = bottomWallRight(x,y);
    bool isStep = step(x,y);
    return isTopWall || isBottomWallLeft || isBottomWallRight || isStep;
  }
};

void computeRecirculationRegion(double &xPoint, double &yPoint, SolutionPtr streamSoln, VarPtr phi) {
  // find the x recirculation region first
  int numIterations = 20;
  double x,y;
  y = 0.01;
  {
    double xGuessLeft = 4.25;
    double xGuessRight = 4.60;
    double leftValue = getSolutionValueAtPoint(xGuessLeft, y, streamSoln, phi);
    double rightValue = getSolutionValueAtPoint(xGuessRight, y, streamSoln, phi);
    if (leftValue * rightValue > 0) {
      cout << "Error: leftValue and rightValue have same sign.\n";
    }
    for (int i=0; i<numIterations; i++) {
      double xGuess = (xGuessLeft + xGuessRight) / 2;
      double middleValue = getSolutionValueAtPoint(xGuess, y, streamSoln, phi);
      if (middleValue * leftValue > 0) { // same sign
        xGuessLeft = xGuess;
        leftValue = middleValue;
      }
      if (middleValue * rightValue > 0) {
        xGuessRight = xGuess;
        rightValue = middleValue;
      }
      xPoint = xGuess;
    }
  }
  {
    double yGuessTop = 0.60;
    double yGuessBottom = 0.20;
    x = 4.01;
    double topValue = getSolutionValueAtPoint(x, yGuessTop, streamSoln, phi);
    double bottomValue = getSolutionValueAtPoint(x, yGuessBottom, streamSoln, phi);
    for (int i=0; i<numIterations; i++) {
      double yGuess = (yGuessTop + yGuessBottom) / 2;
      double middleValue = getSolutionValueAtPoint(x, yGuess, streamSoln, phi);
      if (middleValue * topValue > 0) { // same sign
        yGuessTop = yGuess;
        topValue = middleValue;
      }
      if (middleValue * bottomValue > 0) {
        yGuessBottom = yGuess;
        bottomValue = middleValue;
      }
      yPoint = yGuess;
    }
  }
}

int main(int argc, char *argv[]) {
  int rank = 0, numProcs = 1;
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
  numProcs=mpiSession.getNProc();
#else
#endif
  int pToAdd = 2; // for optimal test function approximation
  double energyThreshold = 0.20; // for mesh refinements
  int pToAddForStreamFunction = pToAdd;
  bool enforceLocalConservation = false;
  bool useExperimentalHdivNorm = false; // attempts to get H(div)-optimality in u (even though u is still L^2 discretely)
  bool useExperimentalH1Norm = false; // attempts to get H^1-optimality in u (even though u is still L^2 discretely)
  bool useGraphNormStrongerTau = false; // mimics the experimental H^1 norm in its requirements on tau
  bool useCompliantGraphNorm = false;   // weights to improve conditioning of the local problems
  bool useExtendedPrecisionForOptimalTestInversion = false;
  bool useCEFormulation = false;
  bool useIterativeRefinementsWithSPDSolve = false;
  bool useSPDLocalSolve = false;
  bool finalSolveUsesStandardGraphNorm = false;
  
  double min_h = 0; //1.0 / 128.0;
  
  // usage: polyOrder [numRefinements]
  // parse args:
  if ((argc != 3) && (argc != 2) && (argc != 4)) {
    cout << "Usage: StokesBackwardFacingStepDriver fieldPolyOrder [numRefinements=10 [adaptThresh=0.20]\n";
    return -1;
  }
  int polyOrder = atoi(argv[1]);
  int numRefs = 10;
  if ( ( argc == 3 ) || (argc == 4)) {
    numRefs = atoi(argv[2]);
  }
  if (argc == 4) {
    energyThreshold = atof(argv[3]);
  }
  if (rank == 0) {
    cout << "numRefinements = " << numRefs << endl;
    if (useCEFormulation) {
      cout << "Using 'cheap experimental' formulation.\n";
    }
  }
  
  /////////////////////////// "VGP_CONFORMING" VERSION ///////////////////////

  // fluxes and traces:
  VarPtr u1hat, u2hat, t1n, t2n;
  // fields for SGP:
  VarPtr phi, p, sigma11, sigma12, sigma21, sigma22;
  // fields specific to VGP:
  VarPtr u1, u2;
  
  BFPtr stokesBF;
  IPPtr qoptIP;
  
  double mu = 1;
  
  FunctionPtr h = Teuchos::rcp( new hFunction() );
  
  VarPtr tau1,tau2,v1,v2,q;
  VarFactory varFactory;
  if ( useCEFormulation ) {
    CEStokesFormulation stokesForm(mu);
    stokesBF = stokesForm.bf();
    qoptIP = stokesForm.graphNorm();
    
    varFactory = stokesForm.ceVarFactory();
    VarPtr v1 = varFactory.testVar(CE_V1_S, HGRAD);
    VarPtr v2 = varFactory.testVar(CE_V2_S, HGRAD);
    VarPtr q = varFactory.testVar(CE_Q_S, HGRAD);
    VarPtr tau1 = varFactory.testVar(CE_TAU1_S, HDIV);
    VarPtr tau2 = varFactory.testVar(CE_TAU2_S, HDIV);
    
    u1hat = varFactory.traceVar(CE_U1HAT_S);
    u2hat = varFactory.traceVar(CE_U2HAT_S);
    
    t1n = varFactory.fluxVar(CE_T1HAT_S);
    t2n = varFactory.fluxVar(CE_T2HAT_S);
    
    u1 = varFactory.fieldVar(CE_U1_S, HGRAD);
    u2 = varFactory.fieldVar(CE_U2_S, HGRAD);
    p = varFactory.fieldVar(CE_P_S);
  } else {
    tau1 = varFactory.testVar("\\tau_1", HDIV);
    tau2 = varFactory.testVar("\\tau_2", HDIV);
    v1 = varFactory.testVar("v_1", HGRAD);
    v2 = varFactory.testVar("v_2", HGRAD);
    q = varFactory.testVar("q", HGRAD);
    
    u1hat = varFactory.traceVar("\\widehat{u}_1");
    u2hat = varFactory.traceVar("\\widehat{u}_2");
    
    t1n = varFactory.fluxVar("\\widehat{t_{1n}}");
    t2n = varFactory.fluxVar("\\widehat{t_{2n}}");
    if (!useCompliantGraphNorm) {
      u1 = varFactory.fieldVar("u_1");
      u2 = varFactory.fieldVar("u_2");
    } else {
      u1 = varFactory.fieldVar("u_1", HGRAD);
      u2 = varFactory.fieldVar("u_2", HGRAD);
    }
    sigma11 = varFactory.fieldVar("\\sigma_11");
    sigma12 = varFactory.fieldVar("\\sigma_12");
    sigma21 = varFactory.fieldVar("\\sigma_21");
    sigma22 = varFactory.fieldVar("\\sigma_22");
    p = varFactory.fieldVar("p");
    
    stokesBF = Teuchos::rcp( new BF(varFactory) );  
    // tau1 terms:
    stokesBF->addTerm(u1,tau1->div());
    stokesBF->addTerm(sigma11,tau1->x()); // (sigma1, tau1)
    stokesBF->addTerm(sigma12,tau1->y());
    stokesBF->addTerm(-u1hat, tau1->dot_normal());
    
    // tau2 terms:
    stokesBF->addTerm(u2, tau2->div());
    stokesBF->addTerm(sigma21,tau2->x()); // (sigma2, tau2)
    stokesBF->addTerm(sigma22,tau2->y());
    stokesBF->addTerm(-u2hat, tau2->dot_normal());
    
    // v1:
    stokesBF->addTerm(mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
    stokesBF->addTerm(mu * sigma12,v1->dy());
    stokesBF->addTerm( - p, v1->dx() );
    stokesBF->addTerm( t1n, v1);
    
    // v2:
    stokesBF->addTerm(mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
    stokesBF->addTerm(mu * sigma22,v2->dy());
    stokesBF->addTerm( -p, v2->dy());
    stokesBF->addTerm( t2n, v2);
    
    if (! useCompliantGraphNorm) {
        // q:
      stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
      stokesBF->addTerm(-u2,q->dy());
      stokesBF->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);
    } else {
      // unsure whether I should divide q through by h in this equation
      // (I divide it in the test norm) -- the RHS here is 0, so mathematically the two are
      // equivalent.  Therefore conditioning should decide, and my conclusion is that
      // we're better conditioned without the division by h.
      // q:
      stokesBF->addTerm(-u1,q->dx());// / h); // (-u, grad q)
      stokesBF->addTerm(-u2,q->dy());// / h);
      stokesBF->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);// / h);
    }
  }
  
  if (rank==0)
    stokesBF->printTrialTestInteractions();
  
  stokesBF->setUseSPDSolveForOptimalTestFunctions(useSPDLocalSolve);
  stokesBF->setUseIterativeRefinementsWithSPDSolve(useIterativeRefinementsWithSPDSolve);
  stokesBF->setUseExtendedPrecisionSolveForOptimalTestFunctions(useExtendedPrecisionForOptimalTestInversion);

  ///////////////////////////////////////////////////////////////////////////
  SpatialFilterPtr nonOutflowBoundary = Teuchos::rcp( new NonOutflowBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowBoundary );
  
  int H1Order = polyOrder + 1;
  Teuchos::RCP<Mesh> mesh, streamMesh;
  
  FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
  A(0) = 0.0; A(1) = 1.0;
  B(0) = 0.0; B(1) = 2.0;
  C(0) = 4.0; C(1) = 2.0;
  D(0) = RIGHT_OUTFLOW; D(1) = 2.0;
  E(0) = RIGHT_OUTFLOW; E(1) = 1.0;
  F(0) = RIGHT_OUTFLOW; F(1) = 0.0;
  G(0) = 4.0; G(1) = 0.0;
  H(0) = 4.0; H(1) = 1.0;
  vector<FieldContainer<double> > vertices;
  vertices.push_back(A); int A_index = 0;
  vertices.push_back(B); int B_index = 1;
  vertices.push_back(C); int C_index = 2;
  vertices.push_back(D); int D_index = 3;
  vertices.push_back(E); int E_index = 4;
  vertices.push_back(F); int F_index = 5;
  vertices.push_back(G); int G_index = 6;
  vertices.push_back(H); int H_index = 7;
  vector< vector<int> > elementVertices;
  vector<int> el1, el2, el3, el4, el5;
  // left patch:
  el1.push_back(A_index); el1.push_back(H_index); el1.push_back(C_index); el1.push_back(B_index);
  // top right:
  el2.push_back(H_index); el2.push_back(E_index); el2.push_back(D_index); el2.push_back(C_index);
  // bottom right:
  el3.push_back(G_index); el3.push_back(F_index); el3.push_back(E_index); el3.push_back(H_index);
  
  elementVertices.push_back(el1);
  elementVertices.push_back(el2);
  elementVertices.push_back(el3);
  
  mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, stokesBF, H1Order, pToAdd) );
  
  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  
  ////////////////////   CREATE RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy ); // zero for now...
  
  IPPtr ip;
  
  if (! useCEFormulation ) {
    qoptIP = Teuchos::rcp(new IP());
    
    double beta = 1.0;
    
    if (useExperimentalHdivNorm) {
      qoptIP->addTerm( sqrt(beta) * v1 );
      qoptIP->addTerm( sqrt(beta) * v2 );
      qoptIP->addTerm( sqrt(beta) * q );
      qoptIP->addTerm( sqrt(beta) * tau1 );
      qoptIP->addTerm( sqrt(beta) * tau2 );
      
      qoptIP->addTerm( mu * v1->grad() + tau1 ); // sigma11, sigma12
      qoptIP->addTerm( mu * v2->grad() + tau2 ); // sigma21, sigma22
      qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
      qoptIP->addTerm( q->dx() );    // u1
      qoptIP->addTerm( q->dy() );    // u2    
    } else if (useExperimentalH1Norm) {
      qoptIP->addTerm( sqrt(beta) * v1 );
      qoptIP->addTerm( sqrt(beta) * v2 );
      qoptIP->addTerm( sqrt(beta) * q );
      qoptIP->addTerm( sqrt(beta) * tau1 );
      qoptIP->addTerm( sqrt(beta) * tau2 );
      
      // then we're "legally" allowed to reverse the integration by parts of both u1 and u2 terms...
      qoptIP->addTerm( mu * v1->grad() + tau1 ); // sigma11, sigma12
      qoptIP->addTerm( mu * v2->grad() + tau2 ); // sigma21, sigma22
      qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
      // For now, we add these in anyway...
      qoptIP->addTerm( tau1->div() );    // u1
      qoptIP->addTerm( tau2->div() );    // u2
    } else if (useCompliantGraphNorm) {
      qoptIP->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
      qoptIP->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
      qoptIP->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
      qoptIP->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
      qoptIP->addTerm( mu * v1->dx() + mu * v2->dy() );   // pressure
      qoptIP->addTerm( h * tau1->div() - h * q->dx() );   // u1
      qoptIP->addTerm( h * tau2->div() - h * q->dy());    // u2
      
      qoptIP->addTerm( (mu / h) * v1 );
      qoptIP->addTerm( (mu / h) * v2 );
      qoptIP->addTerm( q );
      qoptIP->addTerm( tau1 );
      qoptIP->addTerm( tau2 );
    } else { // some version of graph norm, then
      qoptIP->addTerm( sqrt(beta) * v1 );
      qoptIP->addTerm( sqrt(beta) * v2 );
      qoptIP->addTerm( sqrt(beta) * q );
      qoptIP->addTerm( sqrt(beta) * tau1 );
      qoptIP->addTerm( sqrt(beta) * tau2 );
      
      qoptIP = stokesBF->graphNorm();
      if (useGraphNormStrongerTau) { // "mix in" a bit of the H^1 experimental norm...
        qoptIP->addTerm( tau1->div() );    // u1
        qoptIP->addTerm( tau2->div() );    // u2
      }
    }
  }
  
  ip = qoptIP;
  
  if (rank==0) 
    ip->printInteractions();
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  
  BFPtr streamBF;
  SolutionPtr streamSolution;
  // define bilinear form for stream function:
  
  VarFactory streamVarFactory;
  VarPtr phi_hat = streamVarFactory.traceVar("\\widehat{\\phi}");
  VarPtr psin_hat = streamVarFactory.fluxVar("\\widehat{\\psi}_n");
  VarPtr psi_1 = streamVarFactory.fieldVar("\\psi_1");
  VarPtr psi_2 = streamVarFactory.fieldVar("\\psi_2");
  phi = streamVarFactory.fieldVar("\\phi");
  VarPtr q_s = streamVarFactory.testVar("q_s", HGRAD);
  VarPtr v_s = streamVarFactory.testVar("v_s", HDIV);
  streamBF = Teuchos::rcp( new BF(streamVarFactory) );
  streamBF->addTerm(psi_1, q_s->dx());
  streamBF->addTerm(psi_2, q_s->dy());
  streamBF->addTerm(-psin_hat, q_s);
  
  streamBF->addTerm(psi_1, v_s->x());
  streamBF->addTerm(psi_2, v_s->y());
  streamBF->addTerm(phi, v_s->div());
  streamBF->addTerm(-phi_hat, v_s->dot_normal());
  
  Teuchos::RCP<BCEasy> streamBC = Teuchos::rcp( new BCEasy );
  //  streamBC->addDirichlet(psin_hat, entireBoundary, u0_cross_n);
  Teuchos::RCP<SpatialFilter> wallBoundary = Teuchos::rcp( new WallBoundary );
  FunctionPtr phi0 = Teuchos::rcp( new PHI_0 );
  streamBC->addDirichlet(phi_hat, nonOutflowBoundary, phi0); // not sure this is right -- phi0 should probably be imposed on outflow, too...
  //  streamBC->addDirichlet(phi_hat, outflowBoundary, phi0);
  //  streamBC->addZeroMeanConstraint(phi);
  
  IPPtr streamIP = Teuchos::rcp( new IP );
  streamIP->addTerm(q_s);
  streamIP->addTerm(q_s->grad());
  streamIP->addTerm(v_s);
  streamIP->addTerm(v_s->div());
  
  streamBF->setUseSPDSolveForOptimalTestFunctions(useSPDLocalSolve);
  streamBF->setUseIterativeRefinementsWithSPDSolve(useIterativeRefinementsWithSPDSolve);
  streamBF->setUseExtendedPrecisionSolveForOptimalTestFunctions(useExtendedPrecisionForOptimalTestInversion);
  
  streamMesh = Teuchos::rcp( new Mesh(vertices, elementVertices, streamBF, H1Order, pToAddForStreamFunction) );
  
  streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC ) );
  
//  mesh->registerObserver(streamMesh); // will refine streamMesh in the same way as mesh.
  
  ////////////////////   CREATE BCs   ///////////////////////
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0 );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  bc->addDirichlet(u1hat, nonOutflowBoundary, u1_0);
  bc->addDirichlet(u2hat, nonOutflowBoundary, u2_0);
  // for now, prescribe same thing on the outflow boundary
  // TODO: replace with zero-traction condition
  // THIS IS A GUESS AT THE ZERO-TRACTION CONDITION
  cout << "SHOULD CONFIRM THAT THE ZERO-TRACTION CONDITION IS RIGHT!!\n";
  FunctionPtr zero = Function::zero();
  bc->addDirichlet(t1n, outflowBoundary, zero);
  bc->addDirichlet(t2n, outflowBoundary, zero);
  // hypothesis: when we impose the no-traction condition, not allowed to impose zero-mean pressure
//  bc->addZeroMeanConstraint(p);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  RefinementStrategy refinementStrategy( solution, energyThreshold, min_h );
  
  // just an experiment:
  //  refinementStrategy.setEnforceOneIrregurity(false);

  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
  mesh->registerObserver(refHistory);
  
  int uniformRefinements = 0;
  for (int refIndex=0; refIndex<uniformRefinements; refIndex++){
    refinementStrategy.refine();
  }
  
  // our elements now have aspect ratio 4:1.  We want to do 2 sets of horizontal refinements to square them up.
  // COMMENTING THESE LINES OUT AS A TEST: TODO: UNCOMMENT THEM.
//  if (rank==0)
//    cout << "NOTE: using anisotropic initial mesh.  Should change back after test complete!\n";
  Teuchos::RCP<RefinementPattern> verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuad();
  mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
  mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
  
  if (rank == 0) {
    cout << "Starting mesh has " << mesh->numActiveElements() << " elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    
    if (enforceLocalConservation) {
      cout << "Enforcing local conservation.\n";
    }
    if (useExperimentalHdivNorm) {
      cout << "NOTE: Using experimental H(div) norm!\n";
    }
    if (useExperimentalH1Norm) {
      cout << "NOTE: Using experimental H^1 norm!\n";
    }
    if (useCompliantGraphNorm) {
      cout << "NOTE: Using unit-compliant graph norm.\n";
    }
    if (useGraphNormStrongerTau) {
      cout << "NOTE: Using \"tau-strengthened\" graph norm.\n";
    }
    if (useExtendedPrecisionForOptimalTestInversion) {
      cout << "NOTE: using extended precision (long double) for Gram matrix inversion.\n";
    }
    if (finalSolveUsesStandardGraphNorm) {
      cout << "NOTE: will use standard graph norm for final solve.\n";
    }
  }
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->solve(false);
    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  
  // one more solve on the final refined mesh:
  solution->solve(false);
  double energyErrorTotal = solution->energyErrorTotal();
  double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, mesh, "bfs_maxConditionIPMatrix.dat");
  if (rank == 0) {
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "Max condition number estimate: " << maxConditionNumber << endl;
  }
  
  
  if (finalSolveUsesStandardGraphNorm) {
    if (rank==0)
      cout << "switching to graph norm for final solve";
    
    IPPtr ipToCompare = stokesBF->graphNorm();
    Teuchos::RCP<Solution> solutionToCompare = Teuchos::rcp( new Solution(mesh, bc, rhs, ipToCompare) );

    solutionToCompare->solve(false);
    
    FunctionPtr u1ToCompare = Function::solution(u1, solutionToCompare);
    FunctionPtr u2ToCompare = Function::solution(u2, solutionToCompare);
    
    FunctionPtr u1_soln = Function::solution(u1, solution);
    FunctionPtr u2_soln = Function::solution(u2, solution);
    
    double u1_l2difference = (u1ToCompare - u1_soln)->l2norm(mesh) / u1_soln->l2norm(mesh);
    double u2_l2difference = (u2ToCompare - u2_soln)->l2norm(mesh) / u2_soln->l2norm(mesh);
    
    double graph_maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ipToCompare, mesh, "bfs_maxConditionIPMatrix_graph.dat");
    
    if (rank==0) {
      cout << "L^2 differences with automatic graph norm:\n";
      cout << "    u1: " << u1_l2difference * 100 << "%" << endl;
      cout << "    u2: " << u2_l2difference * 100 << "%" << endl;
    }  
    solution = solutionToCompare;
    
    double energyErrorTotal = solution->energyErrorTotal();
    if (rank == 0) {
      cout << "Final energy error (standard graph norm): " << energyErrorTotal << endl;
      cout << "Max condition number estimate (standard graph norm): " << graph_maxConditionNumber << endl;
    }
  }
  
  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );
  Teuchos::RCP<RHSEasy> streamRHS = Teuchos::rcp( new RHSEasy );
  streamRHS->addTerm(vorticity * q_s);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
  
  
  //  FunctionPtr u1_sq = u1_prev * u1_prev;
  //  FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
  //  FunctionPtr u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dx() + u2->dy() ) );
  FunctionPtr massFlux = Teuchos::rcp( new PreviousSolutionFunction(solution, u1hat->times_normal_x() + u2hat->times_normal_y()) );
  
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
  double totalAbsMassFluxInterior = 0;
  double totalAbsMassFluxBoundary = 0;
  
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
      if (mesh->boundary().boundaryElement(cellID)) {
        totalAbsMassFluxBoundary += abs( massFluxIntegral[cellID] );
      } else {
        totalAbsMassFluxInterior += abs( massFluxIntegral[cellID] );
      }
    }
  }
  if (rank==0) {
    cout << "largest mass flux: " << maxMassFluxIntegral << endl;
    cout << "total mass flux: " << totalMassFlux << endl;
    cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
    cout << "sum of mass flux absolute value (interior elements): " << totalAbsMassFluxInterior << endl;
    cout << "sum of mass flux absolute value (boundary elements): " << totalAbsMassFluxBoundary << endl;
    cout << "largest h: " << sqrt(maxCellMeasure) << endl;
    cout << "smallest h: " << sqrt(minCellMeasure) << endl;
    cout << "ratio of largest / smallest h: " << sqrt(maxCellMeasure) / sqrt(minCellMeasure) << endl;
  }
  
  ///////// SET UP & SOLVE STREAM SOLUTION /////////
  streamSolution->setIP(streamIP);
  streamSolution->setRHS(streamRHS);
  
  refHistory->playback(streamMesh);
  
  if (rank == 0) {
    cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
    cout << "solving for approximate stream function...\n";
  }
  
  streamSolution->solve(false);
  energyErrorTotal = streamSolution->energyErrorTotal();
  // commenting out the recirculation region computation, because it doesn't work yet.
//  double x,y;
//  computeRecirculationRegion(x, y, streamSolution, phi);
  if (rank == 0) {  
    cout << "...solved.\n";
    cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
//    cout << "Recirculation region top: y=" << y << endl;
//    cout << "Recirculation region right: x=" << x << endl;
  }
  
  if (rank==0){
    massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
    solution->writeFieldsToFile(u1->ID(), "u1.m");
    solution->writeFieldsToFile(u2->ID(), "u2.m");
    streamSolution->writeFieldsToFile(phi->ID(), "phi.m");
    
    VTKExporter exporter(solution, mesh, varFactory);
    exporter.exportSolution("backStepSoln", H1Order*2);
    
    VTKExporter streamExporter(streamSolution, streamMesh, streamVarFactory);
    streamExporter.exportSolution("backStepStreamSoln", H1Order*2);
    
    solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
    solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
    solution->writeFieldsToFile(p->ID(), "p.m");

    writePatchValues(0, RIGHT_OUTFLOW, 0, 2, streamSolution, phi, "phi_patch.m");
    writePatchValues(4, 5, 0, 1, streamSolution, phi, "phi_patch_east.m");
    
    FieldContainer<double> eastPoints = pointGrid(4, RIGHT_OUTFLOW, 0, 2, 100);
    FieldContainer<double> eastPointData = solutionData(eastPoints, streamSolution, phi);
    GnuPlotUtil::writeXYPoints("phi_east.dat", eastPointData);

    FieldContainer<double> westPoints = pointGrid(0, 4, 1, 2, 100);
    FieldContainer<double> westPointData = solutionData(westPoints, streamSolution, phi);
    GnuPlotUtil::writeXYPoints("phi_west.dat", westPointData);
    
    set<double> contourLevels = diagonalContourLevels(eastPointData,4);
    
    vector<string> dataPaths;
    dataPaths.push_back("phi_east.dat");
    dataPaths.push_back("phi_west.dat");
    GnuPlotUtil::writeContourPlotScript(contourLevels, dataPaths, "backStepContourPlot.p");
    
    double xTics = 0.1, yTics = -1;
    FieldContainer<double> eastPatchPoints = pointGrid(4, 4.4, 0, 0.45, 200);
    FieldContainer<double> eastPatchPointData = solutionData(eastPatchPoints, streamSolution, phi);
    GnuPlotUtil::writeXYPoints("phi_patch_east.dat", eastPatchPointData);
    set<double> patchContourLevels = diagonalContourLevels(eastPatchPointData,4);
    // be sure to the 0 contour, where the direction should change:
    patchContourLevels.insert(0);
    
    vector<string> patchDataPath;
    patchDataPath.push_back("phi_patch_east.dat");
    GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "backStepEastContourPlot.p", xTics, yTics);
    
    GnuPlotUtil::writeComputationalMeshSkeleton("backStepMesh", mesh);
      
//      ofstream fout("phiContourLevels.dat");
//      fout << setprecision(15);
//      for (set<double>::iterator levelIt = contourLevels.begin(); levelIt != contourLevels.end(); levelIt++) {
//        fout << *levelIt << ", ";
//      }
//      fout.close();
    //    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
    //    writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
    
    cout << "wrote files.\n";
  }
  return 0;
}
