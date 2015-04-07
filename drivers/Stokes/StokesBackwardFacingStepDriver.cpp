//
//  StokesBackwardFacingStep.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/27/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "choice.hpp"
#include "mpi_choice.hpp"

#include "RefinementStrategy.h"
#include "RefinementPattern.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "GnuPlotUtil.h"

#include "HDF5Exporter.h"

#include "StokesFormulation.h"
#include "MeshUtilities.h"

#include "MeshPolyOrderFunction.h"

#include "RefinementPattern.h"
#include "BackwardFacingStepRefinementStrategy.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "RefinementHistory.h"
#include "PenaltyConstraints.h"

#include "StreamDriverUtil.h"

// Epetra includes
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

using namespace std;

static double tol=1e-14;

static double RIGHT_OUTFLOW = 8.0;
static double MESH_BOTTOM = 0.0;
static double LEFT_INFLOW = 0.0;

bool topWall(double x, double y) {
  return abs(y-2.0) < tol;
}

bool bottomWallRight(double x, double y) {
  return (abs(y-MESH_BOTTOM) < tol) && (x-4.0 >= tol);
}

bool bottomWallLeft(double x, double y) {
  return (abs(y-1.0) < tol) && (x-4.0 < tol);
}

bool step(double x, double y) {
  return (abs(x-4.0) < tol) && (y-1.0 <=tol);
}

bool inflow(double x, double y) {
  return abs(x-LEFT_INFLOW) < tol;
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
//      cout << "(" << x << ", " << y << "): imposing inflow condition on U1_0\n";
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

bool canReadFile(string fileName) {
  bool canRead = false;
  ifstream fin(fileName.c_str());
  if (fin.good())
  {
    canRead = true;
  }
  fin.close();
  return canRead;
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
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
  Epetra_SerialComm Comm;
#endif
  
#ifdef HAVE_MPI
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args(argc, argv );
#endif
  bool useExtendedPrecisionForOptimalTestInversion = false;
  bool useIterativeRefinementsWithSPDSolve = false;
  bool useSPDLocalSolve = false;
  bool finalSolveUsesStandardGraphNorm = false;
  
  int polyOrder = args.Input<int>("--polyOrder", "L^2 (field) polynomial order");
  int pToAdd = args.Input<int>("--pToAdd", "polynomial enrichment for test functions", 2);
  int pToAddForStreamFunction = pToAdd;
  
  int numRefs = args.Input<int>("--numRefs", "Number of refinements", 10);
  double energyThreshold = args.Input<double>("--adaptiveThreshold", "threshold parameter for greedy adaptivity", 0.20);
  bool enforceLocalConservation = args.Input<bool>("--enforceLocalConservation", "Enforce local conservation.", false);
  bool useCompliantGraphNorm = args.Input<bool>("--useCompliantNorm", "use the 'scale-compliant' graph norm", false);
  bool useExperimentalNorm = args.Input<bool>("--useExperimentalNorm", "use whatever the current experimental norm is", false);
  bool verboseRefinements = args.Input<bool>("--verboseRefinements", "verbose refinement output", false);
  
  bool useBiswasGeometry = args.Input<bool>("--useBiswasGeometry", "use an expansion ratio of 1.9423", false);
  
  int maxPolyOrder = args.Input<int>("--maxPolyOrder", "maximum polynomial order allowed in refinements", polyOrder);
  double min_h = args.Input<double>("--minh", "minimum element diameter for h-refinements", 0);
  bool induceCornerRefinements = args.Input<bool>("--induceCornerRefinements", "induce refinements in the recirculating corner", false);
  bool compareWithOverkill = args.Input<bool>("--compareWithOverkill", "compare with an overkill solution", false);
  int numOverkillRefinements = args.Input<int>("--numOverkillRefinements", "number of uniform refinements for overkill mesh compared with starting adaptive mesh", 4); // 4 for the final version --> 3072 elements with h=1/16
  int H1OrderOverkill = 1 + args.Input<int>("--overkillPolyOrder", "polynomial order for overkill solution", 5);
  string overkillSolnFile = args.Input<string>("--overkillSolnFile", "file to which to save / from which to load overkill solution.", "stokesBFSOverkill_3072_k5.soln");
  
  double eps = args.Input<double>("--testNormL2Weight", "weight for L^2 test terms (smaller often better)", 1.0);
  
  args.Process();
  
  // usage: polyOrder [numRefinements]
  // parse args:
//  if ((argc != 3) && (argc != 2) && (argc != 4)) {
//    cout << "Usage: StokesBackwardFacingStepDriver fieldPolyOrder [numRefinements=10 [adaptThresh=0.20]\n";
//    return -1;
//  }
//  int polyOrder = atoi(argv[1]);
//  int numRefs = 10;
//  if ( ( argc == 3 ) || (argc == 4)) {
//    numRefs = atoi(argv[2]);
//  }
//  if (argc == 4) {
//    energyThreshold = atof(argv[3]);
//  }
//  if (rank == 0) {
//    cout << "numRefinements = " << numRefs << endl;
//  }
  
  if (rank==0) {
    cout << "polyOrder = " << polyOrder << endl;
    if (useBiswasGeometry) {
      cout << "using Biswas geometry.\n";
    }
    if (compareWithOverkill) {
      cout << "Will compare with overkill mesh.\n";
    }
    if (verboseRefinements) {
      cout << "Verbose refinements is true.\n";
    }
  }
  
  if (useBiswasGeometry) {
    MESH_BOTTOM = 1.0 - 0.9423;
    RIGHT_OUTFLOW = 7.0;
    LEFT_INFLOW = 3.0;
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
  
    // q:
  stokesBF->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBF->addTerm(-u2,q->dy());
  stokesBF->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);
  
//  if (rank==0)
//    stokesBF->printTrialTestInteractions();
  
  stokesBF->setUseSPDSolveForOptimalTestFunctions(useSPDLocalSolve);
  stokesBF->setUseIterativeRefinementsWithSPDSolve(useIterativeRefinementsWithSPDSolve);
  stokesBF->setUseExtendedPrecisionSolveForOptimalTestFunctions(useExtendedPrecisionForOptimalTestInversion);

  ///////////////////////////////////////////////////////////////////////////
  SpatialFilterPtr nonOutflowBoundary = Teuchos::rcp( new NonOutflowBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowBoundary );
  
  int H1Order = polyOrder + 1;
  Teuchos::RCP<Mesh> mesh, streamMesh;
  
  vector<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
  A[0] = LEFT_INFLOW; A[1] = 1.0;
  B[0] = LEFT_INFLOW; B[1] = 2.0;
  C[0] = 4.0; C[1] = 2.0;
  D[0] = RIGHT_OUTFLOW; D[1] = 2.0;
  E[0] = RIGHT_OUTFLOW; E[1] = 1.0;
  F[0] = RIGHT_OUTFLOW; F[1] = MESH_BOTTOM;
  G[0] = 4.0; G[1] = MESH_BOTTOM;
  H[0] = 4.0; H[1] = 1.0;
  vector< vector<double> > vertices;
  vertices.push_back(A); int A_index = 0;
  vertices.push_back(B); int B_index = 1;
  vertices.push_back(C); int C_index = 2;
  vertices.push_back(D); int D_index = 3;
  vertices.push_back(E); int E_index = 4;
  vertices.push_back(F); int F_index = 5;
  vertices.push_back(G); int G_index = 6;
  vertices.push_back(H); int H_index = 7;
  vector< vector<IndexType> > elementVertices;
  vector<IndexType> el1, el2, el3, el4, el5;
  // left patch:
  el1.push_back(A_index); el1.push_back(H_index); el1.push_back(C_index); el1.push_back(B_index);
  // top right:
  el2.push_back(H_index); el2.push_back(E_index); el2.push_back(D_index); el2.push_back(C_index);
  // bottom right:
  el3.push_back(G_index); el3.push_back(F_index); el3.push_back(E_index); el3.push_back(H_index);
  
  elementVertices.push_back(el1);
  elementVertices.push_back(el2);
  elementVertices.push_back(el3);
  
  FieldContainer<double> bottomCornerPoint(1,2);
  // want to cheat just inside the bottom corner element:
  bottomCornerPoint(0,0) = G[0] + 1e-10;
  bottomCornerPoint(0,1) = G[1] + 1e-10;
  
  /*
   Mesh(const vector<vector<double> > &vertices, vector< vector<IndexType> > &elementVertices,
   BFPtr bilinearForm, int H1Order, int pToAddTest, bool useConformingTraces = true,
   map<int,int> trialOrderEnhancements=_emptyIntIntMap, map<int,int> testOrderEnhancements=_emptyIntIntMap,
   vector< PeriodicBCPtr > periodicBCs = vector< PeriodicBCPtr >());
   */
  
  
  mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, stokesBF, H1Order, pToAdd) );
  
  FunctionPtr one = Function::constant(1.0);
  double meshMeasure = one->integrate(mesh);
  
  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
  mesh->registerObserver(refHistory);
  
  Teuchos::RCP<RefinementPattern> verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuad();
  if (! useBiswasGeometry) {
    // our elements now have aspect ratio 4:1.  We want to do 2 sets of horizontal refinements to square them up.
    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
  } else {
    // for the Biswas geometry, the only thing we can conveniently do is approximate squares.
    FieldContainer<double> inflowPoint(1,2);
    inflowPoint(0,0) = A[0] + 1e-10;
    inflowPoint(0,1) = A[1] + 1e-10;
    
    int inflowCell = mesh->elementsForPoints(inflowPoint)[0]->cellID();
    set<GlobalIndexType> activeCellIDs = mesh->getActiveCellIDs();
    activeCellIDs.erase(activeCellIDs.find(inflowCell));
    mesh->hRefine(activeCellIDs, verticalCut);
  }
  
  MeshPtr overkillMesh;
  if (compareWithOverkill) {
    overkillMesh = Teuchos::rcp( new Mesh(vertices, elementVertices, stokesBF, H1OrderOverkill, pToAdd) );
    overkillMesh->hRefine(overkillMesh->getActiveCellIDs(), verticalCut);
    overkillMesh->hRefine(overkillMesh->getActiveCellIDs(), verticalCut);
    
    for (int i=0; i<numOverkillRefinements; i++) {
      overkillMesh->hRefine(overkillMesh->getActiveCellIDs(), RefinementPattern::regularRefinementPatternQuad());
    }
    if (rank==0) {
      cout << "Overkill mesh has " << overkillMesh->numActiveElements() << " elements and ";
      cout << overkillMesh->numGlobalDofs() << " dofs.\n";
    }
  }
  
  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  
  ////////////////////   CREATE RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs(); // zero for now...
  
  IPPtr ip;
  
  qoptIP = Teuchos::rcp(new IP());
  
  if (useCompliantGraphNorm) {
    qoptIP->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
    qoptIP->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
    qoptIP->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
    qoptIP->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
    qoptIP->addTerm( mu * v1->dx() + mu * v2->dy() );   // pressure
    qoptIP->addTerm( h * tau1->div() - h * q->dx() );   // u1
    qoptIP->addTerm( h * tau2->div() - h * q->dy());    // u2
    
    qoptIP->addTerm( eps * (mu / h) * v1 );
    qoptIP->addTerm( eps * (mu / h) * v2 );
    qoptIP->addTerm( eps * q );
    qoptIP->addTerm( eps * tau1 );
    qoptIP->addTerm( eps * tau2 );
  } else if (useExperimentalNorm) {
    qoptIP->addTerm( v1->dx() + tau1->x() ); // sigma11
    qoptIP->addTerm( v1->dy() + tau1->y() ); // sigma12
    qoptIP->addTerm( v2->dx() + tau2->x() ); // sigma21
    qoptIP->addTerm( v2->dy() + tau2->y() ); // sigma22
    qoptIP->addTerm( v1->dx() + v2->dy() );   // pressure
    
    double eps = 1e-4;
    qoptIP->addTerm( eps * v1 );
    qoptIP->addTerm( eps * v2 );
    qoptIP->addTerm( eps * q );
    qoptIP->addTerm( eps * tau1 );
    qoptIP->addTerm( eps * tau2 );
  } else { // some version of graph norm, then
    qoptIP = stokesBF->graphNorm(eps);
  }
  
  ip = qoptIP;
  
  if (rank==0) 
    ip->printInteractions();
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  
  SolutionPtr overkillSolution;
  map<int, double> dofsToL2error; // key: numGlobalDofs, value: total L2error compared with overkill
  map<int, double> dofsToBestL2error;
  
  vector< VarPtr > fields;
  fields.push_back(u1);
  fields.push_back(u2);
  fields.push_back(sigma11);
  fields.push_back(sigma12);
  fields.push_back(sigma21);
  fields.push_back(sigma22);
  fields.push_back(p);
  
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
  
  BCPtr streamBC = BC::bc();
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
  
  // will use refinement history to playback refinements on streamMesh (no need to register streamMesh)
  
  ////////////////////   CREATE BCs   ///////////////////////
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0 );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  
  if (false) {
    cout << "EXPERIMENTING: replacing Dirichlet with penalty constraints.\n";
    
    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
    pc->addConstraint(u1hat==u1_0,nonOutflowBoundary);
    pc->addConstraint(u2hat==u2_0,nonOutflowBoundary);
    
    pc->addConstraint(t1n==Function::zero(),outflowBoundary);
    pc->addConstraint(t2n==Function::zero(),outflowBoundary);
    
    solution->setFilter(pc);
  } else {
    bc->addDirichlet(u1hat, nonOutflowBoundary, u1_0);
    bc->addDirichlet(u2hat, nonOutflowBoundary, u2_0);

    // impose zero-traction condition:
    FunctionPtr zero = Function::zero();
    bc->addDirichlet(t1n, outflowBoundary, zero);
    bc->addDirichlet(t2n, outflowBoundary, zero);
    // when we impose the no-traction condition, not allowed to impose zero-mean pressure
    // bc->addZeroMeanConstraint(p);
  }
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  double L2normOverkill = -1;
  if (compareWithOverkill) {
    overkillSolution = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
    
    if (enforceLocalConservation) {
      overkillSolution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==Function::zero());
    }
    
    if ((overkillSolnFile.length() > 0) && canReadFile(overkillSolnFile)) {
      // then load solution from file, and skip solve
      if (rank==0) {
        cout << "Loading overkill solution from " << overkillSolnFile << "." << endl;
      }
      overkillSolution = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
      overkillSolution->readFromFile(overkillSolnFile);
      if (rank==0) {
        cout << "Loaded." << endl;
      }
    } else {
      if (rank==0) {
        cout << "Solving on overkill mesh";
      }
      Epetra_Time timer(Comm);
      if (!enforceLocalConservation) {
        if (rank==0) {
          cout << " (using condensed solve)... " << endl;
        }
        overkillSolution->condensedSolve();
      } else {
        if (rank==0) {
          cout << "... " << endl;
        }
        // condensed solve doesn't support lagrange constraints yet...
        overkillSolution->solve(true);
      }
      
      double overkillSolutionTime = timer.ElapsedTime();
      if (rank==0) {
        cout << "... solved in " << overkillSolutionTime << " seconds." << endl;
      }
      
      if (rank == 0) {
        if (overkillSolnFile.length() > 0) {
          cout << "writing to disk...\n";
          overkillSolution->writeToFile(overkillSolnFile);
          cout << "Wrote overkill solution to " << overkillSolnFile << endl;
        }
      }

    }
    double overkillEnergyError = overkillSolution->energyErrorTotal();
    if (rank == 0)
      cout << "overkill energy error: " << overkillEnergyError << endl;
    
    double L2normSquared = 0;
    for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
      VarPtr var = *fieldIt;
      FunctionPtr fieldFxn = Function::solution(var, overkillSolution);
      if (var->ID() == p->ID()) {
        // pressure: subtract off the average value:
        double pAvg = fieldFxn->integrate(overkillMesh) / meshMeasure;
        fieldFxn = fieldFxn - pAvg;
      }
      
      double L2norm = fieldFxn->l2norm(overkillMesh);
      L2normSquared += L2norm * L2norm;
      if (rank==0) {
        cout << "L^2 norm for overkill solution of field " << var->name() << ": " << L2norm << endl;
      }
    }
    L2normOverkill = sqrt(L2normSquared);
    if (rank==0) {
      cout << "L^2 norm of all overkill fields: " << L2normOverkill << endl;
    }
  }
  
  Teuchos::RCP<BackwardFacingStepRefinementStrategy> bfsRefinementStrategy = Teuchos::rcp( new BackwardFacingStepRefinementStrategy(solution, energyThreshold,
                                                                                                                                    min_h, maxPolyOrder,
                                                                                                                                    (rank==0) && verboseRefinements) );
  bfsRefinementStrategy->addCorner(G[0], G[1]);
  bfsRefinementStrategy->addCorner(H[0], H[1]);
  
//  Teuchos::RCP<RefinementStrategy> refinementStrategy( solution, energyThreshold, min_h );
  
  // just an experiment:
  //  refinementStrategy.setEnforceOneIrregurity(false);
  
  if (rank == 0) {
    cout << "Starting mesh has " << mesh->numActiveElements() << " elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    
    if (enforceLocalConservation) {
      cout << "Enforcing local conservation.\n";
    }
    if (useCompliantGraphNorm) {
      cout << "NOTE: Using unit-compliant graph norm.\n";
    }
    if (useExtendedPrecisionForOptimalTestInversion) {
      cout << "NOTE: using extended precision (long double) for Gram matrix inversion.\n";
    }
    if (finalSolveUsesStandardGraphNorm) {
      cout << "NOTE: will use standard graph norm for final solve.\n";
    }
  }
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    if (!enforceLocalConservation) {
      solution->condensedSolve();
    } else {
      // condensed solve doesn't support lagrange constraints yet...
      solution->solve(true);
    }
    if (compareWithOverkill) {
      Teuchos::RCP<Solution> bestSoln = Teuchos::rcp( new Solution(solution->mesh(), bc, rhs, ip) );
      overkillSolution->projectFieldVariablesOntoOtherSolution(bestSoln);
      Teuchos::RCP<Solution> bestSolnOnOverkillMesh = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
      bestSoln->projectFieldVariablesOntoOtherSolution(bestSolnOnOverkillMesh);
      
      // determine error as difference between our solution and overkill
      bestSolnOnOverkillMesh->addSolution(overkillSolution,-1.0);
      
      Teuchos::RCP<Solution> adaptiveSolnOnOverkillMesh = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
      solution->projectFieldVariablesOntoOtherSolution(adaptiveSolnOnOverkillMesh);
      
      // determine error as difference between our solution and overkill
      adaptiveSolnOnOverkillMesh->addSolution(overkillSolution,-1.0);
      
      double L2errorSquared = 0.0;
      double bestL2errorSquared = 0.0;
      for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
        VarPtr var = *fieldIt;
        int fieldID = var->ID();
        FunctionPtr fieldErrorFxn = Function::solution(var, adaptiveSolnOnOverkillMesh);
        
        double L2error = fieldErrorFxn->l2norm(adaptiveSolnOnOverkillMesh->mesh());
        L2errorSquared += L2error * L2error;
        double bestL2error = bestSolnOnOverkillMesh->L2NormOfSolutionGlobal(fieldID);
        bestL2errorSquared += bestL2error * bestL2error;
        if (rank==0) {
          cout << "L^2 error for " << var->name() << ": " << L2error;
          cout << " (vs. best error of " << bestL2error << ")\n";
        }
      }
      int numGlobalDofs = mesh->numGlobalDofs();
      if (rank==0) {
        cout << "for " << numGlobalDofs << " dofs, total L2 error: " << sqrt(L2errorSquared);
        cout << " (vs. best error of " << sqrt(bestL2errorSquared) << ")\n";
      }
      dofsToL2error[numGlobalDofs] = sqrt(L2errorSquared) / L2normOverkill;
      dofsToBestL2error[numGlobalDofs] = sqrt(bestL2errorSquared) / L2normOverkill;
//      if (rank==0) {
//        VTKExporter exporter(adaptiveSolnOnOverkillMesh, mesh, varFactory);
//        ostringstream errorForRefinement;
//        errorForRefinement << "overkillError_refinement_" << refIndex;
//        exporter.exportSolution(errorForRefinement.str());
//      }
    }

    bfsRefinementStrategy->refine(rank==0); // print to console on rank 0
    
    if (induceCornerRefinements) {
      // induce refinements in bottom corner:
      vector< Teuchos::RCP<Element> > corners = mesh->elementsForPoints(bottomCornerPoint);
      vector<GlobalIndexType> cornerIDs;
      cornerIDs.push_back(corners[0]->cellID());
      mesh->hRefine(cornerIDs, RefinementPattern::regularRefinementPatternQuad());
    }
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
  RHSPtr streamRHS = RHS::rhs();
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
  
  CellTopoPtr quad = CellTopology::quad();
  DofOrderingFactory dofOrderingFactory(fakeBF);
  int fakeTestOrder = H1Order;
  DofOrderingPtr testOrdering = dofOrderingFactory.testOrdering(fakeTestOrder, quad);
  
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
//  for (vector< ElementTypePtr >::iterator elemTypeIt = elemTypes.begin(); elemTypeIt != elemTypes.end(); elemTypeIt++) {
//    ElementTypePtr elemType = *elemTypeIt;
//    vector< ElementPtr > elems = mesh->elementsOfTypeGlobal(elemType);
//    vector<GlobalIndexType> cellIDs;
//    for (int i=0; i<elems.size(); i++) {
//      cellIDs.push_back(elems[i]->cellID());
//    }
//    FieldContainer<double> physicalCellNodes = mesh->physicalCellNodesGlobal(elemType);
//    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh) );
//    basisCache->setPhysicalCellNodes(physicalCellNodes,cellIDs,true); // true: create side caches
//    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
//    FieldContainer<double> fakeRHSIntegrals(elems.size(),testOrdering->totalDofs());
//    massFluxTerm->integrate(fakeRHSIntegrals,testOrdering,basisCache,true); // true: force side evaluation
//    //      cout << "fakeRHSIntegrals:\n" << fakeRHSIntegrals;
//    for (int i=0; i<elems.size(); i++) {
//      int cellID = cellIDs[i];
//      // pick out the ones for testOne:
//      massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
//    }
//    //      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
//    //        for (int i=0; i<elems.size(); i++) {
//    //          int cellID = cellIDs[i];
//    //          // pick out the ones for testOne:
//    //          massFluxIntegral[cellID] += fakeRHSIntegrals(i,testOneIndex);
//    //        }
//    //      }
//    // find the largest:
//    for (int i=0; i<elems.size(); i++) {
//      int cellID = cellIDs[i];
//      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
//    }
//    for (int i=0; i<elems.size(); i++) {
//      int cellID = cellIDs[i];
//      maxCellMeasure = max(maxCellMeasure,cellMeasures(i));
//      minCellMeasure = min(minCellMeasure,cellMeasures(i));
//      maxMassFluxIntegral = max(abs(massFluxIntegral[cellID]), maxMassFluxIntegral);
//      totalMassFlux += massFluxIntegral[cellID];
//      totalAbsMassFlux += abs( massFluxIntegral[cellID] );
//      if (mesh->boundary().boundaryElement(cellID)) {
//        totalAbsMassFluxBoundary += abs( massFluxIntegral[cellID] );
//      } else {
//        totalAbsMassFluxInterior += abs( massFluxIntegral[cellID] );
//      }
//    }
//  }
//  if (rank==0) {
//    cout << "largest mass flux: " << maxMassFluxIntegral << endl;
//    cout << "total mass flux: " << totalMassFlux << endl;
//    cout << "sum of mass flux absolute value: " << totalAbsMassFlux << endl;
//    cout << "sum of mass flux absolute value (interior elements): " << totalAbsMassFluxInterior << endl;
//    cout << "sum of mass flux absolute value (boundary elements): " << totalAbsMassFluxBoundary << endl;
//    cout << "largest h: " << sqrt(maxCellMeasure) << endl;
//    cout << "smallest h: " << sqrt(minCellMeasure) << endl;
//    cout << "ratio of largest / smallest h: " << sqrt(maxCellMeasure) / sqrt(minCellMeasure) << endl;
//  }
  
  ///////// SET UP & SOLVE STREAM SOLUTION /////////
  streamSolution->setIP(streamIP);
  streamSolution->setRHS(streamRHS);
  
  refHistory->playback(streamMesh);
  
  if (rank == 0) {
    cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
    cout << "solving for approximate stream function...\n";
  }
  
  streamSolution->condensedSolve();
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

  HDF5Exporter exporter(mesh, "backStepSoln", ".");
  exporter.exportSolution(solution);

  HDF5Exporter exporterVorticity(mesh, "backStepVorticity", ".");
  exporterVorticity.exportFunction(vorticity, "vorticity");
  
  HDF5Exporter streamExporter(streamMesh, "backStepStreamSoln", ".");
  streamExporter.exportSolution(streamSolution);

  
//  if (rank==0){
////    massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
////    solution->writeFieldsToFile(u1->ID(), "u1.m");
////    solution->writeFieldsToFile(u2->ID(), "u2.m");
////    streamSolution->writeFieldsToFile(phi->ID(), "phi.m");
//    
////    FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
////    exporter.exportFunction(polyOrderFunction,"backStepPolyOrders");
//    
////    solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
////    solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
////    solution->writeFieldsToFile(p->ID(), "p.m");
//
////    writePatchValues(0, RIGHT_OUTFLOW, 0, 2, streamSolution, phi, "phi_patch.m");
////    writePatchValues(4, 5, 0, 1, streamSolution, phi, "phi_patch_east.m");
//    
//    FieldContainer<double> eastPoints = pointGrid(4, RIGHT_OUTFLOW, 0, 2, 100);
//    FieldContainer<double> eastPointData = solutionData(eastPoints, streamSolution, phi);
//    GnuPlotUtil::writeXYPoints("phi_east.dat", eastPointData);
//
//    FieldContainer<double> westPoints = pointGrid(0, 4, 1, 2, 100);
//    FieldContainer<double> westPointData = solutionData(westPoints, streamSolution, phi);
//    GnuPlotUtil::writeXYPoints("phi_west.dat", westPointData);
//    
//    set<double> contourLevels = diagonalContourLevels(eastPointData,4);
//    
//    vector<string> dataPaths;
//    dataPaths.push_back("phi_east.dat");
//    dataPaths.push_back("phi_west.dat");
//    GnuPlotUtil::writeContourPlotScript(contourLevels, dataPaths, "backStepContourPlot.p");
//    
//    double xTics = 0.1, yTics = -1;
//    FieldContainer<double> eastPatchPoints = pointGrid(4, 4.4, 0, 0.45, 200);
//    FieldContainer<double> eastPatchPointData = solutionData(eastPatchPoints, streamSolution, phi);
//    GnuPlotUtil::writeXYPoints("phi_patch_east.dat", eastPatchPointData);
//    set<double> patchContourLevels = diagonalContourLevels(eastPatchPointData,4);
//    // be sure to the 0 contour, where the direction should change:
//    patchContourLevels.insert(0);
//    
//    vector<string> patchDataPath;
//    patchDataPath.push_back("phi_patch_east.dat");
//    GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "backStepEastContourPlot.p", xTics, yTics);
//
//    {
//      map<double,string> scaleToName;
//      scaleToName[1]   = "bfsPatch";
//      scaleToName[0.4] = "bfsPatchEddy1";
//      scaleToName[0.05] = "bfsPatchEddy2";
//      scaleToName[0.005] = "bfsPatchEddy3";
//      
//      for (map<double,string>::iterator entryIt=scaleToName.begin(); entryIt != scaleToName.end(); entryIt++) {
//        double scale = entryIt->first;
//        string name = entryIt->second;
//        ostringstream fileNameStream;
//        fileNameStream << name << ".dat";
//        FieldContainer<double> patchPoints = pointGrid(4, 4+scale, MESH_BOTTOM, MESH_BOTTOM + scale, 200);
//        FieldContainer<double> patchPointData = solutionData(patchPoints, streamSolution, phi);
//        GnuPlotUtil::writeXYPoints(fileNameStream.str(), patchPointData);
//        ostringstream scriptNameStream;
//        scriptNameStream << name << ".p";
//        set<double> contourLevels = diagonalContourLevels(patchPointData,4);
//        vector<string> dataPaths;
//        dataPaths.push_back(fileNameStream.str());
//        GnuPlotUtil::writeContourPlotScript(contourLevels, dataPaths, scriptNameStream.str());
//      }
//      
//      double xTics = 0.1, yTics = -1;
//      FieldContainer<double> eastPatchPoints = pointGrid(4, 4.4, MESH_BOTTOM, MESH_BOTTOM + 0.45, 200);
//      FieldContainer<double> eastPatchPointData = solutionData(eastPatchPoints, streamSolution, phi);
//      GnuPlotUtil::writeXYPoints("phi_patch_east.dat", eastPatchPointData);
//      set<double> patchContourLevels = diagonalContourLevels(eastPatchPointData,4);
//      // be sure to the 0 contour, where the direction should change:
//      patchContourLevels.insert(0);
//      
//      vector<string> patchDataPath;
//      patchDataPath.push_back("phi_patch_east.dat");
//      GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "backStepEastContourPlot.p", xTics, yTics);
//    }
//    
//    GnuPlotUtil::writeComputationalMeshSkeleton("backStepMesh", mesh);
//    
////      ofstream fout("phiContourLevels.dat");
////      fout << setprecision(15);
////      for (set<double>::iterator levelIt = contourLevels.begin(); levelIt != contourLevels.end(); levelIt++) {
////        fout << *levelIt << ", ";
////      }
////      fout.close();
//    //    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
//    //    writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
//    
//    
//    if (compareWithOverkill) {
//      if (rank==0) {
//        cout << "******* Adaptivity Convergence Report *******\n";
//        cout << "dofs\tL2 error\n";
//        for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
//          int dofs = entryIt->first;
//          double err = entryIt->second;
//          cout << dofs << "\t" << err;
//          double bestError = dofsToBestL2error[dofs];
//          cout << "\t" << bestError << endl;
//        }
//        ofstream fout("stokesBFSOverkillComparison.txt");
//        fout << "dofs\tsoln_error\tbest_error\n";
//        for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
//          int dofs = entryIt->first;
//          double err = entryIt->second;
//          fout << dofs << "\t" << err;
//          double bestError = dofsToBestL2error[dofs];
//          fout << "\t" << bestError << endl;
//        }
//        fout.close();
//      }
//    }
//    
//    cout << "wrote files.\n";
//  }
  return 0;
}
