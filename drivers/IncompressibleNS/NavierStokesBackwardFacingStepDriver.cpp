//
//  NavierStokesBackwardFacingStep.cpp
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
#include "SolutionExporter.h"

#include "NavierStokesFormulation.h"
#include "MeshUtilities.h"

#include "MeshPolyOrderFunction.h"

#include "RefinementPattern.h"
#include "BackwardFacingStepRefinementStrategy.h"

#include "ParameterFunction.h"

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
      return 9 * y * y - 2 * y * y * y - 12 * y + 5; // 5 just to make it 0 at y=1
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
  
  double Re = args.Input<double>("--Re", "Reynolds number", 100);
  
  int numRefs = args.Input<int>("--numRefs", "Number of refinements", 10);
  double energyThreshold = args.Input<double>("--adaptiveThreshold", "threshold parameter for greedy adaptivity", 0.20);
  bool enforceLocalConservation = args.Input<bool>("--enforceLocalConservation", "Enforce local conservation.", false);
  bool useCompliantGraphNorm = args.Input<bool>("--useCompliantNorm", "use the 'scale-compliant' graph norm", false);
  bool useExperimentalNorm = args.Input<bool>("--useExperimentalNorm", "use whatever the current experimental norm is", false);
  bool longDoubleGramInversion = args.Input<bool>("--longDoubleGramInversion", "use long double Cholesky factorization for Gram matrix", false);
  
  bool outputStiffnessMatrix = args.Input<bool>("--writeFinalStiffnessToDisk", "write the final stiffness matrix to disk.", false);
  bool computeMaxConditionNumber = args.Input<bool>("--computeMaxConditionNumber", "compute the maximum Gram matrix condition number for final mesh.", false);
  bool useCondensedSolve = args.Input<bool>("--useCondensedSolve", "use static condensation", true);
  bool reportConditionNumber = args.Input<bool>("--reportGlobalConditionNumber", "report the 2-norm condition number for the global system matrix", false);
  
  bool useBiswasGeometry = args.Input<bool>("--useBiswasGeometry", "use an expansion ratio of 1.9423", false);
  
  double dt = args.Input<double>("--timeStep", "time step (0 for none)", 0); // 0.5 used to be the standard value
  
  int numUniformRefinements = args.Input<int>("--initialUniformRefinements", "Number of uniform refinements to perform before starting adaptive refinements", 0);
  
  int maxPolyOrder = args.Input<int>("--maxPolyOrder", "maximum polynomial order allowed in refinements", polyOrder);
  double min_h = args.Input<double>("--minh", "minimum element diameter for h-refinements", 0);
  bool induceCornerRefinements = args.Input<bool>("--induceCornerRefinements", "induce refinements in the recirculating corner", false);
  int numCornerRefinementsToInduce = args.Input<int>("--numCornerRefinementsToInduce", "number of refinements to induce in the recirculating corner", 5);
  bool compareWithOverkill = args.Input<bool>("--compareWithOverkill", "compare with an overkill solution", false);
  int numOverkillRefinements = args.Input<int>("--numOverkillRefinements", "number of uniform refinements for overkill mesh compared with starting adaptive mesh", 4); // 4 for the final version --> 3072 elements with h=1/16
  int H1OrderOverkill = 1 + args.Input<int>("--overkillPolyOrder", "polynomial order for overkill solution", 5);
  
  bool useTractionBCsOnOutflow = args.Input<bool>("--useTractionBCsOnOutflow", "impose zero-traction boundary conditions on outflow (otherwise, impose zero-mean pressure)", true);
  
  bool weightIncrementL2Norm = useCompliantGraphNorm; // if using the compliant graph norm, weight the measure of the L^2 increment accordingly
  
  int maxIters = args.Input<int>("--maxIters", "maximum number of Newton-Raphson iterations to take to try to match tolerance", 50);
  double minL2Increment = args.Input<double>("--NRtol", "Newton-Raphson tolerance, L^2 norm of increment", 3e-8);
  string replayFile = args.Input<string>("--replayFile", "file with refinement history to replay", "");
  string solnFile = args.Input<string>("--solnFile", "file with solution data", "");
  string solnSaveFile = args.Input<string>("--solnSaveFile", "file to which to save solution data", "");
  string saveFile = args.Input<string>("--saveReplay", "file to which to save refinement history", "");
  
  double finalSolveMinL2Increment = args.Input<double>("--finalNRtol", "Newton-Raphson tolerance for final solve, L^2 norm of increment", minL2Increment / 10);
  
  args.Process();
  
  if (useBiswasGeometry) {
    MESH_BOTTOM = 1.0 - 0.9423;
    RIGHT_OUTFLOW = 7.0;
    LEFT_INFLOW = 3.0;
  }
  
  bool enforceOneIrregularity = true;
  bool reportPerCellErrors  = true;
  bool useMumps = true;
  bool compareWithOverkillMesh = false;
  bool useAdHocHPRefinements = false;
  bool startWithZeroSolutionAfterRefinement = false;
  
  bool useLineSearch = false;
  
  bool artificialTimeStepping = (dt > 0);

  if (rank == 0) {
    cout << "numRefinements = " << numRefs << endl;
    cout << "Re = " << Re << endl;
    if (artificialTimeStepping) cout << "dt = " << dt << endl;
    if (!startWithZeroSolutionAfterRefinement) {
      cout << "NOT starting with 0 solution after refinement...\n";
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
    
  FunctionPtr h = Teuchos::rcp( new hFunction() );
  
  VarPtr tau1,tau2,v1,v2,q;
  // get variable definitions:
  VarFactory varFactory = VGPStokesFormulation::vgpVarFactory();
  u1 = varFactory.fieldVar(VGP_U1_S);
  u2 = varFactory.fieldVar(VGP_U2_S);
  sigma11 = varFactory.fieldVar(VGP_SIGMA11_S);
  sigma12 = varFactory.fieldVar(VGP_SIGMA12_S);
  sigma21 = varFactory.fieldVar(VGP_SIGMA21_S);
  sigma22 = varFactory.fieldVar(VGP_SIGMA22_S);
  p = varFactory.fieldVar(VGP_P_S);
  
  u1hat = varFactory.traceVar(VGP_U1HAT_S);
  u2hat = varFactory.traceVar(VGP_U2HAT_S);
  t1n = varFactory.fluxVar(VGP_T1HAT_S);
  t2n = varFactory.fluxVar(VGP_T2HAT_S);
  
  v1 = varFactory.testVar(VGP_V1_S, HGRAD);
  v2 = varFactory.testVar(VGP_V2_S, HGRAD);
  tau1 = varFactory.testVar(VGP_TAU1_S, HDIV);
  tau2 = varFactory.testVar(VGP_TAU2_S, HDIV);
  q = varFactory.testVar(VGP_Q_S, HGRAD);
  
  FunctionPtr zero = Function::zero();
  
  ///////////////////////////////////////////////////////////////////////////
  SpatialFilterPtr nonOutflowBoundary = Teuchos::rcp( new NonOutflowBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowBoundary );
  
  int H1Order = polyOrder + 1;
  
  FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
  A(0) = LEFT_INFLOW; A(1) = 1.0;
  B(0) = LEFT_INFLOW; B(1) = 2.0;
  C(0) = 4.0; C(1) = 2.0;
  D(0) = RIGHT_OUTFLOW; D(1) = 2.0;
  E(0) = RIGHT_OUTFLOW; E(1) = 1.0;
  F(0) = RIGHT_OUTFLOW; F(1) = MESH_BOTTOM;
  G(0) = 4.0; G(1) = MESH_BOTTOM;
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
  
  FieldContainer<double> bottomCornerPoint(1,2);
  // want to cheat just inside the bottom corner element:
  bottomCornerPoint(0,0) = G(0) + 1e-10;
  bottomCornerPoint(0,1) = G(1) + 1e-10;
  
  ParameterFunctionPtr Re_param = ParameterFunction::parameterFunction(Re);
  MeshGeometryPtr geometry = Teuchos::rcp( new MeshGeometry(vertices, elementVertices) );
  VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re_param, geometry,
                                                          H1Order, pToAdd,
                                                          zero, zero,  // zero forcing function
                                                          useCompliantGraphNorm); // enrich velocity if using compliant graph norm
  
  SolutionPtr solution = problem.backgroundFlow();
  solution->setReportConditionNumber(reportConditionNumber);
  SolutionPtr solnIncrement = problem.solutionIncrement();
  solnIncrement->setReportConditionNumber(reportConditionNumber);
  
  problem.bf()->setUseExtendedPrecisionSolveForOptimalTestFunctions(longDoubleGramInversion);
  
  Teuchos::RCP<Mesh> mesh = problem.mesh();
  mesh->registerSolution(solution);
  mesh->registerSolution(solnIncrement);
  
  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
  mesh->registerObserver(refHistory);
  
  FunctionPtr one = Function::constant(1.0);
  double meshMeasure = one->integrate(mesh);
  
  ParameterFunctionPtr dt_inv = ParameterFunction::parameterFunction(1.0 / dt);
  if (artificialTimeStepping) {
    //    // LHS gets u_inc / dt:
    BFPtr bf = problem.bf();
    FunctionPtr dt_inv_fxn = Teuchos::rcp(dynamic_cast< Function* >(dt_inv.get()), false);
    bf->addTerm(-dt_inv_fxn * u1, v1);
    bf->addTerm(-dt_inv_fxn * u2, v2);
    problem.setIP( bf->graphNorm() ); // graph norm has changed...
  }
  
  if (useCompliantGraphNorm) {
    if (artificialTimeStepping) {
      problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm(dt_inv));
    } else {
      problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm());
    }
    // (otherwise, will use graph norm)
  }

  Teuchos::RCP<RefinementPattern> verticalCut = RefinementPattern::xAnisotropicRefinementPatternQuad();
  if (! useBiswasGeometry) {
    // our elements now have aspect ratio 4:1.  We want to do 2 sets of horizontal refinements to square them up.
    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
  } else {
    // for the Biswas geometry, the only thing we can conveniently do is approximate squares.
    FieldContainer<double> inflowPoint(1,2);
    inflowPoint(0,0) = A(0) + 1e-10;
    inflowPoint(0,1) = A(1) + 1e-10;
    
    int inflowCell = mesh->elementsForPoints(inflowPoint)[0]->cellID();
    set<int> activeCellIDs = mesh->getActiveCellIDs();
//    if (rank==0)
//      cout << activeCellIDs.size() << " active cellIDs before erasure\n";
    activeCellIDs.erase(activeCellIDs.find(inflowCell));
//    if (rank==0)
//      cout << activeCellIDs.size() << " active cellIDs after erasure\n";
    mesh->hRefine(activeCellIDs, verticalCut);
    
//    inflowCell = mesh->elementsForPoints(inflowPoint)[0]->cellID();
//    activeCellIDs = mesh->getActiveCellIDs();
//    if (rank==0)
//      cout << activeCellIDs.size() << " active cellIDs before erasure\n";
//    activeCellIDs.erase(activeCellIDs.find(inflowCell));
//    if (rank==0)
//      cout << activeCellIDs.size() << " active cellIDs after erasure\n";
//    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
  }

  for (int i=0; i<numUniformRefinements; i++) {
    mesh->hRefine(mesh->getActiveCellIDs(), RefinementPattern::regularRefinementPatternQuad());
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

  ////////////////////   SOLVE & REFINE   ///////////////////////
  
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
  
  MeshPtr streamMesh = Teuchos::rcp( new Mesh(vertices, elementVertices, streamBF, H1Order, pToAddForStreamFunction) );
  streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC ) );
  
  // will use refinement history to playback refinements on streamMesh (no need to register streamMesh)
  
  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  FunctionPtr u1_0 = Teuchos::rcp( new U1_0 );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  
  bc->addDirichlet(u1hat, nonOutflowBoundary, u1_0);
  bc->addDirichlet(u2hat, nonOutflowBoundary, u2_0);

  if (useTractionBCsOnOutflow) {
    // impose zero-traction condition:
    bc->addDirichlet(t1n, outflowBoundary, zero);
    bc->addDirichlet(t2n, outflowBoundary, zero);
  } else {
    // do nothing on outflow, but do impose zero-mean pressure
    bc->addZeroMeanConstraint(p);
  }
  
  solution->setBC(bc);
  solnIncrement->setBC(bc);
  // when we impose the no-traction condition, not allowed to impose zero-mean pressure
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  Teuchos::RCP<BackwardFacingStepRefinementStrategy> bfsRefinementStrategy = Teuchos::rcp( new BackwardFacingStepRefinementStrategy(solnIncrement, energyThreshold,
                                                                                                                                    min_h, maxPolyOrder, rank==0) );
  bfsRefinementStrategy->addCorner(G(0), G(1));
  bfsRefinementStrategy->addCorner(H(0), H(1));
  
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
  
  bool printToConsole = rank==0;
  FunctionPtr u1_incr = Function::solution(u1, solnIncrement);
  FunctionPtr u2_incr = Function::solution(u2, solnIncrement);
  FunctionPtr sigma11_incr = Function::solution(sigma11, solnIncrement);
  FunctionPtr sigma12_incr = Function::solution(sigma12, solnIncrement);
  FunctionPtr sigma21_incr = Function::solution(sigma21, solnIncrement);
  FunctionPtr sigma22_incr = Function::solution(sigma22, solnIncrement);
  FunctionPtr p_incr = Function::solution(p, solnIncrement);
  
  FunctionPtr l2_incr;
  
  if (! weightIncrementL2Norm) {
    l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + sigma11_incr * sigma11_incr + sigma12_incr * sigma12_incr
    + sigma21_incr * sigma21_incr + sigma22_incr * sigma22_incr;
  } else {
    double Re2 = Re * Re;
    l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + Re2 * sigma11_incr * sigma11_incr + Re2 * sigma12_incr * sigma12_incr
    + Re2 * sigma21_incr * sigma21_incr + Re2 * sigma22_incr * sigma22_incr;
  }
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){
    if (startWithZeroSolutionAfterRefinement) {
      // start with a fresh (zero) initial guess for each adaptive mesh:
      solution->clear();
      problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
    }
    
    if (computeMaxConditionNumber) {
      IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(problem.solutionIncrement()->ip().get()), false );
      bool jacobiScalingTrue = true;
      double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, problem.solutionIncrement()->mesh(), jacobiScalingTrue);
      if (rank==0) {
        cout << "max jacobi-scaled Gram matrix condition number estimate with zero background flow: " << maxConditionNumber << endl;
      }
    }
    
    double incr_norm;
    do {
      problem.iterate(useLineSearch, useCondensedSolve);
      incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
      //          // update time step
      //          double new_dt = min(1.0/incr_norm, 1000.0);
      //          dt_inv->setValue(1/new_dt);
      
      if (rank==0) {
        cout << "\x1B[2K"; // Erase the entire current line.
        cout << "\x1B[0E"; // Move to the beginning of the current line.
        cout << "Refinement # " << refIndex << ", iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
        flush(cout);
      }
    } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
    
    if (rank==0)
      cout << "\nFor refinement " << refIndex << ", num iterations: " << problem.iterationCount() << endl;
    
    if (computeMaxConditionNumber) {
      IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(problem.solutionIncrement()->ip().get()), false );
      bool jacobiScalingTrue = true;
      double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, problem.solutionIncrement()->mesh(), jacobiScalingTrue);
      if (rank==0) {
        cout << "max jacobi-scaled Gram matrix condition number estimate with nonzero background flow: " << maxConditionNumber << endl;
      }
    }
    
    // reset iteration count to 1 (for the background flow):
    problem.setIterationCount(1);
    // reset iteration count to 0 (to start from 0 initial guess):
    //      problem.setIterationCount(0);
    
    //      solveStrategy->solve(printToConsole);
    
    double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
    if (rank == 0) {
      cout << "For refinement " << refIndex << ", mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
      cout << "  Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".\n";
    }

    bfsRefinementStrategy->refine(false); //rank==0); // print to console on rank 0
    
    if (refIndex < numCornerRefinementsToInduce) {
      if (induceCornerRefinements) {
        // essentially, refine the four elements nearest the bottom corner...
        ElementPtr bottomCorner = mesh->elementsForPoints(bottomCornerPoint)[0];
        Element* bottomCornerParent;
        if (bottomCorner->getParent() != NULL) {
          bottomCornerParent = bottomCorner->getParent();
        } else {
          bottomCornerParent = bottomCorner.get();
        }
        
        // induce refinements in bottom corner:
        set<int> cornerIDs = bottomCornerParent->getDescendants();
        mesh->hRefine(cornerIDs, RefinementPattern::regularRefinementPatternQuad());
        mesh->enforceOneIrregularity();
      }
    }
    
    if (saveFile.length() > 0) {
      if (rank == 0) {
        refHistory->saveToFile(saveFile);
      }
    }
  }
  // one more solve on the final refined mesh:
  if (rank==0) cout << "Final solve:\n";
  if (startWithZeroSolutionAfterRefinement) {
    // start with a fresh (zero) initial guess for each adaptive mesh:
    solution->clear();
    problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
  }
  double incr_norm;
  do {
    problem.iterate(useLineSearch, useCondensedSolve);
    incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
    if (rank==0) {
      cout << "\x1B[2K"; // Erase the entire current line.
      cout << "\x1B[0E"; // Move to the beginning of the current line.
      cout << "Iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
      flush(cout);
    }
  } while ((incr_norm > finalSolveMinL2Increment ) && (problem.iterationCount() < maxIters));
  if (rank==0) cout << endl;
  
  if (computeMaxConditionNumber) {
    string fileName = "nsCavity_maxConditionIPMatrix.dat";
    IPPtr ip = Teuchos::rcp( dynamic_cast< IP* >(problem.solutionIncrement()->ip().get()), false );
    bool jacobiScalingTrue = true;
    double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, problem.solutionIncrement()->mesh(), jacobiScalingTrue, fileName);
    if (rank==0) {
      cout << "max Gram matrix condition number estimate: " << maxConditionNumber << endl;
      cout << "putative worst-conditioned Gram matrix written to: " << fileName << "." << endl;
    }
  }
  
  if (outputStiffnessMatrix) {
    if (rank==0) {
      cout << "performing one extra iteration and outputting its stiffness matrix to disk.\n";
    }
    problem.solutionIncrement()->setWriteMatrixToFile(true, "nsCavity_final_stiffness.dat");
    problem.iterate(useLineSearch, useCondensedSolve);
    if (rank==0) {
      cout << "Final iteration, L^2(incr) = " << incr_norm << endl;
    }
  }
  
  double energyErrorTotal = solution->energyErrorTotal();
  double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
  if (rank == 0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "  (Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".)\n";
  }
  
  if (rank==0) {
    if (solnSaveFile.length() > 0) {
      solution->writeToFile(solnSaveFile);
    }
  }

  
//  for (int refIndex=0; refIndex<numRefs; refIndex++){
//    if (!enforceLocalConservation) {
//      solution->condensedSolve();
//    } else {
//      // condensed solve doesn't support lagrange constraints yet...
//      solution->solve(true);
//    }
//    if (compareWithOverkill) {
//      Teuchos::RCP<Solution> bestSoln = Teuchos::rcp( new Solution(solution->mesh(), bc, rhs, ip) );
//      overkillSolution->projectFieldVariablesOntoOtherSolution(bestSoln);
//      if (rank==0) {
//        VTKExporter exporter(solution, mesh, varFactory);
//        ostringstream cavityRefinement;
//        cavityRefinement << "backstep_solution_refinement_" << refIndex;
//        exporter.exportSolution(cavityRefinement.str());
//        VTKExporter exporterBest(bestSoln, mesh, varFactory);
//        ostringstream bestRefinement;
//        bestRefinement << "backstep_best_refinement_" << refIndex;
//        exporterBest.exportSolution(bestRefinement.str());
//      }
//      Teuchos::RCP<Solution> bestSolnOnOverkillMesh = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
//      bestSoln->projectFieldVariablesOntoOtherSolution(bestSolnOnOverkillMesh);
//      
//      FunctionPtr p_best = Teuchos::rcp( new PreviousSolutionFunction(bestSoln,p) );
//      double p_avg = p_best->integrate(mesh);
//      if (rank==0)
//        cout << "Integral of best solution pressure: " << p_avg << endl;
//      
//      // determine error as difference between our solution and overkill
//      bestSolnOnOverkillMesh->addSolution(overkillSolution,-1.0);
//      
//      Teuchos::RCP<Solution> adaptiveSolnOnOverkillMesh = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
//      solution->projectFieldVariablesOntoOtherSolution(adaptiveSolnOnOverkillMesh);
//      
//      // determine error as difference between our solution and overkill
//      adaptiveSolnOnOverkillMesh->addSolution(overkillSolution,-1.0);
//      
//      double L2errorSquared = 0.0;
//      double bestL2errorSquared = 0.0;
//      for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
//        VarPtr var = *fieldIt;
//        int fieldID = var->ID();
//        FunctionPtr fieldErrorFxn = Function::solution(var, adaptiveSolnOnOverkillMesh);
//        if (var->ID() == p->ID()) {
//          // pressure: subtract off the average difference:
//          double pAvg = fieldErrorFxn->integrate(adaptiveSolnOnOverkillMesh->mesh()) / meshMeasure;
//          fieldErrorFxn = fieldErrorFxn - pAvg;
//        }
//        
//        double L2error = fieldErrorFxn->l2norm(adaptiveSolnOnOverkillMesh->mesh());
//        L2errorSquared += L2error * L2error;
//        double bestL2error = bestSolnOnOverkillMesh->L2NormOfSolutionGlobal(fieldID);
//        bestL2errorSquared += bestL2error * bestL2error;
//        if (rank==0) {
//          cout << "L^2 error for " << var->name() << ": " << L2error;
//          cout << " (vs. best error of " << bestL2error << ")\n";
//        }
//      }
//      int numGlobalDofs = mesh->numGlobalDofs();
//      if (rank==0) {
//        cout << "for " << numGlobalDofs << " dofs, total L2 error: " << sqrt(L2errorSquared);
//        cout << " (vs. best error of " << sqrt(bestL2errorSquared) << ")\n";
//      }
//      dofsToL2error[numGlobalDofs] = sqrt(L2errorSquared);
//      dofsToBestL2error[numGlobalDofs] = sqrt(bestL2errorSquared);
//      if (rank==0) {
//        VTKExporter exporter(adaptiveSolnOnOverkillMesh, mesh, varFactory);
//        ostringstream errorForRefinement;
//        errorForRefinement << "overkillError_refinement_" << refIndex;
//        exporter.exportSolution(errorForRefinement.str());
//      }
//    }
//
//    bfsRefinementStrategy->refine(rank==0); // print to console on rank 0
//    
//    if (induceCornerRefinements) {
//      // induce refinements in bottom corner:
//      vector< Teuchos::RCP<Element> > corners = mesh->elementsForPoints(bottomCornerPoint);
//      vector<int> cornerIDs;
//      cornerIDs.push_back(corners[0]->cellID());
//      mesh->hRefine(cornerIDs, RefinementPattern::regularRefinementPatternQuad());
//    }
//  }
//  
//  // one more solve on the final refined mesh:
//  solution->solve(false);
//  double energyErrorTotal = solution->energyErrorTotal();
//  double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ip, mesh, "bfs_maxConditionIPMatrix.dat");
//  if (rank == 0) {
//    cout << "Final energy error: " << energyErrorTotal << endl;
//    cout << "Max condition number estimate: " << maxConditionNumber << endl;
//  }
//  
//  if (finalSolveUsesStandardGraphNorm) {
//    if (rank==0)
//      cout << "switching to graph norm for final solve";
//    
//    IPPtr ipToCompare = stokesBF->graphNorm();
//    Teuchos::RCP<Solution> solutionToCompare = Teuchos::rcp( new Solution(mesh, bc, rhs, ipToCompare) );
//
//    solutionToCompare->solve(false);
//    
//    FunctionPtr u1ToCompare = Function::solution(u1, solutionToCompare);
//    FunctionPtr u2ToCompare = Function::solution(u2, solutionToCompare);
//    
//    FunctionPtr u1_soln = Function::solution(u1, solution);
//    FunctionPtr u2_soln = Function::solution(u2, solution);
//    
//    double u1_l2difference = (u1ToCompare - u1_soln)->l2norm(mesh) / u1_soln->l2norm(mesh);
//    double u2_l2difference = (u2ToCompare - u2_soln)->l2norm(mesh) / u2_soln->l2norm(mesh);
//    
//    double graph_maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(ipToCompare, mesh, "bfs_maxConditionIPMatrix_graph.dat");
//    
//    if (rank==0) {
//      cout << "L^2 differences with automatic graph norm:\n";
//      cout << "    u1: " << u1_l2difference * 100 << "%" << endl;
//      cout << "    u2: " << u2_l2difference * 100 << "%" << endl;
//    }  
//    solution = solutionToCompare;
//    
//    double energyErrorTotal = solution->energyErrorTotal();
//    if (rank == 0) {
//      cout << "Final energy error (standard graph norm): " << energyErrorTotal << endl;
//      cout << "Max condition number estimate (standard graph norm): " << graph_maxConditionNumber << endl;
//    }
//  }
  
  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, -u1->dy() + u2->dx() ) );
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
  
  if (rank==0){
//    massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
//    solution->writeFieldsToFile(u1->ID(), "u1.m");
//    solution->writeFieldsToFile(u2->ID(), "u2.m");
//    streamSolution->writeFieldsToFile(phi->ID(), "phi.m");
    
    VTKExporter exporter(solution, mesh, varFactory);
    exporter.exportSolution("backStepSoln", H1Order*2);
    
    VTKExporter streamExporter(streamSolution, streamMesh, streamVarFactory);
    streamExporter.exportSolution("backStepStreamSoln", H1Order*2);
    
    FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
    exporter.exportFunction(polyOrderFunction,"backStepPolyOrders");
    exporter.exportFunction(vorticity, "backStepVorticity");
    
    cout << "exported vorticity to backStepVorticity\n";
//    solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
//    solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
//    solution->writeFieldsToFile(p->ID(), "p.m");

//    writePatchValues(0, RIGHT_OUTFLOW, 0, 2, streamSolution, phi, "phi_patch.m");
//    writePatchValues(4, 5, 0, 1, streamSolution, phi, "phi_patch_east.m");
    
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
    FieldContainer<double> eastPatchPoints = pointGrid(4, 4.4, MESH_BOTTOM, 0.45, 200);
    FieldContainer<double> eastPatchPointData = solutionData(eastPatchPoints, streamSolution, phi);
    GnuPlotUtil::writeXYPoints("phi_patch_east.dat", eastPatchPointData);
    set<double> patchContourLevels = diagonalContourLevels(eastPatchPointData,4);
    // be sure to the 0 contour, where the direction should change:
    patchContourLevels.insert(0);
    
    vector<string> patchDataPath;
    patchDataPath.push_back("phi_patch_east.dat");
    GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "backStepEastContourPlot.p", xTics, yTics);

    {
      map< pair<double,double> ,string> scaleToName;
      scaleToName[make_pair(1.05, 1.20)]   = "bfsPatch";
      scaleToName[make_pair(0.22, 0.20)] = "bfsPatchEddy1";
      scaleToName[make_pair(0.05, 0.05)] = "bfsPatchEddy2";
      
      for (map< pair<double,double>, string>::iterator entryIt=scaleToName.begin(); entryIt != scaleToName.end(); entryIt++) {
        double scaleX = (entryIt->first).first;
        double scaleY = (entryIt->first).second;
        string name = entryIt->second;
        double xTics = scaleX / 4, yTics = -1;
        ostringstream fileNameStream;
        fileNameStream << name << ".dat";
        FieldContainer<double> patchPoints = pointGrid(4, 4+scaleX, MESH_BOTTOM, MESH_BOTTOM + scaleY, 200);
        FieldContainer<double> patchPointData = solutionData(patchPoints, streamSolution, phi);
        GnuPlotUtil::writeXYPoints(fileNameStream.str(), patchPointData);
        ostringstream scriptNameStream;
        scriptNameStream << name << ".p";
        set<double> contourLevels = diagonalContourLevels(patchPointData,4);
        vector<string> dataPaths;
        dataPaths.push_back(fileNameStream.str());
        GnuPlotUtil::writeContourPlotScript(contourLevels, dataPaths, scriptNameStream.str(), xTics, yTics);
      }
      
      double xTics = 0.1, yTics = -1;
      FieldContainer<double> eastPatchPoints = pointGrid(4, 4.4, MESH_BOTTOM, MESH_BOTTOM + 0.45, 200);
      FieldContainer<double> eastPatchPointData = solutionData(eastPatchPoints, streamSolution, phi);
      GnuPlotUtil::writeXYPoints("phi_patch_east.dat", eastPatchPointData);
      set<double> patchContourLevels = diagonalContourLevels(eastPatchPointData,4);
      // be sure to the 0 contour, where the direction should change:
      patchContourLevels.insert(0);
      
      vector<string> patchDataPath;
      patchDataPath.push_back("phi_patch_east.dat");
      GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "backStepEastContourPlot.p", xTics, yTics);
    }
    
    GnuPlotUtil::writeComputationalMeshSkeleton("backStepMesh", mesh);
    
//      ofstream fout("phiContourLevels.dat");
//      fout << setprecision(15);
//      for (set<double>::iterator levelIt = contourLevels.begin(); levelIt != contourLevels.end(); levelIt++) {
//        fout << *levelIt << ", ";
//      }
//      fout.close();
    //    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
    //    writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
    
    
    if (compareWithOverkill) {
      if (rank==0) {
        cout << "******* Adaptivity Convergence Report *******\n";
        cout << "dofs\tL2 error\n";
        for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
          int dofs = entryIt->first;
          double err = entryIt->second;
          cout << dofs << "\t" << err;
          double bestError = dofsToBestL2error[dofs];
          cout << "\t" << bestError << endl;
        }
        ofstream fout("backstepOverkillComparison.txt");
        fout << "******* Adaptivity Convergence Report *******\n";
        fout << "dofs\tL2 error\tBest error\n";
        for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
          int dofs = entryIt->first;
          double err = entryIt->second;
          fout << dofs << "\t" << err;
          double bestError = dofsToBestL2error[dofs];
          fout << "\t" << bestError << endl;
        }
        fout.close();
      }
    }
    
    cout << "wrote files.\n";
  }
  return 0;
}
