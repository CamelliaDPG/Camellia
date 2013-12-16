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

#include "MLSolver.h"

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

static double MESH_TOP = 2.0;
static double STEP_Y = 1.0;
static double STEP_X = 4.0;
static double RIGHT_OUTFLOW = 8.0;
static double MESH_BOTTOM = 0.0;
static double LEFT_INFLOW = 0.0;

bool topWall(double x, double y) {
  return abs(y-MESH_TOP) < tol;
}

bool bottomWallRight(double x, double y) {
  return (abs(y-MESH_BOTTOM) < tol) && (x-STEP_X >= tol);
}

bool bottomWallLeft(double x, double y) {
  return (abs(y-STEP_Y) < tol) && (x-STEP_X < tol);
}

bool step(double x, double y) {
  return (abs(x-STEP_X) < tol) && (y-STEP_Y <=tol);
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
      // U1_0 should be 1.5 in the middle of the inflow (yields average of 1.0)
      double inflowHeight = MESH_TOP - STEP_Y;
      double weight = 6 * (1.0 / inflowHeight) * (1.0 / inflowHeight);
      return - weight * (y-MESH_TOP) * (y-STEP_Y);
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
    if (isBottomWallLeft || isBottomWallRight || isStep ) { // walls: no slip
      return 0.0;
    } else if (isTopWall || isInflow) {
      double inflowHeight = MESH_TOP - STEP_Y;
      double weight = 6 * (1.0 / inflowHeight) * (1.0 / inflowHeight);
      return - weight * ( (y * y * y  - STEP_Y * STEP_Y * STEP_Y) / 3.0
                         - (MESH_TOP + STEP_Y) * (y * y - STEP_Y * STEP_Y) / 2.0
                         + MESH_TOP * STEP_Y * (y - STEP_Y) );
//      return 9 * y * y - 2 * y * y * y - 12 * y + 5; // 5 just to make it 0 at y=1
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

double findHorizontalSignReversal(double y, double xGuessLeft, double xGuessRight, SolutionPtr soln, VarPtr u1) {
  int rank = Teuchos::GlobalMPISession::getRank();
  double leftValue = getSolutionValueAtPoint(xGuessLeft, y, soln, u1);
  double rightValue = getSolutionValueAtPoint(xGuessRight, y, soln, u1);
  if (leftValue * rightValue > 0) {
    double xGuess = (xGuessLeft + xGuessRight) / 2;
    if (rank==0) {
      cout << "Error: u1(" << xGuessLeft << ") = " << leftValue << " and u1(" << xGuessRight << ") = " << rightValue;
      cout << " have the same sign.  Returning " << -xGuess << endl;
    }
    return -xGuess;
  }
  int numIterations = 30;
  double x = 0;
  for (int i=0; i<numIterations; i++) {
    double xGuess = (xGuessLeft + xGuessRight) / 2;
    double middleValue = getSolutionValueAtPoint(xGuess, y, soln, u1);
    if (middleValue * leftValue > 0) { // same sign
      xGuessLeft = xGuess;
      leftValue = middleValue;
    }
    if (middleValue * rightValue > 0) {
      xGuessRight = xGuess;
      rightValue = middleValue;
    }
    x = xGuess;
  }
  return x;
}

void computeRecirculationRegion(double &xPoint, double &yPoint, SolutionPtr streamSoln, VarPtr phi) {
  // find the x recirculation region first
  int numIterations = 20;
  double x,y;
  y = 0.01;
  int rank = Teuchos::GlobalMPISession::getRank();
  {
    double xGuessLeft = 4.25;
    double xGuessRight = 4.60;
    double leftValue = getSolutionValueAtPoint(xGuessLeft, y, streamSoln, phi);
    double rightValue = getSolutionValueAtPoint(xGuessRight, y, streamSoln, phi);
    if (leftValue * rightValue > 0) {
      if (rank==0) cout << "Error: leftValue and rightValue have same sign.\n";
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

void computeLengthsForGartling(SolutionPtr soln, VarPtr u1, double epsDistance,
                               double &primaryReattachmentLength, double &secondarySeparationLength, double &secondaryReattachmentLength) {
  // find the sign reversals in u1
  double yNearBottom = MESH_BOTTOM + epsDistance;
  double yNearTop = MESH_TOP - epsDistance;
  primaryReattachmentLength = findHorizontalSignReversal(yNearBottom, 1.0, 15.0, soln, u1);
  secondarySeparationLength = findHorizontalSignReversal(yNearTop, 1.0, 7, soln, u1);
  secondaryReattachmentLength = findHorizontalSignReversal(yNearTop, 7, 15, soln, u1);
}

void printLengthsForGartling(SolutionPtr soln, VarPtr u1, double epsDistance) {
  // find the sign reversals in u1
  double yNearBottom = MESH_BOTTOM + epsDistance;
  double yNearTop = MESH_TOP - epsDistance;
  double primaryReattachmentLength = findHorizontalSignReversal(yNearBottom, 1.0, 15.0, soln, u1);
  double secondarySeparationLength = findHorizontalSignReversal(yNearTop, 1.0, 7, soln, u1);
  double secondaryReattachmentLength = findHorizontalSignReversal(yNearTop, 7, 15, soln, u1);
  
//  double primaryReattachmentLength = findZeroPointHorizontalSearch(yNearBottom, 1.0, 15.0, soln, u1);
//  double secondarySeparationLength = findZeroPointHorizontalSearch(yNearTop, 1.0, 7, soln, u1);
//  double secondaryReattachmentLength = findZeroPointHorizontalSearch(yNearTop, 7, 15, soln, u1);
  
  int rank = Teuchos::GlobalMPISession::getRank();
  if (rank==0) {
    cout << setw(30) << "primary reattachment length: " << primaryReattachmentLength << endl;
    cout << setw(30) << "secondary separation length: "  << secondarySeparationLength << endl;
    cout << setw(30) << "secondary reattachment length: " << secondaryReattachmentLength << endl;
  }
}

vector<double> verticalLinePoints() {
  // points where values are often reported in the literature (in Gartling)
  vector<double> yPoints;
  double y = 0.50;
  for (y = 0.5; y >= -0.5; y -= 0.05) {
    yPoints.push_back(y);
  }
  return yPoints;
}

struct VerticalLineSolutionValues {
  double x;
  vector<double> yPoints;
  vector<double> u1;
  vector<double> u2;
  vector<double> omega;
  vector<double> p;
};

VerticalLineSolutionValues computeVerticalLineSolutionValues(double xValue, FunctionPtr u1_prev, FunctionPtr u2_prev,
                                      FunctionPtr p_prev, FunctionPtr vorticity, bool computePOffsetAtOrigin = true) {
  
  
  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(false); // allows Function::evaluate() call, below
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) p_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(false);
  // the next bit commented out -- Function::evaluate() must depend only on space, not on the mesh (prev soln depends on mesh)
  
  double pOffset = computePOffsetAtOrigin ? Function::evaluate(p_prev, 0, 0) : 0; // Gartling sets p at (0,0) = 0.
  
  VerticalLineSolutionValues values;
  
  double y;
  double x = xValue;
  values.x = xValue;
  values.yPoints = verticalLinePoints();
  for (int i=0; i<values.yPoints.size(); i++) {
    y = values.yPoints[i];
    values.u1.push_back( Function::evaluate(u1_prev, x, y) );
    values.u2.push_back( Function::evaluate(u2_prev, x, y) );
    values.p.push_back( Function::evaluate(p_prev, x, y) - pOffset );
    values.omega.push_back( Function::evaluate(vorticity, x, y) );
  }
  return values;
}

void reportVerticalLineSolutionValues(double xValue, FunctionPtr u1_prev, FunctionPtr u2_prev,
                                      FunctionPtr p_prev, FunctionPtr vorticity, bool computePOffsetAtOrigin = true) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(false); // allows Function::evaluate() call, below
  ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) p_prev.get())->setOverrideMeshCheck(false);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(false);
  // the next bit commented out -- Function::evaluate() must depend only on space, not on the mesh (prev soln depends on mesh)
  vector<double> u1values, u2values, pValues, vorticityValues;
  
  double pOffset = computePOffsetAtOrigin ? Function::evaluate(p_prev, 0, 0) : 0; // Gartling sets p at (0,0) = 0.
  
  double x,y;
  x = xValue;
  vector<double> yPoints = verticalLinePoints();
  for (int i=0; i<yPoints.size(); i++) {
    y = yPoints[i];
    u1values.push_back( Function::evaluate(u1_prev, x, y) );
    u2values.push_back( Function::evaluate(u2_prev, x, y) );
    pValues.push_back( Function::evaluate(p_prev, x, y) - pOffset );
    vorticityValues.push_back( Function::evaluate(vorticity, x, y) );
  }
  
  if (rank==0) {
    cout << "**** x=" << xValue << ", values ****\n";
    int w = 20;
    cout << setw(w) << "y" << setw(w) << "u1" << setw(w) << "u2" << setw(w) << "p" << setw(w) << "omega" << endl;
    for (int i=0; i<yPoints.size(); i++) {
      cout << setw(w) << yPoints[i] << setw(w) << u1values[i] << setw(w) << u2values[i] << setw(w) << pValues[i] << setw(w) << vorticityValues[i] << endl;
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
  
  bool useGartlingParameters = args.Input<bool>("--useGartling", "Use parameters from D.K. Gartling's 1990 paper", false);
  double Re_default = 100;
  if (useGartlingParameters) { // default to the 800 Re used there
    Re_default = 800;
  }
  double Re = args.Input<double>("--Re", "Reynolds number", Re_default);
  
  int numRefs = args.Input<int>("--numRefs", "Number of refinements", 10);
  int numStreamSolutionRefs = args.Input<int>("--numStreamRefs", "Number of refinements in stream solution", 0);
  double energyThreshold = args.Input<double>("--adaptiveThreshold", "threshold parameter for greedy adaptivity", 0.20);
  bool enforceLocalConservation = args.Input<bool>("--enforceLocalConservation", "Enforce local conservation.", false);
  bool useCompliantGraphNorm = args.Input<bool>("--useCompliantNorm", "use the 'scale-compliant' graph norm", false);

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
  
  double maxAspectRatioForGartling = args.Input<double>("--maxAspectRatioForGartling", "maximum allowable aspect ratio for initial mesh when using Gartling parameters", 2.0);
  
  int maxIters = args.Input<int>("--maxIters", "maximum number of Newton-Raphson iterations to take to try to match tolerance", 50);
  double minL2Increment = args.Input<double>("--NRtol", "Newton-Raphson tolerance, L^2 norm of increment", 3e-8);
  string replayFile = args.Input<string>("--replayFile", "file with refinement history to replay", "");
  string solnFile = args.Input<string>("--solnFile", "file with solution data", "");
  string solnSaveFile = args.Input<string>("--solnSaveFile", "file to which to save solution data", "");
  string saveFile = args.Input<string>("--saveReplay", "file to which to save refinement history", "");
  
  bool useMumps = args.Input<bool>("--useMumps", "use MUMPS for global linear solves", true);
  bool useML = args.Input<bool>("--useML", "use ML for global linear solves", false);
  double mlTol = args.Input<double>("--mlTol", "tolerance for ML convergence", 1e-6);
  if (useML) useMumps = false; // mutually exclusive...
  
  args.Process();
  
  if (useBiswasGeometry) {
    MESH_BOTTOM = 1.0 - 0.9423;
    RIGHT_OUTFLOW = 7.0;
    LEFT_INFLOW = 3.0;
  } else if (useGartlingParameters) {
    RIGHT_OUTFLOW = 30.0;
    LEFT_INFLOW = 0.0;
    STEP_Y = 0.0;
    STEP_X = 0.0;
    MESH_TOP = 0.5;
    MESH_BOTTOM = -0.5;
  }
  
  vector<double> relativeEnergyErrors(numRefs + 1);
  vector<long> dofCounts(numRefs+1);
  vector<long> fluxDofCounts(numRefs+1);
  vector<long> elementCounts(numRefs+1);
  vector<int> iterationCounts(numRefs+1);
  vector<double> tolerances(numRefs+1);
  
  map<double, vector< VerticalLineSolutionValues > > verticalCutValues; // keys are x locations
  
  vector<double> gartlingPrimaryReattachmentLengths,gartlingSecondarySeparationLengths,gartlingSecondaryReattachmentLengths;
  if (useGartlingParameters) {
    gartlingPrimaryReattachmentLengths = vector<double>(numRefs + 1);
    gartlingSecondarySeparationLengths = vector<double>(numRefs + 1);
    gartlingSecondaryReattachmentLengths = vector<double>(numRefs + 1);
    verticalCutValues[0] = vector< VerticalLineSolutionValues >();
    verticalCutValues[7] = vector< VerticalLineSolutionValues >();
    verticalCutValues[15] = vector< VerticalLineSolutionValues >();
    verticalCutValues[30] = vector< VerticalLineSolutionValues >();
  }
  
  Teuchos::RCP<Solver> solver;
  if (useMumps) {
#ifdef USE_MUMPS
    solver = Teuchos::rcp(new MumpsSolver());
#else
    if (rank==0)
      cout << "useMumps = true, but USE_MUMPS is unset.  Exiting...\n";
    exit(1);
#endif
  } else if (useML) {
    solver = Teuchos::rcp(new MLSolver(mlTol));
  } else {
    solver = Teuchos::rcp(new KluSolver());
  }
  
  bool startWithZeroSolutionAfterRefinement = false;
  
  bool useLineSearch = false;
  
  bool artificialTimeStepping = (dt > 0);

  if (rank == 0) {
    cout << "numRefinements = " << numRefs << endl;
    cout << "Re = " << Re << endl;
    cout << "using L^2 tolerance relative to L^2 norm of background flow.\n";
    if (artificialTimeStepping) cout << "dt = " << dt << endl;
    if (!startWithZeroSolutionAfterRefinement) {
      cout << "NOT starting with 0 solution after refinement...\n";
    }
    if (useCondensedSolve) {
      cout << "Using condensed solve.\n";
    } else {
      cout << "Not using condensed solve.\n";
    }
    if (useMumps) {
      cout << "Using MUMPS for global linear solves.\n";
    } else if (useML) {
      cout << "Using ML for global linear solves.\n";
    } else {
      cout << "Using KLU for global linear solves.\n";
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
  A(0) = LEFT_INFLOW; A(1) = STEP_Y;
  B(0) = LEFT_INFLOW; B(1) = MESH_TOP;
  C(0) = STEP_X; C(1) = MESH_TOP;
  D(0) = RIGHT_OUTFLOW; D(1) = MESH_TOP;
  E(0) = RIGHT_OUTFLOW; E(1) = STEP_Y;
  F(0) = RIGHT_OUTFLOW; F(1) = MESH_BOTTOM;
  G(0) = STEP_X; G(1) = MESH_BOTTOM;
  H(0) = STEP_X; H(1) = STEP_Y;
  vector<FieldContainer<double> > vertices;
  vector< vector<int> > elementVertices;
  vector<int> el1, el2, el3, el4, el5;
  
  if (! useGartlingParameters) {
    vertices.push_back(A); int A_index = 0;
    vertices.push_back(B); int B_index = 1;
    vertices.push_back(C); int C_index = 2;
    vertices.push_back(D); int D_index = 3;
    vertices.push_back(E); int E_index = 4;
    vertices.push_back(F); int F_index = 5;
    vertices.push_back(G); int G_index = 6;
    vertices.push_back(H); int H_index = 7;

    // left patch:
    el1.push_back(A_index); el1.push_back(H_index); el1.push_back(C_index); el1.push_back(B_index);
    // top right:
    el2.push_back(H_index); el2.push_back(E_index); el2.push_back(D_index); el2.push_back(C_index);
    // bottom right:
    el3.push_back(G_index); el3.push_back(F_index); el3.push_back(E_index); el3.push_back(H_index);
    
    elementVertices.push_back(el1);
    elementVertices.push_back(el2);
    elementVertices.push_back(el3);
  } else {
    // Gartling's domain begins just at the step edge...
    vertices.push_back(C); int C_index = 0;
    vertices.push_back(D); int D_index = 1;
    vertices.push_back(E); int E_index = 2;
    vertices.push_back(F); int F_index = 3;
    vertices.push_back(G); int G_index = 4;
    vertices.push_back(H); int H_index = 5;
    
    // top right:
    el2.push_back(H_index); el2.push_back(E_index); el2.push_back(D_index); el2.push_back(C_index);
    // bottom right:
    el3.push_back(G_index); el3.push_back(F_index); el3.push_back(E_index); el3.push_back(H_index);
    
    elementVertices.push_back(el2);
    elementVertices.push_back(el3);
  }
  
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
  problem.setSolver(solver);
  
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
  if (useGartlingParameters) {
    // our elements now have aspect ratio 30:1.  We want to do 5 sets of horizontal refinements to square them up (less if the user relaxed the maxAspectRatioForGartling)
    double aspectRatio = 30;
    while (aspectRatio > maxAspectRatioForGartling) {
      mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
      aspectRatio /= 2;
    }
//    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
//    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
//    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
//    mesh->hRefine(mesh->getActiveCellIDs(), verticalCut);
  } else if (! useBiswasGeometry) {
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

  Teuchos::RCP<PenaltyConstraints> pc;
  
  if (useTractionBCsOnOutflow) {
    pc = Teuchos::rcp(new PenaltyConstraints);
    
    // define traction components in terms of field variables
    FunctionPtr n = Function::normal();
    LinearTermPtr t1 = n->x() * (2 * sigma11 - p) + n->y() * (sigma12 + sigma21);
    LinearTermPtr t2 = n->x() * (sigma12 + sigma21) + n->y() * (2 * sigma22 - p);
    
    if (rank==0)
      cout << "Imposing zero traction at outflow with penalty constraints.\n";
    // outflow: both traction components are 0
    pc->addConstraint(t1==zero, outflowBoundary);
    pc->addConstraint(t2==zero, outflowBoundary);
  } else {
    // do nothing on outflow, but do impose zero-mean pressure
    bc->addZeroMeanConstraint(p);
  }
  
  // set pc and bc -- pc in particular may be null
  solution->setFilter(pc);
  solnIncrement->setFilter(pc);
  
  solution->setBC(bc);
  solnIncrement->setBC(bc);
  // when we impose the no-traction condition, not allowed to impose zero-mean pressure
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  bool printRefsToConsole = false;
//  bool printRefsToConsole = rank==0;
  Teuchos::RCP<BackwardFacingStepRefinementStrategy> bfsRefinementStrategy = Teuchos::rcp( new BackwardFacingStepRefinementStrategy(solnIncrement, energyThreshold,
                                                                                                                                    min_h, maxPolyOrder, printRefsToConsole) );
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
  
//  FunctionPtr u1_prev = Function::solution(u1, solution);
//  FunctionPtr u2_prev = Function::solution(u2, solution);
  FunctionPtr sigma11_prev = Function::solution(sigma11, solution);
  FunctionPtr sigma12_prev = Function::solution(sigma12, solution);
  FunctionPtr sigma21_prev = Function::solution(sigma21, solution);
  FunctionPtr sigma22_prev = Function::solution(sigma22, solution);
//  FunctionPtr p_prev = Function::solution(p, solution);

  // Function::evaluate() can be used with PreviousSolutionFunction, not with SimpleSolutionFunction.
  FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u1 ) );
  FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, u2 ) );
  FunctionPtr p_prev = Teuchos::rcp( new PreviousSolutionFunction(solution, p ) );
  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, (-Re) * sigma12 + Re * sigma21 ) );
//  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, -u1->dy() + u2->dx() ) );
  
  FunctionPtr l2_incr, l2_prev;
  
  if (! weightIncrementL2Norm) {
    l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + sigma11_incr * sigma11_incr + sigma12_incr * sigma12_incr
    + sigma21_incr * sigma21_incr + sigma22_incr * sigma22_incr;
    
    l2_prev = u1_prev * u1_prev + u2_prev * u2_prev + p_prev * p_prev
    + sigma11_prev * sigma11_prev + sigma12_prev * sigma12_prev
    + sigma21_prev * sigma21_prev + sigma22_prev * sigma22_prev;
  } else {
    double Re2 = Re * Re;
    l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + Re2 * sigma11_incr * sigma11_incr + Re2 * sigma12_incr * sigma12_incr
    + Re2 * sigma21_incr * sigma21_incr + Re2 * sigma22_incr * sigma22_incr;
    
    l2_prev = u1_prev * u1_prev + u2_prev * u2_prev + p_prev * p_prev
    + Re2 * sigma11_prev * sigma11_prev + Re2 * sigma12_prev * sigma12_prev
    + Re2 * sigma21_prev * sigma21_prev + Re2 * sigma22_prev * sigma22_prev;
  }
  
  double initialMinL2Increment = minL2Increment;
  if (rank==0) cout << "Initial relative L^2 tolerance: " << minL2Increment << endl;
  
  LinearTermPtr backgroundSolnFunctional = problem.bf()->testFunctional(problem.backgroundFlow());
  RieszRep solnRieszRep(mesh, problem.solutionIncrement()->ip(), backgroundSolnFunctional);
  
  LinearTermPtr incrementalSolnFunctional = problem.bf()->testFunctional(problem.solutionIncrement());
  RieszRep incrementRieszRep(mesh, problem.solutionIncrement()->ip(), incrementalSolnFunctional);
  
  tolerances[0] = initialMinL2Increment;
  
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
    
    double incr_norm, prev_norm;
    do {
      problem.iterate(useLineSearch, useCondensedSolve, false);
//      problem.iterate(useLineSearch, useCondensedSolve, problem.iterationCount() >= 2); // skip initialization after the 0 and 1 guesses -- but so far, condensed solve doesn't support preconditioner reuse.
      
      incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
      prev_norm = sqrt(l2_prev->integrate(problem.mesh()));
      if (prev_norm > 0) {
        incr_norm /= prev_norm;
      }

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
    
    double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
    solnRieszRep.computeRieszRep();
    double solnEnergyNormTotal = solnRieszRep.getNorm();
//    incrementRieszRep.computeRieszRep();
//    double incrementEnergyNormTotal = incrementRieszRep.getNorm();
    
    double relativeEnergyError = incrementalEnergyErrorTotal / solnEnergyNormTotal;
    minL2Increment = initialMinL2Increment * relativeEnergyError;
    
    relativeEnergyErrors[refIndex] = relativeEnergyError;
    dofCounts[refIndex] = mesh->numGlobalDofs();
    fluxDofCounts[refIndex] = mesh->numFluxDofs();
    elementCounts[refIndex] = mesh->numActiveElements();
    
    iterationCounts[refIndex] = (refIndex==0) ? problem.iterationCount() : problem.iterationCount() - 1;
    tolerances[refIndex+1] = minL2Increment;
    
    
    // reset iteration count to 1 (for the background flow):
    problem.setIterationCount(1);
    // reset iteration count to 0 (to start from 0 initial guess):
    //      problem.setIterationCount(0);
    
    //      solveStrategy->solve(printToConsole);
    
    
    if (rank == 0) {
      cout << "For refinement " << refIndex << ", mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
//      cout << "  Incremental solution's energy is " << incrementEnergyNormTotal << ".\n";
      cout << "  Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".\n";
      cout << "  Background flow's energy norm is " << solnEnergyNormTotal << ".\n";
//      cout << "  Relative energy of increment is " << incrementEnergyNormTotal / solnEnergyNormTotal * 100.0 << "%" << endl;
      cout << "  Relative energy error: " << incrementalEnergyErrorTotal / solnEnergyNormTotal * 100.0 << "%" << endl;
      cout << "  Updated L^2 tolerance for nonlinear iteration: " << minL2Increment << endl;
      if (useGartlingParameters) {
      }
    }
  
    if (useGartlingParameters) {
      if (rank==0) cout << "Using sigma12 sign reversals and 0 eps:\n";
      printLengthsForGartling(solution, sigma12, 0);
      
      computeLengthsForGartling(solution, sigma12, 0,
                                gartlingPrimaryReattachmentLengths[refIndex],
                                gartlingSecondarySeparationLengths[refIndex],
                                gartlingSecondaryReattachmentLengths[refIndex]);
      
      for (map<double, vector< VerticalLineSolutionValues > >::iterator cutVectorsIt = verticalCutValues.begin();
           cutVectorsIt != verticalCutValues.end(); cutVectorsIt++) {
        double x = cutVectorsIt->first;
        VerticalLineSolutionValues values = computeVerticalLineSolutionValues(x, u1_prev, u2_prev, p_prev, vorticity);
        cutVectorsIt->second.push_back(values);
      }
      
      reportVerticalLineSolutionValues( 0.0, u1_prev, u2_prev, p_prev, vorticity);
      reportVerticalLineSolutionValues( 7.0, u1_prev, u2_prev, p_prev, vorticity);
      reportVerticalLineSolutionValues(15.0, u1_prev, u2_prev, p_prev, vorticity);
      
      
      ((PreviousSolutionFunction*) p_prev.get())->setOverrideMeshCheck(false);
      double pOffset = Function::evaluate(p_prev, 0, 0); // Gartling sets p at (0,0) = 0.
      if (rank==0) cout << "(pOffset for the above: " << pOffset << ")" << endl;
      
      reportVerticalLineSolutionValues(30.0, u1_prev, u2_prev, p_prev, vorticity, false);

    }

    bfsRefinementStrategy->refine(false); //rank==0); // print to console on rank 0
    if (rank==0)
      cout << "After refinement " << refIndex + 1 << ", mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " total dofs and " << mesh->numFluxDofs() << " flux dofs.\n";
    
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
  } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
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
  
  solnRieszRep.computeRieszRep();
  double solnEnergyNormTotal = solnRieszRep.getNorm();
  
  double energyErrorTotal = solution->energyErrorTotal();
  double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
  
  double relativeEnergyError = incrementalEnergyErrorTotal / solnEnergyNormTotal;
  
  relativeEnergyErrors[numRefs] = relativeEnergyError;
  dofCounts[numRefs] = mesh->numGlobalDofs();
  fluxDofCounts[numRefs] = mesh->numFluxDofs();
  elementCounts[numRefs] = mesh->numActiveElements();
  iterationCounts[numRefs] = (numRefs > 0) ? problem.iterationCount() - 1 : problem.iterationCount();
  
  if (rank == 0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "  (Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".)\n";
  }

  if (useGartlingParameters) {
//    if (rank==0) cout << "Using u1 sign reversals and .01 eps:\n";
//    printLengthsForGartling(solution, u1, 0.01);
    if (rank==0) cout << "Using sigma12 sign reversals and 0 eps:\n";
    printLengthsForGartling(solution, sigma12, 0);
    computeLengthsForGartling(solution, sigma12, 0,
                              gartlingPrimaryReattachmentLengths[numRefs],
                              gartlingSecondarySeparationLengths[numRefs],
                              gartlingSecondaryReattachmentLengths[numRefs]);
    
    for (map<double, vector< VerticalLineSolutionValues > >::iterator cutVectorsIt = verticalCutValues.begin();
         cutVectorsIt != verticalCutValues.end(); cutVectorsIt++) {
      double x = cutVectorsIt->first;
      VerticalLineSolutionValues values = computeVerticalLineSolutionValues(x, u1_prev, u2_prev, p_prev, vorticity);
      cutVectorsIt->second.push_back(values);
    }
    
    reportVerticalLineSolutionValues( 0.0, u1_prev, u2_prev, p_prev, vorticity);
    reportVerticalLineSolutionValues( 7.0, u1_prev, u2_prev, p_prev, vorticity);
    reportVerticalLineSolutionValues(15.0, u1_prev, u2_prev, p_prev, vorticity);
    
    ((PreviousSolutionFunction*) p_prev.get())->setOverrideMeshCheck(false);
    double pOffset = Function::evaluate(p_prev, 0, 0); // Gartling sets p at (0,0) = 0.
    if (rank==0) cout << "(pOffset for the above: " << pOffset << ")" << endl;
    
    reportVerticalLineSolutionValues(30.0, u1_prev, u2_prev, p_prev, vorticity, false);
  }
  
  if (rank==0) {
    if (solnSaveFile.length() > 0) {
      solution->writeToFile(solnSaveFile);
    }
  }
  
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
  
  // register the main solution's mesh with streamMesh, so that refinements propagate appropriately:
  streamMesh->registerObserver(solution->mesh());
  Teuchos::RCP<BackwardFacingStepRefinementStrategy> streamRefinementStrategy = Teuchos::rcp( new BackwardFacingStepRefinementStrategy(streamSolution, energyThreshold,
                                                                                                                                       min_h, maxPolyOrder, rank==0) );
  for (int refIndex=0; refIndex<numStreamSolutionRefs; refIndex++) {
    cout << "Stream refinement # " << refIndex + 1 << ":\n";
    streamSolution->condensedSolve(solver);
    streamRefinementStrategy->refine();
  }
  
  streamSolution->condensedSolve(solver);
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

#ifdef USE_VTK
    VTKExporter exporter(solution, mesh, varFactory);
    exporter.exportSolution("backStepSoln", H1Order*2);
    
    VTKExporter streamExporter(streamSolution, streamMesh, streamVarFactory);
    streamExporter.exportSolution("backStepStreamSoln", H1Order*2);
    
    FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
    exporter.exportFunction(polyOrderFunction,"backStepPolyOrders");
    exporter.exportFunction(vorticity, "backStepVorticity");
    
    cout << "exported vorticity to backStepVorticity\n";
#endif
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
    
    // write out the separation values
    if (useGartlingParameters) {
      if (rank==0) {
        ofstream fout("backstepSeparationLengths.txt");
        fout << "ref. #\tprimary reattachment\tsecondary separation\tsecondary reattachment\trel. energy error\tL^2 tol.\tNR iterations\telements\tdofs\tflux dofs\n";
        for (int refIndex=0; refIndex<=numRefs; refIndex++) {
          fout << setprecision(8) << fixed;
          fout << refIndex << "\t" << gartlingPrimaryReattachmentLengths[refIndex];
          fout << "\t" << gartlingSecondarySeparationLengths[refIndex];
          fout << "\t" << gartlingSecondaryReattachmentLengths[refIndex];
          fout << setprecision(3) << scientific;;
          fout << "\t" << relativeEnergyErrors[refIndex];
          fout << "\t" << tolerances[refIndex];
          fout << "\t" << iterationCounts[refIndex];
          fout << "\t" << elementCounts[refIndex];
          fout << "\t" << dofCounts[refIndex];
          fout << "\t" << fluxDofCounts[refIndex];
          fout << endl;
        }
        fout.close();
        
        fout.open("backStepVerticalCutData.txt");
        fout << "ref. #\tx\ty\tu1\tu2\t|u|\tp\tomega\n";
        for (int refIndex=0; refIndex<=numRefs; refIndex++) {
          for (map<double, vector< VerticalLineSolutionValues > >::iterator cutVectorsIt = verticalCutValues.begin();
               cutVectorsIt != verticalCutValues.end(); cutVectorsIt++) {
            double x = cutVectorsIt->first;
            vector< VerticalLineSolutionValues > valuesList = cutVectorsIt->second;
            VerticalLineSolutionValues values = valuesList[refIndex];
            int yCount = values.yPoints.size();
            for (int yIndex=0; yIndex<yCount; yIndex++) {
              double y = values.yPoints[yIndex];
              double u1 = values.u1[yIndex];
              double u2 = values.u2[yIndex];
              double u = sqrt(u1*u1 + u2*u2);
              double p = values.p[yIndex];
              double omega = values.omega[yIndex];
              fout << refIndex << "\t" << x << "\t" << y;
              fout << "\t" << u1 << "\t" << u2 << "\t" << u;
              fout << "\t" << p << "\t" << omega << endl;
            }
          }
        }
        fout.close();
      }
    }
    
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
