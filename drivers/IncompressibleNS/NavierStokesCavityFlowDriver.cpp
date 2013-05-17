//
//  NavierStokesCavityFlowDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "choice.hpp"
#include "mpi_choice.hpp"

#include "HConvergenceStudy.h"

#include "InnerProductScratchPad.h"

#include "PreviousSolutionFunction.h"

#include "LagrangeConstraints.h"

#include "BasisFactory.h"

#include "ParameterFunction.h"

#include "RefinementHistory.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "NavierStokesFormulation.h"

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
//#include "LidDrivenFlowRefinementStrategy.h"
#include "RefinementPattern.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"
#include "MeshPolyOrderFunction.h"
#include "MeshTestUtility.h"
#include "NonlinearSolveStrategy.h"
#include "PenaltyConstraints.h"

#include "GnuPlotUtil.h"

#include "MeshUtilities.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

// static double REYN = 100;
static double Re = 400; // matches John Evans's dissertation, p. 183

VarFactory varFactory; 
// test variables:
VarPtr tau1, tau2, v1, v2, q;
// traces and fluxes:
VarPtr u1hat, u2hat, t1n, t2n;
// field variables:
VarPtr u1, u2, sigma11, sigma12, sigma21, sigma22, p;


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

FieldContainer<double> pointGrid(double xMin, double xMax, double yMin, double yMax, int numPoints) {
  vector<double> points1D_x, points1D_y;
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
  return points;
}

FieldContainer<double> solutionData(FieldContainer<double> &points, SolutionPtr solution, VarPtr u1) {
  int numPoints = points.dimension(0);
  FieldContainer<double> values(numPoints);
  solution->solutionValues(values, u1->ID(), points);
  
  FieldContainer<double> xyzData(numPoints, 3);
  for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
    xyzData(ptIndex,0) = points(ptIndex,0);
    xyzData(ptIndex,1) = points(ptIndex,1);
    xyzData(ptIndex,2) = values(ptIndex);
  }
  return xyzData;
}

set<double> diagonalContourLevels(FieldContainer<double> &pointData, int pointsPerLevel=1) {
  // traverse diagonal of (i*numPoints + j) data from solutionData()
  int numPoints = sqrt(pointData.dimension(0));
  set<double> levels;
  for (int i=0; i<numPoints; i++) {
    levels.insert(pointData(i*numPoints + i,2)); // format for pointData has values at (ptIndex, 2)
  }
  // traverse the counter-diagonal
  for (int i=0; i<numPoints; i++) {
    levels.insert(pointData(i*numPoints + numPoints-1-i,2)); // format for pointData has values at (ptIndex, 2)
  }
  set<double> filteredLevels;
  int i=0;
  pointsPerLevel *= 2;
  for (set<double>::iterator levelIt = levels.begin(); levelIt != levels.end(); levelIt++) {
    if (i%pointsPerLevel==0) {
      filteredLevels.insert(*levelIt);
    }
    i++;
  }
  return filteredLevels;
}

void writePatchValues(double xMin, double xMax, double yMin, double yMax,
                      SolutionPtr solution, VarPtr u1, string filename,
                      int numPoints=100) {
  FieldContainer<double> points = pointGrid(xMin,xMax,yMin,yMax,numPoints);
  FieldContainer<double> values(numPoints*numPoints);
  solution->solutionValues(values, u1->ID(), points);
  ofstream fout(filename.c_str());
  fout << setprecision(15);
  
  fout << "X = zeros(" << numPoints << ",1);\n";
  //    fout << "Y = zeros(numPoints);\n";
  fout << "U = zeros(" << numPoints << "," << numPoints << ");\n";
  for (int i=0; i<numPoints; i++) {
    fout << "X(" << i+1 << ")=" << points(i,0) << ";\n";
  }
  for (int i=0; i<numPoints; i++) {
    fout << "Y(" << i+1 << ")=" << points(i,1) << ";\n";
  }
  
  for (int i=0; i<numPoints; i++) {
    for (int j=0; j<numPoints; j++) {
      int pointIndex = i*numPoints + j;
      fout << "U("<<i+1<<","<<j+1<<")=" << values(pointIndex) << ";" << endl;
    }
  }
  fout.close();
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
#ifdef HAVE_MPI
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args(argc, argv );
#endif
  
  try {
  // read args:
    int polyOrder = args.Input<int>("--polyOrder", "L^2 (field) polynomial order");
    int numRefs = args.Input<int>("--numRefinements", "Number of refinements", 6);
    double Re = args.Input<double>("--Re", "Reynolds number", 400);
    bool longDoubleGramInversion = args.Input<bool>("--longDoubleGramInversion", "use long double Cholesky factorization for Gram matrix", false);
    int horizontalCells = args.Input<int>("--horizontalCells", "horizontal cell count for initial mesh (if vertical unspecified, will match horizontal)", 2);
    int verticalCells = args.Input<int>("--verticalCells", "vertical cell count for initial mesh", horizontalCells);
    bool outputStiffnessMatrix = args.Input<bool>("--writeFinalStiffnessToDisk", "write the final stiffness matrix to disk.", false);
    bool computeMaxConditionNumber = args.Input<bool>("--computeMaxConditionNumber", "compute the maximum Gram matrix condition number for final mesh.", false);
    bool enforceLocalConservation = args.Input<bool>("--enforceLocalConservation", "enforce local conservation using Lagrange constraints", false);
    bool useCompliantGraphNorm = args.Input<bool>("--useCompliantNorm", "use the 'scale-compliant' graph norm", false);
    bool reportConditionNumber = args.Input<bool>("--reportGlobalConditionNumber", "report the 2-norm condition number for the global system matrix", false);

    bool weightIncrementL2Norm = useCompliantGraphNorm; // if using the compliant graph norm, weight the measure of the L^2 increment accordingly
    
    int maxIters = args.Input<int>("--maxIters", "maximum number of Newton-Raphson iterations to take to try to match tolerance", 50);
    double minL2Increment = args.Input<double>("--NRtol", "Newton-Raphson tolerance, L^2 norm of increment", 3e-8);
    string replayFile = args.Input<string>("--replayFile", "file with refinement history to replay", "");
    string saveFile = args.Input<string>("--saveReplay", "file to which to save refinement history", "");
    
    double finalSolveMinL2Increment = args.Input<double>("--finalNRtol", "Newton-Raphson tolerance for final solve, L^2 norm of increment", minL2Increment / 10);
    
    args.Process();
    
    bool useLineSearch = false;
    
    int pToAdd = 2; // for optimal test function approximation
    int pToAddForStreamFunction = 2;
    double nonlinearStepSize = 1.0;
    double dt = 0.5;
    double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  //  double nonlinearRelativeEnergyTolerance = 0.15; // used to determine convergence of the nonlinear solution
    double eps = 1.0/64.0; // width of ramp up to 1.0 for top BC;  eps == 0 ==> soln not in H1
    // epsilon above is chosen to match our initial 16x16 mesh, to avoid quadrature errors.
  //  double eps = 0.0; // John Evans's problem: not in H^1
    bool enforceLocalConservationInFinalSolve = false; // only works correctly for Picard (and maybe not then!)
    bool enforceOneIrregularity = true;
    bool reportPerCellErrors  = true;
    bool useMumps = true;
    bool compareWithOverkillMesh = false;
    bool useAdHocHPRefinements = false;
    bool startWithZeroSolutionAfterRefinement = true;
    
    bool artificialTimeStepping = false;
    
    int overkillMeshSize = 8;
    int overkillPolyOrder = 7; // H1 order
    
    
  //  // usage: polyOrder [numRefinements]
  //  // parse args:
  //  if ((argc != 4) && (argc != 3) && (argc != 2) && (argc != 5)) {
  //    cout << "Usage: NavierStokesCavityFlowDriver fieldPolyOrder [numRefinements=10 [Reyn=400]]\n";
  //    return -1;
  //  }
  //  int polyOrder = atoi(argv[1]);
  //  int numRefs = 10;
  //  if ( argc == 3) {
  //    numRefs = atoi(argv[2]);
  //  }
  //  if ( argc == 4) {
  //    numRefs = atoi(argv[2]);
  //    Re = atof(argv[3]);
  //  }
  //  if ( argc == 5) {
  //    numRefs = atoi(argv[2]);
  //    Re = atof(argv[3]);
  //    horizontalCells = atoi(argv[4]);
  //    verticalCells = horizontalCells;
  //  }
    if (rank == 0) {
      cout << "numRefinements = " << numRefs << endl;
      cout << "Re = " << Re << endl;
      cout << "initial mesh: " << horizontalCells << " x " << verticalCells << endl;
      if (artificialTimeStepping) cout << "dt = " << dt << endl;
      if (!startWithZeroSolutionAfterRefinement) {
        cout << "NOTE: experimentally, NOT starting with 0 solution after refinement...\n";
      }
    }
    
    FieldContainer<double> quadPoints(4,2);
    
    quadPoints(0,0) = 0.0; // x1
    quadPoints(0,1) = 0.0; // y1
    quadPoints(1,0) = 1.0;
    quadPoints(1,1) = 0.0;
    quadPoints(2,0) = 1.0;
    quadPoints(2,1) = 1.0;
    quadPoints(3,0) = 0.0;
    quadPoints(3,1) = 1.0;

    // define meshes:
    int H1Order = polyOrder + 1;
    bool useTriangles = false;
    bool meshHasTriangles = useTriangles;
    
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
    
  //  // create a pointer to a new mesh:
  //  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
  //                                                navierStokesBF, H1Order, H1Order+pToAdd, useTriangles);

  //  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy ); // zero for now...
  //  IPPtr ip = initGraphInnerProductStokes(mu);

  //  SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) ); // accumulated solution
  //  SolutionPtr solnIncrement = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  //  solnIncrement->setReportConditionNumber(false);
    
    FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
    FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
    FunctionPtr zero = Function::zero();
    ParameterFunctionPtr Re_param = ParameterFunction::parameterFunction(Re);
    VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re_param,quadPoints,
                                                            horizontalCells,verticalCells,
                                                            H1Order, pToAdd,
                                                            u1_0, u2_0,  // BC for u
                                                            zero, zero); // zero forcing function
    
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
    
  //  if ( ! usePicardIteration ) { // we probably could afford to do pseudo-time with Picard, but choose not to
  //    // add time marching terms for momentum equations (v1 and v2):
    ParameterFunctionPtr dt_inv = ParameterFunction::parameterFunction(1.0 / dt); //Teuchos::rcp( new ConstantScalarFunction(1.0 / dt, "\\frac{1}{dt}") );
    if (artificialTimeStepping) {
  //    // LHS gets u_inc / dt:
      BFPtr bf = problem.bf();
      FunctionPtr dt_inv_fxn = Teuchos::rcp(dynamic_cast< Function* >(dt_inv.get()), false);
      bf->addTerm(-dt_inv_fxn * u1, v1);
      bf->addTerm(-dt_inv_fxn * u2, v2);
      problem.setIP( bf->graphNorm() ); // graph norm has changed...
    }
    
    if (useCompliantGraphNorm) {
      problem.setIP(problem.vgpNavierStokesFormulation()->scaleCompliantGraphNorm(dt_inv));
      // (otherwise, will use graph norm)
    }
    
  //  }
    
  //  if (rank==0) {
  //    cout << "********** STOKES BF **********\n";
  //    stokesBFMath->printTrialTestInteractions();
  //    cout << "\n\n********** NAVIER-STOKES BF **********\n";
  //    navierStokesBF->printTrialTestInteractions();
  //    cout << "\n\n";
  //  }
    
    // set initial guess (all zeros is probably a decent initial guess here)
  //  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0) );
  //  map< int, FunctionPtr > initialGuesses;
  //  initialGuesses[u1->ID()] = zero;
  //  initialGuesses[u2->ID()] = zero;
  //  initialGuesses[sigma11->ID()] = zero;
  //  initialGuesses[sigma12->ID()] = zero;
  //  initialGuesses[sigma21->ID()] = zero;
  //  initialGuesses[sigma22->ID()] = zero;
  //  initialGuesses[p->ID()] = zero;
  //  initialGuesses[u1hat->ID()] = zero;
  //  initialGuesses[u2hat->ID()] = zero;
  //  initialGuesses[t1n->ID()] = zero;
  //  initialGuesses[t2n->ID()] = zero;
  //  solution->projectOntoMesh(initialGuesses);
    
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
                                     streamBF, H1Order+pToAddForStreamFunction,
                                     H1Order+pToAdd+pToAddForStreamFunction, useTriangles);

    mesh->registerObserver(streamMesh); // will refine streamMesh in the same way as mesh.
    
    if (replayFile.length() > 0) {
      RefinementHistory refHistory;
      refHistory.loadFromFile(replayFile);
      refHistory.playback(mesh);
    }
    
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
      if (saveFile.length() > 0) {
        cout << "will save refinement history to file " << saveFile << endl;
      }
      if (replayFile.length() > 0) {
        cout << "will replay refinements from file " << replayFile << endl;
      }
    }
    
    ////////////////////   CREATE BCs   ///////////////////////
    SpatialFilterPtr entireBoundary = Teuchos::rcp( new SpatialFilterUnfiltered );

    FunctionPtr u1_prev = Function::solution(u1,solution);
    FunctionPtr u2_prev = Function::solution(u2,solution);
    
    FunctionPtr u1hat_prev = Function::solution(u1hat,solution);
    FunctionPtr u2hat_prev = Function::solution(u2hat,solution);
      
  //  if ( ! usePicardIteration ) {
  //    bc->addDirichlet(u1hat, entireBoundary, u1_0 - u1hat_prev);
  //    bc->addDirichlet(u2hat, entireBoundary, u2_0 - u2hat_prev);
  //  // as long as we don't subtract from the RHS, I think the following is actually right:
  ////    bc->addDirichlet(u1hat, entireBoundary, u1_0);
  ////    bc->addDirichlet(u2hat, entireBoundary, u2_0);
  //  } else {
  ////    bc->addDirichlet(u1hat, entireBoundary, u1_0);
  ////    bc->addDirichlet(u2hat, entireBoundary, u2_0);
  //    // experiment:
  //    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  //    pc->addConstraint(u1hat==u1_0,entireBoundary);
  //    pc->addConstraint(u2hat==u2_0,entireBoundary);
  //    solnIncrement->setFilter(pc);
  //  }
  //  bc->addZeroMeanConstraint(p);
  //  
    /////////////////// SOLVE OVERKILL //////////////////////
  //  if (compareWithOverkillMesh) {
  //    // TODO: fix this to make it work with Navier-Stokes
  //    cout << "WARNING: still need to switch overkill to handle nonlinear iteration...\n";
  //    overkillMesh = Mesh::buildQuadMesh(quadPoints, overkillMeshSize, overkillMeshSize,
  //                                       stokesBFMath, overkillPolyOrder, overkillPolyOrder+pToAdd, useTriangles);
  //    
  //    if (rank == 0) {
  //      cout << "Solving on overkill mesh (" << overkillMeshSize << " x " << overkillMeshSize << " elements, ";
  //      cout << overkillMesh->numGlobalDofs() <<  " dofs).\n";
  //    }
  //    overkillSolution = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
  //    overkillSolution->solve();
  //    if (rank == 0)
  //      cout << "...solved.\n";
  //    double overkillEnergyError = overkillSolution->energyErrorTotal();
  //    if (rank == 0)
  //      cout << "overkill energy error: " << overkillEnergyError << endl;
  //  }
    
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
      FunctionPtr zero = Function::zero();
      solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
      solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
    }
    
    FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
    
    double energyThreshold = 0.20; // for mesh refinements
    Teuchos::RCP<RefinementStrategy> refinementStrategy;
    if (useAdHocHPRefinements) {
  //    refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solution, energyThreshold, 1.0 / horizontalCells )); // no h-refinements allowed
  //    refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solnIncrement, energyThreshold, 1.0 / overkillMeshSize, overkillPolyOrder, rank==0 ));
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "need to build against LidDrivenFlowRefinementStrategy before using ad hoc hp refinements");
    } else {
//      if (rank==0) cout << "NOTE: using solution, not solnIncrement, for refinement strategy.\n";
//      refinementStrategy = Teuchos::rcp( new RefinementStrategy( solution, energyThreshold ));
      refinementStrategy = Teuchos::rcp( new RefinementStrategy( solnIncrement, energyThreshold ));
    }
    
    refinementStrategy->setEnforceOneIrregularity(enforceOneIrregularity);
    refinementStrategy->setReportPerCellErrors(reportPerCellErrors);

    Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
    Teuchos::RCP<NonlinearSolveStrategy> solveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
                                                                                                 stepSize,
                                                                                                 nonlinearRelativeEnergyTolerance));
    
    Teuchos::RCP<NonlinearSolveStrategy> finalSolveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
                                                                                                 stepSize,
                                                                                                 nonlinearRelativeEnergyTolerance / 10));

    
    
  //  solveStrategy->setUsePicardIteration(usePicardIteration);
    
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
          problem.iterate(useLineSearch);
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
        
        refinementStrategy->refine(false); //rank==0); // print to console on rank 0
        
        if (saveFile.length() > 0) {
          if (rank == 0) {
            refHistory->saveToFile(saveFile);
          }
        }
        
        // find top corner cells:
        vector< Teuchos::RCP<Element> > topCorners = mesh->elementsForPoints(topCornerPoints);
        if (rank==0) {// print out top corner cellIDs
          cout << "Refinement # " << refIndex+1 << " complete.\n";
          vector<int> cornerIDs;
          cout << "top-left corner ID: " << topCorners[0]->cellID() << endl;
          cout << "top-right corner ID: " << topCorners[1]->cellID() << endl;
          cout << mesh->activeElements().size() << " elements, " << mesh->numGlobalDofs() << " dofs.\n";
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
        problem.iterate(useLineSearch);
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
        problem.iterate(useLineSearch);
        if (rank==0) {
          cout << "Final iteration, L^2(incr) = " << incr_norm << endl;
        }
      }
      
  //    if (enforceLocalConservationInFinalSolve && !enforceLocalConservation) {
  //      solnIncrement->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  //    }
  //    
  //    finalSolveStrategy->solve(printToConsole);
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
      BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType,mesh,polyOrder) ); // enrich by trial space order
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
    //  mesh->unregisterObserver(streamMesh);
    //  streamMesh->registerObserver(mesh);
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
      
      FieldContainer<double> points = pointGrid(0, 1, 0, 1, 100);
      FieldContainer<double> pointData = solutionData(points, streamSolution, phi);
      GnuPlotUtil::writeXYPoints("phi_patch_navierStokes_cavity.dat", pointData);
      set<double> patchContourLevels = diagonalContourLevels(pointData,1);
      vector<string> patchDataPath;
      patchDataPath.push_back("phi_patch_navierStokes_cavity.dat");
      GnuPlotUtil::writeContourPlotScript(patchContourLevels, patchDataPath, "lidCavityNavierStokes.p");
      GnuPlotUtil::writeComputationalMeshSkeleton("nsCavityMesh", mesh);

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
    
  } catch ( choice::ArgException& e )
  {
    // There is no reason to do anything
  }
  
  return 0;
}
