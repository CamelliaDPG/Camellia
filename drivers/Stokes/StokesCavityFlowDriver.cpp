//
//  StokesCavityFlowDriver.cpp
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
#include "PenaltyConstraints.h"
#include "Solver.h"

#include "MLSolver.h"
//#include "CGSolver.h"
#include "MPIWrapper.h"

#include <Teuchos_GlobalMPISession.hpp>

#include "StreamDriverUtil.h"
#include "GnuPlotUtil.h"
#include "MeshUtilities.h"

#include "RefinementHistory.h"

#include "MeshFactory.h"

#include "SolutionExporter.h"

#include "QoIFilter.h"

#include "CamelliaConfig.h"

using namespace std;

//// just testing the mass flux integration
//// this one should mean that a 1x1 mesh has mass flux of 2.0
//class U1_0 : public SimpleFunction {
//  double _eps;
//public:
//  U1_0(double eps) {
//    _eps = eps;
//  }
//  double value(double x, double y) {
//    double tol = 1e-14;
//    if (abs(x-1.0) < tol) { // right boundary
//      return 1.0; // flow out
//    } else if (abs(x) < tol) { // left boundary
//      return -1.0; // flow out
//    } else {
//      return 0;
//    }
//  }
//};

// trying a smoother version
//class U1_0 : public SimpleFunction {
//  double _eps;
//public:
//  U1_0(double eps) {
//    _eps = eps;
//  }
//  double value(double x, double y) {
//    double tol = 1e-14;
//    if (abs(y-1.0) < tol) { // top boundary
//      if ( (abs(x) < _eps) ) { // top left
//        double x_s = x / _eps; // x_s: scaled x
//        return 3 * x_s * x_s - 2 * x_s * x_s * x_s;
//      } else if ( abs(1.0-x) < _eps) { // top right
//        double x_s = (1 - x) / _eps;
//        return 3 * x_s * x_s - 2 * x_s * x_s * x_s;
//      } else { // top middle
//        return 1;
//      }
//    } else { // not top boundary: 0.0
//      return 0.0;
//    }
//  }
//};

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

void writeStreamlines(double xMin, double xMax, double yMin, double yMax,
                      SolutionPtr solution, VarPtr u1, VarPtr u2, string filename) {
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
  solution->solutionValues(values2, u2->ID(), points);
  ofstream fout(filename.c_str());
  fout << setprecision(15);
  
  fout << "X = zeros(" << numPoints << ",1);\n";
  //    fout << "Y = zeros(numPoints);\n";
  fout << "U = zeros(" << numPoints << "," << numPoints << ");\n";
  fout << "V = zeros(" << numPoints << "," << numPoints << ");\n";
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
      fout << "V("<<i+1<<","<<j+1<<")=" << values2(pointIndex) << ";" << endl;
    }
  }
  fout.close();
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

void streamSolve(MeshPtr streamMesh, VarPtr q_s, VarPtr v_s, VarPtr phi, VarPtr phi_hat, FunctionPtr vorticity,
                 Teuchos::RCP<Solver> solver, bool useCondensedSolve, string refSuffix) {
  int rank = Teuchos::GlobalMPISession::getRank();
  
  ///////// SET UP & SOLVE STREAM SOLUTION /////////
  //  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution,sigma12 - sigma21) );
  RHSPtr streamRHS = RHS::rhs();
  streamRHS->addTerm(vorticity * q_s);
  ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
  
  BCPtr streamBC = BC::bc();
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0) );
  //  streamBC->addDirichlet(psin_hat, entireBoundary, u0_cross_n);
  streamBC->addDirichlet(phi_hat, SpatialFilter::allSpace(), zero);
  //  streamBC->addZeroMeanConstraint(phi);
  
  IPPtr streamIP = Teuchos::rcp( new IP );
  streamIP->addTerm(q_s);
  streamIP->addTerm(q_s->grad());
  streamIP->addTerm(v_s);
  streamIP->addTerm(v_s->div());
  SolutionPtr streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );
  
  //  mesh->unregisterObserver(streamMesh);
  //  streamMesh->registerObserver(mesh);
  //  RefinementStrategy streamRefinementStrategy( streamSolution, energyThreshold );
  //  for (int refIndex=0; refIndex < 3; refIndex++) {
  //    streamSolution->solve(solver);
  //    streamRefinementStrategy.refine(rank==0);
  //  }
  
  if (useCondensedSolve) {
    streamSolution->condensedSolve(solver);
  } else {
    streamSolution->solve(solver);
  }
  double energyErrorTotal = streamSolution->energyErrorTotal();
  if (rank == 0) {
    cout << "...solved.\n";
    cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
  }
  
  if (rank==0){
    //    writePatchValues(0, 1, 0, 1, streamSolution, phi, "phi_patch.m");
    //    writePatchValues(0, .1, 0, .1, streamSolution, phi, "phi_patch_detail.m");
    //    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
    //    writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
    
    map<double,string> scaleToName;
    scaleToName[1]   = "cavityPatch";
    scaleToName[0.1] = "cavityPatchEddy1";
    scaleToName[0.006] = "cavityPatchEddy2";
    scaleToName[0.0004] = "cavityPatchEddy3";
    scaleToName[0.00004] = "cavityPatchEddy4";
    
    for (map<double,string>::iterator entryIt=scaleToName.begin(); entryIt != scaleToName.end(); entryIt++) {
      double scale = entryIt->first;
      string name = entryIt->second;
      ostringstream fileNameStream;
      fileNameStream << name << refSuffix << ".dat";
      FieldContainer<double> patchPoints = pointGrid(0, scale, 0, scale, 100);
      FieldContainer<double> patchPointData = solutionData(patchPoints, streamSolution, phi);
      GnuPlotUtil::writeXYPoints(fileNameStream.str(), patchPointData);
      ostringstream scriptNameStream;
      scriptNameStream << name << ".p";
      set<double> contourLevels = diagonalContourLevels(patchPointData,4);
      vector<string> dataPaths;
      dataPaths.push_back(fileNameStream.str());
      GnuPlotUtil::writeContourPlotScript(contourLevels, dataPaths, scriptNameStream.str());
    }
    GnuPlotUtil::writeComputationalMeshSkeleton("cavityMesh", streamMesh);
  }
}

int main(int argc, char *argv[]) {
  Teuchos::GlobalMPISession mpiSession(&argc, &argv);
  int rank = Teuchos::GlobalMPISession::getRank();
  int pToAdd = 2; // for optimal test function approximation
  int pToAddForStreamFunction = pToAdd;
  double eps = 1.0/64.0; // width of ramp up to 1.0 for top BC;  eps == 0 ==> soln not in H1
  // epsilon above is chosen to match our initial 16x16 mesh, to avoid quadrature errors.
//  double eps = 0.0; // John Evans's problem: not in H^1
  bool induceCornerRefinements = false;
  bool symmetricRefinements = false; // symmetric across the horizontal midline
  bool singularityAvoidingInitialMesh = false;
  bool enforceLocalConservation = false;
  bool enforceOneIrregularity = true;
  bool reportPerCellErrors  = true;
  bool useMumps = true;
  bool useCG = false;
  bool useML = false;
  bool compareWithOverkillMesh = false;
  bool useDivergenceFreeVelocity = false;
  bool useWeightedGraphNorm = false;
  
  bool adaptForLRCornerVorticity = true;
  
  bool useExtendedPrecisionForOptimalTestInversion = false;
  bool useAdHocHPRefinements = false;
  bool usePenaltyConstraintsForDiscontinuousBC = false;
  bool useCondensedSolve = true;
  bool writeOverkillRHSToFile = false;
  string overkillRHSFile = "stokesCavityOverkillRHS_192_k2.rhs";
  int overkillMeshSize = 192;
  int overkillH1Order = 3; // H1 order
  string overkillSolnFile = "stokesCavityOverkill_192_k2.soln";
  
  double cgTol = 1e-8;
  int cgMaxIt = 400000;
  double energyThreshold = 0.20; // for mesh refinements
  
  string saveFile = "";//"stokesCavityReplay.replay";
  string replayFile = "";//"stokesCavityReplay.replay";

  Teuchos::RCP<Solver> solver;
  if (useMumps) {
#ifdef HAVE_AMESOS_MUMPS
    solver = Teuchos::rcp(new MumpsSolver());
#else
    if (rank==0)
      cout << "useMumps = true, but HAVE_AMESOS_MUMPS is unset.  Exiting...\n";
    exit(1);
#endif
  } else if (useML) {
    solver = Teuchos::rcp( new MLSolver );
  } else {
    solver = Teuchos::rcp(new KluSolver());
  }

//  Teuchos::RCP<Solver> cgSolver = Teuchos::rcp( new CGSolver(cgMaxIt, cgTol) );
  
  if (usePenaltyConstraintsForDiscontinuousBC) {
    // then eps should be 0 (NO RAMP)
    eps = 0;
  }
  
  // usage: polyOrder [numRefinements]
  // parse args:
  if ((argc != 3) && (argc != 2) && (argc != 4)) {
    cout << "Usage: StokesCavityFlowDriver fieldPolyOrder [numRefinements=10 [adaptThresh=0.20]\n";
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
  
  double minH = 0; // 1.0 / 8192.0;
  int maxPolyOrder = polyOrder; // forces just h-refinements
  
  if (compareWithOverkillMesh) {
    if (polyOrder + 1 > overkillH1Order) {
      cout << "Error: H1 order of coarse mesh exceeds the overkill mesh's H1 order.\n";
      exit(1);
    }
  }
  
  if (rank == 0) {
    cout << "numRefinements = " << numRefs << endl;
    cout << "polyOrder = " << polyOrder << endl;
    cout << "maxPolyOrder = " << maxPolyOrder << endl;
  }
  
  /////////////////////////// "VGP_CONFORMING" VERSION ///////////////////////
  VarFactory varFactory; 
  VarPtr tau1 = varFactory.testVar("\\tau_1", HDIV);  // tau_1
  VarPtr tau2 = varFactory.testVar("\\tau_2", HDIV);  // tau_2
  VarPtr v1 = varFactory.testVar("v_1", HGRAD); // v_1
  VarPtr v2 = varFactory.testVar("v_2", HGRAD); // v_2
  VarPtr q = varFactory.testVar("q", HGRAD); // q
  
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  VarPtr sigma1n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{1n}}");
  VarPtr sigma2n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{2n}}");
  VarPtr u1, u2, u;
  if (useDivergenceFreeVelocity) {
    u = varFactory.fieldVar("u", HDIV_FREE);
    u1 = u->x();
    u2 = u->y();
  } else {
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
  }
  
  VarPtr sigma11 = varFactory.fieldVar("\\sigma_11");
  VarPtr sigma12 = varFactory.fieldVar("\\sigma_12");
  VarPtr sigma21 = varFactory.fieldVar("\\sigma_21");
  VarPtr sigma22 = varFactory.fieldVar("\\sigma_22");
  VarPtr p = varFactory.fieldVar("p");
  
  double mu = 1;
  
  BFPtr stokesBFMath = Teuchos::rcp( new BF(varFactory) );  
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
  
  // v1:
  stokesBFMath->addTerm(mu * sigma11,v1->dx()); // (mu sigma1, grad v1) 
  stokesBFMath->addTerm(mu * sigma12,v1->dy());
  stokesBFMath->addTerm( - p, v1->dx() );
  stokesBFMath->addTerm( sigma1n, v1);
  
  // v2:
  stokesBFMath->addTerm(mu * sigma21,v2->dx()); // (mu sigma2, grad v2)
  stokesBFMath->addTerm(mu * sigma22,v2->dy());
  stokesBFMath->addTerm( -p, v2->dy());
  stokesBFMath->addTerm( sigma2n, v2);
  
  // q:
  stokesBFMath->addTerm(-u1,q->dx()); // (-u, grad q)
  stokesBFMath->addTerm(-u2,q->dy());
  stokesBFMath->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), q);
  
  stokesBFMath->setUseExtendedPrecisionSolveForOptimalTestFunctions(useExtendedPrecisionForOptimalTestInversion);
  
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
  
  streamBF->setUseExtendedPrecisionSolveForOptimalTestFunctions(useExtendedPrecisionForOptimalTestInversion);
  
  // define meshes:
  int H1Order = polyOrder + 1;
  int horizontalCells = 2, verticalCells = 2;
  bool useTriangles = false;
  bool nonConformingTraces = false;
  bool meshHasTriangles = useTriangles | singularityAvoidingInitialMesh;
  Teuchos::RCP<Mesh> mesh, streamMesh, overkillMesh;
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
  
  map< int, int > trialEnhancements;
  if (useWeightedGraphNorm) {
    if (useDivergenceFreeVelocity) {
      trialEnhancements[u->ID()] = 1;
    } else {
      trialEnhancements[u1->ID()] = 1;
      trialEnhancements[u2->ID()] = 1;
    }
  }
  
  if ( ! singularityAvoidingInitialMesh ) {
    // create a pointer to a new mesh:
    mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                               stokesBFMath, H1Order, H1Order+pToAdd, useTriangles, nonConformingTraces, trialEnhancements);
    streamMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                     streamBF, H1Order, H1Order+pToAddForStreamFunction, useTriangles);
  } else {
    vector<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
    // top (left to right):
    A[0] = 0.0;            A[1] = 1.0;
    B[0] = 1.0 / 64.0;     B[1] = 1.0;
    C[0] = 1.0 - 1 / 64.0; C[1] = 1.0;
    D[0] = 1.0;            D[1] = 1.0;
    // bottom (right to left):
    E[0] = 1.0;            E[1] = 0.0;
    F[0] = 1.0 - 1/64.0;   F[1] = 0.0;
    G[0] = 1.0 / 64.0;     G[1] = 0.0;
    H[0] = 0;              H[1] = 0.0;
    
    vector<vector<double> > vertices;
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
    // must go counterclockwise:
    el1.push_back(A_index); el1.push_back(H_index); el1.push_back(G_index); el1.push_back(B_index);
    el2.push_back(B_index); el2.push_back(G_index); el2.push_back(F_index); el2.push_back(C_index);
    el3.push_back(C_index); el3.push_back(F_index); el3.push_back(E_index); el3.push_back(D_index);
    elementVertices.push_back(el1);
    elementVertices.push_back(el2);
    elementVertices.push_back(el3);
//    FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
//    // outer square
//    A(0) = 0.0; A(1) = 0.0;
//    B(0) = 1.0; B(1) = 0.0;
//    C(0) = 1.0; C(1) = 1.0;
//    D(0) = 0.0; D(1) = 1.0;
//    // center point:
//    E(0) = 0.5; E(1) = 0.5;
//    // bisectors of outer square:
//    F(0) = 0.0; F(1) = 0.5;
//    G(0) = 1.0; G(1) = 0.5;
//    H(0) = 0.5; H(1) = 0.0;
//    
//    vector<FieldContainer<double> > vertices;
//    vertices.push_back(A); int A_index = 0;
//    vertices.push_back(B); int B_index = 1;
//    vertices.push_back(C); int C_index = 2;
//    vertices.push_back(D); int D_index = 3;
//    vertices.push_back(E); int E_index = 4;
//    vertices.push_back(F); int F_index = 5;
//    vertices.push_back(G); int G_index = 6;
//    vertices.push_back(H); int H_index = 7;
//    
//    vector< vector<int> > elementVertices;
//    vector<int> el1, el2, el3, el4, el5;
//    // must go counterclockwise:
//    el1.push_back(A_index); el1.push_back(H_index); el1.push_back(E_index); el1.push_back(F_index);
//    el2.push_back(H_index); el2.push_back(B_index); el2.push_back(G_index); el2.push_back(E_index);
//    el3.push_back(F_index); el3.push_back(E_index); el3.push_back(D_index); 
//    el4.push_back(D_index); el4.push_back(E_index); el4.push_back(C_index); 
//    el5.push_back(E_index); el5.push_back(G_index); el5.push_back(C_index);
//    elementVertices.push_back(el1);
//    elementVertices.push_back(el2);
//    elementVertices.push_back(el3);
//    elementVertices.push_back(el4);
//    elementVertices.push_back(el5);
    
    mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, stokesBFMath, H1Order, pToAdd) );
    streamMesh = Teuchos::rcp( new Mesh(vertices, elementVertices, streamBF, H1Order, pToAddForStreamFunction) );
    
    // trapezoidal singularity-avoiding mesh below.  (triangular above)
    //    FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
    //    // outer square
    //    A(0) = 0.0; A(1) = 0.0;
    //    B(0) = 1.0; B(1) = 0.0;
    //    C(0) = 1.0; C(1) = 1.0;
    //    D(0) = 0.0; D(1) = 1.0;
    //    // inner square:
    //    E(0) = 0.25; E(1) = 0.25;
    //    F(0) = 0.75; F(1) = 0.25;
    //    G(0) = 0.75; G(1) = 0.75;
    //    H(0) = 0.25; H(1) = 0.75;
    //    vector<FieldContainer<double> > vertices;
    //    vertices.push_back(A); int A_index = 0;
    //    vertices.push_back(B); int B_index = 1;
    //    vertices.push_back(C); int C_index = 2;
    //    vertices.push_back(D); int D_index = 3;
    //    vertices.push_back(E); int E_index = 4;
    //    vertices.push_back(F); int F_index = 5;
    //    vertices.push_back(G); int G_index = 6;
    //    vertices.push_back(H); int H_index = 7;
    //    vector< vector<int> > elementVertices;
    //    vector<int> el1, el2, el3, el4, el5;
    //    // outside trapezoidal elements:
    //    el1.push_back(A_index); el1.push_back(B_index); el1.push_back(F_index); el1.push_back(E_index);
    //    el2.push_back(B_index); el2.push_back(C_index); el2.push_back(G_index); el2.push_back(F_index);
    //    el3.push_back(C_index); el3.push_back(D_index); el3.push_back(H_index); el3.push_back(G_index);
    //    el4.push_back(D_index); el4.push_back(A_index); el4.push_back(E_index); el4.push_back(H_index);
    //    // interior square element
    //    el5.push_back(E_index); el5.push_back(F_index); el5.push_back(G_index); el5.push_back(H_index);
    //    elementVertices.push_back(el1);
    //    elementVertices.push_back(el2);
    //    elementVertices.push_back(el3);
    //    elementVertices.push_back(el4);
    //    elementVertices.push_back(el5);
    //    mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, stokesBFMath, H1Order, pToAdd) );
  }
  
  // for divergence-free solutions, we need to project onto a standard L2 space so that we can take derivatives
  // and solve for the streamfunction
  SolutionPtr L2VelocitySolution;
  VarPtr u1_L2, u2_L2;
  if (useDivergenceFreeVelocity) {
    VarFactory velocityVarFactory;
    u1_L2 = velocityVarFactory.fieldVar("u1");
    u2_L2 = velocityVarFactory.fieldVar("u2");
    VarPtr dummyFlux = velocityVarFactory.fluxVar("un");
    VarPtr dummyTest = velocityVarFactory.testVar("v", HGRAD);
    BFPtr velocityBF = Teuchos::rcp( new BF(velocityVarFactory) );
    MeshPtr velocityMesh = MeshFactory::quadMesh(velocityBF, H1Order, pToAdd, 1.0, 1.0, horizontalCells, verticalCells);
    mesh->registerObserver(velocityMesh);
    L2VelocitySolution = Teuchos::rcp( new Solution(velocityMesh) );
  }
  
  mesh->registerObserver(streamMesh); // will refine streamMesh in the same way as mesh.
  mesh->registerObserver(refHistory);
  
  if (replayFile.length() > 0) {
    RefinementHistory refHistory;
    refHistory.loadFromFile(replayFile);
    refHistory.playback(mesh);
  }
  
  Teuchos::RCP<Solution> overkillSolution;
  map<int, double> dofsToL2error; // key: numGlobalDofs, value: total L2error compared with overkill
  map<int, double> dofsToBestL2error;

  vector< VarPtr > fields;
  if (useDivergenceFreeVelocity) {
    fields.push_back(u);
  } else {
    fields.push_back(u1);
    fields.push_back(u2);
  }
  fields.push_back(sigma11);
  fields.push_back(sigma12);
  fields.push_back(sigma21);
  fields.push_back(sigma22);
  fields.push_back(p);
  
  if (rank == 0) {
    if ( ! singularityAvoidingInitialMesh )
      cout << "Starting mesh has " << mesh->activeElements().size() << " elements and ";
    else
      cout << "Using singularity-avoiding initial mesh: 5 elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    cout << "eps for top BC = " << eps << endl;
    
    if (useTriangles) {
      cout << "Using triangles.\n";
    }
    if (useWeightedGraphNorm) {
      cout << "Using h-weighted graph norm.\n";
    }
    if (induceCornerRefinements) {
      cout << "Artificially inducing refinements in bottom corners.\n";
    }
    if (symmetricRefinements) {
      cout << "Imposing symmetric refinements on top and bottom of mesh.\n";
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
    
    if (usePenaltyConstraintsForDiscontinuousBC) {
      cout << "Using penalty constraints for discontinuous BC (==> NO RAMP).\n";
    }
    if (useDivergenceFreeVelocity) {
      cout << "Using divergence-free velocity.\n";
    }
    
    if (useCondensedSolve) {
      cout << "Using condensed solve.\n";
    } else {
      cout << "Not using condensed solve.\n";
    }
  }
  
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
  
  IPPtr ip;
  
  IPPtr qoptIP = Teuchos::rcp(new IP());
  
  double beta = 1.0;
  FunctionPtr h = Teuchos::rcp( new hFunction() );
  
  if (useWeightedGraphNorm) {
    qoptIP->addTerm( mu * v1->dx() + tau1->x() ); // sigma11
    qoptIP->addTerm( mu * v1->dy() + tau1->y() ); // sigma12
    qoptIP->addTerm( mu * v2->dx() + tau2->x() ); // sigma21
    qoptIP->addTerm( mu * v2->dy() + tau2->y() ); // sigma22
    qoptIP->addTerm( v1->dx() + v2->dy() );       // pressure
    qoptIP->addTerm( h * tau1->div() - h * q->dx() ); // u1
    qoptIP->addTerm( h * tau2->div() - h * q->dy() ); // u2
    qoptIP->addTerm( (mu / h) * v1 );
    qoptIP->addTerm( (mu / h) * v2 );
    qoptIP->addTerm(  q );
    qoptIP->addTerm( tau1 );
    qoptIP->addTerm( tau2 );
  } else {
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
  }
  
  ip = qoptIP;
  
  if (rank==0) {
    int cellID = mesh->activeElements()[0]->cellID(); // just sample the first active element
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID, true);
    DofOrderingPtr testSpace = mesh->getElement(cellID)->elementType()->testOrderPtr;
    double conditionNumber = qoptIP->computeMaxConditionNumber(testSpace,basisCache);
    cout << "Gram matrix cond # for cell " << cellID << " : " << conditionNumber << endl;
  }
  
  if (rank==0) 
    ip->printInteractions();
  
  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr entireBoundary = Teuchos::rcp( new UnitSquareBoundary );
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  FunctionPtr un_0 = Teuchos::rcp( new Un_0(eps) );
  FunctionPtr u0_cross_n = Teuchos::rcp( new U0_cross_n(eps) );
  
  if (! usePenaltyConstraintsForDiscontinuousBC ) {
    bc->addDirichlet(u1hat, entireBoundary, u1_0);
  }
  bc->addDirichlet(u2hat, entireBoundary, u2_0);
  bc->addZeroMeanConstraint(p);
  
  ////////////////////   CREATE RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs(); // zero for now...
  rhs->addTerm(Function::zero() * v1); // just because goal-oriented doesn't handle empty RHS well just yet...
  
  /////////////////// SOLVE OVERKILL //////////////////////
  if (compareWithOverkillMesh) {
    overkillMesh = MeshFactory::buildQuadMesh(quadPoints, overkillMeshSize, overkillMeshSize,
                                       stokesBFMath, overkillH1Order, overkillH1Order+pToAdd, useTriangles);
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
      if (rank == 0) {
        cout << "Solving on overkill mesh (" << overkillMeshSize << " x " << overkillMeshSize << " elements, ";
        cout << overkillMesh->numGlobalDofs() <<  " dofs).\n";
      }
      overkillSolution = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
      overkillSolution->setWriteRHSToMatrixMarketFile(writeOverkillRHSToFile, overkillRHSFile);
      if (useCondensedSolve) {
        overkillSolution->condensedSolve(solver);
      } else {
        overkillSolution->solve(solver);
      }
      if (rank == 0) {
        cout << "...solved.\n";
        if (overkillSolnFile.length() > 0) {
          cout << "writing to disk...\n";
          overkillSolution->writeToFile(overkillSolnFile);
          cout << "Wrote overkill solution to " << overkillSolnFile << endl;
        }
      }
      double overkillEnergyError = overkillSolution->energyErrorTotal();
      if (rank == 0)
        cout << "overkill energy error: " << setprecision(15) << overkillEnergyError << endl;
    }
  }
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  SolutionPtr qoiSolution;
  if (adaptForLRCornerVorticity) {
    qoiSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  }
  
  FunctionPtr vorticity;
  if (useDivergenceFreeVelocity) {
    vorticity = Teuchos::rcp( new PreviousSolutionFunction(L2VelocitySolution, - u1_L2->dy() + u2_L2->dx() ) );
  } else {
    vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );
  }
//  solution->setReportConditionNumber(true);
  if (usePenaltyConstraintsForDiscontinuousBC) {
    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
    pc->addConstraint(u1hat==u1_0,entireBoundary);
    solution->setFilter(pc);
  }
  
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
  
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  if (useAdHocHPRefinements) 
//    refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solution, energyThreshold, 1.0 / horizontalCells )); // no h-refinements allowed
    if (compareWithOverkillMesh) {
      refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solution, energyThreshold, 1.0 / overkillMeshSize, overkillH1Order, rank==0 ));
    } else {
      refinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solution, energyThreshold, 0, 15, rank==0 ));
    }
  else if (symmetricRefinements) {
    // we again use the problem-specific LidDrivenFlowRefinementStrategy, but now with hMin = 0, and maxP = H1Order-1 (i.e. never refine in p)
    Teuchos::RCP<LidDrivenFlowRefinementStrategy> lidRefinementStrategy = Teuchos::rcp( new LidDrivenFlowRefinementStrategy( solution, energyThreshold, minH,
                                                                                                                            maxPolyOrder, rank==0 ));
    lidRefinementStrategy->setSymmetricRefinements(true);
    refinementStrategy = lidRefinementStrategy;
  } else if (adaptForLRCornerVorticity) {
    // define goal:
    vector< Teuchos::RCP<Element> > corners = mesh->elementsForPoints(bottomCornerPoints);
    int lrCornerCellID = corners[0]->cellID();
    FunctionPtr weight = Function::cellCharacteristic(lrCornerCellID);
    LinearTermPtr trialFunctional = weight * ( - sigma12 + sigma21 ); // weighted vorticity
    qoiSolution->setFilter(QoIFilter::qoiFilter(trialFunctional));
    refinementStrategy = Teuchos::rcp( new RefinementStrategy( qoiSolution, energyThreshold, minH ));
  } else {
    refinementStrategy = Teuchos::rcp( new RefinementStrategy( solution, energyThreshold, minH ));
  }
  
  refinementStrategy->setEnforceOneIrregularity(enforceOneIrregularity);
  refinementStrategy->setReportPerCellErrors(reportPerCellErrors);
  
  if (!useCG) {
    if (useCondensedSolve) {
      solution->condensedSolve(solver);
      if (qoiSolution.get()) {
        qoiSolution->condensedSolve(solver);
      }
    } else {
      solution->solve(solver);
      if (qoiSolution.get()) {
        qoiSolution->solve(solver);
      }
    }
    streamSolve(streamMesh, q_s, v_s, phi, phi_hat, vorticity, solver, useCondensedSolve, "_ref0");
  } else {
    cout << "WARNING: cgSolver unset.\n";
  }
  
  if (useDivergenceFreeVelocity) {
    FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u->x()) );
    FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u->y()) );
    map<int, FunctionPtr > projectionMap;
    projectionMap[u1_L2->ID()] = u1_prev;
    projectionMap[u2_L2->ID()] = u2_prev;
    ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true);
    ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);
    L2VelocitySolution->projectOntoMesh(projectionMap);
  }
  
  polyOrderFunction->writeValuesToMATLABFile(mesh, "cavityFlowPolyOrders_0.m");
  FunctionPtr ten = Teuchos::rcp( new ConstantScalarFunction(10) );
  ten->writeBoundaryValuesToMATLABFile(mesh, "skeleton_0.dat");
  
  // report vorticity value that's often reported in the literature
  double vort_x = 0.0, vort_y = 0.95;
  double vorticityValue = Function::evaluate(vorticity, vort_x, vort_y);
  if (rank==0) {
    cout << setprecision(15) << endl;
    cout << "vorticity at (0,0.95) = " << vorticityValue << endl;
  }
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){
    if (compareWithOverkillMesh) {
      Teuchos::RCP<Solution> bestSoln = Teuchos::rcp( new Solution(solution->mesh(), bc, rhs, ip) );
      overkillSolution->projectFieldVariablesOntoOtherSolution(bestSoln);
      if (rank==0) {
        VTKExporter exporter(solution, mesh, varFactory);
        ostringstream cavityRefinement;
        cavityRefinement << "cavity_solution_refinement_" << refIndex;
        exporter.exportSolution(cavityRefinement.str());
        VTKExporter exporterBest(bestSoln, mesh, varFactory);
        ostringstream bestRefinement;
        bestRefinement << "cavity_best_refinement_" << refIndex;
        exporterBest.exportSolution(bestRefinement.str());
      }
      Teuchos::RCP<Solution> bestSolnOnOverkillMesh = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
      bestSoln->projectFieldVariablesOntoOtherSolution(bestSolnOnOverkillMesh);
      
      FunctionPtr p_best = Teuchos::rcp( new PreviousSolutionFunction(bestSoln,p) );
      double p_avg = p_best->integrate(mesh);
      if (rank==0)
        cout << "Integral of best solution pressure: " << p_avg << endl;
      
      // determine error as difference between our solution and overkill
      bestSolnOnOverkillMesh->addSolution(overkillSolution,-1.0);
      
      Teuchos::RCP<Solution> adaptiveSolnOnOverkillMesh = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
      solution->projectFieldVariablesOntoOtherSolution(adaptiveSolnOnOverkillMesh);
//      if (refIndex==numRefs-1) { // last refinement
//        solution->writeFieldsToFile(p->ID(),"pressure.m");
//        overkillSolution->writeFieldsToFile(p->ID(), "pressure_overkill.m");
//      }

      // determine error as difference between our solution and overkill
      adaptiveSolnOnOverkillMesh->addSolution(overkillSolution,-1.0);

//      if (refIndex==numRefs-1) { // last refinement
//        projectedSoln->writeFieldsToFile(p->ID(), "pressure_error_vs_overkill.m");
//      }
      double L2errorSquared = 0.0;
      double bestL2errorSquared = 0.0;
      for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
        VarPtr var = *fieldIt;
        int fieldID = var->ID();
        double L2error = adaptiveSolnOnOverkillMesh->L2NormOfSolutionGlobal(fieldID);
        L2errorSquared += L2error * L2error;
        double bestL2error = bestSolnOnOverkillMesh->L2NormOfSolutionGlobal(fieldID);
        bestL2errorSquared += bestL2error * bestL2error;
        if (rank==0) {
          cout << "L^2 error for " << var->name() << ": " << L2error;
          cout << " (vs. best error of " << bestL2error << ")\n";
        }
      }
      if (rank==0) {
        VTKExporter exporter(adaptiveSolnOnOverkillMesh, mesh, varFactory);
        ostringstream errorForRefinement;
        errorForRefinement << "overkillError_refinement_" << refIndex;
        exporter.exportSolution(errorForRefinement.str());
      }

//      double maxError = 0.0;
//      for (int i=0; i<overkillMesh->numElements(); i++) {
//        double error = projectedSoln->L2NormOfSolutionInCell(p->ID(),i);
//        if (error > maxError) {
//          if (rank == 0)
//            cout << "New maxError for pressure found in cell " << i << ": " << error << endl;
//          maxError = error;
//        }
//      }
      
      int numGlobalDofs = mesh->numGlobalDofs();
      if (rank==0) {
        cout << "for " << numGlobalDofs << " dofs, total L2 error: " << sqrt(L2errorSquared);
        cout << " (vs. best error of " << sqrt(bestL2errorSquared) << ")\n";
      }
      dofsToL2error[numGlobalDofs] = sqrt(L2errorSquared);
      dofsToBestL2error[numGlobalDofs] = sqrt(bestL2errorSquared);
    }
    if (adaptForLRCornerVorticity) {
      // update refinement strategy
      vector< Teuchos::RCP<Element> > corners = mesh->elementsForPoints(bottomCornerPoints);
      int lrCornerCellID = corners[0]->cellID();
      FunctionPtr weight = Function::cellCharacteristic(lrCornerCellID);
      LinearTermPtr trialFunctional = weight * ( - sigma12 + sigma21 ); // weighted vorticity
      qoiSolution->setFilter(QoIFilter::qoiFilter(trialFunctional));
    }
    refinementStrategy->refine(rank==0); // print to console on rank 0
    if (! MeshTestUtility::checkMeshConsistency(mesh)) {
      if (rank==0) cout << "checkMeshConsistency returned false after refinement.\n";
    }
    // find top corner cells:
    vector< Teuchos::RCP<Element> > topCorners = mesh->elementsForPoints(topCornerPoints);
    if (rank==0) {// print out top corner cellIDs
      cout << "Refinement # " << refIndex+1 << " complete.\n";
      vector<int> cornerIDs;
      cout << "top-left corner ID: " << topCorners[0]->cellID() << endl;
      cout << "top-right corner ID: " << topCorners[1]->cellID() << endl;
      if (singularityAvoidingInitialMesh) {
        cout << "other top-left corner ID: " << topCorners[2]->cellID() << endl;
        cout << "other top-right corner ID: " << topCorners[3]->cellID() << endl;
      }
    }
    if (induceCornerRefinements) {
      // induce refinements in bottom corners:
      vector< Teuchos::RCP<Element> > corners = mesh->elementsForPoints(bottomCornerPoints);
      vector<GlobalIndexType> cornerIDs;
      cornerIDs.push_back(corners[0]->cellID());
      cornerIDs.push_back(corners[1]->cellID());
      mesh->hRefine(cornerIDs, RefinementPattern::regularRefinementPatternQuad());
    }
    // solve on the refined mesh:
    if (!useCG) {
      if (useCondensedSolve) {
        solution->condensedSolve(solver);
        if (qoiSolution.get()) {
          qoiSolution->condensedSolve(solver);
        }
      } else {
        solution->solve(solver);
        if (qoiSolution.get()) {
          qoiSolution->solve(solver);
        }
      }
    } else {
      cout << "WARNING: cgSolver unset.\n";
    }
    ostringstream refSuffix;
    refSuffix << "_ref" << refIndex;
    streamSolve(streamMesh, q_s, v_s, phi, phi_hat, vorticity, solver, useCondensedSolve, refSuffix.str());
    
    if (useDivergenceFreeVelocity) {
      FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u->x()) );
      FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u->y()) );
      map<int, FunctionPtr > projectionMap;
      projectionMap[u1_L2->ID()] = u1_prev;
      projectionMap[u2_L2->ID()] = u2_prev;
      ((PreviousSolutionFunction*) u1_prev.get())->setOverrideMeshCheck(true);
      ((PreviousSolutionFunction*) u2_prev.get())->setOverrideMeshCheck(true);
      L2VelocitySolution->projectOntoMesh(projectionMap);
    }
    
    // report vorticity value that's often reported in the literature
    double vort_x = 0.0, vort_y = 0.95;
    double vorticityValue = Function::evaluate(vorticity, vort_x, vort_y);
    if (rank==0) {
      cout << setprecision(15) << endl;
      cout << "vorticity at (0,0.95) = " << vorticityValue << endl;
    }
    
    ostringstream meshOutputFileName, skeletonOutputFileName;
    meshOutputFileName << "cavityFlowPolyOrders_" << refIndex + 1 << ".m";
    polyOrderFunction->writeValuesToMATLABFile(mesh, meshOutputFileName.str());
    skeletonOutputFileName << "skeleton_" << refIndex + 1 << ".dat";
    ten->writeBoundaryValuesToMATLABFile(mesh, skeletonOutputFileName.str());
  }
  if (compareWithOverkillMesh) {
    Teuchos::RCP<Solution> bestSoln = Teuchos::rcp( new Solution(solution->mesh(), bc, rhs, ip) );
    overkillSolution->projectFieldVariablesOntoOtherSolution(bestSoln);
    Teuchos::RCP<Solution> bestSolnOnOverkillMesh = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
    bestSoln->projectFieldVariablesOntoOtherSolution(bestSolnOnOverkillMesh);
    FunctionPtr p_best = Teuchos::rcp( new PreviousSolutionFunction(bestSoln,p) );
    double p_avg = p_best->integrate(mesh);
    if (rank==0)
      cout << "Integral of best solution pressure: " << p_avg << endl;
    
    // determine error as difference between our solution and overkill
    bestSolnOnOverkillMesh->addSolution(overkillSolution,-1.0);
    
    Teuchos::RCP<Solution> projectedSoln = Teuchos::rcp( new Solution(overkillMesh, bc, rhs, ip) );
    solution->projectFieldVariablesOntoOtherSolution(projectedSoln);
    
    projectedSoln->addSolution(overkillSolution,-1.0);
    double L2errorSquared = 0.0;
    double bestL2errorSquared = 0.0;
    for (vector< VarPtr >::iterator fieldIt=fields.begin(); fieldIt !=fields.end(); fieldIt++) {
      VarPtr var = *fieldIt;
      int fieldID = var->ID();
      double L2error = projectedSoln->L2NormOfSolutionGlobal(fieldID);
      double bestL2error = bestSolnOnOverkillMesh->L2NormOfSolutionGlobal(fieldID);
      bestL2errorSquared += bestL2error * bestL2error;
      if (rank==0) {
        cout << "L^2 error for " << var->name() << ": " << L2error;
        cout << " (vs. best error of " << bestL2error << ")\n";
      }
      L2errorSquared += L2error * L2error;
    }
    int numGlobalDofs = mesh->numGlobalDofs();
    dofsToL2error[numGlobalDofs] = sqrt(L2errorSquared);
    dofsToBestL2error[numGlobalDofs] = sqrt(bestL2errorSquared);
    if (rank==0) {
      VTKExporter exporter(solution, mesh, varFactory);
      ostringstream errorForRefinement;
      errorForRefinement << "overkillError_refinement_" << numRefs;
      exporter.exportSolution(errorForRefinement.str());
    }
  }
  
//  cout << "on rank " << rank << ", about to compute max local condition number" << endl;
//  double maxConditionNumber = MeshUtilities::computeMaxLocalConditionNumber(qoptIP, mesh, "cavity_maxConditionIPMatrix.dat");
  
  double energyErrorTotal = solution->energyErrorTotal();
  if (rank == 0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
//    cout << "Max Gram matrix condition number: " << maxConditionNumber << endl;
  }
  
  streamSolve(streamMesh, q_s, v_s, phi, phi_hat, vorticity, solver, useCondensedSolve, "");
  
  FunctionPtr u_prev, u_div;
  if (useDivergenceFreeVelocity) {
    u_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u) );
    u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u->div() ) );
  } else {
    FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u1) );
    FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u2) );
    u_prev = Function::vectorize(u1_prev, u2_prev);
    u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dx() + u2->dy() ) );
  }

  FunctionPtr u_dot_u = u_prev * u_prev;
  FunctionPtr u_mag = Teuchos::rcp( new SqrtFunction( u_dot_u ) );
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
  
  CellTopoPtrLegacy quadTopoPtr = Teuchos::rcp(new shards::CellTopology(shards::getCellTopologyData<shards::Quadrilateral<4> >() ));
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
    vector<GlobalIndexType> cellIDs;
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
  
  if (saveFile.length() > 0) {
    if (rank == 0) {
      refHistory->saveToFile(saveFile);
    }
  }
  
  if (rank==0) {
    if (compareWithOverkillMesh) {
      cout << "******* Adaptivity Convergence Report *******\n";
      cout << "dofs\tL2 error\n";
      for (map<int,double>::iterator entryIt=dofsToL2error.begin(); entryIt != dofsToL2error.end(); entryIt++) {
        int dofs = entryIt->first;
        double err = entryIt->second;
        cout << dofs << "\t" << err;
        double bestError = dofsToBestL2error[dofs];
        cout << "\t" << bestError << endl;
      }
      ofstream fout("overkillComparison.txt");
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
  
  return 0;
}
