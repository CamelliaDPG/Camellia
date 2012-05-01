//
//  StokesCavityFlowDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "RefinementPattern.h"
#include "PreviousSolutionFunction.h"
#include "LagrangeConstraints.h"

typedef Teuchos::RCP<Element> ElementPtr;
typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;


#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

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

int main(int argc, char *argv[]) {
  int rank = 0;
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
#else
#endif
  int pToAdd = 1; // for optimal test function approximation
  bool weightTestNormDerivativesByH = false;
  double eps = 1.0/64.0; // width of ramp up to 1.0 for top BC;  eps == 0 ==> soln not in H1
  // epsilon above is chosen to match our initial 16x16 mesh, to avoid quadrature errors.
//  double eps = 0.0; // John Evans's problem: not in H^1
  bool induceCornerRefinements = true;
  bool singularityAvoidingInitialMesh = false;
  bool enforceLocalConservation = true;
  
  // usage: polyOrder [numRefinements]
  // parse args:
  if ((argc != 3) && (argc != 2)) {
    cout << "Usage: StokesCavityFlowDriver fieldPolyOrder [numRefinements=10]\n";
    return -1;
  }
  int polyOrder = atoi(argv[1]);
  int numRefs = 10;
  if ( argc == 3) {
    numRefs = atoi(argv[2]);
  }
  if (rank == 0)
    cout << "numRefinements = " << numRefs << endl;
  
  /////////////////////////// "MATH_CONFORMING" VERSION ///////////////////////
  VarFactory varFactory; 
  VarPtr q1 = varFactory.testVar("q_1", HDIV);
  VarPtr q2 = varFactory.testVar("q_2", HDIV);
  VarPtr v1 = varFactory.testVar("v_1", HGRAD);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD);
  VarPtr v3 = varFactory.testVar("v_3", HGRAD);
  //  VarPtr testOne; // used for local conservation, if requested
  //  if (enforceLocalConservation) {
  //    testOne = varFactory.testVar("1", CONSTANT_SCALAR);
  //  }
  
  VarPtr u1hat = varFactory.traceVar("\\widehat{u}_1");
  VarPtr u2hat = varFactory.traceVar("\\widehat{u}_2");
  //  VarPtr uhatn = varFactory.fluxVar("\\widehat{u}_n");
  VarPtr sigma1n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{1n}}");
  VarPtr sigma2n = varFactory.fluxVar("\\widehat{P - \\mu \\sigma_{2n}}");
  VarPtr u1 = varFactory.fieldVar("u_1");
  VarPtr u2 = varFactory.fieldVar("u_2");
  VarPtr sigma11 = varFactory.fieldVar("\\sigma_11");
  VarPtr sigma12 = varFactory.fieldVar("\\sigma_12");
  VarPtr sigma21 = varFactory.fieldVar("\\sigma_21");
  VarPtr sigma22 = varFactory.fieldVar("\\sigma_22");
  VarPtr p = varFactory.fieldVar("p");
  
  double mu = 1;
  
  BFPtr stokesBFMath = Teuchos::rcp( new BF(varFactory) );  
  // q1 terms:
  stokesBFMath->addTerm(u1,q1->div());
  stokesBFMath->addTerm(sigma11,q1->x()); // (sigma1, q1)
  stokesBFMath->addTerm(sigma12,q1->y());
  stokesBFMath->addTerm(-u1hat, q1->dot_normal());
  
  // q2 terms:
  stokesBFMath->addTerm(u2, q2->div());
  stokesBFMath->addTerm(sigma21,q2->x()); // (sigma2, q2)
  stokesBFMath->addTerm(sigma22,q2->y());
  stokesBFMath->addTerm(-u2hat, q2->dot_normal());
  
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
  
  // v3:
  stokesBFMath->addTerm(-u1,v3->dx()); // (-u, grad v3)
  stokesBFMath->addTerm(-u2,v3->dy());
  stokesBFMath->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), v3);
  
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
  streamBF->addTerm(-psi_1, q_s->dx());
  streamBF->addTerm(-psi_2, q_s->dy());
  streamBF->addTerm(psin_hat, q_s);
  
  streamBF->addTerm(-psi_1, v_s->x());
  streamBF->addTerm(-psi_2, v_s->y());
  streamBF->addTerm(-phi, v_s->div());
  streamBF->addTerm(phi_hat, v_s->dot_normal());
  
  // define meshes:
  int H1Order = polyOrder + 1;
  int horizontalCells = 2, verticalCells = 2;
  bool useTriangles = false;
  bool meshHasTriangles = useTriangles | singularityAvoidingInitialMesh;
  Teuchos::RCP<Mesh> mesh, streamMesh;
  
  if ( ! singularityAvoidingInitialMesh ) {
    FieldContainer<double> quadPoints(4,2);
    
    quadPoints(0,0) = 0.0; // x1
    quadPoints(0,1) = 0.0; // y1
    quadPoints(1,0) = 1.0;
    quadPoints(1,1) = 0.0;
    quadPoints(2,0) = 1.0;
    quadPoints(2,1) = 1.0;
    quadPoints(3,0) = 0.0;
    quadPoints(3,1) = 1.0;
    
    // create a pointer to a new mesh:
    mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                               stokesBFMath, H1Order, H1Order+pToAdd, useTriangles);
    streamMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                     streamBF, H1Order+2, H1Order+pToAdd+2, useTriangles);
  } else {
    FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
    // outer square
    A(0) = 0.0; A(1) = 0.0;
    B(0) = 1.0; B(1) = 0.0;
    C(0) = 1.0; C(1) = 1.0;
    D(0) = 0.0; D(1) = 1.0;
    // center point:
    E(0) = 0.5; E(1) = 0.5;
    // bisectors of outer square:
    F(0) = 0.0; F(1) = 0.5;
    G(0) = 1.0; G(1) = 0.5;
    H(0) = 0.5; H(1) = 0.0;
    
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
    // must go counterclockwise:
    el1.push_back(A_index); el1.push_back(H_index); el1.push_back(E_index); el1.push_back(F_index);
    el2.push_back(H_index); el2.push_back(B_index); el2.push_back(G_index); el2.push_back(E_index);
    el3.push_back(F_index); el3.push_back(E_index); el3.push_back(D_index); 
    el4.push_back(D_index); el4.push_back(E_index); el4.push_back(C_index); 
    el5.push_back(E_index); el5.push_back(G_index); el5.push_back(C_index);
    elementVertices.push_back(el1);
    elementVertices.push_back(el2);
    elementVertices.push_back(el3);
    elementVertices.push_back(el4);
    elementVertices.push_back(el5);

    mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, stokesBFMath, H1Order, pToAdd) );
    streamMesh = Teuchos::rcp( new Mesh(vertices, elementVertices, streamBF, H1Order+2, pToAdd+2) );
    
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
  
  mesh->registerMesh(streamMesh); // will refine streamMesh in the same way as mesh.
  
  if (rank == 0) {
    if ( ! singularityAvoidingInitialMesh )
      cout << "Starting mesh has " << horizontalCells << " x " << verticalCells << " elements and ";
    else
      cout << "Using singularity-avoiding initial mesh: 5 elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    cout << "eps for top BC = " << eps << endl;
    
    if (useTriangles) {
      cout << "Using triangles.\n";
    }
    if (weightTestNormDerivativesByH) {
      cout << "Weighting test norm derivatives by h.\n";
    }
    if (induceCornerRefinements) {
      cout << "Artificially inducing refinements in bottom corners.\n";
    }
    if (enforceLocalConservation) {
      cout << "Enforcing local conservation.\n";
    }
  }
  
  Teuchos::RCP<DPGInnerProduct> ip;
  
  IPPtr qoptIP = Teuchos::rcp(new IP());
  
  double beta = 1.0;
  FunctionPtr h = Teuchos::rcp( new hFunction() );
  
  if (weightTestNormDerivativesByH) {
    qoptIP->addTerm( mu * v1->dx() + q1->x() ); // sigma11
    qoptIP->addTerm( mu * v1->dy() + q1->y() ); // sigma12
    qoptIP->addTerm( mu * v2->dx() + q2->x() ); // sigma21
    qoptIP->addTerm( mu * v2->dy() + q2->y() ); // sigma22
    qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
    qoptIP->addTerm( h * q1->div() - v3->dx() );    // u1
    qoptIP->addTerm( h * q2->div() - v3->dy() );    // u2
  } else {
    qoptIP->addTerm( mu * v1->dx() + q1->x() ); // sigma11
    qoptIP->addTerm( mu * v1->dy() + q1->y() ); // sigma12
    qoptIP->addTerm( mu * v2->dx() + q2->x() ); // sigma21
    qoptIP->addTerm( mu * v2->dy() + q2->y() ); // sigma22
    qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
    qoptIP->addTerm( q1->div() - v3->dx() );    // u1
    qoptIP->addTerm( q2->div() - v3->dy() );    // u2
  }
  qoptIP->addTerm( sqrt(beta) * v1 );
  qoptIP->addTerm( sqrt(beta) * v2 );
  qoptIP->addTerm( sqrt(beta) * v3 );
  qoptIP->addTerm( sqrt(beta) * q1 );
  qoptIP->addTerm( sqrt(beta) * q2 );
  
  ip = qoptIP;
  
  if (rank==0) 
    ip->printInteractions();
  
  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr entireBoundary = Teuchos::rcp( new UnitSquareBoundary );
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0(eps) );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  FunctionPtr un_0 = Teuchos::rcp( new Un_0(eps) );
  FunctionPtr u0_cross_n = Teuchos::rcp( new U0_cross_n(eps) );

  bc->addDirichlet(u1hat, entireBoundary, u1_0);
  bc->addDirichlet(u2hat, entireBoundary, u2_0);
  bc->addZeroMeanConstraint(p);
  
  ////////////////////   CREATE RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy ); // zero for now...
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dy() - u2->dx() ) );
//  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution,sigma12 - sigma21) );
  Teuchos::RCP<RHSEasy> streamRHS = Teuchos::rcp( new RHSEasy );
  streamRHS->addTerm(vorticity * q_s);
  
  Teuchos::RCP<BCEasy> streamBC = Teuchos::rcp( new BCEasy );
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0) );
  streamBC->addDirichlet(psin_hat, entireBoundary, u0_cross_n);
  streamBC->addZeroMeanConstraint(phi);
  
  IPPtr streamIP = Teuchos::rcp( new IP );
  streamIP->addTerm(q_s);
  streamIP->addTerm(q_s->grad());
  streamIP->addTerm(v_s);
  streamIP->addTerm(v_s->div());
  SolutionPtr streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );
  
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  double energyThreshold = 0.20; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  // just an experiment:
  //  refinementStrategy.setEnforceOneIrregurity(false);
  
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
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->solve(false);
    refinementStrategy.refine(rank==0); // print to console on rank 0
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
      vector<int> cornerIDs;
      cornerIDs.push_back(corners[0]->cellID());
      cornerIDs.push_back(corners[1]->cellID());
      mesh->hRefine(cornerIDs, RefinementPattern::regularRefinementPatternQuad());
    }
  }
  // one more solve on the final refined mesh:
  solution->solve(false);
  double energyErrorTotal = solution->energyErrorTotal();
  if (rank == 0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
  }
  
  FunctionPtr u1_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u1) );
  FunctionPtr u2_prev = Teuchos::rcp( new PreviousSolutionFunction(solution,u2) );
  FunctionPtr u1_sq = u1_prev * u1_prev;
  FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
  FunctionPtr u_mag = Teuchos::rcp( new SqrtFunction( u_dot_u ) );
  FunctionPtr u_div = Teuchos::rcp( new PreviousSolutionFunction(solution, u1->dx() + u2->dy() ) );
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
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType) );
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

  streamSolution->solve(false);
  energyErrorTotal = streamSolution->energyErrorTotal();
  if (rank == 0) {  
    cout << "...solved.\n";
    cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
  }

  
  if (rank==0){
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
      
      cout << "wrote files: u_mag.m, u_div.m, u1.m, u1_hat.dat, u2.m, u2_hat.dat, p.m, phi.m, vorticity.m.\n";
    } else {
      solution->writeToFile(u1->ID(), "u1.dat");
      solution->writeToFile(u2->ID(), "u2.dat");
      solution->writeToFile(u2->ID(), "p.dat");
      cout << "wrote files: u1.dat, u2.dat, p.dat\n";
    }
    writeStreamlines(0, 1, 0, 1, solution, u1, u2, "u_streamlines.m");
    // corner detail
    writeStreamlines(0, .1, 0, .1, solution, u1, u2, "u_detail_streamlines.m");
    writeStreamlines(0, .01, 0, .01, solution, u1, u2, "u_minute_detail_streamlines.m");
    writeStreamlines(0, .001, 0, .001, solution, u1, u2, "u_minute_minute_detail_streamlines.m");
    writePatchValues(0, 1, 0, 1, streamSolution, phi, "phi_patch.m");
    writePatchValues(0, .1, 0, .1, streamSolution, phi, "phi_patch_detail.m");
    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
  }
  return 0;
}