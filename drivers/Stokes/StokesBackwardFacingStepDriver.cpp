//
//  StokesBackwardFacingStep.cpp
//  Camellia-debug
//
//  Created by Nathan Roberts on 4/27/12.
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

static double tol=1e-14;

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
  return abs(x-8.0) < tol;
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
  int pToAdd = 1; // for optimal test function approximation
  int pToAddForStreamFunction = pToAdd;
  bool enforceLocalConservation = true;
  bool useSGP = false; // stream-gradient-pressure formulation (velocity-gradient-pressure is the alternative)
  //  if (useSGP) {
  //    enforceLocalConservation = false; // automatic for SGP
  //  }
  
  // usage: polyOrder [numRefinements]
  // parse args:
  if ((argc != 3) && (argc != 2)) {
    cout << "Usage: StokesBackwardFacingStepDriver fieldPolyOrder [numRefinements=10]\n";
    return -1;
  }
  int polyOrder = atoi(argv[1]);
  int numRefs = 10;
  if ( argc == 3) {
    numRefs = atoi(argv[2]);
  }
  if (rank == 0)
    cout << "numRefinements = " << numRefs << endl;
  
  /////////////////////////// "VGP_CONFORMING" VERSION ///////////////////////
  VarFactory varFactory; 
  VarPtr q1 = varFactory.testVar("q_1", HDIV);
  VarPtr q2 = varFactory.testVar("q_2", HDIV);
  VarPtr q3;
  VarPtr v1 = varFactory.testVar("v_1", HGRAD);
  VarPtr v2 = varFactory.testVar("v_2", HGRAD);
  VarPtr v3;
  if ( useSGP ) {
    v3 = varFactory.testVar("v_3", HGRAD);
  } else {
    v3 = varFactory.testVar("v_3", HGRAD);
  }
  //  VarPtr testOne; // used for local conservation, if requested
  //  if (enforceLocalConservation) {
  //    testOne = varFactory.testVar("1", CONSTANT_SCALAR);
  //  }
  
  // fluxes and traces:
  VarPtr u1hat, u2hat, t1n, t2n;
  // fields for SGP:
  VarPtr phi, p, sigma11, sigma12, sigma21, sigma22;
  // fields specific to VGP:
  VarPtr u1, u2;
  // trace specific to SGP:
  VarPtr phi_hat; // used to weakly enforce continuity of phi
  
  u1hat = varFactory.traceVar("\\widehat{u}_1");
  u2hat = varFactory.traceVar("\\widehat{u}_2");
  
  if (useSGP) {
    phi_hat = varFactory.traceVar("\\widehat{\\phi}");
  }
  
  t1n = varFactory.fluxVar("\\widehat{t_{1n}}");
  t2n = varFactory.fluxVar("\\widehat{t_{2n}}");
  if (useSGP) {
    phi = varFactory.fieldVar("\\phi", HGRAD);
  } else {
    u1 = varFactory.fieldVar("u_1");
    u2 = varFactory.fieldVar("u_2");
  }
  sigma11 = varFactory.fieldVar("\\sigma_11");
  sigma12 = varFactory.fieldVar("\\sigma_12");
  sigma21 = varFactory.fieldVar("\\sigma_21");
  sigma22 = varFactory.fieldVar("\\sigma_22");
  p = varFactory.fieldVar("p");
  
  double mu = 1;
  
  BFPtr stokesBF = Teuchos::rcp( new BF(varFactory) );  
  // q1 terms:
  if (useSGP) {
    stokesBF->addTerm( phi->dy(), q1->div() );
  } else {
    stokesBF->addTerm(u1,q1->div());
  }
  stokesBF->addTerm(sigma11,q1->x()); // (sigma1, q1)
  stokesBF->addTerm(sigma12,q1->y());
  stokesBF->addTerm(-u1hat, q1->dot_normal());
  
  // q2 terms:
  if (useSGP) {
    stokesBF->addTerm( -phi->dx(), q2->div());
  } else {
    stokesBF->addTerm(u2, q2->div());
  }
  stokesBF->addTerm(sigma21,q2->x()); // (sigma2, q2)
  stokesBF->addTerm(sigma22,q2->y());
  stokesBF->addTerm(-u2hat, q2->dot_normal());
  
  if (useSGP) { // equation to enforce continuity of phi weakly -- not certain this will work!
    stokesBF->addTerm(phi->grad(), q1);
    stokesBF->addTerm(phi, q1->div());
    stokesBF->addTerm(-phi_hat, q1->dot_normal());
    stokesBF->addTerm(phi->grad(), q2);
    stokesBF->addTerm(phi, q2->div());
    stokesBF->addTerm(-phi_hat, q2->dot_normal());
    // q3 terms
    //    stokesBF->addTerm(phi->grad(), q3);
    //    stokesBF->addTerm(phi, q3->div());
    //    stokesBF->addTerm(-phi_hat, q3->dot_normal());
    //    stokesBF->addTerm(phi->curl(), q3);
    //    stokesBF->addTerm(-phi, q3->curl());
    //    stokesBF->addTerm(phi_hat, q3->cross_normal());
  }
  
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
  
  if ( ! useSGP ) {
    // v3:
    stokesBF->addTerm(-u1,v3->dx()); // (-u, grad v3)
    stokesBF->addTerm(-u2,v3->dy());
    stokesBF->addTerm(u1hat->times_normal_x() + u2hat->times_normal_y(), v3);
  }
  
  if (rank==0)
    stokesBF->printTrialTestInteractions();
  
  ///////////////////////////////////////////////////////////////////////////
  SpatialFilterPtr nonOutflowBoundary = Teuchos::rcp( new NonOutflowBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowBoundary );
  
  int H1Order = polyOrder + 1;
  Teuchos::RCP<Mesh> mesh, streamMesh;
  
  FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
  A(0) = 0.0; A(1) = 1.0;
  B(0) = 0.0; B(1) = 2.0;
  C(0) = 4.0; C(1) = 2.0;
  D(0) = 8.0; D(1) = 2.0;
  E(0) = 8.0; E(1) = 1.0;
  F(0) = 8.0; F(1) = 0.0;
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
  
  Teuchos::RCP<DPGInnerProduct> ip;
  
  IPPtr qoptIP = Teuchos::rcp(new IP());
  
  double beta = 1.0;
  FunctionPtr h = Teuchos::rcp( new hFunction() );
  
  if (useSGP) { // then graph norm equivalent to naive norm
    qoptIP->addTerm(v1->grad());
    qoptIP->addTerm(v2->grad());
    qoptIP->addTerm(v3->grad());
    qoptIP->addTerm(q1->div());
    qoptIP->addTerm(q2->div());
    //    qoptIP->addTerm(q3->div());
  } else {
    qoptIP->addTerm( mu * v1->grad() + q1 ); // sigma11, sigma12
    qoptIP->addTerm( mu * v2->grad() + q2 ); // sigma21, sigma22
    qoptIP->addTerm( v1->dx() + v2->dy() );     // pressure
    qoptIP->addTerm( q1->div() - v3->dx() );    // u1
    qoptIP->addTerm( q2->div() - v3->dy() );    // u2
  }
  qoptIP->addTerm( sqrt(beta) * v1 );
  qoptIP->addTerm( sqrt(beta) * v2 );
  if ( ! useSGP ) {
    qoptIP->addTerm( sqrt(beta) * v3 );
  }
  qoptIP->addTerm( sqrt(beta) * q1 );
  qoptIP->addTerm( sqrt(beta) * q2 );
  
  ip = qoptIP;
  
  if (rank==0) 
    ip->printInteractions();
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  
  BFPtr streamBF;
  SolutionPtr streamSolution;
  // define bilinear form for stream function:
  if (! useSGP) {
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
    
    FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );
    Teuchos::RCP<RHSEasy> streamRHS = Teuchos::rcp( new RHSEasy );
    streamRHS->addTerm(vorticity * q_s);
    ((PreviousSolutionFunction*) vorticity.get())->setOverrideMeshCheck(true);
    
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
    
    streamMesh = Teuchos::rcp( new Mesh(vertices, elementVertices, streamBF, H1Order, pToAdd+pToAddForStreamFunction) );
    
    streamSolution = Teuchos::rcp( new Solution( streamMesh, streamBC, streamRHS, streamIP ) );
  }
  
  if (!useSGP) {
    mesh->registerMesh(streamMesh); // will refine streamMesh in the same way as mesh.
  }
  
  ////////////////////   CREATE BCs   ///////////////////////
  FunctionPtr u1_0 = Teuchos::rcp( new U1_0 );
  FunctionPtr u2_0 = Teuchos::rcp( new U2_0 );
  bc->addDirichlet(u1hat, nonOutflowBoundary, u1_0);
  bc->addDirichlet(u2hat, nonOutflowBoundary, u2_0);
  // for now, prescribe same thing on the outflow boundary
  // TODO: replace with zero-traction condition
  // THIS IS A GUESS AT THE ZERO-TRACTION CONDITION
  cout << "SHOULD CONFIRM THAT THE ZERO-TRACTION CONDITION IS RIGHT!!\n";
  FunctionPtr zero = Teuchos::rcp(new ConstantScalarFunction(0.0));
  //  bc->addDirichlet(sigma1n, outflowBoundary, zero);
  //  bc->addDirichlet(sigma2n, outflowBoundary, zero);
  bc->addZeroMeanConstraint(p);
  FunctionPtr phi0 = Teuchos::rcp( new PHI_0 );
  
  if (useSGP) {
    bc->addZeroMeanConstraint(phi);
    //    bc->addDirichlet(phi_hat, nonOutflowBoundary, phi0); // not sure this is right -- phi0 should probably be imposed on outflow, too...
  }
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  if (enforceLocalConservation) {
    FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
    solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y()==zero);
  }
  
  double energyThreshold = 0.20; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  // just an experiment:
  //  refinementStrategy.setEnforceOneIrregurity(false);
  
  int uniformRefinements = 0;
  for (int refIndex=0; refIndex<uniformRefinements; refIndex++){
    refinementStrategy.refine();
  }
  
  if (rank == 0) {
    cout << "Starting mesh has " << mesh->numActiveElements() << " elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    
    if (useSGP) {
      cout << "Using stream-gradient-pressure formulation.\n";
    }
    if (enforceLocalConservation) {
      cout << "Enforcing local conservation.\n";
    }
  }
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->solve(false);
    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  // one more solve on the final refined mesh:
  solution->solve(false);
  double energyErrorTotal = solution->energyErrorTotal();
  if (rank == 0) 
    cout << "Final energy error: " << energyErrorTotal << endl;
  
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
  
  ///////// SET UP & SOLVE STREAM SOLUTION /////////
  if (! useSGP ) {
    if (rank == 0) {
      cout << "streamMesh has " << streamMesh->numActiveElements() << " elements.\n";
      cout << "solving for approximate stream function...\n";
    }
    
    streamSolution->solve(false);
    energyErrorTotal = streamSolution->energyErrorTotal();
    if (rank == 0) {  
      cout << "...solved.\n";
      cout << "Stream mesh has energy error: " << energyErrorTotal << endl;
    }
  }
  
  if (rank==0){
    massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
    if (useSGP) {
      solution->writeFieldsToFile(phi->ID(), "phi.m");
      
      FunctionPtr u1_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, phi->dy()) );
      FunctionPtr u2_soln = Teuchos::rcp( new PreviousSolutionFunction(solution, -phi->dx()) );
      
      u1_soln->writeValuesToMATLABFile(solution->mesh(), "u1.m");
      u2_soln->writeValuesToMATLABFile(solution->mesh(), "u2.m");
    } else {
      solution->writeFieldsToFile(u1->ID(), "u1.m");
      solution->writeFieldsToFile(u2->ID(), "u2.m");
      streamSolution->writeFieldsToFile(phi->ID(), "phi.m");
    }
    solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
    solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
    solution->writeFieldsToFile(p->ID(), "p.m");
    if (useSGP) {
      writePatchValues(0, 8, 0, 2, solution, phi, "phi_patch.m");
      writePatchValues(4, 8, 0, 2, solution, phi, "phi_patch_east.m");
    } else {
      writePatchValues(0, 8, 0, 2, streamSolution, phi, "phi_patch.m");
      writePatchValues(4, 8, 0, 2, streamSolution, phi, "phi_patch_east.m");
    }
    //    writePatchValues(0, .01, 0, .01, streamSolution, phi, "phi_patch_minute_detail.m");
    //    writePatchValues(0, .001, 0, .001, streamSolution, phi, "phi_patch_minute_minute_detail.m");
    
    cout << "wrote files: u_div.m, u1.m, u1_hat.dat, u2.m, u2_hat.dat, p.m, phi_patch.m\n";
    
  }
  return 0;
}