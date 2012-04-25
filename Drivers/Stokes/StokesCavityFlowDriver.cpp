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
        return 1.0;
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
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
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
  bool weightTestNormDerivativesByH = false;
//  double eps = 1.0/16.0; // width of ramp up to 1.0 for top BC;  eps == 0 ==> soln not in H1
  // epsilon above is chosen to match our initial 16x16 mesh, to avoid quadrature errors.
  double eps = 0.0; // John Evans's problem: not in H^1
  bool induceCornerRefinements = false;
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
  
  int H1Order = polyOrder + 1;
  int horizontalCells = 2, verticalCells = 2;
  bool useTriangles = false;
  Teuchos::RCP<Mesh> mesh;
  
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
  } else {
    FieldContainer<double> A(2), B(2), C(2), D(2), E(2), F(2), G(2), H(2);
    // outer square
    A(0) = 0.0; A(1) = 0.0;
    B(0) = 1.0; B(1) = 0.0;
    C(0) = 1.0; C(1) = 1.0;
    D(0) = 0.0; D(1) = 1.0;
    // inner square:
    E(0) = 0.25; E(1) = 0.25;
    F(0) = 0.75; F(1) = 0.25;
    G(0) = 0.75; G(1) = 0.75;
    H(0) = 0.25; H(1) = 0.75;
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
    // outside trapezoidal elements:
    el1.push_back(A_index); el1.push_back(B_index); el1.push_back(F_index); el1.push_back(E_index);
    el2.push_back(B_index); el2.push_back(C_index); el2.push_back(G_index); el2.push_back(F_index);
    el3.push_back(C_index); el3.push_back(D_index); el3.push_back(H_index); el3.push_back(G_index);
    el4.push_back(D_index); el4.push_back(A_index); el4.push_back(E_index); el4.push_back(H_index);
    // interior square element
    el5.push_back(E_index); el5.push_back(F_index); el5.push_back(G_index); el5.push_back(H_index);
    elementVertices.push_back(el1);
    elementVertices.push_back(el2);
    elementVertices.push_back(el3);
    elementVertices.push_back(el4);
    elementVertices.push_back(el5);
    mesh = Teuchos::rcp( new Mesh(vertices, elementVertices, stokesBFMath, H1Order, pToAdd) );
  }
  
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
  bc->addDirichlet(u1hat, entireBoundary, u1_0);
  bc->addDirichlet(u2hat, entireBoundary, u2_0);
  bc->addZeroMeanConstraint(p);
  
  ////////////////////   CREATE RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy ); // zero for now...
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
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
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){    
    solution->solve(false);
    refinementStrategy.refine(rank==0); // print to console on rank 0
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
  if (rank == 0) 
    cout << "Final energy error: " << solution->energyErrorTotal() << endl;
  
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
  
  int numSides = 4;
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
  
  if (rank==0){
    if (! useTriangles ) {
      massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
      u_mag->writeValuesToMATLABFile(solution->mesh(), "u_mag.m");
      u_div->writeValuesToMATLABFile(solution->mesh(), "u_div.m");
      solution->writeFieldsToFile(u1->ID(), "u1.m");
      solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
      solution->writeFieldsToFile(u2->ID(), "u2.m");
      solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
      solution->writeFieldsToFile(p->ID(), "p.m");
      
      cout << "wrote files: u_mag.m, u_div.m, u1.m, u1_hat.dat, u2.m, u2_hat.dat, p.m\n";
    } else {
      solution->writeToFile(u1->ID(), "u1.dat");
      solution->writeToFile(u2->ID(), "u2.dat");
      solution->writeToFile(u2->ID(), "p.dat");
      cout << "wrote files: u1.dat, u2.dat, p.dat\n";
    }
  }
  return 0;
}