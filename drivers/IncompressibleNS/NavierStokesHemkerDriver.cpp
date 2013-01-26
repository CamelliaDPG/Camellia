//
//  NavierStokesCavityFlowDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/14/12.
//  Copyright (c) 2012 __MyCompanyName__. All rights reserved.
//

#include "HConvergenceStudy.h"

#include "InnerProductScratchPad.h"

#include "PreviousSolutionFunction.h"

#include "LagrangeConstraints.h"

#include "BasisFactory.h"

#include "GnuPlotUtil.h"

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

#include "MeshFactory.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

using namespace std;

static double Re = 5;

VarFactory varFactory; 
// test variables:
VarPtr tau1, tau2, v1, v2, q;
// traces and fluxes:
VarPtr u1hat, u2hat, t1n, t2n;
// field variables:
VarPtr u1, u2, sigma11, sigma12, sigma21, sigma22, p;

class LeftBoundary : public SpatialFilter {
  double _left;
public:
  LeftBoundary(double width) {
    double tol = 1e-14;
    _left = - width / 2.0 + tol;
  }
  bool matchesPoint(double x, double y) {
    bool matches = x < _left;
//    cout << "Left boundary ";
//    if (matches) {
//      cout << "matches";
//    } else {
//      cout << "does not match";
//    }
//    cout << " point (" << x << ", " << y << ")\n";
    return matches;
  }
};

class RightBoundary : public SpatialFilter {
  double _right;
public:
  RightBoundary(double width) {
    double tol = 1e-14;
    _right = width / 2.0 - tol;
  }
  bool matchesPoint(double x, double y) {
    return x > _right;
  }
};


class TopBoundary : public SpatialFilter {
  double _top;
public:
  TopBoundary(double height) {
    double tol = 1e-14;
    _top = height / 2.0 - tol;
  }
  bool matchesPoint(double x, double y) {
    return y > _top;
  }
};

class BottomBoundary : public SpatialFilter {
  double _bottom;
public:
  BottomBoundary(double height) {
    double tol = 1e-14;
    _bottom = - height / 2.0 + tol;
  }
  bool matchesPoint(double x, double y) {
    return y < _bottom;
  }
};

class NearCylinder : public SpatialFilter {
  double _enlarged_radius;
public:
  NearCylinder(double radius) {
    double enlargement_factor = 1.2;
    _enlarged_radius = radius * enlargement_factor;
  }
  bool matchesPoint(double x, double y) {
    if (x*x + y*y < _enlarged_radius * _enlarged_radius) {
      return true;
    } else {
      return false;
    }
  }
};

class BoundaryVelocity : public SimpleFunction {
  double _left, _right, _top, _bottom, _radius;
  int _comp;
public:
  BoundaryVelocity(double width, double height, double radius, int component) {
    double tol = 1e-14;
    _left = - width / 2.0 + tol;
    _right = width / 2.0 - tol;
    _top = height / 2.0 - tol;
    _bottom = - height / 2.0 + tol;
    _radius = radius;
    _comp = component; // 0 for x, 1 for y
  }
  
  double value(double x, double y) {
    // widen the radius to allow for some geometry error
    double enlarged_radius = _radius * 1.2;
    if (x*x + y*y < enlarged_radius * enlarged_radius) {
      // then we're on the cylinder
      return 0.0;
    }
    if (_comp == 0) { // u0 == 1 everywhere that we set u0
      return 1.0;
    } else {
      return 0.0;
    }
  }
};

int main(int argc, char *argv[]) {
  int rank = 0;
#ifdef HAVE_MPI
  // TODO: figure out the right thing to do here...
  // may want to modify argc and argv before we make the following call:
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  rank=mpiSession.getRank();
#else
#endif
  bool useLineSearch = false;
  
  int pToAdd = 2; // for optimal test function approximation
  double nonlinearStepSize = 1.0;
  double dt = 0.5;
  double nonlinearRelativeEnergyTolerance = 0.015; // used to determine convergence of the nonlinear solution
  bool enforceOneIrregularity = true;
  bool reportPerCellErrors  = true;

  bool startWithZeroSolutionAfterRefinement = true;
  
  bool artificialTimeStepping = false;
  
  int maxIters = 50; // for nonlinear steps
  
  // usage: polyOrder [numRefinements]
  // parse args:
  if ((argc != 4) && (argc != 3) && (argc != 2) && (argc != 5)) {
    cout << "Usage: NavierStokesHemkerDriver fieldPolyOrder [numRefinements=10 [Reyn=5]]\n";
    return -1;
  }
  int polyOrder = atoi(argv[1]);
  int numRefs = 10;
  if ( argc == 3) {
    numRefs = atoi(argv[2]);
  }
  if ( argc == 4) {
    numRefs = atoi(argv[2]);
    Re = atof(argv[3]);
  }
  if (rank == 0) {
    cout << "numRefinements = " << numRefs << endl;
    cout << "Re = " << Re << endl;
    if (artificialTimeStepping) cout << "dt = " << dt << endl;
    if (!startWithZeroSolutionAfterRefinement) {
      cout << "NOTE: experimentally, NOT starting with 0 solution after refinement...\n";
    }
  }

  // define meshes:
  int H1Order = polyOrder + 1;
  double minL2Increment = 1e-8;

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
  
  double width = 10, height = 10;
  double radius = 1;
  FunctionPtr zero = Function::zero();
  FunctionPtr inflowSpeed = Function::constant(1.0);
  
  MeshGeometryPtr geometry = MeshFactory::hemkerGeometry(width,height,radius);
//  MeshGeometryPtr geometry = MeshFactory::quadGeometry(width,height); // hemker domain, but without the cylinder, and without curvilinear geometry!
  
  VGPNavierStokesProblem problem = VGPNavierStokesProblem(Re,geometry,
                                                          H1Order, pToAdd); // zero forcing function
  SolutionPtr solution = problem.backgroundFlow();
  SolutionPtr solnIncrement = problem.solutionIncrement();
  
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr nearCylinder = Teuchos::rcp( new NearCylinder(radius) );
  SpatialFilterPtr top          = Teuchos::rcp( new TopBoundary(height) );
  SpatialFilterPtr bottom       = Teuchos::rcp( new BottomBoundary(height) );
  SpatialFilterPtr left         = Teuchos::rcp( new LeftBoundary(width) );
  SpatialFilterPtr right        = Teuchos::rcp( new RightBoundary(width) );
  // could simplify the below using SpatialFilter's Or-ing capability
  bc->addDirichlet(u1hat,nearCylinder,zero);
  bc->addDirichlet(u2hat,nearCylinder,zero);
  bc->addDirichlet(u1hat,left,inflowSpeed);
  bc->addDirichlet(u2hat,left,zero);
  bc->addDirichlet(u1hat,top,inflowSpeed);
  bc->addDirichlet(u2hat,top,zero);
  bc->addDirichlet(u1hat,bottom,inflowSpeed);
  bc->addDirichlet(u2hat,bottom,zero);
  // finally, no-traction conditions
  bc->addDirichlet(t1n,right,zero);
  bc->addDirichlet(t2n,right,zero);
  
  bc->addZeroMeanConstraint(p);
  problem.setBC(bc);
  
  Teuchos::RCP<Mesh> mesh = problem.mesh();
  mesh->registerSolution(solution);
  mesh->registerSolution(solnIncrement);
  
  if (rank == 0) {
    cout << "Starting mesh has " << problem.mesh()->numElements() << " elements and ";
    cout << mesh->numGlobalDofs() << " total dofs.\n";
    cout << "polyOrder = " << polyOrder << endl; 
    cout << "pToAdd = " << pToAdd << endl;
    
    if (enforceOneIrregularity) {
      cout << "Enforcing 1-irregularity.\n";
    } else {
      cout << "NOT enforcing 1-irregularity.\n";
    }
  }
  
  map< int, FunctionPtr > initialGuess;
  initialGuess[u1->ID()] = Function::constant(1.0);
  initialGuess[u1hat->ID()] = Function::constant(1.0);
  // all other variables: use zero initial guess (the implicit one)
  
  ////////////////////   CREATE BCs   ///////////////////////
  FunctionPtr u1_prev = Function::solution(u1,solution);
  FunctionPtr u2_prev = Function::solution(u2,solution);
  
  FunctionPtr u1hat_prev = Function::solution(u1hat,solution);
  FunctionPtr u2hat_prev = Function::solution(u2hat,solution);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  
  FunctionPtr vorticity = Teuchos::rcp( new PreviousSolutionFunction(solution, - u1->dy() + u2->dx() ) );

  FunctionPtr polyOrderFunction = Teuchos::rcp( new MeshPolyOrderFunction(mesh) );
  
  double energyThreshold = 0.20; // for mesh refinements
  Teuchos::RCP<RefinementStrategy> refinementStrategy;
  if (rank==0) cout << "NOTE: using solution, not solnIncrement, for refinement strategy.\n";
  refinementStrategy = Teuchos::rcp( new RefinementStrategy( solution, energyThreshold ));
//  refinementStrategy = Teuchos::rcp( new RefinementStrategy( solnIncrement, energyThreshold ));
  
  refinementStrategy->setEnforceOneIrregularity(enforceOneIrregularity);
  refinementStrategy->setReportPerCellErrors(reportPerCellErrors);

  Teuchos::RCP<NonlinearStepSize> stepSize = Teuchos::rcp(new NonlinearStepSize(nonlinearStepSize));
  Teuchos::RCP<NonlinearSolveStrategy> solveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
                                                                                               stepSize,
                                                                                               nonlinearRelativeEnergyTolerance));
  
  Teuchos::RCP<NonlinearSolveStrategy> finalSolveStrategy = Teuchos::rcp(new NonlinearSolveStrategy(solution, solnIncrement, 
                                                                                               stepSize,
                                                                                               nonlinearRelativeEnergyTolerance / 10));
  
  if (true) { // do regular refinement strategy...
    bool printToConsole = rank==0;
    FunctionPtr u1_incr = Function::solution(u1, solnIncrement);
    FunctionPtr u2_incr = Function::solution(u2, solnIncrement);
    FunctionPtr sigma11_incr = Function::solution(sigma11, solnIncrement);
    FunctionPtr sigma12_incr = Function::solution(sigma12, solnIncrement);
    FunctionPtr sigma21_incr = Function::solution(sigma21, solnIncrement);
    FunctionPtr sigma22_incr = Function::solution(sigma22, solnIncrement);
    FunctionPtr p_incr = Function::solution(p, solnIncrement);
    
    FunctionPtr l2_incr = u1_incr * u1_incr + u2_incr * u2_incr + p_incr * p_incr
    + sigma11_incr * sigma11_incr + sigma12_incr * sigma12_incr
    + sigma21_incr * sigma21_incr + sigma22_incr * sigma22_incr;

    for (int refIndex=0; refIndex<numRefs; refIndex++){
      if (startWithZeroSolutionAfterRefinement) {
        // start with a fresh initial guess for each adaptive mesh:
        solution->clear();
        solution->projectOntoMesh(initialGuess);
        problem.setIterationCount(0); // must be zero to force solve with background flow again (instead of solnIncrement)
      }
      
      double incr_norm;
      do {
        problem.iterate(useLineSearch);
        incr_norm = sqrt(l2_incr->integrate(problem.mesh()));
        if (rank==0) {
          cout << "\x1B[2K"; // Erase the entire current line.
          cout << "\x1B[0E"; // Move to the beginning of the current line.
          cout << "Refinement # " << refIndex << ", iteration: " << problem.iterationCount() << "; L^2(incr) = " << incr_norm;
          flush(cout);
        }
      } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));

      if (rank==0)
        cout << "\nFor refinement " << refIndex << ", num iterations: " << problem.iterationCount() << endl;
      
      // reset iteration count to 1 (for the background flow):
      problem.setIterationCount(1);
      
      refinementStrategy->refine(rank==0); // print to console on rank 0
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
    } while ((incr_norm > minL2Increment ) && (problem.iterationCount() < maxIters));
    if (rank==0) cout << endl;
  }
  
  double energyErrorTotal = solution->energyErrorTotal();
  double incrementalEnergyErrorTotal = solnIncrement->energyErrorTotal();
  if (rank == 0) {
    cout << "Final mesh has " << mesh->numActiveElements() << " elements and " << mesh->numGlobalDofs() << " dofs.\n";
    cout << "Final energy error: " << energyErrorTotal << endl;
    cout << "  (Incremental solution's energy error is " << incrementalEnergyErrorTotal << ".)\n";
  }
  
  FunctionPtr u1_sq = u1_prev * u1_prev;
  FunctionPtr u_dot_u = u1_sq + (u2_prev * u2_prev);
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
    for (int i=0; i<elems.size(); i++) {
      int cellID = cellIDs[i];
      // pick out the ones for testOne:
      massFluxIntegral[cellID] = fakeRHSIntegrals(i,testOneIndex);
    }
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
    GnuPlotUtil::writeComputationalMeshSkeleton("finalHemkerMesh", mesh);
    
    solution->writeToVTK("nsHemkerSoln.vtk.vtu");
    solution->writeTracesToVTK("nsHemkerSoln.vtk.vtu");
    massFlux->writeBoundaryValuesToMATLABFile(solution->mesh(), "massFlux.dat");
    u_div->writeValuesToMATLABFile(solution->mesh(), "u_div.m");
    solution->writeFieldsToFile(u1->ID(), "u1.m");
    solution->writeFluxesToFile(u1hat->ID(), "u1_hat.dat");
    solution->writeFieldsToFile(u2->ID(), "u2.m");
    solution->writeFluxesToFile(u2hat->ID(), "u2_hat.dat");
    solution->writeFieldsToFile(p->ID(), "p.m");
    
    FunctionPtr ten = Teuchos::rcp( new ConstantScalarFunction(10) );
    ten->writeBoundaryValuesToMATLABFile(solution->mesh(), "skeleton.dat");
    cout << "wrote files: u_mag.m, u_div.m, u1.m, u1_hat.dat, u2.m, u2_hat.dat, p.m, phi.m, vorticity.m.\n";
    polyOrderFunction->writeValuesToMATLABFile(mesh, "cavityFlowPolyOrders.m");
  }
  
  return 0;
}
