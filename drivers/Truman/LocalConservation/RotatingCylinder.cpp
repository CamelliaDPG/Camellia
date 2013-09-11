//  SimpleConvection.cpp
//  Driver for Conservative Convection
//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CheckConservation.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

bool enforceLocalConservation = false;
double dt = 1.0/32.;
int numTimeSteps = 200; // max time steps
double halfWidth = 1;
int H1Order = 3, pToAdd = 2;

class ScalarParamFunction : public Function {
  double _a;
  public:
  ScalarParamFunction(double a) : Function(0){
    _a = a;
  }
  void set_param(double a){
    _a = a;
  }
  double get_param(){
    return _a;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    CHECK_VALUES_RANK(values);
    values.initialize(_a);
  }
};

class InflowSquareBoundary : public SpatialFilter {
  FunctionPtr _beta;
  public:
  InflowSquareBoundary(FunctionPtr beta){
    _beta = beta;
  }
  // bool matchesPoint(double x, double y) {
  //   return true;
  // }
  bool matchesPoints(FieldContainer<bool> &pointsMatch, BasisCachePtr basisCache) {
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());    
    const FieldContainer<double> *normals = &(basisCache->getSideNormals());
    int numCells = (*points).dimension(0);
    int numPoints = (*points).dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints,2);
    _beta->values(beta_pts,basisCache);

    double tol=1e-14;
    bool somePointMatches = false;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double n1 = (*normals)(cellIndex,ptIndex,0);
        double n2 = (*normals)(cellIndex,ptIndex,1);
        double beta_n = beta_pts(cellIndex,ptIndex,0)*n1 + beta_pts(cellIndex,ptIndex,1)*n2 ;
        // cout << "beta = (" << beta_pts(cellIndex,ptIndex,0)<<", "<<
        //   beta_pts(cellIndex,ptIndex,1)<<"), n = ("<<n1<<", "<<n2<<")"<<endl;
        pointsMatch(cellIndex,ptIndex) = false;
        if (beta_n < 0){
          pointsMatch(cellIndex,ptIndex) = true;
          somePointMatches = true;
        }
      }
    }
    return somePointMatches;
  }
};

class Beta : public Function {
public:
  Beta() : Function(1) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = values.dimension(2);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d = 0; d < spaceDim; d++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex,ptIndex,0) = -y;
          values(cellIndex,ptIndex,1) = x;
        }
      }
    }
  }
};

class InitialCondition : public Function {
public:
  InitialCondition() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = values.dimension(2);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d = 0; d < spaceDim; d++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if ((x+.5)*(x+.5)+y*y < .25*.25)
            values(cellIndex,ptIndex) = 1.0;
          else
            values(cellIndex,ptIndex) = 0.0;
        }
      }
    }
  }
};

int main(int argc, char *argv[]) {
  // Process command line arguments
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr beta_n_u_hat = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");
  
  FunctionPtr beta = Teuchos::rcp(new Beta());
  
  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  
  meshBoundary(0,0) = -1.0; // x1
  meshBoundary(0,1) = -1.0; // y1
  meshBoundary(1,0) =  1.0;
  meshBoundary(1,1) = -1.0;
  meshBoundary(2,0) =  1.0;
  meshBoundary(2,1) =  1.0;
  meshBoundary(3,0) = -1.0;
  meshBoundary(3,1) =  1.0;

  int horizontalCells = 32, verticalCells = 32;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr prevTimeFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  
  SolutionPtr flowResidual = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );  

  FunctionPtr u_prev_time = Teuchos::rcp( new PreviousSolutionFunction(prevTimeFlow, u) );
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr invDt = Teuchos::rcp(new ScalarParamFunction(1.0/dt));    
  
  // v terms:
  confusionBF->addTerm( beta * u, - v->grad() );
  confusionBF->addTerm( beta_n_u_hat, v);

  confusionBF->addTerm( u, invDt*v );
  rhs->addTerm( u_prev_time * invDt * v );
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // robust test norm
  IPPtr ip = confusionBF->graphNorm();
  // IPPtr ip = Teuchos::rcp(new IP);
  // ip->addTerm(v);
  // ip->addTerm(invDt*v - beta*v->grad());

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary(beta) );
  FunctionPtr u0 = Teuchos::rcp( new ConstantScalarFunction(0) );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  bc->addDirichlet(beta_n_u_hat, inflowBoundary, beta*n*u0);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);
  mesh->registerSolution(prevTimeFlow);
  mesh->registerSolution(flowResidual);

  // ==================== SET INITIAL GUESS ==========================
  FunctionPtr u_init = Teuchos::rcp(new InitialCondition());
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()]      = u_init;

  prevTimeFlow->projectOntoMesh(functionMap);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  // if (enforceLocalConservation) {
  //   // FunctionPtr parity = Teuchos::rcp<Function>( new SideParityFunction );
  //   // LinearTermPtr conservedQuantity = Teuchos::rcp<LinearTerm>( new LinearTerm(parity, beta_n_u_minus_sigma_n) );
  //   LinearTermPtr conservedQuantity = Teuchos::rcp<LinearTerm>( new LinearTerm(1.0, beta_n_u_minus_sigma_n) );
  //   LinearTermPtr sourcePart = Teuchos::rcp<LinearTerm>( new LinearTerm(invDt, u) );
  //   conservedQuantity->addTerm(sourcePart, true);
  //   solution->lagrangeConstraints()->addConstraint(conservedQuantity == u_prev_time * invDt);
  // }
  
  int timestepCount = 0;
  double time_tol = 1e-8;
  double L2_time_residual = 1e9;
  while((L2_time_residual > time_tol) && (timestepCount < numTimeSteps))
  {
    solution->solve(false);
    // Subtract solutions to get residual
    flowResidual->setSolution(solution);
    flowResidual->addSolution(prevTimeFlow, -1.0);       
    L2_time_residual = flowResidual->L2NormOfSolutionGlobal(u->ID());

    if (rank == 0)
    {
      cout << endl << "Timestep: " << timestepCount << ", dt = " << dt << ", Time residual = " << L2_time_residual << endl;    	

      stringstream outfile;
      outfile << "rotatingCylinder_" << timestepCount;
      solution->writeToVTK(outfile.str(), 5);

      // Check local conservation
      FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
      FunctionPtr source = Teuchos::rcp( new PreviousSolutionFunction(flowResidual, u) );
      source = invDt * source;
      Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, source, varFactory, mesh);
      cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
        << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
    }

    prevTimeFlow->setSolution(solution); // reset previous time solution to current time sol
    timestepCount++;
  }
  
  return 0;
}

