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
bool steady = true;
double dt = 0.25;
int numTimeSteps = 20; // max time steps
double halfWidth = 1;
int numRefs = 0;
int H1Order = 2, pToAdd = 2;

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

class LeftBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      return (abs(x) < tol);
    }
};

// boundary value for sigma_n
class InletBC : public Function {
  public:
    InletBC() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (abs(y) <= halfWidth)
            // values(cellIndex, ptIndex) = -1.0*(1.0-y/halfWidth*y/halfWidth);
            values(cellIndex, ptIndex) = -1.0;
          else
            values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

int main(int argc, char *argv[]) {
  // Process command line arguments
  if (argc > 1)
    numRefs = atof(argv[1]);
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

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
  
  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  
  meshBoundary(0,0) =  0.0; // x1
  meshBoundary(0,1) = -2.0; // y1
  meshBoundary(1,0) =  4.0;
  meshBoundary(1,1) = -2.0;
  meshBoundary(2,0) =  4.0;
  meshBoundary(2,1) =  2.0;
  meshBoundary(3,0) =  0.0;
  meshBoundary(3,1) =  2.0;

  int horizontalCells = 8, verticalCells = 8;
  
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

  if (!steady)
  {
    confusionBF->addTerm( u, invDt*v );
    rhs->addTerm( u_prev_time * invDt * v );
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr lBoundary = Teuchos::rcp( new LeftBoundary );
  FunctionPtr u1 = Teuchos::rcp( new InletBC );
  bc->addDirichlet(beta_n_u_hat, lBoundary, u1);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);
  mesh->registerSolution(prevTimeFlow);
  mesh->registerSolution(flowResidual);

  // ==================== SET INITIAL GUESS ==========================
  double u_free = 0.0;
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()]      = Teuchos::rcp( new ConstantScalarFunction(u_free) );

  prevTimeFlow->projectOntoMesh(functionMap);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  // if (enforceLocalConservation) {
  //   if (steady)
  //   {
  //     FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  //     solution->lagrangeConstraints()->addConstraint(beta_n_u_minus_sigma_n == zero);
  //   }
  //   else
  //   {
  //     // FunctionPtr parity = Teuchos::rcp<Function>( new SideParityFunction );
  //     // LinearTermPtr conservedQuantity = Teuchos::rcp<LinearTerm>( new LinearTerm(parity, beta_n_u_minus_sigma_n) );
  //     LinearTermPtr conservedQuantity = Teuchos::rcp<LinearTerm>( new LinearTerm(1.0, beta_n_u_minus_sigma_n) );
  //     LinearTermPtr sourcePart = Teuchos::rcp<LinearTerm>( new LinearTerm(invDt, u) );
  //     conservedQuantity->addTerm(sourcePart, true);
  //     solution->lagrangeConstraints()->addConstraint(conservedQuantity == u_prev_time * invDt);
  //   }
  // }
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    if (steady)
    {
      solution->solve(false);

      if (rank == 0)
      {
        stringstream outfile;
        outfile << "Convection_" << refIndex;
        solution->writeToVTK(outfile.str());

        // Check local conservation
        FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
        FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
        Teuchos::Tuple<double, 3> fluxImbalances = checkConservation(flux, zero, varFactory, mesh);
        cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
          << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
      }
    }
    else
    {
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
        cout << endl << "Timestep: " << timestepCount << ", dt = " << dt << ", Time residual = " << L2_time_residual << endl;    	

        if (rank == 0)
        {
          stringstream outfile;
          outfile << "TransientConvection_" << refIndex << "-" << timestepCount;
          solution->writeToVTK(outfile.str());

          // Check local conservation
          FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
          FunctionPtr source = Teuchos::rcp( new PreviousSolutionFunction(flowResidual, u) );
          source = invDt * source;
          Teuchos::Tuple<double, 3> fluxImbalances = checkConservationTest(flux, source, varFactory, mesh);
          cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
            << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
        }

        prevTimeFlow->setSolution(solution); // reset previous time solution to current time sol
        timestepCount++;
      }
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(rank==0); // print to console on rank 0
    cout << endl;
  }
  
  return 0;
}

