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
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double halfWidth;
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
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int commRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  // Required arguments
  int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
  bool enforceLocalConservation = args.Input<bool>("--conserve", "enforce local conservation");
  bool steady = args.Input<bool>("--steady", "run steady rather than transient");

  // Optional arguments (have defaults)
  double dt = args.Input("--dt", "time step", 0.25);
  int numTimeSteps = args.Input("--nt", "number of time steps", 20);
  halfWidth = args.Input("--halfWidth", "half width of inlet profile", 1.0);
  args.Process();

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
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
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
                                                bf, H1Order, H1Order+pToAdd);

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
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( beta_n_u_hat, v);

  if (!steady)
  {
    bf->addTerm( u, invDt*v );
    rhs->addTerm( u_prev_time * invDt * v );
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();
  // ip->addTerm(v);
  // ip->addTerm(beta*v->grad());

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
  VTKExporter exporter(solution, mesh, varFactory);
  
  for (int refIndex=0; refIndex<=numRefs; refIndex++){    
    if (steady)
    {
      solution->solve(false);

      if (commRank == 0)
      {
        stringstream outfile;
        outfile << "Convection_" << refIndex;
        exporter.exportSolution(outfile.str());

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
      // cout << L2_time_residual <<" "<< time_tol << timestepCount << numTimeSteps << endl;
      while((L2_time_residual > time_tol) && (timestepCount < numTimeSteps))
      {
        solution->solve(false);
        // Subtract solutions to get residual
        flowResidual->setSolution(solution);
        flowResidual->addSolution(prevTimeFlow, -1.0);       
        L2_time_residual = flowResidual->L2NormOfSolutionGlobal(u->ID());

        if (commRank == 0)
        {
          cout << endl << "Timestep: " << timestepCount << ", dt = " << dt << ", Time residual = " << L2_time_residual << endl;    	

          stringstream outfile;
          outfile << "TransientConvection_" << refIndex << "-" << timestepCount;
          exporter.exportSolution(outfile.str());

          // Check local conservation
          // FunctionPtr flux = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u_hat) );
          // FunctionPtr source = Teuchos::rcp( new PreviousSolutionFunction(flowResidual, u) );
          // source = invDt * source;
          // Teuchos::Tuple<double, 3> fluxImbalances = checkConservationTest(flux, source, varFactory, mesh);
          // cout << "Mass flux: Largest Local = " << fluxImbalances[0] 
          //   << ", Global = " << fluxImbalances[1] << ", Sum Abs = " << fluxImbalances[2] << endl;
        }

        prevTimeFlow->setSolution(solution); // reset previous time solution to current time sol
        timestepCount++;
      }
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }
  
  return 0;
}

