//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double pi = 2.0*acos(0.0);

int H1Order = 3, pToAdd = 2;

class ConstantXBoundary : public SpatialFilter {
   private:
      double xval;
   public:
      ConstantXBoundary(double xval): xval(xval) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(x-xval) < tol);
      }
};

class ConstantYBoundary : public SpatialFilter {
   private:
      double yval;
   public:
      ConstantYBoundary(double yval): yval(yval) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(y-yval) < tol);
      }
};

class ExactU : public Function {
  public:
    ExactU() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex, ptIndex) = 1-2*x;
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

  // Optional arguments (have defaults)
  int numRefs = args.Input("--numRefs", "number of refinement steps", 0);
  double nu = args.Input("--nu", "viscosity", 1e-2);
  int numX = args.Input("--numX", "number of cells in x direction", 1);
  int maxNewtonIterations = args.Input("--maxIterations", "maximum number of Newton iterations", 10);
  int numY = numX;
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("tau", HGRAD);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma");
  VarPtr fhat = varFactory.fluxVar("fhat");
  VarPtr uhat = varFactory.spatialTraceVar("uhat");

  ////////////////////   INITIALIZE USEFUL VARIABLES   ///////////////////////
  // Define useful functions
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();
  FunctionPtr uExact = Teuchos::rcp( new ExactU );

  // Initialize useful variables
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  ////////////////////   BUILD MESH   ///////////////////////
  FieldContainer<double> meshBoundary(4,2);
  double xmin = 0.0;
  double xmax = 1.0;
  double ymin = 0.0;
  double ymax = 1.0;

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  ymin; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  ymin;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  ymax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  ymax;

  MeshPtr mesh = Mesh::buildQuadMesh(meshBoundary, numX, numY,
      bf, H1Order, H1Order+pToAdd, false);

  ////////////////////   SET INITIAL CONDITIONS   ///////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  map<int, Teuchos::RCP<Function> > initialConditions;
  initialConditions[u->ID()]     = uExact;
  initialConditions[sigma->ID()] = zero;

  backgroundFlow->projectOntoMesh(initialConditions);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  // Set up problem

  FunctionPtr u_prev     = Function::solution(u, backgroundFlow);
  FunctionPtr sigma_prev = Function::solution(sigma, backgroundFlow);

  // tau terms:
  bf->addTerm( sigma/nu, tau);
  bf->addTerm( u, tau->dx());
  bf->addTerm( -uhat, tau->times_normal_x());

  // v terms:
  bf->addTerm( -u_prev*u, v->dx());
  bf->addTerm( sigma, v->dx());
  bf->addTerm( -u, v->dy());
  bf->addTerm( fhat, v);

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  rhs->addTerm( 0.5*u_prev*u_prev * v->dx() );
  rhs->addTerm( -sigma_prev * v->dx() );
  rhs->addTerm( u_prev * v->dy() );

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // Graph norm
  IPPtr ip = bf->graphNorm();

  ////////////////////   CREATE BCs   ///////////////////////
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(ymin) );
  bc->addDirichlet(fhat, bottom, -uExact);
  bc->addDirichlet(fhat, left,   -0.5*one);
  bc->addDirichlet(fhat, right,  0.5*one);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  mesh->registerSolution(backgroundFlow);
  mesh->registerSolution(solution);
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(backgroundFlow, mesh, varFactory);

  double nonlinearRelativeEnergyTolerance = 1e-5; // used to determine convergence of the nonlinear solution
  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    double L2Update = 1e7;
    int iterCount = 0;
    while (L2Update > nonlinearRelativeEnergyTolerance && iterCount < maxNewtonIterations)
    {
      solution->solve(false);
      double uL2Update = solution->L2NormOfSolutionGlobal(u->ID());
      L2Update = uL2Update;

      backgroundFlow->addSolution(solution, 1.0);
      iterCount++;
      if (commRank == 0)
        cout << "L2 Norm of Update = " << L2Update << endl;
    }

     if (commRank == 0)
     {
        stringstream outfile;
        outfile << "spacetimeburgers_" << refIndex;
        exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

