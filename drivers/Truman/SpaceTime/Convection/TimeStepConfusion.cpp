//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"
#include "TimeIntegrator.h"
#include "PreviousSolutionFunction.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

int H1Order = 3, pToAdd = 2;

class EpsilonScaling : public hFunction {
  double _epsilon;
  public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

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

class InitialCondition : public Function {
  public:
    InitialCondition() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (abs(x) <= 0.25)
            // values(cellIndex, ptIndex) = -4*(abs(x)-0.25);
            values(cellIndex, ptIndex) = 1;
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

  // Optional arguments (have defaults)
  double epsilon = args.Input<double>("--epsilon", "diffusion parameter", 1e-2);
  int norm = args.Input<int>("--norm", "0 = graph\n    1 = coupled robust", 0);
  args.Process();

  cout << "Running with epsilon = " << epsilon << " and " << ((norm) ? "robust norm" : "graph norm") << endl;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);
  VarPtr uhat = varFactory.traceVar("uhat");
  VarPtr fhat = varFactory.fluxVar("fhat");

  // Initialize useful variables
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  IPPtr ip = Teuchos::rcp(new IP);
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  // Define useful functions
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();

  ////////////////////   BUILD MESH   ///////////////////////
  FieldContainer<double> meshBoundary(4,2);
  double xmin = -0.5;
  double xmax = 1.0;
  double ymin = 0.0;
  double ymax = 0.1;

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  ymin; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  ymin;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  ymax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  ymax;

  int horizontalCells = 64, verticalCells = 1;

  MeshPtr mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
      bf, H1Order, H1Order+pToAdd, false);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);

  // tau terms:
  bf->addTerm( sigma / epsilon, tau);
  bf->addTerm( u, tau->div());
  bf->addTerm( -uhat, tau->dot_normal());

  // v terms:
  bf->addTerm( sigma, v->grad() );
  bf->addTerm( beta * u, - v->grad() );
  bf->addTerm( fhat, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // Graph norm
  // if (norm == 0)
  // {
  //   ip = bf->graphNorm();
  // }
  // // Coupled robust norm
  // else if (norm == 1)
  // {
  //   // coupled robust test norm
  //   FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) );
  //   ip->addTerm( v );
  //   ip->addTerm( sqrt(epsilon) * v->grad() );
  //   ip->addTerm( beta * v->grad() );
  //   ip->addTerm( tau->div() - beta*v->grad() );
  //   ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
  // }

  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(ymin) );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(ymax) );
  FunctionPtr u0 = Teuchos::rcp( new InitialCondition );
  bc->addDirichlet(fhat, left, zero);
  bc->addDirichlet(fhat, bottom, zero);
  bc->addDirichlet(uhat, right, zero);
  bc->addDirichlet(fhat, top, zero);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  SolutionPtr solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );


  // ==================== SET INITIAL GUESS ==========================
  double u_free = 0.0;
  double sigma1_free = 0.0;
  double sigma2_free = 0.0;
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()]      = u0;
  functionMap[sigma->ID()] = Function::vectorize(zero, zero);

  // ImplicitEulerIntegrator timeIntegrator(bf, rhs, mesh, solution, functionMap);
  ESDIRK4Integrator timeIntegrator(bf, rhs, mesh, solution, functionMap);
  timeIntegrator.addTimeTerm(u, v, one);

  solution->setIP( bf->graphNorm() );

  double dt = 2e-2;
  double Dt = 1e-1;
  VTKExporter exporter(solution, mesh, varFactory);

  timeIntegrator.runToTime(1*Dt, dt);
  exporter.exportSolution("timestep_confusion0");

  timeIntegrator.runToTime(2*Dt, dt);
  exporter.exportSolution("timestep_confusion1");

  // timeIntegrator.runToTime(3*Dt, dt);
  // exporter.exportSolution("timestep_confusion2");

  return 0;
}

