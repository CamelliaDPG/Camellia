//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"
#include "TimeIntegrator.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double pi = 2.0*acos(0.0);

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

class InitialU : public Function {
  public:
    InitialU() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (abs(x) <= 0.25)
            values(cellIndex, ptIndex) = 1+cos(4*pi*x);
          else
            values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

class ConfusionSteadyResidual : public SteadyResidual{
  private:
    vector<double> beta;
    double epsilon;
  public:
    ConfusionSteadyResidual(VarFactory &varFactory, vector<double> beta, double epsilon):
      SteadyResidual(varFactory), beta(beta), epsilon(epsilon) {};
    LinearTermPtr createResidual(SolutionPtr solution)
    {
      VarPtr tau = varFactory.testVar("tau", HDIV);
      VarPtr v = varFactory.testVar("v", HGRAD);

      VarPtr u = varFactory.fieldVar("u");
      VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);
      VarPtr uhat = varFactory.traceVar("uhat");
      VarPtr fhat = varFactory.fluxVar("fhat");

      FunctionPtr u_prev = Function::solution(u, solution);
      FunctionPtr sigma_prev = Function::solution(sigma, solution);

      LinearTermPtr residual = Teuchos::rcp( new LinearTerm );
      residual->addTerm( beta*u_prev * -v->grad() );
      residual->addTerm( sigma_prev * v->grad() );

      residual->addTerm( sigma_prev / epsilon * tau);
      residual->addTerm( u_prev * tau->div());

      return residual;
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

  ////////////////////   INITIALIZE USEFUL VARIABLES   ///////////////////////
  // Define useful functions
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();
  FunctionPtr u0 = Teuchos::rcp( new InitialU );
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);

  // Initialize useful variables
  IPPtr ip = Teuchos::rcp(new IP);
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  BFPtr steadyJacobian = Teuchos::rcp( new BF(varFactory) );
  ConfusionSteadyResidual steadyResidual(varFactory, beta, epsilon);

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
      steadyJacobian, H1Order, H1Order+pToAdd, false);

  ////////////////////   SET INITIAL CONDITIONS   ///////////////////////
  map<int, Teuchos::RCP<Function> > initialConditions;
  initialConditions[u->ID()]     = u0;
  initialConditions[sigma->ID()] = Function::vectorize(zero, zero);
  initialConditions[uhat->ID()]  = zero;
  initialConditions[fhat->ID()]  = zero;

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  // Set up problem
  // ImplicitEulerIntegrator timeIntegrator(steadyJacobian, steadyResidual, mesh, bc, ip, initialConditions, true);
  ESDIRKIntegrator timeIntegrator(steadyJacobian, steadyResidual, mesh, bc, ip, initialConditions, 2, true);

  // tau terms:
  steadyJacobian->addTerm( sigma / epsilon, tau);
  steadyJacobian->addTerm( u, tau->div());
  steadyJacobian->addTerm( -uhat, tau->dot_normal());

  // v terms:
  steadyJacobian->addTerm( sigma, v->grad() );
  steadyJacobian->addTerm( beta * u, - v->grad() );
  steadyJacobian->addTerm( fhat, v);

  // time terms:
  timeIntegrator.addTimeTerm(u, v, one);

  ////////////////////   DEFINE INNER PRODUCT   ///////////////////////
  // Graph norm
  if (norm == 0)
  {
    ip = steadyJacobian->graphNorm();
    timeIntegrator.solutionUpdate()->setIP(ip);
  }
  // Coupled robust norm
  else if (norm == 1)
  {
    // coupled robust test norm
    FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) );
    ip->addTerm( v );
    ip->addTerm( sqrt(epsilon) * v->grad() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->div() - beta*v->grad() );
    ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
  }
  ip->addTerm( timeIntegrator.invDt() * v );

  ////////////////////   CREATE BCs   ///////////////////////
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(ymin) );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(ymax) );
  bc->addDirichlet(fhat, left, zero);
  bc->addDirichlet(fhat, bottom, zero);
  bc->addDirichlet(uhat, right, zero);
  bc->addDirichlet(fhat, top, zero);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double dt = 1e-1;
  double Dt = 1e-1;
  VTKExporter exporter(timeIntegrator.solution(), mesh, varFactory);
  VTKExporter prevExporter(timeIntegrator.prevSolution(), mesh, varFactory);

  prevExporter.exportSolution("timestep_confusion0");

  timeIntegrator.runToTime(1*Dt, dt);
  exporter.exportSolution("timestep_confusion1");

  timeIntegrator.runToTime(2*Dt, dt);
  exporter.exportSolution("timestep_confusion2");

  timeIntegrator.runToTime(3*Dt, dt);
  exporter.exportSolution("timestep_confusion3");

  return 0;
}

