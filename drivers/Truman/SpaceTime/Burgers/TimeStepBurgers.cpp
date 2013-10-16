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

class ExactU1 : public Function {
  public:
    double R;
    ExactU1(double R) : Function(0), R(R) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex, ptIndex) = 0.75-1./(4*(1+exp(R*(-getTime()-4*x+4*y)/32)));
        }
      }
    }
};

class ExactU2 : public Function {
  public:
    double R;
    ExactU2(double R) : Function(0), R(R) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex, ptIndex) = 0.75+1./(4*(1+exp(R*(-getTime()-4*x+4*y)/32)));
        }
      }
    }
};

class ExactSigma1 : public Function {
  public:
    double R;
    ExactSigma1(double R) : Function(1), R(R) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex, ptIndex, 0) = -R/32*exp(R/32*(-getTime()-4*x+4*y))/pow(1+exp(R*(-getTime()-4*x+4*y)/32),2);
          values(cellIndex, ptIndex, 1) = R/32*exp(R/32*(-getTime()-4*x+4*y))/pow(1+exp(R*(-getTime()-4*x+4*y)/32),2);
        }
      }
    }
};

class ExactSigma2 : public Function {
  public:
    double R;
    ExactSigma2(double R) : Function(1), R(R) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex, ptIndex, 0) = R/32*exp(R/32*(-getTime()-4*x+4*y))/pow(1+exp(R*(-getTime()-4*x+4*y)/32),2);
          values(cellIndex, ptIndex, 1) = -R/32*exp(R/32*(-getTime()-4*x+4*y))/pow(1+exp(R*(-getTime()-4*x+4*y)/32),2);
        }
      }
    }
};

class BurgersSteadyResidual : public SteadyResidual{
  private:
    vector<double> beta;
    double R;
  public:
    BurgersSteadyResidual(VarFactory &varFactory, double R):
      SteadyResidual(varFactory), R(R) {};
    LinearTermPtr createResidual(SolutionPtr solution)
    {
      VarPtr tau1 = varFactory.testVar("tau1", HDIV);
      VarPtr tau2 = varFactory.testVar("tau2", HDIV);
      VarPtr v1 = varFactory.testVar("v1", HGRAD);
      VarPtr v2 = varFactory.testVar("v2", HGRAD);

      // define trial variables
      VarPtr u1 = varFactory.fieldVar("u1");
      VarPtr u2 = varFactory.fieldVar("u2");
      VarPtr sigma1 = varFactory.fieldVar("sigma1", VECTOR_L2);
      VarPtr sigma2 = varFactory.fieldVar("sigma2", VECTOR_L2);
      VarPtr u1hat = varFactory.traceVar("u1hat");
      VarPtr u2hat = varFactory.traceVar("u2hat");
      VarPtr f1hat = varFactory.fluxVar("f1hat");
      VarPtr f2hat = varFactory.fluxVar("f2hat");

      FunctionPtr u1_prev = Function::solution(u1, solution);
      FunctionPtr u2_prev = Function::solution(u2, solution);
      FunctionPtr sigma1_prev = Function::solution(sigma1, solution);
      FunctionPtr sigma2_prev = Function::solution(sigma2, solution);

      vector<double> e1(2); // (1,0)
      e1[0] = R;
      vector<double> e2(2); // (0,1)
      e2[1] = R;
      FunctionPtr Rbeta = e1 * u1_prev + e2 * u2_prev;
      FunctionPtr Rfunc = Function::constant(R);

      LinearTermPtr residual = Teuchos::rcp( new LinearTerm );
      residual->addTerm( Rfunc*sigma1_prev * tau1 );
      residual->addTerm( Rfunc*sigma2_prev * tau2 );
      residual->addTerm( u1_prev * tau1->div() );
      residual->addTerm( u2_prev * tau2->div() );

      residual->addTerm( Rbeta*sigma1_prev * v1);
      residual->addTerm( Rbeta*sigma2_prev * v2);
      residual->addTerm( sigma1_prev * v1->grad() );
      residual->addTerm( sigma2_prev * v2->grad() );

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
  // int numRefs = args.Input<int>("--numRefs", "number of refinement steps");

  // Optional arguments (have defaults)
  double R = args.Input<double>("--R", "effective Reynolds number", 80);
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau1 = varFactory.testVar("tau1", HDIV);
  VarPtr tau2 = varFactory.testVar("tau2", HDIV);
  VarPtr v1 = varFactory.testVar("v1", HGRAD);
  VarPtr v2 = varFactory.testVar("v2", HGRAD);

  // define trial variables
  VarPtr u1 = varFactory.fieldVar("u1");
  VarPtr u2 = varFactory.fieldVar("u2");
  VarPtr sigma1 = varFactory.fieldVar("sigma1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("sigma2", VECTOR_L2);
  VarPtr u1hat = varFactory.traceVar("u1hat");
  VarPtr u2hat = varFactory.traceVar("u2hat");
  VarPtr f1hat = varFactory.fluxVar("f1hat");
  VarPtr f2hat = varFactory.fluxVar("f2hat");

  ////////////////////   INITIALIZE USEFUL VARIABLES   ///////////////////////
  // Define useful functions
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();
  FunctionPtr u1Exact     = Teuchos::rcp( new ExactU1(R) );
  FunctionPtr u2Exact     = Teuchos::rcp( new ExactU2(R) );
  FunctionPtr sigma1Exact = Teuchos::rcp( new ExactSigma1(R) );
  FunctionPtr sigma2Exact = Teuchos::rcp( new ExactSigma2(R) );

  // Initialize useful variables
  IPPtr ip = Teuchos::rcp(new IP);
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  BFPtr steadyJacobian = Teuchos::rcp( new BF(varFactory) );
  BurgersSteadyResidual steadyResidual(varFactory, R);

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

  int horizontalCells = 8, verticalCells = 8;

  MeshPtr mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
      steadyJacobian, H1Order, H1Order+pToAdd, false);

  ////////////////////   SET INITIAL CONDITIONS   ///////////////////////
  map<int, Teuchos::RCP<Function> > initialConditions;
  initialConditions[u1->ID()]     = u1Exact;
  initialConditions[u2->ID()]     = u2Exact;
  initialConditions[sigma1->ID()] = sigma1Exact;
  initialConditions[sigma2->ID()] = sigma2Exact;

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  // Set up problem
  ImplicitEulerIntegrator timeIntegrator(steadyJacobian, steadyResidual, mesh, bc, ip, initialConditions, true);

  FunctionPtr u1_prev     = Function::solution(u1, timeIntegrator.prevSolution());
  FunctionPtr u2_prev     = Function::solution(u2, timeIntegrator.prevSolution());
  FunctionPtr sigma1_prev = Function::solution(sigma1, timeIntegrator.prevSolution());
  FunctionPtr sigma2_prev = Function::solution(sigma2, timeIntegrator.prevSolution());

  vector<double> e1(2); // (1,0)
  e1[0] = R;
  vector<double> e2(2); // (0,1)
  e2[1] = R;
  FunctionPtr Rbeta = e1 * u1_prev + e2 * u2_prev;

  // tau terms:
  steadyJacobian->addTerm( R*sigma1, tau1);
  steadyJacobian->addTerm( R*sigma2, tau2);
  steadyJacobian->addTerm( u1, tau1->div());
  steadyJacobian->addTerm( u2, tau2->div());
  steadyJacobian->addTerm( -u1hat, tau1->dot_normal());
  steadyJacobian->addTerm( -u2hat, tau2->dot_normal());

  // v terms:
  steadyJacobian->addTerm( R*Function::xPart(sigma1_prev)*u1, v1 );
  steadyJacobian->addTerm( R*Function::yPart(sigma1_prev)*u2, v1 );
  steadyJacobian->addTerm( R*Function::xPart(sigma2_prev)*u1, v2 );
  steadyJacobian->addTerm( R*Function::yPart(sigma2_prev)*u2, v2 );
  steadyJacobian->addTerm( Rbeta*sigma1, v1 );
  steadyJacobian->addTerm( Rbeta*sigma2, v2 );
  steadyJacobian->addTerm( sigma1, v1->grad() );
  steadyJacobian->addTerm( sigma2, v2->grad() );
  steadyJacobian->addTerm( -f1hat, v1);
  steadyJacobian->addTerm( -f2hat, v2);

  // time terms:
  timeIntegrator.addTimeTerm(u1, v1, one);
  timeIntegrator.addTimeTerm(u2, v2, one);


  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // Graph norm
  ip = steadyJacobian->graphNorm();
  ip->addTerm( timeIntegrator.invDt() * v1 );
  ip->addTerm( timeIntegrator.invDt() * v2 );
  timeIntegrator.solutionUpdate()->setIP(ip);

  ////////////////////   CREATE BCs   ///////////////////////
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(ymin) );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(ymax) );
  bc->addDirichlet(u1hat, left,   u1Exact);
  bc->addDirichlet(u1hat, bottom, u1Exact);
  bc->addDirichlet(u1hat, right,  u1Exact);
  bc->addDirichlet(u1hat, top,    u1Exact);
  bc->addDirichlet(u2hat, left,   u2Exact);
  bc->addDirichlet(u2hat, bottom, u2Exact);
  bc->addDirichlet(u2hat, right,  u2Exact);
  bc->addDirichlet(u2hat, top,    u2Exact);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double dt = 2e-1;
  double Dt = 1e-0;
  VTKExporter exporter(timeIntegrator.solution(), mesh, varFactory);
  VTKExporter prevExporter(timeIntegrator.prevSolution(), mesh, varFactory);

  prevExporter.exportSolution("timestep_burgers0");

  timeIntegrator.runToTime(1*Dt, dt);
  exporter.exportSolution("timestep_burgers1");

  timeIntegrator.runToTime(2*Dt, dt);
  exporter.exportSolution("timestep_burgers2");

  timeIntegrator.runToTime(3*Dt, dt);
  exporter.exportSolution("timestep_burgers3");

  // double t = dt;
  // int timestep = 0;
  // while (t <= 3)
  // {
  //   timestep++;
  //   t += dt;
  //   dynamic_cast< ExactU1* >(u1Exact.get())->t += dt;
  //   dynamic_cast< ExactU2* >(u2Exact.get())->t += dt;
  //   dynamic_cast< ExactSigma1* >(sigma1Exact.get())->t += dt;
  //   dynamic_cast< ExactSigma2* >(sigma2Exact.get())->t += dt;
  //   double uUpdateL2 = 1e9;
  //   while (uUpdateL2 > 1e-6)
  //   {
  //     solution->solve();
  //     uUpdateL2 = solution->L2NormOfSolution(0);
  //     cout << "Update size = " << uUpdateL2 << endl;
  //     backgroundFlow->addSolution(solution, 1);
  //   }
  //   stringstream outfile;
  //   outfile << "Burgers_" << timestep;
  //   exporter.exportSolution(outfile.str());
  // }

  // // ImplicitEulerIntegrator timeIntegrator(bf, rhs, mesh, solution, functionMap);
  // ESDIRKIntegrator timeIntegrator(bf, rhs, mesh, solution, functionMap, 6);
  // timeIntegrator.addTimeTerm(u, v, one);

  // solution->setIP( bf->graphNorm() );

  // double dt = 5e-2;
  // double Dt = 1e-1;
  // VTKExporter exporter(solution, mesh, varFactory);

  // timeIntegrator.runToTime(1*Dt, dt);
  // exporter.exportSolution("timestep_confusion0");

  // timeIntegrator.runToTime(2*Dt, dt);
  // exporter.exportSolution("timestep_confusion1");

  // timeIntegrator.runToTime(3*Dt, dt);
  // exporter.exportSolution("timestep_confusion2");

  return 0;
}

