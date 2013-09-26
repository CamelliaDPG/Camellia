//  SimpleConvection.cpp
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

int H1Order = 3, pToAdd = 2;

class TimeZeroBoundary : public SpatialFilter {
  public:
    bool matchesPoint(double x, double y) {
      double tol = 1e-14;
      return (abs(y) < tol);
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
            values(cellIndex, ptIndex) = 1.0;
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
  double alpha = args.Input<double>("--alpha", "dudx multiplier", 1);
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr v = varFactory.testVar("v", HGRAD);
  // VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr tau = varFactory.testVar("tau", VECTOR_HGRAD);

  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma_x = varFactory.fieldVar("sigma_x");
  VarPtr sigma_t = varFactory.fieldVar("sigma_t");
  VarPtr uhat = varFactory.traceVar("uhat");
  VarPtr fhat = varFactory.fluxVar("fhat");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  double xmin = -0.5;
  double xmax = 0.5;
  double tmax = 1.0;

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  0.0; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  0.0;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  tmax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  tmax;

  int horizontalCells = 8, verticalCells = 8;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                                                bf, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  double epsilon_x = 1e-2;
  double epsilon_t = 1e-22;

  // // v terms:
  // bf->addTerm( -alpha*u, v->dx() );
  // bf->addTerm( -u, v->dy() );
  // bf->addTerm( sigma_x, v->dx() );
  // bf->addTerm( sigma_t, v->dy() );
  // bf->addTerm( fhat, v);

  // // tau terms:
  // bf->addTerm( sigma_x/epsilon_x, tau->x() );
  // bf->addTerm( sigma_t/epsilon_t, tau->y() );
  // bf->addTerm( u, tau->div() );
  // bf->addTerm( -uhat, tau->dot_normal() );

  FunctionPtr n = Function::normal();
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();
  FunctionPtr one_zero = Function::vectorize(one,zero);
  FunctionPtr zero_one = Function::vectorize(zero,one);
  FunctionPtr zero_zero = Function::vectorize(zero,zero);
  FunctionPtr xxPart = sqrt(epsilon_x)*Function::vectorize(one_zero, zero_zero);
  FunctionPtr yyPart = sqrt(epsilon_t)*Function::vectorize(zero_zero, zero_one);
  FunctionPtr Dsqrt = xxPart + yyPart;

  // tau terms:
  bf->addTerm( sigma_x, tau->x() );
  bf->addTerm( sigma_t, tau->y() );
  bf->addTerm( u, Dsqrt*tau->grad() );
  bf->addTerm( -uhat, sqrt(epsilon_x)*n->x()*tau->x()+sqrt(epsilon_t)*n->y()*tau->y() );

  // v terms:
  bf->addTerm( -alpha*u, v->dx() );
  bf->addTerm( -u, v->dy() );
  bf->addTerm( sqrt(epsilon_x)*sigma_x, v->dx() );
  bf->addTerm( sqrt(epsilon_t)*sigma_t, v->dy() );
  bf->addTerm( fhat, v);

  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();
  // ip->addTerm(v);
  // ip->addTerm(beta*v->grad());

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr timezero = Teuchos::rcp( new TimeZeroBoundary );
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  FunctionPtr u0 = Teuchos::rcp( new InitialCondition );
  FunctionPtr uLeft = zero;
  FunctionPtr uRight = zero;
  bc->addDirichlet(fhat, timezero, -u0);
  bc->addDirichlet(uhat, right, zero);
  bc->addDirichlet(fhat, left, zero);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(solution, mesh, varFactory);

  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
     solution->solve(false);

     if (commRank == 0)
     {
        stringstream outfile;
        outfile << "Confusion_" << refIndex;
        exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

