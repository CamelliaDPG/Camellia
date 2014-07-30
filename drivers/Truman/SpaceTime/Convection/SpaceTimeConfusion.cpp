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
double pi = 2.0*acos(0.0);

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
      double val;
   public:
      ConstantXBoundary(double val): val(val) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(x-val) < tol);
      }
};

class ConstantYBoundary : public SpatialFilter {
   private:
      double val;
   public:
      ConstantYBoundary(double val): val(val) {};
      bool matchesPoint(double x, double y) {
         double tol = 1e-14;
         return (abs(y-val) < tol);
      }
};

// class UExact : public Function {
//   public:
//     UExact() : Function(0) {}
//     void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
//       int numCells = values.dimension(0);
//       int numPoints = values.dimension(1);

//       const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
//       for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//         for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//           double x = (*points)(cellIndex,ptIndex,0);
//           double y = (*points)(cellIndex,ptIndex,1);
//           values(cellIndex, ptIndex) = cos(2*pi*x)*exp(-4*pi*pi*epsilon*y);
//         }
//       }
//     }
// };

class Forcing : public Function {
  public:
    Forcing() : Function(0) {}
    void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
      int numCells = values.dimension(0);
      int numPoints = values.dimension(1);

      const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
      for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
        for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          if (y >= .25 && y <= 0.5 && x >= 0.25 && x <= 0.5)
            values(cellIndex, ptIndex) = 1;
          // else if (y >= .5 && y <= 0.75 && x >= 0.5-1./8. && x <= 0.5+1./8.)
          //   values(cellIndex, ptIndex) = -1;
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

  // Optional arguments (have defaults)
  int norm = args.Input("--norm", "test norm", 1);
  int numRefs = args.Input("--numRefs", "number of refinement steps", 0);
  double epsilon = args.Input<double>("--epsilon", "diffusion parameter");
  args.Process();

  int numX = 6;
  int numY = 4;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("tau", HGRAD);
  // VarPtr tau = varFactory.testVar("tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", L2);
  // VarPtr sigma = varFactory.fieldVar("sigma", VECTOR_L2);
  VarPtr uhat = varFactory.spatialTraceVar("uhat");
  // VarPtr uhat = varFactory.traceVar("uhat");
  VarPtr fhat = varFactory.fluxVar("fhat");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  double xmin = 0.0;
  double xmax = 1.5;
  double tmax = 1.0;

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  0.0; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  0.0;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  tmax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  tmax;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, numX, numY,
                                                bf, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  // tau terms:
  bf->addTerm( sigma/epsilon, tau );
  bf->addTerm( u, tau->dx() );
  bf->addTerm( -uhat, tau->times_normal_x() );

  // v terms:
  bf->addTerm( -u, v->dx() );
  bf->addTerm( sigma, v->dx() );
  bf->addTerm( -u, v->dy() );
  bf->addTerm( fhat, v);

  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new Forcing );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(epsilon) );
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);
  switch (norm)
  {
    // Automatic graph norm
    case 0:
    ip = bf->graphNorm();
    break;

    // Modified Robust Norm
    case 1:
    ip->addTerm( v );
    ip->addTerm( sqrt(epsilon) * v->grad() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->dx() - beta*v->grad() );
    ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
    break;

    // Modified Robust Norm 2
    case 2:
    ip->addTerm( v );
    ip->addTerm( sqrt(epsilon) * v->dx() );
    ip->addTerm( beta * v->grad() );
    ip->addTerm( tau->dx() - beta*v->grad() );
    ip->addTerm( ip_scaling/sqrt(epsilon) * tau );
    break;

    default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid problem number");
  }

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(0) );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(tmax) );
  // FunctionPtr u_exact = Teuchos::rcp( new UExact );
  FunctionPtr uLeft = zero;
  FunctionPtr uRight = zero;
  bc->addDirichlet(fhat, right, zero);
  bc->addDirichlet(fhat, left, zero);
  // bc->addDirichlet(fhat, bottom, -u_exact);
  bc->addDirichlet(fhat, bottom, zero);

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

     FunctionPtr u_soln = Function::solution(u, solution);
     if (commRank == 0)
     {
        stringstream outfile;
        outfile << "confusion_" << refIndex;
        exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

