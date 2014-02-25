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

double epsilon = 1e-2;

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
          // if (y >= .25 && y <= 0.5 && x >= 0.5-1./8. && x <= 0.5+1./8.)
          if (x >= 0.5-1./8. && x <= 0.5+1./8.)
            values(cellIndex, ptIndex) = 1;
          // else if (y >= .5 && y <= 0.75 && x >= 0.5-1./8. && x <= 0.5+1./8.)
          //   values(cellIndex, ptIndex) = -1;
          else
            values(cellIndex, ptIndex) = 0;
        }
      }
    }
};

// class ShiftedTimeFunction : public Function {
//   private:
//     FunctionPtr _original;
//   public:
//     ShiftedTimeFunction(exportFunction original) : Function(0), _original(original) {}
//     void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
//       int numCells = values.dimension(0);
//       int numPoints = values.dimension(1);

//       const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
//       for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
//         for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
//           double x = (*points)(cellIndex,ptIndex,0);
//           double y = (*points)(cellIndex,ptIndex,1);
//           values(cellIndex, ptIndex) = 1;
//         }
//       }
//     }
// };

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
  int numX = args.Input("--numX", "number of cells in x direction", 4);
  int numRefs = args.Input("--numRefs", "number of refinement steps", 0);
  int numSlabs = args.Input("--numSlabs", "number of time slabs", 2);
  int numY = numX/numSlabs;
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("tau", HGRAD);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma = varFactory.fieldVar("sigma", L2);
  VarPtr uhat = varFactory.spatialTraceVar("uhat");
  VarPtr fhat = varFactory.fluxVar("fhat");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  vector< Teuchos::RCP<Mesh> > meshes;
  double xmin, xmax, tmin, tmax;
  vector<double> tmins;
  vector<double> tmaxs;
  for (int i=0; i < numSlabs; i++)
  {
    // define nodes for mesh
    FieldContainer<double> meshBoundary(4,2);
    xmin = 0.0;
    xmax = 1.0;
    tmin = 1.0*double(i)/numSlabs;
    tmax = 1.0*double(i+1)/numSlabs;
    cout << "Creating time slab [" << tmin << "," << tmax << "]" << endl;

    meshBoundary(0,0) =  xmin; // x1
    meshBoundary(0,1) =  tmin; // y1
    meshBoundary(1,0) =  xmax;
    meshBoundary(1,1) =  tmin;
    meshBoundary(2,0) =  xmax;
    meshBoundary(2,1) =  tmax;
    meshBoundary(3,0) =  xmin;
    meshBoundary(3,1) =  tmax;

    // create a pointer to a new mesh:
    Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, numX, numY,
      bf, H1Order, H1Order+pToAdd);
    meshes.push_back(mesh);
    tmins.push_back(tmin);
    tmaxs.push_back(tmax);
  }

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  // tau terms:
  bf->addTerm( sigma/epsilon, tau );
  bf->addTerm( u, tau->dx() );
  bf->addTerm( -uhat, tau->times_normal_x() );

  // v terms:
  bf->addTerm( sigma, v->dx() );
  bf->addTerm( -u, v->dy() );
  bf->addTerm( fhat, v);

  FunctionPtr one = Function::constant(1.0);
  FunctionPtr zero = Function::zero();

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new Forcing );
  // rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();

  ////////////////////   CREATE BCs   ///////////////////////
  vector< Teuchos::RCP<BCEasy> > bcs;
  FunctionPtr fhat_prev;
  for (int slab=0; slab < numSlabs; slab++)
  {
    Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
    SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
    SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
    SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(tmins[slab]) );
    SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(tmaxs[slab]) );
    FunctionPtr uLeft = zero;
    FunctionPtr uRight = zero;
    bc->addDirichlet(fhat, right, zero);
    bc->addDirichlet(fhat, left, zero);
    if (slab == 0)
      bc->addDirichlet(fhat, bottom, f);
      // bc->addDirichlet(fhat, bottom, zero);
    // else
    //   bc->addDirichlet(fhat, bottom, fhat_prev);
    bcs.push_back(bc);
  }

  vector< Teuchos::RCP<Solution> > solutions;
  for (int slab=0; slab < numSlabs; slab++)
  {
    Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(meshes[slab], bcs[slab], rhs, ip) );
    solutions.push_back(solution);
    if (slab > 0)
    {
      fhat_prev = Function::solution(fhat, solutions[slab-1]);
      // fhat_prev = f;
      SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(tmins[slab]) );
      bcs[slab]->addDirichlet(fhat, bottom, -fhat_prev);
    }

    // ==================== Register Solutions ==========================
    meshes[slab]->registerSolution(solution);

    ////////////////////   SOLVE & REFINE   ///////////////////////
    double energyThreshold = 0.2; // for mesh refinements
    RefinementStrategy refinementStrategy( solution, energyThreshold );
    VTKExporter exporter(solution, meshes[slab], varFactory);
    // exporter.exportFunction(fhat_prev, "fhat_prev");

    for (int refIndex=0; refIndex<=numRefs; refIndex++)
    {
       solution->solve(false);

       FunctionPtr u_soln = Function::solution(u, solution);
       if (commRank == 0)
       {
          stringstream outfile;
          outfile << "heat_" << slab << "_" << refIndex;
          exporter.exportSolution(outfile.str());
      }

      if (refIndex < numRefs)
        refinementStrategy.refine(commRank==0); // print to console on commRank 0
    }
  }

  return 0;
}

