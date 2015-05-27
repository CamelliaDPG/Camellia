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

int H1Order = 5, pToAdd = 2;

class TimeZero : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y) < tol);
  }
};

class LeftBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(x) < tol);
  }
};

class RightBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(x-1) < tol);
  }
};

// boundary value for sigma_n
class InitialCondition : public Function
{
public:
  InitialCondition() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        if (abs(x-0.5) <= 0.1)
          values(cellIndex, ptIndex) = pow(sin((x-0.4)/0.2),1);
        else
          values(cellIndex, ptIndex) = 0;
      }
    }
  }
};

int main(int argc, char *argv[])
{
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
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr v = varFactory.testVar("v", HGRAD);
  VarPtr q = varFactory.testVar("q", HGRAD);

  // define trial variables
  VarPtr rho = varFactory.fieldVar("rho");
  VarPtr rhohat = varFactory.fluxVar("rhohat");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr uhat = varFactory.fluxVar("uhat");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);

  meshBoundary(0,0) =  0.0; // x1
  meshBoundary(0,1) =  0.0; // y1
  meshBoundary(1,0) =  1.0;
  meshBoundary(1,1) =  0.0;
  meshBoundary(2,0) =  1.0;
  meshBoundary(2,1) =  0.25;
  meshBoundary(3,0) =  0.0;
  meshBoundary(3,1) =  0.25;

  int horizontalCells = 8, verticalCells = 2;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                            bf, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  double U = 0.5;
  double a = 1.0;

  // q terms:
  bf->addTerm( rho, -q->dy() );
  bf->addTerm( rhohat, q );
  bf->addTerm( a*u, -q->dx() );
  bf->addTerm( a*uhat, q );
  bf->addTerm( U*rho, -q->dx() );
  bf->addTerm( U*rhohat, q );

  // v terms:
  bf->addTerm( u, -v->dy() );
  bf->addTerm( uhat, v );
  bf->addTerm( U*u, -v->dx() );
  bf->addTerm( U*uhat, v );
  bf->addTerm( a*rho, -v->dx() );
  bf->addTerm( a*rhohat, v );

  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();
  // ip->addTerm(v);
  // ip->addTerm(beta*v->grad());

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr timezero = Teuchos::rcp( new TimeZero );
  SpatialFilterPtr left = Teuchos::rcp( new LeftBoundary );
  SpatialFilterPtr right = Teuchos::rcp( new RightBoundary );
  FunctionPtr u0 = Teuchos::rcp( new InitialCondition );
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  bc->addDirichlet(uhat, timezero, -u0);
  bc->addDirichlet(rhohat, timezero, zero);
  bc->addDirichlet(uhat, left, zero);
  bc->addDirichlet(rhohat, left, zero);
  bc->addDirichlet(uhat, right, zero);
  bc->addDirichlet(rhohat, right, zero);

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
      outfile << "IsentropicSinePulse_" << refIndex;
      exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

