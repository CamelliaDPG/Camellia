//  SimpleConvection.cpp
//  Driver for Conservative Convection
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

class Inlet : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(x+1) < tol);
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
        if (abs(x) <= 0.25)
          values(cellIndex, ptIndex) = 1.0;
        else
          values(cellIndex, ptIndex) = 0;
      }
    }
  }
};

// boundary value for sigma_n
class InletValue : public Function
{
public:
  InletValue() : Function(0) {}
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
        values(cellIndex, ptIndex) = 0.0;
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
  double alpha = args.Input<double>("--alpha", "dudx multiplier");

  // Optional arguments (have defaults)
  args.Process();

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory.fluxVar("uhat");
  VarPtr u = varFactory.fieldVar("u");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);

  meshBoundary(0,0) = -1.0; // x1
  meshBoundary(0,1) =  0.0; // y1
  meshBoundary(1,0) =  1.0;
  meshBoundary(1,1) =  0.0;
  meshBoundary(2,0) =  1.0;
  meshBoundary(2,1) =  1.0;
  meshBoundary(3,0) = -1.0;
  meshBoundary(3,1) =  1.0;

  int horizontalCells = 8, verticalCells = 8;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                            bf, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  // v terms:
  bf->addTerm( alpha*u, -v->dx() );
  bf->addTerm( u, -v->dy() );
  bf->addTerm( uhat, v);

  ////////////////////   SPECIFY RHS   ///////////////////////
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();
  // ip->addTerm(v);
  // ip->addTerm(beta*v->grad());

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr Boundary0 = Teuchos::rcp( new TimeZero );
  SpatialFilterPtr inlet = Teuchos::rcp( new Inlet );
  FunctionPtr u0 = Teuchos::rcp( new InitialCondition );
  FunctionPtr u_in = Teuchos::rcp( new InletValue );
  bc->addDirichlet(uhat, Boundary0, -u0);
  bc->addDirichlet(uhat, inlet, u_in);

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
      outfile << "Convection_" << refIndex;
      exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

