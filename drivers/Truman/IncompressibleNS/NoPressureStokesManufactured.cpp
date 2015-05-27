//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "RefinementHistory.h"
#include "SolutionExporter.h"
#include "MeshFactory.h"
#include "CheckConservation.h"
#include "LagrangeConstraints.h"
#include "PenaltyConstraints.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

double pi = 2.0*acos(0.0);

class ConstantXBoundary : public SpatialFilter
{
private:
  double xval;
public:
  ConstantXBoundary(double xval): xval(xval) {};
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(x-xval) < tol);
  }
};

class ConstantYBoundary : public SpatialFilter
{
private:
  double yval;
public:
  ConstantYBoundary(double yval): yval(yval) {};
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y-yval) < tol);
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
  int norm = args.Input<int>("--norm", "0 = graph\n    1 = robust\n    2 = coupled robust");

  // Optional arguments (have defaults)
  int polyOrder = args.Input("--polyOrder", "polynomial order for field variables", 2);
  int deltaP = args.Input("--deltaP", "how much to enrich test space", 2);
  // string saveFile = args.Input<string>("--meshSaveFile", "file to which to save refinement history", "");
  // string replayFile = args.Input<string>("--meshLoadFile", "file with refinement history to replay", "");
  args.Process();

  // if (commRank==0)
  // {
  //   cout << "saveFile is " << saveFile << endl;
  //   cout << "loadFile is " << replayFile << endl;
  // }

  ////////////////////   PROBLEM DEFINITIONS   ///////////////////////
  int H1Order = polyOrder+1;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau11 = varFactory.testVar("tau11", HGRAD);
  VarPtr tau12 = varFactory.testVar("tau12", HGRAD);
  VarPtr tau22 = varFactory.testVar("tau22", HGRAD);
  VarPtr v1 = varFactory.testVar("v1", HGRAD);
  VarPtr v2 = varFactory.testVar("v2", HGRAD);

  // define trial variables
  VarPtr u1 = varFactory.fieldVar("u1");
  VarPtr u2 = varFactory.fieldVar("u2");
  VarPtr sigma11 = varFactory.fieldVar("sigma11");
  VarPtr sigma12 = varFactory.fieldVar("sigma12");
  VarPtr sigma22 = varFactory.fieldVar("sigma22");
  VarPtr u1hat = varFactory.traceVar("u1hat");
  VarPtr u2hat = varFactory.traceVar("u2hat");
  VarPtr t1hat = varFactory.fluxVar("t1hat");
  VarPtr t2hat = varFactory.fluxVar("t2hat");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );

  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);
  double xmin =  0.0;
  double xmax =  1.0;
  double ymin =  0.0;
  double ymax =  1.0;

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  ymin; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  ymin;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  ymax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  ymax;

  int horizontalCells = 4, verticalCells = 4;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, horizontalCells, verticalCells,
                            bf, H1Order, H1Order+deltaP);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  FunctionPtr u1_prev = Function::solution(u1, backgroundFlow);
  FunctionPtr u2_prev = Function::solution(u2, backgroundFlow);

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  FunctionPtr y = Function::yn(1);

  // ==================== SET INITIAL GUESS ==========================
  // map<int, Teuchos::RCP<Function> > functionMap;
  // functionMap[u1->ID()] = u1Exact;
  // functionMap[u2->ID()] = u2Exact;

  // backgroundFlow->projectOntoMesh(functionMap);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  // stress equation
  bf->addTerm( sigma11, tau11 );
  bf->addTerm( sigma12, tau12 );
  bf->addTerm( sigma12, tau12 );
  bf->addTerm( sigma22, tau22 );
  bf->addTerm( -0.5*sigma11, tau11 );
  bf->addTerm( -0.5*sigma22, tau11 );
  bf->addTerm( -0.5*sigma11, tau22 );
  bf->addTerm( -0.5*sigma22, tau22 );
  bf->addTerm( 2*u1, tau11->dx() );
  bf->addTerm( 2*u1, tau12->dy() );
  bf->addTerm( 2*u2, tau12->dx() );
  bf->addTerm( 2*u2, tau22->dy() );
  bf->addTerm( -2*u1hat, tau11->times_normal_x() );
  bf->addTerm( -2*u1hat, tau12->times_normal_y() );
  bf->addTerm( -2*u2hat, tau12->times_normal_x() );
  bf->addTerm( -2*u2hat, tau22->times_normal_y() );

  // momentum equation
  bf->addTerm( sigma11, v1->dx() );
  bf->addTerm( sigma12, v1->dy() );
  bf->addTerm( sigma12, v2->dx() );
  bf->addTerm( sigma22, v2->dy() );
  bf->addTerm( t1hat, v1);
  bf->addTerm( t2hat, v2);

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  // manufactured solution
  rhs->addTerm( y*tau12 );
  rhs->addTerm( y*tau12 );
  rhs->addTerm( -v1 );

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  if (norm == 0)
  {
    ip = bf->graphNorm();
  }
  else if (norm == 1)
  {
    ip = Teuchos::rcp( new IP );
    ip->addTerm( 0.5*tau11-0.5*tau22 + v1->dx() );
    ip->addTerm( tau12 + v1->dy() );
    ip->addTerm( tau12 + v2->dx() );
    ip->addTerm( 0.5*tau22-0.5*tau11 + v2->dy() );

    ip->addTerm( 2*tau11->dx() + 2*tau12->dy() );
    ip->addTerm( 2*tau12->dx() + 2*tau22->dy() );

    ip->addTerm( v1 );
    ip->addTerm( v2 );
    ip->addTerm( tau11 );
    ip->addTerm( tau12 );
    ip->addTerm( tau12 );
    ip->addTerm( tau22 );
  }
  else if (norm == 2)
  {
    ip = Teuchos::rcp( new IP );
    ip->addTerm( 0.5*tau11-0.5*tau22 + v1->dx() );
    ip->addTerm( tau12 + 0.5*one*(v1->dy() + v2->dx()) );
    ip->addTerm( tau12 + 0.5*one*(v1->dy() + v2->dx()) );
    ip->addTerm( 0.5*tau22-0.5*tau11 + v2->dy() );

    ip->addTerm( 2*tau11->dx() + 2*tau12->dy() );
    ip->addTerm( 2*tau12->dx() + 2*tau22->dy() );

    ip->addTerm( v1 );
    ip->addTerm( v2 );
    ip->addTerm( tau11 );
    ip->addTerm( tau12 );
    ip->addTerm( tau12 );
    ip->addTerm( tau22 );
  }

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  // Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp( new PenaltyConstraints );
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(0) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(1) );
  SpatialFilterPtr bottom = Teuchos::rcp( new ConstantYBoundary(0) );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(1) );
  bc->addDirichlet(u1hat, left, zero);
  bc->addDirichlet(u2hat, left, zero);
  bc->addDirichlet(u1hat, bottom, zero);
  bc->addDirichlet(u2hat, bottom, zero);
  bc->addDirichlet(t1hat, right, zero);
  bc->addDirichlet(t2hat, right, -Function::yn(1));
  bc->addDirichlet(t1hat, top, -one);
  bc->addDirichlet(t2hat, top, zero);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  // if (enforceLocalConservation) {
  //   solution->lagrangeConstraints()->addConstraint(u1hat->times_normal_x() + u2hat->times_normal_y() == zero);
  // }

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);

  // Teuchos::RCP< RefinementHistory > refHistory = Teuchos::rcp( new RefinementHistory );
  // mesh->registerObserver(refHistory);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(solution, mesh, varFactory);

  double nonlinearRelativeEnergyTolerance = 1e-5; // used to determine convergence of the nonlinear solution
  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    solution->solve(false);

    if (commRank == 0)
    {
      stringstream outfile;
      outfile << "manufacturedStokes" << "_" << refIndex;
      exporter.exportSolution(outfile.str());
    }

    if (commRank == 0)
    {
      FunctionPtr u1Solution = Function::solution(u1, solution);
      FunctionPtr u2Solution = Function::solution(u2, solution);
      FunctionPtr sigma11Solution = Function::solution(sigma11, solution);
      FunctionPtr sigma12Solution = Function::solution(sigma12, solution);
      FunctionPtr sigma22Solution = Function::solution(sigma22, solution);
      FunctionPtr u1Diff = u1Solution;
      FunctionPtr u2Diff = u2Solution;
      FunctionPtr sigma11Diff = sigma11Solution;
      FunctionPtr sigma12Diff = sigma12Solution-y;
      FunctionPtr sigma22Diff = sigma22Solution;
      FunctionPtr u1Sqr = u1Diff*u1Diff;
      FunctionPtr u2Sqr = u2Diff*u2Diff;
      FunctionPtr sigma11Sqr = sigma11Diff*sigma11Diff;
      FunctionPtr sigma12Sqr = sigma12Diff*sigma12Diff;
      FunctionPtr sigma22Sqr = sigma22Diff*sigma22Diff;

      double u1Error = u1Sqr->integrate(mesh);
      double u2Error = u2Sqr->integrate(mesh);
      double sigma11Error = sigma11Sqr->integrate(mesh);
      double sigma12Error = sigma12Sqr->integrate(mesh);
      double sigma22Error = sigma22Sqr->integrate(mesh);
      double L2Error = sqrt(u1Error + u2Error);
      cout << "L2 Error = " << endl
           << sqrt(u1Error) << endl
           << sqrt(u2Error) << endl
           << sqrt(sigma11Error) << endl
           << sqrt(sigma12Error) << endl
           << sqrt(sigma22Error) << endl;
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

