//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "PenaltyConstraints.h"
#include "HDF5Exporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

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

class BottomFree : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y) < tol && abs(x) > .5);
  }
};

class BottomPlate : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y) < tol && abs(x) <= 0.5);
  }
};

class IPWeight : public Function
{
public:
  IPWeight() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++)
    {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++)
      {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        values(cellIndex, ptIndex) = 0.1+1-abs(x);
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
  int Re = args.Input<int>("--Re", "Reynolds number");
  int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
  int maxNewtonIterations = args.Input<int>("--maxIterations", "maximum number of Newton iterations");
  int norm = args.Input<int>("--norm", "0 = graph\n    1 = robust\n    2 = coupled robust");

  // Optional arguments (have defaults)
  int polyOrder = args.Input("--polyOrder", "polynomial order for field variables", 2);
  int deltaP = args.Input("--deltaP", "how much to enrich test space", 2);
  int xCells = args.Input("--xCells", "number of cells in the x direction", 4);
  int yCells = args.Input("--yCells", "number of cells in the t direction", 2);
  args.Process();

  ////////////////////   PROBLEM DEFINITIONS   ///////////////////////
  int H1Order = polyOrder+1;

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau1 = varFactory.testVar("tau1", HDIV);
  VarPtr tau2 = varFactory.testVar("tau2", HDIV);
  VarPtr v1 = varFactory.testVar("v1", HGRAD);
  VarPtr v2 = varFactory.testVar("v2", HGRAD);
  VarPtr q = varFactory.testVar("q", HGRAD);

  // define trial variables
  VarPtr u1 = varFactory.fieldVar("u1");
  VarPtr u2 = varFactory.fieldVar("u2");
  VarPtr p = varFactory.fieldVar("p");
  VarPtr u1hat = varFactory.traceVar("u1hat");
  VarPtr u2hat = varFactory.traceVar("u2hat");
  VarPtr t1hat = varFactory.fluxVar("t1hat");
  VarPtr t2hat = varFactory.fluxVar("t2hat");
  VarPtr sigma1 = varFactory.fieldVar("sigma1", VECTOR_L2);
  VarPtr sigma2 = varFactory.fieldVar("sigma2", VECTOR_L2);

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);

  double xmin = -1;
  double xmax =  1;
  double ymin =  0;
  double ymax =  1;

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  ymin; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  ymin;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  ymax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  ymax;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, xCells, yCells,
                            bf, H1Order, H1Order+deltaP);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  vector<double> e1(2); // (1,0)
  e1[0] = 1;
  vector<double> e2(2); // (0,1)
  e2[1] = 1;

  FunctionPtr u1_prev = Function::solution(u1, backgroundFlow);
  FunctionPtr u2_prev = Function::solution(u2, backgroundFlow);

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  FunctionPtr beta = e1 * u1_prev + e2 * u2_prev;

  // ==================== SET INITIAL GUESS ==========================
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u1->ID()] = one;
  functionMap[u2->ID()] = zero;

  backgroundFlow->projectOntoMesh(functionMap);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  // stress equation
  bf->addTerm( sigma1, tau1 );
  bf->addTerm( sigma2, tau2 );
  bf->addTerm( u1, tau1->div() );
  bf->addTerm( u2, tau2->div() );
  bf->addTerm( -u1hat, tau1->dot_normal() );
  bf->addTerm( -u2hat, tau2->dot_normal() );

  // momentum equation
  bf->addTerm( -0.5*u1_prev*u1, v1->dx() );
  bf->addTerm( -0.5*u1_prev*u2, v1->dy() );
  bf->addTerm( -0.5*u2_prev*u1, v2->dx() );
  bf->addTerm( -0.5*u2_prev*u2, v2->dy() );
  bf->addTerm( -0.5*u1_prev*u1, v1->dx() );
  bf->addTerm( -0.5*u2_prev*u1, v1->dy() );
  bf->addTerm( -0.5*u1_prev*u2, v2->dx() );
  bf->addTerm( -0.5*u2_prev*u2, v2->dy() );
  bf->addTerm( 1./Re*sigma1, v1->grad() );
  bf->addTerm( 1./Re*sigma2, v2->grad() );
  bf->addTerm( -p, v1->dx() );
  bf->addTerm( -p, v2->dy() );
  bf->addTerm( t1hat, v1);
  bf->addTerm( t2hat, v2);

  // continuity equation
  bf->addTerm( -u1, q->dx() );
  bf->addTerm( -u2, q->dy() );
  bf->addTerm( u1hat, q->times_normal_x() );
  bf->addTerm( u2hat, q->times_normal_y() );

  ////////////////////   SPECIFY RHS   ///////////////////////
  RHSPtr rhs = RHS::rhs();

  // stress equation
  rhs->addTerm( -u1_prev * tau1->div() );
  rhs->addTerm( -u2_prev * tau2->div() );

  // momentum equation
  rhs->addTerm( 0.5*u1_prev*u1_prev * v1->dx() );
  rhs->addTerm( 0.5*u1_prev*u2_prev * v1->dy() );
  rhs->addTerm( 0.5*u2_prev*u1_prev * v2->dx() );
  rhs->addTerm( 0.5*u2_prev*u2_prev * v2->dy() );

  // continuity equation
  rhs->addTerm( u1_prev * q->dx() );
  rhs->addTerm( u2_prev * q->dy() );

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = Teuchos::rcp(new IP);
  if (norm == 0)
  {
    ip = bf->graphNorm();
  }
  else if (norm == 1)
  {
    // ip = bf->l2Norm();
    FunctionPtr ipw = Teuchos::rcp( new IPWeight );
    ip->addTerm( ipw*tau1 );
    ip->addTerm( ipw*tau2 );
    ip->addTerm( ipw*sqrt(1./Re)*v1->grad() );
    ip->addTerm( ipw*sqrt(1./Re)*v2->grad() );
    ip->addTerm( ipw*tau1->div()
                 -0.5*ipw*u1_prev*v1->dx()
                 -0.5*ipw*u2_prev*v1->dy()
                 -0.5*ipw*u1_prev*v1->dx()
                 -0.5*ipw*u2_prev*v2->dx()
                 -ipw*q->dx() );
    ip->addTerm( ipw*tau2->div()
                 -0.5*ipw*u1_prev*v2->dx()
                 -0.5*ipw*u2_prev*v2->dy()
                 -0.5*ipw*u1_prev*v1->dy()
                 -0.5*ipw*u2_prev*v2->dy()
                 -ipw*q->dy() );
    ip->addTerm( ipw*v1->dx() );
    ip->addTerm( ipw*v2->dy() );
    ip->addTerm( ipw*v1 );
    ip->addTerm( ipw*v2 );
    ip->addTerm( ipw*q );
  }

  ////////////////////   CREATE BCs   ///////////////////////
  BCPtr bc = BC::bc();
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  SpatialFilterPtr bottomFree = Teuchos::rcp( new BottomFree );
  SpatialFilterPtr bottomPlate = Teuchos::rcp( new BottomPlate );
  SpatialFilterPtr top = Teuchos::rcp( new ConstantYBoundary(ymax) );
  bc->addDirichlet(u2hat, bottomFree, zero);
  // bc->addDirichlet(t2hat, bottomFree, zero);
  bc->addDirichlet(u1hat, bottomPlate, zero);
  bc->addDirichlet(u2hat, bottomPlate, zero);
  bc->addDirichlet(u1hat, top, one);
  bc->addDirichlet(t2hat, top, zero);
  bc->addDirichlet(u1hat, left, one);
  bc->addDirichlet(u2hat, left, zero);
  // bc->addDirichlet(t1hat, right, zero);

  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  // LinearTermPtr xflux = Teuchos::rcp<LinearTerm>( new LinearTerm(1, t1hat) );
  // LinearTermPtr yflux = Teuchos::rcp<LinearTerm>( new LinearTerm(1, t1hat) );
  // LinearTermPtr pTerm = Teuchos::rcp<LinearTerm>( new LinearTerm(-1, p) );
  // LinearTermPtr uTerm = Teuchos::rcp<LinearTerm>( new LinearTerm(-0.5*u1_prev, u1) );
  vector<double> weight1;
  vector<double> weight2;
  weight1.push_back(1);
  weight1.push_back(0);
  weight2.push_back(0);
  weight2.push_back(1);
  LinearTermPtr sigma1Term = Teuchos::rcp<LinearTerm>( new LinearTerm(weight1, sigma1) );
  // LinearTermPtr sigma2Term = Teuchos::rcp<LinearTerm>( new LinearTerm(weight2, sigma2) );
  // xflux->addTerm(pTerm, true);
  // xflux->addTerm(uTerm, true);
  // // yflux->addTerm(pTerm, true);
  // pc->addConstraint(sigma1Term==zero, right);
  // pc->addConstraint(sigma2Term==zero, top);
  // pc->addConstraint(sigma2Term==zero, bottomFree);
  // pc->addConstraint(yflux==zero, left);
  // pc->addConstraint(t1hat+0.5*u1hat==zero, right);

  // zero mean constraint on pressure
  // bc->addZeroMeanConstraint(p);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  // solution->setFilter(pc);

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  HDF5Exporter exporter(mesh, "FlatPlateConfusion");

  double nonlinearRelativeEnergyTolerance = 1e-5; // used to determine convergence of the nonlinear solution
  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    double L2Update = 1e10;
    int iterCount = 0;
    while (L2Update > nonlinearRelativeEnergyTolerance && iterCount < maxNewtonIterations)
    {
      solution->solve(false);
      double u1L2Update = solution->L2NormOfSolutionGlobal(u1->ID());
      double u2L2Update = solution->L2NormOfSolutionGlobal(u2->ID());
      L2Update = sqrt(u1L2Update*u1L2Update + u2L2Update*u2L2Update);

      // line search algorithm
      double alpha = 1.0;
      // bool useLineSearch = false;
      // int posEnrich = 5; // amount of enriching of grid points on which to ensure positivity
      // if (useLineSearch){ // to enforce positivity of density rho
      //   double lineSearchFactor = .5; double eps = .001; // arbitrary
      //   FunctionPtr rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution) - Function::constant(eps);
      //   FunctionPtr eTemp = Function::solution(e,backgroundFlow) + alpha*Function::solution(e,solution) - Function::constant(eps);
      //   bool rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
      //   bool eIsPositive = eTemp->isPositive(mesh,posEnrich);
      //   int iter = 0; int maxIter = 20;
      //   while (!(rhoIsPositive && eIsPositive) && iter < maxIter){
      //     alpha = alpha*lineSearchFactor;
      //     rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution);
      //     eTemp = Function::solution(e,backgroundFlow) + alpha*Function::solution(e,solution);
      //     rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
      //     eIsPositive = eTemp->isPositive(mesh,posEnrich);
      //     iter++;
      //   }
      //   if (commRank==0 && alpha < 1.0){
      //     cout << "line search factor alpha = " << alpha << endl;
      //   }
      // }

      backgroundFlow->addSolution(solution, alpha, false, true);
      iterCount++;
      if (commRank == 0)
        cout << "L2 Norm of Update = " << L2Update << endl;
    }
    if (commRank == 0)
      cout << endl;

    exporter.exportSolution(backgroundFlow, varFactory, refIndex, 2, cellIDToSubdivision(mesh, 4));

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

