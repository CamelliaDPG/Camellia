//  Camellia
//
//  Created by Truman Ellis on 6/4/2012.

#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "RefinementStrategy.h"
#include "SolutionExporter.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

typedef map< int, FunctionPtr > sparseFxnVector;    // dim = {trialID}
typedef map< int, sparseFxnVector > sparseFxnMatrix; // dim = {testID, trialID}

class ZeroTimeBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    return (abs(y) < tol);
  }
};

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

class DiscontinuousInitialCondition : public Function
{
private:
  double xloc;
  double valL;
  double valR;
public:
  DiscontinuousInitialCondition(double xloc, double valL, double valR) : Function(0), xloc(xloc), valL(valL), valR(valR) {}
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
        if (x <= xloc)
          values(cellIndex, ptIndex) = valL;
        else
          values(cellIndex, ptIndex) = valR;
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
  int problem = args.Input<int>("--problem", "which problem to run");
  int numRefs = args.Input<int>("--numRefs", "number of refinement steps");
  int maxNewtonIterations = args.Input<int>("--maxIterations", "maximum number of Newton iterations");

  // Optional arguments (have defaults)
  int polyOrder = args.Input("--polyOrder", "polynomial order for field variables", 2);
  int deltaP = args.Input("--deltaP", "how much to enrich test space", 2);
  int xCells = args.Input("--xCells", "number of cells in the x direction", 32);
  int tCells = args.Input("--tCells", "number of cells in the t direction", 8);
  args.Process();

  ////////////////////   PROBLEM DEFINITIONS   ///////////////////////
  int H1Order = polyOrder+1;

  double xmin, xmax, xint, tmax;
  double GAMMA, rhoL, rhoR, uL, uR, pL, pR, eL, eR;
  string problemName;

  switch (problem)
  {
  case 1:
    // Sod shock tube
    problemName = "Sod";
    xmin = 0;
    xmax = 1;
    xint = 0.5;
    tmax = 0.2;

    GAMMA = 1.4;
    rhoL = 1;
    rhoR = 0.125;
    uL = 0;
    uR = 0;
    pL = 1;
    pR = 0.1;
    eL = pL/(rhoL*(GAMMA-1));
    eR = pR/(rhoR*(GAMMA-1));
    break;
  case 2:
    // Double Rarefaction
    problemName = "DoubleRarefaction";
    xmin = -1;
    xmax = 1;
    xint = 0;
    tmax = 0.2;

    GAMMA = 1.4;
    rhoL = 7;
    rhoR = 7;
    uL = -1;
    uR = 1;
    pL = 0.2;
    pR = 0.2;
    eL = pL/(rhoL*(GAMMA-1));
    eR = pR/(rhoR*(GAMMA-1));
    break;
  case 3:
    // Single Rarefaction
    problemName = "SingleRarefaction";
    xmin = -0.2;
    xmax = 1;
    xint = 0;
    tmax = 0.5;

    GAMMA = 1.4;
    rhoL = 7;
    rhoR = 7;
    uL = 0;
    uR = 1;
    pL = 0.2;
    pR = 0.2;
    eL = pL/(rhoL*(GAMMA-1));
    eR = pR/(rhoR*(GAMMA-1));
    break;
  default:
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "invalid problem number");
  }
  cout << "Running the " << problemName << " problem" << endl;


  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr vm = varFactory.testVar("vm", HGRAD);
  VarPtr vx = varFactory.testVar("vx", HGRAD);
  VarPtr ve = varFactory.testVar("ve", HGRAD);

  // define trial variables
  VarPtr rho = varFactory.fieldVar("rho");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr e = varFactory.fieldVar("e");
  VarPtr Fm = varFactory.fluxVar("Fm");
  VarPtr Fx = varFactory.fluxVar("Fx");
  VarPtr Fe = varFactory.fluxVar("Fe");

  ////////////////////   BUILD MESH   ///////////////////////
  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // define nodes for mesh
  FieldContainer<double> meshBoundary(4,2);

  meshBoundary(0,0) =  xmin; // x1
  meshBoundary(0,1) =  0.0; // y1
  meshBoundary(1,0) =  xmax;
  meshBoundary(1,1) =  0.0;
  meshBoundary(2,0) =  xmax;
  meshBoundary(2,1) =  tmax;
  meshBoundary(3,0) =  xmin;
  meshBoundary(3,1) =  tmax;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(meshBoundary, xCells, tCells,
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

  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u) );
  FunctionPtr rho_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, rho) );
  FunctionPtr e_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, e) );

  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );

  // ==================== SET INITIAL GUESS ==========================
  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[rho->ID()] = Teuchos::rcp( new DiscontinuousInitialCondition(xint, rhoL, rhoR) ) ;
  functionMap[u->ID()]   = Teuchos::rcp( new DiscontinuousInitialCondition(xint, uL, uR) );
  functionMap[e->ID()]   = Teuchos::rcp( new DiscontinuousInitialCondition(xint, eL, eR) );

  backgroundFlow->projectOntoMesh(functionMap);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  // conservation law fluxes
  bf->addTerm(Fm, vm);
  bf->addTerm(Fx, vx);
  bf->addTerm(Fe, ve);

  // Jacobians
  sparseFxnMatrix Jm;
  sparseFxnMatrix Jx;
  sparseFxnMatrix Je;

  map<int, VarPtr> U;
  U[rho->ID()] = rho;
  U[u->ID()] = u;
  U[e->ID()] = e;

  int x_comp = 0;
  int t_comp = 1;

  Jm[x_comp][rho->ID()] = u_prev;
  Jm[x_comp][u->ID()]   = rho_prev;
  Jm[x_comp][e->ID()]   = zero;
  Jm[t_comp][rho->ID()] = one;
  Jm[t_comp][u->ID()]   = zero;
  Jm[t_comp][e->ID()]   = zero;

  Jx[x_comp][rho->ID()] = u_prev*u_prev+(GAMMA-1)*e_prev;
  Jx[x_comp][u->ID()]   = 2*rho_prev*u_prev;
  Jx[x_comp][e->ID()]   = rho_prev*(GAMMA-1);
  Jx[t_comp][rho->ID()] = u_prev;
  Jx[t_comp][u->ID()]   = rho_prev;
  Jx[t_comp][e->ID()]   = zero;

  Je[x_comp][rho->ID()] = GAMMA*e_prev*u_prev+0.5*u_prev*u_prev*u_prev;
  Je[x_comp][u->ID()]   = rho_prev*GAMMA*e_prev+1.5*rho_prev*u_prev*u_prev;
  Je[x_comp][e->ID()]   = rho_prev*GAMMA*u_prev;
  Je[t_comp][rho->ID()] = e_prev+0.5*u_prev*u_prev;
  Je[t_comp][u->ID()]   = rho_prev*u_prev;
  Je[t_comp][e->ID()]   = rho_prev;

  for (int j=0; j < 3; j++)
  {
    bf->addTerm( Jm[x_comp][j]*U[j], -vm->dx() );
    bf->addTerm( Jm[t_comp][j]*U[j], -vm->dy() );
    bf->addTerm( Jx[x_comp][j]*U[j], -vx->dx() );
    bf->addTerm( Jx[t_comp][j]*U[j], -vx->dy() );
    bf->addTerm( Je[x_comp][j]*U[j], -ve->dx() );
    bf->addTerm( Je[t_comp][j]*U[j], -ve->dy() );
  }

  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  rhs->addTerm( (rho_prev*u_prev) * vm->dx() );
  rhs->addTerm( (rho_prev) * vm->dy() );

  rhs->addTerm( (rho_prev*u_prev*u_prev + rho_prev*(GAMMA-1)*e_prev) * vx->dx() );
  rhs->addTerm( (rho_prev*u_prev) * vx->dy() );

  rhs->addTerm( (rho_prev*e_prev*u_prev + 0.5*rho_prev*u_prev*u_prev*u_prev + rho_prev*(GAMMA-1)*e_prev*u_prev) * ve->dx() );
  rhs->addTerm( (rho_prev*e_prev + 0.5*rho_prev*u_prev*u_prev) * ve->dy() );

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  IPPtr ip = bf->graphNorm();

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr timezero = Teuchos::rcp( new ZeroTimeBoundary );
  SpatialFilterPtr left = Teuchos::rcp( new ConstantXBoundary(xmin) );
  SpatialFilterPtr right = Teuchos::rcp( new ConstantXBoundary(xmax) );
  FunctionPtr rho0  = Teuchos::rcp( new DiscontinuousInitialCondition(xint, -rhoL, -rhoR) );
  FunctionPtr mass0 = Teuchos::rcp( new DiscontinuousInitialCondition(xint, -uL*rhoL, -uR*rhoR) );
  FunctionPtr E0    = Teuchos::rcp( new DiscontinuousInitialCondition(xint, -(rhoL*eL+0.5*rhoL*uL*uL), -(rhoR*eR+0.5*rhoR*uR*uR)) );
  bc->addDirichlet(Fm, timezero, rho0);
  bc->addDirichlet(Fx, timezero, mass0);
  bc->addDirichlet(Fe, timezero, E0);
  bc->addDirichlet(Fm, left, -rhoL*uL*one);
  bc->addDirichlet(Fm, right, rhoR*uR*one);
  bc->addDirichlet(Fx, left, -(rhoL*uL*uL+pL)*one);
  bc->addDirichlet(Fx, right, (rhoR*uR*uR+pR)*one);
  bc->addDirichlet(Fe, left, -(rhoL*eL+0.5*rhoL*uL*uL+pL)*uL*one);
  bc->addDirichlet(Fe, right, (rhoR*eR+0.5*rhoR*uR*uR+pR)*uR*one);

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  // ==================== Register Solutions ==========================
  mesh->registerSolution(solution);
  mesh->registerSolution(backgroundFlow);

  ////////////////////   SOLVE & REFINE   ///////////////////////
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  VTKExporter exporter(backgroundFlow, mesh, varFactory);

  double nonlinearRelativeEnergyTolerance = 1e-5; // used to determine convergence of the nonlinear solution
  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {
    double L2Update = 1e7;
    int iterCount = 0;
    while (L2Update > nonlinearRelativeEnergyTolerance && iterCount < maxNewtonIterations)
    {
      solution->solve(false);
      double rhoL2Update = solution->L2NormOfSolutionGlobal(rho->ID());
      double uL2Update = solution->L2NormOfSolutionGlobal(u->ID());
      double eL2Update = solution->L2NormOfSolutionGlobal(e->ID());
      L2Update = sqrt(rhoL2Update*rhoL2Update + uL2Update*uL2Update + eL2Update*eL2Update);

      // line search algorithm
      double alpha = 1.0;
      bool useLineSearch = true;
      int posEnrich = 5; // amount of enriching of grid points on which to ensure positivity
      if (useLineSearch)  // to enforce positivity of density rho
      {
        double lineSearchFactor = .5;
        double eps = .001; // arbitrary
        FunctionPtr rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution) - Function::constant(eps);
        FunctionPtr eTemp = Function::solution(e,backgroundFlow) + alpha*Function::solution(e,solution) - Function::constant(eps);
        bool rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
        bool eIsPositive = eTemp->isPositive(mesh,posEnrich);
        int iter = 0;
        int maxIter = 20;
        while (!(rhoIsPositive && eIsPositive) && iter < maxIter)
        {
          alpha = alpha*lineSearchFactor;
          rhoTemp = Function::solution(rho,backgroundFlow) + alpha*Function::solution(rho,solution);
          eTemp = Function::solution(e,backgroundFlow) + alpha*Function::solution(e,solution);
          rhoIsPositive = rhoTemp->isPositive(mesh,posEnrich);
          eIsPositive = eTemp->isPositive(mesh,posEnrich);
          iter++;
        }
        if (commRank==0 && alpha < 1.0)
        {
          cout << "line search factor alpha = " << alpha << endl;
        }
      }

      backgroundFlow->addSolution(solution, alpha);
      iterCount++;
      if (commRank == 0)
        cout << "L2 Norm of Update = " << L2Update << endl;
    }
    if (commRank == 0)
      cout << endl;

    if (commRank == 0)
    {
      stringstream outfile;
      outfile << problemName << "_" << refIndex;
      exporter.exportSolution(outfile.str());
    }

    if (refIndex < numRefs)
      refinementStrategy.refine(commRank==0); // print to console on commRank 0
  }

  return 0;
}

