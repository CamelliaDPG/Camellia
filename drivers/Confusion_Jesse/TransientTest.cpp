#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "LagrangeConstraints.h"
#include "PreviousSolutionFunction.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

typedef Teuchos::RCP<IP> IPPtr;
typedef Teuchos::RCP<BF> BFPtr;

double pi = 2.0*acos(0.0);

class EpsilonScaling : public hFunction
{
  double _epsilon;
public:
  EpsilonScaling(double epsilon)
  {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h)
  {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class invSqrtHScaling : public hFunction
{
public:
  double value(double x, double y, double h)
  {
    return sqrt(1.0/h);
  }
};

class InflowSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol); // left inflow
    bool topMatch = (abs(y-1.0)<tol );
    bool bottomMatch = (abs(y)<tol );
    bool freeStreamMatch = ((abs(x)<.5) && abs(y)<tol );
    //    return xMatch || freeStreamMatch || topMatch;
    return xMatch || bottomMatch || topMatch;
  }
};

class OutflowSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool wallMatch = (abs(x-1.0)<tol);
    bool plateMatch = ((abs(x)>.5) && abs(y)<tol);
    bool halfWallMatch = (abs(x-1.0)<tol && y<.5);
    return wallMatch;
    //    return halfWallMatch || plateMatch;
    //    return plateMatch;
  }
};

class ZeroStressOutflow : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool halfWallMatch = (abs(x-1.0)<tol && y>=.5);
    return halfWallMatch;
  }
};

class Uinflow : public Function
{
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0; i<cellIDs.size(); i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        double x = points(i,j,0);
        double y = points(i,j,1);
        values(i,j) = 0.0;
        if (y>.25 && y<.75)
        {
          values(i,j) = 1.0;
        }
      }
    }
  }
};

int main(int argc, char *argv[])
{
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif

  int nCells = 2;
  if ( argc > 1)
  {
    nCells = atoi(argv[1]);
    if (rank==0)
    {
      cout << "numCells = " << nCells << endl;
    }
  }

  int numRefs = 0;
  if ( argc > 2)
  {
    numRefs = atoi(argv[2]);
    if (rank==0)
    {
      cout << "numRefs = " << numRefs << endl;
    }
  }

  double eps = 1e-3;
  if ( argc > 3)
  {
    eps = atof(argv[3]);
    if (rank==0)
    {
      cout << "eps = " << eps << endl;
    }
  }

  double dt = .1;
  if (argc > 4)
  {
    dt = atof(argv[4]);
    if (rank==0)
    {
      cout << "dt = " << dt << endl;
    }
  }


  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");

  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(uhat, -tau->dot_normal());

  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( -u, beta * v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 2, pToAdd = 2;

  FieldContainer<double> quadPoints(4,2);

  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;

  int horizontalCells = nCells, verticalCells = nCells;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                            confusionBF, H1Order, H1Order+pToAdd);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);
  qoptIP->addTerm( v );
  qoptIP->addTerm( tau / eps + v->grad() );
  qoptIP->addTerm( beta * v->grad() - tau->div() );

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr ip_scaling = Teuchos::rcp( new EpsilonScaling(eps) );

  robIP->addTerm( v); // no ip scaling?
  robIP->addTerm( sqrt(eps) * v->grad() );
  robIP->addTerm( beta * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( 1.0/sqrt(eps) * tau ); // no ip scaling

  ////////////////////   CREATE BCs   ///////////////////////

  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary);
  SpatialFilterPtr outflowNoWall = Teuchos::rcp(new ZeroStressOutflow);
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  FunctionPtr zero = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );

  //  bc->addDirichlet(uhat, outflowBoundary, zero); // wall BC - constant throughout

  FunctionPtr u0 = Function::constant(0.0);
  FunctionPtr uIn = Teuchos::rcp(new Uinflow);
  bool useRobustBC = true;
  if (useRobustBC)
  {
    bc->addDirichlet(beta_n_u_minus_sigma_n, inflowBoundary, beta*n*uIn);
  }
  else
  {
    bc->addDirichlet(uhat, inflowBoundary, uIn);
  }

  ////////////////////   SPECIFY RHS   ///////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  ////////////////////   SPECIFY TRANSIENT BITS   ///////////////////////

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);

  SolutionPtr prevTimeSoln = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
  bool useTimeSteps = true;
  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(prevTimeSoln, u) );
  if (useTimeSteps)
  {
    map<int, FunctionPtr > functionMap;
    functionMap[u->ID()] = u0;
    prevTimeSoln->projectOntoMesh(functionMap);

    confusionBF->addTerm(u, (1.0/dt) * v); // transient bf term
    rhs->addTerm( (1.0/dt) * u_prev * v );
    robIP->addTerm( (1.0/dt) * v);
    mesh->registerSolution(prevTimeSoln); // u_t(i-1)
  }

  ////////////////////   DEFINE SOLUTION/REFINEMENT STRATEGY   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );

  bool useZeroStressOutflow = false;
  if (useZeroStressOutflow)
  {
    Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
    LinearTermPtr stress = Teuchos::rcp(new LinearTerm(-1.0,beta_n_u_minus_sigma_n));
    cout << "adding stress term w/overriding" << endl;
    stress->addTerm( LinearTerm( beta * n, uhat ), true );
    cout << "adding constraint" << endl;
    pc->addConstraint(stress == zero, outflowNoWall);
    cout << "adding filter" << endl;
    solution->setFilter(pc);
  }
  mesh->registerSolution(solution); // u_t(i)
  LinearTermPtr conserved = Teuchos::rcp(new LinearTerm((1.0/dt),u));
  LinearTermPtr flux = Teuchos::rcp(new LinearTerm(1.0,beta_n_u_minus_sigma_n));
  conserved->addTerm(flux,true);
  solution->lagrangeConstraints()->addConstraint( conserved  == (1.0/dt)*u_prev);
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );

  ////////////////////   SOLVE & REFINE   ///////////////////////

  for (int refIndex=0; refIndex<=numRefs; refIndex++)
  {

    if (useTimeSteps)
    {
      int i = 0;
      double time_tol = 1e-8;

      double u_time_residual = 1e7+time_tol;
      while (u_time_residual > time_tol)
      {
        solution->solve(false);

        // get transient residual
        prevTimeSoln->addSolution(solution,-1.0);
        u_time_residual = (prevTimeSoln->L2NormOfSolutionGlobal(u->ID()))/dt;
        if (rank==0)
        {
          cout << "time residual on timestep " << i << " = " << u_time_residual << endl;
          std::ostringstream oss;
          oss << refIndex << "_" << i;
          std::ostringstream vtu;
          vtu<<".vtu";
          solution->writeToVTK("time"+oss.str()+vtu.str(),min(H1Order+1,4));

        }

        // reset solution
        prevTimeSoln->setSolution(solution); // reset u(t(i-1)) = this u
        i++;
      }
    }
    else
    {
      solution->solve(false);
    }

    if (rank==0)
    {
      std::ostringstream oss;
      oss << refIndex;
      std::ostringstream dat;
      dat<<".dat";
      std::ostringstream vtu;
      vtu<<".vtu";

      solution->writeFluxesToFile(uhat->ID(), "uhatTime" + oss.str()+dat.str());
      solution->writeFluxesToFile(beta_n_u_minus_sigma_n->ID(), "fhatTime"+oss.str()+dat.str());
      solution->writeToVTK("time"+oss.str()+vtu.str(),min(H1Order+1,4));

      cout << "wrote files: time.vtu, uhat.dat\n";
    }
    if (refIndex<numRefs)
    {
      cout << "on refinement number " << refIndex << endl;
      refinementStrategy.refine(rank==0); // print to console on rank 0
    }

  }

  return 0;

}
