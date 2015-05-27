#include "OptimalInnerProduct.h"
#include "Mesh.h"
#include "Solution.h"

#include "RefinementStrategy.h"
#include "NonlinearStepSize.h"
#include "NonlinearSolveStrategy.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

#include "InnerProductScratchPad.h"
#include "PreviousSolutionFunction.h"
#include "RefinementPattern.h"
#include "RieszRep.h"
#include "HessianFilter.h"

#include "TestingUtilities.h"
#include "FiniteDifferenceUtilities.h"
#include "MeshUtilities.h"

#include "SolutionExporter.h"

#include <sstream>

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;

class PositivePart : public Function
{
  FunctionPtr _f;
public:
  PositivePart(FunctionPtr f)
  {
    _f = f;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints);
    _f->values(values,basisCache);

    for (int i = 0; i<numCells; i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        if (values(i,j)<0)
        {
          values(i,j) = 0.0;
        }
      }
    }
  }
};

class U0 : public SimpleFunction
{
public:
  double value(double x, double y)
  {
    return 1 - 2 * x;
  }
};

class TopBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool yMatch = (abs(y-1.0) < tol);
    return yMatch;
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
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  int nCells = args.Input<int>("--nCells", "num cells",2);
  int numSteps = args.Input<int>("--numSteps", "num NR steps",20);

  int polyOrder = 0;

  // define our manufactured solution or problem bilinear form:
  bool useTriangles = false;

  int pToAdd = 1;

  args.Process();

  int H1Order = polyOrder + 1;

  ////////////////////////////////////////////////////////////////////
  // DEFINE VARIABLES
  ////////////////////////////////////////////////////////////////////

  // new-style bilinear form definition
  VarFactory varFactory;
  VarPtr fn = varFactory.fluxVar("\\widehat{\\beta_n_u}");
  VarPtr u = varFactory.fieldVar("u");

  VarPtr v = varFactory.testVar("v",HGRAD);
  BFPtr bf = Teuchos::rcp( new BF(varFactory) ); // initialize bilinear form

  ////////////////////////////////////////////////////////////////////
  // CREATE MESH
  ////////////////////////////////////////////////////////////////////

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells , bf, H1Order, H1Order+pToAdd);

  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );

  vector<double> e1(2),e2(2);
  e1[0] = 1;
  e2[1] = 1;

  FunctionPtr u_prev = Teuchos::rcp( new PreviousSolutionFunction(backgroundFlow, u) );
  FunctionPtr beta = e1 * u_prev + Teuchos::rcp( new ConstantVectorFunction( e2 ) );

  ////////////////////////////////////////////////////////////////////
  // DEFINE BILINEAR FORM
  ////////////////////////////////////////////////////////////////////

  // v:
  bf->addTerm( -u, beta * v->grad());
  bf->addTerm( fn, v);

  ////////////////////////////////////////////////////////////////////
  // DEFINE RHS
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr u_prev_squared_div2 = 0.5 * u_prev * u_prev;
  rhs->addTerm((e1 * u_prev_squared_div2 + e2 * u_prev) * v->grad());

  // ==================== SET INITIAL GUESS ==========================

  mesh->registerSolution(backgroundFlow);
  FunctionPtr zero = Function::constant(0.0);
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  //  FunctionPtr parity = Teuchos::rcp(new SideParityFunction);

  FunctionPtr u0_squared_div_2 = 0.5 * u0 * u0;

  map<int, Teuchos::RCP<Function> > functionMap;
  functionMap[u->ID()] = u0;
  //  functionMap[fn->ID()] = -(e1 * u0_squared_div_2 + e2 * u0) * n * parity;
  backgroundFlow->projectOntoMesh(functionMap);

  // ==================== END SET INITIAL GUESS ==========================

  ////////////////////////////////////////////////////////////////////
  // DEFINE INNER PRODUCT
  ////////////////////////////////////////////////////////////////////

  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm( v );
  ip->addTerm(v->grad());
  //  ip->addTerm( beta * v->grad() ); // omitting term to make IP non-dependent on u

  ////////////////////////////////////////////////////////////////////
  // DEFINE DIRICHLET BC
  ////////////////////////////////////////////////////////////////////

  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new TopBoundary);
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new NegatedSpatialFilter(outflowBoundary) );
  Teuchos::RCP<BCEasy> inflowBC = Teuchos::rcp( new BCEasy );
  inflowBC->addDirichlet(fn,inflowBoundary,
                         ( e1 * u0_squared_div_2 + e2 * u0) * n );

  ////////////////////////////////////////////////////////////////////
  // CREATE SOLUTION OBJECT
  ////////////////////////////////////////////////////////////////////

  Teuchos::RCP<Solution> solution = Teuchos::rcp(new Solution(mesh, inflowBC, rhs, ip));
  mesh->registerSolution(solution);
  solution->setCubatureEnrichmentDegree(10);

  ////////////////////////////////////////////////////////////////////
  // HESSIAN BIT + CHECKS ON GRADIENT + HESSIAN
  ////////////////////////////////////////////////////////////////////

  VarFactory hessianVars = varFactory.getBubnovFactory(VarFactory::BUBNOV_TRIAL);
  VarPtr du = hessianVars.test(u->ID());
  //  BFPtr hessianBF = Teuchos::rcp( new BF(hessianVars) ); // initialize bilinear form

  FunctionPtr du_current  = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );

  FunctionPtr fnhat = Teuchos::rcp(new PreviousSolutionFunction(solution,fn));
  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  residual->addTerm(fnhat*v,true);
  residual->addTerm( - (e1 * (u_prev_squared_div2) + e2 * (u_prev)) * v->grad(),true);

  LinearTermPtr Bdu = Teuchos::rcp(new LinearTerm);// residual
  Bdu->addTerm( - du_current*(beta*v->grad()));

  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  Teuchos::RCP<RieszRep> duRiesz = Teuchos::rcp(new RieszRep(mesh, ip, Bdu));
  riesz->computeRieszRep();
  FunctionPtr e_v = Teuchos::rcp(new RepFunction(v,riesz));
  e_v->writeValuesToMATLABFile(mesh, "e_v.m");
  FunctionPtr posErrPart = Teuchos::rcp(new PositivePart(e_v->dx()));
  //  hessianBF->addTerm(e_v->dx()*u,du);
  //  hessianBF->addTerm(posErrPart*u,du);
  //  Teuchos::RCP<NullFilter> nullFilter = Teuchos::rcp(new NullFilter);
  //  Teuchos::RCP<HessianFilter> hessianFilter = Teuchos::rcp(new HessianFilter(hessianBF));

  Teuchos::RCP< LineSearchStep > LS_Step = Teuchos::rcp(new LineSearchStep(riesz));

  double NL_residual = 9e99;
  for (int i = 0; i<numSteps; i++)
  {
    // write matrix to file and then resollve without hessian
    /*
    solution->setFilter(hessianFilter);
    stringstream oss;
    oss << "hessianMatrix" << i << ".dat";
    solution->setWriteMatrixToFile(true,oss.str());
    solution->solve(false);

    solution->setFilter(nullFilter);
    oss.str(""); // clear
    oss << "stiffnessMatrix" << i << ".dat";
    solution->setWriteMatrixToFile(false,oss.str());
    */

    solution->solve(false); // do one solve to initialize things...
    double stepLength = 1.0;
    stepLength = LS_Step->stepSize(backgroundFlow,solution, NL_residual);

    //      solution->setWriteMatrixToFile(true,"stiffness.dat");

    backgroundFlow->addSolution(solution,stepLength);
    NL_residual = LS_Step->getNLResidual();
    if (rank==0)
    {
      cout << "NL residual after adding = " << NL_residual << " with step size " << stepLength << endl;
    }

    double fd_gradient;
    for (int dofIndex = 0; dofIndex<mesh->numGlobalDofs(); dofIndex++)
    {
      TestingUtilities::initializeSolnCoeffs(solnPerturbation);
      TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,1.0,dofIndex);
      fd_gradient = FiniteDifferenceUtilities::finiteDifferenceGradient(mesh, riesz, backgroundFlow, dofIndex);

      // CHECK GRADIENT
      LinearTermPtr b_u =  bf->testFunctional(solnPerturbation);
      map<int,FunctionPtr> NL_err_rep_map;

      NL_err_rep_map[v->ID()] = Teuchos::rcp(new RepFunction(v,riesz));
      FunctionPtr gradient = b_u->evaluate(NL_err_rep_map, TestingUtilities::isFluxOrTraceDof(mesh,dofIndex)); // use boundary part only if flux or trace
      double grad;
      if (TestingUtilities::isFluxOrTraceDof(mesh,dofIndex))
      {
        grad = gradient->integralOfJump(mesh,10);
      }
      else
      {
        grad = gradient->integrate(mesh,10);
      }
      double fdgrad = fd_gradient;
      double diff = grad-fdgrad;
      if (abs(diff)>1e-6 && i>0)
      {
        cout << "Found difference of " << diff << ", " << " with fd val = " << fdgrad << " and gradient = " << grad << " in dof " << dofIndex << ", isTraceDof = " << TestingUtilities::isFluxOrTraceDof(mesh,dofIndex) << endl;
      }
    }
  }

  VTKExporter exporter(solution, mesh, varFactory);
  if (rank==0)
  {
    exporter.exportSolution("qopt");
    cout << endl;
  }

  return 0;
}

