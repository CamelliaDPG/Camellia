#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"
#include "CondensationSolver.h"
#include "ZoltanMeshPartitionPolicy.h"
#include "LagrangeConstraints.h"
#include "RieszRep.h"
#include "BasisFactory.h" // for test
#include "HessianFilter.h"
#include "MeshUtilities.h"
#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

double pi = 2.0*acos(0.0);

class InflowSquareBoundary : public SpatialFilter
{
public:
  bool matchesPoint(double x, double y)
  {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol); // left inflow
    bool yMatch = (abs(y)<tol); // top/bottom
    return xMatch || yMatch;
  }
};


class Uinflow : public Function
{
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    double tol = 1e-11;
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
        if (abs(y)<tol)
        {
          values(i,j) = 1.0-x;
        }
        if (abs(x)<tol)
        {
          values(i,j) = 1.0-y;
        }

      }
    }
  }
};

class LinearOrthogPoly : public Function
{
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    double tol = 1e-11;
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0; i<cellIDs.size(); i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        double x = points(i,j,0);
        double y = points(i,j,1);
        if (abs(y)<tol || abs(y-1.0)<tol)  // if on top or bottom
        {
          //	  values(i,j) = (6*x*x-6*x+1);
          values(i,j) = 20*x*x*x - 30*x*x + 12 * x - 1;
        }
        else if (abs(x)<tol || abs(x-1.0)<tol)   // if on sides
        {
          //	  values(i,j) = (6*y*y-6*y+1);
          values(i,j) = 20*y*y*y - 30*y*y + 12*y - 1;
        }
      }
    }
  }
};

class EdgeFunction : public Function
{
public:
  bool boundaryValueOnly()
  {
    return true;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache)
  {
    double tol = 1e-11;
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0; i<cellIDs.size(); i++)
    {
      for (int j = 0; j<numPoints; j++)
      {
        double x = points(i,j,0);
        double y = points(i,j,1);
        values(i,j) = x*y+1.0;
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
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  int rank = 0;
  int numProcs = 1;
  Epetra_SerialComm Comm;
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

  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr v = varFactory.testVar("v", HGRAD);

  // define trial variables
  VarPtr beta_n_u = varFactory.fluxVar("\\widehat{\\beta \\cdot n }");
  VarPtr u = varFactory.fieldVar("u");

  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(1.0);

  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////

  BFPtr convectionBF = Teuchos::rcp( new BF(varFactory) );

  // v terms:
  convectionBF->addTerm( -u, beta * v->grad() );
  convectionBF->addTerm( beta_n_u, v);
  //  convectionBF->addTerm( u, v);

  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());
  //  ip->addTerm(v->grad());

  ////////////////////   SPECIFY RHS   ///////////////////////

  FunctionPtr zero = Function::constant(0.0);
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );

  ////////////////////   CREATE BCs   ///////////////////////

  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );

  FunctionPtr uIn = Teuchos::rcp(new Uinflow);
  FunctionPtr n = Teuchos::rcp(new UnitNormalFunction);
  bc->addDirichlet(beta_n_u, inflowBoundary, beta*n*uIn);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 8;
  int H1Order = order+1;
  int pToAdd = 2;

  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(2,1, convectionBF, H1Order, H1Order+pToAdd);

  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  double energyThreshold = .2; // for mesh refinements - just to make mesh irregular
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  int numRefs = 0;
  if (rank==0)
  {
    cout << "solving/refining..." << endl;
  }
  for (int i = 0; i<numRefs; i++)
  {
    solution->solve(false);
    refinementStrategy.refine(rank==0);
  }
  solution->solve(false);
  FunctionPtr uCopy = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  FunctionPtr fnhatCopy = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u));

  ////////////////////   get residual   ///////////////////////

  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  residual->addTerm(-fnhatCopy*v + (beta*uCopy)*v->grad());

  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  riesz->computeRieszRep();
  cout << "riesz error = " << riesz->getNorm() << endl;
  cout << "energy error = " << solution->energyErrorTotal() << endl;
  FunctionPtr rieszRepFxn = Teuchos::rcp(new RepFunction(v,riesz));

  map<int,FunctionPtr> err_rep_map;
  err_rep_map[v->ID()] = rieszRepFxn;

  FunctionPtr edgeResidual = residual->evaluate(err_rep_map, true) ;
  FunctionPtr elemResidual = residual->evaluate(err_rep_map, false);

  LinearTermPtr edgeOnlyRes = Teuchos::rcp(new LinearTerm);// residual
  edgeOnlyRes->addTerm(-fnhatCopy*v);
  FunctionPtr edgeOnlyFxn = edgeOnlyRes->evaluate(err_rep_map,true);

  double edgeRes = edgeResidual->integrate(mesh,10);
  double elemRes = elemResidual->integrate(mesh,10);
  double edgeOnlyResVal = edgeOnlyFxn->integrate(mesh,10);
  cout << "residual eval'd at edge = " << edgeRes << ", vs edge only residual " << edgeOnlyResVal << endl;
  //  cout << "eleme residual = " << elemRes << " and total residual = " << elemRes + edgeRes << endl;

  BCPtr nullBC = Teuchos::rcp((BC*)NULL);
  RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL);
  IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC,
                                 nullRHS, nullIP) );
  int numGlobalDofs = mesh->numGlobalDofs();
  for (int dofIndex = 0; dofIndex<numGlobalDofs; dofIndex++)
  {
    if (solution->isFluxOrTraceDof(dofIndex))
    {
      solnPerturbation->clearSolution(); // clear all solns
      solnPerturbation->setSolnCoeffForGlobalDofIndex(1.0,dofIndex);
      LinearTermPtr b_u =  convectionBF->testFunctional(solnPerturbation);
      map<int,FunctionPtr> NL_err_rep_map;
      NL_err_rep_map[v->ID()] = Teuchos::rcp(new RepFunction(v,riesz));
// NL_err_rep_map[v->ID()] = Teuchos::rcp(new LinearOrthogPoly);
      FunctionPtr b_e = b_u->evaluate(NL_err_rep_map, solution->isFluxOrTraceDof(dofIndex)); // use boundary part only if flux or trace
      double be_int = b_e->integrate(mesh,10);
      cout << "bilinear form evaluated at flux dof/error test = " << be_int << endl;
    }
  }


  if (rank==0)
  {
    rieszRepFxn->writeValuesToMATLABFile(mesh,"err_rep.m");
    solution->writeFluxesToFile(beta_n_u->ID(), "fhat.dat");
    solution->writeToVTK("U.vtu",min(H1Order+1,4));

    cout << "wrote files: rates.vtu, uhat.dat\n";
  }

  return 0;
}


