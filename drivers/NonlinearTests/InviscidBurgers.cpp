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

#include "TestingUtilities.h"
#include "FiniteDifferenceUtilities.h" 
#include "MeshUtilities.h"

#include "SolutionExporter.h"

#include "StandardAssembler.h"
#include "Epetra_Vector.h"
#include "Solver.h"

#include <sstream>

typedef Teuchos::RCP<shards::CellTopology> CellTopoPtr;

class PositivePart : public Function {
  FunctionPtr _f;
public:
  PositivePart(FunctionPtr f) {
    _f = f;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    FieldContainer<double> beta_pts(numCells,numPoints);
    _f->values(values,basisCache);
    
    for (int i = 0;i<numCells;i++){
      for (int j = 0;j<numPoints;j++){
	if (values(i,j)<0){
	  values(i,j) = 0.0;
	}
      }
    }
  }
};

class U0 : public SimpleFunction {
public:
  double value(double x, double y) {
    return 1 - 2 * x;
  }
};

class TopBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool yMatch = (abs(y-1.0) < tol);
    return yMatch;
  }
};

int main(int argc, char *argv[]) {

#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  choice::MpiArgs args( argc, argv );
#else
  choice::Args args( argc, argv );
#endif
  int rank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();

  bool useTriangles = false;  

  int nCells = args.Input<int>("--nCells", "num cells",2);
  int numSteps = args.Input<int>("--numSteps", "num NR steps",20);

  int polyOrder = 1; 
  int pToAdd = 1;

  args.Process();
 
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
  
  int H1Order = polyOrder + 1;
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells , bf, H1Order, H1Order+pToAdd);
  
  ////////////////////////////////////////////////////////////////////
  // INITIALIZE BACKGROUND FLOW FUNCTIONS
  ////////////////////////////////////////////////////////////////////
  BCPtr nullBC = Teuchos::rcp((BC*)NULL); RHSPtr nullRHS = Teuchos::rcp((RHS*)NULL); IPPtr nullIP = Teuchos::rcp((IP*)NULL);
  SolutionPtr backgroundFlow = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
  SolutionPtr solnPerturbation = Teuchos::rcp(new Solution(mesh, nullBC, nullRHS, nullIP) );
  
  vector<double> e1(2),e2(2);
  e1[0] = 1; e2[1] = 1;
  
  FunctionPtr u_prev = Function::solution(u,backgroundFlow);
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
  // CHECKS ON GRADIENT 
  ////////////////////////////////////////////////////////////////////

  FunctionPtr fnhat = Function::solution(fn,solution);

  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual
  residual->addTerm(fnhat*v,true);
  residual->addTerm(-(e1 * (u_prev_squared_div2) + e2 * (u_prev)) * v->grad(),true);

  Teuchos::RCP<RieszRep> rieszResidual = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  rieszResidual->computeRieszRep();

  Teuchos::RCP< LineSearchStep > LS_Step = Teuchos::rcp(new LineSearchStep(rieszResidual));

  set<int> myGlobalIndicesSet = mesh->globalDofIndicesForPartition(rank);
  FieldContainer<int> bcGlobalIndices;
  FieldContainer<double> bcGlobalValues;
  mesh->boundary().bcsToImpose(bcGlobalIndices,bcGlobalValues,*(solution->bc()), myGlobalIndicesSet);
  set<int> bcInds;
  for (int i=0;i<bcGlobalIndices.dimension(0);i++){
    bcInds.insert(bcGlobalIndices(i));
  }

  double NL_residual = 9e99;
  for (int i = 0;i<numSteps;i++){

    Teuchos::RCP<StandardAssembler> assembler = Teuchos::rcp(new StandardAssembler(solution));
    Epetra_FECrsMatrix K = assembler->initializeMatrix();
    Epetra_FEVector b = assembler->initializeVector();
    Epetra_FEVector x = assembler->initializeVector();
    assembler->assembleProblem(K,b);  
    Teuchos::RCP<Epetra_LinearProblem> problem = Teuchos::rcp(new Epetra_LinearProblem(&K,&x,&b));
    Teuchos::RCP<Solver> solver = Teuchos::rcp(new KluSolver());    
    solver->setProblem(problem);
    solver->solve();
    assembler->distributeDofs(x);

    double stepLength = 1.0;
    stepLength = LS_Step->stepSize(backgroundFlow,solution, NL_residual);

    backgroundFlow->addSolution(solution,stepLength);
    NL_residual = LS_Step->getNLResidual();
    if (rank==0){
      cout << "NL residual after adding = " << NL_residual << " with step size " << stepLength << endl;    
    }
 
   // test FIELD DOFS
    double fd_gradient;
    for (int dofIndex = 0;dofIndex<mesh->numGlobalDofs();dofIndex++){
      bool isBCIndex = bcInds.find(dofIndex) != bcInds.end();      

      // preset BCs in all solution perturbations
      TestingUtilities::initializeSolnCoeffs(solnPerturbation);
      for (int i = 0;i<bcGlobalIndices.dimension(0);i++){
	TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,bcGlobalValues(i),bcGlobalIndices(i));
      }
      if (!isBCIndex){	
	TestingUtilities::setSolnCoeffForGlobalDofIndex(solnPerturbation,1.0,dofIndex);
      }
      
      fd_gradient = FiniteDifferenceUtilities::finiteDifferenceGradient(mesh, rieszResidual, backgroundFlow, dofIndex);
      
      // CHECK RHS
      if (!isBCIndex){
	cout << "fd gradient = " << fd_gradient << ", while b[i] = " << (*b(0))[dofIndex] << endl;
      }
    }    
    cout << endl;
  }
  
  VTKExporter exporter(solution, mesh, varFactory);
  if (rank==0){
    exporter.exportSolution("inviscidBurgers");
    cout << endl;
  }

  return 0;
}

