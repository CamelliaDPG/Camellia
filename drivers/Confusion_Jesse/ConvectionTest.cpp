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

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

double pi = 2.0*acos(0.0);

class InflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol); // left inflow
    bool yMatch = (abs(y)<tol); // top/bottom
    return xMatch || yMatch;
  }
};


class Uinflow : public Function {
public:
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    double tol = 1e-11;
    vector<int> cellIDs = basisCache->cellIDs();
    int numPoints = values.dimension(1);
    FieldContainer<double> points = basisCache->getPhysicalCubaturePoints();
    for (int i = 0;i<cellIDs.size();i++){
      for (int j = 0;j<numPoints;j++){
	double x = points(i,j,0);
	double y = points(i,j,1);
	values(i,j) = 0.0;
	if (x<.25 && abs(y)<tol){
	  values(i,j) = 1.0-x/.25;
	}
	if (y<.25 && abs(x)<tol){
	  values(i,j) = 1.0-y/.25;
	}

      }
    }
  }
};

int main(int argc, char *argv[]) {
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
  if ( argc > 1) {
    nCells = atoi(argv[1]);
    if (rank==0){
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
  convectionBF->addTerm( u, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

   // robust test norm
  IPPtr ip = Teuchos::rcp(new IP);
  ip->addTerm(v);
  ip->addTerm(beta*v->grad());
  //  ip->addTerm(v->grad());
  
  ////////////////////   SPECIFY RHS   ///////////////////////

  FunctionPtr zero = Function::constant(0.0);
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  //  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary(beta) );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );

  //  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  FunctionPtr uIn = Teuchos::rcp(new Uinflow);
  FunctionPtr n = Teuchos::rcp(new UnitNormalFunction);
  bc->addDirichlet(beta_n_u, inflowBoundary, beta*n*uIn);  

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 0;
  int H1Order = order+1; int pToAdd = 2;
  
  FieldContainer<double> quadPoints(4,2);
  double squareSize = 1.0;
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = squareSize;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = squareSize;
  quadPoints(2,1) = squareSize;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = squareSize;
 
  int horizontalCells = nCells, verticalCells = nCells;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                convectionBF, H1Order, H1Order+pToAdd);
  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));

  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  /*
  LinearTermPtr conserved = Teuchos::rcp(new LinearTerm(1.0,u));
  LinearTermPtr flux = Teuchos::rcp(new LinearTerm(1.0,beta_n_u));  
  conserved->addTerm(flux,true);
  solution->lagrangeConstraints()->addConstraint( conserved == Function::constant(0.0));
  */
      
  ElementTypePtr elemType = mesh->getElement(0)->elementType();
  BasisCachePtr basisCache = Teuchos::rcp(new BasisCache(elemType, mesh)); 
  vector<int> cellIDs;
  vector< ElementPtr > allElems = mesh->activeElements();
  vector< ElementPtr >::iterator elemIt;
  for (elemIt=allElems.begin();elemIt!=allElems.end();elemIt++){
    cellIDs.push_back((*elemIt)->cellID());
  }
  bool createSideCacheToo = true; 
  basisCache->setPhysicalCellNodes(mesh->physicalCellNodes(elemType), cellIDs, createSideCacheToo);

  int numTrialDofs = elemType->trialOrderPtr->totalDofs();
  cout << "num Trial dofs " << numTrialDofs << endl;
  int numCells = mesh->numElements();
  FieldContainer<double> lhs(numCells,numTrialDofs);
  LinearTermPtr linTerm = Teuchos::rcp(new LinearTerm(1.0,beta_n_u));
  LinearTermPtr field = Teuchos::rcp(new LinearTerm(1.0,u));
  linTerm->addTerm(field,true);
  linTerm->integrate(lhs, elemType->trialOrderPtr, basisCache->getSideBasisCache(0));  
  for (int c = 0;c<numCells;c++){
    for (int i = 0;i<numTrialDofs;i++){
      cout << "linTerm at cell " << c << " and dof " << i << "= " << lhs(c,i) << endl;
    }
  }
 

  double energyThreshold = .2; // for mesh refinements - just to make mesh irregular
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  int numRefs = 0;
  if (rank==0){
    cout << "solving/refining..." << endl;
  }
  for (int i = 0;i<numRefs;i++){
    solution->solve(false);
    refinementStrategy.refine(rank==0); 
  }
  solution->solve(false);  
  FunctionPtr uCopy = Teuchos::rcp( new PreviousSolutionFunction(solution, u) );
  FunctionPtr fnhatCopy = Teuchos::rcp( new PreviousSolutionFunction(solution, beta_n_u));
  
  ////////////////////   get residual   ///////////////////////

  LinearTermPtr residual = Teuchos::rcp(new LinearTerm);// residual 
  residual->addTerm(-fnhatCopy*v + (beta*uCopy)*v->grad() - uCopy*v);
  Teuchos::RCP<RieszRep> riesz = Teuchos::rcp(new RieszRep(mesh, ip, residual));
  riesz->computeRieszRep();
  cout << "riesz error = " << riesz->getNorm() << endl;
  cout << "energy error = " << solution->energyErrorTotal() << endl;
  FunctionPtr rieszRepFxn = Teuchos::rcp(new RepFunction(v,riesz));

  if (rank==0){    
    rieszRepFxn->writeValuesToMATLABFile(mesh,"err_rep.m");
    solution->writeFluxesToFile(beta_n_u->ID(), "fhat.dat");
    solution->writeToVTK("U.vtu",min(H1Order+1,4));
    
    cout << "wrote files: rates.vtu, uhat.dat\n";
  }

  return 0;
}


