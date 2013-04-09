#include "Solver.h"
#include "Amesos.h"
#include "Amesos_Utils.h"

#include "SolutionExporter.h"
#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"
#include "PreviousSolutionFunction.h"
#include "ZoltanMeshPartitionPolicy.h"

#include "RieszRep.h"
#include "BasisFactory.h" // for test
#include "HessianFilter.h"

#include "MeshUtilities.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#include "mpi_choice.hpp"
#else
#include "choice.hpp"
#endif

#include "Epetra_LinearProblem.h"
#include "EpetraExt_RowMatrixOut.h"
#include "EpetraExt_MultiVectorOut.h"
#include "Epetra_FECrsMatrix.h"
#include "Epetra_FEVector.h"
#include "StandardAssembler.h" // for system assembly
#include "SerialDenseWrapper.h" // for system assembly
#include "TestingUtilities.h" 
#include "MeshPolyOrderFunction.h"

double pi = 2.0*acos(0.0);

class EpsilonScaling : public hFunction {
  double _epsilon;
public:
  EpsilonScaling(double epsilon) {
    _epsilon = epsilon;
  }
  double value(double x, double y, double h) {
    // should probably by sqrt(_epsilon/h) instead (note parentheses)
    // but this is what was in the old code, so sticking with it for now.
    double scaling = min(_epsilon/(h*h), 1.0);
    // since this is used in inner product term a like (a,a), take square root
    return sqrt(scaling);
  }
};

class AnisotropicHScaling : public Function {
  int _spatialCoord;
public:
  AnisotropicHScaling(int spatialCoord){
    _spatialCoord = spatialCoord;
  }
  void values(FieldContainer<double> &values, BasisCachePtr basisCache){
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);

    MeshPtr mesh = basisCache->mesh();
    vector<int> cellIDs = basisCache->cellIDs();

    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      double h = 1.0;
      if (_spatialCoord==0){
	h = mesh->getCellXSize(cellIDs[cellIndex]);
      }else if (_spatialCoord==1){
	h = mesh->getCellYSize(cellIDs[cellIndex]);
      }	
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
	values(cellIndex,ptIndex) = sqrt(h);
      }
    }
  }
};

class SqrtHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return sqrt(h);
  }
};
class InvSqrtHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return sqrt(1.0/h);
  }
};
class InvHScaling : public hFunction {
public:
  double value(double x, double y, double h) {
    return 1.0/h;
  }
};

class InflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x)<tol); // x = 0 is inflow
    return xMatch;
  }
};

class WallBoundary : public SpatialFilter{
public: 
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool topWall = (x>1.0) && (abs(y-1.0)<tol);
    bool faceWall = (y<1.0) && (abs(x-1.0)<tol);
    //    return topWall || faceWall;
    return topWall;
  }
};

class FreeStreamBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool bottom = (abs(y)<tol) && (x < 1.0);
    bool top = abs(y-2.0)<tol;
    return top || bottom;
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
  
  int numRefs = args.Input<int>("--numRefs","num adaptive refinements",0);
  double eps = args.Input<double>("--epsilon","diffusion parameter",1e-2);
  if (rank==0){
    cout << "numrefs  = " << numRefs << endl;
    cout << "eps = " << eps << endl;
  }

  FunctionPtr zero = Function::constant(0.0);
  FunctionPtr one = Function::constant(1.0);
  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );
  vector<double> e1,e2;
  e1.push_back(1.0);e1.push_back(0.0);
  e2.push_back(0.0);e2.push_back(1.0);

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

  // first order term with magnitude alpha
  double alpha = 0.0;
  confusionBF->addTerm(alpha * u, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

   // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);
  FunctionPtr C_h = Teuchos::rcp( new EpsilonScaling(eps) );  
  FunctionPtr invH = Teuchos::rcp(new InvHScaling);
  FunctionPtr invSqrtH = Teuchos::rcp(new InvSqrtHScaling);
  FunctionPtr sqrtH = Teuchos::rcp(new SqrtHScaling);
  robIP->addTerm(v*alpha);
  robIP->addTerm(invSqrtH*v);
  //  robIP->addTerm(v);
  robIP->addTerm(sqrt(eps) * v->grad() );
  robIP->addTerm(sqrtH * beta * v->grad() );
  robIP->addTerm(tau->div() );
  robIP->addTerm(C_h/sqrt(eps) * tau );  

  LinearTermPtr vVecLT = Teuchos::rcp(new LinearTerm);
  LinearTermPtr tauVecLT = Teuchos::rcp(new LinearTerm);
  vVecLT->addTerm(sqrt(eps)*v->grad());
  tauVecLT->addTerm(C_h/sqrt(eps)*tau);

  ////////////////////   SPECIFY RHS   ///////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  //  f = one;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );

  SpatialFilterPtr inflow = Teuchos::rcp( new InflowBoundary);
  SpatialFilterPtr wall = Teuchos::rcp( new WallBoundary);
  SpatialFilterPtr freeStream = Teuchos::rcp( new FreeStreamBoundary);
  
  //  bc->addDirichlet(uhat, halfWallOutflow, one);
  bc->addDirichlet(uhat, wall, one);
  bc->addDirichlet(beta_n_u_minus_sigma_n, inflow, zero);
  bc->addDirichlet(beta_n_u_minus_sigma_n, freeStream, zero);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int order = 2;
  int H1Order = order+1; int pToAdd = 2;
  
  // create a pointer to a new mesh:
  MeshPtr mesh = MeshUtilities::buildFrontFacingStep(confusionBF, H1Order, H1Order+pToAdd);
  //  mesh->setPartitionPolicy(Teuchos::rcp(new ZoltanMeshPartitionPolicy("HSFC")));  
  
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  //  solution->solve(false);
  solution->condensedSolve();
  
  LinearTermPtr residual = rhs->linearTermCopy();
  residual->addTerm(-confusionBF->testFunctional(solution));  
  RieszRepPtr rieszResidual = Teuchos::rcp(new RieszRep(mesh, robIP, residual));
  rieszResidual->computeRieszRep();
  FunctionPtr e_v = Teuchos::rcp(new RepFunction(v,rieszResidual));
  FunctionPtr e_tau = Teuchos::rcp(new RepFunction(tau,rieszResidual));
  map<int,FunctionPtr> errRepMap;
  errRepMap[v->ID()] = e_v;
  errRepMap[tau->ID()] = e_tau;
  FunctionPtr errTau = tauVecLT->evaluate(errRepMap,false);
  FunctionPtr errV = vVecLT->evaluate(errRepMap,false);
  FunctionPtr xErr = (errTau->x())*(errTau->x()) + (errV->dx())*(errV->dx());
  FunctionPtr yErr = (errTau->y())*(errTau->y()) + (errV->dy())*(errV->dy());

  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );  

  for (int refIndex=0;refIndex<numRefs;refIndex++){
    if (rank==0){
      cout << "on ref index " << refIndex << endl;
    }    
    rieszResidual->computeRieszRep(); // in preparation to get anisotropy    

    vector<int> cellIDs;
    refinementStrategy.getCellsAboveErrorThreshhold(cellIDs);
    map<int,double> xErrMap = xErr->cellIntegrals(cellIDs,mesh,15,true);
    map<int,double> yErrMap = yErr->cellIntegrals(cellIDs,mesh,15,true);
    //    refinementStrategy.refine(rank==0,xErrMap,yErrMap); //anisotropic refinements
    refinementStrategy.refine(rank==0); // no anisotropy
    solution->condensedSolve();
  }

  ////////////////////   print to file   ///////////////////////

  FunctionPtr orderFxn = Teuchos::rcp(new MeshPolyOrderFunction(mesh));
  VTKExporter exporter(solution, mesh, varFactory);
  std::ostringstream oss;
  oss << numRefs;
  if (rank==0){
    exporter.exportSolution(string("robustIP")+oss.str());
    exporter.exportFunction(orderFxn, "meshOrder");
    cout << endl;
  }
 
  return 0;
} 


