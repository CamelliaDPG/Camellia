//  NewConfusionDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"

#include "Solution.h"

#include "PreviousSolutionFunction.h"

#include "MeshUtilities.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

class EntireBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    return true;
  }
};

class UnitSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) || (abs(x-1.0) < tol);
    bool yMatch = (abs(y) < tol) || (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

class InflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x) < tol) ;
    bool yMatch = (abs(y) < tol) ;
    return xMatch || yMatch;
  }
};

class OutflowSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x-1.0) < tol);
    bool yMatch = (abs(y-1.0) < tol);
    return xMatch || yMatch;
  }
};

class Beta : public Function {
public:
  Beta() : Function(1) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    int spaceDim = values.dimension(2);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        for (int d = 0; d < spaceDim; d++) {
          double x = (*points)(cellIndex,ptIndex,0);
          double y = (*points)(cellIndex,ptIndex,1);
          values(cellIndex,ptIndex,0) = y;
          values(cellIndex,ptIndex,0) = -x;
        }
      }
    }
  }
};

string fileNameForRefinement(string fileName, int refinementNumber) {
  ostringstream fileNameStream;
  fileNameStream << fileName << "_r" << refinementNumber << ".dat";
  return fileNameStream.str();
}

int main(int argc, char *argv[]) {
#ifdef HAVE_MPI
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);
  int rank=mpiSession.getRank();
  int numProcs=mpiSession.getNProc();
#else
  int rank = 0;
  int numProcs = 1;
#endif
  bool enforceOneIrregularity = true;
  bool writeStiffnessMatrices = true;
  bool writeWorstCaseGramMatrices = true;
  int numRefs = 3;
  
  int H1Order = 3, pToAdd = 2;
  int horizontalCells = 1, verticalCells = 1;
  
  // problem parameters:
  double eps = 1e-4;
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
  FunctionPtr beta = Function::constant(beta_const);
  
  if (rank==0) {
    cout << "eps = " << eps << endl;
    cout << "numRefs = " << numRefs << endl;
    cout << "p = " << H1Order-1 << endl;
  }
  
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr sigma_n = varFactory.fluxVar("\\widehat{\\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  vector< VarPtr > nonSigmaTrials;
  vector< VarPtr > sigmaTrials;
  nonSigmaTrials.push_back(uhat);
  nonSigmaTrials.push_back(u);
  
  sigmaTrials.push_back(sigma1);
  sigmaTrials.push_back(sigma2);
  sigmaTrials.push_back(sigma_n);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());
  
  FunctionPtr n = Function::normal();
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( beta * u, - v->grad() );
  confusionBF->addTerm( beta * n * uhat, v);
  confusionBF->addTerm( - sigma_n, v);
  
  /////////////////// DEFINE OPERATOR SPLIT /////////////
  // set up var factory with compatible IDs:
  VarFactory sigmaVarFactory = varFactory.trialSubFactory(sigmaTrials);
  VarFactory nonSigmaVarFactory = varFactory.trialSubFactory(nonSigmaTrials);
  
  // bf1 gets the sigma terms
  BFPtr bf1 = Teuchos::rcp( new BF(sigmaVarFactory) );
  bf1->addTerm( sigma1, tau->x() );
  bf1->addTerm( sigma2, tau->y() );
  bf1->addTerm( sigma1, v->dx() );
  bf1->addTerm( sigma2, v->dy() );
  bf1->addTerm( -sigma_n, v);
  
  // bf2 gets everything else:
  BFPtr bf2 = Teuchos::rcp( new BF(nonSigmaVarFactory) );
  bf2->addTerm(u, tau->div());
  bf2->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  bf2->addTerm( beta_const * u, - v->grad() );
  bf2->addTerm(beta * n * uhat, v);
  
  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  FieldContainer<double> quadPoints(4,2);
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  Teuchos::RCP<Mesh> mesh1 = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                 bf1, H1Order, H1Order+pToAdd);
  Teuchos::RCP<Mesh> mesh2 = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                 bf2, H1Order, H1Order+pToAdd);
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);
  // refine the split meshes in tandem with the combined mesh:
  mesh->registerObserver(mesh1);
  mesh->registerObserver(mesh2);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  
  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);
  qoptIP->addTerm( tau / eps + v->grad() );
  qoptIP->addTerm( beta_const * v->grad() - tau->div() );
  qoptIP->addTerm( v );
  
  IPPtr ip1 = bf1->graphNorm();
  IPPtr ip2 = bf2->graphNorm();
  
  /////////// DEFINE PREVIOUS SOLUTION FUNCTIONS /////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh) );
  SolutionPtr soln1 = Teuchos::rcp( new Solution(mesh1) ); // bc inactive here because it doesn't address any live trial IDs...
  SolutionPtr soln2 = Teuchos::rcp( new Solution(mesh2) );
  
  // define trial variables
  FunctionPtr sigma1_prev1 = Teuchos::rcp( new PreviousSolutionFunction( soln1, sigma1) );
  FunctionPtr sigma2_prev1 = Teuchos::rcp( new PreviousSolutionFunction( soln1, sigma2) );
  FunctionPtr sigma_n_prev1 = Teuchos::rcp( new PreviousSolutionFunction( soln1,sigma_n) );
  
  static_cast< PreviousSolutionFunction* >(sigma1_prev1.get())->setOverrideMeshCheck(true);
  static_cast< PreviousSolutionFunction* >(sigma2_prev1.get())->setOverrideMeshCheck(true);
  static_cast< PreviousSolutionFunction* >(sigma_n_prev1.get())->setOverrideMeshCheck(true);
  
  map< int, FunctionPtr > solnMap1;
  solnMap1[sigma1->ID()] = sigma1_prev1;
  solnMap1[sigma2->ID()] = sigma2_prev1;
  solnMap1[sigma_n->ID()] = sigma_n_prev1;
  
  FunctionPtr uhat_prev2 = Teuchos::rcp( new PreviousSolutionFunction(soln2, uhat) );
  FunctionPtr u_prev2 = Teuchos::rcp( new PreviousSolutionFunction(soln2, u) );
  static_cast< PreviousSolutionFunction* >(u_prev2.get())->setOverrideMeshCheck(true);
  static_cast< PreviousSolutionFunction* >(uhat_prev2.get())->setOverrideMeshCheck(true);
  
  ////////////////////   SPECIFY RHSes   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  Teuchos::RCP<RHSEasy> rhs1 = Teuchos::rcp( new RHSEasy );
  rhs1->addTerm( f * v );
  rhs1->addTerm( beta * u_prev2 * v->grad() - beta * n * uhat_prev2 * v );
  rhs1->addTerm( -eps * u_prev2 * tau->div() + eps * uhat_prev2 * tau->dot_normal() );
  
  Teuchos::RCP<RHSEasy> rhs2 = Teuchos::rcp( new RHSEasy );
  rhs2->addTerm( f * v );
  rhs2->addTerm( (1.0 / eps) * sigma1_prev1 * tau->x() );
  rhs2->addTerm( (1.0 / eps) * sigma2_prev1 * tau->y() );
  rhs2->addTerm( -sigma1_prev1 * v->dx() - sigma2_prev1 * v->dy());
  rhs2->addTerm( sigma_n_prev1 * v );
  
  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary );
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr u0 = (1-x) * (1-y);
  bc->addDirichlet(uhat, outflowBoundary, u0);
  bc->addDirichlet(uhat, inflowBoundary, u0);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  solution->setBC(bc);
  solution->setRHS(rhs);
  solution->setIP(qoptIP);
  
  soln1->setBC(bc);
  soln1->setRHS(rhs1);
  soln1->setIP( ip1 );
  
  soln2->setBC(bc);
  soln2->setRHS(rhs2);
  soln2->setIP( ip2 );
  
  map< int, FunctionPtr > solnMap2;
  solnMap2[uhat->ID()] = uhat_prev2;
  solnMap2[u->ID()] = u_prev2;
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  refinementStrategy.setEnforceOneIrregularity(enforceOneIrregularity);
  
  // project initial guess onto soln1 (u must interpolate the boundary data…)
  map< int, FunctionPtr > initialGuess1;
  initialGuess1[u->ID()] = u0;
  initialGuess1[uhat->ID()] = u0;
  soln1->projectOntoMesh(initialGuess1);
  
  map< int, FunctionPtr > initialGuess2;
  initialGuess2[sigma1->ID()] = eps * u0->dx();
  initialGuess2[sigma2->ID()] = eps * u0->dy();
  initialGuess2[sigma_n->ID()] = eps * u0->dx() * n->x() + eps * u0->dy() * n->y();
  soln2->projectOntoMesh(initialGuess2);
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){
    if (writeStiffnessMatrices) {
      string stiffnessFile = fileNameForRefinement("bf1_stiffness", refIndex);
      soln1->setWriteMatrixToFile(true, stiffnessFile);
    }
    if (writeStiffnessMatrices) {
      string stiffnessFile = fileNameForRefinement("bf2_stiffness", refIndex);
      soln2->setWriteMatrixToFile(true, stiffnessFile);
    }
    // take a bunch of steps--we can be more sophisticated later...
    for (int i=0; i<3; i++) {
//      soln1->solve();
      soln2->solve();
    }
    
    solution->projectOntoMesh(solnMap1);
    solution->projectOntoMesh(solnMap2);
    
    if (writeWorstCaseGramMatrices) {
      string gramFile1 = fileNameForRefinement("bf1_gram", refIndex);
      bool jacobiScaling = true;
      double condNum = MeshUtilities::computeMaxLocalConditionNumber(soln1->ip(), mesh1, jacobiScaling, gramFile1);
      if (rank==0) {
        cout << "estimated worst-case Gram matrix condition number for bf1: " << condNum << endl;
        cout << "putative worst-case Gram matrix written to file for bf1 " << gramFile1 << endl;
      }
      string gramFile2 = fileNameForRefinement("bf2_gram", refIndex);
      condNum = MeshUtilities::computeMaxLocalConditionNumber(soln2->ip(), mesh2, jacobiScaling, gramFile2);
      if (rank==0) {
        cout << "estimated worst-case Gram matrix condition number for bf2: " << condNum << endl;
        cout << "putative worst-case Gram matrix written to file for bf2 " << gramFile2 << endl;
      }
    }
    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  if (writeStiffnessMatrices) {
    string stiffnessFile = fileNameForRefinement("bf1_stiffness", numRefs);
    soln1->setWriteMatrixToFile(true, stiffnessFile);
  }
  if (writeStiffnessMatrices) {
    string stiffnessFile = fileNameForRefinement("bf2_stiffness", numRefs);
    soln2->setWriteMatrixToFile(true, stiffnessFile);
  }
  if (writeWorstCaseGramMatrices) {
    string gramFile1 = fileNameForRefinement("bf1_gram", numRefs);
    bool jacobiScaling = true;
    double condNum = MeshUtilities::computeMaxLocalConditionNumber(soln1->ip(), mesh1, jacobiScaling, gramFile1);
    if (rank==0) {
      cout << "estimated worst-case Gram matrix condition number for bf1: " << condNum << endl;
      cout << "putative worst-case Gram matrix written to file for bf1 " << gramFile1 << endl;
    }
    string gramFile2 = fileNameForRefinement("bf2_gram", numRefs);
    condNum = MeshUtilities::computeMaxLocalConditionNumber(soln2->ip(), mesh2, jacobiScaling, gramFile2);
    if (rank==0) {
      cout << "estimated worst-case Gram matrix condition number for bf2: " << condNum << endl;
      cout << "putative worst-case Gram matrix written to file for bf2 " << gramFile2 << endl;
    }
  }
  
  // one more solve on the final refined mesh:
  soln2->solve();
//  soln1->solve();
  
  solution->projectOntoMesh(solnMap1);
  solution->projectOntoMesh(solnMap2);
  
  if (rank==0){
    soln2->writeFieldsToFile(u->ID(), "u_iter.m");
    soln2->writeFluxesToFile(uhat->ID(), "u_iter_hat.dat");
    
    cout << "wrote files: u_iter.m, u_iter_hat.dat\n";
  }
  
  return 0;
}