//  NewConfusionDriver.cpp
//  Camellia
//
//  Created by Nathan Roberts on 4/2/12.

#include "InnerProductScratchPad.h"
#include "RefinementStrategy.h"
#include "Constraint.h"
#include "PenaltyConstraints.h"

#include "Solution.h"

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

// boundary value for u
class U0 : public Function {
public:
  U0() : Function(0) {}
  void values(FieldContainer<double> &values, BasisCachePtr basisCache) {
    int numCells = values.dimension(0);
    int numPoints = values.dimension(1);
    
    const FieldContainer<double> *points = &(basisCache->getPhysicalCubaturePoints());
    double tol=1e-14;
    for (int cellIndex=0; cellIndex<numCells; cellIndex++) {
      for (int ptIndex=0; ptIndex<numPoints; ptIndex++) {
        double x = (*points)(cellIndex,ptIndex,0);
        double y = (*points)(cellIndex,ptIndex,1);
        // solution with a boundary layer (section 5.2 in DPG Part II)
        // for x = 1, y = 1: u = 0
        if ( ( abs(x-1.0) < tol ) || (abs(y-1.0) < tol ) ) {
          values(cellIndex,ptIndex) = 0;  
        } else if ( abs(x) < tol ) { // for x=0: u = 1 - y
          values(cellIndex,ptIndex) = 1.0 - y;
        } else { // for y=0: u=1-x
          values(cellIndex,ptIndex) = 1.0 - x;   
        }

      }
    }
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
  bool useCompliantGraphNorm = false;
  bool enforceOneIrregularity = true;
  bool writeStiffnessMatrices = true;
  bool writeWorstCaseGramMatrices = true;
  int numRefs = 0;
  
  int H1Order = 3, pToAdd = 2;
  int horizontalCells = 1, verticalCells = 1;
  
  // problem parameters:
  double eps = 1e-4;
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
  if (rank==0) {
    string normChoice = useCompliantGraphNorm ? "unit-compliant graph norm" : "standard graph norm";
    cout << "Using " << normChoice << "." << endl;
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
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u;
  if (useCompliantGraphNorm) {
    u = varFactory.fieldVar("u",HGRAD);
  } else {
    u = varFactory.fieldVar("u");
  }
  
  VarPtr sigma1 = varFactory.fieldVar("\\sigma_1");
  VarPtr sigma2 = varFactory.fieldVar("\\sigma_2");
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma1 / eps, tau->x());
  confusionBF->addTerm(sigma2 / eps, tau->y());
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma1, v->dx() );
  confusionBF->addTerm( sigma2, v->dy() );
  confusionBF->addTerm( beta_const * u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////
  // mathematician's norm
  IPPtr mathIP = Teuchos::rcp(new IP());
  mathIP->addTerm(tau);
  mathIP->addTerm(tau->div());

  mathIP->addTerm(v);
  mathIP->addTerm(v->grad());

  // quasi-optimal norm
  IPPtr qoptIP = Teuchos::rcp(new IP);
  
  if (!useCompliantGraphNorm) {
    qoptIP->addTerm( tau / eps + v->grad() );
    qoptIP->addTerm( beta_const * v->grad() - tau->div() );
    
    qoptIP->addTerm( v );
  } else {
    FunctionPtr h = Teuchos::rcp( new hFunction );
    
    // here, we're aiming at optimality in 1/h^2 |u|^2 + 1/eps^2 |sigma|^2
    
    qoptIP->addTerm( tau + eps * v->grad() );
    qoptIP->addTerm( h * beta_const * v->grad() - tau->div() );
    qoptIP->addTerm(v);
    qoptIP->addTerm(tau);
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Teuchos::rcp( new ConstantScalarFunction(0.0) );
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflowBoundary = Teuchos::rcp( new InflowSquareBoundary );
  SpatialFilterPtr outflowBoundary = Teuchos::rcp( new OutflowSquareBoundary );
  FunctionPtr u0 = Teuchos::rcp( new U0 );
  bc->addDirichlet(uhat, outflowBoundary, u0);

  // bc->addDirichlet(uhat, inflowBoundary, u0);
  
  Teuchos::RCP<PenaltyConstraints> pc = Teuchos::rcp(new PenaltyConstraints);
  pc->addConstraint(uhat==u0,inflowBoundary);

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

  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);
  
  ////////////////////   SOLVE & REFINE   ///////////////////////
  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, qoptIP) );
  solution->setFilter(pc);
  
  double energyThreshold = 0.2; // for mesh refinements
  RefinementStrategy refinementStrategy( solution, energyThreshold );
  refinementStrategy.setEnforceOneIrregularity(enforceOneIrregularity);
  
  for (int refIndex=0; refIndex<numRefs; refIndex++){
    if (writeStiffnessMatrices) {
      string stiffnessFile = fileNameForRefinement("confusion_stiffness", refIndex);
      solution->setWriteMatrixToFile(true, stiffnessFile);
    }
    solution->solve();
    if (writeWorstCaseGramMatrices) {
      string gramFile = fileNameForRefinement("confusion_gram", refIndex);
      bool jacobiScaling = true;
      double condNum = MeshUtilities::computeMaxLocalConditionNumber(qoptIP, mesh, jacobiScaling, gramFile);
      if (rank==0) {
        cout << "estimated worst-case Gram matrix condition number: " << condNum << endl;
        cout << "putative worst-case Gram matrix written to file " << gramFile << endl;
      }
    }
    refinementStrategy.refine(rank==0); // print to console on rank 0
  }
  if (writeStiffnessMatrices) {
    string stiffnessFile = fileNameForRefinement("confusion_stiffness", numRefs);
    solution->setWriteMatrixToFile(true, stiffnessFile);
  }
  if (writeWorstCaseGramMatrices) {
    string gramFile = fileNameForRefinement("confusion_gram", numRefs);
    bool jacobiScaling = true;
    double condNum = MeshUtilities::computeMaxLocalConditionNumber(qoptIP, mesh, jacobiScaling, gramFile);
    if (rank==0) {
      cout << "estimated worst-case Gram matrix condition number: " << condNum << endl;
      cout << "putative worst-case Gram matrix written to file " << gramFile << endl;
    }
  }
  // one more solve on the final refined mesh:
  solution->solve();
  
  if (rank==0){
    solution->writeFieldsToFile(u->ID(), "u.m");
    solution->writeFluxesToFile(uhat->ID(), "u_hat.dat");
    
    cout << "wrote files: u.m, u_hat.dat\n";
  }
  
  return 0;
}
