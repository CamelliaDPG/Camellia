//
//  GDAMinimumRuleTests.cpp
//  Camellia-debug
//
//  Created by Nate Roberts on 2/26/14.
//
//

#include "GDAMinimumRuleTests.h"

#include "MeshFactory.h"

#include "BF.h"

#include "BCEasy.h"
#include "RHSEasy.h"

#include "Solution.h"
#include "PreviousSolutionFunction.h"

#include "Epetra_SerialComm.h"

// boundary value for u
class GDAMinimumRuleTests_U0 : public Function {
public:
  GDAMinimumRuleTests_U0() : Function(0) {}
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


GDAMinimumRuleTests::GDAMinimumRuleTests() {
  
}

SolutionPtr GDAMinimumRuleTests::quadMeshSolution(bool useMinRule, int horizontalCells, int verticalCells) {
  double eps = 1e-2;
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
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
  
  MeshPtr mesh;
  
  int H1Order = 2;
  int pToAddTest = 2;
  double width = 1.0, height = 1.0;
  if (useMinRule) {
    mesh = MeshFactory::quadMeshMinRule(confusionBF, H1Order, pToAddTest, width, height, horizontalCells, verticalCells);
  } else {
    mesh = MeshFactory::quadMesh(confusionBF, H1Order, pToAddTest, width, height, horizontalCells, verticalCells);
  }
  
  ////////////////////   SPECIFY RHS   ///////////////////////
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = Function::zero();
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!
  
  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  FunctionPtr u0 = Teuchos::rcp( new GDAMinimumRuleTests_U0 );
  bc->addDirichlet(uhat, SpatialFilter::allSpace(), u0);
  
  IPPtr ip = confusionBF->graphNorm();
  
  SolutionPtr soln = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  return soln;
}

void GDAMinimumRuleTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testGlobalToLocalToGlobalConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testLocalInterpretationConsistency()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testSingleCellMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}
void GDAMinimumRuleTests::setup() {
  
}

void GDAMinimumRuleTests::teardown() {
  
}

bool GDAMinimumRuleTests::testGlobalToLocalToGlobalConsistency() {
  bool success = true;
  
  SolutionPtr minRuleSoln = quadMeshSolution(true, 1, 1);
  GlobalDofAssignmentPtr gda = minRuleSoln->mesh()->globalDofAssignment();

  int numGlobalDofs = gda->globalDofCount();
  FieldContainer<double> globalData(numGlobalDofs);
  for (int i=0; i<numGlobalDofs; i++) {
    globalData(i) = 2*i + 1; // arbitrary data
  }
  FieldContainer<double> globalDataExpected = globalData;
  FieldContainer<double> globalDataActual;
  FieldContainer<double> localData;
  GlobalIndexType cellID = 0;
  
  Epetra_SerialComm Comm;
  Epetra_BlockMap map(numGlobalDofs, 1, 0, Comm);
  Epetra_Vector globalDataVector(map);
  for (int i=0; i<numGlobalDofs; i++) {
    globalDataVector[i] = globalData(i);
  }
  
  bool dontAccumulate = false;
  gda->interpretGlobalData(cellID, localData, globalDataVector,dontAccumulate);
  FieldContainer<GlobalIndexType> globalDofIndices;
  gda->interpretLocalData(cellID, localData, globalDataActual, globalDofIndices,dontAccumulate);
  
  double tol=1e-14;
  double maxDiff;
  if (TestSuite::fcsAgree(globalDataActual, globalDataExpected, tol, maxDiff)) {
    //    cout << "global data actual and expected AGREE; max difference is " << maxDiff << endl;
    //    cout << "globalDataActual:\n" << globalDataActual;
  } else {
    cout << "global data actual and expected DISAGREE; max difference is " << maxDiff << endl;
    success = false;
    cout << "Expected:\n" << globalDataExpected;
    cout << "Actual:\n" << globalDataActual;
  }
  
  return success;
}

bool GDAMinimumRuleTests::testLocalInterpretationConsistency() {
  bool success = true;
  SolutionPtr minRuleSoln = quadMeshSolution(true, 1, 1);

  GlobalDofAssignmentPtr gda = minRuleSoln->mesh()->globalDofAssignment();
  
  GlobalIndexType cellID = 0;
  DofOrderingPtr trialOrdering = gda->elementType(cellID)->trialOrderPtr;
  FieldContainer<double> localData(trialOrdering->totalDofs());
  // initialize with dummy data
  for (int dofOrdinal=0; dofOrdinal<localData.size(); dofOrdinal++) {
    localData(dofOrdinal) = dofOrdinal;
  }
  FieldContainer<double> globalDataExpected;
  FieldContainer<GlobalIndexType> globalDofIndices;
  gda->interpretLocalData(cellID, localData, globalDataExpected, globalDofIndices);
  
  FieldContainer<double> globalDataActual(globalDataExpected.size());
  set<int> varIDs = trialOrdering->getVarIDs();
  
  set<int> indicesForMappedData;
  
  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
    int varID = *varIt;
    int sideCount = trialOrdering->getNumSidesForVarID(varID);
    for (int sideOrdinal=0; sideOrdinal<sideCount; sideOrdinal++) {
      BasisPtr basis = trialOrdering->getBasis(varID,sideOrdinal);
      FieldContainer<double> basisData(basis->getCardinality());
      // on the assumption that minimum rule does *not* enforce conformity locally, we can consistently pull basis data
      // from the local data without worrying about "double dipping"
      vector<int> dofIndices = trialOrdering->getDofIndices(varID, sideOrdinal);
      int basisOrdinal = 0;
      for (vector<int>::iterator dofIndexIt = dofIndices.begin(); dofIndexIt != dofIndices.end(); dofIndexIt++, basisOrdinal++) {
        int dofIndex = *dofIndexIt;
        
        if (indicesForMappedData.find(dofIndex) != indicesForMappedData.end()) {
          cout << "Error: local dofIndex " << dofIndex << " belongs to multiple bases.  This is not expected in the design of testLocalInterpretationConsistency().\n";
          success = false;
        }
        
        basisData(basisOrdinal) = localData(dofIndex);
        indicesForMappedData.insert(dofIndex);
      }
      FieldContainer<double> globalDataForBasis;
      FieldContainer<GlobalIndexType> globalDofIndicesForBasis;
      gda->interpretLocalBasisData(cellID, varID, sideOrdinal, basisData, globalDataForBasis, globalDofIndicesForBasis);
      int globalEntryCount = globalDofIndicesForBasis.size();
      for (int entryOrdinal=0; entryOrdinal<globalEntryCount; entryOrdinal++) {
        GlobalIndexType entryIndex = globalDofIndicesForBasis(entryOrdinal);
        globalDataActual(entryIndex) += globalDataForBasis(entryOrdinal);
      }
    }
  }
  
  double tol=1e-14;
  double maxDiff;
  if (TestSuite::fcsAgree(globalDataActual, globalDataExpected, tol, maxDiff)) {
//    cout << "global data actual and expected AGREE; max difference is " << maxDiff << endl;
//    cout << "globalDataActual:\n" << globalDataActual;
  } else {
    cout << "global data actual and expected DISAGREE; max difference is " << maxDiff << endl;
    success = false;
  }

  
  return success;
}

bool GDAMinimumRuleTests::testSingleCellMesh() {
  bool success = true;
  
  // up to a permutation of the rows/columns, everything should be identical for min and max rule on a single cell mesh.
  SolutionPtr minRuleSoln = quadMeshSolution(true, 1, 1);
  SolutionPtr maxRuleSoln = quadMeshSolution(false, 1, 1);
  
  if (minRuleSoln->mesh()->numGlobalDofs() != maxRuleSoln->mesh()->numGlobalDofs()) {
    cout << "testSingleCellMesh() failure: min rule mesh doesn't have the same # of global dofs as max rule mesh.  For a one-element mesh, these should be identical.\n";
    success = false;
  }

  minRuleSoln->solve();
  maxRuleSoln->solve();
  
  GlobalIndexType cellID = 0;
  
  DofOrderingPtr trialOrderingMaxRule = maxRuleSoln->mesh()->getElementType(cellID)->trialOrderPtr;
  DofOrderingPtr trialOrderingMinRule = minRuleSoln->mesh()->getElementType(cellID)->trialOrderPtr;
  
  VarFactory vf = maxRuleSoln->mesh()->bilinearForm()->varFactory();
  
  set<int> varIDs = trialOrderingMaxRule->getVarIDs();

  double tol=1e-13;
  for (set<int>::iterator varIt = varIDs.begin(); varIt != varIDs.end(); varIt++) {
    int varID = *varIt;
    VarPtr var = vf.trial(varID);
    string varName = var->name();
    
    FunctionPtr maxRuleSolnFxn = Function::solution(var, maxRuleSoln); //Teuchos::rcp( new PreviousSolutionFunction( maxRuleSoln, var ));
    FunctionPtr minRuleSolnFxn = Function::solution(var, minRuleSoln); //Teuchos::rcp( new PreviousSolutionFunction( minRuleSoln, var ));
    
//    ((PreviousSolutionFunction*) maxRuleSolnFxn.get())->setOverrideMeshCheck(true);
//    ((PreviousSolutionFunction*) minRuleSolnFxn.get())->setOverrideMeshCheck(true);
    
    double l2diff = (maxRuleSolnFxn - minRuleSolnFxn)->l2norm(maxRuleSoln->mesh());
    
    if (l2diff > tol) {
      cout << "Max and min rule solution for var " << varName << " differ; L^2 norm of difference: " << l2diff << endl;
      success = false;
    }
  }
  
  return success;
}