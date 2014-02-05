
#include "SolutionTests.h"

#include "Intrepid_FieldContainer.hpp"
#include "Mesh.h"
#include "MeshFactory.h"

#ifdef HAVE_MPI
#include <Teuchos_GlobalMPISession.hpp>
#else
#endif

#include "ConfusionBilinearForm.h"
#include "ConfusionManufacturedSolution.h"
#include "PoissonBilinearForm.h"
#include "PoissonExactSolution.h"
#include "Function.h"
#include "MathInnerProduct.h"

#include "InnerProductScratchPad.h"

#include "NavierStokesFormulation.h"

#include "ExactSolution.h"

#include "GnuPlotUtil.h"

#include "BC.h"

#include "MeshUtilities.h"

class NewQuadraticFunction : public SimpleFunction {
public:
  double value(double x, double y) {
    return x*y + 3.0 * x * x;
  }
};

class SqrtFunction : public SimpleFunction {
public:
  double value(double x, double y) {
    return sqrt(abs(x));
  }
};

class QuadraticFunction : public AbstractFunction {
public:    
  void getValues(FieldContainer<double> &functionValues, const FieldContainer<double> &physicalPoints) {
    int numCells = physicalPoints.dimension(0);
    int numPoints = physicalPoints.dimension(1);
    functionValues.resize(numCells,numPoints);
    for (int i=0;i<numCells;i++){
      for (int j=0;j<numPoints;j++){
        double x = physicalPoints(i,j,0);
        double y = physicalPoints(i,j,1);
        functionValues(i,j) = x*y + 3.0*x*x;
      }
    }  
  }
  
};

class UnitSquareBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch = (abs(x-1.0)<tol) || (abs(x)<tol);
    bool yMatch = (abs(y-1.0)<tol) || (abs(y)<tol);
    return xMatch || yMatch;
  }
};

class InflowBoundary : public SpatialFilter {
public:
  bool matchesPoint(double x, double y) {
    double tol = 1e-14;
    bool xMatch =  (abs(x)<tol);
    bool yMatch =  (abs(y)<tol);
    return xMatch || yMatch;
  }
};

bool SolutionTests::solutionCoefficientsAreConsistent(Teuchos::RCP<Solution> soln, bool printDetailsToConsole) {
  Teuchos::RCP<BilinearForm> bf = soln->mesh()->bilinearForm();
  
  vector<int> trialIDs = bf->trialIDs();

  map<int, double> globalBasisCoefficients; // maps from globalDofID --> coefficient
  
  bool success = true;
  
  double tol = 1e-14;
  for (int i=0; i<trialIDs.size(); i++) {
    int trialID = trialIDs[i];
    if (bf->isFluxOrTrace(trialID) ) {
      // then there's a chance at inconsistency
      set<GlobalIndexType> activeCellIDs = soln->mesh()->getActiveCellIDs();
      for (set<GlobalIndexType>::iterator cellIt = activeCellIDs.begin(); cellIt != activeCellIDs.end(); cellIt++) {
        GlobalIndexType cellID = *cellIt;
        DofOrderingPtr trialSpace = soln->mesh()->getElement(cellID)->elementType()->trialOrderPtr;
        int numSides = trialSpace->getNumSidesForVarID(trialID);
        for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
          
          vector<int> localDofIndices = trialSpace->getDofIndices(trialID,sideIndex);
          int basisCardinality = localDofIndices.size();
          
          FieldContainer<double> solnCoeffs(basisCardinality);
          soln->solnCoeffsForCellID(solnCoeffs, cellID, trialID, sideIndex);
          
          for (int dofOrdinal = 0; dofOrdinal < basisCardinality; dofOrdinal++) {
            int localDofIndex = localDofIndices[dofOrdinal];
            int globalDofIndex = soln->mesh()->globalDofIndex(cellID,localDofIndex);
            if ( globalBasisCoefficients.find(globalDofIndex) != globalBasisCoefficients.end() ) {
              // compare previous entry
              double diff = abs(globalBasisCoefficients[globalDofIndex] - solnCoeffs[dofOrdinal]);
              if (diff > tol) {
                if (printDetailsToConsole) {
                  cout << "coefficients inconsistent for cellID " << cellID << " and dofOrdinal " << dofOrdinal;
                  cout << " (on side " << sideIndex << "; globalDofIndex = " << globalDofIndex << ")";
                  cout << " and trialID " << trialID << " (diff = " << diff;
                  cout << "; values are " << globalBasisCoefficients[globalDofIndex];
                  cout << " and " << solnCoeffs[dofOrdinal] << ")" << endl;
                }
                success = false;
              }
            }
            // store
            globalBasisCoefficients[globalDofIndex] = solnCoeffs[dofOrdinal];
          }
        }
      }
    }
  }
  return success;
}

// unclear on why these initializers are necessary but others (e.g. _confusionSolution1_2x2) are not
// maybe a bug in Teuchos::RCP?
SolutionTests::SolutionTests() :
_confusionExactSolution(Teuchos::rcp( (ConfusionManufacturedSolution*) NULL )),
_poissonExactSolution(Teuchos::rcp( (PoissonExactSolution*) NULL ))
{}

void SolutionTests::setup() {
  // first, build a simple mesh
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
  
  double epsilon = 1e-2;
  double beta_x = 1.0, beta_y = 1.0;
  _confusionExactSolution = Teuchos::rcp( new ConfusionManufacturedSolution(epsilon,beta_x,beta_y) ); 
  
  bool useConformingTraces = true;
  int polyOrder = 2; // 2 is minimum for projecting QuadraticFunction exactly
  _poissonExactSolution = 
    Teuchos::rcp( new PoissonExactSolution(PoissonExactSolution::POLYNOMIAL, 
					   polyOrder, useConformingTraces) );  
  _poissonExactSolution->setUseSinglePointBCForPHI(false); // impose zero-mean constraint

  int H1Order = polyOrder+1;
  int horizontalCells = 2; int verticalCells = 2;
  
  // before we hRefine, compute a solution for comparison after refinement
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(_confusionExactSolution->bilinearForm()));
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _confusionExactSolution->bilinearForm(), H1Order, H1Order+1);

  Teuchos::RCP<Mesh> poissonMesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _poissonExactSolution->bilinearForm(), H1Order, H1Order+2);
  Teuchos::RCP<Mesh> poissonMesh1x1 = MeshFactory::buildQuadMesh(quadPoints, 1, 1, _poissonExactSolution->bilinearForm(), H1Order, H1Order+2);
  Teuchos::RCP<DPGInnerProduct> poissonIp = Teuchos::rcp(new MathInnerProduct(_poissonExactSolution->bilinearForm()));

  _confusionSolution1_2x2 = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->bc(), _confusionExactSolution->ExactSolution::rhs(), ip) );
  _confusionSolution2_2x2 = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->bc(), _confusionExactSolution->ExactSolution::rhs(), ip) );
  _poissonSolution = Teuchos::rcp( new Solution(poissonMesh, _poissonExactSolution->bc(),_poissonExactSolution->ExactSolution::rhs(), ip));
  _poissonSolution_1x1 = Teuchos::rcp( new Solution(poissonMesh1x1, _poissonExactSolution->bc(),_poissonExactSolution->ExactSolution::rhs(), ip));
  _poissonSolution_1x1_unsolved = Teuchos::rcp( new Solution(poissonMesh1x1, _poissonExactSolution->bc(),_poissonExactSolution->ExactSolution::rhs(), ip));
  
  _confusionUnsolved = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->bc(), _confusionExactSolution->ExactSolution::rhs(), ip) );

  _confusionSolution1_2x2->solve();
  _confusionSolution2_2x2->solve();
  _poissonSolution->solve();
  _poissonSolution_1x1->solve();
  
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  double y[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  
  _testPoints = FieldContainer<double>(NUM_POINTS_1D*NUM_POINTS_1D,2);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    for (int j=0; j<NUM_POINTS_1D; j++) {
      _testPoints(i*NUM_POINTS_1D + j, 0) = x[i];
      _testPoints(i*NUM_POINTS_1D + j, 1) = y[j];
    }
  }
}

void SolutionTests::teardown() {
  _confusionSolution1_2x2 = Teuchos::rcp( (Solution*)NULL );
  _confusionSolution2_2x2 = Teuchos::rcp( (Solution*)NULL );  
  _testPoints.resize(0);
}

void SolutionTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  
  if (testProjectVectorValuedSolution()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testHRefinementInitialization()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testNewProjectFunction()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testSolutionsAreConsistent()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testProjectSolutionOntoOtherMesh()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testSolutionEvaluationBasisCache() ) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testAddSolution()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testProjectFunction()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testAddRefinedSolutions()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testEnergyError()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testPRefinementInitialization()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

 setup();
  if (testScratchPadSolution()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

  setup();
  if (testCondensationSolve()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
}

bool SolutionTests::storageSizesAgree(Teuchos::RCP< Solution > soln1, Teuchos::RCP< Solution > soln2) {
  const map< GlobalIndexType, FieldContainer<double> >* solnMap1 = &(soln1->solutionForCellIDGlobal());
  const map< GlobalIndexType, FieldContainer<double> >* solnMap2 = &(soln2->solutionForCellIDGlobal());
  if (solnMap1->size() != solnMap2->size() ) {
    cout << "SOLUTION 1 entries: ";
    for(map< GlobalIndexType, FieldContainer<double> >::const_iterator soln1It = (*solnMap1).begin();
        soln1It != (*solnMap1).end(); soln1It++) {
      int cellID = soln1It->first;
      cout << cellID << " ";
    }
    cout << endl;
    cout << "SOLUTION 2 entries: ";
    for(map< GlobalIndexType, FieldContainer<double> >::const_iterator soln2It = (*solnMap2).begin();
        soln2It != (*solnMap2).end(); soln2It++) {
      int cellID = soln2It->first;
      cout << cellID << " ";
    }
    cout << endl;
    cout << "active elements: ";
    int numElements = soln1->mesh()->activeElements().size();
    for (int elemIndex=0; elemIndex<numElements; elemIndex++){
      int cellID = soln1->mesh()->activeElements()[elemIndex]->cellID();
      cout << cellID << " ";
    }
    cout << endl;
    
    return false;
  }
  for(map< GlobalIndexType, FieldContainer<double> >::const_iterator soln1It = (*solnMap1).begin();
      soln1It != (*solnMap1).end(); soln1It++) {
    GlobalIndexType cellID = soln1It->first;
    int size = soln1It->second.size();
    map< GlobalIndexType, FieldContainer<double> >::const_iterator soln2It = (*solnMap2).find(cellID);
    if (soln2It == (*solnMap2).end()) {
      return false;
    }
    if ((soln2It->second).size() != size) {
      return false;
    }
  }
  return true;
}

bool SolutionTests::testAddSolution() {
  bool success = true;
  
  double weight = 3.141592;
  double tol = 1e-12;
  
  FieldContainer<double> expectedValuesU(_testPoints.dimension(0));
  FieldContainer<double> expectedValuesSIGMA1(_testPoints.dimension(0));
  FieldContainer<double> expectedValuesSIGMA2(_testPoints.dimension(0));
  _confusionSolution2_2x2->solutionValues(expectedValuesU, ConfusionBilinearForm::U, _testPoints);
  _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA1, ConfusionBilinearForm::SIGMA_1, _testPoints);
  _confusionSolution2_2x2->solutionValues(expectedValuesSIGMA2, ConfusionBilinearForm::SIGMA_2, _testPoints);
  
  BilinearForm::multiplyFCByWeight(expectedValuesU, weight+1.0);
  BilinearForm::multiplyFCByWeight(expectedValuesSIGMA1, weight+1.0);
  BilinearForm::multiplyFCByWeight(expectedValuesSIGMA2, weight+1.0);
  
  _confusionSolution1_2x2->addSolution(_confusionSolution2_2x2, weight);
  FieldContainer<double> valuesU(_testPoints.dimension(0));
  FieldContainer<double> valuesSIGMA1(_testPoints.dimension(0));
  FieldContainer<double> valuesSIGMA2(_testPoints.dimension(0));
  
  _confusionSolution1_2x2->solutionValues(valuesU, ConfusionBilinearForm::U, _testPoints);
  _confusionSolution1_2x2->solutionValues(valuesSIGMA1, ConfusionBilinearForm::SIGMA_1, _testPoints);
  _confusionSolution1_2x2->solutionValues(valuesSIGMA2, ConfusionBilinearForm::SIGMA_2, _testPoints);
  
  for (int pointIndex=0; pointIndex < valuesU.size(); pointIndex++) {
    double diff = abs(valuesU[pointIndex] - expectedValuesU[pointIndex]);
    if (diff > tol) {
      success = false;
      cout << "expected value of U: " << expectedValuesU[pointIndex] << "; actual: " << valuesU[pointIndex] << endl;
    }
    
    diff = abs(valuesSIGMA1[pointIndex] - expectedValuesSIGMA1[pointIndex]);
    if (diff > tol) {
      success = false;
      cout << "expected value of SIGMA1: " << expectedValuesSIGMA1[pointIndex] << "; actual: " << valuesSIGMA1[pointIndex] << endl;
    }
    
    diff = abs(valuesSIGMA2[pointIndex] - expectedValuesSIGMA2[pointIndex]);
    if (diff > tol) {
      success = false;
      cout << "expected value of SIGMA2: " << expectedValuesSIGMA2[pointIndex] << "; actual: " << valuesSIGMA2[pointIndex] << endl;
    }
  }
  
  return success;
}

bool SolutionTests::testProjectFunction() {
  bool success = true;
  double tol = 1e-14;
  Teuchos::RCP<QuadraticFunction> quadraticFunction = Teuchos::rcp(new QuadraticFunction );
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  functionMap[ConfusionBilinearForm::U] = quadraticFunction;
  functionMap[ConfusionBilinearForm::SIGMA_1] = quadraticFunction;
  functionMap[ConfusionBilinearForm::SIGMA_2] = quadraticFunction;

  _confusionUnsolved->projectOntoMesh(functionMap);  
  
  FieldContainer<double> valuesU(_testPoints.dimension(0));
  FieldContainer<double> valuesSIGMA1(_testPoints.dimension(0));
  FieldContainer<double> valuesSIGMA2(_testPoints.dimension(0));
  
  _confusionUnsolved->solutionValues(valuesU, ConfusionBilinearForm::U, _testPoints);
  _confusionUnsolved->solutionValues(valuesSIGMA1, ConfusionBilinearForm::SIGMA_1, _testPoints);
  _confusionUnsolved->solutionValues(valuesSIGMA2, ConfusionBilinearForm::SIGMA_2, _testPoints);

  FieldContainer<double> allCellTestPoints = _testPoints;
  allCellTestPoints.resize(1,_testPoints.dimension(0),_testPoints.dimension(1));
  FieldContainer<double> functionValues(1,_testPoints.dimension(0));
  quadraticFunction->getValues(functionValues,allCellTestPoints);
  int numValues = functionValues.size();
  for (int valueIndex = 0;valueIndex<numValues;valueIndex++){
    double diff = abs(functionValues[valueIndex]-valuesU[valueIndex]);
    if (diff>tol){
      success = false;
      cout << "Test failed: difference in projected and computed values is " << diff << endl;
    }
    diff = abs(functionValues[valueIndex]-valuesSIGMA1[valueIndex]);
    if (diff>tol){
      success = false;
      cout << "Test failed: difference in projected and computed values is " << diff << endl;
    }

    diff = abs(functionValues[valueIndex]-valuesSIGMA2[valueIndex]);
    if (diff>tol){
      success = false;
      cout << "Test failed: difference in projected and computed values is " << diff << endl;
    }

  }      

  return success;  
}

bool SolutionTests::testProjectVectorValuedSolution() {
  bool success = true;
  double tol = 1e-14;
  
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory;
  VarPtr tau = varFactory.testVar("\\tau", HDIV);
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr uhat = varFactory.traceVar("\\widehat{u}");
  VarPtr beta_n_u_minus_sigma_n = varFactory.fluxVar("\\widehat{\\beta \\cdot n u - \\sigma_{n}}");
  VarPtr u = varFactory.fieldVar("u");
  
  VarPtr sigma = varFactory.fieldVar("\\sigma", VECTOR_L2);
  
  // problem parameters:
  double eps = 1e-4;
  vector<double> beta_const;
  beta_const.push_back(2.0);
  beta_const.push_back(1.0);
  
  ////////////////////   DEFINE BILINEAR FORM   ///////////////////////
  BFPtr confusionBF = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  confusionBF->addTerm(sigma / eps, tau);
  confusionBF->addTerm(u, tau->div());
  confusionBF->addTerm(-uhat, tau->dot_normal());
  
  // v terms:
  confusionBF->addTerm( sigma, v->grad() );
  confusionBF->addTerm( beta_const * u, - v->grad() );
  confusionBF->addTerm( beta_n_u_minus_sigma_n, v);
  
  map<int, VarPtr > trialVars = varFactory.trialVars();

  map<int, FunctionPtr > functionMap;
  FunctionPtr quadraticFunction = Teuchos::rcp(new NewQuadraticFunction );
  FunctionPtr vectorFunction = Function::vectorize(Function::xn(1), quadraticFunction);
  
  for (map<int, VarPtr >::iterator varIt = trialVars.begin();
       varIt != trialVars.end(); varIt++) {
    VarPtr var = varIt->second;
    int varID = var->ID();
    if (var->space() == VECTOR_L2) {
      functionMap[varID] = vectorFunction;
    } else {
      functionMap[varID] = quadraticFunction;
    }
  }
  
  int spaceDim = 2;
  FieldContainer<double> quadPoints(4,spaceDim);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
  
  int horizontalCells = 3;
  int verticalCells = 3;
  
  int H1Order = 3; // that way, L^2 space contains quadratics
  
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, confusionBF, H1Order, H1Order+1);

  SolutionPtr confusionSoln = Teuchos::rcp( new Solution(mesh) );
  
  confusionSoln->projectOntoMesh(functionMap);
  
  if ( ! solutionCoefficientsAreConsistent(confusionSoln) ) {
    cout << "testProjectVectorValuedSolution: for quadraticFunction projection, solution coefficients are inconsistent.\n";
    success = false;
  }
  
  for (int testIndex=0; testIndex<2; testIndex++) {
    if (testIndex == 1) {
      // test in which we project the Solution itself
      // here, we just project the Solution onto itself, nothing terribly interesting, but should exercise the vector-valuedness anyway
      // conveniently, this means that the expected values don't change, so we can use the same verification logic below
      set<GlobalIndexType> activeCellIDs = confusionSoln->mesh()->getActiveCellIDs();
      for (set<GlobalIndexType>::iterator cellIDIt = activeCellIDs.begin(); cellIDIt != activeCellIDs.end(); cellIDIt++) {
        GlobalIndexType cellID = *cellIDIt;
        ElementTypePtr elemType = confusionSoln->mesh()->getElement(cellID)->elementType();
        vector<GlobalIndexType> childIDs;
        childIDs.push_back(cellID);
        confusionSoln->projectOldCellOntoNewCells(cellID, elemType, childIDs);
      }
    }
    for (int i=0; i<confusionSoln->mesh()->numElements(); i++) {
      int numCells = 1;
      ElementPtr elem = confusionSoln->mesh()->getElement(i);
      int cellID = elem->cellID();
      vector<GlobalIndexType> cellIDs(1,cellID);
      BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elem->elementType(),confusionSoln->mesh()) );
      
      basisCache->setPhysicalCellNodes( confusionSoln->mesh()->physicalCellNodesForCell(cellID),
                                       cellIDs, true); // true: create side cache, too
      int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
      int sideToTest = 2;
      int numPointsSide = basisCache->getSideBasisCache(sideToTest)->getPhysicalCubaturePoints().dimension(1);
      
      FieldContainer<double> functionValues(numCells, numPoints);
      quadraticFunction->values(functionValues,basisCache);
      
      FieldContainer<double> vectorFunctionValues(numCells, numPoints, spaceDim);
      vectorFunction->values(vectorFunctionValues, basisCache);
      
      FieldContainer<double> functionValuesSide(numCells, numPointsSide);
      quadraticFunction->values(functionValuesSide,basisCache->getSideBasisCache(sideToTest));

      FieldContainer<double> vectorFunctionValuesSide(numCells, numPointsSide, spaceDim);
      vectorFunction->values(vectorFunctionValuesSide, basisCache->getSideBasisCache(sideToTest));
      
      for (map<int, VarPtr >::iterator varIt = trialVars.begin();
           varIt != trialVars.end(); varIt++) {
        VarPtr var = varIt->second;
        int varID = var->ID();
        // for second test, we only want to test fields, since that's what's supported in projections...
        if ((testIndex==1) && ((var->varType()==FLUX) || (var->varType()==TRACE))) continue;
        FieldContainer<double> valuesExpected;
        FieldContainer<double> valuesActual;
        if ( confusionBF->isFluxOrTrace(varID) ) {
          FieldContainer<double> values;
          if (var->rank()==0) {
            values = FieldContainer<double>(numCells, numPointsSide);
            valuesExpected = functionValuesSide;
          } else if (var->rank()==1) {
            values = FieldContainer<double>(numCells, numPointsSide, spaceDim);
            valuesExpected = vectorFunctionValuesSide;
          }
          confusionSoln->solutionValues(values, varID, basisCache->getSideBasisCache(sideToTest));
          valuesActual = values;
        } else { // volume
          FieldContainer<double> values;
          if (var->rank()==0) {
            values = FieldContainer<double>(numCells, numPoints);
            valuesExpected = functionValues;
          } else if (var->rank()==1) {
            values = FieldContainer<double>(numCells, numPoints, spaceDim);
            valuesExpected = vectorFunctionValues;
          }
          confusionSoln->solutionValues(values, varID, basisCache);
          valuesActual = values;
        }
        double maxDiff;
        if ( !fcsAgree(valuesExpected, valuesActual, tol, maxDiff) ) {
          cout << "testProjectVectorValuedSolution() failure: maxDiff is " << maxDiff << " for trial variable " << var->name() << endl;
          cout << "expectedValues:\n" << valuesExpected << endl;
          cout << "actualValues:\n" << valuesActual << endl;
          success = false;
        }
      }
    }
    if (testIndex==1) {
      // then let's try adding vector-valued Solutions together
      // (just checking that this doesn't throw an exception)
      SolutionPtr soln2 = Teuchos::rcp(new Solution(mesh));
      map<int, FunctionPtr > functionMap;
      functionMap[ sigma->ID() ] = Function::zero(1); // assumes 2D (tensor zero of rank 1)
      functionMap[ u->ID() ] = Function::zero();
      soln2->projectOntoMesh(functionMap);
      
      soln2->addSolution(confusionSoln, 1.0);
      soln2 = Teuchos::rcp(new Solution(mesh));
      soln2->projectOntoMesh(functionMap);
      confusionSoln->addSolution(confusionSoln, 1.0);
    }
  }
  
  return success;
}

bool SolutionTests::testNewProjectFunction() {
  bool success = true;
  double tol = 1e-14;
  
  Teuchos::RCP<BilinearForm> bf = _confusionUnsolved->mesh()->bilinearForm();
  
  vector<int> trialIDs = bf->trialIDs();
  
  map<int, FunctionPtr > functionMap;
  FunctionPtr quadraticFunction = Teuchos::rcp(new NewQuadraticFunction );
  for (int i=0; i<trialIDs.size(); i++) {
    int trialID = trialIDs[i];
    functionMap[trialID] = quadraticFunction;
  }
  
  _confusionUnsolved->projectOntoMesh(functionMap);
  
  if ( ! solutionCoefficientsAreConsistent(_confusionUnsolved) ) {
    cout << "testNewProjectFunction: for quadraticFunction projection, solution coefficients are inconsistent.\n";
    success = false;
  }
  
  for (int i=0; i<_confusionUnsolved->mesh()->numElements(); i++) {
    int numCells = 1;
    ElementPtr elem = _confusionUnsolved->mesh()->getElement(i);
    int cellID = elem->cellID();
    vector<GlobalIndexType> cellIDs(1,cellID);
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elem->elementType(),_confusionUnsolved->mesh()) );
        
    basisCache->setPhysicalCellNodes( _confusionUnsolved->mesh()->physicalCellNodesForCell(cellID),
                                     cellIDs, true); // true: create side cache, too
    int numPoints = basisCache->getPhysicalCubaturePoints().dimension(1);
    int sideToTest = 2;
    int numPointsSide = basisCache->getSideBasisCache(sideToTest)->getPhysicalCubaturePoints().dimension(1);

    FieldContainer<double> functionValues(numCells, numPoints);
    quadraticFunction->values(functionValues,basisCache);
    
    FieldContainer<double> functionValuesSide(numCells, numPointsSide);
    quadraticFunction->values(functionValuesSide,basisCache->getSideBasisCache(sideToTest));
    
    for (int trialIndex=0; trialIndex<trialIDs.size(); trialIndex++) {
      int trialID = trialIDs[trialIndex];
      FieldContainer<double> valuesExpected;
      FieldContainer<double> valuesActual;
      if ( bf->isFluxOrTrace(trialID) ) {
        FieldContainer<double> values(numCells, numPointsSide);
        _confusionUnsolved->solutionValues(values, trialID, basisCache->getSideBasisCache(sideToTest));
        valuesActual = values;
        valuesExpected = functionValuesSide;
      } else { // volume
        FieldContainer<double> values(numCells, numPoints);
        _confusionUnsolved->solutionValues(values, trialID, basisCache);
        valuesActual = values;
        valuesExpected = functionValues;
      }
      double maxDiff;
      if ( !fcsAgree(valuesExpected, valuesActual, tol, maxDiff) ) {
        cout << "testNewProjectFunction() failure: maxDiff is " << maxDiff << " for trialID " << trialID << endl;
        cout << "expectedValues:\n" << valuesExpected << endl;
        cout << "actualValues:\n" << valuesActual << endl;
        success = false;
      }
    }
  }

  // now, try something a little different: project various functions onto
  // a constant mesh.
  
  FieldContainer<double> quadPoints(4,2);
  
//  quadPoints(0,0) = 0.0; // x1
//  quadPoints(0,1) = 0.0; // y1
//  quadPoints(1,0) = 1.0;
//  quadPoints(1,1) = 0.0;
//  quadPoints(2,0) = 1.0;
//  quadPoints(2,1) = 1.0;
//  quadPoints(3,0) = 0.0;
//  quadPoints(3,1) = 1.0;
  
  // Domain from Evans Hughes for Navier-Stokes Kovasznay flow:
  quadPoints(0,0) =  0.0; // x1
  quadPoints(0,1) = -0.5; // y1
  quadPoints(1,0) =  1.0;
  quadPoints(1,1) = -0.5;
  quadPoints(2,0) =  1.0;
  quadPoints(2,1) =  0.5;
  quadPoints(3,0) =  0.0;
  quadPoints(3,1) =  0.5;
  
  VarFactory varFactory;
  VarPtr u = varFactory.fieldVar("u");
  VarPtr v = varFactory.testVar("v", HGRAD);
  BFPtr emptyBF = Teuchos::rcp(new BF(varFactory));
  Teuchos::RCP< BCEasy > bc = Teuchos::rcp( new BCEasy );
  
  int H1Order = 1; // constant L^2 ==> the projection on each cell should be the L^2 norm on that cell
  int horizontalCells = 2, verticalCells = 2;
  int pTest = 1; // doesn't matter
  
  Teuchos::RCP<Mesh> fineMesh = MeshFactory::buildQuadMesh(quadPoints, 16, 16, emptyBF, H1Order, pTest);
  
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, emptyBF, H1Order, pTest);
  SolutionPtr soln = Teuchos::rcp( new Solution(mesh, bc) );
  
  // set up the functions
  vector< FunctionPtr > functions;
  double Re = 40;
  FunctionPtr u1_exact, u2_exact, p_exact;
  NavierStokesFormulation::setKovasznay( Re, mesh, u1_exact, u2_exact, p_exact );
  functions.push_back(u1_exact);
  functions.push_back(u2_exact);
  functions.push_back(p_exact);
  
  FunctionPtr sqrtFunction = Teuchos::rcp( new SqrtFunction );
  functions.push_back(sqrtFunction);
  
  for (int j=0; j<functions.size(); j++) {
    FunctionPtr f = functions[j];
    functionMap.clear();
    functionMap[u->ID()] = f;
    
    FieldContainer<double> expectedValues( mesh->numActiveElements() );
    FieldContainer<double> actualValues( mesh->numActiveElements() );
    
    ElementTypePtr elemType = mesh->elementTypes()[0];
    bool testVsTest=false;
    int cubatureDegreeEnrichment = 5;
    
    double fIntegral = f->integrate(mesh,cubatureDegreeEnrichment);
//    cout << "testNewProjectFunction: integral of f on whole mesh = " << fIntegral << endl;
    
    double l2ErrorOfAverage = (Function::constant(fIntegral) - f)->l2norm(fineMesh,cubatureDegreeEnrichment);
//    cout << "testNewProjectFunction: l2 error of fIntegral: " << l2ErrorOfAverage << endl;
    
    vector<GlobalIndexType> cellIDs = mesh->cellIDsOfTypeGlobal(elemType);
    
    BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elemType, mesh, testVsTest, cubatureDegreeEnrichment) );
    basisCache->setPhysicalCellNodes(mesh->physicalCellNodesGlobal(elemType), cellIDs, false); // false: no side cache
    
    f->integrate(expectedValues, basisCache);
    FieldContainer<double> cellMeasures = basisCache->getCellMeasures();
    
    for (int i=0; i<expectedValues.size(); i++) {
      expectedValues(i) /= cellMeasures(i);
    }
    
    soln->setCubatureEnrichmentDegree(cubatureDegreeEnrichment);
    soln->projectOntoMesh(functionMap);
    
    if ( ! solutionCoefficientsAreConsistent(soln) ) {
      cout << "testNewProjectFunction: in projection of Function " << j << " onto constants, solution coefficients are inconsistent.\n";
      success = false;
    }

    FunctionPtr projectedFunction = Function::solution(u, soln);
    projectedFunction->integrate(actualValues, basisCache); // these will be weighted by cellMeasures
    for (int i=0; i<actualValues.size(); i++) {
      actualValues(i) /= cellMeasures(i);
    }
    
//    cout << "expectedValues for projection of f onto constants:\n" << expectedValues;
    
    double maxDiff;
    if ( !fcsAgree(expectedValues, actualValues, tol, maxDiff) ) {
      cout << "testNewProjectFunction() failure in projection of Function " << j << " onto constants: maxDiff is " << maxDiff << endl;
      cout << "expectedValues:\n" << expectedValues << endl;
      cout << "actualValues:\n" << actualValues << endl;
      
      success = false;
    }
    
    int trialID = u->ID();
    VarPtr trialVar = u;
    
    // this maybe doesn't exactly belong here (better to have an ExactSolution test),
    // but this is convenient for now:
    Teuchos::RCP<ExactSolution> exactSoln = Teuchos::rcp( new ExactSolution );
    exactSoln->setSolutionFunction(trialVar, f);
    // test the L2 error measured in two ways
    double l2errorActual = exactSoln->L2NormOfError(*soln, trialID, 15);
    
    FunctionPtr bestFxnError = Function::solution(trialVar, soln) - f;
    int matchingCubatureEnrichment = 15 - (pTest + H1Order - 1); // chosen so that the effective cubature degree below will match that above
    double l2errorExpected = bestFxnError->l2norm(soln->mesh(),matchingCubatureEnrichment); // here the cubature is actually an enrichment....
    
    if (abs(l2errorExpected - l2errorActual) > tol) {
      success = false;
      cout << "testNewProjectFunction: for function " << j << ", ExactSolution error doesn't match";
      cout << " that measured by Function: " << l2errorActual << " vs " << l2errorExpected << endl;
    }
  }
  
  return success;
}

bool SolutionTests::testProjectSolutionOntoOtherMesh() {
  bool success = true;
  double tol = 1e-14;
  _poissonSolution_1x1->projectFieldVariablesOntoOtherSolution(_poissonSolution);
//  _poissonSolution_1x1->writeFieldsToFile(PoissonBilinearForm::PHI, "phi_1x1.m");
//  _poissonSolution->writeFieldsToFile(PoissonBilinearForm::PHI, "phi_1x1_projected.m");
  _poissonSolution->projectFieldVariablesOntoOtherSolution(_poissonSolution_1x1_unsolved);
  // difference should be zero:
  _poissonSolution_1x1_unsolved->addSolution(_poissonSolution_1x1,-1.0);
  // test for all field variables:
  vector<int> fieldIDs = _poissonSolution_1x1->mesh()->bilinearForm()->trialVolumeIDs();
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    double diffL2 = _poissonSolution_1x1_unsolved->L2NormOfSolutionGlobal(fieldID);
    if (diffL2 > tol) {
      string varName = _poissonSolution_1x1->mesh()->bilinearForm()->trialName(fieldID);
      cout << "testProjectSolutionOntoOtherMesh: Failure for trial ID " << varName << ": ";
      cout << "coarse solution projected onto fine mesh then back onto coarse differs from original by L2 norm of " << diffL2 << endl;
      cout << "L2 norm of original solution: " << _poissonSolution_1x1->L2NormOfSolutionGlobal(fieldID) << endl;
      success = false;
    }
  }
  
  return success;
}

bool SolutionTests::testAddRefinedSolutions() {
  bool success = true;

  Teuchos::RCP<QuadraticFunction> quadraticFunction = Teuchos::rcp(new QuadraticFunction );
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  functionMap[ConfusionBilinearForm::U] = quadraticFunction;
  functionMap[ConfusionBilinearForm::SIGMA_1] = quadraticFunction;
  functionMap[ConfusionBilinearForm::SIGMA_2] = quadraticFunction;
  _confusionSolution2_2x2->projectOntoMesh(functionMap);  // pretend confusionSolution1 is the linearized solution

  // solve
  _confusionSolution1_2x2->solve(false);

  // add the two solutions together
  _confusionSolution2_2x2->addSolution(_confusionSolution1_2x2,1.0);    // pretend confusionSolution2 is the accumulated solution

  // refine the mesh
  vector<GlobalIndexType> quadCellsToRefine;
  quadCellsToRefine.push_back(0); // refine first cell
  _confusionSolution1_2x2->mesh()->registerSolution(_confusionSolution1_2x2);
  _confusionSolution1_2x2->mesh()->registerSolution(_confusionSolution2_2x2);
  _confusionSolution1_2x2->mesh()->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());

  // solve
  _confusionSolution1_2x2->solve(false); // resolve for du on new mesh
  
  if ( ! storageSizesAgree(_confusionSolution1_2x2, _confusionSolution2_2x2) ) {
    cout << "Storage sizes disagree, so add will fail.\n";
    return false;
  }
  
  // add the two solutions together
  _confusionSolution1_2x2->addSolution(_confusionSolution2_2x2,1.0);    

  return success;  
}


bool SolutionTests::testEnergyError(){

  double tol = 1e-11;

  bool success = true;
  // First test: exact solution has zero energy error:
  map<GlobalIndexType, double> energyError = _poissonSolution->energyError();
  vector< Teuchos::RCP< Element > > activeElements = _poissonSolution->mesh()->activeElements();
  vector< Teuchos::RCP< Element > >::iterator activeElemIt;
  
  double totalEnergyErrorSquared = 0.0;
  for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    totalEnergyErrorSquared += energyError[cellID]*energyError[cellID];
  }
  if (totalEnergyErrorSquared > tol) {
    success = false;
    cout << "testEnergyError failed: energy error is " << totalEnergyErrorSquared << endl;
  }
  
  // second test: test and trial spaces the same, define b(u,v) = (u,v)
  // then the energy norm of u = (u,u)^1/2
  VarFactory varFactory;
  VarPtr u = varFactory.fieldVar("u");
  VarPtr v = varFactory.testVar("v", L2); // L2 so that the orders for u and v can match

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  bf->addTerm(u,v); // L2 norm
  
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr uSoln = Teuchos::rcp( new ConstantScalarFunction(3.0) );
  rhs->addTerm(uSoln * v);
  
  BCPtr bc = Teuchos::rcp( new BCEasy ); // no bcs
  
  IPPtr ip = Teuchos::rcp( new IP );
  ip->addTerm(v); // L^2
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;  
    
  int horizontalElements = 10, verticalElements = 5;
  int H1Order = 3;
  int pTest = H1Order;
  
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalElements, verticalElements, bf, H1Order, pTest);
  
  SolutionPtr soln = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  
  double expectedEnergyError = sqrt( (uSoln * uSoln )->integrate(mesh) ); // integral of a constant
  
  double actualEnergyError = soln->energyErrorTotal();
  
  if (abs(actualEnergyError - expectedEnergyError) > tol) {
    cout << "Expected energy error to be " << expectedEnergyError << "; was " << actualEnergyError << endl;
    success = false;
  }
  
  // 3rd test: much the same, but different RHS
  uSoln = Teuchos::rcp( new NewQuadraticFunction );
  
  rhs = Teuchos::rcp( new RHSEasy );
  rhs->addTerm(uSoln * v);
  
  soln = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  
  // compute the L^2 norm of uSoln:
  expectedEnergyError = sqrt( (uSoln * uSoln)->integrate(mesh) );
  
  actualEnergyError = soln->energyErrorTotal();
  
  if (abs(actualEnergyError - expectedEnergyError) > tol) {
    cout << "Expected energy error to be " << expectedEnergyError << "; was " << actualEnergyError << endl;
    success = false;
  }
  
  // 4th test: try a non-zero solution, zero RHS
  rhs = Teuchos::rcp( new RHSEasy );
  soln = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );

  map<int, FunctionPtr > functionMap;
  functionMap[u->ID()] = uSoln;
  soln->projectOntoMesh(functionMap);
  
  // compute the L^2 norm of uSoln:
  expectedEnergyError = sqrt( (uSoln * uSoln)->integrate(mesh) );
  actualEnergyError = soln->energyErrorTotal();
  
  if (abs(actualEnergyError - expectedEnergyError) > tol) {
    cout << "Expected energy error to be " << expectedEnergyError << "; was " << actualEnergyError << endl;
    success = false;
  }
  
  return success;
}


bool SolutionTests::testHRefinementInitialization() {

  double tol = 2e-14;

  bool success = true;
  Teuchos::RCP< Mesh > mesh = _poissonSolution->mesh();
  
  //_poissonSolution->solve(false);
  
  // for now, instead of using a true solution, let's just project some known functions onto the mesh
  // start with an exactly representable polynomial, but this should work for arbitrary functions...
  FunctionPtr x = Function::xn();
  FunctionPtr y = Function::yn();
  FunctionPtr phiFxn = x;// * x * x + x * y;
  FunctionPtr psiFxn = phiFxn->grad();
  map< int, FunctionPtr > fxnMap;
  fxnMap[ PoissonBilinearForm::PHI ] = phiFxn;
  fxnMap[ PoissonBilinearForm::PSI_1 ] = psiFxn->x();
  fxnMap[ PoissonBilinearForm::PSI_2 ] = psiFxn->y();
  
  _poissonSolution->projectOntoMesh(fxnMap);
  
//  int trialIDToWrite = PoissonBilinearForm::PHI;
  string filePrefix = "phi";
  string fileSuffix = ".m";
//  _poissonSolution->writeFieldsToFile(trialIDToWrite, filePrefix + "BeforeRefinement" + fileSuffix);
  
  // test for all field variables:
  vector<int> fieldIDs = _poissonSolution->mesh()->bilinearForm()->trialVolumeIDs();
  
  map<int, FieldContainer<double> > expectedMap;
  
  FieldContainer<double> expectedValues(_testPoints.dimension(0)); 
  FieldContainer<double> actualValues(_testPoints.dimension(0)); 
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(expectedValues,fieldID,_testPoints);
    expectedMap[fieldID] = expectedValues;
  }
  
//  _poissonSolution->writeFieldsToFile(PoissonBilinearForm::PHI,"phi_preRef.m");
  vector<GlobalIndexType> quadCellsToRefine;
  quadCellsToRefine.push_back(1);
  mesh->registerSolution(_poissonSolution);
//  vector< Teuchos::RCP<Solution> > solutions;
//  solutions.push_back(_poissonSolution);
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad());
  
  GnuPlotUtil::writeComputationalMeshSkeleton("/tmp/poissonRefinedMesh",mesh);
//  _poissonSolution->writeFieldsToFile(PoissonBilinearForm::PHI,"phi_postRef.m");
  
//  _poissonSolution->writeFieldsToFile(trialIDToWrite, filePrefix + "AfterRefinement" + fileSuffix);
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(actualValues,fieldID,_testPoints);
    double maxDiff = 0;
    if ( ! fcsAgree(expectedMap[fieldID],actualValues,tol,maxDiff) ) {
      success = false;
      cout << "testHRefinementInitialization failed: max difference in " 
           << _poissonSolution->mesh()->bilinearForm()->trialName(fieldID) << " is " << maxDiff << endl;
      
      FieldContainer<double> points = _testPoints;
      int numPoints = _testPoints.dimension(0);
      points.resize(1,numPoints,_testPoints.dimension(1));
      FieldContainer<double> expected = expectedMap[fieldID];
      expected.resize(1,numPoints);
      FieldContainer<double> actual = actualValues;
      actual.resize(1,numPoints);
      reportFunctionValueDifferences(points, expected, actual, tol);
    }
  }

//  _poissonSolution->solve(false);
//  _poissonSolution->writeFieldsToFile(PoissonBilinearForm::PHI,"phi_postSolve.m");
  
  return success;
}


bool SolutionTests::testPRefinementInitialization() {
  
  double tol = 1e-14;
  
  bool success = true;
  
  _poissonSolution->solve(false);
  
  // test for all field variables:
  vector<int> fieldIDs = _poissonSolution->mesh()->bilinearForm()->trialVolumeIDs();
  
  map<int, FieldContainer<double> > expectedMap;
  
  FieldContainer<double> expectedValues(_testPoints.dimension(0)); 
  FieldContainer<double> actualValues(_testPoints.dimension(0)); 
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(expectedValues,fieldID,_testPoints);
    expectedMap[fieldID] = expectedValues;
//    cout << "expectedValues:\n" << expectedValues;
  }
  
  vector<GlobalIndexType> quadCellsToRefine;
  quadCellsToRefine.push_back(0); // just refine first cell  
  
  _poissonSolution->mesh()->registerSolution(_poissonSolution);
  _poissonSolution->mesh()->pRefine(quadCellsToRefine);
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(actualValues,fieldID,_testPoints);
    double maxDiff;
    if ( ! fcsAgree(expectedMap[fieldID],actualValues,tol,maxDiff) ) {
      success = false;
      cout << "testHRefinementInitialization failed: max difference in " 
      << _poissonSolution->mesh()->bilinearForm()->trialName(fieldID) << " is " << maxDiff << endl;
    }
  }
  
  return success;
}

bool SolutionTests::testSolutionEvaluationBasisCache() {
  bool success = true;
  double tol = 1e-12;
  
  // remap _testPoints from (0,1)^2 to (-1,1)^2:
  int numPoints = _testPoints.dimension(0);
  for (int i=0; i<numPoints; i++) {
    double x = _testPoints(i,0);
    double y = _testPoints(i,1);
    _testPoints(i,0) = x * 2.0 - 1.0;
    _testPoints(i,1) = y * 2.0 - 1.0;
  }
  
  ElementPtr elem = _poissonSolution_1x1->mesh()->getElement(0); // "spectral" mesh--just one element
  int cellID = elem->cellID();
  vector<GlobalIndexType> cellIDs(1,cellID);
  BasisCachePtr basisCache = Teuchos::rcp( new BasisCache(elem->elementType(), _poissonSolution_1x1->mesh()) );
  
  basisCache->setRefCellPoints( _testPoints ); // don't use cubature points...
//  cout << "physicalCellNodes:\n" << _poissonSolution_1x1->mesh()->physicalCellNodesForCell(cellID);
  basisCache->setPhysicalCellNodes( _poissonSolution_1x1->mesh()->physicalCellNodesForCell(cellID),
                                   cellIDs, true); // true: create side cache, too

  FieldContainer<double> testPoints = basisCache->getPhysicalCubaturePoints();
  
  map<int, FieldContainer<double> > expectedMap;
  
  int numCells = testPoints.dimension(0);
  
  FieldContainer<double> expectedValues(numCells,numPoints); 
  FieldContainer<double> actualValues(numCells,numPoints); 
  
  vector<int> fieldIDs = _poissonSolution_1x1->mesh()->bilinearForm()->trialVolumeIDs();
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonExactSolution->solutionValues(expectedValues,fieldID,testPoints);
    expectedMap[fieldID] = expectedValues;
  }
    
  // test for all field variables:
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution_1x1->solutionValues(actualValues,fieldID,basisCache);
    double maxDiff;
    expectedValues = expectedMap[fieldID];
    if ( ! fcsAgree(expectedValues,actualValues,tol,maxDiff) ) {
      success = false;
      cout << "testSolutionEvaluationBasisCache failed: max difference in " 
      << _poissonSolution_1x1->mesh()->bilinearForm()->trialName(fieldID) << " is " << maxDiff << endl;
      
      cout << "Expected:\n" << expectedValues;
      cout << "Actual:\n" << actualValues;
    }
  }
  
  // TODO: test for side caches, too (both traces & fluxes and fields restricted to sides...)
  
  return success;
}

bool SolutionTests::testScratchPadSolution() {

  bool success = true;
  double tol = 1e-12;
  double eps = .1;
  
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
  
  ////////////////////   DEFINE INNER PRODUCT(S)   ///////////////////////

  // robust test norm
  IPPtr robIP = Teuchos::rcp(new IP);

  robIP->addTerm( v);
  robIP->addTerm( sqrt(eps) * v->grad() );
  robIP->addTerm( beta * v->grad() );
  robIP->addTerm( tau->div() );
  robIP->addTerm( Function::constant(1.0)/sqrt(eps) * tau );
  
  ////////////////////   SPECIFY RHS   ///////////////////////

  FunctionPtr zero = Function::constant(0.0);
  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  FunctionPtr f = zero;
  rhs->addTerm( f * v ); // obviously, with f = 0 adding this term is not necessary!

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr squareBoundary = Teuchos::rcp( new UnitSquareBoundary );

  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  FunctionPtr one = Teuchos::rcp( new ConstantScalarFunction(1.0) );
  bc->addDirichlet(uhat, squareBoundary, one);

  ////////////////////   BUILD MESH   ///////////////////////
  // define nodes for mesh
  int H1Order = 1; int pToAdd = 1;
  
  FieldContainer<double> quadPoints(4,2);
  
  quadPoints(0,0) = 0.0; // x1
  quadPoints(0,1) = 0.0; // y1
  quadPoints(1,0) = 1.0;
  quadPoints(1,1) = 0.0;
  quadPoints(2,0) = 1.0;
  quadPoints(2,1) = 1.0;
  quadPoints(3,0) = 0.0;
  quadPoints(3,1) = 1.0;
 
  int nCells = 2;
  int horizontalCells = nCells, verticalCells = nCells;
  
  // create a pointer to a new mesh:
  Teuchos::RCP<Mesh> mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells,
                                                confusionBF, H1Order, H1Order+pToAdd);
    
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution;
  solution = Teuchos::rcp( new Solution(mesh, bc, rhs, robIP) );
  solution->solve(false);
  double uL2Norm = solution->L2NormOfSolutionGlobal(u->ID());
  double sigma1L2Norm = solution->L2NormOfSolutionGlobal(sigma1->ID()); // should be 0
  double sigma2L2Norm = solution->L2NormOfSolutionGlobal(sigma2->ID()); // shoudl be 0
  double L1L2Norm = uL2Norm + sigma1L2Norm + sigma2L2Norm;

  if (abs(L1L2Norm-1.0)>tol){
    success = false;
  }  
  return success;
}



bool SolutionTests::testCondensationSolve() {

  bool success = true;
  double tol = 1e-12;

  int numProcs=1;
  int rank=0;

#ifdef HAVE_MPI
  rank     = Teuchos::GlobalMPISession::getRank();
  numProcs = Teuchos::GlobalMPISession::getNProc();
//  Epetra_MpiComm Comm(MPI_COMM_WORLD);
//  //cout << "rank: " << rank << " of " << numProcs << endl;
#else
//  Epetra_SerialComm Comm;
#endif
  
  vector<double> beta;
  beta.push_back(1.0);
  beta.push_back(0.0);
 
  ////////////////////   DECLARE VARIABLES   ///////////////////////
  // define test variables
  VarFactory varFactory; 
  VarPtr v = varFactory.testVar("v", HGRAD);
  
  // define trial variables
  VarPtr beta_n_u= varFactory.fluxVar("\\widehat{\\beta \\cdot n u");
  VarPtr u = varFactory.fieldVar("u");
  
  ////////////////////   DEFINE BILINEAR FORM/IP   ///////////////////////

  BFPtr bf = Teuchos::rcp( new BF(varFactory) );
  // tau terms:
  bf->addTerm( -u, beta * v->grad() );
  bf->addTerm( beta_n_u, v);
  
  // robust test norm
  IPPtr ip = bf->graphNorm();
  
  ////////////////////   SPECIFY RHS   ///////////////////////

  Teuchos::RCP<RHSEasy> rhs = Teuchos::rcp( new RHSEasy );
  rhs->addTerm( 1.0* v ); 

  ////////////////////   CREATE BCs   ///////////////////////
  Teuchos::RCP<BCEasy> bc = Teuchos::rcp( new BCEasy );
  SpatialFilterPtr inflow = Teuchos::rcp( new InflowBoundary );

  FunctionPtr n = Teuchos::rcp( new UnitNormalFunction );

  bc->addDirichlet(beta_n_u, inflow, Function::constant(1.0)*beta*n);

  ////////////////////   BUILD MESH   ///////////////////////

  int H1Order = 3; int pToAdd = 3; int nCells = 4;  
  Teuchos::RCP<Mesh> mesh = MeshUtilities::buildUnitQuadMesh(nCells, bf, H1Order, H1Order+pToAdd);
    
  ////////////////////   SOLVE & REFINE   ///////////////////////

  Teuchos::RCP<Solution> solution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  Teuchos::RCP<Solution> condensationSolution = Teuchos::rcp( new Solution(mesh, bc, rhs, ip) );
  solution->solve(false);
  condensationSolution->solve(false);
  FunctionPtr uF = Function::solution(u,solution);
  FunctionPtr uCond = Function::solution(u,condensationSolution);
  double diff = (uF-uCond)->l2norm(mesh,H1Order);
  if (diff>tol){
    cout << "Failing test: Condensation solve does not match regular solve" << endl;
    success=false;
  }

  return success;
}

bool SolutionTests::testSolutionsAreConsistent() {
  bool success = true;
  
  if (! solutionCoefficientsAreConsistent(_confusionSolution1_2x2) ) {
    success = false;
    cout << "_confusionSolution1_2x2 coefficients are inconsistent.\n";
  }
  
  if (! solutionCoefficientsAreConsistent(_confusionSolution2_2x2) ) {
    success = false;
    cout << "_confusionSolution1_2x2 coefficients are inconsistent.\n";
  }
  
  return success;
}
