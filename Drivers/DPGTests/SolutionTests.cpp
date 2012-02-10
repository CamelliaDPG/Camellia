
#include "SolutionTests.h"

#include "Intrepid_FieldContainer.hpp"
#include "Mesh.h"

#include "ConfusionBilinearForm.h"
#include "ConfusionManufacturedSolution.h"
#include "PoissonBilinearForm.h"
#include "PoissonExactSolution.h"

#include "MathInnerProduct.h"
#include "SimpleFunction.h"

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
  
  // h-convergence
  int sqrtElements = 2;
  
  double epsilon = 1e-2;
  double beta_x = 1.0, beta_y = 1.0;
  _confusionExactSolution = Teuchos::rcp( new ConfusionManufacturedSolution(epsilon,beta_x,beta_y) ); 
  
  bool useConformingTraces = true;
  int polyOrder = 2;
  _poissonExactSolution = 
    Teuchos::rcp( new PoissonExactSolution(PoissonExactSolution::POLYNOMIAL, 
					   polyOrder, useConformingTraces) );  
  _poissonExactSolution->setUseSinglePointBCForPHI(false); // impose zero-mean constraint

  int H1Order = 3;
  int horizontalCells = 2; int verticalCells = 2;
  
  // before we hRefine, compute a solution for comparison after refinement
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp(new MathInnerProduct(_confusionExactSolution->bilinearForm()));
  Teuchos::RCP<Mesh> mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _confusionExactSolution->bilinearForm(), H1Order, H1Order+1);

  Teuchos::RCP<Mesh> poissonMesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, _poissonExactSolution->bilinearForm(), H1Order, H1Order+2);
  Teuchos::RCP<DPGInnerProduct> poissonIp = Teuchos::rcp(new MathInnerProduct(_poissonExactSolution->bilinearForm()));

  _confusionSolution1_2x2 = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->bc(), _confusionExactSolution->ExactSolution::rhs(), ip) );
  _confusionSolution2_2x2 = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->bc(), _confusionExactSolution->ExactSolution::rhs(), ip) );
  _poissonSolution = Teuchos::rcp( new Solution(poissonMesh, _poissonExactSolution->bc(),_poissonExactSolution->ExactSolution::rhs(), ip));
  _confusionUnsolved = Teuchos::rcp( new Solution(mesh, _confusionExactSolution->bc(), _confusionExactSolution->ExactSolution::rhs(), ip) );

  _confusionSolution1_2x2->solve();
  _confusionSolution2_2x2->solve();
  _poissonSolution->solve();
  
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  double y[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  
  _testPoints = FieldContainer<double>(NUM_POINTS_1D*NUM_POINTS_1D,2);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    for (int j=0; j<NUM_POINTS_1D; j++) {
      _testPoints(i*NUM_POINTS_1D + j, 0) = x[i];
      _testPoints(i*NUM_POINTS_1D + j, 1) = y[i];
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
  if (testHRefinementInitialization()) {
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
  
}

bool SolutionTests::storageSizesAgree(Teuchos::RCP< Solution > soln1, Teuchos::RCP< Solution > soln2) {
  const map< int, FieldContainer<double> >* solnMap1 = &(soln1->solutionForCellIDGlobal());
  const map< int, FieldContainer<double> >* solnMap2 = &(soln2->solutionForCellIDGlobal());
  if (solnMap1->size() != solnMap2->size() ) {
    cout << "SOLUTION 1 entries: ";
    for(map< int, FieldContainer<double> >::const_iterator soln1It = (*solnMap1).begin();
        soln1It != (*solnMap1).end(); soln1It++) {
      int cellID = soln1It->first;
      cout << cellID << " ";
    }
    cout << "\n";
    cout << "SOLUTION 2 entries: ";
    for(map< int, FieldContainer<double> >::const_iterator soln2It = (*solnMap2).begin();
        soln2It != (*solnMap2).end(); soln2It++) {
      int cellID = soln2It->first;
      cout << cellID << " ";
    }
    cout << "\n";
    
    return false;
  }
  for(map< int, FieldContainer<double> >::const_iterator soln1It = (*solnMap1).begin();
      soln1It != (*solnMap1).end(); soln1It++) {
    int cellID = soln1It->first;
    int size = soln1It->second.size();
    map< int, FieldContainer<double> >::const_iterator soln2It = (*solnMap2).find(cellID);
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
  Teuchos::RCP<SimpleFunction> simpleFunction = Teuchos::rcp(new SimpleFunction());
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  functionMap[ConfusionBilinearForm::U] = simpleFunction;
  functionMap[ConfusionBilinearForm::SIGMA_1] = simpleFunction;
  functionMap[ConfusionBilinearForm::SIGMA_2] = simpleFunction;

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
  simpleFunction->getValues(functionValues,allCellTestPoints);
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



bool SolutionTests::testAddRefinedSolutions(){
  bool success = true;
  double tol = 1e-14;

  Teuchos::RCP<SimpleFunction> simpleFunction = Teuchos::rcp(new SimpleFunction());
  map<int, Teuchos::RCP<AbstractFunction> > functionMap;
  functionMap[ConfusionBilinearForm::U] = simpleFunction;
  functionMap[ConfusionBilinearForm::SIGMA_1] = simpleFunction;
  functionMap[ConfusionBilinearForm::SIGMA_2] = simpleFunction;
  _confusionSolution2_2x2->projectOntoMesh(functionMap);  // pretend confusionSolution1 is the linearized solution

  // solve
  _confusionSolution1_2x2->solve(false);

  // add the two solutions together
  _confusionSolution2_2x2->addSolution(_confusionSolution1_2x2,1.0);    // pretend confusionSolution2 is the accumulated solution

  // refine the mesh
  vector<int> quadCellsToRefine;
  quadCellsToRefine.push_back(0); // refine first cell
  vector< Teuchos::RCP<Solution> > solutions;
  solutions.push_back(_confusionSolution1_2x2);
  solutions.push_back(_confusionSolution2_2x2);
  _confusionSolution1_2x2->mesh()->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad(),solutions);

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
  map<int, double> energyError;
  _poissonSolution->energyError(energyError);
  vector< Teuchos::RCP< Element > > activeElements = _poissonSolution->mesh()->activeElements();
  vector< Teuchos::RCP< Element > >::iterator activeElemIt;
  
  double totalEnergyErrorSquared = 0.0;
  for (activeElemIt = activeElements.begin();activeElemIt != activeElements.end(); activeElemIt++){
    Teuchos::RCP< Element > current_element = *(activeElemIt);
    int cellID = current_element->cellID();
    totalEnergyErrorSquared += energyError[cellID]*energyError[cellID];
  }
  if (totalEnergyErrorSquared>tol){
    success = false;
    cout << "testEnergyError failed: energy error is " << totalEnergyErrorSquared << endl;
  }
  
  
  return success;
}


bool SolutionTests::testHRefinementInitialization(){

  double tol = 1e-14;

  bool success = true;
  Teuchos::RCP< Mesh > mesh = _poissonSolution->mesh();
  
  _poissonSolution->solve(false);
  int trialIDToWrite = PoissonBilinearForm::PHI;
  string filePrefix = "phi";
  string fileSuffix = ".m";
  _poissonSolution->writeFieldsToFile(trialIDToWrite, filePrefix + "BeforeRefinement" + fileSuffix);
  
  // test for all field variables:
  vector<int> fieldIDs = _poissonSolution->mesh()->bilinearForm().trialVolumeIDs();
  
  map<int, FieldContainer<double> > expectedMap;
  
  FieldContainer<double> expectedValues(_testPoints.dimension(0)); 
  FieldContainer<double> actualValues(_testPoints.dimension(0)); 
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(expectedValues,fieldID,_testPoints);
    expectedMap[fieldID] = expectedValues;
  }
  
  _poissonSolution->writeFieldsToFile(PoissonBilinearForm::PHI,"phi_preRef.m");
  vector<int> quadCellsToRefine;
  quadCellsToRefine.push_back(1);
  vector< Teuchos::RCP<Solution> > solutions;
  solutions.push_back(_poissonSolution);
  mesh->hRefine(quadCellsToRefine,RefinementPattern::regularRefinementPatternQuad(),solutions); // passing in solution to reinitialize stuff
  _poissonSolution->writeFieldsToFile(PoissonBilinearForm::PHI,"phi_postRef.m");
  
  _poissonSolution->writeFieldsToFile(trialIDToWrite, filePrefix + "AfterRefinement" + fileSuffix);
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(actualValues,fieldID,_testPoints);
    double maxDiff;
    if ( ! fcsAgree(expectedMap[fieldID],actualValues,tol,maxDiff) ) {
      success = false;
      cout << "testHRefinementInitialization failed: max difference in " 
           << _poissonSolution->mesh()->bilinearForm().trialName(fieldID) << " is " << maxDiff << endl;
    }
  }

  _poissonSolution->solve(false);
  _poissonSolution->writeFieldsToFile(PoissonBilinearForm::PHI,"phi_postSolve.m");
  
  return success;
}


bool SolutionTests::testPRefinementInitialization() {
  
  double tol = 1e-14;
  
  bool success = true;
  
  _poissonSolution->solve(false);
  
  // test for all field variables:
  vector<int> fieldIDs = _poissonSolution->mesh()->bilinearForm().trialVolumeIDs();
  
  map<int, FieldContainer<double> > expectedMap;
  
  FieldContainer<double> expectedValues(_testPoints.dimension(0)); 
  FieldContainer<double> actualValues(_testPoints.dimension(0)); 
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(expectedValues,fieldID,_testPoints);
    expectedMap[fieldID] = expectedValues;
//    cout << "expectedValues:\n" << expectedValues;
  }
  
  vector<int> quadCellsToRefine;
  quadCellsToRefine.push_back(0); // just refine first cell  
  vector< Teuchos::RCP<Solution> > solutions;
  solutions.push_back(_poissonSolution);
  _poissonSolution->mesh()->pRefine(quadCellsToRefine,solutions); // passing in solution to reinitialize stuff
  
  for (vector<int>::iterator fieldIDIt=fieldIDs.begin(); fieldIDIt != fieldIDs.end(); fieldIDIt++) {
    int fieldID = *fieldIDIt;
    _poissonSolution->solutionValues(actualValues,fieldID,_testPoints);
    double maxDiff;
    if ( ! fcsAgree(expectedMap[fieldID],actualValues,tol,maxDiff) ) {
      success = false;
      cout << "testHRefinementInitialization failed: max difference in " 
      << _poissonSolution->mesh()->bilinearForm().trialName(fieldID) << " is " << maxDiff << endl;
    }
  }
  
  return success;
}
