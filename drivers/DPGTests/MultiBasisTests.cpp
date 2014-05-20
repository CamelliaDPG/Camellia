
#include "MultiBasisTests.h"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CellTools.hpp"

#include "BasisFactory.h"

#include "BilinearForm.h" // defines IntrepidExtendedTypes
#include "PoissonBilinearForm.h"

#include "BasisEvaluation.h"

#include "ConfusionManufacturedSolution.h"
#include "ConfusionProblemLegacy.h"
#include "ConfusionBilinearForm.h"
#include "ConfusionInnerProduct.h"

#include "MeshTestUtility.h" // used for checkMeshConsistency
#include "MeshTestSuite.h"
#include "MeshFactory.h"

#include "GDAMaximumRule2D.h"

#include "GnuPlotUtil.h"

typedef Teuchos::RCP< FieldContainer<double> > FCPtr;

// for some reason, we throw an exception (at least in debug mode) if we don't
// explicitly initialize the _mesh variable
MultiBasisTests::MultiBasisTests() : _mesh(Teuchos::rcp((Mesh *)NULL)) {}

void MultiBasisTests::runTests(int &numTestsRun, int &numTestsPassed) {
  
//  try {
//    setup();
//    if (testSolveUniformMesh()) {
//      numTestsPassed++;
//    }
//    numTestsRun++;
//    teardown();
  
  setup();
  if (testNeighborPRefinementSimple()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testChildPRefinementSimple()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testChildPRefinementMultiLevel()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testNeighborPRefinementMultiLevel()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testSimpleRefinement()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testMultiLevelRefinement()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  setup();
  if (testMultiBasisLegacyTest()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();

//  } catch (...) {
//    cout << "MultiBasisTests: caught exception while running tests.\n";
//    teardown();
//  }
}

void MultiBasisTests::reportPOrders(vector< map<int,int> > &polyOrders) {
  map<int,int> fieldOrders = polyOrders[0];
  for (map<int,int>::iterator fieldOrderIt=fieldOrders.begin(); fieldOrderIt != fieldOrders.end(); fieldOrderIt++) {
    cout << "var ID " << fieldOrderIt->first << ": " << fieldOrderIt->second << endl;
  }
  for (int sideOrdinal=0; sideOrdinal<polyOrders.size()-1; sideOrdinal++) {
    cout << "side ordinal " << sideOrdinal << ":" << endl;
    map<int, int> sideOrders = polyOrders[sideOrdinal+1];
    for (map<int,int>::iterator sideOrderIt=sideOrders.begin(); sideOrderIt != sideOrders.end(); sideOrderIt++) {
      cout << "var ID " << sideOrderIt->first << ": " << sideOrderIt->second << endl;
    }
  }
}

bool MultiBasisTests::basisValuesAgreeWithPermutedNeighbor(Teuchos::RCP<Mesh> mesh) {
  // for every side (MultiBasis or no), compute values for that side, and values for its neighbor along
  // the same physical points.  (Imitate the comparison between parent and child, only remember that
  // the neighbor involves a flip: (-1,1) --> (1,-1).)
 
  return MeshTestSuite::neighborBasesAgreeOnSides(mesh, _testPoints1D, true);
}

bool MultiBasisTests::doPRefinementAndTestIt(GlobalIndexType cellID, const string &testName) {
  bool success = true;
  
  ElementPtr elem = _mesh->getElement(cellID);
  
  if (elem->isChild()) {
    if ( ! childPolyOrdersAgreeWithParent(elem) ) {
      cout << testName << ": before refinement, parent and child don't agree on p-order.\n";
      return false;
    }
  }
  
  vector< map< int, int> > elemPOrdersBeforeRefinement; // includes all fields and fluxes
  getPolyOrders(elemPOrdersBeforeRefinement,elem);
  
//  cout << "trialOrdering for cell " << elem->cellID() << " before p-refinement:\n";
//  cout << *(elem->elementType()->trialOrderPtr);
  
  vector<GlobalIndexType> cellsToRefine;
  cellsToRefine.push_back(elem->cellID());
  _mesh->pRefine(cellsToRefine);
  
  elem = _mesh->getElement(cellID);
  
//  cout << "trialOrdering for cell " << elem->cellID() << " after p-refinement:\n";
//  cout << *(elem->elementType()->trialOrderPtr);
  
  if (elem->isChild()) {
    if ( ! childPolyOrdersAgreeWithParent(elem) ) {
      cout << testName << ": after refinement, parent and child don't agree on p-order.\n";
      return false;
    }
  }
  
  // (check both that p-refinement was done in child, and that meshLooksGood())
  vector< map< int, int> > elemPOrdersAfterRefinement; // map from varID to p-order
  getPolyOrders(elemPOrdersAfterRefinement,elem);
  
  if ( ! pRefined( elemPOrdersBeforeRefinement, elemPOrdersAfterRefinement ) ) {
    cout << testName << ": after p-refinement, element doesn't have increased p-order.\n";
    success = false;
    cout << "p-orders before refinement:\n";
    reportPOrders(elemPOrdersBeforeRefinement);
    cout << "p-orders after refinement:\n";
    reportPOrders(elemPOrdersAfterRefinement);
  }
  
  if ( !meshLooksGood() ) {
    success = false;
  }
  
  if ( !refinementsHaveNotIncreasedError() ) {
    success = false;
  }
  
  if ( !success ) {
    cout << "Failed " << testName << ".\n";
  }
  return success;
}

bool MultiBasisTests::childPolyOrdersAgreeWithParent(ElementPtr child) {
  vector< map< int, int> > elemPOrdersAlongSharedSidesBeforeRefinement; // map from varID to p-order
  vector< map< int, int> > parentPOrdersAlongSharedSidesBeforeRefinement;
  
  getPolyOrdersAlongSharedSides(elemPOrdersAlongSharedSidesBeforeRefinement,
                                parentPOrdersAlongSharedSidesBeforeRefinement,
                                child);
  return polyOrdersAgree( elemPOrdersAlongSharedSidesBeforeRefinement, parentPOrdersAlongSharedSidesBeforeRefinement );
}

void MultiBasisTests::getPolyOrders(vector< map<int, int> > &polyOrderMapVector, ElementPtr elem) {
  // polyOrderMapVector has the following maps (in the order given):
  // - fieldID -> p-order for field variable
  // - for each sideIndex: fluxID -> p-order for flux/trace variable
  polyOrderMapVector.clear();
  vector<int>::iterator varIt;
  map<int, int> polyOrders;
  for (varIt = _fieldIDs.begin(); varIt != _fieldIDs.end(); varIt++) {
    int fieldID = *varIt;
    int polyOrder = BasisFactory::basisPolyOrder(elem->elementType()->trialOrderPtr->getBasis(fieldID));
    polyOrders[fieldID] = polyOrder;
  }
  polyOrderMapVector.push_back(polyOrders);
  int numSides = elem->numSides();
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    polyOrders.clear();
    vector<int>::iterator varIt;
    for (varIt = _fluxIDs.begin(); varIt != _fluxIDs.end(); varIt++) {
      int fluxID = *varIt;
      int polyOrder =  BasisFactory::basisPolyOrder(elem->elementType()->trialOrderPtr->getBasis(fluxID,sideIndex));
      polyOrders[fluxID] = polyOrder;
    }
    polyOrderMapVector.push_back(polyOrders);
  }
}

void MultiBasisTests::getPolyOrdersAlongSharedSides(vector< map<int, int> > &childPOrderMapForSide,
                                                    vector< map<int, int> > &parentPOrderMapForSide,
                                                    ElementPtr child) {
  childPOrderMapForSide.clear();
  parentPOrderMapForSide.clear();
  ElementPtr parent = child->getParent();
  int numSides = child->numSides();
  for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
    int parentSideIndex = child->parentSideForSideIndex(sideIndex);
    if (parentSideIndex >= 0) { // they share this side
      map<int, int> childPOrdersForSide;
      map<int, int> parentPOrdersForSide;
      vector<int>::iterator varIt;
      for (varIt = _fluxIDs.begin(); varIt != _fluxIDs.end(); varIt++) {
        int fluxID = *varIt;
        int childPolyOrder = child->elementType()->trialOrderPtr->getBasis(fluxID,sideIndex)->getDegree();
        int parentPolyOrder = parent->elementType()->trialOrderPtr->getBasis(fluxID,parentSideIndex)->getDegree();
        childPOrdersForSide[fluxID] = childPolyOrder;
        parentPOrdersForSide[fluxID] = parentPolyOrder;
      }
      parentPOrderMapForSide.push_back(childPOrdersForSide);
      parentPOrderMapForSide.push_back(parentPOrdersForSide);
    }
  }
}

void MultiBasisTests::hRefineAllActiveCells(Teuchos::RCP<Mesh> mesh) {
  vector<GlobalIndexType> cellIDsToRefine;
  for (vector< ElementPtr >::const_iterator elemIt=mesh->activeElements().begin();
       elemIt != mesh->activeElements().end(); elemIt++) {
    int cellID = (*elemIt)->cellID();
    cellIDsToRefine.push_back(cellID);
  }
  mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());
}

void MultiBasisTests::makeSimpleRefinement() {
  vector<GlobalIndexType> cellIDsToRefine;
  //cout << "refining SW element (cellID " << _sw->cellID() << ")\n";
  cellIDsToRefine.push_back(_sw->cellID()); // this is cellID 0, as things are right now implemented
  // the next line will throw an exception in Mesh right now, because Mesh doesn't yet support MultiBasis
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());
}

void MultiBasisTests::makeMultiLevelRefinement() {
  makeSimpleRefinement();
  
  vector<GlobalIndexType> cellIDsToRefine;
  // now, find the southeast element in the refined element, and refine it
  // the southeast element should have (0.375, 0.125) at its center
  FieldContainer<double> point(1,2);
  point(0,0) = 0.375; point(0,1) = 0.125;
  ElementPtr elem = _mesh->elementsForPoints(point)[0];
  cellIDsToRefine.push_back(elem->cellID());
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());
}

bool MultiBasisTests::meshLooksGood() {
  bool looksGood = true;
  if ( !multiBasisCorrectlyAppliedInMesh(_mesh,_fluxIDs,_fieldIDs) ) {
    cout << "multiBasisCorrectlyAppliedInMesh returned false.\n";
    looksGood = false;
  }
//  if ( !patchBasesAgreeWithParentInMesh() ) {
//    cout << "patchBasesAgreeWithParentInMesh returned false.\n";
//    looksGood = false;
//  }
  if ( !MeshTestUtility::checkMeshConsistency(_mesh) ) {
    cout << "MeshTestUtility::checkMeshConsistency() returned false.\n";
    looksGood = false;
  }
  if ( !basisValuesAgreeWithPermutedNeighbor(_mesh) ) {
    cout << "basisValuesAgreeWithPermutedNeighbor returned false.\n";
    looksGood = false;
  }
  return looksGood;
}

bool MultiBasisTests::multiBasisCorrectlyAppliedInMesh(Teuchos::RCP<Mesh> mesh, vector<int> fluxIDs, vector<int> fieldIDs) {
  // checks that the right elements have some  in the right places
  vector< ElementPtr > activeElements = mesh->activeElements();
  
  // depending on our debugging needs, could revise this to return more information
  // about the nature and extent of the incorrectness when correct == false.
  
  bool correct = true;
  
  vector< ElementPtr >::iterator elemIt;
  for (elemIt = activeElements.begin(); elemIt != activeElements.end(); elemIt++) {
    ElementPtr elem = *elemIt;
    vector<int>::iterator varIt;
    for (varIt = fluxIDs.begin(); varIt != fluxIDs.end(); varIt++) {
      int fluxID = *varIt;
      int numSides = elem->numSides();
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        BasisPtr basis = elem->elementType()->trialOrderPtr->getBasis(fluxID,sideIndex);
        bool hasMultiBasis = BasisFactory::isMultiBasis(basis);
        bool shouldHaveMultiBasis;
        // check who the (ancestor's) neighbor is on this side:
        int sideIndexInNeighbor;
        ElementPtr neighbor = mesh->ancestralNeighborForSide(elem,sideIndex,sideIndexInNeighbor);
        
        if (neighbor.get() != NULL) {
          // then we'll check *neighbor's* basis instead:
          BasisPtr neighborBasis = neighbor->elementType()->trialOrderPtr->getBasis(fluxID,sideIndexInNeighbor);
          hasMultiBasis = BasisFactory::isMultiBasis(neighborBasis);
        }
        
        // check whether the neighbor relationship is symmetric:
        if (neighbor.get() == NULL) {
          shouldHaveMultiBasis = false;
        } else if (neighbor->getNeighborCellID(sideIndexInNeighbor) != elem->cellID()) {
          // i.e. neighbor's neighbor is our parent/ancestor--so we should have a MultiBasis
          shouldHaveMultiBasis = true;
        } else {
          shouldHaveMultiBasis = false;
        }
        if (shouldHaveMultiBasis != hasMultiBasis) {
          correct = false;
        }
      }
    }
    for (varIt = fieldIDs.begin(); varIt != fieldIDs.end(); varIt++) {
      int fieldID = *varIt;
      bool shouldHaveMultiBasis = false; // false for all fields
      BasisPtr basis = elem->elementType()->trialOrderPtr->getBasis(fieldID);
      bool hasMultiBasis = BasisFactory::isMultiBasis(basis);
      if (shouldHaveMultiBasis != hasMultiBasis) {
        correct = false;
      }
    }
  }
  return correct;
}
  
bool MultiBasisTests::polyOrdersAgree(const vector< map<int, int> > &pOrderMapVector1,
                                      const vector< map<int, int> > &pOrderMapVector2) {
  vector< map<int, int> >::const_iterator mapVectorIt1;
  vector< map<int, int> >::const_iterator mapVectorIt2 = pOrderMapVector2.begin();
  for (mapVectorIt1 = pOrderMapVector1.begin(); mapVectorIt1 != pOrderMapVector1.end(); mapVectorIt1++) {
    map<int, int> map1 = *mapVectorIt1;
    map<int, int> map2 = *mapVectorIt2;
    map<int, int>::iterator map1It;
    for (map1It=map1.begin(); map1It != map1.end(); map1It++) {
      pair<int,int> entry = *map1It;
      if (map2[entry.first] != entry.second) {
        return false;
      }
    }
    mapVectorIt2++;
  }
  return true;
}

void MultiBasisTests::pRefineAllActiveCells() {
  vector<GlobalIndexType> cellIDsToRefine;
  for (vector< ElementPtr >::const_iterator elemIt=_mesh->activeElements().begin();
       elemIt != _mesh->activeElements().end(); elemIt++) {
    int cellID = (*elemIt)->cellID();
    cellIDsToRefine.push_back(cellID);
  }
  _mesh->pRefine(cellIDsToRefine);
}

bool MultiBasisTests::pRefined(const vector< map<int, int> > &pOrderMapForSideBefore,
                               const vector< map<int, int> > &pOrderMapForSideAfter) {
  vector< map<int, int> >::const_iterator beforeVectorIt;
  vector< map<int, int> >::const_iterator afterVectorIt = pOrderMapForSideAfter.begin();
  for (beforeVectorIt = pOrderMapForSideBefore.begin(); beforeVectorIt != pOrderMapForSideBefore.end(); beforeVectorIt++) {
    map<int, int> beforeMap = *beforeVectorIt;
    map<int, int> afterMap = *afterVectorIt;
    map<int, int>::iterator beforeMapIt;
    for (beforeMapIt=beforeMap.begin(); beforeMapIt != beforeMap.end(); beforeMapIt++) {
      pair<int,int> entry = *beforeMapIt;
      int sideOrdinal = entry.first;
      int pOrderAfter = afterMap[sideOrdinal];
      int pOrderBefore = entry.second;
      if (pOrderAfter != (pOrderBefore + 1)) {
        return false;
      }
    }
    afterVectorIt++;
  }  
  return true;
}

void MultiBasisTests::setup() {
  
  _useMumps = false; // false because Jesse reports trouble with MUMPS

  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.99};
  
  _testPoints1D = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    _testPoints1D(i, 0) = x[i];
  }
  
  /**** SUPPORT FOR TESTS THAT MULTIBASIS IS CORRECTLY ASSIGNED WITHIN MESH *****/
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
  
  int H1Order = 3;
  int delta_p = 3; // for tests
  int horizontalCells = 2; int verticalCells = 2;
  
  double eps = 1.0; // not really testing for sharp gradients right now--just want to see if things basically work
  double beta_x = 1.0;
  double beta_y = 1.0;
  // _confusionExactSolution = Teuchos::rcp( new ConfusionManufacturedSolution(eps,beta_x,beta_y) );
  
  Teuchos::RCP<ConfusionBilinearForm> confusionBF = Teuchos::rcp( new ConfusionBilinearForm(eps,beta_x,beta_y) );
  
  Teuchos::RCP<ConfusionProblemLegacy> confusionProblem = Teuchos::rcp( new ConfusionProblemLegacy(confusionBF) );

//  Teuchos::RCP<ConfusionBilinearForm> confusionBF = Teuchos::rcp( (ConfusionBilinearForm*) _confusionExactSolution->bilinearForm.get(), false); // false: doesn't own the memory, since the RCP _confusionExactSolution does that);

  _mesh = MeshFactory::buildQuadMesh(quadPoints, horizontalCells, verticalCells, confusionBF, H1Order, H1Order+delta_p);
  
  Teuchos::RCP<DPGInnerProduct> ip = Teuchos::rcp( new ConfusionInnerProduct( confusionBF, _mesh ) );
  
  _confusionSolution = Teuchos::rcp( new Solution(_mesh, confusionProblem, confusionProblem, ip) );
  
  // the right way to determine the southwest element, etc. is as follows:
  FieldContainer<double> points(4,2);
  // southwest center:
  points(0,0) = 0.25; points(0,1) = 0.25;
  // southeast center:
  points(1,0) = 0.75; points(1,1) = 0.25;
  // northwest center:
  points(2,0) = 0.25; points(2,1) = 0.75;
  // northeast center:
  points(3,0) = 0.75; points(3,1) = 0.75;
  vector<ElementPtr> elements = _mesh->elementsForPoints(points);
  
  _sw = elements[0];
  _se = elements[1];
  _nw = elements[2];
  _ne = elements[3];
  
//  cout << "SW nodes:\n" << _mesh->physicalCellNodesForCell(_sw->cellID());
//  cout << "SE nodes:\n" << _mesh->physicalCellNodesForCell(_se->cellID());
//  cout << "NW nodes:\n" << _mesh->physicalCellNodesForCell(_nw->cellID());
//  cout << "NE nodes:\n" << _mesh->physicalCellNodesForCell(_ne->cellID());
  
  _confusionSolution->solve(_useMumps);
  
//  for (vector<int>::iterator fieldIt=_fieldIDs.begin(); fieldIt != _fieldIDs.end(); fieldIt++) {
//    int fieldID = *fieldIt;
//    double err = _confusionExactSolution->L2NormOfError(*(_confusionSolution.get()),fieldID);
//    _confusionL2ErrorForOriginalMesh[fieldID] = err;
//  }
  
  _confusionEnergyErrorForOriginalMesh = _confusionSolution->energyErrorTotal();
  
//  _confusionSolution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_multiBasis_before_refinement.m");
  
  _mesh->setUsePatchBasis(false);
  
  _fluxIDs = confusionBF->trialBoundaryIDs();
  _fieldIDs = confusionBF->trialVolumeIDs();
  
}

bool MultiBasisTests::refinementsHaveNotIncreasedError() {
  return refinementsHaveNotIncreasedError(_confusionSolution);
}

bool MultiBasisTests::refinementsHaveNotIncreasedError(Teuchos::RCP<Solution> solution) {
  double tol = 1e-11;
  
  bool success = true;
  
  solution->solve(_useMumps);

  double err = _confusionSolution->energyErrorTotal();
  double diff = err - _confusionEnergyErrorForOriginalMesh;
  if (diff > tol) {
    cout << "MultiBasisTests: increase in error after refinement " << diff << " > tol " << tol << ".\n";
    
    solution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_multiBasis.m");
    solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "confusion_u_hat_multiBasis.m");
    
    success = false;
  }
  
//  for (vector<int>::iterator fieldIt=_fieldIDs.begin(); fieldIt != _fieldIDs.end(); fieldIt++) {
//    int fieldID = *fieldIt;
//    double err = _confusionExactSolution->L2NormOfError(*(_confusionSolution.get()),fieldID);
//    double originalErr = _confusionL2ErrorForOriginalMesh[fieldID];
//    if (err - originalErr > tol) {
//      cout << "MultiBasisTests: increase in error after refinement " << err - originalErr << " > tol " << tol << " for ";
//      cout << _confusionExactSolution->bilinearForm()->trialName(fieldID) << endl;
//      
//      solution->writeFieldsToFile(ConfusionBilinearForm::U, "confusion_u_multiBasis.m");
//      solution->writeFluxesToFile(ConfusionBilinearForm::U_HAT, "confusion_u_hat_multiBasis.m");
//      
//      success = false;
//    }
//  }
  
  return success;
}


void MultiBasisTests::teardown() {
  _parentBasis = Teuchos::rcp((Camellia::Basis<> *)NULL);
  
  _mesh = Teuchos::rcp((Mesh *)NULL);
  _sw = Teuchos::rcp((Element *)NULL);
  _se = Teuchos::rcp((Element *)NULL);
  _nw = Teuchos::rcp((Element *)NULL);
  _ne = Teuchos::rcp((Element *)NULL);
}

bool MultiBasisTests::testMultiBasisLegacyTest() {
  // this test copied from the old DPGTests method.  Not yet integrated with the other tests...
  // (I.e. we should make this use setup and the mesh that's there, etc.)
  
  // 1. create trialOrdering for side
  // 2. make MultiBasis for a side along a side broken in 2, with trialOrdering in each
  // 3. test that MultiBasis agrees at the vertices
  bool success = true;
  double tol = 1e-15;
  
  BFPtr bilinearForm = PoissonBilinearForm::poissonBilinearForm();
  
  int polyOrder = 2; 
  Teuchos::RCP<DofOrdering> trialOrdering;
  shards::CellTopology quad_4(shards::getCellTopologyData<shards::Quadrilateral<4> >() );
  
  DofOrderingFactory dofOrderingFactory(bilinearForm);
  
  trialOrdering = dofOrderingFactory.trialOrdering(polyOrder, quad_4, true);
  
  // suppose that the broken element is on the west side of the big element
  int parentSideIndexInNeighbor = 3;
  int numChildren = 2;
  
  vector< pair< Teuchos::RCP<DofOrdering>, int> > childTrialOrdersForSide;
  for (int childIndex=0; childIndex<numChildren; childIndex++) {
    int childSideIndex = 1;
    childTrialOrdersForSide.push_back(make_pair(trialOrdering,childSideIndex));
  }
  
  Teuchos::RCP<DofOrdering> mbTrialOrdering = trialOrdering;
  
  dofOrderingFactory.assignMultiBasis( mbTrialOrdering, parentSideIndexInNeighbor, 
                                      quad_4, childTrialOrdersForSide );
  
  VarPtr phi_hat = PoissonBilinearForm::poissonBilinearForm()->varFactory().traceVar(PoissonBilinearForm::S_PHI_HAT);
  
  int trialID = phi_hat->ID(), sideIndex = 1;
  int numFields = trialOrdering->getBasisCardinality(trialID,sideIndex), numPoints = 1;
  int spaceDim = 1; // along sides...
  // TODO: make the two trialOrderings different... (or maybe just add a test with multiple MB levels)
  FieldContainer<double> values1(numFields,numPoints);
  FieldContainer<double> points1(numPoints,spaceDim);
  FieldContainer<double> values2(numFields,numPoints);
  FieldContainer<double> points2(numPoints,spaceDim);
  points1(0,0) = -0.75;
  //  points1(1,0) = -0.25;
  //  points1(2,0) = 0.75;
  points2(0,0) = -0.35;
  //  points2(1,0) = -0.15;
  //  points2(2,0) = 0.40;
  int numMBPoints = numChildren*numPoints;
  int numMBFields = mbTrialOrdering->getBasisCardinality(trialID,parentSideIndexInNeighbor);
  if (numMBFields != 2*numFields) {
    success = false;
    cout << "FAILURE: in testMultiBasis, MB doesn't have the expected cardinality\n";
  }
  FieldContainer<double> mbValues(numMBFields,numMBPoints);
  FieldContainer<double> mbPoints(numMBPoints,spaceDim);
  
  for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
    mbPoints(pointIndex,0) = -(points1(pointIndex,0) - 1.0) / 2.0;
    mbPoints(pointIndex + numPoints,0) = -(points2(pointIndex,0) + 1.0) / 2.0;
  }
  
  trialOrdering->getBasis(trialID,sideIndex)->getValues(values1,points1,Intrepid::OPERATOR_VALUE);
  trialOrdering->getBasis(trialID,sideIndex)->getValues(values2,points2,Intrepid::OPERATOR_VALUE);
  mbTrialOrdering->getBasis(trialID,parentSideIndexInNeighbor)->getValues(mbValues,mbPoints,Intrepid::OPERATOR_VALUE);
  for (int fieldIndex=0; fieldIndex<numFields; fieldIndex++) {
    for (int pointIndex=0; pointIndex<numPoints; pointIndex++) {
      for (int childIndex=0; childIndex<numChildren; childIndex++) { 
        int childSubSideIndexInMB = GDAMaximumRule2D::neighborChildPermutation(childIndex, numChildren);
        FieldContainer<double> values = (childIndex == 0) ? values1 : values2;
        int permutedFieldIndex = GDAMaximumRule2D::neighborDofPermutation(fieldIndex,numFields); // within the subbasis
        int mbPointIndex = childIndex*numPoints + pointIndex;
        MultiBasis<>* multiBasis = (MultiBasis<>*) mbTrialOrdering->getBasis(trialID,parentSideIndexInNeighbor).get();
        //      TODO: figure out which of the following is right, and fix whatever bug(s) are implied--
        //        the uncommented one is the one that (mostly) works--only failure is on the shared dof, and we know
        //        whats up with that one--we need to allow it to be nonzero on the second side...
        int mbFieldIndex = multiBasis->relativeToAbsoluteDofOrdinal(permutedFieldIndex,childSubSideIndexInMB);
        //        int mbFieldIndex = multiBasis->relativeToAbsoluteDofOrdinal(fieldIndex,childIndex);
        double diff = abs( mbValues(mbFieldIndex, mbPointIndex) - values(fieldIndex,pointIndex) );
        if (diff > tol) {
          success = false;
          cout << "FAILURE: in testMultiBasis, MB and sub-basis differ by " << diff << " for child " << childIndex << ", point " << pointIndex;
          cout << ", fieldIndex " << fieldIndex << endl;
          cout << mbValues(mbFieldIndex, mbPointIndex) << " != " << values(fieldIndex,pointIndex) << endl;
          cout << "(mbFieldIndex, mbPointIndex) = (" << mbFieldIndex << ", " << mbPointIndex << "); ";
          cout << "child " << childIndex << ": (fieldIndex, pointIndex) = (" << fieldIndex << ", " << pointIndex << ")\n";
        }
      }
    }
  }
  if ( ! success ) {
    cout << "MultiBasis values:\n" << mbValues;
    cout << "sub-basis 1 values:\n" << values1;
    cout << "sub-basis 2 values:\n" << values2;
  }  
  return success;
}

bool MultiBasisTests::testSimpleRefinement() {
  // refine in the sw, and then check that the right elements have PatchBases
  bool success = true;
  makeSimpleRefinement();
  
  if ( !meshLooksGood() || (! refinementsHaveNotIncreasedError()) ) {
    success = false;
    cout << "Failed testSimpleRefinement.\n";
  }
  
  return success;
}

bool MultiBasisTests::testMultiLevelRefinement() {
  // refine in the sw, then refine in its se, and check the mesh
  bool success = true;
  makeMultiLevelRefinement();
  
  if ( !meshLooksGood() || (! refinementsHaveNotIncreasedError())) {
    success = false;
    cout << "Failed testMultiLevelRefinement.\n";
  }
  
  return success;
}

bool MultiBasisTests::testChildPRefinementSimple() {
  // in same mesh as the simple h-refinement test, p-refine the child.  Check that its parent also gets p-refined...
  makeSimpleRefinement();
    
  // the child we'd like to p-refine is the upper-right quadrant of the lower-left cell of the original mesh.
  // since we're on a unit square, that element contains the point (0.375, 0.375)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.375; cellPoint(0,1) = 0.375;
  ElementPtr child = _mesh->elementsForPoints(cellPoint)[0];
  
  return doPRefinementAndTestIt(child->cellID(),"testChildPRefinementSimple");
}

bool MultiBasisTests::testChildPRefinementMultiLevel() { 
  // in same mesh as the multi-level h-refinement test, p-refine the child.  Check that its parent and grandparent also get p-refined...
  makeMultiLevelRefinement();
  
  // the child we'd like to p-refine is NE quad. of the SE quad. of the SW element of the original mesh.
  // since we're on a unit square, that element contains the point (0.4375, 0.1875)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.4375; cellPoint(0,1) = 0.1875;
  ElementPtr child = _mesh->elementsForPoints(cellPoint)[0];
  
  return doPRefinementAndTestIt(child->cellID(),"testChildPRefinementMultiLevel");
}

bool MultiBasisTests::testNeighborPRefinementSimple() {
  // in same mesh as the simple h-refinement test, p-refine a big neighbor.  Check that its parent also gets p-refined...
  makeSimpleRefinement();
  
  // the neighbor we'd like to p-refine is SE quad of the original mesh
  // since we're on a unit square, that element contains the point (0.75, 0.25)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.75; cellPoint(0,1) = 0.25;
  ElementPtr neighbor = _mesh->elementsForPoints(cellPoint)[0];
  
//  GnuPlotUtil::writeComputationalMeshSkeleton("MultiBasisPRefinementMesh", _mesh, true); // true: label cells
  
  return doPRefinementAndTestIt(neighbor->cellID(),"testNeighborPRefinementSimple");
}

bool MultiBasisTests::testNeighborPRefinementMultiLevel() {
  // in same mesh as the multi-level h-refinement test, p-refine a big neighbor.  Check that its parent and grandparent also get p-refined...
  makeMultiLevelRefinement();
  
  // the neighbor we'd like to p-refine is SE quad of the original mesh
  // since we're on a unit square, that element contains the point (0.75, 0.25)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.75; cellPoint(0,1) = 0.25;
  ElementPtr neighbor = _mesh->elementsForPoints(cellPoint)[0];
  
  return doPRefinementAndTestIt(neighbor->cellID(),"testNeighborPRefinementMultiLevel");
} 

bool MultiBasisTests::testSolveUniformMesh() {
  // TODO: write this test, and make it a proper test... (Right now, a container for debug code!)

  cout << "MultiBasisTests::testSolveUniformMesh() not yet implemented." << endl;
  return false;
}