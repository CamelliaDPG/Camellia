
#include "PatchBasisTests.h"

#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_CellTools.hpp"

#include "BasisFactory.h"

#include "BilinearForm.h" // defines IntrepidExtendedTypes
#include "StokesBilinearForm.h"
#include "BasisEvaluation.h"

#include "MeshTestSuite.h" // used for checkMeshConsistency

typedef Teuchos::RCP< FieldContainer<double> > FCPtr;

// for some reason, we throw an exception (at least in debug mode) if we don't
// explicitly initialize the _mesh variable
PatchBasisTests::PatchBasisTests() : _mesh(Teuchos::rcp((Mesh *)NULL)) {}

void PatchBasisTests::runTests(int &numTestsRun, int &numTestsPassed) {
  setup();
  if (testPatchBasis1D()) {
    numTestsPassed++;
  }
  numTestsRun++;
  teardown();
  
  try {
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
    
    // for now, disable the p-refinement tests:
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
    if (testNeighborPRefinementSimple()) {
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
  } catch (...) {
    cout << "PatchBasisTests: caught exception while running tests.\n";
    teardown();
  }
}

bool PatchBasisTests::doPRefinementAndTestIt(ElementPtr elem, const string &testName) {
  bool success = true;
  
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
  
  vector<int> cellsToRefine;
  cellsToRefine.push_back(elem->cellID());
  _mesh->pRefine(cellsToRefine);
  
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
    cout << testName << ": after p-refinement, child doesn't have increased p-order.\n";
    success = false;
  }
  
  if ( !meshLooksGood() ) {
    success = false;
  }
  
  if ( !success ) {
    cout << "Failed " << testName << ".\n";
  }
  return success;
}

bool PatchBasisTests::childPolyOrdersAgreeWithParent(ElementPtr child) {
  vector< map< int, int> > elemPOrdersAlongSharedSidesBeforeRefinement; // map from varID to p-order
  vector< map< int, int> > parentPOrdersAlongSharedSidesBeforeRefinement;
  
  getPolyOrdersAlongSharedSides(elemPOrdersAlongSharedSidesBeforeRefinement,
                                parentPOrdersAlongSharedSidesBeforeRefinement,
                                child);
  return polyOrdersAgree( elemPOrdersAlongSharedSidesBeforeRefinement, parentPOrdersAlongSharedSidesBeforeRefinement );
}

void PatchBasisTests::getPolyOrders(vector< map<int, int> > &polyOrderMapVector, ElementPtr elem) {
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

void PatchBasisTests::getPolyOrdersAlongSharedSides(vector< map<int, int> > &childPOrderMapForSide,
                                                    vector< map<int, int> > &parentPOrderMapForSide,
                                                    ElementPtr child) {
  childPOrderMapForSide.clear();
  parentPOrderMapForSide.clear();
  Element* parent = child->getParent();
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

void PatchBasisTests::makeSimpleRefinement() {
  vector<int> cellIDsToRefine;
  //cout << "refining SW element (cellID " << _sw->cellID() << ")\n";
  cellIDsToRefine.push_back(_sw->cellID()); // this is cellID 0, as things are right now implemented
  // the next line will throw an exception in Mesh right now, because Mesh doesn't yet support PatchBasis
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());
}

void PatchBasisTests::makeMultiLevelRefinement() {
  makeSimpleRefinement();
  
  vector<int> cellIDsToRefine;
  // now, find the southeast element in the refined element, and refine it
  // the southeast element should have (0.375, 0.125) at its center
  FieldContainer<double> point(1,2);
  point(0,0) = 0.375; point(0,1) = 0.125;
  ElementPtr elem = _mesh->elementsForPoints(point)[0];
  cellIDsToRefine.push_back(elem->cellID());
  _mesh->hRefine(cellIDsToRefine,RefinementPattern::regularRefinementPatternQuad());
}

bool PatchBasisTests::meshLooksGood() {
  bool looksGood = true;
  if ( !patchBasisCorrectlyAppliedInMesh() ) {
    cout << "patchBasisCorrectlyAppliedInMesh returned false.\n";
    looksGood = false;
  }
  if ( !patchBasesAgreeWithParentInMesh() ) {
    cout << "patchBasesAgreeWithParentInMesh returned false.\n";
    looksGood = false;
  }
  if ( !MeshTestSuite::checkMeshConsistency(*_mesh) ) {
    cout << "MeshTestSuite::checkMeshConsistency() returned false.\n";
    looksGood = false;
  }
  return looksGood;
}

bool PatchBasisTests::patchBasisCorrectlyAppliedInMesh() {
  // checks that the right elements have some PatchBasis in the right places
  vector< ElementPtr > activeElements = _mesh->activeElements();
  
  // depending on our debugging needs, could revise this to return more information
  // about the nature and extent of the incorrectness when correct == false.
  
  bool correct = true;
  
  vector< ElementPtr >::iterator elemIt;
  for (elemIt = activeElements.begin(); elemIt != activeElements.end(); elemIt++) {
    ElementPtr elem = *elemIt;
    vector<int>::iterator varIt;
    for (varIt = _fluxIDs.begin(); varIt != _fluxIDs.end(); varIt++) {
      int fluxID = *varIt;
      int numSides = elem->numSides();
      for (int sideIndex=0; sideIndex<numSides; sideIndex++) {
        BasisPtr basis = elem->elementType()->trialOrderPtr->getBasis(fluxID,sideIndex);
        bool hasPatchBasis = BasisFactory::isPatchBasis(basis);
        bool shouldHavePatchBasis;
        // check who the (ancestor's) neighbor is on this side:
        int sideIndexInNeighbor;
        ElementPtr neighbor = _mesh->ancestralNeighborForSide(elem,sideIndex,sideIndexInNeighbor);
        int neighborCellID = neighbor->cellID();
        
        // check whether the neighbor relationship is symmetric:
        if (neighborCellID == -1) {
          shouldHavePatchBasis = false;
        } else if (_mesh->getElement(neighborCellID)->getNeighborCellID(sideIndexInNeighbor) != elem->cellID()) {
          // i.e. neighbor's neighbor is our parent/ancestor--so we should have a PatchBasis
          shouldHavePatchBasis = true;
        } else {
          shouldHavePatchBasis = false;
        }
        if (shouldHavePatchBasis != hasPatchBasis) {
          correct = false;
        }
      }
    }
    for (varIt = _fieldIDs.begin(); varIt != _fieldIDs.end(); varIt++) {
      int fieldID = *varIt;
      bool shouldHavePatchBasis = false; // false for all fields
      BasisPtr basis = elem->elementType()->trialOrderPtr->getBasis(fieldID);
      bool hasPatchBasis = BasisFactory::isPatchBasis(basis);
      if (shouldHavePatchBasis != hasPatchBasis) {
        correct = false;
      }
    }
  }
  return correct;
}
  
bool PatchBasisTests::patchBasesAgreeWithParentInMesh() {
  // checks that those elements with PatchBases compute values that agree with their parents 
  
  // iterate through all elements (including inactive!), looking for those that have PatchBases.
  //  - anytime a PatchBasis is found, take the _testPoints1D as the subcell reference for the PatchBasis
  //  - check that parent and child agree along the shared edge to within a very small tolerance (1e-15, say)
  
  double tol = 1e-15;
  bool valuesAgree = true;
  int numElements = _mesh->numElements();
  for (int cellID=0; cellID < numElements; cellID++) {
    ElementPtr elem = _mesh->getElement(cellID);
    
    Teuchos::RCP< DofOrdering > trialOrdering = elem->elementType()->trialOrderPtr;
    vector<int> varIDs = trialOrdering->getVarIDs();
    for (vector<int>::iterator varIDIt = varIDs.begin(); varIDIt != varIDs.end(); varIDIt++) {
      int varID = *varIDIt;
      int numSides = trialOrdering->getNumSidesForVarID(varID);
      for (int sideIndex = 0; sideIndex < numSides; sideIndex++) {
        BasisPtr basis = trialOrdering->getBasis(varID,sideIndex);
        if (BasisFactory::isPatchBasis(basis)) {
          // get parent basis:
          int parentSideIndex = elem->parentSideForSideIndex(sideIndex);
          Element* parent = elem->getParent();
          BasisPtr parentBasis = parent->elementType()->trialOrderPtr->getBasis(varID, parentSideIndex);
        
          FieldContainer<double> childCellNodes = _mesh->physicalCellNodesForCell(cellID);
          FieldContainer<double> parentCellNodes = _mesh->physicalCellNodesForCell(parent->cellID());
        
          /*
           It appears that there's no way (built in) to go from 2D points along the edge to the 1D ref cell
           So we want to keep things in 1D points.  The basic idea here is to treat the parent's edge as the
           "physical" space.  The trick is to figure out what the child's edge nodes should be inside the
           parent.  Because our edge divisions can be assumed to be in exactly two pieces wherever they are divided, 
           we can look for the shared vertex between parent and child along the side.  If the shared vertex is at the
           "earlier" vertex for parent (where earlier is understood modulo the side: so on the side 3 of a quad,
           vertex 3 is earlier than vertex 0), then the edge nodes are -1 and 0.  Otherwise, they are 0 and -1.
           
           This assumes a fair bit about the nature of our cell topologies, etc.  This is a test in which
           we control all that.  It's less clear what we should do if we wanted a similar feature in the core
           code, or to write a similar test for a more general topology or refinement pattern.
           */
          FieldContainer<double> childEdgeNodesInParentRef(1,2,1);
          int childIndexInParentSide = elem->indexInParentSide(parentSideIndex);
          if (childIndexInParentSide == 0) {
            childEdgeNodesInParentRef(0,0,0) = -1;
            childEdgeNodesInParentRef(0,1,0) = 0;
          } else if (childIndexInParentSide == 1) {
            childEdgeNodesInParentRef(0,0,0) = 0;
            childEdgeNodesInParentRef(0,1,0) = 1;
          } else {
            TEST_FOR_EXCEPTION( true, std::invalid_argument, "indexInParentSide isn't 0 or 1" );
          }
          
          FieldContainer<double> parentTestPoints(_testPoints1D);
          shards::CellTopology line_2(shards::getCellTopologyData<shards::Line<2> >() );
          CellTools<double>::mapToPhysicalFrame(parentTestPoints,_testPoints1D,childEdgeNodesInParentRef,line_2,0);
          
          // evaluate testPoints and parentTestPoints in respective bases
          FCPtr parentValues = BasisEvaluation::getValues(parentBasis,IntrepidExtendedTypes::OPERATOR_VALUE,parentTestPoints);
          FCPtr childValues =  BasisEvaluation::getValues(basis,IntrepidExtendedTypes::OPERATOR_VALUE,_testPoints1D);
          
          // check that they agree
          TEST_FOR_EXCEPTION(parentValues->size() != childValues->size(), std::invalid_argument,
                             "parentValues and childValues don't have the same size--perhaps parentBasis and child don't have the same order?");
          for (int i=0; i<parentValues->size(); i++) {
            double diff = abs((*parentValues)[i]-(*childValues)[i]);
            if (diff > tol) {
              cout << "For child cellID " << cellID << " on side " << sideIndex << ", ";
              cout << "parent value != childValue (" << (*parentValues)[i] << " != " << (*childValues)[i] << ")\n";
              valuesAgree = false;
            }
          }
        }
      }
    }
  }
  
  return valuesAgree; // unimplemented
}

bool PatchBasisTests::polyOrdersAgree(const vector< map<int, int> > &pOrderMapVector1,
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

bool PatchBasisTests::pRefined(const vector< map<int, int> > &pOrderMapForSideBefore,
                               const vector< map<int, int> > &pOrderMapForSideAfter) {
  vector< map<int, int> >::const_iterator beforeVectorIt;
  vector< map<int, int> >::const_iterator afterVectorIt = pOrderMapForSideAfter.begin();
  for (beforeVectorIt = pOrderMapForSideBefore.begin(); beforeVectorIt != pOrderMapForSideBefore.end(); beforeVectorIt++) {
    map<int, int> beforeMap = *beforeVectorIt;
    map<int, int> afterMap = *afterVectorIt;
    map<int, int>::iterator beforeMapIt;
    for (beforeMapIt=beforeMap.begin(); beforeMapIt != beforeMap.end(); beforeMapIt++) {
      pair<int,int> entry = *beforeMapIt;
      int sideIndex = entry.first;
      int pOrderAfter = afterMap[sideIndex];
      int pOrderBefore = entry.second;
      if (pOrderAfter != (pOrderBefore + 1)) {
        return false;
      }
    }
    afterVectorIt++;
  }  
  return true;
}

void PatchBasisTests::setup() {
  
  /**** SUPPORT FOR TESTS THAT PATCHBASIS COMPUTES THE CORRECT VALUES *****/
  // for tests, we'll do a simple division of a line segment into thirds
  // (for now, PatchBasis only supports 1D bases--sufficient for 2D DPG meshes)
  // setup bases:
  int polyOrder = 3;
  _parentBasis = BasisFactory::getBasis( polyOrder, shards::Line<2>::key, IntrepidExtendedTypes::FUNCTION_SPACE_HGRAD );
  FieldContainer<double> nodesLeft(2,1), nodesMiddle(2,1), nodesRight(2,1);
  nodesLeft(0,0)   = -1.0;
  nodesLeft(1,0)   = -1.0 / 3.0;
  nodesMiddle(0,0) = -1.0 / 3.0;
  nodesMiddle(1,0) = 1.0 / 3.0;
  nodesRight(0,0)  = 1.0 / 3.0;
  nodesRight(1,0)  = 1.0;
  _patchBasisLeft   = BasisFactory::getPatchBasis(_parentBasis,nodesLeft);
  _patchBasisMiddle = BasisFactory::getPatchBasis(_parentBasis,nodesMiddle);
  _patchBasisRight  = BasisFactory::getPatchBasis(_parentBasis,nodesRight);
  
  double refCellLeft = -1.0;
  double refCellRight = 1.0;
  
  // setup test points:
  static const int NUM_POINTS_1D = 10;
  double x[NUM_POINTS_1D] = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
  
  _testPoints1D = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    _testPoints1D(i, 0) = x[i];
  }
  
  _testPoints1DLeftParent   = FieldContainer<double>(NUM_POINTS_1D,1);
  _testPoints1DMiddleParent = FieldContainer<double>(NUM_POINTS_1D,1);
  _testPoints1DRightParent  = FieldContainer<double>(NUM_POINTS_1D,1);
  for (int i=0; i<NUM_POINTS_1D; i++) {
    double offset = (x[i] - refCellLeft) / 3.0;
    _testPoints1DLeftParent(i,0)   = -1.0       + offset;
    _testPoints1DMiddleParent(i,0) = -1.0 / 3.0 + offset;
    _testPoints1DRightParent(i,0)  =  1.0 / 3.0 + offset;
  }
  
  /**** SUPPORT FOR TESTS THAT PATCHBASIS IS CORRECTLY ASSIGNED WITHIN MESH *****/
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
  
  double mu = 1.0;
  Teuchos::RCP<BilinearForm> stokesBF = Teuchos::rcp(new StokesBilinearForm(mu) );
  
  _fluxIDs = stokesBF->trialBoundaryIDs();
  _fieldIDs = stokesBF->trialVolumeIDs();
  
  int H1Order = 3;
  int horizontalCells = 2; int verticalCells = 2;
  
  _mesh = Mesh::buildQuadMesh(quadPoints, horizontalCells, verticalCells, stokesBF, H1Order, H1Order+1);
  
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
  
  _mesh->setUsePatchBasis(true);
}

void PatchBasisTests::teardown() {
  _testPoints1D.resize(0);
  _testPoints1DLeftParent.resize(0);
  _testPoints1DMiddleParent.resize(0);
  _testPoints1DRightParent.resize(0);
  _parentBasis = Teuchos::rcp((DoubleBasis *)NULL);
  _patchBasisLeft = Teuchos::rcp((PatchBasis *)NULL);
  _patchBasisMiddle = Teuchos::rcp((PatchBasis *)NULL);
  _patchBasisRight = Teuchos::rcp((PatchBasis *)NULL);
  
  _mesh = Teuchos::rcp((Mesh *)NULL);
  _sw = Teuchos::rcp((Element *)NULL);
  _se = Teuchos::rcp((Element *)NULL);
  _nw = Teuchos::rcp((Element *)NULL);
  _ne = Teuchos::rcp((Element *)NULL);
  
}

bool PatchBasisTests::testPatchBasis1D() {
  bool success = true;
  
  double tol = 1e-15;
  int numPoints = _testPoints1D.size();
  int numFields = _parentBasis->getCardinality();
  FieldContainer<double> valuesLeft(numFields,numPoints),   expectedValuesLeft(numFields,numPoints);
  FieldContainer<double> valuesMiddle(numFields,numPoints), expectedValuesMiddle(numFields,numPoints);
  FieldContainer<double> valuesRight(numFields,numPoints),  expectedValuesRight(numFields,numPoints);
  
  // get the expected values
  _parentBasis->getValues(expectedValuesLeft,   _testPoints1DLeftParent,   Intrepid::OPERATOR_VALUE);
  _parentBasis->getValues(expectedValuesMiddle, _testPoints1DMiddleParent, Intrepid::OPERATOR_VALUE);
  _parentBasis->getValues(expectedValuesRight,  _testPoints1DRightParent,  Intrepid::OPERATOR_VALUE);
  
  // get the actual values:
  _patchBasisLeft  ->getValues(valuesLeft,   _testPoints1D, Intrepid::OPERATOR_VALUE);
  _patchBasisMiddle->getValues(valuesMiddle, _testPoints1D, Intrepid::OPERATOR_VALUE);
  _patchBasisRight ->getValues(valuesRight,  _testPoints1D, Intrepid::OPERATOR_VALUE);
  
  for (int fieldIndex=0; fieldIndex < numFields; fieldIndex++) {
    for (int pointIndex=0; pointIndex < numPoints; pointIndex++) {
      double diff = abs(valuesLeft(fieldIndex,pointIndex) - expectedValuesLeft(fieldIndex,pointIndex));
      if (diff > tol) {
        success = false;
        cout << "expected value of left basis: " << expectedValuesLeft(fieldIndex,pointIndex) << "; actual: " << valuesLeft(fieldIndex,pointIndex) << endl;
      }
      
      diff = abs(valuesMiddle(fieldIndex,pointIndex) - expectedValuesMiddle(fieldIndex,pointIndex));
      if (diff > tol) {
        success = false;
        cout << "expected value of middle basis: " << expectedValuesMiddle(fieldIndex,pointIndex) << "; actual: " << valuesMiddle(fieldIndex,pointIndex) << endl;
      }
      
      diff = abs(valuesRight(fieldIndex,pointIndex) - expectedValuesRight(fieldIndex,pointIndex));
      if (diff > tol) {
        success = false;
        cout << "expected value of right basis: " << expectedValuesRight(fieldIndex,pointIndex) << "; actual: " << valuesRight(fieldIndex,pointIndex) << endl;
      }
    }
  }
  
  return success;
}

bool PatchBasisTests::testSimpleRefinement() {
  // refine in the sw, and then check that the right elements have PatchBases
  bool success = true;
  makeSimpleRefinement();
  
  if ( !meshLooksGood() ) {
    success = false;
    cout << "Failed testSimpleRefinement.\n";
  }
  
  return success;
}

bool PatchBasisTests::testMultiLevelRefinement() {
  // refine in the sw, then refine in its se, and check the mesh
  bool success = true;
  makeMultiLevelRefinement();
  
  if ( !meshLooksGood() ) {
    success = false;
    cout << "Failed testMultiLevelRefinement.\n";
  }
  
  return success;
}

bool PatchBasisTests::testChildPRefinementSimple() {
  // in same mesh as the simple h-refinement test, p-refine the child.  Check that its parent also gets p-refined...
  makeSimpleRefinement();
  
  bool success = true;
  
  // the child we'd like to p-refine is the upper-right quadrant of the lower-left cell of the original mesh.
  // since we're on a unit square, that element contains the point (0.375, 0.375)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.375; cellPoint(0,1) = 0.375;
  ElementPtr child = _mesh->elementsForPoints(cellPoint)[0];
  
  return doPRefinementAndTestIt(child,"testChildPRefinementSimple");
}

bool PatchBasisTests::testChildPRefinementMultiLevel() { 
  // in same mesh as the multi-level h-refinement test, p-refine the child.  Check that its parent and grandparent also get p-refined...
  bool success = true;
  makeMultiLevelRefinement();
  
  // the child we'd like to p-refine is NE quad. of the SE quad. of the SW element of the original mesh.
  // since we're on a unit square, that element contains the point (0.4375, 0.1875)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.4375; cellPoint(0,1) = 0.1875;
  ElementPtr child = _mesh->elementsForPoints(cellPoint)[0];
  
  return doPRefinementAndTestIt(child,"testChildPRefinementMultiLevel");
}

bool PatchBasisTests::testNeighborPRefinementSimple() {
  // in same mesh as the simple h-refinement test, p-refine a big neighbor.  Check that its parent also gets p-refined...
  makeSimpleRefinement();
  
  // the neighbor we'd like to p-refine is SE quad of the original mesh
  // since we're on a unit square, that element contains the point (0.75, 0.25)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.75; cellPoint(0,1) = 0.25;
  ElementPtr neighbor = _mesh->elementsForPoints(cellPoint)[0];
  
  return doPRefinementAndTestIt(neighbor,"testNeighborPRefinementSimple");
}

bool PatchBasisTests::testNeighborPRefinementMultiLevel() {
  // in same mesh as the multi-level h-refinement test, p-refine a big neighbor.  Check that its parent and grandparent also get p-refined...
  makeMultiLevelRefinement();
  
  // the neighbor we'd like to p-refine is SE quad of the original mesh
  // since we're on a unit square, that element contains the point (0.75, 0.25)
  FieldContainer<double> cellPoint(1,2);
  cellPoint(0,0) = 0.75; cellPoint(0,1) = 0.25;
  ElementPtr neighbor = _mesh->elementsForPoints(cellPoint)[0];
  
  return doPRefinementAndTestIt(neighbor,"testNeighborPRefinementMultiLevel");
} 