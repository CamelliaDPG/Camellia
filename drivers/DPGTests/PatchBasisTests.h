#ifndef CAMELLIA_PATCH_BASIS_TESTS
#define CAMELLIA_PATCH_BASIS_TESTS

#include "PatchBasis.h"

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Mesh.h"
#include "Solution.h"
#include "ExactSolution.h"

typedef Basis<double, FieldContainer<double> > DoubleBasis;
typedef Teuchos::RCP< DoubleBasis > BasisPtr;

typedef Teuchos::RCP< PatchBasis > PatchBasisPtr;
typedef Teuchos::RCP<Element> ElementPtr;

class PatchBasisTests : public TestSuite {
private:
  FieldContainer<double> _testPoints1D;
  FieldContainer<double> _testPoints1DLeftParent, _testPoints1DMiddleParent, _testPoints1DRightParent;
  BasisPtr _parentBasis;
  PatchBasisPtr _patchBasisLeft, _patchBasisMiddle, _patchBasisRight;
  
  bool _useMumps;
  
  // stuff for mesh/refinement tests
  Teuchos::RCP<Mesh> _mesh; // a 2x2 mesh with usePatchBasis==true
  ElementPtr _sw, _se, _nw, _ne;
  
  Teuchos::RCP<Solution> _confusionSolution;
  //Teuchos::RCP<ExactSolution> _confusionExactSolution;
  //  map<int, double> _confusionL2ErrorForOriginalMesh; // a baseline to compare against
  double _confusionEnergyErrorForOriginalMesh; // a baseline to compare against

  
  vector<int> _fluxIDs;
  vector<int> _fieldIDs;
  
  bool basisValuesAgreeWithPermutedNeighbor(Teuchos::RCP<Mesh> mesh);
  
  bool childPolyOrdersAgreeWithParent(ElementPtr child);
  
  void getPolyOrders(vector< map<int, int> > &polyOrderMapVector, ElementPtr elem);
  void getPolyOrdersAlongSharedSides(vector< map<int, int> > &childPOrderMapForSide,
                                     vector< map<int, int> > &parentPOrderMapForSide,
                                     ElementPtr child);
  
  static void hRefineAllActiveCells(Teuchos::RCP<Mesh>);
  
  bool meshLooksGood();
  
  static bool patchBasisCorrectlyAppliedInMesh(Teuchos::RCP<Mesh> mesh,vector<int> fluxIDs,vector<int> fieldIDs); // checks that the right elements have some PatchBasis in the right places
  bool patchBasesAgreeWithParentInMesh(); // checks that those elements with PatchBases compute values that agree with their parents
  
  bool polyOrdersAgree(const vector< map<int, int> > &polyOrderMapVector1,
                       const vector< map<int, int> > &polyOrderMapVector2);
  
  void pRefineAllActiveCells();
  
  bool pRefined(const vector< map<int, int> > &pOrderMapForSideBefore,
                const vector< map<int, int> > &pOrderMapForSideAfter);

  bool refinementsHaveNotIncreasedError();
  bool refinementsHaveNotIncreasedError(Teuchos::RCP<Solution> solution);
  
  bool doPRefinementAndTestIt(ElementPtr elem, const string &testName);
  
  void makeSimpleRefinement();
  void makeMultiLevelRefinement();
  
  void setup();
  void teardown();
public:
  PatchBasisTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "PatchBasisTests"; }
  // 1D patches are all that's supported right now (suffices for 2D DPG.)
  bool testPatchBasis1D(); // check the correctness of the gimmicky divide-into-thirds PatchBasis

  bool testSimpleRefinement();  // h-refine in the sw, and then check that the right elements have PatchBases
  bool testMultiLevelRefinement(); // h-refine in the sw, and then in the se of the sw--check for multi-level PatchBasis, and correct values…
  
  bool testChildPRefinementSimple(); // in same mesh as the simple h-refinement test, p-refine the child.  Check that its parent also gets p-refined...
  bool testChildPRefinementMultiLevel(); // in same mesh as the multi-level h-refinement test, p-refine the child.  Check that its parent and grandparent also get p-refined...
  
  bool testNeighborPRefinementSimple(); // in same mesh as the simple h-refinement test, p-refine a big neighbor.  Check that its parent also gets p-refined...
  bool testNeighborPRefinementMultiLevel(); // in same mesh as the multi-level h-refinement test, p-refine a big neighbor.  Check that its parent and grandparent also get p-refined...
  
  bool testSolveUniformMesh(); // in original, unrefined mesh: do we solve correctly?
};


#endif