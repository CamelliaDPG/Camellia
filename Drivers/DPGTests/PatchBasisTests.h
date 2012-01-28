#ifndef CAMELLIA_PATCH_BASIS_TESTS
#define CAMELLIA_PATCH_BASIS_TESTS

#include "PatchBasis.h"

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Mesh.h"

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
  
  // stuff for mesh/refinement tests
  Teuchos::RCP<Mesh> _mesh; // a 2x2 mesh set to use patchBasis
  ElementPtr _sw, _se, _nw, _ne;
  
  void setup();
  void teardown();
public:
  PatchBasisTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "PatchBasisTests"; }
  // 1D patches are all that's supported right now (suffices for 2D DPG.)
  bool testPatchBasis1D(); // check the correctness of the gimmicky divide-into-thirds PatchBasis

  bool testSimpleRefinement();  // refine in the sw, and then check that the right elements have PatchBases
  bool testMultiLevelRefinement(); // refine in the sw, and then in the se of the sw--check for multi-level PatchBasis, and correct values…
};


#endif