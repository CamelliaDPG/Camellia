#ifndef CAMELLIA_PATCH_BASIS_TESTS
#define CAMELLIA_PATCH_BASIS_TESTS

#include "PatchBasis.h"

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

typedef Basis<double, FieldContainer<double> > DoubleBasis;
typedef Teuchos::RCP< DoubleBasis > BasisPtr;

typedef Teuchos::RCP< PatchBasis > PatchBasisPtr;

class PatchBasisTests : public TestSuite {
private:
  FieldContainer<double> _testPoints1D;
  FieldContainer<double> _testPoints1DLeftParent, _testPoints1DMiddleParent, _testPoints1DRightParent;
  BasisPtr _parentBasis;
  PatchBasisPtr _patchBasisLeft, _patchBasisMiddle, _patchBasisRight;
  
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "PatchBasisTests."; }
  bool testPatchBasis1D(); // 1D patches are all that's supported right now (suffices for 2D DPG.)
};


#endif