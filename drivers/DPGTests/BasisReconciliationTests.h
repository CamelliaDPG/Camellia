//
//  BasisReconciliationTests.h
//  Camellia-debug
//
//  Created by Nate Roberts on 11/19/13.
//
//

#ifndef Camellia_debug_BasisReconciliationTests_h
#define Camellia_debug_BasisReconciliationTests_h

#include "BasisReconciliation.h"

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"


class BasisReconciliationTests : public TestSuite {
private:
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "BasisReconciliationTests"; }
  
  bool testP();
  bool testPSide();
  bool testH();
  
private:
  bool pConstraintSideBasisSubTest(BasisPtr fineBasis, unsigned fineSideIndex, FieldContainer<double> &finePhysicalCellNodes,
                                   BasisPtr coarseBasis, unsigned coarseSideIndex, FieldContainer<double> &coarsePhysicalCellNodes);
  bool pConstraintWholeBasisSubTest(BasisPtr fineBasis, BasisPtr coarseBasis);
};


#endif
