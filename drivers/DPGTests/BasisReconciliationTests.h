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
  bool testHSide();
  
private:
  bool pConstraintSideBasisSubTest(BasisPtr fineBasis, unsigned fineSideIndex, FieldContainer<double> &finePhysicalCellNodes,
                                   BasisPtr coarseBasis, unsigned coarseSideIndex, FieldContainer<double> &coarsePhysicalCellNodes);
  bool pConstraintInternalBasisSubTest(BasisPtr fineBasis, BasisPtr coarseBasis);

  bool hConstraintSideBasisSubTest(BasisPtr fineBasis, unsigned fineSideIndex, FieldContainer<double> &fineCellAncestralNodes,
                                   RefinementBranch &volumeRefinements,
                                   BasisPtr coarseBasis, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes);
  bool hConstraintInternalBasisSubTest(BasisPtr fineBasis, RefinementBranch &refinements, BasisPtr coarseBasis);
  
  FieldContainer<double> permutedSidePoints(shards::CellTopology &sideTopo, FieldContainer<double> &pointsRefCell, unsigned permutation);
  
  FieldContainer<double> translateQuad(const FieldContainer<double> &quad, double x, double y);
  FieldContainer<double> translateHex(const FieldContainer<double> &hex, double x, double y, double z);

  unsigned vertexPermutation(shards::CellTopology &fineTopo, unsigned fineSideIndex, FieldContainer<double> &fineCellNodes,
                             shards::CellTopology &coarseTopo, unsigned coarseSideIndex, FieldContainer<double> &coarseCellNodes);
  
  void addDummyCellDimensionToFC(FieldContainer<double> &fc);
  void stripDummyCellDimensionFromFC(FieldContainer<double> &fc);
  
};


#endif
