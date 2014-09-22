//
//  CellTopologyTests.h
//  Camellia
//
//  Created by Nate Roberts on 9/16/14.
//
//

#ifndef __Camellia__CellTopologyTests__
#define __Camellia__CellTopologyTests__

#include "TestSuite.h"
#include "CellTopology.h"

class CellTopologyTests : public TestSuite {
private:
  void setup();
  void teardown();
  
  bool checkDimension(CellTopoPtr cellTopo);
  bool checkPermutationCount(CellTopoPtr cellTopo);
  bool checkPermutations(CellTopoPtr cellTopo); // true if all permutations are distinct and inverses do in fact invert
  
  vector< CellTopoPtr > _shardsTopologies; // populated on test construction
public:
  CellTopologyTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "CellTopologyTests"; }
  
  bool testShardsTopologiesPermutations();
  bool testOneTensorTopologiesPermutations(); // one tensorial dimension
  bool testMultiTensorTopologiesPermutations();
};


#endif /* defined(__Camellia__CellTopologyTests__) */
