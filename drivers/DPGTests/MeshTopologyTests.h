#ifndef CAMELLIA_NEW_MESH_TESTS
#define CAMELLIA_NEW_MESH_TESTS

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "MeshTopology.h"

class MeshTopologyTests : public TestSuite {
private:
  void setup();
  void teardown();
public:
  MeshTopologyTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "MeshTopologyTests"; }
  
  bool test1DMesh();
  bool test2DMesh();
  bool test3DMesh();
  
  bool testEntityConstraints();
  bool testCellsForEntity();
  bool testConstraintRelaxation();
  
  bool testNeighborRelationships();
};

#endif
