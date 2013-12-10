#ifndef CAMELLIA_NEW_MESH_TESTS
#define CAMELLIA_NEW_MESH_TESTS

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "NewMesh.h"

class NewMeshTests : public TestSuite {
private:
  void setup();
  void teardown();
public:
  NewMeshTests();
  void runTests(int &numTestsRun, int &numTestsPassed);
  string testSuiteName() { return "NewMeshTests"; }
  
  bool test1DMesh();
  bool test2DMesh();
  bool test3DMesh();
  
  bool testEntityConstraints();
};

#endif
