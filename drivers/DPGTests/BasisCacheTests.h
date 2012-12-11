#ifndef Camellia_BasisCacheTests_h
#define Camellia_BasisCacheTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

#include "Mesh.h"
#include "InnerProductScratchPad.h"
#include "BasisCache.h"

class BasisCacheTests : public TestSuite {  
  Teuchos::RCP<Mesh> _spectralConfusionMesh; // 1x1 mesh, H1 order = 1, pToAdd = 0
  Teuchos::RCP<BF> _confusionBF; // standard confusion bilinear form
  VarPtr _uhat_confusion; // confusion variable u_hat
  FieldContainer<double> _testPoints;
  ElementTypePtr _elemType;
  BasisCachePtr _basisCache;
  
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testSetRefCellPoints();
  
  std::string testSuiteName();
};

#endif
