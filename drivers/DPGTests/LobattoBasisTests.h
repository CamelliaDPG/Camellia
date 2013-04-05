#ifndef Camellia_LobattoBasisTests_h
#define Camellia_LobattoBasisTests_h

#include "TestSuite.h"

// Teuchos includes
#include "Teuchos_RCP.hpp"

class LobattoBasisTests : public TestSuite {  
  void setup();
  void teardown();
public:
  void runTests(int &numTestsRun, int &numTestsPassed);
  
  bool testLegendreValues();
  
  bool testLobattoValues();
  bool testLobattoDerivativeValues();
  
  
  std::string testSuiteName();
};

#endif
