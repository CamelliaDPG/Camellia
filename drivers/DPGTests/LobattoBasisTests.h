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
  
  bool testH1Classifications(); // checks that edge functions, vertex functions, etc. are correctly listed for the H^1 Lobatto basis
  
  std::string testSuiteName();
};

#endif
