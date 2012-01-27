#ifndef CAMELLIA_TEST_SUITE
#define CAMELLIA_TEST_SUITE

using namespace std;
#include <string>

// abstract class for tests
class TestSuite {
public:
  virtual void runTests(int &numTestsRun, int &numTestsPassed) = 0;
  virtual string testSuiteName() = 0;

};

#endif