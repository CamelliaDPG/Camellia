#include "gtest/gtest.h"

using namespace ::testing;

class MPIUnitTestResultPrinter : public TestEventListener {
 public:
  MPIUnitTestResultPrinter(int _commRank, TestEventListener* _default_printer) :
    commRank(_commRank), default_printer(_default_printer) {}

  // The following methods override what's in the TestEventListener class.
  virtual void OnTestProgramStart(const UnitTest& /*unit_test*/) {}
  virtual void OnTestIterationStart(const UnitTest& unit_test, int iteration);
  virtual void OnEnvironmentsSetUpStart(const UnitTest& unit_test);
  virtual void OnEnvironmentsSetUpEnd(const UnitTest& /*unit_test*/) {}
  virtual void OnTestCaseStart(const TestCase& test_case);
  virtual void OnTestStart(const TestInfo& test_info);
  virtual void OnTestPartResult(const TestPartResult& result);
  virtual void OnTestEnd(const TestInfo& test_info);
  virtual void OnTestCaseEnd(const TestCase& test_case);
  virtual void OnEnvironmentsTearDownStart(const UnitTest& unit_test);
  virtual void OnEnvironmentsTearDownEnd(const UnitTest& /*unit_test*/) {}
  virtual void OnTestIterationEnd(const UnitTest& unit_test, int iteration);
  virtual void OnTestProgramEnd(const UnitTest& /*unit_test*/) {}

 private:
  int commRank;
  TestEventListener* default_printer;
  static void PrintFailedTests(const UnitTest& unit_test);

  internal::String test_case_name_;
};
