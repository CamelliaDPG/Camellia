#include "gtest/gtest.h"

#include "Epetra_MpiComm.h"
#ifdef HAVE_MPI
#include "Epetra_MpiComm.h"
#else
#include "Epetra_SerialComm.h"
#endif

using namespace ::testing;

class ParallelUnitTestPrinter : public TestEventListener {
 public:
  ParallelUnitTestPrinter(int _commRank, int _numProcs, TestEventListener* _default_printer) :
    commRank(_commRank), numProcs(_numProcs), default_printer(_default_printer) {}

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
  // Epetra_MpiComm Comm;
  int commRank;
  int numProcs;
  TestEventListener* default_printer;

  static void PrintFailedTests(const UnitTest& unit_test);

  static bool allSuccess(bool mySuccess);

  internal::String test_case_name_;
};
