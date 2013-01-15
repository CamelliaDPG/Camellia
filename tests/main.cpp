#include "gtest/gtest.h"
#include "ParallelUnitTestPrinter.h"

#include "Teuchos_GlobalMPISession.hpp"

// class MPIPrinter : public ::testing::internal::PrettyUnitTestResultPrinter {
// private:
// };

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  Teuchos::GlobalMPISession mpiSession(&argc, &argv,0);

  ::testing::UnitTest& unit_test = *::testing::UnitTest::GetInstance();
  ::testing::TestEventListeners& listeners = unit_test.listeners();
  ::testing::TestEventListener* default_printer = listeners.Release(listeners.default_result_printer());
  // listeners.SetDefaultResultPrinter(new ParallelUnitTestPrinter);
  int commRank = Teuchos::GlobalMPISession::getRank();
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  listeners.Append(new ParallelUnitTestPrinter(commRank, numProcs, default_printer) );

  return RUN_ALL_TESTS();
}
