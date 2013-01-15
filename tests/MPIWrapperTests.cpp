#include "MPIWrapperTests.h"
#include "TestUtil.h"

#include "Teuchos_GlobalMPISession.hpp"

bool allSuccess(bool mySuccess)
{
  int mySuccessInt = mySuccess ? 0 : -1;
  int successSum = MPIWrapper::sum(mySuccessInt);
  return successSum == 0;
}

void MPIWrapperTests::SetUp()
{
}

TEST_F(MPIWrapperTests, TestEntryWiseSum)
{
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int rank = Teuchos::GlobalMPISession::getRank();
  
  FieldContainer<double> expectedValues(2);
  for (int i=0; i<numProcs; i++) {
    expectedValues[0] += i*i;
    expectedValues[1] += 1;
  }
  FieldContainer<double> values(2);
  values[0] = rank*rank;
  values[1] = 1;
  
  MPIWrapper::entryWiseSum(values);
  double tol = 1e-16;
  
  double maxDiff = 0;
  EXPECT_TRUE(fcsAgree(values, expectedValues, tol, maxDiff))
    << "MPIWrapperTests::testentryWiseSum() failed with maxDiff " << maxDiff << endl;
  // EXPECT_EQ(rank, 2) << "rank " << rank << endl;
}

TEST_F(MPIWrapperTests, TestSimpleSum)
{
  int numProcs = Teuchos::GlobalMPISession::getNProc();
  int rank = Teuchos::GlobalMPISession::getRank();
  
  double expectedValue = 0;
  for (int i=0; i<numProcs; i++) {
    expectedValue += i;
  }
  FieldContainer<double> values(1);
  values[0] = rank;
  
  double sum = MPIWrapper::sum(values);
  double tol = 1e-16;
  EXPECT_LT(abs(sum-expectedValue), tol)
    << "MPIWrapperTests::testSimpleSum() failed: expected " << expectedValue << " but got " << sum << endl;
}
// 
// TEST_F(MPIWrapperTests, TestAllSuccess)
// {
//   int numProcs = Teuchos::GlobalMPISession::getNProc();
//   int rank = Teuchos::GlobalMPISession::getRank();
//   int val = 5;
//   if (rank == 3)
//     val = 0;
//   EXPECT_LT(val, 1) << "Wrong rank" << endl;
// }
