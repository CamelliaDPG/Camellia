#include "ParallelUnitTestPrinter.h"

#include "MPIWrapper.h"

#include <iostream>

void ParallelUnitTestPrinter::OnTestIterationStart(const UnitTest& unit_test, int iteration)
{
  if (commRank == 0)
    default_printer->OnTestIterationStart(unit_test, iteration);
}

void ParallelUnitTestPrinter::OnEnvironmentsSetUpStart(const UnitTest& unit_test)
{
  if (commRank == 0)
    default_printer->OnEnvironmentsSetUpStart(unit_test);
}

void ParallelUnitTestPrinter::OnTestCaseStart(const TestCase& test_case)
{
  if (commRank == 0)
    default_printer->OnTestCaseStart(test_case);
}

void ParallelUnitTestPrinter::OnTestStart(const TestInfo& test_info)
{
  if (commRank == 0)
    default_printer->OnTestStart(test_info);
}

void ParallelUnitTestPrinter::OnTestPartResult(const TestPartResult& result)
{
  for (int i = 0; i < numProcs; i++)
  {
    if (commRank == i)
    {
      default_printer->OnTestPartResult(result);
    }
  }
}

void ParallelUnitTestPrinter::OnTestEnd(const TestInfo& test_info)
{
#ifdef HAVE_MPI
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
#else
  Epetra_SerialComm Comm;
#endif

  if (allSuccess(test_info.result()->Passed()))
  {
    if (commRank == 0)
      default_printer->OnTestEnd(test_info);
  }
  else
  {
    for (int i=0; i < numProcs; i++)
    {
      if (i == commRank)
      {
        // cout << "Printing from " << commRank << endl;
        default_printer->OnTestEnd(test_info);
      }
      Comm.Barrier();
    }
  }

}

void ParallelUnitTestPrinter::OnTestCaseEnd(const TestCase& test_case)
{
  if (commRank == 0)
    default_printer->OnTestCaseEnd(test_case);
}

void ParallelUnitTestPrinter::OnEnvironmentsTearDownStart(const UnitTest& unit_test)
{
  if (commRank == 0)
    default_printer->OnEnvironmentsTearDownStart(unit_test);
}

void ParallelUnitTestPrinter::OnTestIterationEnd(const UnitTest& unit_test, int iteration)
{
  if (commRank == 0)
    default_printer->OnTestIterationEnd(unit_test, iteration);
}

bool ParallelUnitTestPrinter::allSuccess(bool mySuccess)
{
  int mySuccessInt = mySuccess ? 0 : -1;
  int successSum = MPIWrapper::sum(mySuccessInt);
  return successSum == 0;
}
