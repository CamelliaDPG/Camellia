#include "MPIUnitTestResultPrinter.h"

void MPIUnitTestResultPrinter::OnTestIterationStart(const UnitTest& unit_test, int iteration)
{
  if (commRank == 0)
    default_printer->OnTestIterationStart(unit_test, iteration);
}

void MPIUnitTestResultPrinter::OnEnvironmentsSetUpStart(const UnitTest& unit_test)
{
  if (commRank == 0)
    default_printer->OnEnvironmentsSetUpStart(unit_test);
}

void MPIUnitTestResultPrinter::OnTestCaseStart(const TestCase& test_case)
{
  if (commRank == 0)
    default_printer->OnTestCaseStart(test_case);
}

void MPIUnitTestResultPrinter::OnTestStart(const TestInfo& test_info)
{
  if (commRank == 0)
    default_printer->OnTestStart(test_info);
}

void MPIUnitTestResultPrinter::OnTestPartResult(const TestPartResult& result)
{
  if (commRank == 0)
    default_printer->OnTestPartResult(result);
}

void MPIUnitTestResultPrinter::OnTestEnd(const TestInfo& test_info)
{
  if (commRank == 0)
    default_printer->OnTestEnd(test_info);
}

void MPIUnitTestResultPrinter::OnTestCaseEnd(const TestCase& test_case)
{
  if (commRank == 0)
    default_printer->OnTestCaseEnd(test_case);
}

void MPIUnitTestResultPrinter::OnEnvironmentsTearDownStart(const UnitTest& unit_test)
{
  if (commRank == 0)
    default_printer->OnEnvironmentsTearDownStart(unit_test);
}

void MPIUnitTestResultPrinter::OnTestIterationEnd(const UnitTest& unit_test, int iteration)
{
  if (commRank == 0)
    default_printer->OnTestIterationEnd(unit_test, iteration);
}

