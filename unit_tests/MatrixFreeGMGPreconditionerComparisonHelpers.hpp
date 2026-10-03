// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//  MatrixFreeGMGPreconditionerComparisonHelpers.hpp
//  Camellia
//

#ifndef Camellia_MatrixFreeGMGPreconditionerComparisonHelpers_hpp
#define Camellia_MatrixFreeGMGPreconditionerComparisonHelpers_hpp

#include "MatrixFreeGMGPreconditioner.h"
#include "TypeDefs.h"

#include "Teuchos_Array.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace Camellia
{
namespace Test
{
  struct ApplyInverseTpetraComparisonResult
  {
    int applyInverseReturnCode = 0;
    double maxDifference = 0.0;
  };

  /** \brief Compare Epetra-style ApplyInverse preconditioner semantics with Tpetra apply semantics.

      GMGOperator::ApplyInverse(X,Y) has preconditioner semantics: it overwrites Y
      with the inverse/preconditioner application and supports callers seeding Y with
      arbitrary values.  Tpetra preconditioners express the same operation through
      apply(X,Y,NO_TRANS,1,0).  This helper intentionally compares only that common
      preconditioner contract; it does not require or imply full alpha/beta support
      from an ApplyInverse-style object.
   */
  struct StageTraceComparisonResult
  {
    bool sizeMatches = true;
    size_t firstMismatchIndex = 0;
    std::string firstMismatchField;
    double maxDifference = 0.0;
  };

  template <class LeftStageTraceEntry, class RightStageTraceEntry>
  inline double maxStageTraceEntryDifference(
      const LeftStageTraceEntry &left,
      const RightStageTraceEntry &right)
  {
    return std::max({std::abs(left.rhsInfNorm - right.rhsInfNorm),
                     std::abs(left.solutionInfNorm - right.solutionInfNorm),
                     std::abs(left.residualInfNorm - right.residualInfNorm)});
  }

  template <class ExpectedStageTraceEntry, class ActualStageTraceEntry>
  inline StageTraceComparisonResult compareStageTraces(
      const std::vector<ExpectedStageTraceEntry> &expected,
      const std::vector<ActualStageTraceEntry> &actual,
      const double tolerance = 1e-12)
  {
    StageTraceComparisonResult result;
    result.sizeMatches = expected.size() == actual.size();

    const size_t entryCount = std::min(expected.size(), actual.size());
    bool foundMismatch = false;
    for (size_t entryOrdinal=0; entryOrdinal<entryCount; entryOrdinal++)
    {
      const auto &expectedEntry = expected[entryOrdinal];
      const auto &actualEntry = actual[entryOrdinal];
      const double entryDifference = maxStageTraceEntryDifference(expectedEntry, actualEntry);
      result.maxDifference = std::max(result.maxDifference, entryDifference);

      if (!foundMismatch)
      {
        if (expectedEntry.levelIndex != actualEntry.levelIndex)
        {
          result.firstMismatchIndex = entryOrdinal;
          result.firstMismatchField = "levelIndex";
          foundMismatch = true;
        }
        else if (expectedEntry.stageName != actualEntry.stageName)
        {
          result.firstMismatchIndex = entryOrdinal;
          result.firstMismatchField = "stageName";
          foundMismatch = true;
        }
        else if (std::abs(expectedEntry.rhsInfNorm - actualEntry.rhsInfNorm) > tolerance)
        {
          result.firstMismatchIndex = entryOrdinal;
          result.firstMismatchField = "rhsInfNorm";
          foundMismatch = true;
        }
        else if (std::abs(expectedEntry.solutionInfNorm - actualEntry.solutionInfNorm) > tolerance)
        {
          result.firstMismatchIndex = entryOrdinal;
          result.firstMismatchField = "solutionInfNorm";
          foundMismatch = true;
        }
        else if (std::abs(expectedEntry.residualInfNorm - actualEntry.residualInfNorm) > tolerance)
        {
          result.firstMismatchIndex = entryOrdinal;
          result.firstMismatchField = "residualInfNorm";
          foundMismatch = true;
        }
      }
    }

    if (!foundMismatch && !result.sizeMatches)
    {
      result.firstMismatchIndex = entryCount;
      result.firstMismatchField = "size";
    }

    return result;
  }

  template <class ExpectedStageTraceEntry, class ActualStageTraceEntry>
  inline void assertStageTracesMatch(
      const std::vector<ExpectedStageTraceEntry> &expected,
      const std::vector<ActualStageTraceEntry> &actual,
      const double tolerance = 1e-12)
  {
    const StageTraceComparisonResult result = compareStageTraces(expected, actual, tolerance);
    TEUCHOS_TEST_FOR_EXCEPTION(!result.sizeMatches, std::runtime_error,
                               "Stage trace size mismatch at entry " << result.firstMismatchIndex);
    TEUCHOS_TEST_FOR_EXCEPTION(!result.firstMismatchField.empty(), std::runtime_error,
                               "Stage trace mismatch at entry " << result.firstMismatchIndex
                               << " field " << result.firstMismatchField
                               << " max difference " << result.maxDifference);
    TEUCHOS_TEST_FOR_EXCEPTION(result.maxDifference > tolerance, std::runtime_error,
                               "Stage trace max difference " << result.maxDifference
                               << " exceeds tolerance " << tolerance);
  }

  template <class ApplyInverseLike, class TpetraPreconditioner>
  ApplyInverseTpetraComparisonResult compareApplyInverseWithTpetraPreconditioner(
      const ApplyInverseLike &applyInverseLike,
      const TpetraPreconditioner &tpetraPreconditioner,
      const TVector<double> &rhs)
  {
    ApplyInverseTpetraComparisonResult result;

    TVector<double> applyInverseOutput(tpetraPreconditioner.getRangeMap(), rhs.getNumVectors());
    TVector<double> tpetraOutput(tpetraPreconditioner.getRangeMap(), rhs.getNumVectors());

    // Deliberately seed both outputs to verify overwrite/beta=0 preconditioner semantics.
    applyInverseOutput.putScalar(123.0);
    tpetraOutput.putScalar(-456.0);

    result.applyInverseReturnCode = applyInverseLike.ApplyInverse(rhs, applyInverseOutput);
    tpetraPreconditioner.apply(rhs, tpetraOutput, Teuchos::NO_TRANS, 1.0, 0.0);

    TVector<double> difference(tpetraPreconditioner.getRangeMap(), rhs.getNumVectors());
    difference.update(1.0, applyInverseOutput, 0.0);
    difference.update(-1.0, tpetraOutput, 1.0);

    Teuchos::Array<double> norms(difference.getNumVectors());
    difference.normInf(norms());
    for (int vectorOrdinal=0; vectorOrdinal<norms.size(); vectorOrdinal++)
    {
      result.maxDifference = std::max(result.maxDifference, norms[vectorOrdinal]);
    }

    return result;
  }

  template <class ApplyInverseLike, class TpetraPreconditioner>
  void assertApplyInverseMatchesTpetraPreconditioner(
      const ApplyInverseLike &applyInverseLike,
      const TpetraPreconditioner &tpetraPreconditioner,
      const TVector<double> &rhs,
      Teuchos::FancyOStream &out,
      bool &success,
      const double tolerance = 1e-12)
  {
    const ApplyInverseTpetraComparisonResult result =
      compareApplyInverseWithTpetraPreconditioner(applyInverseLike, tpetraPreconditioner, rhs);

    TEST_EQUALITY(result.applyInverseReturnCode, 0);
    TEST_COMPARE(result.maxDifference, <=, tolerance);
  }
}
}

#endif
