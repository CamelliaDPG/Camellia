#include "MatrixFreeGMGPreconditioner.h"
#include "MatrixFreeGMGPreconditionerComparisonHelpers.hpp"

#include "Teuchos_DefaultComm.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_UnitTestHarness.hpp"

using namespace Camellia;

namespace
{
  TMapConstPtr serialMap(const GlobalIndexType numGlobalEntries)
  {
    return Teuchos::rcp(new TMap(numGlobalEntries, 0, Teuchos::DefaultComm<int>::getComm()));
  }

  MatrixFreeGMGPreconditioner::Level oneLevelScalingLevel(const TMapConstPtr &map,
                                                          const double scale)
  {
    MatrixFreeGMGPreconditioner::Level level;
    level.map = map;
    level.coarseSolve = [scale](const TVector<double> &rhs, TVector<double> &solution)
    {
      solution.update(scale, rhs, 0.0);
    };
    return level;
  }

  class ScalingApplyInverse
  {
  public:
    explicit ScalingApplyInverse(const double scale, const int returnCode = 0)
    : _scale(scale), _returnCode(returnCode)
    {}

    int ApplyInverse(const TVector<double> &X, TVector<double> &Y) const
    {
      if (_returnCode != 0)
        return _returnCode;

      Y.update(_scale, X, 0.0);
      return 0;
    }

  private:
    double _scale;
    int _returnCode;
  };
}

TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, ApplyInverseComparisonHarnessMatchesTpetraPreconditionerSemantics )
{
  auto map = serialMap(4);
  std::vector<MatrixFreeGMGPreconditioner::Level> levels;
  levels.push_back(oneLevelScalingLevel(map, 0.25));
  MatrixFreeGMGPreconditioner tpetraPreconditioner(levels, "ComparisonHarnessScalingPreconditioner");

  TVector<double> rhs(map, 2);
  rhs.putScalar(8.0);

  const ScalingApplyInverse applyInverseLike(0.25);
  Test::assertApplyInverseMatchesTpetraPreconditioner(applyInverseLike,
                                                      tpetraPreconditioner,
                                                      rhs,
                                                      out,
                                                      success);
}

TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, ApplyInverseComparisonHarnessReportsMismatch )
{
  auto map = serialMap(3);
  std::vector<MatrixFreeGMGPreconditioner::Level> levels;
  levels.push_back(oneLevelScalingLevel(map, 0.25));
  MatrixFreeGMGPreconditioner tpetraPreconditioner(levels, "ComparisonHarnessMismatchPreconditioner");

  TVector<double> rhs(map, 1);
  rhs.putScalar(8.0);

  const ScalingApplyInverse mismatchedApplyInverse(0.5);
  const Test::ApplyInverseTpetraComparisonResult result =
    Test::compareApplyInverseWithTpetraPreconditioner(mismatchedApplyInverse,
                                                     tpetraPreconditioner,
                                                     rhs);

  TEST_EQUALITY(result.applyInverseReturnCode, 0);
  TEST_COMPARE(result.maxDifference, >, 1e-12);
}

TEUCHOS_UNIT_TEST( MatrixFreeInfrastructure, ApplyInverseComparisonHarnessPropagatesApplyInverseErrorCode )
{
  auto map = serialMap(2);
  std::vector<MatrixFreeGMGPreconditioner::Level> levels;
  levels.push_back(oneLevelScalingLevel(map, 0.25));
  MatrixFreeGMGPreconditioner tpetraPreconditioner(levels, "ComparisonHarnessErrorPreconditioner");

  TVector<double> rhs(map, 1);
  rhs.putScalar(8.0);

  const ScalingApplyInverse failingApplyInverse(0.25, -17);
  const Test::ApplyInverseTpetraComparisonResult result =
    Test::compareApplyInverseWithTpetraPreconditioner(failingApplyInverse,
                                                     tpetraPreconditioner,
                                                     rhs);

  TEST_EQUALITY(result.applyInverseReturnCode, -17);
  TEST_COMPARE(result.maxDifference, >, 1e-12);
}
