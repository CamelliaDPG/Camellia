#ifndef Camellia_TensorPlanarMITLMomentComparisonAdapter_h
#define Camellia_TensorPlanarMITLMomentComparisonAdapter_h

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "TensorMaxwellRHSMomentData.h"

namespace Camellia
{
  /**
   * \brief Phase 8 comparison helper for tensor-native Planar MITL moment dry-runs.
   *
   * The first production-facing Phase 8 backend is intentionally a dry-run/comparison
   * path: the legacy SolutionMoments update remains authoritative while tensor-native
   * reductions are sampled, timed, and compared.  This helper keeps the comparison
   * policy independent of the Planar MITL driver and of the Maxwell RHS wiring so that
   * tests and instrumentation can share one convention for rho/Jx/Jz error metrics.
   */
  class TensorPlanarMITLMomentComparisonAdapter
  {
  public:
    struct FieldComparison
    {
      double maxAbsDifference = 0.0;
      double maxRelDifference = 0.0;
      int maxAbsEntry = -1;
      int maxRelEntry = -1;

      bool within(double absTolerance, double relTolerance) const
      {
        return (maxAbsDifference <= absTolerance) && (maxRelDifference <= relTolerance);
      }
    };

    struct Comparison
    {
      FieldComparison rhoRHS;
      FieldComparison Jx;
      FieldComparison Jz;

      bool within(double absTolerance, double relTolerance) const
      {
        return rhoRHS.within(absTolerance, relTolerance) &&
               Jx.within(absTolerance, relTolerance) &&
               Jz.within(absTolerance, relTolerance);
      }
    };

  private:
    static void update(FieldComparison &field,
                       double actual,
                       double expected,
                       int entry,
                       double relativeFloor)
    {
      const double absDifference = std::abs(actual - expected);
      const double denominator = std::max(std::abs(expected), relativeFloor);
      const double relDifference = absDifference / denominator;
      if ((field.maxAbsEntry < 0) || (absDifference > field.maxAbsDifference))
      {
        field.maxAbsDifference = absDifference;
        field.maxAbsEntry = entry;
      }
      if ((field.maxRelEntry < 0) || (relDifference > field.maxRelDifference))
      {
        field.maxRelDifference = relDifference;
        field.maxRelEntry = entry;
      }
    }

    static void requireCompatible(const TensorMaxwellRHSMomentData &actual,
                                  const TensorMaxwellRHSMomentData &expected)
    {
      if (actual.numConfigurationCells() != expected.numConfigurationCells())
      {
        throw std::invalid_argument("TensorPlanarMITLMomentComparisonAdapter configuration cell count mismatch.");
      }
      if (actual.numPointsPerConfigurationCell() != expected.numPointsPerConfigurationCell())
      {
        throw std::invalid_argument("TensorPlanarMITLMomentComparisonAdapter point count mismatch.");
      }
    }

  public:
    static Comparison compare(const TensorMaxwellRHSMomentData &actual,
                              const TensorMaxwellRHSMomentData &expected,
                              double relativeFloor = 1e-14)
    {
      if (relativeFloor <= 0.0)
      {
        throw std::invalid_argument("TensorPlanarMITLMomentComparisonAdapter requires a positive relative floor.");
      }
      requireCompatible(actual, expected);

      Comparison comparison;
      const int pointsPerCell = actual.numPointsPerConfigurationCell();
      for (int cellOrdinal=0; cellOrdinal<actual.numConfigurationCells(); cellOrdinal++)
      {
        for (int pointOrdinal=0; pointOrdinal<pointsPerCell; pointOrdinal++)
        {
          const int entry = cellOrdinal * pointsPerCell + pointOrdinal;
          update(comparison.rhoRHS,
                 actual.rhoRHS(cellOrdinal, pointOrdinal),
                 expected.rhoRHS(cellOrdinal, pointOrdinal),
                 entry,
                 relativeFloor);
          update(comparison.Jx,
                 actual.Jx(cellOrdinal, pointOrdinal),
                 expected.Jx(cellOrdinal, pointOrdinal),
                 entry,
                 relativeFloor);
          update(comparison.Jz,
                 actual.Jz(cellOrdinal, pointOrdinal),
                 expected.Jz(cellOrdinal, pointOrdinal),
                 entry,
                 relativeFloor);
        }
      }
      return comparison;
    }
  };
}

#endif
