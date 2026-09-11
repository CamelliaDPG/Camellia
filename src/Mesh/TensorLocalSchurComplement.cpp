#include "TensorLocalSchurComplement.h"

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

using namespace Camellia;

namespace
{
  void decodeOrdinal(const TensorFactorSpaceShape &shape,
                     IndexType ordinal,
                     std::vector<int> &componentOrdinals)
  {
    TEUCHOS_TEST_FOR_EXCEPTION((ordinal < 0) || (ordinal >= shape.totalDofs()), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder ordinal out of range.");
    componentOrdinals.assign(shape.componentCount(), 0);
    for (int d=0; d<shape.componentCount(); d++)
    {
      componentOrdinals[d] = static_cast<int>((ordinal / shape.componentStride(d)) % shape.componentCardinality(d));
    }
  }

  std::vector<double> coefficientValuesForComponent(const TensorFunctionPtr &coefficient,
                                                    const TensorBasisCache1D &componentCache,
                                                    int componentCount,
                                                    int componentOrdinal)
  {
    TensorFunction::PointView points("TensorLocalSchurComplementBuilder::componentCoefficientPoints",
                                     componentCache.pointCount(),
                                     componentCount);
    auto pointsHost = Kokkos::create_mirror_view(points);
    for (int pointOrdinal=0; pointOrdinal<componentCache.pointCount(); pointOrdinal++)
    {
      for (int d=0; d<componentCount; d++)
      {
        pointsHost(pointOrdinal,d) = 0.0;
      }
      const double xi = componentCache.referencePoints().h_view(pointOrdinal);
      pointsHost(pointOrdinal,componentOrdinal) = componentCache.cellLeft() + 0.5 * (xi + 1.0) * componentCache.cellWidth();
    }
    Kokkos::deep_copy(points, pointsHost);

    TensorFunction::ValueView values("TensorLocalSchurComplementBuilder::componentCoefficientValues",
                                     componentCache.pointCount());
    coefficient->values(points, values);
    auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);

    std::vector<double> coefficientValues(componentCache.pointCount(), 0.0);
    for (int pointOrdinal=0; pointOrdinal<componentCache.pointCount(); pointOrdinal++)
    {
      coefficientValues[pointOrdinal] = valuesHost(pointOrdinal);
    }
    return coefficientValues;
  }

  void validateShapeAgainstCache(const TensorFactorSpaceShape &shape,
                                 const TensorBasisCacheND &basisCache,
                                 const std::string &label)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(shape.componentCount() != basisCache.componentCount(), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder shape/cache component-count mismatch for " + label + ".");
    for (int d=0; d<shape.componentCount(); d++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION(shape.componentCardinality(d) != basisCache.componentCache(d).basisCardinality(), std::invalid_argument,
                                 "TensorLocalSchurComplementBuilder shape/cache basis-cardinality mismatch for " + label + ".");
    }
  }
}

TensorProductLocalOperator TensorLocalSchurComplementBuilder::buildVolumeBOperator(const TensorBilinearTerm &term,
                                                                                   const TensorFactorSpaceShape &trialShape,
                                                                                   const TensorFactorSpaceShape &testShape,
                                                                                   const TensorBasisCacheND &trialBasisCache,
                                                                                   const TensorBasisCacheND &testBasisCache,
                                                                                   const std::vector<TensorFunctionPtr> &coefficientFactors)
{
  const int componentCount = term.componentCount();
  TEUCHOS_TEST_FOR_EXCEPTION(componentCount != trialBasisCache.componentCount(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder term/trial-cache rank mismatch.");
  TEUCHOS_TEST_FOR_EXCEPTION(componentCount != testBasisCache.componentCount(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder term/test-cache rank mismatch.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(coefficientFactors.size()) != componentCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires one coefficient factor slot per tensor component.");
  validateShapeAgainstCache(trialShape, trialBasisCache, "trial space");
  validateShapeAgainstCache(testShape, testBasisCache, "test space");

  std::vector<TensorComponentLocalMatrix> componentOperators;
  componentOperators.reserve(componentCount);

  for (int d=0; d<componentCount; d++)
  {
    const auto &trialCache = trialBasisCache.componentCache(d);
    const auto &testCache  = testBasisCache.componentCache(d);
    TEUCHOS_TEST_FOR_EXCEPTION(trialCache.pointCount() != testCache.pointCount(), std::logic_error,
                               "TensorLocalSchurComplementBuilder expected matching trial/test quadrature rules.");

    auto trialValues = trialCache.valuesAtQuadrature();
    auto trialDerivatives = trialCache.derivativesAtQuadrature();
    auto testValues = testCache.valuesAtQuadrature();
    auto testDerivatives = testCache.derivativesAtQuadrature();
    auto weights = trialCache.referenceWeights();
    trialValues.sync_host();
    trialDerivatives.sync_host();
    testValues.sync_host();
    testDerivatives.sync_host();
    weights.sync_host();

    std::vector<double> coefficientValues;
    if (coefficientFactors[d] != Teuchos::null)
    {
      coefficientValues = coefficientValuesForComponent(coefficientFactors[d], trialCache, componentCount, d);
    }

    std::vector<double> componentMatrix(testCache.basisCardinality() * trialCache.basisCardinality(), 0.0);
    for (int testOrdinal=0; testOrdinal<testCache.basisCardinality(); testOrdinal++)
    {
      for (int trialOrdinal=0; trialOrdinal<trialCache.basisCardinality(); trialOrdinal++)
      {
        double entry = 0.0;
        for (int pointOrdinal=0; pointOrdinal<trialCache.pointCount(); pointOrdinal++)
        {
          const double trialValue = term.trialOperator().takesDerivativeInComponent(d)
              ? trialDerivatives.h_view(trialOrdinal,pointOrdinal) * trialCache.derivativeScale()
              : trialValues.h_view(trialOrdinal,pointOrdinal);
          const double testValue = term.testOperator().takesDerivativeInComponent(d)
              ? testDerivatives.h_view(testOrdinal,pointOrdinal) * testCache.derivativeScale()
              : testValues.h_view(testOrdinal,pointOrdinal);
          const double coefficientValue = (coefficientFactors[d] != Teuchos::null) ? coefficientValues[pointOrdinal] : 1.0;
          entry += weights.h_view(pointOrdinal) * trialCache.jacobian() * coefficientValue * trialValue * testValue;
        }
        componentMatrix[testOrdinal * trialCache.basisCardinality() + trialOrdinal] = entry;
      }
    }

    componentOperators.push_back(TensorComponentLocalMatrix(testCache.basisCardinality(),
                                                            trialCache.basisCardinality(),
                                                            componentMatrix));
  }

  return TensorProductLocalOperator(trialShape, testShape, componentOperators);
}

std::vector<double> TensorLocalSchurComplementBuilder::denseBMatrix(const TensorProductLocalOperator &bOperator)
{
  std::vector<double> b(static_cast<size_t>(bOperator.rangeSize() * bOperator.domainSize()), 0.0);
  for (IndexType row=0; row<bOperator.rangeSize(); row++)
  {
    for (IndexType col=0; col<bOperator.domainSize(); col++)
    {
      b[static_cast<size_t>(row * bOperator.domainSize() + col)] = bOperator.entry(row, col);
    }
  }
  return b;
}

std::vector<std::vector<double>> TensorLocalSchurComplementBuilder::l2TestMassDiagonalFactors(const TensorBasisCacheND &testBasisCache)
{
  std::vector<std::vector<double>> componentDiagonals(testBasisCache.componentCount());
  for (int d=0; d<testBasisCache.componentCount(); d++)
  {
    const auto &testCache = testBasisCache.componentCache(d);
    auto testValues = testCache.valuesAtQuadrature();
    auto weights = testCache.referenceWeights();
    testValues.sync_host();
    weights.sync_host();

    componentDiagonals[d].assign(testCache.basisCardinality(), 0.0);
    for (int testOrdinal=0; testOrdinal<testCache.basisCardinality(); testOrdinal++)
    {
      double entry = 0.0;
      for (int pointOrdinal=0; pointOrdinal<testCache.pointCount(); pointOrdinal++)
      {
        const double value = testValues.h_view(testOrdinal, pointOrdinal);
        entry += weights.h_view(pointOrdinal) * testCache.jacobian() * value * value;
      }
      componentDiagonals[d][testOrdinal] = entry;
    }
  }
  return componentDiagonals;
}

std::vector<double> TensorLocalSchurComplementBuilder::l2TestMassDiagonal(const TensorFactorSpaceShape &testShape,
                                                                          const TensorBasisCacheND &testBasisCache)
{
  validateShapeAgainstCache(testShape, testBasisCache, "test space");
  return tensorProductDiagonal(testShape, l2TestMassDiagonalFactors(testBasisCache));
}

std::vector<double> TensorLocalSchurComplementBuilder::tensorProductDiagonal(const TensorFactorSpaceShape &shape,
                                                                             const std::vector<std::vector<double>> &componentDiagonals)
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentDiagonals.size()) != shape.componentCount(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires one diagonal factor per tensor component.");
  for (int d=0; d<shape.componentCount(); d++)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(componentDiagonals[d].size()) != shape.componentCardinality(d), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder component diagonal length must match the factor-space cardinality.");
  }

  std::vector<double> diagonal(shape.totalDofs(), 1.0);
  std::vector<int> componentOrdinals;
  for (IndexType ordinal=0; ordinal<shape.totalDofs(); ordinal++)
  {
    decodeOrdinal(shape, ordinal, componentOrdinals);
    double value = 1.0;
    for (int d=0; d<shape.componentCount(); d++)
    {
      value *= componentDiagonals[d][componentOrdinals[d]];
    }
    diagonal[ordinal] = value;
  }
  return diagonal;
}

std::vector<double> TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(const TensorProductLocalOperator &bOperator,
                                                                                           const std::vector<double> &gDiagonal)
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(gDiagonal.size()) != bOperator.rangeSize(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires one G diagonal entry per local test dof.");

  const IndexType trialDofCount = bOperator.domainSize();
  const IndexType testDofCount  = bOperator.rangeSize();
  std::vector<double> k(static_cast<size_t>(trialDofCount * trialDofCount), 0.0);

  for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
  {
    const double gEntry = gDiagonal[testOrdinal];
    TEUCHOS_TEST_FOR_EXCEPTION(gEntry == 0.0, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder cannot form a Schur complement with a zero G diagonal entry.");
    for (IndexType trialRow=0; trialRow<trialDofCount; trialRow++)
    {
      const double left = bOperator.entry(testOrdinal, trialRow) / gEntry;
      for (IndexType trialCol=0; trialCol<trialDofCount; trialCol++)
      {
        k[static_cast<size_t>(trialRow * trialDofCount + trialCol)] += left * bOperator.entry(testOrdinal, trialCol);
      }
    }
  }
  return k;
}
