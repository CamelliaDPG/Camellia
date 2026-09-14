#include "TensorLocalSchurComplement.h"

#include "TensorCoefficientExpansion.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>

#include "Basis.h"
#include "CamelliaIntrepidExtendedTypes.h"
#include "CellTopology.h"
#include "CubatureFactory.h"
#include "MPIWrapper.h"
#include "Teuchos_TestForException.hpp"
#include "Var.h"

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

  double cachedBasisEvaluationAtPoint(const std::vector<int> &componentBasisOrdinals,
                                      const TensorBasisCacheND &basisCache,
                                      const TensorComponentOperatorDescriptor &descriptor,
                                      const std::vector<int> &pointOrdinals)
  {
    double value = 1.0;
    for (int d=0; d<descriptor.componentCount(); d++)
    {
      const auto &componentCache = basisCache.componentCache(d);
      auto values = componentCache.valuesAtQuadrature();
      auto derivatives = componentCache.derivativesAtQuadrature();
      values.sync_host();
      derivatives.sync_host();
      const int basisOrdinal = componentBasisOrdinals[d];
      const int pointOrdinal = pointOrdinals[d];
      value *= descriptor.takesDerivativeInComponent(d)
          ? derivatives.h_view(basisOrdinal, pointOrdinal) * componentCache.derivativeScale()
          : values.h_view(basisOrdinal, pointOrdinal);
    }
    return value;
  }

  std::vector<double> fullTensorCoefficientValues(const TensorFunctionPtr &coefficient,
                                                  const TensorBasisCacheND &basisCache)
  {
    TensorFunction::ValueView values("TensorLocalSchurComplementBuilder::fullTensorCoefficientValues",
                                     basisCache.tensorPointCount());
    coefficient->values(basisCache.physicalPoints().d_view, values);
    auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
    std::vector<double> result(basisCache.tensorPointCount(), 0.0);
    for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
    {
      result[pointOrdinal] = valuesHost(pointOrdinal);
    }
    return result;
  }

  std::vector<double> componentTestGramMatrix(const TensorBasisCache1D &testCache,
                                              bool derivative)
  {
    auto values = testCache.valuesAtQuadrature();
    auto derivatives = testCache.derivativesAtQuadrature();
    auto weights = testCache.referenceWeights();
    values.sync_host();
    derivatives.sync_host();
    weights.sync_host();

    std::vector<double> matrix(testCache.basisCardinality() * testCache.basisCardinality(), 0.0);
    for (int row=0; row<testCache.basisCardinality(); row++)
    {
      for (int col=0; col<testCache.basisCardinality(); col++)
      {
        double entry = 0.0;
        for (int pointOrdinal=0; pointOrdinal<testCache.pointCount(); pointOrdinal++)
        {
          const double rowValue = derivative ? derivatives.h_view(row, pointOrdinal) * testCache.derivativeScale() : values.h_view(row, pointOrdinal);
          const double colValue = derivative ? derivatives.h_view(col, pointOrdinal) * testCache.derivativeScale() : values.h_view(col, pointOrdinal);
          entry += weights.h_view(pointOrdinal) * testCache.jacobian() * rowValue * colValue;
        }
        matrix[row * testCache.basisCardinality() + col] = entry;
      }
    }
    return matrix;
  }

  double tensorProductMatrixEntry(const TensorFactorSpaceShape &shape,
                                  const std::vector<std::vector<double>> &componentMatrices,
                                  IndexType rowOrdinal,
                                  IndexType colOrdinal)
  {
    std::vector<int> rowComponentOrdinals;
    std::vector<int> colComponentOrdinals;
    decodeOrdinal(shape, rowOrdinal, rowComponentOrdinals);
    decodeOrdinal(shape, colOrdinal, colComponentOrdinals);

    double value = 1.0;
    for (int d=0; d<shape.componentCount(); d++)
    {
      const int cardinality = shape.componentCardinality(d);
      value *= componentMatrices[d][rowComponentOrdinals[d] * cardinality + colComponentOrdinals[d]];
    }
    return value;
  }

  std::vector<double> solveDenseLinearSystem(std::vector<double> matrix,
                                             std::vector<double> rhs,
                                             IndexType dimension)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(matrix.size()) != dimension * dimension, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder dense solve matrix has incompatible dimensions.");
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(rhs.size()) != dimension, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder dense solve RHS has incompatible dimensions.");
    for (IndexType pivot=0; pivot<dimension; pivot++)
    {
      IndexType pivotRow = pivot;
      double pivotAbs = std::abs(matrix[static_cast<size_t>(pivot * dimension + pivot)]);
      for (IndexType row=pivot + 1; row<dimension; row++)
      {
        const double candidateAbs = std::abs(matrix[static_cast<size_t>(row * dimension + pivot)]);
        if (candidateAbs > pivotAbs)
        {
          pivotAbs = candidateAbs;
          pivotRow = row;
        }
      }
      TEUCHOS_TEST_FOR_EXCEPTION(pivotAbs == 0.0, std::invalid_argument,
                                 "TensorLocalSchurComplementBuilder cannot invert a singular dense test-norm matrix.");
      if (pivotRow != pivot)
      {
        for (IndexType col=0; col<dimension; col++) std::swap(matrix[static_cast<size_t>(pivot * dimension + col)], matrix[static_cast<size_t>(pivotRow * dimension + col)]);
        std::swap(rhs[pivot], rhs[pivotRow]);
      }
      for (IndexType row=pivot + 1; row<dimension; row++)
      {
        const double factor = matrix[static_cast<size_t>(row * dimension + pivot)] / matrix[static_cast<size_t>(pivot * dimension + pivot)];
        matrix[static_cast<size_t>(row * dimension + pivot)] = 0.0;
        for (IndexType col=pivot + 1; col<dimension; col++) matrix[static_cast<size_t>(row * dimension + col)] -= factor * matrix[static_cast<size_t>(pivot * dimension + col)];
        rhs[row] -= factor * rhs[pivot];
      }
    }
    std::vector<double> solution(dimension, 0.0);
    for (IndexType reverseOrdinal=0; reverseOrdinal<dimension; reverseOrdinal++)
    {
      const IndexType row = dimension - 1 - reverseOrdinal;
      double value = rhs[row];
      for (IndexType col=row + 1; col<dimension; col++) value -= matrix[static_cast<size_t>(row * dimension + col)] * solution[col];
      solution[row] = value / matrix[static_cast<size_t>(row * dimension + row)];
    }
    return solution;
  }

  std::vector<double> formDenseTestNormCrossSchurComplement(const std::vector<double> &rowBMatrix,
                                                            IndexType testDofCount,
                                                            IndexType rowTrialDofCount,
                                                            const std::vector<double> &colBMatrix,
                                                            IndexType colTrialDofCount,
                                                            const std::vector<double> &gMatrix)
  {
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(gMatrix.size()) != testDofCount * testDofCount, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder dense cross G dimensions do not match the supplied test dof count.");
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(rowBMatrix.size()) != testDofCount * rowTrialDofCount, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder dense row B dimensions do not match the supplied test/trial dof counts.");
    TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(colBMatrix.size()) != testDofCount * colTrialDofCount, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder dense column B dimensions do not match the supplied test/trial dof counts.");

    TensorLocalSchurComplementBuilder::LocalGraphNormAction graphAction(testDofCount, gMatrix);
    auto gInvB = graphAction.applyInverseToColumns(colBMatrix, colTrialDofCount);

    std::vector<double> k(static_cast<size_t>(rowTrialDofCount * colTrialDofCount), 0.0);
    for (IndexType rowTrialOrdinal=0; rowTrialOrdinal<rowTrialDofCount; rowTrialOrdinal++)
    {
      for (IndexType colTrialOrdinal=0; colTrialOrdinal<colTrialDofCount; colTrialOrdinal++)
      {
        double entry = 0.0;
        for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
        {
          entry += rowBMatrix[static_cast<size_t>(testOrdinal * rowTrialDofCount + rowTrialOrdinal)]
                   * gInvB[static_cast<size_t>(testOrdinal * colTrialDofCount + colTrialOrdinal)];
        }
        k[static_cast<size_t>(rowTrialOrdinal * colTrialDofCount + colTrialOrdinal)] = entry;
      }
    }
    return k;
  }

  int quadratureDegreeForComponent(int trialPolyOrder,
                                   int testPolyOrder,
                                   bool trialDerivative,
                                   bool testDerivative)
  {
    int degree = trialPolyOrder + testPolyOrder;
    if (trialDerivative) degree -= 1;
    if (testDerivative) degree -= 1;
    return std::max(1, degree + 1);
  }

  using CoefficientFactorTuple = TensorCoefficientFactorTuple;
  using CoefficientFactorExpansion = TensorCoefficientFactorExpansion;

  bool termSupportsAffineL2VolumeCellSchurComplement(const TensorBilinearTerm &term,
                                                     CoefficientFactorExpansion &coefficientFactorTerms)
  {
    if (!term.isVolumeTerm()) return false;
    if (term.trialVar()->isTrace() || term.testVar()->isTrace()) return false;

    coefficientFactorTerms.clear();
    if (!appendTensorSeparableCoefficientFactorTerms(term.coefficient(), term.componentCount(), coefficientFactorTerms)) return false;
    return !coefficientFactorTerms.empty();
  }

  bool termSupportsFullTensorCoefficientVolumeCellSchurComplement(const TensorBilinearTerm &term)
  {
    if (!term.isVolumeTerm()) return false;
    if (term.trialVar()->isTrace() || term.testVar()->isTrace()) return false;
    if (term.coefficient() == Teuchos::null) return false;
    if (term.coefficient()->hasSolutionDependencies()) return false;
    return true;
  }

  bool hasMatchingVolumeTerms(const TensorBilinearFormPtr &bilinearForm,
                              const std::string &trialVarName,
                              const std::string &testVarName)
  {
    for (const auto &term : bilinearForm->terms())
    {
      if (term.trialVar()->name() != trialVarName) continue;
      if (term.testVar()->name()  != testVarName)  continue;
      if (term.isVolumeTerm()) return true;
    }
    return false;
  }

  bool hasMatchingBoundaryTerms(const TensorBilinearFormPtr &bilinearForm,
                                const std::string &trialVarName,
                                const std::string &testVarName)
  {
    for (const auto &term : bilinearForm->terms())
    {
      if (term.trialVar()->name() != trialVarName) continue;
      if (term.testVar()->name()  != testVarName)  continue;
      if (term.isBoundaryTerm()) return true;
    }
    return false;
  }

  struct LineQuadratureRule
  {
    std::vector<double> points;
    std::vector<double> weights;
  };

  struct BoundaryBasisData1D
  {
    std::vector<double> values;
    int fieldCount = 0;
    int pointCount = 0;
  };

  bool tensorVarIsTraceLike(const TensorVarPtr &tensorVar)
  {
    return (tensorVar != Teuchos::null) && tensorVar->isTrace();
  }

  bool tensorVarIsFluxLike(const TensorVarPtr &tensorVar)
  {
    return (tensorVar != Teuchos::null) && tensorVar->isFluxLikeTrace();
  }

  LineQuadratureRule lineQuadratureFromCubatureFactory(int exactnessDegree)
  {
    CubatureFactory cubFactory;
    auto cubature = cubFactory.create(CellTopology::line(), exactnessDegree);

    const int pointCount = cubature->getNumPoints();
    auto points = cubature->allocateCubaturePoints();
    auto weights = cubature->allocateCubatureWeights();
    cubature->getCubature(points, weights);

    auto pointsView  = points.getTensorComponent(0);
    auto weightsView = weights.getTensorComponent(0).getUnderlyingView();
    auto pointsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), pointsView);
    auto weightsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), weightsView);

    LineQuadratureRule rule;
    rule.points.resize(pointCount);
    rule.weights.resize(pointCount);
    for (int p=0; p<pointCount; p++)
    {
      rule.points[p] = pointsHost(p,0);
      rule.weights[p] = weightsHost(p);
    }
    return rule;
  }

  BoundaryBasisData1D buildBoundaryBasisDataForComponent(const TensorMeshPtr &tensorMesh,
                                                         const TensorVarDiscretizationPtr &discretization,
                                                         int componentOrdinal,
                                                         bool isTest,
                                                         const ComponentCellID1D &cellID,
                                                         const std::vector<double> &quadraturePoints)
  {
    BoundaryBasisData1D data;
    auto basisFamily = tensorMesh->effectiveBasisFamily(discretization);
    VarPtr componentVar = discretization->tensorVar()->componentVar(componentOrdinal);
    const int polyOrder = tensorMesh->effectivePolyOrder(discretization, cellID, componentOrdinal, isTest);
    auto basis = basisFamily->getBasis(CellTopology::line(), efsForSpace(componentVar->space()), polyOrder);

    data.fieldCount = basis->getCardinality();
    data.pointCount = static_cast<int>(quadraturePoints.size());
    data.values.assign(data.fieldCount * data.pointCount, 0.0);

    Camellia::FieldContainer<double> refPointContainer(data.pointCount, 1);
    for (int p=0; p<data.pointCount; p++) refPointContainer(p,0) = quadraturePoints[p];

    Camellia::FieldContainer<double> basisValues(data.fieldCount, data.pointCount);
    basis->getValues(basisValues, refPointContainer, OP_VALUE);
    for (int fieldOrdinal=0; fieldOrdinal<data.fieldCount; fieldOrdinal++)
    {
      for (int pointOrdinal=0; pointOrdinal<data.pointCount; pointOrdinal++)
      {
        data.values[fieldOrdinal * data.pointCount + pointOrdinal] = basisValues(fieldOrdinal, pointOrdinal);
      }
    }
    return data;
  }

  double boundaryBasisValue(const TensorVarPtr &tensorVar,
                            const std::vector<int> &componentBasisOrdinals,
                            const std::vector<BoundaryBasisData1D> &basisData,
                            int orthogonalComponentOrdinal,
                            const std::vector<int> &pointOrdinals)
  {
    double value = 1.0;
    for (int d=0; d<static_cast<int>(componentBasisOrdinals.size()); d++)
    {
      if (tensorVarIsTraceLike(tensorVar) && (d == tensorVar->traceComponentOrdinal()))
      {
        value *= (componentBasisOrdinals[d] == 0) ? 1.0 : 0.0;
        continue;
      }
      const int pointOrdinal = (d == orthogonalComponentOrdinal) ? 0 : pointOrdinals[d];
      value *= basisData[d].values[componentBasisOrdinals[d] * basisData[d].pointCount + pointOrdinal];
    }
    return value;
  }

  ComponentCellID1D firstIncidentOrthogonalCell(const TensorMeshPtr &tensorMesh,
                                                const TensorMeshTopology::TensorSkeletonEntity &entity)
  {
    auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(entity.orthogonalComponentOrdinal());
    const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
    const IndexType incidentEnd   = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal() + 1);
    TEUCHOS_TEST_FOR_EXCEPTION(incidentBegin >= incidentEnd, std::logic_error,
                               "TensorLocalSchurComplementBuilder expected at least one incident orthogonal-component cell for boundary assembly.");
    return orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
  }

  bool termSupportsAffineBoundaryB(const TensorBilinearTerm &term)
  {
    if (!term.isBoundaryTerm()) return false;
    if (!term.trialOperator().isValueOnly()) return false;
    if (!term.testOperator().isValueOnly()) return false;
    const bool trialTrace = tensorVarIsTraceLike(term.trialVar());
    const bool testTrace  = tensorVarIsTraceLike(term.testVar());
    if (trialTrace == testTrace) return false;
    if (term.coefficient() == Teuchos::null) return false;
    if (term.coefficient()->hasSolutionDependencies()) return false;
    return true;
  }
}

namespace
{
  std::vector<double> denseVolumeBMatrixByTensorQuadrature(const TensorBilinearTerm &term,
                                                           const TensorFactorSpaceShape &trialShape,
                                                           const TensorFactorSpaceShape &testShape,
                                                           const TensorBasisCacheND &trialBasisCache,
                                                           const TensorBasisCacheND &testBasisCache)
  {
    validateShapeAgainstCache(trialShape, trialBasisCache, "trial space");
    validateShapeAgainstCache(testShape,  testBasisCache,  "test space");

    std::vector<double> coefficientValues = fullTensorCoefficientValues(term.coefficient(), trialBasisCache);
    auto weights = trialBasisCache.quadratureWeights();
    weights.sync_host();

    std::vector<int> pointCounts(term.componentCount(), 1);
    for (int d=0; d<term.componentCount(); d++)
    {
      pointCounts[d] = trialBasisCache.componentCache(d).pointCount();
      TEUCHOS_TEST_FOR_EXCEPTION(pointCounts[d] != testBasisCache.componentCache(d).pointCount(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder expected matching trial/test quadrature rules for full tensor coefficient fallback.");
    }

    std::vector<double> b(static_cast<size_t>(testShape.totalDofs() * trialShape.totalDofs()), 0.0);
    std::vector<int> trialBasisOrdinals(term.componentCount(), 0);
    std::vector<int> testBasisOrdinals(term.componentCount(), 0);
    std::vector<int> pointOrdinals(term.componentCount(), 0);

    for (IndexType testOrdinal=0; testOrdinal<testShape.totalDofs(); testOrdinal++)
    {
      decodeOrdinal(testShape, testOrdinal, testBasisOrdinals);
      for (IndexType trialOrdinal=0; trialOrdinal<trialShape.totalDofs(); trialOrdinal++)
      {
        decodeOrdinal(trialShape, trialOrdinal, trialBasisOrdinals);
        double entry = 0.0;
        for (IndexType flatPointOrdinal=0; flatPointOrdinal<trialBasisCache.tensorPointCount(); flatPointOrdinal++)
        {
          IndexType workingOrdinal = flatPointOrdinal;
          for (int d=0; d<term.componentCount(); d++)
          {
            pointOrdinals[d] = workingOrdinal % pointCounts[d];
            workingOrdinal /= pointCounts[d];
          }
          entry += weights.h_view(flatPointOrdinal)
                   * coefficientValues[flatPointOrdinal]
                   * cachedBasisEvaluationAtPoint(trialBasisOrdinals, trialBasisCache, term.trialOperator(), pointOrdinals)
                   * cachedBasisEvaluationAtPoint(testBasisOrdinals,  testBasisCache,  term.testOperator(),  pointOrdinals);
        }
        b[static_cast<size_t>(testOrdinal * trialShape.totalDofs() + trialOrdinal)] = entry;
      }
    }
    return b;
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

std::vector<double> TensorLocalSchurComplementBuilder::graphTestNormMatrix(const TensorFactorSpaceShape &testShape,
                                                                           const TensorBasisCacheND &testBasisCache,
                                                                           double l2Weight)
{
  validateShapeAgainstCache(testShape, testBasisCache, "test space");
  TEUCHOS_TEST_FOR_EXCEPTION(l2Weight < 0.0, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a nonnegative L2 graph-norm weight.");

  std::vector<std::vector<double>> massMatrices(testShape.componentCount());
  std::vector<std::vector<double>> stiffnessMatrices(testShape.componentCount());
  for (int d=0; d<testShape.componentCount(); d++)
  {
    massMatrices[d] = componentTestGramMatrix(testBasisCache.componentCache(d), false);
    stiffnessMatrices[d] = componentTestGramMatrix(testBasisCache.componentCache(d), true);
  }

  std::vector<double> gMatrix(static_cast<size_t>(testShape.totalDofs() * testShape.totalDofs()), 0.0);
  for (IndexType row=0; row<testShape.totalDofs(); row++)
  {
    for (IndexType col=0; col<testShape.totalDofs(); col++)
    {
      double entry = l2Weight * tensorProductMatrixEntry(testShape, massMatrices, row, col);
      for (int derivativeComponent=0; derivativeComponent<testShape.componentCount(); derivativeComponent++)
      {
        std::vector<std::vector<double>> componentMatrices = massMatrices;
        componentMatrices[derivativeComponent] = stiffnessMatrices[derivativeComponent];
        entry += tensorProductMatrixEntry(testShape, componentMatrices, row, col);
      }
      gMatrix[static_cast<size_t>(row * testShape.totalDofs() + col)] = entry;
    }
  }
  return gMatrix;
}

TensorLocalSchurComplementBuilder::LocalGraphNormAction::LocalGraphNormAction(IndexType testDofCount,
                                                                               const std::vector<double> &gMatrix)
: _testDofCount(testDofCount), _gMatrix(gMatrix), _gFactorization(gMatrix)
{
  TEUCHOS_TEST_FOR_EXCEPTION(_testDofCount < 0, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction requires a nonnegative local test dof count.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(_gMatrix.size()) != _testDofCount * _testDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction requires a square dense G matrix matching the test dof count.");
  factorGraphNormMatrix();
}

IndexType TensorLocalSchurComplementBuilder::LocalGraphNormAction::testDofCount() const
{
  return _testDofCount;
}

const std::vector<double> &TensorLocalSchurComplementBuilder::LocalGraphNormAction::gMatrix() const
{
  return _gMatrix;
}

void TensorLocalSchurComplementBuilder::LocalGraphNormAction::factorGraphNormMatrix()
{
  _pivotRows.assign(_testDofCount, 0);
  for (IndexType pivot=0; pivot<_testDofCount; pivot++)
  {
    IndexType pivotRow = pivot;
    double pivotAbs = std::abs(_gFactorization[static_cast<size_t>(pivot * _testDofCount + pivot)]);
    for (IndexType row=pivot + 1; row<_testDofCount; row++)
    {
      const double candidateAbs = std::abs(_gFactorization[static_cast<size_t>(row * _testDofCount + pivot)]);
      if (candidateAbs > pivotAbs)
      {
        pivotAbs = candidateAbs;
        pivotRow = row;
      }
    }
    TEUCHOS_TEST_FOR_EXCEPTION(pivotAbs == 0.0, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder cannot factor a singular dense test-norm matrix.");
    _pivotRows[pivot] = pivotRow;
    if (pivotRow != pivot)
    {
      for (IndexType col=0; col<_testDofCount; col++) std::swap(_gFactorization[static_cast<size_t>(pivot * _testDofCount + col)], _gFactorization[static_cast<size_t>(pivotRow * _testDofCount + col)]);
    }
    for (IndexType row=pivot + 1; row<_testDofCount; row++)
    {
      const double factor = _gFactorization[static_cast<size_t>(row * _testDofCount + pivot)] / _gFactorization[static_cast<size_t>(pivot * _testDofCount + pivot)];
      _gFactorization[static_cast<size_t>(row * _testDofCount + pivot)] = factor;
      for (IndexType col=pivot + 1; col<_testDofCount; col++) _gFactorization[static_cast<size_t>(row * _testDofCount + col)] -= factor * _gFactorization[static_cast<size_t>(pivot * _testDofCount + col)];
    }
  }
}

std::vector<double> TensorLocalSchurComplementBuilder::LocalGraphNormAction::applyInverse(const std::vector<double> &rhs) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(rhs.size()) != _testDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction RHS size does not match the local test dof count.");

  std::vector<double> working = rhs;
  for (IndexType pivot=0; pivot<_testDofCount; pivot++)
  {
    const IndexType pivotRow = _pivotRows[pivot];
    if (pivotRow != pivot) std::swap(working[pivot], working[pivotRow]);
    for (IndexType row=pivot + 1; row<_testDofCount; row++)
    {
      working[row] -= _gFactorization[static_cast<size_t>(row * _testDofCount + pivot)] * working[pivot];
    }
  }

  std::vector<double> solution(_testDofCount, 0.0);
  for (IndexType reverseOrdinal=0; reverseOrdinal<_testDofCount; reverseOrdinal++)
  {
    const IndexType row = _testDofCount - 1 - reverseOrdinal;
    double value = working[row];
    for (IndexType col=row + 1; col<_testDofCount; col++) value -= _gFactorization[static_cast<size_t>(row * _testDofCount + col)] * solution[col];
    solution[row] = value / _gFactorization[static_cast<size_t>(row * _testDofCount + row)];
  }
  return solution;
}

std::vector<double> TensorLocalSchurComplementBuilder::LocalGraphNormAction::applyInverseToColumns(const std::vector<double> &inputMatrix,
                                                                                                   IndexType columnCount) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(columnCount < 0, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction requires a nonnegative column count.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(inputMatrix.size()) != _testDofCount * columnCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction input matrix dimensions do not match.");

  std::vector<double> output(inputMatrix.size(), 0.0);
  for (IndexType columnOrdinal=0; columnOrdinal<columnCount; columnOrdinal++)
  {
    std::vector<double> rhs(_testDofCount, 0.0);
    for (IndexType testOrdinal=0; testOrdinal<_testDofCount; testOrdinal++)
    {
      rhs[testOrdinal] = inputMatrix[static_cast<size_t>(testOrdinal * columnCount + columnOrdinal)];
    }

    auto solution = applyInverse(rhs);
    for (IndexType testOrdinal=0; testOrdinal<_testDofCount; testOrdinal++)
    {
      output[static_cast<size_t>(testOrdinal * columnCount + columnOrdinal)] = solution[testOrdinal];
    }
  }
  return output;
}

std::vector<double> TensorLocalSchurComplementBuilder::LocalGraphNormAction::applySchurComplement(const std::vector<double> &bMatrix,
                                                                                                  IndexType trialDofCount,
                                                                                                  const std::vector<double> &trialVector) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(trialDofCount < 0, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction requires a nonnegative trial dof count.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(bMatrix.size()) != _testDofCount * trialDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction dense B dimensions do not match.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(trialVector.size()) != trialDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder LocalGraphNormAction trial vector size does not match.");

  std::vector<double> bTimesX(_testDofCount, 0.0);
  for (IndexType testOrdinal=0; testOrdinal<_testDofCount; testOrdinal++)
  {
    for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
    {
      bTimesX[testOrdinal] += bMatrix[static_cast<size_t>(testOrdinal * trialDofCount + trialOrdinal)] * trialVector[trialOrdinal];
    }
  }

  auto gInverseBTimesX = applyInverse(bTimesX);
  std::vector<double> result(trialDofCount, 0.0);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    for (IndexType testOrdinal=0; testOrdinal<_testDofCount; testOrdinal++)
    {
      result[trialOrdinal] += bMatrix[static_cast<size_t>(testOrdinal * trialDofCount + trialOrdinal)] * gInverseBTimesX[testOrdinal];
    }
  }
  return result;
}

TensorLocalSchurComplementBuilder::LocalGraphNormAction TensorLocalSchurComplementBuilder::localGraphNormAction(IndexType testDofCount,
                                                                                                                 const std::vector<double> &gMatrix)
{
  return LocalGraphNormAction(testDofCount, gMatrix);
}

std::vector<double> TensorLocalSchurComplementBuilder::applyAffineGraphVolumeSchurComplement(const TensorMeshPtr &tensorMesh,
                                                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                                                             const TensorBilinearFormPtr &bilinearForm,
                                                                                             const std::string &trialVarName,
                                                                                             const std::string &testVarName,
                                                                                             const std::vector<double> &trialVector,
                                                                                             double l2Weight)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for matrix-free graph-norm Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for matrix-free graph-norm Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for matrix-free graph-norm Schur action.");

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(trialVector.size()) != trialDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder matrix-free graph-norm Schur action input size does not match the trial coefficient count.");

  std::vector<double> output(trialDofCount, 0.0);
  auto blocks = affineL2VolumeCellSchurComplements(tensorMesh,
                                                   tensorDofLayout,
                                                   bilinearForm,
                                                   trialVarName,
                                                   testVarName);
  auto testDiscretization = tensorMesh->testDiscretization(testVarName);
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localTrialDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                          trialVarName,
                                                                          tensorCellOrdinal);
    TEUCHOS_TEST_FOR_EXCEPTION(localTrialDofCount != block.trialShape.totalDofs(), std::logic_error,
                               "TensorLocalSchurComplementBuilder local graph-norm action size does not match the local trial block size.");

    std::vector<double> localTrialVector(localTrialDofCount, 0.0);
    for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
    {
      const IndexType trialGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                                       trialVarName,
                                                                       tensorCellOrdinal,
                                                                       localTrialOrdinal);
      localTrialVector[localTrialOrdinal] = trialVector[trialGid];
    }

    std::vector<int> gQuadratureDegrees(block.testShape.componentCount(), 1);
    for (int d=0; d<block.testShape.componentCount(); d++)
    {
      const int testPolyOrder = tensorMesh->effectivePolyOrder(testDiscretization, block.cellTuple[d], d, true);
      gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
    }
    TensorBasisCacheND gTestBasisCache(tensorMesh, testDiscretization, true, block.cellTuple, gQuadratureDegrees);
    auto graphAction = localGraphNormAction(block.testShape.totalDofs(), graphTestNormMatrix(block.testShape, gTestBasisCache, l2Weight));
    auto localOutput = graphAction.applySchurComplement(block.bMatrix,
                                                        localTrialDofCount,
                                                        localTrialVector);

    for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
    {
      const IndexType trialGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                                       trialVarName,
                                                                       tensorCellOrdinal,
                                                                       localTrialOrdinal);
      output[trialGid] += localOutput[localTrialOrdinal];
    }
  }

  return output;
}

std::vector<double> TensorLocalSchurComplementBuilder::applyAffineGraphVolumeSchurComplement(const TensorMeshPtr &tensorMesh,
                                                                                             const TensorDofLayoutPtr &tensorDofLayout,
                                                                                             const TensorBilinearFormPtr &bilinearForm,
                                                                                             const std::vector<std::string> &trialVarNames,
                                                                                             const std::vector<std::string> &testVarNames,
                                                                                             const std::vector<double> &trialVector,
                                                                                             double l2Weight)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for multi-variable matrix-free graph-norm Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for multi-variable matrix-free graph-norm Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for multi-variable matrix-free graph-norm Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVarNames.empty() || testVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder multi-variable matrix-free graph-norm Schur action requires trial and test variables.");

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType trialVarCount = static_cast<IndexType>(trialVarNames.size());
  const IndexType testVarCount  = static_cast<IndexType>(testVarNames.size());
  std::vector<IndexType> trialOffsets(trialVarCount, 0);
  IndexType totalTrialDofCount = 0;
  for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(trialVector.size()) != totalTrialDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder multi-variable matrix-free graph-norm Schur action input size does not match the concatenated trial coefficient count.");

  std::vector<std::vector<bool>> pairHasBlocks(testVarCount, std::vector<bool>(trialVarCount, false));
  std::vector<std::vector<std::vector<LocalVolumeSchurComplementBlock>>> pairBlocks(testVarCount,
                                                                                    std::vector<std::vector<LocalVolumeSchurComplementBlock>>(trialVarCount));
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
    {
      const std::string &trialVarName = trialVarNames[trialVarOrdinal];
      const std::string &testVarName  = testVarNames[testVarOrdinal];
      if (!hasMatchingVolumeTerms(bilinearForm, trialVarName, testVarName)) continue;
      pairBlocks[testVarOrdinal][trialVarOrdinal] = affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                       tensorDofLayout,
                                                                                       bilinearForm,
                                                                                       trialVarName,
                                                                                       testVarName);
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(pairBlocks[testVarOrdinal][trialVarOrdinal].size()) != tensorMesh->topology()->activeCellCount(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder expected one local B block per active tensor cell for multi-variable matrix-free graph action.");
      pairHasBlocks[testVarOrdinal][trialVarOrdinal] = true;
    }
  }

  std::vector<double> output(totalTrialDofCount, 0.0);
  const IndexType activeCellCount = tensorMesh->topology()->activeCellCount();
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    const std::string &testVarName = testVarNames[testVarOrdinal];
    const auto &testVariableData = operatorData.variableData(TENSOR_TEST_SPACE, testVarName);
    TEUCHOS_TEST_FOR_EXCEPTION(testVariableData.isTraceVariable(), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder matrix-free graph volume action currently requires volume test variables.");
    const auto testDiscretization = tensorMesh->testDiscretization(testVarName);

    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<activeCellCount; tensorCellOrdinal++)
    {
      const auto cellTuple = tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
      auto testShape = tensorDofLayout->factorSpaceShape(testVarName, TENSOR_TEST_SPACE, cellTuple);
      const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(localTestDofCount != testShape.totalDofs(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder local test shape does not match TensorOperatorData for multi-variable graph action.");
      std::vector<double> localB(static_cast<size_t>(localTestDofCount * totalTrialDofCount), 0.0);

      for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
      {
        if (!pairHasBlocks[testVarOrdinal][trialVarOrdinal]) continue;
        const std::string &trialVarName = trialVarNames[trialVarOrdinal];
        const auto &block = pairBlocks[testVarOrdinal][trialVarOrdinal][tensorCellOrdinal];
        const IndexType localTrialDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
        TEUCHOS_TEST_FOR_EXCEPTION((localTrialDofCount != block.trialShape.totalDofs()) || (localTestDofCount != block.testShape.totalDofs()), std::logic_error,
                                   "TensorLocalSchurComplementBuilder local volume B block shape does not match TensorOperatorData for multi-variable graph action.");
        for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
        {
          for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
          {
            const IndexType trialGid = trialOffsets[trialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localTrialOrdinal);
            localB[static_cast<size_t>(localTestOrdinal * totalTrialDofCount + trialGid)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * localTrialDofCount + localTrialOrdinal)];
          }
        }
      }

      std::vector<int> gQuadratureDegrees(testShape.componentCount(), 1);
      for (int d=0; d<testShape.componentCount(); d++)
      {
        const int testPolyOrder = tensorMesh->effectivePolyOrder(testDiscretization, cellTuple[d], d, true);
        gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
      }
      TensorBasisCacheND gTestBasisCache(tensorMesh, testDiscretization, true, cellTuple, gQuadratureDegrees);
      auto graphAction = localGraphNormAction(localTestDofCount, graphTestNormMatrix(testShape, gTestBasisCache, l2Weight));
      auto localOutput = graphAction.applySchurComplement(localB, totalTrialDofCount, trialVector);
      for (IndexType gid=0; gid<totalTrialDofCount; gid++) output[gid] += localOutput[gid];
    }
  }

  return output;
}

std::vector<double> TensorLocalSchurComplementBuilder::applyAffineGraphVolumeBoundarySchurComplement(const TensorMeshPtr &tensorMesh,
                                                                                                     const TensorDofLayoutPtr &tensorDofLayout,
                                                                                                     const TensorBilinearFormPtr &bilinearForm,
                                                                                                     const std::vector<std::string> &trialVarNames,
                                                                                                     const std::vector<std::string> &testVarNames,
                                                                                                     const std::vector<double> &trialVector,
                                                                                                     double l2Weight)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for matrix-free graph volume/boundary Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for matrix-free graph volume/boundary Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for matrix-free graph volume/boundary Schur action.");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVarNames.empty() || testVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder matrix-free graph volume/boundary Schur action requires trial and test variables.");

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType trialVarCount = static_cast<IndexType>(trialVarNames.size());
  std::vector<IndexType> trialOffsets(trialVarCount, 0);
  IndexType totalTrialDofCount = 0;
  for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(trialVector.size()) != totalTrialDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder matrix-free graph volume/boundary Schur action input size does not match the concatenated trial coefficient count.");

  std::vector<double> output(totalTrialDofCount, 0.0);
  for (const std::string &testVarName : testVarNames)
  {
    const auto &testVariableData = operatorData.variableData(TENSOR_TEST_SPACE, testVarName);
    TEUCHOS_TEST_FOR_EXCEPTION(testVariableData.isTraceVariable(), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder matrix-free graph volume/boundary action currently requires volume test variables.");
    TensorCoefficientMap testMap(tensorMesh, tensorDofLayout, TENSOR_TEST_SPACE, testVarName);
    const auto testDiscretization = tensorMesh->testDiscretization(testVarName);

    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<tensorMesh->topology()->activeCellCount(); tensorCellOrdinal++)
    {
      const auto cellTuple = tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
      auto testShape = tensorDofLayout->factorSpaceShape(testVarName, TENSOR_TEST_SPACE, cellTuple);
      const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(localTestDofCount != testShape.totalDofs(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder local test shape does not match TensorOperatorData for graph volume/boundary action.");

      std::vector<int> gQuadratureDegrees(testShape.componentCount(), 1);
      for (int d=0; d<testShape.componentCount(); d++)
      {
        const int testPolyOrder = tensorMesh->effectivePolyOrder(testDiscretization, cellTuple[d], d, true);
        gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
      }
      TensorBasisCacheND gTestBasisCache(tensorMesh, testDiscretization, true, cellTuple, gQuadratureDegrees);
      std::vector<double> localB(static_cast<size_t>(localTestDofCount * totalTrialDofCount), 0.0);

      for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
      {
        const std::string &trialVarName = trialVarNames[trialVarOrdinal];
        if (hasMatchingVolumeTerms(bilinearForm, trialVarName, testVarName))
        {
          auto volumeBlocks = affineL2VolumeCellSchurComplements(tensorMesh,
                                                                 tensorDofLayout,
                                                                 bilinearForm,
                                                                 trialVarName,
                                                                 testVarName);
          const auto &block = volumeBlocks[tensorCellOrdinal];
          const IndexType localTrialDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
          TEUCHOS_TEST_FOR_EXCEPTION((localTrialDofCount != block.trialShape.totalDofs()) || (localTestDofCount != block.testShape.totalDofs()), std::logic_error,
                                     "TensorLocalSchurComplementBuilder local graph volume B block shape does not match TensorOperatorData for graph action.");
          for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
          {
            for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
            {
              const IndexType trialGid = trialOffsets[trialVarOrdinal]
                  + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localTrialOrdinal);
              localB[static_cast<size_t>(localTestOrdinal * totalTrialDofCount + trialGid)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * localTrialDofCount + localTrialOrdinal)];
            }
          }
        }

        if (hasMatchingBoundaryTerms(bilinearForm, trialVarName, testVarName))
        {
          auto boundaryBlocks = affineBoundaryBBlocks(tensorMesh,
                                                      tensorDofLayout,
                                                      bilinearForm,
                                                      trialVarName,
                                                      testVarName);
          const auto &trialVariableData = operatorData.variableData(TENSOR_TRIAL_SPACE, trialVarName);
          TensorCoefficientMap trialMap(tensorMesh, tensorDofLayout, TENSOR_TRIAL_SPACE, trialVarName);
          for (const auto &block : boundaryBlocks)
          {
            const IndexType entityOrdinal = block.entityOrdinal;
            const int orthogonalComponentOrdinal = block.entity.orthogonalComponentOrdinal();
            const auto retainedComponents = tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);
            auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
            const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(block.entity.vertexOrdinal());
            const ComponentCellID1D incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
            std::vector<ComponentCellID1D> testCells(tensorMesh->topology()->componentCount(), incidentCell);
            for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
            {
              testCells[retainedComponents[retainedOrdinal]] = block.entity.retainedCellTuple()[retainedOrdinal];
            }
            if (tensorMesh->topology()->tensorCellOrdinal(testCells) != tensorCellOrdinal) continue;

            auto testIndices = testMap.indicesForVolumeCell(testCells);
            TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(testIndices.size()) != block.testShape.totalDofs(), std::logic_error,
                                       "TensorLocalSchurComplementBuilder graph boundary test indices do not match block test shape.");

            std::vector<IndexType> trialIndices(block.trialShape.totalDofs(), -1);
            if (trialVariableData.isTraceVariable())
            {
              for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
              {
                trialIndices[localTrialOrdinal] = operatorData.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, entityOrdinal, localTrialOrdinal);
              }
            }
            else
            {
              trialIndices = trialMap.indicesForVolumeCell(testCells);
            }

            for (IndexType localTestOrdinal=0; localTestOrdinal<block.testShape.totalDofs(); localTestOrdinal++)
            {
              const IndexType testGid = testIndices[localTestOrdinal];
              IndexType cellLocalTestOrdinal = -1;
              for (IndexType candidate=0; candidate<localTestDofCount; candidate++)
              {
                if (operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal, candidate) == testGid)
                {
                  cellLocalTestOrdinal = candidate;
                  break;
                }
              }
              TEUCHOS_TEST_FOR_EXCEPTION(cellLocalTestOrdinal < 0, std::logic_error,
                                         "TensorLocalSchurComplementBuilder could not map a boundary test index into the graph volume cell.");
              for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
              {
                const IndexType trialGid = trialOffsets[trialVarOrdinal] + trialIndices[localTrialOrdinal];
                localB[static_cast<size_t>(cellLocalTestOrdinal * totalTrialDofCount + trialGid)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * block.trialShape.totalDofs() + localTrialOrdinal)];
              }
            }
          }
        }
      }

      auto graphAction = localGraphNormAction(localTestDofCount, graphTestNormMatrix(testShape, gTestBasisCache, l2Weight));
      auto localOutput = graphAction.applySchurComplement(localB, totalTrialDofCount, trialVector);
      for (IndexType gid=0; gid<totalTrialDofCount; gid++) output[gid] += localOutput[gid];
    }
  }
  return output;
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
  return formDiagonalTestNormSchurComplement(denseBMatrix(bOperator),
                                             bOperator.rangeSize(),
                                             bOperator.domainSize(),
                                             gDiagonal);
}

std::vector<double> TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(const std::vector<double> &bMatrix,
                                                                                           IndexType testDofCount,
                                                                                           IndexType trialDofCount,
                                                                                           const std::vector<double> &gDiagonal)
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(gDiagonal.size()) != testDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires one G diagonal entry per local test dof.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(bMatrix.size()) != testDofCount * trialDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder dense B dimensions do not match the supplied test/trial dof counts.");

  std::vector<double> k(static_cast<size_t>(trialDofCount * trialDofCount), 0.0);
  for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
  {
    const double gEntry = gDiagonal[testOrdinal];
    TEUCHOS_TEST_FOR_EXCEPTION(gEntry == 0.0, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder cannot form a Schur complement with a zero G diagonal entry.");
    for (IndexType trialRow=0; trialRow<trialDofCount; trialRow++)
    {
      const double left = bMatrix[static_cast<size_t>(testOrdinal * trialDofCount + trialRow)] / gEntry;
      for (IndexType trialCol=0; trialCol<trialDofCount; trialCol++)
      {
        k[static_cast<size_t>(trialRow * trialDofCount + trialCol)] += left * bMatrix[static_cast<size_t>(testOrdinal * trialDofCount + trialCol)];
      }
    }
  }
  return k;
}

std::vector<double> TensorLocalSchurComplementBuilder::formDenseTestNormSchurComplement(const std::vector<double> &bMatrix,
                                                                                         IndexType testDofCount,
                                                                                         IndexType trialDofCount,
                                                                                         const std::vector<double> &gMatrix)
{
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(gMatrix.size()) != testDofCount * testDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder dense G dimensions do not match the supplied test dof count.");
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(bMatrix.size()) != testDofCount * trialDofCount, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder dense B dimensions do not match the supplied test/trial dof counts.");

  TensorLocalSchurComplementBuilder::LocalGraphNormAction graphAction(testDofCount, gMatrix);
  auto gInvB = graphAction.applyInverseToColumns(bMatrix, trialDofCount);

  std::vector<double> k(static_cast<size_t>(trialDofCount * trialDofCount), 0.0);
  for (IndexType trialRow=0; trialRow<trialDofCount; trialRow++)
  {
    for (IndexType trialCol=0; trialCol<trialDofCount; trialCol++)
    {
      double entry = 0.0;
      for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
      {
        entry += bMatrix[static_cast<size_t>(testOrdinal * trialDofCount + trialRow)] * gInvB[static_cast<size_t>(testOrdinal * trialDofCount + trialCol)];
      }
      k[static_cast<size_t>(trialRow * trialDofCount + trialCol)] = entry;
    }
  }
  return k;
}

std::vector<TensorLocalSchurComplementBuilder::LocalVolumeSchurComplementBlock>
TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(const TensorMeshPtr &tensorMesh,
                                                                      const TensorDofLayoutPtr &tensorDofLayout,
                                                                      const TensorBilinearFormPtr &bilinearForm,
                                                                      const std::string &trialVarName,
                                                                      const std::string &testVarName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for local assembled cells.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for local assembled cells.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for local assembled cells.");

  auto trialDiscretization = tensorMesh->trialDiscretization(trialVarName);
  auto testDiscretization  = tensorMesh->testDiscretization(testVarName);
  std::vector<LocalVolumeSchurComplementBlock> blocks;
  blocks.reserve(tensorMesh->topology()->activeCellCount());

  for (const auto &cellTuple : tensorMesh->topology()->activeCellTuples())
  {
    auto localContext = tensorDofLayout->volumeContext(trialVarName, testVarName, cellTuple);
    const auto &trialShape = localContext.trialShape();
    const auto &testShape  = localContext.testShape();

    LocalVolumeSchurComplementBlock block;
    block.cellTuple = cellTuple;
    block.trialShape = trialShape;
    block.testShape = testShape;
    block.bMatrix.assign(static_cast<size_t>(testShape.totalDofs() * trialShape.totalDofs()), 0.0);

    bool assembledAnyTerm = false;
    for (const auto &term : bilinearForm->terms())
    {
      if (term.trialVar()->name() != trialVarName) continue;
      if (term.testVar()->name()  != testVarName)  continue;
      if (!term.isVolumeTerm()) continue;

      CoefficientFactorExpansion coefficientFactorTerms;
      const bool supportsSeparableCoefficientFactors = termSupportsAffineL2VolumeCellSchurComplement(term, coefficientFactorTerms);
      TEUCHOS_TEST_FOR_EXCEPTION(!supportsSeparableCoefficientFactors && !termSupportsFullTensorCoefficientVolumeCellSchurComplement(term), std::invalid_argument,
                                 "TensorLocalSchurComplementBuilder encountered an unsupported affine L2 volume term for local assembled-cell construction.");

      std::vector<int> quadratureDegrees(term.componentCount(), 1);
      for (int d=0; d<term.componentCount(); d++)
      {
        const int trialPolyOrder = tensorMesh->effectivePolyOrder(trialDiscretization, cellTuple[d], d, false);
        const int testPolyOrder  = tensorMesh->effectivePolyOrder(testDiscretization,  cellTuple[d], d, true);
        quadratureDegrees[d] = quadratureDegreeForComponent(trialPolyOrder,
                                                            testPolyOrder,
                                                            term.trialOperator().takesDerivativeInComponent(d),
                                                            term.testOperator().takesDerivativeInComponent(d));
      }

      TensorBasisCacheND trialBasisCache(tensorMesh, trialDiscretization, false, cellTuple, quadratureDegrees);
      TensorBasisCacheND testBasisCache (tensorMesh, testDiscretization,  true,  cellTuple, quadratureDegrees);
      if (supportsSeparableCoefficientFactors)
      {
        for (const auto &coefficientFactors : coefficientFactorTerms)
        {
          auto bOperator = buildVolumeBOperator(term,
                                                trialShape,
                                                testShape,
                                                trialBasisCache,
                                                testBasisCache,
                                                coefficientFactors);
          auto termB = denseBMatrix(bOperator);
          for (size_t entryOrdinal=0; entryOrdinal<block.bMatrix.size(); entryOrdinal++)
          {
            block.bMatrix[entryOrdinal] += termB[entryOrdinal];
          }
        }
      }
      else
      {
        auto termB = denseVolumeBMatrixByTensorQuadrature(term,
                                                          trialShape,
                                                          testShape,
                                                          trialBasisCache,
                                                          testBasisCache);
        for (size_t entryOrdinal=0; entryOrdinal<block.bMatrix.size(); entryOrdinal++)
        {
          block.bMatrix[entryOrdinal] += termB[entryOrdinal];
        }
      }
      assembledAnyTerm = true;
    }

    TEUCHOS_TEST_FOR_EXCEPTION(!assembledAnyTerm, std::invalid_argument,
                               "TensorLocalSchurComplementBuilder found no supported matching affine L2 volume terms for local assembled-cell construction.");

    std::vector<int> gQuadratureDegrees(testShape.componentCount(), 1);
    for (int d=0; d<testShape.componentCount(); d++)
    {
      const int testPolyOrder = tensorMesh->effectivePolyOrder(testDiscretization, cellTuple[d], d, true);
      gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
    }
    TensorBasisCacheND gTestBasisCache(tensorMesh, testDiscretization, true, cellTuple, gQuadratureDegrees);
    block.gDiagonal = l2TestMassDiagonal(testShape, gTestBasisCache);
    block.gMatrix.assign(static_cast<size_t>(testShape.totalDofs() * testShape.totalDofs()), 0.0);
    for (IndexType testOrdinal=0; testOrdinal<testShape.totalDofs(); testOrdinal++)
    {
      block.gMatrix[static_cast<size_t>(testOrdinal * testShape.totalDofs() + testOrdinal)] = block.gDiagonal[testOrdinal];
    }
    block.kMatrix = formDiagonalTestNormSchurComplement(block.bMatrix,
                                                        testShape.totalDofs(),
                                                        trialShape.totalDofs(),
                                                        block.gDiagonal);
    blocks.push_back(block);
  }

  return blocks;
}

TMatrixPtr<double> TensorLocalSchurComplementBuilder::affineL2VolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                                          const TensorDofLayoutPtr &tensorDofLayout,
                                                                                          const TensorBilinearFormPtr &bilinearForm,
                                                                                          const std::string &trialVarName,
                                                                                          const std::string &testVarName,
                                                                                          Teuchos_CommPtr comm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for CRS assembly.");
  if (comm == Teuchos::null)
  {
    comm = MPIWrapper::TeuchosCommSerial();
  }

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType coefficientCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);

  using Matrix = TMatrix<double>;
  using GO = typename Matrix::global_ordinal_type;
  using Scalar = double;

  Teuchos::Array<GO> rowGids(coefficientCount);
  for (IndexType gid=0; gid<coefficientCount; gid++)
  {
    rowGids[gid] = static_cast<GO>(gid);
  }
  TMapPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                      rowGids(),
                                      0,
                                      comm));

  auto blocks = affineL2VolumeCellSchurComplements(tensorMesh,
                                                   tensorDofLayout,
                                                   bilinearForm,
                                                   trialVarName,
                                                   testVarName);

  std::vector<std::set<GO>> rowColumns(coefficientCount);
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
    TEUCHOS_TEST_FOR_EXCEPTION(localDofCount != block.trialShape.totalDofs(), std::logic_error,
                               "TensorLocalSchurComplementBuilder local CRS graph size does not match the local K block size.");

    std::vector<GO> localGids(localDofCount);
    for (IndexType localOrdinal=0; localOrdinal<localDofCount; localOrdinal++)
    {
      localGids[localOrdinal] = static_cast<GO>(operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                                                       trialVarName,
                                                                                       tensorCellOrdinal,
                                                                                       localOrdinal));
    }

    for (IndexType localRow=0; localRow<localDofCount; localRow++)
    {
      TEUCHOS_TEST_FOR_EXCEPTION((localGids[localRow] < 0) || (localGids[localRow] >= coefficientCount), std::logic_error,
                                 "TensorLocalSchurComplementBuilder encountered an out-of-range global row index during CRS graph construction.");
      for (IndexType localCol=0; localCol<localDofCount; localCol++)
      {
        TEUCHOS_TEST_FOR_EXCEPTION((localGids[localCol] < 0) || (localGids[localCol] >= coefficientCount), std::logic_error,
                                   "TensorLocalSchurComplementBuilder encountered an out-of-range global column index during CRS graph construction.");
        rowColumns[localGids[localRow]].insert(localGids[localCol]);
      }
    }
  }

  size_t maxEntriesPerRow = 1;
  for (const auto &columns : rowColumns)
  {
    maxEntriesPerRow = std::max(maxEntriesPerRow, columns.size());
  }

  TMatrixPtr<double> matrix = Teuchos::rcp(new Matrix(map, maxEntriesPerRow));
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);

    Teuchos::Array<GO> cols(localDofCount);
    for (IndexType localCol=0; localCol<localDofCount; localCol++)
    {
      cols[localCol] = static_cast<GO>(operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                                              trialVarName,
                                                                              tensorCellOrdinal,
                                                                              localCol));
    }

    for (IndexType localRow=0; localRow<localDofCount; localRow++)
    {
      const GO row = static_cast<GO>(operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                                           trialVarName,
                                                                           tensorCellOrdinal,
                                                                           localRow));
      Teuchos::Array<Scalar> values(localDofCount);
      for (IndexType localCol=0; localCol<localDofCount; localCol++)
      {
        values[localCol] = block.kMatrix[static_cast<size_t>(localRow * localDofCount + localCol)];
      }
      matrix->insertGlobalValues(row, cols(), values());
    }
  }

  matrix->fillComplete(map, map);
  return matrix;
}

std::vector<TensorLocalSchurComplementBuilder::LocalVolumeSchurComplementBlock>
TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(const TensorMeshPtr &tensorMesh,
                                                                         const TensorDofLayoutPtr &tensorDofLayout,
                                                                         const TensorBilinearFormPtr &bilinearForm,
                                                                         const std::string &trialVarName,
                                                                         const std::string &testVarName,
                                                                         double l2Weight)
{
  auto blocks = affineL2VolumeCellSchurComplements(tensorMesh,
                                                   tensorDofLayout,
                                                   bilinearForm,
                                                   trialVarName,
                                                   testVarName);
  auto testDiscretization = tensorMesh->testDiscretization(testVarName);
  for (auto &block : blocks)
  {
    std::vector<int> gQuadratureDegrees(block.testShape.componentCount(), 1);
    for (int d=0; d<block.testShape.componentCount(); d++)
    {
      const int testPolyOrder = tensorMesh->effectivePolyOrder(testDiscretization, block.cellTuple[d], d, true);
      gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
    }
    TensorBasisCacheND gTestBasisCache(tensorMesh, testDiscretization, true, block.cellTuple, gQuadratureDegrees);
    block.gMatrix = graphTestNormMatrix(block.testShape, gTestBasisCache, l2Weight);
    block.gDiagonal.assign(block.testShape.totalDofs(), 0.0);
    for (IndexType testOrdinal=0; testOrdinal<block.testShape.totalDofs(); testOrdinal++)
    {
      block.gDiagonal[testOrdinal] = block.gMatrix[static_cast<size_t>(testOrdinal * block.testShape.totalDofs() + testOrdinal)];
    }
    block.kMatrix = formDenseTestNormSchurComplement(block.bMatrix,
                                                     block.testShape.totalDofs(),
                                                     block.trialShape.totalDofs(),
                                                     block.gMatrix);
  }
  return blocks;
}

TMatrixPtr<double> TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                                              const TensorDofLayoutPtr &tensorDofLayout,
                                                                                              const TensorBilinearFormPtr &bilinearForm,
                                                                                              const std::string &trialVarName,
                                                                                              const std::string &testVarName,
                                                                                              double l2Weight,
                                                                                              Teuchos_CommPtr comm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for graph-norm CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for graph-norm CRS assembly.");
  if (comm == Teuchos::null)
  {
    comm = MPIWrapper::TeuchosCommSerial();
  }

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType coefficientCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);

  using Matrix = TMatrix<double>;
  using GO = typename Matrix::global_ordinal_type;
  using Scalar = double;

  Teuchos::Array<GO> rowGids(coefficientCount);
  for (IndexType gid=0; gid<coefficientCount; gid++) rowGids[gid] = static_cast<GO>(gid);
  TMapPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), rowGids(), 0, comm));

  auto blocks = affineGraphVolumeCellSchurComplements(tensorMesh, tensorDofLayout, bilinearForm, trialVarName, testVarName, l2Weight);

  std::vector<std::set<GO>> rowColumns(coefficientCount);
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
    TEUCHOS_TEST_FOR_EXCEPTION(localDofCount != block.trialShape.totalDofs(), std::logic_error,
                               "TensorLocalSchurComplementBuilder local graph-norm CRS graph size does not match the local K block size.");
    std::vector<GO> localGids(localDofCount);
    for (IndexType localOrdinal=0; localOrdinal<localDofCount; localOrdinal++)
    {
      localGids[localOrdinal] = static_cast<GO>(operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localOrdinal));
    }
    for (IndexType localRow=0; localRow<localDofCount; localRow++)
    {
      for (IndexType localCol=0; localCol<localDofCount; localCol++) rowColumns[localGids[localRow]].insert(localGids[localCol]);
    }
  }

  size_t maxEntriesPerRow = 1;
  for (const auto &columns : rowColumns) maxEntriesPerRow = std::max(maxEntriesPerRow, columns.size());

  TMatrixPtr<double> matrix = Teuchos::rcp(new Matrix(map, maxEntriesPerRow));
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
    Teuchos::Array<GO> cols(localDofCount);
    for (IndexType localCol=0; localCol<localDofCount; localCol++)
    {
      cols[localCol] = static_cast<GO>(operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localCol));
    }
    for (IndexType localRow=0; localRow<localDofCount; localRow++)
    {
      const GO row = static_cast<GO>(operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localRow));
      Teuchos::Array<Scalar> values(localDofCount);
      for (IndexType localCol=0; localCol<localDofCount; localCol++)
      {
        values[localCol] = block.kMatrix[static_cast<size_t>(localRow * localDofCount + localCol)];
      }
      matrix->insertGlobalValues(row, cols(), values());
    }
  }

  matrix->fillComplete(map, map);
  return matrix;
}

TMatrixPtr<double> TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                                              const TensorDofLayoutPtr &tensorDofLayout,
                                                                                              const TensorBilinearFormPtr &bilinearForm,
                                                                                              const std::vector<std::string> &trialVarNames,
                                                                                              const std::vector<std::string> &testVarNames,
                                                                                              double l2Weight,
                                                                                              Teuchos_CommPtr comm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for multi-variable graph-norm CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for multi-variable graph-norm CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for multi-variable graph-norm CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder multi-variable graph-norm CRS assembly requires at least one trial variable.");
  TEUCHOS_TEST_FOR_EXCEPTION(testVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder multi-variable graph-norm CRS assembly requires at least one test variable.");
  if (comm == Teuchos::null)
  {
    comm = MPIWrapper::TeuchosCommSerial();
  }

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType trialVarCount = static_cast<IndexType>(trialVarNames.size());
  const IndexType testVarCount  = static_cast<IndexType>(testVarNames.size());
  std::vector<IndexType> trialOffsets(trialVarCount, 0);
  IndexType totalTrialDofCount = 0;
  for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }

  using Matrix = TMatrix<double>;
  using GO = typename Matrix::global_ordinal_type;
  using Scalar = double;

  Teuchos::Array<GO> rowGids(totalTrialDofCount);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++) rowGids[gid] = static_cast<GO>(gid);
  TMapPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), rowGids(), 0, comm));

  std::vector<std::vector<bool>> pairHasBlocks(testVarCount, std::vector<bool>(trialVarCount, false));
  std::vector<std::vector<std::vector<LocalVolumeSchurComplementBlock>>> pairBlocks(testVarCount,
                                                                                    std::vector<std::vector<LocalVolumeSchurComplementBlock>>(trialVarCount));
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
    {
      const std::string &trialVarName = trialVarNames[trialVarOrdinal];
      const std::string &testVarName  = testVarNames[testVarOrdinal];
      if (!hasMatchingVolumeTerms(bilinearForm, trialVarName, testVarName)) continue;

      pairBlocks[testVarOrdinal][trialVarOrdinal] = affineGraphVolumeCellSchurComplements(tensorMesh,
                                                                                          tensorDofLayout,
                                                                                          bilinearForm,
                                                                                          trialVarName,
                                                                                          testVarName,
                                                                                          l2Weight);
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(pairBlocks[testVarOrdinal][trialVarOrdinal].size()) != tensorMesh->topology()->activeCellCount(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder expected one local graph block per active tensor cell for multi-variable CRS assembly.");
      pairHasBlocks[testVarOrdinal][trialVarOrdinal] = true;
    }
  }

  std::vector<std::set<GO>> rowColumns(totalTrialDofCount);
  const IndexType activeCellCount = tensorMesh->topology()->activeCellCount();
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<activeCellCount; tensorCellOrdinal++)
    {
      for (IndexType rowTrialVarOrdinal=0; rowTrialVarOrdinal<trialVarCount; rowTrialVarOrdinal++)
      {
        if (!pairHasBlocks[testVarOrdinal][rowTrialVarOrdinal]) continue;
        const auto &rowBlock = pairBlocks[testVarOrdinal][rowTrialVarOrdinal][tensorCellOrdinal];
        const IndexType rowLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                            trialVarNames[rowTrialVarOrdinal],
                                                                            tensorCellOrdinal);
        TEUCHOS_TEST_FOR_EXCEPTION(rowLocalDofCount != rowBlock.trialShape.totalDofs(), std::logic_error,
                                   "TensorLocalSchurComplementBuilder row trial local DOF count does not match local graph B block shape.");

        for (IndexType colTrialVarOrdinal=0; colTrialVarOrdinal<trialVarCount; colTrialVarOrdinal++)
        {
          if (!pairHasBlocks[testVarOrdinal][colTrialVarOrdinal]) continue;
          const auto &colBlock = pairBlocks[testVarOrdinal][colTrialVarOrdinal][tensorCellOrdinal];
          const IndexType colLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                              trialVarNames[colTrialVarOrdinal],
                                                                              tensorCellOrdinal);
          TEUCHOS_TEST_FOR_EXCEPTION(colLocalDofCount != colBlock.trialShape.totalDofs(), std::logic_error,
                                     "TensorLocalSchurComplementBuilder column trial local DOF count does not match local graph B block shape.");
          TEUCHOS_TEST_FOR_EXCEPTION(rowBlock.testShape.totalDofs() != colBlock.testShape.totalDofs(), std::logic_error,
                                     "TensorLocalSchurComplementBuilder requires matching test shapes when forming multi-variable graph cross blocks.");

          for (IndexType localRow=0; localRow<rowLocalDofCount; localRow++)
          {
            const IndexType rowGid = trialOffsets[rowTrialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                        trialVarNames[rowTrialVarOrdinal],
                                                        tensorCellOrdinal,
                                                        localRow);
            for (IndexType localCol=0; localCol<colLocalDofCount; localCol++)
            {
              const IndexType colGid = trialOffsets[colTrialVarOrdinal]
                  + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                          trialVarNames[colTrialVarOrdinal],
                                                          tensorCellOrdinal,
                                                          localCol);
              rowColumns[rowGid].insert(static_cast<GO>(colGid));
            }
          }
        }
      }
    }
  }

  size_t maxEntriesPerRow = 1;
  for (const auto &columns : rowColumns) maxEntriesPerRow = std::max(maxEntriesPerRow, columns.size());

  TMatrixPtr<double> matrix = Teuchos::rcp(new Matrix(map, maxEntriesPerRow));
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<activeCellCount; tensorCellOrdinal++)
    {
      for (IndexType rowTrialVarOrdinal=0; rowTrialVarOrdinal<trialVarCount; rowTrialVarOrdinal++)
      {
        if (!pairHasBlocks[testVarOrdinal][rowTrialVarOrdinal]) continue;
        const auto &rowBlock = pairBlocks[testVarOrdinal][rowTrialVarOrdinal][tensorCellOrdinal];
        const IndexType rowLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                            trialVarNames[rowTrialVarOrdinal],
                                                                            tensorCellOrdinal);
        const IndexType testDofCount = rowBlock.testShape.totalDofs();

        for (IndexType colTrialVarOrdinal=0; colTrialVarOrdinal<trialVarCount; colTrialVarOrdinal++)
        {
          if (!pairHasBlocks[testVarOrdinal][colTrialVarOrdinal]) continue;
          const auto &colBlock = pairBlocks[testVarOrdinal][colTrialVarOrdinal][tensorCellOrdinal];
          const IndexType colLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                              trialVarNames[colTrialVarOrdinal],
                                                                              tensorCellOrdinal);
          auto localK = formDenseTestNormCrossSchurComplement(rowBlock.bMatrix,
                                                              testDofCount,
                                                              rowLocalDofCount,
                                                              colBlock.bMatrix,
                                                              colLocalDofCount,
                                                              rowBlock.gMatrix);

          Teuchos::Array<GO> cols(colLocalDofCount);
          for (IndexType localCol=0; localCol<colLocalDofCount; localCol++)
          {
            cols[localCol] = static_cast<GO>(trialOffsets[colTrialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                        trialVarNames[colTrialVarOrdinal],
                                                        tensorCellOrdinal,
                                                        localCol));
          }

          for (IndexType localRow=0; localRow<rowLocalDofCount; localRow++)
          {
            const GO row = static_cast<GO>(trialOffsets[rowTrialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                        trialVarNames[rowTrialVarOrdinal],
                                                        tensorCellOrdinal,
                                                        localRow));
            Teuchos::Array<Scalar> values(colLocalDofCount);
            for (IndexType localCol=0; localCol<colLocalDofCount; localCol++)
            {
              values[localCol] = localK[static_cast<size_t>(localRow * colLocalDofCount + localCol)];
            }
            matrix->insertGlobalValues(row, cols(), values());
          }
        }
      }
    }
  }

  matrix->fillComplete(map, map);
  return matrix;
}

std::vector<TensorLocalSchurComplementBuilder::LocalBoundaryBBlock>
TensorLocalSchurComplementBuilder::affineBoundaryBBlocks(const TensorMeshPtr &tensorMesh,
                                                         const TensorDofLayoutPtr &tensorDofLayout,
                                                         const TensorBilinearFormPtr &bilinearForm,
                                                         const std::string &trialVarName,
                                                         const std::string &testVarName)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for local boundary blocks.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for local boundary blocks.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for local boundary blocks.");

  auto trialDiscretization = tensorMesh->trialDiscretization(trialVarName);
  auto testDiscretization  = tensorMesh->testDiscretization(testVarName);
  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  std::vector<LocalBoundaryBBlock> blocks;

  for (const auto &term : bilinearForm->terms())
  {
    if (term.trialVar()->name() != trialVarName) continue;
    if (term.testVar()->name()  != testVarName)  continue;
    if (!term.isBoundaryTerm()) continue;
    TEUCHOS_TEST_FOR_EXCEPTION(!termSupportsAffineBoundaryB(term), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder encountered an unsupported affine boundary term for local B construction.");

    const bool trialTrace = tensorVarIsTraceLike(term.trialVar());
    const bool testTrace  = tensorVarIsTraceLike(term.testVar());
    const int orthogonalComponentOrdinal = trialTrace ? term.trialVar()->traceComponentOrdinal()
                                                      : term.testVar()->traceComponentOrdinal();
    auto skeleton = tensorMesh->topology()->tensorSkeleton(orthogonalComponentOrdinal);
    const auto retainedComponents = tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);

    for (IndexType entityOrdinal=0; entityOrdinal<skeleton.entityCount(); entityOrdinal++)
    {
      const auto &entity = skeleton.entity(entityOrdinal);
      auto localContext = tensorDofLayout->skeletonContext(trialVarName, testVarName, entity);
      const auto &trialShape = localContext.trialShape();
      const auto &testShape  = localContext.testShape();

      const ComponentCellID1D incidentCell = firstIncidentOrthogonalCell(tensorMesh, entity);
      std::vector<ComponentCellID1D> componentCells(term.componentCount(), incidentCell);
      for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
      {
        componentCells[retainedComponents[retainedOrdinal]] = entity.retainedCellTuple()[retainedOrdinal];
      }

      std::vector<int> pointCounts(term.componentCount(), 1);
      std::vector<std::vector<double>> refPoints(term.componentCount());
      std::vector<std::vector<double>> refWeights(term.componentCount());
      std::vector<double> measureScales(term.componentCount(), 1.0);
      std::vector<double> fixedCoordinates(term.componentCount(), 0.0);
      auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
      fixedCoordinates[orthogonalComponentOrdinal] = orthogonalTopo->activeVertexCoordinates().h_view(entity.vertexOrdinal());
      const IndexType leftVertex  = orthogonalTopo->cellVertexIndex(incidentCell, 0);
      const IndexType rightVertex = orthogonalTopo->cellVertexIndex(incidentCell, 1);
      double orthogonalReferenceCoordinate = 0.0;
      if (entity.vertexOrdinal() == leftVertex)
      {
        orthogonalReferenceCoordinate = -1.0;
      }
      else if (entity.vertexOrdinal() == rightVertex)
      {
        orthogonalReferenceCoordinate = 1.0;
      }
      else
      {
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                   "TensorLocalSchurComplementBuilder boundary entity vertex is not incident on the selected orthogonal cell.");
      }

      for (int d : retainedComponents)
      {
        const int trialPolyOrder = tensorMesh->effectivePolyOrder(trialDiscretization, componentCells[d], d, false);
        const int testPolyOrder  = tensorMesh->effectivePolyOrder(testDiscretization,  componentCells[d], d, true);
        const int quadratureDegree = quadratureDegreeForComponent(trialPolyOrder, testPolyOrder, false, false);
        LineQuadratureRule quadrature = lineQuadratureFromCubatureFactory(quadratureDegree);
        refPoints[d] = quadrature.points;
        refWeights[d] = quadrature.weights;
        pointCounts[d] = static_cast<int>(quadrature.points.size());
        auto componentTopo = tensorMesh->topology()->componentMeshTopology(d);
        measureScales[d] = 0.5 * componentTopo->cellWidth(componentCells[d]);
      }
      refPoints[orthogonalComponentOrdinal] = {orthogonalReferenceCoordinate};
      refWeights[orthogonalComponentOrdinal] = {1.0};

      std::vector<BoundaryBasisData1D> trialBasisData(term.componentCount());
      std::vector<BoundaryBasisData1D> testBasisData(term.componentCount());
      for (int d=0; d<term.componentCount(); d++)
      {
        if (!(trialTrace && (d == orthogonalComponentOrdinal)))
        {
          trialBasisData[d] = buildBoundaryBasisDataForComponent(tensorMesh, trialDiscretization, d, false, componentCells[d], refPoints[d]);
        }
        if (!(testTrace && (d == orthogonalComponentOrdinal)))
        {
          testBasisData[d] = buildBoundaryBasisDataForComponent(tensorMesh, testDiscretization, d, true, componentCells[d], refPoints[d]);
        }
      }

      IndexType tensorPointCount = 1;
      for (int d : retainedComponents) tensorPointCount *= pointCounts[d];
      TensorFunction::PointView physicalPoints("TensorLocalSchurComplementBuilder::boundaryPhysicalPoints",
                                               tensorPointCount,
                                               term.componentCount());
      auto physicalPointsHost = Kokkos::create_mirror_view(physicalPoints);
      std::vector<double> quadratureWeights(tensorPointCount, 1.0);
      std::vector<int> pointOrdinals(term.componentCount(), 0);
      for (IndexType flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
      {
        IndexType workingOrdinal = flatPointOrdinal;
        double weight = 1.0;
        for (int d=0; d<term.componentCount(); d++)
        {
          if (d == orthogonalComponentOrdinal) physicalPointsHost(flatPointOrdinal,d) = fixedCoordinates[d];
        }
        for (int d : retainedComponents)
        {
          pointOrdinals[d] = workingOrdinal % pointCounts[d];
          workingOrdinal /= pointCounts[d];
          const double xi = refPoints[d][pointOrdinals[d]];
          auto componentTopo = tensorMesh->topology()->componentMeshTopology(d);
          physicalPointsHost(flatPointOrdinal,d) = componentTopo->cellLeft(componentCells[d]) + 0.5 * (xi + 1.0) * componentTopo->cellWidth(componentCells[d]);
          weight *= refWeights[d][pointOrdinals[d]] * measureScales[d];
        }
        quadratureWeights[flatPointOrdinal] = weight;
      }
      Kokkos::deep_copy(physicalPoints, physicalPointsHost);

      TensorFunction::ValueView coefficientValues("TensorLocalSchurComplementBuilder::boundaryCoefficientValues",
                                                  tensorPointCount);
      term.coefficient()->values(physicalPoints, coefficientValues);
      auto coefficientValuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coefficientValues);

      const double trialOutwardNormalSign = tensorVarIsFluxLike(term.trialVar())
          ? static_cast<double>(operatorData.skeletonEntityOutwardNormalSign(TENSOR_TRIAL_SPACE, trialVarName, entityOrdinal))
          : 1.0;
      const double testOutwardNormalSign = tensorVarIsFluxLike(term.testVar())
          ? static_cast<double>(operatorData.skeletonEntityOutwardNormalSign(TENSOR_TEST_SPACE, testVarName, entityOrdinal))
          : 1.0;

      LocalBoundaryBBlock block;
      block.entity = entity;
      block.entityOrdinal = entityOrdinal;
      block.trialShape = trialShape;
      block.testShape = testShape;
      block.bMatrix.assign(static_cast<size_t>(testShape.totalDofs() * trialShape.totalDofs()), 0.0);
      const auto trialLayout = tensorDofLayout->layout(trialVarName, TENSOR_TRIAL_SPACE);
      const auto testLayout  = tensorDofLayout->layout(testVarName,  TENSOR_TEST_SPACE);
      std::vector<int> trialBasisOrdinals(term.componentCount(), 0);
      std::vector<int> testBasisOrdinals(term.componentCount(), 0);
      for (IndexType localTestOrdinal=0; localTestOrdinal<testShape.totalDofs(); localTestOrdinal++)
      {
        tensorDofLayout->decodeLocalDofOrdinal(testLayout, testShape, TENSOR_VARIABLE_ORDERING, localTestOrdinal, testBasisOrdinals);
        for (IndexType localTrialOrdinal=0; localTrialOrdinal<trialShape.totalDofs(); localTrialOrdinal++)
        {
          tensorDofLayout->decodeLocalDofOrdinal(trialLayout, trialShape, TENSOR_VARIABLE_ORDERING, localTrialOrdinal, trialBasisOrdinals);
          double entry = 0.0;
          for (IndexType flatPointOrdinal=0; flatPointOrdinal<tensorPointCount; flatPointOrdinal++)
          {
            IndexType workingOrdinal = flatPointOrdinal;
            for (int d : retainedComponents)
            {
              pointOrdinals[d] = workingOrdinal % pointCounts[d];
              workingOrdinal /= pointCounts[d];
            }
            pointOrdinals[orthogonalComponentOrdinal] = 0;
            entry += quadratureWeights[flatPointOrdinal]
                     * coefficientValuesHost(flatPointOrdinal)
                     * trialOutwardNormalSign
                     * boundaryBasisValue(term.trialVar(), trialBasisOrdinals, trialBasisData, orthogonalComponentOrdinal, pointOrdinals)
                     * testOutwardNormalSign
                     * boundaryBasisValue(term.testVar(), testBasisOrdinals, testBasisData, orthogonalComponentOrdinal, pointOrdinals);
          }
          block.bMatrix[static_cast<size_t>(localTestOrdinal * trialShape.totalDofs() + localTrialOrdinal)] = entry;
        }
      }
      blocks.push_back(block);
    }
  }

  TEUCHOS_TEST_FOR_EXCEPTION(blocks.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder found no supported matching affine boundary terms for local B construction.");
  return blocks;
}

TMatrixPtr<double> TensorLocalSchurComplementBuilder::affineL2VolumeBoundarySchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                                                  const TensorDofLayoutPtr &tensorDofLayout,
                                                                                                  const TensorBilinearFormPtr &bilinearForm,
                                                                                                  const std::vector<std::string> &trialVarNames,
                                                                                                  const std::vector<std::string> &testVarNames,
                                                                                                  Teuchos_CommPtr comm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for volume/boundary CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for volume/boundary CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for volume/boundary CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVarNames.empty() || testVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder volume/boundary CRS assembly requires trial and test variables.");
  if (comm == Teuchos::null)
  {
    comm = MPIWrapper::TeuchosCommSerial();
  }

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType trialVarCount = static_cast<IndexType>(trialVarNames.size());
  std::vector<IndexType> trialOffsets(trialVarCount, 0);
  IndexType totalTrialDofCount = 0;
  for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }

  using Matrix = TMatrix<double>;
  using GO = typename Matrix::global_ordinal_type;
  using Scalar = double;

  std::vector<double> denseK(static_cast<size_t>(totalTrialDofCount * totalTrialDofCount), 0.0);

  for (const std::string &testVarName : testVarNames)
  {
    const auto &testVariableData = operatorData.variableData(TENSOR_TEST_SPACE, testVarName);
    TEUCHOS_TEST_FOR_EXCEPTION(testVariableData.isTraceVariable(), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder volume/boundary CRS assembly currently requires volume test variables.");
    const IndexType testDofCount = testVariableData.coefficientCount();
    std::vector<double> globalB(static_cast<size_t>(testDofCount * totalTrialDofCount), 0.0);
    std::vector<double> globalGDiagonal(testDofCount, 0.0);
    TensorCoefficientMap testMap(tensorMesh, tensorDofLayout, TENSOR_TEST_SPACE, testVarName);

    const auto testDiscretization = tensorMesh->testDiscretization(testVarName);
    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<tensorMesh->topology()->activeCellCount(); tensorCellOrdinal++)
    {
      const auto cellTuple = tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
      auto testShape = tensorDofLayout->factorSpaceShape(testVarName, TENSOR_TEST_SPACE, cellTuple);
      std::vector<int> gQuadratureDegrees(testShape.componentCount(), 1);
      for (int d=0; d<testShape.componentCount(); d++)
      {
        const int testPolyOrder = tensorMesh->effectivePolyOrder(testDiscretization, cellTuple[d], d, true);
        gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
      }
      TensorBasisCacheND gTestBasisCache(tensorMesh, testDiscretization, true, cellTuple, gQuadratureDegrees);
      auto localG = l2TestMassDiagonal(testShape, gTestBasisCache);
      const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(localTestDofCount != testShape.totalDofs(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder local test shape does not match TensorOperatorData for G assembly.");
      for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
      {
        const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal, localTestOrdinal);
        globalGDiagonal[testGid] = localG[localTestOrdinal];
      }
    }

    for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
    {
      const std::string &trialVarName = trialVarNames[trialVarOrdinal];
      if (hasMatchingVolumeTerms(bilinearForm, trialVarName, testVarName))
      {
        auto blocks = affineL2VolumeCellSchurComplements(tensorMesh,
                                                         tensorDofLayout,
                                                         bilinearForm,
                                                         trialVarName,
                                                         testVarName);
        for (const auto &block : blocks)
        {
          const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
          const IndexType localTrialDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
          const IndexType localTestDofCount  = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE,  testVarName,  tensorCellOrdinal);
          TEUCHOS_TEST_FOR_EXCEPTION((localTrialDofCount != block.trialShape.totalDofs()) || (localTestDofCount != block.testShape.totalDofs()), std::logic_error,
                                     "TensorLocalSchurComplementBuilder local volume B block shape does not match TensorOperatorData.");
          for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
          {
            const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal, localTestOrdinal);
            for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
            {
              const IndexType trialGid = trialOffsets[trialVarOrdinal]
                  + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localTrialOrdinal);
              globalB[static_cast<size_t>(testGid * totalTrialDofCount + trialGid)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * localTrialDofCount + localTrialOrdinal)];
            }
          }
        }
      }

      if (hasMatchingBoundaryTerms(bilinearForm, trialVarName, testVarName))
      {
        auto boundaryBlocks = affineBoundaryBBlocks(tensorMesh,
                                                    tensorDofLayout,
                                                    bilinearForm,
                                                    trialVarName,
                                                    testVarName);
        const auto &trialVariableData = operatorData.variableData(TENSOR_TRIAL_SPACE, trialVarName);
        TensorCoefficientMap trialMap(tensorMesh, tensorDofLayout, TENSOR_TRIAL_SPACE, trialVarName);
        for (const auto &block : boundaryBlocks)
        {
          const IndexType entityOrdinal = block.entityOrdinal;
          const int orthogonalComponentOrdinal = block.entity.orthogonalComponentOrdinal();
          const auto retainedComponents = tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);
          auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
          const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(block.entity.vertexOrdinal());
          const ComponentCellID1D incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
          std::vector<ComponentCellID1D> testCells(tensorMesh->topology()->componentCount(), incidentCell);
          for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
          {
            testCells[retainedComponents[retainedOrdinal]] = block.entity.retainedCellTuple()[retainedOrdinal];
          }
          auto testIndices = testMap.indicesForVolumeCell(testCells);
          TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(testIndices.size()) != block.testShape.totalDofs(), std::logic_error,
                                     "TensorLocalSchurComplementBuilder boundary test indices do not match block test shape.");

          std::vector<IndexType> trialIndices(block.trialShape.totalDofs(), -1);
          if (trialVariableData.isTraceVariable())
          {
            for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
            {
              trialIndices[localTrialOrdinal] = operatorData.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, entityOrdinal, localTrialOrdinal);
            }
          }
          else
          {
            trialIndices = trialMap.indicesForVolumeCell(testCells);
          }

          for (IndexType localTestOrdinal=0; localTestOrdinal<block.testShape.totalDofs(); localTestOrdinal++)
          {
            const IndexType testGid = testIndices[localTestOrdinal];
            for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
            {
              const IndexType trialGid = trialOffsets[trialVarOrdinal] + trialIndices[localTrialOrdinal];
              globalB[static_cast<size_t>(testGid * totalTrialDofCount + trialGid)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * block.trialShape.totalDofs() + localTrialOrdinal)];
            }
          }
        }
      }
    }

    for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
    {
      const double gEntry = globalGDiagonal[testOrdinal];
      TEUCHOS_TEST_FOR_EXCEPTION(gEntry == 0.0, std::invalid_argument,
                                 "TensorLocalSchurComplementBuilder cannot assemble volume/boundary Schur complement without a positive L2 test diagonal.");
      for (IndexType row=0; row<totalTrialDofCount; row++)
      {
        const double left = globalB[static_cast<size_t>(testOrdinal * totalTrialDofCount + row)];
        if (left == 0.0) continue;
        for (IndexType col=0; col<totalTrialDofCount; col++)
        {
          const double right = globalB[static_cast<size_t>(testOrdinal * totalTrialDofCount + col)];
          if (right == 0.0) continue;
          denseK[static_cast<size_t>(row * totalTrialDofCount + col)] += left * right / gEntry;
        }
      }
    }
  }

  Teuchos::Array<GO> rowGids(totalTrialDofCount);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++)
  {
    rowGids[gid] = static_cast<GO>(gid);
  }
  TMapPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                      rowGids(),
                                      0,
                                      comm));

  size_t maxEntriesPerRow = 1;
  for (IndexType row=0; row<totalTrialDofCount; row++)
  {
    size_t rowEntryCount = 0;
    for (IndexType col=0; col<totalTrialDofCount; col++)
    {
      if (denseK[static_cast<size_t>(row * totalTrialDofCount + col)] != 0.0) rowEntryCount++;
    }
    maxEntriesPerRow = std::max(maxEntriesPerRow, rowEntryCount);
  }

  TMatrixPtr<double> matrix = Teuchos::rcp(new Matrix(map, maxEntriesPerRow));
  for (IndexType row=0; row<totalTrialDofCount; row++)
  {
    Teuchos::Array<GO> cols;
    Teuchos::Array<Scalar> values;
    for (IndexType col=0; col<totalTrialDofCount; col++)
    {
      const double value = denseK[static_cast<size_t>(row * totalTrialDofCount + col)];
      if (value == 0.0) continue;
      cols.push_back(static_cast<GO>(col));
      values.push_back(value);
    }
    if (!cols.empty())
    {
      matrix->insertGlobalValues(static_cast<GO>(row), cols(), values());
    }
  }

  matrix->fillComplete(map, map);
  return matrix;
}

TMatrixPtr<double> TensorLocalSchurComplementBuilder::affineGraphVolumeBoundarySchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                                                      const TensorDofLayoutPtr &tensorDofLayout,
                                                                                                      const TensorBilinearFormPtr &bilinearForm,
                                                                                                      const std::vector<std::string> &trialVarNames,
                                                                                                      const std::vector<std::string> &testVarNames,
                                                                                                      double l2Weight,
                                                                                                      Teuchos_CommPtr comm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for graph volume/boundary CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for graph volume/boundary CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for graph volume/boundary CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVarNames.empty() || testVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder graph volume/boundary CRS assembly requires trial and test variables.");
  if (comm == Teuchos::null)
  {
    comm = MPIWrapper::TeuchosCommSerial();
  }

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType trialVarCount = static_cast<IndexType>(trialVarNames.size());
  std::vector<IndexType> trialOffsets(trialVarCount, 0);
  IndexType totalTrialDofCount = 0;
  for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }

  using Matrix = TMatrix<double>;
  using GO = typename Matrix::global_ordinal_type;
  using Scalar = double;

  std::vector<double> denseK(static_cast<size_t>(totalTrialDofCount * totalTrialDofCount), 0.0);

  for (const std::string &testVarName : testVarNames)
  {
    const auto &testVariableData = operatorData.variableData(TENSOR_TEST_SPACE, testVarName);
    TEUCHOS_TEST_FOR_EXCEPTION(testVariableData.isTraceVariable(), std::invalid_argument,
                               "TensorLocalSchurComplementBuilder graph volume/boundary CRS assembly currently requires volume test variables.");
    TensorCoefficientMap testMap(tensorMesh, tensorDofLayout, TENSOR_TEST_SPACE, testVarName);
    const auto testDiscretization = tensorMesh->testDiscretization(testVarName);

    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<tensorMesh->topology()->activeCellCount(); tensorCellOrdinal++)
    {
      const auto cellTuple = tensorMesh->topology()->tensorCellTuple(tensorCellOrdinal);
      auto testShape = tensorDofLayout->factorSpaceShape(testVarName, TENSOR_TEST_SPACE, cellTuple);
      const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal);
      TEUCHOS_TEST_FOR_EXCEPTION(localTestDofCount != testShape.totalDofs(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder local test shape does not match TensorOperatorData for graph G assembly.");

      std::vector<int> gQuadratureDegrees(testShape.componentCount(), 1);
      for (int d=0; d<testShape.componentCount(); d++)
      {
        const int testPolyOrder = tensorMesh->effectivePolyOrder(testDiscretization, cellTuple[d], d, true);
        gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
      }
      TensorBasisCacheND gTestBasisCache(tensorMesh, testDiscretization, true, cellTuple, gQuadratureDegrees);
      auto localG = graphTestNormMatrix(testShape, gTestBasisCache, l2Weight);
      std::vector<double> localB(static_cast<size_t>(localTestDofCount * totalTrialDofCount), 0.0);

      for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
      {
        const std::string &trialVarName = trialVarNames[trialVarOrdinal];
        if (hasMatchingVolumeTerms(bilinearForm, trialVarName, testVarName))
        {
          auto volumeBlocks = affineL2VolumeCellSchurComplements(tensorMesh,
                                                                 tensorDofLayout,
                                                                 bilinearForm,
                                                                 trialVarName,
                                                                 testVarName);
          const auto &block = volumeBlocks[tensorCellOrdinal];
          const IndexType localTrialDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal);
          TEUCHOS_TEST_FOR_EXCEPTION((localTrialDofCount != block.trialShape.totalDofs()) || (localTestDofCount != block.testShape.totalDofs()), std::logic_error,
                                     "TensorLocalSchurComplementBuilder local graph volume B block shape does not match TensorOperatorData.");
          for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
          {
            for (IndexType localTrialOrdinal=0; localTrialOrdinal<localTrialDofCount; localTrialOrdinal++)
            {
              const IndexType trialGid = trialOffsets[trialVarOrdinal]
                  + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, tensorCellOrdinal, localTrialOrdinal);
              localB[static_cast<size_t>(localTestOrdinal * totalTrialDofCount + trialGid)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * localTrialDofCount + localTrialOrdinal)];
            }
          }
        }

        if (hasMatchingBoundaryTerms(bilinearForm, trialVarName, testVarName))
        {
          auto boundaryBlocks = affineBoundaryBBlocks(tensorMesh,
                                                      tensorDofLayout,
                                                      bilinearForm,
                                                      trialVarName,
                                                      testVarName);
          const auto &trialVariableData = operatorData.variableData(TENSOR_TRIAL_SPACE, trialVarName);
          TensorCoefficientMap trialMap(tensorMesh, tensorDofLayout, TENSOR_TRIAL_SPACE, trialVarName);
          for (const auto &block : boundaryBlocks)
          {
            const IndexType entityOrdinal = block.entityOrdinal;
            const int orthogonalComponentOrdinal = block.entity.orthogonalComponentOrdinal();
            const auto retainedComponents = tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);
            auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
            const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(block.entity.vertexOrdinal());
            const ComponentCellID1D incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
            std::vector<ComponentCellID1D> testCells(tensorMesh->topology()->componentCount(), incidentCell);
            for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
            {
              testCells[retainedComponents[retainedOrdinal]] = block.entity.retainedCellTuple()[retainedOrdinal];
            }
            if (tensorMesh->topology()->tensorCellOrdinal(testCells) != tensorCellOrdinal) continue;

            auto testIndices = testMap.indicesForVolumeCell(testCells);
            TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(testIndices.size()) != block.testShape.totalDofs(), std::logic_error,
                                       "TensorLocalSchurComplementBuilder graph boundary test indices do not match block test shape.");

            std::vector<IndexType> trialIndices(block.trialShape.totalDofs(), -1);
            if (trialVariableData.isTraceVariable())
            {
              for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
              {
                trialIndices[localTrialOrdinal] = operatorData.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, trialVarName, entityOrdinal, localTrialOrdinal);
              }
            }
            else
            {
              trialIndices = trialMap.indicesForVolumeCell(testCells);
            }

            for (IndexType localTestOrdinal=0; localTestOrdinal<block.testShape.totalDofs(); localTestOrdinal++)
            {
              const IndexType globalTestGid = testIndices[localTestOrdinal];
              IndexType cellLocalTestOrdinal = -1;
              for (IndexType candidate=0; candidate<localTestDofCount; candidate++)
              {
                if (operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, testVarName, tensorCellOrdinal, candidate) == globalTestGid)
                {
                  cellLocalTestOrdinal = candidate;
                  break;
                }
              }
              TEUCHOS_TEST_FOR_EXCEPTION(cellLocalTestOrdinal < 0, std::logic_error,
                                         "TensorLocalSchurComplementBuilder could not map a boundary test index into the graph volume cell.");
              for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
              {
                const IndexType trialGid = trialOffsets[trialVarOrdinal] + trialIndices[localTrialOrdinal];
                localB[static_cast<size_t>(cellLocalTestOrdinal * totalTrialDofCount + trialGid)] += block.bMatrix[static_cast<size_t>(localTestOrdinal * block.trialShape.totalDofs() + localTrialOrdinal)];
              }
            }
          }
        }
      }

      auto localK = formDenseTestNormCrossSchurComplement(localB,
                                                          localTestDofCount,
                                                          totalTrialDofCount,
                                                          localB,
                                                          totalTrialDofCount,
                                                          localG);
      for (IndexType row=0; row<totalTrialDofCount; row++)
      {
        for (IndexType col=0; col<totalTrialDofCount; col++)
        {
          denseK[static_cast<size_t>(row * totalTrialDofCount + col)] += localK[static_cast<size_t>(row * totalTrialDofCount + col)];
        }
      }
    }
  }

  Teuchos::Array<GO> rowGids(totalTrialDofCount);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++) rowGids[gid] = static_cast<GO>(gid);
  TMapPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(), rowGids(), 0, comm));

  size_t maxEntriesPerRow = 1;
  for (IndexType row=0; row<totalTrialDofCount; row++)
  {
    size_t rowEntryCount = 0;
    for (IndexType col=0; col<totalTrialDofCount; col++)
    {
      if (denseK[static_cast<size_t>(row * totalTrialDofCount + col)] != 0.0) rowEntryCount++;
    }
    maxEntriesPerRow = std::max(maxEntriesPerRow, rowEntryCount);
  }

  TMatrixPtr<double> matrix = Teuchos::rcp(new Matrix(map, maxEntriesPerRow));
  for (IndexType row=0; row<totalTrialDofCount; row++)
  {
    Teuchos::Array<GO> cols;
    Teuchos::Array<Scalar> values;
    for (IndexType col=0; col<totalTrialDofCount; col++)
    {
      const double value = denseK[static_cast<size_t>(row * totalTrialDofCount + col)];
      if (value == 0.0) continue;
      cols.push_back(static_cast<GO>(col));
      values.push_back(value);
    }
    if (!cols.empty())
    {
      matrix->insertGlobalValues(static_cast<GO>(row), cols(), values());
    }
  }

  matrix->fillComplete(map, map);
  return matrix;
}

TMatrixPtr<double> TensorLocalSchurComplementBuilder::affineL2VolumeSchurComplementMatrix(const TensorMeshPtr &tensorMesh,
                                                                                          const TensorDofLayoutPtr &tensorDofLayout,
                                                                                          const TensorBilinearFormPtr &bilinearForm,
                                                                                          const std::vector<std::string> &trialVarNames,
                                                                                          const std::vector<std::string> &testVarNames,
                                                                                          Teuchos_CommPtr comm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(tensorMesh == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorMesh for multi-variable CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(tensorDofLayout == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorDofLayout for multi-variable CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(bilinearForm == Teuchos::null, std::invalid_argument,
                             "TensorLocalSchurComplementBuilder requires a non-null TensorBilinearForm for multi-variable CRS assembly.");
  TEUCHOS_TEST_FOR_EXCEPTION(trialVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder multi-variable CRS assembly requires at least one trial variable.");
  TEUCHOS_TEST_FOR_EXCEPTION(testVarNames.empty(), std::invalid_argument,
                             "TensorLocalSchurComplementBuilder multi-variable CRS assembly requires at least one test variable.");
  if (comm == Teuchos::null)
  {
    comm = MPIWrapper::TeuchosCommSerial();
  }

  TensorOperatorData operatorData(tensorMesh, tensorDofLayout);
  const IndexType trialVarCount = static_cast<IndexType>(trialVarNames.size());
  const IndexType testVarCount  = static_cast<IndexType>(testVarNames.size());
  std::vector<IndexType> trialOffsets(trialVarCount, 0);
  IndexType totalTrialDofCount = 0;
  for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }

  using Matrix = TMatrix<double>;
  using GO = typename Matrix::global_ordinal_type;
  using Scalar = double;

  Teuchos::Array<GO> rowGids(totalTrialDofCount);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++)
  {
    rowGids[gid] = static_cast<GO>(gid);
  }
  TMapPtr map = Teuchos::rcp(new TMap(Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid(),
                                      rowGids(),
                                      0,
                                      comm));

  std::vector<std::vector<bool>> pairHasBlocks(testVarCount, std::vector<bool>(trialVarCount, false));
  std::vector<std::vector<std::vector<LocalVolumeSchurComplementBlock>>> pairBlocks(testVarCount,
                                                                                    std::vector<std::vector<LocalVolumeSchurComplementBlock>>(trialVarCount));
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    for (IndexType trialVarOrdinal=0; trialVarOrdinal<trialVarCount; trialVarOrdinal++)
    {
      const std::string &trialVarName = trialVarNames[trialVarOrdinal];
      const std::string &testVarName  = testVarNames[testVarOrdinal];
      if (!hasMatchingVolumeTerms(bilinearForm, trialVarName, testVarName)) continue;

      pairBlocks[testVarOrdinal][trialVarOrdinal] = affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                       tensorDofLayout,
                                                                                       bilinearForm,
                                                                                       trialVarName,
                                                                                       testVarName);
      TEUCHOS_TEST_FOR_EXCEPTION(static_cast<IndexType>(pairBlocks[testVarOrdinal][trialVarOrdinal].size()) != tensorMesh->topology()->activeCellCount(), std::logic_error,
                                 "TensorLocalSchurComplementBuilder expected one local block per active tensor cell for multi-variable CRS assembly.");
      pairHasBlocks[testVarOrdinal][trialVarOrdinal] = true;
    }
  }

  std::vector<std::set<GO>> rowColumns(totalTrialDofCount);
  const IndexType activeCellCount = tensorMesh->topology()->activeCellCount();
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<activeCellCount; tensorCellOrdinal++)
    {
      for (IndexType rowTrialVarOrdinal=0; rowTrialVarOrdinal<trialVarCount; rowTrialVarOrdinal++)
      {
        if (!pairHasBlocks[testVarOrdinal][rowTrialVarOrdinal]) continue;
        const auto &rowBlock = pairBlocks[testVarOrdinal][rowTrialVarOrdinal][tensorCellOrdinal];
        const IndexType rowLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                            trialVarNames[rowTrialVarOrdinal],
                                                                            tensorCellOrdinal);
        TEUCHOS_TEST_FOR_EXCEPTION(rowLocalDofCount != rowBlock.trialShape.totalDofs(), std::logic_error,
                                   "TensorLocalSchurComplementBuilder row trial local DOF count does not match local B block shape.");

        for (IndexType colTrialVarOrdinal=0; colTrialVarOrdinal<trialVarCount; colTrialVarOrdinal++)
        {
          if (!pairHasBlocks[testVarOrdinal][colTrialVarOrdinal]) continue;
          const auto &colBlock = pairBlocks[testVarOrdinal][colTrialVarOrdinal][tensorCellOrdinal];
          const IndexType colLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                              trialVarNames[colTrialVarOrdinal],
                                                                              tensorCellOrdinal);
          TEUCHOS_TEST_FOR_EXCEPTION(colLocalDofCount != colBlock.trialShape.totalDofs(), std::logic_error,
                                     "TensorLocalSchurComplementBuilder column trial local DOF count does not match local B block shape.");
          TEUCHOS_TEST_FOR_EXCEPTION(rowBlock.testShape.totalDofs() != colBlock.testShape.totalDofs(), std::logic_error,
                                     "TensorLocalSchurComplementBuilder requires matching test shapes when forming multi-variable cross blocks.");

          for (IndexType localRow=0; localRow<rowLocalDofCount; localRow++)
          {
            const IndexType rowGid = trialOffsets[rowTrialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                        trialVarNames[rowTrialVarOrdinal],
                                                        tensorCellOrdinal,
                                                        localRow);
            for (IndexType localCol=0; localCol<colLocalDofCount; localCol++)
            {
              const IndexType colGid = trialOffsets[colTrialVarOrdinal]
                  + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                          trialVarNames[colTrialVarOrdinal],
                                                          tensorCellOrdinal,
                                                          localCol);
              rowColumns[rowGid].insert(static_cast<GO>(colGid));
            }
          }
        }
      }
    }
  }

  size_t maxEntriesPerRow = 1;
  for (const auto &columns : rowColumns)
  {
    maxEntriesPerRow = std::max(maxEntriesPerRow, columns.size());
  }

  TMatrixPtr<double> matrix = Teuchos::rcp(new Matrix(map, maxEntriesPerRow));
  for (IndexType testVarOrdinal=0; testVarOrdinal<testVarCount; testVarOrdinal++)
  {
    for (IndexType tensorCellOrdinal=0; tensorCellOrdinal<activeCellCount; tensorCellOrdinal++)
    {
      for (IndexType rowTrialVarOrdinal=0; rowTrialVarOrdinal<trialVarCount; rowTrialVarOrdinal++)
      {
        if (!pairHasBlocks[testVarOrdinal][rowTrialVarOrdinal]) continue;
        const auto &rowBlock = pairBlocks[testVarOrdinal][rowTrialVarOrdinal][tensorCellOrdinal];
        const IndexType rowLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                            trialVarNames[rowTrialVarOrdinal],
                                                                            tensorCellOrdinal);
        const IndexType testDofCount = rowBlock.testShape.totalDofs();

        for (IndexType colTrialVarOrdinal=0; colTrialVarOrdinal<trialVarCount; colTrialVarOrdinal++)
        {
          if (!pairHasBlocks[testVarOrdinal][colTrialVarOrdinal]) continue;
          const auto &colBlock = pairBlocks[testVarOrdinal][colTrialVarOrdinal][tensorCellOrdinal];
          const IndexType colLocalDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE,
                                                                              trialVarNames[colTrialVarOrdinal],
                                                                              tensorCellOrdinal);

          Teuchos::Array<GO> cols(colLocalDofCount);
          for (IndexType localCol=0; localCol<colLocalDofCount; localCol++)
          {
            cols[localCol] = static_cast<GO>(trialOffsets[colTrialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                        trialVarNames[colTrialVarOrdinal],
                                                        tensorCellOrdinal,
                                                        localCol));
          }

          for (IndexType localRow=0; localRow<rowLocalDofCount; localRow++)
          {
            const GO row = static_cast<GO>(trialOffsets[rowTrialVarOrdinal]
                + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE,
                                                        trialVarNames[rowTrialVarOrdinal],
                                                        tensorCellOrdinal,
                                                        localRow));
            Teuchos::Array<Scalar> values(colLocalDofCount);
            for (IndexType localCol=0; localCol<colLocalDofCount; localCol++)
            {
              double entry = 0.0;
              for (IndexType localTestOrdinal=0; localTestOrdinal<testDofCount; localTestOrdinal++)
              {
                const double gEntry = rowBlock.gDiagonal[localTestOrdinal];
                TEUCHOS_TEST_FOR_EXCEPTION(gEntry == 0.0, std::invalid_argument,
                                           "TensorLocalSchurComplementBuilder cannot form a multi-variable Schur complement with a zero G diagonal entry.");
                entry += rowBlock.bMatrix[static_cast<size_t>(localTestOrdinal * rowLocalDofCount + localRow)]
                         * colBlock.bMatrix[static_cast<size_t>(localTestOrdinal * colLocalDofCount + localCol)]
                         / gEntry;
              }
              values[localCol] = entry;
            }
            matrix->insertGlobalValues(row, cols(), values());
          }
        }
      }
    }
  }

  matrix->fillComplete(map, map);
  return matrix;
}
