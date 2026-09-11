#include "Teuchos_UnitTestHarness.hpp"

#include "TensorBasisCache.h"
#include "TensorBilinearForm.h"
#include "TensorCoefficientMap.h"
#include "TensorDofLayout.h"
#include "TensorLocalSchurComplement.h"
#include "TensorMesh.h"
#include "TensorOperatorData.h"
#include "TensorOperatorFactory.h"
#include "BasisFamily.h"
#include "Var.h"

#include <algorithm>
#include <cmath>
#include <vector>

using namespace Camellia;

namespace
{
TensorFactorSpaceShape makeShape(const std::vector<int> &cardinalities)
{
  std::vector<IndexType> strides(cardinalities.size(), 1);
  IndexType totalDofs = 1;
  for (int d=0; d<static_cast<int>(cardinalities.size()); d++)
  {
    strides[d] = totalDofs;
    totalDofs *= cardinalities[d];
  }
  return TensorFactorSpaceShape(cardinalities, strides, totalDofs);
}

TensorVarPtr makeFieldTensorVar(const std::string &name,
                                int componentCount)
{
  std::vector<VarPtr> componentVars;
  for (int d=0; d<componentCount; d++)
  {
    componentVars.push_back(Teuchos::rcp(new Var(d, 0, name + "_" + std::to_string(d), Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS)));
  }
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, componentVars));
}

TensorVarPtr makeFluxTensorVar(const std::string &name,
                               int componentCount,
                               int traceComponentOrdinal)
{
  std::vector<VarPtr> componentVars;
  for (int d=0; d<componentCount; d++)
  {
    componentVars.push_back(Teuchos::rcp(new Var(d, 0, name + "_" + std::to_string(d), Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  }
  return Teuchos::rcp(new TensorVar(name, TENSOR_TRACE, componentVars, traceComponentOrdinal));
}

TensorMeshPtr makeSingleCellTensorMesh(const std::vector<std::pair<double,double>> &componentExtents)
{
  std::vector<Teuchos::RCP<ComponentMeshTopology1D>> components;
  for (int d=0; d<static_cast<int>(componentExtents.size()); d++)
  {
    components.push_back(Teuchos::rcp(new ComponentMeshTopology1D({componentExtents[d].first, componentExtents[d].second},
                                                                  { ActiveLeafSet1D({{0,0}}) },
                                                                  false,
                                                                  1)));
  }
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology(components));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  const int componentCount = static_cast<int>(componentExtents.size());
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u", componentCount), Teuchos::null, std::vector<int>(componentCount, 0))));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("v", componentCount), Teuchos::null, std::vector<int>(componentCount, 0))));
  return tensorMesh;
}

TensorMeshPtr makeVlasovLikeVolumeAffineTensorMesh1X1T1V()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 2));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-2.0, 2.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1, component2}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("f", 3), Teuchos::null, std::vector<int>{0,0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("w", 3), Teuchos::null, std::vector<int>{0,0,0})));
  return tensorMesh;
}

TensorMeshPtr makeVlasovLikeVolumeAffineTensorMesh1D1T2V()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 2));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-2.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 2));
  auto component3 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1, component2, component3}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("f", 4), Teuchos::null, std::vector<int>{0,0,0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("w", 4), Teuchos::null, std::vector<int>{0,0,0,0})));
  return tensorMesh;
}

TensorMeshPtr makeTwoByTwoAffineTensorMesh()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 3.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u", 2), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("v", 2), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
}

TensorMeshPtr makeTwoTrialOneTestAffineTensorMesh()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 3.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("q", 2), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u", 2), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("v", 2), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
}

TensorMeshPtr makeTraceVolumeAffineTensorMesh()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("f", 2), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFluxTensorVar("tn", 2, 0), Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("w", 2), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
}

TensorMeshPtr makeDirectionalTraceVolumeAffineTensorMesh()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 2));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("f", 2), Teuchos::null, std::vector<int>{0,0})));

  std::vector<VarPtr> componentVars;
  componentVars.push_back(Teuchos::rcp(new Var(10, 0, "tn_x_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  componentVars.push_back(Teuchos::rcp(new Var(11, 0, "tn_t_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  auto directionalTraces = TensorVarFactory::directionalTraceFamily("tn", {"x", "t"}, componentVars);
  for (const auto &traceVar : directionalTraces)
  {
    tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(traceVar, Teuchos::null, std::vector<int>{0,0})));
  }
  tensorMesh->addTestDiscretization (Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("w", 2), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
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

std::vector<int> quadratureDegreesForTerm(const TensorMeshPtr &tensorMesh,
                                          const TensorBilinearTerm &term,
                                          const TensorMeshTopology::TensorCellTuple &cellTuple)
{
  auto trialDiscretization = tensorMesh->trialDiscretization(term.trialVar()->name());
  auto testDiscretization  = tensorMesh->testDiscretization(term.testVar()->name());
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
  return quadratureDegrees;
}

void decodeOrdinal(const TensorFactorSpaceShape &shape,
                   IndexType ordinal,
                   std::vector<int> &componentOrdinals)
{
  componentOrdinals.assign(shape.componentCount(), 0);
  for (int d=0; d<shape.componentCount(); d++)
  {
    componentOrdinals[d] = static_cast<int>((ordinal / shape.componentStride(d)) % shape.componentCardinality(d));
  }
}

double linearBasisValue(int basisOrdinal,
                        double xi)
{
  return (basisOrdinal == 0) ? 0.5 * (1.0 - xi) : 0.5 * (1.0 + xi);
}

IndexType localVolumeOrdinalForGlobalIndex(const TensorOperatorData &operatorData,
                                           TensorSpaceRole role,
                                           const std::string &varName,
                                           IndexType tensorCellOrdinal,
                                           IndexType globalIndex)
{
  const IndexType localDofCount = operatorData.volumeLocalDofCount(role, varName, tensorCellOrdinal);
  for (IndexType localOrdinal=0; localOrdinal<localDofCount; localOrdinal++)
  {
    if (operatorData.volumeLocalToGlobalIndex(role, varName, tensorCellOrdinal, localOrdinal) == globalIndex)
    {
      return localOrdinal;
    }
  }
  return -1;
}

std::vector<double> componentMassEntries(double cellWidth)
{
  const double a = 1.0 / std::sqrt(3.0);
  const std::vector<double> points{-a, a};
  const double jacobian = 0.5 * cellWidth;

  std::vector<double> entries(4, 0.0);
  for (int row=0; row<2; row++)
  {
    for (int col=0; col<2; col++)
    {
      double value = 0.0;
      for (double xi : points)
      {
        value += jacobian * linearBasisValue(row, xi) * linearBasisValue(col, xi);
      }
      entries[row * 2 + col] = value;
    }
  }
  return entries;
}

std::vector<double> componentMassDiagonal(double cellWidth)
{
  auto entries = componentMassEntries(cellWidth);
  return {entries[0], entries[3]};
}

double explicitTensorBEntry(const TensorFactorSpaceShape &testShape,
                            const TensorFactorSpaceShape &trialShape,
                            IndexType testOrdinal,
                            IndexType trialOrdinal,
                            const std::vector<double> &cellWidths)
{
  const double a = 1.0 / std::sqrt(3.0);
  const std::vector<double> points{-a, a};
  std::vector<int> testComponentOrdinals;
  std::vector<int> trialComponentOrdinals;
  decodeOrdinal(testShape, testOrdinal, testComponentOrdinals);
  decodeOrdinal(trialShape, trialOrdinal, trialComponentOrdinals);

  double entry = 0.0;
  for (double xi0 : points)
  {
    for (double xi1 : points)
    {
      const std::vector<double> xis{xi0, xi1};
      double value = 1.0;
      for (int d=0; d<testShape.componentCount(); d++)
      {
        value *= (0.5 * cellWidths[d])
                 * linearBasisValue(testComponentOrdinals[d], xis[d])
                 * linearBasisValue(trialComponentOrdinals[d], xis[d]);
      }
      entry += value;
    }
  }
  return entry;
}

double explicitTensorGDiagonalEntry(const TensorFactorSpaceShape &testShape,
                                    IndexType testOrdinal,
                                    const std::vector<double> &cellWidths)
{
  const double a = 1.0 / std::sqrt(3.0);
  const std::vector<double> points{-a, a};
  std::vector<int> testComponentOrdinals;
  decodeOrdinal(testShape, testOrdinal, testComponentOrdinals);

  double entry = 0.0;
  for (double xi0 : points)
  {
    for (double xi1 : points)
    {
      const std::vector<double> xis{xi0, xi1};
      double value = 1.0;
      for (int d=0; d<testShape.componentCount(); d++)
      {
        const double basisValue = linearBasisValue(testComponentOrdinals[d], xis[d]);
        value *= (0.5 * cellWidths[d]) * basisValue * basisValue;
      }
      entry += value;
    }
  }
  return entry;
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

std::vector<double> coefficientValues(const TensorFunctionPtr &coefficient,
                                      const TensorBasisCacheND &basisCache)
{
  TensorFunction::ValueView values("local-schur-test-coefficient-values", basisCache.tensorPointCount());
  coefficient->values(basisCache.physicalPoints().d_view, values);
  auto valuesHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), values);
  std::vector<double> result(basisCache.tensorPointCount(), 0.0);
  for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
  {
    result[pointOrdinal] = valuesHost(pointOrdinal);
  }
  return result;
}

bool closeEnough(double actual,
                 double expected,
                 double relTol,
                 double absTol)
{
  const double scale = std::max(std::abs(actual), std::abs(expected));
  if (scale <= absTol)
  {
    return std::abs(actual - expected) <= absTol;
  }
  return std::abs(actual - expected) <= relTol * scale;
}

std::vector<double> solveDenseLinearSystem(std::vector<double> matrix,
                                           std::vector<double> rhs,
                                           IndexType dimension)
{
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
                               "TensorLocalSchurComplementTests cannot solve a singular dense matrix.");
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

std::vector<double> applyDenseSchurAction(const std::vector<double> &bMatrix,
                                          IndexType testDofCount,
                                          IndexType trialDofCount,
                                          const std::vector<double> &rhs,
                                          const std::vector<double> &gMatrix)
{
  auto gInverseRhs = solveDenseLinearSystem(gMatrix, rhs, testDofCount);
  std::vector<double> result(trialDofCount, 0.0);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
    {
      result[trialOrdinal] += bMatrix[static_cast<size_t>(testOrdinal * trialDofCount + trialOrdinal)] * gInverseRhs[testOrdinal];
    }
  }
  return result;
}

double explicitCachedBEntry(const TensorBilinearTerm &term,
                            const TensorDofLayoutPtr &dofLayout,
                            const TensorDofLayout::VariableLayout &trialLayout,
                            const TensorDofLayout::VariableLayout &testLayout,
                            const TensorFactorSpaceShape &trialShape,
                            const TensorFactorSpaceShape &testShape,
                            const TensorBasisCacheND &trialCache,
                            const TensorBasisCacheND &testCache,
                            const std::vector<double> &coefficientValues,
                            IndexType testOrdinal,
                            IndexType trialOrdinal)
{
  std::vector<int> trialBasisOrdinals(term.componentCount(), 0);
  std::vector<int> testBasisOrdinals(term.componentCount(), 0);
  std::vector<int> pointOrdinals(term.componentCount(), 0);
  dofLayout->decodeLocalDofOrdinal(trialLayout, trialShape, TENSOR_VARIABLE_ORDERING, trialOrdinal, trialBasisOrdinals);
  dofLayout->decodeLocalDofOrdinal(testLayout,  testShape,  TENSOR_VARIABLE_ORDERING, testOrdinal,  testBasisOrdinals);

  auto weights = trialCache.quadratureWeights();
  weights.sync_host();

  double entry = 0.0;
  for (IndexType flatPointOrdinal=0; flatPointOrdinal<trialCache.tensorPointCount(); flatPointOrdinal++)
  {
    IndexType workingOrdinal = flatPointOrdinal;
    for (int d=0; d<term.componentCount(); d++)
    {
      pointOrdinals[d] = workingOrdinal % trialCache.componentCache(d).pointCount();
      workingOrdinal /= trialCache.componentCache(d).pointCount();
    }
    entry += weights.h_view(flatPointOrdinal)
             * coefficientValues[flatPointOrdinal]
             * cachedBasisEvaluationAtPoint(trialBasisOrdinals, trialCache, term.trialOperator(), pointOrdinals)
             * cachedBasisEvaluationAtPoint(testBasisOrdinals,  testCache,  term.testOperator(),  pointOrdinals);
  }
  return entry;
}

double explicitCachedGraphGEntry(const TensorDofLayoutPtr &dofLayout,
                                 const TensorDofLayout::VariableLayout &testLayout,
                                 const TensorFactorSpaceShape &testShape,
                                 const TensorBasisCacheND &testCache,
                                 double l2Weight,
                                 IndexType rowOrdinal,
                                 IndexType colOrdinal)
{
  std::vector<int> rowBasisOrdinals(testShape.componentCount(), 0);
  std::vector<int> colBasisOrdinals(testShape.componentCount(), 0);
  std::vector<int> pointOrdinals(testShape.componentCount(), 0);
  dofLayout->decodeLocalDofOrdinal(testLayout, testShape, TENSOR_VARIABLE_ORDERING, rowOrdinal, rowBasisOrdinals);
  dofLayout->decodeLocalDofOrdinal(testLayout, testShape, TENSOR_VARIABLE_ORDERING, colOrdinal, colBasisOrdinals);

  auto weights = testCache.quadratureWeights();
  weights.sync_host();
  double entry = 0.0;
  for (IndexType flatPointOrdinal=0; flatPointOrdinal<testCache.tensorPointCount(); flatPointOrdinal++)
  {
    IndexType workingOrdinal = flatPointOrdinal;
    for (int d=0; d<testShape.componentCount(); d++)
    {
      pointOrdinals[d] = workingOrdinal % testCache.componentCache(d).pointCount();
      workingOrdinal /= testCache.componentCache(d).pointCount();
    }

    const auto valueDescriptor = TensorComponentOperatorDescriptor::valueOnly(testShape.componentCount());
    const double rowValue = cachedBasisEvaluationAtPoint(rowBasisOrdinals, testCache, valueDescriptor, pointOrdinals);
    const double colValue = cachedBasisEvaluationAtPoint(colBasisOrdinals, testCache, valueDescriptor, pointOrdinals);
    entry += weights.h_view(flatPointOrdinal) * l2Weight * rowValue * colValue;
    for (int derivativeComponent=0; derivativeComponent<testShape.componentCount(); derivativeComponent++)
    {
      const auto derivativeDescriptor = TensorComponentOperatorDescriptor::singleDerivative(testShape.componentCount(), derivativeComponent);
      const double rowDerivative = cachedBasisEvaluationAtPoint(rowBasisOrdinals, testCache, derivativeDescriptor, pointOrdinals);
      const double colDerivative = cachedBasisEvaluationAtPoint(colBasisOrdinals, testCache, derivativeDescriptor, pointOrdinals);
      entry += weights.h_view(flatPointOrdinal) * rowDerivative * colDerivative;
    }
  }
  return entry;
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, LocalBAndDiagonalGMatchExplicitTensorQuadratureReference )
{
  auto shape = makeShape({2,2});
  const std::vector<double> cellWidths{2.0, 3.0};
  TensorProductLocalOperator bOperator(shape,
                                       shape,
                                       {TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[0])),
                                        TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[1]))});

  auto bMatrix = TensorLocalSchurComplementBuilder::denseBMatrix(bOperator);
  TEST_EQUALITY(bMatrix.size(), static_cast<size_t>(shape.totalDofs() * shape.totalDofs()));
  for (IndexType row=0; row<shape.totalDofs(); row++)
  {
    for (IndexType col=0; col<shape.totalDofs(); col++)
    {
      const double expected = explicitTensorBEntry(shape, shape, row, col, cellWidths);
      TEST_FLOATING_EQUALITY(bMatrix[row * shape.totalDofs() + col], expected, 1e-14);
    }
  }

  auto gDiagonal = TensorLocalSchurComplementBuilder::tensorProductDiagonal(shape,
                                                                            {componentMassDiagonal(cellWidths[0]),
                                                                             componentMassDiagonal(cellWidths[1])});
  TEST_EQUALITY(gDiagonal.size(), static_cast<size_t>(shape.totalDofs()));
  for (IndexType row=0; row<shape.totalDofs(); row++)
  {
    const double expected = explicitTensorGDiagonalEntry(shape, row, cellWidths);
    TEST_FLOATING_EQUALITY(gDiagonal[row], expected, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, LocalKMatchesExplicitDiagonalTestNormReference )
{
  auto shape = makeShape({2,2});
  const std::vector<double> cellWidths{2.0, 3.0};
  TensorProductLocalOperator bOperator(shape,
                                       shape,
                                       {TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[0])),
                                        TensorComponentLocalMatrix(2, 2, componentMassEntries(cellWidths[1]))});
  auto gDiagonal = TensorLocalSchurComplementBuilder::tensorProductDiagonal(shape,
                                                                            {componentMassDiagonal(cellWidths[0]),
                                                                             componentMassDiagonal(cellWidths[1])});

  auto kMatrix = TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, gDiagonal);
  TEST_EQUALITY(kMatrix.size(), static_cast<size_t>(shape.totalDofs() * shape.totalDofs()));

  for (IndexType row=0; row<shape.totalDofs(); row++)
  {
    for (IndexType col=0; col<shape.totalDofs(); col++)
    {
      double expected = 0.0;
      for (IndexType testOrdinal=0; testOrdinal<shape.totalDofs(); testOrdinal++)
      {
        const double bRow = explicitTensorBEntry(shape, shape, testOrdinal, row, cellWidths);
        const double bCol = explicitTensorBEntry(shape, shape, testOrdinal, col, cellWidths);
        const double gEntry = explicitTensorGDiagonalEntry(shape, testOrdinal, cellWidths);
        expected += bRow * bCol / gEntry;
      }
      TEST_FLOATING_EQUALITY(kMatrix[row * shape.totalDofs() + col], expected, 1e-14);
      TEST_FLOATING_EQUALITY(kMatrix[row * shape.totalDofs() + col], kMatrix[col * shape.totalDofs() + row], 1e-14);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, CachedAffineVolumeBAndGMatchExplicitReference )
{
  auto tensorMesh = makeSingleCellTensorMesh({{0.0, 2.0}, {-1.0, 2.0}});
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBilinearTerm term(makeFieldTensorVar("u", 2),
                          makeFieldTensorVar("v", 2),
                          TensorComponentOperatorDescriptor::valueOnly(2),
                          TensorComponentOperatorDescriptor::valueOnly(2),
                          TensorFunction::componentCoordinate(0) * TensorFunction::componentCoordinate(1));
  auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, cellTuple);
  TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("u"), false, cellTuple, quadratureDegrees);
  TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("v"),  true,  cellTuple, quadratureDegrees);
  auto localContext = dofLayout->volumeContext("u", "v", cellTuple);

  auto bOperator = TensorLocalSchurComplementBuilder::buildVolumeBOperator(term,
                                                                           localContext.trialShape(),
                                                                           localContext.testShape(),
                                                                           trialCache,
                                                                           testCache,
                                                                           {TensorFunction::componentCoordinate(0), TensorFunction::componentCoordinate(1)});
  auto bMatrix = TensorLocalSchurComplementBuilder::denseBMatrix(bOperator);
  auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
  const auto trialLayout = dofLayout->layout("u", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("v", TENSOR_TEST_SPACE);
  for (IndexType row=0; row<localContext.testShape().totalDofs(); row++)
  {
    for (IndexType col=0; col<localContext.trialShape().totalDofs(); col++)
    {
      const double expected = explicitCachedBEntry(term,
                                                   dofLayout,
                                                   trialLayout,
                                                   testLayout,
                                                   localContext.trialShape(),
                                                   localContext.testShape(),
                                                   trialCache,
                                                   testCache,
                                                   fullCoefficientValues,
                                                   row,
                                                   col);
      TEST_ASSERT(closeEnough(bMatrix[row * localContext.trialShape().totalDofs() + col], expected, 1e-12, 1e-14));
    }
  }

  auto gDiagonal = TensorLocalSchurComplementBuilder::l2TestMassDiagonal(localContext.testShape(), testCache);
  TEST_EQUALITY(gDiagonal.size(), static_cast<size_t>(localContext.testShape().totalDofs()));
  for (double entry : gDiagonal)
  {
    TEST_ASSERT(entry > 0.0);
  }

  auto kMatrix = TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, gDiagonal);
  for (IndexType row=0; row<localContext.trialShape().totalDofs(); row++)
  {
    for (IndexType col=0; col<localContext.trialShape().totalDofs(); col++)
    {
      TEST_FLOATING_EQUALITY(kMatrix[row * localContext.trialShape().totalDofs() + col],
                             kMatrix[col * localContext.trialShape().totalDofs() + row],
                             1e-12);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, CachedVlasovLikeVolumeBMatchesExplicitReference )
{
  auto tensorMesh = makeSingleCellTensorMesh({{0.0, 1.0}, {0.0, 1.0}, {-2.0, 2.0}});
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBilinearTerm term(makeFieldTensorVar("u", 3),
                          makeFieldTensorVar("v", 3),
                          TensorComponentOperatorDescriptor::valueOnly(3),
                          TensorComponentOperatorDescriptor::singleDerivative(3, 0),
                          TensorFunction::constant(-1.0) * TensorFunction::componentCoordinate(2));
  auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, cellTuple);
  TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("u"), false, cellTuple, quadratureDegrees);
  TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("v"),  true,  cellTuple, quadratureDegrees);
  auto localContext = dofLayout->volumeContext("u", "v", cellTuple);

  auto bOperator = TensorLocalSchurComplementBuilder::buildVolumeBOperator(term,
                                                                           localContext.trialShape(),
                                                                           localContext.testShape(),
                                                                           trialCache,
                                                                           testCache,
                                                                           {TensorFunction::constant(-1.0), Teuchos::null, TensorFunction::componentCoordinate(2)});
  auto bMatrix = TensorLocalSchurComplementBuilder::denseBMatrix(bOperator);
  auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
  const auto trialLayout = dofLayout->layout("u", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("v", TENSOR_TEST_SPACE);

  for (IndexType row=0; row<localContext.testShape().totalDofs(); row++)
  {
    for (IndexType col=0; col<localContext.trialShape().totalDofs(); col++)
    {
      const double expected = explicitCachedBEntry(term,
                                                   dofLayout,
                                                   trialLayout,
                                                   testLayout,
                                                   localContext.trialShape(),
                                                   localContext.testShape(),
                                                   trialCache,
                                                   testCache,
                                                   fullCoefficientValues,
                                                   row,
                                                   col);
      TEST_ASSERT(closeEnough(bMatrix[row * localContext.trialShape().totalDofs() + col], expected, 1e-11, 1e-13));
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineL2VolumeCellSchurComplementsSupportRelativisticVelocityFallback )
{
  auto tensorMesh = makeVlasovLikeVolumeAffineTensorMesh1D1T2V();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("local-assembled-relativistic-velocity"));
  const double c = 5.0;
  auto ux = TensorFunction::componentCoordinate(2);
  auto uy = TensorFunction::componentCoordinate(3);
  auto gamma = sqrt(TensorFunction::constant(1.0) + TensorFunction::constant(1.0 / (c * c)) * (ux * ux + uy * uy));
  auto vx = TensorFunction::constant(-1.0) * ux * reciprocal(gamma);
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4),
                                   makeFieldTensorVar("w", 4),
                                   TensorComponentOperatorDescriptor::valueOnly(4),
                                   TensorComponentOperatorDescriptor::singleDerivative(4, 0),
                                   vx));

  auto blocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                      dofLayout,
                                                                                      form,
                                                                                      "f",
                                                                                      "w");
  TEST_EQUALITY(blocks.size(), static_cast<size_t>(tensorMesh->topology()->activeCellCount()));

  const auto &term = form->term(0);
  const auto trialLayout = dofLayout->layout("f", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("w", TENSOR_TEST_SPACE);
  for (const auto &block : blocks)
  {
    auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, block.cellTuple);
    TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, block.cellTuple, quadratureDegrees);
    TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("w"), true,  block.cellTuple, quadratureDegrees);
    auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
    for (IndexType row=0; row<block.testShape.totalDofs(); row++)
    {
      for (IndexType col=0; col<block.trialShape.totalDofs(); col++)
      {
        const double expected = explicitCachedBEntry(term,
                                                     dofLayout,
                                                     trialLayout,
                                                     testLayout,
                                                     block.trialShape,
                                                     block.testShape,
                                                     trialCache,
                                                     testCache,
                                                     fullCoefficientValues,
                                                     row,
                                                     col);
        TEST_ASSERT(closeEnough(block.bMatrix[row * block.trialShape.totalDofs() + col], expected, 1e-11, 1e-13));
      }
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineL2VolumeCellSchurComplementsSupportSeparableCoefficientSums )
{
  auto tensorMesh = makeSingleCellTensorMesh({{0.0, 1.0}, {0.0, 1.0}, {-2.0, 2.0}});
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("local-assembled-coefficient-sum"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 3),
                                   makeFieldTensorVar("v", 3),
                                   TensorComponentOperatorDescriptor::valueOnly(3),
                                   TensorComponentOperatorDescriptor::singleDerivative(3, 2),
                                   TensorFunction::constant(-1.0) * TensorFunction::componentCoordinate(2)
                                     + TensorFunction::constant(0.5) * TensorFunction::componentCoordinate(0) * TensorFunction::componentCoordinate(1)));

  auto blocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                      dofLayout,
                                                                                      form,
                                                                                      "u",
                                                                                      "v");
  TEST_EQUALITY(blocks.size(), static_cast<size_t>(1));

  const auto &term = form->term(0);
  const auto &block = blocks[0];
  auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, block.cellTuple);
  TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("u"), false, block.cellTuple, quadratureDegrees);
  TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("v"), true,  block.cellTuple, quadratureDegrees);
  auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
  const auto trialLayout = dofLayout->layout("u", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("v", TENSOR_TEST_SPACE);

  for (IndexType row=0; row<block.testShape.totalDofs(); row++)
  {
    for (IndexType col=0; col<block.trialShape.totalDofs(); col++)
    {
      const double expected = explicitCachedBEntry(term,
                                                   dofLayout,
                                                   trialLayout,
                                                   testLayout,
                                                   block.trialShape,
                                                   block.testShape,
                                                   trialCache,
                                                   testCache,
                                                   fullCoefficientValues,
                                                   row,
                                                   col);
      TEST_ASSERT(closeEnough(block.bMatrix[row * block.trialShape.totalDofs() + col], expected, 1e-11, 1e-13));
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, SeparableCoefficientSumMatchesEquivalentSeparateTerms )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();

  auto sumForm = Teuchos::rcp(new TensorBilinearForm("coefficient-sum-single-term"));
  sumForm->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                      makeFieldTensorVar("v", 2),
                                      TensorComponentOperatorDescriptor::valueOnly(2),
                                      TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                      TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1)));

  auto splitForm = Teuchos::rcp(new TensorBilinearForm("coefficient-sum-split-terms"));
  splitForm->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                        makeFieldTensorVar("v", 2),
                                        TensorComponentOperatorDescriptor::valueOnly(2),
                                        TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                        TensorFunction::componentCoordinate(0)));
  splitForm->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                        makeFieldTensorVar("v", 2),
                                        TensorComponentOperatorDescriptor::valueOnly(2),
                                        TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                        TensorFunction::componentCoordinate(1)));

  auto sumBlocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                         dofLayout,
                                                                                         sumForm,
                                                                                         "u",
                                                                                         "v");
  auto splitBlocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                           dofLayout,
                                                                                           splitForm,
                                                                                           "u",
                                                                                           "v");
  TEST_EQUALITY(sumBlocks.size(), splitBlocks.size());
  for (size_t blockOrdinal=0; blockOrdinal<sumBlocks.size(); blockOrdinal++)
  {
    TEST_EQUALITY(sumBlocks[blockOrdinal].bMatrix.size(), splitBlocks[blockOrdinal].bMatrix.size());
    TEST_EQUALITY(sumBlocks[blockOrdinal].gDiagonal.size(), splitBlocks[blockOrdinal].gDiagonal.size());
    TEST_EQUALITY(sumBlocks[blockOrdinal].kMatrix.size(), splitBlocks[blockOrdinal].kMatrix.size());
    for (size_t entryOrdinal=0; entryOrdinal<sumBlocks[blockOrdinal].bMatrix.size(); entryOrdinal++)
    {
      TEST_ASSERT(closeEnough(sumBlocks[blockOrdinal].bMatrix[entryOrdinal], splitBlocks[blockOrdinal].bMatrix[entryOrdinal], 1e-12, 1e-14));
    }
    for (size_t entryOrdinal=0; entryOrdinal<sumBlocks[blockOrdinal].gDiagonal.size(); entryOrdinal++)
    {
      TEST_FLOATING_EQUALITY(sumBlocks[blockOrdinal].gDiagonal[entryOrdinal], splitBlocks[blockOrdinal].gDiagonal[entryOrdinal], 1e-14);
    }
    for (size_t entryOrdinal=0; entryOrdinal<sumBlocks[blockOrdinal].kMatrix.size(); entryOrdinal++)
    {
      TEST_ASSERT(closeEnough(sumBlocks[blockOrdinal].kMatrix[entryOrdinal], splitBlocks[blockOrdinal].kMatrix[entryOrdinal], 1e-11, 1e-13));
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineL2VolumeCellSchurComplementsMatchDenseReference )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("local-assembled-affine-l2"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(2.0)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  auto blocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                      dofLayout,
                                                                                      form,
                                                                                      "u",
                                                                                      "v");
  TEST_EQUALITY(blocks.size(), static_cast<size_t>(tensorMesh->topology()->activeCellCount()));

  const auto trialLayout = dofLayout->layout("u", TENSOR_TRIAL_SPACE);
  const auto testLayout  = dofLayout->layout("v", TENSOR_TEST_SPACE);
  for (const auto &block : blocks)
  {
    auto localContext = dofLayout->volumeContext("u", "v", block.cellTuple);
    TEST_ASSERT(block.trialShape == localContext.trialShape());
    TEST_ASSERT(block.testShape  == localContext.testShape());
    TEST_EQUALITY(block.bMatrix.size(), static_cast<size_t>(block.testShape.totalDofs() * block.trialShape.totalDofs()));
    TEST_EQUALITY(block.gDiagonal.size(), static_cast<size_t>(block.testShape.totalDofs()));
    TEST_EQUALITY(block.kMatrix.size(), static_cast<size_t>(block.trialShape.totalDofs() * block.trialShape.totalDofs()));

    std::vector<double> referenceB(block.bMatrix.size(), 0.0);
    for (const auto &term : form->terms())
    {
      auto quadratureDegrees = quadratureDegreesForTerm(tensorMesh, term, block.cellTuple);
      TensorBasisCacheND trialCache(tensorMesh, tensorMesh->trialDiscretization("u"), false, block.cellTuple, quadratureDegrees);
      TensorBasisCacheND testCache (tensorMesh, tensorMesh->testDiscretization("v"), true,  block.cellTuple, quadratureDegrees);
      auto fullCoefficientValues = coefficientValues(term.coefficient(), trialCache);
      for (IndexType row=0; row<block.testShape.totalDofs(); row++)
      {
        for (IndexType col=0; col<block.trialShape.totalDofs(); col++)
        {
          referenceB[row * block.trialShape.totalDofs() + col] += explicitCachedBEntry(term,
                                                                                       dofLayout,
                                                                                       trialLayout,
                                                                                       testLayout,
                                                                                       block.trialShape,
                                                                                       block.testShape,
                                                                                       trialCache,
                                                                                       testCache,
                                                                                       fullCoefficientValues,
                                                                                       row,
                                                                                       col);
        }
      }
    }

    for (size_t entryOrdinal=0; entryOrdinal<referenceB.size(); entryOrdinal++)
    {
      TEST_ASSERT(closeEnough(block.bMatrix[entryOrdinal], referenceB[entryOrdinal], 1e-12, 1e-14));
    }

    std::vector<int> gReferenceQuadratureDegrees(block.testShape.componentCount(), 1);
    for (int d=0; d<block.testShape.componentCount(); d++)
    {
      const int testPolyOrder = tensorMesh->effectivePolyOrder(tensorMesh->testDiscretization("v"), block.cellTuple[d], d, true);
      gReferenceQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
    }
    TensorBasisCacheND gTestCache(tensorMesh, tensorMesh->testDiscretization("v"), true, block.cellTuple, gReferenceQuadratureDegrees);
    auto referenceG = TensorLocalSchurComplementBuilder::l2TestMassDiagonal(block.testShape, gTestCache);
    auto referenceK = TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(referenceB,
                                                                                            block.testShape.totalDofs(),
                                                                                            block.trialShape.totalDofs(),
                                                                                            referenceG);
    for (IndexType testOrdinal=0; testOrdinal<block.testShape.totalDofs(); testOrdinal++)
    {
      TEST_ASSERT(block.gDiagonal[testOrdinal] > 0.0);
      TEST_FLOATING_EQUALITY(block.gDiagonal[testOrdinal], referenceG[testOrdinal], 1e-14);
    }
    for (size_t entryOrdinal=0; entryOrdinal<referenceK.size(); entryOrdinal++)
    {
      TEST_ASSERT(closeEnough(block.kMatrix[entryOrdinal], referenceK[entryOrdinal], 1e-12, 1e-14));
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineGraphNormLocalSchurComplementsMatchExplicitReference )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("assembled-affine-graph-local"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.5)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::componentCoordinate(0)));

  const double l2Weight = 3.0;
  auto blocks = TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "u",
                                                                                         "v",
                                                                                         l2Weight);
  TEST_EQUALITY(blocks.size(), static_cast<size_t>(tensorMesh->topology()->activeCellCount()));
  const auto testLayout = dofLayout->layout("v", TENSOR_TEST_SPACE);
  for (const auto &block : blocks)
  {
    std::vector<int> gQuadratureDegrees(block.testShape.componentCount(), 1);
    for (int d=0; d<block.testShape.componentCount(); d++)
    {
      const int testPolyOrder = tensorMesh->effectivePolyOrder(tensorMesh->testDiscretization("v"), block.cellTuple[d], d, true);
      gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
    }
    TensorBasisCacheND gTestCache(tensorMesh, tensorMesh->testDiscretization("v"), true, block.cellTuple, gQuadratureDegrees);

    TEST_EQUALITY(block.gMatrix.size(), static_cast<size_t>(block.testShape.totalDofs() * block.testShape.totalDofs()));
    TEST_EQUALITY(block.kMatrix.size(), static_cast<size_t>(block.trialShape.totalDofs() * block.trialShape.totalDofs()));
    auto referenceK = TensorLocalSchurComplementBuilder::formDenseTestNormSchurComplement(block.bMatrix,
                                                                                         block.testShape.totalDofs(),
                                                                                         block.trialShape.totalDofs(),
                                                                                         block.gMatrix);
    for (IndexType row=0; row<block.testShape.totalDofs(); row++)
    {
      TEST_FLOATING_EQUALITY(block.gDiagonal[row], block.gMatrix[row * block.testShape.totalDofs() + row], 1e-14);
      for (IndexType col=0; col<block.testShape.totalDofs(); col++)
      {
        const double expected = explicitCachedGraphGEntry(dofLayout, testLayout, block.testShape, gTestCache, l2Weight, row, col);
        TEST_ASSERT(closeEnough(block.gMatrix[row * block.testShape.totalDofs() + col], expected, 1e-12, 1e-14));
        TEST_ASSERT(closeEnough(block.gMatrix[row * block.testShape.totalDofs() + col],
                                block.gMatrix[col * block.testShape.totalDofs() + row],
                                1e-12,
                                1e-14));
      }
    }
    for (IndexType row=0; row<block.trialShape.totalDofs(); row++)
    {
      for (IndexType col=0; col<block.trialShape.totalDofs(); col++)
      {
        TEST_ASSERT(closeEnough(block.kMatrix[row * block.trialShape.totalDofs() + col], referenceK[row * block.trialShape.totalDofs() + col], 1e-12, 1e-14));
        TEST_ASSERT(closeEnough(block.kMatrix[row * block.trialShape.totalDofs() + col],
                                block.kMatrix[col * block.trialShape.totalDofs() + row],
                                1e-12,
                                1e-14));
      }
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, LocalGraphNormActionMatchesDenseSchurMatrixAction )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("local-graph-action"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.5)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::componentCoordinate(0)));

  const double l2Weight = 2.25;
  auto blocks = TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "u",
                                                                                         "v",
                                                                                         l2Weight);
  TEST_ASSERT(!blocks.empty());
  for (const auto &block : blocks)
  {
    auto action = TensorLocalSchurComplementBuilder::localGraphNormAction(block.testShape.totalDofs(), block.gMatrix);
    TEST_EQUALITY(action.testDofCount(), block.testShape.totalDofs());
    TEST_EQUALITY(action.gMatrix().size(), block.gMatrix.size());

    std::vector<double> localTrialVector(block.trialShape.totalDofs(), 0.0);
    for (IndexType trialOrdinal=0; trialOrdinal<block.trialShape.totalDofs(); trialOrdinal++)
    {
      localTrialVector[trialOrdinal] = 0.2 + 0.05 * static_cast<double>(trialOrdinal + 1);
    }

    std::vector<double> expected(block.trialShape.totalDofs(), 0.0);
    for (IndexType row=0; row<block.trialShape.totalDofs(); row++)
    {
      for (IndexType col=0; col<block.trialShape.totalDofs(); col++)
      {
        expected[row] += block.kMatrix[static_cast<size_t>(row * block.trialShape.totalDofs() + col)] * localTrialVector[col];
      }
    }

    auto actual = action.applySchurComplement(block.bMatrix, block.trialShape.totalDofs(), localTrialVector);
    TEST_EQUALITY(actual.size(), expected.size());
    for (size_t i=0; i<actual.size(); i++)
    {
      TEST_ASSERT(closeEnough(actual[i], expected[i], 1e-11, 1e-13));
    }

    std::vector<double> bTimesX(block.testShape.totalDofs(), 0.0);
    for (IndexType testOrdinal=0; testOrdinal<block.testShape.totalDofs(); testOrdinal++)
    {
      for (IndexType trialOrdinal=0; trialOrdinal<block.trialShape.totalDofs(); trialOrdinal++)
      {
        bTimesX[testOrdinal] += block.bMatrix[static_cast<size_t>(testOrdinal * block.trialShape.totalDofs() + trialOrdinal)] * localTrialVector[trialOrdinal];
      }
    }

    auto gInverseBTimesX = action.applyInverse(bTimesX);
    for (IndexType row=0; row<block.testShape.totalDofs(); row++)
    {
      double reconstructed = 0.0;
      for (IndexType col=0; col<block.testShape.totalDofs(); col++)
      {
        reconstructed += block.gMatrix[static_cast<size_t>(row * block.testShape.totalDofs() + col)] * gInverseBTimesX[col];
      }
      TEST_ASSERT(closeEnough(reconstructed, bTimesX[row], 1e-11, 1e-13));
    }
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeAffineGraphVolumeSchurActionMatchesAssembledCrs )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("matrix-free-graph-action"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.5)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::componentCoordinate(0)));

  const double l2Weight = 2.5;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "u",
                                                                                         "v",
                                                                                         l2Weight);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "u");
  std::vector<double> input(trialDofCount, 0.0);
  for (IndexType gid=0; gid<trialDofCount; gid++) input[gid] = -0.3 + 0.04 * static_cast<double>((gid % 11) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  auto actual = TensorLocalSchurComplementBuilder::applyAffineGraphVolumeSchurComplement(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "u",
                                                                                         "v",
                                                                                         input,
                                                                                         l2Weight);
  TEST_EQUALITY(actual.size(), static_cast<size_t>(trialDofCount));
  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(actual[gid], yHost(localOrdinal,0), 1e-11, 1e-13));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeAffineGraphVolumeSchurActionMatchesAssembledCrsRelativistic1D1T2V )
{
  auto tensorMesh = makeVlasovLikeVolumeAffineTensorMesh1D1T2V();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("matrix-free-relativistic-graph-action"));
  const double c = 5.0;
  auto ux = TensorFunction::componentCoordinate(2);
  auto uy = TensorFunction::componentCoordinate(3);
  auto gamma = sqrt(TensorFunction::constant(1.0) + TensorFunction::constant(1.0 / (c * c)) * (ux * ux + uy * uy));
  auto vx = TensorFunction::constant(-1.0) * ux * reciprocal(gamma);
  auto vy = TensorFunction::constant(0.5) * uy * reciprocal(gamma);
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 0), vx));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 1), TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 2), TensorFunction::constant(-0.5) * (TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1))));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 3), vy + TensorFunction::constant(-0.75) * TensorFunction::componentCoordinate(1)));

  const double l2Weight = 1.25;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "f",
                                                                                         "w",
                                                                                         l2Weight);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "f");
  std::vector<double> input(trialDofCount, 0.0);
  for (IndexType gid=0; gid<trialDofCount; gid++) input[gid] = 0.1 + 0.025 * static_cast<double>((gid % 17) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  auto actual = TensorLocalSchurComplementBuilder::applyAffineGraphVolumeSchurComplement(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "f",
                                                                                         "w",
                                                                                         input,
                                                                                         l2Weight);
  TEST_EQUALITY(actual.size(), static_cast<size_t>(trialDofCount));
  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(actual[gid], yHost(localOrdinal,0), 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeGraphNormSchurFactoryBuildsTrialToTrialOperator )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("graph-schur-facade"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::constant(1.0)));

  MatrixFreeGraphNormSchurTensorOperatorFactory factory(2.5);
  TEST_ASSERT(factory.canBuild(tensorMesh, dofLayout, form,
                               TENSOR_TRIAL_SPACE, "u",
                               TENSOR_TRIAL_SPACE, "u"));
  TEST_ASSERT(!factory.canBuild(tensorMesh, dofLayout, form,
                                TENSOR_TRIAL_SPACE, "u",
                                TENSOR_TEST_SPACE, "v"));
  const auto diagnostic = factory.validationMessage(tensorMesh, dofLayout, form,
                                                    TENSOR_TRIAL_SPACE, "u",
                                                    TENSOR_TEST_SPACE, "v");
  TEST_ASSERT(diagnostic.find("trial-to-trial") != std::string::npos);

  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TRIAL_SPACE, "u");

  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "u");
  TEST_ASSERT(op != Teuchos::null);
  TEST_EQUALITY(op->backend(), TENSOR_OPERATOR_BACKEND_MATRIX_FREE);
  TEST_EQUALITY(op->domainRole(), TENSOR_TRIAL_SPACE);
  TEST_EQUALITY(op->rangeRole(), TENSOR_TRIAL_SPACE);
  TEST_EQUALITY(op->domainVarName(), "u");
  TEST_EQUALITY(op->rangeVarName(), "u");
  TEST_EQUALITY(op->domainSize(), trialDofCount);
  TEST_EQUALITY(op->rangeSize(), trialDofCount);
  TEST_EQUALITY(factory.l2Weight(), 2.5);
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeGraphNormSchurOperatorApplyMatchesAssembledCrs )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("graph-schur-facade-crs"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.5)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::componentCoordinate(0)));

  const double l2Weight = 2.5;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "u",
                                                                                         "v",
                                                                                         l2Weight);
  MatrixFreeGraphNormSchurTensorOperatorFactory factory(l2Weight);
  auto op = factory.build(tensorMesh, dofLayout, form,
                          TENSOR_TRIAL_SPACE, "u",
                          TENSOR_TRIAL_SPACE, "u");

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType gid=0; gid<op->domainSize(); gid++) input[gid] = -0.3 + 0.04 * static_cast<double>((gid % 11) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  auto initialStats = op->instrumentation();
  TEST_EQUALITY(initialStats.graphNormSchurApplyCount(), 0);

  std::vector<double> actual;
  op->apply(input, actual);
  TEST_EQUALITY(actual.size(), static_cast<size_t>(op->rangeSize()));
  auto stats = op->instrumentation();
  TEST_EQUALITY(stats.graphNormSchurApplyCount(), 1);
  TEST_ASSERT(stats.graphNormSchurActiveTrialDofCount() > 0);
  TEST_ASSERT(stats.graphNormSchurActiveTrialDofCount() < stats.graphNormSchurApplyCount() * op->domainSize() * tensorMesh->topology()->activeCellCount());
  TEST_ASSERT(stats.count(TensorOperatorInstrumentation::LOCAL_ACTION) > 0);
  op->resetInstrumentation();
  TEST_EQUALITY(op->instrumentation().graphNormSchurApplyCount(), 0);
  TEST_EQUALITY(op->instrumentation().graphNormSchurActiveTrialDofCount(), 0);

  std::vector<double> secondActual;
  op->apply(input, secondActual);
  auto secondStats = op->instrumentation();
  TEST_EQUALITY(secondStats.graphNormSchurApplyCount(), 1);
  TEST_ASSERT(secondStats.count(TensorOperatorInstrumentation::LOCAL_ACTION) > 0);
  TEST_EQUALITY(secondStats.count(TensorOperatorInstrumentation::BASIS_CACHE_CONSTRUCTION), 0);
  TEST_EQUALITY(secondStats.count(TensorOperatorInstrumentation::CONTRACTION_1D_CONSTRUCTION), 0);
  TEST_EQUALITY(secondActual.size(), actual.size());
  for (size_t i=0; i<actual.size(); i++)
  {
    TEST_ASSERT(closeEnough(secondActual[i], actual[i], 1e-14, 1e-14));
  }
  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(actual[gid], yHost(localOrdinal,0), 1e-11, 1e-13));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineGraphNormVolumeSchurComplementMatrixActionMatchesLocalBlockScatter )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("assembled-affine-graph-crs"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.5)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::componentCoordinate(0)));

  const double l2Weight = 2.5;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "u",
                                                                                         "v",
                                                                                         l2Weight);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType globalDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "u");
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(globalDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(globalDofCount));

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = 0.4 + 0.075 * static_cast<double>(gid + 1);
  }
  matrix->apply(x, y);

  std::vector<double> input(globalDofCount, 0.0);
  for (IndexType gid=0; gid<globalDofCount; gid++) input[gid] = 0.4 + 0.075 * static_cast<double>(gid + 1);

  std::vector<double> expected(globalDofCount, 0.0);
  auto blocks = TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         "u",
                                                                                         "v",
                                                                                         l2Weight);
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, "u", tensorCellOrdinal);
    for (IndexType localRow=0; localRow<localDofCount; localRow++)
    {
      const IndexType rowGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "u", tensorCellOrdinal, localRow);
      for (IndexType localCol=0; localCol<localDofCount; localCol++)
      {
        const IndexType colGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "u", tensorCellOrdinal, localCol);
        expected[rowGid] += block.kMatrix[static_cast<size_t>(localRow * localDofCount + localCol)] * input[colGid];
      }
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-11, 1e-13));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineL2VolumeSchurComplementMatrixActionMatchesLocalBlockScatter )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("assembled-affine-l2-crs"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.5)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::componentCoordinate(0)));

  auto matrix = TensorLocalSchurComplementBuilder::affineL2VolumeSchurComplementMatrix(tensorMesh,
                                                                                       dofLayout,
                                                                                       form,
                                                                                       "u",
                                                                                       "v");
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType globalDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "u");
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(globalDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(globalDofCount));

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = 0.25 + 0.125 * static_cast<double>(gid + 1);
  }
  matrix->apply(x, y);

  std::vector<double> input(globalDofCount, 0.0);
  for (IndexType gid=0; gid<globalDofCount; gid++)
  {
    input[gid] = 0.25 + 0.125 * static_cast<double>(gid + 1);
  }

  std::vector<double> expected(globalDofCount, 0.0);
  auto blocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                      dofLayout,
                                                                                      form,
                                                                                      "u",
                                                                                      "v");
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localDofCount = operatorData.volumeLocalDofCount(TENSOR_TRIAL_SPACE, "u", tensorCellOrdinal);
    TEST_EQUALITY(localDofCount, block.trialShape.totalDofs());
    for (IndexType localRow=0; localRow<localDofCount; localRow++)
    {
      const IndexType rowGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "u", tensorCellOrdinal, localRow);
      for (IndexType localCol=0; localCol<localDofCount; localCol++)
      {
        const IndexType colGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "u", tensorCellOrdinal, localCol);
        expected[rowGid] += block.kMatrix[static_cast<size_t>(localRow * localDofCount + localCol)] * input[colGid];
      }
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-11, 1e-13));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineL2VolumeSchurComplementMatrixActionMatchesMatrixFreeVlasovLike1X1T1V )
{
  auto tensorMesh = makeVlasovLikeVolumeAffineTensorMesh1X1T1V();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("vlasov-like-1x1t1v-volume-crs"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 3),
                                   makeFieldTensorVar("w", 3),
                                   TensorComponentOperatorDescriptor::valueOnly(3),
                                   TensorComponentOperatorDescriptor::singleDerivative(3, 0),
                                   TensorFunction::constant(-1.0) * TensorFunction::componentCoordinate(2)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 3),
                                   makeFieldTensorVar("w", 3),
                                   TensorComponentOperatorDescriptor::valueOnly(3),
                                   TensorComponentOperatorDescriptor::singleDerivative(3, 1),
                                   TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 3),
                                   makeFieldTensorVar("w", 3),
                                   TensorComponentOperatorDescriptor::valueOnly(3),
                                   TensorComponentOperatorDescriptor::singleDerivative(3, 2),
                                   TensorFunction::constant(-1.0) * (TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1))));

  auto matrix = TensorLocalSchurComplementBuilder::affineL2VolumeSchurComplementMatrix(tensorMesh,
                                                                                       dofLayout,
                                                                                       form,
                                                                                       "f",
                                                                                       "w");
  MatrixFreeTensorOperatorFactory factory;
  auto bOperator = factory.build(tensorMesh,
                                 dofLayout,
                                 form,
                                 TENSOR_TRIAL_SPACE,
                                 "f",
                                 TENSOR_TEST_SPACE,
                                 "w");
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "f");
  const IndexType testDofCount  = operatorData.coefficientCount(TENSOR_TEST_SPACE,  "w");
  TEST_EQUALITY(bOperator->domainSize(), trialDofCount);
  TEST_EQUALITY(bOperator->rangeSize(),  testDofCount);
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(trialDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(trialDofCount));

  std::vector<double> input(trialDofCount, 0.0);
  for (IndexType gid=0; gid<trialDofCount; gid++) input[gid] = 0.3 + 0.05 * static_cast<double>((gid % 17) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = input[gid];
  }
  matrix->apply(x, y);

  std::vector<double> bInputImage;
  bOperator->apply(input, bInputImage);

  std::vector<std::vector<double>> bColumns(trialDofCount);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    std::vector<double> basisVector(trialDofCount, 0.0);
    basisVector[trialOrdinal] = 1.0;
    bOperator->apply(basisVector, bColumns[trialOrdinal]);
    TEST_EQUALITY(bColumns[trialOrdinal].size(), static_cast<size_t>(testDofCount));
  }

  std::vector<double> globalGDiagonal(testDofCount, 0.0);
  auto blocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                      dofLayout,
                                                                                      form,
                                                                                      "f",
                                                                                      "w");
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, "w", tensorCellOrdinal);
    TEST_EQUALITY(localTestDofCount, block.testShape.totalDofs());
    for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
    {
      const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, "w", tensorCellOrdinal, localTestOrdinal);
      TEST_ASSERT(globalGDiagonal[testGid] == 0.0);
      globalGDiagonal[testGid] = block.gDiagonal[localTestOrdinal];
    }
  }

  std::vector<double> expected(trialDofCount, 0.0);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
    {
      TEST_ASSERT(globalGDiagonal[testOrdinal] > 0.0);
      expected[trialOrdinal] += bColumns[trialOrdinal][testOrdinal] * bInputImage[testOrdinal] / globalGDiagonal[testOrdinal];
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineGraphVolumeSchurComplementMatrixActionMatchesMatrixFreeVlasovLike1D1T2V )
{
  auto tensorMesh = makeVlasovLikeVolumeAffineTensorMesh1D1T2V();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("vlasov-like-1d1t2v-volume-graph-crs"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 0), TensorFunction::constant(-1.0) * TensorFunction::componentCoordinate(2)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 1), TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 2), TensorFunction::constant(-0.5) * (TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1))));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 3), TensorFunction::constant(0.25) * TensorFunction::componentCoordinate(0) * TensorFunction::componentCoordinate(3) + TensorFunction::constant(-0.75) * TensorFunction::componentCoordinate(1)));

  const double l2Weight = 1.75;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh, dofLayout, form, "f", "w", l2Weight);
  MatrixFreeTensorOperatorFactory factory;
  auto bOperator = factory.build(tensorMesh, dofLayout, form, TENSOR_TRIAL_SPACE, "f", TENSOR_TEST_SPACE, "w");
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "f");
  const IndexType testDofCount = operatorData.coefficientCount(TENSOR_TEST_SPACE, "w");
  TEST_EQUALITY(bOperator->domainSize(), trialDofCount);
  TEST_EQUALITY(bOperator->rangeSize(), testDofCount);

  std::vector<double> input(trialDofCount, 0.0);
  for (IndexType gid=0; gid<trialDofCount; gid++) input[gid] = -0.15 + 0.04 * static_cast<double>((gid % 19) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  std::vector<double> bInputImage;
  bOperator->apply(input, bInputImage);
  std::vector<std::vector<double>> bColumns(trialDofCount);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    std::vector<double> basisVector(trialDofCount, 0.0);
    basisVector[trialOrdinal] = 1.0;
    bOperator->apply(basisVector, bColumns[trialOrdinal]);
  }

  std::vector<double> expected(trialDofCount, 0.0);
  auto blocks = TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(tensorMesh, dofLayout, form, "f", "w", l2Weight);
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, "w", tensorCellOrdinal);
    std::vector<double> localBInputImage(localTestDofCount, 0.0);
    std::vector<double> localBColumns(static_cast<size_t>(localTestDofCount * trialDofCount), 0.0);
    for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
    {
      const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, "w", tensorCellOrdinal, localTestOrdinal);
      localBInputImage[localTestOrdinal] = bInputImage[testGid];
      for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++) localBColumns[static_cast<size_t>(localTestOrdinal * trialDofCount + trialOrdinal)] = bColumns[trialOrdinal][testGid];
    }
    auto localAction = applyDenseSchurAction(localBColumns, localTestDofCount, trialDofCount, localBInputImage, block.gMatrix);
    for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++) expected[trialOrdinal] += localAction[trialOrdinal];
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineGraphVolumeSchurComplementMatrixActionMatchesMatrixFreeRelativistic1D1T2V )
{
  auto tensorMesh = makeVlasovLikeVolumeAffineTensorMesh1D1T2V();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("relativistic-1d1t2v-volume-graph-crs"));
  const double c = 5.0;
  auto ux = TensorFunction::componentCoordinate(2);
  auto uy = TensorFunction::componentCoordinate(3);
  auto gamma = sqrt(TensorFunction::constant(1.0) + TensorFunction::constant(1.0 / (c * c)) * (ux * ux + uy * uy));
  auto vx = TensorFunction::constant(-1.0) * ux * reciprocal(gamma);
  auto vy = TensorFunction::constant(0.5) * uy * reciprocal(gamma);
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 0), vx));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 1), TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 2), TensorFunction::constant(-0.5) * (TensorFunction::componentCoordinate(0) + TensorFunction::componentCoordinate(1))));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 4), makeFieldTensorVar("w", 4), TensorComponentOperatorDescriptor::valueOnly(4), TensorComponentOperatorDescriptor::singleDerivative(4, 3), vy + TensorFunction::constant(-0.75) * TensorFunction::componentCoordinate(1)));

  const double l2Weight = 1.25;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh, dofLayout, form, "f", "w", l2Weight);
  MatrixFreeTensorOperatorFactory factory;
  auto bOperator = factory.build(tensorMesh, dofLayout, form, TENSOR_TRIAL_SPACE, "f", TENSOR_TEST_SPACE, "w");
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "f");
  const IndexType testDofCount = operatorData.coefficientCount(TENSOR_TEST_SPACE, "w");
  TEST_EQUALITY(bOperator->domainSize(), trialDofCount);
  TEST_EQUALITY(bOperator->rangeSize(), testDofCount);
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(trialDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(trialDofCount));

  std::vector<double> input(trialDofCount, 0.0);
  for (IndexType gid=0; gid<trialDofCount; gid++) input[gid] = 0.2 - 0.035 * static_cast<double>((gid % 23) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  std::vector<double> bInputImage;
  bOperator->apply(input, bInputImage);
  std::vector<std::vector<double>> bColumns(trialDofCount);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    std::vector<double> basisVector(trialDofCount, 0.0);
    basisVector[trialOrdinal] = 1.0;
    bOperator->apply(basisVector, bColumns[trialOrdinal]);
    TEST_EQUALITY(bColumns[trialOrdinal].size(), static_cast<size_t>(testDofCount));
  }

  std::vector<double> expected(trialDofCount, 0.0);
  auto blocks = TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(tensorMesh, dofLayout, form, "f", "w", l2Weight);
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, "w", tensorCellOrdinal);
    std::vector<double> localBInputImage(localTestDofCount, 0.0);
    std::vector<double> localBColumns(static_cast<size_t>(localTestDofCount * trialDofCount), 0.0);
    for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
    {
      const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, "w", tensorCellOrdinal, localTestOrdinal);
      localBInputImage[localTestOrdinal] = bInputImage[testGid];
      for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++) localBColumns[static_cast<size_t>(localTestOrdinal * trialDofCount + trialOrdinal)] = bColumns[trialOrdinal][testGid];
    }
    auto localAction = applyDenseSchurAction(localBColumns, localTestDofCount, trialDofCount, localBInputImage, block.gMatrix);
    for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++) expected[trialOrdinal] += localAction[trialOrdinal];
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineL2VolumeSchurComplementMatrixActionMatchesMatrixFreeVolumePath )
{
  auto tensorMesh = makeTwoByTwoAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("assembled-affine-l2-matrix-free-comparison"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(2.0)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  auto matrix = TensorLocalSchurComplementBuilder::affineL2VolumeSchurComplementMatrix(tensorMesh,
                                                                                       dofLayout,
                                                                                       form,
                                                                                       "u",
                                                                                       "v");
  MatrixFreeTensorOperatorFactory factory;
  auto bOperator = factory.build(tensorMesh,
                                 dofLayout,
                                 form,
                                 TENSOR_TRIAL_SPACE,
                                 "u",
                                 TENSOR_TEST_SPACE,
                                 "v");
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "u");
  const IndexType testDofCount  = operatorData.coefficientCount(TENSOR_TEST_SPACE,  "v");
  TEST_EQUALITY(bOperator->domainSize(), trialDofCount);
  TEST_EQUALITY(bOperator->rangeSize(),  testDofCount);

  std::vector<double> input(trialDofCount, 0.0);
  for (IndexType gid=0; gid<trialDofCount; gid++)
  {
    input[gid] = 0.5 + 0.2 * static_cast<double>((gid % 7) + 1);
  }

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = input[gid];
  }
  matrix->apply(x, y);

  std::vector<double> bInputImage;
  bOperator->apply(input, bInputImage);

  std::vector<std::vector<double>> bColumns(trialDofCount);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    std::vector<double> basisVector(trialDofCount, 0.0);
    basisVector[trialOrdinal] = 1.0;
    bOperator->apply(basisVector, bColumns[trialOrdinal]);
    TEST_EQUALITY(bColumns[trialOrdinal].size(), static_cast<size_t>(testDofCount));
  }

  std::vector<double> globalGDiagonal(testDofCount, 0.0);
  auto blocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                      dofLayout,
                                                                                      form,
                                                                                      "u",
                                                                                      "v");
  for (const auto &block : blocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, "v", tensorCellOrdinal);
    TEST_EQUALITY(localTestDofCount, block.testShape.totalDofs());
    for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
    {
      const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, "v", tensorCellOrdinal, localTestOrdinal);
      TEST_ASSERT(globalGDiagonal[testGid] == 0.0);
      globalGDiagonal[testGid] = block.gDiagonal[localTestOrdinal];
    }
  }

  std::vector<double> expected(trialDofCount, 0.0);
  for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
  {
    for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
    {
      TEST_ASSERT(globalGDiagonal[testOrdinal] > 0.0);
      expected[trialOrdinal] += bColumns[trialOrdinal][testOrdinal] * bInputImage[testOrdinal] / globalGDiagonal[testOrdinal];
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MultiVariableAffineL2VolumeSchurComplementMatrixActionMatchesBlockReference )
{
  auto tensorMesh = makeTwoTrialOneTestAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("multi-variable-affine-l2-crs"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.25)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("q", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  const std::vector<std::string> trialVarNames{"u", "q"};
  const std::vector<std::string> testVarNames{"v"};
  auto matrix = TensorLocalSchurComplementBuilder::affineL2VolumeSchurComplementMatrix(tensorMesh,
                                                                                       dofLayout,
                                                                                       form,
                                                                                       trialVarNames,
                                                                                       testVarNames);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  std::vector<IndexType> trialOffsets(trialVarNames.size(), 0);
  IndexType totalTrialDofCount = 0;
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }
  const IndexType testDofCount = operatorData.coefficientCount(TENSOR_TEST_SPACE, "v");
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));

  std::vector<double> input(totalTrialDofCount, 0.0);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++)
  {
    input[gid] = 0.35 + 0.1 * static_cast<double>((gid % 11) + 1);
  }

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = input[gid];
  }
  matrix->apply(x, y);

  MatrixFreeTensorOperatorFactory factory;
  std::vector<std::vector<double>> bInputImages(trialVarNames.size());
  std::vector<std::vector<std::vector<double>>> bColumns(trialVarNames.size());
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    const std::string &trialVarName = trialVarNames[trialVarOrdinal];
    const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);
    std::vector<double> localInput(trialDofCount, 0.0);
    for (IndexType localOrdinal=0; localOrdinal<trialDofCount; localOrdinal++)
    {
      localInput[localOrdinal] = input[trialOffsets[trialVarOrdinal] + localOrdinal];
    }

    auto bOperator = factory.build(tensorMesh,
                                   dofLayout,
                                   form,
                                   TENSOR_TRIAL_SPACE,
                                   trialVarName,
                                   TENSOR_TEST_SPACE,
                                   "v");
    bOperator->apply(localInput, bInputImages[trialVarOrdinal]);
    TEST_EQUALITY(bInputImages[trialVarOrdinal].size(), static_cast<size_t>(testDofCount));

    bColumns[trialVarOrdinal].resize(trialDofCount);
    for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
    {
      std::vector<double> basisVector(trialDofCount, 0.0);
      basisVector[trialOrdinal] = 1.0;
      bOperator->apply(basisVector, bColumns[trialVarOrdinal][trialOrdinal]);
      TEST_EQUALITY(bColumns[trialVarOrdinal][trialOrdinal].size(), static_cast<size_t>(testDofCount));
    }
  }

  std::vector<double> combinedBInputImage(testDofCount, 0.0);
  for (const auto &image : bInputImages)
  {
    for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
    {
      combinedBInputImage[testOrdinal] += image[testOrdinal];
    }
  }

  std::vector<double> globalGDiagonal(testDofCount, 0.0);
  auto gBlocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                       dofLayout,
                                                                                       form,
                                                                                       "u",
                                                                                       "v");
  for (const auto &block : gBlocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    const IndexType localTestDofCount = operatorData.volumeLocalDofCount(TENSOR_TEST_SPACE, "v", tensorCellOrdinal);
    for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
    {
      const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, "v", tensorCellOrdinal, localTestOrdinal);
      globalGDiagonal[testGid] = block.gDiagonal[localTestOrdinal];
    }
  }

  std::vector<double> expected(totalTrialDofCount, 0.0);
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
    for (IndexType localTrialOrdinal=0; localTrialOrdinal<trialDofCount; localTrialOrdinal++)
    {
      double value = 0.0;
      for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
      {
        TEST_ASSERT(globalGDiagonal[testOrdinal] > 0.0);
        value += bColumns[trialVarOrdinal][localTrialOrdinal][testOrdinal] * combinedBInputImage[testOrdinal] / globalGDiagonal[testOrdinal];
      }
      expected[trialOffsets[trialVarOrdinal] + localTrialOrdinal] = value;
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, AffineBoundaryBBlocksMatchTraceAwareMatrixFreeApply )
{
  auto tensorMesh = makeTraceVolumeAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("trace-boundary-b-blocks"));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn", 2, 0),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  MatrixFreeTensorOperatorFactory factory;
  auto boundaryOperator = factory.build(tensorMesh,
                                        dofLayout,
                                        form,
                                        TENSOR_TRIAL_SPACE,
                                        "tn",
                                        TENSOR_TEST_SPACE,
                                        "w");
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  TensorCoefficientMap testMap(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "w");
  std::vector<double> input(operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "tn"), 0.0);
  for (IndexType gid=0; gid<static_cast<IndexType>(input.size()); gid++)
  {
    input[gid] = 0.75 + 0.125 * static_cast<double>(gid + 1);
  }

  std::vector<double> actual;
  boundaryOperator->apply(input, actual);
  std::vector<double> expected(operatorData.coefficientCount(TENSOR_TEST_SPACE, "w"), 0.0);

  auto blocks = TensorLocalSchurComplementBuilder::affineBoundaryBBlocks(tensorMesh,
                                                                         dofLayout,
                                                                         form,
                                                                         "tn",
                                                                         "w");
  auto skeleton = tensorMesh->topology()->tensorSkeleton(0);
  TEST_EQUALITY(blocks.size(), static_cast<size_t>(skeleton.entityCount()));
  const auto retainedComponents = tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(0);
  for (IndexType entityOrdinal=0; entityOrdinal<skeleton.entityCount(); entityOrdinal++)
  {
    const auto &entity = skeleton.entity(entityOrdinal);
    const auto &block = blocks[entityOrdinal];
    TEST_ASSERT(block.entity == entity);
    TEST_EQUALITY(block.bMatrix.size(), static_cast<size_t>(block.testShape.totalDofs() * block.trialShape.totalDofs()));

    auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(entity.orthogonalComponentOrdinal());
    const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
    const ComponentCellID1D incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
    std::vector<ComponentCellID1D> testCells(tensorMesh->topology()->componentCount(), incidentCell);
    for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
    {
      testCells[retainedComponents[retainedOrdinal]] = entity.retainedCellTuple()[retainedOrdinal];
    }
    auto testIndices = testMap.indicesForVolumeCell(testCells);

    for (IndexType localTestOrdinal=0; localTestOrdinal<block.testShape.totalDofs(); localTestOrdinal++)
    {
      const IndexType testGid = testIndices[localTestOrdinal];
      for (IndexType localTrialOrdinal=0; localTrialOrdinal<block.trialShape.totalDofs(); localTrialOrdinal++)
      {
        const IndexType trialGid = operatorData.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "tn", entityOrdinal, localTrialOrdinal);
        expected[testGid] += block.bMatrix[static_cast<size_t>(localTestOrdinal * block.trialShape.totalDofs() + localTrialOrdinal)] * input[trialGid];
      }
    }
  }

  TEST_EQUALITY(actual.size(), expected.size());
  for (size_t i=0; i<actual.size(); i++)
  {
    TEST_ASSERT(closeEnough(actual[i], expected[i], 1e-11, 1e-13));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, DirectionalTraceBoundaryBBlocksUseRequestedSkeletonFamilies )
{
  auto tensorMesh = makeDirectionalTraceVolumeAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("directional-trace-boundary-b-blocks"));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn_x", 2, 0),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn_t", 2, 1),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(2.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  auto xBlocks = TensorLocalSchurComplementBuilder::affineBoundaryBBlocks(tensorMesh,
                                                                          dofLayout,
                                                                          form,
                                                                          "tn_x",
                                                                          "w");
  auto tBlocks = TensorLocalSchurComplementBuilder::affineBoundaryBBlocks(tensorMesh,
                                                                          dofLayout,
                                                                          form,
                                                                          "tn_t",
                                                                          "w");
  auto xSkeleton = tensorMesh->topology()->tensorSkeleton(0);
  auto tSkeleton = tensorMesh->topology()->tensorSkeleton(1);
  TEST_EQUALITY(xBlocks.size(), static_cast<size_t>(xSkeleton.entityCount()));
  TEST_EQUALITY(tBlocks.size(), static_cast<size_t>(tSkeleton.entityCount()));

  TensorOperatorData operatorData(tensorMesh, dofLayout);
  TEST_EQUALITY(operatorData.variableData(TENSOR_TRIAL_SPACE, "tn_x").traceComponentOrdinal(), 0);
  TEST_EQUALITY(operatorData.variableData(TENSOR_TRIAL_SPACE, "tn_t").traceComponentOrdinal(), 1);
  TEST_EQUALITY(operatorData.skeletonEntityCount(TENSOR_TRIAL_SPACE, "tn_x"), xSkeleton.entityCount());
  TEST_EQUALITY(operatorData.skeletonEntityCount(TENSOR_TRIAL_SPACE, "tn_t"), tSkeleton.entityCount());

  for (IndexType entityOrdinal=0; entityOrdinal<xSkeleton.entityCount(); entityOrdinal++)
  {
    TEST_EQUALITY(xBlocks[entityOrdinal].entity, xSkeleton.entity(entityOrdinal));
    TEST_EQUALITY(xBlocks[entityOrdinal].entity.orthogonalComponentOrdinal(), 0);
    TEST_ASSERT(!xBlocks[entityOrdinal].bMatrix.empty());
  }
  for (IndexType entityOrdinal=0; entityOrdinal<tSkeleton.entityCount(); entityOrdinal++)
  {
    TEST_EQUALITY(tBlocks[entityOrdinal].entity, tSkeleton.entity(entityOrdinal));
    TEST_EQUALITY(tBlocks[entityOrdinal].entity.orthogonalComponentOrdinal(), 1);
    TEST_ASSERT(!tBlocks[entityOrdinal].bMatrix.empty());
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, VolumeBoundarySchurComplementMatrixMatchesMatrixFreeReference )
{
  auto tensorMesh = makeTraceVolumeAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("volume-boundary-schur"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 2),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn", 2, 0),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  const std::vector<std::string> trialVarNames{"f", "tn"};
  const std::vector<std::string> testVarNames{"w"};
  auto matrix = TensorLocalSchurComplementBuilder::affineL2VolumeBoundarySchurComplementMatrix(tensorMesh,
                                                                                                dofLayout,
                                                                                                form,
                                                                                                trialVarNames,
                                                                                                testVarNames);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  std::vector<IndexType> trialOffsets(trialVarNames.size(), 0);
  IndexType totalTrialDofCount = 0;
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }
  const IndexType testDofCount = operatorData.coefficientCount(TENSOR_TEST_SPACE, "w");
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));

  std::vector<double> input(totalTrialDofCount, 0.0);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++)
  {
    input[gid] = 0.2 + 0.05 * static_cast<double>((gid % 13) + 1);
  }

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = input[gid];
  }
  matrix->apply(x, y);

  MatrixFreeTensorOperatorFactory factory;
  std::vector<double> combinedBInputImage(testDofCount, 0.0);
  std::vector<std::vector<std::vector<double>>> bColumns(trialVarNames.size());
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    const std::string &trialVarName = trialVarNames[trialVarOrdinal];
    const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);
    std::vector<double> localInput(trialDofCount, 0.0);
    for (IndexType localOrdinal=0; localOrdinal<trialDofCount; localOrdinal++)
    {
      localInput[localOrdinal] = input[trialOffsets[trialVarOrdinal] + localOrdinal];
    }

    auto bOperator = factory.build(tensorMesh,
                                   dofLayout,
                                   form,
                                   TENSOR_TRIAL_SPACE,
                                   trialVarName,
                                   TENSOR_TEST_SPACE,
                                   "w");
    std::vector<double> image;
    bOperator->apply(localInput, image);
    TEST_EQUALITY(image.size(), static_cast<size_t>(testDofCount));
    for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
    {
      combinedBInputImage[testOrdinal] += image[testOrdinal];
    }

    bColumns[trialVarOrdinal].resize(trialDofCount);
    for (IndexType trialOrdinal=0; trialOrdinal<trialDofCount; trialOrdinal++)
    {
      std::vector<double> basisVector(trialDofCount, 0.0);
      basisVector[trialOrdinal] = 1.0;
      bOperator->apply(basisVector, bColumns[trialVarOrdinal][trialOrdinal]);
      TEST_EQUALITY(bColumns[trialVarOrdinal][trialOrdinal].size(), static_cast<size_t>(testDofCount));
    }
  }

  std::vector<double> globalGDiagonal(testDofCount, 0.0);
  auto gBlocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                       dofLayout,
                                                                                       form,
                                                                                       "f",
                                                                                       "w");
  for (const auto &block : gBlocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(block.cellTuple);
    for (IndexType localTestOrdinal=0; localTestOrdinal<block.testShape.totalDofs(); localTestOrdinal++)
    {
      const IndexType testGid = operatorData.volumeLocalToGlobalIndex(TENSOR_TEST_SPACE, "w", tensorCellOrdinal, localTestOrdinal);
      globalGDiagonal[testGid] = block.gDiagonal[localTestOrdinal];
    }
  }

  std::vector<double> expected(totalTrialDofCount, 0.0);
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    const IndexType trialDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
    for (IndexType localTrialOrdinal=0; localTrialOrdinal<trialDofCount; localTrialOrdinal++)
    {
      double value = 0.0;
      for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
      {
        TEST_ASSERT(globalGDiagonal[testOrdinal] > 0.0);
        value += bColumns[trialVarOrdinal][localTrialOrdinal][testOrdinal] * combinedBInputImage[testOrdinal] / globalGDiagonal[testOrdinal];
      }
      expected[trialOffsets[trialVarOrdinal] + localTrialOrdinal] = value;
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeMultiVariableAffineGraphVolumeSchurActionMatchesAssembledCrs )
{
  auto tensorMesh = makeTwoTrialOneTestAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("matrix-free-multi-variable-graph-action"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.25)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("q", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  const std::vector<std::string> trialVarNames{"u", "q"};
  const std::vector<std::string> testVarNames{"v"};
  const double l2Weight = 2.25;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                          dofLayout,
                                                                                          form,
                                                                                          trialVarNames,
                                                                                          testVarNames,
                                                                                          l2Weight);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  IndexType totalTrialDofCount = 0;
  for (const auto &trialVarName : trialVarNames)
  {
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);
  }

  std::vector<double> input(totalTrialDofCount, 0.0);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++) input[gid] = -0.2 + 0.055 * static_cast<double>((gid % 13) + 1);
  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  auto actual = TensorLocalSchurComplementBuilder::applyAffineGraphVolumeSchurComplement(tensorMesh,
                                                                                         dofLayout,
                                                                                         form,
                                                                                         trialVarNames,
                                                                                         testVarNames,
                                                                                         input,
                                                                                         l2Weight);
  TEST_EQUALITY(actual.size(), static_cast<size_t>(totalTrialDofCount));
  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(actual[gid], yHost(localOrdinal,0), 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeGraphNormSchurFactoryBuildsMultiVariableOperator )
{
  auto tensorMesh = makeTwoTrialOneTestAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("graph-schur-multivariable-facade"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.25)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("q", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  const std::vector<std::string> trialVarNames{"u", "q"};
  const std::vector<std::string> testVarNames{"v"};
  MatrixFreeGraphNormSchurTensorOperatorFactory factory(2.25);
  TEST_ASSERT(factory.canBuild(tensorMesh, dofLayout, form, trialVarNames, testVarNames));

  auto op = factory.build(tensorMesh, dofLayout, form, trialVarNames, testVarNames);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  const IndexType expectedDofCount = operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "u")
                                   + operatorData.coefficientCount(TENSOR_TRIAL_SPACE, "q");
  TEST_EQUALITY(op->domainRole(), TENSOR_TRIAL_SPACE);
  TEST_EQUALITY(op->rangeRole(), TENSOR_TRIAL_SPACE);
  TEST_EQUALITY(op->domainVarName(), "u,q");
  TEST_EQUALITY(op->rangeVarName(),  "u,q");
  TEST_EQUALITY(op->domainSize(), expectedDofCount);
  TEST_EQUALITY(op->rangeSize(),  expectedDofCount);
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeGraphNormSchurMultiVariableOperatorApplyMatchesAssembledCrs )
{
  auto tensorMesh = makeTwoTrialOneTestAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("graph-schur-multivariable-facade-action"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.25)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("q", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  const std::vector<std::string> trialVarNames{"u", "q"};
  const std::vector<std::string> testVarNames{"v"};
  const double l2Weight = 2.25;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                          dofLayout,
                                                                                          form,
                                                                                          trialVarNames,
                                                                                          testVarNames,
                                                                                          l2Weight);
  MatrixFreeGraphNormSchurTensorOperatorFactory factory(l2Weight);
  auto op = factory.build(tensorMesh, dofLayout, form, trialVarNames, testVarNames);

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType gid=0; gid<op->domainSize(); gid++) input[gid] = 0.18 + 0.035 * static_cast<double>((gid % 11) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  std::vector<double> actual;
  op->apply(input, actual);
  TEST_EQUALITY(actual.size(), static_cast<size_t>(op->rangeSize()));
  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(actual[gid], yHost(localOrdinal,0), 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MultiVariableAffineGraphVolumeSchurComplementMatrixActionMatchesDenseLocalReference )
{
  auto tensorMesh = makeTwoTrialOneTestAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("multi-variable-affine-graph-crs"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("u", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.25)));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("q", 2),
                                   makeFieldTensorVar("v", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 0),
                                   TensorFunction::componentCoordinate(1)));

  const std::vector<std::string> trialVarNames{"u", "q"};
  const std::vector<std::string> testVarNames{"v"};
  const double l2Weight = 2.25;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeSchurComplementMatrix(tensorMesh,
                                                                                          dofLayout,
                                                                                          form,
                                                                                          trialVarNames,
                                                                                          testVarNames,
                                                                                          l2Weight);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  std::vector<IndexType> trialOffsets(trialVarNames.size(), 0);
  IndexType totalTrialDofCount = 0;
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));

  std::vector<double> input(totalTrialDofCount, 0.0);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++) input[gid] = 0.45 + 0.065 * static_cast<double>((gid % 9) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = input[gid];
  }
  matrix->apply(x, y);

  auto uBlocks = TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(tensorMesh,
                                                                                          dofLayout,
                                                                                          form,
                                                                                          "u",
                                                                                          "v",
                                                                                          l2Weight);
  auto qBlocks = TensorLocalSchurComplementBuilder::affineGraphVolumeCellSchurComplements(tensorMesh,
                                                                                          dofLayout,
                                                                                          form,
                                                                                          "q",
                                                                                          "v",
                                                                                          l2Weight);
  TEST_EQUALITY(uBlocks.size(), qBlocks.size());

  std::vector<double> expected(totalTrialDofCount, 0.0);
  for (size_t blockOrdinal=0; blockOrdinal<uBlocks.size(); blockOrdinal++)
  {
    const auto &uBlock = uBlocks[blockOrdinal];
    const auto &qBlock = qBlocks[blockOrdinal];
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(uBlock.cellTuple);
    TEST_EQUALITY(tensorCellOrdinal, tensorMesh->topology()->tensorCellOrdinal(qBlock.cellTuple));
    const IndexType testDofCount = uBlock.testShape.totalDofs();
    const IndexType uLocalDofCount = uBlock.trialShape.totalDofs();
    const IndexType qLocalDofCount = qBlock.trialShape.totalDofs();
    const IndexType localTrialDofCount = uLocalDofCount + qLocalDofCount;
    std::vector<double> localB(static_cast<size_t>(testDofCount * localTrialDofCount), 0.0);
    std::vector<IndexType> localGids(localTrialDofCount, -1);
    for (IndexType testOrdinal=0; testOrdinal<testDofCount; testOrdinal++)
    {
      for (IndexType localTrialOrdinal=0; localTrialOrdinal<uLocalDofCount; localTrialOrdinal++)
      {
        localB[static_cast<size_t>(testOrdinal * localTrialDofCount + localTrialOrdinal)] = uBlock.bMatrix[static_cast<size_t>(testOrdinal * uLocalDofCount + localTrialOrdinal)];
      }
      for (IndexType localTrialOrdinal=0; localTrialOrdinal<qLocalDofCount; localTrialOrdinal++)
      {
        localB[static_cast<size_t>(testOrdinal * localTrialDofCount + uLocalDofCount + localTrialOrdinal)] = qBlock.bMatrix[static_cast<size_t>(testOrdinal * qLocalDofCount + localTrialOrdinal)];
      }
    }
    for (IndexType localTrialOrdinal=0; localTrialOrdinal<uLocalDofCount; localTrialOrdinal++)
    {
      localGids[localTrialOrdinal] = trialOffsets[0] + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "u", tensorCellOrdinal, localTrialOrdinal);
    }
    for (IndexType localTrialOrdinal=0; localTrialOrdinal<qLocalDofCount; localTrialOrdinal++)
    {
      localGids[uLocalDofCount + localTrialOrdinal] = trialOffsets[1] + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "q", tensorCellOrdinal, localTrialOrdinal);
    }

    auto localK = TensorLocalSchurComplementBuilder::formDenseTestNormSchurComplement(localB,
                                                                                      testDofCount,
                                                                                      localTrialDofCount,
                                                                                      uBlock.gMatrix);
    for (IndexType localRow=0; localRow<localTrialDofCount; localRow++)
    {
      for (IndexType localCol=0; localCol<localTrialDofCount; localCol++)
      {
        expected[localGids[localRow]] += localK[static_cast<size_t>(localRow * localTrialDofCount + localCol)] * input[localGids[localCol]];
      }
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, GraphVolumeBoundarySchurComplementMatrixMatchesDenseLocalReference )
{
  auto tensorMesh = makeTraceVolumeAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("graph-volume-boundary-schur"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 2),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn", 2, 0),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  const std::vector<std::string> trialVarNames{"f", "tn"};
  const std::vector<std::string> testVarNames{"w"};
  const double l2Weight = 1.75;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeBoundarySchurComplementMatrix(tensorMesh,
                                                                                                  dofLayout,
                                                                                                  form,
                                                                                                  trialVarNames,
                                                                                                  testVarNames,
                                                                                                  l2Weight);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  std::vector<IndexType> trialOffsets(trialVarNames.size(), 0);
  IndexType totalTrialDofCount = 0;
  for (size_t trialVarOrdinal=0; trialVarOrdinal<trialVarNames.size(); trialVarOrdinal++)
  {
    trialOffsets[trialVarOrdinal] = totalTrialDofCount;
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarNames[trialVarOrdinal]);
  }
  TEST_EQUALITY(matrix->getGlobalNumRows(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));
  TEST_EQUALITY(matrix->getGlobalNumCols(), static_cast<Tpetra::global_size_t>(totalTrialDofCount));

  std::vector<double> input(totalTrialDofCount, 0.0);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++) input[gid] = 0.3 + 0.04 * static_cast<double>((gid % 17) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getDomainMap()->getGlobalElement(localOrdinal);
    xHost(localOrdinal,0) = input[gid];
  }
  matrix->apply(x, y);

  TensorCoefficientMap testMap(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "w");
  auto volumeBlocks = TensorLocalSchurComplementBuilder::affineL2VolumeCellSchurComplements(tensorMesh,
                                                                                            dofLayout,
                                                                                            form,
                                                                                            "f",
                                                                                            "w");
  auto boundaryBlocks = TensorLocalSchurComplementBuilder::affineBoundaryBBlocks(tensorMesh,
                                                                                 dofLayout,
                                                                                 form,
                                                                                 "tn",
                                                                                 "w");
  std::vector<double> expected(totalTrialDofCount, 0.0);
  for (const auto &volumeBlock : volumeBlocks)
  {
    const IndexType tensorCellOrdinal = tensorMesh->topology()->tensorCellOrdinal(volumeBlock.cellTuple);
    const IndexType localTestDofCount = volumeBlock.testShape.totalDofs();
    std::vector<int> gQuadratureDegrees(volumeBlock.testShape.componentCount(), 1);
    for (int d=0; d<volumeBlock.testShape.componentCount(); d++)
    {
      const int testPolyOrder = tensorMesh->effectivePolyOrder(tensorMesh->testDiscretization("w"), volumeBlock.cellTuple[d], d, true);
      gQuadratureDegrees[d] = std::max(1, 2 * testPolyOrder + 1);
    }
    TensorBasisCacheND gTestCache(tensorMesh, tensorMesh->testDiscretization("w"), true, volumeBlock.cellTuple, gQuadratureDegrees);
    auto localG = TensorLocalSchurComplementBuilder::graphTestNormMatrix(volumeBlock.testShape, gTestCache, l2Weight);
    std::vector<double> localB(static_cast<size_t>(localTestDofCount * totalTrialDofCount), 0.0);

    for (IndexType localTestOrdinal=0; localTestOrdinal<localTestDofCount; localTestOrdinal++)
    {
      for (IndexType localTrialOrdinal=0; localTrialOrdinal<volumeBlock.trialShape.totalDofs(); localTrialOrdinal++)
      {
        const IndexType trialGid = trialOffsets[0] + operatorData.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "f", tensorCellOrdinal, localTrialOrdinal);
        localB[static_cast<size_t>(localTestOrdinal * totalTrialDofCount + trialGid)] += volumeBlock.bMatrix[static_cast<size_t>(localTestOrdinal * volumeBlock.trialShape.totalDofs() + localTrialOrdinal)];
      }
    }

    for (const auto &boundaryBlock : boundaryBlocks)
    {
      const int orthogonalComponentOrdinal = boundaryBlock.entity.orthogonalComponentOrdinal();
      const auto retainedComponents = tensorMesh->topology()->retainedComponentOrdinalsForSkeleton(orthogonalComponentOrdinal);
      auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(orthogonalComponentOrdinal);
      const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(boundaryBlock.entity.vertexOrdinal());
      const ComponentCellID1D incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
      std::vector<ComponentCellID1D> testCells(tensorMesh->topology()->componentCount(), incidentCell);
      for (size_t retainedOrdinal=0; retainedOrdinal<retainedComponents.size(); retainedOrdinal++)
      {
        testCells[retainedComponents[retainedOrdinal]] = boundaryBlock.entity.retainedCellTuple()[retainedOrdinal];
      }
      if (tensorMesh->topology()->tensorCellOrdinal(testCells) != tensorCellOrdinal) continue;

      auto testIndices = testMap.indicesForVolumeCell(testCells);
      for (IndexType boundaryTestOrdinal=0; boundaryTestOrdinal<boundaryBlock.testShape.totalDofs(); boundaryTestOrdinal++)
      {
        const IndexType testGid = testIndices[boundaryTestOrdinal];
        const IndexType localTestOrdinal = localVolumeOrdinalForGlobalIndex(operatorData, TENSOR_TEST_SPACE, "w", tensorCellOrdinal, testGid);
        TEST_ASSERT(localTestOrdinal >= 0);
        for (IndexType localTrialOrdinal=0; localTrialOrdinal<boundaryBlock.trialShape.totalDofs(); localTrialOrdinal++)
        {
          const IndexType trialGid = trialOffsets[1] + operatorData.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "tn", boundaryBlock.entityOrdinal, localTrialOrdinal);
          localB[static_cast<size_t>(localTestOrdinal * totalTrialDofCount + trialGid)] += boundaryBlock.bMatrix[static_cast<size_t>(boundaryTestOrdinal * boundaryBlock.trialShape.totalDofs() + localTrialOrdinal)];
        }
      }
    }

    auto localK = TensorLocalSchurComplementBuilder::formDenseTestNormSchurComplement(localB,
                                                                                      localTestDofCount,
                                                                                      totalTrialDofCount,
                                                                                      localG);
    for (IndexType row=0; row<totalTrialDofCount; row++)
    {
      for (IndexType col=0; col<totalTrialDofCount; col++)
      {
        expected[row] += localK[static_cast<size_t>(row * totalTrialDofCount + col)] * input[col];
      }
    }
  }

  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(yHost(localOrdinal,0), expected[gid], 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeGraphVolumeBoundarySchurActionMatchesAssembledCrs )
{
  auto tensorMesh = makeTraceVolumeAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("matrix-free-graph-volume-boundary-action"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 2),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn", 2, 0),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  const std::vector<std::string> trialVarNames{"f", "tn"};
  const std::vector<std::string> testVarNames{"w"};
  const double l2Weight = 1.75;
  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeBoundarySchurComplementMatrix(tensorMesh,
                                                                                                  dofLayout,
                                                                                                  form,
                                                                                                  trialVarNames,
                                                                                                  testVarNames,
                                                                                                  l2Weight);
  TensorOperatorData operatorData(tensorMesh, dofLayout);
  IndexType totalTrialDofCount = 0;
  for (const auto &trialVarName : trialVarNames)
  {
    totalTrialDofCount += operatorData.coefficientCount(TENSOR_TRIAL_SPACE, trialVarName);
  }

  std::vector<double> input(totalTrialDofCount, 0.0);
  for (IndexType gid=0; gid<totalTrialDofCount; gid++) input[gid] = 0.15 + 0.03 * static_cast<double>((gid % 19) + 1);
  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  auto actual = TensorLocalSchurComplementBuilder::applyAffineGraphVolumeBoundarySchurComplement(tensorMesh,
                                                                                                 dofLayout,
                                                                                                 form,
                                                                                                 trialVarNames,
                                                                                                 testVarNames,
                                                                                                 input,
                                                                                                 l2Weight);
  TEST_EQUALITY(actual.size(), static_cast<size_t>(totalTrialDofCount));
  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(actual[gid], yHost(localOrdinal,0), 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, MatrixFreeGraphNormSchurVolumeDirectionalTraceOperatorApplyMatchesAssembledCrs )
{
  auto tensorMesh = makeDirectionalTraceVolumeAffineTensorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  auto form = Teuchos::rcp(new TensorBilinearForm("graph-schur-directional-trace-facade-action"));
  form->addTerm(TensorBilinearTerm(makeFieldTensorVar("f", 2),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::singleDerivative(2, 1),
                                   TensorFunction::constant(-1.0)));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn_x", 2, 0),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(1.0),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));
  form->addTerm(TensorBilinearTerm(makeFluxTensorVar("tn_t", 2, 1),
                                   makeFieldTensorVar("w", 2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorComponentOperatorDescriptor::valueOnly(2),
                                   TensorFunction::constant(0.5),
                                   TENSOR_BILINEAR_TERM_BOUNDARY));

  const std::vector<std::string> trialVarNames{"f", "tn_x", "tn_t"};
  const std::vector<std::string> testVarNames{"w"};
  const double l2Weight = 1.75;
  MatrixFreeGraphNormSchurTensorOperatorFactory factory(l2Weight);
  TEST_ASSERT(!factory.canBuild(tensorMesh, dofLayout, form, trialVarNames, testVarNames, false));
  TEST_ASSERT(factory.canBuild(tensorMesh, dofLayout, form, trialVarNames, testVarNames, true));

  auto matrix = TensorLocalSchurComplementBuilder::affineGraphVolumeBoundarySchurComplementMatrix(tensorMesh,
                                                                                                  dofLayout,
                                                                                                  form,
                                                                                                  trialVarNames,
                                                                                                  testVarNames,
                                                                                                  l2Weight);
  auto op = factory.build(tensorMesh, dofLayout, form, trialVarNames, testVarNames, true);
  TEST_EQUALITY(op->domainVarName(), "f,tn_x,tn_t");
  TEST_EQUALITY(op->rangeVarName(),  "f,tn_x,tn_t");

  std::vector<double> input(op->domainSize(), 0.0);
  for (IndexType gid=0; gid<op->domainSize(); gid++) input[gid] = -0.1 + 0.025 * static_cast<double>((gid % 17) + 1);

  TVector<double> x(matrix->getDomainMap(), 1);
  TVector<double> y(matrix->getRangeMap(), 1);
  auto xHost = x.getLocalViewHost(Tpetra::Access::ReadWrite);
  for (size_t localOrdinal=0; localOrdinal<matrix->getDomainMap()->getLocalNumElements(); localOrdinal++) xHost(localOrdinal,0) = input[matrix->getDomainMap()->getGlobalElement(localOrdinal)];
  matrix->apply(x, y);

  std::vector<double> actual;
  op->apply(input, actual);
  TEST_EQUALITY(actual.size(), static_cast<size_t>(op->rangeSize()));
  auto yHost = y.getLocalViewHost(Tpetra::Access::ReadOnly);
  for (size_t localOrdinal=0; localOrdinal<matrix->getRangeMap()->getLocalNumElements(); localOrdinal++)
  {
    const auto gid = matrix->getRangeMap()->getGlobalElement(localOrdinal);
    TEST_ASSERT(closeEnough(actual[gid], yHost(localOrdinal,0), 1e-10, 1e-12));
  }
}

TEUCHOS_UNIT_TEST( TensorLocalSchurComplement, RejectsInvalidGDiagonal )
{
  auto shape = makeShape({2});
  TensorProductLocalOperator bOperator(shape,
                                       shape,
                                       {TensorComponentLocalMatrix(2, 2, {1.0, 0.0,
                                                                          0.0, 1.0})});
  TEST_THROW(TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, {1.0}), std::invalid_argument);
  TEST_THROW(TensorLocalSchurComplementBuilder::formDiagonalTestNormSchurComplement(bOperator, {1.0, 0.0}), std::invalid_argument);
}
}
