#include "Teuchos_UnitTestHarness.hpp"

#include "TensorBasisCache.h"
#include "TensorCoefficientMap.h"
#include "TensorCoefficientValueEvaluator.h"
#include "TensorDofLayout.h"
#include "TensorMaxwell1D2VMomentReductionPipeline.h"
#include "TensorMesh.h"
#include "Var.h"

#include <cmath>
#include <map>
#include <vector>

using namespace Camellia;

namespace
{
TensorVarPtr makeFieldTensorVar1C2V(const std::string &name)
{
  VarPtr x  = Teuchos::rcp(new Var(0, 0, name + "_x",  Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr ux = Teuchos::rcp(new Var(1, 0, name + "_ux", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr uz = Teuchos::rcp(new Var(2, 0, name + "_uz", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, std::vector<VarPtr>{x, ux, uz}));
}

TensorVarPtr makeTraceTensorVar1C2V(const std::string &name, int traceComponentOrdinal)
{
  VarPtr x  = Teuchos::rcp(new Var(0, 0, name + "_x",  Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  VarPtr ux = Teuchos::rcp(new Var(1, 0, name + "_ux", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  VarPtr uz = Teuchos::rcp(new Var(2, 0, name + "_uz", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_TRACE, std::vector<VarPtr>{x, ux, uz}, traceComponentOrdinal));
}

TensorMeshPtr makeTensorCoefficientValueEvaluatorMesh(bool includeTrace = false)
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-2.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1, component2}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar1C2V("f"), Teuchos::null, std::vector<int>{0,0,0})));
  tensorMesh->addTestDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar1C2V("w"), Teuchos::null, std::vector<int>{0,0,0})));
  if (includeTrace)
  {
    tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeTraceTensorVar1C2V("tn", 0), Teuchos::null, std::vector<int>{0,0,0})));
  }
  return tensorMesh;
}

std::vector<double> deterministicCoefficients(IndexType coefficientCount)
{
  std::vector<double> coefficients(coefficientCount, 0.0);
  for (IndexType i=0; i<coefficientCount; i++) coefficients[i] = 0.125 + 0.25 * static_cast<double>(i + 1);
  return coefficients;
}

TEUCHOS_UNIT_TEST( TensorCoefficientValueEvaluator, VolumeValuesOnTensorCellMatchBasisExpansion )
{
  auto tensorMesh = makeTensorCoefficientValueEvaluatorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});
  auto coefficients = deterministicCoefficients(coefficientMap.coefficientCount());

  auto values = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap, basisCache, coefficients, cellTuple);
  auto localToGlobal = coefficientMap.indicesForVolumeCell(cellTuple);
  for (IndexType tensorPointOrdinal=0; tensorPointOrdinal<basisCache.tensorPointCount(); tensorPointOrdinal++)
  {
    std::vector<int> pointOrdinals(basisCache.componentCount(), 0);
    for (int d=0; d<basisCache.componentCount(); d++)
      pointOrdinals[d] = static_cast<int>((tensorPointOrdinal / basisCache.pointStride(d)) % basisCache.componentCache(d).pointCount());

    double expected = 0.0;
    for (IndexType localOrdinal=0; localOrdinal<static_cast<IndexType>(localToGlobal.size()); localOrdinal++)
    {
      IndexType workingLocalOrdinal = localOrdinal;
      double basisProduct = 1.0;
      for (int d=0; d<basisCache.componentCount(); d++)
      {
        const auto &componentCache = basisCache.componentCache(d);
        const int basisOrdinal = static_cast<int>(workingLocalOrdinal % componentCache.basisCardinality());
        workingLocalOrdinal /= componentCache.basisCardinality();
        basisProduct *= componentCache.valuesAtQuadrature().view_host()(basisOrdinal, pointOrdinals[d]);
      }
      expected += coefficients[localToGlobal[localOrdinal]] * basisProduct;
    }
    TEST_FLOATING_EQUALITY(values(tensorPointOrdinal), expected, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorCoefficientValueEvaluator, VolumeValuesUseOwnedRangeAndGhostCoefficients )
{
  auto tensorMesh = makeTensorCoefficientValueEvaluatorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});
  auto coefficients = deterministicCoefficients(coefficientMap.coefficientCount());
  auto expected = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap, basisCache, coefficients, cellTuple);

  const auto ownedRange = std::make_pair<IndexType,IndexType>(0, coefficientMap.coefficientCount() - 1);
  std::vector<double> ownedCoefficients(coefficients.begin(), coefficients.end() - 1);
  std::map<IndexType,double> ghostCoefficients;
  ghostCoefficients[coefficientMap.coefficientCount() - 1] = coefficients.back();

  auto actual = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap,
                                                                          basisCache,
                                                                          ownedRange,
                                                                          ownedCoefficients,
                                                                          ghostCoefficients,
                                                                          cellTuple);
  for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
  {
    TEST_FLOATING_EQUALITY(actual(pointOrdinal), expected(pointOrdinal), 1e-14);
  }

  ghostCoefficients.clear();
  TEST_THROW(TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap,
                                                                       basisCache,
                                                                       ownedRange,
                                                                       ownedCoefficients,
                                                                       ghostCoefficients,
                                                                       cellTuple),
             std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorCoefficientValueEvaluator, OwnedCoefficientValuesCanBePreparedFromRankLocalNeededIndices )
{
  auto tensorMesh = makeTensorCoefficientValueEvaluatorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});
  auto coefficients = deterministicCoefficients(coefficientMap.coefficientCount());
  auto expected = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap, basisCache, coefficients, cellTuple);

  const int rankOrdinal = 0;
  const int rankCount = 2;
  const auto ownedRange = coefficientMap.ownedRangeForRank(rankOrdinal, rankCount);
  std::vector<double> ownedCoefficients(coefficients.begin() + ownedRange.first,
                                        coefficients.begin() + ownedRange.second);
  std::map<IndexType,double> ghostCoefficients;
  for (IndexType ghostIndex : coefficientMap.neededIndicesForVolumeCell(cellTuple, rankOrdinal, rankCount))
  {
    ghostCoefficients[ghostIndex] = coefficients[ghostIndex];
  }

  TensorCoefficientValueEvaluator::OwnedCoefficientValues ownedValues(ownedRange, ownedCoefficients, ghostCoefficients);
  auto actual = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap, basisCache, ownedValues, cellTuple);
  for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
  {
    TEST_FLOATING_EQUALITY(actual(pointOrdinal), expected(pointOrdinal), 1e-14);
  }

  TEST_ASSERT(!ghostCoefficients.empty());
  ghostCoefficients.erase(ghostCoefficients.begin());
  TensorCoefficientValueEvaluator::OwnedCoefficientValues incompleteOwnedValues(ownedRange, ownedCoefficients, ghostCoefficients);
  TEST_THROW(TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap, basisCache, incompleteOwnedValues, cellTuple),
             std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorCoefficientValueEvaluator, DeviceVolumeValuesMatchHostOwnedCoefficientEvaluation )
{
  auto tensorMesh = makeTensorCoefficientValueEvaluatorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});
  auto coefficients = deterministicCoefficients(coefficientMap.coefficientCount());

  const int rankOrdinal = 1;
  const int rankCount = 2;
  auto ownedRange = coefficientMap.ownedRangeForRank(rankOrdinal, rankCount);
  std::vector<double> ownedCoefficients(coefficients.begin() + ownedRange.first,
                                        coefficients.begin() + ownedRange.second);
  std::map<IndexType,double> ghostCoefficients;
  for (IndexType ghostIndex : coefficientMap.neededIndicesForVolumeCell(cellTuple, rankOrdinal, rankCount))
  {
    ghostCoefficients[ghostIndex] = coefficients[ghostIndex];
  }
  TensorCoefficientValueEvaluator::OwnedCoefficientValues ownedValues(ownedRange, ownedCoefficients, ghostCoefficients);

  auto expected = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap,
                                                                            basisCache,
                                                                            ownedValues,
                                                                            cellTuple);
  auto actualDevice = TensorCoefficientValueEvaluator::volumeValuesOnTensorCellDevice(coefficientMap,
                                                                                      basisCache,
                                                                                      ownedValues,
                                                                                      cellTuple);
  auto actual = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), actualDevice);
  for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
  {
    TEST_FLOATING_EQUALITY(actual(pointOrdinal), expected(pointOrdinal), 1e-14);
  }

  TensorCoefficientValueEvaluator::OwnedCoefficientValues missingDeviceMirrors(ownedValues.ownedRange,
                                                                              ownedValues.ownedCoefficients,
                                                                              ownedValues.ghostCoefficientsByGlobalIndex);
  missingDeviceMirrors.ownedCoefficientsDevice = TensorCoefficientValueEvaluator::OwnedCoefficientValues::DeviceScalarView();
  TEST_THROW(TensorCoefficientValueEvaluator::volumeValuesOnTensorCellDevice(coefficientMap,
                                                                             basisCache,
                                                                             missingDeviceMirrors,
                                                                             cellTuple),
             std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorCoefficientValueEvaluator, ConfigurationVelocityMatrixFeedsMaxwellMomentPipeline )
{
  auto tensorMesh = makeTensorCoefficientValueEvaluatorMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});
  auto coefficients = deterministicCoefficients(coefficientMap.coefficientCount());

  auto values = TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(coefficientMap,
                                                                                               basisCache,
                                                                                               coefficients,
                                                                                               cellTuple,
                                                                                               std::vector<int>{0},
                                                                                               std::vector<int>{1,2});
  TEST_EQUALITY(values.extent_int(0), basisCache.componentCache(0).pointCount());
  TEST_EQUALITY(values.extent_int(1), basisCache.componentCache(1).pointCount() * basisCache.componentCache(2).pointCount());

  const auto ownedRange = std::make_pair<IndexType,IndexType>(0, coefficientMap.coefficientCount() - 1);
  std::vector<double> ownedCoefficients(coefficients.begin(), coefficients.end() - 1);
  std::map<IndexType,double> ghostCoefficients;
  ghostCoefficients[coefficientMap.coefficientCount() - 1] = coefficients.back();
  TensorCoefficientValueEvaluator::OwnedCoefficientValues ownedValues(ownedRange, ownedCoefficients, ghostCoefficients);
  auto ownedValuesMatrix = TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(coefficientMap,
                                                                                                          basisCache,
                                                                                                          ownedValues,
                                                                                                          cellTuple,
                                                                                                          std::vector<int>{0},
                                                                                                          std::vector<int>{1,2});
  for (int configPoint=0; configPoint<values.extent_int(0); configPoint++)
  {
    for (int velocityPoint=0; velocityPoint<values.extent_int(1); velocityPoint++)
    {
      TEST_FLOATING_EQUALITY(ownedValuesMatrix(configPoint,velocityPoint), values(configPoint,velocityPoint), 1e-14);
    }
  }

  ghostCoefficients.clear();
  TensorCoefficientValueEvaluator::OwnedCoefficientValues missingGhostValues(ownedRange, ownedCoefficients, ghostCoefficients);
  TEST_THROW(TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(coefficientMap,
                                                                                            basisCache,
                                                                                            missingGhostValues,
                                                                                            cellTuple,
                                                                                            std::vector<int>{0},
                                                                                            std::vector<int>{1,2}),
             std::invalid_argument);

  TensorMomentReduction::HostMatrix velocityPoints("velocity points", values.extent_int(1), 2);
  TensorMomentReduction::HostVector velocityWeights("velocity weights", values.extent_int(1));
  for (int uxPoint=0; uxPoint<basisCache.componentCache(1).pointCount(); uxPoint++)
  {
    for (int uzPoint=0; uzPoint<basisCache.componentCache(2).pointCount(); uzPoint++)
    {
      const int velocityOrdinal = uxPoint + basisCache.componentCache(1).pointCount() * uzPoint;
      const auto &uxCache = basisCache.componentCache(1);
      const auto &uzCache = basisCache.componentCache(2);
      velocityPoints(velocityOrdinal,0) = uxCache.cellLeft() + 0.5 * (uxCache.referencePoints().view_host()(uxPoint) + 1.0) * uxCache.cellWidth();
      velocityPoints(velocityOrdinal,1) = uzCache.cellLeft() + 0.5 * (uzCache.referencePoints().view_host()(uzPoint) + 1.0) * uzCache.cellWidth();
      velocityWeights(velocityOrdinal) = uxCache.referenceWeights().view_host()(uxPoint) * uxCache.jacobian()
                                       * uzCache.referenceWeights().view_host()(uzPoint) * uzCache.jacobian();
    }
  }

  std::vector<double> n0(values.extent_int(0), 0.0);
  const double q = 2.0;
  const double eps0 = 4.0;
  auto data = TensorMaxwell1D2VMomentReductionPipeline::buildRHSMomentData(basisCache.componentCache(0).pointCount(),
                                                                          1,
                                                                          {q},
                                                                          eps0,
                                                                          {values},
                                                                          velocityPoints,
                                                                          velocityWeights,
                                                                          n0);
  for (int configPoint=0; configPoint<values.extent_int(0); configPoint++)
  {
    double expectedDensity = 0.0;
    double expectedJx = 0.0;
    double expectedJz = 0.0;
    for (int velocityOrdinal=0; velocityOrdinal<values.extent_int(1); velocityOrdinal++)
    {
      expectedDensity += values(configPoint,velocityOrdinal) * velocityWeights(velocityOrdinal);
      expectedJx += q * values(configPoint,velocityOrdinal) * velocityWeights(velocityOrdinal) * velocityPoints(velocityOrdinal,0);
      expectedJz += q * values(configPoint,velocityOrdinal) * velocityWeights(velocityOrdinal) * velocityPoints(velocityOrdinal,1);
    }
    TEST_FLOATING_EQUALITY(data.rhoRHS(configPoint, 0), q / eps0 * expectedDensity, 1e-14);
    TEST_FLOATING_EQUALITY(data.Jx(configPoint, 0), expectedJx, 1e-14);
    TEST_FLOATING_EQUALITY(data.Jz(configPoint, 0), expectedJz, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorCoefficientValueEvaluator, RejectsInvalidInputs )
{
  auto tensorMesh = makeTensorCoefficientValueEvaluatorMesh(true);
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});
  auto coefficients = deterministicCoefficients(coefficientMap.coefficientCount());
  coefficients.pop_back();
  TEST_THROW(TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap, basisCache, coefficients, cellTuple), std::invalid_argument);

  coefficients = deterministicCoefficients(coefficientMap.coefficientCount());
  TEST_THROW(TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(coefficientMap,
                                                                                           basisCache,
                                                                                           coefficients,
                                                                                           cellTuple,
                                                                                           std::vector<int>{0},
                                                                                           std::vector<int>{1}),
             std::invalid_argument);

  TensorCoefficientMap traceCoefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "tn");
  TEST_THROW(TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(traceCoefficientMap, basisCache, coefficients, cellTuple), std::invalid_argument);
}
}
