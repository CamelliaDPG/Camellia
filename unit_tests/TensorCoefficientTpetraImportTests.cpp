#include "Teuchos_UnitTestHarness.hpp"

#include "TensorBasisCache.h"
#include "TensorCoefficientMap.h"
#include "TensorCoefficientTpetraImport.h"
#include "TensorCoefficientValueEvaluator.h"
#include "TensorMaxwell1D2VMomentReductionPipeline.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "Var.h"

#include <Tpetra_Core.hpp>
#include <Kokkos_Core.hpp>

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

TensorMeshPtr makeTensorCoefficientTpetraImportMesh(bool includeTrace = false,
                                                     int configurationElementCount = 1)
{
  std::vector<double> configurationVertices;
  std::vector<ActiveLeafSet1D> configurationLeafSets;
  for (int elementOrdinal=0; elementOrdinal<=configurationElementCount; elementOrdinal++)
  {
    configurationVertices.push_back(static_cast<double>(elementOrdinal) / static_cast<double>(configurationElementCount));
  }
  for (int elementOrdinal=0; elementOrdinal<configurationElementCount; elementOrdinal++)
  {
    configurationLeafSets.push_back(ActiveLeafSet1D({{0,0}}));
  }
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D(configurationVertices, configurationLeafSets, false, 1));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-2.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1, component2}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar1C2V("f"), Teuchos::null, std::vector<int>{0,0,0})));
  if (includeTrace)
  {
    tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeTraceTensorVar1C2V("tn", 0), Teuchos::null, std::vector<int>{0,0,0})));
  }
  return tensorMesh;
}

double deterministicCoefficient(IndexType globalIndex)
{
  return 0.125 + 0.25 * static_cast<double>(globalIndex + 1);
}
}

TEUCHOS_UNIT_TEST( TensorCoefficientTpetraImport, ImportsVolumeCellGhostsIntoOwnedCoefficientView )
{
  auto tensorMesh = makeTensorCoefficientTpetraImportMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});

  std::vector<double> globalCoefficients(coefficientMap.coefficientCount(), 0.0);
  for (IndexType gid=0; gid<coefficientMap.coefficientCount(); gid++)
  {
    globalCoefficients[gid] = deterministicCoefficient(gid);
  }
  auto expected = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap,
                                                                            basisCache,
                                                                            globalCoefficients,
                                                                            cellTuple);

  auto ownedMap = TensorCoefficientTpetraImport::ownedMapForDefaultPartition(coefficientMap, Tpetra::getDefaultComm());
  TensorCoefficientTpetraImport::multivector_type ownedVector(ownedMap, 1);
  auto ownedHost = ownedVector.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (size_t localOrdinal=0; localOrdinal<ownedMap->getLocalNumElements(); localOrdinal++)
  {
    const IndexType gid = static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<TensorCoefficientTpetraImport::map_type::local_ordinal_type>(localOrdinal)));
    ownedHost(localOrdinal,0) = deterministicCoefficient(gid);
  }

  auto ownedValues = TensorCoefficientTpetraImport::ownedValuesForVolumeCell(coefficientMap,
                                                                             ownedVector,
                                                                             cellTuple);
  auto actual = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap,
                                                                          basisCache,
                                                                          ownedValues,
                                                                          cellTuple);
  for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
  {
    TEST_FLOATING_EQUALITY(actual(pointOrdinal), expected(pointOrdinal), 1e-14);
  }

  auto comm = Tpetra::getDefaultComm();
  const auto expectedGhosts = coefficientMap.neededIndicesForVolumeCell(cellTuple, comm->getRank(), comm->getSize());
  TEST_EQUALITY(ownedValues.ghostCoefficientsByGlobalIndex.size(), expectedGhosts.size());
  for (IndexType gid : expectedGhosts)
  {
    TEST_ASSERT(ownedValues.ghostCoefficientsByGlobalIndex.find(gid) != ownedValues.ghostCoefficientsByGlobalIndex.end());
    TEST_FLOATING_EQUALITY(ownedValues.ghostCoefficientsByGlobalIndex[gid], deterministicCoefficient(gid), 1e-14);
  }

  TEST_ASSERT(ownedValues.hasDeviceMirrors());
  TEST_EQUALITY(ownedValues.ownedCoefficientsDevice.extent(0), ownedValues.ownedCoefficients.size());
  TEST_EQUALITY(ownedValues.ghostGlobalIndicesDevice.extent(0), expectedGhosts.size());
  TEST_EQUALITY(ownedValues.ghostCoefficientsDevice.extent(0), expectedGhosts.size());

  auto ownedDevice = ownedValues.ownedCoefficientsDevice;
  const IndexType ownedStart = ownedValues.ownedRange.first;
  double ownedMirrorError = 0.0;
  Kokkos::parallel_reduce("check tensor owned coefficient device mirror",
                          static_cast<int>(ownedValues.ownedCoefficients.size()),
                          KOKKOS_LAMBDA(const int localOrdinal, double &error)
                          {
                            const IndexType gid = ownedStart + static_cast<IndexType>(localOrdinal);
                            const double expectedValue = 0.125 + 0.25 * static_cast<double>(gid + 1);
                            const double diff = ownedDevice(localOrdinal) - expectedValue;
                            error += diff * diff;
                          },
                          ownedMirrorError);
  TEST_FLOATING_EQUALITY(ownedMirrorError, 0.0, 1e-14);

  auto ghostIndicesDevice = ownedValues.ghostGlobalIndicesDevice;
  auto ghostValuesDevice = ownedValues.ghostCoefficientsDevice;
  double ghostMirrorError = 0.0;
  Kokkos::parallel_reduce("check tensor ghost coefficient device mirror",
                          static_cast<int>(expectedGhosts.size()),
                          KOKKOS_LAMBDA(const int ghostOrdinal, double &error)
                          {
                            const IndexType gid = ghostIndicesDevice(ghostOrdinal);
                            const double expectedValue = 0.125 + 0.25 * static_cast<double>(gid + 1);
                            const double diff = ghostValuesDevice(ghostOrdinal) - expectedValue;
                            error += diff * diff;
                          },
                          ghostMirrorError);
  TEST_FLOATING_EQUALITY(ghostMirrorError, 0.0, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorCoefficientTpetraImport, ImportsVolumeCellWorksetGhostsIntoOwnedCoefficientViews )
{
  auto tensorMesh = makeTensorCoefficientTpetraImportMesh(false, 2);
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");

  std::vector<double> globalCoefficients(coefficientMap.coefficientCount(), 0.0);
  for (IndexType gid=0; gid<coefficientMap.coefficientCount(); gid++)
  {
    globalCoefficients[gid] = deterministicCoefficient(gid);
  }

  std::vector<TensorMeshTopology::TensorCellTuple> cellTuples;
  cellTuples.push_back(tensorMesh->topology()->tensorCellTuple(0));
  cellTuples.push_back(tensorMesh->topology()->tensorCellTuple(1));

  auto ownedMap = TensorCoefficientTpetraImport::ownedMapForDefaultPartition(coefficientMap, Tpetra::getDefaultComm());
  TensorCoefficientTpetraImport::multivector_type ownedVector(ownedMap, 1);
  auto ownedHost = ownedVector.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (size_t localOrdinal=0; localOrdinal<ownedMap->getLocalNumElements(); localOrdinal++)
  {
    const IndexType gid = static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<TensorCoefficientTpetraImport::map_type::local_ordinal_type>(localOrdinal)));
    ownedHost(localOrdinal,0) = deterministicCoefficient(gid);
  }

  auto ownedValuesByCell = TensorCoefficientTpetraImport::ownedValuesForVolumeCells(coefficientMap,
                                                                                    ownedVector,
                                                                                    cellTuples);
  TEST_EQUALITY(ownedValuesByCell.size(), cellTuples.size());

  auto comm = Tpetra::getDefaultComm();
  for (size_t cellOrdinal=0; cellOrdinal<cellTuples.size(); cellOrdinal++)
  {
    TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuples[cellOrdinal], std::vector<int>{2,2,2});
    auto expected = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap,
                                                                              basisCache,
                                                                              globalCoefficients,
                                                                              cellTuples[cellOrdinal]);
    auto actual = TensorCoefficientValueEvaluator::volumeValuesOnTensorCell(coefficientMap,
                                                                            basisCache,
                                                                            ownedValuesByCell[cellOrdinal],
                                                                            cellTuples[cellOrdinal]);
    for (IndexType pointOrdinal=0; pointOrdinal<basisCache.tensorPointCount(); pointOrdinal++)
    {
      TEST_FLOATING_EQUALITY(actual(pointOrdinal), expected(pointOrdinal), 1e-14);
    }

    const auto expectedGhosts = coefficientMap.neededIndicesForVolumeCell(cellTuples[cellOrdinal], comm->getRank(), comm->getSize());
    TEST_EQUALITY(ownedValuesByCell[cellOrdinal].ghostCoefficientsByGlobalIndex.size(), expectedGhosts.size());
    TEST_ASSERT(ownedValuesByCell[cellOrdinal].hasDeviceMirrors());
  }
}

TEUCHOS_UNIT_TEST( TensorCoefficientTpetraImport, ReusableVolumeWorksetImportPlanFeedsSuccessiveCoefficientVectors )
{
  auto tensorMesh = makeTensorCoefficientTpetraImportMesh(false, 2);
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");

  std::vector<TensorMeshTopology::TensorCellTuple> cellTuples;
  cellTuples.push_back(tensorMesh->topology()->tensorCellTuple(0));
  cellTuples.push_back(tensorMesh->topology()->tensorCellTuple(1));

  auto ownedMap = TensorCoefficientTpetraImport::ownedMapForDefaultPartition(coefficientMap, Tpetra::getDefaultComm());
  TensorCoefficientTpetraImport::VolumeCellWorksetImportPlan importPlan(coefficientMap, ownedMap, cellTuples);
  TEST_EQUALITY(importPlan.worksetSize(), cellTuples.size());
  TEST_ASSERT(importPlan.overlapMap() != Teuchos::null);

  for (double scale : {1.0, -2.0})
  {
    TensorCoefficientTpetraImport::multivector_type ownedVector(ownedMap, 1);
    auto ownedHost = ownedVector.getLocalViewHost(Tpetra::Access::OverwriteAll);
    for (size_t localOrdinal=0; localOrdinal<ownedMap->getLocalNumElements(); localOrdinal++)
    {
      const IndexType gid = static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<TensorCoefficientTpetraImport::map_type::local_ordinal_type>(localOrdinal)));
      ownedHost(localOrdinal,0) = scale * deterministicCoefficient(gid);
    }

    auto ownedValuesByCell = importPlan.ownedValues(ownedVector);
    TEST_EQUALITY(ownedValuesByCell.size(), cellTuples.size());
    for (size_t cellOrdinal=0; cellOrdinal<cellTuples.size(); cellOrdinal++)
    {
      const auto expectedGhosts = importPlan.neededGhostIndicesByCell()[cellOrdinal];
      TEST_EQUALITY(ownedValuesByCell[cellOrdinal].ghostCoefficientsByGlobalIndex.size(), expectedGhosts.size());
      TEST_ASSERT(ownedValuesByCell[cellOrdinal].hasDeviceMirrors());
      for (IndexType gid : expectedGhosts)
      {
        TEST_FLOATING_EQUALITY(ownedValuesByCell[cellOrdinal].ghostCoefficientsByGlobalIndex[gid],
                               scale * deterministicCoefficient(gid),
                               1e-14);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( TensorCoefficientTpetraImport, ImportedOwnedCoefficientValuesFeedMaxwellMomentPipeline )
{
  auto tensorMesh = makeTensorCoefficientTpetraImportMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "f");
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});

  std::vector<double> globalCoefficients(coefficientMap.coefficientCount(), 0.0);
  for (IndexType gid=0; gid<coefficientMap.coefficientCount(); gid++)
  {
    globalCoefficients[gid] = deterministicCoefficient(gid);
  }

  auto ownedMap = TensorCoefficientTpetraImport::ownedMapForDefaultPartition(coefficientMap, Tpetra::getDefaultComm());
  TensorCoefficientTpetraImport::multivector_type ownedVector(ownedMap, 1);
  auto ownedHost = ownedVector.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (size_t localOrdinal=0; localOrdinal<ownedMap->getLocalNumElements(); localOrdinal++)
  {
    const IndexType gid = static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<TensorCoefficientTpetraImport::map_type::local_ordinal_type>(localOrdinal)));
    ownedHost(localOrdinal,0) = deterministicCoefficient(gid);
  }

  auto ownedValues = TensorCoefficientTpetraImport::ownedValuesForVolumeCell(coefficientMap,
                                                                             ownedVector,
                                                                             cellTuple);
  const std::vector<int> configurationComponentOrdinals{0};
  const std::vector<int> velocityComponentOrdinals{1,2};
  TensorMaxwell1D2VMomentReductionPipeline::HostMatrix velocityPoints("velocityPoints", 4, 2);
  velocityPoints(0,0) = -1.0; velocityPoints(0,1) = -2.0;
  velocityPoints(1,0) =  0.0; velocityPoints(1,1) = -2.0;
  velocityPoints(2,0) = -1.0; velocityPoints(2,1) =  2.0;
  velocityPoints(3,0) =  0.0; velocityPoints(3,1) =  2.0;
  TensorMaxwell1D2VMomentReductionPipeline::HostVector velocityWeights("velocityWeights", 4);
  velocityWeights(0) = 0.5;
  velocityWeights(1) = 1.0;
  velocityWeights(2) = 1.5;
  velocityWeights(3) = 2.0;

  const std::vector<double> charges{2.0};
  const double eps0 = 4.0;
  const std::vector<double> n0{0.25, 0.75};
  auto expectedSpeciesValues = TensorCoefficientValueEvaluator::volumeValuesAsConfigurationVelocityMatrix1D2V(coefficientMap,
                                                                                                              basisCache,
                                                                                                              globalCoefficients,
                                                                                                              cellTuple,
                                                                                                              configurationComponentOrdinals,
                                                                                                              velocityComponentOrdinals);
  auto expected = TensorMaxwell1D2VMomentReductionPipeline::buildRHSMomentData(1,
                                                                               2,
                                                                               charges,
                                                                               eps0,
                                                                               {expectedSpeciesValues},
                                                                               velocityPoints,
                                                                               velocityWeights,
                                                                               n0);
  auto actual = TensorMaxwell1D2VMomentReductionPipeline::buildRHSMomentDataFromOwnedCoefficientValues(1,
                                                                                                       2,
                                                                                                       charges,
                                                                                                       eps0,
                                                                                                       {ownedValues},
                                                                                                       coefficientMap,
                                                                                                       basisCache,
                                                                                                       cellTuple,
                                                                                                       configurationComponentOrdinals,
                                                                                                       velocityComponentOrdinals,
                                                                                                       velocityPoints,
                                                                                                       velocityWeights,
                                                                                                       n0);

  for (int point=0; point<2; point++)
  {
    TEST_FLOATING_EQUALITY(actual.rhoRHS(0, point), expected.rhoRHS(0, point), 1e-14);
    TEST_FLOATING_EQUALITY(actual.Jx(0, point), expected.Jx(0, point), 1e-14);
    TEST_FLOATING_EQUALITY(actual.Jz(0, point), expected.Jz(0, point), 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorCoefficientTpetraImport, ImportsSkeletonEntityGhostsIntoOwnedCoefficientView )
{
  auto tensorMesh = makeTensorCoefficientTpetraImportMesh(true);
  auto dofLayout = tensorMesh->createDofLayout();
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "tn");
  const auto &skeleton = tensorMesh->topology()->tensorSkeleton(0);
  const auto &entity = skeleton.entity(0);

  auto ownedMap = TensorCoefficientTpetraImport::ownedMapForDefaultPartition(coefficientMap, Tpetra::getDefaultComm());
  TensorCoefficientTpetraImport::multivector_type ownedVector(ownedMap, 1);
  auto ownedHost = ownedVector.getLocalViewHost(Tpetra::Access::OverwriteAll);
  for (size_t localOrdinal=0; localOrdinal<ownedMap->getLocalNumElements(); localOrdinal++)
  {
    const IndexType gid = static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<TensorCoefficientTpetraImport::map_type::local_ordinal_type>(localOrdinal)));
    ownedHost(localOrdinal,0) = deterministicCoefficient(gid);
  }

  auto ownedValues = TensorCoefficientTpetraImport::ownedValuesForSkeletonEntity(coefficientMap,
                                                                                 ownedVector,
                                                                                 entity);
  auto comm = Tpetra::getDefaultComm();
  const auto expectedGhosts = coefficientMap.neededIndicesForSkeletonEntity(entity, comm->getRank(), comm->getSize());
  TEST_EQUALITY(ownedValues.ghostCoefficientsByGlobalIndex.size(), expectedGhosts.size());
  for (IndexType gid : expectedGhosts)
  {
    TEST_ASSERT(ownedValues.ghostCoefficientsByGlobalIndex.find(gid) != ownedValues.ghostCoefficientsByGlobalIndex.end());
    TEST_FLOATING_EQUALITY(ownedValues.ghostCoefficientsByGlobalIndex[gid], deterministicCoefficient(gid), 1e-14);
  }
}
