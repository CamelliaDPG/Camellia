#include "Teuchos_UnitTestHarness.hpp"

#include "TensorCoefficientMap.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "TensorOperatorData.h"
#include "Var.h"

#include <Kokkos_Core.hpp>

using namespace Camellia;

namespace
{
TensorVarPtr makeFieldTensorVar(const std::string &name)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, std::vector<VarPtr>{vx, vt}));
}

TensorVarPtr makeFluxTraceTensorVar(const std::string &name, int traceComponentOrdinal)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_TRACE, std::vector<VarPtr>{vx, vt}, traceComponentOrdinal));
}

TensorMeshPtr makeTensorMeshForOperatorDataTests(bool includeTrace = false)
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }, false, 3));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));

  auto trialVar = makeFieldTensorVar("u");
  auto testVar = trialVar->copyWithNewName("v");
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,0})));
  tensorMesh->addTestDiscretization(Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{0,0})));
  if (includeTrace)
  {
    tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFluxTraceTensorVar("tn", 1), Teuchos::null, std::vector<int>{0,0})));
  }
  return tensorMesh;
}

TensorMeshPtr makeDirectionalTraceTensorMeshForOperatorDataTests()
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests(false);
  std::vector<VarPtr> componentVars;
  componentVars.push_back(Teuchos::rcp(new Var(10, 0, "tn_x_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  componentVars.push_back(Teuchos::rcp(new Var(11, 0, "tn_t_component", Camellia::OP_VALUE, L2, FLUX, Teuchos::null, true, UNKNOWN_FS)));
  auto directionalTraces = TensorVarFactory::directionalTraceFamily("tn", {"x", "t"}, componentVars);
  for (const auto &traceVar : directionalTraces)
  {
    tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(traceVar, Teuchos::null, std::vector<int>{0,0})));
  }
  return tensorMesh;
}

TEUCHOS_UNIT_TEST( TensorOperatorData, ComponentGeometryCompiledToViews )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);

  TEST_EQUALITY(data.componentCount(), 2);
  TEST_EQUALITY(data.componentData(0).activeCellCount(), 2);
  TEST_EQUALITY(data.componentData(1).activeCellCount(), 3);
  TEST_EQUALITY(data.componentData(1).vertexCount(), 4);

  auto widths = data.componentData(1).cellWidths();
  auto jacobians = data.componentData(1).cellJacobians();
  auto polyOrders = data.componentData(1).cellPolyOrders();
  widths.sync_host();
  jacobians.sync_host();
  polyOrders.sync_host();

  TEST_FLOATING_EQUALITY(widths.view_host()(0), 1.0, 1e-15);
  TEST_FLOATING_EQUALITY(widths.view_host()(1), 0.5, 1e-15);
  TEST_FLOATING_EQUALITY(widths.view_host()(2), 0.5, 1e-15);
  TEST_FLOATING_EQUALITY(jacobians.view_host()(1), 0.25, 1e-15);
  TEST_EQUALITY(polyOrders.view_host()(0), 3);
}

TEUCHOS_UNIT_TEST( TensorOperatorData, DeviceViewsExposeComponentGeometry )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);

  auto widths = data.componentData(1).cellWidths();
  auto jacobians = data.componentData(1).cellJacobians();
  auto polyOrders = data.componentData(1).cellPolyOrders();
  auto widthsDevice = widths.view_device();
  auto jacobiansDevice = jacobians.view_device();
  auto polyOrdersDevice = polyOrders.view_device();

  double widthSum = 0.0;
  double jacobianSum = 0.0;
  int polyOrderSum = 0;
  Kokkos::parallel_reduce("TensorOperatorDataDeviceWidthSum", static_cast<int>(widths.extent(0)),
                          KOKKOS_LAMBDA(const int i, double &sum) { sum += widthsDevice(i); }, widthSum);
  Kokkos::parallel_reduce("TensorOperatorDataDeviceJacobianSum", static_cast<int>(jacobians.extent(0)),
                          KOKKOS_LAMBDA(const int i, double &sum) { sum += jacobiansDevice(i); }, jacobianSum);
  Kokkos::parallel_reduce("TensorOperatorDataDevicePolyOrderSum", static_cast<int>(polyOrders.extent(0)),
                          KOKKOS_LAMBDA(const int i, int &sum) { sum += polyOrdersDevice(i); }, polyOrderSum);

  TEST_FLOATING_EQUALITY(widthSum, 2.0, 1e-15);
  TEST_FLOATING_EQUALITY(jacobianSum, 1.0, 1e-15);
  TEST_EQUALITY(polyOrderSum, 9);
}

TEUCHOS_UNIT_TEST( TensorOperatorData, ActiveTensorCellsStoreComponentOrdinals )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);

  TEST_EQUALITY(data.activeCellCount(), 6);
  TEST_EQUALITY(data.activeTensorCellComponentOrdinal(0,0), 0);
  TEST_EQUALITY(data.activeTensorCellComponentOrdinal(0,1), 0);
  TEST_EQUALITY(data.activeTensorCellComponentOrdinal(1,0), 1);
  TEST_EQUALITY(data.activeTensorCellComponentOrdinal(1,1), 0);
  TEST_EQUALITY(data.activeTensorCellComponentOrdinal(2,0), 0);
  TEST_EQUALITY(data.activeTensorCellComponentOrdinal(2,1), 1);
}

TEUCHOS_UNIT_TEST( TensorOperatorData, BoundarySideFlagsMatchTopologyQueries )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);

  for (IndexType cellOrdinal=0; cellOrdinal<tensorMesh->topology()->activeCellCount(); cellOrdinal++)
  {
    auto cellTuple = tensorMesh->topology()->tensorCellTuple(cellOrdinal);
    for (int d=0; d<tensorMesh->topology()->componentCount(); d++)
    {
      for (int sideOrdinalInComponent=0; sideOrdinalInComponent<2; sideOrdinalInComponent++)
      {
        const int tensorSideOrdinal = 2 * d + sideOrdinalInComponent;
        TensorMeshTopology::TensorSide side(cellTuple, d, sideOrdinalInComponent);
        TEST_EQUALITY(data.isBoundarySide(cellOrdinal, tensorSideOrdinal),
                      tensorMesh->topology()->isBoundarySide(side));
      }
    }
  }
}

TEUCHOS_UNIT_TEST( TensorOperatorData, VariableCoefficientCountsMatchCoefficientMap )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);

  TensorCoefficientMap trialMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");
  TensorCoefficientMap testMap(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "v");

  TEST_EQUALITY(data.variableCount(), 2);
  TEST_EQUALITY(data.coefficientCount(TENSOR_TRIAL_SPACE, "u"), trialMap.coefficientCount());
  TEST_EQUALITY(data.coefficientCount(TENSOR_TEST_SPACE, "v"), testMap.coefficientCount());

  bool foundTrial = false;
  bool foundTest = false;
  for (IndexType variableOrdinal=0; variableOrdinal<data.variableCount(); variableOrdinal++)
  {
    const auto &variableData = data.variableData(variableOrdinal);
    if ((variableData.role() == TENSOR_TRIAL_SPACE) && (variableData.name() == "u"))
    {
      foundTrial = true;
      TEST_EQUALITY(variableData.ambientShape().componentCount(), 2);
      TEST_EQUALITY(variableData.coefficientCount(), trialMap.coefficientCount());
    }
    if ((variableData.role() == TENSOR_TEST_SPACE) && (variableData.name() == "v"))
    {
      foundTest = true;
      TEST_EQUALITY(variableData.ambientShape().componentCount(), 2);
      TEST_EQUALITY(variableData.coefficientCount(), testMap.coefficientCount());
    }
  }
  TEST_ASSERT(foundTrial);
  TEST_ASSERT(foundTest);
}

TEUCHOS_UNIT_TEST( TensorOperatorData, VolumeLocalToGlobalViewsMatchCoefficientMap )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");

  const auto &variableData = data.variableData(TENSOR_TRIAL_SPACE, "u");
  auto localCounts = variableData.volumeLocalDofCounts();
  auto localToGlobal = variableData.volumeLocalToGlobalIndices();
  localCounts.sync_host();
  localToGlobal.sync_host();

  for (IndexType cellOrdinal=0; cellOrdinal<tensorMesh->topology()->activeCellCount(); cellOrdinal++)
  {
    auto expectedIndices = coefficientMap.indicesForVolumeCell(tensorMesh->topology()->tensorCellTuple(cellOrdinal));
    TEST_EQUALITY(data.volumeLocalDofCount(TENSOR_TRIAL_SPACE, "u", cellOrdinal), static_cast<IndexType>(expectedIndices.size()));
    TEST_EQUALITY(localCounts.view_host()(cellOrdinal), static_cast<IndexType>(expectedIndices.size()));
    for (IndexType localOrdinal=0; localOrdinal<static_cast<IndexType>(expectedIndices.size()); localOrdinal++)
    {
      TEST_EQUALITY(data.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "u", cellOrdinal, localOrdinal), expectedIndices[localOrdinal]);
      TEST_EQUALITY(localToGlobal.view_host()(cellOrdinal,localOrdinal), expectedIndices[localOrdinal]);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorOperatorData, DeviceViewsExposeVolumeLocalToGlobalMaps )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");

  IndexType expectedCountSum = 0;
  IndexType expectedIndexSum = 0;
  for (IndexType cellOrdinal=0; cellOrdinal<tensorMesh->topology()->activeCellCount(); cellOrdinal++)
  {
    auto expectedIndices = coefficientMap.indicesForVolumeCell(tensorMesh->topology()->tensorCellTuple(cellOrdinal));
    expectedCountSum += expectedIndices.size();
    for (IndexType index : expectedIndices) expectedIndexSum += index;
  }

  const auto &variableData = data.variableData(TENSOR_TRIAL_SPACE, "u");
  auto localCounts = variableData.volumeLocalDofCounts();
  auto localToGlobal = variableData.volumeLocalToGlobalIndices();
  auto localCountsDevice = localCounts.view_device();
  auto localToGlobalDevice = localToGlobal.view_device();

  IndexType actualCountSum = 0;
  IndexType actualIndexSum = 0;
  Kokkos::parallel_reduce("TensorOperatorDataDeviceVolumeLocalDofCountSum", static_cast<int>(localCounts.extent(0)),
                          KOKKOS_LAMBDA(const int cellOrdinal, IndexType &sum) { sum += localCountsDevice(cellOrdinal); }, actualCountSum);
  Kokkos::parallel_reduce("TensorOperatorDataDeviceVolumeLocalToGlobalIndexSum", static_cast<int>(localCounts.extent(0)),
                          KOKKOS_LAMBDA(const int cellOrdinal, IndexType &sum)
                          {
                            for (IndexType localOrdinal=0; localOrdinal<localCountsDevice(cellOrdinal); localOrdinal++)
                            {
                              sum += localToGlobalDevice(cellOrdinal, localOrdinal);
                            }
                          }, actualIndexSum);

  TEST_EQUALITY(actualCountSum, expectedCountSum);
  TEST_EQUALITY(actualIndexSum, expectedIndexSum);
}

TEUCHOS_UNIT_TEST( TensorOperatorData, TraceSkeletonLocalToGlobalViewsAndParityMatchTopology )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests(true);
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "tn");
  auto skeleton = tensorMesh->topology()->tensorSkeleton(1);

  const auto &traceData = data.variableData(TENSOR_TRIAL_SPACE, "tn");
  TEST_ASSERT(traceData.isTraceVariable());
  TEST_EQUALITY(traceData.traceComponentOrdinal(), 1);
  TEST_EQUALITY(data.skeletonEntityCount(TENSOR_TRIAL_SPACE, "tn"), skeleton.entityCount());

  auto localCounts = traceData.skeletonLocalDofCounts();
  auto localToGlobal = traceData.skeletonLocalToGlobalIndices();
  auto vertexOrdinals = traceData.skeletonEntityVertexOrdinals();
  auto sideParities = traceData.skeletonEntitySideParities();
  auto outwardNormalSigns = traceData.skeletonEntityOutwardNormalSigns();
  localCounts.sync_host();
  localToGlobal.sync_host();
  vertexOrdinals.sync_host();
  sideParities.sync_host();
  outwardNormalSigns.sync_host();

  auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(1);
  for (IndexType entityOrdinal=0; entityOrdinal<skeleton.entityCount(); entityOrdinal++)
  {
    const auto &entity = skeleton.entity(entityOrdinal);
    auto expectedIndices = coefficientMap.indicesForSkeletonEntity(entity);
    TEST_EQUALITY(data.skeletonLocalDofCount(TENSOR_TRIAL_SPACE, "tn", entityOrdinal), static_cast<IndexType>(expectedIndices.size()));
    TEST_EQUALITY(localCounts.view_host()(entityOrdinal), static_cast<IndexType>(expectedIndices.size()));
    TEST_EQUALITY(vertexOrdinals.view_host()(entityOrdinal), entity.vertexOrdinal());

    const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
    const ComponentCellID1D &incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
    const IndexType leftVertex = orthogonalTopo->cellVertexIndex(incidentCell, 0);
    const int expectedParity = (entity.vertexOrdinal() == leftVertex) ? -1 : 1;
    TEST_EQUALITY(data.skeletonEntitySideParity(TENSOR_TRIAL_SPACE, "tn", entityOrdinal), expectedParity);
    TEST_EQUALITY(data.skeletonEntityOutwardNormalSign(TENSOR_TRIAL_SPACE, "tn", entityOrdinal), expectedParity);
    TEST_EQUALITY(sideParities.view_host()(entityOrdinal), expectedParity);
    TEST_EQUALITY(outwardNormalSigns.view_host()(entityOrdinal), expectedParity);

    for (IndexType localOrdinal=0; localOrdinal<static_cast<IndexType>(expectedIndices.size()); localOrdinal++)
    {
      TEST_EQUALITY(data.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "tn", entityOrdinal, localOrdinal), expectedIndices[localOrdinal]);
      TEST_EQUALITY(localToGlobal.view_host()(entityOrdinal,localOrdinal), expectedIndices[localOrdinal]);
    }
  }
}

TEUCHOS_UNIT_TEST( TensorOperatorData, DeviceViewsExposeSkeletonTraceMapsAndNormalSigns )
{
  auto tensorMesh = makeTensorMeshForOperatorDataTests(true);
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);
  TensorCoefficientMap coefficientMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "tn");
  auto skeleton = tensorMesh->topology()->tensorSkeleton(1);

  IndexType expectedCountSum = 0;
  IndexType expectedIndexSum = 0;
  int expectedNormalSignSum = 0;
  for (IndexType entityOrdinal=0; entityOrdinal<skeleton.entityCount(); entityOrdinal++)
  {
    const auto &entity = skeleton.entity(entityOrdinal);
    auto expectedIndices = coefficientMap.indicesForSkeletonEntity(entity);
    expectedCountSum += expectedIndices.size();
    for (IndexType index : expectedIndices) expectedIndexSum += index;
    expectedNormalSignSum += data.skeletonEntityOutwardNormalSign(TENSOR_TRIAL_SPACE, "tn", entityOrdinal);
  }

  const auto &traceData = data.variableData(TENSOR_TRIAL_SPACE, "tn");
  auto localCounts = traceData.skeletonLocalDofCounts();
  auto localToGlobal = traceData.skeletonLocalToGlobalIndices();
  auto outwardNormalSigns = traceData.skeletonEntityOutwardNormalSigns();
  auto localCountsDevice = localCounts.view_device();
  auto localToGlobalDevice = localToGlobal.view_device();
  auto outwardNormalSignsDevice = outwardNormalSigns.view_device();

  IndexType actualCountSum = 0;
  IndexType actualIndexSum = 0;
  int actualNormalSignSum = 0;
  Kokkos::parallel_reduce("TensorOperatorDataDeviceSkeletonLocalDofCountSum", static_cast<int>(localCounts.extent(0)),
                          KOKKOS_LAMBDA(const int entityOrdinal, IndexType &sum) { sum += localCountsDevice(entityOrdinal); }, actualCountSum);
  Kokkos::parallel_reduce("TensorOperatorDataDeviceSkeletonLocalToGlobalIndexSum", static_cast<int>(localCounts.extent(0)),
                          KOKKOS_LAMBDA(const int entityOrdinal, IndexType &sum)
                          {
                            for (IndexType localOrdinal=0; localOrdinal<localCountsDevice(entityOrdinal); localOrdinal++)
                            {
                              sum += localToGlobalDevice(entityOrdinal, localOrdinal);
                            }
                          }, actualIndexSum);
  Kokkos::parallel_reduce("TensorOperatorDataDeviceSkeletonOutwardNormalSignSum", static_cast<int>(outwardNormalSigns.extent(0)),
                          KOKKOS_LAMBDA(const int entityOrdinal, int &sum) { sum += outwardNormalSignsDevice(entityOrdinal); }, actualNormalSignSum);

  TEST_EQUALITY(actualCountSum, expectedCountSum);
  TEST_EQUALITY(actualIndexSum, expectedIndexSum);
  TEST_EQUALITY(actualNormalSignSum, expectedNormalSignSum);
}

TEUCHOS_UNIT_TEST( TensorOperatorData, DirectionalTraceVariablesCompileToDistinctSkeletonFamilies )
{
  auto tensorMesh = makeDirectionalTraceTensorMeshForOperatorDataTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorOperatorData data(tensorMesh, dofLayout);

  const auto &tnXData = data.variableData(TENSOR_TRIAL_SPACE, "tn_x");
  const auto &tnTData = data.variableData(TENSOR_TRIAL_SPACE, "tn_t");
  TEST_ASSERT(tnXData.isTraceVariable());
  TEST_ASSERT(tnTData.isTraceVariable());
  TEST_EQUALITY(tnXData.traceComponentOrdinal(), 0);
  TEST_EQUALITY(tnTData.traceComponentOrdinal(), 1);
  auto xSkeleton = tensorMesh->topology()->tensorSkeleton(0);
  auto tSkeleton = tensorMesh->topology()->tensorSkeleton(1);
  TEST_EQUALITY(data.skeletonEntityCount(TENSOR_TRIAL_SPACE, "tn_x"), xSkeleton.entityCount());
  TEST_EQUALITY(data.skeletonEntityCount(TENSOR_TRIAL_SPACE, "tn_t"), tSkeleton.entityCount());

  auto expectedOutwardNormalSign = [tensorMesh](int componentOrdinal, IndexType vertexOrdinal) {
    auto componentTopo = tensorMesh->topology()->componentMeshTopology(componentOrdinal);
    const IndexType incidentBegin = componentTopo->vertexToCellOffsetHost(vertexOrdinal);
    const ComponentCellID1D &incidentCell = componentTopo->activeCells()[componentTopo->vertexToCellOrdinalHost(incidentBegin)];
    const IndexType leftVertex = componentTopo->cellVertexIndex(incidentCell, 0);
    return (vertexOrdinal == leftVertex) ? -1 : 1;
  };

  TensorCoefficientMap tnXMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "tn_x");
  TensorCoefficientMap tnTMap(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "tn_t");
  for (IndexType entityOrdinal=0; entityOrdinal<xSkeleton.entityCount(); entityOrdinal++)
  {
    const auto &entity = xSkeleton.entity(entityOrdinal);
    auto expectedIndices = tnXMap.indicesForSkeletonEntity(entity);
    TEST_EQUALITY(data.skeletonLocalDofCount(TENSOR_TRIAL_SPACE, "tn_x", entityOrdinal), static_cast<IndexType>(expectedIndices.size()));
    TEST_EQUALITY(data.skeletonEntityOutwardNormalSign(TENSOR_TRIAL_SPACE, "tn_x", entityOrdinal),
                  expectedOutwardNormalSign(0, entity.vertexOrdinal()));
    for (IndexType localOrdinal=0; localOrdinal<static_cast<IndexType>(expectedIndices.size()); localOrdinal++)
    {
      TEST_EQUALITY(data.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "tn_x", entityOrdinal, localOrdinal), expectedIndices[localOrdinal]);
    }
  }
  for (IndexType entityOrdinal=0; entityOrdinal<tSkeleton.entityCount(); entityOrdinal++)
  {
    const auto &entity = tSkeleton.entity(entityOrdinal);
    auto expectedIndices = tnTMap.indicesForSkeletonEntity(entity);
    TEST_EQUALITY(data.skeletonLocalDofCount(TENSOR_TRIAL_SPACE, "tn_t", entityOrdinal), static_cast<IndexType>(expectedIndices.size()));
    TEST_EQUALITY(data.skeletonEntityOutwardNormalSign(TENSOR_TRIAL_SPACE, "tn_t", entityOrdinal),
                  expectedOutwardNormalSign(1, entity.vertexOrdinal()));
    for (IndexType localOrdinal=0; localOrdinal<static_cast<IndexType>(expectedIndices.size()); localOrdinal++)
    {
      TEST_EQUALITY(data.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "tn_t", entityOrdinal, localOrdinal), expectedIndices[localOrdinal]);
    }
  }
}
}
