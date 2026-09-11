#include "Teuchos_UnitTestHarness.hpp"

#include "TensorCoefficientMap.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "TensorOperatorData.h"
#include "Var.h"

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

  TEST_FLOATING_EQUALITY(widths.h_view(0), 1.0, 1e-15);
  TEST_FLOATING_EQUALITY(widths.h_view(1), 0.5, 1e-15);
  TEST_FLOATING_EQUALITY(widths.h_view(2), 0.5, 1e-15);
  TEST_FLOATING_EQUALITY(jacobians.h_view(1), 0.25, 1e-15);
  TEST_EQUALITY(polyOrders.h_view(0), 3);
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
    TEST_EQUALITY(localCounts.h_view(cellOrdinal), static_cast<IndexType>(expectedIndices.size()));
    for (IndexType localOrdinal=0; localOrdinal<static_cast<IndexType>(expectedIndices.size()); localOrdinal++)
    {
      TEST_EQUALITY(data.volumeLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "u", cellOrdinal, localOrdinal), expectedIndices[localOrdinal]);
      TEST_EQUALITY(localToGlobal.h_view(cellOrdinal,localOrdinal), expectedIndices[localOrdinal]);
    }
  }
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
  localCounts.sync_host();
  localToGlobal.sync_host();
  vertexOrdinals.sync_host();
  sideParities.sync_host();

  auto orthogonalTopo = tensorMesh->topology()->componentMeshTopology(1);
  for (IndexType entityOrdinal=0; entityOrdinal<skeleton.entityCount(); entityOrdinal++)
  {
    const auto &entity = skeleton.entity(entityOrdinal);
    auto expectedIndices = coefficientMap.indicesForSkeletonEntity(entity);
    TEST_EQUALITY(data.skeletonLocalDofCount(TENSOR_TRIAL_SPACE, "tn", entityOrdinal), static_cast<IndexType>(expectedIndices.size()));
    TEST_EQUALITY(localCounts.h_view(entityOrdinal), static_cast<IndexType>(expectedIndices.size()));
    TEST_EQUALITY(vertexOrdinals.h_view(entityOrdinal), entity.vertexOrdinal());

    const IndexType incidentBegin = orthogonalTopo->vertexToCellOffsetHost(entity.vertexOrdinal());
    const ComponentCellID1D &incidentCell = orthogonalTopo->activeCells()[orthogonalTopo->vertexToCellOrdinalHost(incidentBegin)];
    const IndexType leftVertex = orthogonalTopo->cellVertexIndex(incidentCell, 0);
    const int expectedParity = (entity.vertexOrdinal() == leftVertex) ? -1 : 1;
    TEST_EQUALITY(data.skeletonEntitySideParity(TENSOR_TRIAL_SPACE, "tn", entityOrdinal), expectedParity);
    TEST_EQUALITY(sideParities.h_view(entityOrdinal), expectedParity);

    for (IndexType localOrdinal=0; localOrdinal<static_cast<IndexType>(expectedIndices.size()); localOrdinal++)
    {
      TEST_EQUALITY(data.skeletonLocalToGlobalIndex(TENSOR_TRIAL_SPACE, "tn", entityOrdinal, localOrdinal), expectedIndices[localOrdinal]);
      TEST_EQUALITY(localToGlobal.h_view(entityOrdinal,localOrdinal), expectedIndices[localOrdinal]);
    }
  }
}
}
