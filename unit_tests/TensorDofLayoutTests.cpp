#include "Teuchos_UnitTestHarness.hpp"

#include "TensorDofLayout.h"
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

TensorVarPtr makeTraceTensorVar(const std::string &name, int traceComponentOrdinal)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  VarPtr vt = Teuchos::rcp(new Var(1, 0, name + "_t", Camellia::OP_VALUE, HGRAD, TRACE, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_TRACE, std::vector<VarPtr>{vx, vt}, traceComponentOrdinal));
}

TensorMeshPtr makeTensorMeshForLayoutTests()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 3));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 2));

  auto trialVar = makeFieldTensorVar("u");
  auto testVar = trialVar->copyWithNewName("v");

  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,1}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{1,0}));

  tensorMesh->addTrialDiscretization(trialDisc);
  tensorMesh->addTestDiscretization(testDisc);
  return tensorMesh;
}

TensorMeshPtr makeTensorMeshForTraceLayoutTests()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 3));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 2));

  auto trialTraceVar = makeTraceTensorVar("uhat", 0);
  auto testTraceVar = makeTraceTensorVar("vhat", 1);

  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialTraceVar, Teuchos::null, std::vector<int>{0,1}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testTraceVar, Teuchos::null, std::vector<int>{1,0}));

  tensorMesh->addTrialDiscretization(trialDisc);
  tensorMesh->addTestDiscretization(testDisc);
  return tensorMesh;
}

TEUCHOS_UNIT_TEST( TensorDofLayout, ConstructTrialAndTestLayouts )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);

  TEST_ASSERT(layout.hasLayout("u", TENSOR_TRIAL_SPACE));
  TEST_ASSERT(layout.hasLayout("v", TENSOR_TEST_SPACE));
  TEST_ASSERT(!layout.hasLayout("u", TENSOR_TEST_SPACE));
  TEST_ASSERT(!layout.hasLayout("v", TENSOR_TRIAL_SPACE));
}

TEUCHOS_UNIT_TEST( TensorDofLayout, TrialAndTestLayoutsHaveExpectedComponentCount )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);

  const auto &uLayout = layout.layout("u", TENSOR_TRIAL_SPACE);
  const auto &vLayout = layout.layout("v", TENSOR_TEST_SPACE);

  TEST_EQUALITY(uLayout.componentCount(), 2);
  TEST_EQUALITY(vLayout.componentCount(), 2);
}

TEUCHOS_UNIT_TEST( TensorDofLayout, LowerIndexedComponentsVaryFastestInVariableOrderingForSuppliedShape )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = layout.factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);
  const auto &uLayout = layout.layout("u", TENSOR_TRIAL_SPACE);

  TEST_EQUALITY(shape.componentStride(0), 1);
  TEST_EQUALITY(shape.componentStride(1), shape.componentCardinality(0));

  TEST_EQUALITY(layout.localDofOrdinal(uLayout, shape, {0,0}), 0);
  TEST_EQUALITY(layout.localDofOrdinal(uLayout, shape, {1,0}), 1);
  TEST_EQUALITY(layout.localDofOrdinal(uLayout, shape, {0,1}), shape.componentCardinality(0));
}

TEUCHOS_UNIT_TEST( TensorDofLayout, TestEnrichmentAffectsTestShapeCardinality )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);

  auto uShape = layout.factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);
  auto vShape = layout.factorSpaceShape("v", TENSOR_TEST_SPACE, cellTuple);

  TEST_ASSERT(vShape.componentCardinality(0) >= uShape.componentCardinality(0));
  TEST_ASSERT(vShape.componentCardinality(1) >= uShape.componentCardinality(1));
}

TEUCHOS_UNIT_TEST( TensorDofLayout, VariableAndCellOrderingsHaveSameLocalDofCountForExactShape )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = layout.factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);

  TEST_EQUALITY(layout.localDofCount(shape, TENSOR_VARIABLE_ORDERING), shape.totalDofs());
  TEST_EQUALITY(layout.localDofCount(shape, TENSOR_CELL_ORDERING), shape.totalDofs());
}

TEUCHOS_UNIT_TEST( TensorDofLayout, VariableAndCellOrderingsAreInvertibleForExactShape )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = layout.factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);
  const auto &uLayout = layout.layout("u", TENSOR_TRIAL_SPACE);

  std::vector<int> tupleVariable, tupleCell;
  for (IndexType ordinal=0; ordinal<shape.totalDofs(); ordinal++)
  {
    layout.decodeLocalDofOrdinal(uLayout, shape, TENSOR_VARIABLE_ORDERING, ordinal, tupleVariable);
    layout.decodeLocalDofOrdinal(uLayout, shape, TENSOR_CELL_ORDERING, ordinal, tupleCell);

    TEST_EQUALITY(layout.localDofOrdinal(uLayout, shape, TENSOR_VARIABLE_ORDERING, tupleVariable), ordinal);
    TEST_EQUALITY(layout.localDofOrdinal(uLayout, shape, TENSOR_CELL_ORDERING, tupleCell), ordinal);
  }
}

TEUCHOS_UNIT_TEST( TensorDofLayout, VolumeContextUsesExactCellLocalShapes )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  auto component0 = tensorMesh->topology()->componentMeshTopology(0);
  component0->setCellPolyOrder(ComponentCellID1D(0,1,1), component0->cellPolyOrder(ComponentCellID1D(0,1,1)) + 1);

  TensorDofLayout layout(tensorMesh);
  auto cellTuple0 = tensorMesh->topology()->tensorCellTuple(0);
  auto cellTuple1 = tensorMesh->topology()->tensorCellTuple(1);

  auto context0 = layout.volumeContext("u", "v", cellTuple0);
  auto context1 = layout.volumeContext("u", "v", cellTuple1);

  TEST_ASSERT(context1.trialShape().componentCardinality(0) > context0.trialShape().componentCardinality(0));
  TEST_ASSERT(context1.testShape().componentCardinality(0) >= context0.testShape().componentCardinality(0));
}

TEUCHOS_UNIT_TEST( TensorDofLayout, TraceLayoutMarksSupportKindsAndTraceComponent )
{
  auto tensorMesh = makeTensorMeshForTraceLayoutTests();
  TensorDofLayout layout(tensorMesh);

  const auto &uhatLayout = layout.layout("uhat", TENSOR_TRIAL_SPACE);
  TEST_ASSERT(uhatLayout.hasTraceComponentOrdinal());
  TEST_EQUALITY(uhatLayout.traceComponentOrdinal(), 0);
  TEST_ASSERT(uhatLayout.isTraceSupportComponent(0));
  TEST_ASSERT(!uhatLayout.isTraceSupportComponent(1));
  TEST_EQUALITY(uhatLayout.componentSupportKind(0), TENSOR_TRACE_SKELETON_SUPPORT);
  TEST_EQUALITY(uhatLayout.componentSupportKind(1), TENSOR_VOLUME_BASIS_SUPPORT);
}

TEUCHOS_UNIT_TEST( TensorDofLayout, SkeletonContextUsesLocalSupportExtentInTraceComponent )
{
  auto tensorMesh = makeTensorMeshForTraceLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto skeleton = tensorMesh->topology()->tensorSkeleton(0);
  auto context = layout.skeletonContext("uhat", "vhat", skeleton.entity(0));

  TEST_EQUALITY(context.trialShape().componentCardinality(0), 1);
  TEST_ASSERT(context.trialShape().componentCardinality(1) > 0);
}

TEUCHOS_UNIT_TEST( TensorDofLayout, TraceTestSkeletonContextUsesLocalSupportExtentInItsOwnTraceComponent )
{
  auto tensorMesh = makeTensorMeshForTraceLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto skeleton = tensorMesh->topology()->tensorSkeleton(1);
  auto context = layout.skeletonContext("uhat", "vhat", skeleton.entity(0));

  TEST_EQUALITY(context.testShape().componentCardinality(1), 1);
}

TEUCHOS_UNIT_TEST( TensorDofLayout, TraceVariableAndCellOrderingsAreInvertibleForSkeletonShape )
{
  auto tensorMesh = makeTensorMeshForTraceLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto skeleton = tensorMesh->topology()->tensorSkeleton(0);
  auto context = layout.skeletonContext("uhat", "vhat", skeleton.entity(0));
  const auto &uhatLayout = layout.layout("uhat", TENSOR_TRIAL_SPACE);

  std::vector<int> tupleVariable, tupleCell;
  for (IndexType ordinal=0; ordinal<context.trialShape().totalDofs(); ordinal++)
  {
    layout.decodeLocalDofOrdinal(uhatLayout, context.trialShape(), TENSOR_VARIABLE_ORDERING, ordinal, tupleVariable);
    layout.decodeLocalDofOrdinal(uhatLayout, context.trialShape(), TENSOR_CELL_ORDERING, ordinal, tupleCell);

    TEST_EQUALITY(layout.localDofOrdinal(uhatLayout, context.trialShape(), TENSOR_VARIABLE_ORDERING, tupleVariable), ordinal);
    TEST_EQUALITY(layout.localDofOrdinal(uhatLayout, context.trialShape(), TENSOR_CELL_ORDERING, tupleCell), ordinal);
  }
}

TEUCHOS_UNIT_TEST( TensorDofLayout, RejectWrongNumberOfComponentBasisOrdinals )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = layout.factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);
  const auto &uLayout = layout.layout("u", TENSOR_TRIAL_SPACE);

  TEST_THROW(layout.localDofOrdinal(uLayout, shape, {0}), std::invalid_argument);
  TEST_THROW(layout.localDofOrdinal(uLayout, shape, TENSOR_CELL_ORDERING, {0}), std::invalid_argument);
}
}
