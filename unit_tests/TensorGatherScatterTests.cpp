#include "Teuchos_UnitTestHarness.hpp"

#include "TensorGatherScatter.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
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

TensorMeshPtr makeTensorMeshForGatherScatterTests()
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

TensorMeshPtr makeTensorMeshForTraceGatherScatterTests()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 3));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 2));

  auto trialVar = makeTraceTensorVar("uhat", 0);
  auto testVar = makeTraceTensorVar("vhat", 1);

  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,1}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{1,0}));

  tensorMesh->addTrialDiscretization(trialDisc);
  tensorMesh->addTestDiscretization(testDisc);
  return tensorMesh;
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, ConstructTrialAndTestGatherScatter )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();

  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");
  TensorGatherScatter testGS(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "v");

  TEST_EQUALITY(trialGS.role(), TENSOR_TRIAL_SPACE);
  TEST_EQUALITY(testGS.role(), TENSOR_TEST_SPACE);
  TEST_EQUALITY(trialGS.varName(), "u");
  TEST_EQUALITY(testGS.varName(), "v");
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, LocalDofCountMatchesSuppliedShape )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = dofLayout->factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);

  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");
  TEST_EQUALITY(trialGS.localDofCount(shape), shape.totalDofs());
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, ScatterLocalToGlobalUsesFlatOffsetForExactShape )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = dofLayout->factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");

  std::vector<double> local(trialGS.localDofCount(shape), 1.0);
  std::vector<double> global(trialGS.localDofCount(shape) + 3, 0.0);

  trialGS.scatterLocalToGlobal(local, shape, global, 3);

  for (IndexType i=0; i<3; i++)
  {
    TEST_EQUALITY(global[i], 0.0);
  }
  for (IndexType i=0; i<trialGS.localDofCount(shape); i++)
  {
    TEST_EQUALITY(global[3+i], 1.0);
  }
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, GatherGlobalToLocalUsesFlatOffsetForExactShape )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = dofLayout->factorSpaceShape("v", TENSOR_TEST_SPACE, cellTuple);
  TensorGatherScatter testGS(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "v");

  std::vector<double> global(testGS.localDofCount(shape) + 2, 0.0);
  for (IndexType i=0; i<testGS.localDofCount(shape); i++)
  {
    global[2+i] = static_cast<double>(i + 1);
  }

  std::vector<double> local;
  testGS.gatherGlobalToLocal(global, shape, local, 2);

  TEST_EQUALITY(local.size(), testGS.localDofCount(shape));
  for (IndexType i=0; i<testGS.localDofCount(shape); i++)
  {
    TEST_EQUALITY(local[i], static_cast<double>(i + 1));
  }
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, VariableAndCellOrderingsHaveSameLocalDofCountForExactShape )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto shape = dofLayout->factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);
  TensorGatherScatter variableGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u", TENSOR_VARIABLE_ORDERING);
  TensorGatherScatter cellGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u", TENSOR_CELL_ORDERING);

  TEST_EQUALITY(variableGS.localDofCount(shape), cellGS.localDofCount(shape));
  TEST_EQUALITY(variableGS.localDofCount(shape), shape.totalDofs());
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, TraceVariableLocalDofCountMatchesSkeletonShape )
{
  auto tensorMesh = makeTensorMeshForTraceGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto skeleton = tensorMesh->topology()->tensorSkeleton(0);
  auto context = dofLayout->skeletonContext("uhat", "vhat", skeleton.entity(0));
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "uhat");

  TEST_EQUALITY(trialGS.localDofCount(context.trialShape()), context.trialShape().totalDofs());
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, TraceVariableScatterAndGatherRemainFlatBufferCompatibleForSkeletonShape )
{
  auto tensorMesh = makeTensorMeshForTraceGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto skeleton = tensorMesh->topology()->tensorSkeleton(0);
  auto context = dofLayout->skeletonContext("uhat", "vhat", skeleton.entity(0));
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "uhat");

  std::vector<double> local(trialGS.localDofCount(context.trialShape()), 0.0);
  for (IndexType i=0; i<trialGS.localDofCount(context.trialShape()); i++)
  {
    local[i] = static_cast<double>(2 * i + 1);
  }

  std::vector<double> global(trialGS.localDofCount(context.trialShape()) + 4, 0.0);
  trialGS.scatterLocalToGlobal(local, context.trialShape(), global, 4);

  std::vector<double> gathered;
  trialGS.gatherGlobalToLocal(global, context.trialShape(), gathered, 4);

  TEST_EQUALITY(gathered.size(), local.size());
  for (IndexType i=0; i<trialGS.localDofCount(context.trialShape()); i++)
  {
    TEST_EQUALITY(gathered[i], local[i]);
  }
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, PNonuniformContextsNaturallyHaveDifferentExactLocalDofCounts )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto component0 = tensorMesh->topology()->componentMeshTopology(0);
  component0->setCellPolyOrder(ComponentCellID1D(0,1,1), component0->cellPolyOrder(ComponentCellID1D(0,1,1)) + 1);
  auto dofLayout = tensorMesh->createDofLayout();

  auto context0 = dofLayout->volumeContext("u", "v", tensorMesh->topology()->tensorCellTuple(0));
  auto context1 = dofLayout->volumeContext("u", "v", tensorMesh->topology()->tensorCellTuple(1));

  TEST_ASSERT(context1.trialShape().totalDofs() > context0.trialShape().totalDofs());
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, TrialAndTestCanHaveDifferentExactLocalDofCounts )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto trialShape = dofLayout->factorSpaceShape("u", TENSOR_TRIAL_SPACE, cellTuple);
  auto testShape = dofLayout->factorSpaceShape("v", TENSOR_TEST_SPACE, cellTuple);
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");
  TensorGatherScatter testGS(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "v");

  TEST_ASSERT(testGS.localDofCount(testShape) >= trialGS.localDofCount(trialShape));
}
}
