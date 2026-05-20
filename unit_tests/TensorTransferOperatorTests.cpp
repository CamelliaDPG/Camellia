#include "Teuchos_UnitTestHarness.hpp"

#include "TensorTransferOperator.h"
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

TensorMeshPtr makeTensorMeshForTransferTests(const std::vector<ActiveLeafSet1D> &leafSets0,
                                             const std::vector<ActiveLeafSet1D> &leafSets1,
                                             int defaultP)
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, leafSets0, false, defaultP));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, leafSets1, false, defaultP));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));

  auto trialVar = makeFieldTensorVar("u");
  auto testVar = trialVar->copyWithNewName("v");
  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,0}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{0,0}));

  tensorMesh->addTrialDiscretization(trialDisc);
  tensorMesh->addTestDiscretization(testDisc);
  return tensorMesh;
}

TEUCHOS_UNIT_TEST( TensorTransferOperator, ConstructTensorProductProlongation )
{
  auto coarseMesh = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{0,0}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto fineMesh   = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{1,0}, {1,1}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  TensorTransferOperator op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                            TENSOR_TRIAL_SPACE, "u", COMPONENT_PROLONGATION);

  TEST_EQUALITY(op.componentCount(), 2);
  TEST_ASSERT(op.rowCount() >= op.colCount());
}

TEUCHOS_UNIT_TEST( TensorTransferOperator, LowerIndexedComponentsVaryFastestInTensorComposition )
{
  auto coarseMesh = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{0,0}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto fineMesh   = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{1,0}, {1,1}}) }, { ActiveLeafSet1D({{1,0}, {1,1}}) }, 1);
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  TensorTransferOperator op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                            TENSOR_TRIAL_SPACE, "u", COMPONENT_PROLONGATION);

  std::vector<double> input(op.colCount(), 1.0);
  std::vector<double> output;
  op.apply(input, output);

  TEST_EQUALITY(output.size(), op.rowCount());
  bool someNonzero = false;
  for (double value : output)
  {
    if (value != 0.0) someNonzero = true;
  }
  TEST_ASSERT(someNonzero);
}

TEUCHOS_UNIT_TEST( TensorTransferOperator, RestrictionConstructionSupported )
{
  auto coarseMesh = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{0,0}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto fineMesh   = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{1,0}, {1,1}}) }, { ActiveLeafSet1D({{1,0}, {1,1}}) }, 1);
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  TensorTransferOperator op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                            TENSOR_TRIAL_SPACE, "u", COMPONENT_RESTRICTION);

  TEST_ASSERT(op.rowCount() > 0);
  TEST_ASSERT(op.colCount() > 0);
}

TEUCHOS_UNIT_TEST( TensorTransferOperator, TestRoleConstructionSupported )
{
  auto coarseMesh = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{0,0}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto fineMesh   = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{0,0}}) }, { ActiveLeafSet1D({{0,0}}) }, 2);
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  TensorTransferOperator op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                            TENSOR_TEST_SPACE, "v", COMPONENT_PROLONGATION);

  TEST_EQUALITY(op.role(), TENSOR_TEST_SPACE);
  TEST_EQUALITY(op.varName(), "v");
}

TEUCHOS_UNIT_TEST( TensorTransferOperator, ComponentFactorsMatchTensorRank )
{
  auto coarseMesh = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{0,0}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto fineMesh   = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{1,0}, {1,1}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  TensorTransferOperator op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                            TENSOR_TRIAL_SPACE, "u", COMPONENT_PROLONGATION);

  TEST_EQUALITY(op.componentCount(), coarseMesh->topology()->componentCount());
  for (int d=0; d<op.componentCount(); d++)
  {
    TEST_EQUALITY(op.componentTransfer(d)->componentOrdinal(), d);
    TEST_EQUALITY(op.componentTransfer(d)->transferType(), COMPONENT_PROLONGATION);
  }
}

TEUCHOS_UNIT_TEST( TensorTransferOperator, ApplyProducesExpectedSizedOutput )
{
  auto coarseMesh = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{0,0}}) }, { ActiveLeafSet1D({{0,0}}) }, 1);
  auto fineMesh   = makeTensorMeshForTransferTests({ ActiveLeafSet1D({{1,0}, {1,1}}) }, { ActiveLeafSet1D({{1,0}, {1,1}}) }, 1);
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  TensorTransferOperator op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                            TENSOR_TRIAL_SPACE, "u", COMPONENT_PROLONGATION);

  std::vector<double> input(op.colCount(), 1.0);
  std::vector<double> output;
  op.apply(input, output);

  TEST_EQUALITY(output.size(), op.rowCount());
}
}
