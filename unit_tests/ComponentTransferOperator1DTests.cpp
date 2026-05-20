#include "Teuchos_UnitTestHarness.hpp"

#include "ComponentTransferOperator1D.h"
#include "TensorDofLayout.h"
#include "TensorMesh.h"
#include "Var.h"

using namespace Camellia;

namespace
{
TensorVarPtr makeFieldTensorVar(const std::string &name)
{
  VarPtr vx = Teuchos::rcp(new Var(0, 0, name + "_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, std::vector<VarPtr>{vx}));
}

TensorMeshPtr makeSingleComponentTensorMesh(int defaultP, int enrichment, int componentDefaultP, const std::vector<ActiveLeafSet1D> &leafSets)
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, leafSets, false, componentDefaultP));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), defaultP, enrichment));

  auto trialVar = makeFieldTensorVar("u");
  auto testVar = trialVar->copyWithNewName("v");
  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{0}));

  tensorMesh->addTrialDiscretization(trialDisc);
  tensorMesh->addTestDiscretization(testDisc);
  return tensorMesh;
}

TEUCHOS_UNIT_TEST( ComponentTransferOperator1D, PProlongationHasGrowingCardinality )
{
  auto coarseMesh = makeSingleComponentTensorMesh(1, 0, 1, { ActiveLeafSet1D({{0,0}}) });
  auto fineMesh   = makeSingleComponentTensorMesh(2, 0, 1, { ActiveLeafSet1D({{0,0}}) });
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  ComponentTransferOperator1D op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                                 TENSOR_TRIAL_SPACE, "u", 0, COMPONENT_PROLONGATION);

  TEST_ASSERT(op.rowCount() >= op.colCount());
}

TEUCHOS_UNIT_TEST( ComponentTransferOperator1D, HProlongationReflectsParentChildStructure )
{
  auto coarseMesh = makeSingleComponentTensorMesh(1, 0, 1, { ActiveLeafSet1D({{0,0}}) });
  auto fineMesh   = makeSingleComponentTensorMesh(1, 0, 1, { ActiveLeafSet1D({{1,0}, {1,1}}) });
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  ComponentTransferOperator1D op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                                 TENSOR_TRIAL_SPACE, "u", 0, COMPONENT_PROLONGATION);

  std::vector<double> coarseInput(op.colCount(), 1.0);
  std::vector<double> fineOutput;
  op.apply(coarseInput, fineOutput);

  TEST_EQUALITY(fineOutput.size(), op.rowCount());
  bool someNonzero = false;
  for (double value : fineOutput)
  {
    if (value != 0.0) someNonzero = true;
  }
  TEST_ASSERT(someNonzero);
}

TEUCHOS_UNIT_TEST( ComponentTransferOperator1D, HRestrictionUsesAveragingSemantics )
{
  auto coarseMesh = makeSingleComponentTensorMesh(1, 0, 1, { ActiveLeafSet1D({{0,0}}) });
  auto fineMesh   = makeSingleComponentTensorMesh(1, 0, 1, { ActiveLeafSet1D({{1,0}, {1,1}}) });
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  ComponentTransferOperator1D op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                                 TENSOR_TRIAL_SPACE, "u", 0, COMPONENT_RESTRICTION);

  std::vector<double> fineInput(op.colCount(), 2.0);
  std::vector<double> coarseOutput;
  op.apply(fineInput, coarseOutput);

  TEST_EQUALITY(coarseOutput.size(), op.rowCount());
  bool someNonzero = false;
  for (double value : coarseOutput)
  {
    if (value != 0.0) someNonzero = true;
  }
  TEST_ASSERT(someNonzero);
}

TEUCHOS_UNIT_TEST( ComponentTransferOperator1D, TestRoleConstructionSupported )
{
  auto coarseMesh = makeSingleComponentTensorMesh(1, 1, 1, { ActiveLeafSet1D({{0,0}}) });
  auto fineMesh   = makeSingleComponentTensorMesh(2, 1, 1, { ActiveLeafSet1D({{0,0}}) });
  auto coarseLayout = coarseMesh->createDofLayout();
  auto fineLayout   = fineMesh->createDofLayout();

  ComponentTransferOperator1D op(coarseMesh, fineMesh, coarseLayout, fineLayout,
                                 TENSOR_TEST_SPACE, "v", 0, COMPONENT_PROLONGATION);

  TEST_EQUALITY(op.role(), TENSOR_TEST_SPACE);
  TEST_EQUALITY(op.varName(), "v");
}
}
