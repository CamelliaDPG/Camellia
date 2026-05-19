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
  TEST_ASSERT(uLayout.totalDofs() > 0);
  TEST_ASSERT(vLayout.totalDofs() > 0);
}

TEUCHOS_UNIT_TEST( TensorDofLayout, LowerIndexedComponentsVaryFastestInLocalOrdinal )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);

  const auto &uLayout = layout.layout("u", TENSOR_TRIAL_SPACE);
  TEST_EQUALITY(uLayout.componentStride(0), 1);
  TEST_EQUALITY(uLayout.componentStride(1), uLayout.componentCardinality(0));

  TEST_EQUALITY(layout.localDofOrdinal("u", TENSOR_TRIAL_SPACE, {0,0}), 0);
  TEST_EQUALITY(layout.localDofOrdinal("u", TENSOR_TRIAL_SPACE, {1,0}), 1);
  TEST_EQUALITY(layout.localDofOrdinal("u", TENSOR_TRIAL_SPACE, {0,1}), uLayout.componentCardinality(0));
}

TEUCHOS_UNIT_TEST( TensorDofLayout, TestEnrichmentAffectsTestLayoutCardinality )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);

  const auto &uLayout = layout.layout("u", TENSOR_TRIAL_SPACE);
  const auto &vLayout = layout.layout("v", TENSOR_TEST_SPACE);

  TEST_ASSERT(vLayout.componentCardinality(0) >= uLayout.componentCardinality(0));
  TEST_ASSERT(vLayout.componentCardinality(1) >= uLayout.componentCardinality(1));
}

TEUCHOS_UNIT_TEST( TensorDofLayout, RejectWrongNumberOfComponentBasisOrdinals )
{
  auto tensorMesh = makeTensorMeshForLayoutTests();
  TensorDofLayout layout(tensorMesh);

  TEST_THROW(layout.localDofOrdinal("u", TENSOR_TRIAL_SPACE, {0}), std::invalid_argument);
}
}
