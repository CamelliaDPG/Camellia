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

TEUCHOS_UNIT_TEST( TensorGatherScatter, LocalDofCountMatchesLayout )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();

  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");
  TEST_EQUALITY(trialGS.localDofCount(), dofLayout->layout("u", TENSOR_TRIAL_SPACE).totalDofs());
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, ScatterLocalToGlobalUsesFlatOffset )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");

  std::vector<double> local(trialGS.localDofCount(), 1.0);
  std::vector<double> global(trialGS.localDofCount() + 3, 0.0);

  trialGS.scatterLocalToGlobal(local, global, 3);

  for (IndexType i=0; i<3; i++)
  {
    TEST_EQUALITY(global[i], 0.0);
  }
  for (IndexType i=0; i<trialGS.localDofCount(); i++)
  {
    TEST_EQUALITY(global[3+i], 1.0);
  }
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, GatherGlobalToLocalUsesFlatOffset )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorGatherScatter testGS(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "v");

  std::vector<double> global(testGS.localDofCount() + 2, 0.0);
  for (IndexType i=0; i<testGS.localDofCount(); i++)
  {
    global[2+i] = static_cast<double>(i + 1);
  }

  std::vector<double> local;
  testGS.gatherGlobalToLocal(global, local, 2);

  TEST_EQUALITY(local.size(), testGS.localDofCount());
  for (IndexType i=0; i<testGS.localDofCount(); i++)
  {
    TEST_EQUALITY(local[i], static_cast<double>(i + 1));
  }
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, VariableAndCellOrderingsHaveSameLocalDofCount )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorGatherScatter variableGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u", TENSOR_VARIABLE_ORDERING);
  TensorGatherScatter cellGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u", TENSOR_CELL_ORDERING);

  TEST_EQUALITY(variableGS.localDofCount(), cellGS.localDofCount());
  TEST_EQUALITY(variableGS.localDofCount(), dofLayout->layout("u", TENSOR_TRIAL_SPACE).totalDofs());
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, TraceVariableLocalDofCountMatchesTraceLayout )
{
  auto tensorMesh = makeTensorMeshForTraceGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "uhat");
  TensorGatherScatter testGS(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "vhat");

  TEST_EQUALITY(trialGS.localDofCount(), dofLayout->layout("uhat", TENSOR_TRIAL_SPACE).totalDofs());
  TEST_EQUALITY(testGS.localDofCount(), dofLayout->layout("vhat", TENSOR_TEST_SPACE).totalDofs());
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, TraceVariableScatterAndGatherRemainFlatBufferCompatible )
{
  auto tensorMesh = makeTensorMeshForTraceGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "uhat");

  std::vector<double> local(trialGS.localDofCount(), 0.0);
  for (IndexType i=0; i<trialGS.localDofCount(); i++)
  {
    local[i] = static_cast<double>(2 * i + 1);
  }

  std::vector<double> global(trialGS.localDofCount() + 4, 0.0);
  trialGS.scatterLocalToGlobal(local, global, 4);

  std::vector<double> gathered;
  trialGS.gatherGlobalToLocal(global, gathered, 4);

  TEST_EQUALITY(gathered.size(), local.size());
  for (IndexType i=0; i<trialGS.localDofCount(); i++)
  {
    TEST_EQUALITY(gathered[i], local[i]);
  }
}

TEUCHOS_UNIT_TEST( TensorGatherScatter, TrialAndTestCanHaveDifferentLocalDofCounts )
{
  auto tensorMesh = makeTensorMeshForGatherScatterTests();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorGatherScatter trialGS(tensorMesh, dofLayout, TENSOR_TRIAL_SPACE, "u");
  TensorGatherScatter testGS(tensorMesh, dofLayout, TENSOR_TEST_SPACE, "v");

  TEST_ASSERT(testGS.localDofCount() >= trialGS.localDofCount());
}
}
