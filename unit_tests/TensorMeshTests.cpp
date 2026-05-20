#include "Teuchos_UnitTestHarness.hpp"

#include "TensorMesh.h"
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

Teuchos::RCP<TensorMeshTopology> makeSimpleTensorTopology()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 3));
  return Teuchos::rcp(new TensorMeshTopology({component0, component1}));
}

TEUCHOS_UNIT_TEST( TensorMesh, ConstructFromTopologyAndDefaults )
{
  auto topology = makeSimpleTensorTopology();
  auto factory = TensorVarFactory::factory();
  auto basisFamily = BasisFamily::hierarchicalBasisFamily();

  TensorMesh tensorMesh(topology, factory, basisFamily, 1, 2);

  TEST_EQUALITY(tensorMesh.topology(), topology);
  TEST_EQUALITY(tensorMesh.tensorVarFactory(), factory);
  TEST_EQUALITY(tensorMesh.defaultBasisFamily(), basisFamily);
  TEST_EQUALITY(tensorMesh.defaultPolyOrder(), 1);
  TEST_EQUALITY(tensorMesh.testSpaceEnrichment(), 2);
}

TEUCHOS_UNIT_TEST( TensorMesh, AddTrialAndTestDiscretizations )
{
  auto topology = makeSimpleTensorTopology();
  auto factory = TensorVarFactory::factory();
  auto basisFamily = BasisFamily::hierarchicalBasisFamily();
  TensorMesh tensorMesh(topology, factory, basisFamily, 1, 1);

  auto trialVar = makeFieldTensorVar("u");
  auto testVar = trialVar->copyWithNewName("v");

  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,1}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{1,0}));

  tensorMesh.addTrialDiscretization(trialDisc);
  tensorMesh.addTestDiscretization(testDisc);

  TEST_ASSERT(tensorMesh.hasTrialVar("u"));
  TEST_ASSERT(tensorMesh.hasTestVar("v"));
  TEST_EQUALITY(tensorMesh.trialDiscretization("u"), trialDisc);
  TEST_EQUALITY(tensorMesh.testDiscretization("v"), testDisc);
}

TEUCHOS_UNIT_TEST( TensorMesh, EffectiveBasisFamily_DefaultAndOverride )
{
  auto topology = makeSimpleTensorTopology();
  auto factory = TensorVarFactory::factory();
  auto defaultBasis = BasisFamily::hierarchicalBasisFamily();
  auto overrideBasis = BasisFamily::nodalBasisFamily();
  TensorMesh tensorMesh(topology, factory, defaultBasis, 1, 0);

  auto trialVar = makeFieldTensorVar("u");
  auto discDefault = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,0}));
  auto discOverride = Teuchos::rcp(new TensorVarDiscretization(trialVar->copyWithNewName("w"), overrideBasis, std::vector<int>{0,0}));

  TEST_EQUALITY(tensorMesh.effectiveBasisFamily(discDefault), defaultBasis);
  TEST_EQUALITY(tensorMesh.effectiveBasisFamily(discOverride), overrideBasis);
}

TEUCHOS_UNIT_TEST( TensorMesh, EffectivePolyOrderCombinesDefaultsDeltasAndTopologyP )
{
  auto topology = makeSimpleTensorTopology();
  auto factory = TensorVarFactory::factory();
  auto basisFamily = BasisFamily::hierarchicalBasisFamily();
  TensorMesh tensorMesh(topology, factory, basisFamily, 1, 2);

  auto trialVar = makeFieldTensorVar("u");
  auto testVar = trialVar->copyWithNewName("v");
  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,1}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{1,0}));

  ComponentCellID1D cellX(0,1,0);
  ComponentCellID1D cellT(0,0,0);

  TEST_EQUALITY(tensorMesh.effectivePolyOrder(trialDisc, cellX, 0, false), 2 + 1 + 0);
  TEST_EQUALITY(tensorMesh.effectivePolyOrder(trialDisc, cellT, 1, false), 3 + 1 + 1);
  TEST_EQUALITY(tensorMesh.effectivePolyOrder(testDisc, cellX, 0, true), 2 + 1 + 1 + 2);
  TEST_EQUALITY(tensorMesh.effectivePolyOrder(testDisc, cellT, 1, true), 3 + 1 + 0 + 2);
}

TEUCHOS_UNIT_TEST( TensorMesh, RejectRankMismatchBetweenDiscretizationAndTopology )
{
  auto topology = makeSimpleTensorTopology();
  auto factory = TensorVarFactory::factory();
  auto basisFamily = BasisFamily::hierarchicalBasisFamily();
  TensorMesh tensorMesh(topology, factory, basisFamily, 1, 0);

  VarPtr vx = Teuchos::rcp(new Var(0, 0, "u_x", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  auto badTensorVar = Teuchos::rcp(new TensorVar("u", TENSOR_FIELD, std::vector<VarPtr>{vx}));
  auto badDisc = Teuchos::rcp(new TensorVarDiscretization(badTensorVar, Teuchos::null, std::vector<int>{0}));

  TEST_THROW(tensorMesh.addTrialDiscretization(badDisc), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMesh, RejectDuplicateTrialOrTestName )
{
  auto topology = makeSimpleTensorTopology();
  auto factory = TensorVarFactory::factory();
  auto basisFamily = BasisFamily::hierarchicalBasisFamily();
  TensorMesh tensorMesh(topology, factory, basisFamily, 1, 0);

  auto trialVar0 = makeFieldTensorVar("u");
  auto trialVar1 = makeFieldTensorVar("u");
  auto disc0 = Teuchos::rcp(new TensorVarDiscretization(trialVar0, Teuchos::null, std::vector<int>{0,0}));
  auto disc1 = Teuchos::rcp(new TensorVarDiscretization(trialVar1, Teuchos::null, std::vector<int>{0,0}));

  tensorMesh.addTrialDiscretization(disc0);
  TEST_THROW(tensorMesh.addTrialDiscretization(disc1), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMesh, CreateDofLayout )
{
  auto topology = makeSimpleTensorTopology();
  auto factory = TensorVarFactory::factory();
  auto basisFamily = BasisFamily::hierarchicalBasisFamily();
  TensorMesh tensorMesh(topology, factory, basisFamily, 1, 1);

  auto trialVar = makeFieldTensorVar("u");
  auto testVar = trialVar->copyWithNewName("v");
  auto trialDisc = Teuchos::rcp(new TensorVarDiscretization(trialVar, Teuchos::null, std::vector<int>{0,0}));
  auto testDisc = Teuchos::rcp(new TensorVarDiscretization(testVar, Teuchos::null, std::vector<int>{0,0}));

  tensorMesh.addTrialDiscretization(trialDisc);
  tensorMesh.addTestDiscretization(testDisc);

  auto dofLayout = tensorMesh.createDofLayout();
  TEST_ASSERT(dofLayout != Teuchos::null);
  TEST_EQUALITY(dofLayout->tensorMesh().get(), &tensorMesh);
}
}
