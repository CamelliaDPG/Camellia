#include "Teuchos_UnitTestHarness.hpp"

#include "TensorSolution.h"
#include "TensorMesh.h"
#include "Var.h"

#include <Tpetra_Core.hpp>

using namespace Camellia;

namespace
{
TensorVarPtr makeTensorSolutionFieldVar1C2V(const std::string &name)
{
  VarPtr x  = Teuchos::rcp(new Var(0, 0, name + "_x",  Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr ux = Teuchos::rcp(new Var(1, 0, name + "_ux", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr uz = Teuchos::rcp(new Var(2, 0, name + "_uz", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, std::vector<VarPtr>{x, ux, uz}));
}

TensorMeshPtr makeTensorSolutionTestMesh(int configurationElementCount = 2)
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
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeTensorSolutionFieldVar1C2V("f"), Teuchos::null, std::vector<int>{0,0,0})));
  tensorMesh->addTestDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeTensorSolutionFieldVar1C2V("w"), Teuchos::null, std::vector<int>{0,0,0})));
  return tensorMesh;
}

double tensorSolutionDeterministicCoefficient(IndexType globalIndex)
{
  return 0.125 + 0.25 * static_cast<double>(globalIndex + 1);
}
}

TEUCHOS_UNIT_TEST( TensorSolution, OwnsLongLivedCoefficientVectorForRegisteredTensorVariable )
{
  auto tensorMesh = makeTensorSolutionTestMesh();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorSolution solution(tensorMesh, dofLayout, Tpetra::getDefaultComm());
  solution.registerCoefficientVariable(TENSOR_TRIAL_SPACE, "f");

  TEST_ASSERT(solution.hasCoefficientVariable(TENSOR_TRIAL_SPACE, "f"));
  const auto &coefficientMap = solution.coefficientMap(TENSOR_TRIAL_SPACE, "f");
  auto ownedMap = solution.ownedMap(TENSOR_TRIAL_SPACE, "f");
  TEST_EQUALITY(ownedMap->getLocalNumElements(), coefficientMap.ownedIndicesForRank(solution.comm()->getRank(), solution.comm()->getSize()).size());
  TEST_EQUALITY(solution.ownedCoefficientVector(TENSOR_TRIAL_SPACE, "f")->getMap(), ownedMap);
}

TEUCHOS_UNIT_TEST( TensorSolution, VolumeImportUsesLongLivedCoefficientVectorValues )
{
  auto tensorMesh = makeTensorSolutionTestMesh(2);
  TensorSolution solution(tensorMesh, Teuchos::null, Tpetra::getDefaultComm());
  solution.registerCoefficientVariable(TENSOR_TRIAL_SPACE, "f");

  auto ownedMap = solution.ownedMap(TENSOR_TRIAL_SPACE, "f");
  std::vector<double> localValues(ownedMap->getLocalNumElements(), 0.0);
  for (size_t localOrdinal=0; localOrdinal<localValues.size(); localOrdinal++)
  {
    const IndexType gid = static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<TensorSolution::map_type::local_ordinal_type>(localOrdinal)));
    localValues[localOrdinal] = tensorSolutionDeterministicCoefficient(gid);
  }
  solution.setOwnedCoefficients(TENSOR_TRIAL_SPACE, "f", localValues);

  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  auto ownedValues = solution.ownedValuesForVolumeCell(TENSOR_TRIAL_SPACE, "f", cellTuple);
  const auto &coefficientMap = solution.coefficientMap(TENSOR_TRIAL_SPACE, "f");
  const auto expectedGhosts = coefficientMap.neededIndicesForVolumeCell(cellTuple, solution.comm()->getRank(), solution.comm()->getSize());
  TEST_EQUALITY(ownedValues.ghostCoefficientsByGlobalIndex.size(), expectedGhosts.size());
  for (IndexType gid : expectedGhosts)
  {
    TEST_ASSERT(ownedValues.ghostCoefficientsByGlobalIndex.find(gid) != ownedValues.ghostCoefficientsByGlobalIndex.end());
    TEST_FLOATING_EQUALITY(ownedValues.ghostCoefficientsByGlobalIndex[gid], tensorSolutionDeterministicCoefficient(gid), 1e-14);
  }
  TEST_ASSERT(ownedValues.hasDeviceMirrors());
}
