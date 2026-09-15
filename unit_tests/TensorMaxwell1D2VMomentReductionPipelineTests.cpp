#include "Teuchos_UnitTestHarness.hpp"

#include "TensorBasisCache.h"
#include "TensorMaxwell1D2VMomentReductionPipeline.h"
#include "TensorSolution.h"
#include "TensorMesh.h"
#include "Var.h"

#include <Tpetra_Core.hpp>

#include <cmath>

using namespace Camellia;

namespace
{
TensorVarPtr makePipelineFieldTensorVar1C2V(const std::string &name)
{
  VarPtr x  = Teuchos::rcp(new Var(0, 0, name + "_x",  Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr ux = Teuchos::rcp(new Var(1, 0, name + "_ux", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  VarPtr uz = Teuchos::rcp(new Var(2, 0, name + "_uz", Camellia::OP_VALUE, L2, FIELD, Teuchos::null, true, UNKNOWN_FS));
  return Teuchos::rcp(new TensorVar(name, TENSOR_FIELD, std::vector<VarPtr>{x, ux, uz}));
}

TensorMeshPtr makePipelineTensorMesh1C2V()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-2.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1, component2}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 0));
  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makePipelineFieldTensorVar1C2V("f"), Teuchos::null, std::vector<int>{0,0,0})));
  return tensorMesh;
}

double pipelineCoefficientValue(IndexType globalIndex, double shift)
{
  return shift + 0.25 * static_cast<double>(globalIndex + 1);
}

void fillTensorSolutionCoefficients(TensorSolution &solution, double shift)
{
  auto ownedMap = solution.ownedMap(TENSOR_TRIAL_SPACE, "f");
  std::vector<double> localValues(ownedMap->getLocalNumElements(), 0.0);
  for (size_t localOrdinal=0; localOrdinal<localValues.size(); localOrdinal++)
  {
    const IndexType gid = static_cast<IndexType>(ownedMap->getGlobalElement(static_cast<TensorSolution::map_type::local_ordinal_type>(localOrdinal)));
    localValues[localOrdinal] = pipelineCoefficientValue(gid, shift);
  }
  solution.setOwnedCoefficients(TENSOR_TRIAL_SPACE, "f", localValues);
}

TEUCHOS_UNIT_TEST( TensorMaxwell1D2VMomentReductionPipeline, ReducesSpeciesAndBuildsMaxwellRHSData )
{
  using Pipeline = TensorMaxwell1D2VMomentReductionPipeline;

  const int numCells = 1;
  const int pointsPerCell = 2;
  const double eps0 = 2.0;
  std::vector<double> charges = {3.0, -1.0};
  std::vector<double> n0 = {0.5, 1.5};

  Pipeline::HostMatrix velocityPoints("velocityPoints", 2, 2);
  velocityPoints(0,0) = 1.0; velocityPoints(0,1) = 2.0;
  velocityPoints(1,0) = 3.0; velocityPoints(1,1) = 4.0;

  Pipeline::HostVector velocityWeights("velocityWeights", 2);
  velocityWeights(0) = 0.25;
  velocityWeights(1) = 0.75;

  Pipeline::HostMatrix species0("species0", 2, 2);
  species0(0,0) = 2.0; species0(0,1) = 4.0;
  species0(1,0) = 6.0; species0(1,1) = 8.0;

  Pipeline::HostMatrix species1("species1", 2, 2);
  species1(0,0) = 1.0; species1(0,1) = 3.0;
  species1(1,0) = 5.0; species1(1,1) = 7.0;

  auto data = Pipeline::buildRHSMomentData(numCells,
                                           pointsPerCell,
                                           charges,
                                           eps0,
                                           {species0, species1},
                                           velocityPoints,
                                           velocityWeights,
                                           n0);

  for (int point=0; point<pointsPerCell; point++)
  {
    const double density0 = species0(point,0) * velocityWeights(0)
                          + species0(point,1) * velocityWeights(1);
    const double density1 = species1(point,0) * velocityWeights(0)
                          + species1(point,1) * velocityWeights(1);
    const double xMoment0 = species0(point,0) * velocityWeights(0) * velocityPoints(0,0)
                          + species0(point,1) * velocityWeights(1) * velocityPoints(1,0);
    const double xMoment1 = species1(point,0) * velocityWeights(0) * velocityPoints(0,0)
                          + species1(point,1) * velocityWeights(1) * velocityPoints(1,0);
    const double zMoment0 = species0(point,0) * velocityWeights(0) * velocityPoints(0,1)
                          + species0(point,1) * velocityWeights(1) * velocityPoints(1,1);
    const double zMoment1 = species1(point,0) * velocityWeights(0) * velocityPoints(0,1)
                          + species1(point,1) * velocityWeights(1) * velocityPoints(1,1);

    const double expectedRho = charges[0] / eps0 * (density0 - n0[point])
                             + charges[1] / eps0 * (density1 - n0[point]);
    const double expectedJx = charges[0] * xMoment0 + charges[1] * xMoment1;
    const double expectedJz = charges[0] * zMoment0 + charges[1] * zMoment1;

    TEST_FLOATING_EQUALITY(data.rhoRHS(0, point), expectedRho, 1e-14);
    TEST_FLOATING_EQUALITY(data.Jx(0, point), expectedJx, 1e-14);
    TEST_FLOATING_EQUALITY(data.Jz(0, point), expectedJz, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorMaxwell1D2VMomentReductionPipeline, ReducesRelativisticCurrentMomentsWhenRequested )
{
  using Pipeline = TensorMaxwell1D2VMomentReductionPipeline;

  const int numCells = 1;
  const int pointsPerCell = 1;
  const double eps0 = 4.0;
  const double c = 2.0;
  const double q = -2.0;
  std::vector<double> charges = {q};
  std::vector<double> n0 = {0.25};

  Pipeline::HostMatrix velocityPoints("velocityPoints", 2, 2);
  velocityPoints(0,0) = 1.0; velocityPoints(0,1) = 0.0;
  velocityPoints(1,0) = 0.0; velocityPoints(1,1) = 2.0;

  Pipeline::HostVector velocityWeights("velocityWeights", 2);
  velocityWeights(0) = 0.5;
  velocityWeights(1) = 1.5;

  Pipeline::HostMatrix species0("species0", 1, 2);
  species0(0,0) = 4.0;
  species0(0,1) = 6.0;

  auto data = Pipeline::buildRHSMomentData(numCells,
                                           pointsPerCell,
                                           charges,
                                           eps0,
                                           {species0},
                                           velocityPoints,
                                           velocityWeights,
                                           n0,
                                           true,
                                           c);

  const double density = species0(0,0) * velocityWeights(0)
                       + species0(0,1) * velocityWeights(1);
  const double gamma0 = std::sqrt(1.0 + (velocityPoints(0,0) * velocityPoints(0,0)
                                      + velocityPoints(0,1) * velocityPoints(0,1)) / (c * c));
  const double gamma1 = std::sqrt(1.0 + (velocityPoints(1,0) * velocityPoints(1,0)
                                      + velocityPoints(1,1) * velocityPoints(1,1)) / (c * c));
  const double expectedJx = q * (species0(0,0) * velocityWeights(0) * velocityPoints(0,0) / gamma0
                              + species0(0,1) * velocityWeights(1) * velocityPoints(1,0) / gamma1);
  const double expectedJz = q * (species0(0,0) * velocityWeights(0) * velocityPoints(0,1) / gamma0
                              + species0(0,1) * velocityWeights(1) * velocityPoints(1,1) / gamma1);

  TEST_FLOATING_EQUALITY(data.rhoRHS(0, 0), q / eps0 * (density - n0[0]), 1e-14);
  TEST_FLOATING_EQUALITY(data.Jx(0, 0), expectedJx, 1e-14);
  TEST_FLOATING_EQUALITY(data.Jz(0, 0), expectedJz, 1e-14);
}

TEUCHOS_UNIT_TEST( TensorMaxwell1D2VMomentReductionPipeline, RejectsSpeciesCountMismatch )
{
  using Pipeline = TensorMaxwell1D2VMomentReductionPipeline;

  Pipeline::HostMatrix velocityPoints("velocityPoints", 1, 2);
  velocityPoints(0,0) = 0.0;
  velocityPoints(0,1) = 0.0;
  Pipeline::HostVector velocityWeights("velocityWeights", 1);
  velocityWeights(0) = 1.0;
  Pipeline::HostMatrix species0("species0", 1, 1);
  species0(0,0) = 1.0;

  TEST_THROW(Pipeline::buildRHSMomentData(1,
                                          1,
                                          {1.0, 2.0},
                                          1.0,
                                          {species0},
                                          velocityPoints,
                                          velocityWeights,
                                          {0.0}),
             std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMaxwell1D2VMomentReductionPipeline, BuildsRHSDataFromTensorSolutions )
{
  using Pipeline = TensorMaxwell1D2VMomentReductionPipeline;

  auto tensorMesh = makePipelineTensorMesh1C2V();
  auto dofLayout = tensorMesh->createDofLayout();
  TensorSolutionPtr species0Solution = Teuchos::rcp(new TensorSolution(tensorMesh, dofLayout, Tpetra::getDefaultComm()));
  TensorSolutionPtr species1Solution = Teuchos::rcp(new TensorSolution(tensorMesh, dofLayout, Tpetra::getDefaultComm()));
  species0Solution->registerCoefficientVariable(TENSOR_TRIAL_SPACE, "f");
  species1Solution->registerCoefficientVariable(TENSOR_TRIAL_SPACE, "f");
  fillTensorSolutionCoefficients(*species0Solution, 0.125);
  fillTensorSolutionCoefficients(*species1Solution, 0.875);

  auto cellTuple = tensorMesh->topology()->tensorCellTuple(0);
  TensorBasisCacheND basisCache(tensorMesh, tensorMesh->trialDiscretization("f"), false, cellTuple, std::vector<int>{2,2,2});
  Pipeline::HostMatrix velocityPoints("velocityPoints", basisCache.componentCache(1).pointCount() * basisCache.componentCache(2).pointCount(), 2);
  Pipeline::HostVector velocityWeights("velocityWeights", velocityPoints.extent_int(0));
  for (int uxPoint=0; uxPoint<basisCache.componentCache(1).pointCount(); uxPoint++)
  {
    for (int uzPoint=0; uzPoint<basisCache.componentCache(2).pointCount(); uzPoint++)
    {
      const int velocityOrdinal = uxPoint + basisCache.componentCache(1).pointCount() * uzPoint;
      velocityPoints(velocityOrdinal,0) = -0.5 + static_cast<double>(uxPoint);
      velocityPoints(velocityOrdinal,1) =  1.0 + 0.25 * static_cast<double>(uzPoint);
      velocityWeights(velocityOrdinal) = 0.5 + 0.125 * static_cast<double>(velocityOrdinal);
    }
  }

  const int pointsPerCell = basisCache.componentCache(0).pointCount();
  const std::vector<double> charges = {2.0, -0.5};
  const double eps0 = 3.0;
  std::vector<double> n0(pointsPerCell, 0.25);

  auto expectedSpecies0 = species0Solution->volumeValuesAsConfigurationVelocityMatrix1D2V(TENSOR_TRIAL_SPACE,
                                                                                           "f",
                                                                                           basisCache,
                                                                                           cellTuple,
                                                                                           std::vector<int>{0},
                                                                                           std::vector<int>{1,2});
  auto expectedSpecies1 = species1Solution->volumeValuesAsConfigurationVelocityMatrix1D2V(TENSOR_TRIAL_SPACE,
                                                                                           "f",
                                                                                           basisCache,
                                                                                           cellTuple,
                                                                                           std::vector<int>{0},
                                                                                           std::vector<int>{1,2});
  auto expected = Pipeline::buildRHSMomentData(1,
                                               pointsPerCell,
                                               charges,
                                               eps0,
                                               {expectedSpecies0, expectedSpecies1},
                                               velocityPoints,
                                               velocityWeights,
                                               n0);
  auto actual = Pipeline::buildRHSMomentDataFromTensorSolutions(1,
                                                                 pointsPerCell,
                                                                 charges,
                                                                 eps0,
                                                                 {species0Solution, species1Solution},
                                                                 TENSOR_TRIAL_SPACE,
                                                                 {"f", "f"},
                                                                 basisCache,
                                                                 cellTuple,
                                                                 std::vector<int>{0},
                                                                 std::vector<int>{1,2},
                                                                 velocityPoints,
                                                                 velocityWeights,
                                                                 n0);
  for (int point=0; point<pointsPerCell; point++)
  {
    TEST_FLOATING_EQUALITY(actual.rhoRHS(0, point), expected.rhoRHS(0, point), 1e-14);
    TEST_FLOATING_EQUALITY(actual.Jx(0, point), expected.Jx(0, point), 1e-14);
    TEST_FLOATING_EQUALITY(actual.Jz(0, point), expected.Jz(0, point), 1e-14);
  }

  TEST_THROW(Pipeline::buildRHSMomentDataFromTensorSolutions(1,
                                                              pointsPerCell,
                                                              charges,
                                                              eps0,
                                                              {species0Solution},
                                                              TENSOR_TRIAL_SPACE,
                                                              {"f"},
                                                              basisCache,
                                                              cellTuple,
                                                              std::vector<int>{0},
                                                              std::vector<int>{1,2},
                                                              velocityPoints,
                                                              velocityWeights,
                                                              n0),
             std::invalid_argument);
}
}
