#include "Teuchos_UnitTestHarness.hpp"

#include "TensorMaxwell1D2VMomentReductionPipeline.h"

using namespace Camellia;

namespace
{
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
}
