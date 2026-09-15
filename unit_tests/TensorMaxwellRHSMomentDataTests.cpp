#include "Teuchos_UnitTestHarness.hpp"

#include "TensorMaxwellRHSMomentData.h"

#include "BasisCache.h"
#include "Camellia_FieldContainer.hpp"

#include <Kokkos_Core.hpp>

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorMaxwellRHSMomentData, AdapterMatchesPlanar1D2VMaxwellSetMomentsAlgebra )
{
  const int numCells = 2;
  const int pointsPerCell = 2;
  const double eps0 = 4.0;
  std::vector<double> charges = {2.0, -1.0};
  std::vector<double> n0 = {0.5, 1.0, 1.5, 2.0};

  TensorPlanar1D2VSpeciesMoments species0({1.0, 2.0, 3.0, 4.0},
                                          {10.0, 20.0, 30.0, 40.0},
                                          {100.0, 200.0, 300.0, 400.0});
  TensorPlanar1D2VSpeciesMoments species1({5.0, 6.0, 7.0, 8.0},
                                          {50.0, 60.0, 70.0, 80.0},
                                          {500.0, 600.0, 700.0, 800.0});

  auto data = TensorMaxwell1D2VMomentAdapter::buildRHSMomentData(numCells,
                                                                 pointsPerCell,
                                                                 charges,
                                                                 eps0,
                                                                 {species0, species1},
                                                                 n0);

  TEST_EQUALITY(data.numConfigurationCells(), numCells);
  TEST_EQUALITY(data.numPointsPerConfigurationCell(), pointsPerCell);
  TEST_EQUALITY(data.numEntries(), 4);

  for (int cell=0; cell<numCells; cell++)
  {
    for (int point=0; point<pointsPerCell; point++)
    {
      const int entry = cell * pointsPerCell + point;
      const double expectedRho = charges[0] / eps0 * (species0._density[entry] - n0[entry])
                               + charges[1] / eps0 * (species1._density[entry] - n0[entry]);
      const double expectedJx = charges[0] * species0._xProperVelocityMoment[entry]
                              + charges[1] * species1._xProperVelocityMoment[entry];
      const double expectedJz = charges[0] * species0._zProperVelocityMoment[entry]
                              + charges[1] * species1._zProperVelocityMoment[entry];
      TEST_FLOATING_EQUALITY(data.rhoRHS(cell, point), expectedRho, 1e-14);
      TEST_FLOATING_EQUALITY(data.Jx(cell, point), expectedJx, 1e-14);
      TEST_FLOATING_EQUALITY(data.Jz(cell, point), expectedJz, 1e-14);
    }
  }

  TEST_THROW(data.rhoRHS(2, 0), std::invalid_argument);
  TEST_THROW(TensorMaxwell1D2VMomentAdapter::buildRHSMomentData(numCells, pointsPerCell, charges, 0.0, {species0, species1}, n0), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMaxwellRHSMomentData, DevicePointAssemblyAppliesMaxwellRHSSignsAndScaling )
{
  const int numCells = 2;
  const int pointsPerCell = 3;
  const double eps0 = 4.0;
  TensorMaxwellRHSMomentData data(numCells, pointsPerCell);
  for (int cell=0; cell<numCells; cell++)
  {
    for (int point=0; point<pointsPerCell; point++)
    {
      data.rhoRHS(cell, point) = 1.0 + 10.0 * cell + point;
      data.Jx(cell, point) = 2.0 + 10.0 * cell + point;
      data.Jz(cell, point) = 3.0 + 10.0 * cell + point;
    }
  }
  data.populateDeviceMirrors();
  TEST_ASSERT(data.hasDeviceMirrors());

  TensorMaxwellRHSMomentData::DeviceVector rhoTerm("rhoTerm", data.numEntries());
  TensorMaxwellRHSMomentData::DeviceVector jxTerm("jxTerm", data.numEntries());
  TensorMaxwellRHSMomentData::DeviceVector jzTerm("jzTerm", data.numEntries());
  TensorMaxwellRHSDevicePointAssembly::assemblePointSourceTerms(data, eps0, rhoTerm, jxTerm, jzTerm);

  auto rhoHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhoTerm);
  auto jxHost  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jxTerm);
  auto jzHost  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jzTerm);
  for (int cell=0; cell<numCells; cell++)
  {
    for (int point=0; point<pointsPerCell; point++)
    {
      const int entry = cell * pointsPerCell + point;
      TEST_FLOATING_EQUALITY(rhoHost(entry), data.rhoRHS(cell, point), 1e-14);
      TEST_FLOATING_EQUALITY(jxHost(entry), -data.Jx(cell, point) / eps0, 1e-14);
      TEST_FLOATING_EQUALITY(jzHost(entry),  data.Jz(cell, point) / eps0, 1e-14);
    }
  }

  TensorMaxwellRHSMomentData::DeviceVector shortOutput("shortOutput", data.numEntries() - 1);
  TEST_THROW(TensorMaxwellRHSDevicePointAssembly::assemblePointSourceTerms(data, eps0, shortOutput, jxTerm, jzTerm), std::invalid_argument);
  TensorMaxwellRHSMomentData missingMirrors(numCells, pointsPerCell);
  TEST_THROW(TensorMaxwellRHSDevicePointAssembly::assemblePointSourceTerms(missingMirrors, eps0, rhoTerm, jxTerm, jzTerm), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMaxwellRHSMomentData, FunctionAdapterProvidesPointValuesForMaxwellRHS )
{
  auto basisCache = BasisCache::basisCache1D(0.0, 1.0, 2);
  const int numCells = static_cast<int>(basisCache->cellIDs().size());
  const int pointsPerCell = basisCache->getPhysicalCubaturePointsFC().dimension(1);
  TensorMaxwellRHSMomentData data(numCells, pointsPerCell);
  for (int cell=0; cell<numCells; cell++)
  {
    for (int point=0; point<pointsPerCell; point++)
    {
      data.rhoRHS(cell, point) = 1.0 + 10.0 * cell + point;
      data.Jx(cell, point) = 2.0 + 10.0 * cell + point;
      data.Jz(cell, point) = 3.0 + 10.0 * cell + point;
    }
  }

  Camellia::FieldContainer<double> values(numCells, pointsPerCell);
  TensorMaxwellRHSFunctionAdapter::rhoRHSFunction(data)->values(values, basisCache);
  for (int cell=0; cell<numCells; cell++)
    for (int point=0; point<pointsPerCell; point++)
      TEST_FLOATING_EQUALITY(values(cell,point), data.rhoRHS(cell, point), 1e-14);

  TensorMaxwellRHSFunctionAdapter::JxFunction(data)->values(values, basisCache);
  for (int cell=0; cell<numCells; cell++)
    for (int point=0; point<pointsPerCell; point++)
      TEST_FLOATING_EQUALITY(values(cell,point), data.Jx(cell, point), 1e-14);

  TensorMaxwellRHSFunctionAdapter::JzFunction(data)->values(values, basisCache);
  for (int cell=0; cell<numCells; cell++)
    for (int point=0; point<pointsPerCell; point++)
      TEST_FLOATING_EQUALITY(values(cell,point), data.Jz(cell, point), 1e-14);

  TensorMaxwellRHSMomentData mismatchedData(numCells + 1, pointsPerCell);
  TEST_THROW(TensorMaxwellRHSFunctionAdapter::rhoRHSFunction(mismatchedData)->values(values, basisCache), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMaxwellRHSBoundaryAssembly, PointVolumeTermsMatchExistingDeviceSourceAssembly )
{
  const int numCells = 1;
  const int pointsPerCell = 3;
  const double eps0 = 5.0;
  TensorMaxwellRHSMomentData data(numCells, pointsPerCell);
  for (int point=0; point<pointsPerCell; point++)
  {
    data.rhoRHS(0, point) = 0.5 + point;
    data.Jx(0, point) = 1.5 + point;
    data.Jz(0, point) = 2.5 + point;
  }
  data.populateDeviceMirrors();

  TensorMaxwellRHSMomentData::DeviceVector rhoTerm("tensor Maxwell assembly rho", data.numEntries());
  TensorMaxwellRHSMomentData::DeviceVector jxTerm("tensor Maxwell assembly jx", data.numEntries());
  TensorMaxwellRHSMomentData::DeviceVector jzTerm("tensor Maxwell assembly jz", data.numEntries());
  TensorMaxwellRHSBoundaryAssembly::assemblePointVolumeTerms(data, eps0, rhoTerm, jxTerm, jzTerm);

  auto rhoHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), rhoTerm);
  auto jxHost  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jxTerm);
  auto jzHost  = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), jzTerm);
  for (int point=0; point<pointsPerCell; point++)
  {
    TEST_FLOATING_EQUALITY(rhoHost(point), data.rhoRHS(0, point), 1e-14);
    TEST_FLOATING_EQUALITY(jxHost(point), -data.Jx(0, point) / eps0, 1e-14);
    TEST_FLOATING_EQUALITY(jzHost(point),  data.Jz(0, point) / eps0, 1e-14);
  }
}

TEUCHOS_UNIT_TEST( TensorMaxwellRHSBoundaryAssembly, H1PointBoundaryCopiesLowerAndUpperValuesWithoutNormalSign )
{
  TensorMaxwellRHSMomentData::DeviceVector traceValues("tensor Maxwell H1 point trace values", 4);
  auto traceHost = Kokkos::create_mirror_view(traceValues);
  for (int i=0; i<4; i++) traceHost(i) = 2.0 + 0.5 * i;
  Kokkos::deep_copy(traceValues, traceHost);

  TensorMaxwellRHSMomentData::DeviceVector lower("tensor Maxwell lower H1 boundary terms", 4);
  TensorMaxwellRHSMomentData::DeviceVector upper("tensor Maxwell upper H1 boundary terms", 4);
  TensorMaxwellRHSBoundaryAssembly::assembleH1PointBoundaryTerms1D(traceValues, lower, upper);

  auto lowerHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lower);
  auto upperHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), upper);
  for (int i=0; i<4; i++)
  {
    TEST_FLOATING_EQUALITY(lowerHost(i), traceHost(i), 1e-14);
    TEST_FLOATING_EQUALITY(upperHost(i), traceHost(i), 1e-14);
  }

  TensorMaxwellRHSMomentData::DeviceVector shortOutput("short tensor Maxwell H1 boundary output", 3);
  TEST_THROW(TensorMaxwellRHSBoundaryAssembly::assembleH1PointBoundaryTerms1D(traceValues, shortOutput, upper), std::invalid_argument);
  TEST_THROW(TensorMaxwellRHSBoundaryAssembly::assembleH1PointBoundaryTerms1D(traceValues, lower, shortOutput), std::invalid_argument);
}
}
