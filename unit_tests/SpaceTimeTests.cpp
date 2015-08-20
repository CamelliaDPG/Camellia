//
//  SpaceTimeTests
//  Camellia
//
//  Created by Nate Roberts on 4/10/15.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "Function.h"
#include "MeshFactory.h"
#include "SpaceTimeHeatFormulation.h"
#include "TypeDefs.h"

using namespace Camellia;

namespace
{
MeshPtr singleElementSpaceTimeMesh(int spaceDim, int H1Order)
{
  vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
  vector<int> elementCounts(spaceDim,1);   // 1^d mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

  double t0 = 0.0, t1 = 1.0;
  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

  double epsilon = 1.0;
  SpaceTimeHeatFormulation form(spaceDim, epsilon);
  int delta_k = 1;
  vector<int> H1OrderVector(2);
  H1OrderVector[0] = H1Order;
  H1OrderVector[1] = H1Order;
  MeshPtr mesh = Teuchos::rcp( new Mesh(spaceTimeMeshTopo, form.bf(), H1OrderVector, delta_k) ) ;
  return mesh;
}
void testIntegrateConstantFunctionVolume(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 1;
  FunctionPtr one = Function::constant(1.0);
  MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
  double timeExtent = 1.0;
  double actualIntegral = one->integrate(mesh);
  double expectedIntegral = pow(2.0,spaceDim) * timeExtent;
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-15);
}

void testIntegrateConstantFunctionSides(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 1;
  FunctionPtr oneOnSides = Function::meshSkeletonCharacteristic();
  MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
  double timeExtent = 1.0;
  double actualIntegral = oneOnSides->integrate(mesh);
  double spatialIntegral = pow(2.0,spaceDim);
  double spatialSideIntegral = pow(2.0,spaceDim-1); // 1.0 in 1D, 2.0 in 2D, 4.0 in 3D
  int numSpatialSides = 2 * spaceDim; // 2 nodes for 1D, 4 edges in 2D, 6 faces in 3D
  int numTemporalSides = 2;
  double expectedIntegral = spatialIntegral * numTemporalSides + timeExtent * numSpatialSides * spatialSideIntegral;
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-15);
}

void testIntegrateTimeVaryingFunctionVolume(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 2;
  FunctionPtr t = Function::tn(1); // integral t^2 / 2; over [0,1] = 1/2
  MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
  double actualIntegral = t->integrate(mesh);
  double temporalIntegral = 0.5;
  double expectedIntegral = pow(2.0,spaceDim) * temporalIntegral;
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-15);
}

void testIntegrateTimeVaryingFunctionSides(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 2;
  FunctionPtr oneOnSides = Function::meshSkeletonCharacteristic();
  FunctionPtr t = Function::tn(1); // integral t^2 / 2; over [0,1] = 1/2
  MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
  double actualIntegral = (oneOnSides * t)->integrate(mesh);
  double spatialIntegral = pow(2.0,spaceDim); // will pick up one of these (where t=1; where t=0 we have no contribution)
  double spatialSideIntegral = pow(2.0,spaceDim-1); // 1.0 in 1D, 2.0 in 2D, 4.0 in 3D
  int numSpatialSides = 2 * spaceDim; // 2 nodes for 1D, 4 edges in 2D, 6 faces in 3D
  double temporalIntegral = 0.5;
  double expectedIntegral = spatialIntegral + temporalIntegral * numSpatialSides * spatialSideIntegral;
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-15);
}

void testIntegrateSpaceVaryingFunctionVolume(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 3;
  FunctionPtr f_x = Function::xn(2); // integral x^3 / 3; over [-1,1] = 2/3
  MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
  double actualIntegral = f_x->integrate(mesh);
  double temporalExtent = 1.0;
  double expectedIntegral_f_x = 2.0 / 3.0;
  double expectedIntegral = pow(2.0,spaceDim-1) * expectedIntegral_f_x * temporalExtent;
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-15);
}

void testIntegrateSpaceVaryingFunctionSides(int spaceDim, Teuchos::FancyOStream &out, bool &success)
{
  int H1Order = 3;
  FunctionPtr oneOnSides = Function::meshSkeletonCharacteristic();
  FunctionPtr f_x = Function::xn(2); // integral x^3 / 3; over [-1,1] = 2/3
  MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
  double actualIntegral = (f_x * oneOnSides)->integrate(mesh);
  double temporalExtent = 1.0;
  double expectedIntegral_f_x = 2.0 / 3.0;
  // on the temporal sides:
  double expectedIntegralTemporalSides = pow(2.0,spaceDim-1) * expectedIntegral_f_x;
  int numTemporalSides = 2;
  // on the sides that have x=-1 or x=1, f(x) = 1, so we just take the area
  double expectedIntegralSides_constant_x = pow(2.0,spaceDim-1) * temporalExtent;
  int numSides_constant_x = 2; // one where x=-1, one where x=1
  double expectedIntegralSides_varying_x = expectedIntegral_f_x * pow(2.0,max(spaceDim-2,0)) * temporalExtent;
  int numSides_varying_x = 2 * (spaceDim - 1); // 1D: 0, 2D: 2, 3D: 4
  double expectedIntegral = expectedIntegralTemporalSides * numTemporalSides
                            + expectedIntegralSides_constant_x * numSides_constant_x
                            + expectedIntegralSides_varying_x * numSides_varying_x;
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-15);
}

TEUCHOS_UNIT_TEST( SpaceTime, IntegrateConstantFunctionSides_1D )
{
  int spaceDim = 1;
  testIntegrateConstantFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateConstantFunctionSides_2D )
{
  int spaceDim = 2;
  testIntegrateConstantFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateConstantFunctionSides_3D )
{
  int spaceDim = 3;
  testIntegrateConstantFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateConstantFunctionVolume_1D )
{
  int spaceDim = 1;
  testIntegrateConstantFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateConstantFunctionVolume_2D )
{
  int spaceDim = 2;
  testIntegrateConstantFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateConstantFunctionVolume_3D )
{
  int spaceDim = 3;
  testIntegrateConstantFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateTimeVaryingFunctionVolume_1D )
{
  int spaceDim = 1;
  testIntegrateTimeVaryingFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateTimeVaryingFunctionVolume_2D )
{
  int spaceDim = 2;
  testIntegrateTimeVaryingFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateTimeVaryingFunctionVolume_3D )
{
  int spaceDim = 3;
  testIntegrateTimeVaryingFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateTimeVaryingFunctionSides_1D )
{
  int spaceDim = 1;
  testIntegrateTimeVaryingFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateTimeVaryingFunctionSides_2D )
{
  int spaceDim = 1;
  testIntegrateTimeVaryingFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateTimeVaryingFunctionSides_3D )
{
  int spaceDim = 1;
  testIntegrateTimeVaryingFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateSpaceVaryingFunctionVolume_1D )
{
  int spaceDim = 1;
  testIntegrateSpaceVaryingFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateSpaceVaryingFunctionVolume_2D )
{
  int spaceDim = 2;
  testIntegrateSpaceVaryingFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateSpaceVaryingFunctionVolume_3D )
{
  int spaceDim = 3;
  testIntegrateSpaceVaryingFunctionVolume(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateSpaceVaryingFunctionSides_1D )
{
  int spaceDim = 1;
  testIntegrateSpaceVaryingFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateSpaceVaryingFunctionSides_2D )
{
  int spaceDim = 2;
  testIntegrateSpaceVaryingFunctionSides(spaceDim, out, success);
}
TEUCHOS_UNIT_TEST( SpaceTime, IntegrateSpaceVaryingFunctionSides_3D )
{
  int spaceDim = 3;
  testIntegrateSpaceVaryingFunctionSides(spaceDim, out, success);
}
  TEUCHOS_UNIT_TEST( SpaceTime, UnitNormalTime_1D )
  {
    int spaceDim = 1;
    int H1Order = 2;
    MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);
    FunctionPtr n_spaceTime = Function::normalSpaceTime();
    GlobalIndexType cellID = 0;
    CellPtr cell = mesh->getTopology()->getCell(cellID);
    BasisCachePtr basisCache = BasisCache::basisCacheForCell(mesh, cellID);
    
    unsigned sideOrdinal_t0 = cell->topology()->getTemporalSideOrdinal(0);
    unsigned sideOrdinal_t1 = cell->topology()->getTemporalSideOrdinal(1);
    
    BasisCachePtr sideCache_t0 = basisCache->getSideBasisCache(sideOrdinal_t0);
    BasisCachePtr sideCache_t1 = basisCache->getSideBasisCache(sideOrdinal_t1);
    
    int numPoints = 1;
    Intrepid::FieldContainer<double> refPoint(numPoints,spaceDim);
    sideCache_t0->setRefCellPoints(refPoint);
    sideCache_t1->setRefCellPoints(refPoint);
    
    int numCells = 1;
    Intrepid::FieldContainer<double> values_t0(numCells,numPoints);
    n_spaceTime->t()->values(values_t0, sideCache_t0);
    Intrepid::FieldContainer<double> values_t1(numCells,numPoints);
    n_spaceTime->t()->values(values_t1, sideCache_t1);
    
    double value_t0_actual = values_t0[0];
    double value_t1_actual = values_t1[0];
    
    double value_t0_expected = -1.0; // outward normal at time 0
    double value_t1_expected = +1.0; // outward normal at time 1
    
    double tol = 1e-15;
    TEUCHOS_TEST_FLOATING_EQUALITY(value_t0_actual, value_t0_expected, tol, out, success);
    TEUCHOS_TEST_FLOATING_EQUALITY(value_t1_actual, value_t1_expected, tol, out, success);
  }
} // namespace
