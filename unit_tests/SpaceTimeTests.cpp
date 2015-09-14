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
MeshPtr singleElementSpaceTimeMesh(int spaceDim, int H1Order, bool conforming = false)
{
  vector<double> dimensions(spaceDim,2.0); // 2.0^d hypercube domain
  vector<int> elementCounts(spaceDim,1);   // 1^d mesh
  vector<double> x0(spaceDim,-1.0);
  MeshTopologyPtr spatialMeshTopo = MeshFactory::rectilinearMeshTopology(dimensions, elementCounts, x0);

  double t0 = 0.0, t1 = 1.0;
  MeshTopologyPtr spaceTimeMeshTopo = MeshFactory::spaceTimeMeshTopology(spatialMeshTopo, t0, t1);

  double epsilon = 1.0;
  SpaceTimeHeatFormulation form(spaceDim, epsilon, conforming);
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
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-14);
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
  TEST_FLOATING_EQUALITY(actualIntegral, expectedIntegral, 1e-14);
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
  
  TEUCHOS_UNIT_TEST( SpaceTime, ConformingSpaceTimeTracesAreContinuousInTime )
  {
    // When a trace is in HGRAD and is not purely spatial, we expect it to be continuous in time.
    int spaceDim = 1;
    int H1Order = 2;
    double epsilon = 1.0;
    bool conforming = true; // will use HGRAD for the trace
    
    SpaceTimeHeatFormulation form(spaceDim, epsilon, conforming);
    VarPtr spaceTimeTrace = form.u_hat();
    
    MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order, conforming);
    GlobalIndexType cellID0 = 0;
    CellPtr cell = mesh->getTopology()->getCell(cellID0);
    
    unsigned sideOrdinal_t0 = cell->topology()->getTemporalSideOrdinal(0);
    unsigned sideOrdinal_t1 = cell->topology()->getTemporalSideOrdinal(1);
    
    // refine to produce 4 cells, two for time interval [0.0,0.5], two for [0.5,1.0]
    mesh->hRefine(vector<GlobalIndexType>{cellID0});
    
    // determine which children are the earlier-time ones:
    vector< pair<GlobalIndexType, unsigned> > childrenTime0 = cell->childrenForSide(sideOrdinal_t0);
    
    map<int,FunctionPtr> projectionMap;
    
    SolutionPtr solution = Solution::solution(mesh->bilinearForm(), mesh);
    
    set<GlobalIndexType> rankLocalCells = mesh->cellIDsInPartition();
    
    // set 1 values for time 0 children
    projectionMap[spaceTimeTrace->ID()] = Function::constant(1.0);
    for (auto childEntry : childrenTime0)
    {
      GlobalIndexType childCellID = childEntry.first;
      if (rankLocalCells.find(childCellID) == rankLocalCells.end())
      {
        for (unsigned sideOrdinal=0; sideOrdinal < cell->getSideCount(); sideOrdinal++)
        {
          if (cell->topology()->sideIsSpatial(sideOrdinal))
          {
            solution->projectOntoCell(projectionMap, childCellID, sideOrdinal);
          }
        }
      }
    }
    
    // set 2.0 values for time 1 children
    vector< pair<GlobalIndexType, unsigned> > childrenTime1 = cell->childrenForSide(sideOrdinal_t1);
    projectionMap[spaceTimeTrace->ID()] = Function::constant(2.0);
    for (auto childEntry : childrenTime1)
    {
      GlobalIndexType childCellID = childEntry.first;
      if (rankLocalCells.find(childCellID) == rankLocalCells.end())
      {
        for (unsigned sideOrdinal=0; sideOrdinal < cell->getSideCount(); sideOrdinal++)
        {
          if (cell->topology()->sideIsSpatial(sideOrdinal))
          {
            solution->projectOntoCell(projectionMap, childCellID, sideOrdinal);
          }
        }
      }
    }
    
    // initialize LHS vector will use existing local coefficients to determine global representation
    solution->initializeLHSVector();
    
    // import solution then uses the LHS vector to determine local coefficients
    solution->importSolution();
    
    FunctionPtr solnFxn = Function::solution(spaceTimeTrace, solution);
    for (auto childEntry : childrenTime0)
    {
      GlobalIndexType cellIDTime0 = childEntry.first;
      CellPtr cellTime0 = mesh->getTopology()->getCell(cellIDTime0);
      CellPtr cellTime1 = cellTime0->getNeighbor(sideOrdinal_t1, mesh->getTopology());
      GlobalIndexType cellIDTime1 = cellTime1->cellIndex();
      Intrepid::FieldContainer<double> refPointTime0(1,spaceDim); // spaceDim because spatial side x time
      Intrepid::FieldContainer<double> refPointTime1(1,spaceDim);
      // we don't really care what point we're talking about, so long as the time coordinate is right
      refPointTime0(0,spaceDim-1) = 1.0; // time 0 should get rightmost time (1.0 in ref space)
      refPointTime1(0,spaceDim-1) = -1.0; // time 1 should get leftmost time (-1.0 in ref space)
      
      BasisCachePtr basisCacheTime0 = BasisCache::basisCacheForCell(mesh, cellIDTime0);
      BasisCachePtr basisCacheTime1 = BasisCache::basisCacheForCell(mesh, cellIDTime1);
      
      for (unsigned sideOrdinal=0; sideOrdinal < cell->getSideCount(); sideOrdinal++)
      {
        if (cell->topology()->sideIsSpatial(sideOrdinal))
        {
          BasisCachePtr sideBasisCacheTime0 = basisCacheTime0->getSideBasisCache(sideOrdinal);
          BasisCachePtr sideBasisCacheTime1 = basisCacheTime1->getSideBasisCache(sideOrdinal);
          sideBasisCacheTime0->setRefCellPoints(refPointTime0);
          sideBasisCacheTime1->setRefCellPoints(refPointTime1);
          Intrepid::FieldContainer<double> valuesTime0(1,1), valuesTime1(1,1);
          
          if (rankLocalCells.find(cellIDTime0) != rankLocalCells.end())
          {
            solnFxn->values(valuesTime0, sideBasisCacheTime0);
          }
          if (rankLocalCells.find(cellIDTime1) != rankLocalCells.end())
          {
            solnFxn->values(valuesTime1, sideBasisCacheTime1);
          }
          
          double valueTime0 = valuesTime0[0], valueTime1 = valuesTime1[0];
          valueTime0 = MPIWrapper::sum(valueTime0);
          valueTime1 = MPIWrapper::sum(valueTime1);
          
          double tol = 1e-15;
          TEUCHOS_TEST_FLOATING_EQUALITY(valueTime0, valueTime1, tol, out, success);
        }
      } 
    }
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

  TEUCHOS_UNIT_TEST( SpaceTime, TimeFunctionEqualsLastDimension ) // tn == zn in 2D
  {
    for (int spaceDim = 1; spaceDim < 3; spaceDim++)
    {
      int H1Order = 2;
      MeshPtr mesh = singleElementSpaceTimeMesh(spaceDim, H1Order);

      FunctionPtr t = Function::tn(1);
      FunctionPtr f;
      if (spaceDim == 1) {
        f = Function::yn(1);
      }
      else
      {
        f = Function::zn(1);
      }
      
      double tol = 1e-15;

      // volume first:
      FunctionPtr volumeDiff = f - t;
      double volumeErr = volumeDiff->l2norm(mesh);
      TEUCHOS_TEST_COMPARE(volumeErr, <, tol, out, success);
      
      // boundary (skeleton):
      FunctionPtr boundaryDiff = (f - t) * Function::meshSkeletonCharacteristic();
      double boundaryErr = boundaryDiff->l2norm(mesh);
      TEUCHOS_TEST_COMPARE(boundaryErr, <, tol, out, success);
    }
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
    
    // next, try spatial side ordinals -- expect 0 values here
    unsigned sideOrdinal_x0 = cell->topology()->getSpatialSideOrdinal(0);
    unsigned sideOrdinal_x1 = cell->topology()->getSpatialSideOrdinal(1);
    
    BasisCachePtr sideCache_x0 = basisCache->getSideBasisCache(sideOrdinal_x0);
    BasisCachePtr sideCache_x1 = basisCache->getSideBasisCache(sideOrdinal_x1);
    
    sideCache_x0->setRefCellPoints(refPoint);
    sideCache_x1->setRefCellPoints(refPoint);
    
    Intrepid::FieldContainer<double> values_x0(numCells,numPoints);
    n_spaceTime->t()->values(values_x0, sideCache_x0);
    Intrepid::FieldContainer<double> values_x1(numCells,numPoints);
    n_spaceTime->t()->values(values_x1, sideCache_x1);
    
    double value_x0_actual = values_x0[0];
    double value_x1_actual = values_x1[0];
    
    double value_x0_expected = 0.0;
    double value_x1_expected = 0.0;
    
    TEUCHOS_TEST_FLOATING_EQUALITY(value_x0_actual, value_x0_expected, tol, out, success);
    TEUCHOS_TEST_FLOATING_EQUALITY(value_x1_actual, value_x1_expected, tol, out, success);
  }
} // namespace
