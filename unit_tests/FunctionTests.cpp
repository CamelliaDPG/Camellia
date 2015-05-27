//
//  FunctionTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 2/20/15.
//
//
#include "Teuchos_UnitTestHarness.hpp"

#include "BasisCache.h"
#include <CamelliaCellTools.h>
#include "CellTopology.h"
#include "Function.h"

using namespace Camellia;
using namespace Intrepid;

namespace
{

void testSpaceTimeNormalTimeComponent(CellTopoPtr spaceTopo, Teuchos::FancyOStream &out, bool &success)
{
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 1;
  bool createSideCache = true;
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  FunctionPtr spaceTimeNormalComponent = Function::normalSpaceTime()->t();
  for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
  {
    BasisCachePtr spaceTimeSideCache = spaceTimeBasisCache->getSideBasisCache(sideOrdinal);

    FieldContainer<double> spaceTimeNormals(1,spaceTimeSideCache->getRefCellPoints().dimension(0));
    spaceTimeNormalComponent->values(spaceTimeNormals,spaceTimeSideCache);

    if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
    {
      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal);
        TEST_COMPARE(abs(spaceTimeTemporalNormalComponent), <, 1e-15);
      }
    }
    else
    {
      // otherwise, we expect 0 in every component, except the last, where we expect ±1
      int temporalSideOrdinal = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
      double expectedValue = (temporalSideOrdinal == 0) ? -1.0 : 1.0;
      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal);
        TEST_FLOATING_EQUALITY(spaceTimeTemporalNormalComponent, expectedValue, 1e-15);
      }
    }
  }
}

FieldContainer<double> getScaledTranslatedRefNodes(CellTopoPtr topo, double nodeScaling, double nodeTranslation)
{
  FieldContainer<double> nodes(topo->getNodeCount(),topo->getDimension());
  CamelliaCellTools::refCellNodesForTopology(nodes, topo);
  for (int nodeOrdinal=0; nodeOrdinal<topo->getNodeCount(); nodeOrdinal++)
  {
    for (int d=0; d<topo->getDimension(); d++)
    {
      nodes(nodeOrdinal,d) *= nodeScaling;
      nodes(nodeOrdinal,d) += nodeTranslation;
    }
  }
  return nodes;
}

void setTemporalNodes(CellTopoPtr spaceTimeTopo, FieldContainer<double> &spaceTimeNodes, double t0, double t1)
{
  int d_time = spaceTimeTopo->getDimension() - 1;
  CellTopoPtr spaceTopo = spaceTimeTopo->getTensorialComponent();
  vector<unsigned> tensorComponentNodes = {0,0};
  for (unsigned spaceNode=0; spaceNode<spaceTopo->getNodeCount(); spaceNode++)
  {
    unsigned timeZeroNode = spaceTimeTopo->getNodeFromTensorialComponentNodes({spaceNode, 0});
    unsigned timeOneNode = spaceTimeTopo->getNodeFromTensorialComponentNodes({spaceNode, 1});
    spaceTimeNodes(timeZeroNode,d_time) = t0;
    spaceTimeNodes(timeOneNode,d_time)  = t1;
  }
}

void testSpaceTimeIntegrateByPartsInTime(CellTopoPtr spaceTopo, FunctionPtr f, Teuchos::FancyOStream &out, bool &success)
{
  /* Use the fact that
     (df/dt, 1)_K = < f, n_t >_dK - (f, 0) = < f, n_t >_dK
   */
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 2;
  bool createSideCache = true;
  double spaceNodeScaling = 0.5;
  double spaceNodeTranslation = 0.5; // scale, then translate
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  FieldContainer<double> physicalNodesSpaceTime = getScaledTranslatedRefNodes(spaceTimeTopo, spaceNodeScaling, spaceNodeTranslation);
  double t0 = 0.0, t1 = 1.0;
  setTemporalNodes(spaceTimeTopo,physicalNodesSpaceTime,t0,t1);
  physicalNodesSpaceTime.resize(1,physicalNodesSpaceTime.dimension(0),physicalNodesSpaceTime.dimension(1));
  spaceTimeBasisCache->setPhysicalCellNodes(physicalNodesSpaceTime, vector<GlobalIndexType>(), createSideCache);

  FunctionPtr n_spacetime = Function::normalSpaceTime();

  double lhs_integral = f->dt()->integrate(spaceTimeBasisCache);
  double rhs_integral = (f * n_spacetime->t())->integrate(spaceTimeBasisCache);

  double diff = abs(lhs_integral-rhs_integral);
  double tol = 1e-14;

  TEST_COMPARE(diff, <, tol);
  if (diff >= tol)
  {
    out << "lhs_integral: " << lhs_integral << endl;
    out << "rhs_integral: " << rhs_integral << endl;
  }
}

void testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopoPtr spaceTopo, FunctionPtr f_spatial, Teuchos::FancyOStream &out, bool &success)
{
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 2;
  bool createSideCache = true;
  double spaceNodeScaling = .5;
  double spaceNodeTranslation = .5; // scale, then translate
  BasisCachePtr spaceBasisCache = BasisCache::basisCacheForReferenceCell(spaceTopo, cubatureDegree, createSideCache);
  FieldContainer<double> physicalNodesSpace = getScaledTranslatedRefNodes(spaceTopo, spaceNodeScaling, spaceNodeTranslation);
  physicalNodesSpace.resize(1,physicalNodesSpace.dimension(0),physicalNodesSpace.dimension(1));
  spaceBasisCache->setPhysicalCellNodes(physicalNodesSpace, vector<GlobalIndexType>(), createSideCache);
  double spatialIntegral = f_spatial->integrate(spaceBasisCache);
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  FieldContainer<double> physicalNodesSpaceTime = getScaledTranslatedRefNodes(spaceTimeTopo, spaceNodeScaling, spaceNodeTranslation);
  double t0 = 0.0, t1 = .5;
  double temporalExtent = t1 - t0;
  setTemporalNodes(spaceTimeTopo,physicalNodesSpaceTime,t0,t1);
  physicalNodesSpaceTime.resize(1,physicalNodesSpaceTime.dimension(0),physicalNodesSpaceTime.dimension(1));
  spaceTimeBasisCache->setPhysicalCellNodes(physicalNodesSpaceTime, vector<GlobalIndexType>(), createSideCache);
  double temporalIntegralActual = f_spatial->integrate(spaceTimeBasisCache);
  double temporalIntegralExpected = spatialIntegral * temporalExtent;
  double diff = abs(temporalIntegralExpected - temporalIntegralActual);

  double tol = 1e-14;
  TEST_COMPARE(diff, <, tol);

  if (diff > tol)
  {
    out << "temporalIntegralActual: " << temporalIntegralActual << endl;
    out << "temporalIntegralExpected: " << temporalIntegralExpected << endl;
  }
}

void testSpaceTimeNormal(CellTopoPtr spaceTopo, Teuchos::FancyOStream &out, bool &success)
{
  CellTopoPtr spaceTimeTopo = CellTopology::cellTopology(spaceTopo, 1);
  int cubatureDegree = 1;
  bool createSideCache = true;
  BasisCachePtr spaceBasisCache = BasisCache::basisCacheForReferenceCell(spaceTopo, cubatureDegree, createSideCache);
  BasisCachePtr spaceTimeBasisCache = BasisCache::basisCacheForReferenceCell(spaceTimeTopo, cubatureDegree, createSideCache);
  FunctionPtr spaceTimeNormal = Function::normalSpaceTime();
  FunctionPtr spaceNormal = Function::normal();
  for (int sideOrdinal=0; sideOrdinal<spaceTimeTopo->getSideCount(); sideOrdinal++)
  {
    BasisCachePtr spaceTimeSideCache = spaceTimeBasisCache->getSideBasisCache(sideOrdinal);

    FieldContainer<double> spaceTimeNormals(1,spaceTimeSideCache->getRefCellPoints().dimension(0),spaceTimeTopo->getDimension());
    spaceTimeNormal->values(spaceTimeNormals,spaceTimeSideCache);

    if (spaceTimeTopo->sideIsSpatial(sideOrdinal))
    {
      // expect spaceTimeNormals to match spatial normals in the first d dimensions, and to be 0 in the final dimension
      int spatialSideOrdinal = spaceTimeTopo->getSpatialComponentSideOrdinal(sideOrdinal);
      BasisCachePtr spaceSideCache = spaceBasisCache->getSideBasisCache(spatialSideOrdinal);

      FieldContainer<double> spaceNormals(1,spaceSideCache->getRefCellPoints().dimension(0),spaceTopo->getDimension());
      spaceNormal->values(spaceNormals, spaceSideCache);

      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        // assume all normals are the same on the (spatial) side, and that there exists at least one point:
        for (int d=0; d<spaceTopo->getDimension(); d++)
        {
          double spaceNormalComponent = spaceNormals(0,0,d);
          double spaceTimeNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,d);
          TEST_FLOATING_EQUALITY(spaceNormalComponent, spaceTimeNormalComponent, 1e-15);
        }
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,spaceTopo->getDimension());
        TEST_COMPARE(abs(spaceTimeTemporalNormalComponent), <, 1e-15);
      }
    }
    else
    {
      // otherwise, we expect 0 in every component, except the last, where we expect ±1
      int temporalSideOrdinal = spaceTimeTopo->getTemporalComponentSideOrdinal(sideOrdinal);
      double expectedValue = (temporalSideOrdinal == 0) ? -1.0 : 1.0;
      for (int spaceTimePointOrdinal=0; spaceTimePointOrdinal < spaceTimeNormals.dimension(1); spaceTimePointOrdinal++)
      {
        // assume all normals are the same on the (spatial) side, and that there exists at least one point:
        for (int d=0; d<spaceTopo->getDimension(); d++)
        {
          double spaceTimeNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,d);
          TEST_COMPARE(abs(spaceTimeNormalComponent), <, 1e-15);
        }
        double spaceTimeTemporalNormalComponent = spaceTimeNormals(0,spaceTimePointOrdinal,spaceTopo->getDimension());

        TEST_FLOATING_EQUALITY(spaceTimeTemporalNormalComponent, expectedValue, 1e-15);
      }
    }
  }
}

TEUCHOS_UNIT_TEST( Function, MinAndMaxFunctions )
{
  FunctionPtr one = Function::constant(1);
  FunctionPtr two = Function::constant(2);
  FunctionPtr minFcn = Function::min(one,two);
  FunctionPtr maxFcn = Function::max(one,two);
  double x0 = 0, y0 = 0;
  double expectedValue = 1.0;
  double actualValue = Function::evaluate(minFcn, x0, y0);
  double tol = 1e-14;
  TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
  expectedValue = 2.0;
  actualValue = Function::evaluate(maxFcn, x0, y0);
  TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralLine )
{
  FunctionPtr x = Function::xn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::line(), x, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralQuad )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::quad(), x * y, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralTriangle )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::triangle(), x * y, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralHexahedron )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::hexahedron(), x * y * z, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegralTetrahedron )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr z = Function::zn(1);
  testSpaceTimeIntegralOfSpatiallyVaryingFunction(CellTopology::tetrahedron(), x * y * z, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegrationByPartsInTimeLine )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr t = Function::tn(1);
  testSpaceTimeIntegrateByPartsInTime(CellTopology::line(), x*t, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegrationByPartsInTimeQuad )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr t = Function::tn(1);
  testSpaceTimeIntegrateByPartsInTime(CellTopology::quad(), x*y*t, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeIntegrationByPartsInTimeTriangle )
{
  FunctionPtr x = Function::xn(1);
  FunctionPtr y = Function::yn(1);
  FunctionPtr t = Function::tn(1);
  testSpaceTimeIntegrateByPartsInTime(CellTopology::triangle(), x*y*t, out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalLine )
{
  testSpaceTimeNormal(CellTopology::line(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalQuad )
{
  testSpaceTimeNormal(CellTopology::quad(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTriangle )
{
  testSpaceTimeNormal(CellTopology::triangle(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalHexahedron )
{
  testSpaceTimeNormal(CellTopology::hexahedron(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTetrahedron )
{
  testSpaceTimeNormal(CellTopology::tetrahedron(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentLine )
{
  testSpaceTimeNormalTimeComponent(CellTopology::line(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentQuad )
{
  testSpaceTimeNormalTimeComponent(CellTopology::quad(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentTriangle )
{
  testSpaceTimeNormalTimeComponent(CellTopology::triangle(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentHexahedron )
{
  testSpaceTimeNormalTimeComponent(CellTopology::hexahedron(), out, success);
}

TEUCHOS_UNIT_TEST( Function, SpaceTimeNormalTimeComponentTetrahedron )
{
  testSpaceTimeNormalTimeComponent(CellTopology::tetrahedron(), out, success);
}

TEUCHOS_UNIT_TEST( Function, VectorMultiply )
{
  FunctionPtr x2 = Function::xn(2);
  FunctionPtr y4 = Function::yn(4);
  vector<double> weight(2);
  weight[0] = 3;
  weight[1] = 2;
  FunctionPtr g = Function::vectorize(x2,y4);
  double x0 = 2, y0 = 3;
  double expectedValue = weight[0] * x0 * x0 + weight[1] * y0 * y0 * y0 * y0;
  double actualValue = Function::evaluate(g * weight, x0, y0);
  double tol = 1e-14;
  TEST_FLOATING_EQUALITY(expectedValue,actualValue,tol);
}
} // namespace
