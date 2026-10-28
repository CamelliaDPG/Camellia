#include "Teuchos_UnitTestHarness.hpp"

#include "TensorBasisCache.h"
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

TensorMeshPtr makeTensorMeshForBasisCacheTests()
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 2));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }, false, 3));
  auto tensorTopo = Teuchos::rcp(new TensorMeshTopology({component0, component1}));
  auto factory = TensorVarFactory::factory();
  auto tensorMesh = Teuchos::rcp(new TensorMesh(tensorTopo, factory, BasisFamily::hierarchicalBasisFamily(), 1, 1));

  tensorMesh->addTrialDiscretization(Teuchos::rcp(new TensorVarDiscretization(makeFieldTensorVar("u"), Teuchos::null, std::vector<int>{0,0})));
  return tensorMesh;
}

TEUCHOS_UNIT_TEST( TensorBasisCache, OneDCacheTabulatesLineBasisAndGeometry )
{
  auto tensorMesh = makeTensorMeshForBasisCacheTests();
  auto discretization = tensorMesh->trialDiscretization("u");
  ComponentCellID1D cellID(0,1,1);
  TensorBasisCache1D cache(tensorMesh, discretization, 0, false, cellID, 4);

  TEST_EQUALITY(cache.componentOrdinal(), 0);
  TEST_ASSERT(!cache.isTestSpace());
  TEST_EQUALITY(cache.polyOrder(), tensorMesh->effectivePolyOrder(discretization, cellID, 0, false));
  TEST_ASSERT(cache.basisCardinality() > 0);
  TEST_ASSERT(cache.pointCount() > 0);
  TEST_FLOATING_EQUALITY(cache.cellLeft(), 0.5, 1e-15);
  TEST_FLOATING_EQUALITY(cache.cellWidth(), 0.5, 1e-15);
  TEST_FLOATING_EQUALITY(cache.jacobian(), 0.25, 1e-15);
  TEST_FLOATING_EQUALITY(cache.derivativeScale(), 4.0, 1e-15);

  auto points = cache.referencePoints();
  auto weights = cache.referenceWeights();
  auto values = cache.valuesAtQuadrature();
  auto derivatives = cache.derivativesAtQuadrature();
  points.sync_host();
  weights.sync_host();
  values.sync_host();
  derivatives.sync_host();

  TEST_EQUALITY(static_cast<int>(points.extent(0)), cache.pointCount());
  TEST_EQUALITY(static_cast<int>(weights.extent(0)), cache.pointCount());
  TEST_EQUALITY(static_cast<int>(values.extent(0)), cache.basisCardinality());
  TEST_EQUALITY(static_cast<int>(values.extent(1)), cache.pointCount());
  TEST_EQUALITY(static_cast<int>(derivatives.extent(0)), cache.basisCardinality());
  TEST_EQUALITY(static_cast<int>(derivatives.extent(1)), cache.pointCount());
}

TEUCHOS_UNIT_TEST( TensorBasisCache, NDCacheUsesLowerIndexedComponentFastestPointOrdering )
{
  auto tensorMesh = makeTensorMeshForBasisCacheTests();
  auto discretization = tensorMesh->trialDiscretization("u");
  TensorMeshTopology::TensorCellTuple cellTuple { ComponentCellID1D(0,1,1), ComponentCellID1D(0,2,2) };
  TensorBasisCacheND cache(tensorMesh, discretization, false, cellTuple, std::vector<int>{3,4});

  TEST_EQUALITY(cache.componentCount(), 2);
  TEST_EQUALITY(cache.pointStride(0), 1);
  TEST_EQUALITY(cache.pointStride(1), cache.componentCache(0).pointCount());
  TEST_EQUALITY(cache.tensorPointCount(), cache.componentCache(0).pointCount() * cache.componentCache(1).pointCount());

  auto physicalPoints = cache.physicalPoints();
  auto quadratureWeights = cache.quadratureWeights();
  physicalPoints.sync_host();
  quadratureWeights.sync_host();

  const auto &cache0 = cache.componentCache(0);
  const auto &cache1 = cache.componentCache(1);
  auto points0 = cache0.referencePoints();
  auto points1 = cache1.referencePoints();
  auto weights0 = cache0.referenceWeights();
  auto weights1 = cache1.referenceWeights();
  points0.sync_host();
  points1.sync_host();
  weights0.sync_host();
  weights1.sync_host();

  for (IndexType tensorPointOrdinal=0; tensorPointOrdinal<cache.tensorPointCount(); tensorPointOrdinal++)
  {
    const int pointOrdinal0 = tensorPointOrdinal % cache0.pointCount();
    const int pointOrdinal1 = tensorPointOrdinal / cache0.pointCount();
    const double expectedX = cache0.cellLeft() + 0.5 * (points0.view_host()(pointOrdinal0) + 1.0) * cache0.cellWidth();
    const double expectedT = cache1.cellLeft() + 0.5 * (points1.view_host()(pointOrdinal1) + 1.0) * cache1.cellWidth();
    const double expectedWeight = weights0.view_host()(pointOrdinal0) * cache0.jacobian()
                                * weights1.view_host()(pointOrdinal1) * cache1.jacobian();

    TEST_FLOATING_EQUALITY(physicalPoints.view_host()(tensorPointOrdinal,0), expectedX, 1e-15);
    TEST_FLOATING_EQUALITY(physicalPoints.view_host()(tensorPointOrdinal,1), expectedT, 1e-15);
    TEST_FLOATING_EQUALITY(quadratureWeights.view_host()(tensorPointOrdinal), expectedWeight, 1e-15);
  }
}
}
