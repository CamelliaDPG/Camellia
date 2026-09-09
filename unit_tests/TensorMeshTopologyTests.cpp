#include "Teuchos_UnitTestHarness.hpp"

#include "TensorMeshTopology.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( TensorMeshTopology, ConstructSingleComponentMatchesComponentCells )
{
  auto component = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  TensorMeshTopology tensorTopo({component});

  TEST_EQUALITY(tensorTopo.componentCount(), 1);
  TEST_EQUALITY(tensorTopo.activeCellCount(), component->activeCellCount());

  const auto &cellTuples = tensorTopo.activeCellTuples();
  TEST_EQUALITY(cellTuples.size(), 2);
  TEST_EQUALITY(cellTuples[0].size(), 1);
  TEST_EQUALITY(cellTuples[0][0], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(cellTuples[1][0], ComponentCellID1D(0,1,1));

  for (IndexType ordinal=0; ordinal<tensorTopo.activeCellCount(); ordinal++)
  {
    TEST_EQUALITY(tensorTopo.tensorCellOrdinal(tensorTopo.tensorCellTuple(ordinal)), ordinal);
  }
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, ConstructTwoComponentFullProductCount )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{2,0}, {2,1}, {2,2}, {2,3}}) }));

  TensorMeshTopology tensorTopo({component0, component1});
  TEST_EQUALITY(tensorTopo.activeCellCount(), component0->activeCellCount() * component1->activeCellCount());
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, TensorCellOrderingUsesLowerIndexedComponentsFastest )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));

  TensorMeshTopology tensorTopo({component0, component1});
  const auto &cellTuples = tensorTopo.activeCellTuples();

  TEST_EQUALITY(cellTuples.size(), 4);
  TEST_EQUALITY(cellTuples[0][0], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(cellTuples[0][1], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(cellTuples[1][0], ComponentCellID1D(0,1,1));
  TEST_EQUALITY(cellTuples[1][1], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(cellTuples[2][0], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(cellTuples[2][1], ComponentCellID1D(0,1,1));
  TEST_EQUALITY(cellTuples[3][0], ComponentCellID1D(0,1,1));
  TEST_EQUALITY(cellTuples[3][1], ComponentCellID1D(0,1,1));
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, FlattenUnflattenRoundTrip )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));

  TensorMeshTopology tensorTopo({component0, component1});

  for (IndexType ordinal=0; ordinal<tensorTopo.activeCellCount(); ordinal++)
  {
    auto tuple = tensorTopo.tensorCellTuple(ordinal);
    TEST_EQUALITY(tensorTopo.tensorCellOrdinal(tuple), ordinal);
  }
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, ContainsTensorCellRecognizesPresentAndAbsentTuples )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));

  TensorMeshTopology tensorTopo({component0, component1});

  TensorMeshTopology::TensorCellTuple present { ComponentCellID1D(0,1,0), ComponentCellID1D(0,2,2) };
  TensorMeshTopology::TensorCellTuple absent0 { ComponentCellID1D(0,0,0), ComponentCellID1D(0,2,2) };
  TensorMeshTopology::TensorCellTuple absent1 { ComponentCellID1D(0,1,0), ComponentCellID1D(0,2,1) };
  TensorMeshTopology::TensorCellTuple wrongRank { ComponentCellID1D(0,1,0) };

  TEST_ASSERT(tensorTopo.containsTensorCell(present));
  TEST_ASSERT(!tensorTopo.containsTensorCell(absent0));
  TEST_ASSERT(!tensorTopo.containsTensorCell(absent1));
  TEST_ASSERT(!tensorTopo.containsTensorCell(wrongRank));
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, MixedRefinementComponentsStillFormFullProduct )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));

  TensorMeshTopology tensorTopo({component0, component1, component2});

  const IndexType expectedCount = component0->activeCellCount() * component1->activeCellCount() * component2->activeCellCount();
  TEST_EQUALITY(tensorTopo.activeCellCount(), expectedCount);

  const auto &cellTuples = tensorTopo.activeCellTuples();
  for (const auto &cellTuple : cellTuples)
  {
    TEST_EQUALITY(cellTuple.size(), 3);
    TEST_ASSERT(component0->containsCell(cellTuple[0]));
    TEST_ASSERT(component1->containsCell(cellTuple[1]));
    TEST_ASSERT(component2->containsCell(cellTuple[2]));
  }
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, SidesForTensorCell_FactorizedRepresentation )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));

  TensorMeshTopology tensorTopo({component0, component1});
  TensorMeshTopology::TensorCellTuple tuple { ComponentCellID1D(0,0,0), ComponentCellID1D(0,1,0) };
  auto sides = tensorTopo.sidesForTensorCell(tuple);

  TEST_EQUALITY(sides.size(), 4);
  TEST_EQUALITY(sides[0], TensorMeshTopology::TensorSide(tuple, 0, 0));
  TEST_EQUALITY(sides[1], TensorMeshTopology::TensorSide(tuple, 0, 1));
  TEST_EQUALITY(sides[2], TensorMeshTopology::TensorSide(tuple, 1, 0));
  TEST_EQUALITY(sides[3], TensorMeshTopology::TensorSide(tuple, 1, 1));
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, BoundarySideDetection_NonPeriodic )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }, false, 1));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }, false, 1));

  TensorMeshTopology tensorTopo({component0, component1});
  TensorMeshTopology::TensorCellTuple leftTuple  { ComponentCellID1D(0,1,0), ComponentCellID1D(0,0,0) };
  TensorMeshTopology::TensorCellTuple rightTuple { ComponentCellID1D(0,1,1), ComponentCellID1D(0,0,0) };

  TensorMeshTopology::TensorSide leftBoundary(leftTuple, 0, 0);
  TensorMeshTopology::TensorSide interiorShared(leftTuple, 0, 1);
  TensorMeshTopology::TensorSide rightBoundary(rightTuple, 0, 1);

  TEST_ASSERT(tensorTopo.isBoundarySide(leftBoundary));
  TEST_ASSERT(!tensorTopo.isBoundarySide(interiorShared));
  TEST_ASSERT(tensorTopo.isBoundarySide(rightBoundary));
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, SliceSingleComponentDropsThatDimension )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));

  TensorMeshTopology tensorTopo({component0, component1});
  TensorMeshTopology sliceTopo = tensorTopo.slice({{1,1}});

  TEST_EQUALITY(sliceTopo.componentCount(), 1);
  TEST_EQUALITY(sliceTopo.originalComponentOrdinal(0), 0);
  for (const auto &cellTuple : sliceTopo.activeCellTuples())
  {
    TEST_EQUALITY(cellTuple.size(), 1);
  }
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, SliceAtBoundaryVertex )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));

  TensorMeshTopology tensorTopo({component0, component1});
  TensorMeshTopology sliceTopo = tensorTopo.slice({{1,0}});

  TEST_EQUALITY(sliceTopo.componentCount(), 1);
  TEST_EQUALITY(sliceTopo.activeCellCount(), component0->activeCellCount());
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, SliceMultipleComponentsProducesLowerDimensionalTensorTopology )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));

  TensorMeshTopology tensorTopo({component0, component1, component2});
  TensorMeshTopology sliceTopo = tensorTopo.slice({{1,1}, {2,0}});

  TEST_EQUALITY(sliceTopo.componentCount(), 1);
  TEST_EQUALITY(sliceTopo.originalComponentOrdinal(0), 0);
  for (const auto &cellTuple : sliceTopo.activeCellTuples())
  {
    TEST_EQUALITY(cellTuple.size(), 1);
  }
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, RetainedComponentOrderingPreserved )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));

  TensorMeshTopology tensorTopo({component0, component1, component2});
  TensorMeshTopology sliceTopo = tensorTopo.slice({{1,1}});

  TEST_EQUALITY(sliceTopo.componentCount(), 2);
  TEST_EQUALITY(sliceTopo.originalComponentOrdinal(0), 0);
  TEST_EQUALITY(sliceTopo.originalComponentOrdinal(1), 2);
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, SlicedFlattenUnflattenRoundTrip )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));

  TensorMeshTopology tensorTopo({component0, component1, component2});
  TensorMeshTopology sliceTopo = tensorTopo.slice({{2,0}});

  for (IndexType ordinal=0; ordinal<sliceTopo.activeCellCount(); ordinal++)
  {
    auto tuple = sliceTopo.tensorCellTuple(ordinal);
    TEST_EQUALITY(sliceTopo.tensorCellOrdinal(tuple), ordinal);
  }
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, TensorSkeletonEntityCountMatchesSliceBasedSemantics )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));

  TensorMeshTopology tensorTopo({component0, component1, component2});
  auto skeleton = tensorTopo.tensorSkeleton(1);

  TEST_EQUALITY(skeleton.orthogonalComponentOrdinal(), 1);
  TEST_EQUALITY(skeleton.retainedComponentCount(), 2);
  TEST_EQUALITY(skeleton.retainedComponentOrdinals()[0], 0);
  TEST_EQUALITY(skeleton.retainedComponentOrdinals()[1], 2);
  TEST_EQUALITY(static_cast<IndexType>(skeleton.vertexOrdinals().size()), component1->vertexCount());

  IndexType expectedEntityCount = 0;
  for (IndexType vertexOrdinal=0; vertexOrdinal<component1->vertexCount(); vertexOrdinal++)
  {
    auto sliceTopo = tensorTopo.slice({{1, vertexOrdinal}});
    expectedEntityCount += sliceTopo.activeCellCount();
  }
  TEST_EQUALITY(skeleton.entityCount(), expectedEntityCount);
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, TensorSkeletonOrderingUsesLowerIndexedRetainedComponentsFastestWithinVertex )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));

  TensorMeshTopology tensorTopo({component0, component1, component2});
  auto skeleton = tensorTopo.tensorSkeleton(1);

  TEST_EQUALITY(skeleton.entityCount(), 8);
  TEST_EQUALITY(skeleton.entity(0).vertexOrdinal(), 0);
  TEST_EQUALITY(skeleton.entity(0).retainedCellTuple()[0], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(skeleton.entity(0).retainedCellTuple()[1], ComponentCellID1D(0,1,0));

  TEST_EQUALITY(skeleton.entity(1).vertexOrdinal(), 0);
  TEST_EQUALITY(skeleton.entity(1).retainedCellTuple()[0], ComponentCellID1D(0,1,1));
  TEST_EQUALITY(skeleton.entity(1).retainedCellTuple()[1], ComponentCellID1D(0,1,0));

  TEST_EQUALITY(skeleton.entity(2).vertexOrdinal(), 0);
  TEST_EQUALITY(skeleton.entity(2).retainedCellTuple()[0], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(skeleton.entity(2).retainedCellTuple()[1], ComponentCellID1D(0,1,1));
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, TensorSkeletonAgreesWithSliceAtEachVertex )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 2.0}, { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) }));
  auto component2 = Teuchos::rcp(new ComponentMeshTopology1D({-1.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));

  TensorMeshTopology tensorTopo({component0, component1, component2});
  auto skeleton = tensorTopo.tensorSkeleton(1);

  std::map<IndexType, int> entitiesPerVertex;
  for (const auto &entity : skeleton.entities())
  {
    entitiesPerVertex[entity.vertexOrdinal()]++;
  }

  for (IndexType vertexOrdinal=0; vertexOrdinal<component1->vertexCount(); vertexOrdinal++)
  {
    auto sliceTopo = tensorTopo.slice({{1,vertexOrdinal}});
    TEST_EQUALITY(entitiesPerVertex[vertexOrdinal], static_cast<int>(sliceTopo.activeCellCount()));
  }
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, TensorSkeletonBoundaryClassificationMatchesVertexIncidence )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {1,1}}) }));

  TensorMeshTopology tensorTopo({component0, component1});
  auto skeleton = tensorTopo.tensorSkeleton(1);

  bool foundBoundary = false;
  bool foundInterior = false;
  for (const auto &entity : skeleton.entities())
  {
    if (entity.vertexOrdinal() == 0 || entity.vertexOrdinal() == component1->vertexCount() - 1)
    {
      TEST_ASSERT(tensorTopo.isBoundarySkeletonEntity(entity));
      foundBoundary = true;
    }
    else
    {
      TEST_ASSERT(!tensorTopo.isBoundarySkeletonEntity(entity));
      foundInterior = true;
    }
  }
  TEST_ASSERT(foundBoundary);
  TEST_ASSERT(foundInterior);
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, RejectEmptyComponentList )
{
  TEST_THROW(TensorMeshTopology({}), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, RejectNullComponent )
{
  Teuchos::RCP<ComponentMeshTopology1D> nullComponent = Teuchos::null;
  TEST_THROW(TensorMeshTopology({nullComponent}), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, RejectInvalidFixedComponentOrdinal )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  TensorMeshTopology tensorTopo({component0});
  TEST_THROW(tensorTopo.slice({{3,0}}), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, RejectFixingAllComponents )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  TensorMeshTopology tensorTopo({component0, component1});
  TEST_THROW(tensorTopo.slice({{0,0},{1,0}}), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( TensorMeshTopology, RejectTensorSkeletonForInvalidComponentOrdinal )
{
  auto component0 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  auto component1 = Teuchos::rcp(new ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) }));
  TensorMeshTopology tensorTopo({component0, component1});
  TEST_THROW(tensorTopo.tensorSkeleton(2), std::invalid_argument);
}
}
