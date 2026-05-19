#include "Teuchos_UnitTestHarness.hpp"

#include "ComponentMeshTopology1D.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ConstructSingleRootSingleLeaf )
{
  ComponentMeshTopology1D topo({0.0, 1.0}, { ActiveLeafSet1D({{0,0}}) });

  TEST_EQUALITY(topo.rootCellCount(), 1);
  TEST_EQUALITY(topo.activeCellCount(), 1);
  TEST_EQUALITY(topo.activeCells()[0], ComponentCellID1D(0,0,0));
  TEST_FLOATING_EQUALITY(topo.rootCellLeft(0), 0.0, 1e-15);
  TEST_FLOATING_EQUALITY(topo.rootCellRight(0), 1.0, 1e-15);
  TEST_FLOATING_EQUALITY(topo.cellLeft(ComponentCellID1D(0,0,0)), 0.0, 1e-15);
  TEST_FLOATING_EQUALITY(topo.cellRight(ComponentCellID1D(0,0,0)), 1.0, 1e-15);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ConstructMultipleRoots )
{
  ComponentMeshTopology1D topo({0.0, 1.0, 3.0},
                               { ActiveLeafSet1D({{0,0}}), ActiveLeafSet1D({{0,0}}) });

  TEST_EQUALITY(topo.rootCellCount(), 2);
  TEST_EQUALITY(topo.activeCellCount(), 2);
  TEST_FLOATING_EQUALITY(topo.rootCellLeft(1), 1.0, 1e-15);
  TEST_FLOATING_EQUALITY(topo.rootCellRight(1), 3.0, 1e-15);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ActiveCellsEnumeratedRootThenLeftToRight )
{
  ComponentMeshTopology1D topo({0.0, 1.0, 2.0},
                               { ActiveLeafSet1D({{1,0}, {1,1}}),
                                 ActiveLeafSet1D({{2,0}, {2,1}, {2,2}, {2,3}}) });

  auto cells = topo.activeCells();
  TEST_EQUALITY(cells.size(), 6);
  TEST_EQUALITY(cells[0], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(cells[1], ComponentCellID1D(0,1,1));
  TEST_EQUALITY(cells[2], ComponentCellID1D(1,2,0));
  TEST_EQUALITY(cells[3], ComponentCellID1D(1,2,1));
  TEST_EQUALITY(cells[4], ComponentCellID1D(1,2,2));
  TEST_EQUALITY(cells[5], ComponentCellID1D(1,2,3));
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, MixedLevelGeometryMatchesExpectedIntervals )
{
  ComponentMeshTopology1D topo({0.0, 2.0},
                               { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) });

  ComponentCellID1D c0(0,1,0);
  ComponentCellID1D c1(0,2,2);
  ComponentCellID1D c2(0,2,3);

  TEST_FLOATING_EQUALITY(topo.cellLeft(c0),  0.0, 1e-15);
  TEST_FLOATING_EQUALITY(topo.cellRight(c0), 1.0, 1e-15);
  TEST_FLOATING_EQUALITY(topo.cellLeft(c1),  1.0, 1e-15);
  TEST_FLOATING_EQUALITY(topo.cellRight(c1), 1.5, 1e-15);
  TEST_FLOATING_EQUALITY(topo.cellLeft(c2),  1.5, 1e-15);
  TEST_FLOATING_EQUALITY(topo.cellRight(c2), 2.0, 1e-15);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ContainsCellRecognizesPresentAndAbsent )
{
  ComponentMeshTopology1D topo({0.0, 2.0},
                               { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) });

  TEST_ASSERT(topo.containsCell(ComponentCellID1D(0,1,0)));
  TEST_ASSERT(!topo.containsCell(ComponentCellID1D(0,0,0)));
  TEST_ASSERT(!topo.containsCell(ComponentCellID1D(0,2,1)));
  TEST_ASSERT(!topo.containsCell(ComponentCellID1D(3,0,0)));
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, RejectMismatchedRootAndLeafSetCounts )
{
  TEST_THROW(ComponentMeshTopology1D({0.0, 1.0, 2.0}, { ActiveLeafSet1D({{0,0}}) }), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, RejectNonIncreasingCoarseVertices )
{
  TEST_THROW(ComponentMeshTopology1D({0.0, 1.0, 1.0},
                                     { ActiveLeafSet1D({{0,0}}), ActiveLeafSet1D({{0,0}}) }), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, RejectInvalidActiveLeafSet )
{
  TEST_THROW(ComponentMeshTopology1D({0.0, 1.0}, { ActiveLeafSet1D({{1,0}, {2,1}}) }), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ConstructPeriodicComponent )
{
  ComponentMeshTopology1D topo({0.0, 1.0, 2.0},
                               { ActiveLeafSet1D({{0,0}}), ActiveLeafSet1D({{0,0}}) },
                               true,
                               1);
  TEST_ASSERT(topo.isPeriodic());
  TEST_EQUALITY(topo.rootCellCount(), 2);
  TEST_EQUALITY(topo.activeCellCount(), 2);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ConstructNonPeriodicComponent )
{
  ComponentMeshTopology1D topo({0.0, 1.0, 2.0},
                               { ActiveLeafSet1D({{0,0}}), ActiveLeafSet1D({{0,0}}) },
                               false,
                               1);
  TEST_ASSERT(!topo.isPeriodic());
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, DefaultPolyOrderAppliesToAllActiveCells )
{
  ComponentMeshTopology1D topo({0.0, 1.0, 2.0},
                               { ActiveLeafSet1D({{1,0}, {1,1}}), ActiveLeafSet1D({{0,0}}) },
                               false,
                               2);

  for (const auto &cellID : topo.activeCells())
  {
    TEST_EQUALITY(topo.cellPolyOrder(cellID), 2);
  }
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, CellPolyOrderOverride )
{
  ComponentMeshTopology1D topo({0.0, 1.0},
                               { ActiveLeafSet1D({{1,0}, {1,1}}) },
                               false,
                               2);

  ComponentCellID1D left(0,1,0);
  ComponentCellID1D right(0,1,1);
  topo.setCellPolyOrder(right, 5);

  TEST_EQUALITY(topo.cellPolyOrder(left), 2);
  TEST_EQUALITY(topo.cellPolyOrder(right), 5);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, RejectPolyOrderQueryForAbsentCell )
{
  ComponentMeshTopology1D topo({0.0, 1.0},
                               { ActiveLeafSet1D({{1,0}, {1,1}}) },
                               false,
                               2);

  TEST_THROW(topo.cellPolyOrder(ComponentCellID1D(0,0,0)), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, RefineSingleActiveCell )
{
  ComponentMeshTopology1D topo({0.0, 1.0},
                               { ActiveLeafSet1D({{0,0}}) },
                               false,
                               3);

  topo.refineCell(ComponentCellID1D(0,0,0));

  TEST_EQUALITY(topo.activeCellCount(), 2);
  TEST_ASSERT(topo.containsCell(ComponentCellID1D(0,1,0)));
  TEST_ASSERT(topo.containsCell(ComponentCellID1D(0,1,1)));
  TEST_ASSERT(!topo.containsCell(ComponentCellID1D(0,0,0)));
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, RefinedChildrenInheritParentPolyOrder )
{
  ComponentMeshTopology1D topo({0.0, 1.0},
                               { ActiveLeafSet1D({{0,0}}) },
                               false,
                               2);

  ComponentCellID1D parent(0,0,0);
  topo.setCellPolyOrder(parent, 5);
  topo.refineCell(parent);

  TEST_EQUALITY(topo.cellPolyOrder(ComponentCellID1D(0,1,0)), 5);
  TEST_EQUALITY(topo.cellPolyOrder(ComponentCellID1D(0,1,1)), 5);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ActiveCellOrderingUpdatedAfterRefinement )
{
  ComponentMeshTopology1D topo({0.0, 1.0, 2.0},
                               { ActiveLeafSet1D({{0,0}}), ActiveLeafSet1D({{0,0}}) },
                               false,
                               1);

  topo.refineCell(ComponentCellID1D(0,0,0));
  auto cells = topo.activeCells();

  TEST_EQUALITY(cells.size(), 3);
  TEST_EQUALITY(cells[0], ComponentCellID1D(0,1,0));
  TEST_EQUALITY(cells[1], ComponentCellID1D(0,1,1));
  TEST_EQUALITY(cells[2], ComponentCellID1D(1,0,0));
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, RejectRefinementOfNonActiveCell )
{
  ComponentMeshTopology1D topo({0.0, 1.0},
                               { ActiveLeafSet1D({{1,0}, {1,1}}) },
                               false,
                               1);

  TEST_THROW(topo.refineCell(ComponentCellID1D(0,0,0)), std::invalid_argument);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, ActiveVertexCountMixedLevel )
{
  ComponentMeshTopology1D topo({0.0, 2.0},
                               { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) });

  TEST_EQUALITY(topo.vertexCount(), 4);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, CellVertexIndices )
{
  ComponentMeshTopology1D topo({0.0, 2.0},
                               { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) });

  ComponentCellID1D c0(0,1,0);
  ComponentCellID1D c1(0,2,2);
  ComponentCellID1D c2(0,2,3);

  TEST_EQUALITY(topo.cellVertexIndex(c0,0), 0);
  TEST_EQUALITY(topo.cellVertexIndex(c0,1), 1);
  TEST_EQUALITY(topo.cellVertexIndex(c1,0), 1);
  TEST_EQUALITY(topo.cellVertexIndex(c1,1), 2);
  TEST_EQUALITY(topo.cellVertexIndex(c2,0), 2);
  TEST_EQUALITY(topo.cellVertexIndex(c2,1), 3);
}

TEUCHOS_UNIT_TEST( ComponentMeshTopology1D, VertexCoordinates )
{
  ComponentMeshTopology1D topo({0.0, 2.0},
                               { ActiveLeafSet1D({{1,0}, {2,2}, {2,3}}) });

  auto coords = topo.activeVertexCoordinates();
  coords.sync_host();

  TEST_FLOATING_EQUALITY(coords.h_view(0), 0.0, 1e-15);
  TEST_FLOATING_EQUALITY(coords.h_view(1), 1.0, 1e-15);
  TEST_FLOATING_EQUALITY(coords.h_view(2), 1.5, 1e-15);
  TEST_FLOATING_EQUALITY(coords.h_view(3), 2.0, 1e-15);
}
}
