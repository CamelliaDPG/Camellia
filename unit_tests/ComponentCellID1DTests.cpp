#include "Teuchos_UnitTestHarness.hpp"

#include "ComponentCellID1D.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( ComponentCellID1D, ValidRootCell )
{
  ComponentCellID1D cell(3, 0, 0);
  TEST_ASSERT(cell.isValid());
  TEST_EQUALITY(cell.rootOrdinal(), 3);
  TEST_EQUALITY(cell.level(), 0);
  TEST_EQUALITY(cell.address(), 0);
}

TEUCHOS_UNIT_TEST( ComponentCellID1D, RejectInvalidAddressForLevel0 )
{
  ComponentCellID1D cell(0, 0, 1);
  TEST_ASSERT(!cell.isValid());
}

TEUCHOS_UNIT_TEST( ComponentCellID1D, RejectInvalidAddressForLevel1 )
{
  ComponentCellID1D cell(0, 1, 2);
  TEST_ASSERT(!cell.isValid());
}

TEUCHOS_UNIT_TEST( ComponentCellID1D, ParentChildRelationships )
{
  ComponentCellID1D parent(2, 1, 1);
  ComponentCellID1D leftChild = parent.leftChild();
  ComponentCellID1D rightChild = parent.rightChild();

  TEST_EQUALITY(leftChild.rootOrdinal(), 2);
  TEST_EQUALITY(leftChild.level(), 2);
  TEST_EQUALITY(leftChild.address(), 2);

  TEST_EQUALITY(rightChild.rootOrdinal(), 2);
  TEST_EQUALITY(rightChild.level(), 2);
  TEST_EQUALITY(rightChild.address(), 3);

  TEST_EQUALITY(leftChild.parent(), parent);
  TEST_EQUALITY(rightChild.parent(), parent);
}

TEUCHOS_UNIT_TEST( ComponentCellID1D, AncestorRelation )
{
  ComponentCellID1D root(1, 0, 0);
  ComponentCellID1D child(1, 2, 3);
  ComponentCellID1D sibling(1, 2, 2);
  ComponentCellID1D otherRoot(2, 2, 3);

  TEST_ASSERT(root.isAncestorOf(child));
  TEST_ASSERT(!child.isAncestorOf(root));
  TEST_ASSERT(!sibling.isAncestorOf(child));
  TEST_ASSERT(!root.isAncestorOf(otherRoot));
}

TEUCHOS_UNIT_TEST( ComponentCellID1D, LeftToRightOrderingSameLevel )
{
  ComponentCellID1D left(0, 3, 1);
  ComponentCellID1D right(0, 3, 2);

  TEST_ASSERT(left < right);
  TEST_ASSERT(!(right < left));
}

TEUCHOS_UNIT_TEST( ComponentCellID1D, RootOrdinalAffectsOrdering )
{
  ComponentCellID1D firstRoot(0, 0, 0);
  ComponentCellID1D secondRoot(1, 0, 0);

  TEST_ASSERT(firstRoot < secondRoot);
}
}
