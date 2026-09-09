#include "Teuchos_UnitTestHarness.hpp"

#include "ActiveLeafSet1D.h"

using namespace Camellia;

namespace
{
TEUCHOS_UNIT_TEST( ActiveLeafSet1D, SingleRootLeafIsValidExactCover )
{
  ActiveLeafSet1D leafSet({{0,0}});
  TEST_ASSERT(leafSet.isValid());
  TEST_ASSERT(leafSet.isExactCover());
  TEST_EQUALITY(leafSet.entries().size(), 1);
}

TEUCHOS_UNIT_TEST( ActiveLeafSet1D, BisectionLeavesAreValidExactCover )
{
  ActiveLeafSet1D leafSet({{1,0}, {1,1}});
  TEST_ASSERT(leafSet.isValid());
  TEST_ASSERT(leafSet.isExactCover());
  TEST_EQUALITY(leafSet.entries().size(), 2);
}

TEUCHOS_UNIT_TEST( ActiveLeafSet1D, MixedLevelExactCoverIsValid )
{
  ActiveLeafSet1D leafSet({{1,0}, {2,2}, {2,3}});
  TEST_ASSERT(leafSet.isValid());
  TEST_ASSERT(leafSet.isExactCover());
}

TEUCHOS_UNIT_TEST( ActiveLeafSet1D, OverlapIsRejected )
{
  ActiveLeafSet1D leafSet({{1,0}, {2,1}, {1,1}});
  TEST_ASSERT(!leafSet.isValid());
}

TEUCHOS_UNIT_TEST( ActiveLeafSet1D, GapIsRejected )
{
  ActiveLeafSet1D leafSet({{2,0}, {2,2}, {2,3}});
  TEST_ASSERT(!leafSet.isValid());
}

TEUCHOS_UNIT_TEST( ActiveLeafSet1D, CanonicalOrderingIsApplied )
{
  ActiveLeafSet1D leafSet({{2,3}, {1,0}, {2,2}});
  TEST_ASSERT(leafSet.isValid());
  TEST_EQUALITY(leafSet.entries()[0], ActiveLeafSet1D::Entry(1,0));
  TEST_EQUALITY(leafSet.entries()[1], ActiveLeafSet1D::Entry(2,2));
  TEST_EQUALITY(leafSet.entries()[2], ActiveLeafSet1D::Entry(2,3));
}

TEUCHOS_UNIT_TEST( ActiveLeafSet1D, DerivedCellIDsUseSpecifiedRootOrdinal )
{
  ActiveLeafSet1D leafSet({{1,0}, {1,1}});
  auto cellIDs = leafSet.cellIDsForRoot(7);

  TEST_EQUALITY(cellIDs.size(), 2);
  TEST_EQUALITY(cellIDs[0], ComponentCellID1D(7,1,0));
  TEST_EQUALITY(cellIDs[1], ComponentCellID1D(7,1,1));
}
}
