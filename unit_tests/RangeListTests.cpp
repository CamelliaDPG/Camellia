//
// © 2016 UChicago Argonne.  For licensing details, see LICENSE-Camellia in the licenses directory.
//
//
//  RangeListTests.cpp
//  Camellia
//
//  Created by Nate Roberts on 4/7/16.
//
//

#include "Teuchos_UnitTestHarness.hpp"

#include "RangeList.h"
#include "TypeDefs.h"

#include <set>

using namespace Camellia;
using namespace std;

namespace
{
  void testSetAndRangeListAgreeInIteration(set<int> &myset, RangeList<int> &mylist, Teuchos::FancyOStream &out, bool &success)
  {
    RangeListIterator<int> listIt = mylist.begin();
    set<int>::iterator setIt = myset.begin();
    int numValues = myset.size();
    for (int i=0; i<numValues; i++)
    {
      TEST_EQUALITY(*listIt, *setIt);
      ++listIt;
      ++setIt;
    }
    
    // try again, now with range-based for loop:
    setIt = myset.begin();
    for (int value : mylist)
    {
      TEST_EQUALITY(value, *setIt);
      setIt++;
    }
  }
  
  TEUCHOS_UNIT_TEST( RangeList, Insert )
  {
    RangeList<int> mylist;
    TEST_EQUALITY(mylist.length(), 0);
    TEST_EQUALITY(mylist.size(), 0);
    TEST_ASSERT(!mylist.contains(0));
    
    mylist.insert(0);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 1);
    TEST_ASSERT(mylist.contains(0));
    
    mylist.insert(0);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 1);
    TEST_ASSERT(!mylist.contains(1));
    
    mylist.insert(1);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 2);
    TEST_ASSERT(mylist.contains(1));
    TEST_ASSERT(!mylist.contains(3));
    
    mylist.insert(3);
    TEST_EQUALITY(mylist.length(), 2);
    TEST_EQUALITY(mylist.size(), 3);
    TEST_ASSERT(mylist.contains(3));
    TEST_ASSERT(!mylist.contains(2));
    
    mylist.insert(2);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 4);
    
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
  }
  
  TEUCHOS_UNIT_TEST( RangeList, Remove )
  {
    RangeList<int> mylist;
    mylist.insertRange(0, 4);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 4);
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(1);
    TEST_EQUALITY(mylist.length(), 2);
    TEST_EQUALITY(mylist.size(), 3);
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(1);
    TEST_EQUALITY(mylist.length(), 2);
    TEST_EQUALITY(mylist.size(), 3);
    TEST_ASSERT(mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(0);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 2);
    TEST_ASSERT(!mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(mylist.contains(3));
    
    mylist.remove(3);
    TEST_EQUALITY(mylist.length(), 1);
    TEST_EQUALITY(mylist.size(), 1);
    TEST_ASSERT(!mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(mylist.contains(2));
    TEST_ASSERT(!mylist.contains(3));
    
    mylist.remove(2);
    TEST_EQUALITY(mylist.length(), 0);
    TEST_EQUALITY(mylist.size(), 0);
    TEST_ASSERT(!mylist.contains(0));
    TEST_ASSERT(!mylist.contains(1));
    TEST_ASSERT(!mylist.contains(2));
    TEST_ASSERT(!mylist.contains(3));
  }
  
  TEUCHOS_UNIT_TEST( RangeList, Iterator )
  {
    vector<int> itemsToAdd = {1,5,4,3,2,0,7,6};
    
    RangeList<int> mylist;
    set<int> myset;
    
    // test empty range as well
    testSetAndRangeListAgreeInIteration(myset, mylist, out, success);
    
    for (int item : itemsToAdd)
    {
      mylist.insert(item);
      myset.insert(item);
      testSetAndRangeListAgreeInIteration(myset, mylist, out, success);
    }
  }
  
  TEUCHOS_UNIT_TEST( RangeList, OutOfOrderInsert )
  {
    // a particular case that broke MeshTopology during cell migration
    
    /*
     insert 1
     insert 5
     insert 4
     insert 3
     insert 2
     insert 0
     insert 7
     insert 6
     */
    
    RangeList<int> mylist;
    // [1]
    mylist.insert(1);
    TEST_ASSERT(mylist.contains(1));
    TEST_EQUALITY(mylist.size(),1);
    TEST_EQUALITY(mylist.length(),1);
    // [1],[5]
    mylist.insert(5);
    TEST_ASSERT(mylist.contains(5));
    TEST_EQUALITY(mylist.size(),2);
    TEST_EQUALITY(mylist.length(),2);
    // [1],[4-5]
    mylist.insert(4);
    TEST_ASSERT(mylist.contains(4));
    TEST_EQUALITY(mylist.size(),3);
    TEST_EQUALITY(mylist.length(),2);
    // [1],[3-5]
    mylist.insert(3);
    TEST_ASSERT(mylist.contains(3));
    TEST_EQUALITY(mylist.size(),4);
    TEST_EQUALITY(mylist.length(),2);
    // [1-5]
    mylist.insert(2);
    TEST_ASSERT(mylist.contains(2));
    TEST_EQUALITY(mylist.size(),5);
    TEST_EQUALITY(mylist.length(),1);
    // [0-5]
    mylist.insert(0);
    TEST_ASSERT(mylist.contains(0));
    TEST_EQUALITY(mylist.size(),6);
    TEST_EQUALITY(mylist.length(),1);
    // [0-5],[7]
    mylist.insert(7);
    TEST_ASSERT(mylist.contains(7));
    TEST_EQUALITY(mylist.size(),7);
    TEST_EQUALITY(mylist.length(),2);
    // [0-7]
    mylist.insert(6);
    TEST_ASSERT(mylist.contains(6));
    TEST_EQUALITY(mylist.size(),8);
    TEST_EQUALITY(mylist.length(),1);
  }
  //  TEUCHOS_UNIT_TEST( Int, Assignment )
  //  {
  //    int i1 = 4;
  //    int i2 = i1;
  //    TEST_EQUALITY( i2, i1 );
  //  }
} // namespace
